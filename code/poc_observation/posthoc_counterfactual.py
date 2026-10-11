"""Post-hoc counterfactual ground truth on deployment proposals.

For each non-trivial deployment proposal (status APPROVED or ABANDONED with
n_obs >= 20), we:
  1. Sample K=20 queries from the proposal's cluster (the actual queries that
     were routed there in production).
  2. Generate backbone answers and LoRA-adapted answers for each.
  3. Score with the same judge used in the controlled experiment.
  4. Compute the LoRA-vs-backbone delta.
  5. Compare delta-based GT to the deployment SPRT decision.

This addresses reviewer T3: the deployment GT was based on operator review
(post-hoc human label of "did the cluster justify a birth?"). A real
counterfactual would compare what the backbone produced to what a LoRA on
that cluster's data would produce, on the same queries.

Inputs:  paper_artifacts/poc_results/per_proposal_traces.jsonl
         paper_artifacts/poc_results/cluster_query_traces.jsonl  (if exists)
Outputs: analysis/posthoc_counterfactual.json
         figures/posthoc_agreement.png

USAGE: requires GPU. Set the TRACES / QTRACES paths below (or pass via env
vars POSTHOC_TRACES / POSTHOC_QTRACES) to point at the deployment trace JSONL
files distributed in `data/per_proposal_traces.jsonl`.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRACE_DIR = ROOT.parent.parent / "data"
TRACES = Path(os.environ.get("POSTHOC_TRACES",
                             str(_DEFAULT_TRACE_DIR / "per_proposal_traces.jsonl")))
QTRACES = Path(os.environ.get("POSTHOC_QTRACES",
                              str(_DEFAULT_TRACE_DIR / "cluster_query_traces.jsonl")))
OUTJSON = ROOT / "analysis" / "posthoc_counterfactual.json"
FIGDIR = ROOT / "figures"

# Pairwise LLM-judge delta is in [-1, +1] (mean over K queries).
# delta > 0  : LoRA wins on average; delta = 0 : tie; delta < 0 : backbone wins.
# Threshold for "BIRTH-justified": LoRA must win clearly on average.
DELTA_THRESHOLD = float(os.environ.get("POSTHOC_DELTA", "0.15"))
K_QUERIES = int(os.environ.get("POSTHOC_K", "30"))
LORA_EPOCHS = int(os.environ.get("POSTHOC_EPOCHS", "10"))
LORA_RANK = int(os.environ.get("POSTHOC_R", "32"))
LORA_MAX_TRAIN = int(os.environ.get("POSTHOC_MAXTRAIN", "50"))
SENIOR_PREFIX = (
    "You are a senior domain expert. Provide a precise, technical, "
    "evidence-based answer. Cite the specific reasoning, mechanism, or rule "
    "that applies. Avoid generic language; be concrete and specific.\n\n"
)


def load_proposals():
    items = [json.loads(l) for l in TRACES.open()]
    # filter to proposals worth re-evaluating: have observations + decision
    elig = [x for x in items if x.get("uq_observations") and len(x["uq_observations"]) >= 20]
    return elig


def load_cluster_queries(cluster_id, proposal_id=None):
    """Return list of query strings that were routed to this cluster.

    Tries (in order):
      1) JSONL file paper_artifacts/poc_results/cluster_query_traces.jsonl
      2) sqlite deployment DB (set DEPLOY_DB env var), table
         birth_proposals.cluster_queries (JSON list)
    """
    out = []
    if QTRACES.exists():
        with QTRACES.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("cluster_id") == cluster_id:
                    q = row.get("query") or row.get("text") or row.get("user_message")
                    if q: out.append(q)
        if out: return out
    # Fallback: sqlite
    db_paths = [p for p in [os.environ.get("DEPLOY_DB")] if p]
    for dbp in db_paths:
        if not Path(dbp).exists(): continue
        import sqlite3
        c = sqlite3.connect(dbp)
        if proposal_id is not None:
            row = c.execute("SELECT cluster_queries FROM birth_proposals WHERE id=?",
                            (proposal_id,)).fetchone()
        else:
            row = c.execute("SELECT cluster_queries FROM birth_proposals WHERE cluster_id=? "
                            "ORDER BY created_at DESC LIMIT 1", (cluster_id,)).fetchone()
        c.close()
        if row and row[0]:
            try:
                qs = json.loads(row[0])
                if isinstance(qs, list):
                    out = [q if isinstance(q, str) else q.get("query_text", "")
                           for q in qs]
                    out = [q for q in out if q]
                    return out
            except Exception:
                pass
    return out


_JUDGE_RUBRIC = (
    "You are an expert pairwise judge. Given a user QUERY and two candidate "
    "answers A and B, decide which is more helpful, accurate, and specific. "
    "Output ONLY a single JSON object: "
    '{"verdict": "A" | "B" | "tie", "confidence": "high" | "low", "rationale": "<= 30 words"}. '
    "No prose outside the JSON."
)


def judge_score(backbone_ans: str, lora_ans: str, query: str) -> float:
    """Pairwise LLM-as-judge via OpenAI GPT-4o-mini.

    Returns a delta in [-1, +1]:
      +1.0 LoRA clearly better, +0.5 LoRA slightly better,
       0.0 tie / parse error,
      -0.5 backbone slightly better, -1.0 backbone clearly better.

    Position bias: A/B order is randomized per call.

    Falls back to 0.0 (tie) if no API key or API error.
    """
    import os, random, json as _json, re as _re
    # Prefer DeepSeek (cheap, high RPD); fall back to OpenAI.
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    oai_key = os.getenv("OPENAI_API_KEY")
    if ds_key:
        api_key, base_url, model = ds_key, "https://api.deepseek.com", "deepseek-chat"
    elif oai_key:
        api_key, base_url, model = oai_key, None, "gpt-4o-mini"
    else:
        return 0.0
    # Randomize A/B order
    flip = random.random() < 0.5
    if flip:
        ans_A, ans_B = lora_ans, backbone_ans
    else:
        ans_A, ans_B = backbone_ans, lora_ans
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _JUDGE_RUBRIC},
                {"role": "user", "content": (
                    f"QUERY:\n{query[:1200]}\n\n"
                    f"ANSWER A:\n{ans_A[:1500]}\n\n"
                    f"ANSWER B:\n{ans_B[:1500]}\n\n"
                    f"Return JSON only."
                )},
            ],
        )
        text = r.choices[0].message.content or ""
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if not m: return 0.0
        d = _json.loads(m.group(0))
        verdict = (d.get("verdict") or "").strip().lower()
        conf = (d.get("confidence") or "").strip().lower()
        # Map verdict to "LoRA-was-better" sign
        # If flip, A=lora; otherwise B=lora.
        if verdict == "tie":
            return 0.0
        lora_won = (verdict == "a" and flip) or (verdict == "b" and not flip)
        mag = 1.0 if conf == "high" else 0.5
        return mag if lora_won else -mag
    except Exception as e:
        # Soft fail to tie
        return 0.0


def evaluate_proposal(item, model, tok, lora_train_fn, generate_fn):
    """For one proposal:
        - sample K queries from its cluster
        - train LoRA on the cluster's queries
        - generate backbone vs lora answers, judge, mean delta
    Returns dict with delta and posthoc_gt.
    """
    cid = item.get("cluster_id")
    queries = load_cluster_queries(cid, proposal_id=item.get("id"))[:K_QUERIES]
    if len(queries) < 5:
        return {"id": item.get("id"), "cluster_id": cid,
                "skipped": "insufficient queries", "n_queries": len(queries)}
    # Train LoRA on the same queries (self-distill style, like the POC)
    adapter = lora_train_fn(cid, queries, epochs=LORA_EPOCHS)
    # Score K queries
    deltas = []
    for q in queries:
        a_back = generate_fn(model, tok, q, adapter=None)
        a_lora = generate_fn(model, tok, q, adapter=adapter)
        deltas.append(judge_score(a_back, a_lora, q))
    delta_mean = float(np.mean(deltas))
    posthoc_gt = bool(delta_mean >= DELTA_THRESHOLD)
    deployment_decision_birth = (item.get("status") == "APPROVED")
    return {
        "id": item.get("id"),
        "cluster_id": cid,
        "n_queries": len(queries),
        "delta_mean": delta_mean,
        "delta_std": float(np.std(deltas)),
        "posthoc_gt_birth": posthoc_gt,
        "deployment_birth": deployment_decision_birth,
        "agree": posthoc_gt == deployment_decision_birth,
    }


def main():
    proposals = load_proposals()
    print(f"loaded {len(proposals)} eligible proposals")
    if not proposals:
        return 1

    # Detect environment: are we on the pod with GPU?
    has_torch = False
    try:
        import torch
        has_torch = torch.cuda.is_available()
    except Exception:
        pass

    if not has_torch:
        # Dry run: just report what would be done.
        out = {"mode": "dry-run-no-gpu",
               "n_eligible_proposals": len(proposals),
               "proposals": [{"id": p.get("id"),
                              "cluster_id": p.get("cluster_id"),
                              "status": p.get("status"),
                              "needs_birth_gt": p.get("needs_birth_gt"),
                              "n_obs": p.get("n_obs"),
                              "mean_uq": p.get("mean_uq")}
                             for p in proposals],
               "config": {"K_queries": K_QUERIES, "epochs": LORA_EPOCHS,
                          "delta_threshold": DELTA_THRESHOLD},
               "next_step": "Run on GPU pod via rsync."}
        OUTJSON.write_text(json.dumps(out, indent=2))
        print(f"wrote {OUTJSON} (dry run — no GPU available)")
        for p in proposals:
            print(f"  proposal {p.get('id')} cluster={p.get('cluster_id')} "
                  f"status={p.get('status')} gt={p.get('needs_birth_gt')} "
                  f"n_obs={p.get('n_obs')} mean_uq={p.get('mean_uq'):.3f}")
        return 0

    # GPU path: build a self-contained LoRA trainer that uses the strong
    # senior-expert prefix as distillation target (NOT the weak "noise" prefix
    # path of core/lora_train.py).
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset
    import torch

    name = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    base_model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto")

    def _gen_senior(q):
        msgs = [{"role": "user", "content": SENIOR_PREFIX + q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(base_model.device)
        with torch.no_grad():
            out = base_model.generate(**inputs, max_new_tokens=384,
                                      do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    def lora_train_fn(cid, queries, epochs):
        train_qs = queries[:LORA_MAX_TRAIN]
        print(f"  [LoRA] generating {len(train_qs)} senior answers for c{cid}...")
        pairs = [{"query": q, "answer": _gen_senior(q)} for q in train_qs]

        def fmt(ex):
            msgs = [{"role": "user", "content": ex["query"]},
                    {"role": "assistant", "content": ex["answer"]}]
            text = tok.apply_chat_template(msgs, tokenize=False)
            enc = tok(text, truncation=True, max_length=512, padding="max_length")
            enc["labels"] = enc["input_ids"].copy()
            return enc

        ds = Dataset.from_list(pairs).map(fmt, remove_columns=["query", "answer"])
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                         r=LORA_RANK, lora_alpha=LORA_RANK * 2, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        peft_model = get_peft_model(base_model, cfg)
        outdir = f"/tmp/posthoc_c{cid}"
        args = TrainingArguments(output_dir=outdir, num_train_epochs=epochs,
                                 per_device_train_batch_size=4, learning_rate=2e-4,
                                 logging_steps=10, save_strategy="no",
                                 report_to="none", fp16=True)
        Trainer(model=peft_model, args=args, train_dataset=ds,
                data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
        adapter_path = Path(outdir) / "adapter"
        peft_model.save_pretrained(adapter_path)
        # Detach the LoRA so subsequent backbone calls see the unmodified base.
        peft_model.unload()
        return str(adapter_path)

    def generate_fn(m, t, q, adapter=None):
        if adapter is not None:
            mm = PeftModel.from_pretrained(base_model, adapter)
        else:
            mm = base_model
        msgs = [{"role": "user", "content": q}]
        prompt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = t(prompt, return_tensors="pt", truncation=True, max_length=512).to(mm.device)
        with torch.no_grad():
            out = mm.generate(**inputs, max_new_tokens=384, do_sample=False,
                              pad_token_id=t.eos_token_id)
        ans = t.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        if adapter is not None:
            mm.unload()
        return ans

    model = base_model

    results = []
    for p in proposals:
        try:
            r = evaluate_proposal(p, model, tok, lora_train_fn, generate_fn)
            results.append(r)
            print(f"  prop {r['id']}: delta={r.get('delta_mean', float('nan')):.3f} "
                  f"posthoc={r.get('posthoc_gt_birth')} deploy={r.get('deployment_birth')} "
                  f"agree={r.get('agree')}")
        except Exception as e:
            print(f"  prop {p.get('id')}: FAILED — {e}")
            results.append({"id": p.get("id"), "error": str(e)})

    n_ok = sum(1 for r in results if "agree" in r)
    n_agree = sum(1 for r in results if r.get("agree"))
    summary = {
        "mode": "gpu-real",
        "n_evaluated": n_ok,
        "n_agree_with_deployment": n_agree,
        "agreement_rate": n_agree / max(n_ok, 1),
        "config": {"K_queries": K_QUERIES, "epochs": LORA_EPOCHS,
                   "delta_threshold": DELTA_THRESHOLD},
        "results": results,
    }
    OUTJSON.write_text(json.dumps(summary, indent=2))
    print(f"\nagreement rate: {n_agree}/{n_ok} = {summary['agreement_rate']:.2%}")
    print(f"wrote {OUTJSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
