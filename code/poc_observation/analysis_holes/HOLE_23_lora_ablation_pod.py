"""LoRA-capacity ablation on the pod.

For each (rank, epochs) cell × each of 7 deployment proposals:
  1. Pull cluster_queries from sqlite (proposal.cluster_queries)
  2. Train LoRA at (r, ep) using senior-prefix self-distillation on first
     LORA_MAX_TRAIN queries
  3. Generate 20 (backbone, lora) answer pairs on a held-out judge slice
  4. Stream rows of {proposal_id, config, query, backbone, lora} to JSONL

Judging happens off-pod via Anthropic API.
"""
from __future__ import annotations
import json, sqlite3, time, traceback
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
DB        = "./pod_workdir/data/anon.db"
OUT       = Path("./pod_workdir/exports/lora_ablation_pairs.jsonl")
ELIGIBLE  = [2, 3, 4, 5, 6, 7, 8]
CONFIGS   = [{"r": 8, "ep": 2}, {"r": 16, "ep": 4}]   # +baseline already done
LORA_MAX_TRAIN = 30      # queries used for LoRA self-distill (was 50)
K_JUDGE        = 20      # queries used for backbone-vs-lora judging (was 30)
MAX_NEW_TEACH  = 256
MAX_NEW_JUDGE  = 256

SENIOR_PREFIX = (
    "You are a senior domain expert. Provide a precise, technical, "
    "evidence-based answer. Cite the specific reasoning, mechanism, or rule "
    "that applies. Avoid generic language; be concrete and specific.\n\n"
)


def setup():
    print("loading tokenizer + 4bit backbone...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                quantization_config=bnb,
                                                device_map="auto")
    base.eval()
    return tok, base


@torch.no_grad()
def gen(model, tok, q, max_new):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=512).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                      skip_special_tokens=True).strip()


@torch.no_grad()
def gen_senior(model, tok, q):
    return gen(model, tok, SENIOR_PREFIX + q, MAX_NEW_TEACH)


def load_queries(pid):
    c = sqlite3.connect(DB, timeout=30)
    row = c.execute("SELECT cluster_queries FROM birth_proposals WHERE id=?", (pid,)).fetchone()
    c.close()
    if not row or not row[0]: return []
    try: arr = json.loads(row[0])
    except Exception: return []
    out = []
    for q in arr:
        if isinstance(q, str): out.append(q)
        elif isinstance(q, dict):
            t = q.get("query_text") or q.get("query") or q.get("text")
            if t: out.append(t)
    return out


def train_lora(base, tok, queries, r, ep, cid, senior_cache):
    pairs = []
    for q in queries[:LORA_MAX_TRAIN]:
        if q not in senior_cache:
            senior_cache[q] = gen_senior(base, tok, q)
        pairs.append({"query": q, "answer": senior_cache[q]})
    def fmt(ex):
        msgs = [{"role": "user", "content": ex["query"]},
                {"role": "assistant", "content": ex["answer"]}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(text, truncation=True, max_length=512, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc
    ds = Dataset.from_list(pairs).map(fmt, remove_columns=["query","answer"])
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=r*2,
                     lora_dropout=0.05,
                     target_modules=["q_proj","k_proj","v_proj","o_proj"])
    pm = get_peft_model(base, cfg)
    outdir = f"/tmp/abl_p{cid}_r{r}_e{ep}"
    args = TrainingArguments(output_dir=outdir, num_train_epochs=ep,
                             per_device_train_batch_size=2,
                             gradient_accumulation_steps=2,
                             learning_rate=2e-4, logging_steps=20,
                             save_strategy="no", report_to="none",
                             fp16=False, bf16=True)
    Trainer(model=pm, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    adapter = Path(outdir) / "adapter"
    pm.save_pretrained(adapter)
    pm.unload()
    return str(adapter)


def main():
    tok, base = setup()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f = OUT.open("a")
    t0 = time.time()
    n_total = 0
    for pid in ELIGIBLE:
        qs = load_queries(pid)
        print(f"[pid={pid}] {len(qs)} cluster queries", flush=True)
        if len(qs) < 5:
            print(f"  skip — too few queries"); continue
        train_qs = qs[:LORA_MAX_TRAIN]
        judge_qs = qs[:K_JUDGE]   # overlap is fine — same setup as baseline

        # backbone answers cached once per pid (independent of (r, ep))
        bb_ans = [gen(base, tok, q, MAX_NEW_JUDGE) for q in judge_qs]
        senior_cache = {}     # query -> senior teacher answer (heuristic-judge target)

        for cfg in CONFIGS:
            r, ep = cfg["r"], cfg["ep"]
            try:
                print(f"  config r={r} ep={ep} — training", flush=True)
                adapter = train_lora(base, tok, train_qs, r, ep, pid, senior_cache)
                print(f"  config r={r} ep={ep} — generating", flush=True)
                m = PeftModel.from_pretrained(base, adapter)
                m.eval()
                lora_ans = [gen(m, tok, q, MAX_NEW_JUDGE) for q in judge_qs]
                m.unload()
                for q, a_b, a_l in zip(judge_qs, bb_ans, lora_ans):
                    f.write(json.dumps({"proposal_id": pid,
                                        "config": cfg,
                                        "query": q,
                                        "backbone": a_b,
                                        "lora":     a_l,
                                        "senior":   senior_cache.get(q, "")}) + "\n")
                    f.flush(); n_total += 1
                print(f"  pid={pid} r={r} ep={ep} done — total rows={n_total} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  ERR pid={pid} r={r} ep={ep}: {e}", flush=True)
                traceback.print_exc()
    f.close()
    print(f"DONE total={n_total} in {time.time()-t0:.0f}s -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
