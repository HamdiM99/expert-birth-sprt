"""Lever 2 — STANDALONE multi-UQ replay (no proprietary stack required).

For each of the 7 deployment-eligible birth proposals, replay the
observation phase by computing 4 UQ signals directly with transformers
+ sentence-transformers (the same definitions used in the paper):

- token_entropy:       mean per-step Shannon entropy of next-token logits
- semantic_entropy:    cluster-Shannon entropy over k=5 sampled responses
- expert_disagreement: 1 - mean cosine sim across k=3 expert generations
                       at temperatures [0.3, 0.8, 1.3] (using ONE adapter at
                       different temperatures = "expert" approximation;
                       captures generation diversity which is what the paper
                       expert_disagreement actually measures via temp diversity)
- backbone_agreement:  cos(emb(expert_resp), emb(backbone_resp))

implicit_feedback is stubbed at 0 (matches paper convention).

Inputs:  cluster_queries pulled from ./pod_workdir/data/anon.db
Output:  ./pod_workdir/exports/multi_uq_observations.jsonl
"""
from __future__ import annotations
import json, sqlite3, sys, time, traceback, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER    = "./pod_workdir/founders/adapter_a"   # any adapter — we use it as "expert"
DB         = "./pod_workdir/data/anon.db"
OUT        = Path("./pod_workdir/exports/multi_uq_observations.jsonl")
ELIGIBLE   = {2, 3, 4, 5, 6, 7, 8}
MAX_OBS    = 30        # >= MIN_OBS=20 used by SPRT
MAX_NEW    = 48        # short generations for speed
DEVICE     = "cuda"

EXPERT_TEMPS = [0.3, 0.8, 1.3]


def setup():
    print("loading tokenizer…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print("loading 4-bit backbone…", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                device_map="auto")
    base.eval()

    print(f"loading adapter from {ADAPTER}…", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    print("loading sentence-transformer embedder…", flush=True)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                    device=DEVICE)
    return tok, base, model, embedder


def encode_chat(tok, query):
    msgs = [{"role": "user", "content": query}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(DEVICE)


@torch.no_grad()
def gen_one(model, tok, inputs, temperature, max_new=MAX_NEW):
    do_sample = temperature > 0.05
    out = model.generate(**inputs, max_new_tokens=max_new,
                         do_sample=do_sample,
                         temperature=temperature if do_sample else None,
                         pad_token_id=tok.pad_token_id,
                         output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(seq, skip_special_tokens=True).strip()
    # token entropy: mean over generation steps
    if out.scores:
        ents = []
        for logit in out.scores:
            p = F.softmax(logit[0].float(), dim=-1)
            e = -(p * (p + 1e-12).log()).sum().item()
            ents.append(e)
        tok_ent = float(np.mean(ents)) if ents else 0.0
    else:
        tok_ent = 0.0
    return text, tok_ent


@torch.no_grad()
def compute_signals(query, tok, base, model, embedder):
    inputs = encode_chat(tok, query)

    # ── primary expert response (T=0.3, factual) — also gives token_entropy
    expert_resp, tok_ent = gen_one(model, tok, inputs, 0.3)

    # ── backbone response with adapter disabled
    with model.disable_adapter():
        bb_resp, _ = gen_one(model, tok, inputs, 0.3)

    # ── expert_disagreement: K=3 generations at diverse temps (proxy for K experts)
    resps = [expert_resp]
    for T in EXPERT_TEMPS[1:]:
        r, _ = gen_one(model, tok, inputs, T)
        resps.append(r)

    # ── semantic_entropy: K=5 samples at T=1.0
    sem_resps = []
    for _ in range(5):
        r, _ = gen_one(model, tok, inputs, 1.0, max_new=32)
        sem_resps.append(r)

    # ── embeddings (batch)
    all_resps = [expert_resp, bb_resp] + resps[1:] + sem_resps
    embs = embedder.encode(all_resps, normalize_embeddings=True)
    emb_expert = embs[0]; emb_bb = embs[1]
    emb_disag  = np.vstack([embs[0:1], embs[2:2+len(EXPERT_TEMPS)-1]])  # shape (3, d)
    emb_sem    = embs[2+len(EXPERT_TEMPS)-1:]                            # shape (5, d)

    bb_agree = float(np.dot(emb_expert, emb_bb))

    # disagreement
    K = emb_disag.shape[0]
    sims = [float(np.dot(emb_disag[i], emb_disag[j]))
            for i in range(K) for j in range(i+1, K)]
    disag = max(0.0, min(1.0, 1.0 - (float(np.mean(sims)) if sims else 1.0)))

    # semantic entropy: cluster sims >= 0.85
    K = emb_sem.shape[0]
    assigned = [False]*K; clusters = []
    for i in range(K):
        if assigned[i]: continue
        c = [i]; assigned[i] = True
        for j in range(i+1, K):
            if not assigned[j] and float(np.dot(emb_sem[i], emb_sem[j])) >= 0.85:
                c.append(j); assigned[j] = True
        clusters.append(c)
    probs = np.array([len(c)/K for c in clusters])
    sem_ent = float(-np.sum(probs * np.log(probs + 1e-9)))

    return {
        "token_entropy":       float(tok_ent),
        "semantic_entropy":    float(sem_ent),
        "expert_disagreement": float(disag),
        "backbone_agreement":  float(bb_agree),
        "implicit_feedback":   0.0,
    }


def main():
    tok, base, model, embedder = setup()
    print("warmup…", flush=True)
    _ = compute_signals("hello", tok, base, model, embedder)
    print("warmup OK", flush=True)

    conn = sqlite3.connect(DB, timeout=30)
    rows = conn.execute(
        "SELECT id, cluster_queries FROM birth_proposals WHERE id IN ("
        + ",".join("?"*len(ELIGIBLE)) + ")", tuple(ELIGIBLE)).fetchall()
    proposals = {}
    for pid, cq in rows:
        try: cq = json.loads(cq) if isinstance(cq,str) else cq
        except Exception: cq = []
        proposals[pid] = list(cq)[:MAX_OBS]
    print(f"proposals: {[(k,len(v)) for k,v in proposals.items()]}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    f = OUT.open("w")
    n_total = 0; t0 = time.time()
    for pid in sorted(proposals.keys()):
        for j, qtext in enumerate(proposals[pid]):
            try:
                sig = compute_signals(qtext, tok, base, model, embedder)
                row = {"proposal_id": pid, "obs_idx": j,
                       "query_text": qtext, "signals": sig}
                f.write(json.dumps(row) + "\n"); f.flush()
                n_total += 1
                elapsed = time.time() - t0
                print(f"  [{n_total}] pid={pid} j={j} "
                      f"TE={sig['token_entropy']:.3f} "
                      f"SE={sig['semantic_entropy']:.3f} "
                      f"ED={sig['expert_disagreement']:.3f} "
                      f"BA={sig['backbone_agreement']:.3f} "
                      f"({elapsed:.0f}s)", flush=True)
            except Exception as e:
                print(f"  ERR pid={pid} j={j}: {e}", flush=True)
                traceback.print_exc()
    f.close()
    print(f"DONE n={n_total} in {time.time()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
