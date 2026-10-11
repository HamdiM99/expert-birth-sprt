"""HOLE 26 (pod side, GPU) — MASS-style gradient-surge as an alternative
birth signal.

Hypothesis: a cluster needs an expert when, on a small LoRA fine-tune,
the gradient norm on cluster-queries decays slowly relative to a control
mix --- the model "wants to learn" something. We probe each of the 7
deployment-eligible proposals with a 5-epoch rank-8 LoRA fine-tune and
record per-step gradient norms; the surge metric is the ratio of mean
grad-norm in the first window vs the last window.

Inputs:
  data/multi_uq_observations.jsonl  (queries grouped by proposal_id)

Outputs:
  results/HOLE_26_mass_gradient_surge.json

Run on the pod:
    cd /workspace/poc_observation/analysis_holes
    python HOLE_26_mass_gradient_surge_pod.py \
        --queries ../../../data/multi_uq_observations.jsonl \
        --out ../../../results/HOLE_26_mass_gradient_surge.json
"""
from __future__ import annotations
import argparse, json, time, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN    = 256
LR         = 3e-4
EPOCHS     = 5
BATCH      = 4
LORA_R     = 8
LORA_ALPHA = 16
TARGET_MOD = ["q_proj", "v_proj"]


class QueryDataset(Dataset):
    def __init__(self, queries, tokenizer):
        self.tok = tokenizer
        self.q = queries

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        text = self.q[i]
        enc = self.tok(text, truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        ids  = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}


def grad_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum().item())
    return total_sq ** 0.5


def train_proposal(pid, queries, tokenizer, device):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device)
    cfg  = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                      target_modules=TARGET_MOD, task_type="CAUSAL_LM",
                      bias="none")
    model = get_peft_model(base, cfg).train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)

    ds  = QueryDataset(queries, tokenizer)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    grads, losses = [], []
    step = 0
    for ep in range(EPOCHS):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            gn = grad_norm(model)
            grads.append(gn)
            losses.append(float(loss.item()))
            optim.step()
            optim.zero_grad()
            step += 1
    del model, base
    torch.cuda.empty_cache()

    g = np.array(grads); l = np.array(losses)
    n = len(g); k = max(1, n // 5)
    head_mean = float(g[:k].mean())
    tail_mean = float(g[-k:].mean())
    surge = head_mean / max(tail_mean, 1e-6)
    head_loss = float(l[:k].mean()); tail_loss = float(l[-k:].mean())
    drop_ratio = (head_loss - tail_loss) / max(head_loss, 1e-6)
    return {
        "proposal_id":   pid,
        "n_steps":       n,
        "grad_head_mean": head_mean,
        "grad_tail_mean": tail_mean,
        "grad_surge":    surge,
        "loss_head":     head_loss,
        "loss_tail":     tail_loss,
        "loss_drop":     drop_ratio,
        "grad_trace":    grads,
        "loss_trace":    losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    by_p = defaultdict(list)
    for line in open(args.queries):
        r = json.loads(line)
        by_p[r["proposal_id"]].append(r["query_text"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = {"config": {
        "model": MODEL_NAME, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "target_modules": TARGET_MOD},
           "proposals": []}
    t0 = time.time()
    for pid in sorted(by_p):
        print(f"=== Proposal {pid} ({len(by_p[pid])} queries) ===")
        try:
            r = train_proposal(pid, by_p[pid], tokenizer, device)
            print(f"  surge={r['grad_surge']:.3f} loss_drop={r['loss_drop']:.3f}")
            out["proposals"].append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            out["proposals"].append({"proposal_id": pid, "error": str(e)})
    out["wall_time_sec"] = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}  ({out['wall_time_sec']:.1f}s total)")


if __name__ == "__main__":
    main()
