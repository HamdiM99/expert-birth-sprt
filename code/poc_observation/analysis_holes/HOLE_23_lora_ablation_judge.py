"""Heuristic judge for LoRA-ablation pairs (no API key required).

For each (proposal, config) cell, the per-query delta is
   delta_q = ROUGE-L(lora_ans, senior_teacher) - ROUGE-L(backbone_ans, senior_teacher)
This measures how much closer the LoRA-adapted answer is to the senior-prefix
backbone answer (the LoRA distillation target). Mean delta across the K judge
queries; declare ablation BIRTH if mean delta >= delta_threshold.

Compared against the (r=32, ep=10) baseline DeepSeek-judged decisions in
analysis/data/posthoc_counterfactual.json.
"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

PAIRS = Path(sys.argv[1] if len(sys.argv) > 1
             else "../../../data/lora_ablation_pairs.jsonl")
OUT   = Path("../../../results/HOLE_23_lora_ablation.json")
DELTA_THRESHOLD = 0.05    # 5pp absolute ROUGE-L lift, calibrated below


def lcs_len(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0: return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i-1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j-1]:
                dp[j] = prev + 1
            elif dp[j-1] > dp[j]:
                dp[j] = dp[j-1]
            prev = tmp
    return dp[m]


def rouge_l(hyp, ref):
    """ROUGE-L F1 on whitespace tokens."""
    h = hyp.split(); r = ref.split()
    if not h or not r: return 0.0
    L = lcs_len(h, r)
    if L == 0: return 0.0
    p = L / len(h); rec = L / len(r)
    return 2 * p * rec / (p + rec)


def main():
    rows = [json.loads(l) for l in PAIRS.open()]
    print(f"loaded {len(rows)} pairs from {PAIRS}", flush=True)
    if not rows: return 1

    by_cell = defaultdict(list)
    n_no_senior = 0
    for r in rows:
        if not r.get("senior"):
            n_no_senior += 1; continue
        cfg = r["config"]
        by_cell[(r["proposal_id"], cfg["r"], cfg["ep"])].append(
            (r["query"], r["backbone"], r["lora"], r["senior"]))
    if n_no_senior:
        print(f"  WARN {n_no_senior} pairs missing senior teacher answer (skipped)")

    cells = sorted(by_cell.keys())
    print(f"{len(cells)} cells; computing ROUGE-L deltas...", flush=True)

    results = {}
    t0 = time.time()
    for (pid, r_, ep) in cells:
        deltas = []
        for q, bb, lora, sen in by_cell[(pid, r_, ep)]:
            d = rouge_l(lora, sen) - rouge_l(bb, sen)
            deltas.append(d)
        m = float(np.mean(deltas)) if deltas else 0.0
        s = float(np.std(deltas)) if deltas else 0.0
        results[f"p{pid}_r{r_}_e{ep}"] = {
            "proposal_id": pid, "rank": r_, "epochs": ep,
            "n": len(deltas), "delta_mean": m, "delta_std": s,
            "ablation_birth": bool(m >= DELTA_THRESHOLD),
        }

    # Baseline (r=32, ep=10) — already DeepSeek-judged
    baseline = json.loads(Path(
        "../../../data/"
        "posthoc_counterfactual.json").read_text())
    base_dec = {p["id"]: bool(p["posthoc_decision"])
                for p in baseline["proposals"]}

    cmp_rows = []
    for cell, rdat in results.items():
        b = base_dec.get(rdat["proposal_id"])
        cmp_rows.append({"cell": cell,
                         "proposal_id": rdat["proposal_id"],
                         "config": {"r": rdat["rank"], "ep": rdat["epochs"]},
                         "delta_mean": rdat["delta_mean"],
                         "delta_std": rdat["delta_std"],
                         "ablation_birth": rdat["ablation_birth"],
                         "baseline_birth": b,
                         "agree": b is not None
                                  and rdat["ablation_birth"] == b})

    n_agree = sum(1 for c in cmp_rows if c["agree"])
    n_total = len(cmp_rows)
    out = {
        "experiment": "HOLE_23_lora_capacity_ablation",
        "judge": "ROUGE-L vs senior-teacher (heuristic, no LLM-as-judge required)",
        "delta_threshold": DELTA_THRESHOLD,
        "n_cells": n_total,
        "n_agree_with_baseline": n_agree,
        "agreement_rate": n_agree / max(n_total, 1),
        "comparison": cmp_rows,
        "by_cell": results,
        "notes": ("Heuristic judge fallback; the baseline (r=32, ep=10) was "
                  "DeepSeek-judged in posthoc_counterfactual.json. The ROUGE-L "
                  "scale is different from the pairwise-verdict scale, so a "
                  "delta_threshold of 0.05 is calibrated as 'noticeable' lift "
                  "in token-level lexical overlap with the teacher target."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")
    print(f"agreement with baseline (r=32, ep=10): "
          f"{n_agree}/{n_total} = {out['agreement_rate']:.0%}\n")
    print(f"{'pid':>3} {'r':>3} {'ep':>3} {'delta':>8} {'std':>6} "
          f"{'abl':>5} {'base':>5} {'agree':>5}")
    print("-"*52)
    for c in sorted(cmp_rows, key=lambda x: (x["proposal_id"],
                                              x["config"]["r"],
                                              x["config"]["ep"])):
        print(f"{c['proposal_id']:>3} {c['config']['r']:>3} {c['config']['ep']:>3} "
              f"{c['delta_mean']:>+8.3f} {c['delta_std']:>6.3f} "
              f"{str(c['ablation_birth']):>5} {str(c['baseline_birth']):>5} "
              f"{str(c['agree']):>5}")


if __name__ == "__main__":
    sys.exit(main())
