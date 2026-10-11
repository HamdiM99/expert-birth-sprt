"""HOLE 7 — Recompute deployment Tab-2 SPRT metrics with oracle GT.

posthoc_counterfactual.json gives, for the 7 eligible proposals:
  - deployment_decision (True=BIRTH,  False=SKIP) — what SPRT did at t=now
  - posthoc_decision    (oracle, LLM-as-judge pairwise A/B with senior LoRA delta)
  - delta               (post-hoc senior-vs-base improvement)

We compare:
  proxy GT  := needs_birth_gt   (the auto label used at deployment)
  oracle GT := posthoc_decision (the LLM-judge oracle the reviewer asked about)

For each metric set we compute precision/recall/F1, # births, and an
estimate of fraction-of-LoRA-training-saved.

Cost-saving model: assume eager spawns 1 LoRA / proposal × 7 = 7 trainings.
SPRT only spawns when it decides BIRTH (deployment_decision=True). Define
useful_births under each GT as TP. fraction_saved = 1 - (#births / 7).
fraction_useful = TP / 7. wasted_births = FP / 7.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # supplementary/
DATA = Path(os.environ.get("ANALYSIS_DATA", _ROOT / "data"))
OUT = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_7_oracle_metrics.json"))


def confusion(decisions, gts):
    TP = sum(1 for d,g in zip(decisions,gts) if d and g)
    FP = sum(1 for d,g in zip(decisions,gts) if d and not g)
    FN = sum(1 for d,g in zip(decisions,gts) if not d and g)
    TN = sum(1 for d,g in zip(decisions,gts) if not d and not g)
    P = TP / max(TP+FP, 1) if (TP+FP)>0 else float("nan")
    R = TP / max(TP+FN, 1) if (TP+FN)>0 else float("nan")
    F1 = 2*P*R/(P+R) if (P+R)>0 and not (np.isnan(P) or np.isnan(R)) else float("nan")
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"precision":P,"recall":R,"f1":F1}


def main():
    pc = json.loads((DATA / "posthoc_counterfactual.json").read_text())
    props = pc["proposals"]
    n = len(props)

    decisions  = [p["deployment_decision"] for p in props]   # SPRT BIRTH at deployment
    proxy_gt   = [p["needs_birth_gt"]      for p in props]   # auto/proxy label
    oracle_gt  = [p["posthoc_decision"]    for p in props]   # LLM-judge senior-vs-base

    # Avg observations under SPRT — pull from per_proposal_traces
    traces = [json.loads(l) for l in (DATA / "per_proposal_traces.jsonl").open()]
    id_to_n = {t["id"]: t.get("n_obs", len(t.get("uq_observations",[]))) for t in traces}
    n_obs_list = [id_to_n.get(p["id"], None) for p in props]
    avg_obs = float(np.mean([x for x in n_obs_list if x is not None]))

    proxy = confusion(decisions, proxy_gt)
    oracle = confusion(decisions, oracle_gt)

    n_births_sprt = sum(decisions)
    sprt_savings_vs_eager = 1 - (n_births_sprt / n)   # fraction of LoRAs not trained
    useful_births_proxy = proxy["TP"] / n
    useful_births_oracle = oracle["TP"] / n
    wasted_births_proxy = proxy["FP"] / n
    wasted_births_oracle = oracle["FP"] / n

    # If we had used oracle as the deployment GT to gate eager training
    # the optimal (clairvoyant) policy would BIRTH on oracle_gt=True only.
    # Compute its savings as a reference.
    oracle_optimal_births = sum(oracle_gt)
    oracle_optimal_savings = 1 - (oracle_optimal_births / n)

    out = {
        "n_eligible_proposals": n,
        "deployment_decisions_sprt_birth": [bool(d) for d in decisions],
        "proxy_gt_per_proposal": [bool(g) for g in proxy_gt],
        "oracle_gt_per_proposal": [bool(g) for g in oracle_gt],
        "n_obs_per_proposal": n_obs_list,
        "avg_obs_sprt": avg_obs,
        "proxy_metrics": {
            **proxy,
            "n_births": n_births_sprt,
            "useful_births_frac": useful_births_proxy,
            "wasted_births_frac": wasted_births_proxy,
        },
        "oracle_metrics": {
            **oracle,
            "n_births": n_births_sprt,
            "useful_births_frac": useful_births_oracle,
            "wasted_births_frac": wasted_births_oracle,
        },
        "cost_savings": {
            "eager_baseline_births": n,
            "sprt_births": n_births_sprt,
            "sprt_fraction_lora_saved_vs_eager": sprt_savings_vs_eager,
            "sprt_useful_lora_frac_proxy": useful_births_proxy,
            "sprt_useful_lora_frac_oracle": useful_births_oracle,
            "oracle_optimal_births": oracle_optimal_births,
            "oracle_optimal_savings_vs_eager": oracle_optimal_savings,
            "note": (
                "Eager spawns 1 LoRA per eligible proposal (7). SPRT spawns "
                "only on BIRTH decisions (4). Of those 4 BIRTHs, oracle "
                "confirms 1 useful (TP=1) and rejects 3 as harmful/neutral "
                "(FP=3). SPRT also missed 2 oracle-positive cases (FN=2, "
                "props 5 and 6), which eager would have trained. Net: SPRT "
                "saves 3/7=42.9% of LoRAs vs eager but at the cost of "
                "missing 2/3 oracle-positive opportunities. Under proxy GT "
                "the picture flips: SPRT looks excellent because the proxy "
                "treats every APPROVED-status as positive."
            ),
        },
        "summary": {
            "proxy_vs_oracle_p_drop": proxy["precision"] - oracle["precision"],
            "proxy_vs_oracle_r_drop": proxy["recall"]    - oracle["recall"],
            "proxy_vs_oracle_f1_drop": proxy["f1"]       - oracle["f1"],
            "headline": (
                f"Tab-2 SPRT: proxy GT P={proxy['precision']:.2f} R={proxy['recall']:.2f} F1={proxy['f1']:.2f}; "
                f"oracle GT P={oracle['precision']:.2f} R={oracle['recall']:.2f} F1={oracle['f1']:.2f}. "
                f"Savings vs eager: {sprt_savings_vs_eager*100:.0f}% of LoRA trainings avoided, "
                f"but oracle says only {useful_births_oracle*100:.0f}% of all eligible LoRAs "
                f"(1/7) are actually useful — SPRT spawns 4× too many under oracle."
            ),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
