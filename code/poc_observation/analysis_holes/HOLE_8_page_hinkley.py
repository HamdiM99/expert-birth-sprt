"""HOLE 8 — Page-Hinkley (PH) baseline against SPRT.

Per-proposal UQ sequences u_t are scanned with the standard PH detector:

    g_0 = 0
    g_t = max(0, g_{t-1} + (u_t - mu_ref - delta))
    BIRTH when g_t > lambda

Tune (mu_ref, delta, lambda) by grid; report the operating point closest
to SPRT's (P=1.0, R=0.6) plus the PH best-F1 point. Compare with batched-probe.

Ground truth here = posthoc oracle (HOLE_7) — that's the honest comparison
the reviewer wanted. We also report results under the proxy GT used in the
deployment for comparability with the paper's existing Tab-2.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # supplementary/
DATA = Path(os.environ.get("ANALYSIS_DATA", _ROOT / "data"))
OUT = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_8_page_hinkley.json"))


def ph_detect(seq, mu_ref, delta, lam, n_min=10, n_max=50):
    """Returns (decision, n_obs_used). decision=True ↔ BIRTH triggered."""
    g = 0.0
    for t, u in enumerate(seq[:n_max], start=1):
        g = max(0.0, g + (u - mu_ref - delta))
        if t >= n_min and g > lam:
            return True, t
    return False, min(len(seq), n_max)


def confusion(decisions, gts):
    TP = sum(1 for d,g in zip(decisions,gts) if d and g)
    FP = sum(1 for d,g in zip(decisions,gts) if d and not g)
    FN = sum(1 for d,g in zip(decisions,gts) if not d and g)
    TN = sum(1 for d,g in zip(decisions,gts) if not d and not g)
    P = TP / (TP+FP) if (TP+FP)>0 else float("nan")
    R = TP / (TP+FN) if (TP+FN)>0 else float("nan")
    F1 = 2*P*R/(P+R) if (P+R)>0 and not (np.isnan(P) or np.isnan(R)) else float("nan")
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"precision":P,"recall":R,"f1":F1}


def main():
    traces = [json.loads(l) for l in (DATA / "per_proposal_traces.jsonl").open()]
    pc = json.loads((DATA / "posthoc_counterfactual.json").read_text())
    eligible_ids = [p["id"] for p in pc["proposals"]]
    oracle_gt = {p["id"]: p["posthoc_decision"] for p in pc["proposals"]}
    proxy_gt  = {p["id"]: p["needs_birth_gt"]   for p in pc["proposals"]}

    # Restrict to the 7 eligible proposals (same as Tab-2 SPRT row)
    by_id = {t["id"]: t for t in traces if t.get("uq_observations")}
    eligible = [by_id[i] for i in eligible_ids if i in by_id]
    gts_oracle = [oracle_gt[t["id"]] for t in eligible]
    gts_proxy  = [proxy_gt[t["id"]]  for t in eligible]

    all_uq = np.concatenate([np.array(t["uq_observations"]) for t in eligible])
    grand_mean = float(all_uq.mean())
    grand_std  = float(all_uq.std())

    # Grid for PH parameters
    mu_refs = [np.percentile(all_uq, p) for p in (30, 40, 50)]
    deltas  = [0.0, 0.05, 0.10, 0.15, 0.20]
    lams    = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

    grid = []
    for mu_ref in mu_refs:
        for delta in deltas:
            for lam in lams:
                decisions, n_obs = [], []
                for t in eligible:
                    d, n = ph_detect(t["uq_observations"], mu_ref, delta, lam)
                    decisions.append(d); n_obs.append(n)
                ms_o = confusion(decisions, gts_oracle)
                ms_p = confusion(decisions, gts_proxy)
                grid.append({
                    "mu_ref": float(mu_ref), "delta": float(delta), "lam": float(lam),
                    "n_births": int(sum(decisions)),
                    "avg_obs": float(np.mean(n_obs)),
                    "metrics_oracle": ms_o,
                    "metrics_proxy": ms_p,
                })

    # Closest to SPRT operating point on PROXY GT (P=1.0, R=0.6)
    target_p, target_r = 1.0, 0.6
    def dist(m):
        if np.isnan(m["precision"]) or np.isnan(m["recall"]):
            return 999
        return (m["precision"]-target_p)**2 + (m["recall"]-target_r)**2

    closest_proxy = min(grid, key=lambda c: dist(c["metrics_proxy"]))
    closest_oracle = min(grid, key=lambda c: dist(c["metrics_oracle"]))

    def best_f1(grid, key):
        best = None; best_f1v = -1
        for c in grid:
            f = c[key]["f1"]
            if not np.isnan(f) and f > best_f1v:
                best_f1v = f; best = c
        return best
    best_f1_proxy  = best_f1(grid, "metrics_proxy")
    best_f1_oracle = best_f1(grid, "metrics_oracle")

    # SPRT reference numbers — taken from posthoc + sprt_diagnostics
    sprt_decisions = [p["deployment_decision"] for p in pc["proposals"]]
    sprt_proxy = confusion(sprt_decisions, gts_proxy)
    sprt_oracle = confusion(sprt_decisions, gts_oracle)
    sprt_avg_obs = float(np.mean([t.get("n_obs", len(t["uq_observations"])) for t in eligible]))

    # Batched-probe baseline numbers (read off existing batched_probe_results
    # if we have them locally — they're not in /workspace/analysis/data, so we
    # cite the paper-reported figure here as a placeholder and note it.)

    summary = {
        "comparison_table": {
            "SPRT": {
                "proxy_gt": sprt_proxy, "oracle_gt": sprt_oracle, "avg_obs": sprt_avg_obs,
                "n_births": sum(sprt_decisions),
            },
            "PageHinkley_closest_to_SPRT_proxy": {
                "params": {k: closest_proxy[k] for k in ("mu_ref","delta","lam")},
                "proxy_gt": closest_proxy["metrics_proxy"],
                "oracle_gt": closest_proxy["metrics_oracle"],
                "avg_obs": closest_proxy["avg_obs"],
                "n_births": closest_proxy["n_births"],
            },
            "PageHinkley_best_F1_proxy": {
                "params": {k: best_f1_proxy[k] for k in ("mu_ref","delta","lam")},
                "proxy_gt": best_f1_proxy["metrics_proxy"],
                "oracle_gt": best_f1_proxy["metrics_oracle"],
                "avg_obs": best_f1_proxy["avg_obs"],
                "n_births": best_f1_proxy["n_births"],
            },
            "PageHinkley_best_F1_oracle": {
                "params": {k: best_f1_oracle[k] for k in ("mu_ref","delta","lam")},
                "proxy_gt": best_f1_oracle["metrics_proxy"],
                "oracle_gt": best_f1_oracle["metrics_oracle"],
                "avg_obs": best_f1_oracle["avg_obs"],
                "n_births": best_f1_oracle["n_births"],
            },
        },
        "n_eligible_proposals": len(eligible),
        "uq_grand_mean": grand_mean,
        "uq_grand_std": grand_std,
        "grid_size": len(grid),
        "grid_full": grid,
        "notes": (
            "Page-Hinkley grid (mu_ref ∈ p30/p40/p50, delta ∈ 0..0.2, lambda ∈ 0.5..12) "
            "tuned on the same 7 eligible proposals as SPRT. The 'closest_to_SPRT_proxy' "
            "row is the apples-to-apples non-sequential alternative; PH's best-F1 row "
            "shows what PH could achieve with full grid search (potentially overfitting "
            "to N=7). Both are reported; the conservative comparison is closest-to-SPRT."
        ),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print("SPRT proxy:", sprt_proxy)
    print("SPRT oracle:", sprt_oracle)
    print("PH closest-to-SPRT proxy:", closest_proxy["metrics_proxy"])
    print("PH best-F1 oracle:", best_f1_oracle["metrics_oracle"], "params:", {k: best_f1_oracle[k] for k in ("mu_ref","delta","lam")})
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
