"""HOLE 15 — Short-calibration sigma_hat per cluster.

Reviewer Q3 mitigation: instead of fixing sigma=0.20 globally, estimate
sigma_hat per cluster from a short calibration period (first n_cal
observations), then run SPRT with that sigma_hat.

Practical, deployable variant of HOLE_11/HOLE_9: no LOO needed, no
prior — just N_CAL=10 observations to lock sigma, then SPRT with
fixed-sigma_hat for the remaining MIN_OBS..TARGET_OBS window.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("ANALYSIS_DATA", "/workspace/analysis/data"))
OUT = Path(os.environ.get("ANALYSIS_OUT",
    "/workspace/analysis/results/HOLE_15_short_cal_sigma.json"))

MU0, MU1 = 0.45, 0.55
ALPHA, BETA = 0.05, 0.05
MIN_OBS, TARGET_OBS = 20, 50


def sprt_with_sigma(seq, sigma, mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA,
                    min_obs=MIN_OBS, target_obs=TARGET_OBS):
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    used = []
    for k in range(min(target_obs, len(seq))):
        used.append(seq[k])
        n = len(used); s = sum(used)
        if n >= min_obs:
            llr = (mu1 - mu0) / sigma**2 * s - n * (mu1**2 - mu0**2) / (2 * sigma**2)
            if llr >= A: return "BIRTH", n
            if llr <= B: return "SKIP", n
    if not used: return "SKIP", 0
    m = float(np.mean(used))
    return ("BIRTH" if m >= 0.5 * (mu0 + mu1) else "SKIP"), len(used)


def confusion(decisions, gts):
    TP = sum(1 for d, g in zip(decisions, gts) if d == "BIRTH" and g)
    FP = sum(1 for d, g in zip(decisions, gts) if d == "BIRTH" and not g)
    FN = sum(1 for d, g in zip(decisions, gts) if d != "BIRTH" and g)
    TN = sum(1 for d, g in zip(decisions, gts) if d != "BIRTH" and not g)
    P = TP / (TP + FP) if (TP + FP) else float("nan")
    R = TP / (TP + FN) if (TP + FN) else float("nan")
    F1 = 2 * P * R / (P + R) if (P + R) > 0 and not (np.isnan(P) or np.isnan(R)) else float("nan")
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": P, "recall": R, "f1": F1}


def main():
    traces = [json.loads(l) for l in (DATA / "per_proposal_traces.jsonl").open()]
    pc = json.loads((DATA / "posthoc_counterfactual.json").read_text())
    eligible_ids = [p["id"] for p in pc["proposals"]]
    oracle_gt = {p["id"]: p["posthoc_decision"] for p in pc["proposals"]}
    proxy_gt = {p["id"]: p["needs_birth_gt"] for p in pc["proposals"]}
    sprt_dep = {p["id"]: p["deployment_decision"] for p in pc["proposals"]}
    by_id = {t["id"]: t for t in traces if t.get("uq_observations")}
    eligible = [(i, by_id[i]) for i in eligible_ids if i in by_id]

    sweep = []
    for n_cal in [5, 8, 10, 12, 15]:
        decs, ns, sgs = [], [], []
        for pid, t in eligible:
            seq = t["uq_observations"]
            cal = seq[:n_cal]
            sg = float(np.std(cal, ddof=1)) if len(cal) > 1 else 0.20
            sg = max(sg, 0.05)  # floor
            sgs.append(sg)
            d, n = sprt_with_sigma(seq[n_cal:], sigma=sg)
            decs.append(d); ns.append(n + n_cal)
        gts_p = [proxy_gt[i] for i, _ in eligible]
        gts_o = [oracle_gt[i] for i, _ in eligible]
        sweep.append({
            "n_cal": n_cal,
            "proxy_gt": confusion(decs, gts_p),
            "oracle_gt": confusion(decs, gts_o),
            "avg_obs_total": float(np.mean(ns)),
            "n_births": sum(1 for d in decs if d == "BIRTH"),
            "sigma_hat_per_proposal": sgs,
            "sigma_hat_mean": float(np.mean(sgs)),
            "sigma_hat_range": [float(np.min(sgs)), float(np.max(sgs))],
        })
        print(f"n_cal={n_cal}: P/R proxy={confusion(decs, gts_p)['precision']:.2f}/"
              f"{confusion(decs, gts_p)['recall']:.2f}  oracle={confusion(decs, gts_o)['precision']:.2f}/"
              f"{confusion(decs, gts_o)['recall']:.2f}  N={np.mean(ns):.1f}  "
              f"sigma=[{np.min(sgs):.3f}, {np.max(sgs):.3f}]")

    sprt_decs = ["BIRTH" if sprt_dep[i] else "SKIP" for i, _ in eligible]
    out = {
        "experiment": "HOLE_15_short_cal_sigma",
        "method": "SPRT with cluster-specific sigma_hat from first n_cal observations",
        "params": dict(mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA,
                       min_obs=MIN_OBS, target_obs=TARGET_OBS),
        "n_eligible": len(eligible),
        "n_cal_sweep": sweep,
        "sprt_baseline": {
            "proxy_gt": confusion(sprt_decs, [proxy_gt[i] for i, _ in eligible]),
            "oracle_gt": confusion(sprt_decs, [oracle_gt[i] for i, _ in eligible]),
            "n_births": sum(1 for d in sprt_decs if d == "BIRTH"),
        },
        "headline": (
            f"Short-calibration SPRT (n_cal in {{5,...,15}}, then SPRT with "
            f"per-cluster sigma_hat) on {len(eligible)} proposals: see n_cal_sweep. "
            "Practical, deployable, no LOO."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
