"""HOLE 11 — GLR-SPRT with unknown variance (Wald-Wolfowitz, t-form).

Reviewer Q2: replace fixed sigma=0.20 with running estimate sigma_hat_t.

For Gaussian X_i with unknown mean mu and unknown variance sigma^2,
testing H0: mu = mu0 vs H1: mu = mu1, the GLR statistic at time n is

    Lambda_n = sup_{sigma} L(mu1, sigma; X) / sup_{sigma} L(mu0, sigma; X)

Plugging in MLE sigma_hat^2(mu) = (1/n) sum (X_i - mu)^2 yields

    log Lambda_n = (n/2) log[ S0_n / S1_n ]
where S_k_n = sum (X_i - mu_k)^2.

Stop when log Lambda_n >= log(1/alpha) (=> H1, BIRTH)
        or log Lambda_n <= log(beta)    (=> H0, SKIP).

Compare to fixed-sigma SPRT on the same traces.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
DATA = Path(os.environ.get("ANALYSIS_DATA", _ROOT / "data"))
OUT = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_11_glr_sprt.json"))

MU0, MU1 = 0.45, 0.55
ALPHA, BETA = 0.05, 0.05
MIN_OBS, TARGET_OBS = 20, 50


def glr_decision(seq, mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA,
                 min_obs=MIN_OBS, target_obs=TARGET_OBS):
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    used = []
    sigma_hat_running = []
    for k in range(min(target_obs, len(seq))):
        used.append(seq[k])
        n = len(used)
        if n < 2:
            continue
        x = np.asarray(used, dtype=float)
        S0 = float(np.sum((x - mu0) ** 2))
        S1 = float(np.sum((x - mu1) ** 2))
        log_lr = 0.5 * n * (math.log(max(S0, 1e-12)) - math.log(max(S1, 1e-12)))
        sigma_hat = math.sqrt(min(S0, S1) / n)
        sigma_hat_running.append(sigma_hat)
        if n >= min_obs:
            if log_lr >= A:
                return "BIRTH", n, log_lr, sigma_hat
            if log_lr <= B:
                return "SKIP", n, log_lr, sigma_hat
    if not used:
        return "SKIP", 0, 0.0, 0.0
    m = float(np.mean(used))
    sg = float(np.std(used, ddof=1)) if len(used) > 1 else 0.0
    dec = "BIRTH" if m >= 0.5 * (mu0 + mu1) else "SKIP"
    return dec, len(used), 0.0, sg


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

    rows, decs, ns, sgs = [], [], [], []
    for pid, t in eligible:
        seq = t["uq_observations"]
        dec, n, llr, sg = glr_decision(seq)
        decs.append(dec); ns.append(n); sgs.append(sg)
        rows.append({
            "id": pid, "n_in_trace": len(seq),
            "glr_decision": dec, "glr_n_obs": n,
            "log_lr_final": llr, "sigma_hat": sg,
            "sprt_decision": "BIRTH" if sprt_dep[pid] else "SKIP",
            "proxy_gt": bool(proxy_gt[pid]),
            "oracle_gt": bool(oracle_gt[pid]),
        })

    gts_proxy = [proxy_gt[i] for i, _ in eligible]
    gts_oracle = [oracle_gt[i] for i, _ in eligible]
    sprt_decs = ["BIRTH" if sprt_dep[i] else "SKIP" for i, _ in eligible]

    out = {
        "experiment": "HOLE_11_glr_sprt",
        "method": "GLR-SPRT with unknown variance (running MLE sigma_hat)",
        "params": dict(mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA,
                       min_obs=MIN_OBS, target_obs=TARGET_OBS),
        "n_eligible": len(eligible),
        "per_proposal": rows,
        "glr_metrics": {
            "proxy_gt": confusion(decs, gts_proxy),
            "oracle_gt": confusion(decs, gts_oracle),
            "avg_obs": float(np.mean(ns)),
            "n_births": sum(1 for d in decs if d == "BIRTH"),
            "sigma_hat_mean": float(np.mean(sgs)),
            "sigma_hat_range": [float(np.min(sgs)), float(np.max(sgs))],
        },
        "sprt_metrics": {
            "proxy_gt": confusion(sprt_decs, gts_proxy),
            "oracle_gt": confusion(sprt_decs, gts_oracle),
            "n_births": sum(1 for d in sprt_decs if d == "BIRTH"),
        },
        "headline": (
            f"GLR-SPRT (unknown sigma) on {len(eligible)} traces: "
            f"P/R proxy={confusion(decs, gts_proxy)['precision']:.2f}/{confusion(decs, gts_proxy)['recall']:.2f}, "
            f"oracle={confusion(decs, gts_oracle)['precision']:.2f}/{confusion(decs, gts_oracle)['recall']:.2f}, "
            f"avg N={np.mean(ns):.1f}; "
            f"sigma_hat range [{np.min(sgs):.3f}, {np.max(sgs):.3f}] "
            f"(SPRT assumes {0.20})."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(out["headline"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
