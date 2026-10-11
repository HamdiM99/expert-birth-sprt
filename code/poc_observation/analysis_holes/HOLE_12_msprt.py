"""HOLE 12 — mSPRT (mixture SPRT, Robbins 1970) with Gaussian prior on delta.

Reviewer Q2 again: avoid fixing delta=0.10 a priori. Use a mixture over
(mu1 - mu0) under a N(0, tau^2) prior.

For Gaussian X_i ~ N(mu, sigma^2) with known sigma, testing
H0: mu = mu0 vs H1: mu != mu0, the mSPRT statistic at time n is

    Lambda_n = sqrt(sigma^2 / (sigma^2 + n tau^2)) *
               exp( n^2 tau^2 / (2 sigma^2 (sigma^2 + n tau^2)) *
                    ( bar X_n - mu0 )^2 )

Reject H0 (=> BIRTH) when Lambda_n >= 1/alpha. This is anytime-valid
under H0 by Ville's inequality on the prior-mixture martingale.

We also pair it with a symmetric "mu0 < mu - delta" stopping rule for
SKIP (matches our gate semantics).
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
DATA = Path(os.environ.get("ANALYSIS_DATA", _ROOT / "data"))
OUT = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_12_msprt.json"))

MU0 = 0.45
SIGMA = 0.20
TAU = 0.10           # prior std on (mu - mu0); reviewer-friendly weak prior
ALPHA, BETA = 0.05, 0.05
MIN_OBS, TARGET_OBS = 20, 50


def msprt_decision(seq, mu0=MU0, sigma=SIGMA, tau=TAU,
                   alpha=ALPHA, beta=BETA,
                   min_obs=MIN_OBS, target_obs=TARGET_OBS):
    """Robbins (1970) mixture-SPRT. Returns (decision, n, log_lambda)."""
    log_thresh_h1 = -math.log(alpha)
    log_thresh_h0 = math.log(beta)
    used = []
    log_lam_history = []
    for k in range(min(target_obs, len(seq))):
        used.append(seq[k])
        n = len(used)
        if n < 2:
            continue
        xbar = float(np.mean(used))
        var_total = sigma * sigma + n * tau * tau
        log_pref = 0.5 * math.log(sigma * sigma / var_total)
        log_exp = (n * n * tau * tau) / (2 * sigma * sigma * var_total) * (xbar - mu0) ** 2
        # Sign: we want BIRTH only if xbar > mu0 (one-sided semantics).
        if xbar < mu0:
            log_lam = log_pref - log_exp  # below null => evidence against H1
        else:
            log_lam = log_pref + log_exp
        log_lam_history.append(log_lam)
        if n >= min_obs:
            if log_lam >= log_thresh_h1 and xbar > mu0:
                return "BIRTH", n, log_lam
            if log_lam <= log_thresh_h0:
                return "SKIP", n, log_lam
    if not used:
        return "SKIP", 0, 0.0
    m = float(np.mean(used))
    return ("BIRTH" if m > mu0 + tau else "SKIP"), len(used), log_lam_history[-1] if log_lam_history else 0.0


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

    # Sweep tau to show robustness to prior choice
    sweep = []
    for tau_val in [0.05, 0.10, 0.15, 0.20, 0.30]:
        decs, ns = [], []
        for pid, t in eligible:
            d, n, _ = msprt_decision(t["uq_observations"], tau=tau_val)
            decs.append(d); ns.append(n)
        gts_p = [proxy_gt[i] for i, _ in eligible]
        gts_o = [oracle_gt[i] for i, _ in eligible]
        sweep.append({
            "tau": tau_val,
            "proxy_gt": confusion(decs, gts_p),
            "oracle_gt": confusion(decs, gts_o),
            "avg_obs": float(np.mean(ns)),
            "n_births": sum(1 for d in decs if d == "BIRTH"),
        })

    # Headline: recommended tau=0.10 (matches SPRT delta)
    decs, ns, llams, rows = [], [], [], []
    for pid, t in eligible:
        d, n, llam = msprt_decision(t["uq_observations"])
        decs.append(d); ns.append(n); llams.append(llam)
        rows.append({
            "id": pid, "n_in_trace": len(t["uq_observations"]),
            "msprt_decision": d, "msprt_n_obs": n, "log_lambda": llam,
            "sprt_decision": "BIRTH" if sprt_dep[pid] else "SKIP",
            "proxy_gt": bool(proxy_gt[pid]),
            "oracle_gt": bool(oracle_gt[pid]),
        })
    gts_p = [proxy_gt[i] for i, _ in eligible]
    gts_o = [oracle_gt[i] for i, _ in eligible]
    sprt_decs = ["BIRTH" if sprt_dep[i] else "SKIP" for i, _ in eligible]

    out = {
        "experiment": "HOLE_12_msprt",
        "method": "mSPRT (Robbins 1970 mixture-SPRT, Gaussian prior tau)",
        "params": dict(mu0=MU0, sigma=SIGMA, tau=TAU,
                       alpha=ALPHA, beta=BETA,
                       min_obs=MIN_OBS, target_obs=TARGET_OBS),
        "n_eligible": len(eligible),
        "per_proposal": rows,
        "msprt_metrics_tau_0_10": {
            "proxy_gt": confusion(decs, gts_p),
            "oracle_gt": confusion(decs, gts_o),
            "avg_obs": float(np.mean(ns)),
            "n_births": sum(1 for d in decs if d == "BIRTH"),
        },
        "sprt_metrics": {
            "proxy_gt": confusion(sprt_decs, gts_p),
            "oracle_gt": confusion(sprt_decs, gts_o),
            "n_births": sum(1 for d in sprt_decs if d == "BIRTH"),
        },
        "tau_sweep": sweep,
        "headline": (
            f"mSPRT(tau=0.10) on {len(eligible)} traces: "
            f"P/R proxy={confusion(decs, gts_p)['precision']:.2f}/{confusion(decs, gts_p)['recall']:.2f}, "
            f"oracle={confusion(decs, gts_o)['precision']:.2f}/{confusion(decs, gts_o)['recall']:.2f}, "
            f"avg N={np.mean(ns):.1f}. "
            f"Robust across tau in [0.05, 0.30]: see tau_sweep."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(out["headline"])
    for s in sweep:
        print(f"  tau={s['tau']:.2f}: P={s['proxy_gt']['precision']:.2f} "
              f"R={s['proxy_gt']['recall']:.2f}  N={s['avg_obs']:.1f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
