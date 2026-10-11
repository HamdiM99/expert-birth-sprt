"""BOCPD (Adams & MacKay 2007) baseline on the same 7 deployment proposals
as Tab. tab:baselines, using the GPU-replayed token-entropy streams.

Hazard: 1/lambda with lambda=50 (mean run length).
Likelihood: Normal with Normal-Inverse-Gamma prior (unknown mean, unknown var).
Decision rule: declare BIRTH on the first observation t (>= n_min=20) at which
   P(run-length collapses, i.e. r_t=0) > 0.5  AND
   posterior predictive mean since the change > mu_1.
Otherwise SKIP at MAX_OBS=50.

Standard reference: Adams & MacKay (2007); NIG conjugate update used.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

INP = Path("../../../data/per_proposal_traces.jsonl")
PC  = Path("../../../data/posthoc_counterfactual.json")
OUT = Path("../../../results/HOLE_22_bocpd_baseline.json")

LAMBDA   = 50.0          # hazard expected run length
HAZARD   = 1.0 / LAMBDA
MIN_OBS  = 20
MAX_OBS  = 50
# μ0/μ1/prior are recalibrated from the same streams (p40/p60) so BOCPD is on
# the SAME scale as GLR-SPRT in HOLE_21 — values populated in main().
MU_1         = None
PRIOR_MU0    = None
PRIOR_KAPPA0 = 1.0
PRIOR_ALPHA0 = 1.0
PRIOR_BETA0  = None      # set from empirical σ²


def student_t_logpdf(x, mu, kappa, alpha, beta):
    """Log posterior-predictive density of x under NIG(mu, kappa, alpha, beta).
    The predictive is a Student-t with 2α dof, location mu, scale^2 = β(κ+1)/(ακ).
    """
    df = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z = (x - mu) ** 2 / scale2
    return (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
            - 0.5 * math.log(df * math.pi * scale2)
            - 0.5 * (df + 1) * math.log1p(z / df))


def bocpd(seq):
    """Run BOCPD; return (decision, n_used, prob_changepoint_history,
    posterior_mean_since_change_at_decision)."""
    T = min(len(seq), MAX_OBS)
    if T == 0:
        return "SKIP", 0, [], float("nan")

    # joint = P(run_length=r, x_{1:t}); we keep an array of run-length probs.
    log_joint = np.full(1, 0.0)  # log P(r=0) = 0 (we know r0=0)
    # NIG params per run-length hypothesis
    mu     = np.array([PRIOR_MU0])
    kappa  = np.array([PRIOR_KAPPA0])
    alpha  = np.array([PRIOR_ALPHA0])
    beta   = np.array([PRIOR_BETA0])

    cp_probs = []          # P(r_t = 0 | x_{1:t})
    pred_mean_since_cp = float("nan")

    for t, x in enumerate(seq[:MAX_OBS], start=1):
        # 1) Predictive prob of x under each run-length hypothesis
        log_pred = np.array([student_t_logpdf(x, mu[i], kappa[i],
                                              alpha[i], beta[i])
                             for i in range(len(mu))])
        # 2) Growth: r_t = r_{t-1} + 1, prob (1 - hazard)
        log_grow = log_joint + log_pred + math.log(1 - HAZARD)
        # 3) Change: r_t = 0, prob hazard, with prior predictive of the new run
        log_cp_each = log_joint + log_pred + math.log(HAZARD)
        log_cp_total = float(np.logaddexp.reduce(log_cp_each))

        new_log_joint = np.concatenate([[log_cp_total], log_grow])

        # 4) Update NIG params for grown run-lengths (these advance from prior i)
        new_mu    = np.concatenate([[PRIOR_MU0],
                                    (kappa * mu + x) / (kappa + 1)])
        new_kappa = np.concatenate([[PRIOR_KAPPA0], kappa + 1])
        new_alpha = np.concatenate([[PRIOR_ALPHA0], alpha + 0.5])
        new_beta  = np.concatenate([[PRIOR_BETA0],
                                    beta + 0.5 * kappa * (x - mu) ** 2 / (kappa + 1)])

        # 5) Normalise
        Z = float(np.logaddexp.reduce(new_log_joint))
        log_joint = new_log_joint - Z
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        # 6) Trim tail (keep run-length probs > 1e-6)
        keep = np.exp(log_joint) > 1e-6
        keep[0] = True
        log_joint = log_joint[keep]; mu = mu[keep]; kappa = kappa[keep]
        alpha = alpha[keep]; beta = beta[keep]

        p_cp = float(np.exp(log_joint[0]))
        cp_probs.append(p_cp)

        # 7) Decision rule: only after MIN_OBS
        if t >= MIN_OBS and p_cp > 0.5:
            # posterior-predictive mean conditioned on the change-point
            # = μ of the r=0 hypothesis = PRIOR_MU0 (uninformed)... that's not
            # what we want. Use the running mean of the most-likely run-length
            # path that ENDED in the change. We approximate by the mean over
            # the most recent ceil(LAMBDA/2) observations.
            window = seq[max(0, t-25):t]
            pred_mean_since_cp = float(np.mean(window))
            if pred_mean_since_cp >= MU_1:
                return "BIRTH", t, cp_probs, pred_mean_since_cp
            else:
                return "SKIP", t, cp_probs, pred_mean_since_cp

    # Fallback at MAX_OBS: threshold on overall mean
    overall = float(np.mean(seq[:MAX_OBS]))
    return ("BIRTH" if overall >= 0.5 * (PRIOR_MU0 + MU_1) else "SKIP"), \
           min(T, MAX_OBS), cp_probs, overall


def confusion(decs, gts):
    TP = sum(1 for d,g in zip(decs,gts) if d=="BIRTH" and g)
    FP = sum(1 for d,g in zip(decs,gts) if d=="BIRTH" and not g)
    FN = sum(1 for d,g in zip(decs,gts) if d!="BIRTH" and g)
    TN = sum(1 for d,g in zip(decs,gts) if d!="BIRTH" and not g)
    P = TP/(TP+FP) if (TP+FP) else float("nan")
    R = TP/(TP+FN) if (TP+FN) else float("nan")
    F1 = 2*P*R/(P+R) if (P+R)>0 and not (np.isnan(P) or np.isnan(R)) else float("nan")
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"P":P,"R":R,"F1":F1}


def main():
    pc = json.loads(PC.read_text())
    oracle_gt = {p["id"]: bool(p["posthoc_decision"]) for p in pc["proposals"]}
    proxy_gt  = {p["id"]: bool(p["needs_birth_gt"])   for p in pc["proposals"]}

    streams = {}
    for line in INP.open():
        r = json.loads(line)
        pid = r.get("id")
        obs = r.get("uq_observations") or []
        if not obs: continue
        streams[pid] = [float(v) for v in obs]

    # Calibrate μ0/μ1/prior from the pooled streams (same scheme as HOLE_21)
    pooled = [v for pid, seq in streams.items() for v in seq if len(seq) >= MIN_OBS]
    global PRIOR_MU0, MU_1, PRIOR_BETA0
    PRIOR_MU0 = float(np.percentile(pooled, 40))
    MU_1      = float(np.percentile(pooled, 60))
    sigma2    = float(np.var(pooled, ddof=1))
    # NIG with α=1, β=σ² gives E[σ²] = σ² (mode at σ²)
    PRIOR_BETA0 = max(sigma2, 1e-4)
    print(f"[calib] μ0={PRIOR_MU0:.4f}, μ1={MU_1:.4f}, σ²={sigma2:.4f}")

    rows = []; decs, ns = [], []
    for pid in sorted(streams.keys()):
        seq = streams[pid]
        if len(seq) < MIN_OBS: continue
        d, n, _, m = bocpd(seq)
        rows.append({"id": pid, "n_obs": len(seq),
                     "decision": d, "n_decision": n,
                     "post_cp_mean": m,
                     "oracle": oracle_gt.get(pid),
                     "proxy":  proxy_gt.get(pid)})
        decs.append(d); ns.append(n)

    gts_o = [r["oracle"] for r in rows]
    gts_p = [r["proxy"]  for r in rows]
    out = {
        "experiment": "HOLE_22_bocpd_baseline",
        "description": "BOCPD (Adams & MacKay 2007) with NIG conjugate prior, "
                       f"hazard 1/{int(LAMBDA)}, on the GPU-replayed token-entropy "
                       "streams of the 7 deployment proposals.",
        "params": {"lambda": LAMBDA, "min_obs": MIN_OBS, "max_obs": MAX_OBS,
                   "mu1": MU_1,
                   "prior": {"mu0": PRIOR_MU0, "kappa0": PRIOR_KAPPA0,
                             "alpha0": PRIOR_ALPHA0, "beta0": PRIOR_BETA0}},
        "rows": rows,
        "oracle": confusion(decs, gts_o),
        "proxy":  confusion(decs, gts_p),
        "avg_n":  float(np.mean(ns)) if ns else 0.0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"wrote {OUT}")
    print(f"\nBOCPD on 7 deployment proposals:")
    print(f"  oracle:  P={out['oracle']['P']:.2f} R={out['oracle']['R']:.2f} "
          f"F1={out['oracle']['F1']:.2f}")
    print(f"  proxy :  P={out['proxy']['P']:.2f}  R={out['proxy']['R']:.2f}  "
          f"F1={out['proxy']['F1']:.2f}")
    print(f"  avg N: {out['avg_n']:.1f}")
    print()
    for r in rows:
        print(f"  pid={r['id']:>2} n_obs={r['n_obs']:>2} → {r['decision']:>5} "
              f"@n={r['n_decision']:>2}  oracle={r['oracle']} proxy={r['proxy']} "
              f"post_mean={r['post_cp_mean']:.3f}")


if __name__ == "__main__":
    main()
