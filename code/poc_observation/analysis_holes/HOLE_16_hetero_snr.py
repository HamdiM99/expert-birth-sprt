"""HOLE 16 — Heterogeneous-SNR controlled workload.

Reviewer Q3/Q6: the deployment trace is saturated (UQ ~3sigma gap)
so SPRT and batched-probe collapse onto the same operating point in
Tab. 5. To *separate* sequential strategies we construct a synthetic
4-quadrant workload (NL/code x high/low SNR) where the assumed
sigma=0.20 is wrong and observe how each gate degrades.

Design (per seed):
  - 4 quadrants:
       NL-high   : mu_d=0.50, sigma_d=0.10  (sep/sigma = 1.0 above mu0)
       NL-low    : mu_d=0.50, sigma_d=0.30  (sep/sigma = 0.33)
       code-high : mu_d=0.50, sigma_d=0.08
       code-low  : mu_d=0.50, sigma_d=0.35
  - 16 clusters per quadrant (8 need-birth at mu=0.55, 8 skip at mu=0.45)
  - 50 obs per cluster, gaussian around true mean with quadrant sigma
  - Run gates: SPRT (sigma=0.20), e-process (sub-Gaussian),
       GLR-SPRT (running MLE sigma_hat), mSPRT (mixture over delta),
       short-cal sigma_hat (n_cal=15)
  - Report per-gate / per-quadrant precision, recall, F1, avg N

Saves /workspace/analysis/results/HOLE_16_hetero_snr.json
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("ANALYSIS_OUT",
    "/workspace/analysis/results/HOLE_16_hetero_snr.json"))

MU0, MU1 = 0.45, 0.55
ALPHA, BETA = 0.05, 0.05
MIN_OBS, TARGET_OBS = 20, 50

QUADRANTS = [
    ("NL-high",   0.10),
    ("NL-low",    0.30),
    ("code-high", 0.08),
    ("code-low",  0.35),
]
N_PER_QUADRANT = 16  # 8 birth + 8 skip


def gen_workload(rng):
    """Return list of (qid, true_mu, sigma_d, gt_birth, obs)."""
    items = []
    for q, sd in QUADRANTS:
        for k in range(N_PER_QUADRANT):
            gt = (k % 2 == 0)
            mu = MU1 if gt else MU0
            obs = rng.normal(loc=mu, scale=sd, size=TARGET_OBS).clip(0, 1)
            items.append({
                "quadrant": q, "cluster_id": f"{q}-{k}",
                "true_mu": float(mu), "sigma_d": float(sd),
                "gt_birth": bool(gt), "obs": obs.tolist(),
            })
    return items


# ---------- gates ----------

def sprt_fixed(obs, sigma=0.20):
    A = math.log((1 - BETA) / ALPHA); B = math.log(BETA / (1 - ALPHA))
    used = []
    for k in range(min(TARGET_OBS, len(obs))):
        used.append(obs[k]); n = len(used); s = sum(used)
        if n >= MIN_OBS:
            llr = (MU1-MU0)/sigma**2*s - n*(MU1**2-MU0**2)/(2*sigma**2)
            if llr >= A: return "BIRTH", n
            if llr <= B: return "SKIP", n
    m = float(np.mean(used)) if used else 0.0
    return ("BIRTH" if m >= 0.5*(MU0+MU1) else "SKIP"), len(used)


def eprocess_subg(obs, sigma_bound=0.30):
    """Sub-Gaussian e-process for H0: mu<=MU0 vs H1: mu>=MU1.

    e_n = exp(lambda * sum(x - mu0) - n * lambda^2 * sigma^2 / 2)
    Reject H0 when e_n >= 1/alpha.
    """
    lam = (MU1 - MU0) / sigma_bound**2
    used = []; log_e = 0.0
    for k in range(min(TARGET_OBS, len(obs))):
        used.append(obs[k]); n = len(used)
        log_e += lam * (obs[k] - MU0) - 0.5 * lam**2 * sigma_bound**2
        if n >= MIN_OBS:
            if log_e >= math.log(1/ALPHA): return "BIRTH", n
            # symmetric for H0 acceptance: low e for opposite hyp
            lam_neg = (MU0 - MU1) / sigma_bound**2
            log_e_neg = sum(lam_neg*(o-MU1) - 0.5*lam_neg**2*sigma_bound**2 for o in used)
            if log_e_neg >= math.log(1/BETA): return "SKIP", n
    m = float(np.mean(used)) if used else 0.0
    return ("BIRTH" if m >= 0.5*(MU0+MU1) else "SKIP"), len(used)


def glr_sprt(obs):
    """GLR-SPRT with running MLE sigma_hat at each step."""
    A = math.log((1-BETA)/ALPHA); B = math.log(BETA/(1-ALPHA))
    used = []
    for k in range(min(TARGET_OBS, len(obs))):
        used.append(obs[k]); n = len(used)
        if n >= MIN_OBS:
            arr = np.asarray(used)
            # MLE sigma under composite (use sample std)
            sg = max(float(np.std(arr, ddof=1)), 0.02)
            s = arr.sum()
            llr = (MU1-MU0)/sg**2*s - n*(MU1**2-MU0**2)/(2*sg**2)
            if llr >= A: return "BIRTH", n
            if llr <= B: return "SKIP", n
    m = float(np.mean(used)) if used else 0.0
    return ("BIRTH" if m >= 0.5*(MU0+MU1) else "SKIP"), len(used)


def msprt(obs, tau=0.10, sigma=0.20):
    """Robbins mSPRT with Gaussian prior on delta=mu-mu_mid, var=tau^2.

    Mixture LLR (Gaussian mixture, known sigma):
      LLR = 0.5*log(sigma^2/(sigma^2+n*tau^2))
            + (n*(mean - mu_mid))^2 * tau^2 / (2*sigma^2*(sigma^2+n*tau^2))
    Reject H0 when LLR >= log(1/alpha).
    """
    mu_mid = 0.5*(MU0+MU1)
    used = []
    thr_pos = math.log(1/ALPHA); thr_neg = math.log(1/BETA)
    for k in range(min(TARGET_OBS, len(obs))):
        used.append(obs[k]); n = len(used)
        if n >= MIN_OBS:
            arr = np.asarray(used); m = arr.mean()
            llr = 0.5*math.log(sigma**2/(sigma**2 + n*tau**2)) \
                + (n*(m - mu_mid))**2 * tau**2 / (2*sigma**2*(sigma**2+n*tau**2))
            if (m > mu_mid) and llr >= thr_pos: return "BIRTH", n
            if (m < mu_mid) and llr >= thr_neg: return "SKIP", n
    mm = float(np.mean(used)) if used else 0.0
    return ("BIRTH" if mm >= mu_mid else "SKIP"), len(used)


def short_cal_sigma(obs, n_cal=15):
    cal = obs[:n_cal]
    sg = max(float(np.std(cal, ddof=1)), 0.02) if len(cal) > 1 else 0.20
    A = math.log((1-BETA)/ALPHA); B = math.log(BETA/(1-ALPHA))
    used = []
    for k in range(n_cal, min(TARGET_OBS, len(obs))):
        used.append(obs[k]); n = len(used); s = sum(used)
        if n >= max(5, MIN_OBS - n_cal):
            llr = (MU1-MU0)/sg**2*s - n*(MU1**2-MU0**2)/(2*sg**2)
            if llr >= A: return "BIRTH", n + n_cal
            if llr <= B: return "SKIP", n + n_cal
    m = float(np.mean(used)) if used else 0.0
    return ("BIRTH" if m >= 0.5*(MU0+MU1) else "SKIP"), len(used) + n_cal


GATES = {
    "sprt_fixed":   sprt_fixed,
    "eprocess":     eprocess_subg,
    "glr_sprt":     glr_sprt,
    "msprt_t0.10":  lambda o: msprt(o, tau=0.10),
    "msprt_t0.20":  lambda o: msprt(o, tau=0.20),
    "short_cal":    short_cal_sigma,
}


def confusion(decs, gts):
    TP = sum(1 for d,g in zip(decs,gts) if d=="BIRTH" and g)
    FP = sum(1 for d,g in zip(decs,gts) if d=="BIRTH" and not g)
    FN = sum(1 for d,g in zip(decs,gts) if d!="BIRTH" and g)
    TN = sum(1 for d,g in zip(decs,gts) if d!="BIRTH" and not g)
    P = TP/(TP+FP) if (TP+FP) else float("nan")
    R = TP/(TP+FN) if (TP+FN) else float("nan")
    F1 = 2*P*R/(P+R) if (P+R) and not (np.isnan(P) or np.isnan(R)) else float("nan")
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"P":P,"R":R,"F1":F1}


def main():
    SEEDS = [0, 1, 2, 3, 4]
    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        items = gen_workload(rng)
        decs_by_gate = {g: [] for g in GATES}
        ns_by_gate = {g: [] for g in GATES}
        for it in items:
            for g, fn in GATES.items():
                d, n = fn(it["obs"])
                decs_by_gate[g].append(d); ns_by_gate[g].append(n)
        gts = [it["gt_birth"] for it in items]
        per_gate = {}
        for g in GATES:
            cm_all = confusion(decs_by_gate[g], gts)
            cm_all["avgN"] = float(np.mean(ns_by_gate[g]))
            # per quadrant
            byq = {}
            for q, _ in QUADRANTS:
                idx = [i for i,it in enumerate(items) if it["quadrant"]==q]
                d_q = [decs_by_gate[g][i] for i in idx]
                n_q = [ns_by_gate[g][i] for i in idx]
                gt_q = [gts[i] for i in idx]
                cm = confusion(d_q, gt_q); cm["avgN"] = float(np.mean(n_q))
                byq[q] = cm
            per_gate[g] = {"all": cm_all, "by_quadrant": byq}
        per_seed.append({"seed": seed, "per_gate": per_gate})

    # Aggregate across seeds (mean P/R/F1/avgN)
    agg = {}
    for g in GATES:
        all_F1 = [s["per_gate"][g]["all"]["F1"] for s in per_seed]
        all_R  = [s["per_gate"][g]["all"]["R"]  for s in per_seed]
        all_P  = [s["per_gate"][g]["all"]["P"]  for s in per_seed]
        all_N  = [s["per_gate"][g]["all"]["avgN"] for s in per_seed]
        byq_agg = {}
        for q,_ in QUADRANTS:
            qF1 = [s["per_gate"][g]["by_quadrant"][q]["F1"] for s in per_seed]
            qR  = [s["per_gate"][g]["by_quadrant"][q]["R"]  for s in per_seed]
            qP  = [s["per_gate"][g]["by_quadrant"][q]["P"]  for s in per_seed]
            qN  = [s["per_gate"][g]["by_quadrant"][q]["avgN"] for s in per_seed]
            byq_agg[q] = {
                "F1_mean": float(np.nanmean(qF1)), "F1_std": float(np.nanstd(qF1)),
                "R_mean":  float(np.nanmean(qR)),  "R_std":  float(np.nanstd(qR)),
                "P_mean":  float(np.nanmean(qP)),  "P_std":  float(np.nanstd(qP)),
                "avgN_mean": float(np.nanmean(qN)),
            }
        agg[g] = {
            "F1_mean": float(np.nanmean(all_F1)), "F1_std": float(np.nanstd(all_F1)),
            "R_mean":  float(np.nanmean(all_R)),  "R_std":  float(np.nanstd(all_R)),
            "P_mean":  float(np.nanmean(all_P)),  "P_std":  float(np.nanstd(all_P)),
            "avgN_mean": float(np.nanmean(all_N)),
            "by_quadrant": byq_agg,
        }

    out = {
        "experiment": "HOLE_16_hetero_snr",
        "method": "synthetic 4-quadrant heterogeneous-SNR workload, 5 seeds",
        "params": {"mu0":MU0,"mu1":MU1,"alpha":ALPHA,"beta":BETA,
                    "min_obs":MIN_OBS,"target_obs":TARGET_OBS,
                    "quadrants":[(q,s) for q,s in QUADRANTS],
                    "n_per_quadrant":N_PER_QUADRANT,"seeds":SEEDS},
        "per_seed": per_seed,
        "aggregate": agg,
    }
    head = []
    for g in GATES:
        a = agg[g]
        head.append(f"{g}: F1={a['F1_mean']:.2f}+-{a['F1_std']:.2f} "
                    f"R={a['R_mean']:.2f} N={a['avgN_mean']:.1f}")
    out["headline"] = " | ".join(head)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(out["headline"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
