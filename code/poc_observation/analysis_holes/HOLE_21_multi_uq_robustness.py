"""HOLE 21 v2 — Multi-UQ-signal SPRT/GLR robustness, fed by GPU-replayed
observations with full multi-signal logging.

Input: data/multi_uq_observations.jsonl (one row per (proposal, obs))
Each row: {"proposal_id": int, "obs_idx": int, "query_text": str,
           "signals": {token_entropy, semantic_entropy, expert_disagreement,
                       backbone_agreement, implicit_feedback}}
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ANALYSIS = Path("../../../data")
INP = Path("../../../data/"
           "multi_uq_observations.jsonl")
OUT = Path("../../../results/"
           "HOLE_21_uq_signal_robustness.json")

ALPHA, BETA = 0.05, 0.05
MIN_OBS, MAX_OBS = 20, 50
A = math.log((1-BETA)/ALPHA)
B = math.log(BETA/(1-ALPHA))

UQ_SIGNALS = ["token_entropy", "semantic_entropy", "expert_disagreement",
              "backbone_agreement", "implicit_feedback", "composite_uq"]

# Sign convention: SPRT fires for "high signal = uncertain".
# Backbone agreement: high = AGREEMENT = LOW uncertainty → invert.
# Implicit feedback: positive = good response = LOW uncertainty → negate.
# Composite UQ: per deployment config, weights already point "high = uncertain".
SIGN_FLIP = {"backbone_agreement": True, "implicit_feedback": True}

# Per deployment configs: composite weights for [sem_ent, tok_ent, disag, 1-bb_agree, -impl_fb]
COMPOSITE_W = {"semantic_entropy": 0.3, "token_entropy": 0.2,
               "expert_disagreement": 0.3, "backbone_agreement": -0.15,
               "implicit_feedback": -0.05}
COMPOSITE_BIAS = 0.15  # +0.15·1 from (1-bb_agree); +0·1 from -impl_fb stub



def fixed_sigma_sprt(seq, mu0, mu1, sigma=0.20):
    used = []
    for x in seq[:MAX_OBS]:
        used.append(x)
        n = len(used); s = sum(used)
        if n >= MIN_OBS:
            llr = (mu1-mu0)/sigma**2 * s - n*(mu1**2-mu0**2)/(2*sigma**2)
            if llr >= A: return "BIRTH", n
            if llr <= B: return "SKIP", n
    if not used: return "SKIP", 0
    return ("BIRTH" if np.mean(used) >= 0.5*(mu0+mu1) else "SKIP"), len(used)


def glr_sprt(seq, mu0, mu1):
    used = []
    for x in seq[:MAX_OBS]:
        used.append(x)
        n = len(used)
        if n >= MIN_OBS:
            arr = np.asarray(used, dtype=float)
            S0 = float(((arr - mu0)**2).sum())
            S1 = float(((arr - mu1)**2).sum())
            if S0 <= 0 or S1 <= 0: continue
            llr = 0.5 * n * math.log(S0/S1)
            if llr >= A: return "BIRTH", n
            if llr <= B: return "SKIP", n
    if not used: return "SKIP", 0
    return ("BIRTH" if np.mean(used) >= 0.5*(mu0+mu1) else "SKIP"), len(used)


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
    pc = json.loads((ANALYSIS / "posthoc_counterfactual.json").read_text())
    oracle_gt = {p["id"]: bool(p["posthoc_decision"]) for p in pc["proposals"]}
    proxy_gt  = {p["id"]: bool(p["needs_birth_gt"])   for p in pc["proposals"]}

    # group observations by proposal × signal
    per_signal = {s: {} for s in UQ_SIGNALS}
    for line in INP.open():
        r = json.loads(line)
        pid = r["proposal_id"]
        sig = r["signals"]
        # raw signals (with sign flip where appropriate)
        for s in [s for s in UQ_SIGNALS if s != "composite_uq"]:
            v = sig.get(s)
            if v is None: continue
            try: v = float(v)
            except: continue
            if not np.isfinite(v): continue
            if SIGN_FLIP.get(s):
                v = -v   # high → low uncertainty so flip to high → high uncertainty
            per_signal[s].setdefault(pid, []).append(v)
        # composite: linear combination per deployment config
        try:
            c = (COMPOSITE_W["semantic_entropy"]    * float(sig["semantic_entropy"]) +
                 COMPOSITE_W["token_entropy"]       * float(sig["token_entropy"])    +
                 COMPOSITE_W["expert_disagreement"] * float(sig["expert_disagreement"]) +
                 COMPOSITE_W["backbone_agreement"]  * float(sig["backbone_agreement"]) +
                 COMPOSITE_W["implicit_feedback"]   * float(sig.get("implicit_feedback", 0.0)) +
                 COMPOSITE_BIAS)
            per_signal["composite_uq"].setdefault(pid, []).append(c)
        except Exception:
            pass

    # calibrate per signal: 40th/60th of pooled values, only proposals with >=MIN_OBS
    calib = {}
    for s, pp in per_signal.items():
        all_vals = [v for pid, seq in pp.items() for v in seq if len(seq) >= MIN_OBS]
        if not all_vals:
            calib[s] = None; continue
        a = float(np.percentile(all_vals, 40))
        b = float(np.percentile(all_vals, 60))
        if b <= a + 1e-6:
            calib[s] = None
        else:
            calib[s] = (a, b)

    results = {}
    for s in UQ_SIGNALS:
        if calib[s] is None:
            results[s] = {"skipped": "degenerate calibration (constant signal)"}
            continue
        mu0, mu1 = calib[s]
        pp = per_signal[s]
        rows = []; decs_s, decs_g, ns_s, ns_g = [], [], [], []
        for pid in sorted(pp.keys()):
            seq = pp[pid]
            if len(seq) < MIN_OBS: continue
            d_s, n_s = fixed_sigma_sprt(seq, mu0, mu1)
            d_g, n_g = glr_sprt(seq, mu0, mu1)
            rows.append({"id":pid, "n_obs":len(seq),
                         "mean":float(np.mean(seq)),
                         "std":float(np.std(seq, ddof=1)) if len(seq)>1 else 0.0,
                         "sprt":d_s, "n_sprt":n_s,
                         "glr":d_g,  "n_glr":n_g,
                         "oracle":oracle_gt.get(pid),
                         "proxy":proxy_gt.get(pid)})
            decs_s.append(d_s); decs_g.append(d_g)
            ns_s.append(n_s);    ns_g.append(n_g)
        gts_o = [r["oracle"] for r in rows]
        gts_p = [r["proxy"]  for r in rows]
        results[s] = {
            "calibration": {"mu0_p40": mu0, "mu1_p60": mu1},
            "n_proposals_eligible": len(rows),
            "fixed_sigma_sprt": {
                "oracle": confusion(decs_s, gts_o),
                "proxy":  confusion(decs_s, gts_p),
                "avg_n":  float(np.mean(ns_s)) if ns_s else 0.0,
            },
            "glr_sprt": {
                "oracle": confusion(decs_g, gts_o),
                "proxy":  confusion(decs_g, gts_p),
                "avg_n":  float(np.mean(ns_g)) if ns_g else 0.0,
            },
            "rows": rows,
        }

    out = {
        "experiment": "HOLE_21_uq_signal_robustness",
        "description": "GLR-SPRT and fixed-σ SPRT replayed on GPU with full "
                       "multi-signal logging across 5 UQ statistics, run on "
                       "the cluster_queries of the same 7 deployment proposals.",
        "params": {"alpha": ALPHA, "beta": BETA,
                   "min_obs": MIN_OBS, "max_obs": MAX_OBS,
                   "A": A, "B": B},
        "per_signal": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT}\n")

    print(f"{'signal':<22s} {'gate':<10s} {'P':>5s} {'R':>5s} {'F1':>5s} "
          f"{'avg_N':>6s}  oracle / proxy")
    print("-"*90)
    for s in UQ_SIGNALS:
        r = results.get(s, {})
        if "skipped" in r:
            print(f"{s:<22s} SKIPPED — {r['skipped']}"); continue
        for k, lbl in [("fixed_sigma_sprt","SPRT-fix"), ("glr_sprt","GLR-SPRT")]:
            o = r[k]["oracle"]; p = r[k]["proxy"]; n = r[k]["avg_n"]
            f1o = o["F1"]; f1p = p["F1"]
            print(f"{s:<22s} {lbl:<10s} "
                  f"{o['P']:>5.2f} {o['R']:>5.2f} {f1o:>5.2f} "
                  f"{n:>6.1f}  oracle | "
                  f"P={p['P']:.2f} R={p['R']:.2f} F1={f1p:.2f}")
        print()


if __name__ == "__main__":
    main()
