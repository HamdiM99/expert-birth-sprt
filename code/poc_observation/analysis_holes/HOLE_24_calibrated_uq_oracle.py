"""HOLE 24 — Multi-UQ + per-domain z-scored composite UQ vs the post-hoc oracle.

Question: does the proxy/oracle gap (1/4 BIRTHs confirmed under token entropy)
close when the gate runs on a calibrated multi-UQ composite signal?

Inputs:
  data/multi_uq_observations.jsonl         (5 UQ signals x 185 obs x 7 props)
  data/posthoc_counterfactual.json         (oracle ground truth per proposal)
  data/per_proposal_traces.jsonl           (cluster metadata: domain, status)

Outputs:
  results/HOLE_24_calibrated_uq_oracle.json
  results/HOLE_24_signal_oracle_table.tex   (paper-ready table)

For each candidate signal-config we (1) replay the GLR-SPRT gate on the per-obs
signal stream, (2) record BIRTH/SKIP per proposal, (3) score the resulting
decisions against the post-hoc oracle (precision/recall/F1, agreement count),
(4) emit a head-to-head table.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT   = Path(__file__).resolve().parents[3]
DATA   = ROOT / "data"
RESULT = ROOT / "results"
INP_UQ = DATA / "multi_uq_observations.jsonl"
INP_TR = DATA / "per_proposal_traces.jsonl"
INP_OR = DATA / "posthoc_counterfactual.json"
OUT_J  = RESULT / "HOLE_24_calibrated_uq_oracle.json"
OUT_T  = RESULT / "HOLE_24_signal_oracle_table.tex"

ALPHA, BETA = 0.05, 0.05
MIN_OBS, MAX_OBS = 20, 50
A_TH = math.log((1 - BETA) / ALPHA)
B_TH = math.log(BETA / (1 - ALPHA))

# Sign convention for SPRT: "high signal => uncertain => candidate for BIRTH"
SIGN_FLIP = {"backbone_agreement": True, "implicit_feedback": True}

# Composite weights matching the deployment config of HOLE_21
COMP_W = {
    "semantic_entropy":     0.30,
    "token_entropy":        0.20,
    "expert_disagreement":  0.30,
    "backbone_agreement":  -0.15,
    "implicit_feedback":   -0.05,
}

# Domain assignment heuristic from cluster_id: matches deployment trace.
# 0 = medical, 1 = legal, 2 = code, 4 = trivia, 5 = math, 6 = generic, 7 = chat
CLUSTER_DOMAIN = {"0": "medical", "1": "legal", "2": "code",
                  "4": "trivia", "5": "math", "6": "generic", "7": "chat"}


def load_oracle():
    d = json.load(open(INP_OR))
    return {p["id"]: p for p in d["proposals"]}


def load_traces():
    """Map proposal_id -> (cluster_id, domain, status)."""
    out = {}
    for line in open(INP_TR):
        r = json.loads(line)
        cid = str(r["cluster_id"])
        out[r["id"]] = {
            "cluster_id": cid,
            "domain": CLUSTER_DOMAIN.get(cid, "generic"),
            "status": r["status"],
        }
    return out


def load_signals():
    """Group multi-UQ observations by proposal_id, returning ordered lists."""
    by_p = defaultdict(list)
    for line in open(INP_UQ):
        r = json.loads(line)
        by_p[r["proposal_id"]].append(r)
    for pid in by_p:
        by_p[pid].sort(key=lambda r: r["obs_idx"])
    return dict(by_p)


def get_signal(obs_list, signal_name):
    """Extract a single signal stream, applying sign-flip convention."""
    flip = SIGN_FLIP.get(signal_name, False)
    if signal_name == "composite_uq":
        out = []
        for r in obs_list:
            v = sum(COMP_W[k] * r["signals"][k] for k in COMP_W)
            out.append(v)
        return out
    return [(-r["signals"][signal_name]) if flip else r["signals"][signal_name]
            for r in obs_list]


def per_domain_zscore(seqs_by_pid, traces):
    """Compute domain mean/std on the pooled stream, then z-score per domain."""
    by_domain = defaultdict(list)
    for pid, seq in seqs_by_pid.items():
        d = traces[pid]["domain"]
        by_domain[d].extend(seq)
    stats = {d: (np.mean(v), np.std(v) + 1e-8) for d, v in by_domain.items()}
    out = {}
    for pid, seq in seqs_by_pid.items():
        d = traces[pid]["domain"]
        m, s = stats[d]
        out[pid] = [(x - m) / s for x in seq]
    return out, stats


def calibrate_mu(seqs_by_pid, q_lo=0.20, q_hi=0.80):
    """Calibrate (mu0, mu1) from the pooled signal distribution by quantiles."""
    pool = [x for seq in seqs_by_pid.values() for x in seq]
    return float(np.quantile(pool, q_lo)), float(np.quantile(pool, q_hi))


def glr_sprt(seq, mu0, mu1):
    """GLR-SPRT with running-MLE sigma; same A/B thresholds as fixed-σ Wald."""
    used = []
    for x in seq[:MAX_OBS]:
        used.append(x)
        n = len(used)
        if n < MIN_OBS:
            continue
        s = np.array(used)
        # Running variance under H0 / H1 (constrained MLE = sample var around mu_k)
        v0 = max(np.mean((s - mu0) ** 2), 1e-6)
        v1 = max(np.mean((s - mu1) ** 2), 1e-6)
        llr = 0.5 * n * (math.log(v0) - math.log(v1))
        if llr >= A_TH:
            return "BIRTH", n, llr
        if llr <= B_TH:
            return "SKIP", n, llr
    if not used:
        return "SKIP", 0, 0.0
    fallback = "BIRTH" if np.mean(used) >= 0.5 * (mu0 + mu1) else "SKIP"
    return fallback, len(used), 0.0


def score_against_oracle(decisions, oracle):
    """decisions: {pid -> 'BIRTH'/'SKIP'}; oracle: {pid -> dict with posthoc_decision}."""
    tp = fp = fn = tn = 0
    pids = sorted(decisions.keys())
    matrix = []
    for pid in pids:
        gate = (decisions[pid] == "BIRTH")
        truth = bool(oracle[pid]["posthoc_decision"])
        cell = ("TP" if gate and truth else
                "FP" if gate and not truth else
                "FN" if not gate and truth else "TN")
        matrix.append({"pid": pid, "gate": decisions[pid],
                       "oracle": "BIRTH" if truth else "SKIP", "cell": cell})
        if cell == "TP": tp += 1
        elif cell == "FP": fp += 1
        elif cell == "FN": fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    agree = tp + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1,
            "agreement": agree, "n": n,
            "agreement_rate": agree / n if n else 0.0,
            "matrix": matrix}


def run_config(name, seqs_by_pid, oracle, normalize=False, traces=None):
    """Run GLR-SPRT on one signal config and score vs oracle."""
    if normalize:
        seqs_by_pid, _stats = per_domain_zscore(seqs_by_pid, traces)
    mu0, mu1 = calibrate_mu(seqs_by_pid)
    decisions = {}
    n_obs = {}
    for pid, seq in seqs_by_pid.items():
        d, n, _llr = glr_sprt(seq, mu0, mu1)
        decisions[pid] = d
        n_obs[pid] = n
    score = score_against_oracle(decisions, oracle)
    score["config"]    = name
    score["mu0"], score["mu1"] = mu0, mu1
    score["mean_n"]   = float(np.mean(list(n_obs.values())))
    score["decisions"] = decisions
    return score


def calibrate_mu_q(seqs_by_pid, q_lo, q_hi):
    pool = [x for seq in seqs_by_pid.values() for x in seq]
    return float(np.quantile(pool, q_lo)), float(np.quantile(pool, q_hi))


def run_config_q(name, seqs_by_pid, oracle, q_lo, q_hi,
                 normalize=False, traces=None):
    if normalize:
        seqs_by_pid, _ = per_domain_zscore(seqs_by_pid, traces)
    mu0, mu1 = calibrate_mu_q(seqs_by_pid, q_lo, q_hi)
    decisions, n_obs = {}, {}
    for pid, seq in seqs_by_pid.items():
        d, n, _ = glr_sprt(seq, mu0, mu1)
        decisions[pid] = d; n_obs[pid] = n
    score = score_against_oracle(decisions, oracle)
    score["config"]    = name
    score["mu0"], score["mu1"] = mu0, mu1
    score["mean_n"]    = float(np.mean(list(n_obs.values())))
    score["decisions"] = decisions
    score["q_lo"], score["q_hi"] = q_lo, q_hi
    return score


def main():
    oracle = load_oracle()
    traces = load_traces()
    obs    = load_signals()

    signals = ["token_entropy", "semantic_entropy", "expert_disagreement",
               "backbone_agreement", "composite_uq"]
    quantile_grid = [(0.20, 0.80), (0.30, 0.70), (0.35, 0.65), (0.40, 0.60)]
    configs = []
    for sig in signals:
        for q_lo, q_hi in quantile_grid:
            seqs = {pid: get_signal(o, sig) for pid, o in obs.items()}
            configs.append(run_config_q(
                f"{sig} (raw, q={q_lo:.2f}/{q_hi:.2f})",
                seqs, oracle, q_lo, q_hi))
            configs.append(run_config_q(
                f"{sig} (z-domain, q={q_lo:.2f}/{q_hi:.2f})",
                seqs, oracle, q_lo, q_hi, normalize=True, traces=traces))

    # Headline: deployment baseline (token_entropy raw) vs best
    by_agree = sorted(configs, key=lambda c: -c["agreement"])
    print(f"{'config':<35s}  TP FP FN TN  P     R     F1    agree  meanN")
    for c in by_agree:
        print(f"{c['config']:<35s}  "
              f"{c['tp']:>2d} {c['fp']:>2d} {c['fn']:>2d} {c['tn']:>2d}  "
              f"{c['precision']:.2f}  {c['recall']:.2f}  {c['f1']:.2f}  "
              f"{c['agreement']:>2d}/7   {c['mean_n']:.1f}")

    # ----- Multi-UQ majority vote across signals -----
    vote_signals = ["token_entropy", "semantic_entropy", "expert_disagreement",
                    "backbone_agreement"]
    for q_lo, q_hi in [(0.20, 0.80), (0.30, 0.70)]:
        per_sig_decisions = {}
        for sig in vote_signals:
            seqs = {pid: get_signal(o, sig) for pid, o in obs.items()}
            mu0, mu1 = calibrate_mu_q(seqs, q_lo, q_hi)
            for pid, seq in seqs.items():
                d, _, _ = glr_sprt(seq, mu0, mu1)
                per_sig_decisions.setdefault(pid, []).append(d)
        # majority vote BIRTH iff >= 2/4 signals say BIRTH
        for k in (2, 3):
            decisions = {pid: ("BIRTH" if sum(1 for d in v if d == "BIRTH") >= k
                               else "SKIP") for pid, v in per_sig_decisions.items()}
            score = score_against_oracle(decisions, oracle)
            score["config"] = f"vote-{k}of4 (q={q_lo:.2f}/{q_hi:.2f})"
            score["mu0"] = score["mu1"] = None
            score["mean_n"] = float("nan")
            score["decisions"] = decisions
            configs.append(score)
            print(f"{score['config']:<35s}  "
                  f"{score['tp']:>2d} {score['fp']:>2d} {score['fn']:>2d} {score['tn']:>2d}  "
                  f"{score['precision']:.2f}  {score['recall']:.2f}  {score['f1']:.2f}  "
                  f"{score['agreement']:>2d}/7")

    OUT_J.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"oracle_truth": [(pid, bool(oracle[pid]["posthoc_decision"]))
                                 for pid in sorted(oracle)],
               "configs": configs}, open(OUT_J, "w"), indent=2)

    # LaTeX-ready table
    rows = ["\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Signal $\\times$ normalization & Agreement (vs oracle) & TP & FP & FN & TN & F1 \\\\",
            "\\midrule"]
    for c in by_agree:
        rows.append(f"{c['config']} & {c['agreement']}/7 & "
                    f"{c['tp']} & {c['fp']} & {c['fn']} & {c['tn']} & "
                    f"{c['f1']:.2f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    OUT_T.write_text("\n".join(rows))
    print(f"\nwrote {OUT_J}")
    print(f"wrote {OUT_T}")


if __name__ == "__main__":
    main()
