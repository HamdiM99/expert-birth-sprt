"""HOLE 19 — UQ-as-classifier AUROC under post-hoc ORACLE labels.

Reviewer Q9: which UQ summary best predicts oracle BIRTH (not the
proxy)? Existing App. L reports AUROC=1.000 for mean-UQ but uses the
*proxy* needs_birth_gt. This re-runs the same scoring against
posthoc_decision (oracle), which on the deployment trace is much
harsher (only 1 of 5 proxy-positives is oracle-positive per HOLE_7).

Outputs AUROC + AUPRC for: mean, max, min, std, slope, p95, p10,
mean-of-first-20, mean-of-last-20, drift, AR1.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("ANALYSIS_DATA",
    "../../../data"))
OUT = Path(os.environ.get("ANALYSIS_OUT",
    "../../../results/HOLE_19_uq_oracle_auroc.json"))


def auroc(scores, labels):
    """Trapezoid AUROC; tie-aware."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    if len(set(y)) < 2: return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    P = y.sum(); N = len(y) - P
    tps = np.cumsum(y_sorted); fps = np.cumsum(1 - y_sorted)
    tpr = tps / P if P else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N else np.zeros_like(fps, dtype=float)
    tpr = np.concatenate([[0.0], tpr]); fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


def auprc(scores, labels):
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=int)
    if len(set(y)) < 2: return float("nan")
    order = np.argsort(-s); y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    prec = tps / np.arange(1, len(y_sorted) + 1)
    rec = tps / max(1, y.sum())
    rec = np.concatenate([[0.0], rec]); prec = np.concatenate([[1.0], prec])
    # average precision (step interpolation)
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i] - rec[i-1]) * prec[i]
    return float(ap)


def features(seq):
    s = np.asarray(seq, dtype=float)
    n = len(s)
    if n < 2: return None
    xs = np.arange(n)
    slope = float(np.polyfit(xs, s, 1)[0])
    half = n // 2
    drift = float(s[half:].mean() - s[:half].mean()) if half else 0.0
    ar1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if n > 2 else 0.0
    if not np.isfinite(ar1): ar1 = 0.0
    feats = {
        "mean": float(s.mean()),
        "max":  float(s.max()),
        "min":  float(s.min()),
        "std":  float(s.std(ddof=1)) if n > 1 else 0.0,
        "slope": slope,
        "p95":  float(np.percentile(s, 95)),
        "p10":  float(np.percentile(s, 10)),
        "mean_first20": float(s[:20].mean()) if n >= 20 else float(s.mean()),
        "mean_last20":  float(s[-20:].mean()) if n >= 20 else float(s.mean()),
        "drift": drift,
        "ar1":   ar1,
    }
    return feats


def main():
    traces = [json.loads(l) for l in (DATA / "per_proposal_traces.jsonl").open()]
    pc = json.loads((DATA / "posthoc_counterfactual.json").read_text())
    proxy_gt = {p["id"]: bool(p["needs_birth_gt"]) for p in pc["proposals"]}
    oracle_gt = {p["id"]: bool(p["posthoc_decision"]) for p in pc["proposals"]}
    eligible_ids = [p["id"] for p in pc["proposals"]]
    by_id = {t["id"]: t for t in traces if t.get("uq_observations")}
    rows = []
    for pid in eligible_ids:
        if pid not in by_id: continue
        f = features(by_id[pid]["uq_observations"])
        if f is None: continue
        rows.append({"id": pid, **f,
                     "proxy": int(proxy_gt[pid]), "oracle": int(oracle_gt[pid])})

    feat_names = [k for k in rows[0] if k not in ("id","proxy","oracle")]
    proxy_y = [r["proxy"] for r in rows]
    oracle_y = [r["oracle"] for r in rows]
    results = {}
    for fn in feat_names:
        scores = [r[fn] for r in rows]
        results[fn] = {
            "AUROC_proxy":  auroc(scores, proxy_y),
            "AUPRC_proxy":  auprc(scores, proxy_y),
            "AUROC_oracle": auroc(scores, oracle_y),
            "AUPRC_oracle": auprc(scores, oracle_y),
        }
        # also for sign-flipped if that helps oracle (e.g. min, ar1)
        scores_neg = [-x for x in scores]
        results[fn]["AUROC_oracle_neg"] = auroc(scores_neg, oracle_y)

    n_proxy_pos = sum(proxy_y); n_oracle_pos = sum(oracle_y)
    out = {
        "experiment": "HOLE_19_uq_oracle_auroc",
        "n_proposals": len(rows),
        "n_proxy_pos": n_proxy_pos, "n_proxy_neg": len(rows) - n_proxy_pos,
        "n_oracle_pos": n_oracle_pos, "n_oracle_neg": len(rows) - n_oracle_pos,
        "per_feature": results,
        "rows": rows,
    }
    # rank features by AUROC_oracle (taking max of original or negated)
    best = []
    for fn in feat_names:
        a = max(results[fn]["AUROC_oracle"], results[fn]["AUROC_oracle_neg"])
        if not np.isnan(a): best.append((fn, a))
    best.sort(key=lambda x: -x[1])
    out["ranking_oracle"] = best
    print(f"N={len(rows)}, proxy+={n_proxy_pos}, oracle+={n_oracle_pos}")
    print("Feature           AUROC_proxy  AUPRC_proxy  AUROC_oracle  AUPRC_oracle")
    for fn in feat_names:
        r = results[fn]
        print(f"{fn:18s} {r['AUROC_proxy']:>11.3f}  {r['AUPRC_proxy']:>11.3f}  "
              f"{r['AUROC_oracle']:>12.3f}  {r['AUPRC_oracle']:>12.3f}")
    print("\nTop oracle-AUROC (taking max of orig/negated):")
    for fn, a in best[:5]:
        print(f"  {fn:18s} {a:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
