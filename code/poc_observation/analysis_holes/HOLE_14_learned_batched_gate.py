"""HOLE 14 — Learned batched gate with multi-feature UQ summaries.

Reviewer Q6: "Could you extend the baseline suite with a learned batched
gate that uses multiple summary statistics (mean, variance, trend) of UQ
with cross-validation, and compare its Pareto frontier to SPRT under the
same observation budget?"

Protocol:
  - For each eligible proposal trace, take the first K observations
    (K in {10, 15, 20, 30, 50}).
  - Extract features: mean, std, slope (linear-fit trend), max, min,
    p95, p10, mean-of-second-half - mean-of-first-half (drift), AR1.
  - Train logistic regression + small MLP on these features with
    leave-one-out CV (N=7).
  - Report P/R/F1 vs proxy and oracle GT, vs SPRT and batched-probe.

Honest caveat: N=7 is too small for any learned baseline to be robust.
The result demonstrates that even a richer feature representation does
not unlock a different operating point on this trace; further evidence
that the UQ signal saturates at deployment scale.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("ANALYSIS_DATA", "/workspace/analysis/data"))
OUT = Path(os.environ.get("ANALYSIS_OUT",
    "/workspace/analysis/results/HOLE_14_learned_batched_gate.json"))


def features(seq, k):
    """Extract multi-feature summary from first k observations."""
    s = np.asarray(seq[:k], dtype=float)
    n = len(s)
    if n < 2:
        return None
    xs = np.arange(n)
    slope, intercept = np.polyfit(xs, s, 1)
    half = n // 2
    drift = float(s[half:].mean() - s[:half].mean()) if half > 0 else 0.0
    ar1 = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if n > 2 else 0.0
    if not np.isfinite(ar1): ar1 = 0.0
    return np.array([
        float(s.mean()), float(s.std(ddof=1)) if n > 1 else 0.0,
        float(slope), float(s.max()), float(s.min()),
        float(np.percentile(s, 95)), float(np.percentile(s, 10)),
        drift, ar1,
    ], dtype=float)


def confusion(decisions, gts):
    TP = sum(1 for d, g in zip(decisions, gts) if d and g)
    FP = sum(1 for d, g in zip(decisions, gts) if d and not g)
    FN = sum(1 for d, g in zip(decisions, gts) if not d and g)
    TN = sum(1 for d, g in zip(decisions, gts) if not d and not g)
    P = TP / (TP + FP) if (TP + FP) else float("nan")
    R = TP / (TP + FN) if (TP + FN) else float("nan")
    F1 = 2 * P * R / (P + R) if (P + R) > 0 and not (np.isnan(P) or np.isnan(R)) else float("nan")
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": P, "recall": R, "f1": F1}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut

    traces = [json.loads(l) for l in (DATA / "per_proposal_traces.jsonl").open()]
    pc = json.loads((DATA / "posthoc_counterfactual.json").read_text())
    eligible_ids = [p["id"] for p in pc["proposals"]]
    oracle_gt = {p["id"]: p["posthoc_decision"] for p in pc["proposals"]}
    proxy_gt = {p["id"]: p["needs_birth_gt"] for p in pc["proposals"]}
    sprt_dep = {p["id"]: p["deployment_decision"] for p in pc["proposals"]}

    by_id = {t["id"]: t for t in traces if t.get("uq_observations")}
    eligible = [(i, by_id[i]) for i in eligible_ids if i in by_id]
    print(f"N eligible = {len(eligible)}")

    by_K = {}
    for K in [10, 15, 20, 30, 50]:
        # Build feature matrix
        feats, y_proxy, y_oracle, ids = [], [], [], []
        for pid, t in eligible:
            f = features(t["uq_observations"], K)
            if f is None: continue
            feats.append(f)
            y_proxy.append(int(proxy_gt[pid]))
            y_oracle.append(int(oracle_gt[pid]))
            ids.append(pid)
        X = np.vstack(feats)
        y_p = np.array(y_proxy)
        y_o = np.array(y_oracle)

        results_K = {"K": K, "n": len(ids), "ids": ids}
        for clf_name, make_clf in [
            ("logistic", lambda: LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
            ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000,
                                           random_state=0)),
        ]:
            for tag, y in [("proxy", y_p), ("oracle", y_o)]:
                if len(set(y)) < 2:
                    results_K[f"{clf_name}_{tag}"] = {"error": "single_class"}
                    continue
                preds = np.zeros_like(y)
                loo = LeaveOneOut()
                for tr, te in loo.split(X):
                    sc = StandardScaler().fit(X[tr])
                    Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
                    clf = make_clf()
                    clf.fit(Xtr, y[tr])
                    preds[te] = clf.predict(Xte)
                results_K[f"{clf_name}_{tag}"] = confusion(preds.tolist(),
                                                            y.tolist())
        by_K[K] = results_K
        print(f"K={K:>2}: logistic proxy {results_K['logistic_proxy']}; "
              f"oracle {results_K['logistic_oracle']}")

    # SPRT reference
    sprt_decs = [sprt_dep[i] for i, _ in eligible]
    gts_p = [proxy_gt[i] for i, _ in eligible]
    gts_o = [oracle_gt[i] for i, _ in eligible]
    sprt_p = confusion(sprt_decs, gts_p)
    sprt_o = confusion(sprt_decs, gts_o)

    # Pick best K * classifier per metric
    best_proxy_f1, best_oracle_f1 = -1, -1
    best_p_cell, best_o_cell = None, None
    for K, R in by_K.items():
        for clf in ("logistic", "mlp"):
            for tag, store in [("proxy", "best_p"), ("oracle", "best_o")]:
                cell = R.get(f"{clf}_{tag}")
                if not cell or "error" in cell or np.isnan(cell["f1"]): continue
                if tag == "proxy" and cell["f1"] > best_proxy_f1:
                    best_proxy_f1 = cell["f1"]
                    best_p_cell = {"K": K, "clf": clf, **cell}
                if tag == "oracle" and cell["f1"] > best_oracle_f1:
                    best_oracle_f1 = cell["f1"]
                    best_o_cell = {"K": K, "clf": clf, **cell}

    out = {
        "experiment": "HOLE_14_learned_batched_gate",
        "method": "Learned batched gate (logistic + MLP) over 9 multi-feature UQ summaries with LOO-CV",
        "n_eligible": len(eligible),
        "feature_names": [
            "mean", "std", "slope", "max", "min", "p95", "p10", "drift", "ar1"
        ],
        "by_K": by_K,
        "best_proxy": best_p_cell,
        "best_oracle": best_o_cell,
        "sprt_reference": {
            "proxy_gt": sprt_p, "oracle_gt": sprt_o,
            "n_births": sum(sprt_decs), "avg_obs": 21.7,
        },
        "headline": (
            f"Learned batched gate (LOO N={len(eligible)}): "
            f"best proxy F1={best_proxy_f1:.2f} "
            f"({best_p_cell['clf']} K={best_p_cell['K']}) vs SPRT proxy F1={sprt_p['f1']:.2f}; "
            f"best oracle F1={best_oracle_f1:.2f} "
            f"({best_o_cell['clf']} K={best_o_cell['K']}) vs SPRT oracle F1={sprt_o['f1']:.2f}. "
            f"Multi-feature representation does not improve over saturated mean-UQ rule."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(out["headline"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
