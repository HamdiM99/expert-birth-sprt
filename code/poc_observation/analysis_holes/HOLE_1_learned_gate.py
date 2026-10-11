"""HOLE 1 — Learned-gate baseline.

Compare logistic regression and shallow MLP gates against SPRT on the
deployment proposals (only labeled source: 7 proposals, 5 positive 2 negative
in per_proposal_traces.jsonl). Controlled-experiment ground truth is all-False
in every seed, so AUROC is undefined there; we use the deployment proposals
with leave-one-out CV as the only labeled regime, and compare against
SPRT TNR=0.36 (mean across seeds at tau=0.8) reported in the paper.

Output: analysis/HOLE_1_learned_gate.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
ENS = ROOT.parent / "poc_observation" / "results_ensemble"
OUTJSON = ROOT / "analysis" / "HOLE_1_learned_gate.json"


def features_from_uq(uq):
    u = np.asarray(uq, dtype=float)
    return np.array([u.mean(), u.std(), u.max(), float(len(u)),
                     np.percentile(u, 25), np.percentile(u, 75)])


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    P = y.sum(); N = (~y).sum()
    if P == 0 or N == 0: return None
    order = np.argsort(-s); y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(~y)
    tpr = tps / P; fpr = fps / N
    fpr = np.concatenate(([0.0], fpr, [1.0])); tpr = np.concatenate(([0.0], tpr, [1.0]))
    return float(np.trapz(tpr, fpr))


def main():
    items = [json.loads(l) for l in TRACES.open() if l.strip()]
    items = [x for x in items if x.get("uq_observations") and x.get("needs_birth_gt") is not None]
    print(f"deployment labeled proposals: {len(items)}")

    X = np.vstack([features_from_uq(x["uq_observations"]) for x in items])
    y = np.array([bool(x["needs_birth_gt"]) for x in items])
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    print(f"y distribution: pos={n_pos}, neg={n_neg}")

    # Also try to enrich with controlled-experiment clusters: every cluster has
    # GT=False in all 3 seeds. Adding them would create perfect class imbalance
    # (only negatives) — AUROC undefined. We document the issue and proceed with
    # deployment-only LOO. But we still measure the # extra clusters available.
    extra = 0
    for s in (0, 1, 2):
        f = ENS / f"seed{s}" / "summary.json"
        if f.exists():
            d = json.loads(f.read_text())
            extra += len(d.get("ground_truth", {}))

    # Standardise
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    # ── (a) Logistic regression — closed-form via IRLS ────────────────
    def fit_logreg(Xtr, ytr, n_iter=200, l2=1.0):
        n, d = Xtr.shape
        Xa = np.hstack([Xtr, np.ones((n, 1))])
        w = np.zeros(Xa.shape[1])
        for _ in range(n_iter):
            z = Xa @ w
            p = 1 / (1 + np.exp(-z))
            grad = Xa.T @ (p - ytr) + l2 * np.r_[w[:-1], 0]
            W = np.diag(p * (1 - p) + 1e-9)
            H = Xa.T @ W @ Xa + l2 * np.eye(Xa.shape[1])
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w_new = w - step
            if np.linalg.norm(w_new - w) < 1e-7: break
            w = w_new
        return w

    def pred_logreg(w, Xte):
        Xa = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
        return 1 / (1 + np.exp(-(Xa @ w)))

    # ── (b) MLP — 1 hidden, dim 32, full-batch GD with early stopping ─
    def fit_mlp(Xtr, ytr, hidden=32, lr=0.05, max_epochs=400, val_frac=0.25, seed=0):
        rng = np.random.default_rng(seed)
        n = len(ytr)
        # Need at least 1 of each class in val. If too few, skip val and just train fixed epochs.
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac))
        val = idx[:n_val]; tr = idx[n_val:]
        if ytr[val].sum() == 0 or ytr[val].sum() == len(val) or len(tr) < 2:
            tr = np.arange(n); val = np.arange(n)  # degenerate; fall back to train-on-all
        d = Xtr.shape[1]
        W1 = rng.normal(0, 0.5, (d, hidden)); b1 = np.zeros(hidden)
        W2 = rng.normal(0, 0.5, (hidden, 1)); b2 = np.zeros(1)
        best_val = 1e18; best = (W1.copy(), b1.copy(), W2.copy(), b2.copy()); patience = 30; bad = 0
        for ep in range(max_epochs):
            xt = Xtr[tr]; yt = ytr[tr].astype(float)
            h = np.maximum(0, xt @ W1 + b1)
            z = (h @ W2 + b2).ravel()
            p = 1 / (1 + np.exp(-z))
            # backward
            dz = (p - yt) / len(yt)
            dW2 = h.T @ dz[:, None]; db2 = dz.sum(keepdims=True)
            dh = dz[:, None] @ W2.T
            dh[h <= 0] = 0
            dW1 = xt.T @ dh; db1 = dh.sum(axis=0)
            W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
            # val loss
            xv = Xtr[val]; yv = ytr[val].astype(float)
            hv = np.maximum(0, xv @ W1 + b1)
            zv = (hv @ W2 + b2).ravel()
            pv = np.clip(1 / (1 + np.exp(-zv)), 1e-9, 1 - 1e-9)
            vloss = -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)).mean()
            if vloss < best_val - 1e-5:
                best_val = vloss; best = (W1.copy(), b1.copy(), W2.copy(), b2.copy()); bad = 0
            else:
                bad += 1
                if bad > patience: break
        return best

    def pred_mlp(model, Xte):
        W1, b1, W2, b2 = model
        h = np.maximum(0, Xte @ W1 + b1)
        z = (h @ W2 + b2).ravel()
        return 1 / (1 + np.exp(-z))

    # Leave-one-out (per proposal)
    n = len(y)
    p_log = np.zeros(n); p_mlp = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        Xtr, ytr = Xs[mask], y[mask]
        Xte = Xs[i:i+1]
        # Logistic
        w = fit_logreg(Xtr, ytr.astype(float), l2=1.0)
        p_log[i] = pred_logreg(w, Xte)[0]
        # MLP — average over 3 seeds for stability
        ps = []
        for sd_ in (0, 1, 2):
            mdl = fit_mlp(Xtr, ytr, seed=sd_)
            ps.append(pred_mlp(mdl, Xte)[0])
        p_mlp[i] = float(np.mean(ps))

    log_auroc = auroc(p_log, y)
    mlp_auroc = auroc(p_mlp, y)
    # also report mean-uq-only AUROC for sanity
    base_auroc = auroc(X[:, 0], y)

    print(f"AUROC logistic={log_auroc}, MLP={mlp_auroc}, baseline_meanUQ={base_auroc}")

    # SPRT TNR (paper-reported mean across seeds at tau=0.8): 0.36 — supplied
    sprt_tnr_paper = 0.36

    out = {
        "n_proposals_used": n,
        "n_features": X.shape[1],
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_clusters_extra_controlled_all_false": extra,
        "logistic_auroc": log_auroc,
        "mlp_auroc": mlp_auroc,
        "mean_uq_auroc": base_auroc,
        "sprt_tnr_paper_mean_seeds_tau08": sprt_tnr_paper,
        "note": ("Controlled-experiment GT is all-False in every seed, so "
                 "AUROC is undefined there. Deployment LOO is the only "
                 "labeled regime; N=7 (5 pos, 2 neg). "
                 "Treat this as a sanity check, not a full benchmark."),
        "predictions_logistic": p_log.tolist(),
        "predictions_mlp": p_mlp.tolist(),
        "labels": y.tolist(),
    }
    OUTJSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTJSON}")


if __name__ == "__main__":
    sys.exit(main())
