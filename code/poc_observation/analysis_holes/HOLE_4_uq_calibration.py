"""HOLE 4 — Isotonic calibration of UQ -> P(NeedsBirth) on deployment.

For each of the 7 deployment proposals (with `needs_birth_gt`), use leave-one-out
cross-validation. The raw 'predicted probability' from UQ is taken as the
min-max scaled mean_uq. Then fit an isotonic regression on the n-1 train
points and evaluate Brier score on the held-out point. Compare against the
uncalibrated raw min-max scaled mean_uq.

Saves a reliability diagram to figures/uq_calibration.png.

N=7 → barely meaningful. We state this explicitly.

Output: analysis/HOLE_4_uq_calibration.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
FIGDIR = ROOT / "figures"
OUTJSON = ROOT / "analysis" / "HOLE_4_uq_calibration.json"


def main():
    from sklearn.isotonic import IsotonicRegression

    items = [json.loads(l) for l in TRACES.open() if l.strip()]
    items = [x for x in items if x.get("uq_observations") and x.get("needs_birth_gt") is not None]
    print(f"deployment labeled proposals: {len(items)}")

    uq_mean = np.array([float(np.mean(x["uq_observations"])) for x in items])
    y = np.array([bool(x["needs_birth_gt"]) for x in items], dtype=int)

    # Min-max scale the raw signal to [0,1] for use as a probability proxy.
    lo, hi = uq_mean.min(), uq_mean.max()
    raw = (uq_mean - lo) / (hi - lo + 1e-9)

    n = len(y)
    p_iso = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        # We fit on the GLOBAL min-max scale of training points (re-derive)
        x_tr = uq_mean[mask]
        lo_t, hi_t = x_tr.min(), x_tr.max()
        x_tr_s = (x_tr - lo_t) / (hi_t - lo_t + 1e-9)
        x_te_s = (uq_mean[i] - lo_t) / (hi_t - lo_t + 1e-9)
        x_te_s = float(np.clip(x_te_s, 0.0, 1.0))
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(x_tr_s, y[mask])
        p_iso[i] = float(ir.predict([x_te_s])[0])

    brier_raw = float(np.mean((raw - y) ** 2))
    brier_iso = float(np.mean((p_iso - y) ** 2))
    print(f"Brier raw={brier_raw:.4f}, isotonic LOO={brier_iso:.4f}")

    # ── Reliability diagram ──────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfectly calibrated")
    # Sort by p
    for arr, name, color in [(raw, "raw min-max UQ", "#c0392b"),
                             (p_iso, "isotonic (LOO)", "#27ae60")]:
        order = np.argsort(arr)
        # bin into thirds (N=7 → 3 bins)
        n_bins = 3
        edges = np.linspace(0, 1, n_bins + 1)
        bx, by = [], []
        for k in range(n_bins):
            mk = (arr >= edges[k]) & (arr <= edges[k+1])
            if mk.sum() < 1: continue
            bx.append(arr[mk].mean()); by.append(y[mk].mean())
        ax.plot(bx, by, "o-", color=color, lw=2, ms=8, label=f"{name}")
        # individual points
        ax.scatter(arr, y + np.random.uniform(-0.02, 0.02, len(y)), color=color, alpha=0.4, s=22)
    ax.set_xlabel("Predicted P(NeedsBirth)")
    ax.set_ylabel("Empirical fraction NeedsBirth=1")
    ax.set_title(f"UQ→P(NeedsBirth) reliability (N={n}, Brier raw={brier_raw:.3f}, iso={brier_iso:.3f})",
                 fontsize=10)
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.06, 1.08)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    out_fig = FIGDIR / "uq_calibration.png"
    plt.savefig(out_fig, dpi=160, bbox_inches="tight"); plt.close()
    print(f"wrote {out_fig}")

    out = {
        "n_proposals": int(n),
        "n_pos": int(y.sum()),
        "n_neg": int((~y.astype(bool)).sum()),
        "brier_uncalibrated_raw_minmax": brier_raw,
        "brier_isotonic_LOO": brier_iso,
        "improvement_absolute": brier_raw - brier_iso,
        "raw_predictions": raw.tolist(),
        "isotonic_predictions": p_iso.tolist(),
        "labels": y.tolist(),
        "figure": str(out_fig),
        "honesty_caveat": ("N=7 with 5 positives and 2 negatives — these "
                           "Brier numbers are barely meaningful and serve "
                           "only as a smoke test that an out-of-the-box "
                           "isotonic post-hoc calibration could be a "
                           "trivial drop-in if more labels were collected."),
    }
    OUTJSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTJSON}")


if __name__ == "__main__":
    sys.exit(main())
