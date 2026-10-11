"""AUROC/PR of token_entropy as a discriminator of needs_birth_gt.

Operates on deployment traces (per_proposal_traces.jsonl). For each proposal
we have a sequence of UQ observations and a binary GT label. We score each
proposal by mean(uq_observations) and compute the discrimination metrics.

This addresses reviewer T4 quantitatively: "is the UQ signal even predictive
of need before SPRT bothers to gate it?"

Outputs:
  figures/uq_roc_pr.png
  analysis/uq_discrimination.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
FIGDIR = ROOT / "figures"
OUTJSON = ROOT / "analysis" / "uq_discrimination.json"


def roc_pr(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-scores)  # descending
    s = scores[order]; y = labels[order]
    P = y.sum(); N = (~y).sum()
    if P == 0 or N == 0:
        return None
    tps = np.cumsum(y); fps = np.cumsum(~y)
    tpr = tps / P; fpr = fps / N
    prec = tps / (tps + fps + 1e-12); rec = tpr
    # prepend (0,0)/(1,0) for ROC, (1,0) for PR
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    auroc = float(np.trapz(tpr, fpr))
    # AUPRC via trapezoid on (rec, prec) sorted by rec
    rp = sorted(zip(rec.tolist(), prec.tolist()))
    rs = [r for r, _ in rp]; ps = [p for _, p in rp]
    auprc = float(np.trapz(ps, rs))
    return {"auroc": auroc, "auprc": auprc,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "rec": rs, "prec": ps,
            "n_pos": int(P), "n_neg": int(N)}


def main():
    items = [json.loads(l) for l in TRACES.open()]
    items = [x for x in items if x.get("uq_observations")]
    print(f"loaded {len(items)} proposals")

    # Score 1: mean UQ over the full sequence
    mean_scores = [float(np.mean(x["uq_observations"])) for x in items]
    # Score 2: max UQ
    max_scores = [float(np.max(x["uq_observations"])) for x in items]
    # Score 3: median first-K=20 UQ (early signal — what SPRT sees by n_min)
    k20 = [float(np.mean(x["uq_observations"][:20])) for x in items
           if len(x["uq_observations"]) >= 20]
    items_k20 = [x for x in items if len(x["uq_observations"]) >= 20]
    labels = [bool(x.get("needs_birth_gt")) for x in items]
    labels_k20 = [bool(x.get("needs_birth_gt")) for x in items_k20]

    res = {
        "n_proposals": len(items),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
    }
    for name, s, lab in [("mean_full", mean_scores, labels),
                         ("max_full", max_scores, labels),
                         ("mean_first20", k20, labels_k20)]:
        r = roc_pr(s, lab)
        if r is None:
            print(f"{name}: degenerate (only one class)")
            res[name] = None
        else:
            print(f"{name}: AUROC={r['auroc']:.3f}  AUPRC={r['auprc']:.3f}  "
                  f"(n+={r['n_pos']}, n-={r['n_neg']})")
            res[name] = {"auroc": r["auroc"], "auprc": r["auprc"],
                         "n_pos": r["n_pos"], "n_neg": r["n_neg"]}

    # Plot ROC + PR if we have at least one usable curve
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"mean_full": "#27ae60", "max_full": "#c0392b",
              "mean_first20": "#388bfd"}
    for name, s, lab in [("mean_full", mean_scores, labels),
                         ("max_full", max_scores, labels),
                         ("mean_first20", k20, labels_k20)]:
        r = roc_pr(s, lab)
        if r is None: continue
        ax1.plot(r["fpr"], r["tpr"], color=colors[name],
                 label=f"{name}  AUROC={r['auroc']:.2f}", lw=2)
        ax2.plot(r["rec"], r["prec"], color=colors[name],
                 label=f"{name}  AUPRC={r['auprc']:.2f}", lw=2)
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC: token_entropy → needs_birth")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    base = sum(labels) / max(len(labels), 1)
    ax2.axhline(base, ls="--", color="gray", alpha=0.5,
                label=f"prevalence={base:.2f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("PR: token_entropy → needs_birth")
    ax2.set_ylim(-0.02, 1.05); ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
    plt.suptitle(f"UQ discrimination on deployment ({len(items)} proposals)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGDIR / "uq_roc_pr.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"wrote {FIGDIR/'uq_roc_pr.png'}")

    OUTJSON.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUTJSON}")


if __name__ == "__main__":
    sys.exit(main())
