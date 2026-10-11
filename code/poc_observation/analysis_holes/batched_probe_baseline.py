"""Batched-probe baseline gate on real deployment UQ traces.

Compares:
  - Eager (always BIRTH on candidate cluster)
  - SPRT gate (sequential, with early stopping)
  - Batched-probe gate (collect K shadow queries, then mean-threshold)

Outputs: figures/batched_vs_sprt.png, analysis/batched_probe_results.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
FIGDIR = ROOT / "figures"
OUTJSON = ROOT / "analysis" / "batched_probe_results.json"


def llr_step(u, mu0, mu1, sigma):
    return ((mu1 - mu0) / sigma**2) * u - (mu1**2 - mu0**2) / (2 * sigma**2)


def sprt(seq, mu0, mu1, sigma, alpha=0.05, beta=0.05, n_min=20, n_max=50):
    A = math.log((1 - beta) / alpha); B = math.log(beta / (1 - alpha))
    s = 0.0
    for n, u in enumerate(seq[:n_max], start=1):
        s += llr_step(u, mu0, mu1, sigma)
        if n >= n_min:
            if s >= A: return ("BIRTH", n)
            if s <= B: return ("SKIP", n)
    if not seq: return ("SKIP", 0)
    return ("BIRTH" if np.mean(seq[:n_max]) >= (mu0 + mu1) / 2 else "SKIP",
            min(len(seq), n_max))


def batched_probe(seq, K, threshold):
    """Collect K observations, threshold their mean."""
    if len(seq) < K: K = len(seq)
    return ("BIRTH" if np.mean(seq[:K]) >= threshold else "SKIP", K)


def confusion(decisions, gt):
    tp = fp = fn = tn = 0
    for d, g in zip(decisions, gt):
        if d == "BIRTH" and g: tp += 1
        elif d == "BIRTH" and not g: fp += 1
        elif d == "SKIP" and g: fn += 1
        else: tn += 1
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
            "n_births": sum(1 for d in decisions if d == "BIRTH")}


def main():
    items = [json.loads(l) for l in TRACES.open()]
    items = [x for x in items if x.get("uq_observations") and len(x["uq_observations"]) >= 20]
    print(f"loaded {len(items)} usable proposals")

    all_uq = np.concatenate([np.array(x["uq_observations"]) for x in items])
    mu0 = float(np.percentile(all_uq, 40))
    mu1 = float(np.percentile(all_uq, 60))
    sigma = float(all_uq.std())
    threshold_mid = (mu0 + mu1) / 2
    print(f"derived μ0={mu0:.3f} μ1={mu1:.3f} σ={sigma:.3f} threshold={threshold_mid:.3f}")

    gt = [bool(x.get("needs_birth_gt")) for x in items]
    eager_dec = ["BIRTH"] * len(items)
    sprt_dec = [sprt(x["uq_observations"], mu0, mu1, sigma)[0] for x in items]
    sprt_n = [sprt(x["uq_observations"], mu0, mu1, sigma)[1] for x in items]

    res = {
        "eager": confusion(eager_dec, gt),
        "sprt": confusion(sprt_dec, gt),
        "sprt_avg_obs": float(np.mean(sprt_n)),
        "batched_probe": {},
    }
    for K in (10, 15, 20, 30, 50):
        bp_dec = [batched_probe(x["uq_observations"], K, threshold_mid)[0] for x in items]
        res["batched_probe"][f"K={K}"] = {**confusion(bp_dec, gt), "n_obs": K}

    print("\n=== Comparison ===")
    print(f"{'Gate':<22} {'P':>5} {'R':>5} {'F1':>5} {'#births':>8} {'avg_obs':>8}")
    print(f"{'-'*60}")

    def f1(c):
        p = c["precision"]; r = c["recall"]
        return 2*p*r/(p+r) if (p+r) > 0 else 0.0

    print(f"{'Eager (no obs)':<22} {res['eager']['precision']:>5.2f} {res['eager']['recall']:>5.2f} "
          f"{f1(res['eager']):>5.2f} {res['eager']['n_births']:>8} {0:>8}")
    print(f"{'SPRT (sequential)':<22} {res['sprt']['precision']:>5.2f} {res['sprt']['recall']:>5.2f} "
          f"{f1(res['sprt']):>5.2f} {res['sprt']['n_births']:>8} {res['sprt_avg_obs']:>8.1f}")
    for K, c in res["batched_probe"].items():
        print(f"{'Batched probe ' + K:<22} {c['precision']:>5.2f} {c['recall']:>5.2f} "
              f"{f1(c):>5.2f} {c['n_births']:>8} {c['n_obs']:>8}")

    OUTJSON.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUTJSON}")

    # ── Plot: Pareto comparison ──
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pts = []
    pts.append(("Eager",
                res["eager"]["n_births"], res["eager"]["recall"],
                f1(res["eager"]), 0, "#c0392b", "X"))
    pts.append(("SPRT (ours)",
                res["sprt"]["n_births"], res["sprt"]["recall"],
                f1(res["sprt"]), res["sprt_avg_obs"], "#27ae60", "o"))
    for K, c in res["batched_probe"].items():
        pts.append((f"Batched-{K}", c["n_births"], c["recall"], f1(c), c["n_obs"],
                    "#388bfd", "s"))

    for label, n, r, F, obs, col, mk in pts:
        ax.scatter(n, r, s=140, c=col, marker=mk, edgecolor="black", lw=0.8, alpha=0.9, label=label)
        ax.annotate(label, (n, r), xytext=(6, -3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("# LoRA trainings")
    ax.set_ylabel("Recall")
    ax.set_title(f"Sequential SPRT vs batched-probe vs eager — deployment {len(items)} proposals")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "batched_vs_sprt.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"wrote {FIGDIR/'batched_vs_sprt.png'}")


if __name__ == "__main__":
    sys.exit(main())
