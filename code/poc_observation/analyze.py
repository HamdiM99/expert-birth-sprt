"""Aggregate per-seed results into paper-ready tables + figures.

Inputs:  results/seed{N}/summary.json  ×  N seeds
Outputs: results/aggregate.json + results/figures/{fig1,fig2,fig3}.png
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    seed_dirs = sorted([d for d in rdir.glob("seed*") if d.is_dir()])
    if not seed_dirs:
        print(f"No seed* dirs under {rdir}"); return 1

    per_seed = []
    for sd in seed_dirs:
        sj = sd / "summary.json"
        if sj.exists():
            per_seed.append(json.loads(sj.read_text()))
    if not per_seed:
        print("No summary.json found"); return 1
    print(f"Aggregating {len(per_seed)} seed(s)")

    # ── Aggregate confusion ──────────────────────────────────────────────
    from eval.confusion import aggregate_seeds
    eager_cms = [s["confusion_eager"] for s in per_seed]
    obs_cms = [s["confusion_observation"] for s in per_seed]
    eager_agg = aggregate_seeds(eager_cms)
    obs_agg = aggregate_seeds(obs_cms)

    # ── Threshold sweep aggregation ──────────────────────────────────────
    thresholds = [t["threshold"] for t in per_seed[0]["threshold_sweep"]]
    sweep_agg = []
    for ti, thr in enumerate(thresholds):
        precs = [s["threshold_sweep"][ti]["precision"] for s in per_seed]
        recs = [s["threshold_sweep"][ti]["recall"] for s in per_seed]
        births = [s["threshold_sweep"][ti]["n_births"] for s in per_seed]
        n = len(per_seed)
        def m(v): return sum(v) / n
        def s(v):
            avg = m(v); return (sum((x - avg) ** 2 for x in v) / n) ** 0.5
        sweep_agg.append({
            "threshold": thr,
            "precision_mean": m(precs), "precision_std": s(precs),
            "recall_mean": m(recs), "recall_std": s(recs),
            "n_births_mean": m(births), "n_births_std": s(births),
        })

    aggregate = {
        "n_seeds": len(per_seed),
        "eager": eager_agg,
        "observation": obs_agg,
        "threshold_sweep": sweep_agg,
        "individual": per_seed,
    }
    (rdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"Wrote {rdir/'aggregate.json'}")

    # ── Figures ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        figdir = rdir / "figures"; figdir.mkdir(exist_ok=True)

        # Fig 1: Pareto threshold sweep
        thr = [s["threshold"] for s in sweep_agg]
        births_m = [s["n_births_mean"] for s in sweep_agg]
        births_s = [s["n_births_std"] for s in sweep_agg]
        rec_m = [s["recall_mean"] for s in sweep_agg]
        rec_s = [s["recall_std"] for s in sweep_agg]
        prec_m = [s["precision_mean"] for s in sweep_agg]
        prec_s = [s["precision_std"] for s in sweep_agg]
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.set_xlabel("UQ gate threshold")
        ax1.errorbar(thr, births_m, yerr=births_s, fmt="o-", color="#fa6e6e",
                     capsize=3, label="LoRA trainings")
        ax1.set_ylabel("# LoRA trainings", color="#fa6e6e")
        ax1.axhline(eager_agg["n_births_mean"], ls="--", color="#fa6e6e", alpha=0.4,
                    label=f"Eager baseline (μ={eager_agg['n_births_mean']:.1f})")
        ax2 = ax1.twinx()
        ax2.errorbar(thr, rec_m, yerr=rec_s, fmt="s-", color="#3fb950",
                     capsize=3, label="Recall")
        ax2.errorbar(thr, prec_m, yerr=prec_s, fmt="^-", color="#388bfd",
                     capsize=3, label="Precision")
        ax2.set_ylabel("Score"); ax2.set_ylim(-0.05, 1.05)
        ax1.legend(loc="upper left"); ax2.legend(loc="lower left")
        plt.title(f"Observation gate trade-off (mean ± std, {len(per_seed)} seeds)")
        fig.tight_layout(); plt.savefig(figdir / "fig1_pareto.png", dpi=150); plt.close()
        print(f"Wrote {figdir/'fig1_pareto.png'}")

        # Fig 3: confusion matrices side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
        for ax, agg, title in [(axes[0], eager_agg, "Eager"),
                               (axes[1], obs_agg, "Observation (UQ≥0.55)")]:
            mat = np.array([[agg["tn_mean"], agg["fp_mean"]],
                            [agg["fn_mean"], agg["tp_mean"]]])
            std = np.array([[agg["tn_std"], agg["fp_std"]],
                            [agg["fn_std"], agg["tp_std"]]])
            ax.imshow(mat, cmap="Blues", aspect="equal")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{mat[i,j]:.1f}±{std[i,j]:.1f}",
                            ha="center", va="center",
                            color="white" if mat[i,j] > mat.max()/2 else "black",
                            fontsize=12, fontweight="bold")
            ax.set_xticks([0,1]); ax.set_xticklabels(["pred ¬birth", "pred birth"])
            ax.set_yticks([0,1]); ax.set_yticklabels(["gt ¬needs", "gt needs"])
            ax.set_title(f"{title}\nP={agg['precision_mean']:.2f}±{agg['precision_std']:.2f}  "
                         f"R={agg['recall_mean']:.2f}±{agg['recall_std']:.2f}  "
                         f"F1={agg['f1_mean']:.2f}±{agg['f1_std']:.2f}")
        fig.suptitle(f"Eager vs Observation — {len(per_seed)} seeds")
        fig.tight_layout(); plt.savefig(figdir / "fig3_confusion.png", dpi=150); plt.close()
        print(f"Wrote {figdir/'fig3_confusion.png'}")

        # Fig 2: per-seed per-cluster trace (uses observation decisions)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {"BIRTH": "#3fb950", "SKIP": "#8b949e"}
        for sj in seed_dirs:
            d = json.loads((sj / "decisions.json").read_text())
            for od in d.get("observation", []):
                uqs = od.get("uq_observations", [])
                if not uqs: continue
                running = np.cumsum(uqs) / np.arange(1, len(uqs) + 1)
                ax.plot(range(1, len(running) + 1), running,
                        color=colors.get(od["decision"], "#888"),
                        alpha=0.5, lw=1)
        ax.axhline(0.55, ls="--", color="red", alpha=0.5, label="gate UQ=0.55")
        ax.set_xlabel("Observation #"); ax.set_ylabel("Running mean UQ")
        ax.set_title(f"Per-cluster observation traces ({len(per_seed)} seeds, all clusters)")
        ax.legend()
        fig.tight_layout(); plt.savefig(figdir / "fig2_traces.png", dpi=150); plt.close()
        print(f"Wrote {figdir/'fig2_traces.png'}")

    except ImportError:
        print("matplotlib unavailable — figures skipped")

    # ── Markdown table ───────────────────────────────────────────────────
    md = ["# POC results — Eager vs Observation\n",
          f"Seeds: {len(per_seed)}\n",
          "## Confusion (mean ± std)\n",
          "| Gate | TP | FP | FN | TN | Precision | Recall | F1 | LoRA trains |",
          "|---|---|---|---|---|---|---|---|---|"]
    for label, agg in [("Eager", eager_agg), ("Observation", obs_agg)]:
        md.append(
            f"| {label} | {agg['tp_mean']:.1f}±{agg['tp_std']:.1f} | "
            f"{agg['fp_mean']:.1f}±{agg['fp_std']:.1f} | "
            f"{agg['fn_mean']:.1f}±{agg['fn_std']:.1f} | "
            f"{agg['tn_mean']:.1f}±{agg['tn_std']:.1f} | "
            f"{agg['precision_mean']:.2f}±{agg['precision_std']:.2f} | "
            f"{agg['recall_mean']:.2f}±{agg['recall_std']:.2f} | "
            f"{agg['f1_mean']:.2f}±{agg['f1_std']:.2f} | "
            f"{agg['n_births_mean']:.1f}±{agg['n_births_std']:.1f} |")
    saved = eager_agg["n_births_mean"] - obs_agg["n_births_mean"]
    rate = saved / max(eager_agg["n_births_mean"], 1) * 100
    md.append(f"\n**LoRA trainings saved: {saved:.1f}/{eager_agg['n_births_mean']:.1f} ({rate:.0f}% rejection)**")
    (rdir / "RESULTS.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {rdir/'RESULTS.md'}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
