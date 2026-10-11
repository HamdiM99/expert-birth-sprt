"""Rebuild seed{N}/summary.json with a configurable needs_birth threshold.

Usage:
  python rebuild_summary.py --results-dir ./results --threshold 0.05
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import confusion as cm_mod
from eval import counterfactual as cf_mod


def rebuild_seed(seed_dir: Path, threshold: float):
    cf_results = json.loads((seed_dir / "counterfactual.json").read_text())
    decisions = json.loads((seed_dir / "decisions.json").read_text())
    summary = json.loads((seed_dir / "summary.json").read_text())

    gt = {}
    for cf in cf_results:
        cid = cf.get("cluster_id")
        if cid is None: continue
        gt[cid] = cf_mod.label_needs_birth(cf, improvement_threshold=threshold)

    cm_eager = cm_mod.confusion(decisions["eager"], gt)
    cm_obs = cm_mod.confusion(decisions["observation"], gt)
    sweep = cm_mod.threshold_sweep(
        decisions["observation"], gt,
        thresholds=[0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80])

    summary["ground_truth"] = {str(k): bool(v) for k, v in gt.items()}
    summary["needs_birth_threshold"] = threshold
    summary["confusion_eager"] = cm_eager
    summary["confusion_observation"] = cm_obs
    summary["threshold_sweep"] = sweep
    summary["delta_distribution"] = sorted(
        [{"cluster_id": cf.get("cluster_id"), "delta": cf.get("delta"),
          "dominant_block": cf.get("dominant_block")} for cf in cf_results
         if cf.get("delta") is not None],
        key=lambda x: -(x["delta"] or 0))

    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return cm_eager, cm_obs, summary["delta_distribution"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    seed_dirs = sorted([d for d in rdir.glob("seed*") if d.is_dir()])
    print(f"Rebuilding {len(seed_dirs)} seed(s) with needs_birth threshold = {args.threshold}\n")
    all_deltas = []
    for sd in seed_dirs:
        try:
            cm_e, cm_o, deltas = rebuild_seed(sd, args.threshold)
            print(f"[{sd.name}] eager  P={cm_e['precision']:.2f} R={cm_e['recall']:.2f} → {cm_e['n_births']} births  "
                  f"obs P={cm_o['precision']:.2f} R={cm_o['recall']:.2f} → {cm_o['n_births']} births")
            for d in deltas:
                print(f"  cluster {d['cluster_id']} ({d['dominant_block']}): Δ={d['delta']:.3f}")
                all_deltas.append(d["delta"])
        except FileNotFoundError as e:
            print(f"[{sd.name}] skipped: {e}")
    if all_deltas:
        all_deltas.sort()
        print(f"\nDelta distribution ({len(all_deltas)} clusters across all seeds):")
        print(f"  min={min(all_deltas):.3f}  median={all_deltas[len(all_deltas)//2]:.3f}  max={max(all_deltas):.3f}")
        print(f"  ≥0.03: {sum(1 for d in all_deltas if d >= 0.03)}/{len(all_deltas)}")
        print(f"  ≥0.05: {sum(1 for d in all_deltas if d >= 0.05)}/{len(all_deltas)}")
        print(f"  ≥0.10: {sum(1 for d in all_deltas if d >= 0.10)}/{len(all_deltas)}")


if __name__ == "__main__":
    sys.exit(main())
