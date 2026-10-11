"""HOLE 2 — u_min sensitivity sweep on cluster-detection filter.

For each u_min in {0.05, 0.075, 0.10, 0.15, 0.20, 0.30}:
  - count proposals (clusters) that survive the filter `mean_uq >= u_min`
  - compute mean cluster size (n_obs) among survivors
  - count how many of the 7 originally surviving proposals (default u_min=0.30)
    would still appear (intersection on cluster_id+id)

Operates on per_proposal_traces.jsonl. We use only proposals with non-empty
uq_observations (i.e. those that triggered observation in deployment).

Output: analysis/HOLE_2_umin_sensitivity.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
OUTJSON = ROOT / "analysis" / "HOLE_2_umin_sensitivity.json"


def main():
    items = [json.loads(l) for l in TRACES.open() if l.strip()]
    obs_items = [x for x in items if x.get("uq_observations")]
    print(f"loaded {len(items)} total, {len(obs_items)} with uq_observations")

    sweep = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
    # The 7 "original" proposals are exactly obs_items (each survived the
    # default u_min=0.30 filter on the deployment side).
    orig_ids = {x["id"] for x in obs_items}

    results = []
    for um in sweep:
        survivors = [x for x in obs_items if x.get("mean_uq", 0.0) >= um]
        sizes = [x["n_obs"] for x in survivors]
        n_orig_kept = sum(1 for x in survivors if x["id"] in orig_ids)
        results.append({
            "u_min": um,
            "n_clusters": len(survivors),
            "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
            "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
            "n_of_original_7_kept": n_orig_kept,
            "kept_ids": sorted([x["id"] for x in survivors]),
        })
        print(f"u_min={um}: n={len(survivors)}, mean_size={np.mean(sizes) if sizes else 0:.1f}, "
              f"orig_kept={n_orig_kept}/7")

    out = {
        "n_observed_proposals": len(obs_items),
        "default_u_min": 0.30,
        "sweep": results,
        "note": ("Filter applied is `mean_uq >= u_min` on deployment "
                 "proposals that already received observations. The "
                 "`mean_uq` field is the deployment-time mean across "
                 "the proposal's UQ observations."),
    }
    OUTJSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTJSON}")


if __name__ == "__main__":
    sys.exit(main())
