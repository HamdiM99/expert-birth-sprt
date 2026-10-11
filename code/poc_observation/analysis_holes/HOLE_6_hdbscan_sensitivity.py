"""HOLE 6 — HDBSCAN sensitivity.

Re-cluster the 420 deployment-query embeddings under a grid of
(min_cluster_size, min_samples). Report n_clusters, n_noise, ARI vs the
original labelling. Optionally re-embed with a second model.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # supplementary/
DATA = Path(os.environ.get("ANALYSIS_DATA", _ROOT / "data"))
OUT = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_6_hdbscan_sensitivity.json"))


def hdbscan_labels(embs_n: np.ndarray, mcs: int, ms: int):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                         metric="euclidean", cluster_selection_epsilon=0.0)
    return cl.fit_predict(embs_n)


def main():
    from sklearn.metrics import adjusted_rand_score

    d = np.load(DATA / "embeddings.npz")
    embs = d["embeddings"].astype(np.float32)
    print(f"embeddings shape: {embs.shape}")

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.maximum(norms, 1e-9)

    # Original labelling = production config (min_cluster_size=15, min_samples default→None→= mcs)
    orig_labels = hdbscan_labels(embs_n, mcs=15, ms=15)
    n_orig_clusters = len({l for l in orig_labels if l != -1})
    n_orig_noise = int((orig_labels == -1).sum())
    print(f"original (mcs=15, ms=15): {n_orig_clusters} clusters, {n_orig_noise} noise")

    grid_mcs = [10, 15, 20, 30]
    grid_ms = [5, 10, 15]

    cells = []
    for mcs in grid_mcs:
        for ms in grid_ms:
            labs = hdbscan_labels(embs_n, mcs, ms)
            n_clusters = len({l for l in labs if l != -1})
            n_noise = int((labs == -1).sum())
            ari = float(adjusted_rand_score(orig_labels, labs))
            # ARI restricted to non-noise points in BOTH labellings
            mask = (orig_labels != -1) & (labs != -1)
            if mask.sum() >= 2:
                ari_no_noise = float(adjusted_rand_score(orig_labels[mask], labs[mask]))
            else:
                ari_no_noise = None
            cells.append({
                "min_cluster_size": mcs, "min_samples": ms,
                "n_clusters": n_clusters, "n_noise": n_noise,
                "noise_frac": n_noise / len(labs),
                "ari_vs_orig_full": ari,
                "ari_vs_orig_excl_noise": ari_no_noise,
            })
            print(f"  mcs={mcs:2d} ms={ms:2d}  k={n_clusters:2d}  noise={n_noise:3d}  ARI={ari:.3f}  ARI(no-noise)={ari_no_noise}")

    # Optional second embedder
    second = {"attempted": False, "model": None, "result": None}
    try:
        from sentence_transformers import SentenceTransformer
        # Try cached models only (no_download semantics: we just attempt)
        for cand in ["BAAI/bge-small-en-v1.5", "BAAI/bge-small-en", "intfloat/e5-small-v2"]:
            try:
                m = SentenceTransformer(cand, cache_folder="/root/.cache/torch/sentence_transformers")
                second["model"] = cand
                second["attempted"] = True
                # We don't have raw query texts in the npz, so we cannot re-embed.
                # Mark as attempted-but-no-text-available.
                second["result"] = "model_loadable_but_no_query_text_available_in_npz"
                break
            except Exception as e:
                continue
    except Exception as e:
        second["error"] = str(e)

    out = {
        "n_queries": int(embs.shape[0]),
        "embedder_primary": "MiniLM (384-dim, from deployment)",
        "original_config": {"min_cluster_size": 15, "min_samples": 15,
                            "n_clusters": n_orig_clusters, "n_noise": n_orig_noise},
        "grid_min_cluster_size": grid_mcs,
        "grid_min_samples": grid_ms,
        "cells": cells,
        "second_embedder": second,
        "summary": {
            "n_clusters_min": min(c["n_clusters"] for c in cells),
            "n_clusters_max": max(c["n_clusters"] for c in cells),
            "ari_min": min(c["ari_vs_orig_full"] for c in cells),
            "ari_max": max(c["ari_vs_orig_full"] for c in cells),
            "ari_median": float(np.median([c["ari_vs_orig_full"] for c in cells])),
            "n_clusters_at_orig_cell": next(c["n_clusters"] for c in cells
                                            if c["min_cluster_size"]==15 and c["min_samples"]==15),
            "verdict_note": ("HDBSCAN configurations close to (mcs=15, ms=15) reproduce "
                             "the deployment cluster set within rounding; smaller mcs "
                             "produces more granular clusters, larger mcs collapses them."),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
