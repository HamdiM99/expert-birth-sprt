"""HOLE 13 — Embedder ablation on the 1379 deployment queries.

Reviewer Q1: how sensitive are results to clustering method and embedding
choice (MiniLM vs BGE/e5)?

Protocol:
  1. Load query_text for the 1379 deployment queries from deployment exports.
  2. Re-embed with two alternative SBERT models:
       - BAAI/bge-small-en-v1.5 (384d, retrieval-tuned)
       - intfloat/e5-small-v2  (384d, contrastive)
  3. Cluster each embedding set with HDBSCAN(mcs=15, ms=15) — same as deployment.
  4. Compute n_clusters, n_noise, ARI vs original deployment labels.
  5. Restrict to non-noise common subset for ARI-no-noise.
  6. Report cluster-mean UQ correlation between embedders (does the gate
     decision land on the same clusters?).

Saves /workspace/analysis/results/HOLE_13_embedder_ablation.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from collections import Counter
import numpy as np

QUERIES_PATH = Path(os.environ.get("HOLE13_QUERIES",
    "./pod_workdir/exports/anon_paper_data/queries.jsonl"))
ORIG_EMBS_PATH = Path(os.environ.get("HOLE13_ORIG_EMBS",
    "/workspace/analysis/data/embeddings.npz"))
OUT = Path(os.environ.get("ANALYSIS_OUT",
    "/workspace/analysis/results/HOLE_13_embedder_ablation.json"))


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import adjusted_rand_score
    import hdbscan

    queries = [json.loads(l) for l in QUERIES_PATH.open() if l.strip()]
    texts = [q["query_text"] for q in queries]
    print(f"Loaded {len(texts)} queries from {QUERIES_PATH.name}")

    # Original deployment embeddings + cluster labels
    d = np.load(ORIG_EMBS_PATH)
    orig_embs = d["embeddings"].astype(np.float32)
    print(f"Original embeddings shape: {orig_embs.shape}")
    if orig_embs.shape[0] != len(texts):
        n = min(orig_embs.shape[0], len(texts))
        print(f"WARN: align to {n} (orig={orig_embs.shape[0]} vs texts={len(texts)})")
        orig_embs = orig_embs[:n]
        texts = texts[:n]
        queries = queries[:n]

    def cluster(embs, mcs=15, ms=15):
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        en = embs / np.maximum(norms, 1e-9)
        return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                metric="euclidean").fit_predict(en)

    orig_labels = cluster(orig_embs)
    n_orig_clusters = len({l for l in orig_labels if l != -1})
    print(f"Original (MiniLM): {n_orig_clusters} clusters, "
          f"{(orig_labels == -1).sum()} noise")

    candidates = [
        ("BAAI/bge-small-en-v1.5", lambda x: x),
        ("intfloat/e5-small-v2", lambda x: ["query: " + t for t in x]),
    ]
    results = []
    for model_name, prep in candidates:
        print(f"\n=== Embedding with {model_name} ===")
        try:
            model = SentenceTransformer(model_name, device="cuda")
        except Exception:
            try:
                model = SentenceTransformer(model_name, device="cpu")
            except Exception as e:
                print(f"  Skipping {model_name}: {e}")
                results.append({"model": model_name, "error": str(e)})
                continue
        prepared = prep(texts)
        embs = model.encode(prepared, convert_to_numpy=True, show_progress_bar=False,
                            batch_size=64, normalize_embeddings=False)
        embs = embs.astype(np.float32)
        labs = cluster(embs)
        n_clusters = len({l for l in labs if l != -1})
        n_noise = int((labs == -1).sum())
        ari_full = float(adjusted_rand_score(orig_labels, labs))
        mask = (orig_labels != -1) & (labs != -1)
        ari_nn = float(adjusted_rand_score(orig_labels[mask], labs[mask])) if mask.sum() >= 2 else None

        # Cluster mean comparison
        orig_clusters = {c: orig_labels == c for c in set(orig_labels) if c != -1}
        new_clusters = {c: labs == c for c in set(labs) if c != -1}
        # For each new cluster, find the best-matching orig cluster (Jaccard)
        match_table = []
        for nc, nm in new_clusters.items():
            best, best_j = None, 0.0
            for oc, om in orig_clusters.items():
                inter = int(np.logical_and(nm, om).sum())
                union = int(np.logical_or(nm, om).sum())
                j = inter / union if union else 0
                if j > best_j: best_j = j; best = oc
            match_table.append({"new_cluster_id": int(nc),
                                "size": int(nm.sum()),
                                "matched_orig_id": int(best) if best is not None else None,
                                "jaccard": best_j})
        results.append({
            "model": model_name,
            "n_clusters": n_clusters, "n_noise": n_noise,
            "noise_frac": n_noise / len(labs),
            "ari_full": ari_full, "ari_no_noise": ari_nn,
            "match_table": match_table,
            "top_jaccards": sorted([m["jaccard"] for m in match_table], reverse=True)[:5],
        })
        print(f"  {model_name}: {n_clusters} clusters, {n_noise} noise, "
              f"ARI={ari_full:.3f} (no-noise {ari_nn})")

    out = {
        "experiment": "HOLE_13_embedder_ablation",
        "n_queries": len(texts),
        "primary_embedder": "MiniLM (deployment, 384d)",
        "primary": {
            "n_clusters": n_orig_clusters,
            "n_noise": int((orig_labels == -1).sum()),
        },
        "alternatives": results,
        "summary": (
            "Re-embedding the 1379 deployment queries with BGE-small and e5-small, "
            "then re-clustering with the same HDBSCAN config: ARI vs MiniLM "
            "deployment labels measures cluster-set stability across embedders."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
