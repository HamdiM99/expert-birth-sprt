"""HDBSCAN clustering on query embeddings, with cluster scoring.

Given a list of {query, embedding, uq} dicts, returns clusters with
per-cluster scores: size, cohesion (1 - mean intra-cluster distance),
mean UQ, and a composite trigger_score.

A cluster is a "candidate" iff:
    size >= min_cluster_size  AND  mean_uq >= min_uq

The candidate is the unit on which gates (eager / observation) decide.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_distances


def cluster_queries(items: List[Dict],
                    min_cluster_size: int = 15,
                    min_uq: float = 0.3,
                    cluster_selection_epsilon: float = 0.0):
    """items: list of dicts with keys {query, embedding (np.ndarray), uq}.

    Returns: list of cluster dicts with keys
        cluster_id, size, member_indices, cohesion, mean_uq, trigger_score,
        block_distribution (for diagnostics)
    """
    import hdbscan

    embs = np.stack([np.asarray(x["embedding"], dtype=np.float32) for x in items])
    # HDBSCAN on cosine via L2-normalized → euclidean
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.maximum(norms, 1e-9)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(embs_n)

    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < min_cluster_size:
            continue
        cluster_embs = embs_n[idx]
        # cohesion = 1 - mean pairwise cosine distance
        if len(idx) > 1:
            d = cosine_distances(cluster_embs)
            cohesion = float(1.0 - d[np.triu_indices_from(d, k=1)].mean())
        else:
            cohesion = 1.0
        member_uqs = [items[i].get("uq", 0.0) for i in idx]
        mean_uq = float(np.mean(member_uqs))
        if mean_uq < min_uq:
            continue
        block_dist = {}
        for i in idx:
            b = items[i].get("block", "?")
            block_dist[b] = block_dist.get(b, 0) + 1
        size_norm = min(len(idx) / 50.0, 1.0)
        trigger_score = float(0.4 * cohesion + 0.4 * mean_uq + 0.2 * size_norm)
        clusters.append({
            "cluster_id": int(cid),
            "size": int(len(idx)),
            "member_indices": idx.tolist(),
            "cohesion": cohesion,
            "mean_uq": mean_uq,
            "trigger_score": trigger_score,
            "block_distribution": block_dist,
            "centroid": cluster_embs.mean(axis=0).tolist(),
        })
    return clusters


def detect_candidates(items: List[Dict], **kwargs) -> List[Dict]:
    """Wrapper: returns clusters sorted by trigger_score desc."""
    clusters = cluster_queries(items, **kwargs)
    clusters.sort(key=lambda c: c["trigger_score"], reverse=True)
    return clusters


def assign_cluster_ground_truth(cluster: Dict, items: List[Dict]) -> Optional[bool]:
    """Ground truth label for a cluster: TRUE iff majority of its members
    are flagged gt_needs_birth=True. None if mostly noise."""
    real = trap = noise = 0
    for i in cluster["member_indices"]:
        gt = items[i].get("gt_needs_birth")
        if gt is True: real += 1
        elif gt is False: trap += 1
        else: noise += 1
    total = real + trap
    if total == 0:
        return None
    return real / total >= 0.5
