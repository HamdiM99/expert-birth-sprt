"""Eager birth baseline: any cluster passing the detection thresholds → BIRTH.

Operationalizes BAM/SEED-MoE-style birth triggers: cluster_size >= MIN_SIZE
and mean_uq >= MIN_UQ. No additional validation, no observation phase.

Decision: birth if cluster qualifies, else skip.
"""
from __future__ import annotations
from typing import List, Dict


def decide(clusters: List[Dict],
           min_size: int = 15,
           min_uq: float = 0.3) -> List[Dict]:
    """Returns one decision dict per cluster:
        {cluster_id, decision, reason, n_observations_used}

    decision: "BIRTH" or "SKIP"
    """
    decisions = []
    for c in clusters:
        if c["size"] >= min_size and c["mean_uq"] >= min_uq:
            decisions.append({
                "cluster_id": c["cluster_id"],
                "decision": "BIRTH",
                "reason": f"size={c['size']}>={min_size} and mean_uq={c['mean_uq']:.3f}>={min_uq}",
                "n_observations_used": 0,  # eager uses 0 shadow observations
                "trigger_score": c["trigger_score"],
            })
        else:
            decisions.append({
                "cluster_id": c["cluster_id"],
                "decision": "SKIP",
                "reason": f"failed threshold (size={c['size']} mean_uq={c['mean_uq']:.3f})",
                "n_observations_used": 0,
                "trigger_score": c["trigger_score"],
            })
    return decisions
