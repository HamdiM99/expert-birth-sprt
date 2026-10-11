"""Confusion matrix + precision/recall/F1 + threshold sweep.

Inputs are decisions per cluster (one per gate) and ground-truth labels
per cluster (from counterfactual.py).
"""
from __future__ import annotations
from typing import Dict, List


def confusion(decisions: List[Dict], gt: Dict[int, bool]) -> Dict:
    """decisions: list of {cluster_id, decision} (decision in {BIRTH, SKIP})
       gt: cluster_id -> bool (True = needs_birth)
    """
    tp = fp = tn = fn = 0
    matched = 0
    for d in decisions:
        cid = d["cluster_id"]
        if cid not in gt:
            continue
        matched += 1
        truth = gt[cid]
        decided = d["decision"] == "BIRTH"
        if decided and truth: tp += 1
        elif decided and not truth: fp += 1
        elif not decided and truth: fn += 1
        else: tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "n_births": tp + fp, "n_skipped": tn + fn,
        "n_matched": matched,
    }


def threshold_sweep(observation_decisions: List[Dict],
                    gt: Dict[int, bool],
                    thresholds: List[float]) -> List[Dict]:
    """Re-decide observation cases at each UQ threshold (using stored
    mean_uq_observed) and recompute confusion. Cheap because we don't
    re-run shadow inference."""
    out = []
    for thr in thresholds:
        rerun = []
        for d in observation_decisions:
            mean_uq = d.get("mean_uq_observed", 0.0)
            new_dec = "BIRTH" if mean_uq >= thr else "SKIP"
            rerun.append({**d, "decision": new_dec})
        cm = confusion(rerun, gt)
        out.append({"threshold": thr, **cm})
    return out


def aggregate_seeds(per_seed_cms: List[Dict]) -> Dict:
    """Mean ± std across seeds."""
    keys = ["tp", "fp", "tn", "fn", "precision", "recall", "f1", "n_births"]
    agg = {}
    n = len(per_seed_cms)
    for k in keys:
        vals = [cm[k] for cm in per_seed_cms]
        mean = sum(vals) / n if n else 0.0
        var = sum((v - mean) ** 2 for v in vals) / n if n else 0.0
        agg[k + "_mean"] = mean
        agg[k + "_std"] = var ** 0.5
    return agg
