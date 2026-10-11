"""Observation-gated birth (the contribution).

For every detected cluster:
  1. Run shadow inference on N member queries (backbone-only)
  2. Record UQ signal per query
  3. Apply Sequential Probability Ratio Test (SPRT) stopping rule:
        H0: backbone competent on cluster   (mean_uq <= μ0)
        H1: backbone struggles               (mean_uq >= μ1)
        with μ0 = uq_threshold - δ, μ1 = uq_threshold + δ.
     Continue observing while α/β bounds not met.
  4. Decide BIRTH (H1 accepted) or SKIP (H0 accepted) or
     UNDECIDED at TARGET_OBS — fall back to mean_uq >= threshold.

The gate is parameterized by:
    target_obs   default 50 (cap)
    min_obs      default 20 (lower bound before SPRT may reject)
    uq_threshold default 0.55 (gate threshold)
    alpha, beta  SPRT type-I and type-II error bounds (default 0.05 each)

Returns one decision dict per cluster, with the full UQ trace
for downstream analysis / figures.
"""
from __future__ import annotations
import math
from typing import List, Dict, Callable


def sprt_decision(observations: List[float],
                  mu0: float, mu1: float,
                  sigma: float = 0.2,
                  alpha: float = 0.05, beta: float = 0.05) -> str:
    """Wald's SPRT for normal mean with known variance.

    Returns "H0" (backbone OK), "H1" (backbone struggles), or "CONTINUE".
    """
    if len(observations) == 0:
        return "CONTINUE"
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    # Log-likelihood ratio under N(μ, σ²) assumption
    n = len(observations)
    mean = sum(observations) / n
    # LLR = sum_i [ log φ(x_i; μ1, σ) - log φ(x_i; μ0, σ) ]
    #     = (μ1-μ0)/σ² · Σ x_i  -  n(μ1²-μ0²)/(2σ²)
    s = sum(observations)
    llr = (mu1 - mu0) / (sigma ** 2) * s - n * (mu1**2 - mu0**2) / (2 * sigma**2)
    if llr >= A:
        return "H1"   # accept "backbone struggles" → BIRTH
    if llr <= B:
        return "H0"   # accept "backbone OK" → SKIP
    return "CONTINUE"


def decide(clusters: List[Dict],
           items: List[Dict],
           shadow_uq_fn: Callable[[str], float],
           target_obs: int = 50,
           min_obs: int = 20,
           uq_threshold: float = 0.55,
           sprt_delta: float = 0.10,
           sprt_alpha: float = 0.05,
           sprt_beta: float = 0.05,
           sprt_sigma: float = 0.20) -> List[Dict]:
    """For each cluster, run observation-gated decision.

    items: full workload (so we can pull back the queries by member_indices)
    shadow_uq_fn: callable(query_text) -> float — runs backbone shadow forward

    Returns: list of decision dicts with full trace.
    """
    mu0 = uq_threshold - sprt_delta
    mu1 = uq_threshold + sprt_delta
    decisions = []
    for c in clusters:
        member_qs = [items[i]["query"] for i in c["member_indices"]]
        # Round-robin sample queries up to target_obs
        observations = []
        decision = None
        reason = ""
        for k in range(target_obs):
            q = member_qs[k % len(member_qs)]
            uq = shadow_uq_fn(q)
            observations.append(uq)
            if len(observations) >= min_obs:
                verdict = sprt_decision(observations, mu0, mu1,
                                        sigma=sprt_sigma,
                                        alpha=sprt_alpha, beta=sprt_beta)
                if verdict == "H1":
                    decision = "BIRTH"
                    reason = f"SPRT-H1 at n={len(observations)} mean={sum(observations)/len(observations):.3f}"
                    break
                if verdict == "H0":
                    decision = "SKIP"
                    reason = f"SPRT-H0 at n={len(observations)} mean={sum(observations)/len(observations):.3f}"
                    break
        if decision is None:
            mean = sum(observations) / len(observations) if observations else 0.0
            decision = "BIRTH" if mean >= uq_threshold else "SKIP"
            reason = f"undecided at TARGET_OBS={target_obs}, fallback mean={mean:.3f} {'>=' if decision=='BIRTH' else '<'} {uq_threshold}"

        decisions.append({
            "cluster_id": c["cluster_id"],
            "decision": decision,
            "reason": reason,
            "n_observations_used": len(observations),
            "uq_observations": observations,
            "mean_uq_observed": sum(observations) / len(observations) if observations else 0.0,
            "trigger_score": c["trigger_score"],
        })
    return decisions
