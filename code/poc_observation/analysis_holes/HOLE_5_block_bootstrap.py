"""HOLE 5 — Circular block bootstrap for SPRT type-I/II error,
addressing reviewer's autocorrelation concern.

We re-run the type-I/II bootstrap from sprt_diagnostics.py but instead of
i.i.d. shuffling we use circular block bootstrap with block size b.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # supplementary/
DATA = Path(os.environ.get("ANALYSIS_DATA", _ROOT / "data"))
OUT = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_5_block_bootstrap.json"))


def llr_step(u, mu0, mu1, sigma):
    return ((mu1 - mu0) / sigma**2) * u - (mu1**2 - mu0**2) / (2 * sigma**2)


def sprt_decide(seq, mu0, mu1, sigma, alpha=0.05, beta=0.05, n_min=20, n_max=50):
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    s = 0.0
    for n, u in enumerate(seq[:n_max], start=1):
        s += llr_step(u, mu0, mu1, sigma)
        if n >= n_min:
            if s >= A: return ("BIRTH", n)
            if s <= B: return ("SKIP", n)
    return ("FALLBACK", min(len(seq), n_max))


def circular_block_bootstrap(seq: np.ndarray, b: int, length: int, rng) -> np.ndarray:
    """Resample a sequence of given `length` by stitching circular blocks of size b."""
    n = len(seq)
    if b <= 1:
        return rng.permutation(seq)[:length]
    n_blocks = int(np.ceil(length / b))
    starts = rng.integers(0, n, size=n_blocks)
    out = np.empty(n_blocks * b, dtype=seq.dtype)
    for i, s in enumerate(starts):
        # circular slice of length b
        idx = (s + np.arange(b)) % n
        out[i*b:(i+1)*b] = seq[idx]
    return out[:length]


def main():
    items = [json.loads(l) for l in (DATA / "per_proposal_traces.jsonl").open()]
    items = [x for x in items if x.get("uq_observations")]

    # Same SPRT scale as deployment
    all_uq = np.concatenate([np.array(x["uq_observations"]) for x in items])
    mu0 = float(np.percentile(all_uq, 40))
    mu1 = float(np.percentile(all_uq, 60))
    sigma = float(all_uq.std())

    # Observed mean lag-1 autocorrelation (informational; we report it).
    ac1s = []
    for it in items:
        u = np.array(it["uq_observations"])
        if len(u) >= 3:
            uc = u - u.mean()
            ac1s.append(float(np.corrcoef(uc[:-1], uc[1:])[0, 1]))
    rho_bar = float(np.nanmean(ac1s))

    abandoned = [x for x in items if x.get("status") in ("ABANDONED", "PROPOSED") and len(x["uq_observations"]) >= 20]
    approved  = [x for x in items if x.get("status") == "APPROVED" and len(x["uq_observations"]) >= 20]

    print(f"abandoned={len(abandoned)} approved={len(approved)} rho_bar={rho_bar:.3f}")
    print(f"mu0={mu0:.3f} mu1={mu1:.3f} sigma={sigma:.3f}")

    block_sizes = [1, 3, 5, 8, 12]
    n_resamples = 200
    rng = np.random.default_rng(7)

    type_I_per_block = {}
    type_II_per_block = {}

    for b in block_sizes:
        n_birth_h0, n_total_h0 = 0, 0
        n_skip_h1, n_total_h1 = 0, 0
        for it in abandoned:
            u = np.array(it["uq_observations"])
            for _ in range(n_resamples):
                resampled = circular_block_bootstrap(u, b, len(u), rng)
                d, _ = sprt_decide(resampled.tolist(), mu0, mu1, sigma)
                n_total_h0 += 1
                if d == "BIRTH": n_birth_h0 += 1
        for it in approved:
            u = np.array(it["uq_observations"])
            for _ in range(n_resamples):
                resampled = circular_block_bootstrap(u, b, len(u), rng)
                d, _ = sprt_decide(resampled.tolist(), mu0, mu1, sigma)
                n_total_h1 += 1
                if d == "SKIP": n_skip_h1 += 1
        type_I = n_birth_h0 / max(n_total_h0, 1)
        type_II = n_skip_h1 / max(n_total_h1, 1)
        type_I_per_block[str(b)] = type_I
        type_II_per_block[str(b)] = type_II
        print(f"b={b:2d}  type-I={type_I:.4f}  ({n_birth_h0}/{n_total_h0})  type-II={type_II:.4f}  ({n_skip_h1}/{n_total_h1})")

    # Recommended b from rho_bar:  b = ceil(1/(1-rho))  capped at 50
    rho_safe = min(max(rho_bar, 0.0), 0.95)
    b_recommended = int(math.ceil(1.0 / max(1.0 - rho_safe, 1e-3)))

    summary = {
        "block_sizes": block_sizes,
        "type_I_per_block": type_I_per_block,
        "type_II_per_block": type_II_per_block,
        "n_resamples_per_proposal": n_resamples,
        "n_abandoned_proposals": len(abandoned),
        "n_approved_proposals": len(approved),
        "n_total_h0_trials_per_block": n_resamples * len(abandoned),
        "n_total_h1_trials_per_block": n_resamples * len(approved),
        "autocorr_observed": {
            "rho_bar_mean": rho_bar,
            "per_proposal": ac1s,
        },
        "alpha_nominal": 0.05,
        "beta_nominal": 0.05,
        "b_recommended_from_rho": b_recommended,
        "sprt_params": {"mu0_p40": mu0, "mu1_p60": mu1, "sigma": sigma,
                        "n_min": 20, "n_max": 50},
        "summary": {
            "iid_shuffle_b1_type_I": type_I_per_block["1"],
            "iid_shuffle_b1_type_II": type_II_per_block["1"],
            "block_b5_type_I": type_I_per_block["5"],
            "block_b5_type_II": type_II_per_block["5"],
            "max_type_I_across_blocks": max(type_I_per_block.values()),
            "max_type_II_across_blocks": max(type_II_per_block.values()),
            "verdict": (
                "Wald bounds (α=β=0.05) PRESERVED across all block sizes"
                if max(type_I_per_block.values()) < 0.05 and max(type_II_per_block.values()) < 0.05
                else "Wald bounds VIOLATED for at least one block size — see per-block table."
            ),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
