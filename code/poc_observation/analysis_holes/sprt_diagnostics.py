"""SPRT robustness diagnostics on real deployment UQ traces.

Inputs:  paper_artifacts/poc_results/per_proposal_traces.jsonl
Outputs: figures/sprt_qqplot.png
         figures/sprt_autocorr.png
         figures/sprt_bootstrap_typeI_II.png
         analysis/sprt_diagnostics.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
FIGDIR = ROOT / "figures"
OUTJSON = ROOT / "analysis" / "sprt_diagnostics.json"


def llr_step(u, mu0, mu1, sigma):
    return ((mu1 - mu0) / sigma**2) * u - (mu1**2 - mu0**2) / (2 * sigma**2)


def sprt_decide(seq, mu0, mu1, sigma, alpha, beta, n_min=20, n_max=50):
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    s = 0.0
    for n, u in enumerate(seq[:n_max], start=1):
        s += llr_step(u, mu0, mu1, sigma)
        if n >= n_min:
            if s >= A: return ("BIRTH", n)
            if s <= B: return ("SKIP", n)
    return ("FALLBACK", min(len(seq), n_max))


def main():
    if not TRACES.exists():
        print(f"missing {TRACES}"); return 1
    items = [json.loads(l) for l in TRACES.open()]
    items = [x for x in items if x.get("uq_observations")]
    print(f"loaded {len(items)} proposals")

    # Adapt μ0/μ1 to the observed UQ scale (deployment uses raw token entropy)
    all_uq = np.concatenate([np.array(x["uq_observations"]) for x in items])
    mu_all = float(all_uq.mean()); sigma_emp = float(all_uq.std())
    mu0 = float(np.percentile(all_uq, 40))
    mu1 = float(np.percentile(all_uq, 60))
    sigma = sigma_emp
    print(f"observed UQ scale: mean={mu_all:.3f} std={sigma_emp:.3f}; using μ0={mu0:.3f} μ1={mu1:.3f}")

    # ── Diagnostic 1: per-cluster normality + autocorrelation ─────────
    diag = []
    for it in items:
        u = np.array(it["uq_observations"])
        if len(u) < 10: continue
        u_centered = u - u.mean()
        # lag-1 autocorrelation
        ac1 = float(np.corrcoef(u_centered[:-1], u_centered[1:])[0, 1]) if len(u) > 2 else 0.0
        # Shapiro-Wilk approximation: skewness + kurtosis
        m = u.mean(); s = u.std() + 1e-9
        z = (u - m) / s
        skew = float((z**3).mean())
        kurt = float((z**4).mean()) - 3.0
        diag.append({
            "proposal_id": it.get("id"),
            "status": it.get("status"),
            "n": len(u), "mean": float(m), "std": float(s),
            "lag1_autocorr": ac1, "skew": skew, "excess_kurt": kurt,
        })
    print(f"per-cluster diagnostics on {len(diag)} proposals")

    # ── Diagnostic 2: bootstrap empirical type-I error ────────────────
    # Take SKIP proposals (status ABANDONED or PROPOSED), shuffle their
    # observations 1000× and run SPRT; count BIRTH decisions.
    skip_proposals = [x for x in items if x.get("status") in ("ABANDONED", "PROPOSED")]
    n_birth_under_h0 = 0; n_total_under_h0 = 0
    rng = np.random.default_rng(7)
    for it in skip_proposals:
        u = np.array(it["uq_observations"])
        if len(u) < 20: continue
        for _ in range(200):
            shuffled = rng.permutation(u)
            d, _ = sprt_decide(shuffled.tolist(), mu0, mu1, sigma, 0.05, 0.05)
            n_total_under_h0 += 1
            if d == "BIRTH": n_birth_under_h0 += 1
    type_I_emp = n_birth_under_h0 / max(n_total_under_h0, 1)

    # And type-II: take APPROVED, shuffle, count SKIP decisions
    birth_proposals = [x for x in items if x.get("status") == "APPROVED"]
    n_skip_under_h1 = 0; n_total_under_h1 = 0
    for it in birth_proposals:
        u = np.array(it["uq_observations"])
        if len(u) < 20: continue
        for _ in range(200):
            shuffled = rng.permutation(u)
            d, _ = sprt_decide(shuffled.tolist(), mu0, mu1, sigma, 0.05, 0.05)
            n_total_under_h1 += 1
            if d == "SKIP": n_skip_under_h1 += 1
    type_II_emp = n_skip_under_h1 / max(n_total_under_h1, 1)

    print(f"bootstrap (shuffled): empirical type-I = {type_I_emp:.3f}  (target α = 0.05)")
    print(f"bootstrap (shuffled): empirical type-II = {type_II_emp:.3f} (target β = 0.05)")

    # ── Diagnostic 3: stopping time distribution ──────────────────────
    stop_times = []
    decisions = []
    for it in items:
        u = it["uq_observations"]
        if len(u) < 20: continue
        d, n = sprt_decide(u, mu0, mu1, sigma, 0.05, 0.05)
        stop_times.append(n); decisions.append(d)

    # ── Plots ────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # QQ plots (4 random clusters)
    sample = items[:6] if len(items) >= 6 else items
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, it in zip(axes.flat, sample):
        u = np.array(it["uq_observations"])
        if len(u) < 10:
            ax.set_visible(False); continue
        u_sorted = np.sort(u)
        from scipy.stats import norm
        q_theo = norm.ppf((np.arange(1, len(u)+1) - 0.5) / len(u))
        ax.scatter(q_theo, u_sorted, s=20, color="#388bfd", alpha=0.7)
        m, b = np.polyfit(q_theo, u_sorted, 1)
        ax.plot(q_theo, m*q_theo + b, color="black", lw=1, alpha=0.7)
        ax.set_title(f"prop #{it.get('id')}  status={it.get('status')}  n={len(u)}", fontsize=9)
        ax.set_xlabel("Theoretical N(0,1) quantile"); ax.set_ylabel("Observed UQ")
        ax.grid(alpha=0.3)
    plt.suptitle("Q-Q plots: per-proposal UQ vs Normal — visually checking the SPRT Gaussian assumption",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(FIGDIR / "sprt_qqplot.png", dpi=160, bbox_inches="tight"); plt.close()
    print(f"wrote {FIGDIR/'sprt_qqplot.png'}")

    # Autocorrelation histogram
    ac1s = [d["lag1_autocorr"] for d in diag]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(ac1s, bins=15, color="#fa6e6e", edgecolor="black", alpha=0.85)
    ax.axvline(0, ls="--", color="black", alpha=0.5)
    ax.set_xlabel("Lag-1 autocorrelation of UQ within a proposal")
    ax.set_ylabel("# proposals")
    ax.set_title(f"SPRT independence assumption check — mean lag-1 AC = {np.mean(ac1s):.3f}")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIGDIR / "sprt_autocorr.png", dpi=160, bbox_inches="tight"); plt.close()
    print(f"wrote {FIGDIR/'sprt_autocorr.png'}")

    # Bootstrap type-I/II bar chart + nominal level
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(["Type-I (FP rate under H₀)", "Type-II (FN rate under H₁)"],
                  [type_I_emp, type_II_emp], color=["#c0392b", "#388bfd"],
                  edgecolor="black", lw=0.8, alpha=0.85)
    ax.axhline(0.05, ls="--", color="black", label="nominal α=β=0.05")
    for b, v in zip(bars, [type_I_emp, type_II_emp]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, max(0.3, max(type_I_emp, type_II_emp) * 1.3))
    ax.set_ylabel("Empirical error rate")
    ax.set_title(f"Bootstrap SPRT robustness on shuffled deployment UQ\n"
                 f"({n_total_under_h0} resamples for type-I, {n_total_under_h1} for type-II)",
                 fontsize=10)
    ax.legend()
    plt.tight_layout(); plt.savefig(FIGDIR / "sprt_bootstrap_typeI_II.png", dpi=160, bbox_inches="tight"); plt.close()
    print(f"wrote {FIGDIR/'sprt_bootstrap_typeI_II.png'}")

    # Stopping time histogram
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(stop_times, bins=range(20, 52, 2), color="#27ae60", edgecolor="black", alpha=0.85)
    ax.axvline(np.mean(stop_times), ls="--", color="black",
               label=f"mean={np.mean(stop_times):.1f}, median={np.median(stop_times):.0f}")
    ax.set_xlabel("Stopping time N (observations)"); ax.set_ylabel("# proposals")
    ax.set_title(f"SPRT stopping time on deployment data ({len(stop_times)} proposals)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIGDIR / "sprt_stopping_times.png", dpi=160, bbox_inches="tight"); plt.close()
    print(f"wrote {FIGDIR/'sprt_stopping_times.png'}")

    # ── Save numbers ─────────────────────────────────────────────────
    out = {
        "n_proposals": len(items),
        "uq_scale": {"mean": mu_all, "std": sigma_emp,
                     "mu0_p40": mu0, "mu1_p60": mu1},
        "per_cluster": diag,
        "lag1_autocorr": {"mean": float(np.mean(ac1s)),
                          "median": float(np.median(ac1s)),
                          "max": float(max(ac1s)),
                          "min": float(min(ac1s))},
        "skew": {"mean": float(np.mean([d["skew"] for d in diag]))},
        "excess_kurt": {"mean": float(np.mean([d["excess_kurt"] for d in diag]))},
        "bootstrap": {
            "type_I_empirical": type_I_emp, "n_resamples_H0": n_total_under_h0,
            "type_II_empirical": type_II_emp, "n_resamples_H1": n_total_under_h1,
            "alpha_nominal": 0.05, "beta_nominal": 0.05,
        },
        "stopping_times": {
            "mean": float(np.mean(stop_times)),
            "median": float(np.median(stop_times)),
            "n_at_cap": int(sum(1 for n in stop_times if n >= 50)),
            "n_total": len(stop_times),
        },
    }
    OUTJSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUTJSON}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_cluster"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
