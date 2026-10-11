"""HOLE 3 — Drift, ADF and KPSS stationarity tests on per-proposal UQ traces.

For each of the 7 deployment proposals with uq_observations:
  - KS-test early half vs late half
  - ADF (statsmodels.tsa.stattools.adfuller)  H0: non-stationary
  - KPSS (statsmodels.tsa.stattools.kpss)     H0: stationary

Summary:
  - % proposals with n>=30 (where ADF is meaningful)
  - % with ADF p<0.05 (rejecting non-stationarity)
  - % with both ADF rejects AND KPSS does NOT reject (robustly stationary)

Output: analysis/HOLE_3_drift_stationarity.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT.parent / "paper_artifacts" / "poc_results" / "per_proposal_traces.jsonl"
OUTJSON = ROOT / "analysis" / "HOLE_3_drift_stationarity.json"


def main():
    from statsmodels.tsa.stattools import adfuller, kpss
    warnings.filterwarnings("ignore")

    items = [json.loads(l) for l in TRACES.open() if l.strip()]
    items = [x for x in items if x.get("uq_observations")]
    print(f"loaded {len(items)} proposals with observations")

    per = []
    for it in items:
        u = np.asarray(it["uq_observations"], float)
        n = len(u)
        rec = {"id": it.get("id"), "cluster_id": it.get("cluster_id"),
               "status": it.get("status"), "n": n}
        # KS drift test on early vs late halves
        if n >= 8:
            half = n // 2
            ks_p = float(ks_2samp(u[:half], u[half:]).pvalue)
            rec["ks_p"] = ks_p
            rec["ks_drift_at_05"] = ks_p < 0.05
        else:
            rec["ks_p"] = None; rec["ks_drift_at_05"] = None
        # ADF
        if n >= 30:
            try:
                adf_p = float(adfuller(u, autolag="AIC")[1])
            except Exception as e:
                adf_p = None
            rec["adf_p"] = adf_p
            rec["adf_rejects_nonstat_at_05"] = (adf_p is not None and adf_p < 0.05)
        else:
            rec["adf_p"] = None; rec["adf_rejects_nonstat_at_05"] = None
        # KPSS
        if n >= 12:
            try:
                kpss_p = float(kpss(u, regression="c", nlags="auto")[1])
            except Exception as e:
                kpss_p = None
            rec["kpss_p"] = kpss_p
            # H0 of KPSS = stationary; "not reject" ↔ p >= 0.05
            rec["kpss_does_not_reject_stat_at_05"] = (kpss_p is not None and kpss_p >= 0.05)
        else:
            rec["kpss_p"] = None; rec["kpss_does_not_reject_stat_at_05"] = None

        rec["robustly_stationary"] = bool(
            rec["adf_rejects_nonstat_at_05"] is True
            and rec["kpss_does_not_reject_stat_at_05"] is True
        )
        per.append(rec)
        print(f"id={rec['id']} n={n} ks_p={rec['ks_p']} adf_p={rec['adf_p']} kpss_p={rec['kpss_p']}")

    n_total = len(per)
    n_geq30 = sum(1 for r in per if r["n"] >= 30)
    n_adf_rej = sum(1 for r in per if r["adf_rejects_nonstat_at_05"] is True)
    n_robust = sum(1 for r in per if r["robustly_stationary"])
    n_ks_drift = sum(1 for r in per if r["ks_drift_at_05"] is True)

    out = {
        "n_proposals": n_total,
        "n_with_n_geq_30": n_geq30,
        "pct_n_geq_30": n_geq30 / max(n_total, 1),
        "pct_adf_rejects_nonstat": (n_adf_rej / max(n_geq30, 1)) if n_geq30 else None,
        "pct_robustly_stationary": (n_robust / max(n_geq30, 1)) if n_geq30 else None,
        "pct_ks_early_vs_late_drift": n_ks_drift / max(n_total, 1),
        "alpha": 0.05,
        "per_proposal": per,
        "note": ("ADF H0=non-stationary (rejecting at p<0.05 supports "
                 "stationarity). KPSS H0=stationary (failing to reject "
                 "supports stationarity). 'Robustly stationary' means "
                 "ADF rejects AND KPSS does not reject."),
    }
    OUTJSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTJSON}")


if __name__ == "__main__":
    sys.exit(main())
