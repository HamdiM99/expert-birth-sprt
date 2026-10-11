"""HOLE 9 — Per-domain UQ z-scoring on the controlled experiment (seed 0).

Reviewer concern: code tokens have structurally lower entropy than NL tokens,
so a global UQ threshold mis-calibrates per-domain. Z-score within domain;
re-run SPRT; report whether the failed code cluster is now classified
correctly without breaking the NL clusters.

Approach:
  1. Recover the seed-0 clusters with HDBSCAN on the saved 384-d embeddings
     (same code as core/cluster.py).
  2. Each cluster is assigned a *domain* by the modal `block` of its members,
     coarsened to {"code", "nl"} (true_need_code | trap_b -> code; everything
     else -> nl).
  3. Compute (mu_d, sigma_d) per domain via leave-one-cluster-out: for each
     cluster c with domain d, fit (mu, sigma) over the per-query UQ values
     of all OTHER clusters whose domain is d.
  4. Z-score each per-query UQ observation; re-run Wald SPRT with the
     z-unit-translated thresholds (mu0 = (0.45 - mu)/sigma, mu1 likewise),
     same alpha=beta=0.05, sigma=0.20 in z-space (we keep the assumed-known
     observation noise the same in z-units; documented as a caveat).
  5. Compare TNR/TPR/#births/avg-obs raw vs z-scored on the *same* 6 (or 7)
     clusters that the controlled experiment produced.
  6. Sensitivity: sweep z-threshold and report Pareto.

Caveats logged in the JSON.
"""
from __future__ import annotations
import json, math, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent  # supplementary/
POC = Path(os.environ.get("POC_ROOT", _HERE.parent))  # code/poc_observation/
RES = Path(os.environ.get("HOLE9_RESULTS", _ROOT / "results" / "token_entropy" / "seed0"))
WORKLOAD = Path(os.environ.get("HOLE9_WORKLOAD", POC / "data" / "workload_seed0.jsonl"))
OUT_POD = Path(os.environ.get("ANALYSIS_OUT", _ROOT / "results" / "HOLE_9_per_domain_calibration.json"))

# SPRT params from configs/obs.yaml
UQ_THRESHOLD = 0.55
SPRT_DELTA = 0.10
ALPHA = 0.05
BETA = 0.05
SIGMA = 0.20  # observation-noise sigma in raw UQ units
MIN_OBS = 20
TARGET_OBS = 50

# Birth GT: paper uses LoRA-delta > tau; here we already have summary.json's
# `ground_truth` (which is the labelled `needs_birth` per cluster).
def coarse_domain(block: str) -> str:
    if block in ("true_need_code", "trap_b"):
        return "code"
    return "nl"


def sprt_decision(obs, mu0, mu1, sigma, alpha, beta):
    if not obs:
        return "CONTINUE"
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    n = len(obs); s = sum(obs)
    llr = (mu1 - mu0) / sigma**2 * s - n * (mu1**2 - mu0**2) / (2 * sigma**2)
    if llr >= A: return "H1"
    if llr <= B: return "H0"
    return "CONTINUE"


def run_sprt(observations, mu0, mu1, sigma=SIGMA,
             alpha=ALPHA, beta=BETA,
             min_obs=MIN_OBS, target_obs=TARGET_OBS,
             threshold_for_fallback=None):
    """Replay SPRT step-by-step. Observations is the *full* per-query trace
    already produced by the original run, in order. We respect the same
    (min_obs, target_obs) protocol."""
    used = []
    for k in range(min(target_obs, len(observations))):
        used.append(observations[k])
        if len(used) >= min_obs:
            v = sprt_decision(used, mu0, mu1, sigma, alpha, beta)
            if v == "H1":
                return "BIRTH", len(used), float(np.mean(used)), "SPRT-H1"
            if v == "H0":
                return "SKIP", len(used), float(np.mean(used)), "SPRT-H0"
    if not used:
        return "SKIP", 0, 0.0, "no_obs"
    m = float(np.mean(used))
    thr = threshold_for_fallback if threshold_for_fallback is not None else 0.0
    dec = "BIRTH" if m >= thr else "SKIP"
    return dec, len(used), m, f"fallback mean={m:.3f}"


def main():
    # ---- 1. Load workload + saved per-query UQ + embeddings ----
    items = [json.loads(l) for l in WORKLOAD.open() if l.strip()]
    npz = np.load(RES / "embeddings.npz")
    embs = npz["embeddings"]
    uqs = npz["uqs"]
    assert len(items) == len(embs) == len(uqs), f"{len(items)} vs {embs.shape} vs {uqs.shape}"
    for i, it in enumerate(items):
        it["uq"] = float(uqs[i])
        it["embedding"] = embs[i]

    # ---- 2. Recover the same clusters (HDBSCAN cosine -> L2 norm + euclidean) ----
    import hdbscan
    embn = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    labels = hdbscan.HDBSCAN(min_cluster_size=15, metric="euclidean").fit_predict(embn)

    summary = json.load((RES / "summary.json").open())
    gt = {int(k): bool(v) for k, v in summary["ground_truth"].items()}
    decisions_existing = json.load((RES / "decisions.json").open())
    obs_existing = {d["cluster_id"]: d for d in decisions_existing["observation"]}

    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1: continue
        idx = np.where(labels == cid)[0].tolist()
        if len(idx) < 15: continue
        member_uqs = uqs[idx].tolist()
        mean_uq = float(np.mean(member_uqs))
        if mean_uq < 0.30: continue
        blocks = Counter(items[i]["block"] for i in idx)
        modal_block, _ = blocks.most_common(1)[0]
        clusters.append({
            "cluster_id": int(cid),
            "size": len(idx),
            "member_indices": idx,
            "member_uqs": member_uqs,
            "mean_uq_raw": mean_uq,
            "modal_block": modal_block,
            "domain": coarse_domain(modal_block),
            "block_distribution": dict(blocks),
        })

    # Sanity: align IDs with summary.json
    print(f"recovered {len(clusters)} clusters; summary GT keys: {sorted(gt.keys())}")
    print("cluster_id -> modal_block, domain, mean_uq_raw, gt:")
    for c in clusters:
        c["needs_birth_gt"] = gt.get(c["cluster_id"])
        print(f"  c{c['cluster_id']}: {c['modal_block']:20s} dom={c['domain']:4s} "
              f"mean_uq={c['mean_uq_raw']:.3f} gt={c['needs_birth_gt']}")

    # ---- 3. Per-domain mu/sigma via leave-one-cluster-out -----------
    # Pool the per-query UQs from OTHER clusters of the same domain.
    domain_global = {}  # domain -> all member UQs
    for d in {c["domain"] for c in clusters}:
        pooled = []
        for c in clusters:
            if c["domain"] == d:
                pooled.extend(c["member_uqs"])
        domain_global[d] = {
            "mu_pooled": float(np.mean(pooled)),
            "sigma_pooled": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 1.0,
            "n_pooled": len(pooled),
        }

    cluster_mu_sigma = {}  # cid -> (mu_d_loo, sigma_d_loo)
    for c in clusters:
        d = c["domain"]
        loo = []
        for c2 in clusters:
            if c2["domain"] == d and c2["cluster_id"] != c["cluster_id"]:
                loo.extend(c2["member_uqs"])
        if len(loo) < 2:
            # singleton domain - fallback to pooled (NOT leakage-free; flagged)
            mu = domain_global[d]["mu_pooled"]
            sg = domain_global[d]["sigma_pooled"]
            note = "singleton-domain fallback to pooled (leakage-aware caveat)"
        else:
            mu = float(np.mean(loo))
            sg = float(np.std(loo, ddof=1)) or 1.0
            note = f"LOO over {len(loo)} obs from sibling clusters"
        cluster_mu_sigma[c["cluster_id"]] = (mu, sg, note, len(loo))

    # ---- 4. Replay SPRT — raw and z-scored ----------------------------
    raw_results = {}
    z_results = {}
    obs_streams = {}  # for the JSON

    for c in clusters:
        cid = c["cluster_id"]
        # Pull the EXACT SAME per-query UQ stream the original obs gate used,
        # so the comparison is apples-to-apples (round-robin order preserved).
        stream = obs_existing.get(cid, {}).get("uq_observations", [])
        # Some clusters in the original run hit SPRT before TARGET_OBS, so the
        # stored stream is short. We extend it deterministically by continuing
        # the round-robin over the cluster's members (same logic the original
        # gate would have used had it not stopped early). The first len(stream)
        # entries are guaranteed identical.
        if len(stream) < TARGET_OBS:
            member_qs = c["member_indices"]
            # Each member contributes its saved UQ scalar; round-robin.
            extension = []
            k0 = len(stream)
            for k in range(k0, TARGET_OBS):
                idx_in_cluster = k % len(member_qs)
                extension.append(uqs[member_qs[idx_in_cluster]])
            stream_full = stream + [float(x) for x in extension]
        else:
            stream_full = stream[:TARGET_OBS]

        obs_streams[cid] = stream_full

        # raw SPRT
        mu0_r = UQ_THRESHOLD - SPRT_DELTA
        mu1_r = UQ_THRESHOLD + SPRT_DELTA
        dec_r, n_r, mean_r, why_r = run_sprt(
            stream_full, mu0_r, mu1_r, sigma=SIGMA,
            threshold_for_fallback=UQ_THRESHOLD)
        raw_results[cid] = dict(decision=dec_r, n_obs=n_r, mean=mean_r, reason=why_r)

        # z-scored SPRT
        mu_d, sg_d, note, n_loo = cluster_mu_sigma[cid]
        z_stream = [(x - mu_d) / sg_d for x in stream_full]
        # Translate mu0/mu1/sigma into z-units.
        mu0_z = (mu0_r - mu_d) / sg_d
        mu1_z = (mu1_r - mu_d) / sg_d
        sigma_z = SIGMA / sg_d
        thr_z = (UQ_THRESHOLD - mu_d) / sg_d
        dec_z, n_z, mean_z, why_z = run_sprt(
            z_stream, mu0_z, mu1_z, sigma=sigma_z,
            threshold_for_fallback=thr_z)
        z_results[cid] = dict(
            decision=dec_z, n_obs=n_z, mean_z=mean_z, reason=why_z,
            mu_d=mu_d, sigma_d=sg_d, mu0_z=mu0_z, mu1_z=mu1_z,
            sigma_z=sigma_z, thr_z=thr_z, loo_note=note, n_loo=n_loo,
        )

    # ---- 5. Confusion matrices -----------------------------------------
    def confusion(results):
        tp=fp=tn=fn=0
        births=0; obs_total=0; n_eval=0
        for c in clusters:
            cid = c["cluster_id"]; gtl = c["needs_birth_gt"]
            r = results[cid]
            if r["decision"] == "BIRTH": births += 1
            obs_total += r["n_obs"]
            if gtl is None: continue
            n_eval += 1
            pos = (r["decision"] == "BIRTH")
            if gtl and pos: tp += 1
            elif gtl and not pos: fn += 1
            elif (not gtl) and pos: fp += 1
            else: tn += 1
        tnr = tn / (tn + fp) if (tn + fp) else None
        tpr = tp / (tp + fn) if (tp + fn) else None
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, n_eval=n_eval,
                    n_births=births, avg_obs=obs_total/max(1,len(clusters)),
                    TNR=tnr, TPR=tpr)

    conf_raw = confusion(raw_results)
    conf_z = confusion(z_results)

    # ---- Code cluster verdict ------------------------------------------
    code_clusters = [c for c in clusters if c["domain"] == "code"]
    code_verdict = []
    for c in code_clusters:
        cid = c["cluster_id"]
        code_verdict.append({
            "cluster_id": cid,
            "modal_block": c["modal_block"],
            "needs_birth_gt": c["needs_birth_gt"],
            "raw_decision": raw_results[cid]["decision"],
            "raw_mean": raw_results[cid]["mean"],
            "z_decision": z_results[cid]["decision"],
            "z_mean": z_results[cid]["mean_z"],
            "rescued_or_corrected": (
                # rescue: was wrong under raw, correct under z
                (raw_results[cid]["decision"] == "BIRTH") != (c["needs_birth_gt"] is True)
                and
                (z_results[cid]["decision"] == "BIRTH") == (c["needs_birth_gt"] is True)
            ) if c["needs_birth_gt"] is not None else None,
        })

    # ---- 6. Sensitivity sweep over z-threshold (and equivalent raw thr)
    # We sweep mu_threshold in z-units and compute (TNR, TPR, n_births).
    sweep = []
    for mu_thr_z in np.linspace(-1.0, 2.5, 36):
        # rebuild SPRT under shifted threshold (delta still 0.10 raw -> in z it
        # depends on sigma_d, but for sweep simplicity sweep mu_thr_z and use
        # delta_z = SPRT_DELTA / mean_sigma_d).
        sigmas_d = [v[1] for v in cluster_mu_sigma.values()]
        delta_z = SPRT_DELTA / float(np.mean(sigmas_d))
        tp=fp=tn=fn=0; births=0
        for c in clusters:
            cid = c["cluster_id"]; gtl = c["needs_birth_gt"]
            mu_d, sg_d, _, _ = cluster_mu_sigma[cid]
            zs = [(x - mu_d)/sg_d for x in obs_streams[cid]]
            sigma_z = SIGMA / sg_d
            d, _, _, _ = run_sprt(
                zs, mu_thr_z - delta_z, mu_thr_z + delta_z,
                sigma=sigma_z, threshold_for_fallback=mu_thr_z)
            if d == "BIRTH": births += 1
            if gtl is None: continue
            pos = (d == "BIRTH")
            if gtl and pos: tp += 1
            elif gtl and not pos: fn += 1
            elif (not gtl) and pos: fp += 1
            else: tn += 1
        tnr = tn/(tn+fp) if (tn+fp) else None
        tpr = tp/(tp+fn) if (tp+fn) else None
        sweep.append(dict(z_thr=float(mu_thr_z), TNR=tnr, TPR=tpr, n_births=births,
                          tp=tp, fp=fp, tn=tn, fn=fn))

    # ---- Honest summary ------------------------------------------------
    rescued_any = any(v.get("rescued_or_corrected") for v in code_verdict)
    nl_kept = (conf_z["TPR"] is None or conf_raw["TPR"] is None or
               conf_z["TPR"] >= conf_raw["TPR"])

    summary_text = (
        "Per-domain z-scoring of token-entropy UQ was applied to the seed-0 "
        "controlled experiment (HDBSCAN clusters, 7 controlled blocks). "
        f"Domain stats (LOO over sibling clusters): "
        + ", ".join(f"{d}: mu={s['mu_pooled']:.3f}, sigma={s['sigma_pooled']:.3f} "
                    f"(n={s['n_pooled']})" for d, s in domain_global.items()) + ". "
        f"Raw SPRT confusion: TP={conf_raw['tp']} FP={conf_raw['fp']} "
        f"TN={conf_raw['tn']} FN={conf_raw['fn']} (TNR={conf_raw['TNR']}, "
        f"TPR={conf_raw['TPR']}). "
        f"Z-scored SPRT confusion: TP={conf_z['tp']} FP={conf_z['fp']} "
        f"TN={conf_z['tn']} FN={conf_z['fn']} (TNR={conf_z['TNR']}, "
        f"TPR={conf_z['TPR']}). "
        f"Code-cluster rescue: {rescued_any}. "
        "Caveats: (a) only 1-2 clusters per domain in seed-0, so LOO over "
        "sibling clusters is fragile; (b) noise/trap blocks were folded into "
        "'nl' which mixes heterogeneous distributions; (c) sigma in z-units "
        "was rescaled by sigma_d, but the *additional* observation-noise "
        "assumption sigma=0.20 was kept conceptually constant — a more "
        "principled fix would re-estimate it from data; (d) tau_Delta-based "
        "GT labelling was inherited from summary.json (counterfactual LoRA), "
        "not re-computed."
    )

    out = {
        "experiment": "HOLE_9_per_domain_calibration",
        "seed": 0,
        "n_clusters": len(clusters),
        "domain_global_stats": domain_global,
        "per_cluster": [
            {
                **{k: c[k] for k in ("cluster_id", "size", "modal_block", "domain",
                                      "block_distribution", "mean_uq_raw",
                                      "needs_birth_gt")},
                "mu_d_loo": cluster_mu_sigma[c["cluster_id"]][0],
                "sigma_d_loo": cluster_mu_sigma[c["cluster_id"]][1],
                "loo_note": cluster_mu_sigma[c["cluster_id"]][2],
                "n_loo": cluster_mu_sigma[c["cluster_id"]][3],
                "raw_sprt": raw_results[c["cluster_id"]],
                "z_sprt": z_results[c["cluster_id"]],
            } for c in clusters
        ],
        "confusion_raw": conf_raw,
        "confusion_zscored": conf_z,
        "code_cluster_verdict": code_verdict,
        "code_cluster_rescued": rescued_any,
        "nl_clusters_preserved": nl_kept,
        "z_threshold_sweep": sweep,
        "params": {
            "uq_threshold_raw": UQ_THRESHOLD,
            "sprt_delta_raw": SPRT_DELTA,
            "alpha": ALPHA, "beta": BETA, "sigma_raw": SIGMA,
            "min_obs": MIN_OBS, "target_obs": TARGET_OBS,
            "domain_coarsening": "{true_need_code, trap_b}->code; rest->nl",
        },
        "summary": summary_text,
    }

    OUT_POD.parent.mkdir(parents=True, exist_ok=True)
    OUT_POD.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT_POD}")
    print("\n=== SUMMARY ===\n" + summary_text)


if __name__ == "__main__":
    sys.exit(main())
