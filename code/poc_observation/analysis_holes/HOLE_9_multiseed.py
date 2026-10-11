"""HOLE 9 multi-seed — run per-domain z-scoring across seeds 0/1/2.

Same protocol as HOLE_9_per_domain_calibration.py but iterated on the
3 seeds we have available, to check whether the code-cluster failure is
a seed-0 artefact (singleton domain) or systematic.
"""
from __future__ import annotations
import json, math, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

POC = Path("/workspace/poc_observation")
RESULTS_BASE = POC / "results_token_entropy"
WORKLOAD_BASE = POC / "data"
OUT = Path(os.environ.get("ANALYSIS_OUT", "/workspace/analysis/results/HOLE_9_multiseed.json"))

UQ_THRESHOLD = 0.55
SPRT_DELTA = 0.10
ALPHA, BETA = 0.05, 0.05
SIGMA = 0.20
MIN_OBS, TARGET_OBS = 20, 50


def coarse_domain(block):
    return "code" if block in ("true_need_code", "trap_b") else "nl"


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


def run_sprt(observations, mu0, mu1, sigma=SIGMA, alpha=ALPHA, beta=BETA,
             min_obs=MIN_OBS, target_obs=TARGET_OBS, threshold_for_fallback=None):
    used = []
    for k in range(min(target_obs, len(observations))):
        used.append(observations[k])
        if len(used) >= min_obs:
            v = sprt_decision(used, mu0, mu1, sigma, alpha, beta)
            if v == "H1": return "BIRTH", len(used), float(np.mean(used))
            if v == "H0": return "SKIP", len(used), float(np.mean(used))
    if not used: return "SKIP", 0, 0.0
    m = float(np.mean(used))
    thr = threshold_for_fallback if threshold_for_fallback is not None else 0.0
    return ("BIRTH" if m >= thr else "SKIP"), len(used), m


def run_for_seed(seed):
    res = RESULTS_BASE / f"seed{seed}"
    wlp = WORKLOAD_BASE / f"workload_seed{seed}.jsonl"
    items = [json.loads(l) for l in wlp.open() if l.strip()]
    npz = np.load(res / "embeddings.npz")
    embs, uqs = npz["embeddings"], npz["uqs"]

    import hdbscan
    embn = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
    labels = hdbscan.HDBSCAN(min_cluster_size=15, metric="euclidean").fit_predict(embn)

    summary = json.load((res / "summary.json").open())
    gt = {int(k): bool(v) for k, v in summary["ground_truth"].items()}
    decisions_existing = json.load((res / "decisions.json").open())
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
            "cluster_id": int(cid), "size": len(idx),
            "member_indices": idx, "member_uqs": member_uqs,
            "mean_uq_raw": mean_uq, "modal_block": modal_block,
            "domain": coarse_domain(modal_block),
            "needs_birth_gt": gt.get(int(cid)),
        })

    # Domain pooled stats
    domains = {c["domain"] for c in clusters}
    n_per_domain = {d: sum(1 for c in clusters if c["domain"] == d) for d in domains}

    # Per-cluster mu_d, sigma_d (LOO over sibling clusters)
    cms = {}
    for c in clusters:
        d = c["domain"]
        loo = []
        for c2 in clusters:
            if c2["domain"] == d and c2["cluster_id"] != c["cluster_id"]:
                loo.extend(c2["member_uqs"])
        if len(loo) < 2:
            pooled = sum((c2["member_uqs"] for c2 in clusters if c2["domain"] == d), [])
            mu, sg = float(np.mean(pooled)), float(np.std(pooled, ddof=1)) or 1.0
            note = "singleton-domain pooled fallback"
        else:
            mu, sg = float(np.mean(loo)), float(np.std(loo, ddof=1)) or 1.0
            note = f"LOO over {len(loo)} obs"
        cms[c["cluster_id"]] = (mu, sg, note, len(loo))

    # SPRT replay raw + z-scored
    raw_decs, z_decs = {}, {}
    for c in clusters:
        cid = c["cluster_id"]
        stream = obs_existing.get(cid, {}).get("uq_observations", [])
        if len(stream) < TARGET_OBS:
            mqi = c["member_indices"]
            ext = [float(uqs[mqi[(k) % len(mqi)]]) for k in range(len(stream), TARGET_OBS)]
            stream_full = stream + ext
        else:
            stream_full = stream[:TARGET_OBS]
        # raw
        d_r, n_r, m_r = run_sprt(stream_full, UQ_THRESHOLD - SPRT_DELTA,
                                  UQ_THRESHOLD + SPRT_DELTA, sigma=SIGMA,
                                  threshold_for_fallback=UQ_THRESHOLD)
        raw_decs[cid] = (d_r, n_r, m_r)
        # z
        mu, sg, _, _ = cms[cid]
        zs = [(x - mu) / sg for x in stream_full]
        mu0_z = (UQ_THRESHOLD - SPRT_DELTA - mu) / sg
        mu1_z = (UQ_THRESHOLD + SPRT_DELTA - mu) / sg
        thr_z = (UQ_THRESHOLD - mu) / sg
        d_z, n_z, m_z = run_sprt(zs, mu0_z, mu1_z, sigma=SIGMA / sg,
                                  threshold_for_fallback=thr_z)
        z_decs[cid] = (d_z, n_z, m_z)

    def confusion(decs):
        tp=fp=tn=fn=0; nb=0; obs_total=0; n_eval=0
        for c in clusters:
            cid = c["cluster_id"]; gtl = c["needs_birth_gt"]
            d, n, _ = decs[cid]
            if d == "BIRTH": nb += 1
            obs_total += n
            if gtl is None: continue
            n_eval += 1
            pos = (d == "BIRTH")
            if gtl and pos: tp += 1
            elif gtl and not pos: fn += 1
            elif (not gtl) and pos: fp += 1
            else: tn += 1
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, n_eval=n_eval, n_births=nb,
                    TNR=(tn/(tn+fp) if (tn+fp) else None),
                    TPR=(tp/(tp+fn) if (tp+fn) else None))

    code_clusters = [c for c in clusters if c["domain"] == "code"]
    code_status = []
    for c in code_clusters:
        cid = c["cluster_id"]
        code_status.append({
            "cluster_id": cid, "modal_block": c["modal_block"],
            "gt": c["needs_birth_gt"],
            "raw": raw_decs[cid][0], "z": z_decs[cid][0],
            "rescued": (raw_decs[cid][0] != ("BIRTH" if c["needs_birth_gt"] else "SKIP"))
                       and (z_decs[cid][0] == ("BIRTH" if c["needs_birth_gt"] else "SKIP"))
                       if c["needs_birth_gt"] is not None else None,
        })

    return {
        "seed": seed,
        "n_clusters": len(clusters),
        "n_per_domain": n_per_domain,
        "confusion_raw": confusion(raw_decs),
        "confusion_z": confusion(z_decs),
        "code_clusters": code_status,
        "any_rescued": any(s.get("rescued") for s in code_status),
    }


def main():
    seeds = []
    for s in [0, 1, 2]:
        try:
            r = run_for_seed(s)
            seeds.append(r)
            print(f"seed {s}: {r['n_clusters']} clusters, domains={r['n_per_domain']}, "
                  f"raw TPR={r['confusion_raw']['TPR']}, z TPR={r['confusion_z']['TPR']}, "
                  f"code rescued: {r['any_rescued']}")
        except Exception as e:
            print(f"seed {s}: ERROR {e}")
            seeds.append({"seed": s, "error": str(e)})

    # Aggregate
    valid = [r for r in seeds if "error" not in r]
    n_total_code = sum(len(r["code_clusters"]) for r in valid)
    n_rescued_total = sum(1 for r in valid for s in r["code_clusters"] if s.get("rescued"))
    agg = {
        "n_seeds": len(valid),
        "n_code_clusters_total": n_total_code,
        "n_code_clusters_rescued": n_rescued_total,
        "rescue_rate": (n_rescued_total / n_total_code) if n_total_code else None,
        "raw_tpr_per_seed": [r["confusion_raw"]["TPR"] for r in valid],
        "z_tpr_per_seed": [r["confusion_z"]["TPR"] for r in valid],
    }
    out = {
        "experiment": "HOLE_9_multiseed",
        "method": "per-domain z-scoring with LOO sibling clusters, replicated across seeds 0/1/2",
        "per_seed": seeds,
        "aggregate": agg,
        "headline": (
            f"Per-domain z-scoring across {len(valid)} seeds: "
            f"{n_rescued_total}/{n_total_code} code clusters rescued. "
            f"Raw TPR per seed: {agg['raw_tpr_per_seed']}; z TPR: {agg['z_tpr_per_seed']}."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(out["headline"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
