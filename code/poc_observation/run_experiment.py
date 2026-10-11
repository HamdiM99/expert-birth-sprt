"""POC orchestrator — Eager vs Observation × seeds, full pipeline.

Stages:
  1. Load workload(seed) (or build it)
  2. Embed all queries (sentence-transformers)
  3. Compute UQ signal per query (default: token_entropy)
  4. Cluster via HDBSCAN → candidate clusters
  5. Run BOTH gates (eager + observation) on the same clusters
  6. Counterfactual: train LoRA on every detected cluster + score on gold
  7. Save per-seed results

Usage:
  python run_experiment.py --seeds 0 1 2
  python run_experiment.py --seeds 0 --skip-counterfactual    # fast path

Outputs: results/seed{N}/{decisions.json, counterfactual.json, embeddings.npz}
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--uq-signal", default="token_entropy",
                    choices=["token_entropy", "ensemble_disagreement", "mc_dropout"])
    ap.add_argument("--use-judge", action="store_true",
                    help="Use Anthropic/OpenAI LLM judge (else heuristic)")
    ap.add_argument("--skip-counterfactual", action="store_true",
                    help="Skip the LoRA counterfactual training (fast smoke test)")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] backbone={args.backbone} embedder={args.embedder} signal={args.uq_signal}")

    # Load models lazily so smoke-tests of data/cluster don't need GPU
    print("[setup] Loading backbone (Qwen2.5-1.5B-Instruct)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    print("[setup] Loading embedder...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embedder)

    from core import uq_signals, cluster as cluster_mod, gate_eager, gate_observation
    from eval import score_answers, counterfactual, confusion

    score_fn = lambda pred, ref, q: score_answers.score_answer(
        pred, ref, q, use_judge=args.use_judge)

    if not args.skip_counterfactual:
        from core.lora_train import train_lora
    else:
        train_lora = None

    for seed in args.seeds:
        print(f"\n{'='*70}\n[seed {seed}] starting\n{'='*70}")
        t0 = time.time()
        seed_dir = results_dir / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)

        # ── Workload ────────────────────────────────────────────
        wpath = ROOT / "data" / f"workload_seed{seed}.jsonl"
        if not wpath.exists():
            print(f"[seed {seed}] building workload...")
            from data.build_workload import build
            counts = {"true_need_med": 50, "true_need_legal": 50, "true_need_code": 50,
                      "trap_a": 50, "trap_b": 50, "noise": 250}
            items = build(seed, counts)
            with wpath.open("w") as f:
                for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
        items = [json.loads(l) for l in wpath.open()]
        print(f"[seed {seed}] {len(items)} queries")

        # ── Embed ───────────────────────────────────────────────
        embs = embedder.encode([it["query"] for it in items], show_progress_bar=True,
                               batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        for i, e in enumerate(embs):
            items[i]["embedding"] = e.tolist()

        # ── UQ per query (used for clustering filter) ───────────
        print(f"[seed {seed}] computing UQ ({args.uq_signal}) per query...")
        for i, it in enumerate(items):
            it["uq"] = uq_signals.compute(args.uq_signal, model, tok, it["query"],
                                          max_new_tokens=24)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(items)} (last uq={it['uq']:.3f})")

        # ── Cluster ─────────────────────────────────────────────
        clusters = cluster_mod.detect_candidates(items, min_cluster_size=15, min_uq=0.30)
        print(f"[seed {seed}] {len(clusters)} candidate clusters")
        for c in clusters:
            print(f"  cluster {c['cluster_id']}: size={c['size']} "
                  f"mean_uq={c['mean_uq']:.3f} cohesion={c['cohesion']:.3f} "
                  f"blocks={c['block_distribution']}")

        # ── Gates ───────────────────────────────────────────────
        print(f"[seed {seed}] running EAGER gate...")
        eager_decisions = gate_eager.decide(clusters)

        print(f"[seed {seed}] running OBSERVATION gate (SPRT, target=50)...")
        shadow_uq = lambda q: uq_signals.compute(args.uq_signal, model, tok, q, max_new_tokens=24)
        obs_decisions = gate_observation.decide(clusters, items, shadow_uq,
                                                target_obs=50, min_obs=20, uq_threshold=0.55)

        # ── Counterfactual (the costly part) ────────────────────
        if not args.skip_counterfactual:
            print(f"[seed {seed}] running counterfactual LoRA training × {len(clusters)}...")
            cf_results = []
            for c in clusters:
                cf = counterfactual.evaluate_cluster(
                    c, model, tok, score_fn,
                    lora_train_fn=train_lora,
                    train_lora=True,
                    output_dir=str(seed_dir / "lora_adapters"),
                    items=items,
                )
                cf_results.append(cf)
                print(f"  cluster {c['cluster_id']}: backbone={cf.get('backbone_score'):.3f} "
                      f"lora={cf.get('lora_score'):.3f} delta={cf.get('delta'):.3f}"
                      if cf.get("delta") is not None else f"  cluster {c['cluster_id']}: {cf.get('error', 'no delta')}")
            gt = {c["cluster_id"]: counterfactual.label_needs_birth(cf)
                  for c, cf in zip(clusters, cf_results)}
        else:
            cf_results = []
            # Cheap fallback: gt = majority block of cluster has gt_needs_birth=True
            gt = {}
            for c in clusters:
                lab = cluster_mod.assign_cluster_ground_truth(c, items)
                if lab is not None: gt[c["cluster_id"]] = lab

        # ── Confusion ──────────────────────────────────────────
        cm_eager = confusion.confusion(eager_decisions, gt)
        cm_obs = confusion.confusion(obs_decisions, gt)
        sweep = confusion.threshold_sweep(
            obs_decisions, gt,
            thresholds=[0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80])

        seed_summary = {
            "seed": seed,
            "n_items": len(items),
            "n_clusters": len(clusters),
            "ground_truth": {str(k): bool(v) for k, v in gt.items()},
            "confusion_eager": cm_eager,
            "confusion_observation": cm_obs,
            "threshold_sweep": sweep,
            "wall_time_s": time.time() - t0,
        }
        (seed_dir / "decisions.json").write_text(json.dumps({
            "eager": eager_decisions,
            "observation": obs_decisions,
        }, indent=2, default=str))
        (seed_dir / "counterfactual.json").write_text(json.dumps(cf_results, indent=2, default=str))
        (seed_dir / "summary.json").write_text(json.dumps(seed_summary, indent=2, default=str))
        # Persist embeddings for reproducibility
        import numpy as np
        np.savez_compressed(seed_dir / "embeddings.npz",
                            embeddings=np.stack([np.asarray(it["embedding"], dtype=np.float32) for it in items]),
                            uqs=np.asarray([it["uq"] for it in items], dtype=np.float32))

        print(f"[seed {seed}] done in {time.time()-t0:.0f}s")
        print(f"  Eager:  P={cm_eager['precision']:.2f} R={cm_eager['recall']:.2f} → {cm_eager['n_births']} births")
        print(f"  Obs:    P={cm_obs['precision']:.2f} R={cm_obs['recall']:.2f} → {cm_obs['n_births']} births")

    print("\n[done] all seeds processed.")
    print(f"Run `python analyze.py --results-dir {results_dir}` for tables/figures.")


if __name__ == "__main__":
    sys.exit(main())
