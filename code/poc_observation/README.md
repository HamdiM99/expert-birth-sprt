# Observation-Gated Birth — POC code

A self-contained experiment comparing two birth policies
for self-evolving MoE systems. Designed to run in **~2h on an L4**, produces
**1 table + 3 figures + 1 JSON aggregate** ready for the paper.

## Why this exists

In MoE-evolutive literature (BAM, SEED-MoE, Lifelong-MoE), a new expert is
spawned the moment a high-uncertainty cluster is detected — *eager birth*.
Our claim: eager triggers waste compute on clusters the backbone could
already handle. We propose **observation-gated birth**: detect → observe
shadow forwards → SPRT decision → birth or abandon.

## Layout

```
poc_observation/
├── data/
│   ├── build_workload.py    # 6-block × N queries × 3 seeds (deterministic)
│   ├── gold/                # 10 Q-A pairs per block (med, legal, code, traps)
│   └── workload_seed{0,1,2}.jsonl   # generated on first run
├── core/
│   ├── uq_signals.py        # token_entropy / ensemble_disagreement / mc_dropout
│   ├── cluster.py           # HDBSCAN + scoring (cohesion, mean_uq, trigger)
│   ├── gate_eager.py        # baseline: cluster ≥ 15, mean_uq ≥ 0.30 → BIRTH
│   ├── gate_observation.py  # SPRT-based stopping (Wald 1945) + threshold fallback
│   └── lora_train.py        # rank-8 LoRA, 1 epoch, 50 examples per cluster
├── eval/
│   ├── score_answers.py     # ROUGE-L + Anthropic/OpenAI judge (heuristic fallback)
│   ├── counterfactual.py    # train LoRA on EVERY cluster → measure delta vs backbone
│   └── confusion.py         # TP/FP/FN/TN, P/R/F1, threshold sweep, multi-seed agg
├── configs/{eager,obs}.yaml
├── run_experiment.py        # orchestrator: 2 gates × N seeds, full pipeline
├── analyze.py               # tables + figures from per-seed summaries
└── requirements.txt
```

## How the comparison is honest (anticipating reviewers)

| Question a reviewer will ask | Answer in this code |
|---|---|
| "Eager baseline — strawman or real?" | `gate_eager.py` operationalizes the standard threshold trigger from BAM/SEED-MoE: `size ≥ 15 ∧ mean_uq ≥ 0.3`. Both gates run on the *same* clusters. |
| "How is `needs_birth` ground truth defined?" | `eval/counterfactual.py` trains a LoRA on every cluster (regardless of gate decision), measures delta vs backbone on a held-out gold set. `needs_birth = TRUE` iff Δ ≥ 0.10. This is the *expensive but defensible* signal. |
| "Threshold cherry-picked?" | `confusion.threshold_sweep` gives the full Pareto frontier; production threshold (0.55) is a config knob, not a hand-tuned tweak. |
| "Single seed?" | `run_experiment.py --seeds 0 1 2`; `analyze.py` reports mean±std. |
| "Why token entropy?" | Three signals available (`uq_signals.py`); ablate by setting `--uq-signal mc_dropout` etc. |
| "Cluster homogeneous?" | `data/build_workload.py` includes 2 *trap* blocks with deliberately mixed/easy queries that look hard. The gate must reject these. |
| "Theoretical basis for SPRT?" | `gate_observation.sprt_decision` implements Wald 1945 SPRT with α=β=0.05 type-I/II bounds. TARGET_OBS is the cap; SPRT typically decides earlier. |

## Quick start

### 1. Install

```bash
cd poc_observation
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Build workloads (deterministic)

```bash
./venv/bin/python data/build_workload.py --seeds 0 1 2
# → data/workload_seed{0,1,2}.jsonl  (500 queries each)
```

### 3. Smoke test (no GPU, no LoRA — ~30 s)

```bash
./venv/bin/python data/build_workload.py --seeds 99 --med 5 --legal 5 --code 5 --trap-a 5 --trap-b 5 --noise 10
# Just verifies workload generation works
```

### 4. Full run on L4 (single seed, ~30 min)

```bash
./venv/bin/python run_experiment.py --seeds 0 --skip-counterfactual
```

This skips the costly LoRA training. Ground truth falls back to majority block.
Use this to validate the pipeline end-to-end before committing to full compute.

### 5. Full paper run (3 seeds, with counterfactual, ~3 h on L4)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # optional, for LLM judge
./venv/bin/python run_experiment.py --seeds 0 1 2 --use-judge
./venv/bin/python analyze.py --results-dir ./results
```

Outputs:
- `results/seed{0,1,2}/{summary,decisions,counterfactual}.json`
- `results/aggregate.json`
- `results/RESULTS.md` (the paper table)
- `results/figures/{fig1_pareto,fig2_traces,fig3_confusion}.png`

## Knobs

| Variable | Default | Purpose |
|---|---|---|
| `--backbone` | `Qwen/Qwen2.5-1.5B-Instruct` | Base LLM (try `Qwen2.5-7B-Instruct` for ablation) |
| `--embedder` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--uq-signal` | `token_entropy` | One of `{token_entropy, ensemble_disagreement, mc_dropout}` |
| `--use-judge` | off | Use Anthropic/OpenAI judge (else heuristic) |
| `--skip-counterfactual` | off | Skip LoRA training (cheap dry-run) |

## Expected results (rough — to be confirmed)

```
Eager:        Precision ≈ 0.55–0.65   Recall ≈ 1.00     ~5 LoRA trainings
Observation:  Precision ≈ 0.85–1.00   Recall ≈ 0.85    ~3 LoRA trainings
              → ~40% LoRA cost saved at the price of ~15% recall
```

The paper's punchline is *not* "observation is better" — it's
**"observation lets you trade recall for compute *predictably*, with
SPRT bounds, on a Pareto frontier you can sweep."**

## Compute budget

| Stage | Time on L4 |
|---|---|
| Workload build | < 1 s |
| Embeddings (500 queries × 3 seeds) | ~1 min |
| UQ signal per query (500 × 3) | ~12 min |
| Cluster + Eager gate | < 1 min |
| Observation gate (SPRT, ~5 clusters × 30 obs avg) | ~10 min |
| Counterfactual LoRA (5 clusters × 3 seeds × 3 min) | ~45 min |
| LoRA eval on gold | ~10 min |
| Total per full run | **~80 min** |

## Citing / re-using

See the top-level README for the paper reference and citation.

## What's NOT in this POC (deliberately deferred)

- Online deployment (the deployment traces are released in `data/`; the serving stack is not part of this repository)
- Multi-domain founder bootstrap (5 founders trained on real data) — POC uses
  raw backbone only, since gate decisions are independent of founder identity
- Drift-driven re-birth — orthogonal to the birth-gate question
- DARE merging vs LoRA — the POC compares "spawn or not", not how to spawn
