# Reproduction Recipe — table / figure index

Each row points to the script and the JSON it writes. Numbers reported in the
paper match these files exactly. All commands assume `cd code/poc_observation`.

## Body tables

| Paper table / fig | Script | Output JSON |
|---|---|---|
| Tab. main (controlled, 3-seed) | `run_experiment.py --seeds 0 1 2 --use-judge` | `../../results/token_entropy/seed{0,1,2}/summary.json` |
| Tab. sweep (Pareto, controlled) | `analyze.py --sweep tau-u` | `../../results/token_entropy/seed*/sweep.json` |
| Tab. deployment (proxy GT) | `analysis_holes/sprt_diagnostics.py` | `../../data/sprt_diagnostics.json` |
| **Tab. proxy-vs-oracle** | `analysis_holes/HOLE_7_oracle_metrics.py` | `../../results/HOLE_7_oracle_metrics.json` |
| Tab. baselines (SPRT vs batched-probe vs Page-Hinkley vs **BOCPD**) | `analysis_holes/HOLE_8_page_hinkley.py` + `analysis_holes/batched_probe_baseline.py` + `analysis_holes/HOLE_22_bocpd_baseline.py` | `../../results/HOLE_8_page_hinkley.json` + `analysis_holes/batched_probe_results.json` + `../../results/HOLE_22_bocpd_baseline.json` |

## Body figures

| Figure | Source | Note |
|---|---|---|
| SPRT decision boundaries (schematic) | `figures/schematic_sprt.png` | static, no script |
| Pareto comparison across studies | `figures/pareto_compare.png` | rebuilt by `analyze.py --plot pareto` |

## Appendix

| Appendix § | Script | Output |
|---|---|---|
| App. UQ signal ablation (3-seed ensemble) | `run_experiment.py --uq ensemble --seeds 0 1 2` | `../../results/ensemble/seed{0,1,2}/summary.json` |
| App. MC dropout (seed 0) | `run_experiment.py --uq mcdropout --seed 0` | `../../results/mcdropout/seed0/summary.json` |
| App. Diagnostics (Q-Q, autocorr, bootstrap, KS, ADF, KPSS) | `analysis_holes/sprt_diagnostics.py` + `analysis_holes/HOLE_3_drift_stationarity.py` | `../../data/sprt_diagnostics.json`, `../../results/HOLE_3_drift_stationarity.json` |
| **App. Block bootstrap (Tab. block-bootstrap)** | `analysis_holes/HOLE_5_block_bootstrap.py` | `../../results/HOLE_5_block_bootstrap.json` |
| App. HDBSCAN sensitivity (ARI sweep) | `analysis_holes/HOLE_6_hdbscan_sensitivity.py` | `../../results/HOLE_6_hdbscan_sensitivity.json` |
| App. Calibration (isotonic Brier) | `analysis_holes/HOLE_4_uq_calibration.py` | `../../results/HOLE_4_uq_calibration.json` |
| App. Learned-gate baseline | `analysis_holes/HOLE_1_learned_gate.py` | `../../results/HOLE_1_learned_gate.json` |
| App. u_min sensitivity | `analysis_holes/HOLE_2_umin_sensitivity.py` | `../../results/HOLE_2_umin_sensitivity.json` |
| App. UQ discrimination (ROC/PR) | `analysis_holes/uq_discrimination.py` | `analysis_holes/uq_discrimination.json` |
| **App. Per-domain z-scoring (honest negative)** | `analysis_holes/HOLE_9_per_domain_calibration.py` | `../../results/HOLE_9_per_domain_calibration.json` |
| App. Post-hoc counterfactual on deployment | `posthoc_counterfactual.py` | `../../data/posthoc_counterfactual.json` |
| **App. multi-uq (Lever 2) — multi-UQ-signal SPRT/GLR** | `analysis_holes/HOLE_21_lever2_standalone_replay.py` → `HOLE_21_multi_uq_robustness.py` → `HOLE_21_decision_agreement_table.py` | `../../data/multi_uq_observations.jsonl`, `../../results/HOLE_21_uq_signal_robustness.json`, `../../results/HOLE_21_decision_agreement_table.tex` |
| **App. lora-cap (Q5) — LoRA-capacity ablation** *(rolling update)* | `analysis_holes/HOLE_23_lora_ablation_pod.py` (GPU) → `HOLE_23_lora_ablation_judge.py` (local heuristic ROUGE-L) | `../../data/lora_ablation_pairs.jsonl`, `../../results/HOLE_23_lora_ablation.json` |
| **App. hole24 — Calibrated multi-UQ vs oracle (2/7→4/7)** | `analysis_holes/HOLE_24_calibrated_uq_oracle.py` (CPU) | `../../results/HOLE_24_calibrated_uq_oracle.json`, `../../results/HOLE_24_signal_oracle_table.tex` |
| **App. hole26 — MASS-style gradient surge baseline (4/7→5/7)** | `analysis_holes/HOLE_26_mass_gradient_surge_pod.py` (GPU, ~84s L4) → `HOLE_26_score_local.py` (CPU) | `../../results/HOLE_26_mass_gradient_surge.json`, `../../results/HOLE_26_mass_oracle_score.json` |
| **HOLE 27 — Judge consensus stability (T={0,0.5,1})** | `analysis_holes/HOLE_27_judge_consensus.py` (needs `DEEPSEEK_API_KEY`) | `../../results/HOLE_27_judge_consensus.json` |

## Environment

```text
Python 3.11
torch >= 2.2 (with CUDA 12.x)
transformers >= 4.42
peft >= 0.11
sentence-transformers >= 2.7
hdbscan >= 0.8.33
scikit-learn >= 1.4
scipy >= 1.13
statsmodels >= 0.14    # ADF, KPSS
arch >= 6.3            # circular block bootstrap
```

Pinned in `code/poc_observation/requirements.txt`. GPU recommended for the
LoRA training in the post-hoc oracle (~3 min per cluster on L4); the rest of
the pipeline runs on CPU.

## Determinism

All seeds set globally (`torch`, `numpy`, `random`, `transformers`). HDBSCAN
is deterministic given fixed input and parameters. The DeepSeek judge is
called with `temperature=0` and `seed=42` where the API exposes it; some
residual non-determinism may persist in the API response and is the reason
each post-hoc judgment is repeated 3× with A/B position randomization and
majority-voted (see `posthoc_counterfactual.py`).
