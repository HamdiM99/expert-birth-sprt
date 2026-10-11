# Sequential Testing for Expert Birth in Self-Evolving Mixture-of-Experts

Code, data and result files for the paper

> Mustapha Hamdi. *Sequential Testing for Expert Birth in Self-Evolving Mixture-of-Experts.* arXiv preprint, 2026.

Self-evolving MoE systems spawn a new expert (a LoRA adapter) when a cluster of queries looks
uncertain. **Observation-gated birth** defers that decision: a candidate cluster enters a
shadow-observation phase and a Sequential Probability Ratio Test (fixed-σ SPRT or GLR-SPRT)
decides BIRTH, SKIP or CONTINUE with explicit type-I/II error targets. A post-hoc counterfactual
oracle (LoRA on every cluster + LLM judge) evaluates the gate non-circularly.

## Contents

```
code/poc_observation/     controlled experiment: gates, SPRT, UQ signals, clustering, LoRA,
                          counterfactual ground truth, analysis scripts (analysis_holes/)
data/                     deployment traces (13 proposals), multi-UQ replay (185 obs.),
                          embeddings, oracle outcomes, diagnostics
results/                  per-seed controlled-experiment runs (token entropy, ensemble,
                          MC dropout) and per-analysis JSON / LaTeX tables
figures/make_figures.py   regenerates every data figure of the paper from data/ + results/
REPRODUCTION.md           paper table / figure → script → output file index
```

## Reproducing

**Figures (CPU, seconds).** All data figures are rebuilt from the released files:

```bash
pip install numpy scipy matplotlib pymupdf
python figures/make_figures.py . figures/out paper.pdf
```

(`paper.pdf` is only used to crop the two schematic diagrams, Figs. 1–2; the other figures are
computed from `data/` and `results/`.)

**Controlled experiment (one NVIDIA L4, ~80 min per 3 seeds).**

```bash
cd code/poc_observation
pip install -r requirements.txt
python run_experiment.py --seeds 0 1 2 --use-judge
```

**Post-hoc counterfactual oracle on deployment proposals** (needs `DEEPSEEK_API_KEY` and a GPU):

```bash
cd code/poc_observation
python posthoc_counterfactual.py --traces ../../data/per_proposal_traces.jsonl \
    --out ../../results/HOLE_oracle_replay.json
```

Analysis scripts in `code/poc_observation/analysis_holes/` read `data/` and write `results/`;
`REPRODUCTION.md` maps each table and figure of the paper to its script and output.

## Citation

```bibtex
@misc{hamdi2026sequential,
  title  = {Sequential Testing for Expert Birth in Self-Evolving Mixture-of-Experts},
  author = {Hamdi, Mustapha},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT — see [LICENSE](LICENSE).
