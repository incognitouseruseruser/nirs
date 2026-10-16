# Repo

Code and data for *Human Language Shapes the Emerging Physical Reasoning in Large Language Models*.

Each model receives the XYZ coordinates of 30 conformers of one molecule and returns their stability ranking. Rankings are scored as conformer-ranking accuracy — reliable-pair Kendall's τ at |ΔE| ≥ 2.0 kcal/mol — against r2SCAN-3c reference energies, with UFF, MMFF94 and GFN-FF as force-field baselines.

## Installation

```bash
conda env create -f environment.yml
conda activate llm_benchmark
```

## Reproduce the figures

```bash
python viz_helper/build_texts.py   # per-run explanations -> texts/ (needed by the concreteness panels)
python viz/first_plots.py          # all manuscript figures and tables -> figures/
```

`viz/first_plots.py` writes:

- `figures/molecules_27_3row.png` (+ `.svg`)
- `figures/tau_vs_release_square_w95_2.png`, `figures/tau_vs_params_square_w95_2.png`, `figures/tau_vs_arcagi2_square_w95_2.png`, `figures/tau_vs_benchmarks_facet_2.png`
- `figures/tau_equalHB_vs_release.png`, `figures/tau_decoy_vs_release.png`, `figures/cra_decoy_vs_equalhb.png`
- `figures/seed_spread_heatmap.png`, `figures/tanimoto_distance_distr.png`, `figures/ensemble_rmsd_tfd_vs_nrot.png`, `figures/deltae_energy_profiles.png`, `figures/deltae_pair_diff_profiles.png`
- `figures/energy_probe_mae_heatmap.png`, `figures/FullPlot.png`
- `figures/ConcretenessTau_GPT5_square.png`, `figures/ConcretenessTau_GPT6_square.png`, `figures/ConcretenessTau_all_horizontal.png`
- tables: `figures/ensemble_flexibility.tsv`, `figures/tau_floor2_table.tsv`, `figures/paired_llm_minus_uff_floor2.tsv`, `figures/terms_old_neutral_museglimmer30b.tsv` (+ `_agg`), `figures/models_table.tsv`, `figures/models_reasoning_params.tsv`

Rerunning inference is optional and needs `OPENROUTER_API_KEY` in `.env`, plus a local Ollama server for the open-weight models.

## License

MIT — see `LICENSE`.
