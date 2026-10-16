#!/usr/bin/env python3

"""Generate the manuscript figures and tables (one command).

    python viz/first_plots.py

Produces:
  figures/molecules_27_3row.png (+ .svg)
  figures/tau_vs_release_square_w95_2.png
  figures/tau_vs_params_square_w95_2.png
  figures/tau_vs_arcagi2_square_w95_2.png
  figures/tau_vs_benchmarks_facet_2.png
  figures/tau_equalHB_vs_release.png
  figures/tau_decoy_vs_release.png
  figures/cra_decoy_vs_equalhb.png
  figures/seed_spread_heatmap.png
  figures/tanimoto_distance_distr.png
  figures/ensemble_rmsd_tfd_vs_nrot.png
  figures/deltae_energy_profiles.png
  figures/deltae_pair_diff_profiles.png
  figures/energy_probe_mae_heatmap.png
  figures/ConcretenessTau_GPT5_square.png
  figures/ConcretenessTau_GPT6_square.png
  figures/ConcretenessTau_all_horizontal.png
  figures/FullPlot.png
  figures/ensemble_flexibility.tsv
  figures/tau_floor2_table.tsv
  figures/paired_llm_minus_uff_floor2.tsv
  figures/terms_old_neutral_museglimmer30b.tsv (+ _agg.tsv)
  figures/models_table.tsv
  figures/models_reasoning_params.tsv

Intermediate tables written by the same steps: concreteness_vs_tau.tsv,
seed_spread_heatmap_molecules.tsv, tau_benchmark_correlations_2.tsv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HELPER_ROOT = PROJECT_ROOT / "viz_helper"


def _log_step_outputs(result: Any) -> None:
    if result is None:
        return
    if isinstance(result, Path):
        print(f"  -> {result}", flush=True)
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            _log_step_outputs(item)


def _run_step(label: str, run: Callable[[], Any]) -> None:
    print(f"\n--- {label} ---", flush=True)
    _log_step_outputs(run())


def main() -> int:
    os.chdir(PROJECT_ROOT)
    sys.path[:0] = [str(PROJECT_ROOT), str(HELPER_ROOT)]

    import mol_study as ms
    from benchmark_corr_table import write_corr_tsv as write_benchmark_corr
    from benchmark_corr_table import write_facet_figure as write_benchmark_facet
    from concreteness import build_table as build_concreteness
    from concreteness import load_setup as load_concreteness_setup
    from energy_probe_mae_heatmap import plot_mae_heatmap as write_energy_probe
    from plot_dataset_diversity_panels import write_figures as write_diversity
    from plot_decoy_vs_equalhb import collect_and_write as write_decoy_vs_equalhb
    from plot_decoy_vs_release import collect_and_write as write_decoy_vs_release
    from plot_equal_hbond_vs_release import collect_and_write as write_equal_hbond
    from plot_tau_vs_arcagi import collect_and_write as write_tau_arcagi
    from plot_tau_vs_release_square import collect_and_write as write_tau_square
    from seed_spread_heatmap import collect_and_write as write_seed_heatmap
    from tau_floor2_table import collect_and_write as write_floor2_table
    from terms_old_neutral import main as write_terms_old_neutral

    from viz.three_mol_rows import main as write_mol_grid

    def _concreteness_table():
        models, mol_to_label, mol_labels = load_concreteness_setup()
        return build_concreteness(models, mol_to_label, mol_labels)

    def _mol_study_figures():
        d, order, _mt = ms.load_prepared(ms.DEFAULT_DATA)
        return [
            ms.plot_full(d, order, ms.DEFAULT_OUTDIR),
            ms.plot_square_model(d, ms.BEST_MODELS[0], ms.DEFAULT_OUTDIR,
                                 "ConcretenessTau_GPT6_square.png",
                                 title="GPT-6 Astra", ylabel=False),
            ms.plot_square_model(d, ms.BEST_MODELS[1], ms.DEFAULT_OUTDIR,
                                 "ConcretenessTau_GPT5_square.png", title="GPT-5"),
            ms.plot_all_models_horizontal(d, order, ms.DEFAULT_OUTDIR,
                                          "ConcretenessTau_all_horizontal.png",
                                          lw=1.5, title="All 28 models"),
        ]

    def _paired_vs_uff():
        subprocess.run(
            [sys.executable, "metrics/run_paired_vs_uff.py"],
            check=True,
        )

    def _cohort_tables():
        out = []
        for name in ("models_table.tsv", "models_reasoning_params.tsv"):
            dst = PROJECT_ROOT / "figures" / name
            shutil.copyfile(PROJECT_ROOT / name, dst)
            out.append(dst)
        return out

    steps: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("tau vs release (|ΔE| >= 2.0)", write_tau_square),
        ("tau vs params (|ΔE| >= 2.0)", lambda: write_tau_square(x_mode="params")),
        ("tau vs ARC-AGI-2", lambda: write_tau_arcagi(benches=["2"])),
        ("benchmark correlations (R^2 table) + facet", write_benchmark_corr),
        ("tau vs all benchmarks (facet)", write_benchmark_facet),
        ("equal-H-bond CRA vs release", write_equal_hbond),
        ("decoy CRA vs release", write_decoy_vs_release),
        ("decoy CRA vs equal-H-bond CRA", write_decoy_vs_equalhb),
        ("seed spread heatmap (all models, min@10, |ΔE| >= 2.0)", write_seed_heatmap),
        ("tau @ 2.0 table (mean / median / SD)", write_floor2_table),
        ("dataset diversity panels + ensemble_flexibility.tsv", write_diversity),
        ("energy probe MAE heatmap", write_energy_probe),
        ("concreteness table (explanation-level counts)", _concreteness_table),
        ("ConcretenessTau + FullPlot", _mol_study_figures),
        ("27-molecule grid (PNG+SVG)", write_mol_grid),
        ("paired vs UFF TSV", _paired_vs_uff),
        ("prompt-vocabulary ablation TSVs (Muse Glimmer 30B)", write_terms_old_neutral),
        ("cohort tables (identities / reasoning+params)", _cohort_tables),
    )
    for label, run in steps:
        _run_step(label, run)
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
