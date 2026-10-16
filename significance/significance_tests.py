#!/usr/bin/env python3
"""Statistical tests: each model vs random baseline (per molecule).

Tests (one-sided, α=0.05 by default):
  Kendall τ     — Wilcoxon signed-rank vs 0
  min@10        — Monte Carlo vs molecule-specific Bernoulli(p_random)

Note: Wilcoxon signed-rank test drops zero-difference pairs (tau=0).
This means the effective sample size may be smaller than the number of molecules.
For models with many tau=0 (random-like performance), the test has less power.

Examples:
  python significance_tests.py
  python significance_tests.py --tsv
  python significance_tests.py --include-partial
  python significance_tests.py --min10-heuristic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import add_baseline_arguments, geom_seed_dirs
from metrics.option2_metrics import load_valid_answers
from significance.significance_common import (
    ModelSignificance,
    build_random_baselines,
    load_bundle_meta,
    min10_pvalue_mc,
    per_molecule_metrics,
    sorted_molecules,
)
from summary.benchmark_summary_table import DEFAULT_MODELS


def test_model_vs_random(
    name: str,
    answers: dict[str, list[str]],
    meta: dict,
    baselines,
    molecules: list[str],
    *,
    n_molecules: int,
    mc_trials_min10: int,
    mc_seed_min10: int,
) -> ModelSignificance | None:
    metrics = per_molecule_metrics(answers, meta, molecules)
    if metrics is None:
        return None

    n = len(metrics.molecules)
    rm10 = np.array([baselines.rand_m10_p[m] for m in metrics.molecules])

    p_tau = stats.wilcoxon(
        metrics.taus, alternative="greater", zero_method="wilcox"
    ).pvalue
    p_m10 = min10_pvalue_mc(
        metrics.m10, rm10, trials=mc_trials_min10, seed=mc_seed_min10
    )

    return ModelSignificance(
        name=name,
        n=n,
        n_molecules=n_molecules,
        partial=n < n_molecules,
        tau_mean=float(metrics.taus.mean()),
        m10_hits=int(metrics.m10.sum()),
        p_tau=float(p_tau),
        p_m10=float(p_m10),
    )


def collect_significance_rows(
    bundle_dir: Path,
    *,
    include_partial: bool,
    mc_trials_min10: int,
    mc_seed_min10: int,
    aggregate_bundles: list[Path] | None = None,
    baseline: str = "gfn2-xtb",
    deltae_json: Path | None = None,
) -> list[ModelSignificance]:
    meta = load_bundle_meta(bundle_dir, baseline=baseline, deltae_json=deltae_json)
    molecules = sorted_molecules(meta)
    n_molecules = len(molecules)
    baselines = build_random_baselines(meta)

    if aggregate_bundles:
        all_meta = {bundle_dir: meta}
        for b in aggregate_bundles:
            all_meta[b] = load_bundle_meta(b, baseline=baseline, deltae_json=deltae_json)

    rows: list[ModelSignificance] = []
    for name, filename in DEFAULT_MODELS:
        if filename is None:
            continue

        if aggregate_bundles:
            per_mol_taus: dict[str, list[float]] = {}
            for bdir, bmeta in all_meta.items():
                ans_path = bdir / filename
                if not ans_path.is_file():
                    continue
                banswers = load_valid_answers(ans_path, bmeta)
                if not banswers:
                    continue
                bmetrics = per_molecule_metrics(banswers, bmeta, molecules)
                if bmetrics is None:
                    continue
                for mol, tau in zip(bmetrics.molecules, bmetrics.taus):
                    per_mol_taus.setdefault(mol, []).append(tau)
            if len(per_mol_taus) < 10:
                continue
            n_bundles_with_answers = sum(
                1 for bdir in all_meta if (bdir / filename).is_file()
            )
            if n_bundles_with_answers < 2:
                continue
            agg_mols = sorted(per_mol_taus.keys())
            agg_taus = np.array([np.mean(per_mol_taus[m]) for m in agg_mols])

            p_tau = stats.wilcoxon(
                agg_taus, alternative="greater", zero_method="wilcox"
            ).pvalue

            rows.append(ModelSignificance(
                name=name,
                n=len(agg_mols),
                n_molecules=n_molecules,
                partial=len(agg_mols) < n_molecules,
                tau_mean=float(agg_taus.mean()),
                m10_hits=0,
                p_tau=float(p_tau),
                p_m10=1.0,
            ))
        else:
            answers = load_valid_answers(bundle_dir / filename, meta)
            if not answers:
                continue
            if not include_partial and len(answers) < n_molecules:
                continue
            row = test_model_vs_random(
                name,
                answers,
                meta,
                baselines,
                molecules,
                n_molecules=n_molecules,
                mc_trials_min10=mc_trials_min10,
                mc_seed_min10=mc_seed_min10,
            )
            if row is not None:
                rows.append(row)
    return rows


def print_significance_table(
    rows: list[ModelSignificance], *, tsv: bool, alpha: float
) -> None:
    sep = "\t" if tsv else "  "
    header = [
        "Model",
        "n",
        "tau",
        "min@10",
        "p(tau)",
        "p(min@10)",
        f"sig a={alpha:g}",
    ]
    print(sep.join(header))
    for r in rows:
        n_show = str(r.n) if not r.partial else f"{r.n}*"
        sig = ",".join(
            label
            for label, p in (
                ("tau", r.p_tau),
                ("min@10", r.p_m10),
            )
            if p < alpha
        ) or "---"
        print(
            sep.join(
                [
                    r.name,
                    n_show,
                    f"{r.tau_mean:+.3f}",
                    f"{r.m10_hits}/{r.n}",
                    f"{r.p_tau:.3f}",
                    f"{r.p_m10:.3f}",
                    sig,
                ]
            )
        )
    if any(r.partial for r in rows):
        print()
        print("* partial run -- not all molecules in bundle")


def print_min10_heuristic(*, alpha: float) -> None:
    """Crude binomial vs 10/30 and note on molecule-specific MC."""
    print("\n=== min@10 heuristic (crude binomial vs 10/30, one-sided) ===")
    p_null = 10 / 30
    for hits, total in ((16, 30), (20, 30), (22, 30)):
        p = stats.binomtest(hits, total, p_null, alternative="greater").pvalue
        mark = " *" if p < alpha else ""
        print(f"  {hits}/{total}: p = {p:.3f}{mark}")
    print("  (MC test with molecule-specific p_random is in main table; see MANUSCRIPT_BRIEF.md)")


def print_bonferroni_note(n_models: int, n_tests: int = 2, alpha: float = 0.05) -> None:
    total = n_models * n_tests
    print()
    print(
        f"Bonferroni (one-sided): {n_tests} tests x {n_models} models = {total} comparisons; "
        f"a_corr ~ {alpha / total:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, default=None)
    add_baseline_arguments(ap)
    ap.add_argument("--aggregate", action="store_true",
                    help="Aggregate per-molecule tau across seeds (mean) and test on that")
    ap.add_argument("--bundle-dirs", type=Path, nargs="+",
                    help="Additional bundles for aggregation (used with --aggregate)")
    ap.add_argument("--tsv", action="store_true")
    ap.add_argument(
        "--include-partial",
        action="store_true",
        help="Include models with incomplete molecule coverage (e.g. GPT-5.5 R)",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--min10-heuristic",
        action="store_true",
        help="Also print crude binomial thresholds for min@10",
    )
    ap.add_argument("--mc-trials-min10", type=int, default=200_000)
    ap.add_argument("--mc-seed-min10", type=int, default=42)
    args = ap.parse_args()

    bundle_dir = args.bundle_dir if args.bundle_dir is not None else geom_seed_dirs()[0][1]

    rows = collect_significance_rows(
        bundle_dir,
        include_partial=args.include_partial,
        mc_trials_min10=args.mc_trials_min10,
        mc_seed_min10=args.mc_seed_min10,
        aggregate_bundles=args.bundle_dirs if args.aggregate else None,
        baseline=args.baseline,
        deltae_json=args.deltae_json,
    )
    print_significance_table(rows, tsv=args.tsv, alpha=args.alpha)
    print_bonferroni_note(len(rows), alpha=args.alpha)
    if args.min10_heuristic:
        print_min10_heuristic(alpha=args.alpha)


if __name__ == "__main__":
    main()
