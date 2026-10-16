#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.models_config import (
    add_baseline_arguments,
    default_models_table,
    geom_seed_dirs,
)
from metrics.baseline import prepare_meta
from metrics.conformer_min_in_topk import letters_at_global_minimum, min_in_topk
from metrics.kendall_tau_ranking import kendall_for_molecule
from metrics.option2_metrics import load_valid_answers
from metrics.random_baseline import min_at_k_random

DEFAULT_MODELS: list[tuple[str, str | None]] = default_models_table()


def score_answers(
    answers: dict[str, list[str]], meta: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    taus: list[float] = []
    m1 = m3 = m5 = m10 = 0
    for mol, rnk in answers.items():
        de = meta[mol]["letter_to_deltaE_kcal_mol"]
        tau, _ = kendall_for_molecule(de, rnk)
        taus.append(tau)
        min_letters = letters_at_global_minimum(de)
        m1 += int(min_in_topk(rnk, min_letters, 1))
        m3 += int(min_in_topk(rnk, min_letters, 3))
        m5 += int(min_in_topk(rnk, min_letters, 5))
        m10 += int(min_in_topk(rnk, min_letters, 10))
    n = len(answers)
    return {
        "n": n,
        "n_molecules": len(meta),
        "tau": round(sum(taus) / n, 3) if n else float("nan"),
        "m1": m1,
        "m3": m3,
        "m5": m5,
        "m10": m10,
    }


def random_row(meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n_mol = len(meta)
    m1 = m3 = m5 = m10 = 0.0
    for row in meta.values():
        de = row["letter_to_deltaE_kcal_mol"]
        n = len(de)
        min_e = min(de.values())
        m = sum(1 for v in de.values() if v == min_e)
        m1 += min_at_k_random(n, m, 1)
        m3 += min_at_k_random(n, m, 3)
        m5 += min_at_k_random(n, m, 5)
        m10 += min_at_k_random(n, m, 10)
    return {
        "n": n_mol,
        "n_molecules": n_mol,
        "tau": 0.0,
        "m1": round(m1),
        "m3": round(m3),
        "m5": round(m5),
        "m10": round(m10),
        "partial": False,
    }


def fmt_tau(x: float) -> str:
    return f"{x:+.3f}"


def fmt_n(n: int, n_molecules: int, partial: bool) -> str:
    s = str(n)
    if partial:
        s += "*"
    return s


def print_table(rows: list[dict[str, Any]], *, tsv: bool) -> None:
    sep = "\t" if tsv else "  "
    header = ["Model", "n", "τ", "min@1", "min@3", "min@5", "min@10"]
    print(sep.join(header))
    for r in rows:
        n = r["n"]
        n_show = n if n else r["n_molecules"]
        print(sep.join([
            r["name"],
            fmt_n(n, r["n_molecules"], r.get("partial", False)),
            fmt_tau(r["tau"]) if n else "—",
            f"{int(r['m1'])}/{n_show}",
            f"{int(r['m3'])}/{n_show}",
            f"{int(r['m5'])}/{n_show}",
            f"{int(r['m10'])}/{n_show}",
        ]))
    if any(r.get("partial") for r in rows):
        print()
        print("* interim — not all molecules in bundle yet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, default=None,
                    help="Seed-42 bundle (default: geom-qm9/prompt_30x30_NEW from models.yaml)")
    add_baseline_arguments(ap)
    ap.add_argument("--tsv", action="store_true", help="Tab-separated (for copy-paste)")
    args = ap.parse_args()

    bundle_dir = args.bundle_dir if args.bundle_dir is not None else geom_seed_dirs()[0][1]

    meta_path = bundle_dir / "prompt_metadata.jsonl"
    meta = prepare_meta(meta_path, baseline=args.baseline, deltae_json=args.deltae_json)
    n_molecules = len(meta)

    rows: list[dict[str, Any]] = []
    for name, filename in DEFAULT_MODELS:
        if filename is None:
            row = random_row(meta)
            row["name"] = name
            row["partial"] = False
            rows.append(row)
            continue
        answers = load_valid_answers(bundle_dir / filename, meta)
        row = score_answers(answers, meta)
        row["name"] = name
        row["partial"] = row["n"] < n_molecules
        rows.append(row)

    print_table(rows, tsv=args.tsv)


if __name__ == "__main__":
    main()
