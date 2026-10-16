"""Explanation-text concreteness analysis.

Two independent pipelines:

1. REGEX analysis (fast, deterministic): per-explanation counts of H-bond
   vocabulary, quantified measurements and energy mentions, plus vague-term
   counts; writes ``concreteness_vs_tau.tsv`` and the grid/quantile figures.
2. LLM extraction (slow, optional via ``--llm``): asks a local Ollama model to
   split each explanation into SPECIFIC vs VAGUE effects; writes
   ``specific_effects_vs_tau.tsv``.

    python viz_helper/concreteness.py            # regex tables + figures
    python viz_helper/concreteness.py --llm      # also run the LLM extractor
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "viz_helper"))

from config.models_config import models_by_id

TEXTS_DIR = Path("texts")
LABELS_TSV = Path("figures/seed_spread_heatmap_molecules.tsv")
OUT_DIR = Path("figures")
TABLE_TSV = OUT_DIR / "concreteness_vs_tau.tsv"
RATIO_CURVE_TSV = OUT_DIR / "concreteness_ratio_curve.tsv"
QTY_CURVE_TSV = OUT_DIR / "concreteness_qty_curve.tsv"
SPECIFIC_TSV = OUT_DIR / "specific_effects_vs_tau.tsv"
SEEDS = ("42", "43", "44")


def load_setup(texts_dir: Path = TEXTS_DIR, labels_tsv: Path = LABELS_TSV):
    """(MODELS, MOL_TO_LABEL, MOL_LABELS) with all models that have per-seed texts."""
    available = sorted(p.stem for p in (texts_dir / "seed42").glob("*.jsonl"))
    specs = models_by_id()
    models = {
        mid: (
            specs[mid].display if mid in specs else mid,
            specs[mid].color if mid in specs else "#6b7280",
        )
        for mid in available
    }
    order = [
        line.split("\t")[1]
        for line in labels_tsv.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("label")
    ]
    mol_to_label = {m: i for i, m in enumerate(order, start=1)}
    return models, mol_to_label, set(range(1, len(order) + 1))


ENERGY_RE = re.compile(
    r"\bkcal\b|\bkilocalor\w*|\bkj\b|\bkilojoule\w*|\bhartree\w*|\ba\.u\.|\bE_?h\b",
    re.IGNORECASE,
)
KCAL_RE = re.compile(r"\bkcal\b|\bkilocalor\w*", re.IGNORECASE)
KJ_RE = re.compile(r"\bkj\b|\bkilojoule\w*", re.IGNORECASE)
HARTREE_RE = re.compile(r"\bhartree\w*|\ba\.u\.|\bE_?h\b", re.IGNORECASE)
ENERGY_VAL_RE = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*[-–—]\s*(\d+(?:\.\d+)?))?\s*"
    r"(\bkcal\b|\bkilocalor\w*|\bkj\b|\bkilojoule\w*|\bhartree\w*|\ba\.u\.|\bE_?h\b)",
    re.IGNORECASE,
)
QTY_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:Å|å|angstroms?|degrees?|\bdeg\b|°)", re.IGNORECASE)
HBOND_RE = re.compile(r"hydrogen bond|h-?bond|donor|acceptor|h[·.]{1,3}o|o[·.]{1,3}h", re.IGNORECASE)
VAGUE_RE = re.compile(
    r"\bgauche\b|\banti\b|\beclips\w*|\bstagger\w*|\bsteric\w*|"
    r"\bstrain\w*|\brepulsion\w*|\bclash\w*|\bdispersion\w*|"
    r"hyperconjugat\w*|anomeric\w*|\btorsion\w*|\bpucker\w*|"
    r"\bnonbond\w*|\bcrowd\w*",
    re.IGNORECASE,
)
_HARTREE_KCAL = 627.5095
_KJ_KCAL = 0.239006


def _energy_to_kcal(unit: str) -> float:
    u = unit.lower().rstrip(".")
    if u.startswith(("kcal", "kilocalor")):
        return 1.0
    if u.startswith(("kj", "kilojoule")):
        return _KJ_KCAL
    return _HARTREE_KCAL


def energy_values(text: str) -> list[float]:
    """All numeric energy values in the text, converted to kcal/mol (ranges expanded)."""
    out: list[float] = []
    for lo, hi, unit in ENERGY_VAL_RE.findall(text):
        factor = _energy_to_kcal(unit)
        out.append(float(lo) * factor)
        if hi:
            out.append(float(hi) * factor)
    return out


def _reasoning_text(row: dict) -> str:
    rea = row.get("reasoning")
    if isinstance(rea, dict):
        return " ".join(str(v) for v in rea.values() if v)
    return str(rea) if rea else ""


def build_table(
    models: dict,
    mol_to_label: dict,
    mol_labels: set[int],
    *,
    texts_dir: Path = TEXTS_DIR,
    out_tsv: Path = TABLE_TSV,
) -> pd.DataFrame:
    """Count H-bond / quantified / vague terms per explanation and write the TSV."""
    rows = []
    for mid, (name, _color) in models.items():
        for seed in SEEDS:
            for line in (texts_dir / f"seed{seed}" / f"{mid}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["tau"] is None or mol_to_label.get(r["molecule"]) not in mol_labels:
                    continue
                txt = _reasoning_text(r)
                evals = energy_values(txt)
                n_kcal = len(KCAL_RE.findall(txt))
                n_kj = len(KJ_RE.findall(txt))
                n_hartree = len(HARTREE_RE.findall(txt))
                n_energy = n_kcal + n_kj + n_hartree
                n_gt1 = int(sum(v > 1.0 for v in evals))
                n_qty = len(QTY_RE.findall(txt)) + n_energy
                n_hb = len(HBOND_RE.findall(txt))
                n_vague = len(VAGUE_RE.findall(txt))
                concrete = n_hb + n_qty
                rows.append({
                    "model_id": mid, "model": name, "seed": seed,
                    "label": mol_to_label[r["molecule"]], "tau": r["tau"],
                    "n_energy": n_energy, "n_kcal": n_kcal, "n_kj": n_kj,
                    "n_hartree": n_hartree, "n_gt1": n_gt1, "n_qty": n_qty,
                    "n_hbond": n_hb, "n_vague": n_vague, "words": len(txt.split()),
                    "concrete": concrete, "ratio": concrete / (n_vague + 1),
                })
    df = pd.DataFrame(rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"{len(df)} explanations from {df.model_id.nunique()} models -> {out_tsv}")
    return df


