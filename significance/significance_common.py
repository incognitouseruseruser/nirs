"""Shared helpers for benchmark significance and release-date correlation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.conformer_min_in_topk import letters_at_global_minimum, min_in_topk
from metrics.kendall_tau_ranking import kendall_for_molecule
from metrics.random_baseline import min_at_k_random


@dataclass(frozen=True)
class MoleculeBaselines:
    """Per-molecule random baselines for min@10."""

    rand_m10_p: dict[str, float]


@dataclass(frozen=True)
class MoleculeMetrics:
    """Per-molecule model scores aligned to ``molecules`` order."""

    molecules: tuple[str, ...]
    taus: np.ndarray
    m10: np.ndarray


@dataclass(frozen=True)
class ModelSignificance:
    name: str
    n: int
    n_molecules: int
    partial: bool
    tau_mean: float
    m10_hits: int
    p_tau: float
    p_m10: float

    @property
    def sig_tau(self) -> bool:
        return self.p_tau < 0.05

    @property
    def sig_m10(self) -> bool:
        return self.p_m10 < 0.05

    @property
    def sig_labels(self) -> list[str]:
        out: list[str] = []
        if self.sig_tau:
            out.append("τ")
        if self.sig_m10:
            out.append("min@10")
        return out


def sorted_molecules(meta: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(meta.keys())


def build_random_baselines(
    meta: dict[str, dict[str, Any]],
) -> MoleculeBaselines:
    rand_m10_p: dict[str, float] = {}
    for mol in sorted_molecules(meta):
        de = meta[mol]["letter_to_deltaE_kcal_mol"]
        n = len(de)
        min_e = min(de.values())
        m_pos = sum(1 for v in de.values() if v == min_e)
        rand_m10_p[mol] = min_at_k_random(n, m_pos, 10)
    return MoleculeBaselines(rand_m10_p=rand_m10_p)


def per_molecule_metrics(
    answers: dict[str, list[str]],
    meta: dict[str, dict[str, Any]],
    molecules: list[str],
) -> MoleculeMetrics | None:
    taus: list[float] = []
    m10: list[int] = []
    used: list[str] = []
    for mol in molecules:
        if mol not in answers:
            continue
        rnk = answers[mol]
        de = meta[mol]["letter_to_deltaE_kcal_mol"]
        tau, _ = kendall_for_molecule(de, rnk)
        hit = int(min_in_topk(rnk, letters_at_global_minimum(de), 10))
        used.append(mol)
        taus.append(tau)
        m10.append(hit)
    if not used:
        return None
    return MoleculeMetrics(
        molecules=tuple(used),
        taus=np.asarray(taus, dtype=float),
        m10=np.asarray(m10, dtype=int),
    )


def min10_pvalue_mc(
    hits: np.ndarray,
    p_random: np.ndarray,
    *,
    trials: int = 200_000,
    seed: int = 42,
) -> float:
    """One-sided P(sum hits >= observed) under independent Bernoulli(p_i).

    Note: This test uses a sum statistic which is conservative. It may lose
    power compared to likelihood-ratio tests or permutation tests that use
    more information about the distribution of hits across molecules.
    The p-value is valid but may be larger than necessary.
    """
    n = len(hits)
    obs = int(hits.sum())
    rng = np.random.default_rng(seed)
    sims = rng.binomial(1, p_random, size=(trials, n)).sum(axis=1)
    return float((sims >= obs).mean())


def load_bundle_meta(
    bundle_dir: Path,
    *,
    baseline: str = "gfn2-xtb",
    deltae_json: Path | None = None,
) -> dict[str, dict[str, Any]]:
    from metrics.baseline import prepare_meta
    return prepare_meta(bundle_dir / "prompt_metadata.jsonl", baseline=baseline, deltae_json=deltae_json)
