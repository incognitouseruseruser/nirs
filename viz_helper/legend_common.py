#!/usr/bin/env python3
"""Standalone legend shared by the tau-vs-(gpqa|params|arcagi2|cost) square plots.

Reproduces exactly the handles those figures use (Random, Reasoning/Non-reasoning,
vendor families) and writes them to a single PNG.

    python viz_helper/legend_common.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_tau_vs_release_square import _legend_handles

from config.models_config import resolve_deltae_json
from metrics.kendall_tau_ranking import load_deltae_override
from viz.cohort import complete_geom_models
from viz.plot_style import FONTSIZE, plt

OUT = _REPO / "figures" / "legend_common.png"


def main() -> int:
    override = load_deltae_override(resolve_deltae_json("r2scan3c", None))
    names = {display for display, _ in complete_geom_models(
        deltae_override=override, warn_unregistered=False)}
    handles = _legend_handles(names, has_uff=False, has_gfnff=False)
    fig = plt.figure(figsize=(7.0, 1.1))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        fontsize=FONTSIZE,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.savefig(OUT, dpi=500, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}  ({len(handles)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
