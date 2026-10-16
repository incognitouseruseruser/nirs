#!/usr/bin/env python3

"""Gauche vs H-bond less-stable accuracy scatter (discrimination control).

For every model, pair accuracy on two trap sets with |ΔE| >= 2.0 kcal/mol
(r2SCAN-3c): H-bond less stable pairs (H-bonded truly higher, 89 pairs) and
gauche less stable pairs (O-C-C-O torsion, gauche truly higher, 36 pairs).
Points above the diagonal do better on gauche traps than H-bond traps.

  python viz_helper/plot_gdec_vs_hdec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoy_common import (
    BUNDLES,
    bench_molecules,
    gauche_pair_tables,
    hb_map,
    hbond_pair_tables,
    load_r2scan,
    override_r2scan3c,
    pooled,
    score_pairs_letters,
)

from config.models_config import (
    MARKER_EDGEWIDTH,
    MARKER_SIZE,
    family_for_display,
    family_legend_color,
    marker_for_display,
    model_files_by_display,
)
from viz.plot_style import FIGSIZE_SQUARE, SAVE_DPI, style_ax

REPO = Path(__file__).resolve().parent.parent
GEOM = REPO / "geom-qm9"
OUT = Path("figures/gdec_vs_hdec.png")


def collect_and_write(out: Path | None = None) -> Path:
    from viz.plot_style import plt

    out = Path(out) if out is not None else OUT
    r2scan = load_r2scan()
    bench = bench_molecules()
    hb = hb_map(r2scan, bench)
    tables = hbond_pair_tables(r2scan, hb)
    gtables = gauche_pair_tables(r2scan, bench)
    pair_sets = {"hdec": tables["dec"], "gdec": gtables["gdec"]}
    override = override_r2scan3c()
    bdirs = [GEOM / b for b in BUNDLES]

    res = {}
    for display, fn in model_files_by_display("geom_benchmark").items():
        pred = fn.replace("answers_", "predictions_")
        if not (bdirs[0] / pred).is_file():
            continue
        stats = score_pairs_letters(bdirs, pred, pair_sets, override)
        res[display] = {k: (pooled(stats[k])[0] / pooled(stats[k])[1]
                            if pooled(stats[k])[1] else float("nan"))
                        for k in pair_sets}
    nh = sum(len(v) for v in pair_sets["hdec"].values())
    ng = sum(len(v) for v in pair_sets["gdec"].values())
    print(f"decoy pairs: H-bond={nh}, gauche={ng}", flush=True)

    names = sorted(res)
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    for display in names:
        fam = family_for_display(display)
        ax.scatter([res[display]["hdec"]], [res[display]["gdec"]], s=MARKER_SIZE,
                   c=family_legend_color(fam, names),
                   marker=marker_for_display(display), edgecolors="white",
                   linewidths=MARKER_EDGEWIDTH, zorder=3)
    lim = (-0.05, 1.05)
    ax.plot(lim, lim, color="grey", ls="--", lw=1.0, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Accuracy on H-bond less stable pairs ({nh} pairs)", fontsize=10)
    ax.set_ylabel(f"Accuracy on gauche less stable pairs ({ng} pairs)", fontsize=10)
    style_ax(ax, square=False)
    ax.tick_params(labelsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(collect_and_write())
