#!/usr/bin/env python3

"""CRA on the 89 decoy pairs vs CRA on the 2,997 equal-H-bond pairs (wide canvas).

x: mean per-molecule CRA on pairs whose two conformers carry the same number of
H-bonds (heuristic silent). y: CRA on the decoy pairs, where the H-bonded
conformer is truly higher in energy. Force fields as squares; the shared
release-plot canvas with a bottom legend strip is used for layout consistency.

    python viz_helper/plot_decoy_vs_equalhb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoy_common import (
    BUNDLES,
    bench_molecules,
    equal_hbond_pair_tables,
    hb_count_map,
    hb_map,
    hbond_pair_tables,
    load_r2scan,
    override_r2scan3c,
    score_pairs_ff,
    score_pairs_letters,
)
from matplotlib.lines import Line2D
from plot_metrics_vs_release import (
    get_gfnff_ranking,
    get_mmff94_ranking,
    get_uff_ranking,
)
from plot_tau_vs_release_square import (
    DATE_AXES,
    DATE_LEGEND,
    FIGSIZE_DATE,
    LEGEND_FONTSIZE,
    _legend_handles,
    open_square_figure,
)

from config.models_config import (
    family_for_display,
    family_legend_color,
    marker_for_display,
    model_files_by_display,
)
from viz.plot_style import (
    REF_GFNFF,
    REF_MMFF,
    REF_UFF,
    style_ax,
)

REPO = Path(__file__).resolve().parent.parent
GEOM = REPO / "geom-qm9"
OUT = Path("figures/cra_decoy_vs_equalhb.png")


def _macro(stats: dict) -> float:
    accs = np.asarray([v[0] / v[1] for v in stats.values() if v[1]])
    return float(np.mean(2 * accs - 1)) if len(accs) else float("nan")


def collect_and_write(out: Path | None = None) -> Path:
    from viz.plot_style import plt

    out = Path(out) if out is not None else OUT
    r2scan = load_r2scan()
    bench = bench_molecules()
    hb = hb_map(r2scan, bench)
    tables = {"equal": equal_hbond_pair_tables(r2scan, hb_count_map(r2scan, bench)),
              "dec": hbond_pair_tables(r2scan, hb)["dec"]}
    n_eq = sum(len(v) for v in tables["equal"].values())
    n_dec = sum(len(v) for v in tables["dec"].values())

    override = override_r2scan3c()
    bdirs = [GEOM / b for b in BUNDLES]
    pts, ff_pts = [], []
    for display, fn in model_files_by_display("geom_benchmark").items():
        pred = fn.replace("answers_", "predictions_")
        if not (bdirs[0] / pred).is_file():
            continue
        st = score_pairs_letters(bdirs, pred, tables, override)
        pts.append((display, _macro(st["equal"]), _macro(st["dec"])))
    for name, getter in (("UFF", get_uff_ranking), ("GFN-FF", get_gfnff_ranking),
                         ("MMFF94", get_mmff94_ranking)):
        st = score_pairs_ff(getter, tables, r2scan)
        ff_pts.append((name, _macro(st["equal"]), _macro(st["dec"])))
    print(f"equal-H-bond {n_eq} pairs, decoy {n_dec} pairs; models {len(pts)}; "
          f"FF: {[(n, f'eq={e:+.3f}', f'dec={d:+.3f}') for n, e, d in ff_pts]}", flush=True)

    present = {p[0] for p in pts}
    fig, ax, legend_ax = open_square_figure(
        figsize=FIGSIZE_DATE, axes_rect=DATE_AXES, legend_rect=DATE_LEGEND)
    for display, eq, dec in pts:
        fam = family_for_display(display)
        ax.scatter([eq], [dec], s=28, c=family_legend_color(fam, list(present)),
                   marker=marker_for_display(display), zorder=3)
    for name, eq, dec in ff_pts:
        col = {"UFF": REF_UFF, "GFN-FF": REF_GFNFF, "MMFF94": REF_MMFF}[name]
        ax.scatter([eq], [dec], s=36, c=col, marker="s", zorder=4)
    ax.axhline(0.0, color="grey", ls="--", lw=0.8, zorder=1)
    ax.axvline(0.0, color="grey", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(-0.15, 1.0)
    ax.set_ylim(-1.05, 1.05)
    style_ax(ax, square=False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel(f"CRA on equal H-bond pairs ({n_eq} pairs)", fontsize=10)
    ax.set_ylabel(f"CRA on decoy pairs ({n_dec} pairs)", fontsize=10)
    handles = _legend_handles(present, has_uff=False, has_gfnff=False, has_mmff=False)
    for name, _eq, _dec in ff_pts:
        col = {"UFF": REF_UFF, "GFN-FF": REF_GFNFF, "MMFF94": REF_MMFF}[name]
        handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=col,
                              markeredgecolor="white", markeredgewidth=0.6,
                              markersize=8, linestyle="None", label=name))
    legend_ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                     ncol=4, fontsize=LEGEND_FONTSIZE, frameon=False, handlelength=1.5,
                     columnspacing=1.0, handletextpad=0.5, borderaxespad=0.0,
                     borderpad=0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500, bbox_inches=None)
    plt.close(fig)
    print(f"Saved {out}", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(collect_and_write())
