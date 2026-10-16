#!/usr/bin/env python3

"""Bias toward H-bond (favorable minus less-stable accuracy) vs ranking skill.

Panel B of the decoy analysis: one dot per model — macro-averaged index over
the molecules containing both pair kinds, with 95% Student-t intervals across
molecules (molecule = unit of analysis). Force fields shown as squares.

    python viz_helper/plot_decoy_panelb.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoy_common import (
    BUNDLES,
    FLOOR,
    bench_molecules,
    contrast_triples,
    hb_map,
    load_meta,
    load_predictions,
    load_r2scan,
    macro_index,
    override_r2scan3c,
)
from plot_metrics_vs_release import (
    get_gfnff_ranking,
    get_mmff94_ranking,
    get_uff_ranking,
)

from config.models_config import (
    family_for_display,
    family_legend_color,
    model_files_by_display,
)
from metrics.option2_metrics import score_molecule
from viz.plot_style import FIGSIZE_SQUARE, REF_GFNFF, REF_MMFF, REF_UFF, style_ax

REPO = Path(__file__).resolve().parent.parent
GEOM = REPO / "geom-qm9"
OUT = Path("figures/decoy_panelB.png")
FFCOL = {"UFF": REF_UFF, "GFN-FF": REF_GFNFF, "MMFF94": REF_MMFF}


def model_row(bdirs, pred_file, pairs, override, floor=FLOOR):
    """(mean_tau, pm) for one LLM: full-ranking tau + per-molecule fav/dec."""
    tm = {}
    pm = defaultdict(lambda: [0, 0, 0, 0])
    for bdir in bdirs:
        meta = load_meta(bdir, override)
        ans = load_predictions(bdir, pred_file)
        for mol in meta:
            if mol not in ans:
                continue
            rk = ans[mol]
            l2c = {lab: int(c) for lab, c in meta[mol]["public_label_to_conf_index"].items()}
            if set(rk) != set(l2c):
                continue
            s = score_molecule(rk, meta[mol]["letter_to_deltaE_kcal_mol"], [floor])
            if s.get("valid") and np.isfinite(s[f"tau_f{floor:.1f}"]):
                tm.setdefault(mol, []).append(float(s[f"tau_f{floor:.1f}"]))
        for mol, ps in pairs.items():
            if mol not in ans or mol not in meta:
                continue
            rk = ans[mol]
            l2c = {lab: int(c) for lab, c in meta[mol]["public_label_to_conf_index"].items()}
            if set(rk) != set(l2c):
                continue
            pos = {lab: i for i, lab in enumerate(rk)}
            c2l = {c: lab for lab, c in l2c.items()}
            for h, o, h_lo in ps:
                if h not in c2l or o not in c2l:
                    continue
                first = pos[c2l[h]] < pos[c2l[o]]
                d = pm[mol]
                if h_lo:
                    d[0] += first
                    d[1] += 1
                else:
                    d[2] += (not first)
                    d[3] += 1
    vals = [np.mean(v) for v in tm.values() if len(v) == len(bdirs)]
    mt = float(np.mean(vals)) if vals else float("nan")
    return mt, dict(pm)


def ff_row(getter, pairs, r2scan, bench, floor=FLOOR):
    tm = {}
    pm = defaultdict(lambda: [0, 0, 0, 0])
    for mol in bench:
        rk = getter(REPO / "geom-qm9" / "output_30_confs" / mol)
        if not rk:
            continue
        rk = [int(c) for c in rk]
        allc = sorted(int(k) for k in r2scan[mol])
        if sorted(rk) != allc:
            continue
        s = score_molecule([str(c) for c in rk],
                           {str(c): float(r2scan[mol][str(c)]) for c in allc}, [floor])
        if s.get("valid") and np.isfinite(s[f"tau_f{floor:.1f}"]):
            tm[mol] = float(s[f"tau_f{floor:.1f}"])
    for mol, ps in pairs.items():
        rk = getter(REPO / "geom-qm9" / "output_30_confs" / mol)
        if not rk:
            continue
        rk = [int(c) for c in rk]
        pos = {c: i for i, c in enumerate(rk)}
        for h, o, h_lo in ps:
            if h not in pos or o not in pos:
                continue
            first = pos[h] < pos[o]
            d = pm[mol]
            if h_lo:
                d[0] += first
                d[1] += 1
            else:
                d[2] += (not first)
                d[3] += 1
    mt = float(np.mean(list(tm.values()))) if tm else float("nan")
    return mt, dict(pm)


def collect_rows():
    r2scan = load_r2scan()
    bench = bench_molecules()
    hb = hb_map(r2scan, bench)
    pairs = contrast_triples(r2scan, hb)
    override = override_r2scan3c()
    bdirs = [GEOM / b for b in BUNDLES]
    rows = []
    for display, fn in model_files_by_display("geom_benchmark").items():
        pred = fn.replace("answers_", "predictions_")
        if not (bdirs[0] / pred).is_file():
            print("skip (no file):", display, flush=True)
            continue
        mt, pm = model_row(bdirs, pred, pairs, override)
        if not any(v[1] and v[3] for v in pm.values()):
            print(f"  {display:<24} NO contrast pairs scored", flush=True)
            continue
        rows.append({"model": display, "kind": "llm", "mean_tau": mt, "pm": pm})
        print(f"  {display:<24} tau={mt:+.3f}", flush=True)
    for name, getter in [("UFF", get_uff_ranking), ("GFN-FF", get_gfnff_ranking),
                         ("MMFF94", get_mmff94_ranking)]:
        mt, pm = ff_row(getter, pairs, r2scan, bench)
        rows.append({"model": name, "kind": "ff", "mean_tau": mt, "pm": pm})
        print(f"  {name:<24} tau={mt:+.3f}", flush=True)
    return rows


def collect_and_write(out: Path | None = None) -> Path:
    from viz.plot_style import plt

    out = Path(out) if out is not None else OUT
    rows = collect_rows()
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    for r in rows:
        if r["kind"] != "llm" or not np.isfinite(r["mean_tau"]):
            continue
        pt, half, _n = macro_index(r["pm"])
        fam = family_for_display(r["model"])
        ax.errorbar(r["mean_tau"], pt, yerr=half,
                    fmt="o", color=family_legend_color(fam, [x["model"] for x in rows if x["kind"] == "llm"]),
                    ecolor="#BDBDBD", ms=4, elinewidth=0.8, capsize=0, zorder=3)
    for r in rows:
        if r["kind"] != "ff" or not np.isfinite(r["mean_tau"]):
            continue
        pt, _, _ = macro_index(r["pm"])
        ax.scatter([r["mean_tau"]], [pt], s=36, c=FFCOL[r["model"]], marker="s", zorder=4)
    ax.axhline(0.0, color="grey", ls="--", lw=0.8, zorder=1)
    ax.set_xlim(-0.2, 1.05)
    ax.set_ylim(-1.05, 1.05)
    style_ax(ax, square=False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Conformers Ranking Accuracy (|ΔE| ≥ 2.0 kcal/mol)", fontsize=10)
    ax.set_ylabel("Bias toward H-bond", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(collect_and_write())
