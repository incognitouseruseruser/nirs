#!/usr/bin/env python3

"""Prompt-vocabulary ablation: original vs neutral bundles (Muse Glimmer 30B).

Favorable / decoy pair accuracies under the original prompt (geom-qm9) and
the neutral prompt (geom-qm9-neutral), same pair definition as the main
decoy analysis (|ΔE| >= 2.0, exactly one H-bonded conformer).

    python viz_helper/neutral_ablation.py
    python viz_helper/neutral_ablation.py --tag muse-glimmer30b
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from decoy_common import FLOOR, bench_molecules, hb_map, load_r2scan

REPO = Path(__file__).resolve().parent.parent
BUNDLES = ["prompt_30x30_NEW", "prompt_30x30_NEW_43", "prompt_30x30_NEW_44"]
ROOTS = {"old": REPO / "geom-qm9", "neutral": REPO / "geom-qm9-neutral"}


def score_condition(root: Path, tag: str, r2scan, hb):
    """(fav_acc_n, dec_acc_n, per-mol decoy stats) pooled over seeds."""
    fav = dec = fok = dok = 0
    mol_dec = {}
    for b in BUNDLES:
        bdir = root / b
        meta = {json.loads(l)["molecule"]: json.loads(l)
                for l in (bdir / "prompt_metadata.jsonl").read_text().splitlines()}
        pred = {}
        for l in (bdir / f"predictions_{tag}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                pred[str(r["molecule"])] = r.get("prediction")
        for mol, row in meta.items():
            if mol not in r2scan or mol not in pred or not pred[mol]:
                continue
            l2c = {lab: int(c) for lab, c in row["public_label_to_conf_index"].items()}
            rk = [str(x) for x in pred[mol]]
            if set(rk) != set(l2c):
                continue
            pos = {lab: i for i, lab in enumerate(rk)}
            de = {lab: float(r2scan[mol][str(l2c[lab])]) for lab in l2c}
            hbl = {lab: hb[mol].get(str(l2c[lab]), False) for lab in l2c}
            for a, b in itertools.combinations(l2c, 2):
                if abs(de[a] - de[b]) < FLOOR or hbl[a] == hbl[b]:
                    continue
                h = a if hbl[a] else b
                o = b if hbl[a] else a
                first = pos[h] < pos[o]
                if de[h] < de[o]:
                    fav += 1
                    fok += first
                else:
                    dec += 1
                    dok += (not first)
                    st = mol_dec.setdefault(mol, [0, 0])
                    st[1] += 1
                    st[0] += (not first)
    return fok, fav, dok, dec, mol_dec


def collect_and_write(tag: str = "muse-glimmer30b") -> dict:
    r2scan = load_r2scan()
    bench = bench_molecules()
    hb = hb_map(r2scan, bench)
    res = {}
    for name, root in ROOTS.items():
        fok, fav, dok, dec, mol_dec = score_condition(root, tag, r2scan, hb)
        if not fav or not dec:
            print(f"{name}: NO pairs scored")
            continue
        res[name] = {"fav": fok / fav, "dec": dok / dec, "n_fav": fav, "n_dec": dec,
                     "mol_dec": mol_dec}
        print(f"{name.upper()}: fav={fok}/{fav}={fok / fav:.3f}  "
              f"decoy={dok}/{dec}={dok / dec:.3f}  mols={len(mol_dec)}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="muse-glimmer30b",
                    help="predictions_<tag>.jsonl stem present in both bundle sets")
    args = ap.parse_args()
    collect_and_write(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
