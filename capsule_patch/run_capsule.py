#!/usr/bin/env python
"""Capsule entry point for v5 unmixing.

Drop-in alternative to the upstream `run_capsule.py`. It does NOT import
aind_spot_spectral_unmixing and does not touch the upstream engine or its capsule; it
reads the same inputs and writes to /root/capsule/results.

Usage inside a capsule:
    python run_capsule.py --mouse-id 790322
    python run_capsule.py --mouse-id 790322 --rounds R2 R3 --no-fgbg

Inputs expected under /root/capsule/data:
    <pairwise-unmixing asset>/<mouse>_<R>/mixed_spots_<R>.pkl
    <pairwise-unmixing asset>/<mouse>_<R>/ds_config.json
    <processed asset>/acquisition.json                    (laser power - REQUIRED)
    <processed asset>/image_spot_detection/...            (fg/bg - optional)

The processed asset for a round is resolved from that round's ds_config.json
`dataset_folder`. There are often two processed assets per round and only one matches
the spot set; picking by timestamp silently produces a bad join.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from aind_hcr_pairwise_unmixing_calibrated import pipeline
from aind_hcr_pairwise_unmixing_calibrated.control import CHANS

DATA_DIR = Path("/root/capsule/data")
OUTPUT_DIR = Path("/root/capsule/results")


def find_asset(mouse_id, data_dir=DATA_DIR):
    """The pairwise-unmixing asset directory for this mouse."""
    hits = [p for p in data_dir.iterdir()
            if p.is_dir() and "pairwise-unmixing" in p.name and mouse_id in p.name]
    if not hits:
        hits = [p for p in data_dir.iterdir()
                if p.is_dir() and any((p / f"{mouse_id}_{r}").exists()
                                      for r in ("R1", "R2", "R3"))]
    if not hits:
        raise SystemExit(f"no pairwise-unmixing asset for {mouse_id} under {data_dir}")
    if len(hits) > 1:
        print(f"WARNING: {len(hits)} candidate assets, using {hits[0].name}")
    return hits[0]


def discover_rounds(asset_dir, mouse_id):
    rounds = sorted(p.name.split("_")[-1] for p in asset_dir.iterdir()
                    if p.is_dir() and p.name.startswith(f"{mouse_id}_R")
                    and (p / f"mixed_spots_{p.name.split('_')[-1]}.pkl").exists())
    return sorted(rounds, key=lambda r: int(r[1:]))


def gene_map_for_round(asset_dir, mouse_id, round_key):
    """{channel: gene} from the round's ds_config manifest."""
    cfg_path = asset_dir / f"{mouse_id}_{round_key}" / "ds_config.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    manifest = cfg.get("manifest") or cfg
    gd = manifest.get("gene_dict") or {}
    out = {}
    for chan, entry in gd.items():
        gene = entry.get("gene") if isinstance(entry, dict) else entry
        out[str(chan)] = gene
    if not out:
        raise SystemExit(f"no gene_dict in {cfg_path}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-id", required=True)
    ap.add_argument("--rounds", nargs="*", default=None,
                    help="subset of rounds; default = every round present")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--no-fgbg", action="store_true",
                    help="skip the fg/bg join (faster; output lacks fg/bg columns)")
    ap.add_argument("--processed-root", default=None,
                    help="parent dir of processed assets; default = --data-dir")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    asset = find_asset(args.mouse_id, data_dir)
    rounds = args.rounds or discover_rounds(asset, args.mouse_id)
    if not rounds:
        raise SystemExit(f"no rounds with mixed_spots_*.pkl under {asset}")
    gene_maps = {r: gene_map_for_round(asset, args.mouse_id, r) for r in rounds}

    print(f"mouse   : {args.mouse_id}")
    print(f"asset   : {asset.name}")
    print(f"rounds  : {', '.join(rounds)}")
    for r in rounds:
        print(f"  {r}: " + ", ".join(f"{c}={gene_maps[r].get(c)}" for c in CHANS))
    print(f"fg/bg   : {'skipped' if args.no_fgbg else 'joined from image_spot_detection'}")

    res = pipeline.run_mouse(
        asset, args.mouse_id, rounds, gene_maps,
        processed_root=args.processed_root or str(data_dir),
        output_dir=args.output_dir,
        use_fgbg=not args.no_fgbg)

    print("\nper-channel spot change:")
    print(res["summary"].to_string(index=False))
    print(f"\ncell x gene: {res['cellxgene'].shape[0]:,} cells x "
          f"{res['cellxgene'].shape[1]} gene-rounds")
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
