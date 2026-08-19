#!/usr/bin/env python
"""Capsule entry point for calibrated pairwise unmixing.

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

# The package lives NEXT TO this file, inside code/. That is deliberate: Code Ocean
# mounts only the capsule's code folder, at /code, so this script runs as
# /code/run_capsule.py and a sibling src/ directory does not exist there. An earlier
# layout kept the package in a top-level src/ and reached it with parent.parent/"src",
# which resolved to /src and failed with ModuleNotFoundError at run time while working
# fine in a git checkout.
#
# Keeping the source in the capsule (rather than pip-installing it into the image)
# means a code edit takes effect on the next run with no environment rebuild. When the
# package IS installed (local dev, `pip install -e .`), the installed copy wins and
# this is a no-op.
_PKG_PARENT = Path(__file__).resolve().parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

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
        # Name what IS attached and what is needed. The bare message this replaced
        # ("no pairwise-unmixing asset") does not say that the fix is to attach a data
        # asset rather than to change code or rebuild the environment.
        present = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        raise SystemExit("\n".join([
            "",
            f"No pairwise-unmixing asset for {mouse_id} found under {data_dir}.",
            "",
            "This capsule reads spot tables from",
            f"  HCR_{mouse_id}_pairwise-unmixing_<date>/{mouse_id}_<R>/mixed_spots_<R>.pkl",
            "and the round-to-gene mapping from ds_config.json in the same folder.",
            "",
            f"Attached assets ({len(present)}):",
            *[f"  {n}" for n in present],
            "",
            "Attach the pairwise-unmixing asset for this mouse and re-run. The",
            "_processed_ assets alone are not sufficient: they carry acquisition.json",
            "and image_spot_detection, but the newest generations do not include the",
            "spot tables, and nothing in them maps an imaging date to a round number.",
            "",
        ]))
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
    # Real ds_config.json files use GENE_DICT (uppercase), keyed by ROUND NUMBER as a
    # string, with {channel: gene} inside:
    #     {"GENE_DICT": {"5": {"488": "Npy", "514": "Pvalb", ...}}, "ROUND_N": 5}
    # An earlier version read a lowercase "gene_dict" off a "manifest" key. Neither
    # exists in these files -- verified against R1 and R5 of 800995 -- so every round
    # failed with "no gene_dict". The lowercase/manifest forms are still accepted in
    # case older assets use them.
    gd = cfg.get("GENE_DICT") or cfg.get("gene_dict") or {}
    if not gd:
        manifest = cfg.get("manifest") or {}
        gd = manifest.get("gene_dict") or manifest.get("GENE_DICT") or {}

    # GENE_DICT is nested one level under the round number. Prefer the entry matching
    # this round (ROUND_N, else the digits of round_key); fall back to the sole entry.
    if gd and all(isinstance(v, dict) for v in gd.values()):
        want = str(cfg.get("ROUND_N", "")) or "".join(ch for ch in round_key if ch.isdigit())
        if want in gd:
            gd = gd[want]
        elif len(gd) == 1:
            gd = next(iter(gd.values()))

    out = {}
    for chan, entry in gd.items():
        gene = entry.get("gene") if isinstance(entry, dict) else entry
        if gene:
            out[str(chan)] = str(gene)
    if not out:
        raise SystemExit(
            f"no gene map in {cfg_path}\n"
            f"  looked for GENE_DICT / gene_dict, then manifest.gene_dict\n"
            f"  top-level keys present: {sorted(cfg)}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-id", required=True)
    ap.add_argument("--rounds", nargs="*", default=None,
                    help="subset of rounds; default = every round present. Include R1 "
                         "(Slc17a7) and R4 (Gad2) or cells cannot be classified.")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--no-fgbg", action="store_true",
                    help="skip the fg/bg join (faster; output lacks fg/bg columns)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the four standard cell x gene heatmaps in results/plots/")
    ap.add_argument("--no-anndata", action="store_true",
                    help="skip the annotated .h5ad (class/subclass/cluster labels)")
    ap.add_argument("--experimenter", default=None,
                    help="name recorded as processor_full_name in processing.json")
    ap.add_argument("--no-metadata", action="store_true",
                    help="skip writing processing.json and copying upstream schema files")
    ap.add_argument("--processed-folder", default=None,
                    help="explicit processed-asset directory name to read acquisition.json "
                         "and image_spot_detection from. Overrides ds_config.json's "
                         "dataset_folder and the newest-asset fallback. Use this when "
                         "several processed assets exist and you know which one you want.")
    ap.add_argument("--processed-root", default=None,
                    help="parent dir of processed assets; default = --data-dir")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    asset = find_asset(args.mouse_id, data_dir)
    rounds = args.rounds or discover_rounds(asset, args.mouse_id)

    # R1 carries Slc17a7, the only excitatory marker in the panel, and R4 carries
    # Gad2. Without both, build_anndata cannot assign a class and every cell comes
    # back "unassigned" with no clusters -- a silent loss if the user simply forgot
    # a round. Warn loudly rather than produce an unlabelled AnnData.
    if not args.no_anndata:
        available = set(discover_rounds(asset, args.mouse_id))
        missing = [r for r in ("R1", "R4") if r not in rounds and r in available]
        if missing:
            print(f"WARNING: {' and '.join(missing)} available but not selected. "
                  f"R1 has Slc17a7 (excitatory) and R4 has Gad2 (inhibitory); "
                  f"without both, cells cannot be classified and the AnnData will "
                  f"carry no cluster labels.")
        absent = [r for r in ("R1", "R4") if r not in available]
        if absent:
            print(f"NOTE: {' and '.join(absent)} not present in this asset; "
                  f"class labels need R1 (Slc17a7) and R4 (Gad2).")
    if not rounds:
        raise SystemExit(f"no rounds with mixed_spots_*.pkl under {asset}")
    gene_maps = {r: gene_map_for_round(asset, args.mouse_id, r) for r in rounds}

    print(f"mouse   : {args.mouse_id}")
    print(f"asset   : {asset.name}")
    print(f"rounds  : {', '.join(rounds)}")
    for r in rounds:
        # Only channels this round actually imaged. R1 uses two of the five, and
        # printing "514=None" for the rest reads like a failure to read the config.
        used = ", ".join(f"{c}={g}" for c, g in sorted(gene_maps[r].items()))
        print(f"  {r}: {used}")
    # Report what will ACTUALLY happen, not what was asked for: the join is silently
    # skipped when the processed asset has no image_spot_detection folder, and a log
    # line claiming otherwise hides a missing-input problem until someone looks for
    # fg/bg columns that are not there.
    if args.no_fgbg:
        fgbg_status = "skipped (--no-fgbg)"
    else:
        found = [r for r in rounds
                 if pipeline.round_inputs_from_asset(
                     asset, args.mouse_id, r, args.processed_root or str(data_dir),
                     processed_folder=args.processed_folder)[1]]
        if not found:
            fgbg_status = ("NOT AVAILABLE - no image_spot_detection/ under the processed "
                           "asset; output will have no fg/bg columns")
        elif len(found) < len(rounds):
            fgbg_status = f"available for {len(found)}/{len(rounds)} rounds: {' '.join(found)}"
        else:
            fgbg_status = "joined from image_spot_detection"
    print(f"fg/bg   : {fgbg_status}")

    res = pipeline.run_mouse(
        asset, args.mouse_id, rounds, gene_maps,
        processed_root=args.processed_root or str(data_dir),
        processed_folder=args.processed_folder,
        output_dir=args.output_dir,
        use_fgbg=not args.no_fgbg,
        write_metadata=not args.no_metadata,
        write_anndata=not args.no_anndata,
        write_plots=not args.no_plots,
        experimenter=args.experimenter)

    print("\nper-channel spot change:")
    # Suppress channels with no detections: a round that imaged 2 of 5 channels would
    # otherwise show three gene=None rows of zeros.
    summ = res["summary"]
    shown = summ[summ.n_detected > 0] if "n_detected" in summ.columns else summ
    print(shown.to_string(index=False))
    print(f"\ncell x gene: {res['cellxgene'].shape[0]:,} cells x "
          f"{res['cellxgene'].shape[1]} gene-rounds")
    meta = res.get("metadata")
    if meta:
        print(f"\nmetadata: {Path(meta['processing']).name}"
              f" ({'extends ' + meta['upstream_processing'] if meta['upstream_processing'] else 'new'})")
        print(f"  copied forward: {', '.join(sorted(meta['copied'])) or 'nothing found'}")
        if meta.get("data_description"):
            import json as _j
            nm = _j.load(open(meta["data_description"]))["name"]
            print(f"  derived asset : {nm}")
        else:
            print("  derived asset : no parent data_description.json found - "
                  "data_description.json NOT written")
    if res.get("anndata"):
        import anndata as _ad
        _a = _ad.read_h5ad(res["anndata"])
        print(f"\nannotated: {Path(res['anndata']).name}  "
              f"{_a.n_obs:,} cells x {_a.n_vars} genes")
        print(f"  class   : {dict(_a.obs['class'].value_counts())}")
        n_cl = int((_a.obs['cluster_id'] >= 0).sum())
        print(f"  clusters: {_a.obs.loc[_a.obs.cluster_id >= 0, 'cluster'].nunique()}"
              f" over {n_cl:,} classified cells")
    if res.get("plots"):
        print(f"\nplots: {len(res['plots'])} figures in results/plots/")
        for _p in res["plots"]:
            print(f"  {_p}")
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
