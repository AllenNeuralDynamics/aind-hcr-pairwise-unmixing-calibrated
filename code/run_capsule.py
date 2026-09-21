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

WHY THE PAIRWISE-UNMIXING ASSET IS STILL REQUIRED
-------------------------------------------------
Not for its results. This capsule re-derives every spot decision from `mixed_spots`
and reads none of that asset's `unmixed_*` outputs -- they are the previous method's
answer to the same question. It is required as the container of two inputs:

  mixed_spots_<R>.pkl   also in the processed asset, under
                        image_spot_spectral_unmixing/, with the round in the filename
                        -- but it is a DIFFERENT table, not a copy. Same spots to
                        within 2-4%, twice the bytes per row: 221 B/row there against
                        110 B/row here for a 5-channel round, and 149 against 99 for
                        2-channel R1. Fitting those, the processed table carries about
                        three extra float64 columns PER CHANNEL that the pairwise
                        table drops. Switching inputs is a schema change, not a path
                        change.

  ds_config.json        GENE_DICT, the round -> channel -> gene map, e.g.
                        {"1": {"488": "GFP", "561": "Slc17a7"}}.

GENE_DICT is NOT original to this asset: it is `manifest.gene_dict` flattened, and
`manifest` is a byte-identical copy of the processed asset's own
processing_manifest.json -- which also carries `round`. So the gene map is fully
derivable from the processed assets, and `gene_map_from_manifests` does that when
ds_config.json is absent. ds_config.json stays the primary source because it is what
the spot tables were produced with; if the two ever disagree, the spot tables follow
it. (An earlier version of this docstring claimed the gene map existed only here.
That was wrong: it was checked against acquisition.json, which indeed carries no gene
symbol, and processing_manifest.json was never opened.)
"""
import argparse
import json
import re
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


def find_asset(mouse_id, data_dir=DATA_DIR, required=True):
    """The pairwise-unmixing asset directory for this mouse, or None when not required.

    `required=False` is for --spots-from processed, where this asset is not an input:
    its absence is then a fact to report, not a failure.
    """
    hits = [p for p in data_dir.iterdir()
            if p.is_dir() and "pairwise-unmixing" in p.name and mouse_id in p.name]
    if not hits:
        hits = [p for p in data_dir.iterdir()
                if p.is_dir() and any((p / f"{mouse_id}_{r}").exists()
                                      for r in ("R1", "R2", "R3"))]
    if not hits and not required:
        return None
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
            "Attach the pairwise-unmixing asset for this mouse and re-run.",
            "",
            "What this asset uniquely provides is the spot tables. The processed assets",
            "hold a table of the same name at",
            "  <processed asset>/image_spot_spectral_unmixing/mixed_spots_<R>.pkl,",
            "but it is a different table -- same spots to within a few percent, twice",
            "the bytes per row, about three extra float64 columns per channel. The gene",
            "map is NOT unique to this asset: GENE_DICT is processing_manifest.json's",
            "gene_dict flattened, and this script falls back to reading that directly",
            "from the processed assets when ds_config.json is missing.",
            "",
        ]))
    if len(hits) > 1:
        print(f"WARNING: {len(hits)} candidate assets, using {hits[0].name}")
    return hits[0]


#: --skip token -> the argparse attribute it sets.
SKIP_TOKENS = {"spots": "no_spots", "anndata": "no_anndata", "plots": "no_plots",
               "metadata": "no_metadata", "fgbg": "no_fgbg"}

SPOTS_FROM_CHOICES = ("auto", "pairwise", "processed")


def normalize_app_panel_args(args):
    """Repair the argv shapes Code Ocean's App Panel produces. Mutates `args`.

    A text parameter is emitted as `--name=value`, always, with the value empty when
    the field is left blank. Three consequences, all of which turn a run into a crash
    or a silently wrong result if unhandled:

      blank fields     `--spots-from=` reaches argparse as the empty string. Treated
                       here as "not set", not as an invalid choice.
      no flags         a store_true option cannot be expressed at all, which is what
                       `--skip spots,anndata` exists to work around.
      one value only   `--rounds=R2 R4` arrives as ONE element "R2 R4" rather than
                       two, because the shell never splits inside the parameter.

    None of this applies when the script is run from a terminal, where the flags
    behave normally; this function is a no-op on that path.
    """
    spots = (getattr(args, "spots_from", None) or "").strip()
    args.spots_from = spots or "auto"
    if args.spots_from not in SPOTS_FROM_CHOICES:
        raise SystemExit(
            f"--spots-from: {args.spots_from!r} is not one of "
            f"{', '.join(SPOTS_FROM_CHOICES)}")

    skip = (getattr(args, "skip", None) or "").strip()
    if skip:
        unknown = []
        for tok in (t.strip().lower() for t in re.split(r"[,\s]+", skip) if t.strip()):
            attr = SKIP_TOKENS.get(tok.removeprefix("no-").removeprefix("no_"))
            if attr is None:
                unknown.append(tok)
            else:
                setattr(args, attr, True)
        if unknown:
            raise SystemExit(
                f"--skip: unknown {', '.join(unknown)}. Known tokens: "
                f"{', '.join(sorted(SKIP_TOKENS))}")

    rounds = getattr(args, "rounds", None)
    if rounds:
        flat = []
        for r in rounds:
            flat += [t for t in re.split(r"[,\s]+", str(r).strip()) if t]
        args.rounds = flat or None
    elif rounds is not None:
        args.rounds = None


def discover_rounds_from_processed(data_dir, mouse_id):
    """Rounds available from the PROCESSED assets alone, via processing_manifest.json.

    Each processed asset declares its own round number, so the round set is readable
    without the pairwise-unmixing asset. This is what makes `--spots-from processed` a
    genuinely processed-only run rather than one that still needs the old mount to
    enumerate its own work.
    """
    out = {}
    for man in sorted(Path(data_dir).glob("*/processing_manifest.json")):
        if mouse_id not in man.parent.name:
            continue
        spots = man.parent / "image_spot_spectral_unmixing"
        try:
            with open(man) as fh:
                n = json.load(fh).get("round")
        except (OSError, ValueError):
            continue
        if n is None:
            continue
        rk = f"R{int(n)}"
        if (spots / f"mixed_spots_{rk}.pkl").exists():
            out[rk] = man.parent.name
    return sorted(out, key=lambda r: int(r[1:])), out


def discover_rounds(asset_dir, mouse_id):
    rounds = sorted(p.name.split("_")[-1] for p in asset_dir.iterdir()
                    if p.is_dir() and p.name.startswith(f"{mouse_id}_R")
                    and (p / f"mixed_spots_{p.name.split('_')[-1]}.pkl").exists())
    return sorted(rounds, key=lambda r: int(r[1:]))


def gene_map_from_manifests(round_key, processed_root):
    """{channel: gene} for a round, from the PROCESSED assets' processing_manifest.json.

    `ds_config.json`'s GENE_DICT is not original: it is `manifest.gene_dict` flattened
    to {round: {channel: gene}}, and `manifest` is a byte-identical copy of the
    processed asset's own `processing_manifest.json` (checked for 800792 R1 and R5).
    That file carries `round` as well, so a round's gene map can be read straight from
    the processed assets with no pairwise-unmixing asset involved.

    Used as a fallback when ds_config.json is absent. Kept as a fallback rather than
    the primary source because ds_config.json is what the spot tables were produced
    with, and if the two ever disagree the spot tables follow ds_config.
    """
    root = Path(processed_root)
    want = int("".join(ch for ch in round_key if ch.isdigit()))
    for man in sorted(root.glob("*/processing_manifest.json")):
        try:
            with open(man) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if int(m.get("round", -1)) != want:
            continue
        gd = m.get("gene_dict") or {}
        out = {str(c): str(v["gene"]) for c, v in gd.items()
               if isinstance(v, dict) and v.get("gene")}
        if out:
            print(f"  {round_key}: gene map from {man.parent.name}/processing_manifest.json")
            return out
    return {}


def gene_map_for_round(asset_dir, mouse_id, round_key, processed_root=None):
    """{channel: gene} for a round: ds_config.json, else the processed asset's manifest."""
    cfg_path = asset_dir / f"{mouse_id}_{round_key}" / "ds_config.json"
    if not cfg_path.exists():
        out = gene_map_from_manifests(round_key, processed_root or asset_dir.parent)
        if out:
            return out
        raise SystemExit(
            f"no gene map for {round_key}: {cfg_path} is absent and no "
            f"processing_manifest.json under {processed_root or asset_dir.parent} "
            f"declares round {round_key}.")
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


def _code_provenance():
    """Package version and git commit of the code actually executing.

    A capsule checked out on a stale branch runs old code and says nothing about it:
    the only symptom is label counts that silently match a previous version. Printing
    the version and commit in the run header makes that visible in the log, which is
    the one artefact you still have after the fact.
    """
    from aind_hcr_pairwise_unmixing_calibrated import __version__
    import subprocess
    parts = [f"v{__version__}"]
    try:
        out = subprocess.run(["git", "-C", str(_PKG_PARENT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        branch = subprocess.run(["git", "-C", str(_PKG_PARENT), "rev-parse",
                                 "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            parts.append(out.stdout.strip())
        if branch.returncode == 0 and branch.stdout.strip():
            parts.append(f"branch {branch.stdout.strip()}")
    except (OSError, subprocess.SubprocessError):
        pass          # not a checkout, or no git in the image -- the version still prints
    return " · ".join(parts)


def find_cellxgene(path, mouse_id):
    """Resolve --relabel-from to a cell x gene CSV.

    Accepts the file itself, or a directory to search: a previous run's results, or
    the registered asset for one mounted under /data. Searched recursively because a
    mounted asset puts the CSV one level down, in a folder named for the asset.
    """
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"--relabel-from: {path} does not exist")
    wanted = f"{mouse_id}_cellxgene.csv"
    hits = sorted(p for p in path.rglob(wanted) if p.is_file())
    if not hits:
        others = sorted({p.name for p in path.rglob("*_cellxgene.csv")})
        msg = f"--relabel-from: no {wanted} under {path}."
        if others:
            msg += f" Found for other mice: {', '.join(others)}"
        else:
            msg += " Nothing matching *_cellxgene.csv is there either."
            # The usual cause. A Reproducible Run's outputs go to that run's result
            # set, not into the workstation's /results, and every new run starts with
            # /results empty -- so a relabel run wipes what it was meant to read.
            if path.name == "results":
                msg += (
                    "\n\n/results is empty at the start of every run, and a "
                    "Reproducible Run's outputs are captured to that run's result "
                    "set rather than left here. Point at the mouse's registered "
                    "unmixed-calibrated asset instead:"
                    "\n  1. attach HCR_<mouse>_unmixed-calibrated_<date> to this capsule"
                    "\n  2. python run_capsule.py --mouse-id <mouse> --relabel-from /data"
                    "\nTo relabel a run that was never registered, register it first:"
                    "\n  python tools/register_result_asset.py --latest")
        raise SystemExit(msg)
    if len(hits) > 1:
        print(f"NOTE: {len(hits)} copies of {wanted} under {path}; using the newest.")
        hits.sort(key=lambda p: p.stat().st_mtime)
    return hits[-1]


def relabel(args):
    """Rebuild class, subclass and cluster labels from an existing cell x gene table.

    The labelling is a minute of work on a table the unmixing took an hour to produce,
    so iterating on label rules should not re-run the unmixing. This reads the counts
    back and writes a new .h5ad and figures beside them.

    Deliberately does NOT write metadata or an asset manifest: the counts came from a
    run whose provenance is already recorded, and a second processing.json describing
    the same spot decisions would misrepresent what happened. Keep the original asset
    and note the relabelling commit.
    """
    import pandas as pd
    from aind_hcr_pairwise_unmixing_calibrated import annotate
    from aind_hcr_pairwise_unmixing_calibrated import plots as _plots

    csv = find_cellxgene(args.relabel_from, args.mouse_id)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"code    : {_code_provenance()}")
    print(f"mouse   : {args.mouse_id}")
    print(f"relabel : {csv}")
    print("unmixing: SKIPPED (--relabel-from); counts are read, not re-derived")

    table = pd.read_csv(csv, index_col=0)
    rounds = sorted({str(c).split("-")[0] for c in table.columns})
    print(f"rounds  : {', '.join(rounds)}")
    print(f"cell x gene: {table.shape[0]:,} cells x {table.shape[1]} gene-rounds")

    adata = annotate.build_anndata(
        table, extra_uns=dict(mouse_id=args.mouse_id, rounds=rounds,
                              relabelled_from=str(csv)))
    h5 = out / f"{args.mouse_id}_cellxgene_annotated.h5ad"
    adata.write_h5ad(h5)
    def counts(col):
        return {k: int(v) for k, v in adata.obs[col].value_counts().items() if v}

    print(f"\nannotated: {h5.name}  {adata.n_obs:,} cells x {adata.n_vars} genes")
    print(f"  class   : {counts('class')}")
    print(f"  subclass: {counts('subclass')}")
    n_cl = int((adata.obs["cluster_id"] >= 0).sum())
    print(f"  clusters: {adata.obs.loc[adata.obs.cluster_id >= 0, 'cluster'].nunique()}"
          f" over {n_cl:,} classified cells")
    inh = adata.uns["unmixing"]["clustering"].get("inhibitory", {})
    if inh.get("block"):
        from collections import Counter
        print(f"  blocks  : {dict(Counter(inh['block'].values()))}")

    if not args.no_plots:
        written = _plots.write_plots(adata, out, args.mouse_id, rounds)
        print(f"\nplots: {len(written)} figures in results/plots/")
    print("\nNo metadata or asset manifest written -- the counts belong to the original "
          "run. Keep that asset and record this commit as the labelling version.")
    return 0


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
    ap.add_argument("--no-spots", action="store_true",
                    help="skip the per-round <M>_<R>_unmixed_spots.parquet tables. They "
                         "are written by DEFAULT and are the primary output -- the only "
                         "record of the per-spot decisions, and the cell x gene table "
                         "cannot be rebuilt without them. Use this only when the cell x "
                         "gene table is all you want: they are ~2 GB per 5-round mouse "
                         "and dominate both the asset size and the post-run upload.")
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
    ap.add_argument("--spots-from", default="auto",
                    # No argparse `choices`: Code Ocean's App Panel emits a text
                    # parameter as --spots-from=<value> even when the field is left
                    # blank, and `choices` would reject the empty string before the
                    # run starts. Validated in normalize_app_panel_args instead, where
                    # blank can mean "unset".
                    help="which mixed_spots_<R>.pkl to read. 'pairwise' is the "
                         "HCR_<mouse>_pairwise-unmixing asset, the spot set every "
                         "result so far was produced from. 'processed' is the copy in "
                         "each processed asset, which additionally carries "
                         "chan_<ch>_fg / _bg so the fg/bg join is skipped -- but it "
                         "holds 2-4%% more spots, so it does NOT reproduce the same "
                         "cell x gene table. 'auto' prefers pairwise when attached.")
    ap.add_argument("--skip", default=None, metavar="LIST",
                    help="comma-separated outputs to skip: spots, anndata, plots, "
                         "metadata, fgbg. Equivalent to the --no-* flags, in one text "
                         "field so a Reproducible Run can set them from the App Panel "
                         "(a store_true flag cannot be expressed as an App Panel "
                         "parameter, which emits --name=value).")
    ap.add_argument("--relabel-from", default=None, metavar="PATH",
                    help="skip the unmixing entirely and rebuild ONLY the labels from "
                         "an existing <mouse>_cellxgene.csv. PATH is that file or the "
                         "directory holding it (a previous run's results, or its "
                         "registered asset under /data). Writes the annotated .h5ad and "
                         "the figures; takes about a minute instead of an hour. Nothing "
                         "here re-derives spot decisions, so the cell x gene counts are "
                         "exactly the ones the unmixing produced.")
    args = ap.parse_args(argv)
    normalize_app_panel_args(args)

    if args.relabel_from:
        return relabel(args)

    data_dir = Path(args.data_dir)
    # With --spots-from processed the pairwise asset is not an input at all, so its
    # absence must not be an error -- otherwise the flag still requires the mount it
    # exists to remove. Rounds then come from the processed assets' own manifests.
    if args.spots_from == "processed":
        asset = find_asset(args.mouse_id, data_dir, required=False)
        proc_rounds, proc_where = discover_rounds_from_processed(data_dir, args.mouse_id)
        if not proc_rounds:
            raise SystemExit(
                f"--spots-from processed: no processed asset under {data_dir} carries "
                f"image_spot_spectral_unmixing/mixed_spots_<R>.pkl for {args.mouse_id}.")
        rounds = args.rounds or proc_rounds
        print(f"spots   : processed assets ({len(proc_rounds)} rounds: "
              f"{', '.join(proc_rounds)})")
        if asset is None:
            print("note    : no pairwise-unmixing asset attached; not needed for this run")
    else:
        asset = find_asset(args.mouse_id, data_dir)
        rounds = args.rounds or discover_rounds(asset, args.mouse_id)

    # R1 carries Slc17a7, the only excitatory marker in the panel, and R4 carries
    # Gad2. Without both, build_anndata cannot assign a class and every cell comes
    # back "unassigned" with no clusters -- a silent loss if the user simply forgot
    # a round. Warn loudly rather than produce an unlabelled AnnData.
    if not args.no_anndata:
        available = set(discover_rounds_from_processed(data_dir, args.mouse_id)[0]
                        if args.spots_from == "processed"
                        else discover_rounds(asset, args.mouse_id))
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
    gene_maps = {r: gene_map_for_round(asset if asset is not None else data_dir,
                                       args.mouse_id, r,
                                       processed_root=args.processed_root or data_dir)
                 for r in rounds}
    # Correct wrong gene symbols at the boundary where the map enters the run. Both
    # known errors (Tac, Slac17a7) originate in the acquisition metadata, and the map
    # feeds more than the cell x gene table -- *_spot_change.csv takes its `gene`
    # column straight from here.
    from aind_hcr_pairwise_unmixing_calibrated.annotate import correct_gene_map
    gene_maps = {r: correct_gene_map(gene_maps[r], round_key=r)[0] for r in rounds}

    print(f"code    : {_code_provenance()}")
    print(f"mouse   : {args.mouse_id}")
    print(f"asset   : {asset.name if asset is not None else '(none attached)'}")
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
    elif args.spots_from == "processed":
        # The processed spot tables carry chan_<ch>_fg / _bg themselves, so the join
        # does not run whatever image_spot_detection holds. Reporting the join here
        # because those folders exist was exactly the kind of claim this block was
        # written to prevent -- the per-round line said "join skipped" one screen later.
        fgbg_status = ("native columns on the processed spot tables; "
                       "image_spot_detection not read")
    else:
        found = [r for r in rounds
                 if pipeline.round_inputs_from_asset(
                     asset if asset is not None else data_dir,
                     args.mouse_id, r, args.processed_root or str(data_dir),
                     processed_folder=args.processed_folder)[1]]
        if not found:
            fgbg_status = ("NOT AVAILABLE - no image_spot_detection/ under the processed "
                           "asset; output will have no fg/bg columns")
        elif len(found) < len(rounds):
            fgbg_status = f"available for {len(found)}/{len(rounds)} rounds: {' '.join(found)}"
        else:
            fgbg_status = "joined from image_spot_detection"
    print(f"fg/bg   : {fgbg_status}")
    # State this up front: --no-spots discards the per-spot decisions irrecoverably, and
    # finding that out from an absent file after a 30-minute run is too late.
    print(f"spots   : {'NOT written (--no-spots)' if args.no_spots else 'written per round'}")

    res = pipeline.run_mouse(
        asset if asset is not None else data_dir,
        args.mouse_id, rounds, gene_maps,
        processed_root=args.processed_root or str(data_dir),
        processed_folder=args.processed_folder,
        output_dir=args.output_dir,
        use_fgbg=not args.no_fgbg,
        spots_from=args.spots_from,
        write_metadata=not args.no_metadata,
        write_anndata=not args.no_anndata,
        write_plots=not args.no_plots,
        write_spots=not args.no_spots,
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
            # NOT a registered asset -- just the name written into
            # data_description.json. Nothing in this run creates a data asset; see
            # docs/REGISTER_ASSET.md and tools/register_result_asset.py.
            print(f"  data_description name : {nm}")
        else:
            print("  data_description : no parent data_description.json found - "
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

    # Asset manifest: what the registered data asset for this run should look like.
    # This run cannot register it -- Code Ocean uploads /results to S3 only after this
    # script exits -- so record the name, description and tags for
    # tools/register_result_asset.py to act on afterwards.
    if not args.no_metadata:
        from aind_hcr_pairwise_unmixing_calibrated import manifest as _manifest
        man = _manifest.write_manifest(
            args.output_dir, args.mouse_id, rounds,
            data_dir=args.data_dir,
            n_cells=res["cellxgene"].shape[0],
            n_genes=res["cellxgene"].shape[1],
            creation_time=res.get("creation_time"),
            capsule_name="aind-hcr-pairwise-unmixing-calibrated",
            experimenter=args.experimenter)
        used = man["input_assets"]
        print(f"\nasset manifest: {man['name']}")
        print(f"  inputs: {len(used['unmixing'])} unmixing, "
              f"{len(used['processed'])} processed, {len(used['raw'])} raw")
        if used["other_mouse"]:
            print(f"  WARNING: {len(used['other_mouse'])} asset(s) for other mice were "
                  f"mounted and NOT used; they are named in the manifest description "
                  f"so the asset does not imply they contributed")
        print("  register: python tools/register_result_asset.py --latest "
              "(see docs/REGISTER_ASSET.md)")

    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
