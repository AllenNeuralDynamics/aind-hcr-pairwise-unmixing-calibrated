"""Per-round driver: spot table in, unmixed spot table + cell x gene out.

This is the only module the capsule needs to call. It is deliberately a thin wrapper --
all algorithm decisions live in core.py -- so that the capsule's own code stays a
data-plumbing script.
"""
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import core
from .control import CHANS, control_matrix, load_powers, resolve_dataset_folder
from . import metadata
from .fgbg import attach_fg_bg, diagonal_stats_path


def _native_fg_bg(spots, channels):
    """(fg, bg) per row from the table's own chan_<ch>_fg / _bg columns, else None.

    Each row belongs to one channel, so the per-row value is that channel's column --
    the same selection `attach_fg_bg` performs after reconstructing the values from a
    separate stats file. Returns None when the columns are absent, which is the
    pairwise table and means the join is still needed.

    A row whose `chan` has no matching column stays NaN rather than borrowing another
    channel's value; the caller's coverage check then fails loudly instead of the
    pipeline proceeding on a silently wrong background.
    """
    from . import spots_io

    fg_cols = spots_io.channel_columns(spots, "fg")
    bg_cols = spots_io.channel_columns(spots, "bg")
    if not fg_cols or set(fg_cols) != set(bg_cols):
        return None

    chan = spots["chan"].astype(str).to_numpy()
    fg = np.full(len(spots), np.nan)
    bg = np.full(len(spots), np.nan)
    for c in fg_cols:
        m = chan == str(c)
        if m.any():
            fg[m] = spots.loc[m, fg_cols[c]].to_numpy(float)
            bg[m] = spots.loc[m, bg_cols[c]].to_numpy(float)
    covered = float(np.isfinite(fg).mean())
    if covered < 0.99:
        raise RuntimeError(
            f"native fg/bg covered only {covered:.1%} of spots: the table has columns "
            f"for channels {sorted(fg_cols)} but `chan` also holds "
            f"{sorted(set(chan) - {str(c) for c in fg_cols})[:6]}.")
    return fg, bg


def run_round(spots, powers, gene_map, round_key, B_ctrl=None,
              diag_paths=None, channels=CHANS, **unmix_kw):
    """Unmix one round.

    spots      the FULL spot table for the round (mixed_spots_{R}.pkl). Do NOT
               pre-filter on dist / r / dist_r: the unmixer needs to see every
               candidate ghost, and those flags are annotated for downstream use.
    powers     {channel: excitation power %} from THIS round's acquisition.json.
    gene_map   {channel: gene name} for this round.
    diag_paths optional {channel: path} of spot-detection stats CSVs; when given, raw
               fg/bg are joined on and written to the output.

    Returns dict with keys: spots, separability, decisions, endmembers, cellxgene.
    """
    if B_ctrl is None:
        B_ctrl = control_matrix(channels)
    spots = spots.copy()
    spots["chan"] = spots["chan"].astype(str)

    fg_bg = None
    native = _native_fg_bg(spots, channels)
    if native is not None:
        # The processed asset's table already carries chan_<ch>_fg / _bg, so the
        # exact-coordinate join is not just unnecessary here, it is strictly worse:
        # the join reconstructs by matching coordinates against a separate stats file,
        # while these columns are the values the intensity was computed FROM.
        fg_bg = native
        print(f"  fg/bg: native columns on the spot table ({len(channels)} channels); "
              f"join skipped", flush=True)
    elif diag_paths:
        fg, bg = attach_fg_bg(spots, diag_paths, channels)
        matched = float(np.isfinite(fg).mean())
        if matched < 0.99:
            raise RuntimeError(
                f"fg/bg join matched only {matched:.1%} of spots -- almost certainly the "
                "wrong processed asset. Resolve it from ds_config.json's dataset_folder, "
                "not by timestamp (see fgbg module docstring).")
        fg_bg = (fg, bg)

    out, sep, log, E, eminfo = core.unmix_v3(
        spots, B_ctrl, powers, channels=channels, fg_bg=fg_bg, **unmix_kw)
    cxg = core.cellxgene(out, gene_map, round_key, apply_v3=True)
    return dict(spots=out, separability=sep, decisions=log,
                endmembers=E, endmember_info=eminfo, cellxgene=cxg)


def candidate_processed_assets(processed_root, mouse_id):
    """Processed-asset directories for this mouse that carry what unmixing needs.

    A directory qualifies when it has acquisition.json (laser power, required). Sorted
    NEWEST FIRST by the trailing _processed_<timestamp> in the name, falling back to
    mtime when the name does not parse.
    """
    root = Path(processed_root)
    if not root.is_dir():
        return []
    hits = [p for p in root.iterdir()
            if p.is_dir() and mouse_id in p.name and "_processed_" in p.name
            and (p / "acquisition.json").exists()]

    def key(p):
        stamp = p.name.split("_processed_")[-1]
        return (stamp, p.stat().st_mtime)

    return sorted(hits, key=key, reverse=True)


def processed_asset_for_round(processed_root, mouse_id, round_key):
    """The processed asset that declares itself to be this round, or None.

    `processing_manifest.json` carries `round`, so the mapping is read from the asset
    rather than guessed from its timestamp. This matters because laser power is per
    round: resolving by recency instead can hand an R2 run R6's acquisition.json, which
    mis-scales every endmember with nothing in the output to show it.
    """
    import json as _json

    want = int("".join(ch for ch in str(round_key) if ch.isdigit()))
    for man in sorted(Path(processed_root).glob("*/processing_manifest.json")):
        if mouse_id not in man.parent.name:
            continue
        try:
            with open(man) as fh:
                n = _json.load(fh).get("round")
        except (OSError, ValueError):
            continue
        if n is not None and int(n) == want and (man.parent / "acquisition.json").exists():
            return man.parent
    return None


def round_inputs_from_asset(asset_dir, mouse_id, round_key, processed_root=None,
                            processed_folder=None):
    """Locate this round's acquisition metadata and fg/bg stats files on disk.

    asset_dir         the pairwise-unmixing asset root (contains {mouse}_{R}/ folders)
    processed_root    parent directory holding the processed assets
    processed_folder  explicit processed-asset directory name, overriding everything
                      below. Use this when you know which asset you want.

    Resolution order:
      1. `processed_folder`, if given -- the caller has decided.
      2. `dataset_folder` from this round's ds_config.json, if that directory exists.
         This is the asset the pairwise-unmixing outputs were generated against.
      3. Newest processed asset for this mouse under `processed_root` that has
         acquisition.json.

    Step 3 exists because a mount may legitimately carry a different (often newer)
    processed asset than the one named in ds_config. Which asset is right is the
    user's call: pass `processed_folder` to force it.

    Returns (acquisition_path, diag_paths); either may be None when unavailable, in
    which case the caller must supply powers explicitly and fg/bg are skipped.
    """
    rdir = Path(asset_dir) / f"{mouse_id}_{round_key}"
    root = None

    if processed_folder:
        root = (Path(processed_root) / processed_folder if processed_root
                else Path(processed_folder))
    else:
        cfg = rdir / "ds_config.json"
        if cfg.exists():
            folder = resolve_dataset_folder(cfg)
            cand = Path(processed_root) / folder if processed_root else Path(folder)
            if cand.is_dir():
                root = cand
        if root is None and processed_root:
            # Ask each processed asset which round it IS. Exact, per round, and the
            # only resolution available when there is no pairwise asset to read a
            # dataset_folder out of.
            root = processed_asset_for_round(processed_root, mouse_id, round_key)
        if root is None and processed_root:
            # Last resort: the newest asset for this mouse. WRONG ROUND IS POSSIBLE --
            # laser power is per round, and taking R6's acquisition.json for an R2 run
            # silently mis-scales every endmember. Announced so it is not invisible.
            cands = candidate_processed_assets(processed_root, mouse_id)
            if cands:
                root = cands[0]
                print(f"WARNING: {round_key}: no processed asset declares this round; "
                      f"falling back to the newest one ({root.name}). Its laser power "
                      f"may belong to a different round -- pass --processed-folder to "
                      f"force the right asset.", flush=True)

    if root is None:
        return None, None

    acq = root / "acquisition.json"
    diag = {}
    for c in CHANS:
        p = Path(diagonal_stats_path(str(root), c))
        if p.exists():
            diag[c] = str(p)
    return (str(acq) if acq.exists() else None), (diag or None)


#: Parquet options for the per-round spot tables. These files dominate the capsule's
#: output: six rounds of 800995 came to 15.4 GB with pandas' default snappy, and Code
#: Ocean uploads /results to S3 AFTER the script exits -- so that volume is wall-clock
#: the run's own timer never sees, and was most of the gap between "32 min unmixed" and
#: an hour and a quarter observed end to end. zstd level 3 takes the same set to 6.3 GB
#: (59% smaller) and is marginally FASTER to write than snappy, because less output means
#: less I/O. Verified value-identical on a full R5 round-trip before adoption.
_PARQUET_OPTS = dict(compression="zstd", compression_level=3)

#: Columns safe to narrow before writing: every one is an index, coordinate or count that
#: cannot exceed int32 on data of this scale, and the four object columns hold a handful
#: of distinct strings each so dictionary encoding is strictly better than repeating them
#: 20M times. Narrowing is applied to the WRITTEN copy only -- it halves in-memory size
#: too (8.6 GB -> 3.5 GB on R5) but the round's own arrays are untouched.
_NARROW_INT = ("spot_id", "spot_uid_int", "chan_spot_id", "cell_id", "z", "y", "x")
_NARROW_CAT = ("chan", "v3_chan", "crosstalk_source_chan", "decision_rule")


def _narrow(spots):
    """Downcast index/coordinate columns and dictionary-encode the low-cardinality ones.

    Values are preserved exactly; only their storage type changes. Anything that does not
    fit int32 is left alone rather than silently wrapping.
    """
    out = spots
    for col in _NARROW_CAT:
        if col in out.columns and str(out[col].dtype) == "object":
            out[col] = out[col].astype("category")
    for col in _NARROW_INT:
        if col in out.columns and str(out[col].dtype) == "int64":
            if out[col].abs().max() < 2 ** 31:
                out[col] = out[col].astype("int32")
    return out


def run_mouse(asset_dir, mouse_id, rounds, gene_maps, processed_root=None,
              powers_by_round=None, output_dir=None, use_fgbg=True,
              processed_folder=None, write_metadata=True, experimenter=None,
              write_anndata=True, write_plots=True, write_spots=True,
              spots_from="auto", **unmix_kw):
    """Unmix every round of one mouse and concatenate the cell x gene tables.

    spots_from  which family of spot table to read: "auto" (pairwise when attached),
                "pairwise", or "processed". See spots_io -- the processed table is a
                column superset and makes the fg/bg join unnecessary, but it carries
                2-4% more spots, so the two do not produce the same cell x gene table.

    Writes per-round spot tables and a combined cell x gene table when output_dir is
    given. Laser power is read per round from acquisition.json unless powers_by_round
    supplies it explicitly.

    write_spots defaults True: the per-round spot tables are the primary output, the only
    place the per-spot decisions survive, and the cell x gene table cannot be re-derived
    without them. Pass False (--no-spots) only when the cell x gene table is all that is
    wanted and the ~2 GB per mouse is not worth keeping.
    """
    import time as _time
    _t_all = _time.time()
    asset_dir = Path(asset_dir)
    cxgs, seps, logs, summary = [], [], [], []
    schemas = {}
    _n_rounds = len(rounds)
    for _i_round, round_key in enumerate(rounds, start=1):
        # A six-round mouse is tens of minutes of work. Announce each round so a long
        # silence is distinguishable from a hang, and so the remaining time is legible.
        print(f"\n=== round {_i_round}/{_n_rounds}: {round_key} "
              f"({_time.time() - _t_all:.0f}s elapsed) ===", flush=True)
        from . import spots_io
        spots, _schema = spots_io.load_spot_table(
            round_key, mouse_id, processed_root or asset_dir.parent,
            source=spots_from, asset_dir=asset_dir)
        print(f"  spots  : {_schema['n_rows']:,} rows from the {_schema['family']} "
              f"table ({_schema['asset']})", flush=True)
        schemas[round_key] = {k: v for k, v in _schema.items()
                              if k in ("family", "has_native_fgbg", "n_rows",
                                       "channels", "asset", "path")}
        acq_path, diag = round_inputs_from_asset(
            asset_dir, mouse_id, round_key, processed_root,
            processed_folder=processed_folder)
        if powers_by_round and round_key in powers_by_round:
            powers = powers_by_round[round_key]
        elif acq_path:
            powers = load_powers(acq_path)
        else:
            raise RuntimeError(
                f"no laser power for {mouse_id} {round_key}: acquisition.json not found "
                "and no explicit powers given. Power must come from the round's own "
                "acquisition metadata -- a stored table from another mouse is wrong.")
        res = run_round(spots, powers, gene_maps[round_key], round_key,
                        diag_paths=(diag if use_fgbg else None), **unmix_kw)
        for frame in (res["separability"], res["decisions"]):
            frame.insert(0, "round", round_key)
            frame.insert(0, "mouse", mouse_id)
        seps.append(res["separability"])
        logs.append(res["decisions"])
        cxgs.append(res["cellxgene"].assign(arm="calibrated"))
        for c in CHANS:
            n_det = int((res["spots"].chan == c).sum())
            n_fin = int(((res["spots"].v3_chan == c)
                         & (res["spots"].v3_action != "delete")).sum())
            summary.append(dict(mouse=mouse_id, round=round_key, chan=c,
                                gene=gene_maps[round_key].get(c),
                                n_detected=n_det, n_final=n_fin,
                                pct_change=round(100 * (n_fin - n_det) / max(n_det, 1), 2)))
        if output_dir and write_spots:
            outp = Path(output_dir)
            outp.mkdir(parents=True, exist_ok=True)
            _t_w = _time.time()
            _narrow(res["spots"]).to_parquet(
                outp / f"{mouse_id}_{round_key}_unmixed_spots.parquet", index=False,
                **_PARQUET_OPTS)
            _mb = (outp / f"{mouse_id}_{round_key}_unmixed_spots.parquet").stat().st_size / 1e6
            print(f"  [7/7] spot table written: {_mb:,.0f} MB in "
                  f"{_time.time() - _t_w:.0f}s", flush=True)
        elif output_dir:
            print("  [7/7] spot table NOT written (--no-spots); per-spot decisions for "
                  f"{round_key} are discarded", flush=True)
        del spots, res
        print(f"  round {round_key} done ({_time.time() - _t_all:.0f}s elapsed)", flush=True)
    print(f"\n=== all {_n_rounds} rounds unmixed in {(_time.time() - _t_all)/60:.1f} min; "
          f"building cell x gene ===", flush=True)
    _t_post = _time.time()
    cxg_all = pd.concat(cxgs, ignore_index=True)
    table = cxg_all.pivot_table(index="cell_id", columns="round_chan_gene",
                                values="spot_count", aggfunc="sum", fill_value=0)
    # Correct wrong gene names once, here, so the CSV header, the .h5ad var and the
    # figure axes all carry the corrected symbol. Announced, never silent.
    from .annotate import rename_gene_aliases
    table, _renamed = rename_gene_aliases(table)
    print(f"  cell x gene: {table.shape[0]:,} cells x {table.shape[1]} genes "
          f"({_time.time() - _t_post:.0f}s)", flush=True)
    result = dict(cellxgene=table,
                  spot_tables=schemas,
                  separability=pd.concat(seps, ignore_index=True),
                  decisions=pd.concat(logs, ignore_index=True),
                  summary=pd.DataFrame(summary))
    if output_dir:
        outp = Path(output_dir)
        # /results always exists in Code Ocean so this never fired there, but a nested
        # --output-dir on any other host loses the entire run at the final write.
        outp.mkdir(parents=True, exist_ok=True)
        print("  writing tables...", flush=True)
        table.to_csv(outp / f"{mouse_id}_cellxgene.csv")
        result["separability"].to_csv(outp / f"{mouse_id}_separability.csv", index=False)
        result["decisions"].to_csv(outp / f"{mouse_id}_decisions.csv", index=False)
        result["summary"].to_csv(outp / f"{mouse_id}_spot_change.csv", index=False)
        if write_anndata:
            try:
                from . import annotate
                adata = annotate.build_anndata(
                    table, extra_uns=dict(mouse_id=mouse_id, rounds=list(rounds)))
                h5 = outp / f"{mouse_id}_cellxgene_annotated.h5ad"
                adata.write_h5ad(h5)
                result["anndata"] = str(h5)
                print(f"  annotated .h5ad written ({_time.time() - _t_post:.0f}s "
                      "since unmixing finished)", flush=True)
                if write_plots:
                    from . import plots as _plots
                    result["plots"] = _plots.write_plots(adata, outp, mouse_id, rounds)
            except ImportError as exc:
                # anndata is an optional extra; a missing package must not lose the
                # unmixing results that already succeeded.
                print(f"WARNING: AnnData not written ({exc}). pip install anndata")
        if write_metadata:
            # One timestamp for data_description.json and the asset manifest, so the
            # asset name and the metadata inside the asset agree.
            result["creation_time"] = _dt.datetime.now(_dt.timezone.utc)
            result["metadata"] = _write_asset_metadata(
                asset_dir, mouse_id, rounds, outp, processed_root, processed_folder,
                result, dict(use_fgbg=use_fgbg, write_spots=write_spots, **unmix_kw),
                experimenter=experimenter,
                creation_time=result["creation_time"],
                write_spots=write_spots)
    return result


def _write_asset_metadata(asset_dir, mouse_id, rounds, outp, processed_root,
                          processed_folder, result, params, experimenter=None,
                          creation_time=None, write_spots=True):
    """Carry upstream schema files forward and record this step in processing.json.

    creation_time is passed in so data_description.json and the asset manifest written
    by run_capsule.py carry the SAME timestamp; deriving it independently in each place
    makes them differ by however long post-processing took.
    """
    proc_dirs = []
    for round_key in rounds:
        acq, _ = round_inputs_from_asset(asset_dir, mouse_id, round_key,
                                         processed_root, processed_folder)
        if acq:
            proc_dirs.append(str(Path(acq).parent))
    seen, source_dirs = set(), []
    for d in proc_dirs + [str(asset_dir)]:
        if d not in seen:
            seen.add(d)
            source_dirs.append(d)

    copied = metadata.copy_upstream_metadata(source_dirs, outp)
    # data_description is WRITTEN, not copied: a derived asset names itself and points
    # at its parent (see metadata.derived_data_description).
    dd = None
    for d in source_dirs:
        dd = metadata.derived_data_description(
            d, outp, creation_time=creation_time,
            investigators=[experimenter] if experimenter else None)
        if dd:
            break
    upstream = metadata.find_upstream_processing(source_dirs)
    summ = result["summary"]
    dp = metadata.unmixing_data_process(
        input_locations=[str(Path(asset_dir) / f"{mouse_id}_{r}") for r in rounds],
        output_location=str(outp),
        parameters={"rounds": list(rounds), "mouse_id": mouse_id,
                    "processed_folder": processed_folder, **_jsonable(params)},
        # Only claim files that were actually written: processing.json is the record of
        # what this asset contains, and naming absent spot tables would make it wrong.
        outputs={"cellxgene": f"{mouse_id}_cellxgene.csv",
                 "spots": ([f"{mouse_id}_{r}_unmixed_spots.parquet" for r in rounds]
                           if write_spots else []),
                 "decisions": f"{mouse_id}_decisions.csv",
                 "spot_change": f"{mouse_id}_spot_change.csv",
                 "annotated_cellxgene": f"{mouse_id}_cellxgene_annotated.h5ad"},
        notes=(f"{int(summ.n_detected.sum()):,} spots in, "
               f"{int(summ.n_final.sum()):,} out across {len(rounds)} round(s). "
               "Geometric QC flags are annotated, not applied."),
    )
    path = metadata.write_processing(outp, dp, upstream_processing=upstream,
                                     processor_full_name=experimenter or "")
    return {"processing": path, "copied": copied, "upstream_processing": upstream,
            "data_description": dd}


def _jsonable(d):
    """Drop values json.dump cannot serialise (arrays, callables)."""
    out = {}
    for k, v in (d or {}).items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (str, int, float, bool)) for x in v):
            out[k] = list(v)
    return out
