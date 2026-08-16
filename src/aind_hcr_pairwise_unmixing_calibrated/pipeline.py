"""Per-round driver: spot table in, unmixed spot table + cell x gene out.

This is the only module the capsule needs to call. It is deliberately a thin wrapper --
all algorithm decisions live in core.py -- so that the capsule's own code stays a
data-plumbing script.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import core
from .control import CHANS, control_matrix, load_powers, resolve_dataset_folder
from .fgbg import attach_fg_bg, diagonal_stats_path


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
    if diag_paths:
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
            cands = candidate_processed_assets(processed_root, mouse_id)
            if cands:
                root = cands[0]

    if root is None:
        return None, None

    acq = root / "acquisition.json"
    diag = {}
    for c in CHANS:
        p = Path(diagonal_stats_path(str(root), c))
        if p.exists():
            diag[c] = str(p)
    return (str(acq) if acq.exists() else None), (diag or None)


def run_mouse(asset_dir, mouse_id, rounds, gene_maps, processed_root=None,
              powers_by_round=None, output_dir=None, use_fgbg=True,
              processed_folder=None, **unmix_kw):
    """Unmix every round of one mouse and concatenate the cell x gene tables.

    Writes per-round spot tables and a combined cell x gene table when output_dir is
    given. Laser power is read per round from acquisition.json unless powers_by_round
    supplies it explicitly.
    """
    asset_dir = Path(asset_dir)
    cxgs, seps, logs, summary = [], [], [], []
    for round_key in rounds:
        pkl = asset_dir / f"{mouse_id}_{round_key}" / f"mixed_spots_{round_key}.pkl"
        spots = pd.read_pickle(pkl)
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
        if output_dir:
            outp = Path(output_dir)
            outp.mkdir(parents=True, exist_ok=True)
            res["spots"].to_parquet(
                outp / f"{mouse_id}_{round_key}_unmixed_calibrated.parquet", index=False)
        del spots, res
    cxg_all = pd.concat(cxgs, ignore_index=True)
    table = cxg_all.pivot_table(index="cell_id", columns="round_chan_gene",
                                values="spot_count", aggfunc="sum", fill_value=0)
    result = dict(cellxgene=table,
                  separability=pd.concat(seps, ignore_index=True),
                  decisions=pd.concat(logs, ignore_index=True),
                  summary=pd.DataFrame(summary))
    if output_dir:
        outp = Path(output_dir)
        table.to_csv(outp / f"{mouse_id}_cellxgene_calibrated.csv")
        result["separability"].to_csv(outp / f"{mouse_id}_separability_calibrated.csv", index=False)
        result["decisions"].to_csv(outp / f"{mouse_id}_decisions_calibrated.csv", index=False)
        result["summary"].to_csv(outp / f"{mouse_id}_spot_change_calibrated.csv", index=False)
    return result
