"""Locate and load a round's spot table, from either asset family.

Two different tables in the wild carry the name `mixed_spots_<R>.pkl`:

  PAIRWISE   HCR_<mouse>_pairwise-unmixing_<date>/<mouse>_<R>/mixed_spots_<R>.pkl
             Written by AllenNeuralDynamics/hcr-pairwise-spot-unmixing. Carries
             `chan_<ch>_intensity` only -- fg and bg are already subtracted and
             discarded. Recovering them costs the exact-coordinate join in `fgbg`,
             with its two documented traps.

  PROCESSED  HCR_<mouse>_<date>_processed_<date>/image_spot_spectral_unmixing/
             mixed_spots_<R>.pkl
             Written by AllenNeuralDynamics/aind-spot-spectral-unmixing
             (`process_round.py`, after `calculate_intensities`). Carries
             `chan_<ch>_fg`, `chan_<ch>_bg` AND `chan_<ch>_intensity`, where
             intensity is defined there as fg - bg -- so it is a SUPERSET of the
             pairwise table's channel columns, and the fg/bg join is unnecessary.

That superset is the whole reason this module exists: reading the processed table
removes a join, a required second asset resolution, and the failure mode where the
wrong processed asset silently drops 3% of spots.

MEASURED DIFFERENCES (800792, from the pickle headers -- see PROCESSED_ONLY.md)
------------------------------------------------------------------------------
    rows        processed has 2-4% MORE spots (R1 58,616,632 vs 56,306,976)
    bytes/row   221 vs 110 for a 5-channel round; the extra is fg and bg
    dtype       pairwise stores spot_id as a Categorical of "<chan>_<index>"

The row difference is why this is a branch and not a patch: more input spots means a
different cell x gene table, and which spot set is correct is a question about the
upstream filter, not about this code.
"""
import json
from pathlib import Path

import pandas as pd

#: Base (non per-channel) columns of the processed table, in the order
#: aind-spot-spectral-unmixing's data_loader writes them.
PROCESSED_BASE_COLS = ("spot_id", "chan", "chan_spot_id", "cell_id", "round",
                       "z", "y", "x", "z_center", "y_center", "x_center", "dist", "r")

#: What this pipeline actually requires of a spot table, whatever wrote it.
REQUIRED_COLS = ("chan", "cell_id")


def channel_columns(frame, kind):
    """{channel: column} for a per-channel family: kind in fg / bg / intensity."""
    out = {}
    for col in frame.columns:
        s = str(col)
        if s.startswith("chan_") and s.endswith(f"_{kind}"):
            out[s[len("chan_"):-len(f"_{kind}")]] = col
    return out


def describe_schema(frame):
    """Which family a loaded spot table belongs to, and what it can supply.

    Detection is by CONTENT, not by path: a table carrying `chan_<ch>_fg` can supply
    fg/bg natively regardless of which asset it came out of. Anything else has to have
    them joined on. This keeps the pairwise path working unchanged while letting the
    processed path skip the join.
    """
    fg, bg = channel_columns(frame, "fg"), channel_columns(frame, "bg")
    inten = channel_columns(frame, "intensity")
    native = bool(fg) and set(fg) == set(bg)
    chans = sorted(set(inten) | set(fg),
                   key=lambda c: (not str(c).isdigit(), int(c) if str(c).isdigit() else c))
    return dict(
        family="processed" if native else "pairwise",
        channels=chans,
        has_native_fgbg=native,
        n_rows=int(len(frame)),
        missing_required=[c for c in REQUIRED_COLS if c not in frame.columns],
        intensity_cols=inten, fg_cols=fg, bg_cols=bg,
    )


def _processed_spot_table(data_dir, mouse_id, round_key):
    """The one processed asset for THIS mouse holding THIS round's spot table.

    Two guards, both learned the hard way. The candidate set is scoped to the mouse,
    because an unscoped glob will happily return another animal's table. And where a
    candidate's `processing_manifest.json` declares a round, that declaration decides
    -- the filename is a label, the manifest is the asset's own statement about what
    it contains.

    Raises when more than one asset survives. Several reprocessing versions of the
    same round exist in the bucket and they are NOT interchangeable; choosing between
    them by sort order is the bug this function was written to remove, so an ambiguous
    mount set is an error the caller has to resolve by attaching one.
    """
    want = int("".join(ch for ch in str(round_key) if ch.isdigit()))
    hits = []
    for cand in sorted(data_dir.glob(
            f"*{mouse_id}*/image_spot_spectral_unmixing/mixed_spots_{round_key}.pkl")):
        man = cand.parent.parent / "processing_manifest.json"
        declared = None
        if man.exists():
            try:
                with open(man) as fh:
                    declared = json.load(fh).get("round")
            except (OSError, ValueError):
                declared = None
        if declared is None or int(declared) == want:
            hits.append(cand)
    if len(hits) > 1:
        raise SystemExit(
            f"{round_key}: {len(hits)} processed assets for {mouse_id} carry this "
            f"round's spot table, and they are not interchangeable:\n  "
            + "\n  ".join(str(h.parent.parent.name) for h in hits)
            + "\nAttach exactly one per round.")
    return hits[0] if hits else None


def find_spot_table(round_key, mouse_id, data_dir, source="auto", asset_dir=None):
    """Path to a round's spot table. source in auto / pairwise / processed.

    `auto` prefers the pairwise asset when it is attached, because that is the spot
    set every result to date was produced from. Switching input is an explicit choice,
    not something to happen because an asset was attached or detached.
    """
    data_dir = Path(data_dir)
    pw = None
    if asset_dir is not None:
        cand = Path(asset_dir) / f"{mouse_id}_{round_key}" / f"mixed_spots_{round_key}.pkl"
        pw = cand if cand.exists() else None
    if pw is None:
        for d in sorted(data_dir.glob(f"*pairwise-unmixing*/{mouse_id}_{round_key}")):
            cand = d / f"mixed_spots_{round_key}.pkl"
            if cand.exists():
                pw = cand
                break

    # Scoped to the mouse, and the round is confirmed from the asset's own manifest
    # rather than trusted from the filename. The first version of this globbed
    # `*/image_spot_spectral_unmixing/...` across every mount and took the first
    # sorted hit: with two mice attached that silently read HCR_800792 for a run
    # invoked with --mouse-id 800995, and produced a complete, plausible, entirely
    # wrong asset. Alphabetical order decided which mouse's data a run used.
    pr = _processed_spot_table(data_dir, mouse_id, round_key)

    if source == "pairwise":
        if pw is None:
            raise SystemExit(f"--spots-from pairwise: no mixed_spots_{round_key}.pkl "
                             f"in a pairwise-unmixing asset under {data_dir}")
        return pw, "pairwise"
    if source == "processed":
        if pr is None:
            raise SystemExit(
                f"--spots-from processed: no image_spot_spectral_unmixing/"
                f"mixed_spots_{round_key}.pkl under {data_dir}. Attach the processed "
                f"asset for this round (its processing_manifest.json declares which "
                f"round it is).")
        return pr, "processed"
    if pw is not None:
        return pw, "pairwise"
    if pr is not None:
        return pr, "processed"
    raise SystemExit(f"no spot table for {round_key} under {data_dir}")


def load_spot_table(round_key, mouse_id, data_dir, source="auto", asset_dir=None):
    """(frame, schema dict) for a round, from whichever family is selected."""
    path, family = find_spot_table(round_key, mouse_id, data_dir, source, asset_dir)
    frame = pd.read_pickle(path)
    schema = describe_schema(frame)
    schema["path"], schema["asset"] = str(path), path.parent.name
    if schema["missing_required"]:
        raise SystemExit(
            f"{path} is missing {schema['missing_required']}, which this pipeline "
            f"needs. Columns present: {sorted(map(str, frame.columns))[:20]}")
    if family == "processed" and not schema["has_native_fgbg"]:
        print(f"WARNING: {path.name} came from a processed asset but carries no "
              f"chan_<ch>_fg columns; fg/bg will have to be joined as before.")
    return frame, schema
