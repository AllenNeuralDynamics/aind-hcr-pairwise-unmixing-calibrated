"""Join raw foreground and local background onto a spot table.

The delivered mixed_spots_{R}.pkl keeps only FG - BG, which cannot distinguish 300 over
a background of 100 from 300 over a background of 900. Local BG varies ~3x within a
single channel, so the pair carries information the difference discards. v5 writes both
columns UNFILTERED and leaves thresholding to cell x gene construction.

Two traps, both hit during development and both silent if you get them wrong:

  1. TWO processed assets per round. Only one matches the pairwise-unmixing spot set.
     Resolve it from ds_config.json's `dataset_folder` (see control.resolve_dataset_folder),
     never by picking a timestamp. Wrong asset -> row counts differ by ~3% and the join
     drops most spots.

  2. `chan_spot_id` is NOT a row index into the stats file, and each channel's file has
     its own row order. Positional indexing "works" and gives r = 0.70/0.56 against the
     pipeline's own subtracted value. The exact-coordinate join per channel gives
     r = 1.000000 on all five channels (max abs diff 0.001, 19,630,523 of 19,630,523
     spots matched on 800995 R5).
"""
import numpy as np
import pandas as pd

CHANS = ["488", "514", "561", "594", "638"]


def _coord_key(z, y, x):
    return (np.asarray(z, dtype=np.int64) * 100_000_000
            + np.asarray(y, dtype=np.int64) * 10_000
            + np.asarray(x, dtype=np.int64))


def diagonal_stats_path(dataset_folder, channel):
    """Path (relative to the processed asset root) of a channel's own-spot stats file.

    The 'diagonal' of the channel x spots grid: image_data_channel_C measured at the
    spots detected in channel C. Off-diagonal files measure one channel's image at
    another channel's spots and are not needed here.
    """
    return (f"{dataset_folder}/image_spot_detection/channel_{channel}_stats/"
            f"image_data_channel_{channel}_versus_spots_{channel}.csv")


def attach_fg_bg(spots, diag_paths, channels=CHANS,
                 z_col="z", y_col="y", x_col="x", chan_col="chan"):
    """Return (fg, bg) arrays aligned to `spots`, joined in the DETECTED channel.

    `diag_paths` maps channel -> local path of that channel's diagonal stats CSV.
    Missing channels leave NaN. Verify with:

        assert np.corrcoef(fg - bg, spots["chan_<c>_intensity"])[0, 1] > 0.9999
    """
    key = _coord_key(spots[z_col], spots[y_col], spots[x_col])
    chan = spots[chan_col].astype(str).to_numpy()
    fg = np.full(len(spots), np.nan, np.float32)
    bg = np.full(len(spots), np.nan, np.float32)
    for c in channels:
        sel = chan == c
        if not sel.any() or c not in diag_paths:
            continue
        tab = pd.read_csv(diag_paths[c], usecols=["Z", "Y", "X", "FG", "BG"],
                          dtype={"Z": np.int32, "Y": np.int32, "X": np.int32,
                                 "FG": np.float32, "BG": np.float32})
        tab_key = _coord_key(tab.Z, tab.Y, tab.X)
        for col, dest in (("FG", fg), ("BG", bg)):
            ser = pd.Series(tab[col].to_numpy(), index=tab_key)
            ser = ser[~ser.index.duplicated()]
            dest[sel] = pd.Series(key[sel]).map(ser).to_numpy()
        del tab, tab_key
    return fg, bg


def bg_mad_threshold(bg, n_mad=2.0):
    """median(BG) + n_mad * MAD(BG): a per-channel spot-quality floor.

    MAD not SD, because BG is right-skewed and heavy-tailed (it tracks tissue density).
    A 2-SD threshold lands ABOVE the FG median in three of five channels and discards
    38-65% of spots. Symmetric bounds on FG fail outright: median(FG) - 2*SD(FG) is
    negative in all five channels.

    NOT applied by default. Measured on two mice this removes 8.7-18.9% (800995) and
    6.1-9.6% (788406) of spots and improves 9 of 16 marker-pair correlations, but
    regresses Lamp5-Calb2 by +0.27 in BOTH mice and costs 1,802 Gad2+ cells in 800995
    against 18 in 788406. Ship the columns; let the downstream step choose.
    """
    bg = np.asarray(bg, dtype=float)
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        return float("nan")
    med = float(np.median(bg))
    return med + n_mad * float(np.median(np.abs(bg - med))) * 1.4826
