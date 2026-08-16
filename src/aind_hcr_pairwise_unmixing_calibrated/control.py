"""Single-dye control matrix and per-round laser power.

The control matrix is GROUND TRUTH shared across all mice: it was recorded in separate
experiments with one probe per sample, all channels imaged, to quantify crosstalk
percentage. It is a property of the dye set and the optics, not of any subject, so the
same matrix is used for every mouse. Only its scaling to a round's laser powers is
per-subject.

Laser power ALWAYS comes from each round's own acquisition.json. Do not use any stored
per-round power table: one such table in the project belongs to mouse 804363 and does
not apply to other subjects. Power genuinely differs between mice -- R5 561 nm is 25%
in 804363, 15% in 800995 and 30% in 788406, and R5 488 nm spans 5% to 30% (a 6x range)
-- and beta scales with the victim/source power ratio, so borrowing another mouse's
table mis-scales every magnitude test.
"""
import json
from pathlib import Path

import numpy as np

CHANS = ["488", "514", "561", "594", "638"]

# B_CTRL[source, victim] = fraction of the source dye's own-channel intensity that it
# deposits in the victim channel, under the control experiment's imaging conditions.
# NaN = not measured. Reconstructed from the project's control direction table
# (control_vs_indata_direction.csv): forward value CTRL_beta, reverse value
# CTRL_beta / CTRL_asym.
CONTROL_BLEED = {
    ("594", "561"): 0.4670,   # Sst  -> Cck   (R5 naming; channel pairs are gene-agnostic)
    ("561", "594"): 0.0137,
    ("514", "488"): 0.2800,   # Pvalb -> Npy
    ("488", "514"): 0.1000,
    ("638", "594"): 0.1410,   # Vip  -> Sst
    ("594", "638"): 0.0186,
    ("561", "514"): 0.0660,   # Cck  -> Pvalb
    ("514", "561"): 0.0077,
    ("514", "594"): 0.0600,   # Pvalb -> Sst
    ("594", "514"): 0.0009,
    ("561", "638"): 0.0060,   # Cck  -> Vip   (distant pair; below ctrl_min, never used
    ("638", "561"): 0.0010,   #   for removal -- an elevated in-tissue ratio here is
                              #   co-expression, by panel design)
    ("488", "594"): 0.0220,   # Npy  -> Sst
    ("594", "488"): 0.0009,
}


def control_matrix(channels=CHANS):
    """Return the 5x5 control bleed matrix as an array, NaN where unmeasured."""
    idx = {c: i for i, c in enumerate(channels)}
    B = np.full((len(channels), len(channels)), np.nan)
    for (s, v), beta in CONTROL_BLEED.items():
        if s in idx and v in idx:
            B[idx[s], idx[v]] = beta
    return B


def powers_from_acquisition(acq):
    """Per-channel excitation power (%) from a parsed acquisition.json.

    Walks data_streams -> configurations -> channels -> light_sources and keys on the
    excitation wavelength. Validated against a stored table for mouse 804363: exact
    match on all six rounds and all channels.
    """
    out = {}
    for stream in acq.get("data_streams", []):
        for conf in stream.get("configurations", []):
            for chan in conf.get("channels", []):
                for src in (chan.get("light_sources") or []):
                    wl = src.get("wavelength")
                    pw = src.get("power")
                    if wl is None or pw is None:
                        continue
                    out[str(int(float(wl)))] = float(pw)
    return out


def load_powers(acquisition_path):
    """Read acquisition.json from disk and return {channel: power}."""
    with open(acquisition_path) as fh:
        return powers_from_acquisition(json.load(fh))


def resolve_dataset_folder(ds_config_path):
    """The processed-asset folder that MATCHES this round's spot table.

    There are often TWO processed assets per round and only one corresponds to the spot
    set the pairwise-unmixing asset carries (for 800995 R5: 2,506,903 rows vs
    2,435,209). The pointer in ds_config.json is authoritative -- never pick by
    timestamp.
    """
    with open(ds_config_path) as fh:
        return json.load(fh)["dataset_folder"]
