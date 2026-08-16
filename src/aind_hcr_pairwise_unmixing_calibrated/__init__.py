"""HCR pairwise spot unmixing, v5.

A standalone reimplementation of the crosstalk-removal stage. It does NOT import from
or modify aind-spot-spectral-unmixing; the upstream engine and its capsule are
untouched.

What v5 changes relative to the pairwise stage in the upstream engine:

  * takes the FULL spot table -- geometric QC is annotated, not applied
  * dye lines (endmembers) estimated from SPATIALLY ISOLATED spots, not self-peaking
    ones (in one mouse 93% of Sst spots peak in the Cck channel, so the self-peak
    criterion is circular)
  * bleed magnitude and its tolerance MEASURED per direction from bright isolated
    spots, not inherited constants
  * the allowlist is control-derived and BIDIRECTIONAL: both directions of any pair the
    single-dye control certifies as bleeding. Distant pairs are never admitted -- an
    elevated in-tissue ratio there is co-expression, by panel design
  * deletion requires spatial co-location AND magnitude AND spectral evidence;
    reassignment is the rare no-partner case; undecidable spots are flagged and kept
  * output is UNFILTERED and carries raw fg / local bg per spot

Entry point: pipeline.run_mouse() or pipeline.run_round().
"""
from . import control, core, fgbg, pipeline  # noqa: F401
from .control import control_matrix, load_powers, powers_from_acquisition  # noqa: F401
from .fgbg import attach_fg_bg, bg_mad_threshold  # noqa: F401
from .pipeline import run_mouse, run_round  # noqa: F401

__version__ = "5.0.0"
