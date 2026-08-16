# aind-hcr-pairwise-unmixing-calibrated

Crosstalk removal ("unmixing") for thick-tissue HCR spot tables.

**Calibrated** is the point of the name. The prior pairwise implementation decided what
to delete using constants and in-data heuristics; every quantity this version acts on is
instead tied to an independent measurement:

| quantity | calibrated against |
|---|---|
| which channel pairs may bleed | the single-dye control experiment (one probe per sample) |
| how much they bleed | that control, scaled by the round's own laser powers, then re-measured on spatially isolated spots |
| how far a spot may deviate and still count as bleed | a measured percentile of the same isolated-spot population, per direction |
| what a pure dye looks like | spots with no other-channel neighbour, so they cannot be ghosts |

Nothing in the removal rule is a tuned constant.

**This repository is standalone.** It does not import from, depend on, or modify
[`aind-spot-spectral-unmixing`](https://github.com/mattjdavis/aind-spot-spectral-unmixing)
or the capsule that uses it. Both remain untouched; run v5 from its own capsule against
the same data assets.

## What v5 does differently

The upstream pairwise stage removes crosstalk by finding same-cell overlapping spots in
adjacent channel pairs and deleting whichever fits its own dye line worse. v5 keeps that
core idea — a ghost sits on top of its source and has a predictable brightness — and
changes six things, each because a measurement said so.

| change | why |
|---|---|
| takes the **full** spot table; geometric QC annotated, not applied | measured over 66M spots in 4 rounds: one of the three filters cannot remove anything by construction, and the other two discard spots no dirtier than the ones they keep, and brighter |
| dye lines from **spatially isolated** spots | the old "brightest in its own channel" purity rule is circular — in one mouse 93% of Sst spots peak in the Cck channel *because of* the bleed being measured. The isolated-spot estimate lands within 1° of what the single-dye control independently predicts |
| bleed magnitude and tolerance **measured per direction** | the inherited 1.5× tolerance had no derivation. The bleed ratio is not constant: flat for bright spots, rising for dim ones (a background floor). v5 fits on the bright quartile of isolated spots and measures the tolerance as an upper percentile of the same labelled set |
| allowlist is **control-derived and bidirectional** | apparent bleed between distant channels is co-expression, not leakage — co-expressed genes are deliberately placed in non-neighbouring channels precisely because bleed there is negligible. But a pair that bleeds one way bleeds both, and the control's asymmetry does not hold in tissue (Vip/Sst: 7.6× predicted, 1.5–3.7× measured) |
| deletion needs spatial **and** magnitude **and** spectral evidence; undecidable spots flagged, not deleted | co-location runs 200–600× above a displaced null, so it is strong evidence; but a spot whose predicted bleed sits under the victim channel's noise floor cannot be judged either way |
| output **unfiltered**, carrying raw fg and local bg | the delivered table keeps only FG−BG, which cannot tell 300-over-100 from 300-over-900. Thresholding belongs at cell × gene construction |

## Install

```bash
pip install -e .
pytest tests/          # synthetic ground truth; no data assets needed
```

## Use

```python
from aind_hcr_pairwise_unmixing_calibrated import pipeline

res = pipeline.run_mouse(
    asset_dir="/root/capsule/data/HCR_790322_pairwise-unmixing_...",
    mouse_id="790322",
    rounds=["R2", "R3", "R4", "R5"],
    gene_maps={"R2": {"488": "Ndnf", ...}, ...},
    processed_root="/root/capsule/data",
    output_dir="/root/capsule/results",
)
res["cellxgene"]      # cells x round-channel-gene
res["decisions"]      # per direction: which rule fired, how many spots moved
```

Or from a capsule, using the entry point in `capsule_patch/`:

```bash
python run_capsule.py --mouse-id 790322
```

## Two traps worth knowing before you run it

**Laser power must come from each round's own `acquisition.json`.** It varies more than
you would expect — R5 561 nm is 25% in mouse 804363, 15% in 800995 and 30% in 788406,
and R5 488 nm spans 5% to 30% across the same three — and β scales with the
victim/source power ratio. A stored per-round power table in the project belongs to
804363 alone and is wrong for every other subject.

**There are often two processed assets per round**, and only one matches the spot set in
the pairwise-unmixing asset (for 800995 R5: 2,506,903 rows versus 2,435,209). Resolve it
from that round's `ds_config.json` → `dataset_folder`, never by timestamp. `run_round`
raises if the fg/bg join matches under 99% of spots, which is the symptom.

## Validation

Marker pairs that should not co-occur in the same cells, correlation across Gad2⁺ cells:

| pair | before unmixing | upstream pipeline | v5 |
|---|---|---|---|
| Sst–Cck (800995) | 0.712 | 0.711 | 0.214 |
| Sst–Cck (788406) | 0.717 | 0.560 | −0.345 |
| Npy–Pvalb (788406) | 0.787 | 0.509 | −0.334 |
| Sst–Vip (800995) | 0.441 | 0.394 | −0.317 |

Not everything improves: Cck–Vip regresses in both mice, a side effect of the added
Cck→Sst direction. See the open questions in the project summary.

## Layout

```
src/aind_hcr_pairwise_unmixing_calibrated/
    core.py       the algorithm - endmembers, beta, co-location, per-spot decisions
    control.py    single-dye control matrix (shared across mice) and laser power
    fgbg.py       raw foreground / local background join, and threshold helper
    pipeline.py   per-round and per-mouse drivers
capsule_patch/
    run_capsule.py    Code Ocean entry point
tests/
    test_core.py  synthetic ground truth - ghosts with known identity
```
