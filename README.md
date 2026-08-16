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
or the capsule that uses it. Both remain untouched; run the calibrated version from its
own capsule against the same data assets.

## What the calibrated version does differently

The upstream pairwise stage removes crosstalk by finding same-cell overlapping spots in
adjacent channel pairs and deleting whichever fits its own dye line worse. The calibrated
version keeps that core idea — a ghost sits on top of its source and has a predictable brightness — and
changes six things, each because a measurement said so.

| change | why |
|---|---|
| takes the **full** spot table; geometric QC annotated, not applied | measured over 66M spots in 4 rounds: one of the three filters cannot remove anything by construction, and the other two discard spots no dirtier than the ones they keep, and brighter |
| dye lines from **spatially isolated** spots | the old "brightest in its own channel" purity rule is circular — in one mouse 93% of Sst spots peak in the Cck channel *because of* the bleed being measured. The isolated-spot estimate lands within 1° of what the single-dye control independently predicts |
| bleed magnitude and tolerance **measured per direction** | the inherited 1.5× tolerance had no derivation. The bleed ratio is not constant: flat for bright spots, rising for dim ones (a background floor). The calibrated version fits on the bright quartile of isolated spots and measures the tolerance as an upper percentile of the same labelled set |
| allowlist is **control-derived and bidirectional** | apparent bleed between distant channels is co-expression, not leakage — co-expressed genes are deliberately placed in non-neighbouring channels precisely because bleed there is negligible. But a pair that bleeds one way bleeds both, and the control's asymmetry does not hold in tissue (Vip/Sst: 7.6× predicted, 1.5–3.7× measured) |
| deletion needs spatial **and** magnitude **and** spectral evidence; undecidable spots flagged, not deleted | co-location runs 200–600× above a displaced null, so it is strong evidence; but a spot whose predicted bleed sits under the victim channel's noise floor cannot be judged either way |
| output **unfiltered**, carrying raw fg and local bg | the delivered table keeps only FG−BG, which cannot tell 300-over-100 from 300-over-900. Thresholding belongs at cell × gene construction |

## How this differs from the original pairwise unmixing

The original is `mattjdavis/aind-spot-spectral-unmixing @ hack-for-capsule-mjd`
(`spot_analysis/unmixer.py`, `_unmix_spots_pairwise` / `_filter_pairwise_crosstalk`), as
run by `AllenNeuralDynamics/hcr-pairwise-spot-unmixing`. It is unmodified and still works
exactly as before; this table is a functional comparison, not a migration notice.

### Side by side

| | original pairwise | calibrated |
|---|---|---|
| **which pairs are checked** | a fixed adjacency list by channel count — for 5 channels `(488,514) (514,561) (561,594) (594,638) (488,594)`, hard-coded in `get_default_channel_pairs` | derived from the single-dye control: any ordered direction with β ≥ 0.05, plus the reverse of every such pair |
| **direction** | undirected — a pair is examined, and either member may be removed | directed — source → victim is decided by the control, and only the victim can be deleted |
| **dye line fit** | optimizer over the round's spot cloud (`ratio_calculator`), subset toward high-intensity spots in their own detection channel | component-wise median of spatially isolated spots (no other-channel spot within 1.5 µm), top 5% by own-channel brightness |
| **what decides deletion** | of two co-located spots, delete whichever has the larger distance to its own dye line — purely relative, no absolute criterion | four conditions, all required: spatial partner present, brightness consistent with measured β within a measured tolerance, five-channel fit favours the source, reverse direction does not explain it better |
| **bleed magnitude** | not used | measured per direction, per mouse, per round; control-predicted value used when the measurement is unavailable |
| **tolerance** | not applicable | measured as an upper percentile of observed ÷ predicted on isolated source spots (floor 1.15) |
| **spatial test** | `cKDTree.query_ball_point` with one isotropic radius (`min_dist`), on scaled coordinates | nearest-neighbour with an anisotropic box: 1.0 µm lateral, 1.3 µm axial, because a spot on a plane boundary can be binned one z-plane away |
| **same-cell requirement** | yes — overlaps in different cells are skipped (`unmixer.py:420`) | yes by default (`same_cell=True`), and switchable off per call. Kept because deleting across a segmentation boundary risks removing a real transcript assigned to a neighbouring cell |
| **laser power** | not used | read per round from that round's own `acquisition.json`; scales the control β |
| **geometric QC** | `apply_qc_filters` sets a `valid_spot` flag from `dist < CENT_CUTOFF`, `r > CORR_CUTOFF`, `dist_r > DIST_CUTOFF`; rows are not dropped, and the flag is applied later at cell × gene construction (`cell_by_gene_table.py:76`) | same structure — flags annotated, applied downstream — but `dist_r` is dropped from the gate set, since it is a ratio of the smallest to second-smallest dye-line distance and so is ≥ 1 by construction, making `dist_r > 1.0` a no-op |
| **spots with no partner** | kept, untouched | reassigned when the spectral evidence is strong; this is the rare case |
| **undecidable spots** | no such category | flagged `v3_ambiguous` and kept when predicted bleed falls under the victim channel's noise floor |
| **output** | filtered table: deleted spots are dropped, `unmixed_chan` = original channel | every input spot survives, with `v3_action`, `v3_chan`, `decision_rule`, `beta_used`, `beta_source`, NNLS coefficients, QC flags, and raw `fg` / `bg` |
| **iteration** | `keep` mask mutated in place across pairs, so a spot removed by one pair is invisible to later pairs | single pass; each direction is evaluated against the full spot set |

### Why each change was made

**Adjacency list → control-derived allowlist.** The fixed list includes `(561,594)` and
`(594,638)`, which do bleed, but also `(488,594)` — and treats all of them as equivalent.
Meanwhile an in-data measurement of every pair shows distant pairs with apparently huge
bleed ratios (up to 200× the control prediction). Those are **co-expressed genes**, placed
in non-neighbouring channels precisely because bleed there is negligible. Deleting on that
signal removes real biology. Two discriminators separate the causes: true bleed sits
sub-voxel on its source and scales with it; co-expression does neither.

**Undirected → directed.** "Delete whichever fits its own dye line worse" has no notion of
which dye is physically capable of contaminating which. When the two dye lines are close
together, that comparison is close to a coin flip.

**Relative fit → absolute, calibrated criterion.** The original never asks whether the
victim's brightness is *plausible as bleed*. A bright genuine spot that happens to sit near
a bright spot of a neighbouring gene can be deleted on a marginal difference in dye-line
distance. Requiring `victim ≤ tolerance × β × source`, with both β and tolerance measured,
makes deletion evidence-based rather than comparative.

**Optimizer fit → isolated-spot fit.** Subsetting toward high-intensity spots in their own
detection channel sounds like a purity filter but is circular: in one mouse 93% of Sst
spots are brightest in the Cck channel *because of* the bleed. The isolated-spot estimate
has no such feedback loop and lands within 1° of the control's independent prediction.

**Isotropic radius → anisotropic box.** Axial and lateral uncertainty differ. A single
radius either misses z-displaced ghosts or admits laterally distant chance co-locations.

**QC gates audited, one dropped.** Both versions annotate rather than drop, so this is a
smaller difference than it first appears. What the audit changed is the gate *set*:
measured across four rounds and 66M spots, `dist_r > 1.0` cannot remove anything, because
`dist_r` is the ratio of a spot's smallest to its second-smallest dye-line distance and is
therefore ≥ 1 by construction. The other two gates do remove spots, but those spots are no
dirtier on a crosstalk proxy than the ones kept, and are brighter — so the calibrated
version leaves the thresholds to the cell × gene step rather than treating them as
settled.

**Filtered output → annotated output.** A deleted row cannot be audited or reversed.
Downstream steps need to make their own thresholding decisions, which requires the raw
foreground and local background the original discards in favour of their difference.

### Measured effect

Marker pairs that should not co-occur in the same cells, correlation across Gad2⁺ cells,
two mice, rounds R2–R5:

| pair | no unmixing | original pairwise | calibrated |
|---|---|---|---|
| Sst–Cck (800995) | 0.712 | 0.711 | 0.214 |
| Sst–Cck (788406) | 0.717 | 0.560 | −0.345 |
| Npy–Pvalb (788406) | 0.787 | 0.509 | −0.334 |
| Npy–Pvalb (800995) | 0.292 | 0.132 | −0.290 |
| Sst–Vip (800995) | 0.441 | 0.394 | −0.317 |
| Lamp5–Calb2 (788406) | 0.617 | 0.456 | −0.340 |
| Calb1–Mme (800995) | 0.917 | 0.817 | 0.498 |

Cck–Vip is the exception and ends up worse than no unmixing at all: 0.393 → 0.470 in
800995 and 0.242 → 0.364 in 788406. The regression is specifically from admitting the
reverse directions — with the allowlist forward-only it reads 0.301 and −0.077, so the
bidirectional rule costs 0.17 and 0.44 on this pair while fixing Sst–Vip by ~0.52 on both
mice. The mechanism is the added Cck→Sst direction deleting Cck spots and re-correlating
Cck with Vip through shared depth loss. That direction has the highest sub-voxel
co-location rate of anything measured (0.585), which is suspicious rather than reassuring
given Cck is the densest channel, and it is the first thing to examine.


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

| pair | before unmixing | upstream pipeline | calibrated |
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
