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
changes six things, each because a measurement said so. The numbered rationale for each
is in the section that follows.

| change | why |
|---|---|
| geometric QC **not applied**; flags passed through | The original computes three gates — `dist < 1.25` (spot centre vs. its fitted centroid, in voxels), `r > 0.25` (correlation to the fitted PSF), `dist_r > 1.0` — into a `valid_spot` flag and applies it at cell × gene construction. The calibrated version applies **none** of them: `dist` and `r` are written through as raw columns and the decision is made explicitly downstream. Audited over 66,059,174 spots (both mice, R2–R5), `dist_r > 1.0` is a no-op by construction, and the other two flag mostly clumped or merged detections rather than junk — real signal a single-spot model fits badly. **Consequence:** on 800995 R5 this leaves 2,083,186 extra spots (12.2%) in the delivered table relative to the original |
| dye lines from **spatially isolated** spots | the old "brightest in its own channel" purity rule is circular — in one mouse 93% of Sst spots peak in the Cck channel *because of* the bleed being measured. The isolated-spot estimate lands within 1° of what the single-dye control independently predicts |
| bleed magnitude and tolerance **measured per direction** | the inherited 1.5× tolerance had no derivation. The bleed ratio is not constant: flat for bright spots, rising for dim ones (a background floor). The calibrated version fits on the bright quartile of isolated spots and measures the tolerance as an upper percentile of the same labelled set |
| allowlist is **control-derived and bidirectional** | apparent bleed between distant channels is co-expression, not leakage — co-expressed genes are deliberately placed in non-neighbouring channels precisely because bleed there is negligible. But a pair that bleeds one way bleeds both, and the control's asymmetry does not hold in tissue (Vip/Sst: 7.6× predicted, 1.5–3.7× measured) |
| deletion needs spatial **and** magnitude **and** spectral evidence; undecidable spots flagged, not deleted | *Is co-location meaningful, or would spots land on top of each other anyway at this density?* Test: shift every source spot 24 µm sideways and re-run the search. On directions the control certifies as bleeding, real co-location runs **212–870× above** that shifted control, so a partner is genuine evidence rather than crowding. But the brightness test has a limit: a source spot dim enough that its *predicted* bleed (β × source) falls at or below the victim channel's own noise floor — taken as the 10th percentile of that channel's intensities — gives the test no dynamic range. Such spots are marked `v3_ambiguous` and kept, rather than silently deleted or silently retained |
| output **unfiltered**, carrying raw fg and local bg | the delivered table keeps only FG−BG, which cannot tell 300-over-100 from 300-over-900. Retaining `fg`, `bg` and their difference in the result allows intensity thresholding prior to cell × gene construction if desired |

## What β is

β is a **bleed fraction**. For an ordered pair of channels it answers: how much intensity
does one dye deposit in another channel, relative to how much it deposits in its own?

> **β(source → victim) = (that dye's intensity in the victim channel) ÷ (its intensity in
> its own channel)**

Both readings come from the **same spot** — every detected spot carries an intensity in all
five channels at its own location, so a single Sst transcript might read
`[12, 30, 210, 240, 18]`: brightest in its own channel (594) but with a substantial 561
reading that is pure bleed. β is the typical ratio of those two numbers, so it is a property
of the dye pair and the imaging conditions, not of any individual spot.

![What beta is and how it is measured](docs/beta_explainer.png)

*Sst(594) → Cck(561) in mouse 800995 round 5.* **A** one real spot's five readings: 496 in
Cck's channel with no Cck molecule present, 300 in its own. **B** β is that ratio, 1.65 for
this spot. **C** the estimation pool: 1,981,893 Sst spots → 168,585 spatially isolated →
42,146 in the brightest quartile. **D** three independent routes converge — control 0.467,
control × power ratio 1.401, measured on isolated spots 1.354. **E** the test in the
intensity plane; gold is pure bleed, blue is genuine Cck, dashed is the deletion boundary.
**F** the tolerance is measured per direction and per mouse, not assumed.

A β above 1 means the bleed reads *brighter* than the source's own signal, which happens
when the victim channel is imaged at higher power — here 561 at 15% against 594 at 5%. The
calibrated version uses the measured value when available and the power-scaled control
prediction otherwise.

β is used in exactly one place, the magnitude test. A victim spot is consistent with pure
bleed when

> `victim intensity ≤ tolerance × β × source intensity`

### The tolerance

β is a *median* — the typical bleed ratio — but individual spots scatter around it, so a
strict `victim ≤ β × source` test would reject half of all true bleed. The tolerance is that
allowance, and it is **measured, not chosen**: spatially isolated source spots are a
labelled ground-truth set (their victim-channel reading is pure bleed by construction), so
the tolerance is simply an upper percentile of observed ÷ predicted on that set.

The percentile is a recall/precision dial, measured here for Sst→Cck in 800995:

| percentile | tolerance | catches this much true bleed | sweeps in this much genuine Cck |
|---|---|---|---|
| p75 | 1.43 | 75% | 1.6% |
| **p90** (default) | **2.77** | **90%** | **4.6%** |
| p95 | 3.65 | 95% | 7.3% |
| p99 | 5.93 | 99% | 23.1% |

p99 is where it breaks down — catching the last 1% of bleed costs a quarter of the genuine
signal.

**It is not one number for all channels.** Measured across the ten allowlisted directions it
spans **1.98 to 5.29**, a 2.7× range, and the same direction differs between mice (Npy→Pvalb:
2.12 in 800995, 5.29 in 788406). That spread is the argument against the inherited constant
of 1.5, which sits below every measured value. Per-direction values are in
`beta_tolerance_by_direction.csv`.

A `TOL_FLOOR` of 1.15 guards one degenerate case: if the isolated-spot ratio has almost no
spread the measured tolerance collapses toward 1.0 and the test becomes exact, which is
brittle. The floor never binds on real data — the lowest measured value is 1.98.

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
| **geometric QC** | `apply_qc_filters` sets a `valid_spot` flag from `dist < CENT_CUTOFF`, `r > CORR_CUTOFF`, `dist_r > DIST_CUTOFF`; rows are not dropped, and the flag **is applied** at cell × gene construction (`cell_by_gene_table.py:76`) | gates are **not computed and not applied**; `dist` and `r` pass through as raw columns for the downstream step to use. On 800995 R5 that is 2,083,186 spots (12.2%) present in the calibrated cell × gene table that the original would have excluded |
| **spots with no partner** | kept, untouched | reassigned when the spectral evidence is strong; this is the rare case |
| **undecidable spots** | no such category | flagged `v3_ambiguous` and kept when predicted bleed falls under the victim channel's noise floor |
| **output** | filtered table: deleted spots are dropped, `unmixed_chan` = original channel | every input spot survives, with `v3_action`, `v3_chan`, `decision_rule`, `beta_used`, `beta_source`, NNLS coefficients, QC flags, and raw `fg` / `bg` |
| **iteration** | `keep` mask mutated in place across pairs, so a spot removed by one pair is invisible to later pairs | single pass; each direction is evaluated against the full spot set |

> **Migration note.** Because the calibrated version does not apply the geometric gates,
> its cell × gene counts are not directly comparable to the original's: on 800995 R5,
> 2,083,186 spots (12.2% of the delivered table) are present that the original would have
> excluded, *in addition to* every difference the unmixing itself makes. When comparing
> tables from the two pipelines, either apply `dist < 1.25 & r > 0.25` to the calibrated
> output first, or expect a systematic offset of roughly this size.

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
smaller difference than it first appears — what the audit changed is which gates are
trusted. The three gates the capsule sets are:

| gate | what it measures | capsule value |
|---|---|---|
| `dist` | distance from the detected spot centre to its fitted centroid, in voxels | `< 1.25` |
| `r` | correlation between the spot and the fitted point-spread function | `> 0.25` |
| `dist_r` | ratio of a spot's distance to its *nearest* dye line over its *second-nearest* | `> 1.0` |

The audit covered every spot in both mice across rounds R2–R5 — 66,059,174 spots, 20
round × channel combinations — which is what makes the per-channel breakdown below
trustworthy rather than anecdotal.

`dist_r` is **a no-op at the capsule's setting**. It is a ratio of a smaller distance to a
larger one, so it is ≥ 1 for every spot by construction, and `> 1.0` therefore removes
0.00% in all four rounds. (The engine's own default is 4, which would remove 48–85%
depending on the round — a very different gate. The capsule overrides it to 1.0.)

`dist` and `r` do remove spots — 1.6–9.9% and 0.5–7.1% depending on channel. In **all 20**
round × channel combinations the spots they discard are *brighter* in their own channel than
the ones they keep (R2 Tac: median 133 vs 57; R3 Calb1: 160 vs 135). Two readings of that
are possible and they were tested against each other:

*Autofluorescence?* Bright non-transcript blobs would be bright in **every** channel, giving
an own-channel-to-other-channel ratio near 1. Measured on 800995 R5, the ratio *rises* when
the gate fails in four of five channels (638: 2.56 → 8.39; 514: 2.46 → 5.62), so the
discarded spots are **more** spectrally specific than the kept ones, not less. They do not
look like flat-spectrum autofluorescence.

*Saturated or merged spots?* Profiling `dist` and `r` against brightness within one channel
shows both degrade as spots get brighter — in 594/Sst the failure rate climbs from 0.3% in
the dimmest sixth to 17% around 377 counts. A tight cluster of transcripts fits a
single-spot PSF badly, so these gates are largely flagging **clumped or merged detections**:
real signal, but with a position and an amplitude that a single-spot model cannot represent.

Neither answer justifies discarding them silently, and neither justifies keeping them
unconditionally — a merged detection of three transcripts counted once undercounts, and its
centroid is unreliable. So the calibrated version computes nothing and applies nothing:
`dist` and `r` pass through as raw columns alongside `fg`, `bg` and `fg_over_bg`, and the
filtering decision is made explicitly at cell × gene construction. **This differs from the
original**, which applies its `valid_spot` flag there; see the note below on the size of
that difference.

**Co-location tested against a displaced null.** "The victim spot has a source spot on top
of it" is only evidence if spots don't land on top of each other by chance at these
densities — and these are dense samples (up to 13M spots in one channel). The test:
translate every source spot 24 µm along one axis and re-run the identical search. Any hits
now are pure chance. Measured on the four Sst/Cck/Vip directions in both mice, real
co-location runs **212–870× above** that shifted control (Sst→Vip in 800995: 8.9% of Vip
spots have a sub-voxel Sst partner, against 0.04% after displacement), so a partner is
genuine evidence.

The same test is a useful negative control. Cck→Vip — a distant pair, where the panel
places co-expressed genes precisely because bleed is negligible — comes in at only 28× and
34×, an order of magnitude below the real bleed directions. That is one of the two
discriminators behind excluding distant pairs from the allowlist.

**An explicit "cannot tell" category.** The brightness test asks whether a victim spot is
dim enough to be pure bleed from its partner. When the partner is itself dim, the predicted
bleed β × source can fall below the victim channel's own background — measured as the 10th
percentile of that channel's intensities — and the test has no dynamic range left. Rather
than deleting on a test that cannot discriminate, or silently keeping and pretending the
question was asked, those spots are flagged `v3_ambiguous` and passed through. Note the
criterion is on the *source* side: a plain victim-brightness cut would be wrong, because
real Cck spots are *dimmer* than true bleed into that channel (median 113 vs 180).

**Filtered output → annotated output.** A deleted row cannot be audited or reversed.
Downstream steps need to make their own thresholding decisions, which requires the raw
foreground and local background the original discards in favour of their difference.

### Still to decide: the delivered cell × gene table

The capsule should emit a cell × gene table alongside the spot-level results, which means
choosing filters — geometric and intensity — rather than leaving them implicit. Nothing is
settled here yet; what the audit has established so far:

| candidate filter | measured effect | status |
|---|---|---|
| `dist < 1.25 & r > 0.25` (the original's) | removes 12.2% on 800995 R5; the removed spots are brighter and more spectrally specific, consistent with clumped/merged detections | not applied; flags available |
| `dist_r > 1.0` | removes 0.00% by construction | dropped |
| `FG ≥ median(BG) + 2·MAD(BG)` per channel | removes 8.7–18.9% (800995) and 6.1–9.6% (788406); improves 9 of 16 marker-pair correlations but regresses Lamp5–Calb2 by +0.27 in **both** mice and costs 1,802 Gad2⁺ cells in one mouse against 18 in the other | tested, not adopted |
| per-spot `FG > 1.5 × BG` | removes 22–42%; the most even across channels because it means the same thing in each | untested end-to-end |
| `median(FG)` or `median(FG) − k·SD(FG)` | the first is a 50% rank cut by construction; the second is inert (negative in all five channels) | rejected |

The columns needed for any of these — `dist`, `r`, `fg`, `bg`, `fg_over_bg` — are all in the
spot-level output, so the choice can be made and revised without re-running the unmixer.

### Measured effect

Each value is a **Pearson correlation between two genes' per-cell transcript counts**
(log1p-transformed), computed across Gad2⁺ cells in one mouse, rounds R2–R5. The gene
pairs are markers of *different* interneuron subclasses, so they should rarely appear in
the same cell: a correlation near zero or below is the expected biology, and a strongly
positive one means one gene's spots are being counted inside the other's cells — the
signature of unresolved crosstalk. Lower is therefore better, and the three columns are
the same cells scored three ways.

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

## Two things worth knowing before you run it

**Laser power must come from each round's own `acquisition.json`.** It varies more than
you would expect — R5 561 nm is 25% in mouse 804363, 15% in 800995 and 30% in 788406,
and R5 488 nm spans 5% to 30% across the same three — and β scales with the
victim/source power ratio. A stored per-round power table in the project belongs to
804363 alone and is wrong for every other subject.

**Choosing the processed asset is yours to make.** A mouse usually has several processed
assets, and they are not interchangeable for the fg/bg join: the per-channel
spot-detection tables must correspond to the same detections as the spot table being
unmixed. Resolution order:

1. `--processed-folder <name>` — an explicit choice, which overrides everything else.
2. `dataset_folder` from the round's `ds_config.json`, when that directory is present.
   This names the asset the pairwise-unmixing outputs were generated against, so it is
   the safe default.
3. Otherwise the **newest** processed asset for that mouse carrying `acquisition.json`.

Steps 2 and 3 are conveniences, not guarantees. If a mismatched asset slips through, the
fg/bg join is the tripwire: `run_round` raises when it matches under 99% of spots rather
than silently attaching the wrong intensities. Laser power is read from whichever asset
is chosen, so the choice matters even with `--no-fgbg`.

## Documentation

[`docs/unmixing_summary.pdf`](docs/unmixing_summary.pdf) — a 14-page plain-language
walkthrough of the whole problem and how this version addresses it, with 16 figures
embedded. Written for someone who has not followed the development: what ghost spots are,
why the earlier approaches fell short, what each step of the algorithm does and why, and
what remains open. Start here if you want the reasoning rather than the API.

[`docs/beta_explainer.png`](docs/beta_explainer.png) — the β figure from the section above,
standalone.

## Metadata (aind-data-schema)

The results directory is written as a proper derived asset, not a bare folder of CSVs:

- **Upstream schema files are carried forward unchanged.** `subject.json`,
  `acquisition.json`, `procedures.json`, `instrument.json`, `quality_control.json`,
  `rig.json`, `session.json` — these describe the subject and the acquisition, which
  unmixing does not alter, so copying them is correct. Whatever is absent upstream is
  skipped without complaint. Both the raw and the processed assets carry the full set,
  so pointing at either works.
- **`data_description.json` is written, not copied.** That file describes the *asset*,
  so copying the parent's would assert that our output is the parent. Following the
  convention the processed assets themselves use, the derived one gets
  `name = <parent>_unmixed-calibrated_<timestamp>`, `data_level = "derived"`, and
  `input_data_name = <parent>`, while inheriting subject, institution, modality,
  funding and licence unchanged. If no parent `data_description.json` is found, none is
  written — inventing those fields would be worse than omitting the file.
  `metadata.nd.json` is excluded for the same reason: it aggregates the others and would
  go stale immediately.
- **`processing.json` records this step.** One `DataProcess` named
  `Image spot spectral unmixing` — a term in the aind-data-schema controlled vocabulary
  — carrying the repository URL and version, input and output locations, the run
  parameters, and a note with the spot counts in and out. Any upstream
  `processing.json` found is extended rather than replaced, so the processing history
  accumulates.

Pass `--experimenter "Your Name"` to fill `processor_full_name`, or `--no-metadata` to
skip the whole step.

**Schema version.** These files are written as **1.1.4**, matching the `processing.json`
already present in the HCR processed assets. The installed `aind-data-schema` library is
2.x and its `Processing` model emits schema 2.3.0 with a different structure — a flat
`data_processes` list, `process_type`/`stage`/`code` fields, required `experimenters`.
Mixing both shapes in one asset would leave a file neither reader handles cleanly, so the
1.1.4 shape is written by hand and verified field-for-field against the real assets in
`tests/test_core.py`. If an upstream `processing.json` turns out to be 2.x-shaped it is
referenced in the note rather than downgraded. Migrating to 2.x is a deliberate future
step for the whole asset at once; `metadata.upgrade_note()` records what it involves,
and the 2.x construction has been verified to validate under aind-data-schema 2.8.1.

## Annotated cell × gene table (AnnData)

Alongside the spot tables, the capsule writes `<mouse>_cellxgene_annotated.h5ad`:

| slot | contents |
|---|---|
| `X` | **raw** transcript counts, all cells × all genes |
| `layers["normalized"]` | the matrix clustering ran on: per-cell counts ÷ cell total × median cell total, then each gene ÷ its 95th percentile, clipped to 1 |
| `obs` | `class`, `subclass`, `cluster`, `cluster_id`, marker counts, `total_counts`, `n_genes` |
| `var` | `round`, `channel`, `gene` per column |
| `uns["unmixing"]` | every parameter the labels were computed with |

Both matrices are stored rather than one: a reader who wants counts should not have to
invert a normalization, and a reader reproducing the clustering should not have to guess
how it was done.

**Clusters** come from k-means run **separately on excitatory and inhibitory cells**,
then merged — one joint clustering spends most of its clusters separating the two
classes instead of resolving structure within them. Names are subclass-first with the
most *enriched* genes appended (enrichment, not raw level, so an abundant gene does not
name every cluster): `Pvalb-2 (Pvalb/Mme/Pthlh)`, `Lamp5-1 (Lamp5/Reln/Hpse)`,
`Exc-1 (...)`. A cluster with no gene above the enrichment floor gets no suffix rather
than an invented one.

**Class labels require R1.** The only excitatory marker in the panel is `Slc17a7`, imaged
in **R1** (`488=GFP, 561=Slc17a7`); `Gad2` is in R4. A cell positive for exactly one
marker gets that class; positive for both, or neither, gets `unassigned` — those are the
cells worth inspecting, and forcing them into a class would hide them. **Run R1 and R4
together to get class labels**; without R1 nothing is called excitatory, because "not
Gad2⁺" would sweep in low-quality cells, mis-segmented cells and non-neuronal cells
alike. `uns` records which markers were available.

R1 is cheap to include: it has only two channels, 488 and 561, which are two apart, so
the control allowlist admits no direction between them and its unmixing is close to a
no-op. It is there for the class marker, not for crosstalk removal.

Pass `--no-anndata` to skip this step.

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
