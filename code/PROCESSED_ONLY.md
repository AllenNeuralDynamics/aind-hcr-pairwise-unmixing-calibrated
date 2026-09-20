# Running from the processed assets alone

Status: **branch `processed-only`, scaffolding in place, not validated on real data.**

Goal: drop the `HCR_<mouse>_pairwise-unmixing_<date>` mount, so the capsule reads only
the per-round processed assets it already needs for `acquisition.json`.

## What the pairwise asset was providing

| input | status |
|---|---|
| `ds_config.json` → `GENE_DICT` | **Solved on `main`.** `GENE_DICT` is `manifest.gene_dict` flattened, and `manifest` is a byte-identical copy of the processed asset's own `processing_manifest.json`, which carries `round` too. `gene_map_from_manifests` reads it directly. |
| `mixed_spots_<R>.pkl` | **This branch.** A table of the same name exists in each processed asset, but it is not the same table. |
| *its unmixing results* | Never used. The capsule re-derives every spot decision; it reads none of `unmixed_spots_*`, `removed_spots_*`, `unmixed_cell_by_gene.csv`. |

## The two spot tables

Both are written from the same detections, by different code:

- **pairwise** — `AllenNeuralDynamics/hcr-pairwise-spot-unmixing`
- **processed** — `AllenNeuralDynamics/aind-spot-spectral-unmixing`, in
  `process_round.py` immediately after `calculate_intensities`

Reading that upstream source settles what the extra bytes are. `data_loader` builds
`spot_col_order = [spot_id, chan, chan_spot_id, cell_id, round, z, y, x, z_center,
y_center, x_center, dist, r]` plus `chan_<ch>_fg` and `chan_<ch>_bg` per channel, and
then `calculate_intensities` adds

```python
spots_df[f'chan_{channel}_intensity'] = (
    spots_df[f'chan_{channel}_fg'] - spots_df[f'chan_{channel}_bg'])
```

**Three float64 columns per channel: fg, bg, intensity.** The pairwise table keeps only
the difference — `fgbg.py`'s docstring has said so all along ("keeps only FG − BG"). That
is exactly the 24 bytes/row/channel measured from the pickle headers before any of this
code was read: 221 B/row for a 5-channel processed round against 110 for the pairwise
one, 149 against 99 for 2-channel R1. Two independent routes to the same answer.

### Consequence: the fg/bg join becomes unnecessary

`fgbg.attach_fg_bg` exists only to recover fg and bg after the pairwise step discarded
them, by matching spot coordinates against `image_spot_detection/channel_<ch>_stats/`.
It carries two documented traps (the wrong processed asset silently dropping ~3% of
spots; `chan_spot_id` not being a row index). The processed table needs none of it — the
columns are the values the intensity was computed *from*, not a reconstruction.

## What this branch adds

- `spots_io.py` — locates a round's spot table in either family and reports a schema.
  Detection is by **column content**, not by path, so a relocated or copied file is read
  correctly.
- `pipeline._native_fg_bg` — per-row fg/bg from `chan_<ch>_fg` / `_bg`, taking each row
  from its own channel. Refuses at <99% coverage rather than borrowing another channel's
  background.
- `run_round` skips the join when native columns are present, and says so in the log.
- `--spots-from {auto,pairwise,processed}`. **`auto` prefers pairwise**: every result to
  date came from that spot set, so switching must be a deliberate choice, not a
  consequence of which assets happen to be attached.
- `result["spot_tables"]` records, per round, which family and asset was read and how
  many rows it had — so a run's provenance is in the output rather than inferred.

Tested on synthetic tables of both schemas, including the controlled comparison: the
**same** spot set through both paths gives an identical cell × gene table, with fg/bg
present on the processed side and absent on the pairwise side. 85 tests pass.

## Where the missing 2–4% of spots went: ROI filtering

Resolved from `hcr-pairwise-spot-unmixing`'s own `code/DATA_FLOW.md`. Its per-round order
is:

```
Load cells from mixed CxG
  → ROI filtering   volume · soma classifier · edge · tile overlap
  → Load spots      FILTERED TO KEPT CELLS
  → ... → Save mixed_spots_RN.pkl
```

**Spots are only ever loaded for cells that survived ROI filtering.** So the pairwise
table is not a filtered copy of the processed one — it was built from a smaller cell set
to begin with, and the 2–4% gap is spots in ROI-rejected cells. It removes **cells**, not
just spots, so switching input can add rows to the cell × gene table and not only counts
within existing rows.

The other candidate, `filter_by_threshold` in `aind-spot-spectral-unmixing`, is **not**
it. That function drops nothing:

```python
spots_df['over_thresh'] = False
spots_df.loc[spots_df['spot_id'].isin(spots_over_thresh), 'over_thresh'] = True
return spots_df, spots_over_thresh
```

It annotates and returns the full frame — the same annotate-don't-delete pattern this
capsule uses, and a fourth channel-independent column the processed table carries.

### Should this capsule implement either filter? No — and that is the argument for switching

`filter_by_threshold` is an annotation, so there is nothing to implement.

ROI filtering is a real decision, and the case for leaving it out is that **this capsule
has been applying someone else's ROI rule invisibly.** The pairwise step's rule is volume
+ soma classifier + edge + tile overlap. The consensus protocol's rule is the
segmentation classifier's own argmax with coregistered-never-dropped and no-signature
rescues. Those are different rules, and the second is applied downstream by the cohort
run, which is where the ROI-shape-metrics asset is consumed.

So reading the processed table does not *add* an unfiltered input — it *removes* a hidden
filter, and leaves the ROI decision in one place, downstream, under a documented rule.
That is the same reasoning that keeps geometric QC annotated rather than applied here,
and it is consistent with keeping ROI quality metrics out of this capsule.

What it costs: the per-mouse `.h5ad` will carry cells the ROI classifier would reject,
and class / subclass / cluster labels are computed on that slightly wider population. The
cohort run drops them at its own ROI stage. Whether the wider population moves any
per-cell label materially is measurable — item 1 below.

## Validated on 800792 R2 (v0.5.1, `7e155a3`)

Both arms run, same mouse, same round, `--no-spots`. Each read exactly the row count
predicted from the pickle headers — 11,339,232 pairwise, 11,570,896 processed — so each
arm read what was intended.

**The unmixing is unchanged.** Per-channel spot change agrees to within 0.2 percentage
points across all five channels (Ndnf −2.09 / −2.03, Hpse −53.87 / −54.05, Pthlh −11.23
/ −11.18, Chat −31.40 / −31.35, Tac1 −4.12 / −4.17; pairwise / processed). Of 105,003
shared cells, 103,110 (**98.2%**) are identical across every gene, the largest count
change on any shared cell is **2**, and per-gene totals on shared cells move by −0.001%
to −0.053%. The residual is the expected one: extra spots from recovered cells change a
few crosstalk neighbourhoods, and the direction is consistently a small loss.

**The recovered cells are nearly empty.** 14,523 cells appear in the processed arm that
were absent from the pairwise arm — **13.8% more cells** — with median total counts of
**4** against **75** for shared cells, and 97.6% of them below the 100-count class floor
*on this single round*.

| | pairwise | processed |
|---|---|---|
| spots in | 11,339,232 | 11,570,896 (+2.0%) |
| cells out | 105,003 | 119,526 (+13.8%) |
| median total counts, shared cells | 75 | 75 |
| median total counts, added cells | — | 4 |

The two gaps differ by nearly an order of magnitude — 2% of spots but 14% of cells —
and that is the whole character of the ROI filter: it was removing many ROIs that
carried almost no transcripts. Anyone reasoning about its effect from the spot counts
alone (as an earlier version of this file did) will badly underestimate how many cells
it touched.

**Caveat on the 97.6%.** The 100-count floor applies to a cell's total across all 27
gene-rounds, and this was measured on 5. The added cells are ~19× dimmer than shared
cells per round, so most would still fall below the floor on a six-round run, but the
exact fraction — and therefore the number of recovered cells that actually receive class
and subclass labels — needs the full run. On R2 alone, 349 of the 14,523 clear the floor.

**Verdict: nothing is ruined.** Shared cells are effectively untouched; the cost is a
13.8% longer table whose added rows are overwhelmingly `low_counts`. That is the
annotate-don't-filter trade this capsule makes everywhere else — the rows are labelled
for what they are and a downstream consumer can drop them, rather than being removed
here by a rule nobody downstream can see.

## Decision: keep the recovered cells (2026-09-20)

The ROI-rejected cells stay in the table, labelled for what they are, and a downstream
consumer drops them. This is the same trade the capsule makes for geometric QC and for
intensity thresholding: annotate here, decide there. The published cell count for a
mouse rises by ~14%, almost all of it `low_counts` rows, and that is a sentence in the
resource paper rather than a surprise for whoever counts rows.

**Sequencing.** `--spots-from auto` still prefers the pairwise table. It flips to
processed once the six-round run confirms that class and subclass do not move on shared
cells — the wider population shifts the class mixture, and that is the one route by
which this change can alter a label on a cell that was already there. Until then the
default stays where every shipped result came from.

## What is NOT settled, and must be checked on real data

1. ~~How much does the cell × gene table move~~ — **answered on R2, above.** What
   remains is the six-round version of the same question: how many of the recovered
   cells clear the 100-count floor across all 27 gene-rounds and therefore receive
   class, subclass and cluster labels. On one round it is 349 of 14,523.
2. **Do the native fg/bg agree with what the join reconstructs?** The join was validated
   to r = 1.000000 against the pipeline's own subtracted value on 800995 R5. Run one
   round both ways and compare `fg` and `bg` per spot. If they disagree, the join's
   coordinate matching is picking a different stats generation, and that matters for
   results already produced.
3. **How much does the cell × gene table move**, on the same mouse, with the extra
   spots included. Compare against the registered `unmixed-calibrated` asset.
4. **`valid_spot` is lost.** The pairwise table carries it — `apply_qc_filters` keeps
   every row and annotates it from `dist < 1`, `r > 0.5`, `dist_r > 4`. The processed
   table has no such column. This capsule never read it (it annotates its own geometric
   QC from `dist` and `r`, which both tables carry), so nothing breaks; but note those
   cutoffs differ from the ones in `ds_config.json` for these mice (`CENT_CUTOFF` 1.25,
   `CORR_CUTOFF` 0.25, `DIST_CUTOFF` 1.0), which is worth understanding before anyone
   relies on either.
5. **Runtime.** Dropping the join should remove a large part of the 1,475 s of
   load/join/assembly measured on 800792, but the processed pickles are ~2× larger to
   read. Net effect unknown.

## Suggested first run

```bash
# same mouse, both families, spot tables only -- no metadata, no asset registration
python run_capsule.py --mouse-id 800792 --rounds R2 --spots-from pairwise  --output-dir /results/pw
python run_capsule.py --mouse-id 800792 --rounds R2 --spots-from processed --output-dir /results/pr
```

R2 is the smallest round (2.56 GB processed, 1.24 GB pairwise). Compare
`800792_cellxgene.csv` and the `fg`/`bg` columns of the two spot tables. Item 2 above is
the one to settle first: it is a statement about results already published, not just
about this branch.
