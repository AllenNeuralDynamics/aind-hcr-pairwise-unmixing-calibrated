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

## What is NOT settled, and must be checked on real data

1. **The processed table holds 2–4% more spots** (R1 58,616,632 vs 56,306,976; R2–R6
   ~98%). This is the whole reason the switch is not a drop-in: more input spots means a
   different cell × gene table. Which spot set is *correct* is a question about the
   upstream filter — `process_round.py` writes `mixed_spots` before its own threshold
   filter, and `filter_by_threshold` runs after — not a question about this code.
2. **Do the native fg/bg agree with what the join reconstructs?** The join was validated
   to r = 1.000000 against the pipeline's own subtracted value on 800995 R5. Run one
   round both ways and compare `fg` and `bg` per spot. If they disagree, the join's
   coordinate matching is picking a different stats generation, and that matters for
   results already produced.
3. **How much does the cell × gene table move**, on the same mouse, with the extra
   spots included. Compare against the registered `unmixed-calibrated` asset.
4. **`valid_spot`** — `hcr-pairwise-spot-unmixing` filters on it when present. Check
   whether the processed table carries it and whether it accounts for the row gap.
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
