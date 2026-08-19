# Registering the capsule result as a Code Ocean data asset

## Can the capsule do it itself?

**No — not from inside the run.** Code Ocean uploads `/results` to S3 only *after* the
run script exits. While `run_capsule.py` is executing there is no S3 location to point a
data asset at, so a `create_data_asset` call made during the run has nothing to register.
(This is the same ordering noted in `pipeline.py`'s `_PARQUET_OPTS` comment: the upload is
wall-clock the run's own timer never sees.)

The capsule also has no credentials: the Docker image installs `anndata`, `matplotlib`,
`pyarrow` and `scikit-learn`, with no Code Ocean client, and no API token is present in
the run environment.

## What this patch does instead

Splits the job at the boundary where it naturally falls.

1. **The capsule writes `results/asset_manifest.json`** — the exact name, description,
   tags and custom metadata the asset should carry. The capsule is the only place that
   knows what was mounted, which rounds ran, and how many cells came out, so it is the
   right place to determine these.
2. **`tools/register_result_asset.py` creates the asset**, run after the computation
   finishes. It reads the manifest from the completed computation and POSTs
   `/api/v1/data_assets` with `source.computation.id` — no values re-derived by hand.

```bash
export CODEOCEAN_DOMAIN=https://codeocean.allenneuraldynamics.org
export CODEOCEAN_TOKEN=...        # needs data-asset create scope
python tools/register_result_asset.py 332a1d1e-8bc6-4ff2-aa4c-a6136937f971 --dry-run
python tools/register_result_asset.py 332a1d1e-8bc6-4ff2-aa4c-a6136937f971
```

An alternative worth knowing about: a **Code Ocean pipeline** can capture a process's
results as an asset as part of the pipeline definition, without any API call. If this
capsule becomes one step of a pipeline, use that instead of this script.

## Will it run automatically?

Nothing inside a Code Ocean run can register its own result, for the ordering reason
above. But you don't have to run anything by hand either — there are both manual and
automatic modes.

Every mode is **idempotent**: the manifest name embeds the run's own UTC timestamp, so it
is unique per run and doubles as the key. A run whose asset already exists is skipped, not
duplicated. `--dry-run` prints the plan without creating anything.

```bash
export CODEOCEAN_DOMAIN=https://codeocean.allenneuraldynamics.org
export CODEOCEAN_TOKEN=...            # needs data-asset create scope
export CODEOCEAN_CAPSULE_ID=f8032cb6-1ee2-4273-a847-800079f9177b
```

### Manual

```bash
python tools/register_result_asset.py --list        # what's ready? register nothing
python tools/register_result_asset.py --latest      # register the newest ready run
python tools/register_result_asset.py <computation_id>   # register one specific run
```

`--list` is the one to start with — it shows every computation on the capsule with a
status and a reason, so you can see what a sweep would do before letting one loose. For
the normal case of "I just ran the capsule, register that result", `--latest` needs no
computation id.

### Automatic

```bash
python tools/register_result_asset.py --watch                 # one pass, then exit
python tools/register_result_asset.py --watch --interval 600  # keep polling
```

Put the single-pass form in cron, a scheduled capsule, or a GitHub Actions schedule.

### Code Ocean pipeline (the structural alternative)

A pipeline can capture a process's results as a data asset as part of the pipeline
definition — genuinely automatic, no external credentials, no polling, and the asset
appears as soon as the upload completes. If this capsule becomes one step of a pipeline,
that is strictly better than either mode above and this script becomes unnecessary.

### What gets skipped, and why

| Condition | Behaviour |
|---|---|
| `exit_code != 0` | skipped — a failed run must not become an asset |
| `has_results` false | skipped — nothing to capture |
| no `asset_manifest.json` | skipped — run predates this feature, or used `--no-metadata` |
| asset with that name exists | skipped — already registered |

On the capsule's 8 computations as of 2026-08-19, a sweep skips 5 before making any
create call: four exited non-zero and one produced no results.

## Asset name

```
HCR_782149_unmixed-calibrated_2026-08-19_07-23-53
HCR_<mouse>_<process slug>_<YYYY-MM-DD>_<HH-MM-SS>      (UTC)
```

Keyed on the **mouse**, not on one parent session. The existing
`metadata.derived_data_description()` names the asset after a single processed session
(`HCR_782149_2025-11-05..._processed_..._unmixed-calibrated_<ts>`), which is misleading
here: the run consumes every round of the mouse, so naming it after the first mounted
session asserts a parentage that is only one-fifth true. `data_description.json` keeps
naming its `input_data_name` parent — that field is *about* the parent and is correct —
but the asset name should not.

**Also change `metadata.py` so `data_description.json`'s `name` matches the asset name.**
Otherwise the asset record and the metadata inside it disagree, which is what happened
with `HCR_782149_cell_gene_table_2026-08-19`: anything reading AIND metadata
programmatically (docDB indexing, provenance walkers) sees two different names for one
asset.

## Description

Generated from the mounted asset folders, since Code Ocean mounts every attached asset as
a directory under `/root/capsule/data` — the listing *is* the attached-asset list. Inputs
are split into unmixing input, processed assets, raw acquisition assets, and a fourth
group for assets mounted but belonging to a different mouse. That last group matters: the
2026-08-19 run had 13 assets for 800995 attached while processing 782149. They
contributed nothing but are permanently recorded in the computation's provenance, so the
description should say so explicitly rather than leave a reader to assume the result
derives from both animals.

Example for the 2026-08-19 run:

```
Spectrally unmixed spot tables and cell x gene table for mouse 782149, rounds R1, R2, R3, R4, R5.
Cell x gene table: 25,860 cells x 22 genes.
Produced by aind-hcr-pairwise-unmixing-calibrated.

INPUT DATA ASSETS
Unmixing input (mixed spot tables) (1):
  - HCR_782149_pairwise-unmixing_2026-07-14_18-11-49
Processed assets (acquisition.json, image_spot_detection fg/bg) (5):
  - HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29
  ... 4 more
Raw acquisition assets (5):
  - HCR_782149_2025-11-05_13-00-00
  ... 4 more

Also mounted but NOT used by this run (different mouse) (13):
  - HCR_800995_2026-03-12_13-00-00
  ... 12 more
```

## How the run writes the manifest

`run_capsule.py` calls `manifest.write_manifest()` after `pipeline.run_mouse()` returns,
under the same `--no-metadata` switch as the other metadata, and prints the name plus a
warning when assets for other mice were mounted.

`pipeline.run_mouse()` stamps one `creation_time` and passes it to both
`metadata.derived_data_description()` and the manifest, so the asset name and the
timestamp inside `data_description.json` agree — deriving it independently in each place
would make them differ by however long post-processing took.

## One unrelated fix worth doing in the same pass

**`--no-spots` (done in this change).** The five per-round spot parquets are 2,032 MB of
the 2,101 MB asset (96.7%); the cell-by-gene table the asset is named for is 1.7 MB. They
had no flag alongside `--no-plots` / `--no-anndata` / `--no-metadata`. There is now
`--no-spots`, but the tables remain **on by default**: they are the only record of the
per-spot decisions and the cell x gene table cannot be rebuilt without them, so the flag is
for the case where the cell x gene table is genuinely all that is wanted. When it is
passed, `processing.json` records no spot outputs rather than naming absent files.

**Reconsider the parent selection.** `pipeline.py` takes the first `source_dirs` entry
that has a `data_description.json`, which is an arbitrary one of the five mounted
sessions. Whichever session should be the recorded parent, choose it deliberately —
perhaps the earliest by acquisition date, or an explicit `--parent-asset` argument.
