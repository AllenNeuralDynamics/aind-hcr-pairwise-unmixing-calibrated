# Adopting the consensus capsule's current class and subclass rules

Status: **branch opened, nothing implemented.** Deliberately deferred until the cohort
has been reprocessed with the current code, so that reprocessing changes one thing —
the spot source — and not the labelling as well.

Target: `https://codeocean.allenneuraldynamics.org/capsule/5718104/tree`

## Why this is a divergence rather than a new feature

`labeling.py` was vendored from the cohort capsule's `hcr_pipeline.py` and states so in
its header: the published constants were copied, not re-derived. So the rules here were
the consensus rules *at the time of vendoring*. If the capsule's rules now differ, they
have moved since — which makes this a re-sync, and the first task is to read
`code/PROTOCOL.md` and `hcr_pipeline.py` at 5718104 and diff them against
`labeling.py` function by function, rather than writing new rules.

Functions in scope: `hcr_class_call`, `hcr_subclass_argmax`, and the constants
`HCR_COUNT_FLOOR`, `HCR_SUBCLASS_COUNT_FLOOR`, `HCR_ENRICHMENT_FLOOR`.

## The observation that motivated this

On 782149 (`HCR_782149_unmixed-calibrated_2026-09-21_16-45-08`), cells with a strong
inhibitory marker are being called excitatory:

| `max(Pvalb, Sst, Vip)` | cells | called excitatory |
|---|---|---|
| ≥ 20 | 5,701 | 3,292 (57.7%) |
| ≥ 50 | 2,724 | 885 (32.5%) |
| ≥ 100 | 2,096 | 483 (23.0%) |
| ≥ 200 | 1,359 | 222 (16.3%) |

Median profile of the 483 at the ≥100 threshold:

```
Pvalb 11 | Sst 21 | Vip 115 | Gad2 6 | Slc17a7 240
total counts 1,831   (all excitatory: 1,106)
```

**Vip drives it and Gad2 is absent.** These are not cells where Gad2 argues inhibitory
and Slc17a7 outvotes it — the class call is a Gad2/Slc17a7 ratio and Gad2 is at
background. Their total counts run high, which is consistent with generally bright
cells rather than specifically Vip-positive ones.

So the proposed override — *strong Pvalb/Sst/Vip ⇒ inhibitory regardless of Slc17a7* —
would flip these 483, and on current evidence it is not established whether that
corrects a rule error or imports bleed-through. **Unresolved.**

### The discriminating test, not yet run

Whether the Vip signal in these cells is accompanied by anything else inhibitory:

- If all of R5 is elevated in them (Vip's round on this mouse: 488 Npy, 514 Pvalb,
  561 Cck, 594 Sst, 638 Vip), it is a per-cell or per-round intensity artefact and the
  override would import it.
- If Vip alone is elevated, the override is justified.

An earlier attempt at this filtered `var["round"]` on `"5"` where the values are `"R5"`,
and silently returned an empty gene list. Redo it against the actual labels.

## What does NOT explain it

Applying the p95 transform before the class call — the change considered alongside this
— was tested on 782149 and moves the result the wrong way:

| | excitatory | inhibitory | ambiguous | low_counts |
|---|---|---|---|---|
| raw counts (current) | 24,399 | **3,001** | 752 | 7,192 |
| whole-table p95 | 21,828 | **1,111** | 5,213 | 7,192 |

The inhibitory count falls by 63%, 1,890 currently-inhibitory cells become ambiguous,
and **none of the 483 is rescued**. Dividing each gene by its own 95th percentile
compresses both markers toward the `+1` pseudocount, flattening the log-ratio: the
posterior gates move from `[-2.86, -1.38]` to `[-0.34, 0.47]`. The transform is a
display and clustering scale, not a classification input, and this is why.

For the record, where the transform IS applied today — all three scopes are unchanged
by this branch:

| consumer | input |
|---|---|
| `assign_class` | raw counts |
| `assign_subclass` | raw counts |
| `layers["normalized"]` | p95 over all cells and genes |
| `layers["normalized_within_class"]` | p95 within each class |
| `obsm["X_cluster"]` | p95 within each class, that class's clustering genes |

## Sequencing

1. Finish reprocessing the cohort on `processed-only` with the rules as they stand.
2. Diff 5718104's rules against `labeling.py`.
3. Run the R5 test above.
4. Only then change a rule — and re-run the cohort again, since every label moves.
