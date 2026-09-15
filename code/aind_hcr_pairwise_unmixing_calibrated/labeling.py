"""Class, subclass and cluster rules from the HCR consensus-clustering protocol.

Vendored from `hcr_pipeline.py` in the cohort capsule
(matchings/hcr-inhibitory-consensus-capsule, `code/PROTOCOL.md`) so this capsule and
the cohort run derive labels the same way. The published constants are copied, not
re-derived: they were chosen on the six-mouse cohort and changing them here would
silently fork the two pipelines.

The one rename from the source: `hcr_transform_2b` is `hcr_transform_p95` here, after
what it does rather than which variant it was during development.

WHAT IS NOT VENDORED
--------------------
The ROI quality stage (`hcr_metric_percentile`, `hcr_severity`, `hcr_roi_keep`) and
the depth proxy. Those need the per-mouse `HCR-ROI-label` asset and a cell z column,
neither of which this capsule mounts. The cohort run applies the ROI filter before
its class call, so its class labels are derived on a filtered cell set and this
capsule's are not -- see the README.

CLUSTERS HERE ARE PER MOUSE
---------------------------
`hcr_cluster_blocks` and `hcr_cluster_names` reproduce the cohort naming rules, but
the k-means fit driving them runs on one animal. Cluster identities therefore do not
correspond across mice; any across-mouse analysis needs the consensus clusters.
"""
import numpy as np
import pandas as pd

#: The fifteen genes the cohort clustering was run on. The rest of the panel --
#: including Gad2 and Slc17a7 -- is held out so it stays available as an independent
#: check on the resulting clusters.
HCR_PANEL_15 = ("Pvalb", "Sst", "Vip", "Lamp5", "Npy", "Ndnf", "Cck", "Crh",
                "Calb2", "Tac1", "Reln", "Pthlh", "Hpse", "Mme", "Chat")

HCR_SUBCLASS_MARKERS = ("Pvalb", "Sst", "Vip", "Lamp5")

HCR_COUNT_FLOOR = 100           # total counts below this -> "low_counts", not classified
HCR_SUBCLASS_COUNT_FLOOR = 20   # winning subclass marker below this -> "unassigned"
HCR_ENRICHMENT_FLOOR = 1.5      # plurality subclass below this enrichment -> "Other"
HCR_NAME_FLOOR = 0.5            # cluster mean in transform units to name a gene
HCR_SNCG_DOMINANCE = 1.5        # Cck must lead the runner-up by this to promote to Sncg


def hcr_transform_p95(counts, group_vec=None):
    """Per-gene 95th percentile within group, then per-cell total rescaled to the
    group's median total.

    Renamed from `hcr_transform_2b`. `group_vec` is the mouse in the cohort pipeline;
    this capsule runs one animal at a time, so it defaults to a single group and the
    percentile is taken over the whole table.

    Compute this ONCE per cell set before splitting further. Rescaling inside each
    subclass amplifies noise in that subclass's own marker column and pulls
    contaminant cells into small unstable clusters.
    """
    counts = np.asarray(counts, dtype=float)
    if group_vec is None:
        group_vec = np.zeros(len(counts), dtype=int)
    group_vec = np.asarray(group_vec)
    out = np.zeros(counts.shape, dtype=float)
    for g in np.unique(group_vec):
        s = group_vec == g
        r = counts[s]
        p = np.percentile(r, 95, axis=0)
        y = r / np.where(p > 0, p, 1)
        t = y.sum(1)
        t[t == 0] = 1
        out[s] = (y / t[:, None]) * np.median(t)
    return out


def hcr_class_call(n_inh_marker, n_exc_marker, total_counts,
                   count_floor=HCR_COUNT_FLOOR, keep_mask=None,
                   lo=0.10, hi=0.90, seed=0):
    """Two-component Gaussian mixture on log2((Gad2+1)/(Slc17a7+1)).

    Returns (klass, posterior, thresholds). klass is one of low_counts /
    excitatory / ambiguous / inhibitory. The mixture is fitted only on cells clearing
    `count_floor` total counts (and `keep_mask` where a caller supplies one), then
    every cell is scored against it.

    A mixture rather than a fixed count threshold on each marker: the boundary is
    placed by the data's own two modes, so it moves with a mouse's detection depth
    instead of being calibrated on one animal. Cells landing between the 0.10 and
    0.90 posterior gates are `ambiguous` rather than forced into a class -- expect
    around 1%, and the fraction does not fall with library size, so it is a real
    population rather than a thresholding artefact.
    """
    from sklearn.mixture import GaussianMixture
    lr = np.log2((np.asarray(n_inh_marker, float) + 1)
                 / (np.asarray(n_exc_marker, float) + 1))
    total_counts = np.asarray(total_counts)
    ok = total_counts >= count_floor
    if keep_mask is not None:
        ok = ok & np.asarray(keep_mask, bool)
    gm = GaussianMixture(2, random_state=seed, n_init=5).fit(lr[ok, None])
    order = np.argsort(gm.means_.ravel())
    p_inh = gm.predict_proba(lr[:, None])[:, order[1]]
    grid = np.linspace(lr.min(), lr.max(), 4000)
    pg = gm.predict_proba(grid[:, None])[:, order[1]]
    thr = (float(grid[np.argmin(np.abs(pg - lo))]),
           float(grid[np.argmin(np.abs(pg - hi))]))
    klass = np.where(total_counts < count_floor, "low_counts",
                     np.where(p_inh >= hi, "inhibitory",
                              np.where(p_inh <= lo, "excitatory", "ambiguous")))
    return klass, p_inh, thr


def hcr_subclass_argmax(marker_values, markers=None,
                        count_floor=HCR_SUBCLASS_COUNT_FLOOR):
    """Per-cell subclass = whichever subclass marker is largest, on RAW COUNTS.

    Do not pass p95-normalised values: that stage divides each gene by its own 95th
    percentile, which inflates dim markers (Lamp5 renders ~3x darker than Sst at
    equal raw counts) and moves the call. Do not use a nearest-centroid rule on the
    profile either -- that is biased by how many panel genes a cell type co-expresses.

    Cells whose winning marker carries fewer than `count_floor` counts are returned
    as "unassigned" rather than forced into a subclass.
    """
    if markers is None:
        markers = list(HCR_SUBCLASS_MARKERS)
    v = np.asarray(marker_values, dtype=float)
    lab = np.asarray(markers)[v.argmax(1)].astype(object)
    lab[v.max(1) < count_floor] = "unassigned"
    return lab


def hcr_sncg_cells(subclass, marker_values, sncg_values,
                   count_floor=HCR_SUBCLASS_COUNT_FLOOR, sncg_block="Sncg"):
    """Relabel `unassigned` cells with high Cck as Sncg, in the PER-CELL call.

    Sncg is the subclass defined by what it lacks: high Cck and no convincing
    Pvalb/Sst/Vip/Lamp5. So a cell is Sncg when Cck clears `count_floor` and every
    one of the four markers falls below it -- the cell has a positive signal and
    nothing else claims it.

    This runs as part of the per-cell subclass call rather than only at the cluster
    level. Deciding Sncg afterwards leaves the per-cell label contradicting the
    cluster: on 800792 the 387 cells of the Cck cluster have a median Cck of 224
    counts against marker medians of 8-31, yet the four-way argmax put 337 of them in
    Pvalb/Sst/Vip/Lamp5 on marker counts barely over the floor.

    Deliberately NOT a five-way argmax with Cck as a fifth candidate: Cck is broadly
    expressed rather than subclass-specific, and letting it compete directly wins
    1,387 cells on 800792 -- 1,124 of them taken from the four subclasses. This rule
    draws only from cells no marker claimed (263 cells, 2.4% of inhibitory).

    A DIVERGENCE from the cohort protocol, whose per-cell call is four-way with Sncg
    applied only to clusters. Cells labelled Sncg here are `unassigned` there.
    """
    v = np.asarray(marker_values, dtype=float)
    cck = np.asarray(sncg_values, dtype=float)
    out = np.asarray(subclass, dtype=object).copy()
    out[(v.max(1) < count_floor) & (cck >= count_floor)] = sncg_block
    return out


def hcr_cluster_blocks(cluster_means, per_cell_subclass, labels,
                       enrichment_floor=HCR_ENRICHMENT_FLOOR,
                       name_floor=HCR_NAME_FLOOR,
                       dominance=HCR_SNCG_DOMINANCE,
                       markers=None, sncg_gene="Cck", sncg_block="Sncg"):
    """Assign each cluster a subclass block from the per-cell subclass calls.

    Plurality subclass, required to be `enrichment_floor` times its share of the
    background composition; clusters below the floor become `Other`. Enrichment
    rather than raw purity because a modest share of a rare subclass is strong
    concentration while the same share of a common one is none. A cluster whose
    plurality call is `unassigned` is `Other` -- `unassigned` is not a block.

    Then the Sncg promotion. Sncg is defined by high Cck with NO other subclass
    marker, so Cck must be the highest gene in the cluster profile outright --
    subclass markers included -- above `name_floor` and leading the runner-up by
    `dominance`.

    Ranking Cck against only the non-marker genes, as the cohort's
    `hcr_sncg_block` does, is wrong here: dropping Pvalb/Sst/Vip/Lamp5 before
    ranking hides exactly the evidence that disqualifies a cluster. On 800792 that
    promoted a 627-cell cluster whose Sst mean was 0.787 against Cck's 0.551 (raw
    medians 444 and 102 counts) -- an Sst cluster relabelled Sncg because Sst had
    been removed from the comparison. Requiring Cck to lead everything leaves that
    cluster in Sst and still promotes the genuine Cck cluster (Cck 1.371, Sst
    0.152). The cohort run was not bitten by this because its k = 18 over six mice
    produced no such cluster; the rule was always weaker than its intent.

    The dominance clause is load-bearing -- without it a Pvalb/Mme cluster is
    promoted on a 0.027 tie. This rule is post-hoc; it was written after inspecting
    which clusters it needed to catch, and should be reported as such.

    Returns (blocks dict, per-cluster diagnostics DataFrame).
    """
    if markers is None:
        markers = list(HCR_SUBCLASS_MARKERS)
    sub = pd.Series(np.asarray(per_cell_subclass, dtype=object))
    labels = np.asarray(labels)
    bg = sub.value_counts(normalize=True)

    blocks, rows = {}, []
    # `unassigned` is not a block, and neither is a plurality of Sncg: Sncg means
    # Cck-defined, and the Cck test below is the only route to it. Without this, Sncg
    # being a rare per-cell label (2.4% of inhibitory on 800792) makes the enrichment
    # floor trivial to clear -- a 192-cell Crh cluster reached 14.5x enrichment on a
    # 35% Sncg plurality while its own Cck median was 40 counts against the genuine
    # Cck cluster's 224. Its top gene is Crh; it is not an Sncg cluster.
    not_a_block = ("unassigned", sncg_block)
    for c in cluster_means.index:
        m = labels == c
        vc = sub[m].value_counts(normalize=True)
        plur, purity = vc.idxmax(), float(vc.max())
        enrich = float(purity / bg[plur]) if bg.get(plur, 0) > 0 else np.nan
        block = plur if (plur not in not_a_block and enrich >= enrichment_floor) else "Other"
        rows.append(dict(cluster=c, plurality=plur, purity=purity,
                         enrichment=enrich, block_before_sncg=block, n=int(m.sum())))
        blocks[c] = block

    # Sncg promotion, applied after every block is set so it can override one. The
    # full profile is ranked -- subclass markers are NOT dropped, so a cluster with a
    # stronger Sst/Pvalb/Vip/Lamp5 signal than Cck cannot be promoted.
    sncg_diag = {}
    for c in cluster_means.index:
        profile = cluster_means.loc[c]
        if len(profile) < 2 or sncg_gene not in profile.index:
            continue
        ranked = profile.sort_values(ascending=False)
        top, lead, runner = ranked.index[0], float(ranked.iloc[0]), float(ranked.iloc[1])
        promote = (top == sncg_gene and lead > name_floor
                   and lead >= dominance * max(runner, 1e-9))
        sncg_diag[c] = dict(top_gene=top, top_value=lead,
                            runner_up=ranked.index[1], runner_up_value=runner,
                            sncg_value=float(profile[sncg_gene]), promoted=promote)
        if promote:
            blocks[c] = sncg_block

    diag = pd.DataFrame(rows).set_index("cluster")
    diag["block"] = [blocks[c] for c in diag.index]
    for col, key in [("sncg_top_gene", "top_gene"), ("sncg_top_value", "top_value"),
                     ("sncg_cck_value", "sncg_value"), ("sncg_promoted", "promoted")]:
        diag[col] = [sncg_diag.get(c, {}).get(key) for c in diag.index]
    return blocks, diag


def hcr_cluster_names(cluster_means, blocks, name_floor=HCR_NAME_FLOOR,
                      n_markers=3, order_by=None, markers=None):
    """Name clusters `<Block>-<n> (GeneA/GeneB/GeneC)`.

    Genes are those whose cluster mean exceeds `name_floor` in transform units. The
    ABSOLUTE level, not the deviation across clusters: a gene can deviate strongly
    and still be low everywhere, which produces a name that reads as a marker for
    something the cluster barely expresses.

    ALL FOUR subclass markers are excluded from the gene list, not just the block's
    own. The block prefix already carries the subclass, and naming a cluster after a
    different subclass's marker asserts a contradiction: `Sncg-1 (Sst/Cck)` reads as
    an Sncg cluster whose strongest gene belongs to another subclass. Where that
    happens the block call is what needs fixing, and a name that hides it is worse
    than one that omits a gene.

    Numbering runs within a block. `order_by` is an optional Series over the cluster
    index giving the within-block sort key (the cohort run uses median depth; this
    capsule has no depth, so it passes descending cluster size).

    Returns (names dict cluster -> label, ordered list of cluster ids).
    """
    if markers is None:
        markers = list(HCR_SUBCLASS_MARKERS)
    by_block = {}
    for c in cluster_means.index:
        by_block.setdefault(blocks[c], []).append(c)

    names, ordered = {}, []
    for block in sorted(by_block, key=lambda b: (b in ("Other", "unassigned"), b)):
        members = by_block[block]
        if order_by is not None:
            members = sorted(members, key=lambda c: order_by[c])
        for i, c in enumerate(members, start=1):
            profile = cluster_means.loc[c].drop(labels=markers, errors="ignore")
            top = [g for g, v in profile.sort_values(ascending=False).items()
                   if v > name_floor][:n_markers]
            # `Sst-2  Reln/Cck`, no brackets -- the cohort figures' format.
            names[c] = f"{block}-{i}" + (f"  {'/'.join(map(str, top))}" if top else "")
            ordered.append(c)
    return names, ordered
