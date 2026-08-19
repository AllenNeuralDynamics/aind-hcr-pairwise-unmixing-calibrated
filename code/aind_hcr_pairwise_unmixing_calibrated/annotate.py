"""Annotated cell x gene table as AnnData.

Produces an .h5ad carrying every cell, the raw transcript counts, and three levels of
label: class (excitatory / inhibitory), subclass (Vip / Sst / Pvalb / Lamp5), and a
named cluster.

WHAT IS STORED WHERE
--------------------
    X                    RAW transcript counts. Integers, no transformation.
    layers["normalized"] the matrix clustering was actually run on: per-cell counts
                         divided by the cell total and rescaled to the median cell
                         total, then each gene divided by its 95th percentile and
                         clipped to 1.
    obs                  class, subclass, cluster, cluster_id, plus the marker counts
                         the class call was made from and total_counts / n_genes.
    var                  round, channel, gene for each column.
    uns                  the parameters every label was computed with.

Raw counts in X and the normalized matrix in a layer, rather than one or the other:
a reader who wants counts should not have to invert a normalization, and a reader who
wants to reproduce the clustering should not have to guess how it was normalized.
Both are cheap at this size.

CLASS LABELS NEED R1
--------------------
The only excitatory marker in the panel is Slc17a7, and it is imaged in R1
(488=GFP, 561=Slc17a7). Gad2 is in R4. When R1 is absent from the run there is no
positive excitatory marker and `class` is left as "unassigned" for non-inhibitory
cells rather than asserting excitatory from the absence of Gad2 -- low-quality cells,
badly segmented cells and non-neuronal cells would all land in that bucket. uns
records which markers were available.
"""
import numpy as np
import pandas as pd

#: Positive markers for the two classes, and the round each is imaged in.
CLASS_MARKERS = {"excitatory": "Slc17a7", "inhibitory": "Gad2"}

#: Any of these above MIN_CLASS_COUNTS makes a cell inhibitory. Gad2 alone is too
#: strict a gate: it under-calls badly (on 800995, 11,334 cells clear Gad2 but a third
#: of them also clear Slc17a7, and Gad2 itself is only moderately expressed in some
#: interneuron types). A cell strongly expressing any canonical interneuron marker is
#: inhibitory whether or not its Gad2 reading cleared the threshold.
#:
#: Npy is included and Lamp5 deliberately is NOT, though both are interneuron-associated.
#: Lamp5 is expressed in 45% of ALL cells here (median 84, against 10/5/8 for
#: Pvalb/Sst/Vip), so gating on it would admit most of the excitatory population; Npy is
#: properly sparse (median 0, 1.9% of cells above threshold). Lamp5 remains a SUBCLASS
#: gene -- it names a group once a cell is already inhibitory -- but it cannot admit one.
INHIBITORY_MARKERS = ("Gad2", "Pvalb", "Vip", "Sst", "Npy")

#: Genes barred from cluster marker names on top of the subclass genes. Round 6
#: (Sncg, Adra1b, Cnr1, Adra1a, Htr3a) is excluded by request: those genes are broadly
#: expressed across types here and crowd out the markers that actually distinguish a
#: cluster. They still participate in clustering and in the plotted matrix.
ROUND6_GENES = ("Sncg", "Adra1b", "Cnr1", "Adra1a", "Htr3a")

#: Barred from marker names. Gad2 and Slc17a7 DEFINE the class, so naming an inhibitory
#: cluster after Gad2 is circular and "Lamp5-1 (Slc17a7/...)" reads as a contradiction.
#: GFP is barred as a reporter rather than biology: with it nameable, k-means produced
#: "Pvalb-4 (GFP)" -- a cluster whose only distinguishing feature was reporter
#: brightness, which tracks labelling efficiency, not cell type. GFP is NOT a class gate
#: (see INHIBITORY_MARKERS) and still participates in clustering and the plotted matrix.
GATE_GENES = ("Gad2", "Slc17a7", "GFP")

#: Canonical inhibitory subclasses, in the order clusters are grouped.
SUBCLASS_GENES = ("Pvalb", "Sst", "Vip", "Lamp5")

DEPTH_SCALE = "median"      # per-cell depth target: "median" or "mean" cell total
GENE_PERCENTILE = 95        # per-gene scale: divide by this percentile, clip to 1
MIN_CLASS_COUNTS = 100      # transcripts of a class marker to call that class
N_CLUSTERS_INH = 20
N_CLUSTERS_EXC = 12
RANDOM_SEED = 0
MIN_SUBCLASS_ENRICHMENT = 1.2   # below this a cluster gets no subclass, not a guess


def parse_columns(columns):
    """['R5-561-Cck', ...] -> DataFrame(round, channel, gene) indexed by column."""
    rows = []
    for c in columns:
        parts = str(c).split("-")
        if len(parts) >= 3:
            rows.append(dict(column=c, round=parts[0], channel=parts[1],
                             gene="-".join(parts[2:])))
        else:
            rows.append(dict(column=c, round=None, channel=None, gene=str(c)))
    return pd.DataFrame(rows).set_index("column")


def gene_column(table, gene):
    """The single column for `gene`, or None. Raises when a gene is ambiguous."""
    hits = [c for c in table.columns if str(c).split("-")[-1] == gene]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(f"{gene} appears in {len(hits)} columns: {hits}. "
                         "A gene imaged in several rounds needs explicit handling.")
    return hits[0]


def normalize_cellxgene(table, depth_scale=DEPTH_SCALE,
                        gene_percentile=GENE_PERCENTILE):
    """Two-stage normalization: per-cell MEAN, then per-gene percentile scale.

    Returns (normalized DataFrame, info dict). `depth_scale` is accepted for backward
    compatibility and ignored -- stage 1 is always the cell's own mean.
    """
    counts = table.to_numpy(float)

    # STAGE 1 -- per-cell depth. Each cell is divided by ITS OWN mean gene count, so the
    # unit becomes "relative to this cell's typical gene". A cell twice as brightly
    # detected has twice the mean and divides out, which is the point: raw counts
    # cluster on detection depth (it spans two orders of magnitude here) rather than on
    # identity.
    #
    # The MEAN rather than the median: with a panel this size (27 genes) a sparse cell's
    # median is frequently 0 -- 11,993 of 76,143 cells (15.8%) on 800995 -- and those
    # cells cannot be scaled at all. The mean is positive whenever ANY gene is detected,
    # so every cell with a single transcript is scalable and none are dropped.
    cell_mean = counts.mean(axis=1)
    keep = cell_mean > 0

    scaled = np.zeros_like(counts)
    scaled[keep] = counts[keep] / cell_mean[keep, None]

    # STAGE 2 -- per-gene scale. Divide each gene by its own high percentile over the
    # scalable cells and clip to 1, so a rare gene and an abundant one are comparable.
    # The percentile rather than the max keeps a few outlier cells from compressing
    # everything else into the bottom of the range.
    pct = np.percentile(scaled[keep], gene_percentile, axis=0)
    pct[pct <= 0] = 1.0
    scaled = np.clip(scaled / pct, 0, 1)

    info = dict(depth_scale="per_cell_mean",
                cell_mean_summary=dict(
                    mean=float(np.mean(cell_mean[keep])) if keep.any() else 0.0,
                    median=float(np.median(cell_mean[keep])) if keep.any() else 0.0,
                    min=float(cell_mean[keep].min()) if keep.any() else 0.0,
                    max=float(cell_mean.max())),
                gene_percentile=gene_percentile,
                gene_scale={c: float(p) for c, p in zip(table.columns, pct)},
                n_zero_mean_cells=int((~keep).sum()))
    return pd.DataFrame(scaled, index=table.index, columns=table.columns), info


def assign_class(table, min_counts=MIN_CLASS_COUNTS, markers=CLASS_MARKERS,
                 inhibitory_markers=INHIBITORY_MARKERS):
    """class per cell, from positive markers only.

    INHIBITORY: any of `inhibitory_markers` (Gad2, Pvalb, Vip, Sst) at or above
    `min_counts`. Gad2 alone under-calls -- it is only moderately expressed in some
    interneuron types -- so a cell strongly expressing any canonical interneuron marker
    counts, whether or not its Gad2 reading cleared the bar.

    EXCITATORY: Slc17a7 at or above `min_counts`, and no inhibitory marker.

    Slc17a7-HIGH cells need Gad2 corroboration: an interneuron marker alone is not
    enough when the cell is also strongly excitatory-positive, since a moderate Pvalb
    reading in an excitatory cell would otherwise admit it.

    UNASSIGNED: everything else. That is two distinct situations, both worth keeping
    visible rather than forced into a class:
      * genuinely ambiguous -- Gad2 AND Slc17a7 both above threshold. A double-positive
        cell is usually a segmentation artefact (two cells merged) or residual
        contamination, and calling it either way propagates that error.
      * below threshold on everything, so there is no evidence either way.
    A cell positive for an interneuron marker AND Slc17a7 but NOT Gad2 is called
    inhibitory: without Gad2 corroboration the Slc17a7 is the more likely stray signal.
    """
    n = len(table)
    out = pd.Series(["unassigned"] * n, index=table.index, dtype=object)

    exc_col = gene_column(table, markers["excitatory"])
    gad_col = gene_column(table, markers["inhibitory"])
    inh_cols = {g: gene_column(table, g) for g in inhibitory_markers}

    pos_exc = (table[exc_col].to_numpy() >= min_counts if exc_col is not None
               else np.zeros(n, bool))
    pos_gad = (table[gad_col].to_numpy() >= min_counts if gad_col is not None
               else np.zeros(n, bool))
    per_marker, pos_inh = {}, np.zeros(n, bool)
    for g, col in inh_cols.items():
        hit = (table[col].to_numpy() >= min_counts if col is not None
               else np.zeros(n, bool))
        per_marker[g] = int(hit.sum())
        pos_inh |= hit

    # A cell clearing an interneuron marker while ALSO strongly Slc17a7-positive needs
    # Gad2 corroboration. Without it, an excitatory cell carrying a moderate Pvalb
    # reading is admitted and then clusters with interneurons: on 800995 that produced
    # 1,012 cells (14.4% of the inhibitory class) across three clusters whose Slc17a7
    # medians were 743-806 with Gad2 medians of 26-76 -- i.e. plainly excitatory, let in
    # on Pvalb medians of 154-196 against a real Pvalb cluster's 262. Requiring Gad2
    # rejects 78% of those while keeping 99.1% of genuine inhibitory cells.
    #
    # Gad2 rather than a per-marker threshold: raising each marker to its own 95th
    # percentile was measured on the same cells and rejected only 1 of the 1,012 while
    # discarding 132 genuine ones -- the contaminants' Pvalb is genuinely above any
    # sensible marker threshold, so only a SECOND marker distinguishes them.
    slc_high = pos_exc
    corroborated = pos_inh & (~slc_high | pos_gad)

    ambiguous = pos_gad & pos_exc          # both class markers -> refuse to call
    out[corroborated & ~ambiguous] = "inhibitory"
    out[pos_exc & ~corroborated] = "excitatory"

    info = dict(markers_available={"excitatory": (exc_col or "none"),
                                   "inhibitory": (gad_col or "none")},
                inhibitory_markers={g: (c or "none") for g, c in inh_cols.items()},
                n_positive_per_inhibitory_marker=per_marker,
                min_counts=min_counts,
                n_inhibitory=int((out == "inhibitory").sum()),
                n_excitatory=int((out == "excitatory").sum()),
                n_ambiguous_gad2_and_slc17a7=int(ambiguous.sum()),
                n_below_threshold_on_all=int((~pos_inh & ~pos_exc).sum()))
    return out, info


def _kmeans(matrix, k, seed=RANDOM_SEED):
    """k-means on the normalized matrix DIRECTLY -- no z-scoring.

    normalize_cellxgene already puts every gene on a common [0, 1] scale, so the
    matrix is the intended clustering space. Z-scoring on top of it re-inflated each
    gene to unit variance, which undoes that: a gene detected in a handful of cells
    gets the same variance budget as a gene carrying real structure, and distances
    stop reflecting expression level.
    """
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix)


def cluster_marker_names(cluster_means, prefix_by_cluster, n_markers=3,
                         min_enrichment=1.25,
                         exclude_genes=SUBCLASS_GENES + ROUND6_GENES + GATE_GENES):
    """Name each cluster `<prefix>-<n> (GeneA/GeneB/GeneC)`.

    Markers are the genes most ENRICHED in the cluster -- cluster mean divided by the
    mean across all clusters in the same group -- not the highest raw values, so an
    abundant gene does not name every cluster. Numbering runs within each prefix.

    `exclude_genes` are barred from the marker list, defaulting to the canonical
    subclass genes. Those already appear as the prefix, so listing them again wastes a
    slot: `Pvalb-2 (Pvalb/Mme/Pthlh)` says less than `Pvalb-2 (Mme/Pthlh/Tac)`. The
    subclass call itself is unaffected -- assign_subclass still uses those genes.
    """
    overall = cluster_means.mean(0)
    overall[overall <= 0] = 1e-9
    enrich = cluster_means / overall
    barred = set(exclude_genes or ())

    counter, names = {}, {}
    for cid in cluster_means.index:
        prefix = prefix_by_cluster[cid]
        counter[prefix] = counter.get(prefix, 0) + 1
        e = enrich.loc[cid].sort_values(ascending=False)
        marks = [str(g).split("-")[-1] for g, v in e.items()
                 if v >= min_enrichment and str(g).split("-")[-1] not in barred][:n_markers]
        names[cid] = (f"{prefix}-{counter[prefix]}"
                      + (f" ({'/'.join(marks)})" if marks else ""))
    return names


def assign_subclass(cluster_means, subclass_genes=SUBCLASS_GENES,
                    min_enrichment=MIN_SUBCLASS_ENRICHMENT):
    """Each cluster's subclass = the canonical marker with the HIGHEST EXPRESSION in it.

    Level, not enrichment. The enrichment version (cluster mean / across-cluster mean)
    produced labels that contradicted the cluster's own data: three Lamp5 clusters where
    Sst or Pvalb was in fact the higher-expressing marker, because Lamp5 was merely more
    ENRICHED relative to other clusters. Reading the label off the highest marker means
    the heatmap and the label always agree -- what you see in the Pvalb column is why
    the row says Pvalb.

    Both matrices are on the same normalized scale, so the four markers are directly
    comparable. `min_enrichment` is still applied as a floor on the winner's enrichment,
    so a cluster with no marker standing out at all gets None instead of whichever of
    four flat values happened to be largest.
    """
    overall = cluster_means.mean(0)
    overall[overall <= 0] = 1e-9
    enrich = cluster_means / overall

    cols = {}
    for g in subclass_genes:
        hits = [c for c in cluster_means.columns if str(c).split("-")[-1] == g]
        if hits:
            cols[g] = hits[0]

    out = {}
    for cid in cluster_means.index:
        if not cols:
            out[cid] = (None, np.nan)
            continue
        # winner by EXPRESSION LEVEL
        best = max(cols, key=lambda g: cluster_means.loc[cid, cols[g]])
        # enrichment of that winner, kept as the evidence floor and reported in uns
        val = float(enrich.loc[cid, cols[best]])
        out[cid] = (best if val >= min_enrichment else None, val)
    return out


def cluster_by_class(normalized, classes, n_inh=N_CLUSTERS_INH, n_exc=N_CLUSTERS_EXC,
                     seed=RANDOM_SEED, n_markers=3):
    """Cluster excitatory and inhibitory cells SEPARATELY, then merge the labels.

    Separately because the two classes differ enormously in which genes are
    informative; one joint k-means spends most of its clusters separating the classes
    rather than resolving structure within them. Inhibitory clusters are named by
    subclass (`Pvalb-2 (Mme/Calb1/Cck)`), excitatory ones `Exc-n (...)` since the
    panel carries no excitatory subclass markers.

    Returns (labels Series, cluster_id Series, subclass Series, info dict).
    """
    labels = pd.Series(["unassigned"] * len(normalized), index=normalized.index,
                       dtype=object)
    cluster_id = pd.Series([-1] * len(normalized), index=normalized.index, dtype=int)
    subclass = pd.Series([None] * len(normalized), index=normalized.index, dtype=object)
    info = {}
    offset = 0

    # Cells that could not be depth-normalized (median gene count 0) are all-zero rows.
    # Clustering them would let a fabricated profile form its own cluster, so they are
    # left unclustered (cluster_id -1) exactly like the unassigned class.
    scalable = normalized.index[normalized.to_numpy().sum(1) > 0]

    for cls, k in (("inhibitory", n_inh), ("excitatory", n_exc)):
        sel = classes[classes == cls].index
        sel = normalized.index.intersection(sel).intersection(scalable)
        if len(sel) < k:
            info[cls] = dict(n_cells=int(len(sel)), n_clusters=0,
                             note="too few cells to cluster")
            continue

        sub = normalized.loc[sel]
        lab = _kmeans(sub.to_numpy(float), k, seed)
        means = pd.DataFrame(sub.to_numpy(float), index=lab,
                             columns=sub.columns).groupby(level=0).mean()

        if cls == "inhibitory":
            sc = assign_subclass(means)
            prefix = {cid: (sc[cid][0] or "Inh") for cid in means.index}
        else:
            sc = {cid: (None, np.nan) for cid in means.index}
            prefix = {cid: "Exc" for cid in means.index}

        names = cluster_marker_names(means, prefix, n_markers=n_markers)
        labels.loc[sel] = [names[c] for c in lab]
        cluster_id.loc[sel] = [c + offset for c in lab]
        subclass.loc[sel] = [sc[c][0] for c in lab]

        # uns is written to HDF5, whose group keys must be strings -- an int key
        # raises deep inside the writer AFTER the unmixing has already run. Keys are
        # stringified here, and the values kept HDF5-writable (no None).
        info[cls] = dict(n_cells=int(len(sel)), n_clusters=int(k),
                         id_offset=int(offset),
                         names={str(int(c)): names[c] for c in means.index},
                         subclass={str(int(c)): (sc[c][0] or "none")
                                   for c in means.index},
                         subclass_enrichment={
                             str(int(c)): (float("nan") if not np.isfinite(sc[c][1])
                                           else round(float(sc[c][1]), 3))
                             for c in means.index})
        offset += k

    return labels, cluster_id, subclass, info


def round_channel_order(columns):
    """Column order by round then channel: R1-488, R1-561, R2-488, ... R6-638.

    The acquisition order. Useful as an alternative to the biology-grouped standard
    gene order because it makes round- and channel-level artefacts visible as vertical
    bands -- a whole round reading high, or one channel across rounds.
    """
    def key(c):
        parts = str(c).split("-")
        rnd = int("".join(ch for ch in parts[0] if ch.isdigit()) or 0)
        chan = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else 0
        return (rnd, chan)
    return sorted(columns, key=key)


def build_anndata(table, min_class_counts=MIN_CLASS_COUNTS, n_inh=N_CLUSTERS_INH,
                  n_exc=N_CLUSTERS_EXC, seed=RANDOM_SEED, depth_scale=DEPTH_SCALE,
                  gene_percentile=GENE_PERCENTILE, extra_uns=None):
    """Annotated AnnData from a cell x gene count table. See module docstring."""
    import anndata as ad

    table = table.sort_index()
    normalized, norm_info = normalize_cellxgene(table, depth_scale, gene_percentile)
    classes, class_info = assign_class(table, min_class_counts)
    labels, cluster_id, subclass, clust_info = cluster_by_class(
        normalized, classes, n_inh=n_inh, n_exc=n_exc, seed=seed)

    var = parse_columns(table.columns)
    obs = pd.DataFrame(index=table.index.astype(str))
    obs["class"] = pd.Categorical(classes.values)
    obs["subclass"] = pd.Categorical([s if s else "none" for s in subclass.values])
    obs["cluster"] = pd.Categorical(labels.values)
    obs["cluster_id"] = cluster_id.values
    obs["total_counts"] = table.to_numpy().sum(1)
    obs["n_genes"] = (table.to_numpy() > 0).sum(1)
    for cls, gene in CLASS_MARKERS.items():
        col = gene_column(table, gene)
        if col is not None:
            obs[f"{gene}_counts"] = table[col].to_numpy()

    adata = ad.AnnData(X=table.to_numpy().astype(np.float32), obs=obs, var=var)
    adata.layers["normalized"] = normalized.to_numpy().astype(np.float32)
    adata.uns["unmixing"] = dict(
        normalization=norm_info,
        classification=class_info,
        clustering=dict(method="kmeans", seed=seed, **clust_info),
        note=("X is raw transcript counts; layers['normalized'] is what clustering "
              "was run on (per-cell depth then per-gene 95th percentile, clipped)."),
    )
    if extra_uns:
        adata.uns["unmixing"].update(extra_uns)
    return adata
