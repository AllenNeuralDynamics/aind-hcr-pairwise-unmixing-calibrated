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

#: Canonical inhibitory subclasses, in the order clusters are grouped.
SUBCLASS_GENES = ("Pvalb", "Sst", "Vip", "Lamp5")

DEPTH_SCALE = "median"      # per-cell depth target: "median" or "mean" cell total
GENE_PERCENTILE = 95        # per-gene scale: divide by this percentile, clip to 1
MIN_CLASS_COUNTS = 50       # transcripts of a class marker to call that class
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
    """Two-stage normalization: per-cell depth, then per-gene scale.

    Returns (normalized DataFrame, info dict). Cells with zero total are dropped from
    the normalized matrix -- they carry no information and would divide by zero.
    """
    counts = table.to_numpy(float)
    totals = counts.sum(1)
    keep = totals > 0
    target = (np.median(totals[keep]) if depth_scale == "median"
              else float(np.mean(totals[keep])))

    scaled = np.zeros_like(counts)
    scaled[keep] = counts[keep] / totals[keep, None] * target

    pct = np.percentile(scaled[keep], gene_percentile, axis=0)
    pct[pct <= 0] = 1.0
    scaled = np.clip(scaled / pct, 0, 1)

    info = dict(depth_scale=depth_scale, depth_target=float(target),
                gene_percentile=gene_percentile,
                gene_scale={c: float(p) for c, p in zip(table.columns, pct)},
                n_zero_total_cells=int((~keep).sum()))
    return pd.DataFrame(scaled, index=table.index, columns=table.columns), info


def assign_class(table, min_counts=MIN_CLASS_COUNTS, markers=CLASS_MARKERS):
    """class per cell from positive markers only.

    A cell positive for exactly one class marker gets that class. Positive for both,
    or neither, gets "unassigned" -- those are the cells worth looking at, and forcing
    them into a class would hide them. When the excitatory marker is missing from the
    panel every non-inhibitory cell is "unassigned" (see module docstring).
    """
    cols = {cls: gene_column(table, g) for cls, g in markers.items()}
    n = len(table)
    out = pd.Series(["unassigned"] * n, index=table.index, dtype=object)

    pos = {}
    for cls, col in cols.items():
        pos[cls] = (table[col].to_numpy() >= min_counts if col is not None
                    else np.zeros(n, bool))

    only_inh = pos["inhibitory"] & ~pos["excitatory"]
    only_exc = pos["excitatory"] & ~pos["inhibitory"]
    out[only_inh] = "inhibitory"
    out[only_exc] = "excitatory"
    # both-positive stays "unassigned"; so does neither-positive
    info = dict(markers_available={c: (cols[c] or "none") for c in cols},
                min_counts=min_counts,
                n_double_positive=int((pos["inhibitory"] & pos["excitatory"]).sum()),
                n_neither=int((~pos["inhibitory"] & ~pos["excitatory"]).sum()))
    return out, info


def _kmeans(matrix, k, seed=RANDOM_SEED):
    from sklearn.cluster import KMeans
    z = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-9)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)


def cluster_marker_names(cluster_means, prefix_by_cluster, n_markers=3,
                         min_enrichment=1.25):
    """Name each cluster `<prefix>-<n> (GeneA/GeneB/GeneC)`.

    Markers are the genes most ENRICHED in the cluster -- cluster mean divided by the
    mean across all clusters in the same group -- not the highest raw values, so an
    abundant gene does not name every cluster. Numbering runs within each prefix.
    """
    overall = cluster_means.mean(0)
    overall[overall <= 0] = 1e-9
    enrich = cluster_means / overall

    counter, names = {}, {}
    for cid in cluster_means.index:
        prefix = prefix_by_cluster[cid]
        counter[prefix] = counter.get(prefix, 0) + 1
        e = enrich.loc[cid].sort_values(ascending=False)
        marks = [str(g).split("-")[-1] for g, v in e.items() if v >= min_enrichment][:n_markers]
        names[cid] = (f"{prefix}-{counter[prefix]}"
                      + (f" ({'/'.join(marks)})" if marks else ""))
    return names


def assign_subclass(cluster_means, subclass_genes=SUBCLASS_GENES,
                    min_enrichment=MIN_SUBCLASS_ENRICHMENT):
    """Each cluster's subclass = the canonical marker most enriched in it.

    Enrichment, not raw level: an abundant gene would otherwise win everywhere. A
    cluster whose best marker falls under min_enrichment gets None rather than a
    subclass it has no evidence for.
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
        best = max(cols, key=lambda g: enrich.loc[cid, cols[g]])
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

    for cls, k in (("inhibitory", n_inh), ("excitatory", n_exc)):
        sel = classes[classes == cls].index
        sel = normalized.index.intersection(sel)
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
