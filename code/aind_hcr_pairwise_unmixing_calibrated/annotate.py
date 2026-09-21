"""Annotated cell x gene table as AnnData.

Produces an .h5ad carrying every cell, the raw transcript counts, and three levels of
label: class (excitatory / inhibitory / ambiguous / low_counts), subclass
(Pvalb / Sst / Vip / Lamp5), and a named cluster.

Every rule here comes from the HCR consensus-clustering protocol and is implemented in
`labeling.py`, vendored from the cohort capsule so the two pipelines agree. This module
is the capsule's wiring: it decides which cells and which genes each rule sees, and
assembles the AnnData.

CLUSTERS HERE ARE PER MOUSE
---------------------------
The k-means fit runs on one animal, so cluster identities do NOT correspond across
mice -- `Sst-2` in one mouse is not `Sst-2` in another. For any across-mouse analysis
use the consensus clusters from the cohort capsule, which fits all animals jointly and
assigns every cell to a shared set of centroids. Class and subclass are per-cell rules
and DO carry across mice.

WHAT IS STORED WHERE
--------------------
    X                    RAW transcript counts. Integers, no transformation.
    layers["normalized"] the p95 transform over all cells and all genes: each gene
                         divided by its 95th percentile, then each cell divided by its
                         own total and rescaled to the median total. This is the
                         display matrix.
    obsm["X_cluster"]    the matrix each cell was actually clustered on, widened to
                         the full gene set with NaN wherever nothing was fitted -- a
                         gene outside that class's clustering space, a cell of no
                         class, a cell dropped as all-zero. Inhibitory cells are
                         clustered on the fifteen protocol genes, excitatory on the
                         rest of the panel.

    Only X and layers["normalized"] cover every cell. The two class-scoped matrices
    are defined per class and are NaN for a cell that has none, because there is no
    class whose percentiles could scale it. NaN rather than 0 because a zero row is a
    measurement -- read at face value it says the cell expressed nothing, when what is
    true is that the matrix has nothing to say about that cell.
    obs                  class, subclass, cluster, cluster_id, the mixture posterior
                         the class call came from, the marker counts it was computed
                         on, and total_counts / n_genes.
    var                  round, channel, gene for each column.
    uns                  every parameter each label was computed with.

Raw counts in X and the transformed matrices alongside, rather than one or the other:
a reader who wants counts should not have to invert a transform, and a reader who wants
to reproduce the clustering should not have to guess how it was normalised.

WHY THE INHIBITORY CLUSTERING SEES ONLY FIFTEEN GENES
-----------------------------------------------------
`HCR_PANEL_15` is the gene set the cohort clustering was run on, and k was swept on it.
The remaining panel genes -- including Gad2 and Slc17a7 -- are held out so they stay
available as an independent check on the resulting clusters: a cluster that separates
on the clustering genes and then also separates on a held-out gene has evidence the
clustering itself could not have manufactured. All genes remain in X and in the
normalised layer; only the distance metric is restricted.

The excitatory clustering has no protocol counterpart -- the cohort work covers
inhibitory cells only -- so it keeps this capsule's previous behaviour of using the
whole panel, less the two class markers and the GFP reporter.

CLASS LABELS NEED R1 AND R4
---------------------------
The class call is a mixture on the Gad2/Slc17a7 ratio, so it needs both markers:
Slc17a7 is imaged in R1 (488=GFP, 561=Slc17a7) and Gad2 in R4. With either round
absent there is no ratio to fit, and `class` is left "unassigned" for every cell rather
than asserting a class from one marker's absence -- low-quality cells, badly segmented
cells and non-neuronal cells would all land in that bucket. uns records which markers
were available.
"""
import numpy as np
import pandas as pd

from .labeling import (HCR_COUNT_FLOOR, HCR_ENRICHMENT_FLOOR, HCR_NAME_FLOOR,
                       HCR_PANEL_15, HCR_SUBCLASS_COUNT_FLOOR, HCR_SUBCLASS_MARKERS,
                       hcr_class_call, hcr_cluster_blocks, hcr_cluster_names,
                       hcr_sncg_cells, hcr_subclass_argmax, hcr_transform_p95)

#: The two markers the class call is a ratio of, and the round each is imaged in.
CLASS_MARKERS = {"excitatory": "Slc17a7", "inhibitory": "Gad2"}

#: Canonical inhibitory subclasses, in the order blocks are grouped for output.
SUBCLASS_GENES = tuple(HCR_SUBCLASS_MARKERS)

#: The fifth subclass and the gene that defines it. Sncg has no positive marker of its
#: own in this panel, so it is called from high Cck together with the ABSENCE of the
#: four markers -- in the per-cell call, alongside them rather than after them.
SNCG_GENE, SNCG_BLOCK = "Cck", "Sncg"

#: Barred from the excitatory clustering space. Gad2 and Slc17a7 define the class, so
#: clustering on them re-separates cells the class call already separated. GFP is a
#: reporter rather than biology: with it in the space, k-means produced a cluster whose
#: only distinguishing feature was reporter brightness, which tracks labelling
#: efficiency, not cell type.
EXCLUDED_FROM_CLUSTERING = ("Gad2", "Slc17a7", "GFP")

N_CLUSTERS_INH = 18         # the cohort protocol's k, applied here per mouse
N_CLUSTERS_EXC = 12         # no protocol counterpart; this capsule's prior value
RANDOM_SEED = 0
MIN_CLASS_COUNTS = HCR_COUNT_FLOOR


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


#: Panel gene names that are WRONG and are corrected on load, with the reason each is
#: wrong. Every correction is announced -- see `rename_gene_aliases` and
#: `correct_gene_map`. A silent rename would leave two names for one gene circulating in
#: downstream analyses with nothing in the log to explain which is which.
#:
#:   Tac        the HCR panel's label for the Tac1 probe. Not a gene symbol, and left
#:              alone it fails to match HCR_PANEL_15, so the inhibitory clustering
#:              silently runs on fourteen genes instead of fifteen.
#:   Slac17a7   a misspelling of Slc17a7, the excitatory marker the class call is a
#:              ratio of. Left alone, `gene_column(table, "Slc17a7")` finds nothing and
#:              every cell comes back unassigned. AllenNeuralDynamics/
#:              hcr-pairwise-spot-unmixing patches the same typo in its own output.
#:
#: Both originate in the acquisition metadata -- `processing_manifest.json`'s gene_dict,
#: which is where the round -> channel -> gene map is read from -- so they are corrected
#: at that boundary, not invented here.
GENE_ALIASES = {"Tac": "Tac1", "Slac17a7": "Slc17a7"}


def correct_gene_map(gene_map, round_key=None):
    """{channel: gene} with wrong symbols corrected, announced per substitution.

    Applied where the map is READ, because the gene name reaches more outputs than the
    cell x gene table: `*_spot_change.csv` takes its `gene` column straight from this
    map, and `rename_gene_aliases` -- which rewrites table column labels -- never sees
    it. Correcting here fixes both from one place.

    Returns (corrected map, [(channel, old, new), ...]).
    """
    out, changed = {}, []
    for chan, gene in gene_map.items():
        fixed = GENE_ALIASES.get(str(gene), str(gene))
        out[chan] = fixed
        if fixed != str(gene):
            changed.append((str(chan), str(gene), fixed))
            where = f" in {round_key}" if round_key else ""
            print(f"WARNING: gene name '{gene}' is not a valid symbol and was corrected "
                  f"to '{fixed}'{where} (channel {chan}). Downstream outputs use the "
                  f"corrected name.", flush=True)
    return out, changed


def gene_name(column):
    """'R5-561-Cck' -> 'Cck'.

    No alias resolution here: corrections happen once, loudly, in
    `rename_gene_aliases`, so this stays a pure parse of whatever the column says.
    """
    return str(column).split("-")[-1]


def rename_gene_aliases(table, aliases=None, warn=True):
    """Correct wrong gene names in the column labels. Returns (table, renames).

    Renames the gene field of `<round>-<channel>-<gene>` columns, so the correction
    reaches everything downstream of the cell x gene table -- the CSV header, the
    AnnData `var`, the figure axes -- rather than only the lookups that happen to go
    through a resolver.

    Idempotent: a table whose names are already correct is returned unchanged with an
    empty rename list. In a normal run `correct_gene_map` has already fixed the map the
    columns were built from, so this pass finds nothing. It still matters for a table
    that came from somewhere else -- `--relabel-from` on a CSV written before the
    correction existed is the case that keeps it here.
    """
    aliases = dict(GENE_ALIASES if aliases is None else aliases)
    renames = {}
    for col in table.columns:
        parts = str(col).split("-")
        if parts[-1] in aliases:
            parts[-1] = aliases[parts[-1]]
            renames[col] = "-".join(parts)
    if not renames:
        return table, []
    if warn:
        for old, new in renames.items():
            print(f"WARNING: gene name '{str(old).split('-')[-1]}' is not a valid symbol"
                  f" and was corrected to '{str(new).split('-')[-1]}'"
                  f" ({old} -> {new}). Downstream outputs use the corrected name.",
                  flush=True)
    return table.rename(columns=renames), [(o, n) for o, n in renames.items()]


def gene_column(table, gene):
    """The single column for `gene`, or None. Raises when a gene is ambiguous."""
    hits = [c for c in table.columns if gene_name(c) == gene]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(f"{gene} appears in {len(hits)} columns: {hits}. "
                         "A gene imaged in several rounds needs explicit handling.")
    return hits[0]


def normalize_cellxgene(table, group_vec=None):
    """The p95 transform over the whole table. Returns (DataFrame, info dict).

    Per gene, divide by that gene's 95th percentile; then per cell, divide by the
    cell's own total and rescale to the median total. The gene stage puts a rare gene
    and an abundant one on a comparable scale; the cell stage removes detection depth,
    which spans two orders of magnitude here and would otherwise dominate any distance.

    This replaced a per-cell-mean-then-percentile transform when the labelling was
    aligned to the cohort protocol. The order matters: normalising cells first and
    genes second makes each gene's percentile depend on the cell composition of the
    table, so the same cell transforms differently in a single-mouse and a cohort run.
    """
    counts = table.to_numpy(float)
    scaled = hcr_transform_p95(counts, group_vec)
    pct = np.percentile(counts, 95, axis=0)
    info = dict(transform="p95_then_cell_total",
                gene_percentile=95,
                gene_scale={c: float(p) for c, p in zip(table.columns, pct)},
                n_zero_total_cells=int((counts.sum(1) == 0).sum()))
    return pd.DataFrame(scaled, index=table.index, columns=table.columns), info


def assign_class(table, min_counts=MIN_CLASS_COUNTS, markers=CLASS_MARKERS, seed=0):
    """Class per cell, from a two-component mixture on the Gad2/Slc17a7 ratio.

    Returns (Series, info dict, posterior array). Labels are inhibitory / excitatory /
    ambiguous / low_counts, or "unassigned" everywhere when a marker is missing.

    The mixture is fitted on log2((Gad2+1)/(Slc17a7+1)) over cells clearing
    `min_counts` total counts, and cells are cut at posterior 0.90 and 0.10. A mixture
    rather than fixed per-marker count thresholds: the boundary is set by the data's
    own two modes, so it tracks a mouse's detection depth instead of being calibrated
    on one animal and carried to the others.

    Cells between the two gates are `ambiguous` and stay out of both classes -- around
    1% in the cohort. They are not a thresholding artefact: the fraction barely moves
    when the mixture is refitted per mouse, and it does not fall with library size.
    Some are merged cells from segmentation; some carry genuine signal from both
    markers. Calling them either way propagates the error into every downstream count.
    """
    n = len(table)
    exc_col = gene_column(table, markers["excitatory"])
    inh_col = gene_column(table, markers["inhibitory"])
    total = table.to_numpy().sum(1)

    if exc_col is None or inh_col is None:
        out = pd.Series(["unassigned"] * n, index=table.index, dtype=object)
        info = dict(method="gaussian_mixture_log_ratio",
                    markers_available={"excitatory": exc_col or "none",
                                       "inhibitory": inh_col or "none"},
                    note="class not called: the ratio needs both markers",
                    min_counts=min_counts)
        return out, info, np.full(n, np.nan)

    klass, posterior, thresholds = hcr_class_call(
        table[inh_col].to_numpy(), table[exc_col].to_numpy(), total,
        count_floor=min_counts, seed=seed)
    out = pd.Series(klass, index=table.index, dtype=object)

    info = dict(method="gaussian_mixture_log_ratio",
                markers_available={"excitatory": exc_col, "inhibitory": inh_col},
                min_counts=int(min_counts),
                posterior_gates=[0.10, 0.90],
                log_ratio_thresholds=[float(t) for t in thresholds],
                n_inhibitory=int((out == "inhibitory").sum()),
                n_excitatory=int((out == "excitatory").sum()),
                n_ambiguous=int((out == "ambiguous").sum()),
                n_low_counts=int((out == "low_counts").sum()))
    return out, info, posterior


def assign_subclass(table, markers=SUBCLASS_GENES,
                    count_floor=HCR_SUBCLASS_COUNT_FLOOR):
    """Subclass per cell: whichever subclass marker carries the most RAW counts.

    Returns (Series, info dict). A cell whose winning marker is below `count_floor` is
    "unassigned" rather than forced into a subclass.

    On raw counts, not on the transformed matrix: the p95 stage divides each gene by
    its own 95th percentile, so Lamp5 renders about three times darker than Sst at
    equal raw counts and the call moves. A nearest-centroid rule on the whole profile
    is also wrong here -- it is biased by how many panel genes a cell type happens to
    co-express, so a type expressing five panel genes is pulled away from a type
    expressing one.

    This is a per-CELL call. Clusters get their own block label by plurality of these
    calls, and the two are allowed to disagree: a cluster is named from its plurality
    while cells keep their own call, and that residual disagreement is what makes
    cluster purity informative.
    """
    cols = {g: gene_column(table, g) for g in markers}
    present = [g for g in markers if cols[g] is not None]
    if not present:
        out = pd.Series(["unassigned"] * len(table), index=table.index, dtype=object)
        return out, dict(markers_available=[], count_floor=int(count_floor))

    values = np.column_stack([table[cols[g]].to_numpy(float) for g in present])
    lab = hcr_subclass_argmax(values, markers=present, count_floor=count_floor)

    # Sncg competes here, not afterwards: a cell with Cck above the floor and no
    # marker above it belongs to the subclass defined by that absence. Drawn only
    # from what the four-way call left unassigned -- see hcr_sncg_cells.
    sncg_col = gene_column(table, SNCG_GENE)
    n_sncg = 0
    if sncg_col is not None:
        lab = hcr_sncg_cells(lab, values, table[sncg_col].to_numpy(float),
                             count_floor=count_floor, sncg_block=SNCG_BLOCK)
        n_sncg = int((lab == SNCG_BLOCK).sum())

    out = pd.Series(lab, index=table.index, dtype=object)
    info = dict(markers_available=present, count_floor=int(count_floor),
                n_per_subclass={g: int((out == g).sum()) for g in present},
                sncg_gene=(SNCG_GENE if sncg_col is not None else None),
                n_sncg=n_sncg,
                n_unassigned=int((out == "unassigned").sum()))
    return out, info


def clustering_genes(table, klass):
    """Columns that class's clustering runs on.

    Inhibitory: the fifteen protocol genes, intersected with what the panel carries --
    a reduced panel simply contributes fewer of them. Excitatory: everything except the
    class markers and the reporter.
    """
    if klass == "inhibitory":
        return [c for c in table.columns if gene_name(c) in HCR_PANEL_15]
    return [c for c in table.columns if gene_name(c) not in EXCLUDED_FROM_CLUSTERING]


def _kmeans(matrix, k, seed=RANDOM_SEED):
    """k-means on the transformed matrix DIRECTLY -- no z-scoring.

    The p95 transform already puts every gene on a common scale, so the matrix is the
    intended clustering space. Z-scoring on top re-inflates each gene to unit variance,
    which undoes that: a gene detected in a handful of cells gets the same variance
    budget as a gene carrying real structure, and distances stop reflecting level.
    """
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix)


def cluster_by_class(table, classes, subclass, n_inh=N_CLUSTERS_INH,
                     n_exc=N_CLUSTERS_EXC, seed=RANDOM_SEED):
    """Cluster inhibitory and excitatory cells independently, then recombine.

    Independently because the two classes differ in which genes are informative and in
    which genes they are clustered on; one joint fit spends most of its clusters
    re-separating the classes rather than resolving structure within them.

    The transform is recomputed on each class's own cells and its own gene space, which
    is what the protocol does for the inhibitory table. Inhibitory clusters take a
    subclass block by plurality of the per-cell calls at `HCR_ENRICHMENT_FLOOR`
    enrichment, with the Cck-dominance promotion to Sncg; excitatory clusters are all
    block `Exc`, the panel carrying no excitatory subclass markers.

    Returns (labels, cluster_id, cluster_matrix DataFrame, info dict).
    """
    index = table.index
    labels = pd.Series(["unassigned"] * len(index), index=index, dtype=object)
    cluster_id = pd.Series([-1] * len(index), index=index, dtype=int)
    # NaN, not zero: this matrix records what k-means actually saw, so every entry
    # that was not part of a fit -- a cell of no class, a cell dropped as all-zero,
    # a gene outside that class's clustering space -- has no value rather than a
    # value of nothing. Zero here is indistinguishable from a measured zero, and the
    # two mean opposite things. To place an unfitted cell in the fitted space, reuse
    # the fitted percentiles and median (`_project_2b` in the cohort protocol);
    # recomputing the transform on it would move the space it is being placed in.
    cluster_matrix = pd.DataFrame(np.nan, index=index, columns=table.columns)
    info, offset = {}, 0

    for klass, k in (("inhibitory", n_inh), ("excitatory", n_exc)):
        sel = index[classes.reindex(index).to_numpy() == klass]
        genes = clustering_genes(table, klass)
        if len(sel) < k or not genes:
            info[klass] = dict(n_cells=int(len(sel)), n_clusters=0,
                               genes=[gene_name(c) for c in genes],
                               note="too few cells to cluster" if genes
                                    else "no genes available")
            continue

        raw = table.loc[sel, genes]
        # All-zero rows cannot be scaled and would let a fabricated profile form its
        # own cluster, so they stay unclustered exactly like the unassigned class.
        nonzero = raw.to_numpy().sum(1) > 0
        sel = sel[nonzero]
        if len(sel) < k:
            info[klass] = dict(n_cells=int(len(sel)), n_clusters=0,
                               genes=[gene_name(c) for c in genes],
                               note="too few nonzero cells to cluster")
            continue

        matrix = hcr_transform_p95(table.loc[sel, genes].to_numpy(float))
        lab = _kmeans(matrix, k, seed)
        cluster_matrix.loc[sel, genes] = matrix

        means = pd.DataFrame(matrix, index=lab,
                             columns=[gene_name(c) for c in genes]
                             ).groupby(level=0).mean().sort_index()

        if klass == "inhibitory":
            blocks, diag = hcr_cluster_blocks(means, subclass.loc[sel].to_numpy(), lab)
        else:
            blocks = {c: "Exc" for c in means.index}
            diag = pd.DataFrame(index=means.index)

        sizes = pd.Series({c: int((lab == c).sum()) for c in means.index})
        names, ordered = hcr_cluster_names(means, blocks, order_by=-sizes)

        renumber = {c: i + offset for i, c in enumerate(ordered)}
        labels.loc[sel] = [names[c] for c in lab]
        cluster_id.loc[sel] = [renumber[c] for c in lab]

        # uns is written to HDF5, whose group keys must be strings -- an int key raises
        # deep inside the writer AFTER the unmixing has already run.
        info[klass] = dict(
            n_cells=int(len(sel)), n_clusters=int(k), id_offset=int(offset),
            genes=[gene_name(c) for c in genes],
            names={str(int(c)): names[c] for c in means.index},
            block={str(int(c)): blocks[c] for c in means.index},
            size={str(int(c)): int(sizes[c]) for c in means.index},
            enrichment={str(int(c)): round(float(diag.loc[c, "enrichment"]), 3)
                        for c in means.index} if "enrichment" in diag else {},
            purity={str(int(c)): round(float(diag.loc[c, "purity"]), 3)
                    for c in means.index} if "purity" in diag else {},
            # The Sncg decision per cluster, so a promotion (or a near miss) can be
            # audited from the file without re-deriving the cluster means.
            sncg={str(int(c)): dict(
                      top_gene=str(diag.loc[c, "sncg_top_gene"]),
                      top_value=round(float(diag.loc[c, "sncg_top_value"]), 3),
                      cck_value=round(float(diag.loc[c, "sncg_cck_value"]), 3),
                      promoted=bool(diag.loc[c, "sncg_promoted"]))
                  for c in means.index
                  if "sncg_top_gene" in diag and pd.notna(diag.loc[c, "sncg_top_gene"])}
                 if "sncg_top_gene" in diag else {})
        offset += k

    return labels, cluster_id, cluster_matrix, info


def within_class_transform(table, classes):
    """The p95 transform computed within each class, over ALL panel genes.

    This is the matrix the cluster figures display. `layers["normalized"]` is
    transformed over every cell at once, which puts the labels and the picture on
    different scales: a gene's 95th percentile and the per-cell totals are then both
    set by the excitatory majority, and on 800792 that renders inhibitory Cck at 0.13
    where the naming matrix has 0.62 -- invisible in the figure while appearing in the
    cluster name. Transformed within the class, the display agrees with the naming
    matrix to about one percent (Cck 0.557 against 0.551 on the promoted cluster), so
    a name above the 0.5 floor is a mark you can see.

    Cells in neither class -- low_counts, ambiguous, unassigned -- are NaN, not zero.
    There is no class to normalise them within, and zero is a measurement: a reader
    who takes a zero row at face value concludes the cell expressed nothing, when the
    truth is that this matrix has nothing to say about it. Their counts are in `X` and
    their whole-table transform is in `layers["normalized"]`, which covers every cell.
    """
    out = pd.DataFrame(np.nan, index=table.index, columns=table.columns)
    classes = classes.reindex(table.index)
    for klass in ("inhibitory", "excitatory"):
        sel = table.index[classes.to_numpy() == klass]
        if len(sel):
            out.loc[sel] = hcr_transform_p95(table.loc[sel].to_numpy(float))
    return out


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
                  n_exc=N_CLUSTERS_EXC, seed=RANDOM_SEED, extra_uns=None):
    """Annotated AnnData from a cell x gene count table. See the module docstring."""
    import anndata as ad

    table = table.sort_index()
    # Idempotent: a no-op when the pipeline already corrected the names upstream.
    table, renames = rename_gene_aliases(table)
    normalized, norm_info = normalize_cellxgene(table)

    classes, class_info, posterior = assign_class(table, min_class_counts, seed=seed)

    subclass, subclass_info = assign_subclass(table)
    # Subclass is an inhibitory taxonomy; carrying it on excitatory cells would invite
    # reading a Pvalb label on a cell the class call placed outside the class.
    subclass = subclass.where(classes == "inhibitory", "none")

    labels, cluster_id, cluster_matrix, clust_info = cluster_by_class(
        table, classes, subclass, n_inh=n_inh, n_exc=n_exc, seed=seed)

    var = parse_columns(table.columns)
    obs = pd.DataFrame(index=table.index.astype(str))
    obs["class"] = pd.Categorical(classes.values)
    obs["subclass"] = pd.Categorical(subclass.values)
    obs["cluster"] = pd.Categorical(labels.values)
    obs["cluster_id"] = cluster_id.values
    obs["p_inhibitory"] = np.asarray(posterior, dtype=float)
    obs["total_counts"] = table.to_numpy().sum(1)
    obs["n_genes"] = (table.to_numpy() > 0).sum(1)
    for _, gene in CLASS_MARKERS.items():
        col = gene_column(table, gene)
        if col is not None:
            obs[f"{gene}_counts"] = table[col].to_numpy()

    adata = ad.AnnData(X=table.to_numpy().astype(np.float32), obs=obs, var=var)
    adata.layers["normalized"] = normalized.to_numpy().astype(np.float32)
    adata.layers["normalized_within_class"] = (
        within_class_transform(table, classes).to_numpy().astype(np.float32))
    adata.obsm["X_cluster"] = cluster_matrix.to_numpy().astype(np.float32)
    adata.uns["unmixing"] = dict(
        gene_name_corrections=[list(r) for r in renames],
        normalization=norm_info,
        classification=class_info,
        subclass=subclass_info,
        clustering=dict(method="kmeans", seed=seed,
                        enrichment_floor=HCR_ENRICHMENT_FLOOR,
                        name_floor=HCR_NAME_FLOOR,
                        panel_15=list(HCR_PANEL_15), **clust_info),
        note=("X is raw transcript counts; layers['normalized'] is the p95 transform "
              "over every cell at once; layers['normalized_within_class'] is the same "
              "transform computed within each class and is what the cluster figures "
              "display and what cluster names are comparable to; obsm['X_cluster'] is "
              "what k-means actually saw. The two class-scoped matrices are NaN "
              "wherever nothing was fitted -- cells of no class in both, and genes "
              "outside a class's clustering space in X_cluster. Only layers"
              "['normalized'] covers every cell. Clusters are PER MOUSE and do not "
              "correspond across animals -- "
              "use the consensus clusters for any across-mouse analysis."),
    )
    if extra_uns:
        adata.uns["unmixing"].update(extra_uns)
    return adata
