"""Standard cell x gene heatmaps, written to results/plots/ by the capsule.

Four figures, all from the capsule's OWN labels read back off the annotated AnnData --
never recomputed at plot time. That distinction is load-bearing: an earlier version of
these plots handed a cluster-label frame to a downstream helper in a different row order
than its cell-id order, which paired every label with the wrong cells and made correct
data look randomised. Reading `obs.cluster` directly removes the opportunity.

    cellxgene_all_std.png          all cells,   biology-grouped gene order
    cellxgene_all_rc.png           all cells,   acquisition (round x channel) order
    cellxgene_inhibitory_std.png   inhibitory,  biology-grouped
    cellxgene_inhibitory_rc.png    inhibitory,  acquisition order

Each shows raw counts beside the normalised matrix. Clustering runs on the normalised
matrix in both cases, so the two panels carry the same rows in the same order -- the
display choice sets only the colour scale.
"""
import numpy as np

#: Biology-grouped gene order. Subclass markers first, then neuropeptides and calcium
#: binding proteins, then the round-6 receptors, with the two class gates last so the
#: Slc17a7/Gad2 pair reads as a block against everything else.
STD_GENE_ORDER = ("GFP", "Pvalb", "Sst", "Vip", "Lamp5", "Npy", "Ndnf", "Cck", "Crh",
                  "Calb1", "Calb2", "Tac1", "Tac2", "Reln", "Pdyn", "Penk", "Pthlh",
                  "Hpse", "Mme", "Chat", "Sncg", "Cnr1", "Adra1a", "Adra1b", "Htr3a",
                  "Slc17a7", "Gad2")

#: Subclass block colours, and the order blocks are stacked in. Inhibitory first: on the
#: all-cells figure they are a small minority of the rows (11% on 800995), so putting
#: them at the top keeps them visible instead of buried under the excitatory block.
SUBCLASS_COLORS = {"Pvalb": "#D93137", "Sst": "#FF9900", "Vip": "#A45FBF",
                   "Lamp5": "#DA808C", "Inh": "#999999", "Exc": "#00A809"}
SUBCLASS_ORDER = ("Pvalb", "Sst", "Vip", "Lamp5", "Inh", "Exc")

#: A block thinner than this fraction of the axis cannot hold a rotated name inside its
#: colour bar, so its name is fanned out to the left on a leader line instead.
MIN_BLOCK_FRAC = 0.045
#: The fan needs a minimum vertical span of its own: confining it to the thin blocks'
#: true extent still collides when those blocks are only a few percent of the height.
MIN_FAN_FRAC = 0.16
#: A cluster thinner than this fraction of the axis gets no name on the right. 0.014 was
#: chosen by sweeping against a real 20-cluster inhibitory panel: 0.017 dropped the
#: smallest cluster (91 of 5,962 cells = 0.0153) while 0.014 names all 20 with zero text
#: overlaps. The caption reports the count this gate actually admits, so a dropped name
#: is stated rather than silently missing.
MIN_CLUSTER_FRAC = 0.014


def gene_order(var, kind="std"):
    """Column order for the heatmap: 'std' biology-grouped, or 'rc' acquisition order.

    `var` is `adata.var`, carrying a `gene` column alongside the round-channel-gene
    column names. Tac is matched to Tac1 -- the panel labels it Tac while the standard
    order uses the Tac1 synonym.
    """
    if kind == "rc":
        # sorted() on the tuple key, not np.argsort -- argsort on a list of tuples builds
        # a 2-D array and then fails on multi-dimensional index selection.
        return sorted(var.index, key=_rc_key)
    out = []
    for want in STD_GENE_ORDER:
        for col in var.index:
            if var.loc[col, "gene"] == ("Tac" if want == "Tac1" else want):
                out.append(col)
    return out + [c for c in var.index if c not in out]


def _rc_key(col):
    """Sort key putting columns in acquisition order: round number, then channel nm."""
    parts = str(col).split("-")
    try:
        return (int(parts[0].lstrip("Rr")), int(parts[1]))
    except (ValueError, IndexError):
        return (99, 99)


def block_layout(adata, cell_class=None):
    """Rows ordered by subclass block then cluster, with block and cluster extents.

    Returns (ordered_adata, clusters, blocks, block_boundaries) where `clusters` is
    [(name, start, end, colour)] and `blocks` is [(subclass, start, end, colour)].
    Cells within a cluster are ordered by total counts so each cluster reads as a
    brightness gradient rather than noise.
    """
    a = adata if cell_class is None else adata[adata.obs["class"].isin(cell_class)]
    a = a[a.obs.cluster_id >= 0]
    info = a.obs.groupby("cluster_id", observed=True).agg(name=("cluster", "first"))
    info["prefix"] = [str(n).split("-")[0] for n in info.name]
    info["index"] = [int(str(n).split("-")[1].split(" ")[0]) if "-" in str(n) else 0
                     for n in info.name]
    info["rank"] = [SUBCLASS_ORDER.index(p) if p in SUBCLASS_ORDER else 9
                    for p in info.prefix]
    info = info.sort_values(["rank", "index"])

    order, clusters, boundaries, starts = [], [], [], {}
    previous = None
    for cid in info.index:
        rows = a.obs.index[a.obs.cluster_id == cid]
        rows = a.obs.loc[rows].sort_values("total_counts").index
        prefix = info.loc[cid, "prefix"]
        if prefix != previous:
            if previous is not None:
                boundaries.append(len(order))
            starts[prefix] = len(order)
        previous = prefix
        start = len(order)
        order += list(rows)
        clusters.append((info.loc[cid, "name"], start, len(order),
                         SUBCLASS_COLORS.get(prefix, "#333333")))
    ends = dict(zip(list(starts), list(starts.values())[1:] + [len(order)]))
    blocks = [(p, starts[p], ends[p], SUBCLASS_COLORS.get(p, "#333333")) for p in starts]
    return a[order], clusters, blocks, boundaries


def _block_labels(ax, blocks, n_rows):
    """Subclass name on its own block; thin blocks fanned left on leader lines."""
    thin = [b for b in blocks if (b[2] - b[1]) / n_rows < MIN_BLOCK_FRAC]
    for name, start, end, colour in blocks:
        if (end - start) / n_rows >= MIN_BLOCK_FRAC:
            ax.text(-0.052, (start + end) / 2, name, rotation=90, va="center",
                    ha="center", fontsize=6.8, color=colour, fontweight="bold",
                    transform=ax.get_yaxis_transform(), clip_on=False)
    if not thin:
        return
    top = min(b[1] for b in thin)
    bottom = max(b[2] for b in thin)
    span = max(bottom - top, MIN_FAN_FRAC * n_rows)
    middle = (top + bottom) / 2
    # inset from both axis ends so a fanned name cannot land on the 0 / n tick labels
    low = min(max(middle - span / 2, 0.035 * n_rows), n_rows - span - 0.035 * n_rows)
    for (name, start, end, colour), y in zip(thin, np.linspace(low, low + span, len(thin))):
        ax.plot([-0.030, -0.046], [(start + end) / 2, y], color=colour, lw=0.5,
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.text(-0.050, y, name, va="center", ha="right", fontsize=6.2, color=colour,
                fontweight="bold", transform=ax.get_yaxis_transform(), clip_on=False)


def _panel(a, ax, columns, display, full_labels, clusters, blocks, boundaries,
           cluster_labels):
    from matplotlib.patches import Rectangle

    n = a.n_obs
    matrix = a[:, columns].layers["normalized"] if display == "normalized" else a[:, columns].X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    image = ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0,
                      vmax=(1.0 if display == "normalized" else 200),
                      interpolation="nearest")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(list(columns) if full_labels
                       else [a.var.loc[c, "gene"] for c in columns],
                       rotation=90, fontsize=(5.6 if full_labels else 6.4),
                       style=("normal" if full_labels else "italic"))
    ax.set_yticks([0, n])
    ax.set_yticklabels(["0", f"{n:,}"], fontsize=6.8)
    for _, _, end, _ in clusters[:-1]:
        ax.axhline(end, color="#2e8b57", lw=0.26, ls="--")
    for b in boundaries:
        ax.axhline(b, color="black", lw=0.9)
    for _, start, end, colour in blocks:
        ax.add_patch(Rectangle((-0.030, start), 0.019, end - start,
                               transform=ax.get_yaxis_transform(), facecolor=colour,
                               edgecolor="none", clip_on=False, zorder=5))
    _block_labels(ax, blocks, n)
    if cluster_labels:
        for name, start, end, colour in clusters:
            if (end - start) / n > MIN_CLUSTER_FRAC:
                ax.text(1.006, (start + end) / 2, name, va="center", ha="left",
                        fontsize=6.1, color=colour,
                        transform=ax.get_yaxis_transform(), clip_on=False)
    return image, n


def write_plots(adata, output_dir, mouse_id, rounds):
    """Write the four standard heatmaps into output_dir/plots/. Returns the filenames.

    Import of matplotlib is deferred so a capsule without it still completes the run --
    figures are a convenience, not a result the pipeline should be lost for.
    """
    from pathlib import Path
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"WARNING: plots not written ({exc}). pip install matplotlib")
        return []

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    orders = [("std", "standard gene order", gene_order(adata.var, "std")),
              ("rc", "round x channel order", gene_order(adata.var, "rc"))]
    written = []
    for cell_class, tag, show_clusters, figsize in [
            (None, "all", False, (11.8, 8.6)),
            (["inhibitory"], "inhibitory", True, (11.8, 9.2))]:
        a, clusters, blocks, boundaries = block_layout(adata, cell_class)
        if a.n_obs == 0:
            continue
        for suffix, order_name, columns in orders:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            fig.subplots_adjust(left=0.105, right=0.815, top=0.855, bottom=0.185,
                                wspace=0.72)
            for j, display in enumerate(["raw", "normalized"]):
                image, n = _panel(a, axes[j], columns, display, suffix == "rc",
                                  clusters, blocks, boundaries, show_clusters)
                axes[j].set_title(f"{display} \u00b7 {n:,} cells", loc="left", fontsize=8)
                if j == 0:
                    axes[j].set_ylabel("Cells, grouped by cluster", fontsize=7.5)
                bar = fig.colorbar(image, ax=axes[j], orientation="horizontal",
                                   pad=0.25, fraction=0.045)
                bar.set_label("fraction of gene 95th pct" if display == "normalized"
                              else "transcript count (clip 200)", fontsize=6.8)
            label = "all cells" if cell_class is None else "inhibitory cells"
            census = ", ".join(f"{p} {e - s:,}" for p, s, e, _ in blocks)
            fig.suptitle(f"{mouse_id} \u00b7 {label} \u00b7 {len(clusters)} clusters \u00b7 "
                         f"{len(rounds)} rounds, {adata.n_vars} genes \u00b7 {order_name}",
                         x=0.105, ha="left", fontsize=10.5, y=0.972)
            # The caption is generated from the same predicate the drawing uses, so it
            # cannot claim a name is shown that the size gate actually dropped.
            n_named = sum(1 for _, s, e, _ in clusters
                          if (e - s) / a.n_obs > MIN_CLUSTER_FRAC)
            if show_clusters:
                extra = (f"All {n_named} cluster names are on the right, each at its own "
                         f"cluster's midpoint." if n_named == len(clusters) else
                         f"{n_named} of {len(clusters)} cluster names fit on the right; "
                         f"the remainder are too thin to label.")
            else:
                share = 100 * sum(e - s for p, s, e, _ in blocks if p != "Exc") / a.n_obs
                extra = (f"Cluster names are omitted here: the inhibitory blocks are "
                         f"{share:.0f}% of the height. See the inhibitory figure.")
            fig.text(0.105, 0.936,
                     "Rows are cells, grouped by the cluster the capsule assigned \u2014 labels "
                     "read from the .h5ad, not recomputed. The coloured bar spans each\n"
                     "subclass block at its true position; a leader line connects a block too "
                     f"thin for an in-place name.  Block sizes: {census}.\n{extra}",
                     ha="left", va="top", fontsize=6.7, color="#555555")
            name = f"cellxgene_{tag}_{suffix}.png"
            fig.savefig(plot_dir / name, dpi=170, bbox_inches="tight")
            plt.close(fig)
            written.append(f"plots/{name}")
    print(f"  plots: {len(written)} figures -> {plot_dir}", flush=True)
    return written
