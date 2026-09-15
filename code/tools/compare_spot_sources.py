"""Compare two arms of the same mouse: pairwise spot input against processed.

    python tools/compare_spot_sources.py \
        --pairwise  /results/pw/800792_cellxgene.csv \
        --processed /results/pr/800792_cellxgene.csv \
        --out       /results/compare

Answers the question the switch actually raises. The pairwise step loads spots only for
cells that survived ITS ROI filter (volume, soma classifier, edge, tile overlap), so the
processed arm carries extra cells. Whether that "ruins anything" splits into three
questions, and they have different answers:

  1. Do cells present in BOTH arms get the same counts?
     They should be near-identical. Unmixing is per round and per channel and knows
     nothing about cell identity, so a shared cell's counts can only move through
     spatial-neighbour effects in the crosstalk decision. Large moves here mean the
     switch changed the unmixing, which is not what it is meant to do.

  2. What are the ADDED cells?
     Below the classification count floor they land in `low_counts` and cost nothing.
     Carrying real signal, the ROI filter was discarding data -- worth knowing
     independently of this branch.

  3. Do per-cell LABELS flip on shared cells?
     Class is a mixture fitted to the population, so adding cells can move the boundary
     for a cell whose own counts did not change. This is the only route by which the
     switch can change a conclusion about a cell that was already there.

Nothing here filters or thresholds. It reports.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_FLOOR = 100        # annotate.MIN_CLASS_COUNTS -- below this, class is low_counts


def load(path):
    t = pd.read_csv(path, index_col=0)
    t.index = t.index.astype(str)
    return t.sort_index()


def compare(pw, pr):
    shared = pw.index.intersection(pr.index)
    genes = [c for c in pw.columns if c in pr.columns]
    rows = [
        dict(metric="cells, pairwise arm", value=len(pw)),
        dict(metric="cells, processed arm", value=len(pr)),
        dict(metric="cells in both", value=len(shared)),
        dict(metric="cells only in pairwise", value=len(pw.index.difference(pr.index))),
        dict(metric="cells only in processed", value=len(pr.index.difference(pw.index))),
        dict(metric="gene-round columns in both", value=len(genes)),
    ]

    a, b = pw.loc[shared, genes], pr.loc[shared, genes]
    same = int((a.to_numpy() == b.to_numpy()).all(1).sum())
    delta = (b.to_numpy() - a.to_numpy()).astype(float)
    tot_a = a.to_numpy().sum(1)
    rows += [
        dict(metric="shared cells identical across every gene", value=same),
        dict(metric="shared cells identical (%)",
             value=round(100 * same / max(len(shared), 1), 2)),
        dict(metric="max abs count change on a shared cell",
             value=float(np.abs(delta).max()) if delta.size else 0.0),
        dict(metric="median total-count change on shared cells (%)",
             value=round(float(np.median(
                 100 * delta.sum(1) / np.maximum(tot_a, 1))), 3) if delta.size else 0.0),
    ]

    added = pr.index.difference(pw.index)
    if len(added):
        at = pr.loc[added, genes].to_numpy().sum(1)
        st = pr.loc[shared, genes].to_numpy().sum(1)
        rows += [
            dict(metric="added cells: median total counts", value=float(np.median(at))),
            dict(metric="shared cells: median total counts", value=float(np.median(st))),
            dict(metric=f"added cells below the {COUNT_FLOOR}-count class floor",
                 value=int((at < COUNT_FLOOR).sum())),
            dict(metric="added cells below the floor (%)",
                 value=round(100 * float((at < COUNT_FLOOR).mean()), 2)),
        ]
    return pd.DataFrame(rows), shared, genes, added


def label_flips(pw, pr, shared):
    """Class and subclass per arm, then the cross-tab on shared cells only.

    Each arm is labelled on ITS OWN population, which is the point: the class call is a
    mixture fitted to whatever cells are present, so this measures the effect of the
    wider population on cells that were in both.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    out = {}
    for name, t in (("pairwise", pw), ("processed", pr)):
        t2, _ = A.rename_gene_aliases(t, warn=False)
        cls, info, _ = A.assign_class(t2)
        sub, _ = A.assign_subclass(t2)
        out[name] = pd.DataFrame({"class": cls.values, "subclass": sub.values},
                                 index=t2.index.astype(str))
        out[name + "_info"] = info
    j = out["pairwise"].loc[shared].join(out["processed"].loc[shared],
                                         lsuffix="_pw", rsuffix="_pr")
    cls_tab = pd.crosstab(j["class_pw"], j["class_pr"])
    sub_tab = pd.crosstab(j["subclass_pw"], j["subclass_pr"])
    agree = float((j["class_pw"] == j["class_pr"]).mean())
    return cls_tab, sub_tab, agree, out


def _plot(pw, pr, shared, added, genes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    a = pw.loc[shared, genes].to_numpy().sum(1)
    b = pr.loc[shared, genes].to_numpy().sum(1)
    axes[0].scatter(a, b, s=3, alpha=0.25, lw=0)
    lim = [1, float(max(a.max(), b.max(), 2))]
    axes[0].plot(lim, lim, color="crimson", lw=1)
    axes[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
                xlabel="total counts, pairwise arm",
                ylabel="total counts, processed arm",
                title=f"shared cells (n={len(shared):,})")

    if len(added):
        at = pr.loc[added, genes].to_numpy().sum(1)
        bins = np.logspace(0, np.log10(float(max(b.max(), at.max(), 10))), 50)
        axes[1].hist(b, bins=bins, alpha=0.6, label=f"shared (n={len(shared):,})")
        axes[1].hist(at, bins=bins, alpha=0.6, label=f"added (n={len(added):,})")
        axes[1].axvline(COUNT_FLOOR, color="k", ls="--", lw=1)
        axes[1].set(xscale="log", xlabel="total counts per cell", ylabel="cells",
                    title="cells the ROI filter had removed")
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "no added cells", ha="center",
                     transform=axes[1].transAxes)

    d = pr.loc[shared, genes].sum() - pw.loc[shared, genes].sum()
    rel = 100 * d / pw.loc[shared, genes].sum().clip(lower=1)
    axes[2].barh(range(len(genes)), rel.to_numpy())
    axes[2].set_yticks(range(len(genes)))
    axes[2].set_yticklabels(genes, fontsize=7)
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set(xlabel="change in total counts on shared cells (%)",
                title="per gene-round")
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairwise", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="compare")
    ap.add_argument("--no-labels", action="store_true",
                    help="skip class/subclass comparison (needs both class markers)")
    args = ap.parse_args(argv)

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)
    pw, pr = load(args.pairwise), load(args.processed)

    summary, shared, genes, added = compare(pw, pr)
    summary.to_csv(outp / "arm_comparison.csv", index=False)
    print(summary.to_string(index=False))

    per_gene = pd.DataFrame({
        "gene_round": genes,
        "total_pairwise": pw.loc[shared, genes].sum().to_numpy(),
        "total_processed": pr.loc[shared, genes].sum().to_numpy(),
        "total_processed_all_cells": pr[genes].sum().to_numpy()})
    per_gene["pct_change_shared"] = (
        100 * (per_gene["total_processed"] - per_gene["total_pairwise"])
        / per_gene["total_pairwise"].clip(lower=1)).round(3)
    per_gene["pct_added_by_new_cells"] = (
        100 * (per_gene["total_processed_all_cells"] - per_gene["total_processed"])
        / per_gene["total_processed"].clip(lower=1)).round(3)
    per_gene.to_csv(outp / "per_gene_change.csv", index=False)
    print("\n" + per_gene.to_string(index=False))

    if not args.no_labels:
        try:
            cls_tab, sub_tab, agree, _ = label_flips(pw, pr, shared)
            cls_tab.to_csv(outp / "class_flips_shared_cells.csv")
            sub_tab.to_csv(outp / "subclass_flips_shared_cells.csv")
            print(f"\nclass agreement on shared cells: {agree:.4f}")
            print(cls_tab.to_string())
        except Exception as exc:          # a single round has no marker pair
            print(f"\nlabel comparison skipped: {type(exc).__name__}: {exc}")

    fig = _plot(pw, pr, shared, added, genes)
    fig.savefig(outp / "arm_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nwritten to {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
