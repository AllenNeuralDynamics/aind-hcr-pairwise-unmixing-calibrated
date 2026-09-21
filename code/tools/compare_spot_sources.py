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


def check_cells(wanted, pw, pr, genes):
    """Membership of a named cell set in each arm, and what the verdict means.

    Written for one question: a set of cells present in an upstream table but absent
    from a shipped cell x gene table. There are three outcomes and they point at
    different causes, so the useful output is the partition, not a single number.

        in BOTH arms        the cells are not missing from unmixing at all. Whatever
                            dropped them happened downstream, or the two tables use
                            different id spaces and the join is what failed.
        PROCESSED only      confirmed: the pairwise step never loaded spots for these
                            cells, because its ROI filter had already removed them.
                            Reading the processed table recovers them.
        NEITHER arm         not an ROI-filter effect. The cell x gene table is a pivot
                            over detected spots, so a cell with no spots above
                            threshold in any round has no row -- and coregistration
                            selects cells on in vivo evidence, not on HCR signal, so
                            a coregistered cell with no transcripts detected is a
                            normal outcome rather than a loss.
    """
    wanted = pd.Index(pd.unique(pd.Series(wanted).astype(str)))
    in_pw, in_pr = wanted.isin(pw.index), wanted.isin(pr.index)
    both = wanted[in_pw & in_pr]
    pr_only = wanted[~in_pw & in_pr]
    pw_only = wanted[in_pw & ~in_pr]
    neither = wanted[~in_pw & ~in_pr]

    rows = [dict(group="requested", n=len(wanted), median_total_counts=np.nan),
            dict(group="in both arms", n=len(both),
                 median_total_counts=float(np.median(
                     pr.loc[both, genes].to_numpy().sum(1))) if len(both) else np.nan),
            dict(group="processed arm ONLY (ROI filter recovered)", n=len(pr_only),
                 median_total_counts=float(np.median(
                     pr.loc[pr_only, genes].to_numpy().sum(1))) if len(pr_only) else np.nan),
            dict(group="pairwise arm only", n=len(pw_only),
                 median_total_counts=float(np.median(
                     pw.loc[pw_only, genes].to_numpy().sum(1))) if len(pw_only) else np.nan),
            dict(group="in NEITHER arm", n=len(neither), median_total_counts=np.nan)]
    return pd.DataFrame(rows), dict(both=both, processed_only=pr_only,
                                    pairwise_only=pw_only, neither=neither)


def load_cell_list(path, column=None):
    """Cell ids from a one-per-line list, or from a named or guessed column of a CSV."""
    p = Path(path)
    txt = [s.strip() for s in p.read_text().splitlines() if s.strip()]
    known = ("cell_id", "hcr_id", "roi_id", "cell", "id")
    if column is None and len(txt) > 1 and "," not in txt[0]:
        # A one-column CSV has no comma on its header line either, so a bare
        # split-on-newline swallows the header as a cell id. It then lands in "in
        # neither arm" and reads as one genuinely missing cell -- which is how this
        # was found: 294 requested from a 293-row file.
        if txt[0].lower() in known:
            txt = txt[1:]
        return txt
    t = pd.read_csv(p)
    if column is not None:
        return t[column].astype(str).tolist()
    for cand in ("cell_id", "hcr_id", "roi_id", "cell", "id"):
        if cand in t.columns:
            print(f"  --cells: using column '{cand}' ({len(t)} rows)")
            return t[cand].astype(str).tolist()
    raise SystemExit(
        f"{path}: pass --cells-column; no obvious id column among {list(t.columns)[:12]}")


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
    ap.add_argument("--cells", default=None, metavar="FILE",
                    help="CSV or one-per-line list of cell ids to test for membership "
                         "in each arm. Use this to ask whether a specific set of cells "
                         "-- e.g. coregistered cells absent from a shipped cell x gene "
                         "table -- is absent because the pairwise step's ROI filter "
                         "removed them. See check_cells().")
    ap.add_argument("--cells-column", default=None,
                    help="column holding the cell id, when --cells is a wide CSV")
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

    if args.cells:
        wanted = load_cell_list(args.cells, args.cells_column)
        verdict, groups = check_cells(wanted, pw, pr, genes)
        verdict.to_csv(outp / "cells_of_interest_verdict.csv", index=False)
        for name, idx in groups.items():
            if len(idx):
                pd.Series(idx, name="cell_id").to_csv(
                    outp / f"cells_{name}.csv", index=False)
        print("\n" + verdict.to_string(index=False))
        n_rec = len(groups["processed_only"])
        n_none = len(groups["neither"])
        if n_rec:
            print(f"\n=> {n_rec} of these cells exist in the processed arm and NOT in the "
                  f"pairwise arm: the pairwise step's ROI filter is why they were absent.")
        if n_none:
            print(f"=> {n_none} are in neither arm: not an ROI-filter effect. Either they "
                  f"have no spots above threshold in this round, or the id spaces differ.")
        if len(groups["both"]):
            print(f"=> {len(groups['both'])} are in BOTH arms, so unmixing did not drop "
                  f"them; look downstream of the cell x gene table, or at the join.")

    fig = _plot(pw, pr, shared, added, genes)
    fig.savefig(outp / "arm_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nwritten to {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
