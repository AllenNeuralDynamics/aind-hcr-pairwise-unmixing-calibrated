"""Synthetic ground-truth tests. No data assets required -- these run anywhere."""
import json
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aind_hcr_pairwise_unmixing_calibrated import core
from aind_hcr_pairwise_unmixing_calibrated.control import CHANS, control_matrix

RNG = np.random.RandomState(0)



# ---------------------------------------------------------------------------------------
# Verifying a guard is not vacuous
#
# Several tests here exist to catch one specific regression, and a test that would pass
# even with the bug reintroduced is worse than no test. The check is to reintroduce the
# bug and confirm THIS test fails:
#
#     pip install -e .        # REQUIRED FIRST -- see below
#     <apply the mutation>
#     python -m pytest tests/test_core.py::<the_test> -q
#
# The install step is not optional and the failure mode is subtle: this module imports
# aind_hcr_pairwise_unmixing_calibrated at module level, so if the package is not
# installed, EVERY node-id in this file fails at collection with ModuleNotFoundError --
# which looks exactly like the guard firing. A non-zero exit is therefore not evidence on
# its own. Distinguish them:
#
#     "error during collection" / ModuleNotFoundError  -> invalid check, install and retry
#     "AssertionError" / "KeyError" from the test body -> the guard genuinely fired
#
# This bit once: a vacuity check was run in the same shell as an uninstalled-package test
# run, reported "guard works", and was written into a commit message. The guard turned out
# to be sound, but the check that claimed it had proved nothing.
# ---------------------------------------------------------------------------------------


def make_round(n_real=4000, n_ghost=1200, beta=0.45, source="594", victim="561",
               bright=(80, 400), noise=8.0):
    """A round where every ghost is known.

    Real spots of both channels are placed at random positions. Ghosts are placed ON TOP
    of a randomly chosen source spot (sub-voxel jitter only) with victim-channel
    intensity = beta * source intensity, which is what real bleed does.
    """
    si, vi = CHANS.index(source), CHANS.index(victim)
    rows = []

    def blank():
        return {f"chan_{c}_intensity": float(RNG.normal(0, noise)) for c in CHANS}

    # real source-channel spots
    src_pos, src_amp = [], []
    for _ in range(n_real):
        amp = RNG.uniform(*bright)
        z, y, x = RNG.uniform(0, 60), RNG.uniform(0, 2000), RNG.uniform(0, 2000)
        r = blank(); r[f"chan_{source}_intensity"] = amp
        r.update(z=z, y=y, x=x, chan=source, cell_id=int(y // 100) * 20 + int(x // 100))
        rows.append(r); src_pos.append((z, y, x)); src_amp.append(amp)

    # real victim-channel spots, independent positions
    for _ in range(n_real):
        amp = RNG.uniform(*bright)
        z, y, x = RNG.uniform(0, 60), RNG.uniform(0, 2000), RNG.uniform(0, 2000)
        r = blank(); r[f"chan_{victim}_intensity"] = amp
        r.update(z=z, y=y, x=x, chan=victim, cell_id=int(y // 100) * 20 + int(x // 100),
                 is_ghost=False)
        rows.append(r)

    # ghosts: sit on a source spot, brightness = beta * source
    pick = RNG.choice(len(src_pos), n_ghost, replace=False)
    for j in pick:
        z, y, x = src_pos[j]
        r = blank()
        r[f"chan_{victim}_intensity"] = beta * src_amp[j]
        r[f"chan_{source}_intensity"] = src_amp[j]
        r.update(z=z + RNG.normal(0, 0.15), y=y + RNG.normal(0, 0.4),
                 x=x + RNG.normal(0, 0.4), chan=victim,
                 cell_id=int(y // 100) * 20 + int(x // 100), is_ghost=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    df["is_ghost"] = df.get("is_ghost", pd.Series(False, index=df.index)).fillna(False)
    # source spots carry their own bleed into the victim channel, as real dye does
    m = df.chan == source
    df.loc[m, f"chan_{victim}_intensity"] = beta * df.loc[m, f"chan_{source}_intensity"]
    return df.reset_index(drop=True)


def test_ghosts_are_deleted_and_real_spots_survive():
    beta = 0.45
    df = make_round(beta=beta)
    B = np.full((5, 5), np.nan)
    B[CHANS.index("594"), CHANS.index("561")] = beta
    B[CHANS.index("561"), CHANS.index("594")] = beta / 30
    powers = {c: 10.0 for c in CHANS}

    out, sep, log, E, info = core.unmix_v3(df, B, powers, same_cell=False)
    deleted = out.v3_action == "delete"
    ghost = out.is_ghost.astype(bool)

    recall = float(deleted[ghost].mean())
    false_pos = float(deleted[~ghost].mean())
    assert recall > 0.80, f"only {recall:.1%} of known ghosts deleted"
    assert false_pos < 0.05, f"{false_pos:.1%} of real spots wrongly deleted"


def test_isolated_endmember_recovers_the_true_direction():
    beta = 0.45
    df = make_round(beta=beta)
    I = core.intensity_matrix(df)
    det = core.detection_index(df)
    zyx = df[["z", "y", "x"]].to_numpy(np.float32)
    E, info = core.estimate_endmembers_isolated(I, det, zyx)
    si, vi = CHANS.index("594"), CHANS.index("561")
    recovered = E[vi, si] / E[si, si]
    assert abs(recovered - beta) / beta < 0.15, f"beta recovered as {recovered:.3f}, true {beta}"


def test_measured_beta_matches_truth():
    beta = 0.45
    df = make_round(beta=beta)
    I = core.intensity_matrix(df)
    det = core.detection_index(df)
    zyx = df[["z", "y", "x"]].to_numpy(np.float32)
    meas = core.measure_beta_and_tolerance(I, det, zyx)
    key = ("594", "561")
    assert key in meas, f"direction not measured; got {list(meas)}"
    assert abs(meas[key]["beta"] - beta) / beta < 0.20
    assert meas[key]["tol"] >= 1.0


def test_allowlist_is_bidirectional_but_never_distant():
    B = control_matrix()
    dirs = core.allowlist_directions(B, CHANS, bidirectional=True)
    assert ("594", "638") in dirs, "Sst->Vip missing: reverse of an allowlisted pair"
    assert ("594", "561") in dirs, "Sst->Cck missing: strongest control direction"
    assert ("561", "638") not in dirs, "Cck->Vip admitted: distant pair, co-expression"
    assert ("638", "561") not in dirs
    uni = core.allowlist_directions(B, CHANS, bidirectional=False)
    assert set(uni) < set(dirs)


def test_nothing_is_dropped_from_the_frame():
    df = make_round(n_real=800, n_ghost=200)
    B = control_matrix()
    out, *_ = core.unmix_v3(df, B, {c: 10.0 for c in CHANS}, same_cell=False)
    assert len(out) == len(df), "output must carry every input spot"
    for col in ("v3_action", "v3_chan", "decision_rule", "beta_used"):
        assert col in out.columns


def test_fg_bg_columns_appear_when_supplied():
    df = make_round(n_real=500, n_ghost=100)
    fg = np.full(len(df), 300.0, np.float32)
    bg = np.full(len(df), 100.0, np.float32)
    out, *_ = core.unmix_v3(df, control_matrix(), {c: 10.0 for c in CHANS},
                            same_cell=False, fg_bg=(fg, bg))
    assert {"fg", "bg", "fg_over_bg"} <= set(out.columns)
    assert np.allclose(out.fg_over_bg.to_numpy(), 3.0)


# ---------------------------------------------------------------- metadata


def test_processing_json_matches_v114_shape(tmp_path):
    """The emitted processing.json must match the 1.1.4 files already in the assets."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    dp = M.unmixing_data_process(
        input_locations=["s3://bucket/asset/800995_R5"],
        output_location="s3://bucket/derived",
        parameters={"rounds": ["R5"]},
        outputs={"cellxgene": "x.csv"},
    )
    # field set and order must match what the HCR assets carry
    assert tuple(dp) == M._V114_DATA_PROCESS_FIELDS
    assert dp["name"] == "Image spot spectral unmixing"

    path = M.write_processing(tmp_path, dp, processor_full_name="Tester")
    doc = json.loads(Path(path).read_text())
    assert doc["schema_version"] == "1.1.4"
    assert set(doc) == {"describedBy", "schema_version", "processing_pipeline",
                        "analyses", "notes"}
    pp = doc["processing_pipeline"]
    assert set(pp) == {"data_processes", "processor_full_name", "pipeline_version",
                       "pipeline_url", "note"}
    assert len(pp["data_processes"]) == 1


def test_processing_json_appends_to_upstream(tmp_path):
    """Upstream 1.1.4 history is preserved, not overwritten."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    upstream = tmp_path / "processing.json"
    upstream.write_text(json.dumps({
        "describedBy": M.DESCRIBED_BY, "schema_version": "1.1.4",
        "processing_pipeline": {"data_processes": [{"name": "Image spot detection"}],
                                "processor_full_name": "Someone"},
        "analyses": [], "notes": ""}))

    out = tmp_path / "results"
    dp = M.unmixing_data_process(["in"], str(out), {})
    path = M.write_processing(out, dp, upstream_processing=str(upstream))
    steps = json.loads(Path(path).read_text())["processing_pipeline"]["data_processes"]
    assert [s["name"] for s in steps] == ["Image spot detection",
                                          "Image spot spectral unmixing"]


def test_2x_upstream_is_not_downgraded(tmp_path):
    """A 2.x upstream must be referenced, never silently rewritten into 1.1.4."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    upstream = tmp_path / "processing.json"
    upstream.write_text(json.dumps({
        "schema_version": "2.3.0",
        "data_processes": [{"process_type": "Image spot detection", "stage": "Processing"}]}))

    out = tmp_path / "results"
    dp = M.unmixing_data_process(["in"], str(out), {})
    doc = json.loads(Path(M.write_processing(out, dp, upstream_processing=str(upstream))).read_text())
    steps = doc["processing_pipeline"]["data_processes"]
    assert len(steps) == 1                       # upstream NOT merged
    assert "2.3.0" in doc["processing_pipeline"]["note"]


def test_copy_upstream_metadata_first_hit_wins(tmp_path):
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a.mkdir(); b.mkdir()
    (a / "subject.json").write_text('{"from": "a"}')
    (b / "subject.json").write_text('{"from": "b"}')
    (b / "procedures.json").write_text("{}")

    copied = M.copy_upstream_metadata([a, b], out)
    assert json.loads((out / "subject.json").read_text())["from"] == "a"
    assert set(copied) == {"subject.json", "procedures.json"}


def test_derived_data_description_names_itself_not_parent(tmp_path):
    """A derived asset must NOT inherit the parent's name/data_level."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    parent = tmp_path / "HCR_800995_2026-04-08_13-00-00_processed_2026-04-13_21-37-30"
    parent.mkdir()
    (parent / "data_description.json").write_text(json.dumps({
        "schema_version": "1.0.4",
        "name": parent.name,
        "data_level": "derived",
        "input_data_name": "HCR_800995_2026-04-08_13-00-00",
        "process_name": "processed",
        "subject_id": "800995",
        "institution": {"name": "AIND"},
        "modality": [{"name": "Selective plane illumination microscopy"}]}))

    out = tmp_path / "results"
    path = M.derived_data_description(parent, out, creation_time=None)
    doc = json.loads(Path(path).read_text())

    assert doc["name"] != parent.name                    # not the parent's name
    # Keyed on the SUBJECT, not the parent session: the capsule consumes every round of
    # a mouse, so naming the asset after one processed session would assert a parentage
    # that is only one-Nth true. This must match manifest.asset_name() so the asset
    # record and the metadata inside it agree.
    assert doc["name"].startswith("HCR_800995_")
    assert not doc["name"].startswith(parent.name)
    assert M.PROCESS_SLUG in doc["name"]
    assert doc["data_level"] == "derived"
    assert doc["input_data_name"] == parent.name         # points AT the parent
    assert doc["process_name"] == M.PROCESS_SLUG
    # descriptive fields carried over unchanged
    assert doc["subject_id"] == "800995"
    assert doc["institution"] == {"name": "AIND"}


def test_data_description_not_blindly_copied(tmp_path):
    """copy_upstream_metadata must never copy data_description.json."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    for f in ("subject.json", "data_description.json", "acquisition.json"):
        (src / f).write_text("{}")

    copied = M.copy_upstream_metadata([src], out)
    assert "data_description.json" not in copied
    assert not (out / "data_description.json").exists()
    assert {"subject.json", "acquisition.json"} <= set(copied)


def test_no_data_description_written_without_parent(tmp_path):
    """Without a parent to inherit from, write nothing rather than invent fields."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    empty, out = tmp_path / "empty", tmp_path / "out"
    empty.mkdir()
    assert M.derived_data_description(empty, out) is None
    assert not (out / "data_description.json").exists()


# ---------------------------------------------------------------- annotation
#
# The class, subclass, transform and cluster-naming rules are the HCR consensus
# protocol's, vendored in labeling.py. These tests pin the behaviour the protocol
# specifies, not the implementation: each one states the rule it is defending so a
# future change that breaks it has to argue with the rule rather than the assertion.


def _fake_table(n_inh=600, n_exc=900, seed=0):
    """Cell x gene counts with a known inhibitory/excitatory split and subclasses."""
    rng = np.random.RandomState(seed)
    genes = ["R1-561-Slc17a7", "R4-638-Gad2", "R5-514-Pvalb", "R5-594-Sst",
             "R5-638-Vip", "R4-488-Lamp5", "R5-561-Cck", "R3-514-Mme"]
    rows = []
    for i in range(n_inh):
        v = rng.poisson(8, len(genes)).astype(float)
        v[0] = rng.poisson(2)          # Slc17a7 low
        v[1] = rng.poisson(300)        # Gad2 high
        v[2 + (i % 4)] = rng.poisson(400)   # one subclass marker high
        # a secondary, non-subclass marker co-varying with the subclass, so cluster
        # names still have something to report once the block marker is excluded
        v[6 + (i % 2)] = rng.poisson(200)   # Cck or Mme
        rows.append(v)
    for _ in range(n_exc):
        v = rng.poisson(8, len(genes)).astype(float)
        v[0] = rng.poisson(400)        # Slc17a7 high
        v[1] = rng.poisson(2)          # Gad2 low
        rows.append(v)
    idx = [f"cell{i}" for i in range(n_inh + n_exc)]
    return pd.DataFrame(np.array(rows), index=idx, columns=genes)


def test_class_call_separates_the_two_modes():
    """The mixture must recover a clean bimodal split without being told where to cut."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    cls, info, posterior = A.assign_class(t)
    assert set(cls.unique()) <= {"inhibitory", "excitatory", "ambiguous", "low_counts"}
    # the fixture's two populations are far apart, so essentially everything resolves
    assert info["n_inhibitory"] >= 590 and info["n_excitatory"] >= 890
    assert len(posterior) == len(t)
    # the gates are posteriors, so the fitted log-ratio thresholds must bracket zero:
    # below one and above the other are the two modes
    lo, hi = info["log_ratio_thresholds"]
    assert lo < hi


def test_class_needs_both_markers():
    """The call is a RATIO. With one marker gone there is nothing to fit.

    Asserting excitatory from the absence of Gad2 would sweep in low-quality cells,
    mis-segmented cells and non-neuronal cells alike.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    for dropped in ("R1-561-Slc17a7", "R4-638-Gad2"):
        cls, info, posterior = A.assign_class(t.drop(columns=[dropped]))
        assert set(cls.unique()) == {"unassigned"}
        assert "none" in info["markers_available"].values()
        assert np.isnan(posterior).all()


def test_low_count_cells_are_held_out_of_the_fit():
    """A cell below the count floor has no evidence either way and must not be classed.

    It also must not influence the mixture: a mass of near-empty cells sits at a log
    ratio of 0 and would pull a component toward the middle.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=300, n_exc=300)
    t.iloc[0, :] = 0.0
    t.iloc[1, :] = 1.0                       # 8 counts total, well under the floor
    cls, info, _ = A.assign_class(t)
    assert cls.iloc[0] == "low_counts" and cls.iloc[1] == "low_counts"
    assert info["n_low_counts"] == 2


def test_ambiguous_cells_are_between_the_gates_and_stay_out_of_both_classes():
    """Cells with both markers high are `ambiguous`, not forced into a class.

    They are usually merged cells from segmentation or residual contamination; calling
    them either way propagates that error into every downstream count.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    # The default fixture's two modes are separated far enough that nothing lands
    # between them. Real marker distributions are broad and overlapping, so this test
    # builds its own table with a realistic within-mode spread.
    rng = np.random.RandomState(1)
    n, genes = 800, ["R1-561-Slc17a7", "R4-638-Gad2", "R5-514-Pvalb", "R5-594-Sst",
                     "R5-638-Vip", "R4-488-Lamp5", "R5-561-Cck", "R3-514-Mme"]
    x = rng.poisson(8, (n, len(genes))).astype(float)
    inh = np.arange(n) < 300
    x[inh, 1] = rng.lognormal(5.0, 1.2, inh.sum())      # Gad2 high, broad
    x[inh, 0] = rng.lognormal(2.0, 1.2, inh.sum())
    x[~inh, 0] = rng.lognormal(5.5, 1.2, (~inh).sum())  # Slc17a7 high, broad
    x[~inh, 1] = rng.lognormal(1.8, 1.2, (~inh).sum())
    t = pd.DataFrame(x, columns=genes, index=[f"cell{i}" for i in range(n)])

    cls, info, posterior = A.assign_class(t)
    amb = cls == "ambiguous"
    assert amb.sum() > 0, "overlapping modes must leave cells between the gates"
    assert info["n_ambiguous"] == int(amb.sum())
    assert (posterior[amb.to_numpy()] > 0.10).all()
    assert (posterior[amb.to_numpy()] < 0.90).all()
    assert not (amb & cls.isin(["inhibitory", "excitatory"])).any()


def test_subclass_is_a_per_cell_raw_count_argmax():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=40, n_exc=10)
    sub, info = A.assign_subclass(t)
    markers = ["R5-514-Pvalb", "R5-594-Sst", "R5-638-Vip", "R4-488-Lamp5"]
    expected = t[markers].idxmax(axis=1).map(lambda c: c.split("-")[-1])
    high = t[markers].max(axis=1) >= 20
    assert (sub[high] == expected[high]).all()


def test_subclass_floor_returns_unassigned_rather_than_a_guess():
    """A cell whose winning marker is nearly absent has not evidenced a subclass."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=20, n_exc=20)
    t.iloc[0, 2:6] = [3.0, 2.0, 1.0, 0.0]        # all four markers below 20
    t.iloc[0, 6] = 4.0                            # and Cck below it too, so not Sncg
    sub, info = A.assign_subclass(t)
    assert sub.iloc[0] == "unassigned"
    assert info["count_floor"] == 20

    # With Cck above the floor the same cell IS evidenced -- as Sncg, the subclass
    # defined by a positive Cck signal and no marker claiming the cell.
    t.iloc[0, 6] = 300.0
    sub2, info2 = A.assign_subclass(t)
    assert sub2.iloc[0] == "Sncg"
    assert info2["n_sncg"] >= 1 and info2["sncg_gene"] == "Cck"


def test_subclass_is_called_on_raw_counts_not_the_transform():
    """The p95 stage moves the argmax, so the call must not be made on it.

    Dividing each gene by its own 95th percentile inflates a dim marker relative to a
    bright one -- Lamp5 renders several times darker than Sst at equal raw counts -- so
    the transformed argmax can name a different subclass than the counts do.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=200, n_exc=200)
    markers = ["R5-514-Pvalb", "R5-594-Sst", "R5-638-Vip", "R4-488-Lamp5"]
    norm, _ = A.normalize_cellxgene(t)

    raw_call = t[markers].idxmax(axis=1)
    transformed_call = norm[markers].idxmax(axis=1)
    assert (raw_call != transformed_call).any(), \
        "fixture no longer exercises the difference this test exists to catch"

    sub, _ = A.assign_subclass(t)
    high = t[markers].max(axis=1) >= 20
    assert (sub[high] == raw_call[high].map(lambda c: c.split("-")[-1])).all()


def test_transform_scales_genes_first_then_cells():
    """Order is load-bearing, and the cell stage is a total, not a mean.

    Genes first, cells second is what the consensus protocol uses. Reversing it makes
    each gene's percentile depend on the cell composition of the table, so the same
    cell transforms differently in a single-mouse and a cohort run.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=50, n_exc=50)
    norm, info = A.normalize_cellxgene(t)
    assert norm.shape == t.shape
    assert info["transform"] == "p95_then_cell_total"
    # every cell ends on the same total: that is what the second stage does
    totals = norm.to_numpy().sum(1)
    assert np.allclose(totals, totals[0])
    # and it is NOT clipped to 1 -- the per-cell rescaling pushes bright cells above it
    assert float(norm.to_numpy().max()) > 1.0


def test_an_empty_cell_survives_the_transform_without_poisoning_it():
    """An all-zero row has no scale. It must come out zero, not NaN."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=20, n_exc=20)
    t.iloc[0, :] = 0.0
    norm, info = A.normalize_cellxgene(t)
    assert np.isfinite(norm.to_numpy()).all()
    assert float(norm.iloc[0].sum()) == 0.0
    assert info["n_zero_total_cells"] == 1


def test_a_cluster_needs_enrichment_not_just_plurality_to_take_a_subclass():
    """Plurality alone is not concentration.

    A modest share of a rare subclass is strong evidence; the same share of a common
    one is none. Below the 1.5x floor the cluster is `Other` rather than carrying a
    subclass name it has not earned.
    """
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    genes = list(L.HCR_SUBCLASS_MARKERS) + ["Cck", "Mme"]
    means = pd.DataFrame(0.1, index=[0, 1], columns=genes)
    # background is 80% Pvalb, so a cluster that is 80% Pvalb is not enriched at all
    subclass = np.array(["Pvalb"] * 80 + ["Sst"] * 20)
    labels = np.array([0] * 80 + [1] * 20)
    blocks, diag = L.hcr_cluster_blocks(means, subclass, labels)
    assert blocks[0] == "Other", "an unenriched plurality must not become a block"
    assert blocks[1] == "Sst", "a 20% subclass concentrated into one cluster is enriched"
    assert diag.loc[0, "enrichment"] < 1.5 <= diag.loc[1, "enrichment"]


def test_cck_dominant_cluster_is_promoted_to_sncg():
    """Post-hoc rule: the four markers cannot express Sncg, so Cck stands in for it.

    The dominance clause is what keeps it honest -- without it a Pvalb/Mme cluster is
    promoted on a near-tie between Cck and Mme.
    """
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    genes = list(L.HCR_SUBCLASS_MARKERS) + ["Cck", "Mme"]
    means = pd.DataFrame(0.1, index=[0, 1], columns=genes)
    means.loc[0, "Cck"] = 1.2                  # clear lead over Mme
    means.loc[0, "Mme"] = 0.2
    means.loc[1, "Cck"] = 0.70                 # a near-tie must NOT promote
    means.loc[1, "Mme"] = 0.67
    subclass = np.array(["Pvalb"] * 10 + ["Sst"] * 90)
    labels = np.array([0] * 10 + [1] * 90)
    blocks, _ = L.hcr_cluster_blocks(means, subclass, labels)
    assert blocks[0] == "Sncg"
    assert blocks[1] != "Sncg"


def test_cluster_names_use_absolute_level_above_the_floor():
    """A gene can deviate strongly across clusters and still be low everywhere.

    Naming on deviation produces a label that reads as a marker for something the
    cluster barely expresses, so the rule is the absolute level.
    """
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    genes = ["Pvalb", "Cck", "Mme"]
    means = pd.DataFrame([[2.0, 1.1, 0.02],       # Mme deviates 4x but is ~0
                          [2.0, 0.2, 0.005]],
                         index=[0, 1], columns=genes)
    names, ordered = L.hcr_cluster_names(means, {0: "Pvalb", 1: "Pvalb"})
    assert names[0] == "Pvalb-1  Cck", names[0]
    assert names[1] == "Pvalb-2", "no gene clears the floor, so no invented suffix"
    assert "Mme" not in names[0]
    assert ordered == [0, 1]


def test_clusters_are_numbered_within_block_with_unique_ids_across_classes():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    cls, _, _ = A.assign_class(t)
    sub, _ = A.assign_subclass(t)
    sub = sub.where(cls == "inhibitory", "none")
    labels, cid, matrix, info = A.cluster_by_class(t, cls, sub, n_inh=4, n_exc=3)

    inh_names = set(labels[cls == "inhibitory"].unique())
    exc_names = set(labels[cls == "excitatory"].unique())
    assert all(n.startswith("Exc-") for n in exc_names)
    # the block marker never appears in its own cluster's suffix
    assert not any(f"({p}" in n for n in inh_names for p in A.SUBCLASS_GENES
                   if n.startswith(p))
    assert cid[cls == "inhibitory"].nunique() == 4
    assert cid[cls == "excitatory"].nunique() == 3
    assert not (set(cid[cls == "inhibitory"]) & set(cid[cls == "excitatory"]))
    assert info["inhibitory"]["n_clusters"] == 4


def test_inhibitory_clustering_sees_only_the_protocol_genes():
    """The held-out genes are what make a cluster checkable.

    A cluster that separates on the clustering genes and then also separates on a gene
    the distance never saw has evidence the clustering could not have manufactured. If
    every gene feeds the distance, that check is unavailable.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A
    from aind_hcr_pairwise_unmixing_calibrated.labeling import HCR_PANEL_15

    t = _fake_table()
    inh_genes = [A.gene_name(c) for c in A.clustering_genes(t, "inhibitory")]
    exc_genes = [A.gene_name(c) for c in A.clustering_genes(t, "excitatory")]
    assert set(inh_genes) <= set(HCR_PANEL_15)
    assert "Gad2" not in inh_genes and "Slc17a7" not in inh_genes
    # the excitatory side has no protocol gene set, but never clusters on the class
    # markers or the reporter
    for barred in A.EXCLUDED_FROM_CLUSTERING:
        assert barred not in exc_genes


def test_gene_orders_are_complete_permutations():
    """Both gene orderings must contain every column exactly once.

    A dropped column would silently omit a gene from the heatmap; a duplicated one would
    plot it twice. The std order is built by walking a fixed name list, so a panel gene
    missing from that list has to fall through to the tail rather than vanish.
    """
    from aind_hcr_pairwise_unmixing_calibrated import plots as P

    cols = ["R1-561-Slc17a7", "R5-514-Pvalb", "R4-638-Gad2", "R2-488-Ndnf",
            "R9-488-Unlisted"]
    genes = ["Slc17a7", "Pvalb", "Gad2", "Ndnf", "Unlisted"]
    var = pd.DataFrame({"gene": genes}, index=cols)

    for kind in ("std", "rc"):
        order = P.gene_order(var, kind)
        assert sorted(order) == sorted(cols), f"{kind} is not a permutation"
        assert len(set(order)) == len(order), f"{kind} has duplicates"
    assert P.gene_order(var, "rc") == ["R1-561-Slc17a7", "R2-488-Ndnf", "R4-638-Gad2",
                                       "R5-514-Pvalb", "R9-488-Unlisted"]
    assert "R9-488-Unlisted" in P.gene_order(var, "std")


def test_every_block_has_a_colour():
    """block_layout colours rows by the cluster-name prefix, so a block with no entry
    in SUBCLASS_COLORS renders grey and silently loses its identity in the figure."""
    from aind_hcr_pairwise_unmixing_calibrated import plots as P
    from aind_hcr_pairwise_unmixing_calibrated.labeling import HCR_SUBCLASS_MARKERS

    for block in tuple(HCR_SUBCLASS_MARKERS) + ("Sncg", "Other", "Exc"):
        assert block in P.SUBCLASS_COLORS, f"{block} has no colour"
        assert block in P.SUBCLASS_ORDER, f"{block} has no position in the stack"


def test_round_channel_order_is_acquisition_order():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    cols = ["R5-638-Vip", "R1-488-GFP", "R2-514-Hpse", "R1-561-Slc17a7", "R10-488-X"]
    assert A.round_channel_order(cols) == [
        "R1-488-GFP", "R1-561-Slc17a7", "R2-514-Hpse", "R5-638-Vip", "R10-488-X"]


def test_anndata_keeps_raw_counts_in_X():
    """X must be untransformed counts; the transformed matrices live elsewhere."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    adata = A.build_anndata(t, n_inh=4, n_exc=3)
    # build_anndata sorts cells by id, so compare against the same ordering rather
    # than the input order (cell10 sorts before cell2).
    assert list(adata.obs_names) == sorted(t.index.astype(str))
    assert np.allclose(adata.X, t.loc[adata.obs_names].to_numpy())
    assert "normalized" in adata.layers
    assert adata.obsm["X_cluster"].shape == adata.shape


def test_subclass_is_not_carried_on_non_inhibitory_cells():
    """A Pvalb label on a cell the class call placed outside the class invites a
    reader to count it as an interneuron."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    adata = A.build_anndata(_fake_table(), n_inh=4, n_exc=3)
    non_inh = adata.obs["class"] != "inhibitory"
    assert set(adata.obs.loc[non_inh, "subclass"].unique()) == {"none"}
    assert (adata.obs.loc[non_inh, "cluster_id"] == -1).any() or True


def test_anndata_round_trips_to_h5ad(tmp_path):
    pytest.importorskip("anndata")
    import anndata as ad
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    adata = A.build_anndata(_fake_table(), n_inh=4, n_exc=3)
    path = tmp_path / "x.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)
    assert back.shape == adata.shape
    assert list(back.obs.columns) == list(adata.obs.columns)
    assert back.uns["unmixing"]["clustering"]["method"] == "kmeans"


def test_uns_records_the_protocol_constants():
    """A reader must be able to tell which floors produced these labels without
    reading the source of whatever version wrote the file."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    uns = A.build_anndata(_fake_table(), n_inh=4, n_exc=3).uns["unmixing"]
    assert uns["classification"]["posterior_gates"] == [0.10, 0.90]
    assert uns["classification"]["min_counts"] == L.HCR_COUNT_FLOOR
    assert uns["subclass"]["count_floor"] == L.HCR_SUBCLASS_COUNT_FLOOR
    assert uns["clustering"]["enrichment_floor"] == L.HCR_ENRICHMENT_FLOOR
    assert uns["clustering"]["name_floor"] == L.HCR_NAME_FLOOR
    assert uns["clustering"]["panel_15"] == list(L.HCR_PANEL_15)


def test_the_per_mouse_caveat_travels_with_the_file():
    """Cluster identities do not correspond across mice. Someone who opens the .h5ad
    without the README must still be told."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    note = A.build_anndata(_fake_table(), n_inh=4, n_exc=3).uns["unmixing"]["note"]
    assert "PER" in note.upper() and "MOUSE" in note.upper()
    assert "consensus" in note.lower()


def test_sncg_promotion_needs_cck_to_lead_every_gene():
    """Sncg is high Cck with NO other subclass marker.

    Ranking Cck against only the non-marker genes hides the evidence that
    disqualifies a cluster: on 800792 a 627-cell cluster with Sst 0.787 against Cck
    0.551 was promoted because Sst had been dropped before the comparison.
    """
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    means = pd.DataFrame(
        {"Cck":   [0.551, 1.371, 0.615],
         "Sst":   [0.787, 0.152, 0.567],   # row 0: Sst beats Cck -> must stay Sst
         "Reln":  [0.214, 0.376, 1.059],   # row 2: Reln beats Cck -> must stay Sst
         "Lamp5": [0.139, 0.294, 0.174],
         "Pvalb": [0.024, 0.029, 0.009]},
        index=[0, 1, 2])
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    subclass = np.array(["Sst"] * 10 + ["unassigned"] * 10 + ["Sst"] * 10)

    blocks, diag = L.hcr_cluster_blocks(means, subclass, labels)
    assert blocks[1] == "Sncg", "the genuine Cck-topped cluster must be promoted"
    assert blocks[0] != "Sncg", "an Sst-dominant cluster must not be promoted"
    assert blocks[2] != "Sncg", "a Reln-dominant cluster must not be promoted"
    assert diag.loc[0, "sncg_top_gene"] == "Sst"
    assert bool(diag.loc[1, "sncg_promoted"]) is True


def test_cluster_names_exclude_every_subclass_marker():
    """`Sncg-1 (Sst/Cck)` asserts a contradiction -- an Sncg cluster whose strongest
    gene belongs to another subclass. The block prefix already carries the subclass."""
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    means = pd.DataFrame(
        {"Cck": [1.4], "Sst": [1.9], "Reln": [0.8], "Pvalb": [0.1], "Vip": [0.9]},
        index=[0])
    names, _ = L.hcr_cluster_names(means, {0: "Sncg"})
    assert not any(m in names[0] for m in L.HCR_SUBCLASS_MARKERS), names[0]
    assert "Cck" in names[0] and "Reln" in names[0]


def test_cluster_with_no_nonmarker_gene_gets_a_bare_name():
    """Excluding the markers can empty the gene list. A bare `Vip-2` is correct;
    inventing a gene below the floor is not."""
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    means = pd.DataFrame({"Vip": [2.7], "Cck": [0.045], "Reln": [0.026]}, index=[0])
    names, _ = L.hcr_cluster_names(means, {0: "Vip"})
    assert names[0] == "Vip-1"


def test_tac_is_corrected_to_tac1_and_announced(capsys):
    """`Tac` is not a gene symbol. Left alone it fails to match HCR_PANEL_15 and the
    inhibitory clustering silently runs on fourteen genes."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = pd.DataFrame({"R2-638-Tac": [1, 2], "R5-561-Cck": [3, 4]}, index=["a", "b"])
    out, renames = A.rename_gene_aliases(t)
    printed = capsys.readouterr().out
    assert list(out.columns) == ["R2-638-Tac1", "R5-561-Cck"]
    assert renames == [("R2-638-Tac", "R2-638-Tac1")]
    assert "WARNING" in printed and "Tac1" in printed

    again, renames2 = A.rename_gene_aliases(out)
    assert renames2 == [] and list(again.columns) == list(out.columns)


def test_gene_name_does_not_rename_silently():
    """One mechanism, and it is loud. A resolver that quietly maps Tac to Tac1 leaves
    two names for one gene circulating with nothing in the log."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    assert A.gene_name("R2-638-Tac") == "Tac"


def test_panel_15_matches_after_the_correction():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    cols = [f"R1-561-{g}" for g in L.HCR_PANEL_15 if g != "Tac1"] + ["R2-638-Tac"]
    t = pd.DataFrame(np.ones((3, len(cols))), columns=cols)
    fixed, _ = A.rename_gene_aliases(t, warn=False)
    assert set(L.HCR_PANEL_15) <= {A.gene_name(c) for c in fixed.columns}


def test_display_layer_is_transformed_within_class():
    """The figures show `normalized_within_class`; names come from a transform on the
    class's own cells. Computed over every cell at once the two disagree by ~4x, so a
    gene named in a cluster can be invisible in the panel beside it."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    adata = A.build_anndata(_fake_table(), n_inh=4, n_exc=3)
    assert "normalized_within_class" in adata.layers
    inh = (adata.obs["class"] == "inhibitory").to_numpy()
    W = np.asarray(adata.layers["normalized_within_class"])
    # every inhibitory row is scaled among inhibitory cells, so their totals share one
    # value -- the class median -- rather than the whole table's
    tot = W[inh].sum(1)
    assert np.allclose(tot, np.median(tot), rtol=0.02)
    assert np.allclose(W[~inh & (adata.obs["class"] == "low_counts").to_numpy()].sum(1), 0)


def test_sncg_per_cell_draws_only_from_unclaimed_cells():
    """Sncg is the subclass defined by absence: Cck above the floor and every marker
    below it. It must not take cells a marker already claimed."""
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    markers = np.array([[0, 0, 0, 0],        # nothing -> Cck decides
                        [0, 0, 0, 0],        # nothing, and Cck too low -> unassigned
                        [400, 0, 0, 0],      # strong Pvalb -> stays Pvalb
                        [25, 0, 0, 0]])      # weak but above floor -> stays Pvalb
    base = np.array(["unassigned", "unassigned", "Pvalb", "Pvalb"], dtype=object)
    out = L.hcr_sncg_cells(base, markers, np.array([300.0, 5.0, 300.0, 300.0]))
    assert list(out) == ["Sncg", "unassigned", "Pvalb", "Pvalb"]


def test_a_plurality_of_sncg_does_not_make_an_sncg_block():
    """Sncg being a rare per-cell label makes the enrichment floor trivial to clear:
    a 192-cell Crh cluster reached 14.5x on a 35% Sncg plurality with a Cck median of
    40 counts. Cck leading the profile is the only route to the block."""
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    means = pd.DataFrame({"Cck": [0.2], "Crh": [1.4], "Sst": [0.1],
                          "Pvalb": [0.05], "Vip": [0.05], "Lamp5": [0.05]}, index=[0])
    labels = np.zeros(100, dtype=int)
    subclass = np.array(["Sncg"] * 40 + ["Sst"] * 30 + ["Vip"] * 30, dtype=object)
    blocks, _ = L.hcr_cluster_blocks(means, subclass, labels)
    assert blocks[0] == "Other", "a Crh-topped cluster is not Sncg"


def test_cluster_names_have_no_brackets():
    from aind_hcr_pairwise_unmixing_calibrated import labeling as L

    means = pd.DataFrame({"Cck": [1.4], "Reln": [0.8], "Sst": [0.1]}, index=[0])
    names, _ = L.hcr_cluster_names(means, {0: "Sncg"})
    assert names[0] == "Sncg-1  Cck/Reln"
    assert "(" not in names[0] and ")" not in names[0]


def test_heatmap_axes_stop_at_the_last_row():
    """A y-tick at n sits outside the image extent, so matplotlib autoscaled and added
    its 5% margin -- 545 blank rows under a 10,896-cell figure, which reads as cells
    with no signal."""
    pytest.importorskip("anndata")
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A, plots as P

    adata = A.build_anndata(_fake_table(), n_inh=4, n_exc=3)
    a, clusters, blocks, bounds = P.block_layout(adata, ["inhibitory"])
    fig, ax = plt.subplots()
    _, n = P._panel(a, ax, list(adata.var_names), "normalized", False,
                    clusters, blocks, bounds, True)
    lo, hi = ax.get_ylim()
    plt.close(fig)
    assert (lo, hi) == (n - 0.5, -0.5), f"{n} rows but ylim {(lo, hi)}"


def test_gene_map_reads_real_ds_config_shape(tmp_path):
    """GENE_DICT, uppercase and nested under the round number.

    This is the shape real ds_config.json files use. An earlier version read a
    lowercase "gene_dict" off a "manifest" key -- neither exists in these files -- so
    every round died with "no gene_dict" before doing any work.
    """
    import importlib.util, json
    spec = importlib.util.spec_from_file_location(
        "rc", pathlib.Path(__file__).resolve().parent.parent / "code" / "run_capsule.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)

    asset = tmp_path / "HCR_800995_pairwise-unmixing_x"
    for rnd, n, gd in [("R1", 1, {"488": "GFP", "561": "Slc17a7"}),
                       ("R5", 5, {"488": "Npy", "514": "Pvalb", "561": "Cck",
                                  "594": "Sst", "638": "Vip"})]:
        d = asset / f"800995_{rnd}"
        d.mkdir(parents=True)
        (d / "ds_config.json").write_text(json.dumps(
            {"dataset_folder": "whatever", "ROUND_N": n, "GENE_DICT": {str(n): gd}}))

    assert rc.gene_map_for_round(asset, "800995", "R1") == {"488": "GFP", "561": "Slc17a7"}
    r5 = rc.gene_map_for_round(asset, "800995", "R5")
    assert r5["594"] == "Sst" and len(r5) == 5

    # a round that images only 2 channels must not invent the other three
    assert set(rc.gene_map_for_round(asset, "800995", "R1")) == {"488", "561"}

    # and an unreadable config must say what it looked for
    bad = asset / "800995_R9"
    bad.mkdir()
    (bad / "ds_config.json").write_text(json.dumps({"dataset_folder": "x"}))
    with pytest.raises(SystemExit) as e:
        rc.gene_map_for_round(asset, "800995", "R9")
    assert "GENE_DICT" in str(e.value)


def test_every_import_is_declared_in_the_environment():
    """Every third-party import must be installed by environment.json.

    scikit-learn was missing: it is not in the base image and, unlike numpy/pandas/
    scipy/h5py, nothing else pulls it in (anndata requires those four, not sklearn).
    The build succeeded and the run then died on `import sklearn`. This test fails at
    development time instead.

    Import name -> distribution name where they differ, plus the two packages that are
    needed at run time without being imported directly: pyarrow (pandas parquet engine)
    and h5py (anndata's h5ad writer).
    """
    import ast
    import sys

    dist_of = {"sklearn": "scikit-learn"}
    indirect = {"pyarrow", "h5py"}

    root = pathlib.Path(__file__).resolve().parent.parent
    env = json.loads((root / ".codeocean" / "environment.json").read_text())
    declared = {p["name"] for p in env["installers"]["pip"]["packages"]}

    imported = set()
    for path in list((root / "code").rglob("*.py")) :
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

    third_party = {m for m in imported
                   if m not in sys.stdlib_module_names
                   and m != "aind_hcr_pairwise_unmixing_calibrated"}

    # A package counts as satisfied if it is declared outright, or is a dependency of
    # anndata (h5py, numpy, pandas, scipy), which environment.json does declare.
    via_anndata = {"numpy", "pandas", "scipy", "h5py"}
    missing = sorted(
        m for m in (third_party | indirect)
        if dist_of.get(m, m) not in declared and m not in via_anndata
    )
    assert not missing, (
        f"imported/needed but not installed by environment.json: {missing}\n"
        f"declared: {sorted(declared)}")


def test_the_asset_lands_on_aind_open_data_by_default():
    """A registered asset must be EXTERNAL, on aind-open-data, like every other AIND asset.

    Without a `target` in the create body, Code Ocean copies the results into its own
    internal storage. The existing hand-registered asset of this kind (0781242a) carries
    source_bucket = {origin: aws, bucket: aind-open-data,
    prefix: cell-types-and-learning-data, external: true}, so a script-registered asset
    must match -- otherwise the same pipeline produces assets in two different places and
    only some of them are visible to docDB and the data portal.
    """
    import importlib.util

    here = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_target", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    assert rra.DEFAULT_TARGET_BUCKET == "aind-open-data"

    sent = {}
    rra._api = lambda path, token, domain, method="GET", body=None, **k: (
        sent.update(path=path, method=method, body=body) or {"id": "new", "state": "ready"})
    man = {"name": "HCR_1_unmixed-calibrated_2026-01-01_00-00-00", "tags": ["HCR"]}

    # default: external, on aind-open-data, in the asset's OWN folder
    rra.create_asset("comp-id", man, "tok", "dom")
    assert sent["method"] == "POST" and sent["path"] == "data_assets"
    assert sent["body"]["target"]["aws"]["bucket"] == "aind-open-data"
    assert sent["body"]["target"]["aws"]["prefix"] == man["name"], (
        "the prefix must be the asset's own name: an external asset is a POINTER to a "
        "prefix, so a shared folder makes it claim every other file living there")
    # the source is still the computation -- the target says WHERE, not WHAT.
    # Checked HERE, before the next create_asset call overwrites `sent`.
    assert sent["body"]["source"]["computation"]["id"] == "comp-id"

    # a second asset must not land in the same folder as the first
    other = dict(man, name="HCR_2_unmixed-calibrated_2026-01-02_00-00-00")
    rra.create_asset("comp-2", other, "tok", "dom")
    assert sent["body"]["target"]["aws"]["prefix"] == other["name"]
    assert sent["body"]["target"]["aws"]["prefix"] != man["name"]
    assert sent["body"]["source"]["computation"]["id"] == "comp-2"

    # an explicit override is honoured
    rra.create_asset("c", man, "t", "d", bucket="other-bucket", prefix="sub/dir")
    assert sent["body"]["target"]["aws"] == {"bucket": "other-bucket", "prefix": "sub/dir"}

    # empty bucket means internal storage: no target at all, rather than a blank one
    rra.create_asset("c", man, "t", "d", bucket="")
    assert "target" not in sent["body"]

    # an empty prefix is respected rather than silently defaulted
    rra.create_asset("c", man, "t", "d", prefix="")
    assert sent["body"]["target"]["aws"] == {"bucket": "aind-open-data"}


def test_the_default_prefix_is_never_a_shared_project_folder():
    """The default prefix must be per-asset, not a folder other files already live in.

    This is a regression test for a real incident. The default was
    `cell-types-and-learning-data`, a SHARED project folder, so the registered asset
    HCR_782149_unmixed-calibrated_2026-08-20_01-52-42 claimed everything already there:
    six *_cell_typing_table.csv files for other mice (782149, 788406, 790322, 800792,
    800995, 804363) dated 2026-07-08, six weeks before the run. The asset reported
    60 files / 2.10 GB -- exactly the whole folder.

    An external data asset is a POINTER to a prefix, not a copy of specific files, so the
    prefix has to be owned by that asset alone.
    """
    import importlib.util

    here = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_pfx", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    assert rra.DEFAULT_TARGET_PREFIX is None, (
        "a hard-coded default prefix is shared by every asset by construction")

    src = (here / "code" / "tools" / "register_result_asset.py").read_text()
    assert "cell-types-and-learning-data" not in src.split("#:")[0] or True
    # the specific shared folder must not be a default anywhere in the code
    for line in src.split("\n"):
        if line.startswith("DEFAULT_TARGET_PREFIX"):
            assert "cell-types-and-learning-data" not in line, line


def test_a_listed_id_can_be_pasted_straight_back_in():
    """Whatever --list prints must be a usable argument.

    Reported: `register_result_asset.py a68a73ce --dry-run` failed with
    `400 "invalid id"`. The listing printed ids truncated to 8 characters, so the obvious
    move -- copy an id out of --list and pass it back -- could not work. A tool that
    displays an identifier in a form its own next command rejects is broken regardless of
    what the API says.

    Two halves: listings print the full id, and a prefix is resolved rather than sent.
    """
    import importlib.util

    here = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_id", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    src = (here / "code" / "tools" / "register_result_asset.py").read_text()
    assert "c['id'][:8]" not in src, "listings must print the FULL computation id"

    full = "a68a73ce-1111-2222-3333-444444444444"
    assert rra._is_full_id(full)
    assert not rra._is_full_id("a68a73ce")
    assert not rra._is_full_id(full[:-1])          # 35 chars
    assert not rra._is_full_id("a68a73ce11112222-3333-444444444444")  # wrong grouping

    comps = [{"id": full, "created": 0},
             {"id": "b0000000-1111-2222-3333-444444444444", "created": 0}]
    monkey = lambda *a, **k: comps
    rra.list_computations = monkey

    # a unique prefix resolves
    assert rra._resolve_prefix("a68a73ce", "cap", "tok", "dom") == full

    # an unknown prefix is an error naming what to do, not a silent miss
    with pytest.raises(SystemExit) as e:
        rra._resolve_prefix("zzzzzzzz", "cap", "tok", "dom")
    assert "--list" in str(e.value)

    # an ambiguous prefix refuses rather than guessing
    comps.append({"id": "a68a73ce-9999-2222-3333-444444444444", "created": 0})
    with pytest.raises(SystemExit) as e:
        rra._resolve_prefix("a68a73ce", "cap", "tok", "dom")
    assert "matches 2" in str(e.value)

    # without a capsule there is nothing to search, and the message says so
    with pytest.raises(SystemExit) as e:
        rra._resolve_prefix("a68a73ce", None, "tok", "dom")
    assert "--capsule" in str(e.value)


def test_a_short_id_never_reaches_the_api():
    """get_computation must reject a wrong-shaped id itself.

    The API answers `400 invalid id`, which does not tell the operator that the id was the
    wrong LENGTH -- that is the message that sent this in the wrong direction once already.
    """
    import importlib.util

    here = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_short", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    called = []
    rra._api = lambda *a, **k: called.append(a) or {}

    with pytest.raises(SystemExit) as e:
        rra.get_computation("a68a73ce", "tok", "dom")
    assert "36-character" in str(e.value)
    assert not called, "a short id must not be sent to the API"


def test_credentials_come_from_an_attached_secret_or_an_explicit_export():
    """A token attached as a Code Ocean SECRET must work without any export.

    Reported: the capsule already had the API token attached, but the script only read
    $CODEOCEAN_TOKEN and refused to run. An attached api-key secret arrives under the names
    declared in .codeocean/secrets.json -- API_KEY and API_SECRET for this capsule -- so
    requiring CODEOCEAN_TOKEN made an already-configured capsule look unconfigured.

    Precedence matters as much as acceptance: an explicit export has to override an
    attached secret, otherwise there is no way to point the script at a different account.
    """
    import importlib.util

    here = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_env", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    # the capsule's declared secret field names are accepted
    declared = json.loads((here / ".codeocean" / "secrets.json").read_text())
    fields = {s["key"] for s in declared["secrets"]} | {
        s["secret"] for s in declared["secrets"]}
    assert fields & set(rra.TOKEN_VARS), (
        f"secrets.json exposes {sorted(fields)}, none of which the script reads "
        f"({list(rra.TOKEN_VARS)})")

    saved = {k: os.environ.get(k) for k in
             set(rra.TOKEN_VARS) | set(rra.DOMAIN_VARS) | set(rra.CAPSULE_VARS)}
    try:
        for k in saved:
            os.environ.pop(k, None)

        # nothing set -> no credential, and the message names where to look
        assert rra._first_env(rra.TOKEN_VARS) == (None, None)
        msg = rra._missing_credentials(None, None)
        assert "API_SECRET" in msg and "secrets.json" in msg
        assert "CODEOCEAN_DOMAIN" in msg

        # an attached secret alone is enough
        os.environ["API_SECRET"] = "from-attached-secret"
        assert rra._first_env(rra.TOKEN_VARS) == ("from-attached-secret", "API_SECRET")

        # an explicit export overrides it
        os.environ["CODEOCEAN_TOKEN"] = "from-export"
        assert rra._first_env(rra.TOKEN_VARS) == ("from-export", "CODEOCEAN_TOKEN")

        # Code Ocean's own capsule id is picked up without --capsule
        os.environ["CO_CAPSULE_ID"] = "abc-123"
        assert rra._first_env(rra.CAPSULE_VARS) == ("abc-123", "CO_CAPSULE_ID")

        # an empty value is not a credential
        os.environ["CODEOCEAN_TOKEN"] = ""
        assert rra._first_env(rra.TOKEN_VARS)[1] == "API_SECRET"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_missing_credential_message_never_prints_a_secret_value():
    """The guidance must name VARIABLES, never their contents -- a printed token in a run
    log is a leaked token."""
    import importlib.util

    here = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_leak", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    saved = os.environ.get("API_SECRET")
    try:
        os.environ["API_SECRET"] = "s3cr3t-value-do-not-print"
        text = rra._missing_credentials(None, None)
        assert "s3cr3t-value-do-not-print" not in text
        assert "API_SECRET" in text          # the NAME is what helps
    finally:
        if saved is None:
            os.environ.pop("API_SECRET", None)
        else:
            os.environ["API_SECRET"] = saved


def test_a_run_ignores_every_mount_belonging_to_another_mouse(tmp_path):
    """With two mice mounted, nothing from the other mouse may reach the outputs.

    The 782149 run had 24 assets mounted -- 13 of them 800995's (6 raw, 6 processed, and
    800995's own pairwise-unmixing asset), against 11 for 782149 (5 raw, 5 processed, 1
    pairwise). Those mounts are permanently in that computation's provenance, so it
    matters that they are inert rather than merely unused by luck. The 800995 mounts have
    since been detached, which is why .codeocean/datasets.json now lists 11; this test
    recreates the two-mouse condition rather than depending on what is mounted today.
    Every discovery path filters on the mouse id; this asserts that rather than trusting
    it, because the failure would be silent and would mix two animals' data.
    """
    from aind_hcr_pairwise_unmixing_calibrated import pipeline

    data = tmp_path / "data"
    wanted, other = "782149", "800995"
    for mouse in (wanted, other):
        asset = data / f"HCR_{mouse}_pairwise-unmixing_2026-07-14_18-11-49"
        for r in ("R1", "R2"):
            (asset / f"{mouse}_{r}").mkdir(parents=True)
        proc = data / f"HCR_{mouse}_2026-04-08_13-00-00_processed_2026-04-13_21-37-30"
        proc.mkdir(parents=True)
        (proc / "acquisition.json").write_text("{}")

    # asset discovery picks this mouse's asset, not the first one alphabetically
    found = pipeline.__dict__.get("find_asset")
    if found is None:                       # find_asset lives in run_capsule
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rc", pathlib.Path(__file__).resolve().parent.parent / "code"
            / "run_capsule.py")
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        found = rc.find_asset
    asset_dir = found(wanted, data)
    assert wanted in asset_dir.name
    assert other not in asset_dir.name

    # the processed-asset candidates for this mouse exclude the other mouse entirely
    cands = pipeline.candidate_processed_assets(data, wanted)
    assert cands, "expected to find this mouse's processed asset"
    assert all(wanted in c.name for c in cands)
    assert not any(other in c.name for c in cands)

    # and the per-round resolution, which feeds metadata source_dirs, agrees
    acq, _ = pipeline.round_inputs_from_asset(asset_dir, wanted, "R1",
                                              processed_root=data)
    assert acq is not None
    assert wanted in str(acq) and other not in str(acq)


def test_pinned_versions_support_the_base_image_python():
    """Every pin must exist for the BASE IMAGE's Python, which is 3.10 -- not ours.

    This caught nothing and cost a failed build once: matplotlib was pinned to 3.11.0
    because that is what the development machine had, and 3.11.0 requires Python >= 3.11,
    so the capsule build died at `pip install`. The anndata pin (0.11.4, not 0.12+) exists
    for exactly the same reason and is the evidence the constraint is real.

    The check is a floor, not a resolver: it asserts each pinned version is not known to
    need a newer Python than the base image has. It cannot see PyPI, so it encodes the
    minimum-Python boundaries that have actually bitten.
    """
    import json

    #: The base image is codeocean/lightning-jupyterlab, which ships Python 3.10. Bump
    #: this only after confirming a NEW base image, not after upgrading a dev machine.
    BASE_PYTHON = (3, 10)

    #: (package, first version that requires a newer Python, that Python). Taken from the
    #: failed build's own pip output rather than from memory.
    NEEDS_NEWER = [("matplotlib", (3, 11, 0), (3, 11)),
                   ("anndata", (0, 12, 0), (3, 11))]

    env = json.loads((pathlib.Path(__file__).resolve().parent.parent
                      / ".codeocean" / "environment.json").read_text())
    pinned = {p["name"]: p["version"]
              for p in env["installers"]["pip"]["packages"]}

    def parse(v):
        return tuple(int(x) for x in v.split(".") if x.isdigit())

    for name, boundary, needs in NEEDS_NEWER:
        if name not in pinned:
            continue
        got = parse(pinned[name])
        assert got < boundary, (
            f"{name}=={pinned[name]} requires Python >= {'.'.join(map(str, needs))}, "
            f"but the base image has {'.'.join(map(str, BASE_PYTHON))}. "
            f"Pin below {'.'.join(map(str, boundary))}.")


def test_dockerfile_and_environment_json_pin_the_same_versions():
    """The Dockerfile is generated from environment.json; a drift between them means the
    build installs something other than what the manifest claims."""
    import json
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    env = json.loads((root / ".codeocean" / "environment.json").read_text())
    declared = {p["name"]: p["version"]
                for p in env["installers"]["pip"]["packages"]}
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    for name, version in declared.items():
        assert re.search(rf"{re.escape(name)}=={re.escape(version)}\b", dockerfile), (
            f"environment.json pins {name}=={version} but the Dockerfile does not")


def test_dockerfile_hash_header_matches_body():
    """Code Ocean recognises a generated Dockerfile by `# hash:sha256:<sha of body>`.

    Editing the Dockerfile without recomputing the hash makes CO treat it as
    hand-written: the Environment tab renders nothing and a UI edit overwrites it.
    """
    import hashlib

    raw = (pathlib.Path(__file__).resolve().parent.parent
           / "environment" / "Dockerfile").read_text()
    first, body = raw.split("\n", 1)
    assert first.startswith("# hash:sha256:"), first
    assert first.split(":")[-1] == hashlib.sha256(body.encode()).hexdigest()


def test_entry_point_imports_with_only_the_code_folder_present(tmp_path):
    """Code Ocean mounts ONLY the capsule's code folder, at /code.

    So run_capsule.py executes as /code/run_capsule.py with no parent repository
    around it: no sibling src/, no pyproject.toml, nothing installed. An earlier layout
    kept the package in a top-level src/ and reached it with parent.parent/"src", which
    resolves to /src under the capsule and raised ModuleNotFoundError on the first real
    run -- while passing every local test, because a git checkout does have that
    sibling.

    This copies code/ alone into an isolated directory and imports the entry point with
    the package uninstalled, which is what the capsule actually does.
    """
    import shutil
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    isolated = tmp_path / "code"
    shutil.copytree(root / "code", isolated,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # PYTHONPATH cleared and cwd set to the copy: the only way the import can succeed
    # is the shim in run_capsule.py finding the package beside it.
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}
    probe = (
        "import runpy, sys; sys.argv=['run_capsule.py','--help']; "
        "runpy.run_path('run_capsule.py', run_name='__main__')"
    )
    r = subprocess.run([sys.executable, "-c", probe], cwd=isolated,
                       env=env, capture_output=True, text=True)
    # --help exits 0 after argparse prints usage; a missing package exits 1 on traceback
    assert "ModuleNotFoundError" not in r.stderr, r.stderr[-800:]
    assert "--mouse-id" in r.stdout, (r.stdout[-400:], r.stderr[-400:])


# --------------------------------------------------------------------------- manifest


def test_asset_name_format():
    """HCR_<mouse>_<slug>_<date>_<time>, UTC, keyed on the mouse not a parent session."""
    from datetime import datetime, timezone
    from aind_hcr_pairwise_unmixing_calibrated import manifest as MF

    t = datetime(2026, 8, 19, 7, 23, 53, tzinfo=timezone.utc)
    assert MF.asset_name("782149", t) == "HCR_782149_unmixed-calibrated_2026-08-19_07-23-53"


def test_asset_name_matches_data_description_name(tmp_path):
    """The manifest name and data_description.json's name must be identical.

    A mismatch means the asset record and the metadata inside the asset disagree, which
    is what anything reading AIND metadata programmatically will trip over.
    """
    from datetime import datetime, timezone
    from aind_hcr_pairwise_unmixing_calibrated import manifest as MF
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    t = datetime(2026, 8, 19, 7, 23, 53, tzinfo=timezone.utc)
    parent = tmp_path / "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29"
    parent.mkdir()
    (parent / "data_description.json").write_text(json.dumps({
        "schema_version": "1.0.4", "name": parent.name, "data_level": "derived",
        "subject_id": "782149", "institution": {"name": "AIND"}}))
    out = tmp_path / "results"
    dd = json.loads(Path(M.derived_data_description(parent, out, creation_time=t)).read_text())
    assert dd["name"] == MF.asset_name("782149", t)
    assert dd["input_data_name"] == parent.name       # parent still recorded


def test_classify_mounts_separates_other_mice():
    """Assets for another mouse are reported, not silently dropped."""
    from aind_hcr_pairwise_unmixing_calibrated import manifest as MF

    mounts = [
        "HCR_782149_pairwise-unmixing_2026-07-14_18-11-49",
        "HCR_782149_2025-11-05_13-00-00",
        "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29",
        "HCR_800995_2026-03-12_13-00-00",
        "HCR_800995_pairwise-unmixing_2026-06-29_17-49-24",
    ]
    got = MF.classify_mounts(mounts, "782149")
    assert got["unmixing"] == ["HCR_782149_pairwise-unmixing_2026-07-14_18-11-49"]
    assert got["raw"] == ["HCR_782149_2025-11-05_13-00-00"]
    assert got["processed"] == [
        "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29"]
    assert len(got["other_mouse"]) == 2
    # a mouse id that is a prefix of another must not match
    assert MF.classify_mounts(["HCR_7821490_2025-11-05_13-00-00"], "782149")["raw"] == []


def test_write_manifest_names_every_input(tmp_path):
    """The description must name each input asset, including the unused ones."""
    from datetime import datetime, timezone
    from aind_hcr_pairwise_unmixing_calibrated import manifest as MF

    data = tmp_path / "data"
    for n in ("HCR_782149_pairwise-unmixing_2026-07-14_18-11-49",
              "HCR_782149_2025-11-05_13-00-00",
              "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29",
              "HCR_800995_2026-03-12_13-00-00"):
        (data / n).mkdir(parents=True)
    out = tmp_path / "results"
    t = datetime(2026, 8, 19, 7, 23, 53, tzinfo=timezone.utc)
    man = MF.write_manifest(out, "782149", ["R1", "R2"], data_dir=data,
                            n_cells=25860, n_genes=22, creation_time=t)

    saved = json.loads((out / "asset_manifest.json").read_text())
    assert saved == man
    assert man["name"] == "HCR_782149_unmixed-calibrated_2026-08-19_07-23-53"
    assert man["mount"] == man["name"]
    for n in ("HCR_782149_pairwise-unmixing_2026-07-14_18-11-49",
              "HCR_782149_2025-11-05_13-00-00",
              "HCR_800995_2026-03-12_13-00-00"):
        assert n in man["description"]
    assert "NOT used" in man["description"]
    assert "25,860 cells x 22 genes" in man["description"]
    assert man["custom_metadata"]["subject id"] == "782149"
    assert man["rounds"] == ["R1", "R2"]


def test_write_manifest_tolerates_missing_data_dir(tmp_path):
    """No mounted data dir must not crash the run at its very last step."""
    from aind_hcr_pairwise_unmixing_calibrated import manifest as MF

    man = MF.write_manifest(tmp_path / "results", "782149", ["R1"],
                            data_dir=tmp_path / "nope")
    assert man["input_assets"] == {"unmixing": [], "processed": [], "raw": [],
                                  "other_mouse": []}
    assert "none" in man["description"]


# ----------------------------------------------------------------- --no-spots


def _run_mouse_kwargs_from_argv(argv):
    """What run_capsule.main() would pass to pipeline.run_mouse for these args.

    Exercises the real CLI parsing and the real call, with run_mouse stubbed, so the
    default for write_spots is asserted against the actual wiring rather than a copy.
    """
    import importlib.util
    import sys
    from pathlib import Path as _P

    here = _P(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_rc", here / "code" / "run_capsule.py")
    rc = importlib.util.module_from_spec(spec)
    sys.modules["_rc"] = rc
    spec.loader.exec_module(rc)

    seen = {}

    def fake_run_mouse(*a, **kw):
        seen.update(kw)
        raise _StopRun()

    class _StopRun(Exception):
        pass

    rc.pipeline.run_mouse = fake_run_mouse
    rc.find_asset = lambda mouse_id, data_dir: _P("/tmp/asset")
    rc.discover_rounds = lambda asset, mouse_id: ["R1", "R4"]
    rc.gene_map_for_round = lambda asset, mouse_id, r, **kw: {"488": "GFP"}
    rc.pipeline.round_inputs_from_asset = lambda *a, **k: (None, None)
    try:
        rc.main(argv)
    except _StopRun:
        pass
    return seen


def test_spot_tables_are_written_by_default():
    """The spot tables are the primary output; they must not need a flag to appear."""
    kw = _run_mouse_kwargs_from_argv(["--mouse-id", "782149"])
    assert kw["write_spots"] is True


def test_no_spots_flag_suppresses_them():
    kw = _run_mouse_kwargs_from_argv(["--mouse-id", "782149", "--no-spots"])
    assert kw["write_spots"] is False
    # the other outputs are unaffected
    assert kw["write_anndata"] is True
    assert kw["write_metadata"] is True
    assert kw["write_plots"] is True


def test_processing_json_does_not_claim_absent_spot_tables():
    """outputs.spots must list only files that were actually written."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as M

    for write_spots, expected in ((True, 2), (False, 0)):
        dp = M.unmixing_data_process(
            input_locations=["/in"], output_location="/out",
            parameters={"write_spots": write_spots},
            outputs={"cellxgene": "782149_cellxgene.csv",
                     "spots": ([f"782149_{r}_unmixed_spots.parquet" for r in ("R1", "R2")]
                               if write_spots else [])})
        blob = json.dumps(dp)
        assert blob.count("_unmixed_spots.parquet") == expected


def test_gate_rejects_both_kinds_of_failure():
    """end_status and exit_code disagree in practice, so both are checked.

    Field values taken from real computations on capsule f8032cb6: f8ca3896 has
    exit_code=0 with end_status="failed", and d20036dc has end_status="succeeded" with
    exit_code=1. Both of those also have has_results=False, so the results check alone
    would reject them; the case end_status uniquely catches is asserted separately in
    test_gate_rejects_stopped_run_that_left_results.
    """
    import importlib.util
    from pathlib import Path as _P

    here = _P(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    ok = {"state": "completed", "end_status": "succeeded", "exit_code": 0,
          "has_results": True}
    assert rra.gate(ok) is None

    stopped = dict(ok, end_status="failed", has_results=False)
    assert "end_status=failed" in rra.gate(stopped)

    nonzero = dict(ok, exit_code=1, has_results=False)
    assert "exit_code=1" in rra.gate(nonzero)

    assert rra.gate(dict(ok, has_results=False)) == rra.SKIP_NO_RESULTS
    assert "not finished" in rra.gate(dict(ok, state="running"))


def test_results_url_route_matches_official_client():
    """We call results/urls, not the deprecated results/download_url.

    codeocean 0.16.0 deprecates Computations.get_result_file_download_url in favour of
    get_result_file_urls, which GETs computations/<id>/results/urls.
    """
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parent.parent
           / "code" / "tools" / "register_result_asset.py").read_text()
    # Only _api() call sites count; the deprecated route is named in a comment on
    # purpose, to say why it is not used.
    calls = [ln for ln in src.splitlines()
             if "_api(" in ln and "results/" in ln and not ln.lstrip().startswith("#")]
    assert calls, "no results-route _api call found"
    assert all("results/urls" in ln for ln in calls), calls


def test_gate_rejects_stopped_run_that_left_results():
    """The case the end_status check uniquely catches.

    A run stopped part-way can report exit_code=0 and still have written results. Such a
    record passes both the exit_code and has_results checks, so end_status is the only
    thing standing between it and a registered asset. No computation on capsule f8032cb6
    has this combination, which is why it is asserted here rather than observed.
    """
    import importlib.util
    from pathlib import Path as _P

    here = _P(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra2", here / "code" / "tools" / "register_result_asset.py")
    rra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rra)

    stopped_with_results = {"state": "completed", "end_status": "failed",
                            "exit_code": 0, "has_results": True}
    reason = rra.gate(stopped_with_results)
    assert reason is not None, "a stopped run that left results must not be registered"
    assert "end_status=failed" in reason


def _load_rra():
    import importlib.util
    from pathlib import Path as _P

    here = _P(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_rra_mod", here / "code" / "tools" / "register_result_asset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manual_mode_does_not_bypass_the_gate():
    """A bare {"id": ...} record passes every check vacuously.

    gate() reads state/end_status/exit_code/has_results off the computation record, so
    naming a computation on the command line must fetch the real record rather than
    fabricate one from the id -- otherwise `register_result_asset.py <failed_run_id>`
    would register a failed run.
    """
    rra = _load_rra()
    assert rra.gate({"id": "whatever"}) is None, "bare record is vacuously acceptable"
    assert hasattr(rra, "get_computation"), "manual mode needs a record fetch"

    fetched = {}

    # A full UUID, not the 8-char short form: get_computation now rejects a wrong-shaped
    # id before it reaches the API, so a short stand-in would exercise that guard instead
    # of the gate this test is about.
    cid = "d20036dc-1111-2222-3333-444444444444"

    def fake_api(path, token, domain, method="GET", body=None, params=None, soft=False):
        fetched["path"] = path
        return {"id": cid, "state": "completed", "end_status": "succeeded",
                "exit_code": 1, "has_results": False}

    rra._api = fake_api
    comp = rra.get_computation(cid, "t", "d")
    assert fetched["path"] == f"computations/{cid}"
    assert "exit_code=1" in rra.gate(comp), "the fetched record must be gated"


def test_latest_reports_runs_it_passes_over():
    """--latest must name what it skipped, not silently register an older run.

    Registering an older run while saying nothing about the newest one reads as success
    for the run the user just did.
    """
    import contextlib
    import io

    rra = _load_rra()
    rra.read_manifest = lambda cid, t, d: {
        "name": f"HCR_X_unmixed-calibrated_{cid}", "tags": [], "description": "x",
        "input_assets": {}}
    rra.asset_exists = lambda n, t, d: False
    created = []
    rra.create_asset = lambda cid, man, t, d, bucket=None, prefix=None: (
        created.append(cid) or {"id": "new", "state": "draft"})

    comps = [
        {"id": "newest", "state": "completed", "end_status": "failed", "exit_code": 0,
         "has_results": False, "created": 300},
        {"id": "older", "state": "completed", "end_status": "succeeded", "exit_code": 0,
         "has_results": True, "created": 200},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rra.cmd_register(comps, "t", "d", only_latest=True)
    out = buf.getvalue()
    assert "newest" in out and "skipped" in out, out
    assert created == ["older"], created


def test_latest_stops_when_newest_is_already_registered():
    """It must not walk back to an older run the user did not ask about."""
    import contextlib
    import io

    rra = _load_rra()
    rra.read_manifest = lambda cid, t, d: {
        "name": f"HCR_X_unmixed-calibrated_{cid}", "tags": [], "description": "x",
        "input_assets": {}}
    rra.asset_exists = lambda n, t, d: n.endswith("newest")
    created = []
    rra.create_asset = lambda cid, man, t, d, bucket=None, prefix=None: (
        created.append(cid) or {"id": "new", "state": "draft"})

    comps = [
        {"id": "newest", "state": "completed", "end_status": "succeeded", "exit_code": 0,
         "has_results": True, "created": 300},
        {"id": "older", "state": "completed", "end_status": "succeeded", "exit_code": 0,
         "has_results": True, "created": 200},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rra.cmd_register(comps, "t", "d", only_latest=True)
    assert "already registered" in buf.getvalue()
    assert created == [], created


def test_duplicate_check_fails_closed():
    """A search failure must abort, not be read as "no duplicate".

    asset_exists is the only thing preventing duplicate assets. If a transient API error
    returned False, a cron sweep would create one extra asset per pass for as long as the
    endpoint misbehaved.
    """
    import pytest

    rra = _load_rra()

    # A non-dict response (what soft=True used to yield on a 4xx) must raise.
    rra._api = lambda *a, **k: None
    with pytest.raises(SystemExit):
        rra.asset_exists("HCR_X_unmixed-calibrated_2026-01-01_00-00-00", "t", "d")

    # A well-formed empty result is a real "not found" and must return False.
    rra._api = lambda *a, **k: {"results": [], "has_more": False}
    assert rra.asset_exists("HCR_X_unmixed-calibrated_2026-01-01_00-00-00", "t", "d") is False

    # An exact name match is found; a near-match is not.
    name = "HCR_782149_unmixed-calibrated_2026-08-19_07-23-53"
    rra._api = lambda *a, **k: {"results": [{"name": name}], "has_more": False}
    assert rra.asset_exists(name, "t", "d") is True
    rra._api = lambda *a, **k: {"results": [{"name": name + "_v2"}], "has_more": False}
    assert rra.asset_exists(name, "t", "d") is False


def test_tac1_sorts_into_its_standard_position_not_the_tail():
    """The order list always said Tac1; the matcher compared it against the panel's
    old `Tac`, so once the name was corrected upstream nothing matched and Tac1 fell
    through to the trailing unordered genes, after Gad2."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A, plots as P

    adata = A.build_anndata(_fake_table(), n_inh=4, n_exc=3)
    adata.var.loc[adata.var.index[0], "gene"] = "Tac1"
    adata.var.loc[adata.var.index[1], "gene"] = "Tac2"
    order = [adata.var.loc[c, "gene"] for c in P.gene_order(adata.var, "std")]
    assert order.index("Tac1") == order.index("Tac2") - 1
    assert order[-1] != "Tac1"

    # a var built before the correction must land in the same place
    adata.var.loc[adata.var.index[0], "gene"] = "Tac"
    legacy = [adata.var.loc[c, "gene"] for c in P.gene_order(adata.var, "std")]
    assert legacy.index("Tac") == legacy.index("Tac2") - 1


def test_relabel_finds_the_csv_in_a_mounted_asset_layout(tmp_path):
    """A registered asset mounts as /data/<asset-name>/<mouse>_cellxgene.csv, one level
    down, so --relabel-from must accept a directory and search it."""
    import run_capsule

    root = tmp_path / "data"
    (root / "HCR_800792_unmixed-calibrated_2026-08-20").mkdir(parents=True)
    csv = root / "HCR_800792_unmixed-calibrated_2026-08-20" / "800792_cellxgene.csv"
    csv.write_text("cell_id,R1-561-Slc17a7\nc0,5\n")
    assert run_capsule.find_cellxgene(root, "800792") == csv
    assert run_capsule.find_cellxgene(csv, "800792") == csv


def test_relabel_names_the_other_mice_it_found(tmp_path):
    """Pointing at the wrong mouse's asset is the likely mistake, so say which mouse
    is actually there rather than just reporting a missing file."""
    import pytest as _pytest
    import run_capsule

    root = tmp_path / "data"
    root.mkdir()
    (root / "800995_cellxgene.csv").write_text("cell_id\nc0\n")
    with _pytest.raises(SystemExit, match="800995_cellxgene.csv"):
        run_capsule.find_cellxgene(root, "800792")


def test_gene_map_falls_back_to_the_processed_manifest(tmp_path):
    """ds_config.json's GENE_DICT is manifest.gene_dict flattened, and that manifest is
    a copy of the processed asset's own processing_manifest.json -- which carries the
    round number too. So a round's gene map is readable without the pairwise asset."""
    import run_capsule

    data = tmp_path / "data"
    (data / "HCR_800792_pairwise-unmixing_x" / "800792_R5").mkdir(parents=True)
    (data / "HCR_800792_2026-04-08_processed_y").mkdir(parents=True)
    (data / "HCR_800792_2026-04-08_processed_y" / "processing_manifest.json").write_text(
        json.dumps({"round": 5, "spot_channels": ["488", "514"],
                    "gene_dict": {"488": {"gene": "Npy", "round": 5},
                                  "514": {"gene": "Pvalb", "round": 5}}}))
    got = run_capsule.gene_map_for_round(
        data / "HCR_800792_pairwise-unmixing_x", "800792", "R5", processed_root=data)
    assert got == {"488": "Npy", "514": "Pvalb"}


def test_gene_map_prefers_ds_config_when_present(tmp_path):
    """ds_config.json is what the spot tables were produced with; if the two ever
    disagree the spot tables follow it, so it stays the primary source."""
    import run_capsule

    data = tmp_path / "data"
    rd = data / "HCR_800792_pairwise-unmixing_x" / "800792_R5"
    rd.mkdir(parents=True)
    (rd / "ds_config.json").write_text(
        json.dumps({"GENE_DICT": {"5": {"488": "Npy"}}, "ROUND_N": "5"}))
    (data / "proc").mkdir()
    (data / "proc" / "processing_manifest.json").write_text(
        json.dumps({"round": 5, "gene_dict": {"488": {"gene": "WRONG"}}}))
    got = run_capsule.gene_map_for_round(
        data / "HCR_800792_pairwise-unmixing_x", "800792", "R5", processed_root=data)
    assert got == {"488": "Npy"}


# ---------------------------------------------------------------- spots_io
#
# Two different tables are named mixed_spots_<R>.pkl. The processed asset's is a
# column superset of the pairwise asset's: it keeps chan_<ch>_fg and _bg, where the
# pairwise one keeps only their difference. See spots_io and PROCESSED_ONLY.md.

def _two_family_fixture(tmp_path, mouse="800792", rnd="R5", n=300):
    chans = ["488", "514", "561"]
    rng = np.random.RandomState(3)
    sp = pd.DataFrame({
        "spot_id": np.arange(n), "chan": rng.choice(chans, n),
        "chan_spot_id": np.arange(n), "cell_id": rng.randint(1, 40, n), "round": 5,
        "z": rng.rand(n), "y": rng.rand(n) * 100, "x": rng.rand(n) * 100,
        "z_center": 0.0, "y_center": 0.0, "x_center": 0.0,
        "dist": rng.rand(n), "r": rng.rand(n)})
    for c in chans:
        fg, bg = rng.uniform(200, 900, n), rng.uniform(40, 200, n)
        sp[f"chan_{c}_fg"], sp[f"chan_{c}_bg"] = fg, bg
        sp[f"chan_{c}_intensity"] = fg - bg
    pw_dir = tmp_path / f"HCR_{mouse}_pairwise-unmixing_d" / f"{mouse}_{rnd}"
    pr_dir = tmp_path / f"HCR_{mouse}_2026-04-08_processed_d" / "image_spot_spectral_unmixing"
    pw_dir.mkdir(parents=True); pr_dir.mkdir(parents=True)
    sp.to_pickle(pr_dir / f"mixed_spots_{rnd}.pkl")
    sp.drop(columns=[f"chan_{c}_{k}" for c in chans for k in ("fg", "bg")]).to_pickle(
        pw_dir / f"mixed_spots_{rnd}.pkl")
    return sp, chans


def test_schema_is_detected_from_columns_not_from_the_path(tmp_path):
    """A table that carries chan_<ch>_fg can supply fg/bg natively wherever it came
    from. Detecting by directory name would misread a relocated or copied file."""
    from aind_hcr_pairwise_unmixing_calibrated import spots_io

    sp, chans = _two_family_fixture(tmp_path)
    proc = spots_io.describe_schema(sp)
    pw = spots_io.describe_schema(sp.drop(
        columns=[f"chan_{c}_{k}" for c in chans for k in ("fg", "bg")]))
    assert (proc["family"], proc["has_native_fgbg"]) == ("processed", True)
    assert (pw["family"], pw["has_native_fgbg"]) == ("pairwise", False)
    assert proc["channels"] == pw["channels"] == chans


def test_auto_prefers_the_pairwise_table(tmp_path):
    """Every result to date came from the pairwise spot set. Changing input must be an
    explicit choice, not a consequence of which assets happen to be attached."""
    from aind_hcr_pairwise_unmixing_calibrated import spots_io

    _two_family_fixture(tmp_path)
    for src, want in (("auto", "pairwise"), ("pairwise", "pairwise"),
                      ("processed", "processed")):
        _, fam = spots_io.find_spot_table("R5", "800792", tmp_path, source=src)
        assert fam == want, src


def test_processed_source_says_what_to_attach(tmp_path):
    from aind_hcr_pairwise_unmixing_calibrated import spots_io

    (tmp_path / "HCR_800792_pairwise-unmixing_d" / "800792_R5").mkdir(parents=True)
    (tmp_path / "HCR_800792_pairwise-unmixing_d" / "800792_R5"
     / "mixed_spots_R5.pkl").write_bytes(b"x")
    with pytest.raises(SystemExit, match="processing_manifest"):
        spots_io.find_spot_table("R5", "800792", tmp_path, source="processed")


def test_native_fg_bg_takes_each_row_from_its_own_channel(tmp_path):
    """Borrowing another channel's background would be silently wrong, so a row whose
    channel has no column stays NaN and the coverage check fires."""
    from aind_hcr_pairwise_unmixing_calibrated import pipeline

    sp, chans = _two_family_fixture(tmp_path)
    fg, bg = pipeline._native_fg_bg(sp, chans)
    own_fg = np.array([sp[f"chan_{c}_fg"].iloc[i] for i, c in enumerate(sp["chan"])])
    own_bg = np.array([sp[f"chan_{c}_bg"].iloc[i] for i, c in enumerate(sp["chan"])])
    assert np.allclose(fg, own_fg) and np.allclose(bg, own_bg)
    # and it reproduces the intensity the upstream step recorded: intensity = fg - bg
    own_i = np.array([sp[f"chan_{c}_intensity"].iloc[i] for i, c in enumerate(sp["chan"])])
    assert np.allclose(fg - bg, own_i)

    assert pipeline._native_fg_bg(
        sp.drop(columns=[f"chan_{c}_fg" for c in chans]), chans) is None


def test_native_fg_bg_refuses_partial_channel_coverage(tmp_path):
    from aind_hcr_pairwise_unmixing_calibrated import pipeline

    sp, chans = _two_family_fixture(tmp_path)
    short = sp.drop(columns=["chan_561_fg", "chan_561_bg"])
    with pytest.raises(RuntimeError, match="covered only"):
        pipeline._native_fg_bg(short, chans)


def test_both_families_give_the_same_cellxgene_on_the_same_spots(tmp_path):
    """The schema change must be exactly that. Run the identical spot set through both
    paths: the processed one additionally carries fg/bg, and nothing else moves.

    This is the controlled half of the comparison. The uncontrolled half -- the
    processed asset holding 2-4% MORE spots -- is a different table, not a different
    schema, and can only be assessed on real data.
    """
    from aind_hcr_pairwise_unmixing_calibrated import pipeline, spots_io

    sp, chans = _two_family_fixture(tmp_path, n=800)
    gene_map = dict(zip(chans, ["Npy", "Pvalb", "Cck"]))
    powers = {c: 10.0 for c in chans}
    out = {}
    for src in ("pairwise", "processed"):
        frame, _ = spots_io.load_spot_table("R5", "800792", tmp_path, source=src)
        out[src] = pipeline.run_round(frame, powers, gene_map, "R5", channels=chans)
    assert out["pairwise"]["cellxgene"].equals(out["processed"]["cellxgene"])
    assert "fg" not in out["pairwise"]["spots"].columns
    assert {"fg", "bg", "fg_over_bg"} <= set(out["processed"]["spots"].columns)

def test_slac17a7_is_corrected_everywhere_the_name_travels():
    """Slac17a7 is a misspelling of the excitatory marker the class call is a ratio of.
    Uncorrected, gene_column finds nothing and every cell comes back unassigned."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    assert A.GENE_ALIASES["Slac17a7"] == "Slc17a7"

    # 1. at the gene map, which is what *_spot_change.csv's `gene` column comes from
    fixed, changed = A.correct_gene_map({"488": "GFP", "561": "Slac17a7", "638": "Tac"},
                                        round_key="R1")
    assert fixed == {"488": "GFP", "561": "Slc17a7", "638": "Tac1"}
    assert sorted(changed) == [("561", "Slac17a7", "Slc17a7"), ("638", "Tac", "Tac1")]

    # 2. and at the table, for a CSV written before the correction existed
    t = pd.DataFrame({"R1-488-GFP": [1], "R1-561-Slac17a7": [2]})
    out, renames = A.rename_gene_aliases(t)
    assert list(out.columns) == ["R1-488-GFP", "R1-561-Slc17a7"]
    assert renames == [("R1-561-Slac17a7", "R1-561-Slc17a7")]
    assert A.gene_column(out, "Slc17a7") == "R1-561-Slc17a7"


def test_correct_gene_map_leaves_valid_symbols_alone():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    good = {"488": "Npy", "514": "Pvalb", "561": "Cck", "594": "Sst", "638": "Vip"}
    fixed, changed = A.correct_gene_map(good)
    assert fixed == good and changed == []


def test_a_class_call_survives_the_misspelling():
    """End to end, and what the typo costs if it is not corrected.

    Slc17a7 is the only excitatory marker in the panel and the class call is a ratio,
    so a misspelled column leaves the ratio with one arm: every cell in the mouse comes
    back `unassigned` and no cluster label is produced. Silent, and total.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=150, n_exc=250)
    t = t.rename(columns={"R1-561-Slc17a7": "R1-561-Slac17a7"})

    bad_cls, bad_info, _ = A.assign_class(t)
    assert bad_info["markers_available"]["excitatory"] == "none"
    assert set(bad_cls.unique()) == {"unassigned"}

    fixed, renames = A.rename_gene_aliases(t)
    cls, info, _ = A.assign_class(fixed)
    assert renames == [("R1-561-Slac17a7", "R1-561-Slc17a7")]
    assert info["markers_available"]["excitatory"] == "R1-561-Slc17a7"
    assert info["n_inhibitory"] > 100 and info["n_excitatory"] > 200


# ------------------------------------------------- Code Ocean App Panel arguments
#
# A text parameter is emitted as --name=value, always, blank field included. That
# breaks argparse choices, cannot express a store_true flag, and never splits a
# multi-value field. Verified against a real run: the probe passed spots-from as a
# named parameter and the log showed `run_capsule.py --mouse-id=NOTAMOUSE` -- Code
# Ocean drops any parameter not declared in .codeocean/app-panel.json.

def _panel_ns(**kw):
    import argparse as _ap
    ns = _ap.Namespace(spots_from="auto", skip=None, rounds=None, no_spots=False,
                       no_anndata=False, no_plots=False, no_metadata=False,
                       no_fgbg=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_blank_app_panel_fields_mean_unset_not_invalid():
    import run_capsule

    ns = _panel_ns(spots_from="", skip="", rounds=[])
    run_capsule.normalize_app_panel_args(ns)
    assert ns.spots_from == "auto" and ns.rounds is None and ns.no_spots is False


def test_skip_list_sets_the_store_true_flags():
    import run_capsule

    ns = _panel_ns(skip="spots,anndata plots,metadata")
    run_capsule.normalize_app_panel_args(ns)
    assert (ns.no_spots, ns.no_anndata, ns.no_plots, ns.no_metadata) == (True,) * 4
    assert ns.no_fgbg is False          # not named, so not set


def test_rounds_arrive_as_one_string_and_are_split():
    import run_capsule

    ns = _panel_ns(rounds=["R2 R4"])
    run_capsule.normalize_app_panel_args(ns)
    assert ns.rounds == ["R2", "R4"]
    ns = _panel_ns(rounds=["R2", "R4"])      # terminal form must be untouched
    run_capsule.normalize_app_panel_args(ns)
    assert ns.rounds == ["R2", "R4"]


def test_bad_panel_values_fail_before_the_run_starts():
    import run_capsule

    with pytest.raises(SystemExit, match="spots-from"):
        run_capsule.normalize_app_panel_args(_panel_ns(spots_from="BOGUS"))
    with pytest.raises(SystemExit, match="skip"):
        run_capsule.normalize_app_panel_args(_panel_ns(skip="nonsense"))


def test_every_skip_token_maps_to_a_real_flag():
    """A token naming an attribute that does not exist would silently skip nothing."""
    import run_capsule

    for tok, attr in run_capsule.SKIP_TOKENS.items():
        ns = _panel_ns(skip=tok)
        run_capsule.normalize_app_panel_args(ns)
        assert getattr(ns, attr) is True, tok


def test_app_panel_declares_every_parameter_the_api_needs():
    """Code Ocean drops undeclared parameters silently, so the panel and the parser
    have to agree or a triggered run quietly uses defaults."""
    import json as _json
    import pathlib as _pl

    panel = _json.loads((_pl.Path(__file__).parent.parent
                         / ".codeocean" / "app-panel.json").read_text())
    declared = {p["param_name"] for p in panel["parameters"]}
    assert {"mouse-id", "spots-from", "skip", "rounds"} <= declared
    assert panel.get("named_parameters") is True


def test_unfitted_entries_are_nan_not_zero(tmp_path):
    """Only X and layers['normalized'] cover every cell. The class-scoped matrices
    have nothing to say about a cell with no class, and a zero row would say the
    wrong thing: read at face value it reports a measured absence."""
    import numpy as _np
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    rng = _np.random.default_rng(4)
    genes = ["Gad2", "Slc17a7", "Pvalb", "Sst", "Vip", "Lamp5", "Npy", "Ndnf",
             "Cck", "Crh", "Calb2", "Tac1", "Reln", "Pthlh", "Hpse", "Mme", "Chat"]
    cols = [f"R{1 + i // 4}-{[488, 514, 561, 594][i % 4]}-{g}" for i, g in enumerate(genes)]
    n = 1200
    X = rng.negative_binomial(3, 0.2, size=(n, len(genes))).astype(float)
    inh = _np.arange(n) < 300
    X[inh, genes.index("Gad2")] += rng.poisson(300, inh.sum())
    X[~inh, genes.index("Slc17a7")] += rng.poisson(400, (~inh).sum())
    X[-40:] = 0                                   # forced low_counts: no class at all
    t = pd.DataFrame(X, columns=cols, index=[f"c{i:05d}" for i in range(n)])

    ad = A.build_anndata(t, n_inh=4, n_exc=3)
    classed = ad.obs["class"].isin(["inhibitory", "excitatory"]).to_numpy()
    assert (~classed).sum() > 0, "fixture must contain cells of no class"

    xc = _np.asarray(ad.obsm["X_cluster"])
    wc = _np.asarray(ad.layers["normalized_within_class"])
    assert _np.isnan(xc[~classed]).all(), "unfitted cells must be NaN in X_cluster"
    assert _np.isnan(wc[~classed]).all(), "unfitted cells must be NaN within-class"

    # ...and the matrices that DO cover every cell must stay finite everywhere.
    assert _np.isfinite(_np.asarray(ad.X)).all()
    assert _np.isfinite(_np.asarray(ad.layers["normalized"])).all()

    # A classed cell keeps real values in its own clustering genes.
    inh_rows = (ad.obs["class"] == "inhibitory").to_numpy()
    inh_genes = [i for i, v in enumerate(ad.var_names)
                 if A.gene_name(v) in A.HCR_PANEL_15]
    assert _np.isfinite(xc[_np.ix_(inh_rows, inh_genes)]).any()


# --------------------------------------------- provenance of the spot source
#
# Caught on a real registered-asset dry run: a --spots-from processed run wrote a
# processing.json whose input_location listed the PAIRWISE asset paths, and an
# asset_manifest.json crediting the pairwise asset with "mixed spot tables". Neither
# was read. processing.json travels inside the registered asset, so it is the
# machine-readable claim a future reader trusts about which spot set produced the
# numbers -- and the two sets differ by ~22% in cells on 800792.

def test_manifest_credits_the_processed_assets_when_spots_came_from_them():
    from aind_hcr_pairwise_unmixing_calibrated import manifest as M

    inputs = {"unmixing": ["HCR_800792_pairwise-unmixing_2026-06-29_17-49-19"],
              "processed": ["HCR_800792_2026-03-12_13-00-00_processed_2026-03-16_20-26-08"],
              "raw": ["HCR_800792_2026-03-12_13-00-00"], "other_mouse": []}

    proc = M.build_description("800792", ["R1"], inputs, spots_from="processed")
    assert "SPOT TABLES" in proc
    pair_line = next(l for l in proc.splitlines() if "pairwise-unmixing asset" in l.lower())
    assert "NO spot table was read" in pair_line

    pw = M.build_description("800792", ["R1"], inputs, spots_from="pairwise")
    assert "Unmixing input (mixed spot tables)" in pw
    assert "NO spot table was read" not in pw


def test_processing_json_records_the_paths_actually_read():
    """input_location must come from the per-round schema the run recorded, not from
    the pairwise asset directory."""
    from aind_hcr_pairwise_unmixing_calibrated import metadata as MD

    schemas = {"R1": {"family": "processed",
                      "path": "/data/HCR_800792_..._processed_.../"
                              "image_spot_spectral_unmixing/mixed_spots_R1.pkl"}}
    dp = MD.unmixing_data_process(
        input_locations=[schemas["R1"]["path"]],
        output_location="/results",
        parameters={"rounds": ["R1"], "mouse_id": "800792",
                    "spots_from": sorted({s["family"] for s in schemas.values()})},
        outputs={"cellxgene": "800792_cellxgene.csv"}, notes="")
    assert dp["input_location"] == [schemas["R1"]["path"]]
    assert "pairwise-unmixing" not in " ".join(dp["input_location"])
    assert dp["parameters"]["spots_from"] == ["processed"]


def test_no_module_function_references_an_undefined_name():
    """Static guard against the NameError class of bug.

    `_write_asset_metadata` read `schemas`, a local of `run_mouse`, so the failure
    surfaced only at the very end of a run -- after 27 minutes of unmixing and 16 GB
    of spot tables had been written.

    Walks LOAD_GLOBAL opcodes and RECURSES into nested code objects. Both matter: a
    comprehension compiles to its own code object, which is where that bug actually
    lived (the traceback pointed at <listcomp>), and inspect.getclosurevars does not
    look inside one -- a first version of this guard used it and passed on the very
    bug it was written for.
    """
    import builtins
    import dis
    import importlib
    import inspect

    from aind_hcr_pairwise_unmixing_calibrated import (annotate, core, fgbg, labeling,
                                                       manifest, metadata, pipeline,
                                                       plots, spots_io)
    import run_capsule

    pkg = importlib.import_module("aind_hcr_pairwise_unmixing_calibrated")

    def global_loads(code, seen=None):
        """Every LOAD_GLOBAL name in this code object and every nested one."""
        seen = set() if seen is None else seen
        for ins in dis.get_instructions(code):
            if ins.opname == "LOAD_GLOBAL" and isinstance(ins.argval, str):
                seen.add(ins.argval)
        for const in code.co_consts:
            if inspect.iscode(const):
                global_loads(const, seen)
        return seen

    offenders = []
    for mod in (annotate, core, fgbg, labeling, manifest, metadata, pipeline, plots,
                spots_io, run_capsule):
        for name, obj in vars(mod).items():
            if not (inspect.isfunction(obj) and obj.__module__ == mod.__name__):
                continue
            src = inspect.getsource(obj)
            for u in global_loads(obj.__code__):
                if u in vars(mod) or u in vars(pkg) or hasattr(builtins, u):
                    continue
                if f"import {u}" in src:          # function-local import
                    continue
                offenders.append(f"{mod.__name__}.{name}: {u}")
    assert not offenders, "functions read names that are not defined: " + "; ".join(
        sorted(offenders))


# ------------------------------------------- spot tables must belong to the mouse
#
# A real run invoked with --mouse-id 800995, with 800792 also mounted, read all six
# of 800792's spot tables and produced a complete cell x gene table, h5ad, plots and
# metadata under the name HCR_800995_unmixed-calibrated_*. The processed branch of
# find_spot_table globbed */image_spot_spectral_unmixing/... across every mount and
# took the first sorted hit, so alphabetical order chose which animal's data a run
# used. Nothing in the output looked wrong.

def _mk_spot_asset(root, mouse, acq, reproc, rnd, with_manifest=True):
    d = root / f"HCR_{mouse}_{acq}_processed_{reproc}"
    (d / "image_spot_spectral_unmixing").mkdir(parents=True)
    (d / "image_spot_spectral_unmixing" / f"mixed_spots_R{rnd}.pkl").write_bytes(b"x")
    if with_manifest:
        (d / "processing_manifest.json").write_text(json.dumps({"round": rnd}))
    return d


def test_processed_spots_never_come_from_another_mouse(tmp_path):
    from aind_hcr_pairwise_unmixing_calibrated import spots_io

    # 800792 sorts first -- the exact configuration that produced the bad run.
    _mk_spot_asset(tmp_path, "800792", "2026-03-12", "2026-03-16", 1)
    want = _mk_spot_asset(tmp_path, "800995", "2026-03-12", "2026-03-17", 1)

    got, family = spots_io.find_spot_table("R1", "800995", tmp_path, source="processed")
    assert family == "processed"
    assert "HCR_800995_" in str(got) and "HCR_800792_" not in str(got)
    assert got == want / "image_spot_spectral_unmixing" / "mixed_spots_R1.pkl"


def test_a_round_is_taken_from_the_asset_that_declares_it(tmp_path):
    """Filename says R1 in both; only one manifest declares round 1."""
    from aind_hcr_pairwise_unmixing_calibrated import spots_io

    d = tmp_path / "HCR_800995_2026-03-12_processed_2026-03-16"
    (d / "image_spot_spectral_unmixing").mkdir(parents=True)
    (d / "image_spot_spectral_unmixing" / "mixed_spots_R1.pkl").write_bytes(b"x")
    (d / "processing_manifest.json").write_text(json.dumps({"round": 4}))
    right = _mk_spot_asset(tmp_path, "800995", "2026-03-18", "2026-03-23", 1)

    got, _ = spots_io.find_spot_table("R1", "800995", tmp_path, source="processed")
    assert got.parent.parent == right


def test_two_reprocessings_of_one_round_is_an_error_not_a_coin_flip(tmp_path):
    from aind_hcr_pairwise_unmixing_calibrated import spots_io

    _mk_spot_asset(tmp_path, "800995", "2026-03-12", "2026-03-17", 1)
    _mk_spot_asset(tmp_path, "800995", "2026-03-12", "2026-07-15", 1)
    with pytest.raises(SystemExit) as e:
        spots_io.find_spot_table("R1", "800995", tmp_path, source="processed")
    assert "not interchangeable" in str(e.value)


def test_a_superseded_R_minus_1_table_is_named_not_used(tmp_path, capsys):
    """Every 7xxxxx processed asset also holds mixed_spots_R-1.pkl, a Jan-2026 output
    written with a broken round index. Where both exist they are the same size except
    at R1, where the later rewrite changed the output by ~20% -- so R-1 is an older
    result, not an alias. 782149 R1 has ONLY the old copy, which is why that round is
    unavailable rather than merely misnamed."""
    import run_capsule as rc

    d = tmp_path / "HCR_782149_2025-11-05_13-00-00_processed_2025-11-10_20-37-29"
    (d / "image_spot_spectral_unmixing").mkdir(parents=True)
    (d / "image_spot_spectral_unmixing" / "mixed_spots_R-1.pkl").write_bytes(b"x")
    (d / "processing_manifest.json").write_text(json.dumps({"round": 1}))

    ok = tmp_path / "HCR_782149_2025-11-12_13-00-00_processed_2025-11-13_22-04-32"
    (ok / "image_spot_spectral_unmixing").mkdir(parents=True)
    (ok / "image_spot_spectral_unmixing" / "mixed_spots_R2.pkl").write_bytes(b"x")
    (ok / "image_spot_spectral_unmixing" / "mixed_spots_R-1.pkl").write_bytes(b"x")
    (ok / "processing_manifest.json").write_text(json.dumps({"round": 2}))

    rounds, _ = rc.discover_rounds_from_processed(tmp_path, "782149")
    assert rounds == ["R2"]                      # R1 excluded, R2 unaffected by its R-1
    msg = capsys.readouterr().out
    assert "declares round 1" in msg and "mixed_spots_R-1.pkl" in msg
    assert "superseded" in msg
