"""Synthetic ground-truth tests. No data assets required -- these run anywhere."""
import json
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aind_hcr_pairwise_unmixing_calibrated import core
from aind_hcr_pairwise_unmixing_calibrated.control import CHANS, control_matrix

RNG = np.random.RandomState(0)


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
    assert doc["name"].startswith(parent.name)           # but derived from it
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
        # names still have something to report once subclass genes are excluded
        v[6 + (i % 2)] = rng.poisson(200)   # Cck or Mme
        rows.append(v)
    for _ in range(n_exc):
        v = rng.poisson(8, len(genes)).astype(float)
        v[0] = rng.poisson(400)        # Slc17a7 high
        v[1] = rng.poisson(2)          # Gad2 low
        rows.append(v)
    idx = [f"cell{i}" for i in range(n_inh + n_exc)]
    return pd.DataFrame(np.array(rows), index=idx, columns=genes)


def test_class_labels_need_a_positive_marker():
    """Without Slc17a7 nothing may be called excitatory."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    cls, info = A.assign_class(t)
    assert set(cls.unique()) == {"inhibitory", "excitatory"}

    no_exc = t.drop(columns=["R1-561-Slc17a7"])
    cls2, info2 = A.assign_class(no_exc)
    assert "excitatory" not in set(cls2.unique())          # never asserted
    # "none" rather than None: uns is written to HDF5, which cannot store None
    assert info2["markers_available"]["excitatory"] == "none"
    assert (cls2 == "inhibitory").sum() == (cls == "inhibitory").sum()


def test_double_positive_is_unassigned():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=10, n_exc=10)
    t.iloc[0, t.columns.get_loc("R1-561-Slc17a7")] = 500     # both markers high
    cls, info = A.assign_class(t)
    assert cls.iloc[0] == "unassigned"
    # key renamed when the rule widened to any interneuron marker; the guarantee is the
    # same -- Gad2 AND Slc17a7 together is refused rather than guessed
    assert info["n_ambiguous_gad2_and_slc17a7"] == 1


def test_any_interneuron_marker_calls_inhibitory():
    """Pvalb/Vip/Sst alone are enough, without Gad2.

    Gad2 alone under-calls: on 800995 it labelled 3,401 cells inhibitory where the
    four-marker rule finds 7,035. A cell strongly expressing a canonical interneuron
    marker is inhibitory whether or not its Gad2 reading cleared the bar.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=4, n_exc=4)
    gad = t.columns[t.columns.str.endswith("Gad2")][0]
    t[gad] = 0                                    # nothing clears Gad2
    # GFP is in the gate too: in this preparation it is an interneuron reporter, and it
    # is the single largest contributor (on 800995: 11,835 positive cells vs Gad2's 6,851)
    tested = 0
    for gene in A.INHIBITORY_MARKERS:
        if gene == "Gad2":
            continue                              # held at 0 above, on purpose
        hits = t.columns[t.columns.str.endswith(gene)]
        if not len(hits):
            continue                              # gene absent from this fixture's panel
        col = hits[0]
        t.loc[t.index[0], col] = 500
        cls, info = A.assign_class(t)
        assert cls.iloc[0] == "inhibitory", f"{gene} alone should call inhibitory"
        t.loc[t.index[0], col] = 0
        tested += 1
    assert tested >= 3, f"expected to exercise >=3 markers, got {tested}"

    # and the threshold is honoured: just under it is not a call
    col = t.columns[t.columns.str.endswith("Pvalb")][0]
    t.loc[t.index[0], col] = A.MIN_CLASS_COUNTS - 1
    cls, _ = A.assign_class(t)
    assert cls.iloc[0] != "inhibitory"


def test_gene_orders_are_complete_permutations():
    """Both gene orderings must contain every column exactly once.

    A dropped column would silently omit a gene from the heatmap; a duplicated one would
    plot it twice. The std order is built by walking a fixed name list, so a panel gene
    missing from that list has to fall through to the tail rather than vanish.
    """
    import pandas as pd
    from aind_hcr_pairwise_unmixing_calibrated import plots as P

    cols = ["R1-561-Slc17a7", "R5-514-Pvalb", "R4-638-Gad2", "R2-488-Ndnf",
            "R9-488-Unlisted"]
    genes = ["Slc17a7", "Pvalb", "Gad2", "Ndnf", "Unlisted"]
    var = pd.DataFrame({"gene": genes}, index=cols)

    for kind in ("std", "rc"):
        order = P.gene_order(var, kind)
        assert sorted(order) == sorted(cols), f"{kind} is not a permutation"
        assert len(set(order)) == len(order), f"{kind} has duplicates"
    # acquisition order really is round-then-channel, numerically not lexically
    assert P.gene_order(var, "rc") == ["R1-561-Slc17a7", "R2-488-Ndnf", "R4-638-Gad2",
                                       "R5-514-Pvalb", "R9-488-Unlisted"]
    # a gene absent from STD_GENE_ORDER still appears, at the tail
    assert "R9-488-Unlisted" in P.gene_order(var, "std")


def test_slc17a7_high_cells_need_gad2_corroboration():
    """An interneuron marker alone cannot admit a strongly Slc17a7-positive cell.

    This is the rule that removed 1,081 cells from the inhibitory class on 800995 --
    three clusters with Slc17a7 medians of 743-806 and Gad2 medians of 26-76, admitted
    on Pvalb medians of 154-196 against a real Pvalb cluster's 262. Every one of the
    1,081 was Slc17a7-positive and none had Gad2 >= threshold, so they moved to
    excitatory rather than to unassigned.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=6, n_exc=6)
    thr = A.MIN_CLASS_COUNTS
    pv = t.columns[t.columns.str.endswith("Pvalb")][0]
    gd = t.columns[t.columns.str.endswith("Gad2")][0]
    sl = t.columns[t.columns.str.endswith("Slc17a7")][0]

    # Pvalb-positive AND Slc17a7-high, no Gad2 -> NOT inhibitory (the contaminant case)
    t.loc[t.index[0], [pv, sl, gd]] = [5 * thr, 8 * thr, 0]
    # the same Pvalb signal WITH Gad2 -> corroborated, but Gad2+Slc17a7 both high is the
    # pre-existing AMBIGUOUS case, which takes precedence -> unassigned, not inhibitory.
    # So corroboration never admits an Slc17a7-high cell; its effect is to move cells
    # that used to be inhibitory into excitatory (no Gad2) or unassigned (Gad2 too).
    t.loc[t.index[1], [pv, sl, gd]] = [5 * thr, 8 * thr, 5 * thr]
    # Pvalb-positive with LOW Slc17a7 -> inhibitory, no corroboration needed
    t.loc[t.index[2], [pv, sl, gd]] = [5 * thr, 0, 0]

    cls, info = A.assign_class(t)
    assert cls.iloc[0] != "inhibitory", "Slc17a7-high + Pvalb, no Gad2 must not be inhibitory"
    assert cls.iloc[0] == "excitatory", "it is Slc17a7-positive, so excitatory is the call"
    assert cls.iloc[1] == "unassigned", "Gad2 AND Slc17a7 both high stays ambiguous"
    assert cls.iloc[2] == "inhibitory", "low Slc17a7 needs no corroboration"
    assert info["n_ambiguous_gad2_and_slc17a7"] >= 1


def test_lamp5_is_a_subclass_gene_but_never_admits():
    """Lamp5 must not be in the admission gate.

    It is expressed in 45% of ALL cells on 800995 (median 84, against 10/5/8 for
    Pvalb/Sst/Vip), so gating on it would admit most of the excitatory population --
    30,966 of its 34,235 positive cells are excitatory. Npy is the sparse alternative
    that was added instead (median 0, 1.9% of cells positive).
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    assert "Lamp5" not in A.INHIBITORY_MARKERS
    assert "Npy" in A.INHIBITORY_MARKERS
    assert "Lamp5" in A.SUBCLASS_GENES, "still names a group once a cell IS inhibitory"

    t = _fake_table(n_inh=4, n_exc=4)
    for g in A.INHIBITORY_MARKERS:                 # zero every admission marker present
        hits = t.columns[t.columns.str.endswith(g)]
        if len(hits):
            t[hits[0]] = 0
    lam = t.columns[t.columns.str.endswith("Lamp5")][0]
    t.loc[t.index[0], lam] = 50 * A.MIN_CLASS_COUNTS
    cls, _ = A.assign_class(t)
    assert cls.iloc[0] != "inhibitory", "Lamp5 alone must not admit a cell"


def test_marker_names_exclude_round6_and_gate_genes():
    """Round-6 genes, the class gates and GFP never appear in a cluster name."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    genes = list(A.ROUND6_GENES) + list(A.GATE_GENES) + ["Mme", "Pthlh"]
    means = pd.DataFrame([[10.0] * len(genes), [1.0] * len(genes)], columns=genes)
    means.loc[0, "Sncg"] = 500      # would dominate enrichment if not barred
    means.loc[0, "Gad2"] = 500
    names = A.cluster_marker_names(means, {0: "Pvalb", 1: "Pvalb"})
    for banned in A.ROUND6_GENES + A.GATE_GENES:
        assert banned not in names[0], f"{banned} leaked into {names[0]}"


def test_round_channel_order_is_acquisition_order():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    cols = ["R5-638-Vip", "R1-488-GFP", "R2-514-Hpse", "R1-561-Slc17a7", "R10-488-X"]
    assert A.round_channel_order(cols) == [
        "R1-488-GFP", "R1-561-Slc17a7", "R2-514-Hpse", "R5-638-Vip", "R10-488-X"]


def test_normalization_is_reversible_in_shape_and_bounded():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    norm, info = A.normalize_cellxgene(t)
    assert norm.shape == t.shape
    assert float(norm.to_numpy().max()) <= 1.0 + 1e-9        # clipped at the 95th pct
    assert float(norm.to_numpy().min()) >= 0.0
    assert info["depth_scale"] == "per_cell_mean"
    # the MEAN is why no cell is dropped: it is positive whenever any gene is detected,
    # where a 27-gene panel leaves ~16% of real cells with a median of 0
    assert info["n_zero_mean_cells"] == 0


def test_no_cell_is_dropped_by_depth_normalization():
    """Every cell with at least one transcript is scalable."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table(n_inh=5, n_exc=5)
    t.iloc[0, :] = 0.0                     # completely empty cell
    t.iloc[1, :] = 0.0
    t.iloc[1, 0] = 1.0                     # a single transcript in one gene
    norm, info = A.normalize_cellxgene(t)
    assert info["n_zero_mean_cells"] == 1, "only the all-zero cell is unscalable"
    assert float(norm.iloc[1].sum()) > 0, "one transcript is enough to be scaled"
    # a median-based stage 1 would have failed BOTH: with 8 genes, one nonzero gene
    # still leaves a median of 0
    assert float(np.median(t.iloc[1].to_numpy())) == 0.0


def test_subclass_is_the_highest_expressing_marker_not_the_most_enriched():
    """Level, not enrichment -- so the label never contradicts the heatmap.

    Pvalb is the highest-expressing marker in cluster 0 (0.90 vs Lamp5's 0.40), but
    Lamp5 is the more ENRICHED one: 0.40/0.20 = 2.00x its across-cluster mean, against
    0.90/0.50 = 1.80x for Pvalb. The old enrichment rule labelled such clusters Lamp5
    while the Pvalb column was visibly darker; three real clusters on 800995 had exactly
    this contradiction.
    """
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    means = pd.DataFrame(
        {"R5-514-Pvalb": [0.90, 0.10],     # cluster 0 highest here
         "R4-488-Lamp5": [0.40, 0.00],     # but far more enriched here
         "R5-594-Sst":   [0.05, 0.80],
         "R5-638-Vip":   [0.05, 0.05]},
        index=[0, 1])
    enrich0 = means.loc[0] / means.mean(0)
    assert enrich0["R4-488-Lamp5"] > enrich0["R5-514-Pvalb"], "fixture must be enrichment-inverted"
    sc = A.assign_subclass(means)
    assert sc[0][0] == "Pvalb", "subclass must follow expression level"
    assert sc[1][0] == "Sst"


def test_clusters_named_by_subclass_and_numbered():
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    norm, _ = A.normalize_cellxgene(t)
    cls, _ = A.assign_class(t)
    labels, cid, subclass, info = A.cluster_by_class(norm, cls, n_inh=4, n_exc=3)

    inh_names = set(labels[cls == "inhibitory"].unique())
    exc_names = set(labels[cls == "excitatory"].unique())
    # A marker suffix appears only when a gene clears the enrichment floor. The
    # synthetic excitatory cells are homogeneous by construction, so some Exc
    # clusters legitimately have no distinguishing marker and no suffix -- naming
    # one anyway would be inventing structure. Inhibitory cells DO differ by
    # subclass here, so every inhibitory cluster must carry markers.
    # subclass genes are excluded from marker lists, so a name carries a suffix only
    # when the cluster has a distinguishing NON-subclass gene; the fixture gives
    # inhibitory cells a secondary marker so they do
    assert all("(" in n and ")" in n for n in inh_names)
    assert not any("(Pvalb" in n or "(Sst" in n or "(Vip" in n or "(Lamp5" in n
                   for n in inh_names)
    assert all(n.startswith("Exc-") for n in exc_names)
    # inhibitory clusters carry a canonical subclass prefix
    assert any(n.split("-")[0] in A.SUBCLASS_GENES for n in inh_names)
    # ids are unique across the two independently-clustered groups
    assert cid[cls == "inhibitory"].nunique() == 4
    assert cid[cls == "excitatory"].nunique() == 3
    assert not (set(cid[cls == "inhibitory"]) & set(cid[cls == "excitatory"]))


def test_anndata_keeps_raw_counts_in_X():
    """X must be untransformed counts; the clustering matrix lives in a layer."""
    pytest.importorskip("anndata")
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    t = _fake_table()
    adata = A.build_anndata(t, n_inh=4, n_exc=3)
    # build_anndata sorts cells by id, so compare against the same ordering rather
    # than the input order (cell10 sorts before cell2).
    assert list(adata.obs_names) == sorted(t.index.astype(str))
    assert np.allclose(adata.X, t.loc[adata.obs_names].to_numpy())   # raw, not normalized
    assert "normalized" in adata.layers
    assert adata.layers["normalized"].max() <= 1.0 + 1e-9
    assert {"class", "subclass", "cluster", "cluster_id"} <= set(adata.obs.columns)
    assert set(adata.var.columns) == {"round", "channel", "gene"}
    assert adata.n_obs == len(t)
    # every cell of a known class gets a cluster
    known = adata.obs["class"].isin(["inhibitory", "excitatory"])
    assert (adata.obs.loc[known, "cluster_id"] >= 0).all()


def test_anndata_round_trips_to_h5ad(tmp_path):
    """uns must be HDF5-writable: int keys and None values raise inside the writer,
    AFTER the expensive unmixing has already run."""
    pytest.importorskip("anndata")
    import anndata as ad
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    adata = A.build_anndata(_fake_table(), n_inh=4, n_exc=3)
    path = tmp_path / "annotated.h5ad"
    adata.write_h5ad(path)                       # must not raise

    back = ad.read_h5ad(path)
    assert back.n_obs == adata.n_obs
    assert set(back.obs["cluster"].unique()) == set(adata.obs["cluster"].unique())
    assert "normalized" in back.layers
    assert np.allclose(back.X, adata.X)


def test_subclass_genes_excluded_from_marker_names():
    """Pvalb-2 (Pvalb/...) wastes a slot -- the subclass is already the prefix."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    means = pd.DataFrame(
        {"R5-514-Pvalb": [10.0, 1.0],       # subclass gene, strongly enriched
         "R3-514-Mme":   [8.0, 1.0],
         "R2-561-Pthlh": [6.0, 1.0],
         "R5-561-Cck":   [1.0, 9.0]},
        index=[0, 1])
    names = A.cluster_marker_names(means, {0: "Pvalb", 1: "Vip"})

    assert "Pvalb" not in names[0].split("(")[1]        # not in the marker list
    assert names[0].startswith("Pvalb-1")               # still the prefix
    assert "Mme" in names[0] and "Pthlh" in names[0]    # replaced by the next best
    # opting out restores the old behaviour
    keep = A.cluster_marker_names(means, {0: "Pvalb", 1: "Vip"}, exclude_genes=())
    assert "Pvalb" in keep[0].split("(")[1]


def test_subclass_call_still_uses_subclass_genes():
    """Excluding them from NAMES must not affect the subclass assignment itself."""
    from aind_hcr_pairwise_unmixing_calibrated import annotate as A

    means = pd.DataFrame(
        {"R5-514-Pvalb": [10.0, 1.0],
         "R5-594-Sst":   [1.0, 10.0],
         "R5-561-Cck":   [5.0, 5.0]},
        index=[0, 1])
    sc = A.assign_subclass(means)
    assert sc[0][0] == "Pvalb"
    assert sc[1][0] == "Sst"


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


def test_a_run_ignores_every_mount_belonging_to_another_mouse(tmp_path):
    """With two mice mounted, nothing from the other mouse may reach the outputs.

    The 782149 run had 24 assets mounted -- 13 of them 800995's, including 800995's own
    pairwise-unmixing asset. Those mounts are permanently in that computation's
    provenance, so it matters that they are inert rather than merely unused by luck.
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
