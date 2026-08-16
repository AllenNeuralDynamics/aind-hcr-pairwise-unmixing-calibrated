"""Synthetic ground-truth tests. No data assets required -- these run anywhere."""
import json
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
