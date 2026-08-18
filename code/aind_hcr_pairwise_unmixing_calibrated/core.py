"""Core calibrated unmixing algorithm.

Vendored from the interactive development session; this module has NO dependency on
aind_spot_spectral_unmixing and does not import anything from the upstream engine. It
takes a spot DataFrame in and returns a spot DataFrame out.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial import cKDTree

CHANS = ["488", "514", "561", "594", "638"]

# Regimes for the separability gate (proposed, not yet swept - see proposal §5)
AUC_REFUSE = 0.60      # below this: do nothing, flag the pair
AUC_TRUST = 0.80       # at/above this: per-spot reassignment allowed
ANGLE_REFUSE = 10.0    # degrees


def intensity_matrix(df, channels=CHANS):
    """(n_spots, n_channels) background-subtracted intensities; missing chan -> 0."""
    I = np.zeros((len(df), len(channels)), np.float32)
    for k, c in enumerate(channels):
        col = f"chan_{c}_intensity"
        if col in df.columns:
            I[:, k] = df[col].to_numpy(np.float32)
    return I


def detection_index(df, channels=CHANS):
    ci = {c: i for i, c in enumerate(channels)}
    return np.array([ci[str(c)] for c in df["chan"]], dtype=np.int8)


def build_trees(det, zyx, channels=CHANS):
    """One KD-tree per channel, in micron space. Build once per round and pass around.

    Six separate call sites used to build these independently; on R5 (19.6M spots) a
    full set costs ~7s, so the duplication was minutes per mouse for nothing.
    """
    pts = zyx * VOXEL_UM
    return {j: cKDTree(pts[det == j]) for j in range(len(channels)) if (det == j).any()}


def isolated_mask(det, zyx, k, channels=CHANS, iso_um=1.5, trees=None):
    """Spots detected in channel k with NO spot of any other channel within iso_um.

    Such a spot cannot be a ghost (a ghost has a co-located source partner by
    definition), so this is a purity criterion that makes NO assumption about which
    channel is brightest. The self-peak filter it replaces is circular: at beta ~ 1 a
    real spot IS brighter in its neighbour's channel, so requiring self-peak selects
    an atypical minority (measured: 0.9% of Sst spots) and the endmember inherits
    that bias.
    """
    pts = zyx * VOXEL_UM
    if trees is None:
        trees = {j: cKDTree(pts[det == j]) for j in range(len(channels)) if (det == j).any()}
    m = det == k
    if not m.any():
        return m
    idx = np.where(m)[0]
    q = pts[m]

    # Two optimisations over the obvious "query every tree for every point, OR the
    # results" loop, together ~5.6x faster on R5 Cck (38.1s -> 6.8s) with a
    # bit-identical result:
    #
    #   * SHORT-CIRCUIT. A spot needs only ONE neighbour to be disqualified, so once
    #     flagged it never has to be queried again. Most spots are near something (only
    #     ~5% of Sst spots are isolated), so the surviving set shrinks fast.
    #   * LARGEST TREE FIRST. Querying the densest channel first flags the most points,
    #     making every subsequent query smaller.
    #
    # workers=-1 spreads each query over all cores; the result does not depend on it.
    alive = np.arange(len(idx))
    near_other = np.zeros(len(idx), bool)
    for j in sorted((j for j in trees if j != k), key=lambda j: -trees[j].n):
        if len(alive) == 0:
            break
        d_, _ = trees[j].query(q[alive], k=1, workers=-1)
        hit = d_ < iso_um
        near_other[alive[hit]] = True
        alive = alive[~hit]

    out = np.zeros(len(det), bool)
    out[idx[~near_other]] = True
    return out


def estimate_endmembers_isolated(I, det, zyx, channels=CHANS, iso_um=1.5, percentile=95,
                                 trees=None, iso_cache=None, progress=None):
    """Endmembers from the brightest tail of SPATIALLY ISOLATED spots.

    Validated on 800995 R5: this lands within 1 deg of the single-dye control's
    predicted Cck-Sst geometry (34.8 vs 35.6), where the self-peak pool gave 12.5 and
    partner-exclusion overshot to 43.7.
    """
    # trees and iso_cache are shared with measure_beta_and_tolerance: both need the
    # same 5 KD-trees and the same 5 isolation masks, and recomputing them doubled the
    # cost of the expensive part of a round.
    if trees is None:
        trees = build_trees(det, zyx, channels)
    if iso_cache is None:
        iso_cache = {}
    E = np.eye(len(channels))
    info = {}
    for k, c in enumerate(channels):
        if progress:
            progress(f"isolation mask {c}")
        if k not in iso_cache:
            iso_cache[k] = isolated_mask(det, zyx, k, channels, iso_um, trees)
        iso = iso_cache[k]
        n_iso = int(iso.sum())
        info[c] = dict(n_detected=int((det == k).sum()), n_isolated=n_iso,
                       iso_frac=float(n_iso / max((det == k).sum(), 1)))
        if n_iso < 50:
            continue
        rows = np.where(iso)[0]
        own = I[rows, k]
        sel = rows[own > np.percentile(own, percentile)]
        if len(sel) < 10:
            sel = rows
        V = I[sel].astype(np.float64)
        nrm = np.linalg.norm(V, axis=1)
        V = V[nrm > 0] / nrm[nrm > 0, None]
        med = np.median(V, axis=0)
        E[:, k] = med / np.linalg.norm(med)
    return E, info


def measure_beta_and_tolerance(I, det, zyx, channels=CHANS, iso_um=1.5,
                               bright_pct=75, tol_pct=90, min_iso=2000,
                               trees=None, iso_cache=None):
    """Per-direction beta AND its tolerance, both measured on a labelled set.

    Isolated source spots have a victim-channel reading that IS pure bleed, so:
      * beta  = median(victim/source) over the BRIGHTEST quartile, where the victim
                channel's background floor is negligible. (Measured: the ratio is flat
                at ~1.35 above source~150 but rises to 2.7 for the dimmest spots --
                that rise is background, not bleed, so fitting on all spots biases
                beta upward.)
      * tol   = the tol_pct-th percentile of observed/(beta*source) over ALL isolated
                spots. This is the spread of genuine bleed around beta, which is
                exactly what the multiplicative tolerance is standing in for.

    Replaces the inherited beta_tol=1.5 constant. Measured p90 spans 2.0-5.3 across
    ten directions and is direction- and mouse-specific; 1.5 sits near p76 and so
    discards about a quarter of genuine bleed.
    """
    if trees is None:
        trees = build_trees(det, zyx, channels)
    if iso_cache is None:
        iso_cache = {}
    out = {}
    for si, s in enumerate(channels):
        if si not in iso_cache:
            iso_cache[si] = isolated_mask(det, zyx, si, channels, iso_um, trees)
        iso = iso_cache[si]
        if iso.sum() < min_iso:
            continue
        rows = np.where(iso)[0]
        sI = I[rows, si].astype(np.float64)
        thr = np.percentile(sI, bright_pct)
        br = sI > thr
        if br.sum() < 200:
            continue
        for vi, v in enumerate(channels):
            if vi == si:
                continue
            vI = I[rows, vi].astype(np.float64)
            beta = float(np.median(vI[br] / np.maximum(sI[br], 1e-9)))
            if not np.isfinite(beta) or beta <= 0:
                continue
            t = vI / np.maximum(beta * sI, 1e-9)
            # TOL_FLOOR guards a degenerate case: when the isolated-spot bleed ratio has
            # (near-)zero spread, the measured percentile collapses to ~1.0 and the test
            # becomes "victim <= exactly beta*source". Any mismatch between the ghost's
            # true parent and the nearest co-located source spot then fails it -- on
            # synthetic ground truth that drops recall from ~98% to 73%. Real data has
            # enough spread that the floor rarely binds.
            out[(s, v)] = dict(beta=beta,
                               tol=max(float(np.percentile(t, tol_pct)), TOL_FLOOR),
                               n_isolated=int(iso.sum()), bright_thr=float(thr))
    return out


def contaminating_sources(B_ctrl, k, channels=CHANS, ctrl_min=0.05):
    """Channels whose dye bleeds INTO channel k above ctrl_min, per the control matrix."""
    return [i for i in range(len(channels))
            if i != k and np.isfinite(B_ctrl[i, k]) and B_ctrl[i, k] >= ctrl_min]


def estimate_endmembers(I, det, channels=CHANS, percentile=95, require_selfpeak=True,
                        cell_id=None, B_ctrl=None, ctrl_min=0.05, max_partner_spots=5):
    """Purified endmembers, optionally excluding partner-rich cells.

    CRITICAL (found on 800995 R5): the self-peak + top-brightness pool is biased
    toward cells rich in the CONTAMINATING gene. The brightest "Cck" spots sat in
    cells with a median of 557 Sst transcripts (vs 11 for the pool at large), so the
    Cck endmember absorbed Sst bleed and read 0.580 in 594 where the control matrix
    predicts 0.005 -- collapsing the Cck-Sst angle to 12.5 deg. Restricting the pool
    to cells with <= max_partner_spots of each contaminating gene restores 45.4 deg,
    comparable to 788406's 47.6 deg. The near-collinearity was a selection artefact
    of endmember estimation, NOT the acquisition.

    Pass cell_id and B_ctrl to enable the exclusion; without them this is the old
    (biased) behaviour.
    """
    peak = I.argmax(1).astype(np.int8)
    E = np.eye(len(channels))
    info = {}
    use_excl = cell_id is not None and B_ctrl is not None
    if use_excl:
        codes, _uniq = pd.factorize(cell_id)
        n_cells = codes.max() + 1
        # spots per cell per channel, for the partner-richness test
        per_cell = np.zeros((n_cells, len(channels)), np.int32)
        np.add.at(per_cell, (codes, det.astype(np.int64)), 1)
    for k, c in enumerate(channels):
        m = det == k
        if require_selfpeak:
            m = m & (peak == k)
        n_selfpeak = int(m.sum())
        n_excluded = 0
        if use_excl:
            srcs = contaminating_sources(B_ctrl, k, channels, ctrl_min)
            if srcs:
                rich = (per_cell[:, srcs] > max_partner_spots).any(axis=1)
                keep_cell = ~rich[codes]
                m2 = m & keep_cell
                n_excluded = int(m.sum() - m2.sum())
                if m2.sum() >= 50:      # only apply if enough spots survive
                    m = m2
        n_pool = int(m.sum())
        info[c] = dict(n_pool=n_pool, n_selfpeak=n_selfpeak, n_excluded_partner_rich=n_excluded,
                       selfpeak_frac=float(n_selfpeak / max((det == k).sum(), 1)))
        if n_pool < 50:
            continue
        own = I[m, k]
        sel = m.copy()
        sel[m] = own > np.percentile(own, percentile)
        if sel.sum() < 10:
            sel = m
        V = I[sel].astype(np.float64)
        nrm = np.linalg.norm(V, axis=1)
        V = V[nrm > 0] / nrm[nrm > 0, None]
        med = np.median(V, axis=0)
        if np.linalg.norm(med) > 0:
            E[:, k] = med / np.linalg.norm(med)
    return E, info


def endmember_angle(a, b):
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = float(labels.sum())
    n0 = float(len(labels) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def separability(I, det, E, directions, channels=CHANS, percentile=95,
                 max_per_class=20000, rng=0):
    """Per ordered direction: purified endmember angle + isolated-spot AUC.

    The 1-D score is the channel log-intensity-ratio, the feature a nearest-line
    rule discards. AUC ~0.5 means the dyes are not separable on this panel.
    """
    ci = {c: i for i, c in enumerate(channels)}
    rs = np.random.RandomState(rng)
    peak = I.argmax(1).astype(np.int8)
    rows = []
    for (s, v) in directions:
        si, vi = ci[s], ci[v]

        def pool(k):
            # NOTE: the AUC pool must NOT use the self-peak purity filter. Purity
            # is right for ESTIMATING an endmember, but a self-peak spot has its
            # own channel brightest by construction, so the two classes become
            # trivially separable on the channel log-ratio and every AUC collapses
            # to 1.0 - the gate would never fire. Prototype v1 pools isolated
            # (non-overlap-candidate) bright spots; with no spatial pass here we
            # pool all bright detections in the channel.
            m = det == k
            if m.sum() == 0:
                return np.array([], int)
            own = I[m, k]
            idx = np.where(m)[0][own > np.percentile(own, percentile)]
            if len(idx) > max_per_class:
                idx = rs.choice(idx, max_per_class, replace=False)
            return idx

        a, b = pool(si), pool(vi)
        ang = endmember_angle(E[:, si], E[:, vi])
        if len(a) < 20 or len(b) < 20:
            rows.append(dict(source=s, victim=v, angle_deg=round(ang, 1), auc=np.nan,
                             n_iso_source=len(a), n_iso_victim=len(b), regime="no_data"))
            continue
        idx = np.concatenate([a, b])
        lab = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
        auc = _auc(np.log((I[idx, vi] + 1.0) / (I[idx, si] + 1.0)), lab)
        auc = max(auc, 1 - auc) if np.isfinite(auc) else auc   # direction-agnostic
        # Regime logic (revised after the two-mouse run).
        #
        # Deletion and reassignment rest on DIFFERENT evidence and must not share a
        # gate. Deletion requires a spatially co-located partner in the source
        # channel -- measured at ~99% above chance -- so it stands on physical
        # evidence that does not depend on spectral separability. Reassignment has
        # no spatial partner by construction: it is a purely spectral inference and
        # is the only decision AUC should gate.
        #
        # Separately, a small endmember angle makes the pairwise NNLS
        # ill-conditioned (projecting onto near-parallel vectors), so NNLS is
        # dropped from the evidence for those pairs rather than the pair being
        # refused. At 12.5 deg only 2.8% of genuinely co-located spots score
        # NNLS-dominant -- the test fails, not the spots.
        illcond = ang < ANGLE_ILLCOND
        if not np.isfinite(auc):
            reg = "no_data"
        elif auc >= AUC_TRUST and not illcond:
            reg = "delete_and_reassign"
        else:
            reg = "delete_only"
        rows.append(dict(source=s, victim=v, angle_deg=round(ang, 1),
                         auc=round(float(auc), 3), n_iso_source=len(a),
                         n_iso_victim=len(b), regime=reg, illcond=bool(illcond)))
    return pd.DataFrame(rows)


def nnls_pair(I_sub, e_own, e_other):
    """2-endmember non-negative least squares. Returns (c_own, c_other)."""
    A = np.column_stack([e_own, e_other]).astype(np.float64)
    G = A.T @ A
    B = I_sub.astype(np.float64) @ A                       # (n, 2)
    det_ = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    if abs(det_) < 1e-12:
        c = np.zeros((len(I_sub), 2))
    else:
        c0 = (B[:, 0] * G[1, 1] - B[:, 1] * G[0, 1]) / det_
        c1 = (B[:, 1] * G[0, 0] - B[:, 0] * G[1, 0]) / det_
        c = np.column_stack([c0, c1])
    # project the unconstrained solution onto the non-negative orthant
    neg0, neg1 = c[:, 0] < 0, c[:, 1] < 0
    if neg0.any():
        c[neg0, 0] = 0.0
        c[neg0, 1] = np.maximum(B[neg0, 1] / max(G[1, 1], 1e-12), 0.0)
    if neg1.any():
        c[neg1, 1] = 0.0
        c[neg1, 0] = np.maximum(B[neg1, 0] / max(G[0, 0], 1e-12), 0.0)
    return c[:, 0], c[:, 1]


def power_scaled_beta(B_ctrl, powers, channels=CHANS):
    """beta_pred(source->victim) = beta_ctrl * P(victim) / P(source)."""
    Bp = np.full_like(B_ctrl, np.nan)
    for i, s in enumerate(channels):
        for j, v in enumerate(channels):
            if i == j or not np.isfinite(B_ctrl[i, j]):
                continue
            ps, pv = powers.get(s), powers.get(v)
            if ps and pv and ps > 0:
                Bp[i, j] = B_ctrl[i, j] * (pv / ps)
    return Bp


def allowlist_directions(B_ctrl, present, channels=CHANS, ctrl_min=0.05,
                         bidirectional=True):
    """Ordered (source, victim) directions the control says physically bleed.

    Forward rule: `B_ctrl[s, v] >= ctrl_min`.

    BIDIRECTIONAL rule: if a PAIR bleeds in one direction it is a physically
    bleeding pair, so the reverse direction is admitted too even when its own control
    value falls under ctrl_min. Its magnitude is then MEASURED from isolated spots
    (see measure_beta_and_tolerance), not taken from the control.

    Why: 594/638 are spectral neighbours. Vip->Sst is 0.141 (allowlisted) while
    Sst->Vip is 0.0186 (just under the 0.05 cut), so the control admits only one
    direction of a pair it already says bleeds. The control's 7.6x asymmetry does NOT
    hold in tissue -- measured on isolated spots it is 1.5x (800995) and 3.7x
    (788406), i.e. the weak direction leaks far more than the single-dye control
    predicts. Sst->Vip's sub-voxel co-location fraction (0.089 in 800995) matches
    known bleed (Sst->Cck, 0.091 in the same mouse) and runs 220-646x above a
    displaced null.

    This is NOT the old empirical allowlist. That admitted DISTANT pairs such as
    Cck->Vip (561 and 638, two channels apart, control 0.006), where an elevated
    in-tissue ratio is the signature of co-expression rather than leakage -- panel
    design places co-expressed genes in non-neighbouring channels precisely because
    bleed there is negligible. The bidirectional rule only ever adds the reverse of a
    pair the control has already certified as bleeding.
    """
    ci = {c: i for i, c in enumerate(channels)}
    out = []
    for s in present:
        for v in present:
            if s == v:
                continue
            b = B_ctrl[ci[s], ci[v]]
            fwd = np.isfinite(b) and b >= ctrl_min
            rev = False
            if bidirectional and not fwd:
                br = B_ctrl[ci[v], ci[s]]
                rev = bool(np.isfinite(br) and br >= ctrl_min
                           and np.isfinite(b) and b > 0)
            if fwd or rev:
                out.append((s, v))
    return out


VOXEL_UM = np.array([1.0, 0.24, 0.24], np.float32)   # z, y, x microns per voxel


def coloc_nearest(victim_zyx, source_zyx, dxy=1.0, dz=1.3):
    """Anisotropic co-location (v2). For each victim spot find the nearest source
    spot and keep it only if dxy < `dxy` AND |dz| < `dz` in microns.

    z is tolerant because a spot on a voxel boundary can be binned one plane away
    (confirmed as a discrete displacement spike at exactly one z-plane), whereas
    a ghost sits essentially on top of its source in xy.

    Returns (has_partner, source_row_index) - index is only valid where
    has_partner is True.
    """
    from scipy.spatial import cKDTree
    if len(source_zyx) == 0 or len(victim_zyx) == 0:
        return np.zeros(len(victim_zyx), bool), np.zeros(len(victim_zyx), np.int64)
    q = victim_zyx * VOXEL_UM
    t = source_zyx * VOXEL_UM
    tree = cKDTree(t)
    radius = float(np.hypot(dxy, dz) * 1.5 + 1e-6)
    d, ti = tree.query(q, k=1, distance_upper_bound=radius)
    matched = np.isfinite(d) & (ti < len(t))
    ok = np.zeros(len(q), bool)
    if matched.any():
        qs, ts = q[matched], t[ti[matched]]
        dz_m = np.abs(qs[:, 0] - ts[:, 0])
        dxy_m = np.hypot(qs[:, 1] - ts[:, 1], qs[:, 2] - ts[:, 2])
        good = (dxy_m < dxy) & (dz_m < dz)
        ok[np.where(matched)[0][good]] = True
    ti = np.where(ti < len(t), ti, 0)
    return ok, ti


TOL_FLOOR = 1.15        # minimum multiplicative tolerance (see measure_beta_and_tolerance)
ANGLE_ILLCOND = 25.0     # below this, NNLS on the pair is ill-conditioned -> skip it
BETA_DISAGREE_TOL = 1.5  # |log| ratio beyond which beta_obs overrides beta_pred
DXY_TIGHT = 0.24         # tight xy gate (one voxel) for abundant-source weak-bleed pairs
# Abundance tightening is DISABLED by default (set to a finite value to enable).
# v2 used it to suppress chance co-locations under an abundant source, and in a
# removal-only pipeline it helped. In v3 it measurably hurts: tested on both mice it
# regressed Npy-Pvalb (800995 -0.097 -> 0.129), Lamp5-Calb2 (0.525 -> 0.627) and
# Calb1-Mme (0.727 -> 0.887), because v3's NNLS + endmember evidence already
# discriminates chance co-locations, and the tight gate mostly removes true ghosts.
SRC_ABUND_RATIO = float("inf")


def isolated_beta(I, det, zyx, si, vi, iso_um=1.5, min_iso=500):
    """Co-expression-immune bleed magnitude (v2 `_emp_beta_iso`).

    Take SOURCE spots with no victim-channel spot within iso_um -- those cannot be
    explained by a co-expressed victim transcript -- and read the median leak into
    the victim channel over their own-channel median, both background-subtracted.
    This is the only magnitude estimate immune to genuine co-expression, so it is
    what detects directions the single-dye control UNDER-predicts in tissue.
    """
    ms = det == si
    mv = det == vi
    if ms.sum() < min_iso or mv.sum() < 1:
        return 0.0, 0
    BGs = float(np.percentile(I[:, si], 10))
    BGv = float(np.percentile(I[:, vi], 10))
    tree = cKDTree(zyx[mv] * VOXEL_UM)
    d, _ = tree.query(zyx[ms] * VOXEL_UM, k=1)
    del tree
    iso = d > iso_um
    if iso.sum() < min_iso:
        return 0.0, int(iso.sum())
    rows = np.where(ms)[0][iso]
    own = float(np.median(I[rows, si])) - BGs
    leak = float(np.median(I[rows, vi])) - BGv
    return max(leak, 0.0) / max(own, 1.0), int(iso.sum())


def ghost_consistency(I, det, zyx, si, vi, beta, beta_tol=1.5, rad=1.0, min_near=200):
    """Two checks on a candidate direction (v2 `_ghost_consistency`):

    `gh`  = fraction of co-located victim spots dim enough to BE bleed at this beta.
    `enr` = spatial enrichment vs a displaced null (source shifted 5 um in x and y).
            enr >> 1 means the co-location is physical, not coincidental density.
    """
    ms = det == si
    mv = det == vi
    if ms.sum() < 1 or mv.sum() < 1:
        return 0.0, 0.0, 0
    BGs = float(np.percentile(I[:, si], 10))
    BGv = float(np.percentile(I[:, vi], 10))
    src = zyx[ms] * VOXEL_UM
    vic = zyx[mv] * VOXEL_UM
    tree = cKDTree(src)
    d, idx = tree.query(vic, k=1)
    vn = vic.copy(); vn[:, 1] += 5.0; vn[:, 2] += 5.0
    dn, _ = tree.query(vn, k=1)
    del tree
    near = d <= rad
    if near.sum() < min_near:
        return 0.0, 0.0, int(near.sum())
    vo = I[np.where(mv)[0][near], vi].astype(float) - BGv
    ss = I[np.where(ms)[0][idx[near]], si].astype(float) - BGs
    gh = float((vo <= beta_tol * beta * np.maximum(ss, 0)).mean())
    enr = float((d <= 0.5).mean() / max((dn <= 0.5).mean(), 1e-4))
    return gh, enr, int(near.sum())


# Directions known from instrument/optics knowledge to carry real contamination that
# neither the single-dye control nor the automatic detector captures. 488<->594 are far
# apart spectrally, so the control matrix has essentially nothing there (488->594 =
# 0.022, 594->488 = 0.0009), and the automatic detector rejects them because their
# dim-ghost fraction is ~0.01 against a 0.40 threshold -- the leak is not a dim
# spectral tail but a filter artifact, so its ghosts are not dim. They DO show 4.0x and
# 4.2x spatial enrichment over a displaced null, which is the evidence that they are
# physical. Magnitude comes from the isolated-spot estimate, as for any other
# empirically-added direction.
# EMPIRICAL ALLOWLIST IS OFF BY DEFAULT.
#
# Measured beta exceeds the power-scaled control prediction by 4-223x on every WEAK
# direction while agreeing within 1.7x on every strong one. That pattern is not
# under-predicted bleed -- it is TRUE CO-EXPRESSION. Panel design deliberately places
# co-expressed genes in non-neighbouring channels precisely because bleed there is
# negligible, so an elevated in-tissue ratio between distant channels is the expected
# signature of co-expression, not of leakage. Removing those "ghosts" would delete real
# co-expressed transcripts.
#
# Two independent measurements confirm this (800995 R5):
#   * fraction of victim spots with a SUB-VOXEL (<0.35 um) source partner --
#     bleed sits on top of its source; co-expressed transcripts are near but not
#     identical. Strong control pairs: 0.146-0.256. Cck->Vip: 0.064. Npy->Sst: 0.019.
#   * intensity coupling r(log I_source, log I_victim) among sub-voxel pairs --
#     bleed scales with its source. Sst->Cck: 0.967. Cck->Vip: 0.032 (none).
#
# A direction may be added empirically ONLY with positive evidence on BOTH axes, which
# no auto-detected direction currently satisfies. KNOWN_LEAK_PAIRS is retained as a
# hook but is likewise off: 488<->594 shows sub-voxel fractions of 0.019/0.048 and a
# median co-located ratio 160-315x its control prediction, which is the co-expression
# signature (Npy is expressed in Sst cells), not a filter leak of that magnitude.
USE_EMPIRICAL_DEFAULT = False
KNOWN_LEAK_PAIRS = []


def empirical_allowlist(I, det, zyx, B_ctrl, channels=CHANS, ctrl_min=0.05,
                        emp_min=0.05, ghost_min=0.40, under_factor=3.0,
                        beta_tol=1.5, enr_min=3.0, known_pairs=KNOWN_LEAK_PAIRS,
                        known_enr_min=2.0):
    """Directions the single-dye control UNDER-predicts in tissue (v2).

    The control matrix is optical ground truth for a dye in isolation, but in tissue
    some directions leak more than the control implies (v2 named Sst->Vip). Those
    directions are absent from the control allowlist entirely, so no amount of
    magnitude correction reaches them -- they need to be ADDED. A direction is added
    only when three independent conditions hold: an isolated-spot beta above
    emp_min AND at least under_factor x the control value; a dim-ghost majority
    (gh >= ghost_min); and spatial enrichment over a displaced null (enr >= enr_min).
    """
    ci = {c: k for k, c in enumerate(channels)}
    present = [c for c in channels if (det == ci[c]).any()]
    allow = {}
    for s in present:
        for v in present:
            if s == v:
                continue
            si, vi = ci[s], ci[v]
            ctrl = float(B_ctrl[si, vi]) if np.isfinite(B_ctrl[si, vi]) else 0.0
            if ctrl >= ctrl_min:
                continue                      # already on the control allowlist
            eb, n_iso = isolated_beta(I, det, zyx, si, vi)
            # A KNOWN_LEAK pair gets a lower magnitude floor: 594->488 leaks at only
            # 0.024 but that is still 27x its control value (0.0009), and a weak leak
            # out of an abundant channel still deposits many ghosts.
            floor = emp_min * 0.25 if (s, v) in (known_pairs or []) else emp_min
            if eb < floor or eb < under_factor * max(ctrl, 1e-3):
                continue
            gh, enr, n_near = ghost_consistency(I, det, zyx, si, vi, eb, beta_tol=beta_tol)
            known = (s, v) in (known_pairs or [])
            # A KNOWN_LEAK pair bypasses the dim-ghost majority test but still has to
            # clear spatial enrichment -- the evidence that the contamination is real
            # is that it co-locates, not that it is dim.
            ok = (gh >= ghost_min and enr >= enr_min) or (known and enr >= known_enr_min)
            if ok:
                allow[(s, v)] = dict(beta=round(float(eb), 4), ctrl=round(ctrl, 4),
                                     ghost_frac=round(gh, 3), enrichment=round(enr, 1),
                                     n_isolated=n_iso, n_near=n_near,
                                     source="known_leak" if known and gh < ghost_min else "auto")
    return allow


def observed_beta(E, si, vi):
    """Bleed magnitude read off the purified SOURCE endmember: how much the source
    dye puts into the victim channel relative to its own. Independent of laser-power
    bookkeeping, so it is the empirical check on the forward prediction."""
    own = float(E[si, si])
    return float(E[vi, si] / own) if own > 1e-9 else np.nan


def unmix_v3(df, B_ctrl, powers, channels=CHANS, ctrl_min=0.05, margin=1.0,
             beta_tol=1.5, require_selfpeak=True, dxy=1.0, dz=1.3,
             same_cell=True, angle_illcond=ANGLE_ILLCOND,
             beta_disagree_tol=BETA_DISAGREE_TOL, dxy_tight=DXY_TIGHT,
             src_abund_ratio=SRC_ABUND_RATIO, use_empirical=USE_EMPIRICAL_DEFAULT,
             use_measured_beta=True, tol_pct=90, amb_bg_mult=1.0,
             isolated_endmembers=True, fg_bg=None, bidirectional=True,
             verbose=True):
    """Run v3 on one round.

    Returns (spots_out, sep_table, decision_log). `spots_out` carries every input
    spot with provenance columns; NOTHING is deleted from the frame - the
    `v3_action` column records what a downstream filter should do.
    """
    import time as _time
    _t0 = _time.time()
    _N_STEPS = 7

    def _step(i, msg):
        """Progress line. A round takes minutes, so silence looks like a hang."""
        if verbose:
            print(f"    [{i}/{_N_STEPS}] {msg}  ({_time.time() - _t0:.0f}s)", flush=True)

    ci = {c: i for i, c in enumerate(channels)}
    I = intensity_matrix(df, channels)
    det = detection_index(df, channels)
    present = [c for c in channels if (det == ci[c]).any()]
    zyx = df[["z", "y", "x"]].to_numpy(np.float32)
    cell_id = df["cell_id"].to_numpy()
    _step(1, f"loaded {len(df):,} spots, {len(present)} channels")

    # ONE set of KD-trees and ONE isolation mask per channel, shared by the endmember
    # estimator, the beta/tolerance measurement and the per-direction co-location.
    trees = build_trees(det, zyx, channels)
    iso_cache = {}
    _step(2, "built spatial index")

    # Endmembers are estimated with partner-rich cells excluded (see
    # estimate_endmembers docstring) -- without this the endmember for an abundant
    # channel absorbs its neighbour's bleed and the pair looks collinear.
    if isolated_endmembers:
        E, eminfo = estimate_endmembers_isolated(
            I, det, zyx, channels, trees=trees, iso_cache=iso_cache,
            progress=(lambda m: _step(3, f"dye lines: {m}")) if verbose else None)
    else:
        E, eminfo = estimate_endmembers(I, det, channels, require_selfpeak=require_selfpeak,
                                        cell_id=cell_id, B_ctrl=B_ctrl, ctrl_min=ctrl_min)
    _step(3, "dye lines estimated from isolated spots")
    dirs = allowlist_directions(B_ctrl, present, channels, ctrl_min,
                                bidirectional=bidirectional)
    sep = separability(I, det, E, dirs, channels)
    _step(4, f"allowlist: {len(dirs)} directions to consider")
    B_pred = power_scaled_beta(B_ctrl, powers, channels)

    n = len(df)
    action = np.zeros(n, np.int8)          # 0 keep, 1 delete, 2 reassign
    assigned = det.copy()
    src_chan = np.full(n, "", object)
    rule = np.full(n, "none", object)
    beta_used = np.full(n, np.nan, np.float32)
    coef_own = np.full(n, np.nan, np.float32)
    coef_cross = np.full(n, np.nan, np.float32)
    is_ambiguous = np.zeros(n, bool)
    bg = np.array([np.percentile(I[:, k], 10) for k in range(len(channels))], np.float64)
    meas = measure_beta_and_tolerance(I, det, zyx, channels, tol_pct=tol_pct,
                                      trees=trees, iso_cache=iso_cache) \
        if use_measured_beta else {}
    log = []

    # Directions the control matrix misses entirely (v2). These are real in-tissue
    # contamination with no control entry, so they can only be ADDED, never fixed by
    # magnitude correction. Their magnitude comes from the isolated-spot estimate.
    _step(5, f"measured beta + tolerance for {len(meas)} directions")
    emp = empirical_allowlist(I, det, zyx, B_ctrl, channels, ctrl_min=ctrl_min) \
        if use_empirical else {}
    sep_rows = list(sep.iterrows())
    known = {(r.source, r.victim) for _, r in sep_rows}
    for (es, ev), info in emp.items():
        if (es, ev) not in known:
            sep_rows.append((None, pd.Series(dict(
                source=es, victim=ev, angle_deg=endmember_angle(E[:, ci[es]], E[:, ci[ev]]),
                auc=np.nan, regime="delete_only", illcond=False, empirical=True))))

    _n_dirs = len(sep_rows)
    for _i_dir, (_, row) in enumerate(sep_rows, start=1):
        if verbose:
            print(f"    [6/{_N_STEPS}] direction {_i_dir}/{_n_dirs}: "
                  f"{row.source}->{row.victim}  ({_time.time() - _t0:.0f}s)", flush=True)
        s, v, reg = row.source, row.victim, row.regime
        si, vi = ci[s], ci[v]
        is_emp = (s, v) in emp
        beta_pred = float(emp[(s, v)]["beta"]) if is_emp else B_pred[si, vi]
        beta_rev = B_ctrl[vi, si]
        illcond = bool(row.get("illcond", row.angle_deg < angle_illcond))

        # Magnitude: the power-scaled control prediction is right on average
        # (RMS log-error 0.293 over 18 directions vs 0.696 unscaled), but on
        # individual directions it can be off by >1.5x. beta_obs, read off the
        # purified source endmember, is the empirical check -- and for the
        # near-collinear R5 594->561 case it is stable across mice (0.873 / 0.909)
        # where beta_pred is not (0.93 / 1.40). Prefer the prediction; fall back to
        # the observation when they disagree beyond tolerance.
        beta_obs = observed_beta(E, si, vi)
        beta, beta_src = beta_pred, "emp" if is_emp else "pred"
        if (not is_emp and np.isfinite(beta_pred) and np.isfinite(beta_obs) and beta_obs > 0
                and max(beta_pred / beta_obs, beta_obs / beta_pred) > beta_disagree_tol):
            beta, beta_src = beta_obs, "obs"

        # Measured beta and measured tolerance, both from the labelled isolated-spot
        # set, override the prediction and the inherited 1.5 constant when available.
        beta_tol_used = beta_tol
        if (s, v) in meas:
            beta, beta_src = meas[(s, v)]["beta"], "measured"
            beta_tol_used = float(meas[(s, v)]["tol"])

        m = det == vi                       # spots detected in the VICTIM channel
        # Abundance-aware gate width (v2). An abundant source produces many CHANCE
        # co-locations with genuinely dim victim spots, and a dim real victim passes
        # the intensity test *because* it is dim -- so intensity cannot protect it and
        # only a tight gate can. Tighten to one voxel when the source is abundant AND
        # the bleed is weak; keep wide for strong bleed (beta >= 1) and for
        # empirically-added directions, whose ghosts are spatially displaced.
        n_src = int((det == si).sum()); n_vic = int(m.sum())
        abund = n_src / max(n_vic, 1)
        tighten = (not is_emp) and np.isfinite(beta) and beta < 1.0 and abund >= src_abund_ratio
        gxy = dxy_tight if tighten else dxy
        base = dict(source=s, victim=v, regime=reg, auc=row.auc, angle_deg=row.angle_deg,
                    illcond=illcond, empirical=is_emp,
                    beta_pred=None if not np.isfinite(beta_pred) else round(float(beta_pred), 3),
                    beta_obs=None if not np.isfinite(beta_obs) else round(float(beta_obs), 3),
                    beta_source=beta_src, beta_tol_used=round(float(beta_tol_used),2), gate_xy=gxy, tightened=bool(tighten),
                    src_abund=round(abund, 2), n_victim_spots=n_vic)
        if reg == "no_data" or not m.any() or not np.isfinite(beta) or beta <= 0:
            log.append(dict(base, n_flagged=0, note="no action"))
            continue

        idx = np.where(m)[0]
        sidx = np.where(det == si)[0]
        c_own, c_cross = nnls_pair(I[idx], E[:, vi], E[:, si])
        dominant = c_cross > margin * c_own

        # --- spatial co-location against the SOURCE channel (v2 / capsule) ---
        # A ghost is a duplicate detection of a real source molecule, so it has a
        # spatially coincident partner. That case is a DELETION (the duplicate is
        # spurious). Only a cross-dominant spot with NO partner can be a genuine
        # transcript detected in the wrong channel -> REASSIGNMENT.
        has_partner, part_i = coloc_nearest(zyx[idx], zyx[sidx], dxy=gxy, dz=dz)
        if same_cell and has_partner.any():
            same = cell_id[idx] == cell_id[sidx[part_i]]
            has_partner &= same

        # intensity consistency with the power-scaled forward magnitude, measured
        # against the PARTNER's source-channel intensity where one exists (v2) and
        # against the spot's own source-channel reading otherwise
        vic_I = I[idx, vi].astype(np.float64)
        src_I = np.where(has_partner, I[sidx[part_i], si], I[idx, si]).astype(np.float64)
        # AMBIGUITY FLAG. Deletion needs positive evidence that the spot IS bleed; a
        # spot too dim to discriminate is neither confidently bleed nor confidently
        # real. Flag those rather than silently keeping them, so the cell x gene step
        # can choose. The criterion is NOT victim brightness alone -- measured on
        # isolated spots, real Cck is DIMMER than true bleed (median 561 of 113 vs
        # 180), so a plain brightness cut removes real signal faster than ghosts. What
        # separates them is the SOURCE reading: a true ghost sits under a bright
        # source (median 594 = 63), a real victim spot does not (median 594 = 9).
        # Ambiguous = the predicted bleed is itself at or below the victim channel's
        # noise floor, so the test has no dynamic range.
        pred_bleed = beta * np.maximum(src_I, 0.0)
        ambiguous = pred_bleed <= amb_bg_mult * bg[vi]
        consistent = vic_I <= beta_tol_used * beta * np.maximum(src_I, 0.0)
        reverse = src_I <= beta_tol_used * max(float(beta_rev) if np.isfinite(beta_rev) else 0.0, 1e-6) * vic_I

        # Evidence for DELETION. Where the pair is well conditioned, require NNLS
        # cross-dominance as well. Where it is ill-conditioned, NNLS carries no
        # information (near-parallel projection) and is dropped: co-location plus
        # magnitude consistency plus the reverse-direction guard is the evidence.
        if illcond:
            del_mask = has_partner & consistent & ~reverse
        else:
            del_mask = has_partner & dominant & consistent & ~reverse
        # An ambiguous spot is never deleted; it is kept and flagged.
        amb_hit = idx[has_partner & ambiguous & ~del_mask]
        is_ambiguous[amb_hit] = True
        del_mask &= ~ambiguous

        # Evidence for REASSIGNMENT. No spatial partner exists by construction, so
        # this is a purely spectral claim: it needs NNLS dominance and a pair the
        # separability diagnostic trusts. Never fires on an ill-conditioned pair.
        # A tightened gate is an admission of SPATIAL uncertainty: the source is
        # abundant, so a wide gate would catch chance co-locations. It is not
        # evidence that a no-partner spot is a genuine wrong-channel transcript --
        # under a tight gate most true ghosts also fail co-location. Allowing
        # reassignment here converts deletions into reassignments and inflates the
        # source channel (measured: Calb1 +12.9%, Lamp5 +16.5%). v2 was removal-only
        # so it never hit this; v3 must suppress reassignment on tightened pairs.
        rea_mask = ((~has_partner) & dominant & consistent & ~reverse
                    & (reg == "delete_and_reassign") & (not tighten) & (not is_emp))

        hd = idx[del_mask]; hd = hd[action[hd] == 0]
        action[hd] = 1
        src_chan[hd] = s
        rule[hd] = "coloc_delete_illcond" if illcond else "coloc_delete"
        beta_used[hd] = beta
        hr = idx[rea_mask]; hr = hr[action[hr] == 0]
        action[hr] = 2; assigned[hr] = si
        src_chan[hr] = s; rule[hr] = "nnls_reassign_nopartner"; beta_used[hr] = beta

        coef_own[idx] = c_own
        coef_cross[idx] = c_cross
        log.append(dict(base, n_ambiguous=int((has_partner & ambiguous).sum()),
                        n_coloc=int(has_partner.sum()),
                        n_cross_dominant=int(dominant.sum()),
                        n_deleted=int(len(hd)), n_reassigned=int(len(hr)),
                        n_flagged=int(len(hd) + len(hr))))

    out = df.copy()
    _step(7, f"decided: {int((action==1).sum()):,} delete, "
             f"{int((action==2).sum()):,} reassign, {int((action==0).sum()):,} keep")
    out["v3_action"] = pd.Categorical.from_codes(action, ["keep", "delete", "reassign"])
    out["v3_ambiguous"] = is_ambiguous
    out["v3_chan"] = [channels[i] for i in assigned]
    out["crosstalk_source_chan"] = src_chan
    out["decision_rule"] = rule
    out["beta_used"] = beta_used
    out["nnls_coef_own"] = coef_own
    out["nnls_coef_cross"] = coef_cross

    # Raw foreground and local background, carried through UNFILTERED so downstream
    # steps can apply their own spot-quality threshold. The delivered
    # mixed_spots_{R}.pkl keeps only FG-BG, which cannot distinguish 300-over-100 from
    # 300-over-900; local BG spans 134-423 within the Cck channel alone. Filtering is
    # deliberately NOT done here -- see attach_fg_bg docstring for why the threshold
    # belongs at cell x gene construction.
    if fg_bg is not None:
        out["fg"] = fg_bg[0]
        out["bg"] = fg_bg[1]
        out["fg_over_bg"] = np.where(fg_bg[1] > 0, fg_bg[0] / np.maximum(fg_bg[1], 1e-9), np.nan)
    return out, sep, pd.DataFrame(log), E, eminfo


def attach_fg_bg(out, diag_paths, channels=CHANS):
    """Join raw FG and local BG onto an unmixed spot table, in the DETECTED channel.

    `diag_paths` maps channel -> path of that round's
    `channel_{c}_stats/image_data_channel_{c}_versus_spots_{c}.csv`, which carries
    Z,Y,X,dist,r,SEG_ID,FG,BG. Join is on exact (Z,Y,X) PER CHANNEL -- each channel's
    stats file has its own row order, and `chan_spot_id` is NOT a row index into it
    (positional indexing gives r=0.70/0.56 against the pipeline's subtracted value;
    the coordinate join gives r=1.000000 on all five channels).

    Two traps, both hit during development:
      * There are often TWO processed assets per round and only one matches the
        pairwise-unmixing spot set. Resolve it from `ds_config.json`'s
        `dataset_folder`, never by picking a timestamp.
      * FG-BG reproduces `chan_{c}_intensity` exactly, so a mismatch means the wrong
        asset, not a units problem.

    Filtering is left to the caller. Measured on both mice, an FG >= median(BG) +
    2*MAD(BG) cut removes 8.7-18.9% (800995) and 6.1-9.6% (788406) of spots, improves
    9 of 16 marker-pair correlations, but regresses Lamp5-Calb2 by +0.27 in BOTH mice
    and costs 1,802 Gad2+ cells in 800995 against 18 in 788406 -- so it is a
    reasonable per-analysis option and a poor default.
    """
    dk = (out.z.astype(np.int64) * 100000000
          + out.y.astype(np.int64) * 10000 + out.x.astype(np.int64)).to_numpy()
    chan = out.chan.astype(str).to_numpy()
    fg = np.full(len(out), np.nan, np.float32)
    bg = np.full(len(out), np.nan, np.float32)
    for c in channels:
        sel = chan == c
        if not sel.any() or c not in diag_paths:
            continue
        a = pd.read_csv(diag_paths[c], usecols=["Z", "Y", "X", "FG", "BG"],
                        dtype={"Z": np.int32, "Y": np.int32, "X": np.int32,
                               "FG": np.float32, "BG": np.float32})
        key = (a.Z.astype(np.int64) * 100000000
               + a.Y.astype(np.int64) * 10000 + a.X.astype(np.int64)).to_numpy()
        for col, dest in (("FG", fg), ("BG", bg)):
            s = pd.Series(a[col].to_numpy(), index=key)
            s = s[~s.index.duplicated()]
            dest[sel] = pd.Series(dk[sel]).map(s).to_numpy()
        del a, key
    return fg, bg


def bg_mad_threshold(bg, n_mad=2.0):
    """median(BG) + n_mad * MAD(BG), the per-channel spot-quality floor.

    MAD not SD: BG is right-skewed and heavy-tailed because it tracks tissue density
    (Vip R5 800995: SD 98 vs MAD 13), so a 2-SD threshold lands ABOVE the FG median in
    three of five channels and discards 38-65% of spots. Symmetric bounds on FG fail
    outright -- median(FG) - 2*SD(FG) is negative in all five channels.
    """
    bg = np.asarray(bg, float)
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        return np.nan
    med = float(np.median(bg))
    return med + n_mad * float(np.median(np.abs(bg - med))) * 1.4826


def cellxgene(spots, gene_map, round_key, apply_v3=True, min_intensity=None):
    """Cell x gene spot counts. `apply_v3` drops deleted spots and honours
    reassignment; otherwise counts every spot in its detected channel."""
    if apply_v3:
        keep = spots.v3_action != "delete"
        chan_col = spots.loc[keep, "v3_chan"].astype(str)
        cells = spots.loc[keep, "cell_id"]
    else:
        chan_col = spots["chan"].astype(str)
        cells = spots["cell_id"]
    gene = chan_col.map(gene_map)
    d = pd.DataFrame({"cell_id": cells.to_numpy(),
                      "chan": chan_col.to_numpy(),
                      "gene": gene.to_numpy()}).dropna(subset=["gene", "cell_id"])
    d["round_chan_gene"] = round_key + "-" + d["chan"].astype(str) + "-" + d["gene"].astype(str)
    return (d.groupby(["cell_id", "round_chan_gene"]).size()
             .rename("spot_count").reset_index())
