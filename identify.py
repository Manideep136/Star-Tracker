"""Star identification: which catalog star is each detected blob?

Idea (angle matching + voting):
  * The angle between two stars does not depend on how the camera is rotated.
  * So we measure the angle between every pair of detected stars and look for
    catalog pairs with the same angle. Each match is one vote for
    "this blob is that catalog star".
  * A blob's true identity collects a vote from every other blob; wrong
    identities get only scattered votes.
  * We then solve the attitude, throw out blobs that disagree with it,
    project the whole catalog into the image to confirm, and re-solve using
    every matched star.
"""
import numpy as np
from scipy.spatial import cKDTree

import simulator as sim
import attitude as att

DEG_PER_PX = sim.FOV_DEG / sim.IMG_SIZE


class StarDatabase:
    """Catalog unit vectors plus a sorted table of all star-pair angles."""

    def __init__(self, catalog, max_mag=6.0, max_angle_deg=None):
        stars = [s for s in catalog if s[3] <= max_mag]
        self.names = [s[0] for s in stars]
        self.mags = np.array([s[3] for s in stars])
        self.vecs = np.array([sim.radec_to_unit(s[1], s[2]) for s in stars])

        if max_angle_deg is None:                       # image diagonal + margin
            max_angle_deg = sim.FOV_DEG * np.sqrt(2) + 1.0
        chord = 2 * np.sin(np.radians(max_angle_deg) / 2)
        pairs = cKDTree(self.vecs).query_pairs(chord, output_type="ndarray")
        dots = np.einsum("ij,ij->i", self.vecs[pairs[:, 0]], self.vecs[pairs[:, 1]])
        ang = np.arccos(np.clip(dots, -1, 1))
        order = np.argsort(ang)
        self.pair_a = pairs[order, 0]
        self.pair_b = pairs[order, 1]
        self.pair_ang = ang[order]


def identify_stars(db, det_xy, tol_px=0.7, min_votes=3, max_stars=12,
                   match_px=2.0, min_fraction=0.6):
    """Identify detected stars and solve the camera attitude.

    det_xy: (n, 2) pixel centroids, brightest first.
    Returns None if no trustworthy answer, else a dict with
      R          attitude matrix (inertial -> camera)
      matches    list of (detection index, catalog index)
      n_detected, n_matched
    """
    det_xy = np.asarray(det_xy, dtype=float).reshape(-1, 2)
    n_det = len(det_xy)
    if n_det < 4:
        return None

    # --- 1. Vote using the brightest few detections --------------------------
    vecs = att.pixels_to_vectors(det_xy[:max_stars])
    n = len(vecs)
    tol = np.radians(tol_px * DEG_PER_PX)
    votes = np.zeros((n, len(db.vecs)), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            ang = np.arccos(np.clip(vecs[i] @ vecs[j], -1, 1))
            lo = np.searchsorted(db.pair_ang, ang - tol)
            hi = np.searchsorted(db.pair_ang, ang + tol)
            if hi == lo:
                continue
            a, b = db.pair_a[lo:hi], db.pair_b[lo:hi]
            # the pair could be matched in either order, so vote both ways
            np.add.at(votes[i], a, 1)
            np.add.at(votes[j], b, 1)
            np.add.at(votes[i], b, 1)
            np.add.at(votes[j], a, 1)

    best = votes.argmax(axis=1)
    best_votes = votes[np.arange(n), best]
    use = [i for i in range(n) if best_votes[i] >= min_votes]

    # --- 2. Solve, drop the worst-fitting blob, repeat until all agree --------
    while len(use) >= 4:
        R = att.solve_attitude(vecs[use], db.vecs[best[use]])
        resid = np.linalg.norm(vecs[use] - db.vecs[best[use]] @ R.T, axis=1)
        worst = int(np.argmax(resid))
        if resid[worst] <= 3 * tol:
            break
        use.pop(worst)
    if len(use) < 4:
        return None

    # --- 3. Verify: project the whole catalog and match every detection -------
    cam = db.vecs @ R.T
    z = cam[:, 2]
    half = sim.IMG_SIZE / 2
    front = np.where(z > 0.1)[0]
    px = sim.FOCAL_PX * cam[front, 0] / z[front] + half
    py = sim.FOCAL_PX * cam[front, 1] / z[front] + half
    inside = (px > -5) & (px < sim.IMG_SIZE + 5) & (py > -5) & (py < sim.IMG_SIZE + 5)
    cand = front[inside]
    if len(cand) == 0:
        return None
    tree = cKDTree(np.column_stack([px[inside], py[inside]]))
    dist, k = tree.query(det_xy, distance_upper_bound=match_px)

    best_for_cat = {}
    for i in range(n_det):
        if np.isfinite(dist[i]):
            c = int(cand[k[i]])
            if c not in best_for_cat or dist[i] < best_for_cat[c][1]:
                best_for_cat[c] = (i, dist[i])
    matches = [(i, c) for c, (i, _) in best_for_cat.items()]
    if len(matches) < 4 or len(matches) < min_fraction * n_det:
        return None

    # --- 4. Final solve using every matched star ------------------------------
    di = [m[0] for m in matches]
    ci = [m[1] for m in matches]
    R = att.solve_attitude(att.pixels_to_vectors(det_xy[di]), db.vecs[ci])
    return {"R": R, "matches": matches, "n_detected": n_det, "n_matched": len(matches)}


def identify_ransac(db, det_xy, tol_px=1.5, score_px=3.0, match_px=2.0, max_stars=15,
                    min_inliers=6, min_sep_px=100.0):
    """Robust identification for real images full of non-catalog blobs.

    Instead of letting every detection vote, try hypotheses: take two detections,
    take every catalog pair with the same angle between them, compute the rotation
    that maps one onto the other, and count how many detections land on catalog
    stars under that rotation. The hypothesis with the most hits wins. A wrong
    hypothesis gets almost no hits, so junk blobs cannot fool it.

    det_xy: (n, 2) pixel centroids, brightest first.  Returns the same dict as
    identify_stars (R, matches, n_detected, n_matched) or None.
    """
    det_xy = np.asarray(det_xy, dtype=float).reshape(-1, 2)
    n_det = len(det_xy)
    if n_det < 4:
        return None
    sub = det_xy[:max_stars]
    n = len(sub)
    vecs = att.pixels_to_vectors(sub)
    tol = np.radians(tol_px * DEG_PER_PX)
    half = sim.IMG_SIZE / 2
    fov_rad = np.radians(sim.IMG_SIZE * DEG_PER_PX * np.sqrt(2) / 2 + 1.0)
    det_tree = cKDTree(sub)

    def project(R, margin=5):
        cam = db.vecs @ R.T
        z = cam[:, 2]
        ok = np.where(z > np.cos(fov_rad))[0]
        px = sim.FOCAL_PX * cam[ok, 0] / z[ok] + half
        py = sim.FOCAL_PX * cam[ok, 1] / z[ok] + half
        inside = (px > -margin) & (px < sim.IMG_SIZE + margin) & (py > -margin) & (py < sim.IMG_SIZE + margin)
        return ok[inside], np.column_stack([px[inside], py[inside]])

    def count_matches(R, radius):
        cand, pts = project(R)
        if len(cand) == 0:
            return [], 0
        dist, k = det_tree.query(pts, distance_upper_bound=radius)
        found = {}
        for ci in range(len(cand)):
            if np.isfinite(dist[ci]):
                d_idx = int(k[ci])
                if d_idx not in found or dist[ci] < found[d_idx][1]:
                    found[d_idx] = (int(cand[ci]), float(dist[ci]))
        return [(d, c) for d, (c, _) in found.items()], len(found)

    # Score every hypothesis for a detection pair in one batch (fast):
    #   rotation from the two vector pairs (TRIAD), then ask how many detections,
    #   rotated back to the sky, land on a catalog star.
    cat_tree = cKDTree(db.vecs)
    score_chord = 2 * np.sin(np.radians(score_px * DEG_PER_PX) / 2)
    best_n, best_R = 0, None
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(sub[i] - sub[j])) < min_sep_px:
                continue
            ang = np.arccos(np.clip(vecs[i] @ vecs[j], -1, 1))
            lo = np.searchsorted(db.pair_ang, ang - tol)
            hi = np.searchsorted(db.pair_ang, ang + tol)
            if hi == lo:
                continue
            CA = np.concatenate([db.pair_a[lo:hi], db.pair_b[lo:hi]])
            CB = np.concatenate([db.pair_b[lo:hi], db.pair_a[lo:hi]])
            t1 = vecs[i]
            t2 = np.cross(vecs[i], vecs[j]); t2 /= np.linalg.norm(t2)
            t3 = np.cross(t1, t2)
            r1 = db.vecs[CA]
            r2 = np.cross(r1, db.vecs[CB]); r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
            r3 = np.cross(r1, r2)
            Rm = (np.einsum("a,mb->mab", t1, r1) + np.einsum("a,mb->mab", t2, r2)
                  + np.einsum("a,mb->mab", t3, r3))                 # (M, 3, 3): inertial -> camera
            dirs = np.einsum("mab,ka->mkb", Rm, vecs)                # detections back on the sky
            dist, _ = cat_tree.query(dirs.reshape(-1, 3), distance_upper_bound=score_chord)
            cnt = np.isfinite(dist).reshape(len(CA), n).sum(axis=1)
            m = int(np.argmax(cnt))
            if cnt[m] > best_n:
                best_n, best_R = int(cnt[m]), Rm[m]
    if best_R is None or best_n < min_inliers:
        return None

    # refine: re-solve from all matches, re-match, repeat
    R = best_R
    for _ in range(3):
        matches, cnt = count_matches(R, match_px + 1.0)
        if cnt < 4:
            return None
        di = [m[0] for m in matches]
        ci = [m[1] for m in matches]
        R = att.solve_attitude(att.pixels_to_vectors(sub[di]), db.vecs[ci])
    matches, cnt = count_matches(R, match_px)
    if cnt < min_inliers:
        return None
    return {"R": R, "matches": matches, "n_detected": n_det, "n_matched": cnt}