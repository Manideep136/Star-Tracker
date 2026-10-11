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
