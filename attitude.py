"""Attitude math: pixels -> direction vectors, solve rotation, measure error.

The problem solved here is Wahba's problem: find the rotation R that best maps
catalog directions (inertial frame) onto measured directions (camera frame).
QUEST and the SVD method below are two solvers for the same problem; SVD is
shorter and very robust, so we use it first.
"""
import numpy as np

import simulator as sim

ARCSEC = 206264.806  # arcseconds per radian


def pixels_to_vectors(xy):
    """Pixel coordinates (n, 2) -> unit direction vectors (n, 3) in the camera frame.

    Inverse of the pinhole projection used in simulator.project_stars.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    half = sim.IMG_SIZE / 2
    v = np.column_stack([(xy[:, 0] - half) / sim.FOCAL_PX,
                         (xy[:, 1] - half) / sim.FOCAL_PX,
                         np.ones(len(xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def solve_attitude(body, ref, weights=None):
    """Best rotation R with  body ~ R @ ref  (rows are unit vectors).

    body: measured directions in the camera frame  (n, 3)
    ref:  matching catalog directions, inertial     (n, 3)
    """
    body, ref = np.asarray(body), np.asarray(ref)
    w = np.ones(len(body)) if weights is None else np.asarray(weights)
    B = (body * w[:, None]).T @ ref                  # sum of w * b * r^T
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))  # force a proper rotation
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def attitude_error(R_est, R_true):
    """Error between two attitudes, in arcseconds.

    Returns total error, cross-boresight error (pointing: about camera x and y)
    and roll error (about the boresight, camera z). Roll is always the worst
    axis on a star tracker, which Liebe's paper explains.
    """
    dR = R_est @ R_true.T
    eps = 0.5 * np.array([dR[2, 1] - dR[1, 2],
                          dR[0, 2] - dR[2, 0],
                          dR[1, 0] - dR[0, 1]])      # small-angle error vector
    return {
        "total": float(np.linalg.norm(eps) * ARCSEC),
        "cross": float(np.hypot(eps[0], eps[1]) * ARCSEC),
        "roll": float(abs(eps[2]) * ARCSEC),
    }
