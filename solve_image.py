"""Solve a real star image: find the stars, identify them, print where the camera points.

Run:  python solve_image.py pleiades_dss.fits
      python solve_image.py photo.jpg --fov 20            (JPG/PNG: give the full width in degrees)
      python solve_image.py pleiades_dss.fits --truth 56.75 24.12

Options:
  --fov DEG       field of view across the image WIDTH, degrees (FITS files: read from header)
  --truth RA DEC  true centre in degrees; the script prints the error against it
  --psf PX        star blur in pixels used by the detector (default 1.5)
  --k SIGMA       detection threshold in noise sigmas (default 5)
  --top N         try only the N brightest detections (default: tries 15, 25, 40)

How it works: detect (detect.py) -> keep the brightest few -> match star-pair angles
against the catalog (identify.py) -> solve the rotation (attitude.py) -> report.
Image convention: north up, east left (as on the sky). Output "north angle" is how far
north is rotated clockwise from straight up.
"""
import argparse
import sys

import numpy as np
import cv2

import simulator as sim
import identify
import attitude as att
import detect


def load_image(path, fov_arg):
    """Returns (gray float image with north up, deg_per_px, truth (ra, dec) or None)."""
    truth, deg_per_px = None, None
    if path.lower().endswith((".fits", ".fit", ".fts")):
        from astropy.io import fits
        with fits.open(path) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            data = np.array(hdu.data, dtype=np.float64)
            hdr = hdu.header
        while data.ndim > 2:
            data = data[0]
        data = np.flipud(data)                      # FITS row 0 is the bottom of the sky image
        if "CDELT2" in hdr:
            deg_per_px = abs(float(hdr["CDELT2"]))
        if "CRVAL1" in hdr and "CRVAL2" in hdr:
            truth = (float(hdr["CRVAL1"]), float(hdr["CRVAL2"]))
        gray = data
    else:
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        if color is None:
            sys.exit("Could not read image: " + path)
        gray = color.astype(np.float64).mean(axis=2)
    if fov_arg is not None:
        deg_per_px = fov_arg / gray.shape[1]
    if deg_per_px is None:
        sys.exit("I need the field of view: add  --fov <degrees across the image width>")
    return gray, deg_per_px, truth


def stretch(gray):
    """Compress the huge dynamic range (bright stars vs faint) to 0-255 for the detector."""
    bg = float(np.median(gray))
    x = gray - bg
    upper = x[x > 0]
    sigma = 1.4826 * float(np.median(upper)) if upper.size else 1.0
    sigma = max(sigma, 1e-6)
    y = np.arcsinh(x / (3 * sigma))
    y -= np.median(y)
    top = float(np.percentile(y, 99.99))
    return np.clip(y * (255.0 / max(top, 1e-6)), 0, 255).astype(np.float32), bg


def raw_flux(gray, bg, x, y, r=8):
    """Brightness of a star from the unstretched image (stretching squashes the ranking)."""
    h, w = gray.shape
    xa, xb = int(max(0, round(x) - r)), int(min(w, round(x) + r + 1))
    ya, yb = int(max(0, round(y) - r)), int(min(h, round(y) + r + 1))
    return float((gray[ya:yb, xa:xb] - bg).clip(0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fov", type=float)
    ap.add_argument("--truth", type=float, nargs=2, metavar=("RA", "DEC"))
    ap.add_argument("--psf", type=float, default=1.5)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--top", type=int)
    a = ap.parse_args()

    gray, deg_per_px, header_truth = load_image(a.path, a.fov)
    truth = tuple(a.truth) if a.truth else header_truth
    h, w = gray.shape
    print("Image %d x %d px, %.5f deg/px (%.1f arcsec/px), width %.2f deg"
          % (w, h, deg_per_px, deg_per_px * 3600, deg_per_px * w))

    # --- detect --------------------------------------------------------------
    img, bg = stretch(gray)
    stars = detect.detect_stars(img, k_sigma=a.k, psf_sigma=a.psf, reject_blends=False,
                                max_pixels=5000, min_pixels=3)
    # rank by true brightness, not by the stretched image
    stars = sorted(((x, y, raw_flux(gray, bg, x, y)) for x, y, _ in stars), key=lambda s: -s[2])
    print("Detected %d star-like blobs" % len(stars))
    if len(stars) < 4:
        sys.exit("Fewer than 4 stars found: try a lower --k (e.g. 4) or a different --psf")

    # --- point the simulator's camera model at this image ---------------------
    size = max(w, h)
    sim.IMG_SIZE = size
    sim.FOCAL_PX = 1.0 / np.radians(deg_per_px)
    sim.FOV_DEG = deg_per_px * size
    identify.DEG_PER_PX = deg_per_px
    db = identify.StarDatabase(sim.CATALOG, max_mag=6.0)
    print("Catalog: %d stars (%s)" % (len(db.vecs), sim.CATALOG_SOURCE))

    # shift so the image centre sits at the middle of the (square) model sensor
    off_x, off_y = size / 2 - w / 2, size / 2 - h / 2
    xy_all = np.array([(x + off_x, y + off_y) for x, y, _ in stars])

    # --- identify + solve (try a few star counts, keep the best) --------------
    best, best_n = None, 0
    tries = [a.top] if a.top else [15, 25, 40]
    for n in tries:
        sub = xy_all[:n]
        for tol_px, match_px in ((1.5, 3.0), (2.5, 4.0)):
            res = identify.identify_stars(db, sub, tol_px=tol_px, min_votes=3, max_stars=12,
                                          match_px=match_px, min_fraction=0.3)
            if res and res["n_matched"] > best_n:
                best, best_n, best_try = res, res["n_matched"], (n, tol_px)
    if best is None or best_n < 5:
        print("\nNo trustworthy solution. Things to try: --k 4, --psf 2.5, --top 30, "
              "or check --fov.")
        return

    R = best["R"]
    bore = R[2]                                           # inertial direction of the centre pixel
    ra = np.degrees(np.arctan2(bore[1], bore[0])) % 360
    dec = np.degrees(np.arcsin(bore[2]))
    north = R @ np.array([0.0, 0.0, 1.0])                 # north pole direction in the camera frame
    north_angle = np.degrees(np.arctan2(north[0], -north[1]))
    print("\nSOLVED  (using the %d brightest detections, tolerance %.1f px)" % best_try)
    print("  matched %d of %d stars" % (best["n_matched"], best["n_detected"]))
    print("  centre RA  = %9.4f deg" % ra)
    print("  centre Dec = %9.4f deg" % dec)
    print("  north angle= %9.2f deg  (0 = north up)" % north_angle)
    if truth:
        t = sim.radec_to_unit(*truth)
        sep = np.degrees(np.arccos(np.clip(bore @ t, -1, 1))) * 3600
        print("  truth      = %.4f, %.4f  ->  centre error %.1f arcsec (%.2f px)"
              % (truth[0], truth[1], sep, sep / (deg_per_px * 3600)))

    # --- per-star residuals + annotated picture --------------------------------
    cam = db.vecs @ R.T
    px = sim.FOCAL_PX * cam[:, 0] / cam[:, 2] + size / 2
    py = sim.FOCAL_PX * cam[:, 1] / cam[:, 2] + size / 2
    res_px = []
    vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for x, y, _ in stars[:60]:
        cv2.circle(vis, (int(round(x)), int(round(y))), 9, (0, 200, 0), 1)
    print("\n  matched stars (catalog id, mag, residual in pixels):")
    for di, ci in sorted(best["matches"], key=lambda m: db.mags[m[1]]):
        d = float(np.hypot(xy_all[di][0] - px[ci], xy_all[di][1] - py[ci]))
        res_px.append(d)
        print("    %-10s mag %4.2f   %.2f px" % (db.names[ci], db.mags[ci], d))
        x, y = xy_all[di][0] - off_x, xy_all[di][1] - off_y
        cv2.circle(vis, (int(round(x)), int(round(y))), 13, (0, 0, 255), 2)
        cv2.putText(vis, str(db.names[ci]), (int(x) + 15, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    print("  RMS residual: %.2f px (%.1f arcsec)"
          % (np.sqrt(np.mean(np.square(res_px))), np.sqrt(np.mean(np.square(res_px))) * deg_per_px * 3600))
    out = a.path.rsplit(".", 1)[0] + "_solved.png"
    cv2.imwrite(out, vis)
    print("\nSaved %s (green = detected, red + name = identified catalog star)" % out)


if __name__ == "__main__":
    main()
