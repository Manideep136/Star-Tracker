"""Star detection + sub-pixel centroiding.

Run:  python detect.py              -> simulates an image, detects stars, compares to truth
      python detect.py my_image.png -> detects stars in your own grayscale image
"""
import sys
import numpy as np
import cv2

import simulator as sim


def detect_stars(img, k_sigma=5.0, min_pixels=3, max_pixels=400):
    """Find stars in a grayscale image.

    Returns a list of (x, y, flux) with sub-pixel centroids, brightest first.
    """
    img = img.astype(np.float64)

    # 1. Estimate background level and noise robustly (stars are rare, so the
    #    median is the background and the MAD gives the noise).
    bg = np.median(img)
    sigma = 1.4826 * np.median(np.abs(img - bg))
    sigma = max(sigma, 1.0)

    # 2. Threshold: anything k_sigma above the background is a star candidate.
    thresh = bg + k_sigma * sigma
    mask = (img > thresh).astype(np.uint8)

    # 3. Group connected bright pixels into blobs.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    stars = []
    for i in range(1, n):                      # label 0 is the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_pixels or area > max_pixels:
            continue                           # noise speck or big artifact

        # 4. Centroid: intensity-weighted mean over a window around the blob.
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        pad = 2
        xa, xb = max(0, x0 - pad), min(img.shape[1], x0 + w + pad)
        ya, yb = max(0, y0 - pad), min(img.shape[0], y0 + h + pad)
        patch = np.clip(img[ya:yb, xa:xb] - bg, 0, None)
        flux = patch.sum()
        if flux <= 0:
            continue
        ys, xs = np.mgrid[ya:yb, xa:xb]
        cx = (patch * xs).sum() / flux
        cy = (patch * ys).sum() / flux
        stars.append((cx, cy, flux))

    stars.sort(key=lambda s: -s[2])
    return stars


def match_to_truth(detected, truth, max_dist=3.0):
    """Pair each true star with the nearest detected centroid. Returns rows of
    (name, true_x, true_y, det_x, det_y, error_px) or None if not found."""
    rows = []
    for tx, ty, mag, name in truth:
        best, best_d = None, max_dist
        for dx, dy, _ in detected:
            d = np.hypot(dx - tx, dy - ty)
            if d < best_d:
                best, best_d = (dx, dy), d
        if best is None:
            rows.append((name, tx, ty, None, None, None))
        else:
            rows.append((name, tx, ty, best[0], best[1], best_d))
    return rows


def main():
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"Could not read image: {sys.argv[1]}")
        stars = detect_stars(img)
        print(f"Detected {len(stars)} stars in {sys.argv[1]}")
        for x, y, f in stars:
            print(f"  x={x:8.2f}  y={y:8.2f}  flux={f:9.1f}")
        return

    # Self-test on a simulated image with known truth
    R = sim.attitude_matrix(84.0, 0.0, 20.0)
    truth = sim.project_stars(R)
    img = sim.render_image(truth)

    detected = detect_stars(img)
    print(f"True stars: {len(truth)}   Detected: {len(detected)}\n")

    rows = match_to_truth(detected, truth)
    print(f"{'Star':12s} {'true (x, y)':>20s} {'detected (x, y)':>22s} {'error px':>10s}")
    errs = []
    for name, tx, ty, dx, dy, e in rows:
        if e is None:
            print(f"{name:12s} ({tx:7.1f},{ty:7.1f})   MISSED")
        else:
            errs.append(e)
            print(f"{name:12s} ({tx:7.1f},{ty:7.1f})   ({dx:7.2f},{dy:7.2f})   {e:8.3f}")
    if errs:
        rms = float(np.sqrt(np.mean(np.square(errs))))
        print(f"\nRMS centroid error: {rms:.3f} px  "
              f"(~{rms * sim.FOV_DEG * 3600 / sim.IMG_SIZE:.1f} arcsec)")

    # Save an annotated image so you can see what was found
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, _ in detected:
        cv2.circle(vis, (int(round(x)), int(round(y))), 14, (0, 255, 0), 1)
    cv2.imwrite("detected_stars.png", vis)
    print("Saved detected_stars.png")


if __name__ == "__main__":
    main()
