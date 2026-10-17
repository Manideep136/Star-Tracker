"""Star detection + sub-pixel centroiding.

Run:  python detect.py              -> simulates an image, detects stars, compares to truth
      python detect.py my_image.png -> detects stars in your own grayscale image

Two detectors live here so you can compare them:
  detect_stars        the improved detector (used by everything else)
  detect_stars_basic  the original day-2 detector, kept for before/after tests

What the improved detector does differently:
  1. Noise estimate that survives clipping. Sensor values cannot go below 0, so at
     high noise a big share of dark pixels sit at exactly 0 and the classic
     median-absolute-deviation estimate reads far too low (we measured 11.6 when the
     true noise was 30), which made the threshold too low and flooded the output
     with false stars. Here the noise is measured from the upper half of the
     pixel values only, which clipping cannot touch.
  2. Matched filter. The image is blurred with the star's own blur shape before
     thresholding. Star light adds up, noise averages out, so faint stars rise
     above the noise (about 4.7x better signal-to-noise for a 1.4 px blur).
  3. Windowed centroid. A Gaussian-weighted centroid on the raw pixels,
     repeated a few times, instead of a plain weighted average over a blob.
  4. Blend rejection. Two stars closer than a few pixels merge into one blob and
     give a wrong position. A blob that is wider than a single star's blur is
     dropped. This test only runs when noise is low enough for the width to be
     measured reliably, and not on saturated stars (their flat tops look wide).
"""
import sys
import numpy as np
import cv2

import simulator as sim


def estimate_background(img):
    """Background level and noise, robust to stars AND to clipping at zero.

    The median gives the background (fine as long as under half the pixels are
    clipped). The noise comes from the upper half only: for pure noise the median
    of the positive deviations is 0.6745 sigma.
    """
    bg = float(np.median(img))
    upper = img[img > bg] - bg
    sigma = 1.4826 * float(np.median(upper)) if upper.size else 1.0
    return bg, max(sigma, 0.5)


def detect_stars(img, k_sigma=5.0, min_pixels=4, max_pixels=600, psf_sigma=None,
                 reject_blends=True, blend_ratio=1.2, info=None, saturation_level=250):
    """Find stars in a grayscale image (improved detector).

    img          2-D image, any numeric type
    k_sigma      detection threshold, in noise sigmas of the filtered image
    min_pixels / max_pixels  smallest / largest accepted blob (pixels, after filtering)
    psf_sigma    star blur in pixels (default: the simulator's value)
    reject_blends  drop blobs that look like two or more merged stars
    saturation_level  pixel value treated as saturated (these skip the blend test)
    info         optional dict; filled with 'noise' and 'blends' (count rejected)

    Returns a list of (x, y, flux), brightest first.
    """
    psf_sigma = sim.SIGMA_PX if psf_sigma is None else psf_sigma
    img = img.astype(np.float32)
    height, width = img.shape

    bg, sigma = estimate_background(img)
    smooth = cv2.GaussianBlur(img, (0, 0), psf_sigma)
    bg_s, sigma_s = estimate_background(smooth)
    mask = (smooth > bg_s + k_sigma * sigma_s).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    r = int(np.ceil(2.5 * psf_sigma)) + 1          # centroid window radius
    stars, n_blend = [], 0
    for i in range(1, n):                          # label 0 is the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_pixels or area > max_pixels:
            continue

        # start from the brightest pixel of the filtered blob
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        blob = np.where(labels[y0:y0 + h, x0:x0 + w] == i, smooth[y0:y0 + h, x0:x0 + w], -1e9)
        py, px = np.unravel_index(np.argmax(blob), blob.shape)
        cx, cy = float(x0 + px), float(y0 + py)

        # windowed centroid: Gaussian weights centred on the current estimate
        # (A plain centroid first gives a close starting point. Flat-topped, saturated
        # stars would otherwise need many iterations to settle.)
        ok = True
        for it in range(40):
            xa, xb = int(max(0, round(cx) - r)), int(min(width, round(cx) + r + 1))
            ya, yb = int(max(0, round(cy) - r)), int(min(height, round(cy) + r + 1))
            ys, xs = np.mgrid[ya:yb, xa:xb]
            if it == 0:
                g = 1.0
            else:
                g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * psf_sigma ** 2))
            d = img[ya:yb, xa:xb] - bg
            den = float((g * d).sum())
            if den <= 0:
                ok = False
                break
            new_x = float((g * d * xs).sum() / den)
            new_y = float((g * d * ys).sum() / den)
            moved = abs(new_x - cx) + abs(new_y - cy)
            cx, cy = new_x, new_y
            if it > 0 and moved < 0.002:
                break
        if not ok:
            continue

        xa, xb = int(max(0, round(cx) - r)), int(min(width, round(cx) + r + 1))
        ya, yb = int(max(0, round(cy) - r)), int(min(height, round(cy) + r + 1))
        ys, xs = np.mgrid[ya:yb, xa:xb]
        d = img[ya:yb, xa:xb] - bg
        flux = float(d.sum())
        if flux <= 0:
            continue

        # blend test: is this blob wider than one star's blur?
        saturated = img[ya:yb, xa:xb].max() >= saturation_level
        r4 = float((((xs - cx) ** 2 + (ys - cy) ** 2) ** 2).sum())
        moment_noise = sigma * np.sqrt(r4) / flux  # how much noise shakes the width
        if reject_blends and moment_noise < 0.2 * psf_sigma ** 2 and not saturated:
            sx = np.sqrt(max(float((d * (xs - cx) ** 2).sum() / flux), 0.0))
            sy = np.sqrt(max(float((d * (ys - cy) ** 2).sum() / flux), 0.0))
            if max(sx, sy) > blend_ratio * psf_sigma:
                n_blend += 1
                continue
        stars.append((cx, cy, flux))

    stars.sort(key=lambda s: -s[2])
    if info is not None:
        info.update(noise=sigma, blends=n_blend)
    return stars


def detect_stars_basic(img, k_sigma=5.0, min_pixels=3, max_pixels=400):
    """The original detector (day 2). Kept so you can compare against it.

    Weakness: its noise estimate reads too low when many pixels are clipped at 0,
    so at high noise it reports hundreds of false stars.
    """
    img = img.astype(np.float64)

    bg = np.median(img)
    sigma = 1.4826 * np.median(np.abs(img - bg))
    sigma = max(sigma, 1.0)

    thresh = bg + k_sigma * sigma
    mask = (img > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    stars = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_pixels or area > max_pixels:
            continue
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
    (name, true_x, true_y, det_x, det_y, error_px); det and error are None when
    no detection lies within max_dist."""
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
        # python detect.py photo.jpg [star_blur_px] [threshold_sigma]
        # Real photos differ from the simulator: stars are wider and often stretched or
        # colour-fringed by the lens, and the brightest stars have big glowing halos.
        # So image mode uses a wider blur, skips the blend test (a stretched star would
        # look like a blend) and accepts large blobs.
        path = sys.argv[1]
        psf = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        k_sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        if color is None:
            sys.exit("Could not read image: " + path)
        # Average the colour channels so blue and red stars count equally.
        img = color.astype(np.float32).mean(axis=2)
        info = {}
        stars = detect_stars(img, k_sigma=k_sigma, psf_sigma=psf, reject_blends=False,
                             max_pixels=3000, info=info)
        print("Detected %d stars in %s (noise estimate %.1f, blends rejected %d)"
              % (len(stars), path, info["noise"], info["blends"]))
        for x, y, f in stars:
            print("  x=%8.2f  y=%8.2f  flux=%9.1f" % (x, y, f))

        # Mark every detection on the image and save it next to the original.
        radius = max(8, int(round(4 * psf)))
        for rank, (x, y, f) in enumerate(stars):
            centre = (int(round(x)), int(round(y)))
            cv2.circle(color, centre, radius, (0, 255, 0), 1)
            if rank < 15:                          # number the 15 brightest
                cv2.putText(color, str(rank + 1), (centre[0] + radius + 2, centre[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        stem = path.rsplit(".", 1)[0]
        out = stem + "_detected.png"
        cv2.imwrite(out, color)
        print("Saved " + out + " (green circles = detected stars, yellow numbers = 15 brightest)")
        return

    # Self-test on a simulated image with known truth
    R = sim.attitude_matrix(84.0, 0.0, 20.0)
    truth = sim.project_stars(R)
    img = sim.render_image(truth)

    info = {}
    detected = detect_stars(img, info=info)
    print("True stars: %d   Detected: %d   Blended blobs rejected: %d   "
          "Noise estimate: %.1f (true %.1f)\n"
          % (len(truth), len(detected), info["blends"], info["noise"], sim.NOISE_STD))

    rows = match_to_truth(detected, truth)
    print("%-12s %20s %22s %10s" % ("Star", "true (x, y)", "detected (x, y)", "error px"))
    errs = []
    for name, tx, ty, dx, dy, e in rows:
        if e is None:
            print("%-12s (%7.1f,%7.1f)   not detected (faint, or part of a rejected blend)"
                  % (name, tx, ty))
        else:
            errs.append(e)
            print("%-12s (%7.1f,%7.1f)   (%7.2f,%7.2f)   %8.3f" % (name, tx, ty, dx, dy, e))
    if errs:
        rms = float(np.sqrt(np.mean(np.square(errs))))
        print("\nRMS centroid error: %.3f px  (~%.1f arcsec)"
              % (rms, rms * sim.FOV_DEG * 3600 / sim.IMG_SIZE))

    # Save an annotated image so you can see what was found
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, _ in detected:
        cv2.circle(vis, (int(round(x)), int(round(y))), 14, (0, 255, 0), 1)
    cv2.imwrite("detected_stars.png", vis)
    print("Saved detected_stars.png")


if __name__ == "__main__":
    main()