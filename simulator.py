"""Star-field simulator: given a camera attitude, render a synthetic star image.

Run:  python simulator.py
Output: sim_star_field.png (+ printed true attitude)
"""
import csv
import os

import numpy as np
import cv2

# --- Camera model ---------------------------------------------------------
IMG_SIZE = 1024            # pixels (square sensor)
FOV_DEG = 15.0             # full field of view, degrees
LIMIT_MAG = 6.0            # faintest star we render
SIGMA_PX = 1.4             # star blur (PSF width) in pixels
NOISE_STD = 3.0            # sensor read noise (8-bit counts)
BACKGROUND = 8.0           # sky + dark level (8-bit counts)
REF_MAG = 6.0              # a star of this magnitude...
PEAK_AT_REF_MAG = 40.0     # ...peaks at this brightness (8-bit counts)

# Pinhole focal length in pixels from field of view
FOCAL_PX = (IMG_SIZE / 2) / np.tan(np.radians(FOV_DEG / 2))

# --- Small built-in catalog: (name, RA deg, Dec deg, magnitude) ------------
# Approximate J2000 positions for bright stars so the script runs offline.
# Replace with Hipparcos (astroquery) for a realistic star density.
BUILTIN_CATALOG = [
    ("Sirius", 101.287, -16.716, -1.46), ("Canopus", 95.988, -52.696, -0.74),
    ("Arcturus", 213.915, 19.182, -0.05), ("Vega", 279.234, 38.784, 0.03),
    ("Capella", 79.172, 45.998, 0.08), ("Rigel", 78.634, -8.202, 0.13),
    ("Procyon", 114.825, 5.225, 0.34), ("Betelgeuse", 88.793, 7.407, 0.50),
    ("Altair", 297.696, 8.868, 0.77), ("Aldebaran", 68.980, 16.509, 0.85),
    ("Spica", 201.298, -11.161, 0.97), ("Antares", 247.352, -26.432, 1.06),
    ("Pollux", 116.329, 28.026, 1.14), ("Fomalhaut", 344.413, -29.622, 1.16),
    ("Deneb", 310.358, 45.280, 1.25), ("Regulus", 152.093, 11.967, 1.35),
    ("Castor", 113.650, 31.888, 1.58), ("Bellatrix", 81.283, 6.350, 1.64),
    ("Alnilam", 84.053, -1.202, 1.69), ("Alnitak", 85.190, -1.943, 1.77),
    ("Mintaka", 83.002, -0.299, 2.23), ("Saiph", 86.939, -9.670, 2.06),
    ("Dubhe", 165.932, 61.751, 1.79), ("Merak", 165.460, 56.382, 2.37),
    ("Alioth", 193.507, 55.960, 1.77), ("Mizar", 200.981, 54.925, 2.04),
    ("Alkaid", 206.886, 49.313, 1.86), ("Polaris", 37.955, 89.264, 1.98),
    ("Hamal", 31.793, 23.463, 2.00), ("Alpheratz", 2.097, 29.091, 2.06),
    ("Mirach", 17.433, 35.621, 2.05), ("Almach", 30.975, 42.330, 2.10),
    ("Schedar", 10.127, 56.537, 2.24), ("Caph", 2.295, 59.150, 2.28),
    ("Gamma Cas", 14.177, 60.717, 2.47), ("Menkar", 45.570, 4.090, 2.53),
    ("Mirfak", 51.081, 49.861, 1.79), ("Algol", 47.042, 40.956, 2.12),
    ("Elnath", 81.573, 28.608, 1.65), ("Alhena", 99.428, 16.399, 1.93),
    ("Mebsuta", 100.983, 25.131, 3.06), ("Tejat", 95.740, 22.514, 2.88),
    ("Wezen", 107.098, -26.393, 1.84), ("Adhara", 104.656, -28.972, 1.50),
    ("Alphard", 141.897, -8.659, 1.98), ("Denebola", 177.265, 14.572, 2.14),
    ("Algieba", 154.993, 19.842, 2.28), ("Zosma", 168.527, 20.524, 2.56),
    ("Vindemiatrix", 195.544, 10.959, 2.83), ("Porrima", 190.415, -1.449, 2.74),
    ("Zubenelgenubi", 222.720, -16.042, 2.75), ("Kochab", 222.676, 74.156, 2.08),
    ("Rasalhague", 263.734, 12.560, 2.08), ("Rasalgethi", 258.662, 14.390, 2.78),
    ("Sabik", 257.595, -15.725, 2.43), ("Shaula", 263.402, -37.104, 1.62),
    ("Sargas", 264.330, -42.998, 1.87), ("Kaus Australis", 276.043, -34.385, 1.85),
    ("Nunki", 283.816, -26.297, 2.02), ("Sadr", 305.557, 40.257, 2.23),
    ("Albireo", 292.680, 27.960, 3.18), ("Enif", 326.046, 9.875, 2.39),
    ("Markab", 346.190, 15.205, 2.49), ("Scheat", 345.944, 28.083, 2.42),
    ("Algenib", 3.309, 15.184, 2.83), ("Diphda", 10.897, -17.987, 2.04),
    ("Achernar", 24.429, -57.237, 0.46), ("Acamar", 44.565, -40.305, 2.88),
    ("Hadar", 210.956, -60.373, 0.61), ("Rigil Kent", 219.900, -60.834, -0.01),
    ("Acrux", 186.650, -63.099, 0.77), ("Mimosa", 191.930, -59.689, 1.25),
    ("Gacrux", 187.791, -57.113, 1.64), ("Miaplacidus", 138.300, -69.717, 1.68),
    ("Avior", 125.629, -59.510, 1.86), ("Atria", 252.166, -69.028, 1.92),
]


def load_catalog(path=None):
    """Load catalog.csv (made by get_catalog.py), else fall back to the built-in list.

    Returns (list of (name, ra_deg, dec_deg, mag), description of the source).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.environ.get("STAR_CATALOG") or os.path.join(here, "catalog.csv")
    if os.path.exists(path):
        stars = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                stars.append((f"HIP{row['hip']}", float(row["ra_deg"]),
                              float(row["dec_deg"]), float(row["mag"])))
        if stars:
            return stars, os.path.basename(path)
    return BUILTIN_CATALOG, "built-in 75-star list; run get_catalog.py for the real catalog"


CATALOG, CATALOG_SOURCE = load_catalog()


def radec_to_unit(ra_deg, dec_deg):
    """Right ascension / declination (deg) -> unit vector in the inertial frame."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


def attitude_matrix(ra_deg, dec_deg, roll_deg):
    """Rotation matrix R that maps inertial vectors into the camera frame.

    Camera frame: +z is the boresight, +x is image right, +y is image down.
    """
    z = radec_to_unit(ra_deg, dec_deg)                  # boresight
    north = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, north)) > 0.999:                   # near the pole
        north = np.array([1.0, 0.0, 0.0])
    x = np.cross(north, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # apply roll about the boresight
    r = np.radians(roll_deg)
    x_r = np.cos(r) * x + np.sin(r) * y
    y_r = -np.sin(r) * x + np.cos(r) * y
    return np.vstack([x_r, y_r, z])                     # rows = camera axes


def rotation_to_quaternion(R):
    """Rotation matrix -> quaternion [w, x, y, z]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = np.argmax(np.diag(R))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def project_stars(R, catalog=None, limit_mag=LIMIT_MAG):
    """Return list of (px, py, magnitude, name) for stars visible in the image."""
    catalog = CATALOG if catalog is None else catalog
    visible = []
    half = IMG_SIZE / 2
    for name, ra, dec, mag in catalog:
        if mag > limit_mag:
            continue
        v = R @ radec_to_unit(ra, dec)
        if v[2] <= 0:                                   # behind the camera
            continue
        px = FOCAL_PX * v[0] / v[2] + half
        py = FOCAL_PX * v[1] / v[2] + half
        if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
            visible.append((px, py, mag, name))
    return visible


def render_image(stars, rng=None):
    """Draw Gaussian star blobs and add background + noise. Returns uint8 image."""
    rng = rng or np.random.default_rng(0)
    img = np.full((IMG_SIZE, IMG_SIZE), BACKGROUND, dtype=np.float64)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    for px, py, mag, _ in stars:
        # brightness: 2.512x per magnitude, anchored so a REF_MAG star peaks at
        # PEAK_AT_REF_MAG counts. Bright stars saturate at 255, like a real sensor.
        peak = PEAK_AT_REF_MAG * 2.512 ** (REF_MAG - mag)
        r = int(4 * SIGMA_PX) + 1
        x0, x1 = max(0, int(px) - r), min(IMG_SIZE, int(px) + r + 1)
        y0, y1 = max(0, int(py) - r), min(IMG_SIZE, int(py) + r + 1)
        g = np.exp(-((xx[y0:y1, x0:x1] - px) ** 2 + (yy[y0:y1, x0:x1] - py) ** 2)
                   / (2 * SIGMA_PX ** 2))
        img[y0:y1, x0:x1] += peak * g
    img += rng.normal(0, NOISE_STD, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    # True attitude: pick any pointing direction. This one looks at Orion.
    ra, dec, roll = 84.0, 0.0, 20.0
    R = attitude_matrix(ra, dec, roll)
    q = rotation_to_quaternion(R)

    stars = project_stars(R)
    img = render_image(stars)
    cv2.imwrite("sim_star_field.png", img)

    print(f"Focal length: {FOCAL_PX:.1f} px, FOV: {FOV_DEG} deg")
    print(f"Catalog: {len(CATALOG)} stars ({CATALOG_SOURCE})")
    print(f"True pointing: RA={ra} deg, Dec={dec} deg, roll={roll} deg")
    print(f"True quaternion [w x y z]: {np.round(q, 5)}")
    print(f"Stars visible: {len(stars)}")
    for px, py, mag, name in sorted(stars, key=lambda s: s[2]):
        print(f"  {name:12s} mag {mag:5.2f}  pixel ({px:7.1f}, {py:7.1f})")
    print("Saved sim_star_field.png")


if __name__ == "__main__":
    main()
