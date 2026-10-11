"""Download the Hipparcos catalog and save the bright stars to catalog.csv.

Needs internet. Run once:   python get_catalog.py
Then simulator.py, detect.py and run_pipeline.py use catalog.csv automatically.

Note: positions are at the Hipparcos epoch (J1991.25) and proper motion is
ignored. That is fine for testing against the simulator. For real night-sky
images you will later add proper-motion correction (a few stars move by tens
of arcseconds over 35 years).
"""
import csv
import sys

import numpy as np

MAX_MAG = 6.0
OUT = "catalog.csv"

try:
    from astroquery.vizier import Vizier
except ImportError:
    sys.exit("astroquery is not installed. Run: pip install -r requirements.txt")

print(f"Downloading Hipparcos stars brighter than magnitude {MAX_MAG} "
      "(can take up to a minute)...")
vizier = Vizier(
    columns=["HIP", "RAICRS", "DEICRS", "_RAJ2000", "_DEJ2000", "Vmag"],
    column_filters={"Vmag": f"<{MAX_MAG}"},
    row_limit=-1,
)
result = vizier.get_catalogs("I/239/hip_main")
if len(result) == 0:
    sys.exit("The catalog query returned nothing. Check your internet connection and retry.")
table = result[0]


def pick(*names):
    for name in names:
        if name in table.colnames:
            return name
    sys.exit(f"Could not find any of {names} in the downloaded columns: {table.colnames}\n"
             "Paste this message to Claude and it will adjust the script.")


ra_col = pick("RAICRS", "_RAJ2000")
dec_col = pick("DEICRS", "_DEJ2000")
mag_col = pick("Vmag")

rows = []
for r in table:
    ra, dec, mag = float(r[ra_col]), float(r[dec_col]), float(r[mag_col])
    if np.isfinite(ra) and np.isfinite(dec) and np.isfinite(mag):
        rows.append((int(r["HIP"]), ra, dec, mag))
rows.sort(key=lambda s: s[3])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["hip", "ra_deg", "dec_deg", "mag"])
    w.writerows(rows)

print(f"Saved {len(rows)} stars to {OUT}")
print("Brightest three (sanity check, expect Sirius first, about mag -1.4):")
for hip, ra, dec, mag in rows[:3]:
    print(f"  HIP {hip}: RA={ra:.3f}  Dec={dec:.3f}  mag={mag:.2f}")
