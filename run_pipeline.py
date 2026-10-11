"""Full star tracker test: simulate -> detect -> identify -> solve -> compare to truth.

Run:  python run_pipeline.py          (50 random sky pointings)
      python run_pipeline.py 200      (200 pointings)

Save the output for your notes:
      python run_pipeline.py 100 | Tee-Object -FilePath pipeline_output.txt
"""
import sys
import time

import numpy as np

import simulator as sim
import detect
import identify
import attitude as att

# An answer off by more than 10 arcminutes cannot be centroid noise: the stars were
# misidentified. Smaller errors (even tens of arcseconds, mostly in roll) are just noise.
WRONG_ARCSEC = 600.0


def random_attitude(rng):
    ra = rng.uniform(0, 360)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))   # uniform over the sphere
    roll = rng.uniform(0, 360)
    return ra, dec, roll


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rng = np.random.default_rng(42)

    print(f"Catalog: {len(sim.CATALOG)} stars ({sim.CATALOG_SOURCE})")
    print(f"Noise: {sim.NOISE_STD} counts, FOV: {sim.FOV_DEG} deg, "
          f"limiting magnitude: {sim.LIMIT_MAG}")
    t0 = time.time()
    db = identify.StarDatabase(sim.CATALOG, max_mag=sim.LIMIT_MAG)
    print(f"Pair database: {len(db.pair_ang)} star pairs, built in {time.time() - t0:.1f}s\n")

    good, wrong, none = [], 0, []
    true_counts, det_counts, solve_times = [], [], []
    for t in range(n_trials):
        ra, dec, roll = random_attitude(rng)
        R_true = sim.attitude_matrix(ra, dec, roll)
        truth = sim.project_stars(R_true)
        img = sim.render_image(truth, rng)
        det = detect.detect_stars(img)
        det_xy = np.array([(x, y) for x, y, _ in det])
        true_counts.append(len(truth))
        det_counts.append(len(det))

        t1 = time.time()
        result = identify.identify_stars(db, det_xy) if len(det) >= 4 else None
        solve_times.append(time.time() - t1)

        if result is None:
            none.append((t, ra, dec, roll, len(truth), len(det)))
            continue
        err = att.attitude_error(result["R"], R_true)
        if err["total"] > WRONG_ARCSEC:
            wrong += 1
        else:
            good.append(err)

    print(f"Trials: {n_trials}")
    print(f"  Correct attitude:          {len(good)}")
    print(f"  Misidentified (wrong sky): {wrong}")
    print(f"  No answer (gave up):       {len(none)}")
    print(f"Stars per image: {np.mean(true_counts):.1f} true, {np.mean(det_counts):.1f} detected")
    print(f"Identify + solve time: {1000 * np.mean(solve_times):.1f} ms per image (laptop Python)")
    if good:
        tot = np.array([g["total"] for g in good])
        cross = np.array([g["cross"] for g in good])
        roll = np.array([g["roll"] for g in good])
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        print("\nAccuracy over the correct solutions (arcseconds):")
        print(f"  Total   RMS {rms(tot):7.2f}   median {np.median(tot):7.2f}   worst {tot.max():7.2f}")
        print(f"  Cross-boresight (pointing) RMS {rms(cross):7.2f}")
        print(f"  Roll (about the boresight) RMS {rms(roll):7.2f}")
    if none:
        print("\nPointings with no answer (trial, RA, Dec, roll, true stars, detected):")
        for row in none[:10]:
            print(f"  #{row[0]:3d}  RA {row[1]:6.1f}  Dec {row[2]:6.1f}  roll {row[3]:6.1f}"
                  f"  true {row[4]:2d}  detected {row[5]:2d}")


if __name__ == "__main__":
    main()
