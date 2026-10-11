"""Stress test: how much noise / how few stars can the star tracker take?

Run:  python sweep.py          (50 trials per setting, about 1-3 minutes)
      python sweep.py 100      (100 trials per setting, slower but smoother)

Two sweeps, using the SAME random sky pointings for every setting so the
comparison is fair:
  1. Sensor noise 3 -> 30 counts       (faintest star kept at magnitude 6)
  2. Faintest star 6.0 -> 4.0          (noise kept at 3)

Writes:
  sweep_results.csv   the full table (open it in Excel)
  sweep_results.png   a plot you can put in your application
"""
import csv
import sys
import time

import numpy as np

import simulator as sim
import detect
import identify
import attitude as att
from run_pipeline import random_attitude, WRONG_ARCSEC

NOISE_LEVELS = [3, 6, 10, 15, 20, 30]           # sensor noise, 8-bit counts
LIMIT_MAGS = [6.0, 5.5, 5.0, 4.5, 4.0]          # faintest star the camera sees
BASE_NOISE = 3.0
BASE_MAG = 6.0


def run_case(noise, limit_mag, n_trials, seed=42):
    """Run the whole pipeline n_trials times at one setting. Returns a dict."""
    sim.NOISE_STD = noise
    rng = np.random.default_rng(seed)             # same pointings for every case
    db = identify.StarDatabase(sim.CATALOG, max_mag=limit_mag)

    good, wrong, none = [], 0, 0
    true_n, det_n = [], []
    for _ in range(n_trials):
        ra, dec, roll = random_attitude(rng)
        R_true = sim.attitude_matrix(ra, dec, roll)
        truth = sim.project_stars(R_true, limit_mag=limit_mag)
        img = sim.render_image(truth, rng)
        det = detect.detect_stars(img)
        true_n.append(len(truth))
        det_n.append(len(det))

        result = None
        if len(det) >= 4:
            det_xy = np.array([(x, y) for x, y, _ in det])
            result = identify.identify_stars(db, det_xy)
        if result is None:
            none += 1
            continue
        err = att.attitude_error(result["R"], R_true)
        if err["total"] > WRONG_ARCSEC:
            wrong += 1
        else:
            good.append(err)

    rms = lambda key: (float(np.sqrt(np.mean([g[key] ** 2 for g in good])))
                       if good else float("nan"))
    return {
        "trials": n_trials,
        "correct": len(good),
        "wrong": wrong,
        "no_answer": none,
        "pct_correct": 100.0 * len(good) / n_trials,
        "true_stars": float(np.mean(true_n)),
        "detected": float(np.mean(det_n)),
        "pointing_rms": rms("cross"),
        "roll_rms": rms("roll"),
    }


def print_table(title, rows, label):
    print(f"\n{title}")
    print(f"{label:>8} {'correct':>8} {'wrong':>6} {'none':>5} {'true':>6} {'found':>6} "
          + "pointing-arcsec".rjust(16) + " " + "roll-arcsec".rjust(12))
    for r in rows:
        print(f"{r['value']:>8} {r['correct']:>8} {r['wrong']:>6} {r['no_answer']:>5} "
              f"{r['true_stars']:>6.1f} {r['detected']:>6.1f} "
              f"{r['pointing_rms']:>16.2f} {r['roll_rms']:>12.2f}")


def make_plot(noise_rows, mag_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"           # categorical slots 1 and 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), facecolor=SURFACE)
    sweeps = [
        (noise_rows, "Sensor noise (counts)", "Noise sweep (faintest star: mag 6)", False),
        (mag_rows, "Faintest star visible (magnitude)", f"Star-count sweep (noise: {BASE_NOISE:g})", True),
    ]
    for col, (rows, xlabel, title, invert) in enumerate(sweeps):
        x = [r["x"] for r in rows]
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=9)
            if invert:
                ax.invert_xaxis()           # fewer stars to the right
        top.set_title(title, color=INK, fontsize=11, loc="left", pad=10)

        # top row: how often is the answer right?
        top.plot(x, [r["pct_correct"] for r in rows], color=BLUE, linewidth=2,
                 marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        top.set_ylim(-3, 105)
        top.set_ylabel("Correct attitude (%)", color=MUTED, fontsize=9)
        top.set_xticks(x)

        # bottom row: how accurate are the correct answers?
        px = [r["x"] for r in rows if not np.isnan(r["roll_rms"])]
        pr = [r["roll_rms"] for r in rows if not np.isnan(r["roll_rms"])]
        pp = [r["pointing_rms"] for r in rows if not np.isnan(r["pointing_rms"])]
        bot.plot(px, pr, color=ORANGE, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
        bot.plot(px, pp, color=BLUE, linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
        bot.set_yscale("log")
        bot.set_ylabel('Error, RMS (arcsec)', color=MUTED, fontsize=9)
        bot.set_xlabel(xlabel, color=MUTED, fontsize=9)
        bot.set_xticks(x)
        if px:                                   # direct labels at the line ends
            bot.annotate("Roll", (px[-1], pr[-1]), xytext=(6, 0), textcoords="offset points",
                         color=INK, fontsize=9, va="center")
            bot.annotate("Pointing", (px[-1], pp[-1]), xytext=(6, 0), textcoords="offset points",
                         color=INK, fontsize=9, va="center")
        bot.margins(x=0.12)
    fig.suptitle("Star tracker simulation: where does it break?", color=INK,
                 fontsize=13, x=0.06, ha="left")
    fig.text(0.06, 0.005, "Simulated images only. Errors are for the correct solutions; "
             "the top row shows how often the answer was correct.",
             color=MUTED, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Catalog: {len(sim.CATALOG)} stars ({sim.CATALOG_SOURCE})")
    print(f"{n_trials} random pointings per setting, same pointings every time.")
    t0 = time.time()

    noise_rows, mag_rows = [], []
    for noise in NOISE_LEVELS:
        r = run_case(noise, BASE_MAG, n_trials)
        r.update(sweep="noise", value=noise, x=noise)
        noise_rows.append(r)
        print(f"  noise {noise:>2}: {r['correct']}/{n_trials} correct", flush=True)
    for mag in LIMIT_MAGS:
        r = run_case(BASE_NOISE, mag, n_trials)
        r.update(sweep="limit_mag", value=mag, x=mag)
        mag_rows.append(r)
        print(f"  mag {mag:>3}: {r['correct']}/{n_trials} correct", flush=True)
    sim.NOISE_STD = BASE_NOISE

    print_table("NOISE SWEEP (faintest star: magnitude 6)", noise_rows, "noise")
    print_table(f"STAR-COUNT SWEEP (noise: {BASE_NOISE:g})", mag_rows, "mag")
    print("\ncorrect/wrong/none = right attitude / misidentified / gave up.  "
          "true/found = stars per image.\npointing-arcsec/roll-arcsec = RMS error in "
          "arcseconds over the correct solutions.")

    cols = ["sweep", "value", "trials", "correct", "wrong", "no_answer", "pct_correct",
            "true_stars", "detected", "pointing_rms", "roll_rms"]
    with open("sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(noise_rows + mag_rows)
    make_plot(noise_rows, mag_rows, "sweep_results.png")
    print(f"\nSaved sweep_results.csv and sweep_results.png  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()