
## Detector upgrade: before vs after (real Hipparcos catalog, 50 pointings per setting)
Improved detector: clipping-proof noise estimate, matched filter, windowed centroid, blend rejection.

| Noise | Correct (old -> new) | Stars found of 28.3 (old -> new) | Pointing error arcsec (old -> new) | Roll error arcsec (old -> new) |
|---|---|---|---|---|
| 3 | 50 -> 50 | 28.0 -> 27.8 | 1.24 -> 0.71 | 9.98 -> 5.89 |
| 6 | 50 -> 50 | 27.8 -> 28.0 | 1.82 -> 1.14 | 11.9 -> 9.4 |
| 10 | 48 -> 50 | 16.6 -> 28.1 | 2.30 -> 1.79 | 14.9 -> 14.4 |
| 15 | 47 -> 50 | 15.1 -> 28.0 | 3.06 -> 2.79 | 21.3 -> 21.1 |
| 20 | 48 -> 50 | 17.3 -> 25.8 | 4.20 -> 3.60 | 31.7 -> 26.8 |
| 30 | 0 -> 47 | 549.6 false -> 18.2 | n/a -> 4.78 | n/a -> 41.9 |

- Root cause of the old noise-30 failure: 41% of pixels clipped at zero made the noise estimate read 11.6 instead of 30, so the threshold was far too low (about 550 false detections).
- New detector: noise estimate 20.8 against a true 20.0.
- Zero misidentifications in all runs. Star-count sweep unchanged (identifier limit: needs at least 4 stars).
- Simulated images only.

## First real-image test: Pleiades photo (1024 x 768, JPEG)
- Detector found 93 stars; the brightest star in the cluster centre is detected after raising the blob size limit.
- Real stars are stretched into comet shapes with colour fringes (lens aberration). The simulator's blend test rejected 38 of them, so photo mode skips it.
- Photo mode: colour channels averaged, star blur 2.0 px, blob limit 3000 px.
- Not yet done: star identification (needs the field of view).
