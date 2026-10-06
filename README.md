# Low-Cost Star Tracker for CubeSats

**Goal:** Build a low-cost star tracker for small satellites.
**Input:** a sky image. **Output:** an attitude quaternion.

## Status
- [x] Star-field simulator (`simulator.py`)
- [ ] Star detection + sub-pixel centroiding
- [ ] Star identification (triangle / pyramid matching)
- [ ] Attitude solve (QUEST)
- [ ] Real night-sky test
- [ ] Embedded port (Raspberry Pi / FPGA)

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python simulator.py
```
This writes `sim_star_field.png` and prints the true attitude used.

## How the simulator works
1. Loads a bright-star catalog (a small built-in set, so it runs offline; swap in Hipparcos later).
2. Picks a camera pointing direction and roll (the true attitude).
3. Rotates every star into the camera frame.
4. Projects stars with a pinhole model (focal length + pixel size).
5. Renders each star as a Gaussian blob with brightness from magnitude, then adds noise.

## Camera defaults (change in `simulator.py`)
- Image: 1024 x 1024 px
- Field of view: about 15 degrees
- Limiting magnitude: 6.0

## Next steps
1. Replace the built-in catalog with Hipparcos via `astroquery`.
2. Write `detect.py`: threshold, label blobs, compute centroids.
3. Write `identify.py`: study tetra3 first, then build your own matcher.
4. Write `quest.py`: solve the attitude from matched star vectors.
5. Compare the solved quaternion with the true one and log the error in `results.md`.

## Resources
- tetra3 (ESA), OpenStarTracker, astrometry.net
- Liebe, "Accuracy Performance of Star Trackers: A Tutorial"
- Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*
- Schaub, *Spacecraft Dynamics and Control* (Coursera)
