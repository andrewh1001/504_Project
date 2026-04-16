
# GeoVision - Traffic Violation Detection System

Real-time traffic violation detection from a fixed camera using object detection, multi-object tracking, and ground-plane homography.

## What It Does

-   **Red light violations** — detects vehicles crossing a stop line while the signal is red
-   **Stop sign compliance** — flags vehicles that do not come to a full stop inside a designated zone
-   **Near-collision forecasting** — projects vehicle trajectories onto a satellite ground plane and warns when two vehicles are predicted to converge within a distance


## Files

File

Description

`define_region.py`

Annotation tool — draw traffic lights, stop lines, stop zones, ROIs

`calibrate_H.py`

Camera-to-satellite correspondence picker

`run_detection.py`

Main inference pipeline

## Setup

```bash
pip install ultralytics opencv-python numpy

```
## Usage

### 1. Annotate regions

Controls:

-   `t` — traffic light quad (4 clicks), then `b` to place red/yellow/green bulb dots
-   `l` — stop line (2 clicks)
-   `z` — stop zone quad (4 clicks)
-   `r` — region of interest (free-form, Enter or click first point to close)
-   `p` — stop sign
-   `u` — undo last shape
-   `s` — save to `selected_regions.json`
-   `q` — quit

### 2. Calibrate homography


Place an overhead screenshot of the intersection in the directory. Click matching physical landmarks in both the camera frame and satellite image windows. Press `s` to compute and save `homography.json`. Aim for 8+ point pairs spread across the intersection.

### 3. Run

```bash
python run_detection.py

```
## Configuration

Key parameters at the top of `run2.py`:
`NEAR_COLLISION_DIST`: 15

Satellite pixels threshold for near-collision warning

`PREDICT_FRAMES`: `fps * 2` 
How far ahead to forecast trajectories

`STOP_MIN_SECONDS`: 0.5

Required stop duration for stop sign compliance

`STILL_STD_THRESH`: 2.0

Pixel std dev threshold for stationarity

`CROSS_LOOKBACK`: 5

Frames to look back for stop line crossing

`BULB_RADIUS`: 8

Pixel radius of bulb sampling region

`GROUND_SMOOTH_ALPHA`: 0.3

EMA smoothing for satellite positions (lower = smoother)

## Known Limitations

-   Stop sign stationarity is evaluated in image space — the threshold is perspective-dependent and effectively stricter for vehicles closer to the camera. Ground-plane evaluation via homography would fix this.
-   Linear trajectory extrapolation produces false near-collision alerts for turning vehicles.
-   Traffic light HSV detection degrades under direct sunlight glare.
-   ByteTrack ID switches during occlusion can reset the stop sign zone timer.
