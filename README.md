# mini_orienter

Stabilise video of a miniature you are painting: track fiducials on its
base or handle, estimate their pose in each frame and warp every frame back
to a reference pose, so the miniature stays locked while the background moves.

Two tracking approaches live side by side:

| Approach | Fiducial | Scripts |
|----------|----------|---------|
| **ArUco** (recommended) | 4 printed ArUco markers (`DICT_4X4_50`, IDs 0-3) glued around a cylindrical handle at 90° intervals | `print_markers.py`, `stabilise_aruco.py`, `detect_aruco.py` |
| **De Bruijn strip** | A hand-drawn binary De Bruijn ring (16 cells, window 4) around the base rim | `stabilise.py`, `detect_strip.py`, `diagnose.py` |

Both can use an optional camera calibration made with `calibrate.py`.

> Status: experimental prototype. Neither pipeline has been validated on a
> real recording yet.

## Install

Python 3.9+ and OpenCV 4.7 or newer (the ArUco API used here does not
exist in older versions):

```bash
pip install "opencv-contrib-python>=4.7" numpy
```

## ArUco workflow

```bash
# 1. Print the markers (SVG, open in a browser, print at 100% scale).
#    --diameter sizes each marker to a quarter of the handle circumference.
python print_markers.py --diameter 22 --out markers.svg

# 2. Cut them out and glue IDs 0, 1, 2, 3 around the handle at 90° steps,
#    all upright and in the same direction. Order: with ID 0 facing you,
#    ID 1 is on its right (IDs increase counter-clockwise seen from above).
#    Other layouts, e.g. the hexagonal fiducial: N markers on a regular
#    N-sided prism, same order, then pass --faces N (and --dict/--first-id).

# 3. (Optional, better accuracy) calibrate your camera once, see below.

# 4. Stabilise. --marker-size is the printed side of the black square, in mm.
python stabilise_aruco.py --video knight.mp4 --marker-size 17.3
python stabilise_aruco.py --video knight.mp4 --marker-size 17.3 --camera camera.yml --debug
```

Check that the markers are detectable in a still photo first:

```bash
python detect_aruco.py --image photo.jpg            # writes photo_detected.jpg
python detect_aruco.py --image photo.jpg --dict 4X4_50
```

### `stabilise_aruco.py` flags

| Flag | Meaning |
|------|---------|
| `--video` | Input video (required) |
| `--marker-size` | Marker side length in mm (required) |
| `--handle-radius` | Distance from the handle axis to the marker surfaces in mm. Default: estimated as `marker-size / (2 tan(pi/faces))`, i.e. a prism whose faces are exactly one marker wide; pass the measured value for accuracy |
| `--faces` | Number of marker faces around the handle (default 4; 6 for a hexagonal prism) |
| `--first-id` | Marker ID on face 0; the other faces use consecutive IDs (default 0) |
| `--dict` | ArUco dictionary the markers were printed from (default `4X4_50`) |
| `--camera` | `camera.yml` from `calibrate.py`. Without it a rough camera matrix (focal length = image width) is used |
| `--out` | Output video (default `stabilised.mp4`) |
| `--reference` | Start searching for the reference pose from this frame (default 0); the first frame with a visible marker becomes the reference |
| `--debug` | Show a live preview with detected markers and axes |
| `--frames-dir` | Also save each output frame as PNG here |
| `--pose-out` | Write the per-frame handle pose (rotation vector, position, quaternion, yaw relative to the reference) to this `.csv` or `.json` file; columns are described in `pose_export.py` |
| `--no-video` | Skip writing the stabilised video (use with `--pose-out`) |

### `print_markers.py` flags

| Flag | Meaning |
|------|---------|
| `--diameter` | Handle diameter in mm; marker size = circumference / faces |
| `--faces` | Markers around the handle (default 4); sets the default IDs and `--diameter` sizing |
| `--size` | Marker size in mm (overrides `--diameter`; default 20) |
| `--ids` | Marker IDs to print (default `0 .. faces-1`) |
| `--margin` | White border around each marker in mm (default 3) |
| `--out` | Output SVG (default `markers.svg`) |

## De Bruijn strip workflow

Draw a ring of 16 cells around the base rim, framed by two solid sync lines,
filling the cells that are `1` in the sequence the detector prints at startup
(`[DeBruijn] Sequence: ...`). Then:

```bash
# Without calibration (2D homography mode)
python stabilise.py --video knight.mp4 --radius 14.5 --strip-height 6 --debug

# With calibration (full 3D pose)
python stabilise.py --video knight.mp4 --radius 14.5 --strip-height 6 \
                    --camera camera.yml --debug
```

### `stabilise.py` flags

| Flag | Meaning |
|------|---------|
| `--video` | Input video (required) |
| `--radius` | Radius of the cell ring in mm, i.e. half the measured diameter (required) |
| `--strip-height` | Strip height in mm (default 6) |
| `--rotation` | Cyclic offset (in cells) applied to the De Bruijn sequence, if you drew the strip starting at a different cell than the printed sequence (default 0) |
| `--camera` | `camera.yml` from `calibrate.py` (enables the 3D path) |
| `--reference` | Frame number used as the reference pose (default 0) |
| `--out` | Output video (default `stabilised.mp4`) |
| `--debug` | Show live annotated windows |
| `--frames-dir` | Also save each output frame as PNG here |

### When the strip is not found

```bash
python diagnose.py --video knight.mp4 --frame 0 --out debug_frames
```

This writes every detection step (thresholds, contours, fitted ellipses,
filter result, brightness profiles) to `debug_frames/`. Start with
`05_ellipses_top20.jpg`, then `06_passed_filter.jpg`.

- **Strip not found**: check lighting, run `diagnose.py`, try `--debug`.
- **Low confidence**: make the filled cells darker (go over them 2-3 times).
- **Wobbly output**: an uneven drawing hurts the 3D path; the 2D mode is more forgiving.

## Camera calibration

Film a printed checkerboard from many angles for 20-30 seconds, or take
still photos, then:

```bash
python calibrate.py --video checkerboard.mp4 --cols 9 --rows 6 --square 25
python calibrate.py --images "calib/*.jpg" --cols 9 --rows 6
```

`--cols`/`--rows` count *inner corners*; `--square` is the square size in mm.
The result is written to `camera.yml` (`--out` to change it). At least 10
frames with a detected board are required.

## Splitting a video into frames (optional)

Not needed (the stabilisers read videos directly), but handy for inspection:

```bash
ffmpeg -i knight.mp4 frames/frame_%04d.png
```
