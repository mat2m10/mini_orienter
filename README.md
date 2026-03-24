# De Bruijn Strip Tracker
Stabilise miniature painting video using a hand-drawn De Bruijn ring on the base.

## Install
```bash
pip install opencv-python numpy
```

## Quick start (no calibration)
```bash
# 1. Film your knight, save as knight.mp4
# 2. Run stabiliser (2D homography mode, no camera.yml needed)
python stabilise.py --video knight.mp4 --radius 14.5 --strip-height 6 --debug
```

## With camera calibration (better accuracy)
```bash
# Film a checkerboard from many angles (~20 seconds)
python calibrate.py --video checkerboard.mp4 --cols 9 --rows 6

# Then stabilise with full 3D pose
python stabilise.py --video knight.mp4 --radius 14.5 --strip-height 6 \
                    --camera camera.yml --debug
```

## Decomposing video to frames first (optional)
```bash
# With ffmpeg:
ffmpeg -i knight.mp4 frames/frame_%04d.png

# Or just pass the video directly — stabilise.py handles it.
```

## Parameters
| Flag | What it is |
|------|-----------|
| `--radius` | Half your base diameter in mm. Measure with ruler. |
| `--strip-height` | How tall the strip is in mm |
| `--rotation` | The rotation value you used in the strip generator |
| `--reference` | Which frame to lock to (default: frame 0) |
| `--debug` | Show live annotated windows while processing |

## Troubleshooting
- **Strip not found**: try `--debug` to see what the detector sees. May need to tune lighting.
- **Wobbly output**: your strip drawing may be uneven — the 2D fallback mode is more forgiving.
- **Low confidence**: make sure pencil cells are dark enough. Press harder, go over 2-3x.
