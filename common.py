"""
common.py
---------
Helpers shared by stabilise.py, stabilise_aruco.py and detect_aruco.py:
camera calibration loading, ArUco detector setup, video input/output and
progress reporting.
"""

import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def load_camera(path):
    """Load camera matrix and distortion coeffs from a calibrate.py file."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    cam = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    print(f"Loaded camera calibration from {path}")
    return cam, dist


def default_camera(w, h):
    """
    Rough camera matrix when no calibration file is available.
    Assumes focal length ≈ image width (reasonable for phone cameras).
    Good enough for stabilisation, not for metric measurements.
    """
    f = w  # focal length estimate in pixels
    cam = np.array([
        [f,   0, w/2],
        [0,   f, h/2],
        [0,   0,   1],
    ], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    print("No calibration file — using estimated camera matrix (focal = image width).")
    print("For better results run: python calibrate.py --video checkerboard.mp4")
    return cam, dist


# ---------------------------------------------------------------------------
# ArUco
# ---------------------------------------------------------------------------

def make_aruco_detector(dict_id=aruco.DICT_4X4_50,
                        min_perimeter_rate=0.03,
                        max_perimeter_rate=4.0,
                        subpix=True,
                        polygonal_approx_rate=None):
    """
    ArUco detector with lenient thresholding, tuned for small printed or
    hand-drawn markers. Returns (detector, dictionary).
    """
    dictionary = aruco.getPredefinedDictionary(dict_id)
    params = aruco.DetectorParameters()

    if subpix:
        # Corner refinement helps with pencil-drawn or printed markers
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # Allow small markers (important for a cork-sized handle)
    params.minMarkerPerimeterRate = min_perimeter_rate
    params.maxMarkerPerimeterRate = max_perimeter_rate

    # More lenient thresholding — helps with uneven pencil/print quality
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10

    if polygonal_approx_rate is not None:
        # More forgiving of wobbly (hand-drawn) edges
        params.polygonalApproxAccuracyRate = polygonal_approx_rate

    return aruco.ArucoDetector(dictionary, params), dictionary


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def open_video(path):
    """Open a video or exit. Returns (cap, fps, width, height, total_frames)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")
    return cap, fps, width, height, total


class FrameSink:
    """Writes output frames to an mp4 and, optionally, to numbered PNGs."""

    def __init__(self, out_path, fps, size, frames_dir=None):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, size)
        self.frames_dir = frames_dir
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)

    def write(self, frame, frame_idx):
        self.writer.write(frame)
        if self.frames_dir:
            cv2.imwrite(f"{self.frames_dir}/frame_{frame_idx:04d}.png", frame)

    def release(self):
        self.writer.release()


def print_progress(frame_idx, total, every=30, **counts):
    """Print a progress line every `every` frames."""
    if frame_idx % every:
        return
    pct = 100 * frame_idx / max(total, 1)
    stats = " ".join(f"{k}={v}" for k, v in counts.items())
    print(f"  {frame_idx}/{total} ({pct:.0f}%)  {stats}")
