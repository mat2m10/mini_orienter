"""
calibrate.py
------------
Run ONCE to calibrate your phone camera.
Film a checkerboard (or print one) from many angles — 20-30 seconds.
Saves camera.yml which the other scripts use.

Usage:
    python calibrate.py --video checkerboard.mp4 --cols 9 --rows 6
    python calibrate.py --images ./calib_frames/*.jpg  # or from still photos

A standard 9x6 checkerboard has 9 inner corners wide, 6 inner corners tall.
Print one free at: https://calib.io/pages/camera-calibration-pattern-generator
"""

import cv2
import numpy as np
import argparse
import glob
import sys

def calibrate_from_frames(frames, cols, rows, square_size_mm=25.0):
    """
    frames       : list of BGR images
    cols, rows   : inner corner count (not squares — corners)
    square_size_mm: physical size of one square in mm
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D points in real world: (0,0,0), (1,0,0), ... scaled to mm
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    obj_points = []   # 3D world points
    img_points = []   # 2D image points
    good = 0

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            good += 1
            print(f"  Frame {i:04d}: found corners ({good} good so far)")
        else:
            print(f"  Frame {i:04d}: no corners found, skipping")

    if good < 10:
        print(f"\nOnly {good} good frames — need at least 10. Film more angles.")
        sys.exit(1)

    print(f"\nCalibrating from {good} frames...")
    h, w = frames[0].shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    print(f"Reprojection error: {ret:.4f}px  (good if < 1.0)")
    return camera_matrix, dist_coeffs, (w, h)


def load_frames_from_video(path, max_frames=80, skip=5):
    """Extract frames from a video file, skipping some for variety."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open video: {path}")
        sys.exit(1)
    frames = []
    i = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip == 0:
            frames.append(frame)
        i += 1
    cap.release()
    print(f"Loaded {len(frames)} frames from video.")
    return frames


def load_frames_from_images(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No images found matching: {pattern}")
        sys.exit(1)
    frames = [cv2.imread(p) for p in paths]
    print(f"Loaded {len(frames)} images.")
    return frames


def save_calibration(path, camera_matrix, dist_coeffs, image_size):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.release()
    print(f"\nCalibration saved to {path}")
    print("Camera matrix:")
    print(camera_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument("--video",  help="Path to checkerboard video")
    parser.add_argument("--images", help="Glob pattern for images, e.g. './calib/*.jpg'")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners horizontally")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners vertically")
    parser.add_argument("--square", type=float, default=25.0, help="Square size in mm")
    parser.add_argument("--out", default="camera.yml", help="Output calibration file")
    args = parser.parse_args()

    if args.video:
        frames = load_frames_from_video(args.video)
    elif args.images:
        frames = load_frames_from_images(args.images)
    else:
        print("Provide --video or --images")
        sys.exit(1)

    cam_mat, dist, img_size = calibrate_from_frames(frames, args.cols, args.rows, args.square)
    save_calibration(args.out, cam_mat, dist, img_size)
