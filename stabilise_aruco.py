"""
stabilise_aruco.py
------------------
Stabilise miniature painting video using ArUco markers placed around
a cylindrical handle/base. At least one marker is always visible
regardless of orientation.

Setup:
  - Print markers_sheet.py output, cut into 4 strips
  - Glue them around your cork/handle at 90° intervals
  - Each marker is a different ID so OpenCV knows which face it's seeing

Usage:
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20 --camera camera.yml
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20 --debug

Arguments:
    --video        Input video file
    --marker-size  Physical side length of each marker in mm
    --camera       Optional camera.yml from calibrate.py (improves accuracy)
    --out          Output video (default: stabilised.mp4)
    --reference    Frame number to use as locked reference pose (default: auto)
    --debug        Show live annotated preview
    --frames-dir   Also save individual PNG frames here
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Marker layout: which ID is on which face of the handle
# Each marker is rotated around the Y axis by face_angle degrees
# 4 markers at 90° intervals covers a full cylinder
# ---------------------------------------------------------------------------

MARKER_IDS   = [0, 1, 2, 3]          # IDs printed on the strip sheet
FACE_ANGLES  = [0, 90, 180, 270]     # degrees around cylinder axis

def marker_offset_rotation(face_angle_deg):
    """
    Rotation matrix for a marker on the face at face_angle_deg around Y axis.
    This encodes where the marker sits on the handle relative to marker 0.
    """
    a = np.deg2rad(face_angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [         0, 1,         0],
        [-np.sin(a), 0, np.cos(a)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera calibration loader
# ---------------------------------------------------------------------------

def load_camera(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    cam  = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    print(f"Loaded calibration from {path}")
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
    dist = np.zeros((4,1), dtype=np.float64)
    print("No calibration file — using estimated camera matrix (focal = image width).")
    print("For better results run: python calibrate.py --video checkerboard.mp4")
    return cam, dist


# ---------------------------------------------------------------------------
# ArUco setup
# ---------------------------------------------------------------------------

def make_detector():
    """Create ArUco detector with tuned parameters for small hand-drawn markers."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()

    # More aggressive corner refinement — helps with pencil-drawn or printed markers
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # Allow smaller markers (important for a cork-sized handle)
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0

    # More lenient thresholding — helps with uneven pencil/print quality
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 53
    params.adaptiveThreshWinSizeStep = 10

    detector = aruco.ArucoDetector(dictionary, params)
    return detector, dictionary


# ---------------------------------------------------------------------------
# Pose from a single detected marker
# ---------------------------------------------------------------------------

def pose_from_marker(corners, marker_id, marker_size_mm, cam_mat, dist):
    """
    Estimate the pose of the HANDLE given a detected marker.

    We know:
      - The marker's pose relative to the camera (solvePnP)
      - The marker's pose relative to the handle (from FACE_ANGLES)

    So: handle_pose = marker_pose * inv(marker_on_handle)
    """
    half = marker_size_mm / 2.0

    # 3D corners of the marker in the marker's own frame
    obj_pts = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    img_pts = corners[0].astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, cam_mat, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None, None

    # Convert marker pose to handle pose using known face angle
    face_idx = MARKER_IDS.index(marker_id) if marker_id in MARKER_IDS else 0
    face_angle = FACE_ANGLES[face_idx]

    R_marker, _ = cv2.Rodrigues(rvec)
    R_face = marker_offset_rotation(face_angle)

    # Handle frame = camera -> marker -> (marker -> handle)
    R_handle = R_marker @ R_face.T
    t_handle = tvec  # translation is the same (marker is on handle surface)

    rvec_handle, _ = cv2.Rodrigues(R_handle)
    return rvec_handle, t_handle


# ---------------------------------------------------------------------------
# Stabilisation warp from two poses
# ---------------------------------------------------------------------------

def poses_to_homography(rvec_ref, tvec_ref, rvec_cur, tvec_cur,
                         cam_mat, dist, marker_size_mm):
    """
    Compute a 2D homography that warps the current frame to match
    the reference frame, using the two handle poses.
    """
    # Project a set of 3D reference points using both poses,
    # then find the homography between the two projected sets.
    half = marker_size_mm * 2  # use a larger virtual plane for stability
    pts_3d = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
        [    0,     0, 0],  # centre
        [    0,  half, 0],
        [ half,     0, 0],
        [-half,     0, 0],
    ], dtype=np.float32)

    pts_ref, _ = cv2.projectPoints(pts_3d, rvec_ref, tvec_ref, cam_mat, dist)
    pts_cur, _ = cv2.projectPoints(pts_3d, rvec_cur, tvec_cur, cam_mat, dist)

    pts_ref = pts_ref.reshape(-1, 2).astype(np.float32)
    pts_cur = pts_cur.reshape(-1, 2).astype(np.float32)

    H, mask = cv2.findHomography(pts_cur, pts_ref, cv2.RANSAC, 3.0)
    return H


# ---------------------------------------------------------------------------
# Main stabilisation loop
# ---------------------------------------------------------------------------

def stabilise(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

    # Camera matrix
    if args.camera:
        cam_mat, dist = load_camera(args.camera)
    else:
        cam_mat, dist = default_camera(width, height)

    # Detector
    detector, dictionary = make_detector()

    # Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)

    # ---- Find reference frame ----
    ref_rvec = None
    ref_tvec = None
    ref_frame_idx = 0

    print("\nSearching for reference frame (first frame with a visible marker)...")
    start_search = args.reference if args.reference >= 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_search)

    for search_idx in range(start_search, min(start_search + 120, total)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            # Use first detected marker
            for i, mid in enumerate(ids.ravel()):
                if mid in MARKER_IDS:
                    rv, tv = pose_from_marker(
                        [corners[i]], mid, args.marker_size, cam_mat, dist
                    )
                    if rv is not None:
                        ref_rvec = rv
                        ref_tvec = tv
                        ref_frame_idx = search_idx
                        print(f"Reference pose found at frame {search_idx} "
                              f"using marker ID {mid}")
                        break
            if ref_rvec is not None:
                break

    if ref_rvec is None:
        print("\nNo markers found in the first 120 frames.")
        print("Check that your markers are visible and well-lit.")
        print("Run with --debug to see what the detector sees.")
        sys.exit(1)

    # ---- Process all frames ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n_stabilised = 0
    n_fallback   = 0
    n_lost       = 0
    last_H       = None
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        warp_frame = frame.copy()
        status_text = "LOST"
        found = False

        if ids is not None:
            # Try each detected marker — use first valid one
            for i, mid in enumerate(ids.ravel()):
                if mid not in MARKER_IDS:
                    continue
                rv, tv = pose_from_marker(
                    [corners[i]], mid, args.marker_size, cam_mat, dist
                )
                if rv is None:
                    continue

                H = poses_to_homography(
                    ref_rvec, ref_tvec, rv, tv,
                    cam_mat, dist, args.marker_size
                )
                if H is not None:
                    last_H = H
                    warp_frame = cv2.warpPerspective(frame, H, (width, height))
                    status_text = f"OK  marker={mid}"
                    n_stabilised += 1
                    found = True

                    if args.debug:
                        aruco.drawDetectedMarkers(warp_frame, corners, ids)
                        # Draw axes on the handle
                        cv2.drawFrameAxes(
                            warp_frame, cam_mat, dist, rv, tv,
                            args.marker_size * 1.5
                        )
                    break

        if not found:
            if last_H is not None:
                warp_frame = cv2.warpPerspective(frame, last_H, (width, height))
                status_text = "HELD (no marker)"
                n_fallback += 1
            else:
                n_lost += 1

        # Status overlay
        col = (0,220,0) if found else (0,140,255)
        cv2.putText(warp_frame, f"[{frame_idx:04d}] {status_text}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        out.write(warp_frame)

        if args.frames_dir:
            cv2.imwrite(
                f"{args.frames_dir}/frame_{frame_idx:04d}.png", warp_frame
            )

        if args.debug:
            cv2.imshow("Stabilised", warp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx % 30 == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  "
                  f"ok={n_stabilised} held={n_fallback} lost={n_lost}")

        frame_idx += 1

    cap.release()
    out.release()
    if args.debug:
        cv2.destroyAllWindows()

    print(f"\nDone.")
    print(f"  Stabilised : {n_stabilised}")
    print(f"  Held       : {n_fallback}  (marker lost, used last known warp)")
    print(f"  Lost       : {n_lost}      (no warp available at all)")
    print(f"  Output     : {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco multi-marker stabiliser")
    parser.add_argument("--video",       required=True)
    parser.add_argument("--marker-size", type=float, required=True,
                        help="Marker side length in mm")
    parser.add_argument("--camera",      default=None)
    parser.add_argument("--out",         default="stabilised.mp4")
    parser.add_argument("--reference",   type=int, default=0,
                        help="Start searching for reference pose from this frame")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--frames-dir",  default=None)
    args = parser.parse_args()
    stabilise(args)
