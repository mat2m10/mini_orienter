"""
stabilise_aruco.py
------------------
Stabilise miniature painting video using ArUco markers placed around
a cylindrical handle/base. At least one marker is always visible
regardless of orientation.

Setup:
  - Print print_markers.py output (markers.svg), cut out the 4 markers
  - Glue them around your cork/handle at 90° intervals
  - Each marker is a different ID so OpenCV knows which face it's seeing
  - Other layouts: N markers on a regular N-sided prism (e.g. the
    hexagonal fiducial from generate_aruco: --faces 6), IDs first-id ..
    first-id+N-1 in order around the axis, all upright, same direction.
    Order: with face 0 facing you, face 1 is on its right (IDs increase
    counter-clockwise seen from the top of the handle, +k*360/N about +Y)

Usage:
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20 --camera camera.yml
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20 --debug
    python stabilise_aruco.py --video data/knight.mp4 --marker-size 20 --faces 6

Arguments:
    --video        Input video file
    --marker-size  Physical side length of each marker in mm
    --handle-radius  Distance from the handle axis to each marker's surface
                   in mm (default: estimated as marker_size / (2 tan(pi/faces)),
                   the apothem of a prism whose faces are one marker wide)
    --camera       Optional camera.yml from calibrate.py (improves accuracy)
    --out          Output video (default: stabilised.mp4)
    --reference    Frame number to use as locked reference pose (default: auto)
    --debug        Show live annotated preview
    --frames-dir   Also save individual PNG frames here
    --faces        Number of marker faces around the axis (default 4)
    --first-id     Marker ID on face 0; faces use consecutive IDs (default 0)
    --dict         ArUco dictionary, e.g. 4X4_50 (default), 5X5_100, 6X6_250
    --smooth       Pose smoothing weight in (0, 1]; 1 = off (default).
                   e.g. 0.5 averages SLERP-wise with the previous pose
    --max-extrapolate  Frames to extrapolate the last motion through a
                   marker dropout before holding (default 0 = hold)
    --pose-out     Write the per-frame handle pose/orientation to this
                   .csv or .json file (see pose_export.py for columns)
    --no-video     Do not write the stabilised video (use with --pose-out)
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os
import sys
from pose_export import pose_record, write_poses
from pose_smoothing import PoseSmoother


# ---------------------------------------------------------------------------
# Marker layout: which ID is on which face of the handle
# Each marker is rotated around the Y axis by face_angle degrees
# Default: 4 markers at 90° intervals covers a full cylinder
# ---------------------------------------------------------------------------

DICTS = {
    "4X4_50":  aruco.DICT_4X4_50,
    "4X4_100": aruco.DICT_4X4_100,
    "4X4_250": aruco.DICT_4X4_250,
    "5X5_50":  aruco.DICT_5X5_50,
    "5X5_100": aruco.DICT_5X5_100,
    "5X5_250": aruco.DICT_5X5_250,
    "6X6_50":  aruco.DICT_6X6_50,
    "6X6_100": aruco.DICT_6X6_100,
    "6X6_250": aruco.DICT_6X6_250,
    "ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}


def face_layout(n_faces=4, first_id=0):
    """
    Marker IDs and face angles (degrees around the handle's Y axis) for a
    regular N-sided prism: face k carries ID first_id + k at k * 360/N.
    """
    if n_faces < 1:
        raise ValueError("n_faces must be >= 1")
    ids = [first_id + k for k in range(n_faces)]
    angles = [360.0 * k / n_faces for k in range(n_faces)]
    return ids, angles


MARKER_IDS, FACE_ANGLES = face_layout(4)   # [0..3], [0, 90, 180, 270]


def estimate_handle_radius(marker_size_mm, n_faces=4):
    """
    Default distance from the handle axis to the marker surfaces when
    --handle-radius is not given: the apothem of a regular N-sided prism
    whose faces are exactly one marker wide, s / (2 tan(pi/N)).
    Markers with a white margin or glued on a round handle sit further out,
    so pass --handle-radius when the real value is known.
    """
    if n_faces < 3:
        raise ValueError("a handle radius can only be estimated for >= 3 faces")
    return marker_size_mm / (2.0 * np.tan(np.pi / n_faces))

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
    if not os.path.isfile(path):
        sys.exit(f"Camera file not found: {path}")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        sys.exit(f"Cannot read camera file: {path}")
    cam  = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if cam is None or cam.shape != (3, 3):
        sys.exit(f"{path}: missing or invalid 'camera_matrix' "
                 f"(expected 3x3, run calibrate.py)")
    if dist is None or dist.size == 0:
        sys.exit(f"{path}: missing 'dist_coeffs' (run calibrate.py)")
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

def make_detector(dict_id=aruco.DICT_4X4_50):
    """Create ArUco detector with tuned parameters for small hand-drawn markers."""
    dictionary = aruco.getPredefinedDictionary(dict_id)
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

def pose_from_marker(corners, marker_id, marker_size_mm, cam_mat, dist,
                     handle_radius_mm=0.0, marker_ids=None, face_angles=None):
    """
    Estimate the pose of the HANDLE given a detected marker.

    We know:
      - The marker's pose relative to the camera (solvePnP)
      - The marker's pose relative to the handle: rotated by its face angle
        around the handle's Y axis and pushed out by handle_radius_mm along
        its own normal (marker +Z, pointing away from the axis)

    So: handle_pose = marker_pose * inv(marker_on_handle)

    The handle origin returned here is on the handle axis, so every marker
    ID yields the same handle pose and switching markers does not jump.
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
    if marker_ids is None:
        marker_ids, face_angles = MARKER_IDS, FACE_ANGLES
    if marker_id not in marker_ids:
        return None, None
    face_angle = face_angles[marker_ids.index(marker_id)]

    R_marker, _ = cv2.Rodrigues(rvec)
    R_face = marker_offset_rotation(face_angle)

    # Marker in handle frame: p_handle = R_face @ p_marker + R_face @ [0,0,r]
    # Marker in camera frame: p_cam = R_marker @ p_marker + tvec
    # => p_cam = (R_marker @ R_face.T) @ p_handle + tvec - R_marker @ [0,0,r]
    R_handle = R_marker @ R_face.T
    offset = np.array([[0.0], [0.0], [float(handle_radius_mm)]])
    t_handle = tvec.reshape(3, 1) - R_marker @ offset

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

    # Detector and marker layout
    detector, dictionary = make_detector(DICTS[args.dict])
    marker_ids, face_angles = face_layout(args.faces, args.first_id)
    print(f"Layout: {args.faces} faces, IDs {marker_ids} "
          f"({360.0 / args.faces:g}° apart), dictionary {args.dict}")

    # Distance from the handle axis to the marker surfaces
    if args.handle_radius is None:
        try:
            handle_radius = estimate_handle_radius(args.marker_size, args.faces)
        except ValueError as e:
            sys.exit(f"{e}; pass --handle-radius")
        print(f"Handle radius not given — estimated {handle_radius:.2f} mm "
              f"({args.faces} faces of {args.marker_size} mm on a regular prism). "
              f"Pass --handle-radius for accuracy.")
    else:
        handle_radius = args.handle_radius

    # Output
    out = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Cannot open output video for writing: {args.out}")
            sys.exit(1)
    elif not args.pose_out:
        print("Warning: --no-video without --pose-out writes no output.")
    pose_records = [] if args.pose_out else None
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)

    # ---- Find reference frame ----
    ref_rvec = None
    ref_tvec = None
    ref_frame_idx = 0

    print("\nSearching for reference frame (first frame with a visible marker)...")
    start_search = args.reference if args.reference >= 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_search)

    # Do not bound this by CAP_PROP_FRAME_COUNT: some containers/backends
    # report 0 frames. The loop stops at end of video via cap.read().
    for search_idx in range(start_search, start_search + 120):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            # Use first detected marker
            for i, mid in enumerate(ids.ravel()):
                if mid in marker_ids:
                    rv, tv = pose_from_marker(
                        [corners[i]], mid, args.marker_size, cam_mat, dist,
                        handle_radius, marker_ids, face_angles
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

    # Optional temporal smoothing / dropout extrapolation of the handle pose
    smoother = None
    if args.smooth < 1.0 or args.max_extrapolate > 0:
        smoother = PoseSmoother(args.smooth, args.max_extrapolate)
        print(f"Pose smoothing: alpha={args.smooth}, "
              f"extrapolate up to {args.max_extrapolate} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        warp_frame = frame.copy()
        status_text = "LOST"
        found = False
        record = pose_record(frame_idx)

        if ids is not None:
            # Try each detected marker — use first valid one
            for i, mid in enumerate(ids.ravel()):
                if mid not in marker_ids:
                    continue
                rv, tv = pose_from_marker(
                    [corners[i]], mid, args.marker_size, cam_mat, dist,
                    handle_radius, marker_ids, face_angles
                )
                if rv is None:
                    continue
                if smoother is not None:
                    rv, tv = smoother.update(rv, tv)

                H = poses_to_homography(
                    ref_rvec, ref_tvec, rv, tv,
                    cam_mat, dist, args.marker_size
                )
                if H is not None:
                    last_H = H
                    record = pose_record(frame_idx, mid, rv, tv, ref_rvec)
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

        if not found and smoother is not None:
            pose = smoother.predict()
            if pose is not None:
                H = poses_to_homography(ref_rvec, ref_tvec, *pose,
                                        cam_mat, dist, args.marker_size)
                if H is not None:
                    last_H = H
                    extrapolating = smoother.missed <= args.max_extrapolate
                    status_text = ("PREDICTED (no marker)" if extrapolating
                                   else "HELD (no marker)")

        if not found:
            if status_text != "LOST":
                warp_frame = cv2.warpPerspective(frame, last_H, (width, height))
                n_fallback += 1
            elif last_H is not None:
                warp_frame = cv2.warpPerspective(frame, last_H, (width, height))
                status_text = "HELD (no marker)"
                n_fallback += 1
            else:
                n_lost += 1

        # Status overlay
        col = (0,220,0) if found else (0,140,255)
        cv2.putText(warp_frame, f"[{frame_idx:04d}] {status_text}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        if pose_records is not None:
            pose_records.append(record)

        if out is not None:
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
    if out is not None:
        out.release()
    if pose_records is not None:
        write_poses(args.pose_out, pose_records)
    if args.debug:
        cv2.destroyAllWindows()

    print(f"\nDone.")
    print(f"  Stabilised : {n_stabilised}")
    print(f"  Held       : {n_fallback}  (marker lost, used last known warp)")
    print(f"  Lost       : {n_lost}      (no warp available at all)")
    if out is not None:
        print(f"  Output     : {args.out}")
    if pose_records is not None:
        print(f"  Poses      : {args.pose_out}  ({len(pose_records)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco multi-marker stabiliser")
    parser.add_argument("--video",       required=True)
    parser.add_argument("--marker-size", type=float, required=True,
                        help="Marker side length in mm")
    parser.add_argument("--handle-radius", type=float, default=None,
                        help="Handle axis to marker surface distance in mm "
                             "(default: marker_size / (2 tan(pi/faces)))")
    parser.add_argument("--camera",      default=None)
    parser.add_argument("--out",         default="stabilised.mp4")
    parser.add_argument("--reference",   type=int, default=0,
                        help="Start searching for reference pose from this frame")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--frames-dir",  default=None)
    parser.add_argument("--faces",       type=int, default=4,
                        help="Marker faces around the axis (6 for a hexagonal prism)")
    parser.add_argument("--first-id",    type=int, default=0,
                        help="Marker ID on face 0 (faces use consecutive IDs)")
    parser.add_argument("--dict",        default="4X4_50", choices=sorted(DICTS),
                        help="ArUco dictionary the markers were printed from")
    parser.add_argument("--smooth",      type=float, default=1.0,
                        help="Pose smoothing weight in (0,1]; 1 = off")
    parser.add_argument("--max-extrapolate", type=int, default=0,
                        help="Frames to extrapolate motion through a dropout")
    parser.add_argument("--pose-out",    default=None,
                        help="Write per-frame handle pose to this .csv/.json")
    parser.add_argument("--no-video",    action="store_true",
                        help="Skip writing the stabilised video")
    args = parser.parse_args()
    stabilise(args)
