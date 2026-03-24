"""
stabilise.py
------------
Main script. Takes your video of the knight, detects the De Bruijn strip
each frame, computes pose, and outputs a stabilised video where the
miniature stays locked and the background moves.

Usage (simplest — no calibration):
    python stabilise.py --video knight.mp4 --radius 14.5 --strip-height 6

Usage (with camera calibration for accurate 3D pose):
    python stabilise.py --video knight.mp4 --radius 14.5 --strip-height 6 \
                        --camera camera.yml

Options:
    --video          Input video file
    --radius         Base radius in mm (half the diameter you measured)
    --strip-height   Strip height in mm
    --rotation       Sequence rotation used when generating the strip (default 0)
    --camera         Path to camera.yml from calibrate.py (optional)
    --out            Output video filename (default: stabilised.mp4)
    --debug          Show annotated frames while processing
    --reference      Frame number to use as the reference pose (default: 0)
    --frames-dir     If set, also save individual PNG frames here
"""

import cv2
import numpy as np
import argparse
import os
import sys
from detect_strip import DeBruijnDetector


def load_camera(path):
    """Load camera matrix and distortion coeffs from calibration file."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    cam = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    print(f"Loaded camera calibration from {path}")
    return cam, dist


def rvec_tvec_to_matrix(rvec, tvec):
    """Convert OpenCV pose to 4x4 homogeneous matrix."""
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3,:3] = R
    M[:3, 3] = tvec.ravel()
    return M


def compute_warp_homography(ref_pts, cur_pts):
    """
    2D fallback: compute homography between reference and current
    cell sample points (no calibration needed).
    """
    if len(ref_pts) < 4 or len(cur_pts) < 4:
        return None
    ref = np.array(ref_pts[:len(cur_pts)], dtype=np.float32)
    cur = np.array(cur_pts[:len(ref_pts)], dtype=np.float32)
    H, mask = cv2.findHomography(cur, ref, cv2.RANSAC, 5.0)
    return H


def project_base_centre(rvec, tvec, cam_mat, dist):
    """Project the 3D origin (base centre) to 2D image coordinates."""
    pts, _ = cv2.projectPoints(
        np.array([[0.,0.,0.]]), rvec, tvec, cam_mat, dist
    )
    return pts[0][0]


def stabilise(args):
    # Load camera if provided
    cam_mat, dist_coeffs = None, None
    if args.camera:
        cam_mat, dist_coeffs = load_camera(args.camera)

    use_3d = cam_mat is not None

    # Set up detector
    detector = DeBruijnDetector(
        base_radius_mm=args.radius,
        strip_height_mm=args.strip_height,
        n_window=4,
        sequence_rotation=args.rotation,
        camera_matrix=cam_mat,
        dist_coeffs=dist_coeffs,
        debug=args.debug
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # Optional frames directory
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)

    # --- Pass 1: find the reference frame ---
    print(f"\nSeeking reference frame {args.reference}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.reference)
    ret, ref_frame = cap.read()
    if not ret:
        print("Cannot read reference frame.")
        sys.exit(1)

    ref_result = detector.detect(ref_frame)
    if not ref_result['found']:
        print("Strip not found in reference frame. Try a different --reference frame number.")
        sys.exit(1)

    ref_rvec   = ref_result.get('rvec')
    ref_tvec   = ref_result.get('tvec')
    ref_pts    = ref_result.get('debug_frame')  # not used here
    ref_ellipse = ref_result.get('ellipse')
    print(f"Reference pose found. start_idx={ref_result['start_idx']}, "
          f"confidence={ref_result['confidence']:.2f}")

    # --- Pass 2: process all frames ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    n_stabilised = 0
    n_fallback   = 0
    n_skipped    = 0
    last_H = None  # carry forward last good warp

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        warp_frame = frame.copy()

        if result['found']:
            if use_3d and ref_rvec is not None and result['rvec'] is not None:
                # --- 3D pose stabilisation ---
                # Compute relative transform: ref_pose * inv(cur_pose)
                M_ref = rvec_tvec_to_matrix(ref_rvec, ref_tvec)
                M_cur = rvec_tvec_to_matrix(result['rvec'], result['tvec'])
                M_rel = M_ref @ np.linalg.inv(M_cur)

                # Project 4 base corners to get homography
                corners_3d = np.array([
                    [ args.radius,  args.radius, 0],
                    [-args.radius,  args.radius, 0],
                    [-args.radius, -args.radius, 0],
                    [ args.radius, -args.radius, 0],
                ], dtype=np.float32)

                # Project using current pose
                cur_r, _ = cv2.Rodrigues(M_cur[:3,:3])
                cur_t    = M_cur[:3, 3]
                pts_cur, _ = cv2.projectPoints(corners_3d, result['rvec'],
                                               result['tvec'], cam_mat, dist_coeffs)

                # Project using reference pose
                pts_ref, _ = cv2.projectPoints(corners_3d, ref_rvec,
                                               ref_tvec, cam_mat, dist_coeffs)

                pts_cur = pts_cur.reshape(-1,2).astype(np.float32)
                pts_ref = pts_ref.reshape(-1,2).astype(np.float32)

                H, _ = cv2.findHomography(pts_cur, pts_ref)
                if H is not None:
                    last_H = H
                    warp_frame = cv2.warpPerspective(frame, H, (width, height))
                    n_stabilised += 1

            else:
                # --- 2D homography fallback ---
                # Use cell sample positions directly
                # This works well for small camera movements / rotations
                if ref_ellipse is not None and result.get('ellipse') is not None:
                    ref_cell_pts = _ellipse_sample_pts(ref_ellipse, detector.n_cells)
                    cur_cell_pts = _ellipse_sample_pts(result['ellipse'], detector.n_cells)

                    # Align by start_idx so we match corresponding cells
                    si = result.get('start_idx', 0)
                    cur_aligned = cur_cell_pts[si:] + cur_cell_pts[:si]

                    H = compute_warp_homography(ref_cell_pts, cur_aligned)
                    if H is not None:
                        last_H = H
                        warp_frame = cv2.warpPerspective(frame, H, (width, height))
                        n_stabilised += 1
                    else:
                        n_fallback += 1

        else:
            # Strip not found — carry forward last known warp
            if last_H is not None:
                warp_frame = cv2.warpPerspective(frame, last_H, (width, height))
                n_fallback += 1
            else:
                n_skipped += 1

        # Overlay status
        status = "OK" if result['found'] else "LOST"
        conf   = result.get('confidence', 0)
        cv2.putText(warp_frame, f"[{frame_idx:04d}] {status} conf={conf:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        out.write(warp_frame)

        if args.frames_dir:
            cv2.imwrite(f"{args.frames_dir}/frame_{frame_idx:04d}.png", warp_frame)

        if args.debug:
            cv2.imshow("Stabilised", warp_frame)
            if result.get('debug_frame') is not None:
                cv2.imshow("Detector", result['debug_frame'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx % 30 == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} ({pct:.0f}%) — "
                  f"stabilised={n_stabilised} fallback={n_fallback} skipped={n_skipped}")

        frame_idx += 1

    cap.release()
    out.release()
    if args.debug:
        cv2.destroyAllWindows()

    print(f"\nDone. {frame_idx} frames processed.")
    print(f"  Stabilised : {n_stabilised}")
    print(f"  Fallback   : {n_fallback}  (strip lost, used last good warp)")
    print(f"  Skipped    : {n_skipped}   (no warp available)")
    print(f"  Output     : {args.out}")


def _ellipse_sample_pts(ellipse, n):
    """Sample n evenly-spaced points along an ellipse midline."""
    (cx, cy), (ma, mi), angle_deg = ellipse
    angle_rad = np.deg2rad(angle_deg)
    pts = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        x = cx + (ma/2)*np.cos(theta)*np.cos(angle_rad) \
               - (mi/2)*np.sin(theta)*np.sin(angle_rad)
        y = cy + (ma/2)*np.cos(theta)*np.sin(angle_rad) \
               + (mi/2)*np.sin(theta)*np.cos(angle_rad)
        pts.append((x, y))
    return pts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De Bruijn strip stabiliser")
    parser.add_argument("--video",        required=True,    help="Input video")
    parser.add_argument("--radius",       type=float, required=True,
                                          help="Base radius in mm")
    parser.add_argument("--strip-height", type=float, default=6.0,
                                          help="Strip height in mm")
    parser.add_argument("--rotation",     type=int,   default=0,
                                          help="Sequence rotation from generator")
    parser.add_argument("--camera",       default=None,     help="camera.yml path")
    parser.add_argument("--out",          default="stabilised.mp4")
    parser.add_argument("--debug",        action="store_true")
    parser.add_argument("--reference",    type=int,   default=0,
                                          help="Reference frame number")
    parser.add_argument("--frames-dir",   default=None,
                                          help="Save individual PNG frames here")
    args = parser.parse_args()
    stabilise(args)
