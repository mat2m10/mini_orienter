"""
pose_export.py
--------------
Per-frame handle orientation export for stabilise_aruco.py (--pose-out).

One record per video frame:
    frame        frame index
    detected     1 if a marker gave a pose this frame, else 0
    marker_id    ID of the marker used
    rx, ry, rz   handle rotation vector (Rodrigues, radians), camera frame
    tx, ty, tz   handle origin in the camera frame (mm)
    qw..qz       the same rotation as a unit quaternion (w, x, y, z)
    yaw_deg      rotation about the handle's own axis (Y) relative to the
                 reference pose, in (-180, 180]; what a turntable viewer
                 needs. Tilt of the axis is ignored by this value.
Undetected frames keep empty pose fields.

The file format follows the extension: .json writes a list of records,
anything else writes CSV.
"""

import csv
import json

import cv2
import numpy as np

from pose_smoothing import quat_from_matrix

FIELDS = ["frame", "detected", "marker_id",
          "rx", "ry", "rz", "tx", "ty", "tz",
          "qw", "qx", "qy", "qz", "yaw_deg"]


def quat_from_rvec(rvec):
    """Unit quaternion (w, x, y, z) from a Rodrigues rotation vector, w >= 0."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    q = quat_from_matrix(R)
    return -q if q[0] < 0 else q


def relative_yaw_deg(rvec_ref, rvec_cur):
    """
    Rotation about the handle Y axis from the reference to the current
    pose. With a fixed camera, R_cur = R_ref @ Ry(yaw).
    """
    R_ref, _ = cv2.Rodrigues(np.asarray(rvec_ref, dtype=np.float64))
    R_cur, _ = cv2.Rodrigues(np.asarray(rvec_cur, dtype=np.float64))
    R_rel = R_ref.T @ R_cur
    return float(np.degrees(np.arctan2(R_rel[0, 2], R_rel[2, 2])))


def pose_record(frame_idx, marker_id=None, rvec=None, tvec=None, rvec_ref=None):
    """Build one export record; pass marker_id=None for an undetected frame."""
    rec = {k: None for k in FIELDS}
    rec["frame"] = int(frame_idx)
    rec["detected"] = 0
    if marker_id is None or rvec is None:
        return rec
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    q = quat_from_rvec(r)
    rec.update(detected=1, marker_id=int(marker_id),
               rx=r[0], ry=r[1], rz=r[2], tx=t[0], ty=t[1], tz=t[2],
               qw=q[0], qx=q[1], qy=q[2], qz=q[3])
    if rvec_ref is not None:
        rec["yaw_deg"] = relative_yaw_deg(rvec_ref, r)
    return {k: (float(v) if isinstance(v, np.floating) else v)
            for k, v in rec.items()}


def write_poses(path, records):
    """Write records as JSON (.json) or CSV (anything else)."""
    if path.lower().endswith(".json"):
        with open(path, "w") as f:
            json.dump(records, f, indent=1)
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in records:
            w.writerow({k: ("" if v is None else v) for k, v in rec.items()})
