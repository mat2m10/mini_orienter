"""Synthetic tests for stabilise_aruco.pose_from_marker (no video needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stabilise_aruco as sa  # noqa: E402

CAM = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
DIST = np.zeros((4, 1))
MARKER = 17.0
RADIUS = 2 * MARKER / np.pi
# SOLVEPNP_IPPE_SQUARE is not exact even on noise-free corners (up to
# ~0.3 px reprojection error, ~1.3 deg on a marker seen at 45 degrees), so
# rotations are compared to ~1.7 deg and translations to 0.5 mm. That is
# still far below the radius*sqrt(2) (~15 mm) jump this fix removes.
R_TOL = 3e-2
T_TOL = 0.5
# Marker facing the camera, upright: x right, y up, z towards camera.
FACING = np.diag([1.0, -1.0, -1.0])


def rot_y(deg):
    return sa.marker_offset_rotation(deg)


def project_marker(R_handle, t_handle, face_angle, radius=RADIUS):
    """Image corners of the marker on `face_angle` for a given handle pose."""
    R_face = rot_y(face_angle)
    R_m = R_handle @ R_face
    t_m = t_handle + R_handle @ R_face @ np.array([0, 0, radius])
    h = MARKER / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    rv, _ = cv2.Rodrigues(R_m)
    img, _ = cv2.projectPoints(obj, rv, t_m.reshape(3, 1), CAM, DIST)
    return img.reshape(1, 4, 2).astype(np.float32)


def handle_pose_from(face_angle, view_angle, radius=RADIUS):
    """Handle pose seen with `view_angle` (handle Y-axis yaw) and project
    the marker on `face_angle`; return the recovered handle pose."""
    R_h = FACING @ rot_y(-view_angle)
    t_h = np.array([4.0, -6.0, 250.0])
    mid = sa.MARKER_IDS[sa.FACE_ANGLES.index(face_angle)]
    corners = project_marker(R_h, t_h, face_angle)
    rv, tv = sa.pose_from_marker([corners], mid, MARKER, CAM, DIST, radius)
    R, _ = cv2.Rodrigues(rv)
    return R_h, t_h, R, tv.ravel()


@pytest.mark.parametrize("face", sa.FACE_ANGLES)
def test_recovers_handle_pose_for_every_face(face):
    # View the handle so that this face is 20 degrees off-axis
    R_h, t_h, R, t = handle_pose_from(face, face + 20)
    assert np.allclose(R, R_h, atol=R_TOL)
    assert np.allclose(t, t_h, atol=T_TOL)


@pytest.mark.parametrize("face_a,face_b", [(0, 90), (90, 180), (180, 270), (270, 0)])
def test_adjacent_markers_agree(face_a, face_b):
    # Handle yawed so that both faces are visible at 45 degrees:
    # switching from one marker ID to the other must not move the pose.
    view = face_a + 45
    _, _, Ra, ta = handle_pose_from(face_a, view)
    _, _, Rb, tb = handle_pose_from(face_b, view)
    assert np.allclose(Ra, Rb, atol=R_TOL)
    assert np.linalg.norm(ta - tb) < T_TOL


def test_zero_radius_reproduces_the_old_jump():
    # Documents the bug fixed by --handle-radius: without the offset the
    # two markers disagree by about radius*sqrt(2).
    R_h = FACING @ rot_y(-45)
    t_h = np.array([0.0, 0.0, 250.0])
    poses = []
    for face in (0, 90):
        corners = project_marker(R_h, t_h, face)
        rv, tv = sa.pose_from_marker([corners], sa.MARKER_IDS[face // 90],
                                     MARKER, CAM, DIST, 0.0)
        poses.append(tv.ravel())
    jump = np.linalg.norm(poses[0] - poses[1])
    assert jump == pytest.approx(RADIUS * np.sqrt(2), rel=0.01)
