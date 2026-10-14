"""Geometry tests for stabilise_aruco.py (no video needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stabilise_aruco as sa  # noqa: E402

CAM = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
DIST = np.zeros((4, 1))


@pytest.mark.parametrize("angle", sa.FACE_ANGLES + [45, -30])
def test_marker_offset_rotation_is_a_rotation(angle):
    R = sa.marker_offset_rotation(angle)
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.linalg.det(R) == pytest.approx(1.0)
    # Rotation about the handle (Y) axis keeps Y fixed
    assert np.allclose(R @ [0, 1, 0], [0, 1, 0])


def test_marker_offset_rotation_quarter_turns():
    assert np.allclose(sa.marker_offset_rotation(0), np.eye(3))
    assert np.allclose(sa.marker_offset_rotation(90) @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(sa.marker_offset_rotation(180) @ [0, 0, 1], [0, 0, -1])
    assert np.allclose(sa.marker_offset_rotation(360), np.eye(3))
    assert np.allclose(sa.marker_offset_rotation(90) @ sa.marker_offset_rotation(180),
                       sa.marker_offset_rotation(270))


def test_face_layout_is_consistent():
    assert len(sa.MARKER_IDS) == len(sa.FACE_ANGLES)
    assert len(set(sa.MARKER_IDS)) == len(sa.MARKER_IDS)


def _pose(rx, ry, rz, t):
    rvec = np.deg2rad(np.array([[rx], [ry], [rz]], float))
    return rvec, np.array(t, float).reshape(3, 1)


def test_identical_poses_give_identity_homography():
    rv, tv = _pose(170, 10, 5, [3, -2, 250])
    H = sa.poses_to_homography(rv, tv, rv, tv, CAM, DIST, 20)
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-4)


def test_homography_round_trips_points_on_the_handle_plane():
    ref = _pose(180, 0, 0, [0, 0, 250])
    cur = _pose(165, 20, 10, [12, -8, 280])
    H = sa.poses_to_homography(*ref, *cur, CAM, DIST, 20)
    assert H is not None

    # Any point on the handle's z=0 plane, seen in the current frame, must
    # land where the reference pose would have imaged it.
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-30, 30, (20, 2)), np.zeros(20)]
    img_cur, _ = cv2.projectPoints(pts, *cur, CAM, DIST)
    img_ref, _ = cv2.projectPoints(pts, *ref, CAM, DIST)
    warped = cv2.perspectiveTransform(img_cur.astype(np.float32), H)
    assert np.abs(warped - img_ref).max() < 0.05


def test_default_camera_is_centred_pinhole():
    cam, dist = sa.default_camera(640, 480)
    assert cam.shape == (3, 3)
    assert cam[0, 0] == cam[1, 1] == 640
    assert (cam[0, 2], cam[1, 2]) == (320, 240)
    assert not dist.any()
