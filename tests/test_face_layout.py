"""Tests for configurable N-face marker layouts (no video needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stabilise_aruco as sa  # noqa: E402

CAM = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
DIST = np.zeros((4, 1))
MARKER = 20.0
FACING = np.diag([1.0, -1.0, -1.0])   # marker upright, facing the camera


def test_default_layout_is_unchanged():
    assert sa.MARKER_IDS == [0, 1, 2, 3]
    assert sa.FACE_ANGLES == [0, 90, 180, 270]


def test_hexagonal_layout():
    ids, angles = sa.face_layout(6, first_id=10)
    assert ids == [10, 11, 12, 13, 14, 15]
    assert angles == pytest.approx([0, 60, 120, 180, 240, 300])


def test_invalid_layout():
    with pytest.raises(ValueError):
        sa.face_layout(0)


@pytest.mark.parametrize("n_faces", [3, 4, 6, 8])
def test_every_face_recovers_the_same_handle_rotation(n_faces):
    ids, angles = sa.face_layout(n_faces)
    h = MARKER / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    for mid, angle in zip(ids, angles):
        # Handle yawed so this face is 15 degrees off the camera axis
        R_h = FACING @ sa.marker_offset_rotation(-(angle + 15))
        R_m = R_h @ sa.marker_offset_rotation(angle)
        rv_m, _ = cv2.Rodrigues(R_m)
        img, _ = cv2.projectPoints(obj, rv_m, np.array([[0.0], [0.0], [250.0]]),
                                   CAM, DIST)
        rv, _ = sa.pose_from_marker([img.reshape(1, 4, 2).astype(np.float32)],
                                    mid, MARKER, CAM, DIST,
                                    marker_ids=ids, face_angles=angles)
        R, _ = cv2.Rodrigues(rv)
        assert np.allclose(R, R_h, atol=3e-2), (n_faces, mid)


def test_unknown_marker_id_is_ignored():
    ids, angles = sa.face_layout(6)
    corners = [np.zeros((1, 4, 2), np.float32)]
    assert sa.pose_from_marker(corners, 42, MARKER, CAM, DIST,
                                 marker_ids=ids, face_angles=angles) == (None, None)


def test_handle_radius_estimate_is_the_prism_apothem():
    # s / (2 tan(pi/N)): square -> s/2, hexagon -> s*sqrt(3)/2
    assert sa.estimate_handle_radius(MARKER, 4) == pytest.approx(MARKER / 2)
    assert sa.estimate_handle_radius(MARKER, 6) == pytest.approx(MARKER * np.sqrt(3) / 2)
    with pytest.raises(ValueError):
        sa.estimate_handle_radius(MARKER, 2)


@pytest.mark.parametrize("n_faces", [4, 6])
def test_every_face_recovers_the_same_handle_position(n_faces):
    # Hexagon/square prism with the estimated radius: every face must give
    # the handle axis position, so switching markers does not jump.
    ids, angles = sa.face_layout(n_faces)
    r = sa.estimate_handle_radius(MARKER, n_faces)
    h = MARKER / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    t_h = np.array([[5.0], [-3.0], [250.0]])
    for mid, angle in zip(ids, angles):
        R_h = FACING @ sa.marker_offset_rotation(-(angle + 15))
        R_face = sa.marker_offset_rotation(angle)
        R_m = R_h @ R_face
        t_m = t_h + R_h @ R_face @ np.array([[0.0], [0.0], [r]])
        rv_m, _ = cv2.Rodrigues(R_m)
        img, _ = cv2.projectPoints(obj, rv_m, t_m, CAM, DIST)
        _, tv = sa.pose_from_marker([img.reshape(1, 4, 2).astype(np.float32)],
                                    mid, MARKER, CAM, DIST, r,
                                    marker_ids=ids, face_angles=angles)
        assert np.allclose(tv, t_h, atol=1.0), (n_faces, mid, tv.ravel())
