"""Synthetic tests for pose_smoothing.py (no video needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_smoothing as ps  # noqa: E402


def rvec_y(deg):
    return np.array([[0.0], [np.deg2rad(deg)], [0.0]])


def angle_between(rv_a, rv_b):
    Ra, _ = cv2.Rodrigues(rv_a)
    Rb, _ = cv2.Rodrigues(rv_b)
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_quaternion_round_trip_random_rotations():
    rng = np.random.default_rng(1)
    for _ in range(200):
        R, _ = cv2.Rodrigues(rng.uniform(-np.pi, np.pi, 3))
        assert np.allclose(ps.matrix_from_quat(ps.quat_from_matrix(R)), R, atol=1e-9)
    # 180-degree rotations exercise the non-trace branches
    for axis in np.eye(3):
        R, _ = cv2.Rodrigues(axis * np.pi)
        assert np.allclose(ps.matrix_from_quat(ps.quat_from_matrix(R)), R, atol=1e-9)


def test_slerp_endpoints_midpoint_and_sign():
    q0 = ps.quat_from_matrix(np.eye(3))
    q1 = ps.quat_from_matrix(cv2.Rodrigues(rvec_y(80))[0])
    assert np.allclose(ps.slerp(q0, q1, 0), q0)
    assert np.allclose(ps.slerp(q0, q1, 1), q1)
    mid = ps.matrix_from_quat(ps.slerp(q0, q1, 0.5))
    assert np.allclose(mid, cv2.Rodrigues(rvec_y(40))[0], atol=1e-9)
    # -q1 is the same rotation: must still go the short way
    mid2 = ps.matrix_from_quat(ps.slerp(q0, -q1, 0.5))
    assert np.allclose(mid2, mid, atol=1e-9)


def test_alpha_one_is_passthrough():
    s = ps.PoseSmoother(alpha=1.0)
    for deg in (0, 10, 35, -20):
        tv = np.array([[deg], [1.0], [250.0]])
        rv, t = s.update(rvec_y(deg), tv)
        assert angle_between(rv, rvec_y(deg)) < 1e-6
        assert np.allclose(t, tv)


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(2)
    s = ps.PoseSmoother(alpha=0.3)
    raw_err, smooth_err = [], []
    for _ in range(300):
        noisy_r = rvec_y(20) + rng.normal(0, np.deg2rad(1.0), (3, 1))
        noisy_t = np.array([[0.0], [0.0], [250.0]]) + rng.normal(0, 1.0, (3, 1))
        rv, tv = s.update(noisy_r, noisy_t)
        raw_err.append(angle_between(noisy_r, rvec_y(20)))
        smooth_err.append(angle_between(rv, rvec_y(20)))
    assert np.mean(smooth_err[50:]) < 0.6 * np.mean(raw_err[50:])


def test_dropout_extrapolates_then_holds():
    s = ps.PoseSmoother(alpha=1.0, max_extrapolate=3)
    for i in range(5):                       # 2 deg/frame, 1 mm/frame
        s.update(rvec_y(2 * i), np.array([[i], [0.0], [250.0]]))
    predicted = [s.predict() for _ in range(6)]
    angles = [np.degrees(p[0][1, 0]) for p in predicted]
    xs = [p[1][0, 0] for p in predicted]
    assert angles == pytest.approx([10, 12, 14, 14, 14, 14], abs=1e-6)
    assert xs == pytest.approx([5, 6, 7, 7, 7, 7], abs=1e-9)
    assert s.missed == 6


def test_hold_only_when_extrapolation_disabled():
    s = ps.PoseSmoother(alpha=1.0, max_extrapolate=0)
    assert s.predict() is None               # nothing seen yet
    s.update(rvec_y(0), np.zeros((3, 1)))
    s.update(rvec_y(5), np.zeros((3, 1)))
    rv, _ = s.predict()
    assert angle_between(rv, rvec_y(5)) < 1e-6


def test_no_velocity_jump_after_a_gap():
    # The first measurement after a dropout must not set a velocity from
    # the (gap-spanning) difference.
    s = ps.PoseSmoother(alpha=1.0, max_extrapolate=2)
    s.update(rvec_y(0), np.zeros((3, 1)))
    s.predict()
    s.update(rvec_y(30), np.zeros((3, 1)))
    rv, _ = s.predict()
    assert angle_between(rv, rvec_y(30)) < 1e-6


def test_invalid_alpha():
    with pytest.raises(ValueError):
        ps.PoseSmoother(alpha=0)
