"""Tests for pose_export.py (no video needed)."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pose_export as pe  # noqa: E402
import stabilise_aruco as sa  # noqa: E402

FACING = np.diag([1.0, -1.0, -1.0])


def rvec_of(R):
    return cv2.Rodrigues(R)[0]


@pytest.mark.parametrize("yaw", [0, 15, 90, -120, 179])
def test_relative_yaw_about_handle_axis(yaw):
    tilt = rvec_of(FACING @ cv2.Rodrigues(np.array([0.3, 0.0, 0.1]))[0])
    R_ref, _ = cv2.Rodrigues(tilt)
    R_cur = R_ref @ sa.marker_offset_rotation(yaw)   # Ry(yaw) in handle frame
    assert pe.relative_yaw_deg(tilt, rvec_of(R_cur)) == pytest.approx(yaw, abs=1e-6)


def test_quaternion_matches_rotation():
    rng = np.random.default_rng(3)
    for _ in range(50):
        r = rng.uniform(-3, 3, 3)
        w, x, y, z = pe.quat_from_rvec(r)
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        assert np.allclose(R, cv2.Rodrigues(r)[0], atol=1e-9)
    assert np.allclose(pe.quat_from_rvec(np.zeros(3)), [1, 0, 0, 0])


def _records():
    ref = rvec_of(FACING)
    cur = rvec_of(FACING @ sa.marker_offset_rotation(30))
    return [pe.pose_record(0, 1, ref, np.array([1.0, 2.0, 250.0]), ref),
            pe.pose_record(1),
            pe.pose_record(2, 2, cur, np.array([1.0, 2.0, 251.0]), ref)]


def test_records():
    recs = _records()
    assert list(recs[0]) == pe.FIELDS
    assert recs[0]["detected"] == 1 and recs[0]["yaw_deg"] == pytest.approx(0, abs=1e-9)
    assert recs[1]["detected"] == 0 and recs[1]["rx"] is None
    assert recs[2]["marker_id"] == 2 and recs[2]["yaw_deg"] == pytest.approx(30)


def test_write_csv(tmp_path):
    p = tmp_path / "poses.csv"
    pe.write_poses(str(p), _records())
    rows = list(csv.DictReader(p.open()))
    assert [r["frame"] for r in rows] == ["0", "1", "2"]
    assert rows[1]["rx"] == "" and rows[1]["detected"] == "0"
    assert float(rows[2]["yaw_deg"]) == pytest.approx(30)


def test_write_json(tmp_path):
    p = tmp_path / "poses.json"
    pe.write_poses(str(p), _records())
    data = json.loads(p.read_text())
    assert len(data) == 3 and data[1]["tz"] is None
    assert data[2]["tz"] == pytest.approx(251.0)
