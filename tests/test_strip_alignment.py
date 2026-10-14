"""Synthetic tests for the De Bruijn 2D fallback cell pairing (no video)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stabilise as st  # noqa: E402
from detect_strip import DeBruijnDetector  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    return DeBruijnDetector(base_radius_mm=14.5, strip_height_mm=6)


@pytest.mark.parametrize("shift", range(16))
def test_decode_start_idx_convention(detector, shift):
    # Sample s shows cell (shift + s): _decode must report start_idx == shift.
    n = detector.n_cells
    bits = [detector.sequence[(shift + s) % n] for s in range(n)]
    start_idx, conf = detector._decode(bits)
    assert conf == 1.0
    assert start_idx == shift


@pytest.mark.parametrize("start_idx", [0, 1, 5, 15])
def test_cells_in_sequence_order(start_idx):
    n = 16
    # point of sample s is tagged with the cell it shows
    samples = [("cell", (start_idx + s) % n) for s in range(n)]
    ordered = st.cells_in_sequence_order(samples, start_idx)
    assert [cell for _, cell in ordered] == list(range(n))


def test_detect_reports_found_without_camera(detector, monkeypatch):
    # Without calibration solvePnP is skipped; a decoded strip must still be
    # reported as found so stabilise.py can use the 2D path.
    ellipse = ((320.0, 240.0), (200.0, 120.0), 0.0)
    monkeypatch.setattr(detector, "_find_strip_ellipse", lambda g, d=None: ellipse)
    monkeypatch.setattr(detector, "_sample_cells",
                        lambda g, e, d=None: (list(detector.sequence),
                                              [(0.0, 0.0)] * detector.n_cells))
    result = detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert result["found"] is True
    assert result["rvec"] is None
    assert result["start_idx"] == 0


@pytest.mark.parametrize("ref_start", [0, 3, 9])
def test_ellipse_homography_with_flipped_parametrisation(ref_start):
    # The same physical ring, translated by (12, -7) px, but fitEllipse
    # reports it with axes swapped and angle +90: its parametrisation origin
    # moves a quarter turn, so the decoder reports start_idx + n/4.
    n = 16
    ref = ((320.0, 240.0), (200.0, 120.0), 0.0)
    cur = ((332.0, 233.0), (120.0, 200.0), 90.0)
    cur_start = (ref_start + n // 4) % n

    H = st.ellipse_cell_homography(ref, ref_start, cur, cur_start, n)
    expected = np.array([[1, 0, -12], [0, 1, 7], [0, 0, 1]], float)
    assert np.allclose(H / H[2, 2], expected, atol=1e-3)


def test_ref_start_idx_matters():
    # Regression for the old code, which re-indexed only the current frame:
    # with a non-zero reference start_idx the pairing (and warp) was wrong.
    n = 16
    ellipse = ((320.0, 240.0), (200.0, 120.0), 30.0)
    H = st.ellipse_cell_homography(ellipse, 5, ellipse, 5, n)
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-6)

    ref_pts = st._ellipse_sample_pts(ellipse, n)
    cur = st._ellipse_sample_pts(ellipse, n)
    old_pairing = cur[5:] + cur[:5]
    H_old = st.compute_warp_homography(ref_pts, old_pairing)
    assert H_old is None or not np.allclose(H_old / H_old[2, 2], np.eye(3), atol=1e-2)
