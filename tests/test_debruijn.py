"""De Bruijn sequence and lookup tests (no video needed)."""
import sys
from itertools import product
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detect_strip import DeBruijnDetector, build_lookup, de_bruijn  # noqa: E402


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_de_bruijn_is_binary_with_length_2_pow_n(n):
    seq = de_bruijn(n)
    assert len(seq) == 2 ** n
    assert set(seq) <= {0, 1}


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_every_cyclic_window_is_unique_and_all_appear(n):
    seq = de_bruijn(n)
    ring = seq + seq
    windows = [tuple(ring[i:i + n]) for i in range(len(seq))]
    assert len(set(windows)) == len(windows)
    assert set(windows) == set(product((0, 1), repeat=n))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_build_lookup_maps_each_window_to_its_start(n):
    seq = de_bruijn(n)
    lookup = build_lookup(seq, n)
    assert len(lookup) == len(seq)
    ring = seq + seq
    for i in range(len(seq)):
        assert lookup[tuple(ring[i:i + n])] == i


@pytest.mark.parametrize("rotation", [0, 1, 7, 15, 20])
def test_detector_lookup_unique_for_any_rotation(rotation):
    det = DeBruijnDetector(base_radius_mm=14.5, strip_height_mm=6,
                           sequence_rotation=rotation)
    assert det.n_cells == 16
    assert len(det.lookup) == 16


@pytest.mark.parametrize("shift", [0, 3, 11])
def test_decode_recovers_shift_and_tolerates_one_bad_cell(shift):
    det = DeBruijnDetector(base_radius_mm=14.5, strip_height_mm=6)
    n = det.n_cells
    bits = [det.sequence[(shift + s) % n] for s in range(n)]
    assert det._decode(bits) == (shift, 1.0)

    noisy = list(bits)
    noisy[(shift + 8) % n] ^= 1  # flip one cell away from the start
    start, conf = det._decode(noisy)
    assert start == shift
    assert conf == pytest.approx(15 / 16)
