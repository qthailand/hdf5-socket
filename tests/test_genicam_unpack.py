"""GenICam Mono12Packed unpack: same values as harvesters' PFNC expansion (needs harvesters, no camera)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip('harvesters')
from harvesters.util.pfnc import Dictionary  # noqa: E402
from modules.camera import genicam  # noqa: E402


def _packed(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w * 3 // 2), dtype=np.uint8)


@pytest.mark.parametrize('shift', [4, 0])
def test_unpack_matches_harvesters_expansion(shift):
    src = _packed(64, 1280)
    reference = Dictionary.get_proxy(symbolic='Mono12Packed').expand(src.ravel().copy()).reshape(64, 1280) << shift
    out = np.empty((64, 1280), np.uint16)
    genicam.unpack_mono12packed(src, out, shift)
    assert np.array_equal(out, reference)
    fallback = np.empty_like(out)
    genicam._unpack_mono12packed_numpy(src, fallback, shift)
    assert np.array_equal(fallback, reference)


def test_unpack_reads_rows_with_padding():
    src = _packed(8, 16, seed=1)
    padded = np.concatenate([src, np.full((8, 5), 0xAB, np.uint8)], axis=1)   # padding_x = 5 bytes
    out, reference = np.empty((8, 16), np.uint16), np.empty((8, 16), np.uint16)
    genicam.unpack_mono12packed(padded, out, 4)
    genicam.unpack_mono12packed(src, reference, 4)
    assert np.array_equal(out, reference)
