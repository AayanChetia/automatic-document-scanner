"""Tests for the enhancement modes."""

import numpy as np
import pytest

from scanner.enhance import (MODES, enhance, black_and_white, otsu,
                             magic_color, to_gray)


@pytest.fixture
def sample():
    """A small colour image with a light 'page' and some dark 'text'."""
    img = np.full((120, 100, 3), 220, np.uint8)
    img[30:40, 20:80] = 30     # a dark bar (text-like)
    return img


def test_all_modes_run(sample):
    """Every declared mode must return a non-empty uint8 image."""
    for mode in MODES:
        out = enhance(sample, mode=mode)
        assert out.dtype == np.uint8
        assert out.shape[0] == sample.shape[0] and out.shape[1] == sample.shape[1]


def test_unknown_mode_raises(sample):
    with pytest.raises(ValueError):
        enhance(sample, mode="sepia")


def test_bw_is_binary(sample):
    """Adaptive B&W output must contain only 0 and 255."""
    out = black_and_white(sample)
    assert out.ndim == 2
    assert set(np.unique(out)).issubset({0, 255})


def test_otsu_is_binary(sample):
    out = otsu(sample)
    assert set(np.unique(out)).issubset({0, 255})


def test_gray_is_single_channel(sample):
    out = to_gray(sample)
    assert out.ndim == 2


def test_magic_keeps_three_channels(sample):
    """Magic colour must stay a 3-channel colour image."""
    out = magic_color(sample)
    assert out.ndim == 3 and out.shape[2] == 3


def test_magic_increases_contrast(sample):
    """CLAHE should not collapse the image to a constant (std stays > 0)."""
    out = magic_color(sample)
    assert out.std() > 0
