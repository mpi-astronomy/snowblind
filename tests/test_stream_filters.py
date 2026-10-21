import numpy as np
import pytest

from snowblind import readout
from snowblind.stream_filters import boxcar_noise, row_median_noise

NROWS, WIDTH = 256, 512


def slow_pattern(nrows, width, direction, period=20000.0, amplitude=1.0):
    """Channel image carrying a sinusoid in clock time with the given period."""
    t = readout.clock_index(nrows, width, direction)
    return (amplitude * np.sin(2 * np.pi * t / period)).astype(np.float32)


def test_boxcar_recovers_slow_pattern_outside_gap():
    """boxcar_noise with a 2048-clock window on a 20000-clock sinusoid plus unit white noise recovers it to 0.05 rms."""
    rng = np.random.default_rng(1)
    pattern = slow_pattern(NROWS, WIDTH, 1)
    image = pattern + rng.normal(size=pattern.shape).astype(np.float32)
    mask = np.ones(pattern.shape, dtype=bool)

    noise = boxcar_noise(image, mask, direction=1, window_length=2048, min_density=0.2)

    inner = slice(8, NROWS - 8)
    assert np.sqrt(np.mean((noise[inner] - pattern[inner]) ** 2)) < 0.05


def test_boxcar_interpolates_across_masked_gap():
    """boxcar_noise fills a 40-pixel masked gap in every row with the slow pattern to within 0.1 rms."""
    rng = np.random.default_rng(2)
    pattern = slow_pattern(NROWS, WIDTH, 1)
    image = pattern + rng.normal(size=pattern.shape).astype(np.float32)
    mask = np.ones(pattern.shape, dtype=bool)
    mask[:, 200:240] = False
    image[~mask] = 1e4

    noise = boxcar_noise(image, mask, direction=1, window_length=2048, min_density=0.2)

    gap = noise[8:-8, 200:240] - pattern[8:-8, 200:240]
    assert np.sqrt(np.mean(gap**2)) < 0.1


def test_boxcar_reverse_direction_matches_pattern():
    """boxcar_noise with direction -1 recovers a pattern laid out in reversed clock order."""
    rng = np.random.default_rng(3)
    pattern = slow_pattern(NROWS, WIDTH, -1)
    image = pattern + rng.normal(size=pattern.shape).astype(np.float32)
    mask = np.ones(pattern.shape, dtype=bool)

    noise = boxcar_noise(image, mask, direction=-1, window_length=2048, min_density=0.2)

    inner = slice(8, NROWS - 8)
    assert np.sqrt(np.mean((noise[inner] - pattern[inner]) ** 2)) < 0.05


def test_boxcar_falls_back_to_row_median_in_sparse_region():
    """boxcar_noise in rows with 0.2 percent usable pixels is within 0.1 of the row-median estimate where a row has one."""
    rng = np.random.default_rng(4)
    image = rng.normal(size=(NROWS, WIDTH)).astype(np.float32) + 5.0
    mask = rng.random(image.shape) < 0.002
    mask[:100] = True

    noise = boxcar_noise(image, mask, direction=1, window_length=2048, min_density=0.2)

    fallback = row_median_noise(image, mask)
    rows = np.flatnonzero(mask[150:].any(axis=1)) + 150
    np.testing.assert_allclose(noise[rows], fallback[rows], atol=0.1)


def test_boxcar_constant_offset_is_recovered():
    """boxcar_noise on a constant image with a random 50 percent mask returns the constant everywhere."""
    rng = np.random.default_rng(5)
    image = np.full((NROWS, WIDTH), 3.0, dtype=np.float32)
    mask = rng.random(image.shape) < 0.5

    noise = boxcar_noise(image, mask, direction=1, window_length=1024, min_density=0.2)

    np.testing.assert_allclose(noise, 3.0, atol=1e-4)


def test_boxcar_window_longer_than_stream_is_clamped():
    """boxcar_noise with a window longer than the stream warns and still returns the constant for a constant image."""
    image = np.full((8, 16), 3.0, dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)

    with pytest.warns(UserWarning, match="exceeds the stream length"):
        noise = boxcar_noise(image, mask, direction=1, window_length=4096, min_density=0.2)

    np.testing.assert_allclose(noise, 3.0, atol=1e-4)
