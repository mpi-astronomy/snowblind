import numpy as np

from snowblind.scene_mask import chisel_mask, clip_mask, remove_stripes


def gaussian_sources(shape, x, y, flux, sigma=1.5):
    yy, xx = np.mgrid[0: shape[0], 0: shape[1]]
    image = np.zeros(shape)
    for xi, yi, fi in zip(x, y, flux):
        image += fi * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return image


def test_chisel_mask_on_pure_noise_masks_little():
    """chisel_mask on a 256 x 256 image of unit noise marks fewer than 2 percent of the pixels as source."""
    rng = np.random.default_rng(1)
    rate = rng.normal(size=(256, 256))

    background = chisel_mask(rate)

    assert (~background).mean() < 0.02


def test_chisel_mask_covers_more_of_the_wings_than_clipping():
    """chisel_mask masks more of a bright source's flux than clip_mask at 2 sigma on the same image."""
    rng = np.random.default_rng(2)
    sources = gaussian_sources((256, 256), [128.0], [128.0], [3000.0])
    rate = sources + rng.normal(size=(256, 256))

    chisel = chisel_mask(rate)
    clip = clip_mask(rate, 2.0)

    assert sources[~chisel].sum() > sources[~clip].sum()
    assert sources[~chisel].sum() / sources.sum() > 0.98


def test_chisel_mask_masks_more_of_a_faint_source_than_clipping():
    """chisel_mask masks at least 10 pixels around a source with a 3-sigma peak, where a 2-sigma clip masks only a few."""
    rng = np.random.default_rng(3)
    sources = gaussian_sources((256, 256), [64.0, 192.0], [64.0, 192.0], [42.0, 42.0])
    rate = sources + rng.normal(size=(256, 256))

    chisel = chisel_mask(rate)
    clip = clip_mask(rate, 2.0)

    box = (slice(56, 73), slice(56, 73))
    assert (~chisel[box]).sum() >= 10
    assert (~chisel[box]).sum() > (~clip[box]).sum()


def test_chisel_mask_masks_non_finite():
    """chisel_mask returns False where the rate is NaN."""
    rng = np.random.default_rng(4)
    rate = rng.normal(size=(128, 128))
    rate[10, 10] = np.nan

    background = chisel_mask(rate)

    assert not background[10, 10]


def test_clip_mask_two_sigma_fraction():
    """clip_mask at 2 sigma on unit noise marks about 2.3 percent of pixels as source."""
    rng = np.random.default_rng(5)
    rate = rng.normal(size=(512, 512))

    background = clip_mask(rate, 2.0)

    assert 0.015 < (~background).mean() < 0.035


def test_chisel_mask_does_not_detect_a_smooth_sky():
    """chisel_mask on unit noise plus a sky sinusoid of amplitude 3 and period 400 rows masks under 3 percent."""
    rng = np.random.default_rng(6)
    yy = np.arange(512)[:, None]
    rate = rng.normal(size=(512, 512)) + 3.0 * np.sin(2 * np.pi * yy / 400.0) + 0.01 * np.arange(512)[None, :]

    background = chisel_mask(rate)

    assert (~background).mean() < 0.03


def test_chisel_mask_finds_sources_on_a_smooth_sky():
    """chisel_mask masks 95 percent of a bright source's flux on top of a sky sinusoid."""
    rng = np.random.default_rng(7)
    yy = np.arange(256)[:, None]
    sources = gaussian_sources((256, 256), [100.0], [150.0], [3000.0])
    rate = sources + rng.normal(size=(256, 256)) + 3.0 * np.sin(2 * np.pi * yy / 200.0)

    background = chisel_mask(rate)

    assert sources[~background].sum() / sources.sum() > 0.95


def test_remove_stripes_takes_out_row_and_column_offsets():
    """remove_stripes on an image of row offsets plus column offsets returns zeros everywhere."""
    rows = np.arange(64, dtype=float)[:, None]
    cols = np.arange(64, dtype=float)[None, :] * 0.5
    image = rows + cols

    out = remove_stripes(image, np.isfinite(image))

    assert np.abs(out).max() < 1e-9


def test_remove_stripes_zeroes_non_finite_pixels():
    """remove_stripes returns 0 at a NaN pixel and finite values elsewhere."""
    image = np.ones((16, 16))
    image[3, 3] = np.nan

    out = remove_stripes(image, np.isfinite(image))

    assert out[3, 3] == 0.0
    assert np.isfinite(out).all()


def test_chisel_mask_is_not_fooled_by_row_banding():
    """chisel_mask on unit noise plus row offsets of amplitude 1 masks under 3 percent."""
    rng = np.random.default_rng(8)
    rate = rng.normal(size=(512, 512)) + rng.normal(scale=1.0, size=(512, 1))

    background = chisel_mask(rate)

    assert (~background).mean() < 0.03
