import numpy as np

from astropy.stats import mad_std

from snowblind.scene_mask import chisel_mask, remove_stripes, saturated_stars, spike_mask


def sigma_clip_mask(rate, n_sigma):
    """The jwst step's mask for comparison: background where the rate is below the median plus n_sigma robust sigmas."""
    finite = np.isfinite(rate)
    level = np.median(rate[finite])
    return finite & (rate - level < n_sigma * mad_std(rate[finite]))


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
    """chisel_mask masks more of a bright source's flux than a 2-sigma clip on the same image."""
    rng = np.random.default_rng(2)
    sources = gaussian_sources((256, 256), [128.0], [128.0], [3000.0])
    rate = sources + rng.normal(size=(256, 256))

    chisel = chisel_mask(rate)
    clip = sigma_clip_mask(rate, 2.0)

    assert sources[~chisel].sum() > sources[~clip].sum()
    assert sources[~chisel].sum() / sources.sum() > 0.98


def test_chisel_mask_masks_more_of_a_faint_source_than_clipping():
    """chisel_mask masks at least 10 pixels around a source with a 3-sigma peak, where a 2-sigma clip masks only a few."""
    rng = np.random.default_rng(3)
    sources = gaussian_sources((256, 256), [64.0, 192.0], [64.0, 192.0], [42.0, 42.0])
    rate = sources + rng.normal(size=(256, 256))

    chisel = chisel_mask(rate)
    clip = sigma_clip_mask(rate, 2.0)

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


def test_chisel_mask_coarse_tier_masks_a_smooth_halo():
    """chisel_mask on noise plus a wide Gaussian halo of 0.3 sigma peak masks most of the halo core only with the coarse tier."""
    rng = np.random.default_rng(31)
    rate = rng.normal(1.0, 0.1, (512, 512))
    yy, xx = np.mgrid[0:512, 0:512]
    halo = 0.03 * np.exp(-((yy - 256) ** 2 + (xx - 256) ** 2) / (2 * 60.0 ** 2))
    rate = rate + halo

    with_tier = chisel_mask(rate)
    without_tier = chisel_mask(rate, coarse_fwhm=None)

    core = (np.hypot(yy - 256, xx - 256) < 60)
    assert (~with_tier[core]).mean() > 0.5
    assert (~without_tier[core]).mean() < 0.2


def star_shaped_mask(shape, cy, cx):
    """A scene mask (True where background) with a star-shaped hole: a 60 px core and 300 x 20 px arms."""
    mask = np.ones(shape, dtype=bool)
    mask[cy - 30:cy + 30, cx - 30:cx + 30] = False
    mask[cy - 150:cy + 150, cx - 10:cx + 10] = False
    mask[cy - 10:cy + 10, cx - 150:cx + 150] = False
    return mask


def test_saturated_stars_takes_spike_shaped_regions_with_saturated_pixels():
    """saturated_stars on a mask with a star-shaped hole around NaNs and a filled square around NaNs lists the star only."""
    finite = np.ones((512, 512), dtype=bool)
    finite[250:263, 250:263] = False  # the star's saturated core
    finite[100:105, 100:105] = False  # saturated pixels in a galaxy-like region
    mask = star_shaped_mask((512, 512), 256, 256)
    mask[40:170, 40:170] = False

    stars = saturated_stars(mask, finite)

    assert len(stars) == 1
    assert abs(stars[0][0] - 256) < 1 and abs(stars[0][1] - 256) < 1


def test_spike_mask_follows_a_spike_beyond_the_core_mask():
    """spike_mask on a saturated star with a vertical spike of 0.1 sky masks the spike where the star mask stopped."""
    rng = np.random.default_rng(41)
    rate = rng.normal(1.0, 0.05, (512, 512))
    rate[20:406, 254:259] += 0.1  # a vertical spike through the star
    rate[250:263, 250:263] = np.nan  # the saturated core
    core = star_shaped_mask((512, 512), 256, 256)  # the arm ends at row 106

    masked, nstars = spike_mask(rate, core)

    assert nstars == 1
    assert (~masked[64:106, 256]).mean() > 0.9
    assert masked[64:106, 300].mean() > 0.95


def test_spike_mask_needs_a_saturated_core():
    """spike_mask on a bright but unsaturated source with a spike treats no star and returns the mask unchanged."""
    rng = np.random.default_rng(42)
    rate = rng.normal(1.0, 0.05, (512, 512))
    rate[256, 256] += 50.0
    rate[20:406, 254:259] += 0.1
    core = star_shaped_mask((512, 512), 256, 256)

    masked, nstars = spike_mask(rate, core)

    assert nstars == 0
    np.testing.assert_array_equal(masked, core & np.isfinite(rate))


def test_saturated_stars_with_zero_border_still_finds_the_star():
    """saturated_stars with border=0 on a star-shaped mask around NaNs lists the star, instead of clearing the frame."""
    finite = np.ones((512, 512), dtype=bool)
    finite[250:263, 250:263] = False
    mask = star_shaped_mask((512, 512), 256, 256)

    stars = saturated_stars(mask, finite, border=0)

    assert len(stars) == 1


def test_spike_mask_follows_a_row_spike_too():
    """spike_mask on a saturated star with a horizontal spike of 0.1 sky masks the spike where the star mask stopped."""
    rng = np.random.default_rng(43)
    rate = rng.normal(1.0, 0.05, (512, 512))
    rate[254:259, 20:406] += 0.1  # a horizontal spike through the star
    rate[250:263, 250:263] = np.nan  # the saturated core
    core = star_shaped_mask((512, 512), 256, 256)  # the arm ends at column 106

    masked, nstars = spike_mask(rate, core)

    assert nstars == 1
    assert (~masked[256, 64:106]).mean() > 0.9
    assert masked[300, 64:106].mean() > 0.95


def test_chisel_mask_without_erosion_or_opening_masks_only_the_source():
    """chisel_mask with erode=0, opening=0 and a 0.99 quantile on sky with one bright square masks under 5 percent."""
    rng = np.random.default_rng(6)
    rate = rng.normal(1.0, 0.05, (256, 256))
    rate[100:120, 100:120] += 5.0

    mask = chisel_mask(rate, qthresh=0.99, erode=0, opening=0, coarse_fwhm=None)

    assert (~mask).mean() < 0.05
    assert not mask[105:115, 105:115].any()
