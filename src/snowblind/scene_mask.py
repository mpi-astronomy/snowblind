"""Scene masks for the stream cleaner.

The jwst ``clean_flicker_noise`` step masks sources by sigma-clipping the
draft rate image pixel by pixel: a pixel is a source if its rate is more
than ``n_sigma`` above the background.  That stops where a source's wings
fall below the clipping level, and the wings of bright sources extend far
beyond it.  `chisel_mask` follows the outline of NoiseChisel (Akhlaghi and
Ichikawa 2015, Gnuastro): detect signal in a convolved image at a low
quantile threshold, remove the detections that noise alone produces by
erosion and opening and by a signal-to-noise test against pseudo-detections
in the noise, then grow what survives into the faint wings.
"""
import numpy as np
from scipy import ndimage

__all__ = ["chisel_mask", "clip_mask", "tiled_sky", "remove_stripes"]


def _robust_sigma(values):
    values = values[np.isfinite(values)]
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def clip_mask(rate, n_sigma=2.0):
    """Background mask by sigma clipping, as the jwst step does: True where background."""
    good = np.isfinite(rate)
    level = np.median(rate[good])
    sigma = _robust_sigma(rate[good])
    return good & (rate - level < n_sigma * sigma)


def _candidates(convolved, threshold, erode, opening):
    binary = convolved > threshold
    if erode:
        binary = ndimage.binary_erosion(binary, ndimage.generate_binary_structure(2, 1), iterations=erode)
    if opening:
        binary = ndimage.binary_opening(binary, ndimage.generate_binary_structure(2, 2), iterations=opening)
    return binary


def _blob_snr(image, labels, nlabels, sigma):
    """Signal-to-noise of each labelled blob: summed signal over sigma times the square root of its area."""
    if nlabels == 0:
        return np.array([])
    index = np.arange(1, nlabels + 1)
    total = ndimage.sum(image, labels, index)
    area = ndimage.sum(np.ones_like(image), labels, index)
    return np.asarray(total) / (sigma * np.sqrt(np.asarray(area)))


def chisel_mask(
    rate,
    kernel_fwhm=2.0,
    qthresh=0.5,
    erode=2,
    opening=1,
    snquant=0.99,
    detgrowquant=0.9,
    max_hole=100,
    background=None,
    sky_box=32,
    passes=2,
    max_grow=16,
):
    """Background mask (True where background) that follows source wings.

    Parameters
    ----------
    rate : ndarray, shape (nrows, ncols)
        Draft rate image.  Non-finite pixels are treated as background-less
        and masked.
    kernel_fwhm : float
        FWHM in pixels of the Gaussian the image is convolved with.
    qthresh : float
        Quantile of the sky-subtracted convolved image above which a pixel
        is a candidate.  NoiseChisel's default is 0.3; here 0.5, because at
        0.3 the convolved noise of these undersampled images percolates
        into clusters of thousands of pixels that survive the erosion and
        pass any signal-to-noise cut, and the growth step then spreads
        from them (18 percent of a pure-noise image masked, against 1.5
        percent at 0.5, with a 3-sigma source still caught).
    erode, opening : int
        Iterations of binary erosion (4-connected) and opening (8-connected)
        applied to the candidates to remove what noise alone produces.
    snquant : float
        Quantile of the signal-to-noise distribution of pseudo-detections
        (the same candidates found in the sign-flipped image, which holds
        noise only) that a detection must exceed.
    detgrowquant : float
        Quantile of the sky-subtracted convolved image into which the
        detections are grown, 8-connected, until nothing changes.
    max_hole : int
        Holes inside grown detections up to this many pixels are filled.
    max_grow : int
        Safety cap on the growth distance in pixels.  The growth is data
        driven, one pixel per iteration into connected pixels above
        ``detgrowquant``, and stops by itself when none is left; on real
        images it takes a few iterations.
    background : ndarray or float or None
        Background to subtract first.  Default: a sky model fitted in
        ``sky_box`` tiles (median with sigma clipping, as NoiseChisel's
        tiled sky), refitted with the detections of the previous pass masked
        for ``passes`` passes in total.  A smooth sky must not be detected.
    sky_box : int
        Tile size in pixels of the default sky model.
    passes : int
        Detection passes; each refits the sky with the last mask applied.

    Returns
    -------
    ndarray of bool
        True for background pixels.
    """
    rate = np.asarray(rate, dtype=np.float64)
    finite = np.isfinite(rate)
    if background is not None:
        return _chisel_pass(rate, finite, np.broadcast_to(background, rate.shape), kernel_fwhm, qthresh, erode, opening,
                            snquant, detgrowquant, max_hole, max_grow)
    mask = finite.copy()
    for _ in range(max(passes, 1)):
        sky = tiled_sky(rate, mask, sky_box)
        mask = _chisel_pass(rate, finite, sky, kernel_fwhm, qthresh, erode, opening, snquant, detgrowquant, max_hole, max_grow)
    return mask


def tiled_sky(rate, mask, box):
    """Sky model from sigma-clipped medians in ``box`` tiles, interpolated to every pixel.

    A light 3-by-3 median over the tile grid, as NoiseChisel's tiled sky
    rather than the heavier smoothing of the cleaner's background model, so
    that sky structure on scales of a few tiles is followed, not detected.
    """
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    image = np.where(np.isfinite(rate), rate, 0.0).astype(np.float64)
    excluded = ~(np.asarray(mask, dtype=bool) & np.isfinite(rate))
    sky = Background2D(
        image,
        box_size=(box, box),
        filter_size=(3, 3),
        mask=excluded,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
        bkg_estimator=MedianBackground(),
        exclude_percentile=50.0,
    )
    return np.asarray(sky.background, dtype=np.float64)


def remove_stripes(image, finite):
    """Subtract the per-row and then per-column medians of the finite pixels.

    The draft rate image carries the 1/f banding that the cleaner exists to
    remove; left in, its ridges are connected along rows in the convolved
    image and let a detection grow along a band.  Compact sources barely
    move a row or column median.
    """
    import warnings

    work = np.where(finite, image, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        rows = np.nanmedian(work, axis=1, keepdims=True)
        work = work - np.where(np.isfinite(rows), rows, 0.0)
        cols = np.nanmedian(work, axis=0, keepdims=True)
        work = work - np.where(np.isfinite(cols), cols, 0.0)
    return np.where(finite, work, 0.0)


def _chisel_pass(rate, finite, background, kernel_fwhm, qthresh, erode, opening, snquant, detgrowquant, max_hole, max_grow):
    image = remove_stripes(np.where(finite, rate - background, 0.0), finite)

    sigma_kernel = kernel_fwhm / 2.3548
    convolved = ndimage.gaussian_filter(image, sigma_kernel, truncate=2.5 * kernel_fwhm / sigma_kernel)
    sigma_conv = _robust_sigma(convolved[finite])

    threshold = np.quantile(convolved[finite], qthresh)
    detections = _candidates(convolved, threshold, erode, opening) & finite
    # pseudo-detections: the same procedure on the sign-flipped image finds what noise alone produces
    noise_only = _candidates(-convolved, threshold, erode, opening) & finite
    labels, nlabels = ndimage.label(detections, ndimage.generate_binary_structure(2, 2))
    noise_labels, n_noise = ndimage.label(noise_only, ndimage.generate_binary_structure(2, 2))
    snr = _blob_snr(convolved, labels, nlabels, sigma_conv)
    noise_snr = _blob_snr(-convolved, noise_labels, n_noise, sigma_conv)
    sn_threshold = np.quantile(noise_snr, snquant) if noise_snr.size >= 20 else 0.0
    keep = np.zeros(nlabels + 1, dtype=bool)
    keep[1:] = snr > sn_threshold
    true_detections = keep[labels]

    # grow back over the eroded border and into the faint wings
    grow_into = (convolved > np.quantile(convolved[finite], detgrowquant)) & finite
    structure = ndimage.generate_binary_structure(2, 2)
    grown = ndimage.binary_dilation(true_detections, structure, iterations=erode + opening) & (grow_into | true_detections)
    for iteration in range(max_grow):
        step = ndimage.binary_dilation(grown, structure) & grow_into
        if not np.any(step & ~grown):
            break
        grown |= step
    _chisel_pass.last_growth_iterations = iteration
    if max_hole > 0:
        holes = ndimage.binary_fill_holes(grown) & ~grown
        hole_labels, n_holes = ndimage.label(holes)
        if n_holes:
            sizes = ndimage.sum(np.ones_like(holes, dtype=float), hole_labels, np.arange(1, n_holes + 1))
            small = np.zeros(n_holes + 1, dtype=bool)
            small[1:] = np.asarray(sizes) <= max_hole
            grown |= small[hole_labels] & holes
    return finite & ~grown
