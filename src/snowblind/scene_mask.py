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
from astropy.stats import mad_std
from scipy import ndimage

__all__ = ["chisel_mask", "clip_mask", "tiled_sky", "remove_stripes", "saturated_stars", "spike_mask"]


def clip_mask(rate, n_sigma=2.0):
    """Background mask by sigma clipping, as the jwst step does: True where background."""
    good = np.isfinite(rate)
    level = np.median(rate[good])
    sigma = mad_std(rate[good], ignore_nan=True)
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
    coarse_fwhm=12.0,
    coarse_qthresh=0.9,
    coarse_erode=4,
    coarse_sky_box=256,
    coarse_max_grow=4,
):
    """Background mask (True where background) that follows source wings.

    Two tiers: the fine pass at ``kernel_fwhm`` finds sources and their
    wings; a coarse pass at ``coarse_fwhm`` with a ``coarse_sky_box`` tiled
    sky and a higher detection quantile finds the smooth halos and
    diffraction spikes of bright stars, which the fine pass misses because
    its 32-pixel tiled sky absorbs them (the halo of a star 400 times the
    sky is 20 percent of the sky within 100 pixels and 8 percent at 100 to
    200, below the per-pixel threshold).  ``coarse_fwhm=None`` disables the
    coarse tier.  Left unmasked, such halos are removed by a linear 1/f
    estimator as slow noise.

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
    if coarse_fwhm:
        sky = tiled_sky(rate, mask, coarse_sky_box)
        coarse = _chisel_pass(
            rate, finite, sky, coarse_fwhm, coarse_qthresh, coarse_erode, opening, snquant, detgrowquant, max_hole,
            coarse_max_grow,
        )
        mask &= coarse
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
    sigma_conv = mad_std(convolved[finite], ignore_nan=True)

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


def saturated_stars(background_mask, finite, min_halo=10000, max_fill=0.4, max_stars=4, border=4):
    """Find the saturated stars in a scene mask by the shape of their masked regions.

    A bright star's mask is a large region around saturated (non-finite)
    pixels with arms along the diffraction spikes, so it fills a small
    fraction of its bounding box; a galaxy's is compact and fills most
    of its box.  Regions of at least ``min_halo`` pixels that contain
    non-finite pixels and fill at most ``max_fill`` of their bounding box
    count as stars, the largest ``max_stars`` of them.  The ``border``
    rows and columns of reference pixels are set aside so that a region
    touching the frame edge is not joined to it.

    Returns a list of (y, x) centres of the largest cluster of saturated
    pixels in each region, the star's core, largest region first.
    """
    interior = ~np.asarray(background_mask, dtype=bool)
    interior[:border] = interior[-border:] = False
    interior[:, :border] = interior[:, -border:] = False
    regions, nregions = ndimage.label(interior)
    stars = []
    for k, window in enumerate(ndimage.find_objects(regions)):
        if window is None:
            continue
        part = regions[window] == k + 1
        area = int(part.sum())
        if area < min_halo or area > max_fill * part.size:
            continue
        core = part & ~finite[window]
        if not core.any():
            continue
        clusters, nclusters = ndimage.label(core)
        largest = np.argmax(ndimage.sum(core, clusters, np.arange(1, nclusters + 1))) + 1
        cy, cx = ndimage.center_of_mass(clusters == largest)
        stars.append((area, cy + window[0].start, cx + window[1].start))
    stars.sort(reverse=True)
    return [(cy, cx) for _, cy, cx in stars[:max_stars]]


def spike_mask(
    rate, background_mask, min_halo=10000, max_fill=0.4, max_stars=4, half_width=40, block=64, near=12,
    min_excess=0.02, nsigma=3.0, patience=3, border=4,
):
    """Mask the column-aligned diffraction spikes of saturated stars along their extent.

    The scene mask covers a spike's bright part but not its wings, a few
    columns wide at a fraction of the sky, and a spike along a column
    raises that column's masked mean and darkens the whole column when the
    offset is removed.  Only the vertical spike matters for that, so this
    follows it alone.  Stars come from :func:`saturated_stars`.  From each
    star the column is walked up and down in blocks of ``block`` rows.
    In each block the profile across ``half_width`` columns either side
    of the current spike centre is the masked mean of the rows, smoothed
    over three columns, less the median of the columns more than
    ``near + 10`` from the centre, which removes the star's halo and the
    local background.  Columns within ``near`` of the centre whose excess
    is above ``min_excess`` times the sky and ``nsigma`` times its noise
    are masked in that block, with one column either side, and their
    centroid becomes the centre for the next block, so a spike that
    drifts is followed.  The walk ends at the frame edge or after
    ``patience`` consecutive blocks without a detection; blocks inside
    the star's own mask are passed over.

    Returns the mask with the spikes removed from the background (True where
    background), and the number of stars treated.
    """
    rate = np.asarray(rate, dtype=np.float64)
    finite = np.isfinite(rate)
    background = np.asarray(background_mask, dtype=bool) & finite
    if not background.any():
        return background, 0
    sky = float(np.median(rate[background]))
    sigma = mad_std(rate[background], ignore_nan=True)
    stars = saturated_stars(background_mask, finite, min_halo, max_fill, max_stars, border)
    if not stars:
        return background, 0
    ny, nx = rate.shape
    signal = np.where(background, rate - sky, 0.0)
    weight = background.astype(np.float64)
    smooth = np.ones(3) / 3.0
    out = background.copy()
    for cy, cx in stars:
        for direction in (-1, 1):
            centre = int(round(cx))
            y = int(round(cy))
            misses = 0
            while True:
                r0, r1 = (y - block, y) if direction < 0 else (y, y + block)
                if r0 < border or r1 > ny - border:
                    break
                y += direction * block
                c0, c1 = max(border, centre - half_width), min(nx - border, centre + half_width + 1)
                total = np.convolve(signal[r0:r1, c0:c1].sum(axis=0), smooth, mode="same")
                count = np.convolve(weight[r0:r1, c0:c1].sum(axis=0), smooth, mode="same")
                dx = np.arange(c0, c1) - centre
                inner = np.abs(dx) <= near
                if count[inner].sum() < 0.25 * block * inner.sum():
                    continue  # inside the star's own mask
                with np.errstate(invalid="ignore", divide="ignore"):
                    profile = np.where(count >= 0.25 * block, total / count, np.nan)
                outer = profile[np.abs(dx) > near + 10]
                if np.isfinite(outer).sum() < 10:
                    continue
                excess = profile - np.nanmedian(outer)
                noise = nsigma * sigma / np.sqrt(np.maximum(count, 1.0))
                with np.errstate(invalid="ignore"):
                    detected = inner & (excess > min_excess * abs(sky)) & (excess > noise)
                if not detected.any():
                    misses += 1
                    if misses >= patience:
                        break
                    continue
                misses = 0
                centre = int(round(np.average(np.arange(c0, c1)[detected], weights=excess[detected])))
                columns = ndimage.binary_dilation(detected, structure=np.ones(3, dtype=bool))
                out[r0:r1, c0:c1][:, columns] = False
    return out, len(stars)
