"""Metrics for the validation harness (plan section C.2).

Each function takes plain arrays in the science frame plus the detector
axis keywords, and returns numbers that the experiment scripts under
``scripts/`` write out and plot.  Keeping them here lets the tests cover
them.
"""
import numpy as np
from astropy.stats import mad_std
from scipy import ndimage

from . import readout
from .noise_spectrum import channel_spectra

__all__ = [
    "channel_psd",
    "row_banding_power",
    "aperture_scatter",
    "robust_sigma",
    "d0_chisq",
    "recovered_fraction",
    "injected_flux_bias",
]


def robust_sigma(values):
    """Standard deviation estimated from the median absolute deviation of finite values (astropy's mad_std).

    Parameters
    ----------
    values : array_like
        Sample values of any shape; non-finite entries are left out.

    Returns
    -------
    float
        The MAD-based standard deviation, or NaN when no finite value is left.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(mad_std(values))


def channel_psd(diff_cube, mask, fastaxis, slowaxis, max_lag=2**14, first_direction=1):
    """Readout-ordered noise spectrum per output channel, median over group differences.

    Parameters
    ----------
    diff_cube : ndarray, shape (ndiffs, nrows, ncols)
        Group differences in the science frame.
    mask : ndarray of bool, shape (nrows, ncols) or (ndiffs, nrows, ncols)
        Background mask, True for usable pixels.  A single plane is applied
        to every difference.
    fastaxis : int
        ``meta.subarray.fastaxis`` of the data, used to turn the arrays into
        the detector frame.
    slowaxis : int
        ``meta.subarray.slowaxis`` of the data.
    max_lag : int
        Autocorrelation length in clocks.
    first_direction : int
        Readout direction of channel 0.

    Returns
    -------
    freq : ndarray
        Cycles per clock.
    spectra : ndarray, shape (4, nfreq)
        Median spectrum of each channel.
    """
    diffs = readout.to_detector_frame(np.asarray(diff_cube), fastaxis, slowaxis)
    masks = readout.to_detector_frame(np.asarray(mask), fastaxis, slowaxis)
    if masks.ndim == 2:
        masks = np.broadcast_to(masks, diffs.shape)
    per_channel = [[] for _ in range(readout.NOUTPUTS)]
    freq = None
    for diff, m in zip(diffs, masks):
        freq, spectra = channel_spectra(diff, m, first_direction, max_lag)
        for k, spectrum in enumerate(spectra):
            if spectrum is not None:
                per_channel[k].append(spectrum)
    return freq, np.array([np.median(s, axis=0) if s else np.zeros(freq.size) for s in per_channel])


def row_banding_power(rate_image, mask, fastaxis, slowaxis):
    """Power spectrum of the per-channel row means of a rate image.

    Parameters
    ----------
    rate_image : ndarray, shape (nrows, ncols)
        Rate image in the science frame, DN/s.
    mask : ndarray of bool, shape (nrows, ncols)
        Background mask, True for usable pixels.
    fastaxis : int
        ``meta.subarray.fastaxis`` of the data, used to turn the arrays into
        the detector frame.
    slowaxis : int
        ``meta.subarray.slowaxis`` of the data.

    Returns
    -------
    freq : ndarray
        Cycles per row.
    power : ndarray, shape (4, nfreq)
        ``|rfft(row_means - mean)|**2 / nrows`` per channel, so that the
        mean over bins equals the variance of the row means.
    row_means : ndarray, shape (4, nrows)
        The masked row means themselves.
    """
    image = readout.to_detector_frame(np.asarray(rate_image, dtype=np.float64), fastaxis, slowaxis)
    good = readout.to_detector_frame(np.asarray(mask, dtype=bool), fastaxis, slowaxis) & np.isfinite(image)
    powers, means = [], []
    freq = None
    for columns in readout.channel_slices(image.shape[-1]):
        block = image[:, columns]
        weight = good[:, columns]
        with np.errstate(invalid="ignore", divide="ignore"):
            row_mean = np.where(weight, block, 0.0).sum(axis=1) / weight.sum(axis=1)
        row_mean = np.where(np.isfinite(row_mean), row_mean, np.nanmedian(row_mean))
        centered = row_mean - row_mean.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2 / centered.size
        freq = np.fft.rfftfreq(centered.size, d=1.0)
        powers.append(power)
        means.append(row_mean)
    return freq, np.array(powers), np.array(means)


def aperture_scatter(rate_image, mask, sizes, pixel_sigma=None, min_fraction=0.9):
    """Scatter of the mean in square apertures of background pixels, versus aperture size.

    Parameters
    ----------
    rate_image : ndarray, shape (nrows, ncols)
        Rate image, DN/s.
    mask : ndarray of bool, shape (nrows, ncols)
        Background mask, True for usable pixels.  The mean in an aperture
        is taken over its usable pixels only.
    sizes : sequence of int
        Aperture side lengths in pixels.
    pixel_sigma : float or None
        Per-pixel standard deviation for the white-noise prediction.
        Default is the robust sigma of the usable pixels.
    min_fraction : float
        Apertures with a smaller fraction of usable pixels are skipped.

    Returns
    -------
    dict
        ``{"size": array, "scatter": array, "white_prediction": array, "n_apertures": array}``.
        ``white_prediction`` is ``pixel_sigma / size``.
    """
    image = np.asarray(rate_image, dtype=np.float64)
    good = np.asarray(mask, dtype=bool) & np.isfinite(image)
    values = np.where(good, image, 0.0)
    if pixel_sigma is None:
        pixel_sigma = robust_sigma(image[good])
    scatter, count = [], []
    for size in sizes:
        summed = ndimage.uniform_filter(values, size=size, mode="constant")
        fraction = ndimage.uniform_filter(good.astype(np.float64), size=size, mode="constant")
        usable = fraction >= min_fraction
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(usable, summed / fraction, np.nan)
        # keep one aperture per non-overlapping block so the samples are independent
        half = size // 2
        picks = np.zeros_like(usable)
        picks[half::size, half::size] = True
        selected = mean[usable & picks]
        scatter.append(robust_sigma(selected) if selected.size > 1 else np.nan)
        count.append(int(selected.size))
    sizes = np.asarray(sizes)
    return {
        "size": sizes,
        "scatter": np.array(scatter),
        "white_prediction": pixel_sigma / sizes,
        "n_apertures": np.array(count),
    }


def d0_chisq(d0_resid, used_reset, mask=None):
    """Mean squared normalized residual of the reset difference (plan metric 5).

    Parameters
    ----------
    d0_resid : ndarray
        Normalized residual planes from `ResetRampFitStep` diagnostics.
    used_reset : ndarray of bool
        Where the reset difference took part in the final fit.
    mask : ndarray of bool or None
        Extra pixel selection, broadcast against ``d0_resid``.

    Returns
    -------
    dict
        ``{"chisq": mean of d0_resid**2, "n": count, "used_fraction": fraction of used_reset}``.
    """
    resid = np.asarray(d0_resid, dtype=np.float64)
    used = np.asarray(used_reset, dtype=bool) & np.isfinite(resid)
    if mask is not None:
        used &= np.broadcast_to(np.asarray(mask, dtype=bool), used.shape)
    n = int(used.sum())
    chisq = float(np.mean(resid[used] ** 2)) if n else float("nan")
    return {"chisq": chisq, "n": n, "used_fraction": float(np.asarray(used_reset, dtype=bool).mean())}


def recovered_fraction(rate, reference_rate, mask=None):
    """Pixels with a finite rate, compared with a reference fit (plan metric 6, count part).

    Parameters
    ----------
    rate : ndarray, shape (nrows, ncols)
        Rate image from the fit under test, DN/s.
    reference_rate : ndarray, shape (nrows, ncols)
        Rate image from the reference fit, DN/s.
    mask : ndarray of bool or None
        Pixels to consider; default all.

    Returns
    -------
    dict
        ``n_considered``, ``n_finite``, ``n_finite_reference``, ``n_recovered``
        (finite in ``rate`` but not in the reference) and ``n_lost`` (the
        reverse).
    """
    consider = np.ones(np.shape(rate), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    finite = np.isfinite(rate) & consider
    finite_ref = np.isfinite(reference_rate) & consider
    return {
        "n_considered": int(consider.sum()),
        "n_finite": int(finite.sum()),
        "n_finite_reference": int(finite_ref.sum()),
        "n_recovered": int((finite & ~finite_ref).sum()),
        "n_lost": int((~finite & finite_ref).sum()),
    }


def injected_flux_bias(
    rate, reference_rate, x, y, true_flux, mask, r_ap=3.0, r_in=6.0, r_out=10.0, density_box=33
):
    """Photometric bias of cleaning on injected sources (plan metric 4).

    For every source the flux in a circular aperture of radius ``r_ap``,
    with the median of an annulus ``r_in`` to ``r_out`` removed, is measured
    on ``rate`` and on ``reference_rate`` (the same data without cleaning).
    Their difference divided by the true flux is the relative bias that the
    cleaning introduced; the common noise realization cancels.  Each source
    is also tagged with the local mask density (fraction of masked pixels in
    a ``density_box`` square around it) and the distance to its nearest
    neighbour, and the biases are binned by both.

    Parameters
    ----------
    rate : ndarray, shape (nrows, ncols)
        Cleaned rate image, DN/s.
    reference_rate : ndarray, shape (nrows, ncols)
        Rate image of the same data without cleaning, DN/s.
    x : ndarray, shape (nsources,)
        Source column positions, zero-based pixel coordinates.
    y : ndarray, shape (nsources,)
        Source row positions, zero-based pixel coordinates.
    true_flux : ndarray, shape (nsources,)
        Injected total flux per source in rate units (DN/s).
    mask : ndarray of bool, shape (nrows, ncols)
        Background mask used by the cleaner, True for background pixels.
    r_ap : float
        Aperture radius in pixels.
    r_in : float
        Inner radius of the background annulus in pixels.
    r_out : float
        Outer radius of the background annulus in pixels.  Sources closer
        than ``ceil(r_out) + 1`` to an image edge get NaN fluxes.
    density_box : int
        Side of the box for the local mask density.

    Returns
    -------
    dict
        Per-source arrays ``relative_bias``, ``density``, ``neighbor_distance``,
        ``flux_reference``; and binned medians ``by_density``, ``by_distance``
        and ``by_flux`` as lists of ``(low, high, n, median_bias, scatter)``.
    """
    rate = np.asarray(rate, dtype=np.float64)
    reference_rate = np.asarray(reference_rate, dtype=np.float64)
    masked = ~np.asarray(mask, dtype=bool)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    true_flux = np.asarray(true_flux, dtype=np.float64)
    nrows, ncols = rate.shape
    half = int(np.ceil(r_out)) + 1
    dbox = density_box // 2

    def aperture_flux(image, xc, yc):
        """Background-subtracted flux of one source in the circular aperture.

        Parameters
        ----------
        image : ndarray, shape (nrows, ncols)
            Rate image to measure on, DN/s.
        xc : float
            Source column position, zero-based pixels.
        yc : float
            Source row position, zero-based pixels.

        Returns
        -------
        float
            Sum over pixels within ``r_ap`` of the image minus the median of
            the ``r_in`` to ``r_out`` annulus; NaN when the cutout leaves the
            image, the aperture has no finite pixel or the annulus has fewer
            than 8.
        """
        x0, x1 = int(round(xc)) - half, int(round(xc)) + half + 1
        y0, y1 = int(round(yc)) - half, int(round(yc)) + half + 1
        if x0 < 0 or y0 < 0 or x1 > ncols or y1 > nrows:
            return np.nan
        cut = image[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(xx - xc, yy - yc)
        inside = (r <= r_ap) & np.isfinite(cut)
        annulus = (r >= r_in) & (r <= r_out) & np.isfinite(cut)
        if inside.sum() == 0 or annulus.sum() < 8:
            return np.nan
        return float(np.sum(cut[inside] - np.median(cut[annulus])))

    flux_c = np.array([aperture_flux(rate, xi, yi) for xi, yi in zip(x, y)])
    flux_r = np.array([aperture_flux(reference_rate, xi, yi) for xi, yi in zip(x, y)])
    relative_bias = (flux_c - flux_r) / true_flux

    density = np.full(x.size, np.nan)
    for k, (xi, yi) in enumerate(zip(x, y)):
        xc, yc = int(round(xi)), int(round(yi))
        box = masked[max(yc - dbox, 0): yc + dbox + 1, max(xc - dbox, 0): xc + dbox + 1]
        density[k] = box.mean() if box.size else np.nan

    if x.size > 1:
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.hypot(dx, dy)
        np.fill_diagonal(dist, np.inf)
        neighbor = dist.min(axis=1)
    else:
        neighbor = np.full(x.size, np.inf)

    def binned(values, edges):
        """Median relative bias of the sources in each bin of ``values``.

        Parameters
        ----------
        values : ndarray, shape (nsources,)
            Per-source quantity to bin on (mask density, neighbour distance
            or true flux).
        edges : ndarray, shape (nbins + 1,)
            Bin edges; a source belongs to a bin when ``low <= value < high``.

        Returns
        -------
        list of tuple
            One ``(low, high, n, median_bias, scatter)`` per bin that holds
            at least one source with a finite relative bias; ``scatter`` is
            the robust sigma of the biases in the bin.
        """
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            selected = (values >= low) & (values < high) & np.isfinite(relative_bias)
            n = int(selected.sum())
            if n:
                rows.append((float(low), float(high), n, float(np.median(relative_bias[selected])),
                             float(robust_sigma(relative_bias[selected]))))
        return rows

    return {
        "relative_bias": relative_bias,
        "density": density,
        "neighbor_distance": neighbor,
        "flux_reference": flux_r,
        "by_density": binned(density, np.array([0.0, 0.05, 0.15, 0.3, 0.5, 1.01])),
        "by_distance": binned(neighbor, np.array([0.0, 5.0, 10.0, 20.0, 40.0, np.inf])),
        "by_flux": binned(true_flux, np.geomspace(max(true_flux.min(), 1e-6), true_flux.max() * (1 + 1e-9), 7)),
        "n_measured": int(np.isfinite(relative_bias).sum()),
        "median_bias": float(np.nanmedian(relative_bias)),
    }
