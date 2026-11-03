"""Simulate outlier-detection reference images with stpsf stars: today's scheme against references from all exposures.

Outlier detection compares each exposure with a prediction of it made from
all the exposures: a reference image, blotted back onto the exposure.  This
script builds a dithered field of stpsf stars at detector scale and
measures two things for several ways of making the reference:

1. Without noise, how far the prediction of each star's brightest pixel is
   from the true value.
2. With noise and cosmic rays, flagging with jwst's own rule (stcal's
   ``flag_resampled_crs``: two passes, snr 5 then 4, and a tolerance
   ``scale * D`` with D the largest difference between a reference pixel
   and its neighbors), or with a tolerance measured from the data (see
   below): the cosmic rays found (overall and within 3 pixels of a star),
   the good pixels flagged on blank sky, next to another exposure's cosmic
   ray, and in star wings, and the star-core pixels flagged.  Only cosmic
   rays of at least 8 times the local noise are counted.

The measured tolerances have no ``scale``: for each fine pixel of a
clipped-mean reference, ``model**2`` is the weighted variance of the input
pixels kept by the clipping minus the weighted mean of their noise squared
("measured"), or the square of 1.4826 times the weighted median absolute
deviation of all its input pixels from the clipped mean minus the weighted
median of their noise squared ("robust"), zero if negative, blotted onto
the exposure; a pixel is flagged where
``|data - prediction| > snr * sqrt(err**2 + model**2)``, in jwst's two
passes (snr 5 and 4) with its 3x3 growth.  ``--defaults`` runs only the
proposed defaults (scale 0.5, pixfrac 0.8, plus correction) with the
measured tolerance and with ``scale * D``, against today's scheme.

The references:

- today: each exposure drizzled alone (square kernel, pixfrac 1, input
  pixel scale), the median of those images, blotted back bilinearly, as
  jwst's OutlierDetectionStep does;
- "median of input pixels": on a grid of ``scale`` input pixels, for each
  output pixel the median of the input pixels of every exposure that land
  on it, weighted by their overlap at ``pixfrac``;
- "clipped mean": the weighted mean of those input pixels within 3 sigma of
  the median, three times;
- "plus correction": the exposures' residuals (data minus prediction)
  combined the same way and added to the reference once.

The exposures are offset from each other by random sub-pixel shifts only:
no rotation or distortion between them, no saturation, no IPC.  The PSF is
stpsf's ``OVERDIST`` (distortion and detector blur), oversampled 10 times
and summed over each detector pixel, so that a star can sit at any
sub-pixel position.

Needs spizzle and stpsf (``pip install spizzle stpsf``) and stpsf's data,
version 2.3.0 or later (184 MB, from
https://stsci.box.com/shared/static/kqfolg2bfzqc4mjkgmujo06d3iaymahv.gz),
with ``STPSF_PATH`` set to its ``stpsf-data`` directory::

    STPSF_PATH=.../stpsf-data python scripts/exp_outlier_reference.py

The defaults (F070W, the most undersampled NIRCam filter, on NRCA1; 4
exposures; 160 stars on 192x192 pixels; realizations 11, 12, 13) give the
tables in PLAN-spizzle-resample-and-outlier-detection.md, section 6.
"""
import argparse

import numba
import numpy as np
import stpsf
from scipy import ndimage, sparse
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from spizzle import Coadd, blot, drizzle_operator

OVERSAMPLE = 10
PSF_PIXELS = 31
MARGIN = 4
SKY_SIGMA = 5.0  # electrons, read noise and sky together


def pixel_summed_psf(filter_name, detector, instrument="NIRCam"):
    """stpsf's OVERDIST PSF summed over each detector pixel, as a spline in the offset from the star in pixels."""
    nc = getattr(stpsf, instrument)()
    nc.filter = filter_name
    nc.detector = detector
    over = nc.calc_psf(oversample=OVERSAMPLE, fov_pixels=PSF_PIXELS)["OVERDIST"].data.astype(np.float64)
    n = over.shape[0]
    center = (n - 1) / 2
    cum = np.zeros((n + 1, n + 1))
    cum[1:, 1:] = over.cumsum(0).cumsum(1)
    # the sum over the OVERSAMPLE x OVERSAMPLE samples of a pixel centered m / OVERSAMPLE pixels from the star
    half = OVERSAMPLE // 2
    m = np.arange(-(n // 2 - half), n // 2 - half + 1)
    lo = np.round(center + m - half + 0.5).astype(int)
    hi = lo + OVERSAMPLE
    summed = cum[np.ix_(hi, hi)] - cum[np.ix_(lo, hi)] - cum[np.ix_(hi, lo)] + cum[np.ix_(lo, lo)]
    return RectBivariateSpline(m / OVERSAMPLE, m / OVERSAMPLE, summed, kx=3, ky=3)


def fwhm(psf):
    """Full width at half maximum of the pixel-summed PSF along x, in pixels."""
    u = np.linspace(-3, 3, 6001)
    profile = psf(np.array([0.0]), u)[0]
    above = u[profile >= profile.max() / 2]
    return above[-1] - above[0]


class Field:
    """Stars at fixed sky positions, seen by exposures that are shifted by fractions of a pixel."""

    def __init__(self, psf, n_exposures, size, n_stars, seed):
        rng = np.random.default_rng(seed)
        self.psf = psf
        self.n_exp = n_exposures
        self.shape = (size, size)
        self.shifts = np.vstack([[0.0, 0.0], rng.uniform(-3.0, 3.0, (n_exposures - 1, 2))])
        self.stars = np.column_stack([rng.uniform(14.0, size - 14.0, n_stars), rng.uniform(14.0, size - 14.0, n_stars)])
        self._operators = {}

    def scene(self, k, fluxes):
        """Exposure k without noise; its pixel (j, i) is centered at sky position (i + dx, j + dy)."""
        dx, dy = self.shifts[k]
        r = PSF_PIXELS // 2
        img = np.zeros(self.shape)
        for (xs, ys), f in zip(self.stars, fluxes):
            x0, y0 = xs - dx, ys - dy
            i0, j0 = int(round(x0)), int(round(y0))
            js = np.arange(max(0, j0 - r), min(self.shape[0], j0 + r + 1))
            cols = np.arange(max(0, i0 - r), min(self.shape[1], i0 + r + 1))
            img[np.ix_(js, cols)] += f * self.psf(js - y0, cols - x0)
        return img

    def pixmap(self, k, scale):
        """Pixel map of exposure k onto a grid of pixels ``scale`` input pixels wide."""
        dx, dy = self.shifts[k]
        y, x = np.indices(self.shape, dtype=float)
        return np.dstack([(x + dx + MARGIN) / scale, (y + dy + MARGIN) / scale])

    def out_shape(self, scale):
        n = int(np.ceil((self.shape[0] + 2 * MARGIN) / scale))
        return (n, n)

    def operators(self, scale, pixfrac):
        """All exposures' drizzle operators side by side: row o lists every input pixel landing on output pixel o."""
        key = (scale, pixfrac)
        if key not in self._operators:
            self._operators[key] = sparse.hstack([
                drizzle_operator(self.pixmap(k, scale), self.out_shape(scale), pixfrac=pixfrac).full()
                for k in range(self.n_exp)
            ]).tocsr()
        return self._operators[key]

    def distance_to_stars(self, k):
        dx, dy = self.shifts[k]
        y, x = np.indices(self.shape)
        return np.min([np.hypot(x - (xs - dx), y - (ys - dy)) for xs, ys in self.stars], axis=0)


@numba.njit
def _weighted_median(indptr, indices, weights, values, out):
    """For each row, the weighted median of the values it lists, NaN values left out."""
    for o in range(out.size):
        a, b = indptr[o], indptr[o + 1]
        k = 0
        v = np.empty(b - a)
        w = np.empty(b - a)
        for e in range(a, b):
            if not np.isnan(values[indices[e]]):
                v[k] = values[indices[e]]
                w[k] = weights[e]
                k += 1
        if k == 0:
            out[o] = np.nan
            continue
        v = v[:k]
        w = w[:k]
        order = np.argsort(v)
        half = 0.5 * w.sum()
        cum = 0.0
        for q in order:
            cum += w[q]
            if cum >= half:
                out[o] = v[q]
                break


@numba.njit
def _clipped_mean(indptr, indices, weights, values, sigma, start, out, model2):
    """For each row, the weighted mean of its values within 3 sigma of the estimate, three times from ``start``;
    and in ``model2`` the weighted variance of the values kept the last time minus their weighted mean sigma**2,
    zero if negative."""
    for o in range(out.size):
        est = start[o]
        model2[o] = np.nan
        if np.isnan(est):
            out[o] = np.nan
            continue
        for _ in range(3):
            num = den = 0.0
            for e in range(indptr[o], indptr[o + 1]):
                v = values[indices[e]]
                if not np.isnan(v) and abs(v - est) <= 3.0 * sigma[indices[e]]:
                    num += weights[e] * v
                    den += weights[e]
            if den > 0.0:
                est = num / den
        out[o] = est
        var = noise = den = 0.0
        for e in range(indptr[o], indptr[o + 1]):
            v = values[indices[e]]
            s = sigma[indices[e]]
            if not np.isnan(v) and abs(v - est) <= 3.0 * s:
                var += weights[e] * (v - est) ** 2
                noise += weights[e] * s * s
                den += weights[e]
        if den > 0.0:
            model2[o] = max(0.0, (var - noise) / den)


@numba.njit
def _robust_spread(indptr, indices, weights, values, sigma, center, out):
    """For each row, (1.4826 * the weighted median absolute deviation of its values from ``center``)**2 minus the
    weighted median of their sigma**2, zero if negative."""
    for o in range(out.size):
        a, b = indptr[o], indptr[o + 1]
        dev = np.empty(b - a)
        noise = np.empty(b - a)
        w = np.empty(b - a)
        n = 0
        for e in range(a, b):
            v = values[indices[e]]
            if not np.isnan(v) and not np.isnan(center[o]):
                dev[n] = abs(v - center[o])
                noise[n] = sigma[indices[e]] ** 2
                w[n] = weights[e]
                n += 1
        if n == 0:
            out[o] = np.nan
            continue
        stats = np.empty(2)
        for t, x in enumerate((dev[:n], noise[:n])):
            order = np.argsort(x)
            half = 0.5 * w[:n].sum()
            cum = 0.0
            for q in order:
                cum += w[q]
                if cum >= half:
                    stats[t] = x[q]
                    break
        out[o] = max(0.0, (1.4826 * stats[0]) ** 2 - stats[1])


def reference_today(field, data, sigma):
    """jwst's reference: the median of the exposures drizzled one at a time; and its grid's scale."""
    singles = []
    for k in range(field.n_exp):
        coadd = Coadd(field.out_shape(1.0))
        coadd.add(drizzle_operator(field.pixmap(k, 1.0), field.out_shape(1.0)), data[k])
        singles.append(coadd.out_img)
    return np.nanmedian(singles, axis=0), 1.0, None


def reference_from_input_pixels(scale, pixfrac, clipped=False, correct=False):
    """A function making the reference from all exposures' input pixels on a finer grid."""
    def make(field, data, sigma):
        W = field.operators(scale, pixfrac)
        values = np.concatenate([d.ravel() for d in data])
        out = np.empty(W.shape[0])
        _weighted_median(W.indptr, W.indices, W.data, values, out)
        model2 = None
        if clipped:
            sig = np.concatenate([s.ravel() for s in sigma])
            kept = np.empty(W.shape[0])
            robust = np.empty(W.shape[0])
            _clipped_mean(W.indptr, W.indices, W.data, values, sig, out.copy(), out, kept)
            _robust_spread(W.indptr, W.indices, W.data, values, sig, out, robust)
            model2 = {"measured": kept.reshape(field.out_shape(scale)), "robust": robust.reshape(field.out_shape(scale))}
        ref = out.reshape(field.out_shape(scale))
        if correct:
            residuals = np.concatenate([(data[k] - blot(ref, field.pixmap(k, scale))).ravel() for k in range(field.n_exp)])
            correction = np.empty(W.shape[0])
            _weighted_median(W.indptr, W.indices, W.data, residuals, correction)
            ref = ref + np.nan_to_num(correction.reshape(field.out_shape(scale)))
        return ref, scale, model2
    return make


def peak_errors(field, make):
    """(data - prediction) / data at the brightest pixel of every star in every exposure, without noise."""
    data = [field.scene(k, np.ones(len(field.stars))) for k in range(field.n_exp)]
    ref, scale, _ = make(field, data, [np.ones(field.shape)] * field.n_exp)
    errors = []
    for k in range(field.n_exp):
        prediction = blot(ref, field.pixmap(k, scale))
        dx, dy = field.shifts[k]
        for xs, ys in field.stars:
            i0, j0 = int(round(xs - dx)), int(round(ys - dy))
            box = data[k][j0 - 1:j0 + 2, i0 - 1:i0 + 2]
            jj, ii = np.unravel_index(np.argmax(box), box.shape)
            j, i = j0 - 1 + jj, i0 - 1 + ii
            errors.append((data[k][j, i] - prediction[j, i]) / data[k][j, i])
    return np.array(errors)


def abs_deriv(a):
    """stcal's _abs_deriv: the largest absolute difference of each pixel with its four neighbors."""
    out = np.zeros_like(a)
    rows = np.abs(np.diff(a, axis=0))
    np.putmask(out[1:], np.isfinite(rows), rows)
    np.putmask(out[:-1], rows > out[:-1], rows)
    cols = np.abs(np.diff(a, axis=1))
    np.putmask(out[:, 1:], cols > out[:, 1:], cols)
    np.putmask(out[:, :-1], cols > out[:, :-1], cols)
    return out


def flag(sci, err, prediction, scale, snr=(5.0, 4.0)):
    """stcal's flag_resampled_crs: the first threshold grown by one pixel, and the second."""
    d = abs_deriv(prediction)
    diff = np.abs(sci - prediction)
    first = ndimage.convolve(diff > scale[0] * d + snr[0] * np.nan_to_num(err), np.ones((3, 3), dtype=int), mode="nearest")
    return first & (diff > scale[1] * d + snr[1] * np.nan_to_num(err))


def flag_measured(sci, err, prediction, model, snr=(5.0, 4.0)):
    """The measured tolerance: jwst's two passes and growth with ``snr * sqrt(err**2 + model**2)``, no ``scale * D``."""
    diff = np.abs(sci - prediction)
    noise = np.sqrt(np.nan_to_num(err) ** 2 + np.nan_to_num(model) ** 2)
    first = ndimage.convolve(diff > snr[0] * noise, np.ones((3, 3), dtype=int), mode="nearest")
    return first & (diff > snr[1] * noise)


def realization(field, seed, cr_fraction):
    """Noisy exposures with cosmic rays, their errors, the detectable cosmic rays, and the classes of good pixels."""
    rng = np.random.default_rng(seed)
    fluxes = 10 ** rng.uniform(3.0, 6.0, len(field.stars))
    truth = [field.scene(k, fluxes) for k in range(field.n_exp)]
    err = [np.sqrt(SKY_SIGMA**2 + np.maximum(t, 0.0)) for t in truth]
    dirty, hits, detectable = [], [], []
    for k in range(field.n_exp):
        hit = rng.uniform(size=field.shape) < cr_fraction
        amp = 10 ** rng.uniform(np.log10(10 * SKY_SIGMA), np.log10(1000 * SKY_SIGMA), field.shape)
        dirty.append(truth[k] + rng.normal(size=field.shape) * err[k] + hit * amp)
        hits.append(hit)
        detectable.append(hit & (amp >= 8 * err[k]))
    inner = np.zeros(field.shape, dtype=bool)
    inner[8:-8, 8:-8] = True
    y, x = np.indices(field.shape)
    classes = []
    for k in range(field.n_exp):
        dx, dy = field.shifts[k]
        dist = field.distance_to_stars(k)
        others = np.vstack([np.column_stack([x[hits[m]] + field.shifts[m][0], y[hits[m]] + field.shifts[m][1]])
                            for m in range(field.n_exp) if m != k])
        # sky distance to the nearest cosmic ray of another exposure
        d_cr = cKDTree(others).query(np.column_stack([x.ravel() + dx, y.ravel() + dy]))[0].reshape(field.shape)
        # leave out each cosmic ray and its neighbors: jwst's rule flags those on purpose
        good = inner & ~ndimage.binary_dilation(hits[k], np.ones((3, 3)))
        classes.append({
            "near star": inner & (dist <= 3.0),
            "core": good & (dist <= 1.0),
            "wings": good & (dist > 1.0) & (dist <= 8.0) & (d_cr > 2.0),
            "sky": good & (dist > 8.0) & (d_cr > 2.0),
            "near another cosmic ray": good & (dist > 8.0) & (d_cr <= 2.0),
        })
    return dirty, err, [d & inner for d in detectable], classes


def part1(field):
    schemes = [
        ("today: median of single images, scale 1, pixfrac 1", reference_today),
        ("median of input pixels, scale 0.67, pixfrac 1", reference_from_input_pixels(2 / 3, 1.0)),
        ("median of input pixels, scale 0.5, pixfrac 1", reference_from_input_pixels(0.5, 1.0)),
        ("median of input pixels, scale 0.5, pixfrac 0.8", reference_from_input_pixels(0.5, 0.8)),
        ("median of input pixels, scale 0.5, pixfrac 0.6", reference_from_input_pixels(0.5, 0.6)),
        ("clipped mean, scale 0.5, pixfrac 1", reference_from_input_pixels(0.5, 1.0, clipped=True)),
        ("clipped mean, scale 0.5, pixfrac 0.8", reference_from_input_pixels(0.5, 0.8, clipped=True)),
        ("clipped mean, scale 0.5, pixfrac 0.8, plus correction", reference_from_input_pixels(0.5, 0.8, True, True)),
        ("clipped mean, scale 0.5, pixfrac 1, plus correction", reference_from_input_pixels(0.5, 1.0, True, True)),
        ("median, scale 0.5, pixfrac 0.6, plus correction", reference_from_input_pixels(0.5, 0.6, correct=True)),
    ]
    print(f"\nPart 1: {field.n_exp} exposures, {len(field.stars)} stars, no noise; "
          "(data - prediction) / data at each star's brightest pixel\n")
    print("| reference image | median | 5% | 95% | largest |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, make in schemes:
        e = peak_errors(field, make)
        print(f"| {name} | {np.median(e):+.3f} | {np.percentile(e, 5):+.3f} | {np.percentile(e, 95):+.3f} | "
              f"{e[np.argmax(np.abs(e))]:+.3f} |", flush=True)


def part2(field, seeds, cr_fraction, defaults=False):
    configs = [
        ("today", reference_today, (1.2, 0.7)),
        ("today", reference_today, (0.5, 0.3)),
        ("clipped mean, scale 0.67, pixfrac 1", reference_from_input_pixels(2 / 3, 1.0, clipped=True), (1.2, 0.7)),
        ("clipped mean, scale 0.67, pixfrac 1", reference_from_input_pixels(2 / 3, 1.0, clipped=True), (0.5, 0.3)),
        ("clipped mean, scale 0.5, pixfrac 1", reference_from_input_pixels(0.5, 1.0, clipped=True), (1.2, 0.7)),
        ("clipped mean, scale 0.5, pixfrac 1", reference_from_input_pixels(0.5, 1.0, clipped=True), (0.5, 0.3)),
        ("clipped mean, scale 0.5, pixfrac 1", reference_from_input_pixels(0.5, 1.0, clipped=True), (0.3, 0.2)),
        ("clipped mean, scale 0.5, pixfrac 0.8", reference_from_input_pixels(0.5, 0.8, clipped=True), (0.3, 0.2)),
        ("clipped mean, scale 0.5, pixfrac 0.8, plus correction", reference_from_input_pixels(0.5, 0.8, True, True), (1.2, 0.7)),
        ("clipped mean, scale 0.5, pixfrac 0.8, plus correction", reference_from_input_pixels(0.5, 0.8, True, True), (0.5, 0.3)),
        ("clipped mean, scale 0.5, pixfrac 1, plus correction", reference_from_input_pixels(0.5, 1.0, True, True), (0.5, 0.3)),
        ("clipped mean, scale 0.5, pixfrac 0.6, plus correction", reference_from_input_pixels(0.5, 0.6, True, True), (0.5, 0.3)),
        ("clipped mean, scale 0.5, pixfrac 0.8, plus correction", reference_from_input_pixels(0.5, 0.8, True, True), "measured"),
        ("clipped mean, scale 0.5, pixfrac 0.8, plus correction", reference_from_input_pixels(0.5, 0.8, True, True), "robust"),
        ("clipped mean, scale 0.5, pixfrac 1, plus correction", reference_from_input_pixels(0.5, 1.0, True, True), "measured"),
        ("clipped mean, scale 0.5, pixfrac 1", reference_from_input_pixels(0.5, 1.0, clipped=True), "measured"),
    ]
    if defaults:
        proposed = "clipped mean, scale 0.5, pixfrac 0.8, plus correction"
        configs = [config for config in configs if config[0] in ("today", proposed)]
    realizations = [realization(field, seed, cr_fraction) for seed in seeds]
    print(f"\nPart 2: {len(field.stars)} stars of 1e3-1e6 electrons, noise {SKY_SIGMA:.0f} e- plus Poisson, cosmic rays on "
          f"{cr_fraction:.1%} of the pixels (10-1000 x {SKY_SIGMA:.0f} e-); {len(seeds)} realizations of {field.n_exp} exposures")
    print("flagged where |data - prediction| > scale * D + snr * err (jwst's rule), or > snr * sqrt(err**2 + model**2) "
          "(measured); "
          "cosmic rays counted if at least 8 x the local noise\n")
    print("| reference image | scale | cosmic rays found | ... within 3 px of a star | good pixels flagged per 10^4: sky "
          "| next to another exposure's cosmic ray | star wings (1-8 px) | star-core pixels flagged (within 1 px) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, make, scale in configs:
        measured = scale in ("measured", "robust")
        found = n_found = near = n_near = 0
        flagged = dict.fromkeys(("sky", "near another cosmic ray", "wings", "core"), 0)
        counted = dict.fromkeys(flagged, 0)
        for dirty, err, detectable, classes in realizations:
            ref, s, model2 = make(field, dirty, err)
            for k in range(field.n_exp):
                prediction = blot(ref, field.pixmap(k, s))
                if measured:
                    model = np.sqrt(blot(model2[scale], field.pixmap(k, s)))
                    flags = flag_measured(dirty[k], err[k], prediction, model)
                else:
                    flags = flag(dirty[k], err[k], prediction, scale)
                found += np.sum(flags & detectable[k])
                n_found += np.sum(detectable[k])
                near += np.sum(flags & detectable[k] & classes[k]["near star"])
                n_near += np.sum(detectable[k] & classes[k]["near star"])
                for c in flagged:
                    flagged[c] += np.sum(flags & classes[k][c])
                    counted[c] += np.sum(classes[k][c])
        rate = {c: flagged[c] / counted[c] for c in flagged}
        tolerance = scale if measured else f"{scale[0]} / {scale[1]}"
        print(f"| {name} | {tolerance} | {found / n_found:.1%} | {near / max(1, n_near):.1%} ({n_near}) | "
              f"{1e4 * rate['sky']:.1f} | {1e4 * rate['near another cosmic ray']:.0f} | {1e4 * rate['wings']:.0f} | "
              f"{rate['core']:.1%} |", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--filter", default="F070W")
    parser.add_argument("--detector", default="NRCA1")
    parser.add_argument("--instrument", default="NIRCam", help="stpsf's class: NIRCam, NIRISS, NIRSpec or MIRI")
    parser.add_argument("--exposures", type=int, default=4)
    parser.add_argument("--size", type=int, default=192, help="pixels on a side of each exposure")
    parser.add_argument("--stars", type=int, default=160)
    parser.add_argument("--layout-seed", type=int, default=7, help="seed of the dither offsets and star positions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13], help="seeds of the noise realizations")
    parser.add_argument("--cr-fraction", type=float, default=0.005, help="fraction of pixels hit by a cosmic ray")
    parser.add_argument("--part", choices=["1", "2", "both"], default="both")
    parser.add_argument("--defaults", action="store_true",
                        help="part 2 for the proposed defaults only, with both tolerances, against today's")
    args = parser.parse_args()

    psf = pixel_summed_psf(args.filter, args.detector, args.instrument)
    print(f"stpsf {stpsf.__version__} {args.instrument} {args.filter} {args.detector}, OVERDIST, oversampled {OVERSAMPLE}x: "
          f"FWHM {fwhm(psf):.2f} pixels, {psf(np.array([0.0]), np.array([0.0]))[0, 0]:.0%} of the light in the "
          "central pixel when centered")
    field = Field(psf, args.exposures, args.size, args.stars, args.layout_seed)
    if args.part in ("1", "both"):
        part1(field)
    if args.part in ("2", "both"):
        part2(field, args.seeds, args.cr_fraction, args.defaults)


if __name__ == "__main__":
    main()
