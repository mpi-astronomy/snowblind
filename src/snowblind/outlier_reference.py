"""Outlier detection against one reference built from all exposures' input pixels, in tiles.

`TiledOutlierStep` flags outliers in a set of images this way, as jwst's
outlier detection does for images with its own reference.

jwst's outlier detection drizzles each exposure alone, takes the median of
those images and blots it back: JWST PSFs are undersampled, so that blurs
star cores, and its threshold is loosened near stars (``scale * D``), which
hides cosmic rays there.  Here the reference is built from the input pixels
of all exposures together, at their true positions, on the outlier mosaic's
grid sampled ``oversample`` times more finely:

1. each exposure's input pixels are drizzled onto that grid with spizzle's
   operator (square kernel, ``pixfrac``), after its sky level is
   subtracted, weighted by their overlap times their inverse read-noise
   variance (zero for DO_NOT_USE);
2. for each fine pixel, the input pixels that land on it are combined:
   their weighted median, then their weighted mean within 3 sigma of it,
   three times; where fewer than three exposures land, their weighted mean
   instead, since a median of two exposures takes one of them, outlier and
   all, while their mean halves an outlier, as jwst's median of two images
   does;
3. the exposures' residuals (an input pixel minus the reference blotted at
   its center) are combined by their weighted median and added once
   (``correct``), where at least three exposures land: with two, the
   correction pulls the reference toward both, an outlier included;
4. the reference is blotted onto every exposure pixel with spizzle's
   ``blot``, and the exposure is compared with it by jwst's two-pass rule.
   Where fewer than three exposures cover a pixel, only pixels above the
   prediction are flagged: a median of two cannot tell which one holds the
   outlier, and cosmic rays are positive.

The fine grid is never allocated: it is processed in tiles of ``tile``
fine pixels with a halo of ``halo``.  For each tile, only the input pixels
landing in it and its halo are drizzled, from geometry computed once per
exposure, and each exposure pixel whose center lies in the tile gets its
prediction; then the tile is dropped.  Memory is one tile's input pixels
plus each exposure's own arrays.  The tile builder and the combinations
are planned for spizzle once the method is settled.

spizzle is an optional dependency, ``pip install 'snowblind[spizzle]'``.
"""
import logging

import numpy as np
from astropy.stats import sigma_clipped_stats
from jwst.datamodels import ModelLibrary
from jwst.lib.pipe_utils import match_nans_and_flags
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep
from jwst.resample.resample_utils import resampled_wcs_from_models
from jwst.stpipe.utilities import record_step_status
from scipy import ndimage
from stcal.outlier_detection.utils import flag_resampled_crs
from stdatamodels.jwst.datamodels import dqflags

from .step import Step

try:
    import spizzle
    from numba import njit, prange
except ImportError:
    # spizzle, which brings numba, is optional: without it this module still imports, its kernels stay plain
    # Python, and Exposure raises before they would run
    spizzle = None
    prange = range

    def njit(*args, **kwargs):
        """Leave the function as it is, numba being missing."""
        return lambda function: function

__all__ = ["TiledOutlierStep", "Exposure", "predict", "run", "flag_outliers", "sky_level", "weighted_median",
           "clipped_mean"]

log = logging.getLogger(__name__)

DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
OUTLIER = dqflags.pixel["OUTLIER"]
# rows of fine pixels each thread combines with one buffer
CHUNK = 4096
# input pixels of margin around the pixels of a tile, for the footprints that reach its halo
MARGIN = 3
# spacing in input pixels of the centers sampled to find the input pixels that reach each tile
SAMPLING = 8
# good pixels sampled for the sky level
SKY_SAMPLE = 200_000
# fine pixels beyond a tile's halo within which an input pixel center may still drop into the halo
REACH = 2.0


@njit(cache=True)
def _insert(v, w, s, n, x, weight, sigma):
    """Insert ``x`` with its weight and sigma into the first ``n`` entries of v, w and s, kept sorted by value."""
    q = n
    while q > 0 and v[q - 1] > x:
        v[q] = v[q - 1]
        w[q] = w[q - 1]
        s[q] = s[q - 1]
        q -= 1
    v[q] = x
    w[q] = weight
    s[q] = sigma


@njit(cache=True)
def _sorted_median(v, w, n):
    """The weighted median of the first ``n`` values of v, sorted: the lowest at which the cumulative weight reaches
    half."""
    total = 0.0
    for q in range(n):
        total += w[q]
    cum = 0.0
    for q in range(n):
        cum += w[q]
        if cum >= 0.5 * total:
            return v[q]
    return v[n - 1]


@njit(parallel=True, cache=True)
def _combine(n_cols, boxes, ptr_start, indptr, indices, weights, values, sigma, longest, median_only, spread,
             fewest_for_median, out, model2, count):
    """For each fine pixel of a tile, combine the input pixels of every part (exposure) that land on it.

    Part ``p`` covers the box ``boxes[p] = (y0, x0, ny, nx)`` of the tile; its rows start at ``ptr_start[p]`` in
    ``indptr``, whose values index ``indices`` and ``weights``, and ``indices`` index ``values`` and ``sigma``.
    ``out`` gets the weighted median, or with ``median_only`` False the clipped mean from it, and their weighted
    mean where fewer than ``fewest_for_median`` parts land; ``model2`` with ``spread`` the robust spread of the
    input pixels; ``count`` the number of parts with an input pixel there.
    """
    n_rows = out.size
    n_parts = boxes.shape[0]
    for c in prange((n_rows + CHUNK - 1) // CHUNK):
        v = np.empty(longest)
        w = np.empty(longest)
        s = np.empty(longest)
        dv = np.empty(longest)
        dw = np.empty(longest)
        ds = np.empty(longest)
        for o in range(c * CHUNK, min((c + 1) * CHUNK, n_rows)):
            oy = o // n_cols
            ox = o - oy * n_cols
            n = 0
            reached = 0
            for p in range(n_parts):
                by, bx, bny, bnx = boxes[p, 0], boxes[p, 1], boxes[p, 2], boxes[p, 3]
                if oy < by or oy >= by + bny or ox < bx or ox >= bx + bnx:
                    continue
                r = ptr_start[p] + (oy - by) * bnx + (ox - bx)
                if indptr[r + 1] > indptr[r]:
                    reached += 1
                for e in range(indptr[r], indptr[r + 1]):
                    x = values[indices[e]]
                    if np.isnan(x) or not weights[e] > 0.0:
                        continue
                    _insert(v, w, s, n, x, weights[e], sigma[indices[e]])
                    n += 1
            count[o] = reached
            model2[o] = np.nan
            if n == 0:
                out[o] = np.nan
                continue
            if reached < fewest_for_median:
                num = 0.0
                den = 0.0
                for q in range(n):
                    num += w[q] * v[q]
                    den += w[q]
                est = num / den
            else:
                est = _sorted_median(v, w, n)
                if not median_only:
                    for _ in range(3):
                        num = 0.0
                        den = 0.0
                        for q in range(n):
                            if abs(v[q] - est) <= 3.0 * s[q]:
                                num += w[q] * v[q]
                                den += w[q]
                        if den > 0.0:
                            est = num / den
            out[o] = est
            if spread and not median_only:
                for q in range(n):
                    _insert(dv, dw, ds, q, abs(v[q] - est), w[q], 0.0)
                mad = _sorted_median(dv, dw, n)
                for q in range(n):
                    _insert(dv, dw, ds, q, s[q] * s[q], w[q], 0.0)
                model2[o] = max(0.0, (1.4826 * mad) ** 2 - _sorted_median(dv, dw, n))


def _combine_parts(parts, shape, values, sigma, median_only, spread=False, fewest_for_median=3):
    """Combine the input pixels of ``parts`` (spizzle operators on a tile of ``shape``) by fine pixel.

    ``values`` and ``sigma`` hold the parts' input pixels one after the other, in their operators' column order.
    Where fewer than ``fewest_for_median`` parts land, their weighted mean is taken instead.  Returns the
    combination, the spread squared (NaN without ``spread``) and the count of parts, each of ``shape``.
    """
    boxes = np.zeros((len(parts), 4), dtype=np.int64)
    indptrs, indices, weights = [], [], []
    ptr_start = np.zeros(len(parts), dtype=np.int64)
    n_ptr = n_entries = n_values = longest = 0
    for p, op in enumerate(parts):
        (ys, xs) = op.box
        boxes[p] = (ys.start, xs.start, ys.stop - ys.start, xs.stop - xs.start)
        m = op.matrix
        ptr_start[p] = n_ptr
        indptrs.append(m.indptr.astype(np.int64) + n_entries)
        indices.append(m.indices.astype(np.int64) + n_values)
        weights.append(m.data)
        n_ptr += m.indptr.size
        n_entries += m.data.size
        n_values += m.shape[1]
        longest += int(np.diff(m.indptr).max()) if m.indptr.size > 1 else 0
    out = np.empty(shape[0] * shape[1])
    model2 = np.empty_like(out)
    count = np.zeros(out.size, dtype=np.int32)
    if parts:
        _combine(
            shape[1], boxes, ptr_start, np.concatenate(indptrs), np.concatenate(indices), np.concatenate(weights),
            np.asarray(values, dtype=np.float64), np.asarray(sigma, dtype=np.float64), max(1, longest),
            median_only, spread, fewest_for_median, out, model2, count,
        )
    else:
        out[:] = np.nan
        model2[:] = np.nan
    return out.reshape(shape), model2.reshape(shape), count.reshape(shape)


class _Whole:
    """One spizzle-like operator part covering a whole sparse array, for the public combinations."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.box = (slice(0, matrix.shape[0]), slice(0, 1))


def weighted_median(W, values):
    """For each row of the sparse array ``W``, the median of the ``values`` it lists, weighted by its entries.

    Parameters
    ----------
    W : scipy.sparse.csr_array, shape (n_out, n_in)
        Row ``o`` lists the input values that land on output pixel ``o``, with their weights.
    values : ndarray, shape (n_in,)
        The input values; NaN ones are left out.

    Returns
    -------
    ndarray, shape (n_out,)
        The lowest value at which the cumulative weight reaches half of the row's; NaN for a row without values.
    """
    out, _, _ = _combine_parts([_Whole(W)], (W.shape[0], 1), values, np.ones(W.shape[1]), median_only=True,
                               fewest_for_median=0)
    return out.ravel()


def clipped_mean(W, values, sigma):
    """For each row of ``W``, the weighted mean of its values within 3 sigma of the estimate, three times from their
    weighted median.

    Parameters
    ----------
    W : scipy.sparse.csr_array, shape (n_out, n_in)
        As for `weighted_median`.
    values, sigma : ndarray, shape (n_in,)
        The input values and their noise.

    Returns
    -------
    ndarray, shape (n_out,)
    """
    out, _, _ = _combine_parts([_Whole(W)], (W.shape[0], 1), values, sigma, median_only=False, fewest_for_median=0)
    return out.ravel()


def sky_level(model):
    """The sky level of an exposure, to subtract before combining it with others.

    ``meta.background.level`` when the skymatch step measured it without
    subtracting it, zero when it subtracted it, and otherwise the
    sigma-clipped median of SKY_SAMPLE good pixels, evenly spaced.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.ImageModel`

    Returns
    -------
    float
    """
    background = model.meta.background
    if background.level is not None:
        return 0.0 if background.subtracted else float(background.level)
    data = model.data[_good(model)]
    return float(sigma_clipped_stats(data[::max(1, data.size // SKY_SAMPLE)], sigma=3.0, maxiters=5)[1])


def _good(model):
    """The pixels that are used: finite data and error, without DO_NOT_USE."""
    return np.isfinite(model.data) & np.isfinite(model.err) & ((model.dq & DO_NOT_USE) == 0)


def _tile_ids(x, y, tile, n_tiles):
    """The flat index of the tile holding each fine position, -1 off the fine grid or without a position."""
    ix = np.floor((x + 0.5) / tile)
    iy = np.floor((y + 0.5) / tile)
    valid = (ix >= 0) & (ix < n_tiles[1]) & (iy >= 0) & (iy < n_tiles[0])
    return np.where(valid, iy * n_tiles[1] + ix, -1).astype(np.int64)


class Exposure:
    """What the reference needs of one exposure, computed once, and the prediction it gets.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.ImageModel`
        The exposure.
    mosaic_wcs : gwcs.wcs.WCS
        The outlier mosaic's WCS, with its ``array_shape``.
    oversample : int
        Fine pixels per mosaic pixel along each axis.
    pixfrac : float
        Size of the drizzled input pixels relative to their own.
    tile, halo : int
        Fine pixels on a side of a tile, and of its halo.
    """

    def __init__(self, model, mosaic_wcs, oversample=2, pixfrac=0.8, tile=1024, halo=4):
        if spizzle is None:
            raise ImportError("TiledOutlierStep and outlier_reference need spizzle: pip install 'snowblind[spizzle]'")
        shape = model.data.shape
        ny_m, nx_m = mosaic_wcs.array_shape
        self.n_tiles = (-(-oversample * ny_m // tile), -(-oversample * nx_m // tile))
        self.fine_shape = (oversample * ny_m, oversample * nx_m)
        pixmap = spizzle.pixel_map(model.meta.wcs, mosaic_wcs, shape=shape)
        self.centers = oversample * (pixmap + 0.5) - 0.5
        del pixmap
        corners = spizzle.corners(self.centers, pixfrac)
        self.jacobian = spizzle.jacobian(corners).astype(np.float32)
        # the corners as offsets from the center: small, so float32 keeps them to about 1e-7 fine pixel
        self.offsets = (corners - self.centers[:, :, None, :]).astype(np.float32)
        del corners
        good = _good(model)
        var_rnoise = model.var_rnoise
        weighted = good & np.isfinite(var_rnoise) & (var_rnoise > 0)
        self.weights = np.zeros(shape, dtype=np.float32)
        self.weights[weighted] = 1.0 / var_rnoise[weighted]
        self.sky = sky_level(model)
        # the pixels used, with their data and error; NaN for the others
        self.data = np.where(weighted, model.data, np.nan).astype(np.float32)
        self.err = np.where(weighted, model.err, np.nan).astype(np.float32)
        # the tile of each pixel center, which gives the pixel its prediction
        self.tile_of = _tile_ids(self.centers[..., 0], self.centers[..., 1], tile, self.n_tiles).astype(np.int32)
        # the input pixels drizzled into each tile and its halo: the box of the centers within its reach, from the
        # centers every SAMPLING pixels (and the last row and column), widened by SAMPLING and MARGIN
        rows = np.r_[0:shape[0]:SAMPLING, shape[0] - 1]
        cols = np.r_[0:shape[1]:SAMPLING, shape[1] - 1]
        sampled = self.centers[np.ix_(rows, cols)]
        reach = halo + REACH
        boxes = {}
        for dx in (-reach, reach):
            for dy in (-reach, reach):
                ids = _tile_ids(sampled[..., 0] + dx, sampled[..., 1] + dy, tile, self.n_tiles)
                for t in np.unique(ids[ids >= 0]).tolist():
                    r, c = np.nonzero(ids == t)
                    box = boxes.get(t, (shape[0], 0, shape[1], 0))
                    boxes[t] = (min(box[0], rows[r.min()]), max(box[1], rows[r.max()]),
                                min(box[2], cols[c.min()]), max(box[3], cols[c.max()]))
        pad = SAMPLING + MARGIN
        self.regions = {
            t: (slice(max(0, r0 - pad), min(shape[0], r1 + 1 + pad)), slice(max(0, c0 - pad), min(shape[1], c1 + 1 + pad)))
            for t, (r0, r1, c0, c1) in boxes.items()
        }
        self.prediction = np.full(shape, np.nan, dtype=np.float32)
        self.spread = np.full(shape, np.nan, dtype=np.float32)
        self.n_exposures = np.zeros(shape, dtype=np.uint8)

    def pixels(self, t, region):
        """The flat indices of the pixels whose centers lie in tile ``t``, all within ``region``."""
        local = np.flatnonzero(self.tile_of[region] == t)
        width = region[1].stop - region[1].start
        return (region[0].start + local // width) * self.tile_of.shape[1] + region[1].start + local % width

    def operator(self, region, origin, shape):
        """The spizzle operator of the input pixels of ``region`` onto a tile of ``shape`` at fine ``origin``."""
        corners = (self.centers[region][:, :, None, :] + self.offsets[region]) - origin
        triplets = spizzle.overlap_triplets(corners, shape)
        if triplets[0].size == 0:
            return None
        return spizzle.build_operator(triplets, self.jacobian[region], shape, self.weights[region])


def _tile(exposures, t, tile, halo, correct, spread):
    """Build the reference of tile ``t`` and write the predictions of the exposure pixels whose centers lie in it."""
    n_tiles = exposures[0].n_tiles
    fine_shape = exposures[0].fine_shape
    ty, tx = divmod(t, n_tiles[1])
    origin = np.array([tx * tile - halo, ty * tile - halo], dtype=np.float64)
    shape = (min(tile, fine_shape[0] - ty * tile) + 2 * halo, min(tile, fine_shape[1] - tx * tile) + 2 * halo)
    parts, regions, users = [], [], []
    for exposure in exposures:
        region = exposure.regions.get(t)
        if region is None:
            continue
        op = exposure.operator(region, origin, shape)
        if op is None:
            continue
        parts.append(op)
        regions.append(region)
        users.append(exposure)
    if not parts:
        return
    values = np.concatenate([(e.data[r] - e.sky).ravel() for e, r in zip(users, regions)])
    sigma = np.concatenate([e.err[r].ravel() for e, r in zip(users, regions)])
    reference, model2, count = _combine_parts(parts, shape, values, sigma, median_only=False, spread=spread)
    if correct:
        residuals = np.concatenate([
            (e.data[r] - e.sky - spizzle.blot(reference, e.centers[r] - origin)).ravel() for e, r in zip(users, regions)
        ])
        correction, _, _ = _combine_parts(parts, shape, residuals, sigma, median_only=True)
        reference += np.where(count >= 3, np.nan_to_num(correction), 0.0)
    for exposure, region in zip(users, regions):
        pix = exposure.pixels(t, region)
        if pix.size == 0:
            continue
        positions = exposure.centers.reshape(-1, 2)[pix][:, None, :] - origin
        exposure.prediction.flat[pix] = spizzle.blot(reference, positions)[:, 0] + exposure.sky
        if spread:
            exposure.spread.flat[pix] = np.sqrt(spizzle.blot(model2, positions)[:, 0])
        i = np.clip(np.floor(positions[:, 0, 0]).astype(np.int64), 0, shape[1] - 2)
        j = np.clip(np.floor(positions[:, 0, 1]).astype(np.int64), 0, shape[0] - 2)
        fewest = np.minimum(np.minimum(count[j, i], count[j, i + 1]), np.minimum(count[j + 1, i], count[j + 1, i + 1]))
        exposure.n_exposures.flat[pix] = np.minimum(fewest, 255)


def predict(models, mosaic_wcs, oversample=2, pixfrac=0.8, correct=True, tile=1024, halo=4, spread=False):
    """The prediction of every exposure from the reference of all of them, tile by tile.

    Parameters
    ----------
    models : list of `~stdatamodels.jwst.datamodels.ImageModel`
        The exposures.
    mosaic_wcs : gwcs.wcs.WCS
        The outlier mosaic's WCS, with its ``array_shape``: the grid that
        the reference samples ``oversample`` times more finely.
    oversample : int
        Fine pixels per mosaic pixel along each axis.
    pixfrac : float
        Size of the drizzled input pixels relative to their own.
    correct : bool
        Add the exposures' residuals, combined by their weighted median, once, where at least three exposures land.
    tile, halo : int
        Fine pixels on a side of a tile, and of its halo.
    spread : bool
        Also measure the robust spread of the input pixels of each fine
        pixel, for the measured tolerance of `flag_outliers`.

    Returns
    -------
    list of `Exposure`
        One per model, with its ``prediction`` (the sky added back),
        ``spread`` and ``n_exposures`` (the fewest exposures covering the
        fine pixels its prediction comes from).
    """
    exposures = [Exposure(m, mosaic_wcs, oversample, pixfrac, tile, halo) for m in models]
    run(exposures, correct, tile, halo, spread)
    return exposures


def run(exposures, correct=True, tile=1024, halo=4, spread=False):
    """Build the reference tile by tile and write each exposure's prediction, as `predict` does.

    Parameters
    ----------
    exposures : list of `Exposure`
        Made with the same mosaic WCS, ``oversample``, ``tile`` and ``halo``.
    correct, tile, halo, spread
        As for `predict`.
    """
    tiles = sorted(set().union(*(e.regions for e in exposures)))
    log.info(f"Reference of {len(exposures)} exposures in {len(tiles)} tiles of {tile} fine pixels")
    for t in tiles:
        _tile(exposures, t, tile, halo, correct, spread)


def flag_outliers(exposure, snr=(5.0, 4.0), scale=(0.5, 0.3)):
    """The outliers of one exposure against its prediction, by jwst's two-pass rule.

    Where fewer than three exposures cover a pixel, only pixels above the
    prediction are flagged.

    Parameters
    ----------
    exposure : `Exposure`
        From `predict`.
    snr : pair of float
        Thresholds in noise of the two passes.
    scale : pair of float or None
        jwst's ``scale`` of the two passes; None uses the measured spread in
        its place, ``|data - prediction| > snr * sqrt(err**2 + spread**2)``
        (`predict` with ``spread``).

    Returns
    -------
    ndarray of bool, the exposure's shape
    """
    data, err = exposure.data, exposure.err
    good = np.isfinite(data)
    prediction = exposure.prediction
    if scale is None:
        diff = np.abs(data - prediction)
        noise = np.sqrt(np.nan_to_num(err) ** 2 + np.nan_to_num(exposure.spread) ** 2)
        first = ndimage.binary_dilation(diff > snr[0] * noise, np.ones((3, 3), dtype=bool))
        flags = first & (diff > snr[1] * noise)
    else:
        flags = flag_resampled_crs(data, err, prediction, snr[0], snr[1], scale[0], scale[1], 0.0)
    flags &= (exposure.n_exposures >= 3) | (data > prediction)
    return flags & good & np.isfinite(prediction)


class TiledOutlierStep(Step):
    """Flag outliers in images against one reference built from all their input pixels, in tiles.

    The reference, the prediction of each exposure and the flags are those
    of this module (`predict` and `flag_outliers`), on the outlier mosaic's
    grid that jwst's outlier detection uses (the input pixel scale, from
    ``resampled_wcs_from_models``) sampled twice as finely, with pixfrac 0.8
    and the correction step.  The outliers get DO_NOT_USE and OUTLIER in
    ``dq``, and NaN in the data, error and variance arrays, as jwst's step
    sets them.

    The parameters are jwst's ``snr`` and ``scale``; ``scale`` is 0.5 / 0.3,
    which the sharper reference allows for every NIRCam, NIRISS and NIRSpec
    imaging filter simulated.  The grid, pixfrac and the correction step are
    fixed.  The step does not read jwst's step parameters from CRDS, and runs
    on images only: it raises for spectra, coronagraphy and time series.
    """

    class_alias = "tiled_outlier"
    fits_keyword = "S_TILOUT"
    fits_comment = "Outliers: snowblind.TiledOutlierStep"

    spec = """
        snr = string(default='5.0 4.0')  # Thresholds in noise of the two passes
        scale = string(default='0.5 0.3')  # Factors of the prediction's largest difference from its neighbors added to them
        search_output_file = boolean(default=False)
        in_memory = boolean(default=True)  # in_memory flag ignored if run within the pipeline; set at pipeline level instead
    """  # noqa: E501

    def process(self, input_data):
        """Flag the outliers of every input image.

        Parameters
        ----------
        input_data : asn file or `~jwst.datamodels.library.ModelLibrary`
            The images.

        Returns
        -------
        `~jwst.datamodels.library.ModelLibrary`
            The images with their outliers flagged.
        """
        # jwst's mode, from the pipeline's ``mode`` attribute or the exposure type, and the association ID of the
        # output names; both are private methods of jwst's step
        mode = OutlierDetectionStep._guess_mode(self, input_data)
        if mode != "imaging":
            raise ValueError(f"TiledOutlierStep supports images only, not mode {mode!r}")
        library = self.prepare_output(input_data, open_models=False)
        if not isinstance(library, ModelLibrary):
            library = ModelLibrary(library, on_disk=not self.in_memory)
        OutlierDetectionStep._get_asn_id(self, input_data)
        if len(library) < 2:
            log.warning(f"Input only contains {len(library)} exposures; outlier detection is skipped")
            record_step_status(library, "outlier_detection", False)
            return library

        snr = [float(v) for v in self.snr.split()]
        scale = [float(v) for v in self.scale.split()]
        mosaic_wcs = resampled_wcs_from_models(library, pixel_scale_ratio=1.0)[0]
        exposures = []
        with library:
            for model in library:
                exposures.append(Exposure(model, mosaic_wcs))
                library.shelve(model, modify=False)
        run(exposures)
        with library:
            for model, exposure in zip(library, exposures):
                flags = flag_outliers(exposure, snr=snr, scale=scale)
                model.dq[flags] |= np.uint32(DO_NOT_USE | OUTLIER)
                log.info(f"{np.count_nonzero(flags)} pixels marked as outliers in {model.meta.filename}")
                match_nans_and_flags(model)
                self.set_cal_step(model, "COMPLETE")
                library.shelve(model, modify=True)
        record_step_status(library, "outlier_detection", True)
        return library
