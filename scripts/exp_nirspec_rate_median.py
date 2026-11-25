"""Banding left by msaexp's column median on the rate image and by the stream clean, on a NIRSpec MOS or IFU exposure.

The column median is grizli's ``exposure_oneoverf_correction`` (grizli
754cb05; ``axis=0``, no erosion) on the uncleaned OLS_C rate, with the
settings msaexp (db7e985) passes. Pixels flagged in DQ (JUMP_DET aside)
and pixels more than 5, then 4, then 3 sigma from the column model, grown
by 3 pixels, are left out; the median of the rest of each science-frame
column (a detector readout row, all 2048 rows), less a smooth fit across
the columns, is subtracted from the whole column. The medians are taken
after subtracting the SExtractor background grizli gets from ``sep`` in
meshes of ``deg_pix`` pixels, here from photutils.

* ``msaexp_ifu`` (``ifu.detector_corrections``): only pixels outside the
  slice footprints grown by 4 pixels (msaexp's s-flat mask; the WCS
  footprints here), unflagged and above -10 sigma; ``deg_pix=2048``, so
  one background level and only the median of the column medians kept.
* ``msaexp_ifu_wcs``: the same with the footprints not grown.
* ``msaexp_mos`` (``pipeline.exposure_oneoverf``): no mask besides the
  DQ and the sigma clipping; ``deg_pix=256``, so a Chebyshev fit of
  degree 8 across the columns is kept.

IFU exposures get the two IFU recipes, MOS exposures ``msaexp_ifu_wcs``
and ``msaexp_mos``.

The stream clean runs with the step's defaults on the ``_jump`` ramp of an
earlier ``exp_nirspec_variants.py`` or ``make_nirspec_field_panels.py`` run
and is fitted with OLS_C. ``--stream-variants`` adds variants of it:
``min_period=1200``, ``fit_method='row_median'`` (one value per readout
row and channel), and the footprints grown by 4 pixels for its fit
(``stream_clean_footprints_grown4``).

Both corrections fit the banding on the pixels they would be measured on,
so the background pixels are split into blocks of ``--block`` rows. Each
correction is fitted with every other block left out, and the banding is
measured on the blocks left out, which stand in for the slits or slices:
background pixels outside the footprints and the stream clean's scene
mask. With ``--sets 2`` every other block is held out and the two sets
swap roles; with ``--sets 4`` one block in four is held out, so the fits
keep three quarters of the background, and each quarter takes its turn.

Two banding measures, in DN/s, each with its white floor:

* the column-mean scatter of the docs: robust sigma of the science-frame
  column means over the measured pixels; floor, the pixel scatter over the
  square root of the median number of measured pixels per column;
* the block-mean scatter, what a spectrum extracted over about ``--block``
  rows sees: the mean of the measured pixels of each column in each
  held-out block (blocks with at least 8 such pixels), less the median of
  its block row across the columns, then the standard deviation after one
  5 sigma clip; floor, the pixel scatter over the square root of each
  block's pixel count, combined in quadrature over the blocks.

``metrics.json`` (``metrics_<n>sets.json`` for other ``--sets``) holds
them per correction and set of held-out blocks.
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import scipy.ndimage as nd
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clip
from jwst.ramp_fitting import RampFitStep
from numpy.polynomial import Chebyshev
from photutils.background import Background2D, SExtractorBackground
from stdatamodels.jwst import datamodels

from snowblind import StreamCleanStep
from snowblind.metrics import robust_sigma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_nirspec_field_panels import banding_metrics, save_rate, usable_pixels  # noqa: E402

CORRECTIONS = {
    "NRS_IFU": ["none", "msaexp_ifu_wcs", "msaexp_ifu", "stream_clean"],
    "NRS_MSASPEC": ["none", "msaexp_ifu_wcs", "msaexp_mos", "stream_clean"],
}
JUMP_DET = 4
# msaexp's growth of its s-flat slice mask (load_ifu_sflat dilate_iterations)
SLICE_GROWTH = 4


# stream clean variants: step options, and how many pixels the slit or slice footprints are grown by for its fit
STREAM_CLEAN_VARIANTS = {
    "stream_clean": ({}, 0),
    "stream_clean_min_period1200": ({"min_period": 1200}, 0),
    "stream_clean_row_median": ({"fit_method": "row_median"}, 0),
    "stream_clean_footprints_grown4": ({}, 4),
}


class HeldOutStreamCleanStep(StreamCleanStep):
    """The stream clean with the ``excluded`` pixels removed from its background mask."""

    class_alias = "stream_clean"

    def _clean_with_mask(self, result, background_mask, flat):
        return super()._clean_with_mask(result, background_mask & ~self.excluded, flat)


def held_out_blocks(shape, block, part, nsets=2):
    """Boolean image, True on one block of ``block`` rows in every ``nsets``, starting with block ``part``."""
    rows = (np.arange(shape[0]) // block) % nsets == part
    return np.repeat(rows[:, None], shape[1], axis=1)


def block_metrics(rate, measured, block, min_pixels=8):
    """Block-mean scatter and its white floor over the ``measured`` pixels, in DN/s."""
    nblocks = rate.shape[0] // block
    shape = (nblocks, block, rate.shape[1])
    cut = np.where(measured, rate, np.nan)[: nblocks * block].reshape(shape)
    counts = measured[: nblocks * block].reshape(shape).sum(axis=1)
    means = np.nanmean(cut, axis=1)
    means[counts < min_pixels] = np.nan
    means -= np.nanmedian(means, axis=1, keepdims=True)
    keep = np.isfinite(means)
    keep[keep] = np.abs(means[keep]) <= 5 * robust_sigma(means[keep])
    return {
        "blocks": int(keep.sum()),
        "block_mean_scatter": float(np.std(means[keep])),
        "block_mean_white_floor": float(robust_sigma(cut) * np.sqrt(np.mean(1.0 / counts[keep]))),
    }


def _sextractor_background(sci, unusable, mesh):
    """SExtractor background in meshes of ``mesh`` pixels, as grizli gets it from ``sep``."""
    if mesh >= max(sci.shape):
        values = sigma_clip(sci[~unusable], sigma=3.0, maxiters=None, masked=False)
        return np.float32(SExtractorBackground(sigma_clip=None).calc_background(values))
    background = Background2D(
        sci, (mesh, mesh), mask=unusable, filter_size=(3, 3), exclude_percentile=100,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=None), bkg_estimator=SExtractorBackground(sigma_clip=None),
    )
    return background.background.astype(np.float32)


def grizli_oneoverf(rate, err, dq, manual_mask, deg_pix, erode=False, thresholds=(5, 4, 3), dilate_iterations=3):
    """``rate`` less grizli's column model (``axis=0``) fitted on the ``manual_mask`` pixels that pass its DQ and sigma cuts.

    ``erode`` erodes each sigma mask before growing it, as grizli does on imaging. Its row model (``axis=1``)
    is this function on the transposed arrays.
    """
    dqmask = ((dq - (dq & JUMP_DET)) == 0) & (err > 0) & np.isfinite(err)

    def grown(sn_mask):
        if erode:
            sn_mask = nd.binary_erosion(sn_mask)
        return nd.binary_dilation(sn_mask, iterations=dilate_iterations)

    sci = rate.astype(np.float32)
    back = _sextractor_background(sci, ~dqmask | ~np.isfinite(sci), deg_pix)
    mask = dqmask & ~grown((sci - back) / err > thresholds[0]) & manual_mask
    for thresh in thresholds:
        sci = rate - back
        sci[~mask] = np.nan
        med = np.nanmedian(sci, axis=0)
        sn = (sci - med) / err
        mask = dqmask & ~grown((sn > thresh) & dqmask) & (sn > -thresh) & manual_mask

    xarr = np.linspace(-1, 1, med.size)
    ok = np.isfinite(med)
    if deg_pix >= med.size:
        smooth = np.nanmedian(med)
    else:
        for _ in range(3):
            smooth = Chebyshev.fit(xarr[ok], med[ok], deg=med.size // deg_pix)(xarr)
            ok = np.isfinite(med) & (np.abs(med - smooth) < 0.05)
    model = med - smooth
    model[~np.isfinite(model)] = 0
    return rate - model[None, :]


def msaexp_ifu_mask(none, footprints):
    """msaexp's IFU fitting mask: outside ``footprints``, finite, without DO_NOT_USE, above -10 sigma."""
    return ~footprints & np.isfinite(none["rate"]) & (none["dq"] & 1 == 0) & (none["rate"] > -10 * none["err"])


def uncleaned_rate(jump_path, out_dir):
    """Rate, dq and err of the uncleaned ramp fitted with OLS_C, cached as an npz file."""
    path = os.path.join(out_dir, "none_ols.npz")
    if not os.path.exists(path):
        with datamodels.open(jump_path) as jump:
            rate, _ = RampFitStep.call(jump, algorithm="OLS_C")
        np.savez_compressed(path, rate=rate.data.astype(np.float32), dq=rate.dq, err=rate.err.astype(np.float32))
        rate.close()
    return np.load(path)


def stream_clean_rate(jump_path, input_dir, out_dir, excluded, tag, name="stream_clean", options=None):
    """Rate and scene mask of a stream clean variant fitted without the ``excluded`` pixels, cached as npz files."""
    rate_path = os.path.join(out_dir, f"{name}_{tag}_ols.npz")
    mask_path = os.path.join(out_dir, f"{name}_{tag}_mask.npz")
    if not os.path.exists(rate_path):
        step = HeldOutStreamCleanStep(output_dir=out_dir, save_mask=True, **(options or {}))
        step.excluded = excluded
        step.input_dir = input_dir
        cleaned = step.run(jump_path)
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save_rate(rate_path, rate)
        cleaned.close()
        rate.close()
        saved_mask = glob.glob(os.path.join(out_dir, "*_mask.fits"))[0]
        with datamodels.open(saved_mask) as mask_model:
            np.savez_compressed(mask_path, mask=np.asarray(mask_model.data, dtype=bool))
        os.remove(saved_mask)
    return np.load(rate_path)["rate"], np.load(mask_path)["mask"]


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
    warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="folder of the earlier run: the _jump ramp and regions.npz")
    parser.add_argument("input_dir", help="folder of the _uncal file and, for MOS, the MSA metafile")
    parser.add_argument("out")
    parser.add_argument("--block", type=int, default=32, help="rows per block of background pixels")
    parser.add_argument("--sets", type=int, default=2, help="one block in this many is held out, each set in turn")
    parser.add_argument(
        "--stream-variants", default="", help="comma-separated stream clean variants to run besides the defaults"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jump_path = glob.glob(os.path.join(args.run, "*_jump.fits"))[0]
    variants = [name for name in args.stream_variants.split(",") if name]
    corrections = CORRECTIONS[fits.getheader(jump_path)["EXP_TYPE"]] + variants
    regions = np.load(os.path.join(args.run, "regions.npz"))["regions"]
    none = uncleaned_rate(jump_path, args.out)
    usable = usable_pixels(none["rate"], none["dq"])
    ifu_masks = {
        "msaexp_ifu_wcs": msaexp_ifu_mask(none, regions),
        "msaexp_ifu": msaexp_ifu_mask(none, nd.binary_dilation(regions, iterations=SLICE_GROWTH)),
    }

    # the two-set runs keep their earlier file names
    tags = [f"half{part}" if args.sets == 2 else f"part{part}of{args.sets}" for part in range(args.sets)]
    metrics_path = os.path.join(args.out, "metrics.json" if args.sets == 2 else f"metrics_{args.sets}sets.json")
    metrics = {"block": args.block, "sets": args.sets, "jump": os.path.basename(jump_path)}
    for part, tag in enumerate(tags):
        held = held_out_blocks(regions.shape, args.block, part, args.sets)
        stream_clean, scene = stream_clean_rate(jump_path, os.path.abspath(args.input_dir), args.out, held, tag)
        measured = usable & ~regions & scene & held
        rates = {"none": none["rate"], "stream_clean": stream_clean}
        for name in variants:
            options, growth = STREAM_CLEAN_VARIANTS[name]
            excluded = held | (nd.binary_dilation(regions, iterations=growth) if growth else False)
            rates[name], _ = stream_clean_rate(
                jump_path, os.path.abspath(args.input_dir), args.out, excluded, tag, name, options
            )
        for name, mask in ifu_masks.items():
            if name in corrections:
                rates[name] = grizli_oneoverf(none["rate"], none["err"], none["dq"], mask & ~held, 2048)
        if "msaexp_mos" in corrections:
            rates["msaexp_mos"] = grizli_oneoverf(none["rate"], none["err"], none["dq"], ~held, 256)
        metrics[tag] = {"measured_pixels": int(measured.sum())}
        for name in corrections:
            metrics[tag][name] = {
                **banding_metrics(rates[name], measured), **block_metrics(rates[name], measured, args.block)
            }
        with open(metrics_path, "w") as handle:
            json.dump(metrics, handle, indent=2)

    print(f"{metrics['jump']}: scatter on the held-out blocks [1e-4 DN/s], mean over the {args.sets} sets")
    for name in corrections:
        values = {
            key: 1e4 * np.mean([metrics[tag][name][key] for tag in tags])
            for key in ("column_mean_scatter", "column_mean_white_floor", "block_mean_scatter", "block_mean_white_floor")
        }
        print(
            f"  {name:30s} column means {values['column_mean_scatter']:6.3f} (floor {values['column_mean_white_floor']:5.3f})"
            f"  block means {values['block_mean_scatter']:6.3f} (floor {values['block_mean_white_floor']:5.3f})"
        )


if __name__ == "__main__":
    main()
