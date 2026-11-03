"""Resample cal files with jwst's ResampleImage and with snowblind's SparseResampleImage, and compare the outputs.

Three runs onto jwst's default output frame, with ivm weights, the
variance planes and ERR from them, and context:

1. jwst's ``ResampleImage`` (drizzle), with the pixel map of every pixel;
2. ``SparseResampleImage(variance="stcal")`` with the pixel map of every
   pixel, which should give the same outputs to float32 round-off;
3. ``SparseResampleImage()`` with its defaults: the exact variance planes,
   the noise factor, and the pixel map from splines through every 16th
   pixel.

For runs 2 and 3 the table gives, for each plane, the relative difference
from run 1 over the output pixels with weight in run 1 (median, 99.9th
percentile, largest) and the pixels that are NaN in only one of the two.
Run 3's variance planes are not stcal's, so only its science image and
weights are compared; its variance is compared through two ratios, median
and 5-95% over the pixels with weight: jwst's summed variance planes over
the exact ones, and over the exact ones times the noise factor.

Usage::

    python scripts/compare_sparse_resample.py DATA/jw*_nrcalong_cal.fits

Needs ``pip install 'snowblind[spizzle]'``.
"""
import argparse
import time
import warnings

import numpy as np
from jwst import datamodels
from jwst.datamodels import ModelLibrary
from jwst.resample.resample import ResampleImage
from jwst.resample.resample_step import GOOD_BITS

from snowblind.resample import SparseResampleImage

PLANES = ["data", "wht", "var_rnoise", "var_flat", "var_poisson", "err"]
KWARGS = {
    "weight_type": "ivm",
    "good_bits": GOOD_BITS,
    "blendheaders": False,
    "enable_ctx": True,
    "enable_var": True,
    "report_var": True,
    "compute_err": "from_var",
}


def timed(resamp_class, library, **kwargs):
    """The output model of ``resamp_class(library, **kwargs).resample_many_to_one()``, the resampler and the time."""
    t0 = time.perf_counter()
    resamp = resamp_class(library, **kwargs)
    output = resamp.resample_many_to_one()
    return output, resamp, time.perf_counter() - t0


def difference_row(name, ours, theirs, weighted):
    """A table row: relative difference of ``ours`` from ``theirs`` over ``weighted``, and NaN in only one."""
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(theirs, dtype=np.float64)
    both = weighted & np.isfinite(a) & np.isfinite(b) & (b != 0)
    rel = np.abs(a[both] / b[both] - 1.0)
    one_nan = int(np.sum(weighted & (np.isfinite(a) != np.isfinite(b))))
    return f"| {name} | {np.median(rel):.1e} | {np.percentile(rel, 99.9):.1e} | {rel.max():.1e} | {one_nan} |"


def ratio_row(name, ratio):
    """A table row: median and 5-95% of ``ratio``."""
    low, high = np.nanpercentile(ratio, [5, 95])
    return f"| {name} | {np.nanmedian(ratio):.3f} | {low:.3f} - {high:.3f} |"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("files", nargs="+", help="cal files of one detector and filter")
    args = parser.parse_args()

    library = ModelLibrary([datamodels.open(f) for f in args.files], on_disk=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        jwst_out, _, t_jwst = timed(ResampleImage, library, pixmap_stepsize=1, pixmap_order=1, **KWARGS)
        stcal_out, _, t_stcal = timed(
            SparseResampleImage, library, variance="stcal", pixmap_stepsize=1, pixmap_order=1, **KWARGS
        )
        exact_out, exact, t_exact = timed(SparseResampleImage, library, **KWARGS)

    weighted = jwst_out.wht > 0
    ny, nx = jwst_out.data.shape
    print(f"{len(args.files)} exposures onto jwst's {ny}x{nx} frame, {weighted.sum()} pixels with weight\n")
    print("| run | time |")
    print("| --- | ---: |")
    print(f"| jwst ResampleImage, every pixel's WCS | {t_jwst:.1f} s |")
    print(f"| SparseResampleImage(variance='stcal'), every pixel's WCS | {t_stcal:.1f} s |")
    print(f"| SparseResampleImage(), defaults, with the noise factor pass | {t_exact:.1f} s |")

    print("\nSparseResampleImage(variance='stcal') against jwst:\n")
    print("| plane | median | 99.9% | max | NaN in one only |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for plane in PLANES:
        print(difference_row(plane.upper(), getattr(stcal_out, plane), getattr(jwst_out, plane), weighted))
    print(f"\nCON equal: {np.array_equal(stcal_out.con, jwst_out.con)}")

    print("\nSparseResampleImage() against jwst:\n")
    print("| plane | median | 99.9% | max | NaN in one only |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for plane in ["data", "wht"]:
        print(difference_row(plane.upper(), getattr(exact_out, plane), getattr(jwst_out, plane), weighted))

    jwst_var = (jwst_out.var_rnoise + jwst_out.var_flat + jwst_out.var_poisson)[weighted]
    exact_var = (exact_out.var_rnoise + exact_out.var_flat + exact_out.var_poisson)[weighted]
    noise_factor = exact.noise_factor[weighted]
    print("\n| quantity | median | 5-95% |")
    print("| --- | ---: | ---: |")
    print(ratio_row("noise factor", noise_factor))
    print(ratio_row("jwst VAR / exact VAR", jwst_var / exact_var))
    print(ratio_row("jwst VAR / (exact VAR * noise factor)", jwst_var / (exact_var * noise_factor)))
    print(ratio_row("jwst ERR / exact ERR", jwst_out.err[weighted] / exact_out.err[weighted]))


if __name__ == "__main__":
    main()
