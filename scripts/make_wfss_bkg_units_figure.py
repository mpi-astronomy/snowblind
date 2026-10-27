"""The NIRCam WFSS background reference subtracted by the standard pipeline, as delivered and multiplied by the flat.

Runs the jwst Detector1 pipeline with its defaults on one NIRCam WFSS
``_uncal`` exposure, assigns the WCS, and runs the jwst ``background``
step twice on the rate image with a user source mask: once with the CRDS
``bkg`` reference as delivered, once with a copy of it multiplied by the
flat CRDS selects for the exposure.  Both results are divided by the flat,
as the ``flat_field`` step would do next, and shown side by side.  The
NIRCam references are flat-fielded while the step subtracts them before
flat fielding, so the first panel keeps the flat's pattern times the sky
and the second does not.  The reference's shape along the dispersion
does not match the sky in the bottom quarter of the frame (its falloff
toward the edge is steeper than the sky's, in every exposure in hand),
which leaves the same gradient in both panels; that one-dimensional row
profile is removed from both for display and statistics, so that the
panels show the two-dimensional difference, which is the flat's.

Usage: ``python scripts/make_wfss_bkg_units_figure.py <uncal> --mask <mask FITS> --out docs/figures/wfss_bkg_units.jpg``
"""
import argparse
import os

import crds
import matplotlib
import numpy as np
from astropy.stats import mad_std
from scipy import ndimage
from jwst.assign_wcs import AssignWcsStep
from jwst.background.background_sub_wfss import subtract_wfss_bkg
from jwst.pipeline import Detector1Pipeline
from stdatamodels.jwst import datamodels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def binned(image, factor=2):
    """Mean of ``factor`` by ``factor`` blocks, NaNs ignored."""
    ny, nx = (s // factor * factor for s in image.shape)
    block = image[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return np.nanmean(np.nanmean(block, axis=3), axis=1)


def main():
    """Console entry: run the pipeline twice and draw the two panels."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uncal")
    parser.add_argument("--mask", required=True, help="FITS image, 1 for background pixels (e.g. a stream_clean mask)")
    parser.add_argument("--out", default="docs/figures/wfss_bkg_units.jpg")
    parser.add_argument("--work", default="results/wfss_bkg_units", help="where the rate image and the reference copy go")
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)

    rate_path = os.path.join(args.work, os.path.basename(args.uncal).replace("_uncal.fits", "_rate_wcs.fits"))
    if os.path.exists(rate_path):
        rate = datamodels.open(rate_path)
    else:
        rate = Detector1Pipeline.call(args.uncal, save_results=False)
        rate = AssignWcsStep.call(rate)
        rate.save(rate_path)
    refs = crds.getreferences(rate.get_crds_parameters(), reftypes=["bkg", "flat", "wavelengthrange"], observatory="jwst")
    with datamodels.FlatModel(refs["flat"]) as flat_model:
        flat = np.array(flat_model.data, dtype=np.float64)
        flat[(flat_model.dq & 1) > 0] = np.nan
    gain = np.where(np.isfinite(flat) & (flat > 0), flat, 1.0)

    with datamodels.open(refs["bkg"]) as bkg_model:
        bkg_times_flat = bkg_model.copy()
        bkg_times_flat.data = (np.asarray(bkg_model.data, dtype=np.float64) * gain).astype(np.float32)
        copy_path = os.path.join(args.work, os.path.basename(refs["bkg"]).replace(".fits", "_times_flat.fits"))
        bkg_times_flat.save(copy_path)

    with datamodels.open(args.mask) as mask_model:
        sky = np.asarray(mask_model.data) > 0

    # the background step itself insists on a source catalog even with a user mask, so its WFSS function is
    # called directly, with the step's default rejection settings
    rescaler = {"p": 1.0, "maxiter": 5, "delta_rms_thresh": 0.0}
    panels = []
    for label, bkg_path in (("reference as delivered", refs["bkg"]), ("reference multiplied by the flat", copy_path)):
        result = subtract_wfss_bkg(rate.copy(), bkg_path, refs["wavelengthrange"], user_mask=sky, rescaler_kwargs=rescaler)
        factor = float(result.meta.background.scaling_factor)
        image = np.asarray(result.data, dtype=np.float64) / flat   # what flat_field does next
        good = sky & np.isfinite(image)
        good[:4] = good[-4:] = False
        good[:, :4] = good[:, -4:] = False
        profile = np.nanmedian(np.where(good, image, np.nan), axis=1)
        rows = mad_std(profile, ignore_nan=True)
        image = image - np.nan_to_num(profile)[:, None]   # the reference's shape error along the dispersion, same in both
        cols = mad_std(np.nanmedian(np.where(good, image, np.nan), axis=0), ignore_nan=True)
        pixel = mad_std(image[good])
        smooth_num = ndimage.uniform_filter(np.where(good, image, 0.0), 33)
        smooth_den = ndimage.uniform_filter(good.astype(float), 33)
        smooth = np.where(smooth_den > 0.3, smooth_num / smooth_den, np.nan)
        structure = np.nanstd(smooth[4:-4, 4:-4])
        panels.append((label, factor, image, rows, cols, pixel, structure))
        result.close()
    np.savez_compressed(
        os.path.join(args.work, "panels.npz"), as_delivered=panels[0][2].astype(np.float32),
        times_flat=panels[1][2].astype(np.float32), factors=[panels[0][1], panels[1][1]],
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.0), constrained_layout=True)
    stretch = 0.06
    for ax, (label, factor, image, rows, cols, pixel, structure) in zip(axes, panels):
        shown = ax.imshow(binned(image), origin="lower", cmap="viridis", vmin=-stretch, vmax=stretch, interpolation="nearest")
        ax.set_title(
            f"jwst background step, {label}\nscaling factor {factor:.4f}; row profile removed (its scatter {rows:.4f})\n"
            f"sky residual: structure on 33 px {structure:.4f}, column-mean scatter {cols:.4f}, pixels {pixel:.3f} DN/s",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(shown, ax=axes, shrink=0.6, label="DN/s")
    name = os.path.basename(args.uncal).replace("_uncal.fits", "")
    fig.suptitle(
        f"{name}: Detector1 defaults, jwst background step, divided by the flat as flat_field does next,\n"
        f"the row profile along the dispersion (the same in both) removed; full frame binned 2 x 2, stretch +- {stretch} DN/s",
        fontsize=10,
    )
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}; factors {panels[0][1]:.5g} and {panels[1][1]:.5g}")


if __name__ == "__main__":
    main()
