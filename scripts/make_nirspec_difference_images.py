"""Difference images, uncleaned minus stream-cleaned, of a NIRSpec exposure run through exp_nirspec_variants.py.

For each output folder given, reads ``none_ols.npz``, ``<variant>_ols.npz``,
``regions.npz`` and ``<variant>_mask.npz`` and writes, named after the
exposure:

* ``<exposure>_uncleaned_minus_<variant>.fits``: the difference in DN/s,
  with the uncleaned and cleaned rates and the slice footprints as further
  extensions, for DS9;
* ``<exposure>_uncleaned_minus_<variant>.png``: the difference binned 4x4
  with the footprints outlined, a full-resolution cutout, and the uncleaned
  rate of the same cutout;
* ``<exposure>_removed_per_row_<variant>.png``: the mean of the difference
  along each science row inside the footprints (constant along the
  dispersion direction is the column-offset artifact).

Usage: python scripts/make_nirspec_difference_images.py results/.../NRS1 [results/.../NRS2]
       [--variant default] [--cutout x y size]
"""

import argparse
import glob
import os

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def exposure_name(folder):
    """The exposure root (``jw..._nrs1``) from the saved jump ramp in the folder."""
    jumps = glob.glob(os.path.join(folder, "*_jump.fits"))
    if not jumps:
        raise FileNotFoundError(f"no *_jump.fits in {folder}")
    return os.path.basename(jumps[0]).replace("_jump.fits", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--cutout", nargs=3, type=int, default=(700, 900, 256), metavar=("X", "Y", "SIZE"))
    args = parser.parse_args()
    for folder in args.folders:
        name = exposure_name(folder)
        none = np.load(os.path.join(folder, "none_ols.npz"))
        cleaned = np.load(os.path.join(folder, f"{args.variant}_ols.npz"))
        regions = np.load(os.path.join(folder, "regions.npz"))["regions"]
        mask_path = os.path.join(folder, f"{args.variant}_mask.npz")
        # a jwst chain saves no mask; its background is everything outside the footprints
        mask = np.load(mask_path)["mask"] if os.path.exists(mask_path) else ~regions
        diff = none["rate"] - cleaned["rate"]
        good = (none["dq"] == 0) & np.isfinite(diff)
        footprint = regions & good
        background = mask & good

        hdus = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(diff.astype(np.float32), name="DIFF"),
            fits.ImageHDU(none["rate"].astype(np.float32), name="UNCLEANED"),
            fits.ImageHDU(cleaned["rate"].astype(np.float32), name="CLEANED"),
            fits.ImageHDU(regions.astype(np.uint8), name="REGIONS"),
            fits.ImageHDU(mask.astype(np.uint8), name="BACKGROUND"),
        ])
        hdus[0].header["COMMENT"] = "DIFF = UNCLEANED - CLEANED rate, DN/s; REGIONS = slit or slice footprints"
        hdus[0].header["COMMENT"] = "BACKGROUND = the mask the noise was fitted on"
        fits_path = os.path.join(folder, f"{name}_uncleaned_minus_{args.variant}.fits")
        hdus.writeto(fits_path, overwrite=True)

        print(f"{name}: median removed inside the footprints {1e4 * np.median(diff[footprint]):+.2f}e-4 DN/s, "
              f"outside {1e4 * np.median(diff[background]):+.2f}e-4; pixel scatter of the difference "
              f"{1e4 * mad_std(diff[footprint]):.2f}e-4 (inside) {1e4 * mad_std(diff[background]):.2f}e-4 (outside); "
              f"uncleaned sky inside the footprints {np.median(none['rate'][footprint]):.4f} DN/s")

        x0, y0, size = args.cutout
        cut = (slice(y0, y0 + size), slice(x0, x0 + size))
        extent = (x0, x0 + size, y0, y0 + size)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
        smoothed = ndimage.uniform_filter(np.nan_to_num(diff), 4)
        axes[0].imshow(smoothed[::4, ::4], origin="lower", cmap="RdBu_r", vmin=-0.003, vmax=0.003, extent=(0, 2048, 0, 2048))
        axes[0].contour(regions[::4, ::4], levels=[0.5], colors="k", linewidths=0.3, extent=(0, 2048, 0, 2048))
        axes[0].set_title(f"{name}: uncleaned - cleaned, binned 4x4, footprints outlined (+-0.003 DN/s)")
        axes[1].imshow(diff[cut], origin="lower", cmap="RdBu_r", vmin=-0.01, vmax=0.01, extent=extent)
        axes[1].contour(regions[cut], levels=[0.5], colors="k", linewidths=0.4, extent=extent)
        axes[1].set_title(f"same, {size} px cutout at full resolution (+-0.01 DN/s)")
        axes[2].imshow(none["rate"][cut], origin="lower", cmap="gray", vmin=-0.005, vmax=0.03, extent=extent)
        axes[2].set_title("uncleaned rate, same cutout (-0.005 to 0.03 DN/s)")
        for ax in axes:
            ax.set_xlabel("science x (dispersion)")
            ax.set_ylabel("science y")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{name}_uncleaned_minus_{args.variant}.png"), dpi=110)
        plt.close(fig)

        with np.errstate(all="ignore"):
            per_row = np.nanmean(np.where(footprint, diff, np.nan), axis=1)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(per_row * 1e4, lw=0.5)
        for amp in range(1, 4):
            ax.axvline(512 * amp, color="k", lw=0.3)
        ax.set_xlabel("science row")
        ax.set_ylabel("mean removed inside the footprints, x1e-4 DN/s")
        ax.set_ylim(-40, 40)
        ax.set_title(f"{name}: per-row mean of uncleaned - cleaned inside the slit or slice footprints")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{name}_removed_per_row_{args.variant}.png"), dpi=110)
        plt.close(fig)
        print(f"  wrote {fits_path} and the two figures")


if __name__ == "__main__":
    main()
