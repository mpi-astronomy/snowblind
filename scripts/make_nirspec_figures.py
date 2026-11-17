"""Figures for the NIRSpec runs of ``make_nirspec_dark_panels.py`` and ``make_nirspec_field_panels.py``.

Usage:
    python scripts/make_nirspec_figures.py RUN_DIR [--label TEXT] [--out DIR]

``RUN_DIR`` holds the ``<chain>_ols.npz`` files of one detector.  Two
figures are written: the central 512 by 512 pixels of the four rate
images, one image pixel per detector pixel, and the full frame binned 2
by 2; the stretch is +- 2.5 pixel sigma about the median of the uncleaned
rate, unusable pixels are drawn at the median and the row- and column-mean
scatters in the titles are over the shown region (dark runs) or over the
background pixels outside the slits or slices (field runs, whose
``regions.npz`` and stream_clean mask are read when present).
"""
import argparse
import glob
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from snowblind.metrics import robust_sigma  # noqa: E402

PANELS = [
    ("none_ols.npz", "no cleaning, OLS_C"),
    ("jwst_median_ols.npz", "jwst clean_flicker_noise\n(median per channel), OLS_C"),
    ("jwst_fft_ols.npz", "jwst clean_flicker_noise\n(fft, NSClean), OLS_C"),
    ("stream_clean_ols.npz", "snowblind stream_clean, OLS_C"),
]
CENTER = (slice(768, 1280), slice(768, 1280))
BORDER = 4


def usable(run_dir, reference):
    """Pixels the scatters are measured over: dq 0, finite, near the median; outside the science regions on a field."""
    good = np.isfinite(reference["rate"]) & (reference["dq"] == 0)
    good[:BORDER] = good[-BORDER:] = False
    good[:, :BORDER] = good[:, -BORDER:] = False
    regions_file = os.path.join(run_dir, "regions.npz")
    if os.path.exists(regions_file):
        good &= ~np.load(regions_file)["regions"]
    for mask_file in glob.glob(os.path.join(run_dir, "*_mask.fits")):
        good &= fits.getdata(mask_file).astype(bool)
    level = np.nanmedian(reference["rate"][good])
    sigma = robust_sigma(reference["rate"][good])
    return good & (np.abs(reference["rate"] - level) < 5 * sigma), level, sigma


def bin2(image):
    """Mean over 2 by 2 blocks, NaNs ignored."""
    ny, nx = image.shape
    blocks = image[: ny // 2 * 2, : nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2)
    return np.nanmean(np.nanmean(blocks, axis=3), axis=1)


def panel_figure(run_dir, label, out_dir, region, binned, name):
    """Four rate images side by side with the row- and column-mean scatter of each in its title."""
    reference = np.load(os.path.join(run_dir, "none_ols.npz"))
    stats, level, sigma = usable(run_dir, reference)
    width = 2.5 * sigma
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.9), dpi=150, constrained_layout=True)
    for ax, (file_name, title) in zip(axes, PANELS):
        path = os.path.join(run_dir, file_name)
        if not os.path.exists(path):
            ax.set_axis_off()
            continue
        rate = np.load(path)["rate"]
        shown = np.where(np.isfinite(rate), rate, level)[region]
        if binned:
            shown = bin2(shown)
        image = ax.imshow(shown, origin="lower", cmap="viridis", vmin=level - width, vmax=level + width, interpolation="nearest")
        cut = np.where(stats, rate, np.nan)[region]
        rows = np.nanmean(cut, axis=1)
        columns = np.nanmean(cut, axis=0)
        ax.set_title(
            f"{title}\nrow-mean scatter {robust_sigma(rows):.4f} DN/s\ncolumn-mean scatter {robust_sigma(columns):.4f} DN/s",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    colorbar.set_label("DN/s")
    fig.suptitle(f"{label}, stretch +- {width:.3f} DN/s", fontsize=11)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None, help="output folder (default: the run folder)")
    args = parser.parse_args()
    out_dir = args.out or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "panels.json")) as handle:
        info = json.load(handle)
    label = args.label or (
        f"{info['detector']} {info.get('exposure_type', 'dark')} {info.get('readpatt', '')}, {info['ngroups']} groups"
    )
    panel_figure(args.run_dir, label + ", central 512 x 512 pixels", out_dir, CENTER, False, "rate_images_center.png")
    full = (slice(None), slice(None))
    panel_figure(args.run_dir, label + ", full frame binned 2 x 2", out_dir, full, True, "rate_images_full.jpg")


if __name__ == "__main__":
    main()
