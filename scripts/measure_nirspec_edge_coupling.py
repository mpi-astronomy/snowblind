"""How strongly each band of a NIRSpec frame couples to the 1/f that a cleaning removed.

Usage: python scripts/measure_nirspec_edge_coupling.py OUT.json OUT.png LABEL:RUN_DIR[:CLEANED[:MASK]] ...

Each run directory holds ``none_ols.npz`` and the cleaned rate (default
``stream_clean_ols.npz``) written by the panel scripts, and optionally a
background mask npz.  For every band of 16 science rows the column means of
the cleaned rate are regressed on the 1/f the cleaning removed from the
frame center (rows 224 to 1824, none minus cleaned); 1 + slope is the
coupling gain of that band: 1 where the band carries the same 1/f as the
center, below 1 where the cleaning over-subtracted.  The same is done along
columns with row means.  Also prints the column-mean scatter of the cleaned
and uncleaned rates in the outer 72-row bands and two interior ones.
"""
import json
import os
import sys
import warnings

import matplotlib
import numpy as np
from astropy.stats import sigma_clipped_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BAND = 16
CENTER = (224, 1824)
REPORT_BANDS = ((4, 76), (76, 148), (1000, 1072), (1984, 2044))


def band_means(image, axis, lo, hi):
    """Mean along ``axis`` of the band [lo, hi) of the other axis, median removed, border pixels left out."""
    sub = image[lo:hi, 4:-4] if axis == 0 else image[4:-4, lo:hi]
    means = np.nanmean(sub, axis=axis)
    return means - np.nanmedian(means)


def coupling_profile(uncleaned, cleaned, axis):
    """1 + slope of the cleaned band means against the 1/f removed from the center, per 16-pixel band."""
    removed = band_means(uncleaned, axis, *CENTER) - band_means(cleaned, axis, *CENTER)
    gains = []
    for lo in range(0, 2048, BAND):
        resid = band_means(cleaned, axis, lo, lo + BAND)
        ok = np.isfinite(resid) & np.isfinite(removed)
        if ok.sum() < 200:
            gains.append(np.nan)
            continue
        center, _, scatter = sigma_clipped_stats(resid[ok], sigma=3)
        ok &= np.abs(resid - center) < 4 * scatter
        design = np.vstack([removed[ok], np.ones(ok.sum())]).T
        gains.append(1 + float(np.linalg.lstsq(design, resid[ok], rcond=None)[0][0]))
    return np.array(gains)


def column_mean_scatter(image, lo, hi):
    """Robust scatter of the column means over rows [lo, hi)."""
    means = band_means(image, 0, lo, hi)
    return float(sigma_clipped_stats(means[np.isfinite(means)], sigma=3)[2])


def load_run(spec):
    """Parse LABEL:RUN_DIR[:CLEANED[:MASK]] and return the label and the two masked rate images."""
    parts = spec.split(":")
    label, run_dir = parts[0], parts[1]
    cleaned_name = parts[2] if len(parts) > 2 else "stream_clean_ols.npz"
    uncleaned = np.load(os.path.join(run_dir, "none_ols.npz"))
    cleaned = np.load(os.path.join(run_dir, cleaned_name))
    good = (uncleaned["dq"] == 0) & (cleaned["dq"] == 0)
    if len(parts) > 3:
        mask_file = np.load(os.path.join(run_dir, parts[3]))
        mask = mask_file[mask_file.files[0]].astype(bool)
        good &= mask.all(axis=0) if mask.ndim == 3 else mask
    return label, np.where(good, uncleaned["rate"], np.nan), np.where(good, cleaned["rate"], np.nan)


def main():
    out_json, out_png, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    warnings.filterwarnings("ignore", "Mean of empty slice")
    results = {}
    fig, axes = plt.subplots(2, len(specs), figsize=(5 * len(specs), 7), sharex=True, sharey=True, squeeze=False)
    positions = np.arange(0, 2048, BAND) + BAND / 2
    for column, spec in enumerate(specs):
        label, uncleaned, cleaned = load_run(spec)
        entry = {"coupling_gain_rows": coupling_profile(uncleaned, cleaned, 0).tolist(),
                 "coupling_gain_columns": coupling_profile(uncleaned, cleaned, 1).tolist(),
                 "band_positions": positions.tolist(), "column_mean_scatter": {}}
        for lo, hi in REPORT_BANDS:
            entry["column_mean_scatter"][f"rows {lo}:{hi}"] = {
                "none": column_mean_scatter(uncleaned, lo, hi), "cleaned": column_mean_scatter(cleaned, lo, hi)}
        results[label] = entry
        gains = np.array(entry["coupling_gain_rows"])
        print(f"{label}: coupling gain along rows, bottom {np.round(gains[:8], 2)}, top {np.round(gains[-8:], 2)}, "
              f"interior median {np.nanmedian(gains[14:114]):.3f}")
        for name, values in entry["column_mean_scatter"].items():
            print(f"    {name:15s} column-mean scatter none {values['none']:.4f}  cleaned {values['cleaned']:.4f} DN/s")
        for row, key in enumerate(("coupling_gain_rows", "coupling_gain_columns")):
            ax = axes[row, column]
            ax.plot(positions, entry[key], lw=1)
            ax.axhline(1, color="0.7", lw=0.7)
            for boundary in (512, 1024, 1536):
                ax.axvline(boundary, color="0.85", lw=0.7)
            ax.set_title(f"{label}, along {'rows' if row == 0 else 'columns'}", fontsize=10)
        axes[1, column].set_xlabel("science row or column")
    axes[0, 0].set_ylabel("coupling gain")
    axes[1, 0].set_ylabel("coupling gain")
    axes[0, 0].set_ylim(0.3, 1.5)
    fig.suptitle("1 + slope of the cleaned band means against the 1/f removed from the frame center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    with open(out_json, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
