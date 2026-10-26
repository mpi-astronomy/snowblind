"""Experiment C.3.3, photometry part: saturated-core stars against unsaturated references.

Input is the ``rates.npz`` written by ``exp_reset_field.py`` (OLS_C, LIKELY
and reset-anchored rate images plus the mask of pixels that saturate from
group 2).  Reference stars are bright unsaturated peaks; their ratio of
core flux (r <= r_core) to ring flux (r_in <= r <= r_out), both sky
subtracted, is the curve-of-growth constant C.  For every star whose core
saturates from group 2 and is fully recovered by the reset fit, the ring
flux (unsaturated pixels, same in every fit) predicts the core flux as
C * ring; the bias is core_reset / (C * ring) - 1.  A bias that grows with
core brightness means the first-read model (k, b) calibrated on 50 to 2000
DN pixels does not hold for brighter ones.

Output: ``saturated_stars.json`` and a per-star table next to the input.
"""
import argparse
import json
import os

import numpy as np
from scipy import ndimage

from snowblind.metrics import robust_sigma

R_CORE = 2.5
R_IN, R_OUT = 3.5, 7.0
R_SKY_IN, R_SKY_OUT = 10.0, 15.0
HALF = int(R_SKY_OUT) + 1


def photometry(image, xc, yc):
    """Core, ring and sky-annulus statistics around (xc, yc); NaN where any pixel in core or ring is not finite."""
    nrows, ncols = image.shape
    x0, x1, y0, y1 = xc - HALF, xc + HALF + 1, yc - HALF, yc + HALF + 1
    if x0 < 4 or y0 < 4 or x1 > ncols - 4 or y1 > nrows - 4:
        return None
    cut = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - xc, yy - yc)
    core, ring = r <= R_CORE, (r >= R_IN) & (r <= R_OUT)
    sky = (r >= R_SKY_IN) & (r <= R_SKY_OUT) & np.isfinite(cut)
    if not (np.isfinite(cut[core]).all() and np.isfinite(cut[ring]).all()) or sky.sum() < 20:
        return None
    level = np.median(cut[sky])
    return {"core": float(np.sum(cut[core] - level)), "ring": float(np.sum(cut[ring] - level)), "peak": float(cut[core].max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rates", help="rates.npz from exp_reset_field.py")
    parser.add_argument("--min-peak", type=float, default=300.0, help="DN/s; minimum peak of a reference star")
    args = parser.parse_args()

    z = np.load(args.rates)
    ols, reset, one_good = z["ols"], z["reset"], z["one_good"]
    out_dir = os.path.dirname(args.rates)

    # reference stars: local maxima of the OLS rate above min_peak with no saturation within R_OUT + 2
    finite_ols = np.where(np.isfinite(ols), ols, -np.inf)
    peaks = (finite_ols == ndimage.maximum_filter(finite_ols, size=7)) & (finite_ols > args.min_peak)
    near_sat = ndimage.binary_dilation(~np.isfinite(ols) | one_good, iterations=int(R_OUT) + 2)
    # isolation against blends: no other peak above a tenth of the star's peak within the sky annulus
    all_peaks = (finite_ols == ndimage.maximum_filter(finite_ols, size=5)) & (finite_ols > 20.0)
    ys, xs = np.nonzero(peaks & ~near_sat)
    ratios = []
    for yc, xc in zip(ys, xs):
        window = all_peaks[max(yc - HALF, 0): yc + HALF + 1, max(xc - HALF, 0): xc + HALF + 1]
        values = finite_ols[max(yc - HALF, 0): yc + HALF + 1, max(xc - HALF, 0): xc + HALF + 1]
        if (window & (values > 0.1 * finite_ols[yc, xc])).sum() > 1:
            continue
        phot = photometry(ols, xc, yc)
        if phot is not None and phot["ring"] > 0:
            ratios.append(phot["core"] / phot["ring"])
    ratios = np.array(ratios)
    growth = float(np.median(ratios))
    growth_scatter = float(robust_sigma(ratios))

    # saturated-core stars: connected groups of pixels saturating from group 2, core fully finite in the reset fit
    labels, n_labels = ndimage.label(one_good)
    centers = ndimage.center_of_mass(one_good, labels, range(1, n_labels + 1))
    rows = []
    for (yc, xc) in centers:
        yc, xc = int(round(yc)), int(round(xc))
        phot = photometry(reset, xc, yc)
        if phot is None or phot["ring"] <= 0:
            continue
        # the saturated core is not a finite peak in the OLS rate; another peak above a tenth of the
        # recovered peak within the sky annulus is a blend
        window = all_peaks[max(yc - HALF, 0): yc + HALF + 1, max(xc - HALF, 0): xc + HALF + 1]
        values = finite_ols[max(yc - HALF, 0): yc + HALF + 1, max(xc - HALF, 0): xc + HALF + 1]
        if (window & (values > 0.1 * phot["peak"])).sum() > 0:
            continue
        core_ols = photometry(ols, xc, yc)
        rows.append(
            {
                "x": xc,
                "y": yc,
                "n_saturated_core_pixels": int((labels[max(yc - 3, 0): yc + 4, max(xc - 3, 0): xc + 4] > 0).sum()),
                "ring": phot["ring"],
                "core_reset": phot["core"],
                "core_predicted": growth * phot["ring"],
                "bias": phot["core"] / (growth * phot["ring"]) - 1.0,
                "peak_reset": phot["peak"],
                "core_ols_finite": core_ols is not None,
            }
        )
    bias = np.array([row["bias"] for row in rows])
    peak = np.array([row["peak_reset"] for row in rows])
    edges = np.array([0, 1000, 2000, 4000, 8000, np.inf])
    by_peak = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (peak >= lo) & (peak < hi)
        if sel.sum() >= 3:
            by_peak.append(
                {
                    "peak_dn_per_s": [float(lo), float(hi)],
                    "n": int(sel.sum()),
                    "median_bias": float(np.median(bias[sel])),
                    "scatter": float(robust_sigma(bias[sel])),
                }
            )
    summary = {
        "rates": os.path.basename(args.rates),
        "n_reference_stars": int(ratios.size),
        "growth_core_over_ring": growth,
        "growth_scatter": growth_scatter,
        "growth_standard_error": growth_scatter / np.sqrt(max(ratios.size, 1)),
        "n_saturated_core_stars": int(len(rows)),
        "n_saturated_core_stars_with_ols_core": int(sum(row["core_ols_finite"] for row in rows)),
        "median_bias": float(np.median(bias)) if bias.size else float("nan"),
        "bias_scatter": float(robust_sigma(bias)) if bias.size else float("nan"),
        "by_peak": by_peak,
    }
    with open(os.path.join(out_dir, "saturated_stars.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    with open(os.path.join(out_dir, "saturated_stars_table.json"), "w") as handle:
        json.dump(rows, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
