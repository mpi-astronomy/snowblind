"""Compare passes of the mosaic scene-model loop on one detector.

For every exposure of every pass: the sky level of the cal image, the
residual of the cal image against the blotted median mosaic of that pass
on the background pixels (pixel scatter and the scatter of its row and
column means, a measure of how consistent the exposure is with the other
dithers), and the fraction of pixels flagged as outliers.  Then aperture
photometry of stars found on the first pass's cal image, measured in the
same place on every pass (radius 4, annulus 8 to 12), as flux ratios to
the first pass by flux bin.  Output: ``compare.json`` in the last pass's
directory and a printed table.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from stdatamodels.jwst import datamodels

from snowblind.metrics import robust_sigma


def exposures(pass_dir):
    out = {}
    for blot in sorted(glob.glob(os.path.join(pass_dir, "*_blot.fits"))):
        root = re.sub(r"_a\d{4}$", "", os.path.basename(blot).replace("_blot.fits", ""))
        out[root] = {"blot": blot, "cal": os.path.join(pass_dir, f"{root}_cal.fits"),
                     "crf": blot.replace("_blot.fits", "_crf.fits"), "mask": os.path.join(pass_dir, f"{root}_mask.fits")}
    return out


def load(path):
    with datamodels.open(path) as model:
        return np.array(model.data, dtype=np.float64), np.array(model.dq)


def residual_stats(cal, dq, blot, mask):
    good = mask & np.isfinite(cal) & np.isfinite(blot) & (dq == 0)
    good[:4] = good[-4:] = False
    good[:, :4] = good[:, -4:] = False
    sky = np.nanmedian(cal[good])
    sigma = robust_sigma(cal[good])
    stats = good & (np.abs(cal - sky) < 5 * sigma)
    resid = np.where(stats, cal - blot, np.nan)
    cut = resid[4:-4, 4:-4]
    return {
        "sky": float(sky), "n_sky_pixels": int(stats.sum()),
        "residual_median": float(np.nanmedian(resid)), "residual_pixel_scatter": float(robust_sigma(resid[stats])),
        "residual_row_mean_scatter": float(robust_sigma(np.nanmean(cut, axis=1))),
        "residual_column_mean_scatter": float(robust_sigma(np.nanmean(cut, axis=0))),
        "cal_row_mean_scatter": float(robust_sigma(np.nanmean(np.where(stats, cal, np.nan)[4:-4, 4:-4], axis=1))),
    }


def star_photometry(image, dq, positions):
    data = np.where(np.isfinite(image) & (dq == 0), image, np.nan)
    aperture = CircularAperture(positions, r=4.0)
    annulus = CircularAnnulus(positions, r_in=8.0, r_out=12.0)
    mask = ~np.isfinite(data)
    filled = np.where(mask, 0.0, data)
    flux = aperture_photometry(filled, aperture, mask=mask)["aperture_sum"].value
    annulus_masks = annulus.to_mask(method="center")
    background = np.array([np.nanmedian(m.get_values(data)) for m in annulus_masks])
    return flux - background * aperture.area


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("passes", nargs="+", help="pass directories in order")
    args = parser.parse_args()

    report = {}
    first = exposures(args.passes[0])
    positions = {}
    for root, files in first.items():
        cal, dq = load(files["cal"])
        finder = DAOStarFinder(fwhm=2.5, threshold=50.0 * robust_sigma(cal[np.isfinite(cal)]), exclude_border=True)
        found = finder(np.where(np.isfinite(cal), cal, 0.0))
        positions[root] = np.transpose([found["xcentroid"], found["ycentroid"]])
        print(f"{root}: {len(found)} stars for photometry")

    fluxes = {}
    for pass_dir in args.passes:
        name = os.path.basename(os.path.normpath(pass_dir))
        report[name] = {}
        for root, files in exposures(pass_dir).items():
            cal, dq = load(files["cal"])
            blot, _ = load(files["blot"])
            with datamodels.open(files["mask"]) as m:
                mask = m.data.astype(bool)
            _crf, crf_dq = load(files["crf"])
            entry = residual_stats(cal, dq, blot, mask)
            entry["outlier_fraction"] = float(np.mean((crf_dq & 16) > 0))
            fluxes[(name, root)] = star_photometry(cal, dq, positions[root])
            report[name][root] = entry
            print(f"{name} {root}: " + ", ".join(f"{k} {v:.4g}" for k, v in entry.items()))

    base = os.path.basename(os.path.normpath(args.passes[0]))
    for pass_dir in args.passes[1:]:
        name = os.path.basename(os.path.normpath(pass_dir))
        ratios, ref = [], []
        for root in first:
            a, b = fluxes[(base, root)], fluxes[(name, root)]
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
            ratios.append(b[ok] / a[ok])
            ref.append(a[ok])
        ratios, ref = np.concatenate(ratios), np.concatenate(ref)
        bins = np.percentile(ref, [0, 25, 50, 75, 100])
        table = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (ref >= lo) & (ref < hi)
            table.append({"flux_min": float(lo), "flux_max": float(hi), "n": int(sel.sum()),
                          "median_ratio": float(np.median(ratios[sel])), "ratio_robust_sigma": float(robust_sigma(ratios[sel]))})
            print(
                f"{name}/{base} flux {lo:8.1f}-{hi:8.1f} DN/s: n {sel.sum():5d} "
                f"median ratio {np.median(ratios[sel]):.4f} scatter {robust_sigma(ratios[sel]):.4f}"
            )
        report[name]["photometry_vs_" + base] = table
    with open(os.path.join(args.passes[-1], "compare.json"), "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
