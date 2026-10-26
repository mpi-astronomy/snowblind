"""Photometry and bright-star spike profiles across the chains of a field panel directory.

Sources are found on the flat-fielded SIRS rate (DAOStarFinder) and
measured in the same apertures (radius 4, annulus 8 to 12) on every
chain, as flux ratios to SIRS by flux bin.  For the brightest source,
the azimuthally resolved change against SIRS (cleaned minus SIRS, in
units of the sky) is reported in 12 directions and radial bins out to
400 pixels, so that a diffraction spike removed by a row or column
estimate shows up as a negative change along that direction.
"""
import argparse
import glob
import json
import os

import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.detection import DAOStarFinder
from scipy import ndimage
from stdatamodels.jwst import datamodels

from snowblind.metrics import robust_sigma


def load_flat(panel_dir):
    from jwst.flatfield import FlatFieldStep

    with datamodels.open(os.path.join(panel_dir, "stream_clean_mask.fits")) as m:
        path = FlatFieldStep().get_reference_file(m, "flat")
    with datamodels.FlatModel(path) as fm:
        flat = np.array(fm.data, dtype=np.float64)
        flat[(fm.dq & 1) > 0] = np.nan
    return flat


def photometry(image, positions):
    """Background-subtracted aperture sums; NaN for an aperture that holds any non-finite pixel.

    A chain that leaves the core of a saturating star without a rate would
    otherwise sum a smaller aperture than one that fills it, and the ratio
    would measure the missing pixels rather than the cleaning.
    """
    data = np.where(np.isfinite(image), image, np.nan)
    aperture = CircularAperture(positions, r=4.0)
    annulus = CircularAnnulus(positions, r_in=8.0, r_out=12.0)
    mask = ~np.isfinite(data)
    flux = aperture_photometry(np.where(mask, 0.0, data), aperture, mask=mask)["aperture_sum"].value
    missing = aperture_photometry(mask.astype(np.float64), aperture)["aperture_sum"].value
    background = np.array([np.nanmedian(m.get_values(data)) for m in annulus.to_mask(method="center")])
    return np.where(missing > 0, np.nan, flux - background * aperture.area)


def spike_profile(change, center, sky, nbins=20, rmax=400.0, ndir=12):
    """Mean change in sectors of 30 degrees and radial bins, in units of the sky."""
    cy, cx = center
    yy, xx = np.mgrid[0:change.shape[0], 0:change.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    theta = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
    edges = np.linspace(0, rmax, nbins + 1)
    out = np.full((ndir, nbins), np.nan)
    for d in range(ndir):
        sector = (theta >= d * 360.0 / ndir) & (theta < (d + 1) * 360.0 / ndir)
        for b in range(nbins):
            sel = sector & (rr >= edges[b]) & (rr < edges[b + 1]) & np.isfinite(change)
            if sel.sum() > 20:
                out[d, b] = np.mean(change[sel]) / sky
    return edges, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("panel_dir")
    parser.add_argument("--threshold", type=float, default=50.0, help="detection threshold in robust sigma of the sky")
    args = parser.parse_args()
    d = args.panel_dir
    flat = load_flat(d)
    with datamodels.open(os.path.join(d, "stream_clean_mask.fits")) as m:
        mask = m.data.astype(bool)
    sirs = np.load(os.path.join(d, "sirs_ols.npz"))["rate"].astype(np.float64) / flat
    good = mask & np.isfinite(sirs)
    good[:4] = good[-4:] = False
    good[:, :4] = good[:, -4:] = False
    sky = float(np.nanmedian(sirs[good]))
    sigma = robust_sigma(sirs[good])

    chains = {"jwst": "sirs_jwst_usage_ols.npz", "remstriping": "sirs_remstriping_ols.npz",
              "remstriping (James cal)": "remstriping_james_ols.npz"}
    for path in sorted(glob.glob(os.path.join(d, "sirs_stream_clean_*_ols.npz"))):
        chains["stream_clean " + os.path.basename(path)[len("sirs_stream_clean_"):-len("_ols.npz")]] = os.path.basename(path)
    chains["stream_clean reset fit"] = "sirs_stream_clean_reset.npz"
    chains = {k: v for k, v in chains.items() if os.path.exists(os.path.join(d, v))}

    finder = DAOStarFinder(fwhm=2.5, threshold=args.threshold * sigma, exclude_border=True)
    found = finder(np.where(np.isfinite(sirs), sirs - sky, 0.0))
    positions = np.transpose([found["xcentroid"], found["ycentroid"]])
    reference = photometry(sirs, positions)
    print(f"{len(found)} sources, {int(np.isfinite(reference).sum())} with complete apertures on SIRS; sky {sky:.4f} DN/s")
    report = {"sky": sky, "n_sources": int(len(found)), "photometry": {}, "spikes": {}}
    bins = np.percentile(reference[reference > 0], [0, 25, 50, 75, 100])
    for label, name in chains.items():
        r = np.load(os.path.join(d, name))["rate"].astype(np.float64) / flat
        flux = photometry(r, positions)
        ok = np.isfinite(flux) & np.isfinite(reference) & (reference > 0)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = ok & (reference >= lo) & (reference < hi)
            ratio = flux[sel] / reference[sel]
            rows.append({"flux_min": float(lo), "flux_max": float(hi), "n": int(sel.sum()),
                         "median_ratio": float(np.median(ratio)), "scatter": float(robust_sigma(ratio))})
        report["photometry"][label] = rows
        cells = [
            f"[{row['flux_min']:.0f}-{row['flux_max']:.0f}: {row['median_ratio']:.4f}+-{row['scatter']:.4f}]" for row in rows
        ]
        print(f"  {label:26s} " + "  ".join(cells))

    # brightest source: spike profiles of the change against SIRS
    core = np.where(np.isfinite(sirs), sirs, 0.0)
    smooth = ndimage.uniform_filter(core, 9)
    cy, cx = np.unravel_index(np.argmax(smooth), smooth.shape)
    print(f"brightest source at (y {cy}, x {cx}), peak {sirs[cy, cx]:.1f} DN/s ({sirs[cy, cx] / sky:.0f} x sky)")
    for label, name in chains.items():
        r = np.load(os.path.join(d, name))["rate"].astype(np.float64) / flat
        change = np.where(mask, r - sirs, np.nan)
        edges, profile = spike_profile(change, (cy, cx), sky)
        worst = np.nanmin(profile, axis=1)
        report["spikes"][label] = {"radii": edges.tolist(), "profile_over_sky": profile.tolist()}
        print(f"  {label:26s} most negative sector-mean change over the sky, per 30-degree sector: "
              + " ".join(f"{w:+.3f}" for w in worst))
    with open(os.path.join(d, "photometry.json"), "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
