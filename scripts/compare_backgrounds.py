"""Tabulate the stream_clean background settings of a field panel run on the sky pixels.

Reads ``sirs_ols.npz``, ``sirs_jwst_usage_ols.npz`` and every
``sirs_stream_clean_<setting>_ols.npz`` in a panel directory, with the
stream_clean mask, and prints for each the row-mean and column-mean
scatter of the background pixels within 5 sigma of the sky (full frame
and the central 512 x 512), the change of the mean sky against SIRS, the
5th and 95th percentiles of the per-pixel change, and the 5th percentile
within 8 pixels of a masked source (the streaks).  Also written as
``backgrounds.json`` in the directory.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage
from stdatamodels.jwst import datamodels

from snowblind.metrics import robust_sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("panel_dir")
    parser.add_argument(
        "--flat", action="store_true",
        help="divide every rate by the CRDS flat of the exposure before measuring, so that the detector's "
        "large-scale response does not count as structure",
    )
    args = parser.parse_args()
    d = args.panel_dir
    flat = 1.0
    with datamodels.open(os.path.join(d, "stream_clean_mask.fits")) as m:
        mask = m.data.astype(bool)
        if args.flat:
            from jwst.flatfield import FlatFieldStep

            flat_path = FlatFieldStep().get_reference_file(m, "flat")
            with datamodels.FlatModel(flat_path) as flat_model:
                flat = np.array(flat_model.data, dtype=np.float64)
                flat[(flat_model.dq & 1) > 0] = np.nan
            print(f"rates divided by {os.path.basename(flat_path)}")
    ref = np.load(os.path.join(d, "sirs_ols.npz"))
    sirs = ref["rate"].astype(np.float64) / flat
    good = mask & np.isfinite(sirs) & (ref["dq"] == 0)
    good[:4] = good[-4:] = False
    good[:, :4] = good[:, -4:] = False
    sky = np.nanmedian(sirs[good])
    sigma = robust_sigma(sirs[good])
    stats = good & (np.abs(sirs - sky) < 5 * sigma)
    dist = ndimage.distance_transform_edt(mask)
    region = (slice(768, 1280), slice(768, 1280))

    files = {"sirs": "sirs_ols.npz", "jwst": "sirs_jwst_usage_ols.npz"}
    for path in sorted(glob.glob(os.path.join(d, "sirs_stream_clean_*_ols.npz"))):
        name = os.path.basename(path)[len("sirs_stream_clean_"):-len("_ols.npz")]
        files[f"stream_clean {name}"] = os.path.basename(path)
    files["stream_clean reset fit"] = "sirs_stream_clean_reset.npz"

    print(f"sky {sky:.4f} DN/s, {stats.sum()} sky pixels")
    print(
        f"{'chain':26s} {'full row':>9s} {'full col':>9s} {'cut row':>8s} {'cut col':>8s} "
        f"{'sky %':>7s} {'LS %':>6s} {'p5':>8s} {'p95':>8s} {'near p5':>8s} {'finite':>8s}"
    )
    out = {}
    for label, name in files.items():
        path = os.path.join(d, name)
        if not os.path.exists(path):
            continue
        r = np.load(path)["rate"].astype(np.float64) / flat
        g = stats & np.isfinite(r)
        full = np.where(g, r, np.nan)
        change = (r - sirs)[g]
        near = g & (dist < 8)
        # large-scale change: scatter of 128-pixel block means of the change over the sky pixels, as a
        # fraction of the sky, which is what a scalar background removes when the sky has real structure
        blocks = np.where(g, r - sirs, np.nan)[:2048, :2048].reshape(16, 128, 16, 128)
        block_change = np.nanmean(blocks, axis=(1, 3))
        row = {
            "large_scale_change_pct": float(robust_sigma(block_change[np.isfinite(block_change)]) / sky * 100),
            "full_row": float(robust_sigma(np.nanmean(full[4:-4, 4:-4], axis=1))),
            "full_col": float(robust_sigma(np.nanmean(full[4:-4, 4:-4], axis=0))),
            "cut_row": float(robust_sigma(np.nanmean(full[region], axis=1))),
            "cut_col": float(robust_sigma(np.nanmean(full[region], axis=0))),
            "sky_change_pct": float(change.mean() / sky * 100),
            "p5": float(np.percentile(change, 5)), "p95": float(np.percentile(change, 95)),
            "near_p5": float(np.percentile((r - sirs)[near], 5)) if near.any() else float("nan"),
            "finite": int(np.isfinite(r).sum()),
        }
        out[label] = row
        print(f"{label:26s} {row['full_row']:9.4f} {row['full_col']:9.4f} {row['cut_row']:8.4f} {row['cut_col']:8.4f} "
              f"{row['sky_change_pct']:+7.2f} {row['large_scale_change_pct']:6.2f} {row['p5']:+8.4f} {row['p95']:+8.4f} "
              f"{row['near_p5']:+8.4f} {row['finite']:8d}")
    with open(os.path.join(d, "backgrounds.json"), "w") as handle:
        json.dump({"sky": float(sky), "n_sky_pixels": int(stats.sum()), "chains": out}, handle, indent=1)


if __name__ == "__main__":
    main()
