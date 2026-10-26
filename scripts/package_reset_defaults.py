"""Write the scalar reset calibration of every detector into the package, one file per detector.

Reads the ``*_reset_calib.asdf`` files of a calibration directory and writes
``src/snowblind/data/reset_calib_<detector>.asdf`` for each, with the scalar
reset scatter ``sigma_b``, the first-read scale and reset offset
(detector-wide and per output channel), the first-group time the timing was
measured at, and the provenance of the file.  ``ResetRampFitStep`` uses the
detector's file when no ``reset_calib`` and no ``sigma_b`` are given; the
pedestal maps stay in the calibration files, which are not packaged.  One
detector can be repackaged alone by pointing the script at a directory that
holds only its file.

Usage: ``python scripts/package_reset_defaults.py results/2026-10-22_reset_calib``
"""
import argparse
import datetime
import glob
import os

import asdf
import numpy as np

from snowblind.reset_calib import add_history, read_calibration

PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "snowblind", "data")


def scalar_entry(tree):
    """The scalar part of a calibration tree, plain Python values."""
    provenance = dict(tree.get("provenance") or {})
    provenance.pop("superbias_note", None)
    entry = {
        "sigma_b": float(tree.get("sigma_b_median", np.nanmedian(tree["sigma_b"]))),
        "read_noise_single_median": tree.get("read_noise_single_median"),
        "t1_scale": float(tree.get("t1_scale", 1.0)),
        "reset_offset": float(tree.get("reset_offset", 0.0) or 0.0),
        "t1_scale_channels": tree.get("t1_scale_channels"),
        "reset_offset_channels": tree.get("reset_offset_channels"),
        "t1_seconds": tree.get("t1_seconds"),
        "n_integrations_max": tree.get("n_integrations_max"),
        "provenance": provenance,
        "timing_provenance": tree.get("timing_provenance"),
    }
    for key in ("t1_scale_channels", "reset_offset_channels"):
        if entry[key] is not None:
            entry[key] = [float(v) for v in entry[key]]
    for key in ("read_noise_single_median", "t1_seconds"):
        if entry[key] is not None:
            entry[key] = float(entry[key])
    if entry["n_integrations_max"] is not None:
        entry["n_integrations_max"] = int(entry["n_integrations_max"])
    return entry


def main(argv=None):
    """Console entry: read the calibration files of a directory and write the package table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("calib_dir", help="directory holding <detector>_reset_calib.asdf files")
    parser.add_argument("--out-dir", default=os.path.normpath(PACKAGE_DIR), help="where the per-detector files go")
    args = parser.parse_args(argv)

    written = 0
    for path in sorted(glob.glob(os.path.join(args.calib_dir, "*_reset_calib.asdf"))):
        tree = read_calibration(path)
        detector = str(tree["detector"]).upper()
        entry = scalar_entry(tree)
        entry.update({
            "detector": detector,
            "description": "Scalar reset calibration, used by ResetRampFitStep when no reset_calib file is given",
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "source": os.path.basename(path),
        })
        out = os.path.join(args.out_dir, f"reset_calib_{detector.lower()}.asdf")
        with asdf.AsdfFile(entry) as handle:
            add_history(handle, f"Scalar reset calibration of {detector} packaged from {os.path.basename(path)}")
            handle.write_to(out)
        written += 1
        print(f"{detector}: sigma_b {entry['sigma_b']:.2f} DN, t1_scale {entry['t1_scale']:.4f} -> {out}")
    print(f"wrote {written} files")


if __name__ == "__main__":
    main()
