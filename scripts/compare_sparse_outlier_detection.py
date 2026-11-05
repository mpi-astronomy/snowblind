"""Flag outliers in cal files with jwst's OutlierDetectionStep and with snowblind's SparseOutlierDetectionStep.

Three runs of outlier detection for images, with the steps' default
thresholds:

1. jwst's ``OutlierDetectionStep`` (drizzle and stcal's blot), with the
   pixel map of every pixel;
2. ``SparseOutlierDetectionStep`` with the pixel map of every pixel;
3. ``SparseOutlierDetectionStep`` with its defaults, the pixel map from
   splines through every 16th pixel.

For each run the table gives the time and the pixels it newly flagged as
OUTLIER, summed over the exposures; for runs 2 and 3, the pixels flagged by
only one of the run and run 1.  drizzle 2.2.0 blots in single precision, so
pixels near the thresholds can differ.

Usage::

    python scripts/compare_sparse_outlier_detection.py DATA/jw*_nrca1_cal.fits

Needs ``pip install 'snowblind[spizzle]'``.  The steps' CRDS parameters are
not read (``STPIPE_DISABLE_CRDS_STEPPARS``), so that both use the same
thresholds.
"""
import argparse
import os
import time

import numpy as np
from jwst import datamodels
from jwst.datamodels import ModelLibrary
from jwst.outlier_detection import OutlierDetectionStep
from stdatamodels.jwst.datamodels import dqflags

from snowblind import SparseOutlierDetectionStep

OUTLIER = dqflags.pixel["OUTLIER"]


def new_flags(step, files):
    """The OUTLIER flags that ``step`` sets in the cal files, one boolean image per exposure, and the time."""
    models = [datamodels.open(f) for f in files]
    before = [(m.dq & OUTLIER) > 0 for m in models]
    step.mode = "imaging"
    t0 = time.perf_counter()
    result = step.run(ModelLibrary(models, on_disk=False))
    elapsed = time.perf_counter() - t0
    after = []
    with result:
        for model in result:
            after.append((model.dq & OUTLIER) > 0)
            result.shelve(model, modify=False)
    return np.array([a & ~b for a, b in zip(after, before)]), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("files", nargs="+", help="cal files of one detector and filter")
    args = parser.parse_args()
    os.environ["STPIPE_DISABLE_CRDS_STEPPARS"] = "True"

    runs = [
        ("jwst OutlierDetectionStep, every pixel's WCS", OutlierDetectionStep(pixmap_stepsize=1, pixmap_order=1)),
        ("SparseOutlierDetectionStep, every pixel's WCS", SparseOutlierDetectionStep(pixmap_stepsize=1, pixmap_order=1)),
        ("SparseOutlierDetectionStep, defaults", SparseOutlierDetectionStep()),
    ]
    results = [(name, *new_flags(step, args.files)) for name, step in runs]

    reference = results[0][1]
    print(f"\n{len(args.files)} exposures, {reference[0].size} pixels each\n")
    print("| run | time | pixels flagged | flagged by only one of it and jwst |")
    print("| --- | ---: | ---: | ---: |")
    for name, flags, elapsed in results:
        differ = "" if flags is reference else f"{int(np.sum(flags != reference))}"
        print(f"| {name} | {elapsed:.1f} s | {int(flags.sum())} | {differ} |")


if __name__ == "__main__":
    main()
