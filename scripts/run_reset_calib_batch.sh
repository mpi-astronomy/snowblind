#!/bin/bash
# Reset-level calibration for the ten NIRCam detectors, one at a time (memory), from the 9022 BRIGHT1 darks,
# then the per-channel first-read timing from the 6154 Omega Cen exposure 00004 of each detector.
# Re-runnable: detectors with an existing calibration file or timing summary are skipped.
set -u
export CRDS_PATH=${CRDS_PATH:-$HOME/crds_cache/jwst_ops} CRDS_SERVER_URL=${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}
P=${PYTHON:-$HOME/miniconda3/envs/jwst/bin/python}
OUT=${OUT:-results/2026-10-22_reset_calib}
FIELD=${FIELD:-results/2026-10-22_reset_field}
LOGS=${LOGS:-$OUT/logs}
mkdir -p "$OUT" "$LOGS"
for det in nrca1 nrca2 nrca3 nrca4 nrcalong nrcb1 nrcb2 nrcb3 nrcb4 nrcblong; do
  calib=$OUT/${det}_reset_calib.asdf
  if [ ! -f "$calib" ]; then
    $P -m snowblind.reset_calib "$HOME/data/jwst_darks/jw09022004001_02101_00001_${det}_uncal.fits" -o "$calib" > "$LOGS/${det}_calib.log" 2>&1
    echo "$det calib exit $?"
  fi
  upper=$(echo "$det" | tr '[:lower:]' '[:upper:]')
  if [ -f "$calib" ] && [ ! -f "$FIELD/$upper/summary.json" ]; then
    $P scripts/exp_reset_field.py "$HOME/data/jwst_6154/reduction_20250516/jw06154001001_02101_00004_${det}_uncal.fits" \
      --calib "$calib" --use-pedestal-mean --write-timing --out "$FIELD" > "$LOGS/${det}_timing.log" 2>&1
    echo "$det timing exit $?"
  fi
done
echo batch done
