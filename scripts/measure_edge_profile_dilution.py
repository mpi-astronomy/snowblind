"""Measure how much of the edge series reference is edge noise rather than pixel noise (the dilution of the profile regression).

Runs StreamCleanStep on a saved NRS2 jump ramp with `detector_edges.edge_stream_sums` wrapped so that the
residual blocks it regresses (interior row mean removed) are kept, then on those blocks:

- splits the 96-column window into interleaved column halves A and B (alternate columns, which an even-odd
  column pattern in the residual spoils, and alternate 8-column blocks, the ones to trust); cov(A, B) is the
  edge-noise variance of a half-window mean, var(A) its total; the dilution of the full window is 2 cov / (var_half + cov);
- compares the step's bin gains (regression on the noisy reference) with cross-half gains
  cov(y_b from A, reference from B without b) / cov(A reference without b, B reference without b),
  which are free of the reference's own pixel noise;
- predicts the signal to noise of a plain window mean against its width, and of the profile-weighted
  series, from the undiluted profile and the per-bin noise.

Usage: python measure_edge_profile_dilution.py RUN_DIR  (writes RUN_DIR/edge_dilution_blocks.npz and edge_dilution.log)
"""
import glob
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from stdatamodels.jwst import datamodels

from snowblind import StreamCleanStep, detector_edges

warnings.filterwarnings("ignore")
run_dir = sys.argv[1]
capture_path = os.path.join(run_dir, "edge_dilution_blocks.npz")
BIN = detector_edges.COUPLING_BIN
SMOOTH = detector_edges.EDGE_SMOOTH
WINDOW = detector_edges.EDGE_WINDOW
REACH = detector_edges.EDGE_REACH
WBINS = WINDOW // BIN

if not os.path.exists(capture_path):
    blocks, masks, gains = [], [], []
    original = detector_edges.edge_stream_sums

    def capturing(residual_det, mask_det, **kwargs):
        nrows, ncols = residual_det.shape
        good = mask_det & np.isfinite(residual_det)
        values = np.where(good, residual_det, 0.0).astype(np.float64)
        edges = detector_edges.edge_columns(ncols, REACH)
        interior = slice(edges[0].stop, edges[1].start)
        interior_mean = values[:, interior].sum(axis=1) / np.maximum(good[:, interior].sum(axis=1), 1)
        values = np.where(good, values - interior_mean[:, np.newaxis], 0.0)
        low = values[:, edges[0]]
        high = values[:, edges[1]][:, ::-1]
        blocks.append(np.stack([low, high]).astype(np.float32))
        masks.append(np.stack([good[:, edges[0]], good[:, edges[1]][:, ::-1]]))
        out = original(residual_det, mask_det, **kwargs)
        gains.append(detector_edges.edge_bin_gains(out[1], out[2]))
        return out

    detector_edges.edge_stream_sums = capturing
    jump_path = glob.glob(os.path.join(run_dir, "*_jump.fits"))[0]
    with datamodels.open(jump_path) as jump:
        cleaned = StreamCleanStep().run(jump)
        cleaned.close()
    detector_edges.edge_stream_sums = original
    np.savez_compressed(capture_path, blocks=np.array(blocks), masks=np.array(masks), gains=np.array(gains))

data = np.load(capture_path)
blocks, masks = data["blocks"].astype(np.float64), data["masks"]
ncalls = blocks.shape[0]
# The group differences come first; the last call is the zeroframe.
ndiff = ncalls - 1
blocks, masks = blocks[:ndiff], masks[:ndiff]
nrows = blocks.shape[2]
nbins = REACH // BIN
lines = [f"{ndiff} differences captured (plus the zeroframe, left out), {nrows} rows, {nbins} bins of {BIN} columns"]


def bin_means(values, good, select):
    """Smoothed row means of each bin over the selected columns, and whether the row has enough pixels."""
    sel = np.zeros(REACH, dtype=bool)
    sel[select] = True
    v = values * (good & sel)
    c = (good & sel).astype(np.float64)
    sums = v.reshape(nrows, nbins, BIN).sum(axis=2)
    counts = c.reshape(nrows, nbins, BIN).sum(axis=2)
    sums = uniform_filter1d(sums, SMOOTH, axis=0, mode="nearest")
    counts = uniform_filter1d(counts, SMOOTH, axis=0, mode="nearest")
    return sums, counts


def window_ref(sums, counts, leave_out=None):
    s = sums[:, :WBINS].sum(axis=1)
    c = counts[:, :WBINS].sum(axis=1)
    if leave_out is not None and leave_out < WBINS:
        s = s - sums[:, leave_out]
        c = c - counts[:, leave_out]
    return s / np.maximum(c, 1e-9), c


def covariance(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


halves = {
    "alternate columns": (np.arange(REACH) % 2 == 0, np.arange(REACH) % 2 == 1),
    "alternate 8-column blocks": ((np.arange(REACH) // 8) % 2 == 0, (np.arange(REACH) // 8) % 2 == 1),
}
results = {}
for e, name in enumerate(("edge of channel 0", "edge of channel 3")):
    lines.append(f"\n{name}")
    for split, (sel_a, sel_b) in halves.items():
        cov_ab, var_a, var_b, var_full = 0.0, 0.0, 0.0, 0.0
        for j in range(ndiff):
            values, good = blocks[j, e], masks[j, e]
            sa, ca = bin_means(values, good, sel_a)
            sb, cb = bin_means(values, good, sel_b)
            sf, cf = bin_means(values, good, slice(None))
            xa, _ = window_ref(sa, ca)
            xb, _ = window_ref(sb, cb)
            xf, _ = window_ref(sf, cf)
            cov_ab += covariance(xa, xb)
            var_a += covariance(xa, xa)
            var_b += covariance(xb, xb)
            var_full += covariance(xf, xf)
        var_half = (var_a + var_b) / 2
        dilution = 2 * cov_ab / (var_half + cov_ab)
        lines.append(
            f"  {split}: edge-noise variance of the window mean {cov_ab / ndiff:.4f} DN^2 (smoothed over {SMOOTH} rows), "
            f"pixel-noise variance of the full window mean {(var_half - cov_ab) / 2 / ndiff:.4f}, "
            f"measured full-window variance {var_full / ndiff:.4f}; "
            f"dilution of the regression (signal share of the reference) {dilution:.3f}"
        )
        results[(e, split)] = dilution
    # Profiles: the step's (regression on the noisy reference) and the cross-half one.
    sel_a, sel_b = halves["alternate columns"]
    xy_step = np.zeros(nbins)
    xx_step = np.zeros(nbins)
    xy_cross = np.zeros(nbins)
    xx_cross = np.zeros(nbins)
    bin_var = np.zeros(nbins)
    bin_cov = np.zeros(nbins)
    for j in range(ndiff):
        values, good = blocks[j, e], masks[j, e]
        sa, ca = bin_means(values, good, sel_a)
        sb, cb = bin_means(values, good, sel_b)
        sf, cf = bin_means(values, good, slice(None))
        for b in range(nbins):
            xf, _ = window_ref(sf, cf, b)
            yf = sf[:, b] / np.maximum(cf[:, b], 1e-9)
            xy_step[b] += covariance(yf, xf)
            xx_step[b] += covariance(xf, xf)
            xa, _ = window_ref(sa, ca, b)
            xb, _ = window_ref(sb, cb, b)
            ya = sa[:, b] / np.maximum(ca[:, b], 1e-9)
            yb = sb[:, b] / np.maximum(cb[:, b], 1e-9)
            # Both products are free of shared pixel noise: y from A, reference from B (and the reverse), averaged.
            xy_cross[b] += (covariance(ya, xb) + covariance(yb, xa)) / 2
            xx_cross[b] += covariance(xa, xb)
            bin_var[b] += covariance(yf, yf)
            bin_cov[b] += covariance(ya, yb)
    p_step = xy_step / xx_step
    p_cross = xy_cross / xx_cross
    step_gains = np.nanmean(data["gains"][:ndiff, e], axis=0)
    lines.append("  bin (columns from the edge) | step's gain (mean of per-difference gains) | same from the sums here | "
                 "cross-half gain | bin edge-noise share")
    for b in range(nbins):
        share = bin_cov[b] / bin_var[b]
        lines.append(
            f"  {b * BIN:4d}-{(b + 1) * BIN:3d} | {step_gains[b]:6.3f} | {p_step[b]:6.3f} | {p_cross[b]:6.3f} | {share:5.3f}"
        )
    # Signal to noise of a plain window mean against its width, and of the weighted series, per bin noise.
    # Per-bin pixel-noise variance (full bin, smoothed): var - edge part.
    noise_var = bin_var - bin_cov  # per difference sums; relative values are all that matter
    p = np.clip(p_cross, 0, None)
    lines.append("  predicted signal to noise (relative) of a plain window mean against its width, and of the weighted series:")
    ref = None
    for wb in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        signal = p[:wb].mean()
        noise = np.sqrt(noise_var[:wb].sum()) / wb
        snr = signal / noise
        ref = snr if ref is None else ref
        lines.append(f"    window {wb * BIN:3d} columns: {snr / ref:.3f}")
    weighted = np.sqrt(np.sum(p**2 / noise_var))
    # Scale the weighted series' SNR like the plain mean's: estimate of unit edge noise with error 1/weighted.
    plain_ref = p[:1].mean() / (np.sqrt(noise_var[:1].sum()) / 1)
    lines.append(f"    profile-weighted over all {REACH} columns: {weighted / plain_ref:.3f}")
    inside_step = p_step[:WBINS].mean() / p_step[WBINS:WBINS + 2].mean()
    inside_cross = p_cross[:WBINS].mean() / p_cross[WBINS:WBINS + 2].mean()
    lines.append(f"  inside-window gains relative to the ones just outside: step {inside_step:.3f}, cross {inside_cross:.3f}")

text = "\n".join(lines)
print(text)
with open(os.path.join(run_dir, "edge_dilution.log"), "w") as f:
    f.write(text + "\n")
