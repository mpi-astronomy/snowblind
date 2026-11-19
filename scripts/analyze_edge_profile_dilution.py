"""Second pass over the captured edge residual blocks (see measure_edge_profile_dilution.py).

- dilution of the 96-column reference from interleaved halves of 1, 2, 4, 8 and 16 columns;
- the even-odd column pattern: variance of (even - odd) against (even + odd) half-window means, per bin;
- the step's bin gains against cross-half gains (8-column blocks), and the predicted signal to noise of
  plain window means against their width and of the weighted series, with and without the clip at 1.5.

Usage: python analyze_edge_profile_dilution.py RUN_DIR
"""
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

from snowblind import detector_edges

run_dir = sys.argv[1]
BIN = detector_edges.COUPLING_BIN
SMOOTH = detector_edges.EDGE_SMOOTH
WINDOW = detector_edges.EDGE_WINDOW
REACH = detector_edges.EDGE_REACH
WBINS = WINDOW // BIN
data = np.load(os.path.join(run_dir, "edge_dilution_blocks.npz"))
blocks, masks = data["blocks"].astype(np.float64)[:-1], data["masks"][:-1]
ndiff, _, nrows, _ = blocks.shape
nbins = REACH // BIN
lines = []


def bin_means(values, good, sel):
    v = values * (good & sel)
    c = (good & sel).astype(np.float64)
    sums = uniform_filter1d(v.reshape(nrows, nbins, BIN).sum(axis=2), SMOOTH, axis=0, mode="nearest")
    counts = uniform_filter1d(c.reshape(nrows, nbins, BIN).sum(axis=2), SMOOTH, axis=0, mode="nearest")
    return sums, counts


def window_ref(sums, counts, leave_out=None):
    s = sums[:, :WBINS].sum(axis=1)
    c = counts[:, :WBINS].sum(axis=1)
    if leave_out is not None and leave_out < WBINS:
        s = s - sums[:, leave_out]
        c = c - counts[:, leave_out]
    return s / np.maximum(c, 1e-9)


def cov(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def split(width):
    k = (np.arange(REACH) // width) % 2
    return k == 0, k == 1


for e, name in enumerate(("edge of channel 0", "edge of channel 3")):
    lines.append(f"\n{name}")
    for width in (1, 2, 4, 8, 16):
        sel_a, sel_b = split(width)
        c_ab = v_h = 0.0
        for j in range(ndiff):
            values, good = blocks[j, e], masks[j, e]
            xa = window_ref(*bin_means(values, good, sel_a))
            xb = window_ref(*bin_means(values, good, sel_b))
            c_ab += cov(xa, xb)
            v_h += (cov(xa, xa) + cov(xb, xb)) / 2
        share = 2 * c_ab / (v_h + c_ab)
        lines.append(f"  halves of alternate {width:2d}-column blocks: signal share of the reference {share:.3f}")
    # Even-odd pattern per bin: rms of (even - odd)/2 against the expected pixel noise, from the half-bin means.
    sel_a, sel_b = split(1)
    diff_var = np.zeros(nbins)
    sum_var = np.zeros(nbins)
    for j in range(ndiff):
        values, good = blocks[j, e], masks[j, e]
        sa, ca = bin_means(values, good, sel_a)
        sb, cb = bin_means(values, good, sel_b)
        ya = sa / np.maximum(ca, 1e-9)
        yb = sb / np.maximum(cb, 1e-9)
        for b in range(nbins):
            diff_var[b] += cov(ya[:, b] - yb[:, b], ya[:, b] - yb[:, b]) / 4
            sum_var[b] += cov(ya[:, b] + yb[:, b], ya[:, b] + yb[:, b]) / 4
    lines.append("  even minus odd column means per bin (smoothed over 32 rows):")
    lines.append("    rms DN, and the ratio to the rms of the bin mean")
    lines.append("    " + " ".join(f"{np.sqrt(d / ndiff):.3f}" for d in diff_var))
    lines.append("    " + " ".join(f"{np.sqrt(d / s):.2f}" for d, s in zip(diff_var, sum_var)))
    # Gains with the 8-column-block split.
    sel_a, sel_b = split(8)
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
        sf, cf = bin_means(values, good, np.ones(REACH, dtype=bool))
        ya, yb, yf = sa / np.maximum(ca, 1e-9), sb / np.maximum(cb, 1e-9), sf / np.maximum(cf, 1e-9)
        for b in range(nbins):
            xf = window_ref(sf, cf, b)
            xa = window_ref(sa, ca, b)
            xb = window_ref(sb, cb, b)
            xy_step[b] += cov(yf[:, b], xf)
            xx_step[b] += cov(xf, xf)
            xy_cross[b] += (cov(ya[:, b], xb) + cov(yb[:, b], xa)) / 2
            xx_cross[b] += cov(xa, xb)
            bin_var[b] += cov(yf[:, b], yf[:, b])
            bin_cov[b] += cov(ya[:, b], yb[:, b])
    p_step = xy_step / xx_step
    p_cross = xy_cross / xx_cross
    lines.append("  bin | step's gain (noisy reference) | cross-half gain (8-column blocks) | ratio | bin edge-noise share")
    for b in range(nbins):
        lines.append(
            f"  {b * BIN:4d}-{(b + 1) * BIN:3d} | {p_step[b]:6.3f} | {p_cross[b]:6.3f} | "
            f"{p_step[b] / p_cross[b]:6.3f} | {bin_cov[b] / bin_var[b]:6.3f}"
        )
    noise_var = np.maximum(bin_var - bin_cov, 1e-9)
    p = np.clip(p_cross, 0, None)
    ref = p[:1].mean() / np.sqrt(noise_var[:1].sum())
    lines.append("  predicted signal to noise, relative to the 16-column window,"
                 " from the cross-half profile and per-bin noise:")
    for wb in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        snr = p[:wb].mean() * wb / np.sqrt(noise_var[:wb].sum())
        lines.append(f"    plain mean over {wb * BIN:3d} columns: {snr / ref:.3f}")
    variants = (
        ("true profile", p),
        ("profile clipped at 1.5", np.clip(p, 0, 1.5)),
        ("step's profile clipped at 1.5", np.clip(p_step, 0, 1.5)),
    )
    for label, used in variants:
        # Estimator sum(u c m)/sum(u^2 c) with weights u: mean = sum(u p)/sum(u^2) times e, error = sqrt(sum(u^2 s^2))/sum(u^2).
        mean_gain = np.sum(used * p) / np.sum(used**2)
        err = np.sqrt(np.sum(used**2 * noise_var)) / np.sum(used**2)
        lines.append(f"    weighted series with the {label}: {mean_gain / err / ref:.3f} (bias of the series {mean_gain:.3f}, "
                     f"bin 0 corrected at {used[0] * mean_gain / p[0]:.2f} of its noise)")

text = "\n".join(lines)
print(text)
with open(os.path.join(run_dir, "edge_dilution_analysis.log"), "w") as f:
    f.write(text + "\n")


# Third part: dilution of a plain-window profile against the window width (8-column-block halves), and the net
# correction per bin (series bias times the weight used over the true gain) for the step's profiles.
more = []
sel_a, sel_b = split(8)
full = np.ones(REACH, dtype=bool)
for e, name in enumerate(("edge of channel 0", "edge of channel 3")):
    more.append(f"\n{name}: signal share of a plain window mean (what a plain-window profile is scaled by) against the window")
    for wcols in (32, 48, 64, 96, 128, 160):
        wb = wcols // BIN
        c_ab = v_h = 0.0
        for j in range(ndiff):
            values, good = blocks[j, e], masks[j, e]
            sa, ca = bin_means(values, good, sel_a)
            sb, cb = bin_means(values, good, sel_b)
            xa = sa[:, :wb].sum(axis=1) / np.maximum(ca[:, :wb].sum(axis=1), 1e-9)
            xb = sb[:, :wb].sum(axis=1) / np.maximum(cb[:, :wb].sum(axis=1), 1e-9)
            c_ab += cov(xa, xb)
            v_h += (cov(xa, xa) + cov(xb, xb)) / 2
        more.append(f"  window {wcols:3d}: {2 * c_ab / (v_h + c_ab):.3f}")
    xy_step = np.zeros(nbins)
    xx_step = np.zeros(nbins)
    xy_cross = np.zeros(nbins)
    xx_cross = np.zeros(nbins)
    for j in range(ndiff):
        values, good = blocks[j, e], masks[j, e]
        sa, ca = bin_means(values, good, sel_a)
        sb, cb = bin_means(values, good, sel_b)
        sf, cf = bin_means(values, good, full)
        ya, yb, yf = sa / np.maximum(ca, 1e-9), sb / np.maximum(cb, 1e-9), sf / np.maximum(cf, 1e-9)
        for b in range(nbins):
            xf = window_ref(sf, cf, b)
            xa = window_ref(sa, ca, b)
            xb = window_ref(sb, cb, b)
            xy_step[b] += cov(yf[:, b], xf)
            xx_step[b] += cov(xf, xf)
            xy_cross[b] += (cov(ya[:, b], xb) + cov(yb[:, b], xa)) / 2
            xx_cross[b] += cov(xa, xb)
    p = np.clip(xy_cross / xx_cross, 0, None)
    p_step = xy_step / xx_step
    more.append("  net correction per bin (bins 0 to 7) with the weighted series, as a fraction of the bin's edge noise:")
    variants = (("step's gains clipped at 1.5", np.clip(p_step, 0, 1.5)), ("step's gains not clipped", np.clip(p_step, 0, None)))
    for label, used in variants:
        bias = np.sum(used * p) / np.sum(used**2)
        net = " ".join(f"{bias * used[b] / p[b]:.2f}" if p[b] > 0.1 else "  -  " for b in range(8))
        more.append(f"    {label} (series bias {bias:.2f}): " + net)
text = "\n".join(more)
print(text)
with open(os.path.join(run_dir, "edge_dilution_analysis.log"), "a") as f:
    f.write(text + "\n")
