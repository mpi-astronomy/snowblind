"""Per-core decay times of snowball core persistence, per detector, from measure_snowball_persistence.py's series files, by
distance from the pixels that saturated.

The series' cores were found with SaturationStep's default n_pix_grow_sat=1, which adds a 1 px ring
around each saturated area in every group.  So a core's pixels whose 8 neighbors are all in the core (its
3x3 erosion) are the ones that saturated, and its other pixels are that 1 px ring.  The selections:
  saturated   core pixels 2 px or more from the core's edge
  1 px        the core's edge pixels: 1 px from a saturated pixel
  2 px        pixels 1 px outside the core
  3 px        pixels 2 px outside the core

Excess as in snowball_persistence_excess.py: a pixel's baseline is taken over the integrations that end before the hits near
it or start more than --clean-after s (default CLEAN_AFTER) after them.  The series files are grouped by
detector, and the cores of a detector's files are pooled.  Stack: the mean over cores of each core's
excess (count-weighted over its selected pixels) per time bin.  For the saturated pixels and the 1 px
ring, the stack from fit_from to FIT_TO s after the end of the integration of the hit is fitted with
A (t / 1000 s)^-alpha, giving alpha for the detector; each core's excess curve is fitted over the same
times with alpha fixed, giving its own A, and its decay time is when that curve falls to THRESHOLD:
1000 s (A / THRESHOLD)^(1/alpha).  Only cores with data in their first EARLY_BY s are used.

Null: for each core, as many control pixels (far from every core) as the core has selected pixels are
given that core's hit integration and go through the same steps, so the null decay times show the spread
that noise alone gives.

usage:
  fit_snowball_persistence.py series1.npz [series2.npz ...] [--fit-from SECONDS] [--clean-after SECONDS]
Writes <detector>_decay.npz next to the first series file of each detector.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from snowball_persistence_excess import (CLEAN_AFTER, THRESHOLD, TIME_EDGES, binned, clean_integrations, load,
                                         per_integration_rates, pixel_excess)

FIT_TO = 16000.0
EARLY_BY = 1000.0
PERCENTILES = [50, 75, 90, 95, 99]
LIMITS = [1000, 2000, 3000, 5000, 10000]
SELECTIONS = {"saturated": (-np.inf, -1.5), "1 px": (-1.5, 0), "2 px": (0, 1.5), "3 px": (1.5, 2.5)}
FITTED = ["saturated", "1 px"]
CORE_SIZES = [(1, 9), (10, 29), (30, 99), (100, 100000)]
DWELLS = [(0, 300), (300, 600), (600, 1e9)]  # s from the hit to the end of its integration


def power_law(t, a, alpha):
    return a * (t / 1000.0) ** -alpha


def weighted_mean_over(groups, select, excess, w, n):
    total = np.zeros((n, excess.shape[1]))
    weight = np.zeros((n, excess.shape[1]))
    np.add.at(total, groups[select], (excess * w)[select])
    np.add.at(weight, groups[select], w[select])
    with np.errstate(invalid="ignore"):
        return total / weight, weight


def in_fit_window(t, weights, fit_from):
    """Rows with data from fit_from to EARLY_BY s: a core whose only data are hours later has its amplitude
    set by points where the power law is tens of times smaller, and so by their noise."""
    return ((t > fit_from) & (t <= EARLY_BY) & (weights > 0)).any(axis=1)


def bin_times(t, weights, rows):
    """Weighted mean time of the blocks in each time bin, over the given rows."""
    mids = []
    for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:]):
        inbin = (t[rows] > lo) & (t[rows] <= hi) & (weights[rows] > 0)
        mids.append(np.average(t[rows][inbin], weights=weights[rows][inbin]) if inbin.any() else np.nan)
    return np.array(mids)


def measured_crossing(table, mids):
    """Time at which the stack falls to THRESHOLD for the last time, interpolated in log-log between the
    last bin above it and the next; nan if no bin is above it or the last one is."""
    ok = np.isfinite(table[:, 0]) & np.isfinite(mids)
    m, t = table[ok, 0], mids[ok]
    above = np.nonzero(m > THRESHOLD)[0]
    if len(above) == 0 or above[-1] + 1 >= len(m):
        return np.nan
    k = above[-1]
    lo = max(m[k + 1], 1e-6)
    return float(np.exp(np.interp(np.log(THRESHOLD), [np.log(lo), np.log(m[k])], [np.log(t[k + 1]), np.log(t[k])])))


def fit_alpha(table, mids, fit_from):
    use = (TIME_EDGES[:-1] >= fit_from) & (TIME_EDGES[1:] <= FIT_TO) & np.isfinite(table[:, 0])
    (a, alpha), cov = curve_fit(power_law, mids[use], table[use, 0], sigma=table[use, 1], p0=(0.005, 1.0),
                                absolute_sigma=True)
    return a, alpha, np.sqrt(np.diag(cov))


def amplitudes(curves, weights, t, alpha, fit_from):
    """Least-squares amplitude of (t / 1000 s)^-alpha for each row, over fit_from < t <= FIT_TO."""
    inside = (t > fit_from) & (t <= FIT_TO) & (weights > 0)
    model = np.where(inside, (np.where(inside, t, 1000.0) / 1000.0) ** -alpha, 0.0)
    wm = weights * model
    with np.errstate(invalid="ignore"):
        return (wm * np.nan_to_num(curves)).sum(axis=1) / (wm * model).sum(axis=1)


def decay_time(a, alpha):
    return 1000.0 * (np.clip(a, 0, None) / THRESHOLD) ** (1 / alpha)


def null_curves(d, usable, control, n_pix, rng, clean_after):
    """For each core, the weighted mean excess of n_pix random usable control pixels with that core's hit
    integration, their baselines taken over the integrations clean of it."""
    per_int = per_integration_rates(d)
    pool = np.nonzero(usable & control)[0]
    n_cores = len(n_pix)
    curves = np.full((n_cores, d["rate"].shape[1]), np.nan)
    weights = np.zeros((n_cores, d["rate"].shape[1]))
    clean_of = {}
    for k in np.nonzero(n_pix)[0]:
        hit = d["core"]["int"][k]
        if hit not in clean_of:
            clean_of[hit] = clean_integrations(d, [hit], clean_after)
        pix = rng.choice(pool, min(n_pix[k], len(pool)), replace=False)
        base = np.nanmedian(np.where(clean_of[hit], per_int[pix], np.nan), axis=1)
        px_excess = d["rate"][pix] - base[:, None]
        px_w = np.where(np.isfinite(px_excess) & (d["count"][pix] > 0), d["count"][pix], 0).astype(float)
        weights[k] = px_w.sum(axis=0)
        with np.errstate(invalid="ignore"):
            curves[k] = (np.nan_to_num(px_excess) * px_w).sum(axis=0) / weights[k]
    return curves, weights


def curves_of_file(path, fit_from, clean_after, rng):
    """Each core's excess curve for every selection (and the null for the fitted ones) in one series file."""
    d = load(path)
    excess, w, usable, control, _ = pixel_excess(d, rng, clean_after)
    n_cores = len(d["core"]["int"])
    t = (d["block_mid"][None, :] - d["core"]["t_end"][:, None]) * 86400
    out = {"detector": str(d["detector"]), "t": t, "n_core": d["core"]["n_core"],
           "dwell": (d["core"]["t_end"] - d["core"]["t_hit"]) * 86400}
    for name, (lo, hi) in SELECTIONS.items():
        select = usable & ~control & (d["pix_dist"] > lo) & (d["pix_dist"] <= hi)
        curves, weights = weighted_mean_over(d["pix_core"], select, excess, w, n_cores)
        out[f"{name}|curves"], out[f"{name}|weights"] = curves, weights
        if name in FITTED:
            n_pix = np.bincount(d["pix_core"][select], minlength=n_cores)
            keep = in_fit_window(t, weights, fit_from)
            null, null_w = null_curves(d, usable, control, np.where(keep, n_pix, 0), rng, clean_after)
            out[f"{name}|null"], out[f"{name}|null_w"] = null, null_w
    return out


def pooled(files, key, fill):
    """The rows of key from every file, padded with fill to the longest block axis."""
    arrays = [f[key] for f in files]
    if arrays[0].ndim == 1:
        return np.concatenate(arrays)
    width = max(a.shape[1] for a in arrays)
    return np.concatenate([np.pad(a, ((0, 0), (0, width - a.shape[1])), constant_values=fill) for a in arrays])


def fit_selection(curves, weights, t, null, null_w, fit_from):
    """Stack, alpha, per-core amplitudes and decay times for one selection, and the null's."""
    keep = in_fit_window(t, weights, fit_from)
    stack = binned(curves, weights, t, keep)
    mids = bin_times(t, weights, keep)
    a, alpha, (a_err, alpha_err) = fit_alpha(stack, mids, fit_from)
    amp = amplitudes(curves, weights, t, alpha, fit_from)
    null_amp = amplitudes(null, null_w, t, alpha, fit_from)
    return {"keep": keep, "stack": stack, "mids": mids, "measured": measured_crossing(stack, mids), "a": a,
            "a_err": a_err, "alpha": alpha, "alpha_err": alpha_err, "amp": amp, "tc": decay_time(amp, alpha),
            "null_amp": null_amp, "null_tc": decay_time(null_amp, alpha)}


def percentiles(values, m):
    return ", ".join(f"{p}% {v:.0f}" for p, v in zip(PERCENTILES, np.percentile(values[m], PERCENTILES)))


def print_fit(name, f, n_core, dwell):
    keep = f["keep"]
    print(f"-- {name}: {keep.sum()} cores")
    print(f"  stack fitted with A (t/1000 s)^-alpha: A = {1000 * f['a']:.2f} +- {1000 * f['a_err']:.2f} mDN/s, "
          f"alpha = {f['alpha']:.2f} +- {f['alpha_err']:.2f}; falls to {1000 * THRESHOLD:.0f} mDN/s at "
          f"{decay_time(f['a'], f['alpha']):.0f} s (fit), {f['measured']:.0f} s (measured stack)")
    print("  time (s) " + "".join(f"{m:8.0f}" for m in f["mids"]))
    print("  measured " + "".join(f"{1000 * v:8.2f}" for v in f["stack"][:, 0]))
    print("  +-       " + "".join(f"{1000 * v:8.2f}" for v in f["stack"][:, 1]))
    print("  fit      " + "".join(f"{1000 * v:8.2f}" for v in power_law(f["mids"], f["a"], f["alpha"])))
    tc, null_tc = f["tc"][keep], f["null_tc"][keep]
    print("  per-core decay time (s), percentiles " + ", ".join(f"{p}%" for p in PERCENTILES))
    print("    cores " + "".join(f"{v:8.0f}" for v in np.percentile(tc, PERCENTILES)))
    print("    null  " + "".join(f"{v:8.0f}" for v in np.percentile(null_tc, PERCENTILES)))
    print("  fraction of cores (of null) with decay time above "
          + ", ".join(f"{lim} s: {np.mean(tc > lim):.3f} ({np.mean(null_tc > lim):.3f})" for lim in LIMITS))
    for lo, hi in CORE_SIZES:
        m = (n_core[keep] >= lo) & (n_core[keep] <= hi)
        if m.sum() >= 5:
            print(f"    core {lo}-{hi} px [{m.sum()}]: {percentiles(tc, m)}")
    for lo, hi in DWELLS:
        m = (dwell[keep] >= lo) & (dwell[keep] < hi)
        if m.sum() >= 5:
            print(f"    saturated {lo:.0f}-{hi:.0f} s [{m.sum()}]: {percentiles(tc, m)}")


def report(paths, fit_from, clean_after, rng):
    files = [curves_of_file(path, fit_from, clean_after, rng) for path in paths]
    detector = files[0]["detector"]
    t = pooled(files, "t", np.nan)
    n_core, dwell = pooled(files, "n_core", 0), pooled(files, "dwell", 0)
    baseline = f"before the hit or more than {clean_after:.0f} s after"
    print(f"\n=== {detector}: {len(n_core)} cores from {len(paths)} series file(s); fits from {fit_from:.0f} to "
          f"{FIT_TO:.0f} s; baselines {baseline}")
    print("Excess (mDN/s) by distance from the saturated pixels, stacked over cores, by time bin (s); "
          "where the stack falls to 5 mDN/s:")
    bins = zip(TIME_EDGES[:-1], TIME_EDGES[1:])
    print("  " + "".ljust(18) + "".join(f"{lo:.0f}-{hi:.0f}".rjust(12) for lo, hi in bins))
    saved = {}
    for name in SELECTIONS:
        curves, weights = pooled(files, f"{name}|curves", np.nan), pooled(files, f"{name}|weights", 0.0)
        keep = in_fit_window(t, weights, fit_from)
        stack, mids = binned(curves, weights, t, keep), bin_times(t, weights, keep)
        key = name.replace(" ", "")
        saved[f"stack_{key}"], saved[f"mids_{key}"] = stack, mids
        print("  " + f"{name} [{keep.sum()}]".ljust(18) + "".join(f"{1000 * v:12.2f}" for v in stack[:, 0])
              + f"   {measured_crossing(stack, mids):.0f} s")
        if name in FITTED:
            null, null_w = pooled(files, f"{name}|null", np.nan), pooled(files, f"{name}|null_w", 0.0)
            null_stack = binned(null, null_w, t, in_fit_window(t, null_w, fit_from))
            saved[f"null_stack_{key}"] = null_stack
            print("  " + "  its null".ljust(18) + "".join(f"{1000 * v:12.2f}" for v in null_stack[:, 0]))
            print("  " + "  +-".ljust(18) + "".join(f"{1000 * v:12.2f}" for v in null_stack[:, 1]))
            f = fit_selection(curves, weights, t, null, null_w, fit_from)
            saved.update({f"{k}_{key}": v for k, v in f.items() if k not in ("stack", "mids")})
            saved[f"print_{key}"] = f
    for name in FITTED:
        print_fit(name, saved.pop(f"print_{name.replace(' ', '')}"), n_core, dwell)

    out = Path(paths[0]).with_name(f"{detector.lower()}_decay.npz")
    np.savez(out, detector=detector, series=[str(p) for p in paths], fit_from=fit_from, fit_to=FIT_TO,
             clean_after=clean_after, threshold=THRESHOLD, time_edges=TIME_EDGES, n_core=n_core, dwell=dwell,
             **saved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", nargs="+")
    parser.add_argument("--fit-from", type=float, default=100.0)
    parser.add_argument("--clean-after", type=float, default=CLEAN_AFTER)
    args = parser.parse_args()
    by_detector = {}
    for path in args.series:
        by_detector.setdefault(str(np.load(path)["detector"]), []).append(path)
    for paths in by_detector.values():
        report(paths, args.fit_from, args.clean_after, np.random.default_rng(3))
