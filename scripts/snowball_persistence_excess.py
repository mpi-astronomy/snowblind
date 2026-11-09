"""Snowball core persistence from measure_snowball_persistence.py's series files: excess rate at the cores
against the time since the end of the integration of the hit, per detector.

Excess: a pixel's block rate minus its baseline, the median of its per-integration rates over the
integrations that are clean for it (ending before the integration of the hit, or starting more than
CLEAN_AFTER s after its end, and likewise for every other core that holds the pixel or has it within
NEAR px).  A core's excess in a block is the count-weighted mean over its core pixels that have no
pixeldq flag and a typical baseline (within 5 robust sigma of the control pixels' baselines).  Run on
its own it prints tables of the excess by time bin; fit_snowball_persistence.py uses its functions.

Null: each control pixel (far from every core) is given the hit integration of a core drawn at random
and goes through the same steps, so its excess shows the floor and the scatter left by the method.

usage:
  snowball_persistence_excess.py series1.npz [series2.npz ...]
"""
import sys

import numpy as np

THRESHOLD = 0.005  # DN/s, as for the PersistenceFlagStep times
CLEAN_AFTER = 100000.0  # s; with 10 ks, baselines on the slow detectors held persistence still at 1-2 mDN/s
TIME_EDGES = np.array([0, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 25000])
CORE_SIZES = [(1, 9), (10, 29), (30, 99), (100, 100000)]
DWELLS = [(0, 300), (300, 1000), (1000, 1e9)]


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    fields = list(d["core_fields"])
    cores = {f: d["cores"][:, k] for k, f in enumerate(fields)}
    cores["int"] = cores["int"].astype(int)
    d["core"] = cores
    d["int_start"], d["int_end"] = d["ints"][:, 2], d["ints"][:, 3]
    d["block_int"] = d["blocks"][:, 0].astype(int)
    d["block_mid"] = d["blocks"][:, 2]
    return d


def per_integration_rates(d):
    """Count-weighted mean rate of each pixel in each integration, (npix, nints)."""
    nints = len(d["int_start"])
    w = d["count"].astype(float)
    r = np.where(w > 0, d["rate"], 0.0)
    total = np.zeros((len(r), nints))
    weight = np.zeros((len(r), nints))
    np.add.at(total.T, d["block_int"], (r * w).T)
    np.add.at(weight.T, d["block_int"], w.T)
    with np.errstate(invalid="ignore"):
        return total / weight


def clean_integrations(d, hit_ints, clean_after=CLEAN_AFTER):
    """(nints,) mask of integrations clean of the hits whose integration indices are in hit_ints: ending before
    the hit's integration starts, or starting more than clean_after s after it ends (np.inf: before only)."""
    clean = np.ones(len(d["int_start"]), bool)
    for i in hit_ints:
        clean &= ~((d["int_end"] > d["int_start"][i]) & (d["int_start"] < d["int_end"][i] + clean_after / 86400))
    return clean


def baselines(d, hit_ints_of_pixel, clean_after=CLEAN_AFTER):
    per_int = per_integration_rates(d)
    base = np.full(len(per_int), np.nan)
    n_clean = np.zeros(len(per_int), int)
    for p, hits in enumerate(hit_ints_of_pixel):
        ok = clean_integrations(d, hits, clean_after) & np.isfinite(per_int[p])
        n_clean[p] = ok.sum()
        if n_clean[p]:
            base[p] = np.median(per_int[p, ok])
    return base, n_clean


def hits_of_pixels(d):
    """For each series pixel, the integrations of the hits of every core that holds it as a core pixel or
    has it within NEAR px."""
    near = d["pix_core"] >= 0
    held = {}
    for k, y, x in zip(d["pix_core"][near], d["pix_y"][near], d["pix_x"][near]):
        held.setdefault((y, x), set()).add(d["core"]["int"][k])
    return [sorted(held.get((y, x), set())) for y, x in zip(d["pix_y"], d["pix_x"])]


def pixel_excess(d, rng, clean_after=CLEAN_AFTER):
    """Excess (npix, nblocks) of every series pixel, its pixel-diff weights (0 where there is no rate), the
    usable-pixel mask, the control-pixel mask, and the core index given to each control pixel for the null."""
    hit_ints = hits_of_pixels(d)
    control = d["pix_core"] < 0
    n_cores = len(d["core"]["int"])
    fake_hits = rng.integers(0, n_cores, control.sum())
    for j, p in enumerate(np.nonzero(control)[0]):
        hit_ints[p] = [d["core"]["int"][fake_hits[j]]]
    base, n_clean = baselines(d, hit_ints, clean_after)

    cb = base[control]
    med = np.nanmedian(cb)
    sigma = 1.4826 * np.nanmedian(np.abs(cb - med))
    typical = np.abs(base - med) <= 5 * sigma
    usable = typical & (d["pix_dq"] == 0) & (n_clean >= 2)

    excess = d["rate"] - base[:, None]
    w = np.where(np.isfinite(excess), d["count"], 0).astype(float)
    excess = np.where(w > 0, excess, 0.0)
    return excess, w, usable, control, fake_hits


def excess_curves(d, rng):
    """Per-core and per-null-pixel excess (n, nblocks) with their pixel-diff weights, plus pixel selections."""
    excess, w, usable, control, fake_hits = pixel_excess(d, rng)
    n_cores = len(d["core"]["int"])

    def mean_over(groups, select, n):
        total = np.zeros((n, excess.shape[1]))
        weight = np.zeros((n, excess.shape[1]))
        np.add.at(total, groups[select], (excess * w)[select])
        np.add.at(weight, groups[select], w[select])
        with np.errstate(invalid="ignore"):
            return total / weight, weight

    shells = {}
    for name, lo, hi in [("core", -np.inf, 0), ("1 px", 0.5, 1.5), ("2 px", 1.5, 2.5)]:
        select = usable & ~control & (d["pix_dist"] > lo) & (d["pix_dist"] < hi)
        shells[name] = mean_over(d["pix_core"], select, n_cores)
    null = (np.where(w[control] > 0, excess[control], np.nan), w[control])
    hit_of_null = d["core"]["int"][fake_hits]
    n_usable_core = np.bincount(d["pix_core"][usable & ~control & (d["pix_dist"] < 0)], minlength=n_cores)
    return shells, null, hit_of_null, n_usable_core, usable


def binned(curves, weights, t, mask=None):
    """Mean excess per time bin over the rows of curves (n, nblocks), each row's blocks weighted by
    pixel-diffs, and its standard error from the scatter between rows (each row's bin mean taken once)."""
    rows = np.ones(len(curves), bool) if mask is None else mask
    out = []
    for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:]):
        inbin = (t[rows] > lo) & (t[rows] <= hi) & (weights[rows] > 0)
        wsum = np.where(inbin, weights[rows], 0).sum(axis=1)
        has = wsum > 0
        if has.sum() < 3:
            out.append((np.nan, np.nan, has.sum()))
            continue
        row_mean = np.where(inbin, np.nan_to_num(curves[rows]) * weights[rows], 0).sum(axis=1)[has] / wsum[has]
        out.append((row_mean.mean(), row_mean.std(ddof=1) / np.sqrt(has.sum()), has.sum()))
    return np.array(out)


def crossing(table):
    """First time-bin upper edge from which every later bin's mean is below THRESHOLD + its error."""
    above = np.nonzero(table[:, 0] - table[:, 1] > THRESHOLD)[0]
    if len(above) == 0:
        return TIME_EDGES[0]
    return TIME_EDGES[above.max() + 1]


def report(path, rng):
    d = load(path)
    shells, (null, null_w), hit_of_null, n_usable_core, usable = excess_curves(d, rng)
    t_end = d["core"]["t_end"]
    t = (d["block_mid"][None, :] - t_end[:, None]) * 86400
    t_null = (d["block_mid"][None, :] - d["int_end"][hit_of_null][:, None]) * 86400
    core, core_w = shells["core"]
    keep = n_usable_core > 0
    dwell = (t_end - d["core"]["t_hit"]) * 86400
    n_core = d["core"]["n_core"]

    print(f"\n=== {d['detector']}: {len(d['exposures'])} exposures, {len(d['int_start'])} integrations, "
          f"{len(t_end)} cores ({keep.sum()} with usable core pixels); "
          f"span {(d['int_end'].max() - d['int_start'].min()) * 86400:.0f} s")
    for name in d["exposures"]:
        print(f"    {name}")
    print(f"Excess rate (mDN/s) +- standard error over cores [number of cores], by time since the end of the "
          f"integration of the hit; threshold {THRESHOLD * 1000:.0f} mDN/s")
    bins = zip(TIME_EDGES[:-1], TIME_EDGES[1:])
    print("  " + "group".ljust(26) + "".join(f"{lo:>5.0f}-{hi:<5.0f}".rjust(19) for lo, hi in bins))

    def line(label, table):
        cells = "".join(("-" if np.isnan(m) else f"{1000 * m:7.2f}+-{1000 * e:5.2f}[{n:.0f}]").rjust(19)
                        for m, e, n in table)
        print("  " + label.ljust(26) + cells)

    tables = {}
    tables["all cores"] = binned(core, core_w, t, keep)
    line("all cores", tables["all cores"])
    for name in ("1 px", "2 px"):
        c, w = shells[name]
        line(f"pixels {name} from core", binned(c, w, t, keep))
    line("null (control pixels)", binned(null, null_w, t_null))
    for lo, hi in CORE_SIZES:
        m = keep & (n_core >= lo) & (n_core <= hi)
        tables[f"core {lo}-{hi} px"] = binned(core, core_w, t, m)
        line(f"core {lo}-{hi} px [{m.sum()}]", tables[f"core {lo}-{hi} px"])
    for lo, hi in DWELLS:
        m = keep & (dwell >= lo) & (dwell < hi)
        tables[f"saturated {lo:.0f}-{hi:.0f} s"] = binned(core, core_w, t, m)
        line(f"saturated {lo:.0f}-{hi:.0f} s [{m.sum()}]", tables[f"saturated {lo:.0f}-{hi:.0f} s"])
    upper = d["core"]["event_cy"] >= 1024
    right = d["core"]["event_cx"] >= 1024
    for qy in (False, True):
        for qx in (False, True):
            m = keep & (upper == qy) & (right == qx)
            name = f"quadrant y{'>' if qy else '<'}1024 x{'>' if qx else '<'}1024"
            tables[name] = binned(core, core_w, t, m)
            line(f"{name} [{m.sum()}]", tables[name])
    print("Time after which every later bin is below the threshold (bin edge, s): "
          + ", ".join(f"{k} {crossing(v):.0f}" for k, v in tables.items()))
    return d, shells, t


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    for path in sys.argv[1:]:
        report(path, rng)
