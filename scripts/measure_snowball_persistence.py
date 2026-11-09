"""Snowball core persistence in dark exposures: the rate at each snowball's core pixels, and at the pixels
up to NEAR px around it, in every block of groups of every integration of a detector's exposures.

Pass 1 (`exposure`), one exposure at a time: each integration goes through Detector1 to the jump step
(snowblind_detector1 settings, without the jwst snowball expansion).  Each group difference, in DN/s, has
the median of its line segment subtracted (a line along the fast axis within one amplifier, over pixels
without a pixeldq flag), which removes the 1/f striping.  The differences are averaged per pixel in
blocks of about BLOCK_SECONDS, leaving out those next to a SATURATED, JUMP_DET or DO_NOT_USE group.
Saved in workdir: the block rates and counts, their times, the pixeldq, and the snowball cores found by
SnowblindStep's rule (pixels saturating in the jump group or the next, in an event holding a disk of
min_radius); a core found again in a later group of the same integration is merged into the first.

Pass 2 (`series`): the block rates over every integration of every exposure in workdir of the core
pixels, of the pixels up to NEAR px around each core, and of N_CONTROL random pixels at least
CONTROL_CLEARANCE px from every core; saved to one npz for the analysis.

Times are MJD: a frame is placed at the middle of its read, the end of an integration is its
int_end_MJD_UTC (the last read; the reset follows), and a block at the mean of its differences' times.

usage:
  measure_snowball_persistence.py list uncal1.fits [uncal2.fits ...]
  measure_snowball_persistence.py run outdir uncal1.fits [uncal2.fits ...] [--jobs N] [--per-series N] [--keep-work]
  measure_snowball_persistence.py exposure uncal.fits workdir
  measure_snowball_persistence.py series workdir out.npz

`run` groups the files by detector and, for each detector, runs pass 1 on every exposure (one Python
process per exposure) and pass 2, writing outdir/<detector>_series.npz and outdir/<detector>.log.  With
--per-series N, each detector's exposures, in name order, are split into sets of N with a series file
each (outdir/<detector>_<k>_series.npz).  With --jobs N, N sets run at a time.  A set whose series file
exists is skipped; an exposure whose pass 1 is in the set's work folder is not redone.  The work folder
(about 1 GB per exposure) is removed after pass 2 unless --keep-work.  It needs a snowblind whose
SnowblindStep requires a saturated core (issues #5 and #16).  The saturation step runs with
n_pix_grow_sat=1, the jwst default, whatever CRDS holds, which fit_snowball_persistence.py relies on.
"""
import argparse
import importlib
import json
import logging
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

DO_NOT_USE, SATURATED, JUMP_DET, REFERENCE_PIXEL = 1, 2, 4, 1 << 31
BLOCK_SECONDS = 100.0
NEAR = 2
N_CONTROL = 5000
CONTROL_CLEARANCE = 10
N_AMPLIFIERS = 4

logging.getLogger("snowblind").setLevel(logging.ERROR)


def _single_integration(uncal, i):
    """Integration i of an uncal file as a ramp, read without the other integrations.

    Opening the file as a RampModel reads and converts every integration at once, about 3.5 times the
    file size in memory.
    """
    from astropy.io import fits
    from jwst import datamodels

    with fits.open(uncal, memmap=False) as hdul:
        names = [hdu.name for hdu in hdul]
        hdus = [fits.PrimaryHDU(header=hdul[0].header.copy()),
                fits.ImageHDU(np.asarray(hdul["SCI"].section[i:i + 1], dtype=np.float32), name="SCI"),
                fits.BinTableHDU(hdul["INT_TIMES"].data[i:i + 1], name="INT_TIMES")]
        if "ZEROFRAME" in names:
            hdus.append(fits.ImageHDU(np.asarray(hdul["ZEROFRAME"].section[i:i + 1], dtype=np.float32),
                                      name="ZEROFRAME"))
        result = datamodels.RampModel(fits.HDUList(hdus))
    result.groupdq = np.zeros(result.data.shape, dtype=np.uint8)
    result.pixeldq = np.zeros(result.data.shape[-2:], dtype=np.uint32)
    result.meta.exposure.nints = 1
    return result


def jump_ramp(uncal, i):
    """Detector1 through the jump step for integration i, with the snowblind_detector1 jump settings."""
    from snowblind.pipeline import JUMP_DEFAULTS
    from snowblind.reset_calib import DETECTOR1_TO_JUMP

    result = _single_integration(uncal, i)
    for module_name, class_name, params in DETECTOR1_TO_JUMP:
        if class_name == "JumpStep":
            params = {**JUMP_DEFAULTS, "expand_large_events": False}
        if class_name == "SaturationStep":
            # the jwst default, whatever CRDS holds: the analysis takes a core's pixels 2 px from its edge as the
            # ones that saturated
            params = {"n_pix_grow_sat": 1}
        step_class = getattr(importlib.import_module(module_name), class_name)
        result = step_class.call(result, **params)
    return result


def line_medians_removed(diff, exclude, fastaxis):
    """diff (ny, nx) minus the median of each line segment within an amplifier, over the pixels not excluded."""
    transpose = abs(fastaxis) == 2
    lines = diff.T if transpose else diff
    left_out = exclude.T if transpose else exclude
    n_lines, n_along = lines.shape
    shape = (n_lines, N_AMPLIFIERS, n_along // N_AMPLIFIERS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(left_out, np.nan, lines).reshape(shape), axis=2)
    corrected = (lines.reshape(shape) - np.nan_to_num(med)[:, :, None]).reshape(n_lines, n_along)
    return corrected.T if transpose else corrected


def group_times(int_start, nframes, groupgap, tframe, ngroups):
    k = np.arange(ngroups)
    return int_start + (k * (nframes + groupgap) + (nframes - 1) / 2 + 0.5) * tframe / 86400


def blocks_of_integration(sci, gdq, pixeldq, times, tgroup, fastaxis):
    """Block-averaged, line-corrected rates (nblocks, ny, nx), their counts, the blocks' mean times and lengths.

    Pixels with a pixeldq flag keep a rate (the analysis decides whether to use them) but are left out of
    the line medians.
    """
    ndiff = sci.shape[0] - 1
    per_block = max(1, int(round(BLOCK_SECONDS / tgroup)))
    starts = range(0, ndiff, per_block)
    rates = np.full((len(starts),) + sci.shape[1:], np.nan, dtype=np.float32)
    counts = np.zeros((len(starts),) + sci.shape[1:], dtype=np.uint8)
    mids, seconds = np.zeros(len(starts)), np.zeros(len(starts))
    flagged = pixeldq != 0
    for b, s in enumerate(starts):
        e = min(s + per_block, ndiff)
        total = np.zeros(sci.shape[1:])
        for d in range(s, e):
            bad = ((gdq[d + 1] & (SATURATED | JUMP_DET | DO_NOT_USE)) != 0) | ((gdq[d] & (SATURATED | DO_NOT_USE)) != 0)
            corrected = line_medians_removed((sci[d + 1] - sci[d]) / tgroup, bad | flagged, fastaxis)
            total += np.where(bad, 0, corrected)
            counts[b] += ~bad
        rates[b] = np.where(counts[b] > 0, total / np.maximum(counts[b], 1), np.nan)
        mids[b] = (times[s:e] + times[s + 1:e + 1]).mean() / 2
        seconds[b] = (e - s) * tgroup
    return rates, counts, mids, seconds


def cores_of_integration(gdq, refpix, times):
    """Cores of the integration's snowballs: list of dicts with the hit group, time and core pixels."""
    from snowblind import SnowblindStep

    sat = (gdq & SATURATED) != 0
    jump = (gdq & JUMP_DET) != 0
    step = SnowblindStep(growth_factor=0)
    saturating = step._saturating_in_group_or_next(sat)
    owner = np.full(gdq.shape[1:], -1)
    found = []
    for g in range(1, gdq.shape[0]):
        if not jump[g].any():
            continue
        events, cores = step.dilate_jump_slice(jump[g], saturating[g], refpix)
        if not cores.any():
            continue
        labels, n = ndimage.label(events, structure=np.ones((3, 3)))
        for k, box in enumerate(ndimage.find_objects(labels), start=1):
            event = labels[box] == k
            core = cores[box] & event
            if not core.any():
                continue
            ys, xs = np.nonzero(core)
            ys, xs = ys + box[0].start, xs + box[1].start
            earlier = np.unique(owner[ys, xs])
            earlier = earlier[earlier >= 0]
            if len(earlier):
                first = found[earlier.min()]
                pixels = set(zip(first["y"], first["x"])) | set(zip(ys.tolist(), xs.tolist()))
                first["y"], first["x"] = map(list, zip(*sorted(pixels)))
                owner[ys, xs] = earlier.min()
                first["merged"] += 1
                continue
            owner[ys, xs] = len(found)
            cy, cx = ndimage.center_of_mass(event)
            found.append({
                "group": g,
                "t_hit": float((times[g - 1] + times[g]) / 2),
                "y": ys.tolist(),
                "x": xs.tolist(),
                "event_area": int(event.sum()),
                "event_cy": float(cy + box[0].start),
                "event_cx": float(cx + box[1].start),
                "merged": 0,
            })
    return found


def run_exposure(uncal, workdir):
    from astropy.io import fits

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    name = Path(uncal).name.replace("_uncal.fits", "")
    meta_path = workdir / f"{name}.json"
    if meta_path.exists():
        print(f"{name}: done before", flush=True)
        return
    h = fits.getheader(uncal)
    int_times = fits.getdata(uncal, "INT_TIMES")
    nframes, groupgap, tframe, tgroup = h["NFRAMES"], h["GROUPGAP"], h["TFRAME"], h["TGROUP"]
    nints, ngroups, fastaxis = h["NINTS"], h["NGROUPS"], h["FASTAXIS"]
    info = {
        "exposure": name, "detector": h["DETECTOR"], "readpatt": h["READPATT"], "nints": nints,
        "ngroups": ngroups, "nframes": nframes, "groupgap": groupgap, "tframe": tframe, "tgroup": tgroup,
        "fastaxis": fastaxis, "integrations": [],
    }
    for i in range(nints):
        start, end = float(int_times["int_start_MJD_UTC"][i]), float(int_times["int_end_MJD_UTC"][i])
        times = group_times(start, nframes, groupgap, tframe, ngroups)
        jumped = jump_ramp(uncal, i)
        sci, gdq, pixeldq = jumped.data[0], jumped.groupdq[0], jumped.pixeldq
        refpix = (pixeldq & REFERENCE_PIXEL) != 0
        rates, counts, mids, seconds = blocks_of_integration(sci, gdq, pixeldq, times, tgroup, fastaxis)
        np.save(workdir / f"{name}_int{i:02d}_rates.npy", rates)
        np.save(workdir / f"{name}_int{i:02d}_counts.npy", counts)
        if i == 0:
            np.save(workdir / f"{name}_pixeldq.npy", pixeldq)
        cores = cores_of_integration(gdq, refpix, times)
        info["integrations"].append({
            "start": start, "end": end, "block_mids": mids.tolist(), "block_seconds": seconds.tolist(),
            "cores": cores,
        })
        print(f"{name} integration {i + 1}/{nints}: {len(cores)} cores", flush=True)
        del jumped, sci, gdq
    meta_path.write_text(json.dumps(info))


def _near_pixels(ys, xs, shape):
    """Core pixels and the pixels within NEAR px of the core, with a signed distance: minus the distance
    to the nearest pixel outside the core for core pixels, the distance to the core for the others."""
    y0, y1 = max(min(ys) - NEAR, 0), min(max(ys) + NEAR + 1, shape[0])
    x0, x1 = max(min(xs) - NEAR, 0), min(max(xs) + NEAR + 1, shape[1])
    core = np.zeros((y1 - y0, x1 - x0), bool)
    core[np.array(ys) - y0, np.array(xs) - x0] = True
    padded = np.pad(core, 1)
    inside = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    outside = ndimage.distance_transform_edt(~core)
    dist = np.where(core, -inside, outside)
    keep = core | (outside <= NEAR)
    yy, xx = np.nonzero(keep)
    return yy + y0, xx + x0, dist[keep]


def run_series(workdir, out):
    workdir = Path(workdir)
    infos = sorted((json.loads(p.read_text()) for p in workdir.glob("*.json")),
                   key=lambda info: info["integrations"][0]["start"])
    pixeldq = np.load(workdir / f"{infos[0]['exposure']}_pixeldq.npy")
    shape = pixeldq.shape

    ints, blocks, cores = [], [], []
    for e, info in enumerate(infos):
        for i, integ in enumerate(info["integrations"]):
            n = len(ints)
            ints.append((e, i, integ["start"], integ["end"], info["tgroup"], info["readpatt"]))
            for b, (mid, seconds) in enumerate(zip(integ["block_mids"], integ["block_seconds"])):
                blocks.append((n, b, mid, seconds))
            for c in integ["cores"]:
                cores.append({**c, "int": n, "t_end": integ["end"]})

    pix_core, pix_y, pix_x, pix_dist = [], [], [], []
    near_any = np.zeros(shape, bool)
    for k, c in enumerate(cores):
        yy, xx, dd = _near_pixels(c["y"], c["x"], shape)
        pix_core.append(np.full(len(yy), k))
        pix_y.append(yy)
        pix_x.append(xx)
        pix_dist.append(dd)
        near_any[c["y"], c["x"]] = True
    clear = ndimage.distance_transform_edt(~near_any) >= CONTROL_CLEARANCE
    clear &= (pixeldq & REFERENCE_PIXEL) == 0
    rng = np.random.default_rng(1)
    cy, cx = np.nonzero(clear)
    pick = rng.choice(len(cy), size=min(N_CONTROL, len(cy)), replace=False)
    pix_core.append(np.full(len(pick), -1))
    pix_y.append(cy[pick])
    pix_x.append(cx[pick])
    pix_dist.append(np.full(len(pick), np.nan))
    pix_core, pix_y, pix_x, pix_dist = map(np.concatenate, (pix_core, pix_y, pix_x, pix_dist))

    rate = np.full((len(pix_y), len(blocks)), np.nan, dtype=np.float32)
    count = np.zeros((len(pix_y), len(blocks)), dtype=np.uint8)
    column = 0
    for e, info in enumerate(infos):
        for i in range(len(info["integrations"])):
            r = np.load(workdir / f"{info['exposure']}_int{i:02d}_rates.npy", mmap_mode="r")
            c = np.load(workdir / f"{info['exposure']}_int{i:02d}_counts.npy", mmap_mode="r")
            for b in range(r.shape[0]):
                rate[:, column] = np.asarray(r[b])[pix_y, pix_x]
                count[:, column] = np.asarray(c[b])[pix_y, pix_x]
                column += 1
    assert column == len(blocks)

    core_fields = ["int", "group", "t_hit", "t_end", "event_area", "event_cy", "event_cx", "merged"]
    np.savez_compressed(
        out,
        detector=infos[0]["detector"],
        exposures=np.array([info["exposure"] for info in infos]),
        ints=np.array([t[:5] for t in ints], dtype=float),
        int_readpatt=np.array([t[5] for t in ints]),
        blocks=np.array(blocks, dtype=float),
        cores=np.array([[c[f] for f in core_fields] + [len(c["y"])] for c in cores], dtype=float),
        core_fields=np.array(core_fields + ["n_core"]),
        pix_core=pix_core, pix_y=pix_y, pix_x=pix_x, pix_dist=pix_dist, pix_dq=pixeldq[pix_y, pix_x],
        rate=rate, count=count,
    )
    print(f"{out}: {len(cores)} cores, {len(pix_y)} pixels, {len(blocks)} blocks", flush=True)


def _detector_of(path):
    return Path(path).name.replace("_uncal.fits", "").split("_")[-1]


def run_list(uncals):
    from astropy.io import fits

    rows = []
    for path in uncals:
        h = fits.getheader(path)
        rows.append((h["DETECTOR"], h["EXPSTART"], Path(path).name, h["READPATT"], h["NINTS"], h["NGROUPS"],
                     h["EXPEND"]))
    for detector, start, name, readpatt, nints, ngroups, end in sorted(rows):
        print(f"{detector:9s} {name:45s} {readpatt:9s} {nints:3d} x {ngroups:3d}  "
              f"start MJD {start:.5f}  length {(end - start) * 86400:6.0f} s")


def _run_set(name, uncals, outdir, keep_work):
    workdir = outdir / f"work_{name}"
    out = outdir / f"{name}_series.npz"
    log = outdir / f"{name}.log"
    if out.exists():
        return f"{name}: {out} exists, skipped"
    with open(log, "a") as fh:
        for uncal in uncals:
            done = subprocess.run([sys.executable, __file__, "exposure", str(uncal), str(workdir)], stdout=fh,
                                  stderr=subprocess.STDOUT)
            if done.returncode:
                return f"{name}: failed on {Path(uncal).name}, see {log}"
        done = subprocess.run([sys.executable, __file__, "series", str(workdir), str(out)], stdout=fh,
                              stderr=subprocess.STDOUT)
        if done.returncode:
            return f"{name}: pass 2 failed, see {log}"
    if not keep_work:
        shutil.rmtree(workdir)
    return f"{name}: {out}"


def run_all(outdir, uncals, jobs, per_series, keep_work):
    import jwst
    import stcal
    import snowblind
    from snowblind import SnowblindStep

    if not hasattr(SnowblindStep, "_events_at_edge"):
        sys.exit(f"snowblind in {Path(snowblind.__file__).parent} predates the saturated-core rule of SnowblindStep")
    print(f"snowblind {snowblind.__version__} from {Path(snowblind.__file__).parent}; jwst {jwst.__version__}, "
          f"stcal {stcal.__version__}", flush=True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_detector = {}
    for path in sorted(uncals, key=lambda p: Path(p).name):
        by_detector.setdefault(_detector_of(path), []).append(Path(path))
    sets = {}
    for detector, files in by_detector.items():
        if per_series:
            for k in range(0, len(files), per_series):
                sets[f"{detector}_{k // per_series + 1:02d}"] = files[k:k + per_series]
        else:
            sets[detector] = files
    with ThreadPoolExecutor(jobs) as pool:
        futures = [pool.submit(_run_set, name, files, outdir, keep_work) for name, files in sets.items()]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["list", "run", "exposure", "series"])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--per-series", type=int, default=0)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    if args.mode == "list":
        run_list(args.paths)
    elif args.mode == "run":
        run_all(args.paths[0], args.paths[1:], args.jobs, args.per_series, args.keep_work)
    elif args.mode == "exposure":
        run_exposure(*args.paths)
    else:
        run_series(*args.paths)
