"""Measure persistence in the _rate files of --input_dir: the excess count rate, in later exposures of the same
detector, at the pixels that saturated or came close to it in an earlier exposure, against the time since then.

For each detector the exposures are taken in order of start time.  Each exposure i is a source: its pixels are put
in classes by how much signal they collected (rate times integration time, in DN), with the saturated pixels in
classes of their own.  Each later exposure j of the same detector, starting up to --max_delay seconds after the end
of i, is a target: its local background is removed (photutils Background2D, with the bright pixels of i and of j
masked), and for each class of i the median of the residual rate of j over the pixels of that class is the
persistence.  Pixels that saturated or collected a lot of signal in any exposure after i, up to and including j,
are left out, so each measurement follows one exposure's illumination.

When i and j point at the same place (a filter change without a dither), the sources of i are still on the same
pixels in j; the table records the pointing shift between them in pixels (shift_pix), and the summary uses only
pairs that moved by more than --min_shift pixels.

A pixel counts as saturated when it has SATURATED in the DQ or its signal (rate times integration time) is above the
top unsaturated class edge: before jwst 2.0 ramp fitting set SATURATED only where every group saturated, so classing
by signal gives the same classes for old and new products.  A SATURATED pixel with little signal is mostly a cosmic
ray that saturated late in the ramp and has a class of its own ("sat 0k-60k").

The excess of a class is its median residual minus the median residual of the target's other usable pixels (those in
no class of the source), and the summary bins it by the time from the start of the source to the start of the target,
the interval PersistenceFlagStep compares with its ``time``.  Neighbouring pixels of one source land together on one
patch of the target, so they are not independent: each row also has the excess measured over sources (the connected
groups of a class's pixels, 8-connected): the median over groups of their mean residual, minus the median residual of
the rest, and its error from the scatter of the group means (1.25 times their robust sigma over the root of their
number).  The rest of a crowded target is not a fair reference (the wings of its own stars, the persistence of other
earlier exposures), so the same group statistic is also measured with the source's groups moved to unrelated places
in the target (shifted by each of NULL_SHIFTS, leaving out the source's own classed pixels), and the mean of those is
the null, "null_groups"; "excess_groups - null_groups" is the persistence of the class.

Each detector runs in a Python process of its own (this script with --one) and writes
<output_dir>/persistence_<detector>.ecsv, one row per (source, target, class), and its progress to
<output_dir>/persistence_<detector>.log.  When all are done a summary per detector, class and delay is printed and
written to <output_dir>/persistence_summary.ecsv.
"""
import argparse
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from photutils.background import Background2D, MedianBackground
from scipy import ndimage
from stdatamodels.jwst.datamodels.dqflags import pixel

DO_NOT_USE, SATURATED = pixel["DO_NOT_USE"], pixel["SATURATED"]
# pixels whose signal does not come from the sky: left out of both sources and targets, in sources with the ring of
# pixels around them, which the saturation step marks saturated next to a saturated hot pixel (DO_NOT_USE is left in
# for sources, as the pixels saturated from the first group carry it)
NOT_SKY = (
    pixel["NON_SCIENCE"] | pixel["REFERENCE_PIXEL"] | pixel["DEAD"] | pixel["HOT"] | pixel["WARM"] | pixel["RC"]
    | pixel["TELEGRAPH"] | pixel["NO_SAT_CHECK"] | pixel["UNRELIABLE_DARK"] | pixel["UNRELIABLE_SLOPE"]
    | pixel["OTHER_BAD_PIXEL"]
)
PIXEL_SCALE = {"short": 0.0311, "long": 0.0630}  # arcsec per pixel
# signal collected in an integration [DN]: class edges for pixels that did not saturate, then for those that did
UNSATURATED_EDGES = [2e3, 5e3, 1e4, 2e4, 3e4, 4e4, 6e4]
SATURATED_EDGES = [0, 6e4, 1e5, 3e5, 1e6, np.inf]
CLASSES = (
    [f"unsat {lo / 1e3:g}k-{hi / 1e3:g}k" for lo, hi in zip(UNSATURATED_EDGES[:-1], UNSATURATED_EDGES[1:])]
    + [f"sat {lo / 1e3:g}k-{hi / 1e3:g}k" for lo, hi in zip(SATURATED_EDGES[:-1], SATURATED_EDGES[1:])]
    + ["sat no rate"]
)
BRIGHT_DN = 2e4  # pixels collecting more than this count as bright: masked in backgrounds, left out of later sources
MIN_PIXELS = 20  # fewest pixels of a class worth a row
NULL_SHIFTS = ((517, 731), (-389, 1103))  # (rows, columns) the groups are moved by for the null measurement
DELAY_EDGES = [0, 150, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400]  # summary bins of delay [s]


def exposure_info(path):
    """Detector, timing and pointing of an exposure from its headers."""
    with fits.open(path) as hdul:
        primary, sci = hdul[0].header, hdul["SCI"].header
    return {
        "path": path,
        "file": os.path.basename(path),
        "detector": primary["DETECTOR"],
        "filter": primary["FILTER"],
        "readpatt": primary["READPATT"],
        "start": primary["EXPSTART"] * 86400.0,
        "end": primary["EXPEND"] * 86400.0,
        "t_int": primary["NGROUPS"] * primary["TGROUP"],
        "ra": sci["RA_REF"],
        "dec": sci["DEC_REF"],
        "roll": sci["ROLL_REF"],
    }


def shift_pixels(a, b, scale):
    """Pointing shift between two exposures in pixels, from the separation of their reference points."""
    ra1, dec1, ra2, dec2 = map(np.radians, (a["ra"], a["dec"], b["ra"], b["dec"]))
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    separation = np.degrees(np.arccos(np.clip(cos_sep, -1, 1))) * 3600
    # a roll about the reference point moves the far side of the detector by about 1000 pixels times the angle
    roll = abs((a["roll"] - b["roll"] + 180) % 360 - 180)
    return separation / scale + np.radians(roll) * 1000


def classify(sci, dq, t_int):
    """Class number of each pixel (0 for none, k for CLASSES[k - 1]) and the bright-pixel mask."""
    signal = sci * t_int
    with np.errstate(invalid="ignore"):
        saturated = ((dq & SATURATED) > 0) | (np.nan_to_num(signal) > UNSATURATED_EDGES[-1])
    usable = ~ndimage.binary_dilation((dq & NOT_SKY) > 0, structure=np.ones((3, 3), dtype=bool))
    classes = np.zeros(sci.shape, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        unsat = np.digitize(signal, UNSATURATED_EDGES)  # 0 below the first edge, len(edges) above the last
        unsat_ok = ~saturated & usable & (unsat > 0) & (unsat < len(UNSATURATED_EDGES))
        classes[unsat_ok] = unsat[unsat_ok]
        offset = len(UNSATURATED_EDGES) - 1
        sat = np.digitize(np.where(np.isfinite(signal), signal, -1), SATURATED_EDGES)
        sat_ok = saturated & usable & np.isfinite(signal) & (sat > 0)
        classes[sat_ok] = offset + sat[sat_ok]
        classes[saturated & usable & ~np.isfinite(signal)] = len(CLASSES)
        bright = saturated | (np.nan_to_num(signal) > BRIGHT_DN)
    return classes, bright


def group_excess(resid, selected, groups):
    """Median over groups of the mean of ``resid`` over each group's selected pixels, the number of groups, its error."""
    _, inverse = np.unique(groups[selected], return_inverse=True)
    means = np.bincount(inverse, weights=resid[selected]) / np.bincount(inverse)
    return np.median(means), len(means), 1.25 * robust_sigma(means) / np.sqrt(len(means))


def class_groups(classes):
    """Label of the connected group (8-connected, within one class) of each classed pixel; 0 elsewhere."""
    labels = np.zeros(classes.shape, dtype=np.int32)
    offset = 0
    for k in range(1, len(CLASSES) + 1):
        group, n = ndimage.label(classes == k, structure=np.ones((3, 3), dtype=bool))
        labels[group > 0] = group[group > 0] + offset
        offset += n
    return labels


def robust_sigma(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def measure_detector(paths, max_delay, log):
    """All (source, target, class) rows of one detector's exposures."""
    infos = sorted((exposure_info(p) for p in paths), key=lambda info: info["start"])
    detector = infos[0]["detector"]
    scale = PIXEL_SCALE["long" if detector.endswith("LONG") else "short"]
    print(f"{detector}: {len(infos)} exposures", file=log, flush=True)

    rows = []
    # (info, classes, bright, groups, moved) of the exposures ending less than max_delay before the current start,
    # moved holding (classes, groups) shifted by each of NULL_SHIFTS
    recent = deque()
    for n, target in enumerate(infos):
        started = time.time()
        while recent and target["start"] - recent[0][0]["end"] > max_delay:
            recent.popleft()
        with fits.open(target["path"]) as hdul:
            sci = hdul["SCI"].data.astype(np.float32)
            dq = hdul["DQ"].data
        classes, bright = classify(sci, dq, target["t_int"])
        bad = ((dq & (DO_NOT_USE | NOT_SKY)) > 0) | ~np.isfinite(sci)

        # background of the target with its own bright pixels and those of the recent exposures masked
        mask = bad | ndimage.binary_dilation(bright, iterations=4)
        for _, _, earlier_bright, _, _ in recent:
            mask |= earlier_bright
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # a box may lose up to half its pixels to the mask, which crowded fields need
                background = Background2D(
                    np.where(mask, 0, sci), 64, filter_size=3, mask=mask, bkg_estimator=MedianBackground(),
                    exclude_percentile=50.0,
                ).background
            except ValueError:
                # too few unmasked pixels in every box: one level for the whole frame
                background = np.full(sci.shape, np.median(sci[~mask]), dtype=np.float32)
                print(f"{target['file']}: background from the median of the unmasked pixels", file=log, flush=True)
        resid = sci - background
        clean = ~mask
        sample = resid[clean][:: max(1, clean.sum() // 200000)]
        noise = robust_sigma(sample)
        _, control, _ = sigma_clipped_stats(sample)

        # the pixels bright in an exposure after the source, up to and including the target
        later_bright = bright.copy()
        for source, source_classes, source_bright, source_groups, source_moved in reversed(recent):  # latest first
            delay = target["start"] - source["end"]
            if delay > 0:
                usable = ~bad & ~later_bright
                counts = np.bincount(source_classes[usable], minlength=len(CLASSES) + 1)
                rest = np.median(resid[usable & (source_classes == 0)])
                for k in np.nonzero(counts[1:] >= MIN_PIXELS)[0] + 1:
                    selected = usable & (source_classes == k)
                    values = resid[selected]
                    median_groups, n_groups, err_groups = group_excess(resid, selected, source_groups)
                    nulls = []
                    for moved_classes, moved_groups in source_moved:
                        moved = (moved_classes == k) & usable & (source_classes == 0)
                        if moved.sum() >= MIN_PIXELS:
                            nulls.append(group_excess(resid, moved, moved_groups)[0])
                    null = np.mean(nulls) - rest if nulls else np.nan
                    rows.append({
                        "detector": detector, "source": source["file"], "target": target["file"],
                        "filter_source": source["filter"], "filter_target": target["filter"],
                        "readpatt_source": source["readpatt"], "delay_start": delay,
                        "delay_starts": target["start"] - source["start"],
                        "delay_mid": (target["start"] + target["end"]) / 2 - source["end"],
                        "duration_target": target["end"] - target["start"],
                        "shift_pix": shift_pixels(source, target, scale), "cls": CLASSES[k - 1],
                        "npix": len(values), "median": np.median(values), "median_rest": rest,
                        "excess": np.median(values) - rest, "sigma": robust_sigma(values),
                        "ngroups": n_groups, "excess_groups": median_groups - rest, "excess_groups_err": err_groups,
                        "null_groups": null,
                        "noise_target": noise, "control_target": control,
                    })
            later_bright |= source_bright
        groups = class_groups(classes)
        moved = [(np.roll(classes, shift, axis=(0, 1)), np.roll(groups, shift, axis=(0, 1))) for shift in NULL_SHIFTS]
        recent.append((target, classes, bright, groups, moved))
        print(
            f"{n + 1}/{len(infos)} {target['file']}: {int((classes > len(UNSATURATED_EDGES) - 1).sum())} saturated, "
            f"{len(recent) - 1} earlier exposures, noise {noise:.4f} DN/s, {time.time() - started:.1f} s",
            file=log, flush=True,
        )
    return rows


def summary(table, min_shift):
    """Per detector, class and delay bin: the persistence over the pairs that moved, with its error.

    The persistence of a pair is ``excess_groups - null_groups``; the pairs of a bin are combined with weights of one
    over ``excess_groups_err`` squared.
    """
    table = table[(table["shift_pix"] > min_shift) & np.isfinite(table["null_groups"]) & (table["excess_groups_err"] > 0)]
    rows = []
    groups = defaultdict(list)
    for i, row in enumerate(table):
        k = np.searchsorted(DELAY_EDGES, row["delay_starts"]) - 1
        groups[(row["detector"], row["cls"], k)].append(i)
    for (detector, cls, k), index in sorted(groups.items()):
        part = table[index]
        weight = 1.0 / np.asarray(part["excess_groups_err"], dtype=float) ** 2
        net = np.asarray(part["excess_groups"] - part["null_groups"], dtype=float)
        rows.append({
            "detector": detector, "cls": cls, "delay_lo": DELAY_EDGES[k], "delay_hi": DELAY_EDGES[k + 1],
            "pairs": len(part), "ngroups": int(part["ngroups"].sum()),
            "persistence": float(np.sum(weight * net) / np.sum(weight)), "error": float(1.0 / np.sqrt(np.sum(weight))),
            "noise": float(np.median(part["noise_target"])),
        })
    return Table(rows=rows) if rows else Table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", default=".", help="directory holding the _rate files")
    parser.add_argument("--output_dir", default="persistence_measurement")
    parser.add_argument("--pattern", default="jw*_rate.fits", help="glob matched within input_dir")
    parser.add_argument("--detectors", nargs="*", help="detectors to measure, e.g. nrca1 nrcalong (default: all)")
    parser.add_argument("--max_delay", type=float, default=20000.0, help="latest target start after a source ends [s]")
    parser.add_argument(
        "--min_shift", type=float, default=100.0, help="pointing shift a pair needs for the summary [pix]"
    )
    parser.add_argument("--processes", type=int, default=10, help="detectors run at once")
    parser.add_argument("--one", help=argparse.SUPPRESS)  # the detector of a process started from the main one
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in sorted(glob(os.path.join(args.input_dir, args.pattern))):
        groups[os.path.basename(path).split("_")[3]].append(path)

    if args.one:
        with open(os.path.join(args.output_dir, f"persistence_{args.one}.log"), "w") as log:
            rows = measure_detector(groups[args.one], args.max_delay, log)
        Table(rows=rows).write(os.path.join(args.output_dir, f"persistence_{args.one}.ecsv"), overwrite=True)
        sys.exit(0)

    detectors = [d.lower() for d in args.detectors] if args.detectors else sorted(groups)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(detectors)} detectors: " + ", ".join(f"{d} {len(groups[d])}" for d in detectors), flush=True)

    def run_in_process(detector):
        command = [
            sys.executable, os.path.abspath(__file__), "--one", detector, "--input_dir", args.input_dir,
            "--output_dir", args.output_dir, "--pattern", args.pattern, "--max_delay", str(args.max_delay),
        ]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        return detector, proc.returncode, proc.stdout

    with ThreadPoolExecutor(max_workers=args.processes) as pool:
        futures = [pool.submit(run_in_process, detector) for detector in detectors]
        for future in as_completed(futures):
            detector, returncode, console = future.result()
            print(f"{'done' if returncode == 0 else f'FAILED ({returncode})'}: {detector}", flush=True)
            if returncode != 0:
                print(console[-3000:], flush=True)

    paths = [os.path.join(args.output_dir, f"persistence_{detector}.ecsv") for detector in detectors]
    tables = [table for table in (Table.read(path) for path in paths if os.path.exists(path)) if len(table)]
    result = summary(vstack(tables), args.min_shift)
    result.write(os.path.join(args.output_dir, "persistence_summary.ecsv"), overwrite=True)
    for column in ("persistence", "error", "noise"):
        result[column].format = ".4f"
    result.pprint_all()
