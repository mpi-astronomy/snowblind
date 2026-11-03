"""Inject cosmic rays into cal files and compare outlier detection: jwst's step against per-exposure references.

Stages, each run on its own and saving its results in ``--out``:

``inject``
    Copy the cal files with single-pixel cosmic rays added: ``--per-exposure``
    in each, of 10 to 1000 times the local noise (uniform in the logarithm),
    up to half of them 1 to 3 pixels from one of the brightest sources
    (local maxima more than ``--source-snr`` sigma above the sky, at most
    300 per exposure), as many as fit at least 2 pixels apart, and the rest
    anywhere.  The positions, amplitudes and sources go in
    ``injected.npz``.
``jwst``
    jwst's OutlierDetectionStep (imaging, its defaults, the models on disk)
    on an association of the copies; flags in ``flags_jwst.npz``.
``reference``
    `snowblind.outlier_reference`: one reference from the input pixels of
    all the copies on jwst's outlier mosaic grid sampled twice as finely,
    in tiles, blotted onto each exposure and compared by jwst's rule with
    each tolerance of ``--tolerances`` (``scale`` of the two passes, or
    ``robust`` for the measured spread); flags in
    ``flags_<variant>_<tolerance>.npz``, and the exposures covering each
    pixel in ``coverage.npz``.
``report``
    For each flags file: the injected cosmic rays flagged, overall, near
    sources and by the number of exposures covering the pixel; and the
    pixels flagged that were not injected (more than a pixel from any
    injected one), at source cores (within 1 pixel of a source), in their
    wings (1 to 8 pixels) and elsewhere.  Those include the real outliers
    of the data, the same for every method.

Usage::

    python scripts/exp_injected_outliers.py inject --out DIR DATA/jw*_nrc?long_cal.fits
    python scripts/exp_injected_outliers.py jwst --out DIR
    python scripts/exp_injected_outliers.py reference --out DIR --pixfrac 0.8
    python scripts/exp_injected_outliers.py report --out DIR

Needs ``pip install 'snowblind[spizzle]'``.  The jwst step's CRDS parameters
are not read.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clipped_stats
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.datamodels import ModelLibrary
from jwst.outlier_detection import OutlierDetectionStep
from jwst.resample.resample_utils import resampled_wcs_from_models
from scipy import ndimage
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags

from snowblind import outlier_reference

OUTLIER = dqflags.pixel["OUTLIER"]
DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
SOURCE_SNR = 50.0
MAX_SOURCES = 300


def good_pixels(model):
    return np.isfinite(model.data) & np.isfinite(model.err) & (model.err > 0) & ((model.dq & DO_NOT_USE) == 0)


def bright_sources(model, good, source_snr=SOURCE_SNR):
    """The (y, x) of the brightest local maxima more than ``source_snr`` sigma above the sky, at most MAX_SOURCES."""
    sky = sigma_clipped_stats(model.data[good], sigma=3.0, maxiters=5)[1]
    snr = np.where(good, (model.data - sky) / model.err, 0.0)
    peaks = (snr > source_snr) & (snr == ndimage.maximum_filter(snr, size=3))
    ys, xs = np.nonzero(peaks)
    order = np.argsort(snr[ys, xs])[::-1][:MAX_SOURCES]
    return np.column_stack([ys[order], xs[order]])


def inject(args):
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for name in args.files:
        with datamodels.open(name) as model:
            good = good_pixels(model)
            sources = bright_sources(model, good, args.source_snr)
            taken = np.zeros_like(good)
            sites = []
            # every position 1 to 3 pixels from a source, in random order, while they fit 2 pixels apart
            offsets = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if max(abs(dy), abs(dx)) >= 1]
            candidates = [(y + dy, x + dx) for y, x in sources for dy, dx in offsets]
            for k in rng.permutation(len(candidates)):
                if len(sites) >= args.per_exposure // 2:
                    break
                y, x = candidates[k]
                if 0 <= y < good.shape[0] and 0 <= x < good.shape[1] and good[y, x] and not taken[y, x]:
                    sites.append((y, x, True))
                    taken[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = True
            while len(sites) < args.per_exposure:
                y, x = rng.integers(0, good.shape[0]), rng.integers(0, good.shape[1])
                if good[y, x] and not taken[y, x]:
                    sites.append((y, x, False))
                    taken[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = True
            ys, xs, near = (np.array(v) for v in zip(*sites))
            amp = 10 ** rng.uniform(1.0, 3.0, len(ys))
            model.data[ys, xs] += amp * model.err[ys, xs]
            out = args.out / Path(name).name
            model.save(out)
            catalog[out.name] = {"y": ys, "x": xs, "near": near, "amp": amp, "sources": sources}
            print(f"{out.name}: {len(ys)} injected ({near.sum()} near {len(sources)} sources)", flush=True)
    np.savez(args.out / "injected.npz", catalog=np.array(catalog, dtype=object))
    names = sorted(catalog)
    _, text = asn_from_list.asn_from_list(names, rule=DMS_Level3_Base, product_name="injected").dump(format="json")
    (args.out / "injected_asn.json").write_text(text)


def load_catalog(out):
    return np.load(out / "injected.npz", allow_pickle=True)["catalog"].item()


def save_flags(out, label, flags, seconds):
    np.savez_compressed(out / f"flags_{label}.npz", seconds=seconds,
                        **{name: np.packbits(f) for name, f in flags.items()})
    print(f"{label}: {sum(int(f.sum()) for f in flags.values())} pixels flagged in {seconds:.0f} s", flush=True)


def run_jwst(args):
    os.environ["STPIPE_DISABLE_CRDS_STEPPARS"] = "True"
    cwd = os.getcwd()
    os.chdir(args.out)
    try:
        step = OutlierDetectionStep(in_memory=False)
        step.mode = "imaging"
        t0 = time.perf_counter()
        result = step.run(ModelLibrary("injected_asn.json", on_disk=True))
        seconds = time.perf_counter() - t0
        flags = {}
        with result:
            for model in result:
                flags[model.meta.filename] = (model.dq & OUTLIER) > 0
                result.shelve(model, modify=False)
    finally:
        os.chdir(cwd)
    save_flags(args.out, "jwst", flags, seconds)


def run_reference(args):
    names = sorted(load_catalog(args.out))
    models = [datamodels.open(args.out / name) for name in names]
    wcs = resampled_wcs_from_models(ModelLibrary(models, on_disk=False), pixel_scale_ratio=1.0)[0]
    variant = f"reference_pixfrac{args.pixfrac:g}" + ("" if args.correct else "_nocorrect")
    t0 = time.perf_counter()
    exposures = outlier_reference.predict(models, wcs, pixfrac=args.pixfrac, correct=args.correct,
                                          spread="robust" in args.tolerances)
    seconds = time.perf_counter() - t0
    for tol in args.tolerances:
        scale = None if tol == "robust" else tuple(float(v) for v in tol.split("/"))
        t1 = time.perf_counter()
        flags = {name: outlier_reference.flag_outliers(e, scale=scale) for name, e in zip(names, exposures)}
        save_flags(args.out, f"{variant}_{tol.replace('/', '-')}", flags, seconds + time.perf_counter() - t1)
    np.savez_compressed(args.out / "coverage.npz", **{name: e.n_exposures for name, e in zip(names, exposures)})


def report(args):
    catalog = load_catalog(args.out)
    coverage = dict(np.load(args.out / "coverage.npz")) if (args.out / "coverage.npz").exists() else None
    rows = []
    for path in sorted(args.out.glob("flags_*.npz")):
        stored = np.load(path)
        found = near_found = n = n_near = 0
        by_cover = {}
        extra = {"cores": [0, 0], "wings": [0, 0], "elsewhere": [0, 0]}
        for name, entry in catalog.items():
            with datamodels.open(args.out / name) as model:
                shape = model.data.shape
                good = good_pixels(model)
            flags = np.unpackbits(stored[name])[: shape[0] * shape[1]].reshape(shape).astype(bool)
            ys, xs, near = entry["y"], entry["x"], entry["near"]
            hit = flags[ys, xs]
            found += hit.sum()
            n += len(ys)
            near_found += hit[near].sum()
            n_near += near.sum()
            if coverage is not None:
                cover = np.minimum(coverage[name][ys, xs], 4)
                for c in np.unique(cover):
                    f, t = by_cover.get(int(c), (0, 0))
                    by_cover[int(c)] = (f + hit[cover == c].sum(), t + (cover == c).sum())
            injected = np.zeros(shape, dtype=bool)
            injected[ys, xs] = True
            injected = ndimage.binary_dilation(injected, np.ones((3, 3)))
            source_map = np.zeros(shape, dtype=bool)
            if len(entry["sources"]):
                source_map[entry["sources"][:, 0], entry["sources"][:, 1]] = True
            cores = ndimage.binary_dilation(source_map, np.ones((3, 3)))
            wings = ndimage.binary_dilation(source_map, np.ones((17, 17))) & ~cores
            for key, region in (("cores", cores), ("wings", wings), ("elsewhere", ~cores & ~wings)):
                region = region & good & ~injected
                extra[key][0] += np.sum(flags & region)
                extra[key][1] += region.sum()
        rows.append((path.stem[6:], float(stored["seconds"]), found / n, near_found / max(1, n_near), by_cover, extra))
    covers = sorted({c for row in rows for c in row[4]})
    head = " | ".join(f"found where {c}{'+' if c == 4 else ''} cover" for c in covers)
    print(f"\n{n} cosmic rays injected per method, {n_near} near sources\n")
    print(f"| method | time | found | near sources | {head} | flagged elsewhere: source cores | wings per 10^4 "
          "| elsewhere per 10^4 |")
    print("| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in covers) + " | ---: | ---: | ---: |")
    for label, seconds, found, near, by_cover, extra in rows:
        cells = " | ".join(f"{by_cover[c][0] / max(1, by_cover[c][1]):.1%}" if c in by_cover else "" for c in covers)
        core = extra["cores"][0] / max(1, extra["cores"][1])
        wings = 1e4 * extra["wings"][0] / max(1, extra["wings"][1])
        elsewhere = 1e4 * extra["elsewhere"][0] / max(1, extra["elsewhere"][1])
        print(f"| {label} | {seconds:.0f} s | {found:.1%} | {near:.1%} | {cells} | {core:.2%} | {wings:.1f} | "
              f"{elsewhere:.2f} |")
    if coverage is not None:
        total = sum(c.size for c in coverage.values())
        shares = np.bincount(np.concatenate([np.minimum(c, 4).ravel() for c in coverage.values()]), minlength=5)
        print("\npixels covered by n exposures:", json.dumps({n: f"{s / total:.1%}" for n, s in enumerate(shares)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("stage", choices=["inject", "jwst", "reference", "report"])
    parser.add_argument("files", nargs="*", help="cal files, for inject")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-exposure", type=int, default=2000, help="cosmic rays injected per exposure")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-snr", type=float, default=SOURCE_SNR, help="sigma above the sky of the sources")
    parser.add_argument("--pixfrac", type=float, default=0.8)
    parser.add_argument("--no-correct", dest="correct", action="store_false", help="leave out the correction step")
    parser.add_argument("--tolerances", nargs="+", default=["1.2/0.7", "0.5/0.3", "robust"],
                        help="jwst's scale of the two passes, or robust")
    args = parser.parse_args()
    {"inject": inject, "jwst": run_jwst, "reference": run_reference, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
