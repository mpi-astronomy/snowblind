"""Experiment C.3.2: injected point sources and a sky gradient on a dark (plan Part A).

Synthetic sources and a smooth sky are added to the group data of a dark
``_jump`` product made in memory with SIRS refpix, as ``rate * t_group``
for every group.  So the injection is after all Detector1 corrections, not
into the ``_uncal`` file; the injected signal carries no Poisson noise and
no nonlinearity.  Each cleaning variant is then ramp fitted and compared
with the ramp fit of the injected but uncleaned data, so the common noise
realization cancels and what remains is the cleaner's own effect on
photometry (plan metric 4) and on the sky gradient.

Sources: Gaussian with sigma 1.3 pixels, total fluxes log-uniform between
``--fmin`` and ``--fmax`` DN/s.  Half are placed at random; the other half
in ten dense patches so that the local mask density varies.

Output: ``summary.json`` and per-source tables in
``results/<date>_inject_part_a/<detector>/``.
"""
import argparse
import datetime
import json
import os

import numpy as np
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst.datamodels import dqflags

from snowblind import StreamCleanStep, readout
from snowblind.metrics import injected_flux_bias, robust_sigma
from snowblind.reset_calib import run_detector1_to_jump

MAX_LAG = 2**16
DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
SIGMA_PSF = 1.3


def make_catalog(rng, nsources, shape):
    """Positions and fluxes: half random, half in ten dense patches."""
    nrows, ncols = shape
    margin = 24
    n_random = nsources // 2
    x = [rng.uniform(margin, ncols - margin, n_random)]
    y = [rng.uniform(margin, nrows - margin, n_random)]
    per_patch = (nsources - n_random) // 10
    for _ in range(10):
        cx, cy = rng.uniform(150, ncols - 150), rng.uniform(150, nrows - 150)
        x.append(rng.normal(cx, 40.0, per_patch))
        y.append(rng.normal(cy, 40.0, per_patch))
    x = np.clip(np.concatenate(x), margin, ncols - margin - 1)
    y = np.clip(np.concatenate(y), margin, nrows - margin - 1)
    return x, y


def render_scene(shape, x, y, flux):
    """Sky gradient plus Gaussian sources, in DN/s."""
    nrows, ncols = shape
    yy, xx = np.mgrid[0:nrows, 0:ncols]
    sky = 0.3 + 0.2 * xx / ncols + 0.1 * np.sin(2 * np.pi * yy / 400.0)
    scene = sky.astype(np.float64)
    half = 8
    for xi, yi, fi in zip(x, y, flux):
        xc, yc = int(round(xi)), int(round(yi))
        ys, xs = slice(yc - half, yc + half + 1), slice(xc - half, xc + half + 1)
        sub_y, sub_x = yy[ys, xs], xx[ys, xs]
        scene[ys, xs] += fi * np.exp(-((sub_x - xi) ** 2 + (sub_y - yi) ** 2) / (2 * SIGMA_PSF**2)) / (
            2 * np.pi * SIGMA_PSF**2
        )
    return scene, sky


def inject(model, scene):
    """Return a copy of the jump model with scene * t_group added to every group."""
    injected = model.copy()
    ngroups = injected.data.shape[1]
    group_time = injected.meta.exposure.group_time
    nframes = injected.meta.exposure.nframes
    frame_time = injected.meta.exposure.frame_time
    t_first = frame_time * (nframes + 1) / 2.0
    times = t_first + group_time * np.arange(ngroups)
    for j in range(ngroups):
        injected.data[0, j] += (scene * times[j]).astype(np.float32)
    return injected


def gradient_residual(rate, reference, sky, mask, box=64):
    """How much of the sky gradient the cleaning removed: binned medians of (rate - reference) against the sky."""
    diff = np.where(mask & np.isfinite(rate) & np.isfinite(reference), rate - reference, np.nan)
    nrows, ncols = diff.shape
    blocks = (nrows // box, box, ncols // box, box)
    trimmed = (slice(0, nrows // box * box), slice(0, ncols // box * box))
    binned = np.nanmedian(diff[trimmed].reshape(blocks), axis=(1, 3))
    sky_binned = sky[trimmed].reshape(blocks).mean(axis=(1, 3))
    finite = np.isfinite(binned)
    slope = np.polyfit(sky_binned[finite] - sky_binned[finite].mean(), binned[finite], 1)[0]
    return {
        "binned_residual_robust_sigma": float(robust_sigma(binned)),
        "slope_of_residual_vs_sky": float(slope),
        "sky_range": [float(sky.min()), float(sky.max())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--ngroups", type=int, default=20)
    parser.add_argument("--nsources", type=int, default=1500)
    parser.add_argument("--fmin", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--variants", default="all")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sirs = run_detector1_to_jump(args.file, max_groups=args.ngroups, max_ints=1, refpix_algorithm="sirs")
    detector = sirs.meta.instrument.detector
    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_inject_part_a")
    out_dir = os.path.join(out_root, detector)
    os.makedirs(out_dir, exist_ok=True)

    shape = sirs.data.shape[-2:]
    x, y = make_catalog(rng, args.nsources, shape)
    flux = np.exp(rng.uniform(np.log(args.fmin), np.log(args.fmax), x.size))
    scene, sky = render_scene(shape, x, y, flux)
    injected = inject(sirs, scene)
    sirs.close()

    reference, _ = RampFitStep.call(injected, algorithm="OLS_C")
    reference_rate = reference.data.copy()
    reference.close()

    science = np.zeros(shape, dtype=bool)
    science[readout.NREFPIX:-readout.NREFPIX, readout.NREFPIX:-readout.NREFPIX] = True

    clipped = {"background_method": "model", "clip_sigma": 2.0, "iterations": 2, "output_dir": out_dir, "save_mask": True}

    def linear(box):
        return {
            "background_method": "integration_model", "background_box_size": (box, box), "output_dir": out_dir,
            "save_mask": True,
        }

    variants = {
        "sirs_jwst_median": lambda: CleanFlickerNoiseStep.call(
            injected, skip=False, fit_method="median", fit_by_channel=True, background_method="model"
        ),
        "sirs_row_median_clipped": lambda: StreamCleanStep.call(injected, fit_method="row_median", **clipped),
        "sirs_wiener_clipped": lambda: StreamCleanStep.call(injected, fit_method="wiener", max_lag=MAX_LAG, **clipped),
        "sirs_row_median_box32": lambda: StreamCleanStep.call(injected, fit_method="row_median", **linear(32)),
        "sirs_wiener_box32": lambda: StreamCleanStep.call(injected, fit_method="wiener", max_lag=MAX_LAG, **linear(32)),
        "sirs_wiener_box128": lambda: StreamCleanStep.call(injected, fit_method="wiener", max_lag=MAX_LAG, **linear(128)),
        "sirs_wiener_box256": lambda: StreamCleanStep.call(injected, fit_method="wiener", max_lag=MAX_LAG, **linear(256)),
        "sirs_wiener_box32_nocolumns": lambda: StreamCleanStep.call(
            injected, fit_method="wiener", max_lag=MAX_LAG, column_offsets=False, **linear(32)
        ),
        "sirs_wiener_box32_minp100": lambda: StreamCleanStep.call(
            injected, fit_method="wiener", max_lag=MAX_LAG, min_period=100, **linear(32)
        ),
        "sirs_wiener_box32_minp300": lambda: StreamCleanStep.call(
            injected, fit_method="wiener", max_lag=MAX_LAG, min_period=300, **linear(32)
        ),
        "sirs_wiener_box32_nsig1": lambda: StreamCleanStep.call(
            injected, fit_method="wiener", max_lag=MAX_LAG, n_sigma=1.0, **linear(32)
        ),
        "sirs_wiener_box32_dilate3": lambda: StreamCleanStep.call(
            injected, fit_method="wiener", max_lag=MAX_LAG, mask_dilation=3, **linear(32)
        ),
        "sirs_wiener_box32_clipmask": lambda: StreamCleanStep.call(
            injected, fit_method="wiener", max_lag=MAX_LAG, scene_mask="clip", **linear(32)
        ),
        "sirs_row_median_box32_clipmask": lambda: StreamCleanStep.call(
            injected, fit_method="row_median", scene_mask="clip", **linear(32)
        ),
    }
    wanted = None if args.variants == "all" else set(args.variants.split(","))

    summaries = []
    for name, make in variants.items():
        if wanted is not None and name not in wanted:
            continue
        print(f"=== {name}", flush=True)
        before = set(os.listdir(out_dir))
        model = make()
        rate, _ = RampFitStep.call(model, algorithm="OLS_C")
        new_masks = sorted(f for f in set(os.listdir(out_dir)) - before if f.endswith("_mask.fits"))
        if new_masks:
            from stdatamodels.jwst import datamodels

            kept = os.path.join(out_dir, f"{name}_mask.fits")
            os.replace(os.path.join(out_dir, new_masks[-1]), kept)
            with datamodels.open(kept) as mask_model:
                mask = mask_model.data.astype(bool)
        else:
            # the jwst step keeps its mask internal; use all pixels
            mask = np.ones(shape, dtype=bool)
        bias = injected_flux_bias(rate.data, reference_rate, x, y, flux, mask)
        gradient = gradient_residual(rate.data, reference_rate, sky, science & mask)
        summary = {
            "variant": name,
            "n_measured": bias["n_measured"],
            "median_relative_bias": bias["median_bias"],
            "relative_bias_robust_sigma": float(robust_sigma(bias["relative_bias"])),
            "by_density": bias["by_density"],
            "by_distance": bias["by_distance"],
            "by_flux": bias["by_flux"],
            "gradient": gradient,
            "rate_change_robust_sigma": float(robust_sigma((rate.data - reference_rate)[science & mask])),
        }
        np.savez_compressed(
            os.path.join(out_dir, f"{name}_sources.npz"),
            x=x, y=y, flux=flux, relative_bias=bias["relative_bias"], density=bias["density"],
            neighbor_distance=bias["neighbor_distance"], rate=rate.data.astype(np.float32),
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=1), flush=True)
        model.close()
        rate.close()

    record = {
        "file": os.path.basename(args.file),
        "detector": detector,
        "ngroups_used": args.ngroups,
        "nsources": int(x.size),
        "flux_range_dn_per_s": [args.fmin, args.fmax],
        "injection": "scene * t_group added to the jump product groups; no Poisson noise, no nonlinearity",
        "variants": summaries,
    }
    np.savez_compressed(
        os.path.join(out_dir, "reference.npz"), reference_rate=reference_rate.astype(np.float32), sky=sky.astype(np.float32)
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
