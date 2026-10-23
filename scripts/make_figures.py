"""Figures for docs/flicker_and_reset.rst from the saved experiment results.

Reads the npz and JSON files under ``results/`` written by the experiment
scripts and writes PNG figures to ``docs/figures/``.  Paths of the runs used
are set at the top so the figures can be regenerated after a rerun.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stdatamodels.jwst import datamodels  # noqa: E402

from snowblind.metrics import robust_sigma  # noqa: E402
from snowblind.noise_spectrum import log_binned_spectrum  # noqa: E402

DARKS = "results/2026-10-22_darks_part_a/NRCA1"
INJECT = "results/2026-10-20_inject_part_a_v3/NRCA1"
INJECT_MASKS = "results/2026-10-21_inject_chisel_v3/NRCA1"
FIELD = "results/2026-10-20_reset_field_final3/NRCA1"
OUT = "docs/figures"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    }
)


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=color, va="center", fontsize=9)


def channel_spectra():
    sirs = np.load(os.path.join(DARKS, "sirs.npz"))
    clean = np.load(os.path.join(DARKS, "sirs_wiener_box32.npz"))
    freq = sirs["freq"]
    pixel_time_ms = 0.01  # one pixel is read every 10 microseconds
    floor = 0.1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = {"SIRS only": (100.0, 1.2e4, 0), "SIRS + Wiener clean": (30.0, 0.3, 0)}
    for data, label, color in ((sirs, "SIRS only", SERIES[0]), (clean, "SIRS + Wiener clean", SERIES[1])):
        bin_freq, bin_power = log_binned_spectrum(freq, data["spectra"][0], nbins=120)
        period = pixel_time_ms / bin_freq
        ax.plot(period, np.maximum(bin_power, floor), color=color, label=label)
        x, y, _ = labels[label]
        ax.text(x, y, label, color=color, fontsize=9, ha="center")
    white = float(np.median(sirs["spectra"][0][freq > 0.25]))
    ax.axhline(white, color=MUTED, linewidth=1, linestyle="--")
    ax.text(200.0, white * 1.3, f"read-noise level {white:.0f}", color=MUTED, fontsize=9)
    ax.axvline(5.24, color=MUTED, linewidth=1, linestyle=":")
    ax.text(5.24 * 1.15, 3e4, "one row (5.24 ms)", color=MUTED, fontsize=9)
    ax.axvline(3.0, color=MUTED, linewidth=1, linestyle="--")
    ax.text(3.0 / 1.15, 3e4, "3 ms cutoff", color=MUTED, fontsize=9, ha="right")
    ax.text(1.0e3, floor * 1.4, f"below {floor:g}", color=MUTED, fontsize=8, ha="right")
    ax.set_ylim(floor, 2e5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("period (ms)")
    ax.set_ylabel("noise power per bin (DN$^2$)")
    ax.set_title("Noise power along the readout stream, NRCA1 dark, channel 0", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "channel_spectra.png"), dpi=150)


def aperture_scatter():
    summary = {v["variant"]: v for v in json.load(open(os.path.join(DARKS, "summary.json")))["variants"]}
    picks = [
        ("sirs", summary, "SIRS only", SERIES[0]),
        ("sirs_jwst_median", summary, "SIRS + jwst median", SERIES[1]),
        ("sirs_wiener_box32", summary, "SIRS + Wiener, box 32 (default)", SERIES[2]),
        ("sirs_wiener_box256", summary, "SIRS + Wiener, box 256", SERIES[3]),
        ("sirs_wiener_none", summary, "SIRS + Wiener, no background", SERIES[4]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = np.array(summary["sirs"]["aperture_sizes"])
    ax.plot(sizes, summary["sirs"]["aperture_white_prediction"], color=MUTED, linestyle="--", linewidth=1, label="white noise")
    offsets = {"sirs_jwst_median": 9, "sirs": -2, "sirs_wiener_box32": -12, "sirs_wiener_box256": 8, "sirs_wiener_none": -8}
    for key, table, label, color in picks:
        values = np.array(table[key]["aperture_scatter"])
        ax.plot(sizes, values, color=color, marker="o", markersize=5, label=label)
        end_label(ax, sizes[-1], values[-1], label, color, dy=offsets.get(key, 0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("aperture side (pixels)")
    ax.set_ylabel("scatter of aperture mean (DN/s)")
    ax.set_title("Aperture scatter versus size, NRCA1 dark, 20 groups", loc="left")
    ax.set_xlim(0.8, 400)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "aperture_scatter.png"), dpi=150)


def flux_bias():
    old = {v["variant"]: v for v in json.load(open(os.path.join(INJECT, "summary.json")))["variants"]}
    masks = {v["variant"]: v for v in json.load(open(os.path.join(INJECT_MASKS, "summary.json")))["variants"]}
    del old  # the full-gain Wiener (12 percent loss at the faintest bin) is quoted in the text, not plotted
    picks = [
        ("sirs_wiener_box32_clipmask", masks, "Wiener, 3 ms cutoff, clip mask", SERIES[1]),
        ("sirs_row_median_box32", masks, "row median, chisel mask", SERIES[0]),
        ("sirs_wiener_box32", masks, "Wiener, 3 ms cutoff, chisel mask", SERIES[2]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, table, label, color in picks:
        rows = table[key]["by_flux"]
        flux = np.array([np.sqrt(r[0] * r[1]) for r in rows])
        bias = 100 * np.array([r[3] for r in rows])
        ax.plot(flux, bias, color=color, marker="o", markersize=5, label=label)
        dy = {"row median, chisel mask": 10, "Wiener, 3 ms cutoff, chisel mask": 0, "Wiener, 3 ms cutoff, clip mask": -10}[label]
        end_label(ax, flux[-1], bias[-1], label, color, dy=dy)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("injected source flux (DN/s)")
    ax.set_ylabel("relative flux bias (percent)")
    ax.set_title("Photometric bias of cleaning on injected sources, NRCA1 dark", loc="left")
    ax.set_xlim(2, 20000)
    ax.set_ylim(-1.4, 0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flux_bias.png"), dpi=150)


def first_read():
    summary = json.load(open(os.path.join(FIELD, "summary.json")))
    rows = summary["t1_calibration"]["by_first_difference"]
    fit = summary["t1_calibration"]["two_parameter_fit"]
    t1 = summary["t1_calibration"]["t1_nominal_seconds"]
    diff = np.array([np.sqrt(r["first_diff_dn"][0] * r["first_diff_dn"][1]) for r in rows])
    ratio = np.array([r["median_ratio"] for r in rows])
    err = np.array([r["ratio_standard_error"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(diff, ratio, yerr=err, color=SERIES[0], marker="o", markersize=6, linestyle="none", label="measured, 6154 NRCA1")
    grid = np.geomspace(40, 2500, 100)
    a_t1 = grid / 2.0  # BRIGHT1: the group time is twice the first interval
    model = fit["k_t1_scale"] + fit["b_reset_offset_dn"] / a_t1
    ax.plot(grid, model, color=SERIES[1], label=f"R1 = {fit['b_reset_offset_dn']:.1f} DN + {fit['k_t1_scale']:.3f} a t1")
    ax.axhline(1.0, color=GRID, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("first group difference (DN)")
    ax.set_ylabel("d$_0$ / a$_{rest}$")
    ax.set_title(f"First read against the slope, t1 = {t1:.2f} s", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "first_read.png"), dpi=150)


NGROUPS_ROOT = "results/2026-10-21_darks_vs_ngroups"
RATE_PANELS = "results/2026-10-21_rate_images_v5/NRCA1"
# the same panels on a NIRISS dark: (panel directory, label for the titles, suffix of the figure files)
# jump step with min_jump_area 8 and expand_factor 2.5 (NRCA1) or 3 (NIRISS); plan section D, 2026-10-23
DARK_PANEL_SETS = [
    ("results/2026-10-23_rate_images_nrca1/NRCA1", "NRCA1 dark", ""),
    ("results/2026-10-23_rate_images_nis_f3/NIS", "NIRISS dark", "_nis"),
]


def noise_vs_groups():
    """Counterpart of Figure 4 of the JWST 1/f known-issues page: rate noise against number of groups."""
    import glob

    runs = []
    for path in sorted(glob.glob(os.path.join(NGROUPS_ROOT, "n*", "NRCA1", "summary.json"))):
        record = json.load(open(path))
        runs.append((record["ngroups_used"], {v["variant"]: v for v in record["variants"]}))
    if not runs:
        return
    runs.sort()
    ngroups = np.array([n for n, _ in runs])
    picks = [("sirs", "SIRS only", SERIES[0]), ("sirs_jwst_median", "SIRS + jwst median", SERIES[1]),
             ("sirs_wiener_box32", "SIRS + stream clean", SERIES[2])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, ylabel in ((axes[0], "row_means_robust_sigma", "row-mean scatter of the rate (DN/s)"),
                            (axes[1], "rate_robust_sigma", "pixel scatter of the rate (DN/s)")):
        for name, label, color in picks:
            values = np.array([np.mean(table[name][key]) if key == "row_means_robust_sigma" else table[name][key]
                               for _, table in runs])
            ax.plot(ngroups, values, color=color, marker="o", markersize=5, label=label)
            if ax is axes[0]:
                end_label(ax, ngroups[-1], values[-1], label, color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of groups")
        ax.set_ylabel(ylabel)
        ax.set_xlim(2, 400)
    axes[0].set_title("Row banding versus ramp length, NRCA1 dark", loc="left")
    axes[1].set_title("Pixel noise versus ramp length", loc="left")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "noise_vs_groups.png"), dpi=150)


def dark_rate_images(panels_dir=RATE_PANELS, label="NRCA1 dark", suffix=""):
    """Counterpart of Figure 1 of the JWST 1/f known-issues page: rate images before and after cleaning."""
    panels = [
        ("default_refpix_ols.npz", "median refpix (pre-SIRS default),\nno cleaning, OLS_C"),
        ("sirs_ols.npz", "SIRS refpix, no cleaning,\nOLS_C"),
        ("sirs_jwst_median_ols.npz", "SIRS + jwst clean_flicker_noise\n(median per channel,\nno background), OLS_C"),
        ("sirs_stream_clean_ols.npz", "SIRS + snowblind stream_clean\n(no background), OLS_C"),
    ]
    if not os.path.exists(os.path.join(panels_dir, panels[-1][0])):
        return
    reference = np.load(os.path.join(panels_dir, "sirs_ols.npz"))
    good = np.isfinite(reference["rate"]) & (reference["dq"] == 0)
    good[:4] = good[-4:] = False
    good[:, :4] = good[:, -4:] = False
    region = (slice(768, 1280), slice(768, 1280))
    level = np.nanmedian(reference["rate"][good])
    width = 2.5 * robust_sigma(reference["rate"][good])
    # 1 x 4 at 14.5 in: each panel is wider than 512 image pixels, so every detector pixel is drawn; the
    # fonts are scaled by the width ratio to the 8 in figures so they read the same at equal page width
    scale = 14.5 / 8.0
    base = plt.rcParams["font.size"]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 5.3), dpi=150, constrained_layout=True)
    stats = good & (np.abs(reference["rate"] - level) < 5 * width / 2.5)
    for ax, (name, title) in zip(axes, panels):
        rate = np.load(os.path.join(panels_dir, name))["rate"]
        # display every pixel, with unusable ones at the median level; hot pixels clip to the top of the scale
        shown = np.where(np.isfinite(rate), rate, level)[region]
        image = ax.imshow(shown, origin="lower", cmap="viridis", vmin=level - width, vmax=level + width, interpolation="nearest")
        cut = np.where(stats, rate, np.nan)[region]
        rows = np.nanmean(cut, axis=1)
        columns = np.nanmean(cut, axis=0)
        ax.set_title(
            f"{title}\nrow-mean scatter {robust_sigma(rows):.4f} DN/s\ncolumn-mean scatter {robust_sigma(columns):.4f} DN/s",
            fontsize=base * scale * 0.6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    colorbar.set_label("DN/s", fontsize=base * scale * 0.6)
    colorbar.ax.tick_params(labelsize=base * scale * 0.6)
    fig.suptitle(
        f"{label}, 20 groups, central 512 x 512 pixels, stretch +- {width:.3f} DN/s\n",
        fontsize=base * scale * 0.85,
    )
    fig.savefig(os.path.join(OUT, f"dark_rate_images{suffix}.png"), dpi=150)


def dark_rate_images_full(panels_dir=RATE_PANELS, label="NRCA1 dark", suffix=""):
    """The same four rate images as `dark_rate_images`, full 2048 x 2048 frame, binned 2 x 2 for display."""
    panels = [
        ("default_refpix_ols.npz", "median refpix (pre-SIRS default), no cleaning, OLS_C"),
        ("sirs_ols.npz", "SIRS refpix, no cleaning, OLS_C"),
        ("sirs_jwst_median_ols.npz", "SIRS + jwst clean_flicker_noise\n(median per channel, no background), OLS_C"),
        ("sirs_stream_clean_ols.npz", "SIRS + snowblind stream_clean\n(no background), OLS_C"),
    ]
    if not os.path.exists(os.path.join(panels_dir, panels[-1][0])):
        return
    reference = np.load(os.path.join(panels_dir, "sirs_ols.npz"))
    good = np.isfinite(reference["rate"]) & (reference["dq"] == 0)
    good[:4] = good[-4:] = False
    good[:, :4] = good[:, -4:] = False
    level = np.nanmedian(reference["rate"][good])
    width = 2.5 * robust_sigma(reference["rate"][good])
    stats = good & (np.abs(reference["rate"] - level) < 5 * width / 2.5)
    binning = 2
    scale = 14.5 / 8.0
    base = plt.rcParams["font.size"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 15.3), dpi=150, constrained_layout=True)
    for ax, (name, title) in zip(axes.ravel(), panels):
        rate = np.load(os.path.join(panels_dir, name))["rate"]
        shown = np.where(np.isfinite(rate), rate, level)
        nrows, ncols = (shown.shape[0] // binning) * binning, (shown.shape[1] // binning) * binning
        binned = shown[:nrows, :ncols].reshape(nrows // binning, binning, ncols // binning, binning).mean(axis=(1, 3))
        image = ax.imshow(binned, origin="lower", cmap="viridis", vmin=level - width, vmax=level + width, interpolation="nearest")
        cut = np.where(stats, rate, np.nan)[4:-4, 4:-4]
        rows = np.nanmean(cut, axis=1)
        columns = np.nanmean(cut, axis=0)
        ax.set_title(
            f"{title}\nrow-mean scatter {robust_sigma(rows):.4f} DN/s, column-mean scatter {robust_sigma(columns):.4f} DN/s",
            fontsize=base * scale * 0.6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
    colorbar.set_label("DN/s", fontsize=base * scale * 0.6)
    colorbar.ax.tick_params(labelsize=base * scale * 0.6)
    fig.suptitle(
        f"{label}, 20 groups, full frame binned {binning} x {binning} for display, stretch +- {width:.3f} DN/s\n",
        fontsize=base * scale * 0.85,
    )
    # 9 MB as PNG; JPEG at quality 88 is 2 MB and shows the same structure
    fig.savefig(
        os.path.join(OUT, f"dark_rate_images_full{suffix}.jpg"), dpi=150, pil_kwargs={"quality": 88, "optimize": True}
    )
    plt.close(fig)


FIELD_PANELS = "results/2026-10-21_field_panels_v4"
# field panel sets: (panel root, detector, field label, file suffix, description of the jwst panel,
# whether to divide every rate by the flat for display and statistics)
FIELD_PANEL_SETS = [
    (FIELD_PANELS, "NRCA1", "Omega Cen", "nrca1", "median, flat applied, as run\nfor this program", False),
    (FIELD_PANELS, "NRCALONG", "Omega Cen", "nrcalong", "median, flat applied, as run\nfor this program", False),
    ("results/2026-10-22_ceers_all/00001", "NRCALONG", "CEERS", "ceers_nrcalong", "median, flat applied", True),
    ("results/2026-10-22_ceers_all/00001", "NRCBLONG", "CEERS", "ceers_nrcblong", "median, flat applied", True),
    ("results/2026-10-23_spike_mask/jw05105014014_05101_00002_nrcb2", "NRCB2", "NEXUS bright star", "nexus_nrcb2",
     "median, flat applied", True),
    ("results/2026-10-23_coarse_mask/jw05105014018_08101_00001_nrca1", "NRCA1", "NEXUS", "nexus_nrca1",
     "median, flat applied", True),
    ("results/2026-10-23_spike_mask/jw09263005003_03201_00001_nrcblong", "NRCBLONG", "NEXUS", "nexus_nrcblong",
     "median, flat applied", True),
]


def _field_panels(root, detector, jwst_label, flat_fielded=False):
    """Rate images, background mask and display parameters for one detector of the field panels."""
    folder = os.path.join(root, detector)
    panels = [
        ("sirs_ols.npz", "SIRS refpix, no cleaning,\nOLS_C (pipeline default)"),
        ("sirs_jwst_usage_ols.npz", f"SIRS + jwst clean_flicker_noise\n({jwst_label}), OLS_C"),
        ("sirs_stream_clean_ols.npz", "SIRS + snowblind stream_clean\n(median background), OLS_C"),
        ("sirs_stream_clean_reset.npz", "SIRS + snowblind stream_clean\n(median background),\nsnowblind reset_rampfit"),
    ]
    if os.path.exists(os.path.join(folder, "sirs_remstriping_ols.npz")):
        # CEERS: the team's remstriping on the SIRS rate takes the place of the reset-fit panel, which equals
        # the OLS_C one there, and goes before the stream clean
        remstriping = ("sirs_remstriping_ols.npz", "SIRS + CEERS remstriping\n(iterated threshold), OLS_C")
        panels = [panels[0], panels[1], remstriping, panels[2]]
    if not os.path.exists(os.path.join(folder, panels[-1][0])):
        return None
    with datamodels.open(os.path.join(folder, "stream_clean_mask.fits")) as mask_model:
        background = mask_model.data.astype(bool)
    with open(os.path.join(folder, "panels.json")) as handle:
        info = json.load(handle)
    flat = 1.0
    if flat_fielded:
        from jwst.flatfield import FlatFieldStep

        with datamodels.open(os.path.join(folder, "stream_clean_mask.fits")) as mask_model:
            flat_path = FlatFieldStep().get_reference_file(mask_model, "flat")
        with datamodels.FlatModel(flat_path) as flat_model:
            flat = np.array(flat_model.data, dtype=np.float32)
            flat[(flat_model.dq & 1) > 0] = np.nan
        info["flat"] = os.path.basename(flat_path)
    reference = np.load(os.path.join(folder, "sirs_ols.npz"))
    ref_rate = reference["rate"] / flat
    good = background & np.isfinite(ref_rate) & (reference["dq"] == 0)
    good[:4] = good[-4:] = False
    good[:, :4] = good[:, -4:] = False
    sky = np.nanmedian(ref_rate[good])
    sigma = robust_sigma(ref_rate[good])
    stats = good & (np.abs(ref_rate - sky) < 5 * sigma)
    return folder, panels, info, sky, 2.5 * sigma, stats, flat


def field_rate_images(root, detector, field, suffix, jwst_label, flat_fielded=False):
    """The dark_rate_images layout on an Omega Cen exposure: central 512 x 512 pixels, 1 x 4."""
    found = _field_panels(root, detector, jwst_label, flat_fielded)
    if found is None:
        return
    folder, panels, info, sky, width, stats, flat = found
    region = (slice(768, 1280), slice(768, 1280))
    scale = 14.5 / 8.0
    base = plt.rcParams["font.size"]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 5.3), dpi=150, constrained_layout=True)
    for ax, (name, title) in zip(axes, panels):
        rate = np.load(os.path.join(folder, name))["rate"] / flat
        shown = np.where(np.isfinite(rate), rate, sky)[region]
        image = ax.imshow(shown, origin="lower", cmap="viridis", vmin=sky - width, vmax=sky + width, interpolation="nearest")
        cut = np.where(stats, rate, np.nan)[region]
        rows = np.nanmean(cut, axis=1)
        columns = np.nanmean(cut, axis=0)
        ax.set_title(
            f"{title}\nrow-mean scatter {robust_sigma(rows):.4f} DN/s\ncolumn-mean scatter {robust_sigma(columns):.4f} DN/s",
            fontsize=base * scale * 0.6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    colorbar.set_label("DN/s", fontsize=base * scale * 0.6)
    colorbar.ax.tick_params(labelsize=base * scale * 0.6)
    fig.suptitle(
        f"{field}, {detector} {info['filter']} {info.get('readpatt', '')}, {info['ngroups']} groups"
        + (", flat-fielded" if flat_fielded else "") + ", 512 x 512 cutout, "
        f"sky {sky:.2f} DN/s, stretch +- {width:.3f} DN/s\n",
        fontsize=base * scale * 0.75,
    )
    fig.savefig(os.path.join(OUT, f"field_rate_images_{suffix}.png"), dpi=150)
    plt.close(fig)


def field_rate_images_full(root, detector, field, suffix, jwst_label, flat_fielded=False):
    """The dark_rate_images_full layout on an Omega Cen exposure: full frame binned 2 x 2, 2 x 2 panels."""
    found = _field_panels(root, detector, jwst_label, flat_fielded)
    if found is None:
        return
    folder, panels, info, sky, width, stats, flat = found
    binning = 2
    scale = 14.5 / 8.0
    base = plt.rcParams["font.size"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 15.3), dpi=150, constrained_layout=True)
    for ax, (name, title) in zip(axes.ravel(), panels):
        rate = np.load(os.path.join(folder, name))["rate"] / flat
        shown = np.where(np.isfinite(rate), rate, sky)
        nrows, ncols = (shown.shape[0] // binning) * binning, (shown.shape[1] // binning) * binning
        binned = shown[:nrows, :ncols].reshape(nrows // binning, binning, ncols // binning, binning).mean(axis=(1, 3))
        image = ax.imshow(binned, origin="lower", cmap="viridis", vmin=sky - width, vmax=sky + width, interpolation="nearest")
        cut = np.where(stats, rate, np.nan)[4:-4, 4:-4]
        rows = np.nanmean(cut, axis=1)
        columns = np.nanmean(cut, axis=0)
        ax.set_title(
            f"{title.replace(chr(10), ' ')}\nrow-mean scatter {robust_sigma(rows):.4f} DN/s, "
            f"column-mean scatter {robust_sigma(columns):.4f} DN/s",
            fontsize=base * scale * 0.6,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
    colorbar.set_label("DN/s", fontsize=base * scale * 0.6)
    colorbar.ax.tick_params(labelsize=base * scale * 0.6)
    fig.suptitle(
        f"{field}, {detector} {info['filter']} {info.get('readpatt', '')}, {info['ngroups']} groups"
        + (", flat-fielded" if flat_fielded else "") + f", full frame binned {binning} x {binning} "
        f"for display, stretch +- {width:.3f} DN/s\n",
        fontsize=base * scale * 0.75,
    )
    fig.savefig(
        os.path.join(OUT, f"field_rate_images_{suffix}_full.jpg"), dpi=150,
        pil_kwargs={"quality": 88, "optimize": True},
    )
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    channel_spectra()
    aperture_scatter()
    flux_bias()
    first_read()
    noise_vs_groups()
    for panels_dir, label, suffix in DARK_PANEL_SETS:
        dark_rate_images(panels_dir, label, suffix)
        dark_rate_images_full(panels_dir, label, suffix)
    for root, detector, field, suffix, jwst_label, flat_fielded in FIELD_PANEL_SETS:
        field_rate_images(root, detector, field, suffix, jwst_label, flat_fielded)
        field_rate_images_full(root, detector, field, suffix, jwst_label, flat_fielded)
    print("figures written to", OUT)
