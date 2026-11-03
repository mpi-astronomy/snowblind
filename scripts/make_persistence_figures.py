"""Figures for docs/persistence.rst: how a pair of exposures is measured, and a detector's decay curve.

The first figure is a schematic.  The second reads a detector's packaged persistence file
(``src/snowblind/data/persistence_<detector>.asdf``, written by ``package_persistence_defaults.py``).

Usage: ``python scripts/make_persistence_figures.py``
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import asdf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "snowblind", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

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


def outline(ax, mask, **style):
    """Draw the outer edges of the pixels of ``mask`` on an image axis."""
    rows, cols = mask.shape
    for y in range(rows):
        for x in range(cols):
            if not mask[y, x]:
                continue
            for dy, dx, segment in ((-1, 0, ((x - .5, x + .5), (y - .5, y - .5))), (1, 0, ((x - .5, x + .5), (y + .5, y + .5))),
                                    (0, -1, ((x - .5, x - .5), (y - .5, y + .5))), (0, 1, ((x + .5, x + .5), (y - .5, y + .5)))):
                if not (0 <= y + dy < rows and 0 <= x + dx < cols) or not mask[y + dy, x + dx]:
                    ax.plot(*segment, **style)


def timeline(ax, title, source, target, with_superposition, arrow_from, arrow_to, label, also_skipped=()):
    """One row of exposures: the source, the target, the delay, and the exposures whose bright pixels the target skips.

    ``with_superposition`` are those whose persistence would add to the source's in the target, a superposition of
    persistence; ``also_skipped`` those skipped although theirs cannot.
    """
    starts = np.arange(6) * 1.4
    for n, start in enumerate(starts):
        if n == source:
            style = {"facecolor": SERIES[1], "edgecolor": SERIES[1]}
        elif n == target:
            style = {"facecolor": SERIES[0], "edgecolor": SERIES[0]}
        elif n in with_superposition:
            style = {"facecolor": SURFACE, "edgecolor": MUTED, "hatch": "///"}
        elif n in also_skipped:
            style = {"facecolor": SURFACE, "edgecolor": MUTED, "hatch": "..", "linestyle": ":"}
        else:
            style = {"facecolor": GRID, "edgecolor": GRID}
        ax.add_patch(Rectangle((start, 0), 1.0, 1.0, linewidth=1, **style))
    ax.text(starts[source] + 0.5, 1.25, "source", ha="center", color=SERIES[1], fontsize=9)
    ax.text(starts[target] + 0.5, 1.25, "target", ha="center", color=SERIES[0], fontsize=9)
    for n in with_superposition:
        ax.text(starts[n] + 0.5, 1.25, "superposition of\npersistence: skipped", ha="center", va="bottom", color=MUTED,
                fontsize=8)
    for n in also_skipped:
        ax.text(starts[n] + 0.5, 1.25, "skipped too, with no\nsuperposition", ha="center", va="bottom",
                color=MUTED, fontsize=8)
    x0, x1 = arrow_from(starts), arrow_to(starts)
    ax.add_patch(FancyArrowPatch((x0, -0.45), (x1, -0.45), arrowstyle="<->", mutation_scale=12, color=TEXT, linewidth=1.2))
    ax.plot([x0, x0], [-0.6, 0.5], color=TEXT, linewidth=0.8, linestyle=":")
    ax.plot([x1, x1], [-0.6, 0.5], color=TEXT, linewidth=0.8, linestyle=":")
    ax.text((x0 + x1) / 2, -0.95, label, ha="center", fontsize=9)
    ax.set_xlim(-0.3, starts[-1] + 1.3)
    ax.set_ylim(-1.3, 2.1)
    ax.set_title(title, loc="left", fontsize=10)
    ax.axis("off")


def pairs_figure():
    """Schematic: a forward and a backward pair in time, and the pixels measured in one pair."""
    rng = np.random.default_rng(4)
    size = 31
    y, x = np.mgrid[:size, :size]
    star = 4000.0 * np.exp(-((x - 15) ** 2 + (y - 15) ** 2) / (2 * 2.2 ** 2))
    source = np.minimum(star, 1000.0) + rng.normal(0, 8, star.shape)
    saturated = star >= 1000.0
    ghost = np.where(saturated, 25.0, 0.0)
    elsewhere = 300.0 * np.exp(-((x - 6) ** 2 + (y - 24) ** 2) / (2 * 1.5 ** 2))
    target = ghost + elsewhere + rng.normal(0, 8, star.shape)
    moved = 220.0 * np.exp(-((x - 24) ** 2 + (y - 8) ** 2) / (2 * 1.4 ** 2)) + rng.normal(0, 8, star.shape)

    fig = plt.figure(figsize=(10.0, 6.9))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.2, 2.3], hspace=0.45, wspace=0.08)
    timeline(
        fig.add_subplot(grid[0, :]), "Forward: a later target, where the source can have left persistence",
        source=2, target=4, with_superposition=[3], arrow_from=lambda s: s[2] + 1.0, arrow_to=lambda s: s[4] + 0.5,
        label="delay: end of the source to the middle of the target",
    )
    timeline(
        fig.add_subplot(grid[1, :]), "Backward: an earlier target, where it cannot",
        source=4, target=2, with_superposition=[0, 1], arrow_from=lambda s: s[2] + 0.5, arrow_to=lambda s: s[4],
        label="delay: middle of the target to the start of the source", also_skipped=[3],
    )
    heading = fig.add_subplot(grid[2, :])
    heading.axis("off")
    heading.text(0.0, 0.0, "The forward pair, pixel by pixel: the source exposure, then the same target exposure twice",
                 transform=heading.transAxes, fontsize=10, ha="left", va="bottom")
    panels = [
        (source, "source exposure:\na saturated star, its pixels outlined", "-", (-10, 1000), SERIES[1]),
        (target, "target exposure, the same pixels:\nwhat the star left behind", "-", (-30, 60), SERIES[0]),
        (moved, "target exposure, the outline moved\nto other pixels: the comparison", "--", (-30, 60), SERIES[0]),
    ]
    for k, (image, title, linestyle, (low, high), frame) in enumerate(panels):
        ax = fig.add_subplot(grid[3, k])
        ax.imshow(image, origin="lower", cmap="viridis", vmin=low, vmax=high)
        outline(ax, saturated, color=SERIES[1], linewidth=1.5, linestyle=linestyle)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(frame)
            spine.set_linewidth(2.5)
    fig.savefig(os.path.join(OUT, "persistence_pairs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def decay_figure(detector, packaged_dir):
    """A detector's packaged decay curve: the excess at its saturated stars' pixels, the part of it that is not
    persistence, the persistence with its fit, and the times."""
    with asdf.open(os.path.join(packaged_dir, f"persistence_{detector.lower()}.asdf"), lazy_load=False) as handle:
        tree = handle.tree
        measured = {key: np.asarray(value) for key, value in tree["measured"].items()}
        model, time, typical = tree["model"], tree["time"], tree["typical_time"]
        threshold, margin = tree["threshold"], tree["margin"]
    mdn = 1e3  # DN/s to mDN/s

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    delay = measured["delay_median"]
    persistence = measured["persistence"] * mdn
    error = measured["error"] * mdn
    shown = persistence > 0
    if "forward" in measured:
        ax.plot(delay, measured["forward"] * mdn, color=SERIES[0], marker="o", markersize=5, markerfacecolor="none",
                linestyle="none", label="excess at the stars' pixels")
    not_persistence = float(measured.get("backward", 0.0)) * mdn
    if not_persistence > 0:
        ax.axhline(not_persistence, color=SERIES[2], linewidth=1.2, linestyle="--",
                   label="part of the excess that is not persistence")
    ax.errorbar(delay[shown], persistence[shown], yerr=error[shown], color=SERIES[0], marker="o", markersize=5,
                linestyle="none", label="persistence")
    grid = np.geomspace(delay.min(), max(delay.max(), time) * 1.3, 200)
    ax.plot(grid, model["amplitude"] * (grid / 1000.0) ** -model["alpha"] * mdn, color=SERIES[1],
            label=f"fit to the persistence, A (delay / 1000 s)$^{{-{model['alpha']:.2f}}}$")
    for level, name in ((threshold, f"threshold, {threshold * mdn:g} mDN/s"), (threshold / margin, "half the threshold")):
        ax.axhline(level * mdn, color=MUTED, linewidth=1, linestyle=":")
        ax.text(grid[0], level * mdn * 1.08, name, color=MUTED, fontsize=8)
    for value, name, height in ((typical, "typical_time", 1.0), (time, "time", 0.4)):
        ax.axvline(value, color=MUTED, linewidth=1, linestyle="--")
        ax.text(value * 1.05, persistence[shown].max() * height, f"{name}\n{value:.0f} s", color=MUTED, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("delay since the end of the saturated exposure (s)")
    ax.set_ylabel("excess rate (mDN/s)")
    ax.set_title(f"{detector}: persistence of saturated stars", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "persistence_decay.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packaged_dir", default=os.path.normpath(PACKAGE_DIR), help="folder of the packaged files")
    parser.add_argument("--detector", default="NRCALONG", help="detector of the decay figure")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pairs_figure()
    decay_figure(args.detector, args.packaged_dir)
    print("figures written to", os.path.normpath(OUT))
