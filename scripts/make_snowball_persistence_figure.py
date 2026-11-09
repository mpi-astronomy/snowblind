"""Figure: snowball core persistence per detector, from fit_snowball_persistence.py's <detector>_decay.npz files.  Each
panel: the mean excess over cores of the saturated pixels, of the pixels 1 px and 2 px from them, against
the time since the end of the hit's integration; the power-law fit to the saturated pixels; the 5 mDN/s
threshold; and the times packaged with snowblind for the detector, which SnowblindStep uses (the 99th
percentile of the per-core decay times, see package_snowball_persistence.py).  NIRCam module A on the left,
module B on the right, NIRSpec NRS1 and NRS2 below them, NIRISS below NRS1.

usage: make_snowball_persistence_figure.py figure.png decay1.npz [decay2.npz ...]
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from fit_snowball_persistence import THRESHOLD  # noqa: E402
from snowblind.snowblind import packaged_snowball_times  # noqa: E402

COLUMNS = [["NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCALONG", "NRS1", "NIS"],
           ["NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG", "NRS2", None]]
SHOWN = {"saturated": ("saturated pixels", "#2a78d6"), "1px": ("1 px from a saturated pixel", "#eb6834"),
         "2px": ("2 px from a saturated pixel", "#1baf7a")}
SURFACE, TEXT, TEXT_2, GRID, BORDER = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

# figure geometry in inches: panel size, gaps between panels, margins
PANEL_W, PANEL_H, GAP_W, GAP_H = 4.2, 1.35, 0.3, 0.4
LEFT, RIGHT, TOP, BOTTOM = 0.7, 0.12, 0.92, 0.5


def draw_panel(ax, z):
    t_fit = np.geomspace(100, 25000, 200)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.axhline(1000 * THRESHOLD, color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1.3)
    for sel, (label, color) in SHOWN.items():
        stack, mids = z[f"stack_{sel}"], z[f"mids_{sel}"]
        pos = np.isfinite(stack[:, 0]) & (stack[:, 0] > 0)
        ax.errorbar(mids[pos], 1000 * stack[pos, 0], yerr=1000 * stack[pos, 1], fmt="o", ms=4.8, color=color,
                    mec=SURFACE, mew=0.7, elinewidth=1.3, capsize=0, zorder=3)
    ax.plot(t_fit, 1000 * z["a_saturated"] * (t_fit / 1000) ** -z["alpha_saturated"], color=SHOWN["saturated"][1],
            linewidth=1.6, alpha=0.6, zorder=2)
    core_time, neighbor_time = packaged_snowball_times(str(z["detector"]))
    for value, color in ((core_time, SHOWN["saturated"][1]), (neighbor_time, SHOWN["1px"][1])):
        ax.axvline(value, color=color, linewidth=1.6, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.set_title(str(z["detector"]), fontsize=9.5, fontweight="bold", loc="left", color=TEXT, pad=3)
    ax.set_title(f"times: saturated {core_time:.0f} s, 1 px {neighbor_time:.0f} s;  "
                 f"alpha {float(z['alpha_saturated']):.2f}",
                 fontsize=8, loc="right", color=TEXT_2, pad=3)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.9)


def legend_handles():
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, ms=6, label=lab) for lab, c in SHOWN.values()]
    return handles + [
        plt.Line2D([], [], color=SHOWN["saturated"][1], alpha=0.6, linewidth=1.6,
                   label="power-law fit, saturated pixels"),
        plt.Line2D([], [], color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1.3, label="5 mDN/s threshold"),
        plt.Line2D([], [], color=TEXT_2, linestyle=(0, (1, 2)), linewidth=1.6,
                   label="SnowblindStep's times: 99% of per-core\ndecay times (blue: saturated, orange: 1 px)")]


def main(figure, paths):
    decays = {str(np.load(p)["detector"]): np.load(p) for p in paths}
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
                         "ytick.color": TEXT_2, "axes.edgecolor": GRID, "font.family": "sans-serif"})
    n_rows, n_cols = len(COLUMNS[0]), len(COLUMNS)
    width = LEFT + n_cols * PANEL_W + (n_cols - 1) * GAP_W + RIGHT
    height = TOP + n_rows * PANEL_H + (n_rows - 1) * GAP_H + BOTTOM
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), sharex=True, sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=LEFT / width, right=1 - RIGHT / width, top=1 - TOP / height, bottom=BOTTOM / height,
                        wspace=GAP_W / PANEL_W, hspace=GAP_H / PANEL_H)
    for col, dets in enumerate(COLUMNS):
        for row, det in enumerate(dets):
            ax = axes[row, col]
            if det is None:
                ax.axis("off")
                ax.legend(handles=legend_handles(), loc="center left", frameon=False, fontsize=8.5,
                          handletextpad=0.6, labelspacing=0.55)
                continue
            draw_panel(ax, decays[det])
            last_in_column = row + 1 == len(dets) or dets[row + 1] is None
            if last_in_column:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel("time since the end of the hit's integration (s)")
    axes[0, 0].set_xlim(50, 25000)
    axes[0, 0].set_ylim(0.1, 800)
    axes[0, 0].text(60, 1000 * THRESHOLD * 1.15, "5 mDN/s", fontsize=8, color=TEXT_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("excess (mDN/s)")
    fig.text(LEFT / width, 1 - 0.12 / height, "Snowball core persistence in NIRCam, NIRSpec and NIRISS darks",
             ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(LEFT / width, 1 - 0.4 / height,
             "Dark programs 9022 (NIRCam), 1121 (NIRSpec) and 9279 (NIRISS). Mean excess over cores (800 to 2700 per "
             "detector) in time bins;\nerror bars: standard error over cores; bins at or below zero not drawn. NRS1's "
             "times are lengthened by its 2026 darks (9273), 23% above 2022.",
             ha="left", va="top", fontsize=8, color=TEXT_2, linespacing=1.4)
    fig.savefig(figure, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
