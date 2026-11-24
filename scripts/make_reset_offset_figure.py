"""Figure for issue #47: the reset offset of each integration against the packaged calibration.

Reads the ``<exposure>.json`` files of ``measure_reset_offset_per_integration.py``
and plots, for the multi-integration darks and images, the mean of the four
output channels' offsets against integration number, one line per exposure,
with the packaged calibration's offset as a dashed line.  WFSS exposures are
left out: the stream clean removes their first-group level.

Output: ``reset_offset_per_integration.png`` in the input directory.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    args = parser.parse_args()

    runs = [json.load(open(path)) for path in glob.glob(os.path.join(args.directory, "jw*.json"))]
    shown = sorted(
        (run for run in runs if len(run["integrations"]) > 1 and run["exp_type"] in ("NIS_DARK", "NIS_IMAGE")),
        key=lambda run: (run["exp_type"] != "NIS_DARK", run["date_obs"]),
    )
    calibration = float(np.mean(runs[0]["calibration_offsets"]))
    ndark = 0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=11)

    ax.axhline(calibration, color=TEXT_SECONDARY, linestyle=(0, (5, 4)), linewidth=1.5, zorder=2)
    for color, marker, run in zip(SERIES, MARKERS, shown):
        numbers = [integ["integration"] for integ in run["integrations"]]
        means = [np.mean([c["offset"] for c in integ["channels"]]) for integ in run["integrations"]]
        if run["exp_type"] == "NIS_DARK":
            ndark += 1
            label = f"Dark {ndark}"
        else:
            label = "Image"
        ax.plot(numbers, means, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)

    nmax = max(len(run["integrations"]) for run in shown)
    ax.set_xticks(range(1, nmax + 1))
    ax.set_xlim(0.7, nmax + 0.3)
    ax.text(nmax + 0.3, calibration + 0.5, f"calibration ({calibration:.1f} DN)", ha="right", va="bottom",
            fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_xlabel("Integration", fontsize=11.5, color=TEXT)
    ax.set_ylabel("Reset offset (DN)", fontsize=11.5, color=TEXT)
    ax.set_title("NIRISS reset offset, per integration", fontsize=13, color=TEXT, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(shown), fontsize=10.5, frameon=False,
              handlelength=1.8, columnspacing=1.4, borderaxespad=0.2, labelcolor=TEXT)
    ax.yaxis.set_major_locator(plt.MultipleLocator(3))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.8, bottom=0.17)
    out = os.path.join(args.directory, "reset_offset_per_integration.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
