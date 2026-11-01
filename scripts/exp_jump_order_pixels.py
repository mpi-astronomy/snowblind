"""Follow-up to ``exp_jump_order.py``: count second-pass flags per event, and find the pixels the downward ones sit on.

Reads ``flags.npz`` and ``summary.json`` from ``results/<date>_jump_order/<detector>/``.  For each data
(raw, cleaned), detector and threshold:

- flags on background pixels, away from the reference border and from first-pass flags of the same group
  (as in the tally of ``exp_jump_order.py``), are kept;
- flags of one pixel and integration in consecutive groups are merged into one event, signed by the sum of
  their excess over the pixel's median difference.  The likelihood search drops two adjacent differences at
  once, and the second of a pair has no sign of its own, so counting differences inflates its downward count;
- the downward events are traced to pixels: how many pixels hold them, how many of those pixels hold more
  than one, how many carry a pixel DQ flag (by name, against the share among all background pixels), and
  their read noise against the frame's median.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy import ndimage
from stdatamodels.jwst.datamodels import dqflags

BORDER = 4


def near_first_pass(first_jump, shape):
    """Boolean cube, True within one pixel (8-neighbours, same group) of a first-pass jump flag."""
    cube = np.zeros(shape, dtype=bool)
    cube[tuple(first_jump.T)] = True
    for i in range(shape[0]):
        for g in range(shape[1]):
            if cube[i, g].any():
                cube[i, g] = ndimage.binary_dilation(cube[i, g], structure=np.ones((3, 3), bool))
    return cube


def events(where, excess):
    """Merge flags of one pixel and integration in consecutive groups; returns (i, y, x, summed excess) per event."""
    order = np.lexsort((where[:, 1], where[:, 3], where[:, 2], where[:, 0]))
    w, e = where[order], excess[order]
    same_pixel = np.all(w[1:, [0, 2, 3]] == w[:-1, [0, 2, 3]], axis=1)
    consecutive = same_pixel & (w[1:, 1] == w[:-1, 1] + 1)
    starts = np.concatenate([[True], ~consecutive])
    ids = np.cumsum(starts) - 1
    total = np.bincount(ids, weights=e)
    first = w[starts]
    return first[:, 0], first[:, 2], first[:, 3], total


def dq_names(value):
    """Names of the pixel DQ bits set in ``value``."""
    return [name for name, bit in dqflags.pixel.items() if bit and value & bit]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="results/<date>_jump_order/<detector>")
    args = parser.parse_args()

    with open(os.path.join(args.directory, "summary.json")) as f:
        summary = json.load(f)
    stored = np.load(os.path.join(args.directory, "flags.npz"))
    shape = tuple(summary["shape"])
    background = stored["background"].copy()
    background[:BORDER], background[-BORDER:], background[:, :BORDER], background[:, -BORDER:] = False, False, False, False
    near = near_first_pass(stored["first_jump"], shape)
    pixeldq = stored["pixeldq"]
    readnoise = stored["readnoise"]
    median_rn = float(np.median(readnoise[background]))
    bg_dq = Counter(name for value in pixeldq[background][pixeldq[background] != 0] for name in dq_names(int(value)))
    nbg = int(background.sum())
    print(f"{summary['detector']} {summary['readpatt']}: {nbg} background pixels, median read noise {median_rn:.2f} DN; "
          "DQ among background pixels: " + ", ".join(f"{k} {100 * v / nbg:.2f}%" for k, v in bg_dq.most_common(5)))
    print("data     detector    k  flags  events_up events_dn  up-dn  dn_pixels  dn_pix>1  dn_on_dq  dn_rn/median  "
          "top DQ of dn pixels")

    rows = []
    keys = sorted({k.rsplit("_", 1)[0] for k in stored.files if k.endswith("_where")},
                  key=lambda k: (k.split("_")[0] != "raw", k.split("_", 1)[1].rsplit("_", 1)[0], float(k.rsplit("_", 1)[1])))
    for key in keys:
        data, rest = key.split("_", 1)
        detector, k = rest.rsplit("_", 1)
        where = stored[f"{key}_where"].astype(np.int64)
        excess = stored[f"{key}_excess"]
        keep = background[where[:, 2], where[:, 3]] & ~near[tuple(where.T)]
        where, excess = where[keep], excess[keep]
        if where.size == 0:
            continue
        i, y, x, total = events(where, excess)
        down = total < 0
        down_pixels = Counter(zip(y[down].tolist(), x[down].tolist()))
        repeat = sum(1 for n in down_pixels.values() if n > 1)
        on_dq = sum(1 for (yy, xx) in down_pixels if pixeldq[yy, xx] != 0)
        dq_counts = Counter(name for (yy, xx) in down_pixels for name in dq_names(int(pixeldq[yy, xx])))
        rn_ratio = float(np.median([readnoise[yy, xx] for (yy, xx) in down_pixels]) / median_rn) if down_pixels else float("nan")
        row = {
            "data": data, "detector": detector, "threshold": float(k), "flags": int(len(where)),
            "events_up": int((~down).sum()), "events_down": int(down.sum()),
            "down_pixels": len(down_pixels), "down_pixels_repeat": repeat, "down_pixels_with_dq": on_dq,
            "down_readnoise_ratio": rn_ratio, "down_dq": dict(dq_counts.most_common(4)),
        }
        rows.append(row)
        top = ", ".join(f"{n} {c}" for n, c in dq_counts.most_common(3))
        print(f"{data:8s} {detector:10s} {float(k):4.1f} {len(where):6d} {row['events_up']:9d} {row['events_down']:9d} "
              f"{row['events_up'] - row['events_down']:6d} {len(down_pixels):10d} {repeat:9d} {on_dq:9d} {rn_ratio:13.2f}  {top}")
    with open(os.path.join(args.directory, "pixels.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
