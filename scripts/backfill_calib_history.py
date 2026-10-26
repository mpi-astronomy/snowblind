"""Add the ASDF history entries to reset calibration files written before the writers recorded them.

A file gets a calibration entry built from its ``provenance`` if it has
none, and a timing entry from ``timing_provenance`` if the file carries
per-channel timing and has no timing entry.
"""
import argparse
import os

import asdf
import numpy as np

from snowblind.reset_calib import add_history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for path in args.files:
        with asdf.open(path, mode="rw") as handle:
            tree = handle.tree
            descriptions = [e["description"] for e in handle.get_history_entries()]
            added = []
            if not any(d.startswith("Reset-level calibration") for d in descriptions):
                prov = tree.get("provenance", {}) or {}
                count = int(np.nanmax(np.asarray(tree["n_integrations"])))
                add_history(
                    handle,
                    f"Reset-level calibration for {tree['detector']} from {count} dark integrations of "
                    f"{', '.join(prov.get('input_files', []))} (refpix {prov.get('refpix_algorithm')}, "
                    f"superbias {prov.get('superbias')}): median sigma_b {float(tree['sigma_b_median']):.2f} DN "
                    f"(entry added after the fact on {os.path.basename(path)}'s provenance)",
                )
                added.append("calibration")
            timing = tree.get("timing_provenance")
            if tree.get("t1_scale_channels") is not None and not any(d.startswith("First-read timing") for d in descriptions):
                source = timing.get("file", "unknown data") if isinstance(timing, dict) else "unknown data"
                add_history(
                    handle,
                    f"First-read timing from {source}: t1_scale {float(tree['t1_scale']):.4f}, "
                    f"reset_offset {float(tree['reset_offset']):.2f} DN; per channel t1_scale "
                    f"{[round(float(v), 4) for v in tree['t1_scale_channels']]} (entry added after the fact)",
                )
                added.append("timing")
            if added:
                handle.update()
            print(f"{path}: added {added or 'nothing'}")


if __name__ == "__main__":
    main()
