#!/usr/bin/env python
"""
Reproduce the "xHI slice at fixed neutral-fraction milestones" comparison
figure from a meraxes run's grids output — one row of panels per run
(showing the xH grid at redshifts closest to a set of target
mass-weighted-global-xH values), optionally with a difference row at the
bottom if exactly two runs are given.

Reads directly from the per-snapshot meraxes_grids_<snap>.hdf5 files (no
`dragons` dependency) — only snapshots with a fully-populated 3D `xH`
dataset are usable (snapshots outside ListOutputSnaps only carry the
scalar summary attributes and are skipped automatically). Note: the box
size in h^-1 Mpc is NOT stored in these grid files — pass it with
--box-size if you want physical axis units instead of cell index.

Usage:
    python plot_xHI_comparison.py RUN_DIR [RUN_DIR2] [output.png]
        [--targets 0.75 0.50 0.25] [--slice-index N] [--box-size L] [--labels A B]

Example (single run):
    python plot_xHI_comparison.py /scratch/.../AGN0 xHI_AGN0.png --box-size 210

Example (two runs + difference row, matching the reference figure):
    python plot_xHI_comparison.py /scratch/.../AGN0 /scratch/.../AGN1 xHI_compare.png \
        --labels AGN0 AGN1 --box-size 210
"""

import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def find_full_snapshots(run_dir):
    """Return {snapshot: (path, mass_weighted_global_xH)} for grid files that
    actually contain a populated (non-empty) xH dataset."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "meraxes_grids_*.hdf5")):
        m = re.search(r"meraxes_grids_(\d+)\.hdf5$", path)
        if not m:
            continue
        snap = int(m.group(1))
        try:
            with h5py.File(path, "r") as f:
                if "xH" not in f or f["xH"].shape == (0,):
                    continue
                xhi = f["xH"].attrs["mass_weighted_global_xH"][0]
        except (OSError, KeyError):
            continue
        out[snap] = (path, xhi)
    return out


def pick_snapshots_for_targets(snaps, targets, run_label):
    """For each target xHI value, return the (path, snap, xHI) whose actual
    mass_weighted_global_xH is closest to it. Warns if two targets end up
    picking the same snapshot (a gap in the available grids)."""
    picks = []
    items = list(snaps.items())  # [(snap, (path, xhi)), ...]
    seen = set()
    for target in targets:
        snap, (path, xhi) = min(items, key=lambda kv: abs(kv[1][1] - target))
        if snap in seen:
            print(f"warning [{run_label}]: target xHI={target} picked snapshot {snap}, "
                  f"already used by an earlier target — no grid is closer; there's a gap "
                  f"in your available output snapshots near this xHI value.", file=sys.stderr)
        seen.add(snap)
        picks.append((path, snap, xhi))
    return picks


def read_slice(path, slice_index):
    with h5py.File(path, "r") as f:
        xh = f["xH"][:]
    dim = xh.shape[0]
    i = slice_index if slice_index is not None else dim // 2
    return xh[:, :, i], dim


def eor_xhi_cmap():
    """Navy (ionized, xHI~0) -> gold (neutral, xHI~1), matching the reference figure."""
    return mcolors.LinearSegmentedColormap.from_list("xHI_colour", ["#0b1d4a", "#f2d94e"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", help="one or two directories of meraxes_grids_<snap>.hdf5 files, "
                                                "optionally followed by the output path")
    p.add_argument("--targets", type=float, nargs="+", default=[0.75, 0.50, 0.25],
                    help="target mass-weighted global xHI values, one column per target")
    p.add_argument("--slice-index", type=int, default=None, help="grid index to slice at (default: dim//2)")
    p.add_argument("--box-size", type=float, default=None,
                    help="box size in h^-1 Mpc for physical axis units (not stored in the grid files, "
                         "so falls back to cell index if omitted)")
    p.add_argument("--labels", nargs="+", default=None, help="row labels, one per run dir")
    p.add_argument("--output", default=None, help="output image path (or pass it as the last positional arg)")
    args = p.parse_args()

    run_dirs = list(args.run_dirs)
    output = args.output
    if output is None and len(run_dirs) > 1 and not os.path.isdir(run_dirs[-1]):
        output = run_dirs.pop()
    if output is None:
        output = "xHI_comparison.png"

    have_box_size = args.box_size is not None
    axis_label = r"[$h^{-1}$ Mpc]" if have_box_size else "cell index"

    labels = args.labels if args.labels else [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    n_runs = len(run_dirs)
    n_cols = len(args.targets)
    show_diff = n_runs == 2
    n_rows = n_runs + (1 if show_diff else 0)

    cmap = eor_xhi_cmap()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows),
                              facecolor="w", squeeze=False)

    slices = []  # slices[run_index][col_index] = 2D array
    extent = None
    for row, run_dir in enumerate(run_dirs):
        snaps = find_full_snapshots(run_dir)
        if not snaps:
            raise SystemExit(f"No fully-populated xH grids found in {run_dir}")
        picks = pick_snapshots_for_targets(snaps, args.targets, labels[row])
        row_slices = []
        for col, (path, snap, xhi_actual) in enumerate(picks):
            img, dim = read_slice(path, args.slice_index)
            if extent is None:
                extent = (0, args.box_size, 0, args.box_size) if have_box_size else (0, dim, 0, dim)
            row_slices.append(img)
            ax = axes[row][col]
            ax.imshow(img.T, origin="lower", cmap=cmap, vmin=0, vmax=1, extent=extent)
            ax.text(0.03, 0.95, rf"$\bar{{x}}_{{\rm HI}}\sim{xhi_actual:.2f}$ (snap {snap})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    color="w", bbox=dict(fc="k", alpha=0.4, pad=2))
            if col == 0:
                ax.set_ylabel(f"{labels[row]}\n{axis_label}")
            if row == n_runs - 1:
                ax.set_xlabel(axis_label)
        slices.append(row_slices)

    if show_diff:
        diff_vmax = max(0.05, max(np.abs(slices[0][c] - slices[1][c]).max() for c in range(n_cols)))
        for col in range(n_cols):
            diff = slices[0][col] - slices[1][col]
            ax = axes[n_runs][col]
            im = ax.imshow(diff.T, origin="lower", cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax, extent=extent)
            if col == 0:
                ax.set_ylabel(f"{labels[0]} - {labels[1]}\n{axis_label}")
            ax.set_xlabel(axis_label)
        cbar = fig.colorbar(im, ax=axes[n_runs, :], shrink=0.8, pad=0.01)
        cbar.set_label(rf"$\Delta \bar{{x}}_{{\rm HI}}$ (scale: $\pm${diff_vmax:.3f})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes[:n_runs, :], shrink=0.8, pad=0.01).set_label(r"$\bar{x}_{\rm HI}$")

    if not have_box_size:
        print("note: no --box-size given, axes are in cell index — pass --box-size L "
              "(e.g. --box-size 210 for a 210 h^-1 Mpc box) for physical units.", file=sys.stderr)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
