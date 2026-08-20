#!/usr/bin/env python
"""
Plot the delta_T lightcone strip produced by meraxes when
Flag_Compute21cmBrightTemp and Flag_ConstructLightcone are both on.

meraxes writes the lightcone into the grids file of the snapshot that
matches EndSnapshotLightcone (i.e. the lowest-redshift snapshot of the
run), as two datasets living at the top level of that file:

    LightconeBox  (ReionGridDim, ReionGridDim, LightconeLength)  [mK]
    lightcone-z   (LightconeLength,)                              redshift of each LOS slice

The third axis of LightconeBox is the line-of-sight (constant comoving-
distance steps, so *not* evenly spaced in z) - this script uses
pcolormesh with the real lightcone-z array so the redshift axis is
correctly nonlinear, rather than assuming a linear mapping.

Usage:
    python plot_lightcone.py <grids_file_at_EndSnapshotLightcone> [output.png]
        [--slice-index N] [--box-size L] [--vmin V] [--vmax V]

Example:
    python plot_lightcone.py /scratch/.../AGN1/meraxes_grids_119.hdf5 lightcone.png
"""

import argparse

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def eor_colour():
    """Approximate the standard 21cmFAST 'EoR' brightness-temperature colormap:
    white (deep absorption) -> yellow -> orange -> red -> black (0 mK) -> blue -> cyan (emission)."""
    stops = [
        (0.00, "#ffffff"),
        (0.21, "#f5e642"),
        (0.42, "#f28c28"),
        (0.63, "#c0392b"),
        (0.83, "#000000"),
        (0.92, "#1f4fd6"),
        (1.00, "#57d9ff"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("EoR_colour", stops)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grids_file", help="grids HDF5 file for the EndSnapshotLightcone snapshot")
    p.add_argument("output", nargs="?", default="lightcone.png", help="output image path")
    p.add_argument("--slice-index", type=int, default=None,
                    help="transverse index to slice at (default: ReionGridDim//2)")
    p.add_argument("--box-size", type=float, default=None,
                    help="box size in h^-1 Mpc for the transverse axis label (default: read from file if present)")
    p.add_argument("--vmin", type=float, default=-150.0, help="colour scale min [mK]")
    p.add_argument("--vmax", type=float, default=50.0, help="colour scale max [mK]")
    args = p.parse_args()

    with h5py.File(args.grids_file, "r") as f:
        if "LightconeBox" not in f or "lightcone-z" not in f:
            raise SystemExit(
                f"{args.grids_file} has no LightconeBox/lightcone-z datasets.\n"
                "Make sure this is the grids file for EndSnapshotLightcone, and that the run had\n"
                "Flag_Compute21cmBrightTemp=1 and Flag_ConstructLightcone=1 set."
            )
        box = f["LightconeBox"][:]          # (nx, ny, n_los), mK
        z_los = f["lightcone-z"][:]         # (n_los,)
        box_size = args.box_size
        if box_size is None:
            box_size = f.attrs.get("BoxSize", [None])[0] if "BoxSize" in f.attrs else None

    nx, ny, n_los = box.shape
    islice = args.slice_index if args.slice_index is not None else ny // 2
    img = box[:, islice, :]  # (nx, n_los)

    if box_size is None:
        transverse = np.arange(nx + 1)
        ylabel = "cell index"
    else:
        transverse = np.linspace(0, box_size, nx + 1)
        ylabel = r"[$h^{-1}$ Mpc]"

    # pcolormesh needs cell *edges*; approximate z edges from the (nonuniform) z centres.
    z_edges = np.empty(n_los + 1)
    z_edges[1:-1] = 0.5 * (z_los[:-1] + z_los[1:])
    z_edges[0] = z_los[0] - (z_edges[1] - z_los[0])
    z_edges[-1] = z_los[-1] + (z_los[-1] - z_edges[-2])

    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor="w")
    mesh = ax.pcolormesh(z_edges, transverse, img, cmap=eor_colour(),
                          vmin=args.vmin, vmax=args.vmax, shading="flat")
    ax.set_xlabel("z")
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()  # high z on the left, matching the reference figure
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label(r"$\delta T_b$ (mK)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
