# CPU performance changes

The `Speed-up` branch replaces the previous CPU implementation directly. There
is no optimization switch and no GPU dependency. Keep the existing physics
parameters and MPI rank count when comparing a fiducial run.

## Changes

- **Shared Fourier windows.** Reionization and heating evaluate each radial
  window once per Fourier cell and apply it to all active fields. Copying and
  filtering share one pass. The three window definitions and the unfiltered
  final reionization step are unchanged.
- **Heating coefficients.** Redshift derivatives and source spectral factors
  are computed once per shell and snapshot. The ionization interpolation index
  is found once per cell, outside the shell loop.
- **Source FFT reuse.** Each newly constructed source field is transformed and
  normalized once. Heating shells combine these cached spectra with the
  existing light-cone time weights. With 40 heating shells, this replaces 40
  source forward transforms per field with one per snapshot. All shell inverse
  transforms and subsequent real-space clipping remain. Density and ionization
  transforms are not included in this reduction.
- **Circular histories.** The existing history allocations hold normalized
  Fourier spectra in a circular buffer. Advancing a snapshot overwrites one
  slot instead of moving all older grids. A second full history allocation is
  not required. This changes the internal meaning of the history pointers in
  `reion_grids_t`; rebuild any code linked against this structure.
- **Sparse MPI exchange.** Source properties for each occupied cell are packed
  together and sent to the rank owning that slab. Feedback requests contain
  unique occupied cells, and replies contain only their values. Empty ranks
  still participate. Transfers beyond MPI-3's integer byte-count limits use
  bounded communication rounds. Galaxy contribution order, source calibration
  and stochastic draw order on each rank are preserved.
- **Selected tree reads.** VELOCIRAPTOR readers distribute the `ForestID` scan,
  exchange row indices with forest owners, then read only the selected rows of
  the other properties. Original row order and global indices are retained for
  descendant and FOF links. Dataset handles stay open across bounded reads.
  The forest ownership directory is still gathered on every rank; this has a
  memory and communication cost proportional to the number of selected forests.

The gains depend on which part of the run dominates. Inverse FFTs, frequency
integrals, galaxy evolution and output I/O still cost time. Dense source fields
can reduce the advantage of sparse packing. No end-to-end speedup on Gadi is
claimed from the small checks below.

## Build and check on Gadi

Use the same MPI, parallel HDF5, FFTW and GSL modules as the previous fiducial
build. Reconfigure and rebuild, retaining its CMake physics options:

```sh
git switch Speed-up
git pull --ff-only origin Speed-up
cmake -S . -B build -DUSE_CUDA=OFF
cmake --build build --parallel
```

For a fresh build directory, supply the same `USE_STOCHASTICITY`,
`USE_MINI_HALOS`, magnitude options and build type as the fiducial executable.
In particular, the existing `Release` configuration enables `-ffast-math`;
changing build type during the comparison would introduce another difference.
No input parameter changes are needed for these optimizations.

The small MPI checks require no simulation data or Criterion:

```sh
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target test_cpu_mpi --parallel
ctest --test-dir build -L cpu --output-on-failure
```

Run these inside a compute allocation. They use 1, 2 and 3 ranks on a 4³ grid,
including a rank with no slab, a rank with no galaxies, duplicate cells,
periodic positions, dead galaxies, both SFR prescriptions, and history
wraparound. They check the shared windows against the scalar filter and compare
cached history combinations with an independent FFT of the real-space sum.

## Validation performed

Base revision: `e0a73833d180264711c40ab70abaec2320e3c2bd`.

- Compiled both the default configuration and
  `USE_STOCHASTICITY=ON, USE_MINI_HALOS=ON`, using `RelWithDebInfo`.
- All six small MPI checks passed across those two builds.
- The L10_N512 IGM run on two ranks completed, and its summaries agreed with a
  freshly built base revision under `compare.py --mode strict`.
- A shortened L10_N512 X-ray run on two ranks (32³ grid, four heating shells,
  output snapshots 10 and 25, HMXB and both AGN X-ray components) completed and
  agreed with the base revision under the same strict comparison.
- The combined stochasticity/mini-halo build completed the same short run with
  Lyman-Werner feedback, no-SFR treatment, X-ray scatter and source recalibration.
  All floating-point grid datasets in both short runs were finite.

MPI reduction order and Fourier-space time weighting can change rounding, so
bitwise equality of a full fiducial simulation is not promised. The strict
summary comparison uses relative tolerance `1e-6` for non-count quantities;
passing it is not a substitute for the full scientific validation on Gadi.

The checked-in IGM reference identifies revision `72800527` and differs from
both the base revision and this branch in this environment. It was not replaced;
the comparison above uses the actual base revision. The Criterion suite could
not start here because its message server returned `Operation not permitted`.
Full fiducial runs and performance measurements are left to Gadi.

## Repeat the supplied fiducial job on Gadi

The launcher matches job `179580303.gadi-pbs` and the supplied PBS script:
`hugemem`, project `qp00`, 32 MPI ranks, 320 GB memory, 12 hours, and
`storage=gdata/qp00+gdata/cm25+scratch/qp00`. It retains `OMP_NUM_THREADS=1`
and GCC 12.2.0, OpenMPI 4.1.4, FFTW 3.3.10-nci1, parallel HDF5 1.12.2p and
GSL 2.7.1. CMake 3.24.2 is additionally loaded for compilation and then unloaded.

After updating a dedicated `Speed-up` checkout on a Gadi login node, run:

```sh
bash tools/gadi_speedup_fiducial.sh
```

`input/params/runs/gadi_fiducial_reference.par` records all 214 parameter values
printed in the supplied log. The launcher reads the original
`Fiducial_0.0propscale/meraxes.hdf5` through
`dragons.meraxes.io.read_input_params(..., raw=True)`, checks every logged value,
and writes the original full-precision values to the new run inputs. This
avoids both changes to today's defaults and rounding the model parameters to
the six significant figures printed in the log. The old HDF5 file is read only.
The launcher uses the existing `venvs/tauefix/bin/python3`; `SPEEDUP_PYTHON` can
select another Python environment containing DRAGONS.

The run uses the same 512³ grid, snapshots 0–119, random seed 1809, recombination
and UVB settings. Spin-temperature evolution, X-ray scatter, escape-fraction
scatter and source recalibration remain off. Thus this fiducial tests the
filter, source-grid, feedback and tree-read optimizations, but not the heating
history FFT cache.

Build type, compiler flags, photometry options and array dimensions are read
from `/g/data/qp00/sz4640/meraxes-devs/build_stochastic/CMakeCache.txt`.
`SPEEDUP_BASELINE_BUILD` can point to the original build directory if moved.
Stochasticity is enabled, mini-halos and CUDA are disabled. The launcher requires
the original build cache and matching HDF5 parameters before submitting.

Each invocation creates a new directory under
`/g/data/qp00/sz4640/Results/Genesis_L210_N4320/Fiducial_0.0propscale_Speed-up`.
The script prints its path and job ID. Outputs go in `output/`, the simulation
log in `run.log`, compilation output in `build.log`, and MPI runtime in
`timing.txt`. PBS streams are saved as `pbs_JOBID.out` and `pbs_JOBID.err`.
The original fiducial output directory is preserved for comparison.

Compilation runs in the PBS allocation. The source is frozen in a detached
worktree and parameters are stored in `provenance/`, so later branch updates
do not change a queued job. Unsupported parameters or mismatches with the
original log stop preparation before submission. `SPEEDUP_OUTPUT_ROOT` changes
only the parent directory for the new run.
