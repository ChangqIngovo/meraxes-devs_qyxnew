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
- **Streaming tree reads.** VELOCIRAPTOR readers read contiguous blocks, with
  one reader per property column, then broadcast the columns and select each
  rank's forests locally. Column reads can proceed concurrently; dataset handles
  stay open across blocks. The workspace is bounded to 1,048,576 rows (about
  88 MiB per rank including selection indices). Original row order and global
  indices are retained for descendant and FOF links. No global forest directory
  or HDF5 point selection is required.

The gains depend on which part of the run dominates. Inverse FFTs, frequency
integrals, galaxy evolution and output I/O still cost time. Dense source fields
can reduce the advantage of sparse packing. Small checks do not establish the
end-to-end performance of the current reader on Gadi.

## Tree reader regression and repair

The full 32-rank, 512-cubed fiducial logs from Gadi showed that revision
`20c5ffb` reduced wall time from 18,881 to 16,030 seconds, while the tree-reading
stage increased from 1,710.9 to 5,072.1 seconds. Both runs covered snapshots
0--119 with spin-temperature evolution disabled. These are measurements from
two different jobs, not an isolated I/O benchmark.

The first CPU optimization replaced the original contiguous column reads and
broadcasts with a global forest directory, row routing, and scattered HDF5 point
reads. The original reader already read each column only once from disk; it did
not make every rank read every column. This repair restores contiguous streaming
and removes the per-snapshot directory gather/sort and row-index exchange, while
retaining the other CPU optimizations. Broadcast traffic is restored in exchange
for sequential disk access and bounded workspace.

Each snapshot now logs maximum-rank accumulated times for disk reads,
broadcast/wait, forest selection, and halo conversion. These maxima may come from
different ranks and must not be added to estimate wall time. Broadcast/wait also
includes waiting for other readers. The existing outer tree timer includes
opening/closing files and other setup.

The repaired reader passed synthetic MPI checks on 1, 2 and 3 ranks in the
default and stochasticity/mini-halo builds. They cover contiguous and chunked
HDF5 storage, normal and augmented trees, empty snapshots and ranks, full and
partial forest selections, global row indices, descendant indices, and FOF links
across read blocks. A 17-rank default-build check also covers ranks that only
receive broadcasts and do not read any property columns. The L10 default, IGM,
and X-ray runs through snapshot 75, as well as the short X-ray setup, passed
strict comparisons against the actual pre-optimization `e0a7383` code.
Full-size timing of this repair remains to be measured on Gadi.

The checked-in default, IGM, and X-ray references at `72800527` still fail their
comparisons in this environment; the actual pre-optimization runs agree with the
repaired code. Those historical reference files were not changed.

Run only the synthetic reader checks with:

```sh
cmake --build build --target test_tree_reader_mpi --parallel
ctest --test-dir build -L '^io$' --output-on-failure
```

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
Full scientific validation and timing of the repaired reader are left to Gadi.
