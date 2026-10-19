# Online H/He excursion-set worker

`snapshot_worker.py` is called by the native Meraxes snapshot loop before its
feedback and output stages. It advances the optimized, fixed-centre spherical
ES model in `es_hhe_reference.py`. This solver version includes the dormant
photon activity floor and sparse-geometry optimizations. The SHA-256 recorded
in `snapshot_worker.REFERENCE_SHA256` identifies the validated solver source.
The worker verifies this hash before accepting a request.

The ES filter catalogue, common photon accounts and competing H/He absorption
retain the reference model. Local case-A recombination and eight coupled OTS
channels operate at **fixed 10,000 K**. This is an isothermal closure, with
uniform mixing inside selected ES spheres, rather than a spatial/angular
radiation-transfer or thermal-evolution calculation. The worker runs in one
Python coordinating process with C/OpenMP candidate filters, sparse sums,
H/He updates and compact common-exposure loops, together with threaded
large-grid FFTs. The native MPI application distributes galaxy/grid work.
The ES worker uses a single node's allocated CPU set; it does not distribute
the spatial ES solve across MPI nodes.

## Run and test

The serial reference needs Python 3, NumPy and h5py. The Gadi native backend
also uses SciPy for large FFTs and a C/OpenMP library compiled with
`BUILD_ES_NATIVE=ON`. `tools/gadi_es_l10.sh build` enables this option.

```bash
python tools/es_hhe/test_snapshot_worker.py
python tools/es_hhe/test_photon_floor.py
python tools/es_hhe/test_sparse_geometry.py
python tools/es_hhe/snapshot_worker.py \
  --request request.hdf5 --response es_state_001.hdf5 \
  --checkpoint .es_checkpoint.hdf5
```

Copy the entire directory when freezing a run: the worker imports the reference
from its own directory. Each physical substep logs its duration and remaining
photon budget. Progress also identifies recombination, candidate selection and
absorption, including support counts and chemistry iterations per batch.
The final JSON contains the cumulative count audit.

`ES_NATIVE_LIBRARY` selects a library explicitly; a prepared worker otherwise
loads `libes_hhe_kernels.so` beside its source. `ES_NATIVE_REQUIRED=1` prevents
fallback when the library is absent. `ES_WORKER_THREADS` controls the maximum
OpenMP/FFT thread count. The MPI adapter supplies `ES_MPI_CPUSET`, the union
of its ranks' bindings, and only the worker expands to that set. Run on one
node for multithreaded execution. Outputs record `execution_backend`; the
checkpoint identity also freezes native-library and dispatch-source hashes.

## Request contract (schema version 1)

The request contains float64 arrays `density` and `source_photons`, with
`grid_dim**3` entries in C-order `(x,y,z)` flattening.

* `density` is local hydrogen density divided by cosmic mean hydrogen density;
  it is **1 + overdensity**, not the overdensity itself.
* `source_photons` contains newly emitted, already escaped photon **counts**
  integrated over this interval. Do not pass cumulative stellar mass, an SFR,
  or a photon rate. The native source prescription uses the end-snapshot SFR
  as constant throughout the preceding interval.

Required attributes are `snapshot`, `grid_dim`, `dt_s`, `cell_volume_cm3`,
`mean_h_atoms_per_cell`, `helium_number_ratio`, `radius_ratio`,
`max_radius_cells`, and `run_identity`. `schema_version=1` is supported;
`temperature_K=10000`, `max_step_s` (default one Myr), `redshift`,
`source_spectrum_file`, and `source_energy_eV` are additional attributes.

`max_radius_cells=0` selects the full periodic minimum-image range. A positive
value restricts the maximum ES radius in grid-cell units. The logarithmic
catalogue always includes radius zero; `radius_ratio` must exceed one. Its
spacing is a spatial-resolution choice requiring a convergence test.

The primary spectrum file has two whitespace-separated columns:
`energy_eV photon_number_weight`, with `#` comments. Energies must be finite
and at least 13.6 eV; weights are nonnegative and normalized by their sum.
An arbitrary number of bands is supported, including duplicate energies with
distinct accounts. With no file, the explicitly supplied `source_energy_eV`
is used (the adapter fallback is 18 eV). Monoenergetic 18 eV is a controlled
source approximation and cannot ionize helium directly; it is not a calibrated
stellar SED. Primary groups precede the eight reference diffuse groups.

## State, physical time and density mapping

Internal gas and photon arrays are numbers divided by a constant
`Q = mean_h_atoms_per_cell`. Physical chemistry therefore uses an effective
volume `cell_volume_cm3/Q`, so its input densities are in physical cm^-3.
Q must be calculated from the comoving cell volume and be invariant throughout
the run. The physical cell volume changes with redshift.

The worker applies a per-cell, per-band activity floor of `1e-30` in these
normalized count units. Smaller accounts remain in `photons` and the ledger,
but do not constrain absorption steps or select ES supports. Emission can
accumulate and reactivate an account. The local recombination/OTS solver uses
the equivalent density threshold, preserving dormant credits across unit
conversion. The generic reference API defaults to a zero absolute floor;
`SnapshotModel` explicitly selects the production floor. Response/checkpoint
metadata records this numerical policy, and the diagnostics report the number
and summed balance of dormant accounts. See `docs/es_hhe_gadi.md` for the
snapshot-29 convergence replay and bounded sparse-geometry implementation.

Independent one-cell H-only supports use an exact terminal count update when
no new crossing can occur. Photon-limited cells consume their active accounts;
atom-limited cells with equal active H opacities share the HI debit in proportion
to their photon balances. Unused and dormant photons remain explicit. He-active,
overlapping and possible-crossing cases retain the coupled solver. This avoids
thousands of steps following an already-ionized cell's vanishing neutral tail.

The first requested snapshot initializes neutral gas and requires `dt_s=0`
and no sources. Every subsequent request must have the next consecutive
snapshot number and the same run identity, spectrum and solver configuration.
The run directory must be fresh for a new native process; arbitrary Meraxes
restart/checkpoint restoration is not implemented by this bridge.

At the next snapshot, the prescribed density retains each species fraction
locally. Gas entering a previously empty cell is neutral. The signed change
of all five species is recorded as `matter_exchange`, and the nucleus and
electron ledgers include that transfer. Photon counts are retained unchanged.
This is a local Eulerian density closure, not a reconstructed gas-advection
history. Minority species are scaled directly to avoid loss through `1-x`.

Each interval is split into steps no longer than `max_step_s`. Each step
performs local recombination/OTS for half a step, injects its portion of the
new primary photons, settles the ES absorption problem, then performs the
remaining half-step of recombination/OTS. The final source remainder makes
the injected interval count equal to the prescribed total. The local
recombination/OTS solver has its own splitting approximation, so this ordering
alone does not establish a second-order global time scheme. Snapshot density,
source prescription, time step and filter ladder need convergence checks.

## Outputs and recovery

Every response stores `xH` (neutral H fraction), `xHeI`, `xHeII`, `xHeIII`,
physical cumulative `HII_recombinations`, `HeII_recombinations`,
`HeIII_recombinations`, and the full normalized `gas`, `photons`,
`capture_counts`, `source_history`, `source_emitted_by_group`, `matter_exchange`,
`initial_nuclei` and `baseline_nuclei` arrays. The energy groups, primary
weights, count normalization and cumulative scalar audit are also recorded.
The gas species order is HI, HII, HeI, HeII, HeIII; recombination events are
HII→HI, HeII→HeI and HeIII→HeII. In empty cells the fractions are reported as
neutral, while all species counts remain zero.

The case-A capture fields are diagnostic counts. They must not also be
subtracted as an old Meraxes case-B/Nrec photon budget: OTS accounting already
contains their emitted and reabsorbed photons. The bridge disables the old
recombination tax on this runtime path.

The cumulative ledgers satisfy, including the explicitly recorded matter
transfer:

* photons remaining = primary emitted + diffuse emitted − absorbed;
* electrons = initial electrons + transferred electrons + absorbed − captures;
* hydrogen/helium nuclei = initial nuclei + transferred nuclei.

The checkpoint is protected by an exclusive file lock. The worker validates
and writes both response and checkpoint to staged files before publishing
them. The native caller consumes the response only after a successful worker
exit. Repeating an identical last request is idempotent; changed, skipped or
out-of-order requests fail. An interruption between the two file renames can
be recovered by the same request, from the previous checkpoint or by replay
of the new one. A different run identity cannot overwrite existing state.

No observational calibration, large-grid MPI scaling claim, finite-light-speed
propagation, angular shadowing or thermal closure follows from this adapter.
