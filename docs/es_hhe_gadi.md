# ES-HHe L10 run on Gadi

`Flag_PhotonConservation=3` runs the optimized ES-HHe reference during Meraxes
evolution through its C bridge. The spatial model uses periodic excursion-set
spheres and a shared photon ledger. This mode does not use ray packets or solve
the spatial radiation-transfer equation. Meraxes remains MPI-parallel. Its
rank-0 worker coordinates C/OpenMP geometry and H/He kernels, with threaded
FFTs for large grids. This is shared-memory parallelism within one node;
the spatial ES solve is not distributed across MPI nodes.

The supplied `Genesis_L10_N512.par` describes a 10 Mpc/h N-body volume with
512^3 particles and **32^3 input density cells**. The runner uses these 32^3
cells. It reads the actual `SimulationDir/a_list.txt`, evolves from snapshot 0
and requests every available snapshot with `OutputSnapshots : :` (the parser's
all-snapshot slice). Galaxy catalogues and supported grid fields are written at every
snapshot, including after reionization. `CALC_MAGS=OFF`; photometric catalogues
are not enabled by this run template.

The input is generated directly from this checkout's
`input/params/simulations/Genesis_L10_N512.par` and uses this checkout's
`input/params/defaults.par` for galaxy/source parameters. Runtime settings select
mode 3 and all snapshots. Existing table data are reused from
`/g/data/qp00/sz4640/meraxes-devs/input/`. The old
Sobacchi reionization feedback, UVB/spin-temperature closure, legacy 21-cm
brightness-temperature/power-spectrum outputs and black-hole
ionizing source are disabled in mode 3. H/He recombination is enabled in the
new solver, without the legacy `Nrec` correction. Mode 3 skips the legacy MHR
table initialization and does not read `RecombinationDir`; this directory need
not exist. Its input parameter is retained, and H/He recombination remains
enabled. The temperature is fixed at
10,000 K. These choices define a controlled ES-HHe test, not an already
recalibrated galaxy-feedback model.

No external input or defaults file is read or merged. The obsolete `L10_INPUT`
environment variable is unused. Galaxy/source values therefore follow the
current branch defaults, including `SfEfficiency=0.08` and
`ReionNionPhotPerBary=4000`, rather than an external calibration. Unknown
parameters in the L10 simulation or branch defaults fail native validation.

By default the source is monochromatic at 18 eV. This can ionize hydrogen but
cannot directly ionize helium. Helium chemistry/recombination is implemented;
an ionizing stellar SED is needed to exercise the helium channels. To use a
specified SED, set `ES_SPECTRUM` to a two-column text file containing photon
energy in eV and nonnegative photon-number weight. The integrated photon
normalization still comes from `ReionNionPhotPerBary` and the existing escape
fraction prescription. The runner does not invent calibrated helium-band
fractions.

## Update, build and prepare

Run on Gadi from a clean `Photon_conservation` checkout:

```bash
(
set -euo pipefail
export PC_REPO=/g/data/qp00/sz4640/meraxes_photon_conservation
export PC_BUILD_DIR="$PC_REPO/build-pc-mpi"
cd "$PC_REPO"
git switch Photon_conservation
git pull --ff-only origin Photon_conservation
git submodule update --init --recursive

module --force purge
module load gcc/12.2.0 openmpi/4.1.4 cmake/3.24.2
module load fftw3/3.3.10-nci1 hdf5/1.12.2p gsl/2.7.1
module load python3/3.11.7 intel-mkl/2023.2.0
source /g/data/qp00/sz4640/venvs/tauefix/bin/activate
export ES_PYTHON="$VIRTUAL_ENV/bin/python"
export PC_ENV_FILE=/dev/null
export L10_TABLES=/g/data/qp00/sz4640/meraxes-devs/input
unset L10_INPUT L10_SIMULATION_DIR

bash tools/gadi_es_l10.sh build
export RUN_DIR=/g/data/qp00/sz4640/pc_runs/l10_es_hhe_$(git rev-parse --short=8 HEAD)_$(date +%Y%m%d_%H%M%S)
bash tools/gadi_es_l10.sh prepare
qsub -v "RUN_DIR=$RUN_DIR" "$RUN_DIR/submit.pbs"
)
```

Before creating a run directory, `prepare` invokes the built executable with
`--check-params` on the generated input and its simulation/default parameters.
This runs the native parser and parameter consistency checks without creating
output or loading tables, grids or merger trees. Blank/indented-comment and CRLF
lines are accepted; unknown parameters report their name, file and line on stderr.
An invalid parameter file stops preparation before `qsub`.

The prepare step does not submit a job. It copies the built binary, Python
solver, input/default/simulation files, snapshot list, optional spectrum and
build manifest into a new run directory, and records their SHA-256 hashes.
The provenance directory retains exact copies of the branch's `defaults.par`
and `Genesis_L10_N512.par`. The manifest records both source paths.
The job checks these files before running. A pre-existing run directory is
rejected; choose a new `RUN_DIR` for another run. The existing ES runs are
untouched.

Environment overrides before `prepare`:

| Variable | Default / meaning |
|---|---|
| `L10_SIMULATION_DIR` | `SimulationDir` in `Genesis_L10_N512.par` |
| `L10_TABLES` | `/g/data/qp00/sz4640/meraxes-devs/input` |
| `ES_PYTHON` | `/g/data/qp00/sz4640/venvs/tauefix/bin/python` |
| `ES_SPECTRUM` | Optional explicit source SED file |
| `ES_MAX_STEP_MYR` | 1 Myr maximum chemistry/source time step |
| `ES_RADIUS_RATIO` | 1.1 between successive ES radii |
| `RUN_DIR` | `pc_runs/l10_es_hhe_<commit8>` |

The time-step and radius-ladder choices require convergence checks before
physical inference. Neither the 48-hour request nor the 320 GB allocation is
a measured completion-time or memory prediction for the full L10 run.

## Output and logs

Output is under `$RUN_DIR/output/`; the job log is
`$RUN_DIR/run_<PBS_JOBID>.log`. The manifest records the exact snapshot count,
redshift endpoints, input hashes and Python dependency versions.

Alongside the standard Meraxes catalogue/grid output, each snapshot has
`output/es_state_NNN.hdf5`. It contains `xH`, `xHeI`, `xHeII`, `xHeIII`, the
five-species `gas` array, surviving `photons`, `source_history`,
`capture_counts`, `matter_exchange`, the source/photon energies and weights,
and cumulative `HII_recombinations`, `HeII_recombinations` and
`HeIII_recombinations`. Gas/photon accounts use the normalization recorded in
the file's `count_normalization_atoms` attribute; the three named
recombination datasets are actual event counts. The `diagnostics` attribute
records the snapshot accounting checks. These archives retain helium and
photon information even where a standard Meraxes grid has no matching field.

The worker uses a per-cell, per-band photon activity floor of `1e-30` in
normalized count units: photon number divided by `count_normalization_atoms`.
Accounts at or below this floor remain in the state and conservation ledger;
they do not set absorption steps or select ES candidates. New source or
recombination emission can accumulate in the same account and reactivate it.
The floor also applies after each absorption debit and to local diffuse
absorption, with the appropriate density-unit conversion. It is not a minimum
physical photon number or a change to an ionization/recombination coefficient.
Output records `photon_activity_floor`, `deferred_photon_accounts` and
`deferred_photons`. For the default nine bands on a 32^3 grid, the instantaneous
sum in accounts at or below this floor is bounded by `2.95e-25` normalized
counts. These retained accounts are distinct from the existing relative
chemistry tolerances.

The PBS template requests 32 ranks, 320 GB and 48 hours on `hugemem`. The rank
count fits the 32 input-grid slabs. NCI's [Queue Limits](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits)
page (checked 2026-10-18) allows 1–48 cores and 48 hours for a single-node
`hugemem` job. `ES_WORKER_THREADS` defaults to `PBS_NCPUS`. The adapter passes
the union of the MPI ranks' CPU bindings to the worker, which uses that CPU
set without changing its parents' bindings. Other ranks poll a nonblocking
barrier with a 10 ms sleep while the worker computes. OpenMP waits passively
between kernels; BLAS and the MPI parents retain one thread each.

The L10 build enables `BUILD_ES_NATIVE=ON`, builds
`lib/libes_hhe_kernels.so`, and records its SHA-256 in the build manifest.
Preparation copies the library into `solver/`; PBS verifies its hash and
requires the native backend. It cannot silently fall back to the serial
reference when the library is missing. The worker log reports the OpenMP
team size and CPU-affinity size. Each output/checkpoint records this execution
metadata; numerical checkpoint identity includes both the native library and
Python dispatch hashes. Thread count can change without changing per-row
arithmetic or checkpoint identity.

## Integration validation

The earlier integration baseline built the full CPU Meraxes executable with GCC 13.3, Open MPI 4.1.6,
FFTW 3.3.10, GSL 2.7.1 and HDF5 1.10.10. The online adapter passed synthetic
consecutive-snapshot tests at 1, 2 and 4 ranks, including empty slabs, physical
source/time units, H/He recombination, density remapping and photon/electron/
nucleus accounting. A two-event star-formation regression verifies that mode 3
adds each event's new SFR once. The worker has nine integration tests; twenty-seven
runner tests cover full-snapshot configuration, provenance, PBS launch, shell
environment isolation and preparation without unused legacy recombination tables.
CTest passes the actual Meraxes executable to these runner tests so they also
exercise native parsing of the generated inputs, whitespace/CRLF handling and
unknown or malformed parameter diagnostics. Standalone runner tests without
`ES_PARAMCHECK_BINARY` use a command fixture and skip the four native-parser cases.
The input-source regressions verify the L10 volume and grid, branch-default
galaxy/source values and exact provenance copies. Stale `L10_INPUT` values,
including nonexistent paths or incompatible external files, have no effect.
These tests do not require or replace a full L10 merger-tree run.

A dilute 32^3 synthetic snapshot with one 18 eV source completed in 0.62 s
with approximately 85 MiB process peak RSS. It exercised the full periodic
radius catalogue and retained a zero photon-count residual. This is a
single-step smoke test, not a timing estimate for dense or late-time snapshots.
Gadi execution and the complete L10 history have not been tested here.

## Small-account convergence and sparse geometry

The supplied snapshot-28 checkpoint and snapshot-29 request exposed four
subnormal photon accounts that controlled the common exposure step while their
computed debits rounded to zero. The old solver reached its 20,000-iteration
guard in substep 3. The activity floor removes these accounts from the active
calculation without deleting their balances or raising the iteration guard.

The sparse implementation also:

- Computes weak FFT support sums over occupied cells when that is cheaper than
  traversing every cell in a sphere, retaining exact periodic membership and
  accurate sums, including subnormal values.
- Evolves only the union of selected supports when it is small. Storage is
  bounded by one million memberships and the grid size; broad overlapping
  supports retain the FFT path after that path proved faster in a benchmark.
- Reuses a completed candidate scan only after an exactly unchanged partial
  update. Any successful update triggers a fresh scan; cached centre indices
  are bounded to eight MB.

A controlled replay transferred the validated old checkpoint's arrays in memory
to the new worker and evolved the supplied snapshot-29 request. All four
substeps completed in approximately 24 seconds locally. The final photon
accounting residual was `1.43e-15` relative to total emission; volume- and
mass-weighted neutral hydrogen fractions were `0.999606519956` and
`0.999173852705`. Thirty dormant accounts retained `6.98e-30` normalized counts.
The gas and photon arrays were nonnegative. This is a solver replay, not a
Meraxes restart or a measurement of complete L10 runtime on Gadi.

A separate instrumented replay of the earlier implementation took 33 seconds including profiling overhead:
selected-support chemistry accounted for about 58% and candidate selection
about 40%. This identified compiled aggregate/scatter and chemistry kernels
as the next implementation targets, alongside reuse of periodic membership
work across photon bands.

New floor regressions cover per-account thresholds, accumulation/reactivation,
post-debit inactivity, subnormal convergence, source/recombination ledgers and
unit-conversion preservation. Geometry regressions compare exact memberships,
aggregate/scatter adjoints and bounded dispatch. They are included in the
Gadi build's `test_photon_es_` CTest selection. Start the updated production run
in a fresh directory from snapshot 0; existing checkpoints retain their original
solver/configuration identity and are not automatically migrated.

## Compiled shared-memory execution

The native backend evaluates periodic sparse support sums and indexed
aggregate/scatter operations in C. Independent output rows run in OpenMP;
each row keeps a fixed ordered expansion sum, including subnormal values.
There is no floating-point reduction across threads. The H/He upward update
uses the reference's exponential maps and small-probability series, with one
independent update per cell. Small workloads stay serial inside the compiled
kernel to avoid launching threads for tens of cells.

The compact multi-support path also compiles the complete common-exposure
midpoint loop: field construction, donor feasibility checks, backtracking and
the earliest crossing event. It uses one shared exposure and photon ledger;
overlapping spheres are never evolved as independent physical regions. A
single support keeps the reference's exact monochromatic endpoint. Every
successful native result still passes the Python photon, nucleus, electron,
positivity and finite-value audits before commit. Invalid native results
raise an error; they are not accepted as partial progress.

Candidate row classification combines eligibility, representable absorption
and necessary photon-count bounds in one OpenMP kernel. Rows have independent
outputs. If contraction roundoff could change a count-bound decision, the
whole radius uses the original NumPy classification, preserving its BLAS
shape and summation behavior.

Large native FFTs use SciPy's threaded transform at every thread count,
including one, so changing thread count does not select a different FFT
implementation. Transforms below 1,048,576 cells retain the original NumPy
path. Compilation disables fast-math and floating-point contraction.

Tests compare one/multiple threads, signed high-dynamic-range sums,
subnormal retention, H/He nucleus and electron budgets, mixed spectra,
overlapping supports, FFT round trips, checkpoint identity, and worker CPU
allocation. A delayed-worker native adapter test checks that waiting MPI ranks
consume little process CPU and preserve their original CPU bindings.

For this change, the native library compiled locally with GCC 13 and strict
floating-point options. The Python suite passed 81 tests; four cases requiring
the complete Meraxes parameter parser were skipped. The four physical
regressions also passed. The new MPI affinity/wait helper compiled against
MPI, but this environment cannot initialize MPI sockets and lacks the full
parallel HDF5/FFTW toolchain, so the updated full adapter tests still require
the Gadi build gate. Earlier adapter results above are not evidence that this
new MPI path has run on Gadi.

The supplied snapshot-29 replay matches the previous commit `b4024054`
bit-for-bit in all 18 output datasets at one, four and eight native threads.
The candidate counts and photon residual are unchanged. Mixed-spectrum unit
cases allow small libm/BLAS rounding differences within the existing account
tolerance; general bitwise equivalence to NumPy is not promised.

A sequential before/after replay in the same environment measured:

| Implementation | Worker threads | Snapshot wall time | Process CPU time |
| --- | ---: | ---: | ---: |
| Previous `b4024054` | 1 | 22.28 s | 22.27 s |
| Compiled kernels and candidate filter | 4 | 5.37 s | 5.78 s |

This pair gives a 4.15x speedup. Other new-code repeats ranged from about
5.1 to 9.0 seconds in this shared environment, so the paired result is not
a production runtime guarantee. The CPU/wall ratio remains about 1.08:
eliminating Python iteration is the main whole-frame gain, while individual
large row kernels run in parallel. A separate 32^3, nine-band candidate-mask
microbenchmark measured median 10.12 ms in NumPy, 1.87 ms in native one-thread
and 0.98 ms in native four-thread execution, with identical classifications.

The remaining L10 bottleneck is the per-field sphere filtering and weak-signal
fallback, not the now-compiled compact absorption loop. Cross-field batched
FFTs are a further optimization to investigate; they are not implemented by
this change. Large-grid transforms already use threaded SciPy FFTs. These
measurements do not establish 32-core scaling or the runtime of later,
strongly ionized snapshots.

## Snapshot 49 slow-progress diagnosis

The supplied snapshot-48 checkpoint and snapshot-49 request came from the
older `b4024054` run. They contain finite, nonnegative states and matching run
identities. Replaying that version on one local CPU completed the first
substep in 38.3 seconds: four complete catalogue scans took 26.3 seconds and
absorption took about 11 seconds. The uploaded log ends at substep start;
the supplied state does not reproduce a deadlock.

Two cells had reached `xHI=1e-5` while retaining 0.2281 and 0.09487 normalized
photons. Their HI counts were only `4.33e-5` and `1.79e-5`. Neutral helium
kept a target eligible, while 18 eV photons continued absorbing the diminishing
HI tail. The 409-cell radius-zero partial batch consequently used 9,973
midpoint steps. The existing photon activity floor cannot remove this work:
the remaining photon accounts are large, and the tiny quantity is HI.

For independent one-cell supports with no possible crossing, the existing
H-only equations have the terminal count solution
`absorbed = min(initial_HI, active_photons)`. Photon-limited cells exhaust
every active account regardless of its H opacity. For atom-limited cells,
equal active H opacities preserve account ratios and determine their individual
debits. The new exact path implements those two cases, retaining all unused
and initially inactive photon credits. It does not floor a species. At that
version, unequal opacities in atom-limited cells, any active He opacity,
overlapping supports and possible crossing events used the coupled solver. The final
photon, nucleus and electron audits apply to both paths.

The other bottleneck was weak-signal support recomputation for dense fields.
It now uses the same ordered, compensated sums in a C/OpenMP periodic-offset
kernel, without constructing a centre-by-support array. With this change
alone, the complete six-substep snapshot-49 replay decreased from 148.65 to
50.20 seconds and all 18 response datasets were bitwise unchanged. New progress
lines identify selection, absorption and recombination so a long substep no
longer hides which phase is active.

With the independent H endpoint enabled as well, the complete snapshot took
36.74 seconds locally. Its largest per-cell `xHI` difference from `34d6ccc9`
was `1.49e-14`; the largest species-count difference was `1.29e-13` in normalized
units. The remaining photon sum changed from `1.19e-12` to `2.16e-13` because
the exact endpoint absorbs the former numerical tails. These counts enter the
same absorption ledger, whose residual is `8.54e-15` relative to total emission.
There were 38 committed batches in both runs. The updated suite passed 99
tests with four native-parser cases skipped in this environment; four physical
regressions also passed. The native ABI is now version 2, requiring a fresh
build and prepared run. These local replays do not time a complete Gadi job.

## Snapshot 65 mixed-opacity convergence failure

The supplied `34d6ccc9` checkpoint is snapshot 64; its request is snapshot 65.
The checkpoint and response share identical state arrays and a consistent
photon ledger. The original failure is native status 4: a common-field update
exhausted its 20,000-step guard. The same failure was reproduced with
`611b9e1` at the first substep's 48th batch, containing 1,104 independent
radius-zero supports. Eight atom-limited cells contain both active 18 eV
primary photons and 13.6 eV recombination photons. Their unequal H opacities
prevented the earlier proportional-debit endpoint from being used.

For an independent H-only cell, let `u` be its integrated neutral exposure.
Each account obeys `P_g(u) = P_g(0) exp(-sigma_g u)`. An atom-limited endpoint
therefore solves the monotone scalar equation

```
sum_g P_g(0) * [-expm1(-sigma_g*u)] = initial_HI.
```

The solver brackets this root and uses safeguarded Newton steps. Each account
is debited by its own capture, including the small recombination contribution.
Photon-limited cells still spend all initially active accounts. He-active or
overlapping supports and possible crossing events retain the coupled solver.
Unrepresentable exposure scales return to that solver; no photon band is
silently removed. This is a terminal solution of the existing absorption
system, with unchanged ionization thresholds and species/count audits.

Repeated large-radius scans are also screened during a fixed-source
illumination stage. For each target (HI, HeI and HeI+HeII), the code stores a
lower bound on the minimum photon deficit over all centres at that radius.
Accepted updates reduce this bound by an upper bound on the actual target
loss, nucleus-sum increase and photon-account increase. A radius can be
skipped only while all three deficits remain positive after roundoff guards.
The retained bounds require only three values per radius. They are discarded
at every illumination exit, including exceptions, and rebuilt after source
injection, density remapping or recombination. Skipped radii are evaluated
when needed to identify the smallest partially absorbing support.

A full eight-substep replay of the uploaded snapshot-65 request passes with
both the H endpoint fix alone and with radius screening enabled. On the same
local two-thread setup:

| Implementation | Wall time | Candidate checks | Committed updates |
| --- | ---: | ---: | ---: |
| Mixed-opacity endpoint, full radius scans | 550.60 s | 678,526,976 | 567 |
| Mixed-opacity endpoint and radius screening | 250.85 s | 184,647,680 | 567 |

All 18 response datasets are bitwise identical between these runs, as are
all 567 ordered update events. Screening skipped 15,072 radius scans and
reduced candidate checks by 72.79%. The relative photon ledger residual is
`5.2064e-15`; volume- and mass-weighted xHI are `0.9582474788` and
`0.9141187953`. These are local worker replays, not full Gadi job timings.
The original `611b9e1` version aborts on this input, so its complete-frame
runtime and final state are not comparison baselines.

Validation passed 119 Python tests, with four full-Meraxes parameter-parser
cases skipped locally, plus four physics regression cases. The new tests
include the captured mixed-opacity tail, analytic spectral-partition checks,
underflow/overflow fallback, H/He event equivalence, partial-radius retries
and screening invalidation after new sources or exceptions. Radius screening
is enabled by default. Its implementation is included in the frozen reference
hash; the native kernel ABI remains 2. Pull, rebuild and prepare a fresh run.
