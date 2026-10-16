# MPI execution and Gadi performance measurements

The Gadi job scripts request **48 CPU cores, 1000 GB total memory and 20 hours**
on `hugemem`. They retain the full quadratic shape refinement. The supplied
runner changes only the output directory, PC enable/shape flags, diagnostic
level and the PC memory guard by default. It preserves the input's grid size,
snapshots, source prescriptions, recombinations, tolerances and solver work
limits. An explicit `PC_MAX_EDGES` override changes the stored-edge resource
limit only.

## What is accelerated

The CPU geometry search uses an exact periodic bounding-box test on its
spatial tree. This rejects tree branches that cannot intersect the query
sphere while retaining the original leaf ordering and accepted neighbours.
Repeated max-flow calls are also skipped when a sink-capacity stage is
unchanged. These CPU improvements apply with or without MPI geometry.

The MPI service distributes independent target queries against a frozen
source tree. Each worker uses the same query kernel; rank zero gathers the
results in the original target and neighbour order. Workers cannot spend
photons or change residual capacities. The global maximum-flow solver,
quadratic master optimization and final physical/conservation certificates
remain on rank zero. This is partial parallelization of the existing
allocator, not a new radiation equation or a change to the PC physical model.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PC_MPI_GEOMETRY` | `1` | Enable geometry workers; `0` retains serial geometry at the same Meraxes rank count. |
| `PC_MPI_MIN_TARGETS` | `16384` | Minimum target count before considering MPI for a query context. |
| `PC_MPI_WORKER_MB` | `1024` | Geometry-service allocation limit per worker, in MiB, including its replicated tree. |

Small contexts and contexts whose replicated tree/buffers exceed the worker
limit fall back to serial queries. The batch size can shrink to fit the
worker limit. The limit excludes other Meraxes/MPI allocations and is
separate from `PhotonConservationMemoryMB`; account for both on every node.
Increasing it on 48 ranks can add a substantial aggregate memory allowance.

Each snapshot logs `PC MPI snapshot=...` followed by service ranks, accepted
and declined contexts, query batches, target count, setup/query seconds,
accounted peak worker bytes, worker limit and target threshold. The scalar
HDF5 diagnostics also contain `geometry_mpi_*` attributes. Enabling the
service alone does not establish that it ran: check for nonzero query batches.
Tree setup, communication and the remaining serial work can offset parallel
query savings on a particular problem.

## Resource choices

NCI requires a multiple of 48 cores for a `hugemem` job spanning multiple
nodes. A request for 64 cores is not a standard accepted allocation. The
relevant alternatives are:

| CPU cores | Total memory | Approximate memory per node | Requested walltime | Default PC guard |
| --- | --- | --- | --- | --- |
| 48 | 1000 GB | 1000 GB on one node | 20 h | 700,000 MiB |
| 96 | 1000 GB | 500 GB on each of two nodes | 20 h | 350,000 MiB |

The first option leaves more memory available on the node holding the global
allocation graph. More nodes do not form one shared memory pool. The PC guard
is a solver accounting limit, not a measured peak RSS or a whole-job memory
limit. Other Meraxes arrays, MPI and worker buffers also consume memory.
The defaults leave headroom; they are not a guarantee that a particular L210
calculation will fit. Check actual per-node memory usage after the pilot.

The [NCI queue limits](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits)
permit up to 1470 GB per `hugemem` node, with 48 h at 48 cores, 24 h at 96
cores and 5 h at 144 or 192 cores. The [queue structure](https://opus.nci.org.au/spaces/Help/pages/90308824/Queue+Structure)
describes 48-core nodes and a standard job maximum of 192 cores. These were
checked when preparing the scripts; consult the live limits before changing
the allocation.

## Update and build

From the existing Gadi checkout:

```sh
cd /g/data/qp00/sz4640/meraxes_photon_conservation
git switch Photon_conservation
git pull --ff-only origin Photon_conservation
git submodule update --init --recursive
bash tools/gadi_pc_build.sh
```

The build script refuses a different branch or modified tracked source. It
loads `pc_env.sh` when available, otherwise the documented Gadi modules, and
builds in `build-pc-mpi`. It uses GCC's Cascade Lake target and `-O3` with
debug symbols. The PC sources retain the repository's ordered floating-point
and finite-check compilation settings. It runs the standalone allocation,
ES completion and shape tests. It does not submit jobs. MPI and full-input
tests require the matching compute environment. Both PBS scripts first run
the MPI test at 1, 2 and 4 ranks and stop on a failure or missing test before
launching the simulation. These tests run inside the allocation, not on a
login node.

The manifest records the commit, submodules, binary digest, compiler version,
loaded modules, linked libraries and CMake cache digest. `PC_BUILD_DIR`,
`PC_ENV_FILE` and `PC_BUILD_JOBS` are
optional overrides. The default two compiler workers avoid a large login-node
build. Additional arguments are passed to CMake; changing compile-time source
physics must match the production model deliberately.

## Submit a production input

```sh
cd /g/data/qp00/sz4640/meraxes_photon_conservation
qsub -v PC_INPUT=/g/data/qp00/sz4640/Results/Genesis_L210_N4320/Fiducial_0.0propscale/input.par \
    tools/gadi_pc_hugemem.pbs
```

`PC_INPUT` must be a complete input, not the small PC parameter fragment.
Its `DefaultsFile` must be explicit. Relative paths are evaluated from the
input file's directory, or from `PC_INPUT_CWD` if the original run used a
different working directory. The original input and included parameter files
are copied into a fresh run directory, together with the exact binary and
build manifest. The simulation runs with that original working directory so
relative table and catalogue paths retain their meaning. Output goes into
the new directory; previous simulation outputs are not reused as a checkpoint.

The default output parent is `/g/data/qp00/sz4640/pc_runs/mpi`. Override it with
`PC_RUN_ROOT`. `run.json` records provenance, MPI command, elapsed wall time
and each snapshot's PC, completion and refinement times. Detailed standard
output and errors go to `run.log`. `geometry_mpi_snapshots` and `geometry_mpi`
record the per-snapshot and aggregate MPI diagnostics; `geometry_mpi_used`
is true only if the log reports at least one executed MPI query batch.
`geometry_mpi_diagnostics_present=false` means the evidence is absent, not
that MPI was measured to be inactive.

The job uses one thread per MPI rank and core binding. `PC_DEBUG=1` writes
scalar PC diagnostics and avoids the much larger per-cell/edge output.
Use `PC_DEBUG=2` for a manageable validation run when independent inventory
and geometric certificate checks are required. Diagnostic level does not
disable the solver's native acceptance checks. `PC_MEMORY_MB` overrides the
guard; `PC_RANKS` can use fewer ranks than allocated cores.

The supplied production input may inherit a stored-edge limit of 20 million.
This is a resource ceiling, not a prediction of the graph size needed at
N512. An edge-limit failure must be investigated before retrying. If a larger
ceiling is justified and fits the memory budget, pass `PC_MAX_EDGES` through
`qsub -v` (or `--max-edges` to the Python runner); omitting it preserves the
input/default. Raising it does not remove integer index limits or guarantee
convergence. The exact override is recorded in `run.json`.

To evaluate 96 cores while retaining a total memory request of 1000 GB:

```sh
qsub -l ncpus=96,mem=1000GB,walltime=20:00:00 \
    -v PC_INPUT=/absolute/path/to/complete/input.par \
    tools/gadi_pc_hugemem.pbs
```

For that allocation the script automatically halves its default PC guard to
350,000 MiB. The request must still fit each node. A lower MPI rank count
does not increase the memory PBS assigns to a node.

Check that all required density snapshots exist before a long PC run. In
particular, enabling PC can require early density fields that a previous ES
run skipped. The scripts do not replace absent snapshots, resample a grid,
relax convergence or alter the source history to make a run proceed.
More RAM also does not remove the allocator's existing integer index limits.
There is no demonstrated guarantee that the complete L210 N512 PC evolution
will finish within 20 h; establish this from a representative pilot.

## Measure the MPI benefit at fixed Meraxes rank count

Use a complete, tractable pilot input with the same physics and grid for
every run. The default scaling job compares `PC_MPI_GEOMETRY=0` and `1`
at the same 48 Meraxes ranks, with two repetitions of each. It reverses pair
order on the second repetition. This keeps upstream galaxy/grid MPI work
fixed and isolates the benefit of the geometry worker service. Both cases
include the new CPU search improvements; this is not a comparison with the
old commit's unoptimized CPU implementation.

```sh
qsub -v PC_INPUT=/absolute/path/to/pilot/input.par tools/gadi_pc_scaling.pbs
```

Use `PC_RANKS` to change the fixed rank count and `PC_REPEATS` to change the
number of pairs. Keep the worker threshold/memory settings identical across
pairs. If all query batches are zero, the comparison measures the fallback
and service overhead, not parallel query speedup.

For a separate whole-application strong-scaling experiment, select
`PC_SCALING_MODE=ranks`. Its default list is 1, 2, 4, 8, 16, 32 and 48 ranks;
`PC_RANK_LIST` overrides it. Choose a pilot whose one-rank run finishes well
inside the queue walltime. For a space-separated list, export it before using
PBS's environment-name form:

```sh
export PC_INPUT=/absolute/path/to/pilot/input.par
export PC_SCALING_MODE=ranks
export PC_RANK_LIST='1 4 16 48'
export PC_REPEATS=3
qsub -v PC_INPUT,PC_SCALING_MODE,PC_RANK_LIST,PC_REPEATS tools/gadi_pc_scaling.pbs
```

Each run starts a new executable process with a fresh output directory.
Filesystem and operating-system caches can be warm on subsequent runs;
the scripts neither flush them nor claim cold-I/O timing. Compare median
PC/refinement times separately from whole-run elapsed times. Changing
Meraxes rank count also changes its upstream parallel reductions; compare
the resulting source inventories, neutral fractions and certificates before
interpreting a timing difference as performance of the same problem.

For the fixed-rank pair report `T(geometry=0)/T(geometry=1)`, separately for
the complete PC phase and the whole simulation. For rank scaling with a
common workload, report `T(reference)/T(ranks)` and efficiency relative to
the reference as `speedup/(ranks/reference_ranks)`.
Use the same completed snapshot set and identical convergence settings.
Do not count a failed or truncated run as a faster result. Rank counts alone
are not evidence of acceleration: global flow/certification work and MPI
communication can limit scaling.

## Validation and measured limits of this change

Native MPICH runs at 1, 2 and 4 ranks pass the new regression suite. The
distributed cases exercise 231 accepted tree generations and 675 query
batches, including a final batch smaller than the worker count. They compare
ordered neighbours against the original three-image periodic formula, and
compare allocation fields, source ledgers, cut masks, edge visitation, ES
decisions and shape certificates bit for bit with the scalar path. Uniform
and asymmetric worker memory failures, repeated contexts, small batches and
an injected query failure are covered. AddressSanitizer and UBSan runs at
1 and 2 ranks also pass. The existing native suites pass, as do 200 independent
explicit-edge LP problems and 100 independent quadratic-shape comparisons.
The full Meraxes/Gadi production evolution has not been benchmarked here.

The deterministic synthetic driver is
`tools/benchmark_photon_conservation_shape.c`. Local measurements used GCC
13.3.0, `-O2`, an eight-CPU shared container and the original branch commit
`30b01044541f5014369a7261416c2d366b869e26` as the baseline:

| Synthetic case | Original serial | Optimized MPI1 | MPI2 | MPI4 |
| --- | ---: | ---: | ---: | ---: |
| Patchy 12 cubed, 4 priority groups | 0.7314 s | 0.3434 s | 0.3025 s | 0.3732 s |
| Bottleneck 16 cubed, 1 priority group | 19.3828 s | 13.6014 s | 13.6399 s | 13.8073 s |

Values are median shape-refinement times over three runs, except the original
16-cubed baseline, which is one measurement. The driver forced
`--mpi-min-targets 1` to exercise MPI; these small cases would normally use
the production scalar fallback. Every reported run succeeded and produced
the same binary64 target field as the original implementation. Field SHA256:

- Patchy: `ec333ff84888af1a2163554e8d3e4bec9126d779591c26e0a0363cd1bf93b591`
- Bottleneck: `379988741d17eb92228ebc5d61b815d5118dce89b673e18a9907940b72333b7f`

These measurements establish a CPU improvement, not reliable MPI scaling
for the full solver. In the bottleneck case, 2-to-4-rank query time fell from
about 0.524 to 0.373 s, while tree setup rose from 0.093 to 0.185 s; serial
residual-network work dominates overall time. Increasing rank count alone
does not accelerate that work. No L210 runtime or speedup is inferred from
these synthetic cases.

Build the small MPI benchmark independently of Meraxes dependencies:

```sh
mpicc -O2 -std=c99 -Wall -Wextra -fno-fast-math -ffp-contract=off \
    -DPC_BENCHMARK_MPI -Isrc/physics \
    tools/benchmark_photon_conservation_shape.c \
    src/physics/photon_conservation.c src/physics/photon_conservation_shape.c \
    src/physics/photon_conservation_geometry.c src/core/photon_conservation_mpi.c \
    -lm -o benchmark_pc_mpi
```

Inside a compute allocation, for example:

```sh
mpirun -np 4 ./benchmark_pc_mpi --n 16 --groups 1 --scenario bottleneck \
    --repeat 3 --mpi-min-targets 1 --output field.bin
```
