# ES-HHe L10 run on Gadi

`Flag_PhotonConservation=3` runs the optimized ES-HHe reference during Meraxes
evolution through its C bridge. The spatial model uses periodic excursion-set
spheres and a shared photon ledger. This mode does not use ray packets or solve
the spatial radiation-transfer equation. Meraxes remains MPI-parallel; the
Python ES-HHe solver runs on rank 0. This integration does not establish
large-grid MPI scaling.

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

The PBS template requests 32 ranks, 320 GB and 48 hours on `hugemem`. The rank
count fits the 32 input-grid slabs. NCI's [Queue Limits](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits)
page (checked 2026-10-18) allows 1–48 cores and 48 hours for a single-node
`hugemem` job. The Python worker is single-process; increasing MPI ranks does
not parallelize that solver.

## Integration validation

The full CPU Meraxes executable was built with GCC 13.3, Open MPI 4.1.6,
FFTW 3.3.10, GSL 2.7.1 and HDF5 1.10.10. The online adapter passed synthetic
consecutive-snapshot tests at 1, 2 and 4 ranks, including empty slabs, physical
source/time units, H/He recombination, density remapping and photon/electron/
nucleus accounting. A two-event star-formation regression verifies that mode 3
adds each event's new SFR once. The worker has nine integration tests; nineteen
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
