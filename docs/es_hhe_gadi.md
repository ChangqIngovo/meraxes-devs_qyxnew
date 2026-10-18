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

The input inherits the existing L210 template's galaxy/source parameters,
replaces all simulation-dependent parameters with the L10 parameters, and
reuses `/g/data/qp00/sz4640/meraxes-devs/input/` table directories. The old
Sobacchi reionization feedback, UVB/spin-temperature closure, legacy 21-cm
brightness-temperature/power-spectrum outputs and black-hole
ionizing source are disabled in mode 3. H/He recombination is enabled in the
new solver, without the legacy `Nrec` correction. Mode 3 skips the legacy MHR
table initialization and does not read `RecombinationDir`; this directory need
not exist. Its input parameter is retained, and H/He recombination remains
enabled. The temperature is fixed at
10,000 K. These choices define a controlled ES-HHe test, not an already
recalibrated galaxy-feedback model.

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
set -euo pipefail
cd /g/data/qp00/sz4640/meraxes_photon_conservation
git switch Photon_conservation
git pull --ff-only origin Photon_conservation
git submodule update --init --recursive

module load python3/3.11.7
export ES_PYTHON=/g/data/qp00/sz4640/venvs/meraxes-es/bin/python
if [[ ! -x "$ES_PYTHON" ]]; then
    python3 -m venv --system-site-packages "$(dirname "$(dirname "$ES_PYTHON")")"
fi
"$ES_PYTHON" -m pip install numpy scipy h5py

bash tools/gadi_es_l10.sh build
export RUN_DIR=/g/data/qp00/sz4640/pc_runs/l10_es_hhe_$(git rev-parse --short=8 HEAD)
bash tools/gadi_es_l10.sh prepare
qsub -v "RUN_DIR=$RUN_DIR" "$RUN_DIR/submit.pbs"
```

The prepare step does not submit a job. It copies the built binary, Python
solver, input/default/simulation files, snapshot list, optional spectrum and
build manifest into a new run directory, and records their SHA-256 hashes.
The job checks these files before running. A pre-existing run directory is
rejected; choose a new `RUN_DIR` for another run. The existing ES runs are
untouched.

Environment overrides before `prepare`:

| Variable | Default / meaning |
|---|---|
| `L10_INPUT` | Existing `Results/Genesis_L210_N4320/Fiducial_0.0propscale/input.par` |
| `L10_SIMULATION_DIR` | `SimulationDir` in `Genesis_L10_N512.par` |
| `L10_TABLES` | `/g/data/qp00/sz4640/meraxes-devs/input` |
| `ES_PYTHON` | `/g/data/qp00/sz4640/venvs/meraxes-es/bin/python` |
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
adds each event's new SFR once. The worker has nine integration tests; thirteen
runner tests cover full-snapshot configuration, provenance, PBS launch, shell
environment isolation and preparation without unused legacy recombination tables.
These tests do not require or replace a full L10 merger-tree run.

A dilute 32^3 synthetic snapshot with one 18 eV source completed in 0.62 s
with approximately 85 MiB process peak RSS. It exercised the full periodic
radius catalogue and retained a zero photon-count residual. This is a
single-step smoke test, not a timing estimate for dense or late-time snapshots.
Gadi execution and the complete L10 history have not been tested here.
