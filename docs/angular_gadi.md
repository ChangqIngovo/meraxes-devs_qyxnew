# Meraxes angular-face simulation on Gadi

This launches **Meraxes on the L10_N512 merger trees and density grids**, with
`Flag_PhotonConservation=4`. It is separate from the standalone sphere and
contact benchmarks. The checked-out simulation file specifies a 10 cMpc/h box
and a 32³ grid; `a_list.txt` determines the snapshot count and redshift range
(the supplied test set has 76 snapshots, approximately z=30 to z=10).

The run uses the current angular-face geometry, eight angular bins per source
cell (`AngularLevel=0`), bounded ES modulation with epsilon=0.1, and a maximum
physical step of 0.1 Myr. It uses periodic boundaries and physical light speed.
The default stellar source spectrum is monochromatic 18 eV, so direct stellar
photons ionize H but cannot ionize He. This is an explicit source assumption;
the H/He chemical solver supports a supplied photon-number spectrum.

## Prepare and submit

Keep the existing working tree, including its uncommitted files. Fetch the
**fork** remote and build a separate detached worktree at the selected commit:

```bash
set -euo pipefail
cd /g/data/qp00/sz4640/meraxes-devs
git fetch fork Photon_conservation
angular_rev=$(git rev-parse fork/Photon_conservation)
angular_tag=$(date -u +%Y%m%dT%H%M%S)
angular_src=/g/data/qp00/sz4640/angular_builds/${angular_rev:0:8}_${angular_tag}
mkdir -p /g/data/qp00/sz4640/angular_builds
git worktree add --detach "$angular_src" "$angular_rev"
git -C "$angular_src" submodule update --init --recursive
cd "$angular_src"

export ANGULAR_PYTHON=/g/data/qp00/sz4640/venvs/tauefix/bin/python
bash tools/gadi_angular_build.sh

# The build script runs in a child shell. Load runtime modules here as well.
module --force purge
module load gcc/12.2.0 openmpi/4.1.4 fftw3/3.3.10-nci1 hdf5/1.12.2p gsl/2.7.1 python3/3.11.7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_DIR=/g/data/qp00/sz4640/angular_runs/l10_${angular_rev:0:8}_${angular_tag}
"$ANGULAR_PYTHON" tools/gadi_angular_l10.py
qsub -v RUN_DIR="$RUN_DIR" "$RUN_DIR/submit.pbs"
```

The build runs the angular unit/integration tests before creating a build
manifest. Preparation reads every density-grid header, checks all required
grid parts and the tree catalogue, validates the parameter chain using the
actual executable, and copies the executable, worker, native library and
parameters into a fresh run directory. The PBS job checks those hashes again.
The original working tree is not reset or merged.

Default external inputs:

| Input | Location |
| --- | --- |
| L10 trees, grids and scale factors | `/g/data/cm25/yq5547/dragons/input_trees/Maynooth/LR_trees` |
| Existing input tables | `/g/data/qp00/sz4640/meraxes-devs/input` |
| Python with NumPy and h5py | `/g/data/qp00/sz4640/venvs/tauefix/bin/python` |

Override these through `L10_SIMULATION_DIR`, `L10_TABLES`, and
`ANGULAR_PYTHON` before preparation. Optional settings are
`ANGULAR_MAX_STEP_MYR`, `ANGULAR_LEVEL`, `ANGULAR_EPSILON`,
`ANGULAR_SPEED_FRACTION` and `ANGULAR_SPECTRUM`. A spectrum file has two columns:
energy in eV and nonnegative photon-number weight; at least one weight must be
positive. Preparation accepts energies in [13.6, 1000] eV and copies the file
into the run. A non-default spectrum changes the physical source assumption.

The initial PBS request is four MPI ranks, 32 GB and 20 hours on `normal`.
This is an initial allocation, not a measured runtime or memory prediction.
Meraxes galaxy/grid work uses MPI; the Python angular worker is serial.
Additional MPI ranks do not accelerate its photon transport. The worker can
stop explicitly on resource limits; photon inventories are never dropped to
make a run fit.

## Outputs

For the run directory printed by preparation:

| File | Meaning |
| --- | --- |
| `run_JOBID.log` | Meraxes stdout/stderr and worker diagnostics |
| `output/` | Meraxes galaxy/grid products and angular state files |
| `output/angular_state_NNN.hdf5` | Per-snapshot H/He response and diagnostics |
| `output/.angular_checkpoint.hdf5` | Worker state, including in-flight photons |
| `run_manifest.json`, `provenance/` | Pinned executable, code, parameters and input hashes |
| `COMPLETED` | Created only after `mpirun` returns success |
| `FAILED` | Exit status if the PBS shell fails |

```bash
qstat -u "$USER"
tail -n 80 "$RUN_DIR"/run_*.log
ls -lh "$RUN_DIR/output"
test -f "$RUN_DIR/COMPLETED" && echo 'Meraxes completed successfully'
```

A vanished queue entry alone does not establish success. Inspect the log and
`COMPLETED`. The snapshot worker supports consecutive updates and replay of an
identical request; this does **not** implement a full Meraxes process restart.
The bridge therefore requires a fresh output directory when starting a new
Meraxes process. Do not delete or reuse an earlier run to bypass this guard.

## Physical and numerical scope

Photon inventory retains source cell, angular bin, energy, unwrapped periodic
image and remaining edge flight distance. Gas in periodic images shares the
same physical cell. Photon travel uses increments of source-centered radial
distance. New source photons are pulsed at substep boundaries; photons that
have just crossed a newly transparent cell continue in a later substep. This
production time splitting is **first order** and should be checked by reducing
`AngularMaxStepMyr` when interpreting a simulation result. The earlier
irreversible, open-boundary solver's global time-sandwich result is not a bound
for the periodic, recombining worker.

H/He recombinations use an isothermal 10,000 K case-A closure with local
on-the-spot diffuse reabsorption. Unabsorbed diffuse photons remain in their
physical cell and channel. Diffuse photons do not undergo spatial transport.
Density changes preserve species fractions and are recorded separately as
matter exchange; they do not consume stellar photon inventory.

The simulation evolves galaxies and feeds their escaped stellar photons into
the new ionization solver. Its calibrated UVB, heating and radiative feedback
closures have not been supplied, so the launch configuration disables spin
temperature, 21-cm products, UVB feedback, reionization infall suppression,
CGM escape suppression, AGN ionizing photons and Pop III. This run tests the
integrated ionization history under those explicit assumptions. It does not
establish a calibrated reionization-feedback model.

Full MPI/HDF5 compilation and the L10 data run must be verified on Gadi; local
standalone and snapshot tests do not substitute for that run.

The worker equations, inventories and accounting conventions are documented
in [PERIODIC.md](../tools/es_angular/PERIODIC.md).
