#!/usr/bin/env bash
# Run on a Gadi login node after updating the Speed-up checkout.
# Network operations happen here; compilation and simulation run in PBS.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
command -v qsub >/dev/null || { echo 'Run this script on a Gadi login node.' >&2; exit 1; }
if ! git diff --quiet HEAD --; then
    echo 'The checkout has tracked modifications. Commit them before freezing the run source.' >&2
    exit 1
fi
revision=$(git rev-parse HEAD)
input="$repo/input/params/runs/gadi_fiducial_reference.par"
baseline_build=${SPEEDUP_BASELINE_BUILD:-/g/data/qp00/sz4640/meraxes-devs/build_stochastic}
output_root=${SPEEDUP_OUTPUT_ROOT:-/g/data/qp00/sz4640/Results/Genesis_L210_N4320/Fiducial_0.0propscale_Speed-up}
mkdir -p "$output_root"
run_dir=$(mktemp -d "$output_root/run_$(date +%Y%m%d_%H%M%S)_${revision:0:8}_XXXXXX")
mkdir "$run_dir/provenance" "$run_dir/output"
printf '%s\n' "$revision" > "$run_dir/provenance/commit.txt"
cp "$input" "$run_dir/provenance/input_requested.par"

# Recover the previous run parameters at full precision and match them to the
# supplied log before allocating the production job.
python_bin=${SPEEDUP_PYTHON:-/g/data/qp00/sz4640/venvs/tauefix/bin/python3}
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
"$python_bin" - "$repo" "$run_dir" "$baseline_build" <<'PY'
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import shutil
import sys
try:
    from dragons.meraxes import io as mio
except ImportError as error:
    raise SystemExit('Use a Python environment with DRAGONS: set SPEEDUP_PYTHON to its python3. ' + str(error))

repo, run, baseline = map(Path, sys.argv[1:])

def params(path):
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if ':' not in line:
            raise SystemExit('{}:{}: invalid parameter line'.format(path, number))
        key, value = (part.strip() for part in line.split(':', 1))
        if key in result:
            raise SystemExit('{}: duplicate parameter {}'.format(path, key))
        result[key] = value
    return result

logged = params(run / 'provenance/input_requested.par')
known = set(re.findall(r'(?:strncpy|strcpy)\(params_tag\[n_param\],\s*"([^"]+)"',
                       (repo / 'src/core/read_params.c').read_text()))
baseline_output = Path(logged['OutputDir']) / (logged['FileNameGalaxies'] + '.hdf5')
if not baseline_output.is_file():
    raise SystemExit('Original fiducial output is required for full-precision parameters: ' + str(baseline_output))

def as_text(value):
    if getattr(value, 'size', None) == 1:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8').rstrip('\0')
    return str(value)

values = {key: as_text(value) for key, value in mio.read_input_params(str(baseline_output), raw=True).items()}
unknown = sorted(set(values) - known)
if unknown:
    raise SystemExit('The original output contains unsupported parameters: ' + ', '.join(unknown))
# The log prints numbers with %g (six significant figures). Check identity at
# that precision, then retain every digit stored in the original HDF5 output.
differences = []
for key, expected in logged.items():
    actual = values.get(key)
    if actual is None:
        differences.append(key + ' is absent from the original output')
        continue
    try:
        match = (Decimal(actual) == Decimal(expected) or
                 Decimal(format(float(actual), '.6g')) == Decimal(expected))
    except (ValueError, InvalidOperation):
        match = (Path(actual) == Path(expected) if expected.startswith('/')
                 else actual.strip() == expected.strip())
    if not match:
        differences.append('{}: log={!r}, HDF5={!r}'.format(key, expected, actual))
if differences:
    raise SystemExit('The fiducial HDF5 does not match job 179580303:\n' + '\n'.join(differences))
(run / 'provenance/baseline_parameters.par').write_text(
    ''.join('{:30s} : {}\n'.format(k, v) for k, v in values.items()))
(run / 'provenance/baseline_output.txt').write_text(str(baseline_output) + '\n')
requested = dict(values)
for key in ['SimulationDir', 'CoolingFuncsDir', 'StellarFeedbackDir', 'RecombinationDir']:
    if not values.get(key) or not Path(values[key]).is_dir():
        raise SystemExit('Missing {}: {}'.format(key, values.get(key, '(unset)')))
for key, filename in [('CoolingFuncsDir', 'SD93.hdf5'),
                      ('StellarFeedbackDir', 'stellar_feedback_tables.hdf5')]:
    if not (Path(values[key]) / filename).is_file():
        raise SystemExit('Missing table: ' + str(Path(values[key]) / filename))
if int(values.get('Flag_IncludeSpinTemp', '0')):
    if not Path(values['TablesForXHeatingDir']).is_dir():
        raise SystemExit('Missing TablesForXHeatingDir')

# Freeze parameters from the completed run, rather than today's mutable defaults.
controls = {'DefaultsFile', 'SimParamsFile', 'OutputDir'}
(run / 'provenance/defaults.par').write_text(
    ''.join('{:30s} : {}\n'.format(k, v) for k, v in values.items() if k not in controls))
simulation_keys = ['SimName', 'SimulationDir', 'TreesID', 'CatalogFilePrefix', 'BoxSize',
                   'Hubble_h', 'OmegaM', 'OmegaK', 'OmegaR', 'OmegaLambda', 'BaryonFrac',
                   'PartMass', 'NPart', 'Sigma8', 'SpectralIndex']
(run / 'provenance/simulation.par').write_text(
    ''.join('{:30s} : {}\n'.format(k, values[k]) for k in simulation_keys))
requested['DefaultsFile'] = str(run / 'provenance/defaults.par')
requested['SimParamsFile'] = str(run / 'provenance/simulation.par')
requested['OutputDir'] = str(run / 'output')
(run / 'input.par').write_text(''.join('{:30s} : {}\n'.format(k, v) for k, v in requested.items()))
print('Matched all {} logged parameters; retained full precision from the baseline HDF5.'.format(len(logged)))

cache = {}
cache_file = baseline / 'CMakeCache.txt'
if cache_file.is_file():
    shutil.copy2(cache_file, run / 'provenance/baseline_CMakeCache.txt')
    for line in cache_file.read_text().splitlines():
        match = re.match(r'([^:#/][^:]*):[^=]+=(.*)$', line)
        if match:
            cache[match.group(1)] = match.group(2)
    print('Build options inherited from', cache_file)
else:
    raise SystemExit('Missing original build cache: ' + str(cache_file))
settings = {'CMAKE_BUILD_TYPE': 'RelWithDebInfo', 'USE_STOCHASTICITY': 'ON',
            'USE_MINI_HALOS': 'OFF', 'CALC_MAGS': 'OFF', 'USE_JWST': 'OFF',
            'USE_HST': 'OFF', 'N_HISTORY_SNAPS': '17', 'MAGS_N_SNAPS': '12', 'MAGS_N_BANDS': '11'}
settings.update({key: cache[key] for key in settings if key in cache})
# The requested run has no minihalo physics. Keep source prescriptions available.
settings.update(USE_STOCHASTICITY='ON', USE_MINI_HALOS='OFF', USE_CUDA='OFF', BUILD_TESTS='OFF', BUILD_SHARED_LIBS='OFF')
for key in ['CMAKE_C_FLAGS', 'CMAKE_C_FLAGS_RELEASE', 'CMAKE_C_FLAGS_RELWITHDEBINFO']:
    if cache.get(key):
        settings[key] = cache[key]
if settings['CALC_MAGS'].upper() in ['ON', 'TRUE', '1']:
    sector = Path(cache.get('SECTOR_ROOT', str(baseline.parent / 'src/sector')))
    if not (sector / 'sector.h').is_file():
        raise SystemExit('Photometry is enabled but SECTOR_ROOT is unavailable: ' + str(sector))
    if not Path(values['PhotometricTablesDir']).is_dir():
        raise SystemExit('Missing PhotometricTablesDir')
    shutil.copytree(sector, run / 'sector')
    settings['SECTOR_ROOT'] = str(run / 'sector')
(run / 'provenance/cmake-options.txt').write_text(''.join('-D{}={}\n'.format(k, v) for k, v in settings.items()))
print('Flag_IncludeSpinTemp =', values.get('Flag_IncludeSpinTemp', '0'))
print('CALC_MAGS =', settings['CALC_MAGS'], '; CMAKE_BUILD_TYPE =', settings['CMAKE_BUILD_TYPE'])
PY

GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$run_dir/source" "$revision"
git -C "$run_dir/source" submodule update --init --recursive
# Freeze the job body too, so a later pull does not change a queued run.
{
    printf '#!/bin/bash\nset -euo pipefail\n'
    printf 'run_dir=%q\n' "$run_dir"
    cat <<'PBS'
cd "$run_dir"
exec > "$run_dir/pbs_${PBS_JOBID}.out" 2> "$run_dir/pbs_${PBS_JOBID}.err"
module --force purge
module load gcc/12.2.0 openmpi/4.1.4 fftw3/3.3.10-nci1 hdf5/1.12.2p gsl/2.7.1 cmake/3.24.2
export OMP_NUM_THREADS=1
mapfile -t options < provenance/cmake-options.txt
cmake -S source -B build \
    -DCMAKE_C_COMPILER=mpicc \
    -DHDF5_ROOT=/apps/hdf5/1.12.2p -DHDF5_PREFER_PARALLEL=ON \
    -DFFTW_ROOT=/apps/fftw3/3.3.10-nci1 "${options[@]}" > build.log 2>&1
cmake --build build --parallel 8 >> build.log 2>&1
cp build/CMakeCache.txt provenance/run_CMakeCache.txt
cp build/bin/meraxes meraxes
sha256sum meraxes > provenance/executable.sha256
module unload cmake/3.24.2
module list > provenance/modules.txt 2>&1
ldd meraxes > provenance/linked_libraries.txt
printf 'Job ID: %s\nHost: %s\nNCPUS: %s\n' "$PBS_JOBID" "$(hostname)" "$PBS_NCPUS"
printf 'Starting Meraxes: %s ranks; output=%s/output\n' "$PBS_NCPUS" "$run_dir"
date -Is > started.txt
set +e
/usr/bin/time -v -o timing.txt mpirun -np "$PBS_NCPUS" ./meraxes input.par > run.log 2>&1
status=$?
set -e
printf '%s\n' "$status" > exit_status.txt
date -Is > finished.txt
exit "$status"
PBS
} > "$run_dir/submit.pbs"
cd "$run_dir"
job_id=$(qsub -P qp00 -q hugemem -N meraxes_stochastic \
    -l ncpus=32 -l mem=320GB -l walltime=12:00:00 \
    -l storage=gdata/qp00+gdata/cm25+scratch/qp00 -l wd submit.pbs)
printf '%s\n' "$job_id" > job.id
printf 'Submitted: %s\nRun directory: %s\nOutput: %s/output\n' "$job_id" "$run_dir" "$run_dir"
printf 'Monitor: qstat -swx %s\nLog: tail -f %q/run.log\n' "$job_id" "$run_dir"
