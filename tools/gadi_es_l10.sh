#!/usr/bin/env bash
# Prepare a new ES-HHe L10 run; submission is an explicit, separate qsub.
set -euo pipefail
es_repo=${PC_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
es_build=${PC_BUILD_DIR:-$es_repo/build-pc-mpi}
es_action=${1:-}
case "$es_action" in
    build)
        shift
        exec bash "$es_repo/tools/gadi_pc_build.sh" -DUSE_MINI_HALOS=OFF -DCALC_MAGS=OFF "$@"
        ;;
    prepare) ;;
    *) echo 'Usage: bash tools/gadi_es_l10.sh {build|prepare}' >&2; exit 2 ;;
esac

export PC_REPO="$es_repo" PC_BUILD_DIR="$es_build"
export ES_PYTHON=${ES_PYTHON:-/g/data/qp00/sz4640/venvs/meraxes-es/bin/python}
test -x "$ES_PYTHON" || {
    echo "ES_PYTHON is not executable: $ES_PYTHON (see docs/es_hhe_gadi.md)" >&2; exit 2;
}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$ES_PYTHON" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import h5py
import numpy
import scipy

repo = Path(os.environ['PC_REPO']).resolve()
build = Path(os.environ['PC_BUILD_DIR']).resolve()
python = Path(os.path.abspath(os.environ['ES_PYTHON']))  # Keep the venv interpreter path.
template = Path(os.environ.get('L10_INPUT', '/g/data/qp00/sz4640/Results/Genesis_L210_N4320/Fiducial_0.0propscale/input.par')).resolve()
sim_file = repo / 'input/params/simulations/Genesis_L10_N512.par'
commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--'], check=True)
run = Path(os.environ.get('RUN_DIR', f'/g/data/qp00/sz4640/pc_runs/l10_es_hhe_{commit[:8]}')).resolve()
tables = Path(os.environ.get('L10_TABLES', '/g/data/qp00/sz4640/meraxes-devs/input')).resolve()

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def params(text):
    values = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)', line)
        if not match:
            raise ValueError(f'Unsupported parameter line: {line!r}')
        name, value = match.groups()
        if name in values:
            raise ValueError(f'Duplicate parameter: {name}')
        values[name] = value.strip()
    return values

def replace(text, key, value):
    if len(str(value).encode()) >= 256:
        raise ValueError(f'{key} exceeds Meraxes STRLEN=256')
    line = f'{key} : {value}'
    pattern = r'(?m)^[ \t]*' + re.escape(key) + r'[ \t]*:[^\r\n]*$'
    return re.sub(pattern, lambda _: line, text) if re.search(pattern, text) else text.rstrip() + '\n' + line + '\n'

original = template.read_text()
given = params(original)
simulation = params(sim_file.read_text())
sim_dir = Path(os.environ.get('L10_SIMULATION_DIR', simulation['SimulationDir'])).resolve()
scale_file = sim_dir / 'a_list.txt'
scale_factors = [float(value) for value in scale_file.read_text().split()]
if not scale_factors or any(not math.isfinite(a) or a <= 0 for a in scale_factors):
    raise ValueError('a_list.txt must contain positive, finite scale factors only')
if any(b <= a for a, b in zip(scale_factors, scale_factors[1:])):
    raise ValueError('a_list.txt must increase strictly')
for path in (sim_dir / 'trees', sim_dir / 'grids'):
    if not path.is_dir():
        raise ValueError(f'Missing L10 simulation directory: {path}')
grid = int(simulation['ReionGridDim'])
if grid != 32:
    raise ValueError('This L10 launch template expects the supplied 32^3 density grids')
snapshots = ':'  # Meraxes expands this against the actual a_list.txt length.

manifest_file = build / 'pc_build_manifest.json'
build_info = json.loads(manifest_file.read_text())
binary = build / 'bin/meraxes'
if build_info['commit'] != commit or digest(binary) != build_info['binary_sha256']:
    raise ValueError('Build manifest does not match HEAD/binary; run the build command first')
cache = (build / 'CMakeCache.txt').read_text()
for name in ('CALC_MAGS', 'USE_MINI_HALOS'):
    if re.search(r'(?m)^' + name + r':BOOL=(ON|TRUE|1)$', cache):
        raise ValueError(f'This runner requires {name}=OFF; run the build command first')
defaults_value = given.get('DefaultsFile', '')
if not defaults_value:
    raise ValueError('The input template must identify DefaultsFile')
defaults = Path(defaults_value.replace('@INPUT_FILE_DIR@', str(repo / 'input')))
defaults = defaults if defaults.is_absolute() else template.parent / defaults
defaults = defaults.resolve(strict=True)
worker_source = repo / 'tools/es_hhe'
if not (worker_source / 'snapshot_worker.py').is_file():
    raise ValueError('Missing tools/es_hhe/snapshot_worker.py; pull the mode-3 implementation first')

max_step = float(os.environ.get('ES_MAX_STEP_MYR', '1'))
ratio = float(os.environ.get('ES_RADIUS_RATIO', '1.1'))
if not math.isfinite(max_step) or max_step <= 0 or not math.isfinite(ratio) or ratio <= 1:
    raise ValueError('Need ES_MAX_STEP_MYR > 0 and ES_RADIUS_RATIO > 1')
spectrum_value = os.environ.get('ES_SPECTRUM', '')
spectrum = Path(spectrum_value).resolve(strict=True) if spectrum_value else None
if spectrum is not None:
    rows = numpy.loadtxt(spectrum, ndmin=2)
    if (rows.shape[1] != 2 or len(rows) < 1 or not numpy.isfinite(rows).all()
            or (rows[:, 0] < 13.6).any() or (rows[:, 0] > 1000).any()
            or (rows[:, 1] < 0).any() or not numpy.any(rows[:, 1] > 0)):
        raise ValueError('ES_SPECTRUM needs finite energy[eV] in [13.6,1000], nonnegative photon weights')

overrides = dict(simulation)  # Remove any stale L210 cosmology, particle/grid size and units.
overrides.update({
    'DefaultsFile': run / 'provenance/defaults.par',
    'SimParamsFile': run / 'provenance/Genesis_L10_N512.par',
    'SimulationDir': sim_dir,
    'FileNameGalaxies': 'meraxes',
    'OutputDir': run / 'output',
    'OutputSnapshots': snapshots,
    'TargetSnaps': snapshots,
    'ForestIDFile': '',
    'Flag_OutputGrids': '1',
    'Flag_OutputGridsPostReion': '1',
    'Flag_PhotonConservation': '3',
    'Flag_PatchyReion': '1',
    'Flag_IncludeRecombinations': '1',
    'Flag_IncludeSpinTemp': '0',
    'Flag_Compute21cmBrightTemp': '0',
    'Flag_ComputePS': '0',
    'Flag_ConstructLightcone': '0',
    'ReionUVBFlag': '0',
    'ReionFilterType': '0',
    'ReionRtoMFilterType': '0',
    'Flag_ReionizationModifier': '0',
    'Flag_FescCGMSuppression': '0',
    'EscapeFracBHNorm': '0',
    'FlagMCMC': '0',
    'FlagInteractive': '0',
    'HHeStartSnapshot': '0',
    'HHeInitialTemperature': '10000',
    'HHeBoundaryPeriodic': '1',
    'HHeSourceEnergyEV': '18',
    'HHeSourceSpectrumFile': run / 'provenance/source_spectrum.txt' if spectrum else '',
    'HHeMaxStepMyr': format(max_step, '.17g'),
    'ESPythonExecutable': python,
    'ESWorkerPath': run / 'solver/snapshot_worker.py',
    'ESRadiusRatio': format(ratio, '.17g'),
    'ESMaxRadiusCells': '0',
    'PhotometricTablesDir': tables / 'photometric_tables',
    'CoolingFuncsDir': tables / 'cooling_functions',
    'StellarFeedbackDir': tables / 'stellar_feedback_tables/Kroupa',
    'TablesForXHeatingDir': tables / '21cmFAST-tables',
    'RecombinationDir': tables / 'recombination_tables',
})
# Mode 3 evolves H/He recombination in the ES solver. init.c skips init_MHR()
# for this mode, so the legacy RecombinationDir is not a runtime dependency.
# Keep its configured path without requiring or creating the unused directory.
for key in ('PhotometricTablesDir', 'CoolingFuncsDir', 'StellarFeedbackDir',
            'TablesForXHeatingDir'):
    if not overrides[key].is_dir():
        raise ValueError(f'Missing existing input table directory {key}: {overrides[key]}')
effective = original
for key, value in overrides.items():
    effective = replace(effective, key, value)
if run.exists():
    raise ValueError(f'Run directory already exists; choose a new RUN_DIR: {run}')
# Check the exact generated parameters with the real executable before creating
# a run or requesting resources. Use the existing sources for the two files
# that have not yet been copied into provenance; all effective values agree.
with tempfile.TemporaryDirectory(prefix='meraxes-paramcheck-') as directory:
    checked_input = Path(directory) / 'input.par'
    checked_text = replace(effective, 'DefaultsFile', defaults)
    checked_text = replace(checked_text, 'SimParamsFile', sim_file)
    checked_input.write_text(checked_text)
    check = subprocess.run([str(binary), '--check-params', str(checked_input)],
                           text=True, capture_output=True)
    if check.returncode != 0:
        raise RuntimeError(f'Native parameter check failed for input generated from {template}:\n'
                           + check.stdout + check.stderr)
run.mkdir(parents=True, exist_ok=False)
(run / 'provenance').mkdir()
shutil.copy2(binary, run / 'meraxes')
shutil.copytree(worker_source, run / 'solver', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(template, run / 'provenance/original_input.par')
shutil.copy2(defaults, run / 'provenance/defaults.par')
shutil.copy2(sim_file, run / 'provenance/Genesis_L10_N512.par')
shutil.copy2(scale_file, run / 'provenance/a_list.txt')
shutil.copy2(manifest_file, run / 'provenance/pc_build_manifest.json')
shutil.copy2(repo / 'tools/gadi_es_l10.pbs', run / 'submit.pbs')
if spectrum:
    shutil.copy2(spectrum, run / 'provenance/source_spectrum.txt')
(run / 'input.par').write_text(effective)
files = sorted(path for path in run.rglob('*') if path.is_file())
record = {
    'created_utc': datetime.now(timezone.utc).isoformat(), 'commit': commit,
    'repository': str(repo), 'run_dir': str(run), 'python': str(python),
    'grid': grid, 'snapshot_count': len(scale_factors), 'output_snapshots': snapshots,
    'first_redshift': 1 / scale_factors[0] - 1, 'last_redshift': 1 / scale_factors[-1] - 1,
    'source_spectrum': str(spectrum) if spectrum else 'controlled monochromatic 18 eV test',
    'max_step_myr': max_step, 'radius_ratio': ratio,
    'python_version': sys.version,
    'python_packages': {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'h5py': h5py.__version__},
    'simulation_snapshot_list': str(scale_file), 'simulation_snapshot_list_sha256': digest(scale_file),
    'files': {str(path.relative_to(run)): digest(path) for path in files},
}
(run / 'run_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(f'Prepared: {run}')
print('Native parameter check: passed')
print(f'Grid: {grid}^3; snapshots: 0 through {len(scale_factors)-1}')
print(f'Outputs: {run / "output"}')
print(f'Source: {record["source_spectrum"]}')
print(f'qsub -v RUN_DIR={run} {run / "submit.pbs"}')
PY
