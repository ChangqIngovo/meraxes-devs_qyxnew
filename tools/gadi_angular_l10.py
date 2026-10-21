#!/usr/bin/env python3
"""Prepare a fresh, pinned mode-4 Meraxes L10 simulation; never submit implicitly."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import h5py
import numpy

repo = Path(os.environ.get('ANGULAR_REPO', Path(__file__).resolve().parents[1])).resolve()
build = Path(os.environ.get('ANGULAR_BUILD_DIR', repo / 'build-angular-mpi')).resolve()
python = Path(os.path.abspath(os.environ.get('ANGULAR_PYTHON', sys.executable)))  # Keep the venv interpreter path.
if python != Path(os.path.abspath(sys.executable)):
    raise ValueError('Run this preparation script using ANGULAR_PYTHON itself')
sim_file = repo / 'input/params/simulations/Genesis_L10_N512.par'
defaults = (repo / 'input/params/defaults.par').resolve(strict=True)
commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--'], check=True)
run = Path(os.environ.get('RUN_DIR', f'/g/data/qp00/sz4640/angular_runs/l10_{commit[:8]}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}')).resolve()
if ',' in str(run):
    raise ValueError('RUN_DIR cannot contain a comma because qsub -v uses comma-separated variables')
tables = Path(os.environ.get('L10_TABLES', '/g/data/qp00/sz4640/meraxes-devs/input')).resolve()

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def params(text, source='<input>'):
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)', line)
        if not match:
            raise ValueError(f"Invalid parameter syntax in '{source}' at line {number}: {line!r}")
        name, value = match.groups()
        if name in values:
            raise ValueError(f'Duplicate parameter: {name} in {source} at line {number}')
        values[name] = value.strip()
    return values

def replace(text, key, value):
    if any(character in str(value) for character in '\r\n#'):
        raise ValueError(f'{key} contains a character that cannot be represented in a parameter value')
    if len(str(value).encode()) >= 256:
        raise ValueError(f'{key} exceeds Meraxes STRLEN=256')
    line = f'{key} : {value}'
    pattern = r'(?m)^[ \t]*' + re.escape(key) + r'[ \t]*:[^\r\n]*$'
    return re.sub(pattern, lambda _: line, text) if re.search(pattern, text) else text.rstrip() + '\n' + line + '\n'

simulation_text = sim_file.read_text()
simulation = params(simulation_text, sim_file)
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
tree_file = sim_dir / 'trees' / simulation['CatalogFilePrefix']
if not tree_file.is_file():
    raise ValueError(f'Missing L10 tree catalogue: {tree_file}')
# Read the actual density-grid headers, not just the parameter-file dimension.
# Match the two input layouts recognized by read_grids-velociraptor.c.
grid_files = []
previous_nfiles = None
for snapshot in range(len(scale_factors)):
    swift = sim_dir / 'grids' / f'snap_{snapshot:04d}.hdf5'
    if swift.is_file():
        resampled_dir = sim_dir / 'grids/resampled' / f'N{grid}'
        selected = resampled_dir / swift.name if resampled_dir.is_dir() else swift
        with h5py.File(selected, 'r') as data:
            shape = data['/PartType1/Grids/Density'].shape
            if shape != (grid, grid, grid):
                raise ValueError(f'Expected {grid}^3 density grid: {selected}, found {shape}')
        grid_files.append(selected)
    else:
        first = sim_dir / 'grids' / f'snapshot_{snapshot:03d}.den.0'
        with h5py.File(first, 'r') as data:
            shape = tuple(int(numpy.asarray(data.attrs[f'Ngrid_{axis}']).item()) for axis in 'XYZ')
            if shape != (grid, grid, grid):
                raise ValueError(f'Expected {grid}^3 density grid: {first}, found {shape}')
            nfiles = data.attrs.get('Num_files', previous_nfiles)
            if nfiles is None or int(numpy.asarray(nfiles).item()) < 1:
                raise ValueError(f'Cannot determine number of density files: {first}')
            previous_nfiles = int(numpy.asarray(nfiles).item())
        for part in range(previous_nfiles):
            path = first.with_name(f'snapshot_{snapshot:03d}.den.{part}')
            if not path.is_file():
                raise ValueError(f'Missing density grid part: {path}')
            grid_files.append(path)
snapshots = ':'  # Meraxes expands this against the actual a_list.txt length.

manifest_file = build / 'angular_build_manifest.json'
build_info = json.loads(manifest_file.read_text())
binary = build / 'bin/meraxes'
if build_info['commit'] != commit or digest(binary) != build_info['binary_sha256']:
    raise ValueError('Build manifest does not match HEAD/binary; run the build command first')
native_library = build / 'lib/libangular_front.so'
if (not native_library.is_file()
        or digest(native_library) != build_info.get('angular_library_sha256')):
    raise ValueError('Build manifest does not match angular-face geometry library; run the build command first')
cache = (build / 'CMakeCache.txt').read_text()
if digest(build / 'CMakeCache.txt') != build_info['cmake_cache_sha256']:
    raise ValueError('CMake cache changed after the recorded build; rebuild before preparing')
for name in ('CALC_MAGS', 'USE_MINI_HALOS', 'USE_CUDA', 'USE_STOCHASTICITY'):
    if re.search(r'(?m)^' + name + r':BOOL=(ON|TRUE|1)$', cache):
        raise ValueError(f'This runner requires {name}=OFF; run the build command first')
worker_source = repo / 'tools/es_angular'
if not (worker_source / 'snapshot_worker.py').is_file():
    raise ValueError('Missing tools/es_angular/snapshot_worker.py; update to the mode-4 integration')

max_step = float(os.environ.get('ANGULAR_MAX_STEP_MYR', '0.1'))
level = int(os.environ.get('ANGULAR_LEVEL', '0'))
epsilon = float(os.environ.get('ANGULAR_EPSILON', '0.1'))
speed = float(os.environ.get('ANGULAR_SPEED_FRACTION', '1'))
if (not math.isfinite(max_step) or max_step <= 0 or not 0 <= level <= 8
        or not math.isfinite(epsilon) or epsilon < 0
        or not math.isfinite(speed) or not 0 < speed <= 1):
    raise ValueError('Need max step > 0, level in [0,8], epsilon >= 0, speed fraction in (0,1]')
spectrum_value = os.environ.get('ANGULAR_SPECTRUM', '')
spectrum = Path(spectrum_value).resolve(strict=True) if spectrum_value else None
if spectrum is not None:
    rows = numpy.loadtxt(spectrum, ndmin=2)
    if (rows.shape[1] != 2 or len(rows) < 1 or not numpy.isfinite(rows).all()
            or (rows[:, 0] < 13.6).any() or (rows[:, 0] > 1000).any()
            or (rows[:, 1] < 0).any() or not numpy.any(rows[:, 1] > 0)):
        raise ValueError('ANGULAR_SPECTRUM needs finite energy[eV] in [13.6,1000], nonnegative photon weights')

# The L10 simulation file and this checkout's defaults are the only parameter
# sources. Runtime mode/output/table settings are applied to a fresh L10 input.
overrides = {
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
    'Flag_PhotonConservation': '4',
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
    'AngularStartSnapshot': '0',
    'AngularInitialTemperature': '10000',
    'AngularSourceEnergyEV': '18',
    'AngularSourceSpectrumFile': run / 'provenance/source_spectrum.txt' if spectrum else '',
    'AngularMaxStepMyr': format(max_step, '.17g'),
    'AngularPythonExecutable': python,
    'AngularWorkerPath': run / 'solver/snapshot_worker.py',
    'AngularLevel': str(level),
    'AngularEpsilon': format(epsilon, '.17g'),
    'AngularSpeedFraction': format(speed, '.17g'),
    'AngularLibraryPath': run / 'solver/libangular_front.so',
    'PhotometricTablesDir': tables / 'photometric_tables',
    'CoolingFuncsDir': tables / 'cooling_functions',
    'StellarFeedbackDir': tables / 'stellar_feedback_tables/Kroupa',
    'TablesForXHeatingDir': tables / '21cmFAST-tables',
    'RecombinationDir': tables / 'recombination_tables',
}
# Mode 4 evolves H/He recombination in the angular worker. init.c skips init_MHR()
# for this mode, so the legacy RecombinationDir is not a runtime dependency.
# Keep its configured path without requiring or creating the unused directory.
for key in ('PhotometricTablesDir', 'CoolingFuncsDir', 'StellarFeedbackDir',
            'TablesForXHeatingDir'):
    if not overrides[key].is_dir():
        raise ValueError(f'Missing existing input table directory {key}: {overrides[key]}')
effective = simulation_text
for key, value in overrides.items():
    effective = replace(effective, key, value)
if run.exists():
    raise ValueError(f'Run directory already exists; choose a new RUN_DIR: {run}')
# Check the generated L10 input and branch defaults before creating a run.
# Only provenance paths differ from the final copies; effective values agree.
with tempfile.TemporaryDirectory(prefix='meraxes-paramcheck-') as directory:
    checked_input = Path(directory) / 'input.par'
    checked_text = replace(effective, 'DefaultsFile', defaults)
    checked_text = replace(checked_text, 'SimParamsFile', sim_file)
    checked_input.write_text(checked_text)
    check = subprocess.run([str(binary), '--check-params', str(checked_input)],
                           text=True, capture_output=True)
    if check.returncode != 0:
        raise RuntimeError(f'Native parameter check failed for L10 simulation {sim_file}, '
                           f'with branch defaults {defaults}:\n'
                           + check.stdout + check.stderr)
run.mkdir(parents=True, exist_ok=False)
(run / 'provenance').mkdir()
shutil.copy2(binary, run / 'meraxes')
shutil.copytree(worker_source, run / 'solver', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(native_library, run / 'solver/libangular_front.so')
shutil.copy2(defaults, run / 'provenance/defaults.par')
shutil.copy2(sim_file, run / 'provenance/Genesis_L10_N512.par')
shutil.copy2(scale_file, run / 'provenance/a_list.txt')
shutil.copy2(manifest_file, run / 'provenance/angular_build_manifest.json')
shutil.copy2(repo / 'tools/gadi_angular_l10.pbs', run / 'submit.pbs')
if spectrum:
    shutil.copy2(spectrum, run / 'provenance/source_spectrum.txt')
(run / 'input.par').write_text(effective)
files = sorted(path for path in run.rglob('*') if path.is_file())
record = {
    'created_utc': datetime.now(timezone.utc).isoformat(), 'commit': commit,
    'repository': str(repo), 'run_dir': str(run), 'python': str(python),
    'grid': grid, 'snapshot_count': len(scale_factors), 'output_snapshots': snapshots,
    'first_redshift': 1 / scale_factors[0] - 1, 'last_redshift': 1 / scale_factors[-1] - 1,
    'source_spectrum': str(spectrum) if spectrum else 'monochromatic 18 eV',
    'simulation_parameter_file': str(sim_file), 'defaults_parameter_file': str(defaults),
    'max_step_myr': max_step, 'angular_level': level, 'epsilon': epsilon,
    'speed_fraction': speed, 'solver': 'angular-face-periodic-hhe-v1',
    'angular_library_sha256': digest(native_library),
    'python_version': sys.version,
    'python_packages': {'numpy': numpy.__version__, 'h5py': h5py.__version__},
    'simulation_snapshot_list': str(scale_file), 'simulation_snapshot_list_sha256': digest(scale_file),
    'density_files': {str(path): digest(path) for path in grid_files},
    'tree_catalogue': str(tree_file),
    'files': {str(path.relative_to(run)): digest(path) for path in files},
}
(run / 'run_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(f'Prepared: {run}')
print('Native parameter check: passed')
print(f'Simulation parameters: {sim_file}')
print(f'Default parameters: {defaults}')
print(f'Grid: {grid}^3; snapshots: 0 through {len(scale_factors)-1}')
print(f'Outputs: {run / "output"}')
print(f'Source: {record["source_spectrum"]}')
print(f'qsub -v {shlex.quote(f"RUN_DIR={run}")} {shlex.quote(str(run / "submit.pbs"))}')
