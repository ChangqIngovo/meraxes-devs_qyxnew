#!/usr/bin/env python3
"""Prepare an original-ES comparison using an existing, frozen PC L10 run.

Only the Python standard library is needed. This never submits a job or changes
the parent run, and never substitutes current branch defaults for its inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def params(text):
    result = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*(.*)', line)
        if not match or match[1] in result:
            raise ValueError(f'Malformed or duplicate parameter: {line}')
        result[match[1]] = match[2].strip()
    return result


def replace(text, name, value):
    value = str(value)
    if len(value.encode()) >= 256 or '\n' in value or '\r' in value:
        raise ValueError(f'Invalid or too long parameter: {name}')
    pattern = rf'(?m)^[ \t]*{re.escape(name)}\s*:[^\r\n]*$'
    line = f'{name} : {value}'
    return re.sub(pattern, lambda _: line, text) if re.search(pattern, text) else text.rstrip() + '\n' + line + '\n'


def local_file(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f'Frozen file is outside run directory: {name}')
    return path


def check_files(root, record):
    for name, expected in record['files'].items():
        if digest(local_file(root, name)) != expected:
            raise ValueError(f'Prepared file changed: {root / name}')
    scale_file = Path(record['simulation_snapshot_list'])
    if digest(scale_file) != record['simulation_snapshot_list_sha256']:
        raise ValueError('Simulation a_list.txt changed after preparation')


def prepare(parent, run):
    parent, run = parent.resolve(strict=True), run.resolve()
    if run.exists() or run.is_relative_to(parent):
        raise ValueError(f'Choose a new RUN_DIR outside the PC run: {run}')
    record = json.loads((parent / 'run_manifest.json').read_text())
    check_files(parent, record)
    required = ['input.par', 'meraxes']
    original = (parent / 'input.par').read_text()
    config = params(original)
    parameter_files = {}
    for key in ('DefaultsFile', 'SimParamsFile'):
        path = local_file(parent, config[key])
        parameter_files[key] = path
        required.append(str(path.relative_to(parent)))
    if any(name not in record['files'] for name in required):
        raise ValueError('Parent manifest must cover the binary, input, defaults and simulation parameters')
    expected = {'Flag_PhotonConservation': '3', 'ReionGridDim': '32',
                'Flag_PatchyReion': '1', 'Flag_IncludeRecombinations': '1',
                'OutputSnapshots': ':', 'Flag_OutputGrids': '1',
                'Flag_OutputGridsPostReion': '1'}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f'Parent must have {name} : {value}')
    if float(config.get('BoxSize', 'nan')) != 10.0:
        raise ValueError('This control runner requires the L10 simulation (BoxSize=10)')
    sim_dir = Path(config['SimulationDir'])
    if not sim_dir.is_absolute() or not all((sim_dir / name).is_dir() for name in ('trees', 'grids')):
        raise ValueError('Missing L10 trees/grids or nonabsolute SimulationDir')
    scale_file = (sim_dir / 'a_list.txt').resolve(strict=True)
    if scale_file != Path(record['simulation_snapshot_list']).resolve(strict=True):
        raise ValueError('Input SimulationDir disagrees with the parent snapshot-list provenance')
    scales = [float(value) for value in scale_file.read_text().split()]
    if (not scales or any(not math.isfinite(a) or a <= 0 for a in scales)
            or any(b <= a for a, b in zip(scales, scales[1:]))):
        raise ValueError('Invalid simulation a_list.txt')
    binary = parent / 'meraxes'
    if not os.access(binary, os.X_OK):
        raise ValueError(f'Parent binary is not executable: {binary}')
    overrides = {
        'Flag_PhotonConservation': '0', 'OutputDir': run / 'output',
        'DefaultsFile': run / 'provenance/defaults.par',
        'SimParamsFile': run / 'provenance/Genesis_L10_N512.par',
        'OutputSnapshots': ':', 'TargetSnaps': ':',
        'Flag_OutputGrids': '1', 'Flag_OutputGridsPostReion': '1',
    }
    effective = original
    for name, value in overrides.items():
        effective = replace(effective, name, value)
    # Validate with this exact frozen binary before creating the destination.
    checked = replace(effective, 'DefaultsFile', parameter_files['DefaultsFile'])
    checked = replace(checked, 'SimParamsFile', parameter_files['SimParamsFile'])
    with tempfile.TemporaryDirectory(prefix='meraxes-es-control-') as directory:
        checked_path = Path(directory) / 'input.par'
        checked_path.write_text(checked)
        result = subprocess.run([str(binary), '--check-params', str(checked_path)],
                                cwd=parent, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Native parameter check failed:\n' + result.stdout + result.stderr)
    pbs = Path(__file__).resolve().parents[1] / 'gadi_es_l10_control.pbs'
    if not pbs.is_file():
        raise ValueError(f'Missing PBS template: {pbs}')
    run.mkdir(parents=True, exist_ok=False)
    provenance = run / 'provenance'
    provenance.mkdir()
    shutil.copy2(binary, run / 'meraxes')
    shutil.copy2(parameter_files['DefaultsFile'], provenance / 'defaults.par')
    shutil.copy2(parameter_files['SimParamsFile'], provenance / 'Genesis_L10_N512.par')
    shutil.copy2(scale_file, provenance / 'a_list.txt')
    shutil.copy2(parent / 'input.par', provenance / 'pc_input.par')
    shutil.copy2(parent / 'run_manifest.json', provenance / 'pc_run_manifest.json')
    shutil.copy2(pbs, run / 'submit.pbs')
    shutil.copy2(__file__, run / 'verify_control.py')
    (run / 'input.par').write_text(effective)
    control = {
        'model': 'original ES (Flag_PhotonConservation=0)',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_run': str(parent), 'commit': record['commit'],
        'run_dir': str(run), 'grid': 32, 'snapshot_count': len(scales),
        'output_snapshots': ':', 'simulation_snapshot_list': str(scale_file),
        'simulation_snapshot_list_sha256': digest(scale_file),
        'parameter_overrides': {key: str(value) for key, value in overrides.items()},
        'files': {str(path.relative_to(run)): digest(path)
                  for path in sorted(run.rglob('*')) if path.is_file()},
    }
    (run / 'run_manifest.json').write_text(json.dumps(control, indent=2) + '\n')
    print(f'PC comparison run: {parent}\nFrozen binary commit: {record["commit"]}')
    print(f'Prepared original ES: {run}\nNative parameter check: passed')
    print(f'Grid: 32^3; snapshots: 0 through {len(scales)-1}')
    print(f'Outputs: {run / "output"}')
    print('Legacy recombination cache is reused if compatible; otherwise MPI rebuilds it in output/.')
    print(f'qsub -v RUN_DIR={run} {run / "submit.pbs"}')


def verify(run, ncpus):
    run = run.resolve(strict=True)
    record = json.loads((run / 'run_manifest.json').read_text())
    if str(run) != record['run_dir']:
        raise ValueError('Run was moved after preparation; prepare a new directory')
    check_files(run, record)
    config = params((run / 'input.par').read_text())
    if config['Flag_PhotonConservation'] != '0' or config['OutputDir'] != str(run / 'output'):
        raise ValueError('Invalid control mode or output path')
    if ncpus < 1 or ncpus > record['grid']:
        raise ValueError('MPI ranks must be positive and no greater than the 32 grid slabs')
    output = run / 'output'
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError('Output is nonempty or not a local directory; prepare a new RUN_DIR')
    output.mkdir(exist_ok=True)
    print(f'Model: {record["model"]}\nPC comparison: {record["parent_run"]}', flush=True)
    print(f'Frozen binary commit: {record["commit"]}\nSnapshots: :', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--pc-run', required=True, type=Path)
    prep.add_argument('--run-dir', required=True, type=Path)
    check = commands.add_parser('verify')
    check.add_argument('--run-dir', required=True, type=Path)
    check.add_argument('--ncpus', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.pc_run, args.run_dir)
    else:
        verify(args.run_dir, args.ncpus)


if __name__ == '__main__':
    main()
