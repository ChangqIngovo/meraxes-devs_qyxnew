#!/usr/bin/env python3
"""Check PC-to-original-ES preparation without executing a simulation."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
PREPARE = REPO / 'tools/es_hhe/prepare_es_control.py'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result


class ControlPreparation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='es-control-test-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.parent = self.base / 'pc'
        self.run = self.base / 'control'
        self.sim = self.base / 'simulation'
        self.tables = self.base / 'tables'
        self.checks = self.base / 'parameter_checks.jsonl'
        for path in (self.parent / 'provenance', self.sim / 'trees', self.sim / 'grids',
                     self.base / 'commands', self.parent / 'output'):
            path.mkdir(parents=True)
        for directory in ('photometric_tables', 'cooling_functions',
                          'stellar_feedback_tables/Kroupa', '21cmFAST-tables'):
            (self.tables / directory).mkdir(parents=True)
        (self.sim / 'a_list.txt').write_text('0.02\n0.04\n0.08\n')
        (self.parent / 'output/meraxes_grids_0.hdf5').write_bytes(b'existing PC output\n')
        shutil.copy2(REPO / 'input/params/defaults.par', self.parent / 'provenance/defaults.par')
        shutil.copy2(REPO / 'input/params/simulations/Genesis_L10_N512.par',
                     self.parent / 'provenance/Genesis_L10_N512.par')
        self.input = self.parent / 'input.par'
        values = {
            'DefaultsFile': self.parent / 'provenance/defaults.par',
            'SimParamsFile': self.parent / 'provenance/Genesis_L10_N512.par',
            'SimulationDir': self.sim, 'BoxSize': '10.0', 'ReionGridDim': '32',
            'FileNameGalaxies': 'meraxes', 'OutputDir': self.parent / 'output',
            'OutputSnapshots': ':', 'TargetSnaps': ':', 'Flag_PhotonConservation': '3',
            'Flag_PatchyReion': '1', 'Flag_IncludeRecombinations': '1',
            'Flag_OutputGrids': '1', 'Flag_OutputGridsPostReion': '1',
            'Flag_IncludeSpinTemp': '0', 'Flag_Compute21cmBrightTemp': '0',
            'Flag_ComputePS': '0', 'Flag_ConstructLightcone': '0', 'ReionUVBFlag': '0',
            'Flag_ReionizationModifier': '0', 'Flag_FescCGMSuppression': '0',
            'EscapeFracBHNorm': '0', 'ReionFilterType': '0', 'ReionRtoMFilterType': '0',
            'FlagMCMC': '0', 'FlagInteractive': '0', 'ForestIDFile': '',
            'SfEfficiency': '0.123456', 'ReionNionPhotPerBary': '4567',
            'EscapeFracNorm': '0.173', 'HHeStartSnapshot': '0',
            'HHeInitialTemperature': '10000', 'HHeBoundaryPeriodic': '1',
            'HHeSourceEnergyEV': '18', 'HHeSourceSpectrumFile': '',
            'HHeMaxStepMyr': '1', 'ESRadiusRatio': '1.1', 'ESMaxRadiusCells': '0',
            'ESPythonExecutable': sys.executable,
            'ESWorkerPath': self.parent / 'solver/snapshot_worker.py',
            'PhotometricTablesDir': self.tables / 'photometric_tables',
            'CoolingFuncsDir': self.tables / 'cooling_functions',
            'StellarFeedbackDir': self.tables / 'stellar_feedback_tables/Kroupa',
            'TablesForXHeatingDir': self.tables / '21cmFAST-tables',
            'RecombinationDir': self.tables / 'recombination_tables',
        }
        self.input.write_text(''.join(f'{key} : {value}\n' for key, value in values.items()))
        self.binary = self.parent / 'meraxes'
        self.binary.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, sys\n'
            'from pathlib import Path\n'
            'assert sys.argv[1] == "--check-params", "No simulation may run in this test"\n'
            'with Path(os.environ["CONTROL_TEST_CHECKS"]).open("a") as f:\n'
            '    f.write(json.dumps({"args": sys.argv, "input": Path(sys.argv[2]).read_text()}) + "\\n")\n'
            'if os.environ.get("CONTROL_TEST_REJECT"):\n'
            '    print("Synthetic native parser rejection", file=sys.stderr)\n'
            '    sys.exit(1)\n'
        )
        self.binary.chmod(0o755)
        self.manifest = {
            'commit': 'a' * 40, 'run_dir': str(self.parent), 'repository': str(REPO),
            'grid': 32, 'snapshot_count': 3, 'output_snapshots': ':',
            'python': sys.executable, 'python_packages': {},
            'simulation_parameter_file': str(REPO / 'input/params/simulations/Genesis_L10_N512.par'),
            'defaults_parameter_file': str(REPO / 'input/params/defaults.par'),
            'simulation_snapshot_list': str(self.sim / 'a_list.txt'),
            'simulation_snapshot_list_sha256': sha256(self.sim / 'a_list.txt'),
            'files': {},
        }
        self.refresh_manifest()
        self.env = {**os.environ, 'CONTROL_TEST_CHECKS': str(self.checks)}
        self.env.pop('CONTROL_TEST_REJECT', None)
        for name in list(self.env):
            if name in {'BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS'} or name.startswith('BASH_FUNC_'):
                self.env.pop(name)

    def refresh_manifest(self):
        names = ('meraxes', 'input.par', 'provenance/defaults.par',
                 'provenance/Genesis_L10_N512.par')
        self.manifest['files'] = {name: sha256(self.parent / name) for name in names}
        (self.parent / 'run_manifest.json').write_text(json.dumps(self.manifest))

    def rewrite_input(self, key, value):
        text = self.input.read_text()
        text = re.sub(r'(?m)^' + re.escape(key) + r'\s*:.*$', f'{key} : {value}', text)
        self.input.write_text(text)
        self.refresh_manifest()

    def command(self, *args, ok=True, script=PREPARE):
        result = subprocess.run([sys.executable, str(script), *map(str, args)],
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def prepare(self, ok=True):
        return self.command('prepare', '--pc-run', self.parent, '--run-dir', self.run, ok=ok)

    def test_prepare_preserves_parent_and_calibration(self):
        before = {path.relative_to(self.parent): path.read_bytes()
                  for path in self.parent.rglob('*') if path.is_file()}
        self.prepare()
        after = {path.relative_to(self.parent): path.read_bytes()
                 for path in self.parent.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        original = parameters(self.input)
        control = parameters(self.run / 'input.par')
        for name, value in original.items():
            if name not in {'DefaultsFile', 'SimParamsFile', 'OutputDir',
                            'Flag_PhotonConservation'}:
                self.assertEqual(control[name], value, name)
        self.assertEqual(control['Flag_PhotonConservation'], '0')
        self.assertEqual(control['OutputDir'], str(self.run / 'output'))
        self.assertEqual(Path(control['DefaultsFile']).read_bytes(),
                         Path(original['DefaultsFile']).read_bytes())
        self.assertEqual(Path(control['SimParamsFile']).read_bytes(),
                         Path(original['SimParamsFile']).read_bytes())
        self.assertEqual((self.run / 'meraxes').read_bytes(), self.binary.read_bytes())
        self.assertFalse((self.run / 'output').exists())
        self.assertTrue(self.checks.is_file())
        check = json.loads(self.checks.read_text().splitlines()[-1])
        self.assertIn('Flag_PhotonConservation : 0\n', check['input'])
        self.assertIn(f'OutputDir : {self.run / "output"}\n', check['input'])
        self.assertEqual(check['args'][1], '--check-params')
        record = json.loads((self.run / 'run_manifest.json').read_text())
        self.assertEqual(record['commit'], self.manifest['commit'])
        self.assertEqual(record['snapshot_count'], 3)
        for name, expected in record['files'].items():
            self.assertEqual(sha256(self.run / name), expected, name)
        self.command('verify', '--run-dir', self.run, '--ncpus', '32',
                     script=self.run / 'verify_control.py')

    def test_missing_recombination_cache_allowed(self):
        self.assertFalse((self.tables / 'recombination_tables').exists())
        self.prepare()
        config = parameters(self.run / 'input.par')
        self.assertEqual(config['Flag_IncludeRecombinations'], '1')
        self.assertEqual(config['RecombinationDir'], str(self.tables / 'recombination_tables'))
        self.assertFalse((self.tables / 'recombination_tables').exists())

    def test_parent_mutation_rejected_without_partial_run(self):
        self.input.write_text(self.input.read_text() + '# unrecorded change\n')
        self.prepare(ok=False)
        self.assertFalse(self.run.exists())

    def test_changed_snapshot_list_rejected(self):
        (self.sim / 'a_list.txt').write_text('0.02\n0.04\n0.09\n')
        self.prepare(ok=False)
        self.assertFalse(self.run.exists())

    def test_native_rejection_precedes_run_creation(self):
        self.env['CONTROL_TEST_REJECT'] = '1'
        result = self.prepare(ok=False)
        self.assertIn('Synthetic native parser rejection', result.stdout + result.stderr)
        self.assertFalse(self.run.exists())

    def test_existing_destination_not_rewritten(self):
        self.prepare()
        old = (self.run / 'input.par').read_bytes()
        self.prepare(ok=False)
        self.assertEqual((self.run / 'input.par').read_bytes(), old)

    def test_non_pc_source_rejected(self):
        self.rewrite_input('Flag_PhotonConservation', '0')
        self.prepare(ok=False)
        self.assertFalse(self.run.exists())

    def test_wrong_simulation_rejected(self):
        self.rewrite_input('BoxSize', '210')
        self.prepare(ok=False)
        self.assertFalse(self.run.exists())

    def test_control_mutation_rejected_at_verify(self):
        self.prepare()
        with (self.run / 'input.par').open('a') as stream:
            stream.write('# modified after preparation\n')
        self.command('verify', '--run-dir', self.run, '--ncpus', '32', ok=False)

    def test_too_many_mpi_ranks_rejected(self):
        self.prepare()
        self.command('verify', '--run-dir', self.run, '--ncpus', '48', ok=False)

    def test_existing_output_and_pc_output_symlink_rejected(self):
        self.prepare()
        output = self.run / 'output'
        output.mkdir()
        (output / 'old_grid.hdf5').write_bytes(b'previous result')
        self.command('verify', '--run-dir', self.run, ok=False)
        self.assertEqual((output / 'old_grid.hdf5').read_bytes(), b'previous result')
        (output / 'old_grid.hdf5').unlink()
        output.rmdir()
        output.symlink_to(self.parent / 'output', target_is_directory=True)
        self.command('verify', '--run-dir', self.run, ok=False)
        self.assertEqual((self.parent / 'output/meraxes_grids_0.hdf5').read_bytes(),
                         b'existing PC output\n')

    def test_pbs_launches_frozen_binary_and_keeps_pc_output(self):
        self.prepare()
        commands = self.base / 'commands'
        module = commands / 'module'
        module.write_text('#!/bin/sh\nexit 0\n')
        module.chmod(0o755)
        mpirun = commands / 'mpirun'
        mpirun.write_text('#!' + sys.executable + '\n'
                          'import json, os, sys\n'
                          'from pathlib import Path\n'
                          'Path(os.environ["RUN_DIR"], "invocation.json").write_text(json.dumps(sys.argv[1:]))\n')
        mpirun.chmod(0o755)
        env = {**self.env, 'RUN_DIR': str(self.run), 'PBS_JOBID': 'test.gadi-pbs',
               'PBS_NCPUS': '32', 'PBS_NNODES': '1',
               'PATH': str(commands) + os.pathsep + self.env['PATH']}
        env.pop('PBS_NODEFILE', None)
        result = subprocess.run(['bash', '--noprofile', '--norc', str(self.run / 'submit.pbs')],
                                env=env, text=True, capture_output=True)
        log = self.run / 'run_test.gadi-pbs.log'
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr +
                         (log.read_text() if log.exists() else ''))
        args = json.loads((self.run / 'invocation.json').read_text())
        self.assertEqual(args[-2:], [str(self.run / 'meraxes'), str(self.run / 'input.par')])
        self.assertEqual((self.parent / 'output/meraxes_grids_0.hdf5').read_bytes(),
                         b'existing PC output\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
