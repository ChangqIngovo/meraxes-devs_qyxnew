#!/usr/bin/env python3
"""Exercise the L10 preparation/PBS boundary without Gadi or simulation execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[2]
HEAD = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
NATIVE_BINARY = os.environ.get('ES_PARAMCHECK_BINARY')

class TimedTextTestResult(unittest.TextTestResult):
    """Expose aggregate startup cost and the active case in CTest logs."""
    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)
        self.stream.flush()

    def stopTest(self, test):
        super().stopTest(test)
        self.stream.writeln(f'  elapsed: {time.monotonic() - self.started:.2f} s')
        self.stream.flush()


class Scripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='es-l10-script-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base/'repo'
        self.build = self.base/'build'
        self.sim = self.base/'simulation'
        self.tables = self.base/'tables'
        for directory in [self.repo/'input/params/simulations', self.repo/'tools/es_hhe', self.build/'bin', self.build/'lib', self.sim/'trees', self.sim/'grids', self.base/'commands']:
            directory.mkdir(parents=True)
        for directory in ['photometric_tables','cooling_functions','stellar_feedback_tables/Kroupa','21cmFAST-tables','recombination_tables']:
            (self.tables/directory).mkdir(parents=True)
        (self.sim/'a_list.txt').write_text('0.02\n0.04\n0.08\n')
        shutil.copy2(SOURCE/'input/params/simulations/Genesis_L10_N512.par',self.repo/'input/params/simulations/Genesis_L10_N512.par')
        shutil.copy2(SOURCE/'input/params/defaults.par',self.repo/'input/params/defaults.par')
        shutil.copy2(SOURCE/'tools/gadi_es_l10.pbs',self.repo/'tools/gadi_es_l10.pbs')
        (self.repo/'tools/es_hhe/snapshot_worker.py').write_text('# fixture\n')
        self.sim_params = self.repo/'input/params/simulations/Genesis_L10_N512.par'
        self.defaults = self.repo/'input/params/defaults.par'
        binary=self.build/'bin/meraxes'
        if NATIVE_BINARY:
            shutil.copy2(NATIVE_BINARY,binary)
        else:
            binary.write_text('#!/bin/sh\nif [ "$#" -eq 2 ] && [ "$1" = "--check-params" ]; then exit 0; fi\necho "Fixture binary must not execute a simulation; expected mock mpirun" >&2\nexit 97\n')
        binary.chmod(0o755)
        (self.build/'CMakeCache.txt').write_text('CALC_MAGS:BOOL=OFF\nUSE_MINI_HALOS:BOOL=OFF\n')
        self.native_library=self.build/'lib/libes_hhe_kernels.so'
        self.native_library.write_bytes(b'fixture native ES kernel; never loaded by mock mpirun\n')
        (self.build/'pc_build_manifest.json').write_text(json.dumps({
            'commit':HEAD,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'es_native_library_sha256':hashlib.sha256(self.native_library.read_bytes()).hexdigest()}))
        git=self.base/'commands/git'
        git.write_text('#!/usr/bin/env python3\nimport sys\nif "rev-parse" in sys.argv: print('+repr(HEAD)+')\n')
        git.chmod(0o755)
        mpirun=self.base/'commands/mpirun'
        mpirun.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["RUN_DIR"],"invocation.json").write_text(json.dumps(sys.argv))\nPath(os.environ["RUN_DIR"],"launch_environment.json").write_text(json.dumps({name:os.environ.get(name) for name in ["ES_WORKER_THREADS","ES_NATIVE_REQUIRED","ES_NATIVE_LIBRARY","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","OMP_WAIT_POLICY","OMP_PROC_BIND","GOMP_CPU_AFFINITY","OMP_PLACES"]}))\n')
        mpirun.chmod(0o755)
        self.run=self.base/'run'
        self.env={**os.environ,'PC_REPO':str(self.repo),'PC_BUILD_DIR':str(self.build),'L10_SIMULATION_DIR':str(self.sim),'L10_TABLES':str(self.tables),'ES_PYTHON':sys.executable,'RUN_DIR':str(self.run),'PATH':str(self.base/'commands')+os.pathsep+os.environ['PATH']}
        for name in ['L10_INPUT','ES_SPECTRUM','ES_RADIUS_RATIO','ES_MAX_STEP_MYR',
                     'ES_WORKER_THREADS','ES_NATIVE_REQUIRED','ES_NATIVE_LIBRARY',
                     'PBS_NNODES','PBS_NODEFILE']:
            self.env.pop(name,None)
    def shell_environment(self, **updates):
        # Noninteractive bash still reads BASH_ENV, and imported functions take
        # precedence over our PATH mocks. Isolate only the fixture subprocesses;
        # keep the caller's Python and shared-library environment intact.
        env = {**self.env, **updates}
        for name in list(env):
            if name in {'BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS'} or name.startswith('BASH_FUNC_'):
                del env[name]
        return env
    def prepare(self,ok=True):
        result=subprocess.run(['bash','--noprofile','--norc',str(SOURCE/'tools/gadi_es_l10.sh'),'prepare'],env=self.shell_environment(),text=True,capture_output=True)
        self.assertEqual(result.returncode==0,ok,result.stdout+result.stderr)
        return result
    def assert_l10_configuration(self):
        config=(self.run/'input.par').read_text()
        for key,value in [('OutputSnapshots',':'),('TargetSnaps',':'),('BoxSize','10.0'),
                          ('ReionGridDim','32'),('Flag_PhotonConservation','3'),
                          ('Flag_IncludeRecombinations','1'),('Flag_Compute21cmBrightTemp','0')]:
            self.assertRegex(config,rf'(?m)^\s*{key}\s*:\s*{re.escape(value)}\s*$')
        self.assertIn(f'DefaultsFile : {self.run}/provenance/defaults.par',config)
        self.assertIn(f'SimParamsFile : {self.run}/provenance/Genesis_L10_N512.par',config)
        self.assertIn(f'SimulationDir : {self.sim}',config)
        # Source/galaxy parameters must come from this branch's defaults, not
        # an unrelated calibrated L210 input or its foreign defaults file.
        for key in ['SfEfficiency','ReionNionPhotPerBary','SnReheatEff']:
            self.assertNotRegex(config,rf'(?m)^\s*{key}\s*:')
        self.assertNotIn('L210',config)
        self.assertEqual((self.run/'provenance/defaults.par').read_bytes(),self.defaults.read_bytes())
        self.assertEqual((self.run/'provenance/Genesis_L10_N512.par').read_bytes(),self.sim_params.read_bytes())
        if NATIVE_BINARY:
            check=subprocess.run([str(self.run/'meraxes'),'--check-params',str(self.run/'input.par')],
                                 env=self.shell_environment(),text=True,capture_output=True)
            self.assertEqual(check.returncode,0,check.stdout+check.stderr)
            for key,value in [('SfEfficiency','0.08'),('ReionNionPhotPerBary','4000'),
                              ('SnReheatEff','10'),('BoxSize','10'),('ReionGridDim','32')]:
                self.assertRegex(check.stdout,rf'\b{key}\s+{re.escape(value)}(?:\s|$)')
    def test_prepare_uses_l10_and_branch_defaults_all_snapshots(self):
        result=self.prepare()
        self.assertIn('Native parameter check: passed',result.stdout)
        self.assertFalse((self.run/'output').exists())
        self.assert_l10_configuration()
        record=json.loads((self.run/'run_manifest.json').read_text())
        self.assertEqual(record['snapshot_count'],3)
        self.assertEqual(record['commit'],HEAD)
        self.assertEqual(record['simulation_parameter_file'],str(self.sim_params))
        self.assertEqual(record['defaults_parameter_file'],str(self.defaults))
        self.assertNotIn('defaults_compatibility',record)
        for name in ['original_input.par','original_defaults.par','branch_defaults.par','defaults_compatibility.json']:
            self.assertFalse((self.run/'provenance'/name).exists())
        for name in ['provenance/defaults.par','provenance/Genesis_L10_N512.par']:
            self.assertIn(name,record['files'])
        self.assertEqual((self.run/'solver/libes_hhe_kernels.so').read_bytes(),self.native_library.read_bytes())
        self.assertEqual(record['es_native_library_sha256'],record['files']['solver/libes_hhe_kernels.so'])
        for name,digest in record['files'].items():
            self.assertEqual(hashlib.sha256((self.run/name).read_bytes()).hexdigest(),digest)
    def test_existing_run_not_rewritten(self):
        self.prepare()
        old=(self.run/'input.par').read_bytes()
        self.prepare(False)
        self.assertEqual(old,(self.run/'input.par').read_bytes())
    def test_missing_tables_no_partial_run(self):
        (self.tables/'cooling_functions').rmdir()
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_missing_legacy_recombination_tables_allowed(self):
        legacy_tables=self.tables/'recombination_tables'
        legacy_tables.rmdir()
        self.prepare()
        config=(self.run/'input.par').read_text()
        self.assertIn('Flag_PhotonConservation : 3',config)
        self.assertIn('Flag_IncludeRecombinations : 1',config)
        self.assertIn(f'RecombinationDir : {legacy_tables}',config)
        self.assert_pbs_launch()
        self.assertFalse(legacy_tables.exists())
    def test_binary_mismatch_no_partial_run(self):
        (self.build/'bin/meraxes').write_text('wrong binary')
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_native_library_mismatch_no_partial_run(self):
        self.native_library.write_bytes(b'changed native kernel')
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_missing_native_library_no_partial_run(self):
        self.native_library.unlink()
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_snapshot_list_validation(self):
        (self.sim/'a_list.txt').write_text('0.1\n0.05\n')
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_duplicate_simulation_parameter_rejected(self):
        with self.sim_params.open('a') as f:f.write('BoxSize : 10\n')
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_explicit_sed_copied(self):
        spectrum=self.base/'spectrum.txt'
        spectrum.write_text('18 0.9\n30 0.09\n60 0.01\n')
        self.env['ES_SPECTRUM']=str(spectrum)
        self.prepare()
        self.assertEqual((self.run/'provenance/source_spectrum.txt').read_bytes(),spectrum.read_bytes())
    def test_stale_l10_input_missing_file_is_ignored(self):
        missing=self.base/'nonexistent_l210_input.par'
        self.env['L10_INPUT']=str(missing)
        self.prepare()
        self.assert_l10_configuration()
        self.assertFalse(missing.exists())
    def test_stale_l10_input_cannot_import_foreign_parameters(self):
        foreign_defaults=self.base/'foreign_defaults.par'
        foreign_defaults.write_text('SfEfficiency : 0.2222\nXrayLogLNorm : 40\nFlag_UseXrayMetallicity : 1\n')
        foreign_input=self.base/'foreign_input.par'
        foreign_input.write_text(f'DefaultsFile : {foreign_defaults}\nSimulationDir : /stale/L210\n'
                                 'BoxSize : 210\nReionGridDim : 512\nSfEfficiency : 0.1476\n'
                                 'ReionNionPhotPerBary : 6000\nUnknownForeignInput : 1\n')
        original={path:path.read_bytes() for path in [foreign_input,foreign_defaults]}
        self.env['L10_INPUT']=str(foreign_input)
        self.prepare()
        self.assert_l10_configuration()
        for path,data in original.items():
            self.assertEqual(path.read_bytes(),data)
        for path in [self.run/'input.par',self.run/'provenance/defaults.par',self.run/'run_manifest.json']:
            for foreign_name in ['XrayLogLNorm','Flag_UseXrayMetallicity','UnknownForeignInput',str(foreign_input),str(foreign_defaults)]:
                self.assertNotIn(foreign_name,path.read_text())
        self.assert_pbs_launch()
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_keeps_original_es_and_rejects_removed_pc_modes(self):
        # prepare() first verifies the valid mode-3 input with the real parser.
        # Check sibling parameter files directly: PBS provenance hashes and
        # mock mpirun must not substitute for this native mode-selection test.
        self.prepare()
        original = (self.run/'input.par').read_text()
        for mode in (0, 1, 2):
            with self.subTest(mode=mode):
                config, replacements = re.subn(
                    r'(?m)^Flag_PhotonConservation\s*:\s*3\s*$',
                    f'Flag_PhotonConservation : {mode}', original)
                self.assertEqual(replacements, 1)
                parameter_file = self.run/f'check_mode_{mode}.par'
                parameter_file.write_text(config)
                result = subprocess.run(
                    [str(self.run/'meraxes'), '--check-params', str(parameter_file)],
                    env=self.shell_environment(), text=True, capture_output=True)
                diagnostic = result.stdout + result.stderr
                if mode == 0:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    self.assertIn('Parameter check passed:', result.stdout)
                    self.assertRegex(result.stdout, r'\bFlag_PhotonConservation\s+0(?:\s|$)')
                else:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertIn('Flag_PhotonConservation must be 0 (original ES) or 3', diagnostic)
                    self.assertIn('modes 1 and 2 have been removed', diagnostic)
        self.assertEqual((self.run/'input.par').read_text(), original)
        self.assertFalse((self.run/'output').exists())
        self.assertFalse((self.run/'invocation.json').exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_accepts_blank_lines_and_comments(self):
        with self.sim_params.open('a') as stream:
            stream.write('   \n\t\n  # indented comment\n')
        defaults=self.defaults
        defaults.write_bytes(defaults.read_bytes().replace(b'\n',b'\r\n')
                             + b'\r\n   \r\n\t# comment in CRLF file\r\n')
        self.prepare()
        self.assertFalse((self.run/'output').exists())
        self.assert_l10_configuration()
        effective_defaults=self.run/'provenance/defaults.par'
        effective_defaults.write_bytes(effective_defaults.read_bytes()
                                       + b'\r\n   \r\n\t# native CRLF comment\r\n')
        # The prepared files themselves, with their final provenance paths,
        # must also pass. No simulation data are present in this fixture.
        result=subprocess.run([str(self.run/'meraxes'),'--check-params',str(self.run/'input.par')],
                              env=self.shell_environment(),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Parameter check passed:',result.stdout)
        self.assertFalse((self.run/'output').exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_reports_unknown_simulation_parameter(self):
        with self.sim_params.open('a') as stream:
            stream.write('UnknownESFixtureParameter : 1\n')
        result=self.prepare(False)
        self.assertIn("Unrecognised parameter 'UnknownESFixtureParameter'",result.stderr)
        self.assertIn('input.par',result.stderr)
        self.assertIn('at line ',result.stderr)
        self.assertFalse(self.run.exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_reports_unknown_default_parameter(self):
        defaults=self.defaults
        with defaults.open('a') as stream:
            stream.write('UnknownDefaultFixtureParameter : 1\n')
        result=self.prepare(False)
        self.assertIn("Unrecognised parameter 'UnknownDefaultFixtureParameter'",result.stderr)
        self.assertIn(str(defaults),result.stderr)
        self.assertIn('at line ',result.stderr)
        self.assertFalse(self.run.exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_reports_malformed_default_line(self):
        defaults=self.defaults
        line=len(defaults.read_text().splitlines())+1
        with defaults.open('a') as stream:
            stream.write('missing separator here\n')
        result=self.prepare(False)
        self.assertIn(f"Invalid parameter syntax in '{defaults}' at line {line}",result.stderr)
        self.assertFalse(self.run.exists())
    def pbs(self, **updates):
        env=self.shell_environment(**{'PBS_NCPUS':'32','PBS_JOBID':'fixture.gadi-pbs', **updates})
        return subprocess.run(['bash','--noprofile','--norc','-c','module() { :; }; export -f module; bash --noprofile --norc "$RUN_DIR/submit.pbs"'],env=env,text=True,capture_output=True)
    def assert_pbs_launch(self):
        result=self.pbs()
        log=(self.run/'run_fixture.gadi-pbs.log').read_text()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+log)
        invocation=self.run/'invocation.json'
        self.assertTrue(invocation.is_file(), 'Mock mpirun did not record a launch.\n'+result.stdout+result.stderr+log)
        command=json.loads(invocation.read_text())
        self.assertEqual(command[1:3],['-np','32'])
        self.assertEqual(command[3:7],['--bind-to','core','--map-by','slot'])
        self.assertEqual(command[-2:],[str(self.run/'meraxes'),str(self.run/'input.par')])
        environment=json.loads((self.run/'launch_environment.json').read_text())
        self.assertEqual(environment['ES_WORKER_THREADS'],'32')
        self.assertEqual(environment['ES_NATIVE_REQUIRED'],'1')
        self.assertEqual(environment['ES_NATIVE_LIBRARY'],str(self.run/'solver/libes_hhe_kernels.so'))
        self.assertEqual(environment['OMP_WAIT_POLICY'],'PASSIVE')
        self.assertEqual(environment['OMP_PROC_BIND'],'false')
        self.assertIsNone(environment['GOMP_CPU_AFFINITY'])
        self.assertIsNone(environment['OMP_PLACES'])
        for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            self.assertEqual(environment[name],'1')
        self.assertTrue((self.run/'output').is_dir())
    def test_pbs_preflight_and_launch(self):
        self.prepare()
        self.env.update(OMP_PROC_BIND='true',GOMP_CPU_AFFINITY='0',OMP_PLACES='{0}')
        self.assert_pbs_launch()
        self.assertNotEqual(self.pbs().returncode,0)
    def test_pbs_worker_thread_override(self):
        self.prepare()
        result=self.pbs(ES_WORKER_THREADS='4',PBS_NNODES='1')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+(self.run/'run_fixture.gadi-pbs.log').read_text())
        environment=json.loads((self.run/'launch_environment.json').read_text())
        self.assertEqual(environment['ES_WORKER_THREADS'],'4')
    def test_pbs_rejects_invalid_worker_threads(self):
        self.prepare()
        for count in ['0','-1','33','1.5','invalid']:
            with self.subTest(count=count):
                result=self.pbs(ES_WORKER_THREADS=count)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse((self.run/'invocation.json').exists())
                (self.run/'.running').rmdir()
    def test_pbs_rejects_multiple_nodes(self):
        self.prepare()
        self.assertNotEqual(self.pbs(PBS_NNODES='2').returncode,0)
        self.assertFalse((self.run/'invocation.json').exists())
    def test_pbs_rejects_multiple_hosts_in_nodefile(self):
        self.prepare()
        nodefile=self.base/'nodefile'
        nodefile.write_text('node-a\nnode-b\n')
        self.assertNotEqual(self.pbs(PBS_NODEFILE=str(nodefile)).returncode,0)
        self.assertFalse((self.run/'invocation.json').exists())
    def test_pbs_accepts_repeated_single_host_in_nodefile(self):
        self.prepare()
        nodefile=self.base/'nodefile'
        nodefile.write_text('node-a\n'*32)
        result=self.pbs(PBS_NODEFILE=str(nodefile))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+(self.run/'run_fixture.gadi-pbs.log').read_text())
    def test_shell_startup_file_cannot_override_mocks(self):
        startup=self.base/'startup.bash'
        marker=self.base/'startup-ran'
        startup.write_text('printf startup > "$STARTUP_MARKER"\nmpirun() { :; }; export -f mpirun\n')
        self.env.update(BASH_ENV=str(startup), ENV=str(startup), STARTUP_MARKER=str(marker))
        self.prepare()
        self.assert_pbs_launch()
        self.assertFalse(marker.exists())
        self.assertEqual(self.env['BASH_ENV'],str(startup))
    def test_exported_functions_cannot_override_mocks(self):
        self.env['BASH_FUNC_mpirun%%']='() {  :\n}'
        self.env['BASH_FUNC_module%%']='() {  exit 96\n}'
        self.env['BASH_FUNC_git%%']='() {  exit 95\n}'
        self.prepare()
        self.assert_pbs_launch()
        self.assertIn('BASH_FUNC_mpirun%%',self.env)
    def test_pbs_detects_mutation(self):
        self.prepare()
        with (self.run/'input.par').open('a') as f:f.write('# changed\n')
        self.assertNotEqual(self.pbs().returncode,0)
        self.assertFalse((self.run/'invocation.json').exists())
        self.assertIn('Prepared file changed', (self.run/'run_fixture.gadi-pbs.log').read_text())
    def test_pbs_detects_alist_mutation(self):
        self.prepare()
        (self.sim/'a_list.txt').write_text('0.1\n0.2\n')
        self.assertNotEqual(self.pbs().returncode,0)
        self.assertFalse((self.run/'invocation.json').exists())
    def test_pbs_detects_native_library_mutation(self):
        self.prepare()
        (self.run/'solver/libes_hhe_kernels.so').write_bytes(b'changed prepared native kernel')
        self.assertNotEqual(self.pbs().returncode,0)
        self.assertFalse((self.run/'invocation.json').exists())
        self.assertIn('Prepared file changed: solver/libes_hhe_kernels.so',
                      (self.run/'run_fixture.gadi-pbs.log').read_text())

if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(
        verbosity=2, resultclass=TimedTextTestResult))
