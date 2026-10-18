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
import unittest

SOURCE = Path(__file__).resolve().parents[2]
HEAD = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
NATIVE_BINARY = os.environ.get('ES_PARAMCHECK_BINARY')

class Scripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='es-l10-script-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base/'repo'
        self.build = self.base/'build'
        self.sim = self.base/'simulation'
        self.tables = self.base/'tables'
        for directory in [self.repo/'input/params/simulations', self.repo/'tools/es_hhe', self.build/'bin', self.sim/'trees', self.sim/'grids', self.base/'commands']:
            directory.mkdir(parents=True)
        for directory in ['photometric_tables','cooling_functions','stellar_feedback_tables/Kroupa','21cmFAST-tables','recombination_tables']:
            (self.tables/directory).mkdir(parents=True)
        (self.sim/'a_list.txt').write_text('0.02\n0.04\n0.08\n')
        shutil.copy2(SOURCE/'input/params/simulations/Genesis_L10_N512.par',self.repo/'input/params/simulations/Genesis_L10_N512.par')
        shutil.copy2(SOURCE/'input/params/defaults.par',self.repo/'input/params/defaults.par')
        shutil.copy2(SOURCE/'tools/gadi_es_l10.pbs',self.repo/'tools/gadi_es_l10.pbs')
        (self.repo/'tools/es_hhe/snapshot_worker.py').write_text('# fixture\n')
        shutil.copy2(SOURCE/'input/params/defaults.par',self.base/'defaults.par')
        self.template = self.base/'template.par'
        self.template.write_text(f'DefaultsFile : {self.base}/defaults.par\nOutputSnapshots : 60:120\nSimulationDir : /stale/l210\nBoxSize : 210\nReionGridDim : 512\nSfEfficiency : 0.1476\nReionNionPhotPerBary : 6000\nTargetSnaps : 48,58,75,119\n')
        binary=self.build/'bin/meraxes'
        if NATIVE_BINARY:
            shutil.copy2(NATIVE_BINARY,binary)
        else:
            binary.write_text('#!/bin/sh\nif [ "$#" -eq 2 ] && [ "$1" = "--check-params" ]; then exit 0; fi\necho "Fixture binary must not execute a simulation; expected mock mpirun" >&2\nexit 97\n')
        binary.chmod(0o755)
        (self.build/'CMakeCache.txt').write_text('CALC_MAGS:BOOL=OFF\nUSE_MINI_HALOS:BOOL=OFF\n')
        (self.build/'pc_build_manifest.json').write_text(json.dumps({'commit':HEAD,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}))
        git=self.base/'commands/git'
        git.write_text('#!/usr/bin/env python3\nimport sys\nif "rev-parse" in sys.argv: print('+repr(HEAD)+')\n')
        git.chmod(0o755)
        mpirun=self.base/'commands/mpirun'
        mpirun.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["RUN_DIR"],"invocation.json").write_text(json.dumps(sys.argv))\n')
        mpirun.chmod(0o755)
        self.run=self.base/'run'
        self.env={**os.environ,'PC_REPO':str(self.repo),'PC_BUILD_DIR':str(self.build),'L10_INPUT':str(self.template),'L10_SIMULATION_DIR':str(self.sim),'L10_TABLES':str(self.tables),'ES_PYTHON':sys.executable,'RUN_DIR':str(self.run),'PATH':str(self.base/'commands')+os.pathsep+os.environ['PATH']}
        for name in ['ES_SPECTRUM','ES_RADIUS_RATIO','ES_MAX_STEP_MYR']:
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
    def test_prepare_preserves_calibration_all_snapshots(self):
        result=self.prepare()
        self.assertIn('Native parameter check: passed',result.stdout)
        self.assertFalse((self.run/'output').exists())
        config=(self.run/'input.par').read_text()
        for value in ['OutputSnapshots : :','TargetSnaps : :','BoxSize : 10.0','ReionGridDim : 32','SfEfficiency : 0.1476','Flag_PhotonConservation : 3','Flag_IncludeRecombinations : 1','Flag_Compute21cmBrightTemp : 0']:
            self.assertIn(value,config)
        record=json.loads((self.run/'run_manifest.json').read_text())
        self.assertEqual(record['snapshot_count'],3)
        self.assertEqual(record['commit'],HEAD)
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
    def test_snapshot_list_validation(self):
        (self.sim/'a_list.txt').write_text('0.1\n0.05\n')
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_duplicate_template_rejected(self):
        with self.template.open('a') as f:f.write('BoxSize : 10\n')
        self.prepare(False)
        self.assertFalse(self.run.exists())
    def test_explicit_sed_copied(self):
        spectrum=self.base/'spectrum.txt'
        spectrum.write_text('18 0.9\n30 0.09\n60 0.01\n')
        self.env['ES_SPECTRUM']=str(spectrum)
        self.prepare()
        self.assertEqual((self.run/'provenance/source_spectrum.txt').read_bytes(),spectrum.read_bytes())
    def test_foreign_defaults_preserve_calibration_and_fill_missing_values(self):
        defaults=self.base/'defaults.par'
        text=defaults.read_text()
        text=re.sub(r'(?m)^SfEfficiency\s*:.*$', 'SfEfficiency : 0.2222', text)
        text=re.sub(r'(?m)^SnReheatEff\s*:.*$', 'SnReheatEff : 7.123', text)
        text=re.sub(r'(?m)^SnMetalRetentionFraction\s*:.*\n', '', text)
        text+='Flag_UseXrayMetallicity : 1\nFFTW3WisdomDir : /fixture/wisdom\n'
        original=text.replace('\n','\r\n').encode()
        defaults.write_bytes(original)
        result=self.prepare()
        self.assertIn('Omitted inactive legacy default Flag_UseXrayMetallicity=1',result.stdout)
        config=(self.run/'input.par').read_text()
        effective=(self.run/'provenance/defaults.par').read_text()
        self.assertIn('SfEfficiency : 0.1476',config)
        self.assertIn('SfEfficiency : 0.2222',effective)
        self.assertIn('SnReheatEff : 7.123',effective)
        self.assertIn('FFTW3WisdomDir : /fixture/wisdom',effective)
        self.assertNotIn('Flag_UseXrayMetallicity',effective)
        self.assertEqual((self.run/'provenance/original_defaults.par').read_bytes(),original)
        self.assertEqual(defaults.read_bytes(),original)
        record=json.loads((self.run/'provenance/defaults_compatibility.json').read_text())
        self.assertEqual(record['preserved_legacy_values']['SnReheatEff'],'7.123')
        self.assertEqual(record['added_branch_values']['SnMetalRetentionFraction'],'0.0')
        self.assertEqual(record['omitted_inactive_values']['Flag_UseXrayMetallicity']['value'],'1')
        if NATIVE_BINARY:
            check=subprocess.run([str(self.run/'meraxes'),'--check-params',str(self.run/'input.par')],
                                 env=self.shell_environment(),text=True,capture_output=True)
            self.assertEqual(check.returncode,0,check.stdout+check.stderr)
            for key,value in [('SfEfficiency','0.1476'),('SnReheatEff','7.123'),
                              ('SnMetalRetentionFraction','0'),('FFTW3WisdomDir','/fixture/wisdom')]:
                self.assertRegex(check.stdout,rf'\b{key}\s+{re.escape(value)}(?:\s|$)')
        self.assert_pbs_launch()
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_foreign_option_in_explicit_input_is_not_silently_removed(self):
        with self.template.open('a') as stream:
            stream.write('Flag_UseXrayMetallicity : 1\n')
        result=self.prepare(False)
        self.assertIn("Unrecognised parameter 'Flag_UseXrayMetallicity'",result.stderr)
        self.assertFalse(self.run.exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_accepts_blank_lines_and_comments(self):
        with self.template.open('a') as stream:
            stream.write('   \n\t\n  # indented comment\n')
        defaults=self.base/'defaults.par'
        defaults.write_bytes(defaults.read_bytes().replace(b'\n',b'\r\n')
                             + b'\r\n   \r\n\t# comment in CRLF file\r\n')
        self.prepare()
        self.assertFalse((self.run/'output').exists())
        effective_defaults=self.run/'provenance/defaults.par'
        effective_defaults.write_bytes(effective_defaults.read_bytes().replace(b'\n',b'\r\n')
                                       + b'\r\n   \r\n\t# native CRLF comment\r\n')
        # The prepared files themselves, with their final provenance paths,
        # must also pass. No simulation data are present in this fixture.
        result=subprocess.run([str(self.run/'meraxes'),'--check-params',str(self.run/'input.par')],
                              env=self.shell_environment(),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Parameter check passed:',result.stdout)
        self.assertFalse((self.run/'output').exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_reports_unknown_template_parameter(self):
        with self.template.open('a') as stream:
            stream.write('UnknownESFixtureParameter : 1\n')
        result=self.prepare(False)
        self.assertIn("Unrecognised parameter 'UnknownESFixtureParameter'",result.stderr)
        self.assertIn('input.par',result.stderr)
        self.assertIn('at line ',result.stderr)
        self.assertFalse(self.run.exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_reports_unknown_default_parameter(self):
        defaults=self.base/'defaults.par'
        with defaults.open('a') as stream:
            stream.write('UnknownDefaultFixtureParameter : 1\n')
        result=self.prepare(False)
        self.assertIn("Unrecognised parameter 'UnknownDefaultFixtureParameter'",result.stderr)
        self.assertIn(f'legacy defaults from {defaults}',result.stderr)
        self.assertIn('at line ',result.stderr)
        self.assertFalse(self.run.exists())
    @unittest.skipUnless(NATIVE_BINARY, 'Needs ES_PARAMCHECK_BINARY (provided by CTest)')
    def test_native_parser_reports_malformed_default_line(self):
        defaults=self.base/'defaults.par'
        line=len(defaults.read_text().splitlines())+1
        with defaults.open('a') as stream:
            stream.write('missing separator here\n')
        result=self.prepare(False)
        self.assertIn(f"Invalid parameter syntax in '{defaults}' at line {line}",result.stderr)
        self.assertFalse(self.run.exists())
    def pbs(self):
        env=self.shell_environment(PBS_NCPUS='32',PBS_JOBID='fixture.gadi-pbs')
        return subprocess.run(['bash','--noprofile','--norc','-c','module() { :; }; export -f module; bash --noprofile --norc "$RUN_DIR/submit.pbs"'],env=env,text=True,capture_output=True)
    def assert_pbs_launch(self):
        result=self.pbs()
        log=(self.run/'run_fixture.gadi-pbs.log').read_text()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+log)
        invocation=self.run/'invocation.json'
        self.assertTrue(invocation.is_file(), 'Mock mpirun did not record a launch.\n'+result.stdout+result.stderr+log)
        command=json.loads(invocation.read_text())
        self.assertEqual(command[1:3],['-np','32'])
        self.assertEqual(command[-2:],[str(self.run/'meraxes'),str(self.run/'input.par')])
        self.assertTrue((self.run/'output').is_dir())
    def test_pbs_preflight_and_launch(self):
        self.prepare()
        self.assert_pbs_launch()
        self.assertNotEqual(self.pbs().returncode,0)
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
