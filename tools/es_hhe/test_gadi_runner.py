#!/usr/bin/env python3
"""Exercise the L10 preparation/PBS boundary without Gadi or simulation execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]
HEAD = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()

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
        shutil.copy2(SOURCE/'tools/gadi_es_l10.pbs',self.repo/'tools/gadi_es_l10.pbs')
        (self.repo/'tools/es_hhe/snapshot_worker.py').write_text('# fixture\n')
        (self.base/'defaults.par').write_text('NSteps : 1\n')
        self.template = self.base/'template.par'
        self.template.write_text(f'DefaultsFile : {self.base}/defaults.par\nOutputSnapshots : 60:120\nSimulationDir : /stale/l210\nBoxSize : 210\nReionGridDim : 512\nSfEfficiency : 0.1476\nReionNionPhotPerBary : 6000\nTargetSnaps : 48,58,75,119\n')
        binary=self.build/'bin/meraxes'
        binary.write_text('#!/bin/sh\nexit 0\n')
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
    def prepare(self,ok=True):
        result=subprocess.run(['bash',str(SOURCE/'tools/gadi_es_l10.sh'),'prepare'],env=self.env,text=True,capture_output=True)
        self.assertEqual(result.returncode==0,ok,result.stdout+result.stderr)
        return result
    def test_prepare_preserves_calibration_all_snapshots(self):
        self.prepare()
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
    def pbs(self):
        env={**self.env,'PBS_NCPUS':'32','PBS_JOBID':'fixture.gadi-pbs'}
        return subprocess.run(['bash','-c','module() { :; }; export -f module; bash "$RUN_DIR/submit.pbs"'],env=env,text=True,capture_output=True)
    def test_pbs_preflight_and_launch(self):
        self.prepare()
        result=self.pbs()
        log=(self.run/'run_fixture.gadi-pbs.log').read_text()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+log)
        command=json.loads((self.run/'invocation.json').read_text())
        self.assertEqual(command[1:3],['-np','32'])
        self.assertEqual(command[-2:],[str(self.run/'meraxes'),str(self.run/'input.par')])
        self.assertTrue((self.run/'output').is_dir())
        self.assertNotEqual(self.pbs().returncode,0)
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
