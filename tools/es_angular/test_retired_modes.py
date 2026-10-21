"""Exercise the production mode guard and incremental stellar source update.

Requires Python's standard library and CC (default: cc). Functions are extracted
from production C and compiled against the actual meraxes.h structures. Only
external library type declarations are stubbed; this is not a full MPI build.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]

CONTEXT = r'''
#include "meraxes.h"
#include "core/misc_tools.h"
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
run_globals_t run_globals;
void myexit(int status) { exit(status); }
void mlog_error(const char *message, ...)
{
  va_list args;
  va_start(args, message);
  vfprintf(stderr, message, args);
  va_end(args);
  fputc('\n', stderr);
}
void mlog(const char *message, int flags, ...)
{
  (void)message;
  (void)flags;
}
'''

# These functions use no FFTW, GSL, HDF5 or MPI calls. Type-only declarations let
# the real Meraxes structs compile without requiring those development packages.
HEADERS = {
    "fftw3.h": "#include <stddef.h>\ntypedef float fftwf_complex[2];\ntypedef void *fftwf_plan;\n",
    "gsl/gsl_rng.h": "typedef struct gsl_rng gsl_rng;\n",
    "hdf5.h": "typedef long hid_t;\n",
    "mlog.h": """#include <stdbool.h>
typedef int MPI_Comm;
#define MLOG_MESG 0
void mlog_error(const char *, ...);
void mlog(const char *, int, ...);
""",
    "meraxes_conf.h": """#define N_HISTORY_SNAPS 50
#define USE_STOCHASTICITY 0
#define MERAXES_GITREF_STR "guard-test"
#define MERAXES_GITDIFF_STR ""
#define FEATURE_FLAG_CALC_MAGS 0
""",
}


def extract_function(path, declaration):
    source = (ROOT / path).read_text()
    match = re.search(r"^" + re.escape(declaration) + r"\([^\n]*\)\s*\n\{.*?^\}",
                      source, re.MULTILINE | re.DOTALL)
    if match is None:
        raise RuntimeError(f"Cannot extract production {declaration} from {path}")
    return match.group(0)


def compile_harness(test_class, name, function, driver):
    temporary = tempfile.TemporaryDirectory(prefix=f"meraxes_{name}_")
    test_class.addClassCleanup(temporary.cleanup)
    folder = Path(temporary.name)
    for path, contents in HEADERS.items():
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    harness = folder / f"{name}.c"
    executable = folder / name
    harness.write_text(CONTEXT + function + "\n" + driver)
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler:
        raise RuntimeError("CC must contain a C compiler command")
    subprocess.run(compiler + ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                               "-fno-fast-math", "-ffp-contract=off",
                               "-I" + str(folder), "-I" + str(ROOT / "src"),
                               str(harness), "-lm", "-o", str(executable)], check=True)
    return executable


VALID_ANGULAR_PARAMS = r'''
  run_params_t *p = &run_globals.params;
  p->Flag_PhotonConservation = 4;
  p->NSteps = 1;
  p->physics.SnMetalRetentionFraction = 0.5;
  p->Flag_PatchyReion = 1;
  p->Flag_IncludeRecombinations = 1;
  p->ReionGridDim = 8;
  p->AngularInitialTemperature = 10000;
  p->physics.Y_He = 0.24;
  p->physics.ReionNionPhotPerBary = 4000;
  p->AngularSourceEnergyEV = 18;
  p->AngularMaxStepMyr = 0.1;
  p->AngularEpsilon = 0.1;
  p->AngularSpeedFraction = 1;
  strcpy(p->AngularPythonExecutable, "/usr/bin/python3");
  strcpy(p->AngularWorkerPath, "/tmp/worker.py");
  strcpy(p->AngularLibraryPath, "/tmp/native.so");
'''

INVALID_MODES = ("-1", "1", "2", "3", "5", "INT_MIN", "INT_MAX")
INVALID_CONFIGS = {
    "nan_step": "p->AngularMaxStepMyr = NAN;",
    "nan_energy": "p->AngularSourceEnergyEV = NAN;",
    "nan_epsilon": "p->AngularEpsilon = NAN;",
    "nan_speed": "p->AngularSpeedFraction = NAN;",
    "bad_temperature": "p->AngularInitialTemperature = 20000;",
    "nan_temperature": "p->AngularInitialTemperature = NAN;",
    "bad_level": "p->AngularLevel = 9;",
    "no_recomb": "p->Flag_IncludeRecombinations = 0;",
    "relative_library": 'strcpy(p->AngularLibraryPath, "lib.so");',
    "relative_spectrum": 'strcpy(p->AngularSourceSpectrumFile, "spectrum.txt");',
    "spin": "p->Flag_IncludeSpinTemp = 1;",
    "uvb": "p->ReionUVBFlag = 1;",
    "agn": "p->physics.EscapeFracBHNorm = 0.1;",
    "speed_superluminal": "p->AngularSpeedFraction = 1.1;",
    "speed_zero": "p->AngularSpeedFraction = 0;",
    "no_patchy": "p->Flag_PatchyReion = 0;",
    "negative_source": "p->physics.ReionNionPhotPerBary = -1;",
}


class RetiredModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        driver = "int main(int argc, char **argv) {\n  if (argc != 2) return 2;\n"
        driver += VALID_ANGULAR_PARAMS
        for mode in ("0", "4") + INVALID_MODES:
            driver += (f'  if (strcmp(argv[1], "{mode}") == 0) '
                       f'p->Flag_PhotonConservation = {mode};\n')
        for name, mutation in INVALID_CONFIGS.items():
            driver += f'  if (strcmp(argv[1], "{name}") == 0) {{ {mutation} }}\n'
        driver += "  check_problem_params(p);\n  return 0;\n}\n"
        cls.executable = compile_harness(
            cls, "guard", extract_function("src/core/read_params.c", "static void check_problem_params"), driver)

    def test_baseline_and_angular_modes_accept(self):
        for mode in ("0", "4"):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.executable), mode], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_retired_and_unknown_modes_abort(self):
        for mode in INVALID_MODES:
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.executable), mode], capture_output=True, text=True)
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertIn("must be 0 (original ES) or 4", result.stderr)
                self.assertIn("modes 1, 2 and 3 have been removed", result.stderr)

    def test_invalid_angular_configs_abort(self):
        for name in INVALID_CONFIGS:
            with self.subTest(config=name):
                result = subprocess.run([str(self.executable), name], capture_output=True, text=True)
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertIn("Angular mode", result.stderr)


SOURCE_DRIVER = r'''
int main(int argc, char **argv)
{
  if (argc != 2)
    return 2;
  double zz[] = {6};
  run_globals.ZZ = zz;
  run_globals.params.Flag_PhotonConservation = strcmp(argv[1], "baseline") == 0 ? 0 : 4;
  run_globals.params.physics.EscapeFracNorm = 0.2;
  galaxy_t gal = {0};
  gal.dt = 2;
  gal.Sfr = 3;
  double new_stars = 6;
  if (strcmp(argv[1], "negative_stars") == 0) new_stars = -1;
  if (strcmp(argv[1], "nan_stars") == 0) new_stars = NAN;
  if (strcmp(argv[1], "bad_timestep") == 0) gal.dt = 0;
  if (strcmp(argv[1], "nan_timestep") == 0) gal.dt = NAN;
  if (strcmp(argv[1], "overflow") == 0) { new_stars = 1e300; gal.dt = 1e-300; }
  if (strcmp(argv[1], "zero_event") == 0) {
    update_galaxy_fesc_vals(&gal, 0, 0);
    return gal.FescWeightedSfr == 0 && gal.FescWeightedGSM == 0 ? 0 : 3;
  }
  update_galaxy_fesc_vals(&gal, new_stars, 0);
  gal.Sfr += 4;
  update_galaxy_fesc_vals(&gal, 8, 0);
  // Two events form 6 and 8 mass units over dt=2 with fesc=0.2.
  // Their total escaped rate is 1.4. Mode zero retains its historical behavior.
  const double expected_rate = run_globals.params.Flag_PhotonConservation == 4 ? 1.4 : 2.0;
  if (fabs(gal.FescWeightedSfr - expected_rate) > 1e-12)
    return 3;
  if (fabs(gal.FescWeightedGSM - 2.8) > 1e-12)
    return 4;
  return 0;
}
'''


class IncrementalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executable = compile_harness(
            cls, "source", extract_function("src/core/reionization.c", "void update_galaxy_fesc_vals"), SOURCE_DRIVER)

    def test_two_star_formation_events_and_baseline(self):
        for case in ("angular", "baseline", "zero_event"):
            with self.subTest(case=case):
                result = subprocess.run([str(self.executable), case], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_star_formation_events_abort(self):
        for case in ("negative_stars", "nan_stars", "bad_timestep", "nan_timestep", "overflow"):
            with self.subTest(case=case):
                result = subprocess.run([str(self.executable), case], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Angular", result.stderr)


if __name__ == "__main__":
    unittest.main()
