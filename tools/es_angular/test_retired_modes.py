"""Compile the actual Meraxes parameter guard and reject retired mode values.

Requires only Python's standard library and CC (default: cc). This exercises
check_problem_params from src/core/read_params.c with a minimal C context; it
is not a substitute for compiling or running the full MPI simulation.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]

# The guard is extracted from production source below, never reimplemented here.
CONTEXT = r'''
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define USE_STOCHASTICITY 0
#define ABS_TOL 1e-12
#define MLOG_MESG 0
#define mlog(...) ((void)0)
#define ABORT(status) exit(status)
static void mlog_error(const char *message, ...)
{
  va_list args;
  va_start(args, message);
  vfprintf(stderr, message, args);
  va_end(args);
  fputc('\n', stderr);
}
typedef struct {
  int Flag_PhotonConservation, NSteps, Flag_IncludeSpinTemp, FlagMCMC;
  char ForestIDFile[64];
  struct {
    double EscapeFracScatterDex, XrayScatterDex, SnMetalRetentionFraction;
    int Flag_RemoveSFRScatter, Flag_SourceRecalibration;
  } physics;
} run_params_t;
static struct { run_params_t params; } run_globals;
'''

DRIVER = r'''
int main(int argc, char **argv)
{
  if (argc != 2)
    return 2;
  if (strcmp(argv[1], "INT_MIN") == 0)
    run_globals.params.Flag_PhotonConservation = INT_MIN;
  else if (strcmp(argv[1], "INT_MAX") == 0)
    run_globals.params.Flag_PhotonConservation = INT_MAX;
  else
    run_globals.params.Flag_PhotonConservation = atoi(argv[1]);
  run_globals.params.NSteps = 1;
  run_globals.params.physics.SnMetalRetentionFraction = 0.5;
  check_problem_params(&run_globals.params);
  return 0;
}
'''


class RetiredModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "src/core/read_params.c").read_text()
        match = re.search(r"^static void check_problem_params\([^\n]*\)\s*\n\{.*?^\}",
                          source, re.MULTILINE | re.DOTALL)
        if match is None:
            raise RuntimeError("Cannot extract actual check_problem_params from read_params.c")
        cls.temporary = tempfile.TemporaryDirectory(prefix="meraxes_retired_modes_")
        cls.addClassCleanup(cls.temporary.cleanup)
        folder = Path(cls.temporary.name)
        harness = folder / "guard.c"
        cls.executable = folder / "guard"
        harness.write_text(CONTEXT + match.group(0) + "\n" + DRIVER)
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler:
            raise RuntimeError("CC must contain a C compiler command")
        subprocess.run(compiler + ["-std=c99", "-Wall", "-Wextra", "-Werror",
                                   str(harness), "-o", str(cls.executable)], check=True)

    def test_baseline_accepts_and_retired_modes_abort(self):
        for mode in ("0", "-1", "1", "2", "3", "4", "INT_MIN", "INT_MAX"):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.executable), mode],
                                        capture_output=True, text=True)
                if mode == "0":
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertIn("Flag_PhotonConservation must be 0", result.stderr)
                    self.assertIn("angular method runs independently", result.stderr)
                    self.assertIn("not yet an online periodic H/He backend", result.stderr)


if __name__ == "__main__":
    unittest.main()
