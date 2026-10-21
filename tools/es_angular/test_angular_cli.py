"""CLI I/O and fail-closed controls, independent of the production worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


FOLDER = Path(__file__).resolve().parent


class AngularCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="angular_cli_")
        cls.folder = Path(cls.temporary.name)
        cls.library = os.environ.get("ES_ANGULAR_LIBRARY")
        if not cls.library:
            cls.library = str(cls.folder / "libangular_front.so")
            subprocess.run([sys.executable, str(FOLDER / "build.py"),
                            "--output", cls.library], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.input = self.folder / (self._testMethodName + "_input.npz")
        self.output = self.folder / (self._testMethodName + "_output.npz")
        self.data = dict(capacity=np.ones((1, 1, 1)), initial_mass=np.zeros((1, 1, 1)),
                         source_indices=np.array([[0, 0, 0]], dtype=np.int32), source_rates=np.ones(1))

    def invoke(self, *args, library=None):
        np.savez(self.input, **self.data)
        return subprocess.run([sys.executable, str(FOLDER / "run.py"), "--input", str(self.input),
                               "--output", str(self.output), "--library", library or self.library,
                               *args], capture_output=True, text=True)

    def test_analytic_roundtrip_and_metadata(self):
        result = self.invoke("--times", "0,.25", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        with np.load(self.output, allow_pickle=False) as out:
            np.testing.assert_allclose(out["mass"].ravel(), [0., .25, 1.])
            np.testing.assert_allclose(out["resource_escaped"].sum(axis=(1, 2)), [0., 0., 1.])
            np.testing.assert_allclose(out["source_mass"].sum(axis=(1, 2, 3, 4)), [0., .25, 1.])
            np.testing.assert_allclose(out["resource_balance_error"], 0., atol=1e-14)
            metadata = json.loads(str(out["metadata_json"]))
            self.assertEqual(metadata["boundary"], "open")
            self.assertFalse(metadata["recombination"])
            self.assertEqual(metadata["production_mode3"], "retired")
            self.assertEqual(len(out["diagnostics_json"]), 3)
        sidecar = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(len(sidecar["diagnostics"]), 3)

    def test_external_score_and_finite_shape_checks(self):
        self.data["es_score"] = np.ones((1, 1, 1)) * 1e300
        result = self.invoke("--times", ".2", "--epsilon", ".1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.output.unlink()
        for score in (np.array([1.]), np.ones((1, 1, 1)) * np.nan, -np.ones((1, 1, 1))):
            with self.subTest(score=repr(score)):
                self.data["es_score"] = score
                result = self.invoke("--epsilon", ".1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("es_score", result.stderr)
                self.assertFalse(self.output.exists())

    def test_invalid_controls_and_fractional_source_fail(self):
        for options in (("--times", "1", ".5"), ("--times", "nan"), ("--dx", "0"),
                        ("--level", "9"), ("--epsilon", ".1"), ("--uniform-density", "2")):
            with self.subTest(options=options):
                result = self.invoke(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
        self.data["source_indices"] = np.array([[.5, 0, 0]])
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("integer coordinates", result.stderr)

    def test_missing_library_does_not_build(self):
        missing = self.folder / "not_built.so"
        result = self.invoke(library=str(missing))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicitly", result.stderr)
        self.assertFalse(missing.exists())

    def test_reproducible_example_needs_no_library(self):
        path = self.folder / "example.npz"
        result = subprocess.run([sys.executable, str(FOLDER / "run.py"),
                                 "--write-example", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["capacity"].shape, (9, 9, 9))
            self.assertEqual(data["initial_mass"].sum(), 0)
            self.assertEqual(data["source_rates"].sum(), 8)


if __name__ == "__main__":
    unittest.main()
