#!/usr/bin/env python3
"""Compare native H/He chemistry with the numerical reference and its ledger."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import es_hhe_reference as reference
from native_backend import NativeBackend


class NativeChemistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = os.environ.get("ES_NATIVE_LIBRARY")
        cls.temporary = None
        if filename is None:
            compiler = shutil.which("gcc")
            if compiler is None:
                raise unittest.SkipTest("Native chemistry requires a built library or GCC/OpenMP")
            cls.temporary = tempfile.TemporaryDirectory(prefix="es-native-chemistry-")
            filename = str(Path(cls.temporary.name) / "libes_hhe_kernels.so")
            source = Path(__file__).resolve().parent
            subprocess.run([compiler, "-O3", "-ffp-contract=off", "-fPIC", "-shared",
                "-fopenmp", str(source / "native_runtime.c"),
                str(source / "native_geometry.c"), str(source / "native_chemistry.c"),
                str(source / "native_joint.c"), str(source / "native_candidates.c"),
                "-lm", "-o", filename], check=True, capture_output=True, text=True)
        cls.filename = filename
        cls.native = NativeBackend(filename, 1)
        cls.python = staticmethod(getattr(reference, "_upward_cells_numpy", reference._upward_cells))

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def assert_reference(self, species, rate, exposure):
        actual = self.native.upward_cells(species, rate, exposure)
        expected = self.python(species, rate, exposure)
        for result, original in zip(actual, expected):
            # libm and NumPy may differ by an ulp in exp/expm1; subtraction
            # of a nearly exhausted survival probability magnifies its
            # relative error, while the error per initial atom stays tiny.
            np.testing.assert_allclose(result, original, rtol=4e-14,
                                       atol=4 * np.nextafter(0., 1.))
            norm = np.asarray(species).sum(axis=1)
            self.assertTrue(np.all(np.abs(result - original)
                                   <= 8 * np.finfo(float).eps * norm[:, None]
                                   + 4 * np.nextafter(0., 1.)))
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertTrue(np.all(result >= 0))
        return actual

    def test_scalar_and_vector_exposures_over_dynamic_range(self):
        rng = np.random.default_rng(121)
        species = rng.random((5000, 5)) * 10.**rng.uniform(-300, 5, (5000, 1))
        rate = 10.**rng.uniform(-300, 3, (5000, 3))
        rate[::4, 2] = rate[::4, 1]
        rate[1::4, 2] = np.nextafter(rate[1::4, 1], np.inf)
        self.assert_reference(species, rate, 1.)
        self.assert_reference(species, rate, 10.**rng.uniform(-3, 0, 5000))

    def test_equal_and_nearly_equal_helium_rates(self):
        scale = np.array([1e-300, 1e-12, 0.1, np.nextafter(.5, 0.), .5, 1., 20., 1e3])
        rate = np.column_stack((scale, scale, np.nextafter(scale, np.inf)))
        species = np.tile([1., .2, .08, .01, .02], (len(rate), 1))
        self.assert_reference(species, rate, 1.)
        rate[:, 2] = rate[:, 1]
        self.assert_reference(species, rate, 1.)

    def test_zero_exposure_and_representable_tiny_capture(self):
        species = np.array([[1., .2, .08, .01, .02]])
        rate = np.array([[1e-250, 1e-100, 1e-100]])
        updated, captured = self.assert_reference(species, rate, 0.)
        np.testing.assert_array_equal(updated, species)
        np.testing.assert_array_equal(captured, 0.)
        _, captured = self.assert_reference(species, rate, 1e-50)
        self.assertGreater(captured[0, 0], 0.)
        self.assertGreater(captured[0, 1], 0.)

    def test_nuclei_and_electron_budget(self):
        rng = np.random.default_rng(65)
        species = rng.random((4096, 5))
        rate = 10.**rng.uniform(-12, 2, (len(species), 3))
        updated, captured = self.assert_reference(species, rate, .7)
        np.testing.assert_allclose(updated[:, :2].sum(axis=1), species[:, :2].sum(axis=1),
                                   rtol=5e-16, atol=0.)
        np.testing.assert_allclose(updated[:, 2:].sum(axis=1), species[:, 2:].sum(axis=1),
                                   rtol=5e-16, atol=0.)
        electron_change = (updated[:, 1] + updated[:, 3] + 2 * updated[:, 4]
                           - species[:, 1] - species[:, 3] - 2 * species[:, 4])
        self.assertTrue(np.all(np.abs(electron_change - captured.sum(axis=1))
                               <= 16 * np.finfo(float).eps * species.sum(axis=1)))

    def test_thread_count_does_not_change_arithmetic(self):
        rng = np.random.default_rng(32)
        species = rng.random((4096, 5))
        rate = 10.**rng.uniform(-20, 2, (len(species), 3))
        exposure = rng.random(len(species))
        expected = self.native.upward_cells(species, rate, exposure)
        for threads in (2, 4):
            if threads > self.native.affinity:
                continue
            parallel = NativeBackend(self.filename, threads)
            for actual, original in zip(parallel.upward_cells(species, rate, exposure), expected):
                np.testing.assert_array_equal(actual, original)

    def test_invalid_inputs_do_not_produce_a_committed_state(self):
        species = np.ones((1, 5))
        rate = np.ones((1, 3))
        for value in (-1., np.nan, np.inf):
            bad_species = species.copy()
            bad_species[0, 0] = value
            bad_rate = rate.copy()
            bad_rate[0, 1] = value
            for n, r, t in ((bad_species, rate, 1.), (species, bad_rate, 1.),
                            (species, rate, value)):
                with self.assertRaises(RuntimeError):
                    self.native.upward_cells(n, r, t)
        with self.assertRaises(ValueError):
            self.native.upward_cells(species, rate, [1., 2.])
        with self.assertRaises(RuntimeError):
            self.native.upward_cells(species, np.full((1, 3), 1e308), 2.)

    def test_overlapping_spheres_keep_shared_photon_budget(self):
        species = np.tile([1., .1, .08, .01, .001], (6, 1))
        photons = np.array([[.2, .03, .01], [.1, .02, .01], [.3, .01, .02],
                            [.02, .01, .01], [.1, .02, .03], [.2, .04, .01]])
        membership = np.array([[1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 1, 1, 1]], dtype=bool)
        results = []
        for kernel in (self.python, self.native.upward_cells):
            with mock.patch.object(reference, "_upward_cells", kernel):
                results.append(reference.joint_exposure_update(species, photons, [18., 30., 60.],
                    membership, photon_floor=1e-30, donor_fraction=.1))
        self.assertEqual(results[0]["stop"], results[1]["stop"])
        self.assertEqual(results[0]["events"], results[1]["events"])
        for key in ("species", "photons", "absorbed", "debit"):
            np.testing.assert_allclose(results[0][key], results[1][key], rtol=5e-12, atol=1e-25)
        self.assertLess(abs(results[1]["photon_error"]), 4e-12 * results[1]["absorbed_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
