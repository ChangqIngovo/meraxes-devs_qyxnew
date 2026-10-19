#!/usr/bin/env python3
"""The compiled shared midpoint loop must retain the reference ES update."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import es_hhe_reference as reference


class NativeJoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = os.environ.get("ES_NATIVE_LIBRARY")
        cls.temporary = None
        if filename is None:
            compiler = shutil.which("gcc")
            if compiler is None:
                raise unittest.SkipTest("Native joint test requires GCC/OpenMP or a built library")
            cls.temporary = tempfile.TemporaryDirectory(prefix="es-native-joint-")
            filename = str(Path(cls.temporary.name) / "libes_hhe_kernels.so")
            source = Path(__file__).resolve().parent
            subprocess.run([compiler, "-O3", "-ffp-contract=off", "-fPIC", "-shared", "-fopenmp",
                *(str(source / name) for name in ("native_runtime.c", "native_geometry.c",
                    "native_chemistry.c", "native_joint.c", "native_candidates.c")), "-lm", "-o", filename],
                check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(filename)
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ip = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        cls.call = cls.library.es_joint_compact
        cls.call.restype = ctypes.c_int
        cls.call.argtypes = ([ctypes.c_int64] * 4 + [fp] * 3 + [ip] * 2
            + [ctypes.c_double] * 4 + [ctypes.c_int64, ctypes.c_int]
            + [fp] * 4 + [ip] * 2 + [fp])

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def native(self, gas, photons, energies, membership, *, threads=1,
               epsilon=1e-5, donor_fraction=.2, count_tolerance=1e-13,
               photon_floor=1e-30, max_iterations=20000, csr=None):
        gas = np.ascontiguousarray(gas, dtype=float)
        photons = np.ascontiguousarray(photons, dtype=float)
        cross = reference.cross_sections(np.asarray(energies, dtype=float))
        if np.any(cross):
            cross /= cross.max()
        if csr is None:
            indices = np.concatenate([np.flatnonzero(row) for row in membership]).astype(np.int64)
            offsets = np.r_[0, np.cumsum(np.sum(membership, axis=1))].astype(np.int64)
        else:
            indices, offsets = csr
        output = [np.empty_like(gas), np.empty_like(photons),
                  np.empty((len(gas), 3)), np.empty_like(photons)]
        events = np.zeros((len(offsets)-1, 3), dtype=np.int64)
        integers, reals = np.zeros(3, dtype=np.int64), np.zeros(1)
        status = self.call(len(gas), photons.shape[1], len(offsets)-1, len(indices),
            gas, photons, cross, indices, offsets, epsilon, donor_fraction,
            count_tolerance, photon_floor, max_iterations, threads, *output, events, integers, reals)
        return status, dict(zip(("species", "photons", "absorbed", "debit"), output),
            stop=("no_absorption", "photon_exhaustion", "absorber_saturation",
                  "numerical_remainder", "crossing")[integers[0]],
            iterations=integers[1], substeps=integers[2], exposure=reals[0],
            events=np.argwhere(events).tolist())

    def compare(self, gas, photons, energies, membership, **options):
        original_gas, original_photons = gas.copy(), photons.copy()
        indices = np.concatenate([np.flatnonzero(row) for row in membership]).astype(np.int64)
        offsets = np.r_[0, np.cumsum(np.sum(membership, axis=1))].astype(np.int64)
        operator = reference._CompactSphereOperator(indices, offsets)
        self.assertEqual(operator.shape[1], len(gas))
        defaults = dict(donor_fraction=.2, photon_floor=1e-30)
        defaults.update(options)
        with mock.patch.object(reference, "get_backend", return_value=None):
            expected = reference.joint_exposure_update(gas, photons, energies, operator, **defaults)
        status, actual = self.native(gas, photons, energies, membership, **defaults)
        self.assertEqual(status, 0)
        self.assertEqual(actual["stop"], expected["stop"])
        self.assertEqual(actual["events"], expected["events"])
        # Cross-section contractions may differ by a few ulps from BLAS;
        # require agreement far inside the midpoint integration tolerance.
        for name in ("species", "photons", "absorbed", "debit"):
            if name == "photons":
                # Feasibility comparisons can select a different final
                # half-step once an account is below its convergence scale.
                # Retained credits remain explicit; compare at the solver's
                # existing per-account relative tolerance, not their tiny
                # terminal values' relative error.
                allowance = (2e-10*np.abs(expected[name]) + 2e-25
                             + defaults.get("count_tolerance", 1e-13)*photons)
                self.assertTrue(np.all(np.abs(actual[name]-expected[name]) <= allowance))
            else:
                np.testing.assert_allclose(actual[name], expected[name], rtol=2e-10, atol=2e-25)
            self.assertTrue(np.all(np.isfinite(actual[name])))
            self.assertTrue(np.all(actual[name] >= 0.))
        np.testing.assert_array_equal(gas, original_gas)
        np.testing.assert_array_equal(photons, original_photons)
        np.testing.assert_allclose(actual["species"][:, :2].sum(1), gas[:, :2].sum(1), rtol=4e-12)
        np.testing.assert_allclose(actual["species"][:, 2:].sum(1), gas[:, 2:].sum(1), rtol=4e-12)
        capture = actual["absorbed"].sum()
        self.assertLessEqual(abs(actual["debit"].sum()-capture), 4e-12*max(capture, np.finfo(float).tiny))
        np.testing.assert_allclose(actual["photons"]+actual["debit"], photons,
                                   rtol=4e-12, atol=4*np.nextafter(0., 1.))
        change = (actual["species"][:, 1]+actual["species"][:, 3]+2*actual["species"][:, 4]
                  - gas[:, 1]-gas[:, 3]-2*gas[:, 4]).sum()
        self.assertLessEqual(abs(change-capture), 4e-12*max(capture, gas.sum()))
        return actual

    @staticmethod
    def supports():
        return np.array([[1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0],
                         [0, 0, 0, 1, 1, 1]], dtype=bool)

    def test_random_overlapping_mixed_hydrogen_helium(self):
        rng = np.random.default_rng(435)
        for _ in range(5):
            gas = rng.uniform(.01, 1., (6, 5))
            photons = rng.uniform(.01, .3, (6, 3))
            self.compare(gas, photons, [18., 30., 60.], self.supports())

    def test_nine_bands_keep_duplicate_opacity_accounts_separate(self):
        rng = np.random.default_rng(817)
        gas = rng.uniform(.01, 1., (6, 5))
        photons = 10.**rng.uniform(-24., -1., (6, 9))
        photons[:, 0] = .1
        actual = self.compare(gas, photons, [18., 18., 30., 30., 60., 60., 13.7, 24.7, 54.5],
                              self.supports())
        self.assertEqual(actual["debit"].shape, photons.shape)
        self.assertTrue(np.all(actual["debit"] <= photons))

    def test_earliest_crossing_leaves_unused_photons(self):
        gas = np.tile([1., 0., .08, 0., 0.], (6, 1))
        gas[:, 0] *= [1., 1.2, .8, 1.5, .9, 1.8]
        photons = np.zeros((6, 3))
        photons[:, 0] = [4., 2., 8., 1., 1., 1.]
        actual = self.compare(gas, photons, [18., 30., 60.], self.supports(), epsilon=.01)
        self.assertEqual(actual["stop"], "crossing")
        self.assertGreater(actual["photons"].sum(), 0.)

    def test_helium_iii_crossing_and_initially_ineligible_species(self):
        gas = np.tile([0., 1., 0., .08, 0.], (6, 1))
        gas[:, 3] *= [1., 1.2, .8, 1.5, .9, 1.8]
        photons = np.full((6, 1), .3)
        actual = self.compare(gas, photons, [60.], self.supports(), epsilon=.01)
        self.assertEqual(actual["stop"], "crossing")
        self.assertTrue(all(species == 2 for _, species in actual["events"]))

    def test_zero_radius_cells_share_earliest_event_exposure(self):
        gas = np.tile([1., 0., .08, 0., 0.], (4, 1))
        photons = np.array([[2., .01, .02], [.3, .03, .01], [.1, .04, .01], [.2, .02, .05]])
        actual = self.compare(gas, photons, [18., 30., 60.], np.eye(4, dtype=bool), epsilon=.01)
        self.assertEqual(actual["stop"], "crossing")
        self.assertTrue(any(cell == 0 for cell, _ in actual["events"]))

    def test_dormant_and_subnormal_credits_are_preserved(self):
        gas = np.tile([1., 0., .08, 0., 0.], (6, 1))
        photons = np.full((6, 3), .1)
        photons[0, 0] = 3.66e-322
        photons[1, 1] = 1e-31
        photons[2, 2] = 1e-30
        actual = self.compare(gas, photons, [18., 30., 60.], self.supports())
        for i, g in ((0, 0), (1, 1), (2, 2)):
            self.assertEqual(actual["photons"][i, g], photons[i, g])
            self.assertEqual(actual["debit"][i, g], 0.)

    def test_no_absorption_and_saturation(self):
        gas = np.tile([0., 1., 0., 0., .08], (6, 1))
        actual = self.compare(gas, np.ones((6, 3)), [18., 30., 60.], self.supports())
        self.assertEqual(actual["stop"], "no_absorption")
        gas[:, 0], gas[:, 1] = 1., 0.
        actual = self.compare(gas, np.zeros((6, 3)), [18., 30., 60.], self.supports())
        self.assertEqual(actual["stop"], "no_absorption")

    def test_iteration_failure_preserves_inputs(self):
        gas = np.tile([1., 0., .08, 0., 0.], (6, 1))
        photons = np.full((6, 3), .1)
        original_gas, original_photons = gas.copy(), photons.copy()
        status, _ = self.native(gas, photons, [18., 30., 60.], self.supports(), max_iterations=1)
        self.assertEqual(status, 4)
        np.testing.assert_array_equal(gas, original_gas)
        np.testing.assert_array_equal(photons, original_photons)

    def test_invalid_and_unsupported_input(self):
        gas = np.tile([1., 0., .08, 0., 0.], (6, 1))
        photons = np.full((6, 3), .1)
        status, _ = self.native(gas, photons, [18., 30., 60.], np.ones((1, 6), dtype=bool))
        self.assertEqual(status, 100)
        gas[0, 0] = np.nan
        status, _ = self.native(gas, photons, [18., 30., 60.], self.supports())
        self.assertEqual(status, 2)
        gas[0, 0] = 1.
        status, _ = self.native(gas, photons, [18., 30., 60.], self.supports(),
            csr=(np.array([0, 0, 1], dtype=np.int64), np.array([0, 2, 3], dtype=np.int64)))
        self.assertEqual(status, 1)

    def test_parallel_thread_count_retains_identical_state(self):
        if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < 2:
            self.skipTest("Only one CPU is available")
        cells = 1024
        gas = np.tile([1., 0., .08, 0., 0.], (cells, 1))
        photons = np.full((cells, 3), .02)
        photons[0, 0] = 3.
        membership = np.eye(cells, dtype=bool)
        status, expected = self.native(gas, photons, [18., 30., 60.], membership,
            epsilon=.1, donor_fraction=.4, threads=1)
        self.assertEqual(status, 0)
        status, actual = self.native(gas, photons, [18., 30., 60.], membership,
            epsilon=.1, donor_fraction=.4, threads=2)
        self.assertEqual(status, 0)
        for name in ("species", "photons", "absorbed", "debit"):
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual(actual["events"], expected["events"])
        self.assertEqual(actual["exposure"], expected["exposure"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
