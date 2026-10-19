#!/usr/bin/env python3
"""Candidate masks preserve the reference, including uncertain boundaries."""
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
import native_backend as backend


def numpy_flags(aggregate, cross, epsilon, count_pruning=True):
    n, p = aggregate[:, :5], aggregate[:, 5:]
    normal = np.column_stack((n[:, :2].sum(1), n[:, 2:].sum(1), n[:, 2:].sum(1)))
    remaining = np.column_stack((n[:, 0], n[:, 2], n[:, 2]+n[:, 3]))
    needed = remaining-epsilon*normal
    eligible = (normal > 0) & (remaining > epsilon*normal+64*np.finfo(float).eps*normal)
    hazard = n[:, [0, 2, 3]] @ cross.T
    usable = np.any((p > 0) & (hazard > 0), axis=1)
    valid = np.any(eligible, axis=1) & usable
    partial = np.zeros(len(n), dtype=bool)
    if count_pruning:
        upper = p @ (cross > 0)
        margin = 64*np.finfo(float).eps*np.maximum(upper, remaining)
        possible = np.any(eligible & (upper+margin >= needed), axis=1)
        partial = valid & ~possible
    return valid, partial


class NativeCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = os.environ.get('ES_NATIVE_LIBRARY')
        cls.temporary = None
        if filename is None:
            compiler = shutil.which('gcc')
            if compiler is None:
                raise unittest.SkipTest('Native candidate tests require GCC/OpenMP')
            cls.temporary = tempfile.TemporaryDirectory(prefix='es-native-candidates-')
            filename = str(Path(cls.temporary.name) / 'libes_hhe_kernels.so')
            source = Path(__file__).resolve().parent
            subprocess.run([compiler, '-O3', '-fno-fast-math', '-ffp-contract=off',
                '-fPIC', '-shared', '-fopenmp',
                *(str(source / name) for name in ('native_runtime.c', 'native_geometry.c',
                    'native_chemistry.c', 'native_joint.c', 'native_candidates.c')),
                '-lm', '-o', filename], check=True, capture_output=True, text=True)
        cls.filename = filename
        cls.lib = ctypes.CDLL(filename)
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        up = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        cls.call = cls.lib.es_candidate_flags
        cls.call.argtypes = [ctypes.c_int64, ctypes.c_int64, fp, fp, ctypes.c_double,
                            ctypes.c_int, up, up, up, ctypes.c_int]
        cls.call.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def raw(self, aggregate, cross, epsilon=1e-5, count_pruning=True, threads=1):
        aggregate = np.ascontiguousarray(aggregate, dtype=float)
        cross = np.ascontiguousarray(cross, dtype=float)
        outputs = [np.empty(len(aggregate), dtype=np.uint8) for _ in range(3)]
        status = self.call(len(aggregate), len(cross), aggregate, cross, epsilon,
                           int(count_pruning), *outputs, threads)
        self.assertEqual(status, 0)
        return tuple(value.view(np.bool_) for value in outputs)

    def test_random_dynamic_range_and_parallel_masks(self):
        rng = np.random.default_rng(61)
        cross = reference.cross_sections(np.r_[18., 30., 60., reference.OPACITY_E_EV])
        cross /= cross.max()
        aggregate = 10.**rng.uniform(-320, 3, size=(16384, 5+len(cross)))
        aggregate[rng.random(aggregate.shape) < .25] = 0.
        for pruning in (False, True):
            expected = numpy_flags(aggregate, cross, 1e-5, pruning)
            masks = self.raw(aggregate, cross, count_pruning=pruning)
            for threads in (2, 4):
                for actual, one in zip(self.raw(aggregate, cross, count_pruning=pruning, threads=threads), masks):
                    np.testing.assert_array_equal(actual, one)
            certain = ~masks[2]
            for actual, wanted in zip(masks[:2], expected):
                np.testing.assert_array_equal(actual[certain], wanted[certain])

    def test_species_eligibility_boundary(self):
        epsilon = .01
        center = epsilon + 64*np.finfo(float).eps
        neutral = np.array([np.nextafter(center, 0.), center, np.nextafter(center, 1.)])
        aggregate = np.zeros((3, 6))
        aggregate[:, 0], aggregate[:, 1], aggregate[:, 5] = neutral, 1.-neutral, 3.
        cross = np.array([[1., 0., 0.]])
        actual = self.raw(aggregate, cross, epsilon)
        self.assertFalse(np.any(actual[2]))
        for result, expected in zip(actual[:2], numpy_flags(aggregate, cross, epsilon)):
            np.testing.assert_array_equal(result, expected)

    def boundary_fields(self):
        epsilon = .01
        cross = np.tile([1., 0., 0.], (9, 1))
        count = (1.-epsilon)/(1.+64*np.finfo(float).eps)/9.
        aggregate = np.zeros((7, 14))
        aggregate[:, 0] = 1.
        aggregate[:, 5:] = count + np.array([-1000., -100., -1., 0., 1., 100., 1000.])[:, None]*np.spacing(count)
        return aggregate, cross, epsilon

    def test_pruning_boundary_requests_original_numpy(self):
        aggregate, cross, epsilon = self.boundary_fields()
        native = backend.NativeBackend(self.filename, 1)
        flags = self.raw(aggregate, cross, epsilon)
        self.assertTrue(flags[2][3])
        self.assertIsNone(native.candidate_flags(aggregate, cross, epsilon, True))
        # Away from the guarded band the C classification is identical.
        certain = ~flags[2]
        for actual, expected in zip(flags[:2], numpy_flags(aggregate, cross, epsilon)):
            np.testing.assert_array_equal(actual[certain], expected[certain])

    def test_subnormal_hazard_requires_representable_absorption(self):
        tiny = np.nextafter(0., 1.)
        cross = np.diag([.25, .25, .25])
        aggregate = np.zeros((4, 8))
        aggregate[0, 0], aggregate[1, 0] = tiny, 4*tiny
        aggregate[2, 2], aggregate[3, 3] = tiny, 4*tiny
        aggregate[:, 5:] = 1.
        actual = self.raw(aggregate, cross)
        np.testing.assert_array_equal(actual[0], [False, True, False, True])
        for result, expected in zip(actual[:2], numpy_flags(aggregate, cross, 1e-5)):
            np.testing.assert_array_equal(result, expected)

    def test_selector_fallback_preserves_proposals_and_counters(self):
        def model():
            value = reference.SpectralES(2, helium=0., epsilon=.01, radii2=[0.])
            value.photons[:, 0] = .99/(1.+64*np.finfo(float).eps)
            return value
        baseline, accelerated = model(), model()
        with mock.patch.object(reference, 'get_backend', return_value=None):
            expected_kind, expected_proposals = baseline._select()
        native = backend.NativeBackend(self.filename, 1)
        with mock.patch.object(native, 'candidate_flags', wraps=native.candidate_flags) as call:
            with mock.patch.object(reference, 'get_backend', return_value=native):
                actual_kind, actual_proposals = accelerated._select()
        self.assertEqual(call.call_count, 1)
        self.assertEqual(actual_kind, expected_kind)
        np.testing.assert_array_equal(actual_proposals.centres, expected_proposals.centres)
        self.assertEqual(actual_proposals.radius_index, expected_proposals.radius_index)
        self.assertEqual(accelerated.counters, baseline.counters)

    def test_nonfinite_negative_and_invalid_controls_rejected(self):
        aggregate = np.array([[1., 0., .08, 0., 0., .2]])
        cross = np.array([[1., 0., 0.]])
        outputs = [np.empty(1, dtype=np.uint8) for _ in range(3)]
        for value in (np.nan, np.inf, -1.):
            bad = aggregate.copy()
            bad[0, 5] = value
            self.assertEqual(self.call(1, 1, bad, cross, 1e-5, 1, *outputs, 1), 2)
        self.assertEqual(self.call(1, 1, aggregate, cross, np.nan, 1, *outputs, 1), 1)
        self.assertEqual(self.call(1, 1, aggregate, cross, .01, 1, *outputs, 0), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
