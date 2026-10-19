#!/usr/bin/env python3
"""Independent math.fsum oracles for deterministic OpenMP geometry kernels."""
import ctypes
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


class NativeGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = None
        path = os.environ.get('ES_NATIVE_LIBRARY')
        if not path:
            compiler = shlex.split(os.environ.get('CC', 'cc'))
            if not compiler or not shutil.which(compiler[0]):
                raise unittest.SkipTest('Native geometry requires a C compiler')
            cls.temporary = tempfile.TemporaryDirectory(prefix='es-native-geometry-')
            path = str(Path(cls.temporary.name) / 'geometry.so')
            subprocess.run(compiler + ['-O3', '-std=c99', '-fopenmp', '-fPIC', '-shared',
                '-fno-fast-math', '-ffp-contract=off',
                str(Path(__file__).with_name('native_geometry.c')), '-lm', '-o', path],
                check=True, capture_output=True, text=True)
        cls.lib = ctypes.CDLL(path)
        integer = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        floating = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i64, i32 = ctypes.c_int64, ctypes.c_int
        cls.lib.es_sparse_sums.argtypes = [i64, integer, integer, i64, floating,
                                          integer, i64, ctypes.c_double, floating, i32]
        cls.lib.es_sparse_sums.restype = i32
        cls.lib.es_periodic_offset_sums.argtypes = [i64, integer, floating,
            integer, i64, integer, i64, floating, i32]
        cls.lib.es_periodic_offset_sums.restype = i32
        cls.lib.es_indexed_sum.argtypes = [floating, i64, i64, integer, i64,
                                          integer, i64, floating, i32]
        cls.lib.es_indexed_sum.restype = i32

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def sparse(self, side, xyz, occupied, column, rows, radius2, threads):
        out = np.empty(len(rows))
        status = self.lib.es_sparse_sums(side, xyz, occupied, len(occupied), column,
                                        rows, len(rows), radius2, out, threads)
        self.assertEqual(status, 0)
        return out

    def indexed(self, values, indices, offsets, threads):
        out = np.empty((len(offsets)-1, values.shape[1]))
        status = self.lib.es_indexed_sum(values, len(values), values.shape[1], indices,
                                        len(indices), offsets, len(out), out, threads)
        self.assertEqual(status, 0)
        return out

    def periodic(self, side, xyz, column, rows, offsets, threads):
        out = np.empty(len(rows))
        status = self.lib.es_periodic_offset_sums(side, xyz, column, rows,
            len(rows), offsets, len(offsets), out, threads)
        self.assertEqual(status, 0)
        return out

    def test_periodic_offsets_signed_dynamic_range_and_threads(self):
        rng = np.random.default_rng(493)
        side = 16
        xyz = np.ascontiguousarray(np.indices((side,)*3).reshape(3, -1).T, dtype=np.int64)
        delta = np.minimum(xyz, side-xyz)
        offsets = xyz[np.sum(delta*delta, axis=1) <= 25]
        column = rng.choice([-1., 0., 1.], len(xyz)) * 10.**rng.uniform(-320, 200, len(xyz))
        tiny = np.nextafter(0., 1.)
        column[:12] = [1e200, 1., -1e200, tiny, tiny, -tiny,
                       1., 2.**-53, tiny, -1., -2.**-53, -tiny]
        rows = np.r_[np.arange(0, len(xyz), 4, dtype=np.int64), len(xyz)-1]
        expected = []
        for row in rows:
            points = (offsets + xyz[row]) % side
            cells = (points[:, 0]*side + points[:, 1])*side + points[:, 2]
            expected.append(math.fsum(column[cells].tolist()))
        expected = np.asarray(expected)
        for threads in (1, 4):
            actual = self.periodic(side, xyz, column, rows, offsets, threads)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    def test_periodic_offsets_cancellation_subnormals_and_empty(self):
        side = 2
        xyz = np.ascontiguousarray(np.indices((side,)*3).reshape(3, -1).T, dtype=np.int64)
        rows = np.arange(len(xyz), dtype=np.int64)
        tiny = np.nextafter(0., 1.)
        for column in (np.array([1e16, 1., -1e16, tiny, tiny, -tiny, 0., -0.]),
                       np.array([tiny, tiny, -tiny, 0., -0., 0., 0., 0.])):
            for offsets in (xyz, xyz[:3], xyz[:0]):
                expected = []
                for row in rows:
                    points = (offsets + xyz[row]) % side
                    cells = (points[:, 0]*side + points[:, 1])*side + points[:, 2]
                    expected.append(math.fsum(column[cells].tolist()))
                np.testing.assert_array_equal(self.periodic(side, xyz, column, rows, offsets, 4), expected)
        self.assertEqual(self.periodic(side, xyz, column, rows[:0], xyz, 4).size, 0)

    def test_periodic_offsets_invalid_and_nonfinite(self):
        side = 2
        xyz = np.ascontiguousarray(np.indices((side,)*3).reshape(3, -1).T, dtype=np.int64)
        rows = np.array([7], dtype=np.int64)
        offsets = np.array([[1, 1, 1]], dtype=np.int64)
        column = np.ones(len(xyz))
        out = np.empty(1)
        def call(centres=rows, support=offsets, positions=xyz, threads=4, grid_side=side):
            return self.lib.es_periodic_offset_sums(grid_side, positions, column,
                centres, len(centres), support, len(support), out, threads)
        for value in (np.nan, np.inf, -np.inf):
            column[0] = value
            self.assertEqual(call(), 2)
        column[0] = np.finfo(float).max
        self.assertEqual(call(support=np.repeat(offsets, 2, axis=0)), 2)
        column[0] = 1.
        for index in (-1, len(xyz)):
            self.assertEqual(call(centres=np.array([index], dtype=np.int64)), 1)
        for value in (-1, side):
            invalid = offsets.copy()
            invalid[0, 0] = value
            self.assertEqual(call(support=invalid), 1)
        invalid = xyz.copy()
        invalid[7, 0] = side
        self.assertEqual(call(positions=invalid), 1)
        for threads in (0, 1025):
            self.assertEqual(call(threads=threads), 1)
        for grid_side in (0, 2097152):
            self.assertEqual(call(grid_side=grid_side), 1)

    def test_sparse_minimum_image_and_parallel_determinism(self):
        rng = np.random.default_rng(661)
        side = 32
        xyz = np.ascontiguousarray(np.indices((side,)*3).reshape(3, -1).T, dtype=np.int64)
        column = np.zeros(side**3)
        occupied = np.sort(rng.choice(len(column), 55, replace=False)).astype(np.int64)
        column[occupied] = rng.choice([-1., 1.], len(occupied)) * 10.**rng.uniform(-320, 3, len(occupied))
        rows = np.arange(len(column), dtype=np.int64)
        for radius2 in (0., 2., 36., 123.4, 768.):
            delta = np.abs(xyz[:, None, :] - xyz[occupied][None, :, :])
            delta = np.minimum(delta, side-delta)
            inside = np.sum(delta*delta, axis=2) <= radius2
            expected = np.array([math.fsum(column[occupied[hit]].tolist()) for hit in inside])
            for threads in (1, 2, 4, 8):
                actual = self.sparse(side, xyz, occupied, column, rows, radius2, threads)
                np.testing.assert_array_equal(actual, expected)

    def test_indexed_signed_dynamic_range_and_parallel_determinism(self):
        rng = np.random.default_rng(921)
        values = rng.choice([-1., 1.], (1000, 5)) * 10.**rng.uniform(-320, 4, (1000, 5))
        sizes = rng.integers(0, 401, 300, dtype=np.int64)
        offsets = np.r_[np.int64(0), np.cumsum(sizes)]
        indices = rng.integers(0, len(values), offsets[-1], dtype=np.int64)
        expected = np.array([[math.fsum(values[indices[a:b], j].tolist()) for j in range(5)]
                             for a, b in zip(offsets[:-1], offsets[1:])])
        for threads in (1, 2, 4, 8):
            np.testing.assert_array_equal(self.indexed(values, indices, offsets, threads), expected)

    def test_cancellation_half_ulp_ties_and_subnormals(self):
        tiny = np.nextafter(0., 1.)
        values = np.array([[1., 2.**-53, tiny], [1., 2.**-53, -tiny],
                           [-1., -2.**-53, -tiny], [-1., -2.**-53, tiny],
                           [1e16, 1., -1e16], [tiny, tiny, -tiny],
                           [0., -0., 0.]]).reshape(-1, 1)
        offsets = np.arange(0, len(values)+1, 3, dtype=np.int64)
        indices = np.arange(len(values), dtype=np.int64)
        expected = np.array([[math.fsum(values[a:b, 0].tolist())]
                             for a, b in zip(offsets[:-1], offsets[1:])])
        np.testing.assert_array_equal(self.indexed(values, indices, offsets, 4), expected)

    def test_empty_supports(self):
        xyz = np.zeros((1, 3), dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        rows = np.array([0], dtype=np.int64)
        np.testing.assert_array_equal(self.sparse(1, xyz, empty, np.zeros(1), rows, 0., 8), [0.])
        np.testing.assert_array_equal(self.indexed(np.ones((1, 2)), empty,
                                                  np.zeros(4, dtype=np.int64), 8), np.zeros((3, 2)))

    def test_invalid_indices_nonfinite_and_overflow_rejected(self):
        offsets = np.array([0, 2], dtype=np.int64)
        indices = np.array([0, 1], dtype=np.int64)
        out = np.empty((1, 1))
        for value in (np.nan, np.inf, -np.inf):
            values = np.array([[value], [1.]])
            self.assertEqual(self.lib.es_indexed_sum(values, 2, 1, indices, 2, offsets, 1, out, 4), 2)
        values = np.full((2, 1), np.finfo(float).max)
        self.assertEqual(self.lib.es_indexed_sum(values, 2, 1, indices, 2, offsets, 1, out, 4), 2)
        indices[1] = 2
        self.assertEqual(self.lib.es_indexed_sum(values, 2, 1, indices, 2, offsets, 1, out, 4), 1)
        xyz = np.zeros((1, 3), dtype=np.int64)
        ix = np.array([0], dtype=np.int64)
        self.assertEqual(self.lib.es_sparse_sums(1, xyz, ix, 1, np.array([np.nan]),
                                               ix, 1, 0., out, 4), 2)
        self.assertEqual(self.lib.es_sparse_sums(1, xyz, ix, 1, np.ones(1),
                                               ix, 1, 0., out, 0), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
