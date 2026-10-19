#!/usr/bin/env python3
"""Runtime allocation, identity and large-transform checks for native ES work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

import native_backend as backend


@unittest.skipUnless(os.environ.get("ES_NATIVE_LIBRARY"), "Needs ES_NATIVE_LIBRARY (provided by CTest)")
class NativeRuntime(unittest.TestCase):
    def setUp(self):
        self.library = os.environ["ES_NATIVE_LIBRARY"]

    def test_missing_required_backend_does_not_fall_back(self):
        with mock.patch.dict(os.environ, ES_NATIVE_REQUIRED="1", ES_NATIVE_DISABLE="0",
                             ES_NATIVE_LIBRARY="/nonexistent/es-kernels.so"):
            with mock.patch.object(backend, "_backend", backend._UNINITIALIZED):
                with self.assertRaises(RuntimeError):
                    backend.get_backend()

    def test_thread_count_is_not_a_checkpoint_physics_change(self):
        cpus = len(os.sched_getaffinity(0))
        one = backend.NativeBackend(self.library, 1)
        many = backend.NativeBackend(self.library, min(4, cpus))
        with mock.patch.object(backend, "_backend", one):
            identity = backend.numerical_identity()
        with mock.patch.object(backend, "_backend", many):
            self.assertEqual(backend.numerical_identity(), identity)
        self.assertEqual(many.team_size, min(4, cpus))

    def test_child_expands_only_to_mpi_allocation(self):
        cpus = sorted(os.sched_getaffinity(0))[:2]
        if len(cpus) < 2:
            self.skipTest("Needs two allowed CPUs")
        env = dict(os.environ, ES_NATIVE_REQUIRED="1", ES_NATIVE_DISABLE="0", ES_WORKER_THREADS="2",
                   ES_MPI_CPUSET=",".join(map(str, cpus)))
        code = ("import json,os; "
                "cpus=list(map(int,os.environ['ES_MPI_CPUSET'].split(','))); "
                "os.sched_setaffinity(0,{cpus[0]}); "
                "from native_backend import get_backend; b=get_backend(); "
                "print(json.dumps(dict(affinity=sorted(os.sched_getaffinity(0)),team=b.team_size)))")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                                env=env, check=True, capture_output=True, text=True)
        actual = json.loads(result.stdout)
        self.assertEqual(actual, dict(affinity=cpus, team=2))

    def test_large_threaded_fft_matches_serial_and_round_trip(self):
        threads = min(4, len(os.sched_getaffinity(0)))
        if threads < 2:
            self.skipTest("Needs two allowed CPUs")
        native = backend.NativeBackend(self.library, threads)
        rng = np.random.default_rng(41)
        field = rng.normal(size=(128,)*3)
        with mock.patch.object(backend, "_backend", native):
            self.assertEqual(backend._fft_workers(32**3), 1)
            self.assertEqual(backend._fft_workers(field.size), threads)
            transformed = backend.rfftn(field)
            restored = backend.irfftn(transformed, field.shape)
        serial_native = backend.NativeBackend(self.library, 1)
        with mock.patch.object(backend, "_backend", serial_native):
            np.testing.assert_array_equal(backend.rfftn(field), transformed)
            np.testing.assert_array_equal(backend.irfftn(transformed, field.shape), restored)
        expected = np.fft.rfftn(field)
        np.testing.assert_allclose(transformed, expected, rtol=5e-12, atol=2e-11)
        np.testing.assert_allclose(restored, field, rtol=3e-14, atol=3e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
