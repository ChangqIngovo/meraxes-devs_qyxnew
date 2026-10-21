"""Ledger, event and geometric-support regressions for the angular solver."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import angular_front
from angular_front import AngularFront


class AngularFrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="angular_tests_")
        if not os.environ.get("ES_ANGULAR_LIBRARY"):
            destination = Path(cls.temporary.name) / "libangular_front.so"
            subprocess.run([sys.executable, str(Path(__file__).with_name("build.py")),
                            "--output", str(destination)], check=True)
            os.environ["ES_ANGULAR_LIBRARY"] = str(destination)
        angular_front._LIB = None

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_ledger(self, state, initial, tolerance=2e-12):
        emitted = state["resource_rate"] * state["diagnostics"]["time"]
        np.testing.assert_allclose(state["resource_absorbed"] + state["resource_escaped"],
                                   emitted, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(state["source_mass"].sum(axis=(1, 2, 3)),
                                   state["resource_absorbed"].sum(axis=1),
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(state["mass"] - initial,
                                   state["source_mass"].sum(axis=0),
                                   rtol=tolerance, atol=tolerance)

    def test_one_cell_two_sources_exact_budget(self):
        cap = np.ones((1, 1, 1))
        with AngularFront(cap, cap * 0, 1, [[0, 0, 0], [0, 0, 0]], [.3, .7], level=1) as solver:
            state = solver.advance(.4)
            np.testing.assert_allclose(state["mass"], .4, atol=1e-15)
            np.testing.assert_allclose(state["source_mass"].ravel(), [.12, .28], atol=1e-15)
            state = solver.advance(2)
            np.testing.assert_allclose(state["mass"], 1, atol=1e-15)
            np.testing.assert_allclose(state["source_mass"].ravel(), [.3, .7], atol=1e-15)
            self.assertAlmostEqual(state["resource_escaped"].sum(), 1)
            self.assert_ledger(state, cap * 0)

    def test_output_one_ulp_before_capacity_event(self):
        cap = np.ones((1, 1, 1)); initial = cap * .9
        with AngularFront(cap, initial, 1, [[0, 0, 0]], [1.]) as solver:
            solver.advance(np.nextafter(1 - .9, -np.inf))
            state = solver.advance(.2)
            self.assertAlmostEqual(state["resource_absorbed"].sum(), .1)
            self.assertAlmostEqual(state["resource_escaped"].sum(), .1)
            self.assert_ledger(state, initial)

    def test_free_streaming_off_centre_and_nonbinary_spacing(self):
        cap = np.zeros((7, 6, 5))
        with AngularFront(cap, cap, .15, [[1, 3, 2]], [2.], level=2) as solver:
            state = solver.advance(.3)
            self.assertAlmostEqual(state["resource_escaped"].sum(), .6)
            self.assertLess(state["diagnostics"]["max_face_balance_absolute"], 2e-13)
            self.assertLess(state["diagnostics"]["max_vacuum_flux_absolute"], 2e-13)
            self.assert_ledger(state, cap)

    def test_neutral_source_cell_is_not_a_free_seed(self):
        cap = np.ones((5, 5, 5))
        with AngularFront(cap, cap * 0, 1, [[2, 2, 2]], [1.], level=1) as solver:
            state = solver.advance(.25)
            self.assertEqual(np.count_nonzero(state["mass"]), 1)
            self.assertAlmostEqual(state["mass"][2, 2, 2], .25)
            self.assert_ledger(state, cap * 0)

    def test_neutral_screen_has_no_transmission(self):
        cap = np.ones((7, 7, 7)); initial = cap.copy()
        initial[4, :, :] = 0
        initial[5:, :, :] = 0
        with AngularFront(cap, initial, 1, [[2, 3, 3]], [1.], level=1) as solver:
            state = solver.snapshot()
            self.assertGreater(state["source_rate"].sum(), 0)
            self.assertEqual(state["source_rate"][:, 5:, :, :].sum(), 0)

    def test_output_schedule_does_not_rescore(self):
        cap = np.ones((5, 5, 5)) * .08
        initial = cap * 0; initial[2, 2, 2] = cap[2, 2, 2]
        kwargs = dict(level=1, epsilon=.1, initial_source_photons=[.08], uniform_density=1)
        with AngularFront(cap, initial, .4, [[2, 2, 2]], [1.], **kwargs) as a, \
             AngularFront(cap, initial, .4, [[2, 2, 2]], [1.], **kwargs) as b:
            direct = a.advance(.7)
            for t in [.001, .01, .052, .11, .237, .479, .7]:
                divided = b.advance(t)
            for name in ("mass", "source_mass", "resource_absorbed", "resource_escaped"):
                np.testing.assert_allclose(direct[name], divided[name], rtol=2e-12, atol=2e-13)
            self.assert_ledger(divided, initial)

    def test_bounded_es_rate_and_time_sandwich(self):
        cap = np.ones((7, 7, 7)) * .025
        initial = cap * 0; initial[2:5, 3, 3] = cap[2:5, 3, 3]
        sources, rates = [[2, 3, 3], [4, 3, 3]], [.4, .6]
        epsilon, t = .1, .37
        score = np.indices(cap.shape)[0].astype(float) ** 3
        with AngularFront(cap, initial, .25, sources, rates, level=1) as ref, \
             AngularFront(cap, initial, .25, sources, rates, level=1, epsilon=epsilon) as es:
            es.set_es_scores(score)
            lo = ref.advance(t / (1 + epsilon))["mass"]
            middle = es.advance(t)
            hi = ref.advance(t * (1 + epsilon))["mass"]
            self.assertLessEqual(np.max(lo - middle["mass"]), 2e-13)
            self.assertLessEqual(np.max(middle["mass"] - hi), 2e-13)
            d = middle["diagnostics"]
            self.assertGreaterEqual(d["min_bias_ratio"], 1 / (1 + epsilon) - 2e-14)
            self.assertLessEqual(d["max_bias_ratio"], 1 + epsilon + 2e-14)
            self.assert_ledger(middle, initial)

    def test_finite_external_scores_do_not_overflow(self):
        cap = np.ones((3, 3, 3)); initial = cap.copy(); initial[0, :, :] = 0
        with AngularFront(cap, initial, 1, [[1, 1, 1]], [1.], epsilon=.1) as solver:
            solver.set_es_scores(np.full(cap.shape, np.finfo(float).max))
            self.assert_ledger(solver.advance(.1), initial)

    def test_zero_budget_source_does_not_change_es(self):
        cap = np.ones((3, 3, 3)); initial = cap * 0; initial[1, 1, 1] = 1
        options = dict(level=1, epsilon=2, uniform_density=1)
        with AngularFront(cap, initial, 1, [[1, 1, 1]], [1.], **options) as base, \
             AngularFront(cap, initial, 1, [[1, 1, 1], [0, 1, 1]], [1., 0.], **options) as ghost:
            np.testing.assert_allclose(base.advance(.1)["mass"], ghost.advance(.1)["mass"], atol=1e-15)

    def test_large_epsilon_zero_score_keeps_positive_weights(self):
        cap = np.ones((1, 1, 1))
        with AngularFront(cap, cap * 0, 1, [[0, 0, 0]], [1.],
                          epsilon=np.finfo(float).max) as solver:
            solver.set_es_scores(cap * 0)
            self.assertAlmostEqual(solver.advance(.25)["mass"].sum(), .25)

    def test_resource_rates_recover_source_rate(self):
        cap = np.ones((3, 3, 3)); initial = cap * 0; initial[1, 1, 1] = 1
        with AngularFront(cap, initial, 1, [[1, 1, 1]], [.7], level=1) as solver:
            total = cap * 0; escape = 0
            for g in range(solver.nbins):
                field, out = solver.resource_rate(0, g)
                total += field; escape += out
            np.testing.assert_allclose(total, solver.snapshot()["source_rate"][0], atol=2e-15)
            self.assertAlmostEqual(total.sum() + escape, .7)

    def test_invalid_indices_and_inputs(self):
        cap = np.ones((2, 2, 2))
        for source in ([[.5, 0., 0.]], [[-1, 0, 0]], [[2, 0, 0]]):
            with self.assertRaises(ValueError):
                AngularFront(cap, cap * 0, 1, source, [1.])
        for level in (-1, .5, 9, True):
            with self.assertRaises(ValueError):
                AngularFront(cap, cap * 0, 1, [[0, 0, 0]], [1.], level=level)
        with self.assertRaises(ValueError):
            AngularFront(cap, cap * 2, 1, [[0, 0, 0]], [1.])
        with AngularFront(cap, cap * 0, 1, [[0, 0, 0]], [1.]) as solver:
            with self.assertRaises(ValueError):
                solver.resource_rate(-1, 0)
            with self.assertRaises(ValueError):
                solver.set_es_scores(cap * np.nan)
            with self.assertRaises(RuntimeError):
                solver.advance(-1)

    def test_closed_wrapper_fails_without_native_dereference(self):
        cap = np.ones((1, 1, 1))
        solver = AngularFront(cap, cap * 0, 1, [[0, 0, 0]], [1.])
        solver.close(); solver.close()
        for operation in (solver.snapshot, solver.diagnostics,
                          lambda: solver.advance(1), lambda: solver.resource_rate(0, 0),
                          lambda: solver.set_es_scores(cap)):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()


if __name__ == "__main__":
    unittest.main()
