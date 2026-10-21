"""Physical limits, second-order refinement and count audits for recombination."""

import unittest
import numpy as np

from recombination import (coefficients_1e4, emit_from_captures, recombine,
                           evolve_local, _downward)


class RecombinationTests(unittest.TestCase):
    def audit(self, initial, result):
        self.assertTrue(np.all(result['gas'] >= 0))
        self.assertTrue(np.all(result['captures'] >= 0))
        self.assertTrue(np.all(result['diffuse_emitted'] >= 0))
        scale = np.maximum(initial.sum(1), np.finfo(float).tiny)
        self.assertLess(np.max(abs(result['charge_error'])/scale, initial=0.), 2e-12)
        self.assertLess(np.max(abs(result['nucleus_error'])/scale[:, None], initial=0.), 2e-12)

    def test_pure_hydrogen_nonlinear_analytic_solution(self):
        initial = np.array([[.2, .8, .1, 0., 0.], [2e60, 8e60, 0., 0., 0.]])
        volume = np.array([1e3, 1e64])
        alpha = coefficients_1e4()[0][0]
        dt = 3.2e15
        result = recombine(initial, volume, dt)
        expected = initial[:, 1]/(1+alpha*initial[:, 1]*dt/volume)
        np.testing.assert_allclose(result['gas'][:, 1], expected, rtol=3e-16)
        self.audit(initial, result)

    def test_exact_frozen_electron_helium_chain(self):
        initial = np.array([[0., 0., 0., 0., 1.]])
        alpha = np.array([2., 3., 3.])
        gas, captures = _downward(initial, np.array([1.]), .7, alpha)
        z = 2.1
        expected = [0., 0., 1-(1+z)*np.exp(-z), z*np.exp(-z), np.exp(-z)]
        np.testing.assert_allclose(gas[0], expected, rtol=1e-14, atol=1e-16)
        self.assertAlmostEqual(captures.sum(), 2-gas[0, 3]-2*gas[0, 4], 14)

    def test_coupled_helium_refines_at_second_order(self):
        initial = np.array([[.1, .9, .005, .02, .055]])
        runs = [recombine(initial, 1e3, 5e15, max_fraction=h)
                for h in (.2, .1, .05, .025)]
        differences = [np.abs(a['gas']-b['gas']).sum()
                       for a, b in zip(runs, runs[1:])]
        ratios = np.array(differences[:-1])/differences[1:]
        self.assertTrue(np.all((ratios > 3.5) & (ratios < 4.5)), ratios)
        for result in runs:
            self.audit(initial, result)

    def test_uniform_density_count_volume_scaling(self):
        initial = np.array([[.2, .8, .01, .04, .03]])
        base = recombine(initial, 3e3, 1e15)
        scaled = recombine(initial*1e60, 3e63, 1e15)
        np.testing.assert_allclose(scaled['gas']/1e60, base['gas'], rtol=2e-14)
        np.testing.assert_allclose(scaled['captures']/1e60, base['captures'], rtol=2e-14)
        self.audit(initial*1e60, scaled)

    def test_no_time_and_neutral_gas(self):
        initial = np.array([[1., 0., .08, 0., 0.], [.2, .8, .01, .02, .05]])
        zero = recombine(initial, 1e3, 0.)
        np.testing.assert_array_equal(zero['gas'], initial)
        np.testing.assert_array_equal(zero['captures'], 0.)
        neutral = recombine(initial[:1], 1e3, 1e18)
        np.testing.assert_array_equal(neutral['gas'], initial[:1])
        self.audit(initial[:1], neutral)

    def test_diffuse_count_branching_is_separate_from_capture_count(self):
        # Independent branching identities: all ground captures plus all
        # excited captures recover the case-A event count.
        alpha_a, alpha_b, n2 = coefficients_1e4()
        output = emit_from_captures(np.eye(3))
        self.assertAlmostEqual(output[0, 0], 1-alpha_b[0]/alpha_a[0])
        self.assertAlmostEqual(output[1, 1]+output[1, 2]/.96, 1.)
        self.assertAlmostEqual(output[2, 3]+output[2, 5]/.715, 1.)
        self.assertAlmostEqual(output[2, 4], n2/alpha_a[2])
        self.assertGreater(output[2].sum(), 1.)

    def test_tiny_recombination_counts_are_retained(self):
        initial = np.array([[0., 1., .08, 0., 0.]])
        result = recombine(initial, 1e3, 1e-30)
        self.assertGreater(result['captures'][0, 0], 0.)
        self.assertGreater(result['diffuse_emitted'][0, 0], 0.)
        self.audit(initial, result)

    def test_inputs_not_mutated_and_invalid_inputs_rejected(self):
        initial = np.array([[.1, .9, .01, .02, .05]])
        copy = initial.copy()
        recombine(initial, 1e3, 1e15)
        np.testing.assert_array_equal(initial, copy)
        for kwargs in ({'dt_s': -1.}, {'volume_cm3': 0.}, {'temperature': 2e4},
                       {'max_fraction': 0.}, {'max_substeps': 0}):
            options = dict(volume_cm3=1e3, dt_s=1e15)
            options.update(kwargs)
            with self.assertRaises(ValueError):
                recombine(initial, **options)
        with self.assertRaises(ValueError):
            recombine([[-1, 1, 0, 0, 0]], 1e3, 1.)

    def test_local_ots_hydrogen_recovers_case_b_analytic_solution(self):
        initial = np.array([[.2, .8, .08, 0., 0.]])
        alpha_a, alpha_b, _ = coefficients_1e4()
        duration, volume = 8e15, 1e3
        result = evolve_local(initial, volume, duration)
        final_hii = initial[0, 1]/(1+alpha_b[0]*initial[0, 1]*duration/volume)
        self.assertAlmostEqual(result['gas'][0, 1], final_hii, 14)
        net = initial[0, 1]-final_hii
        self.assertAlmostEqual(result['captures'][0, 0], net*alpha_a[0]/alpha_b[0], 14)
        self.assertAlmostEqual(result['diffuse_absorbed'][0, 0],
                               result['diffuse_emitted'][0, 0], 14)
        np.testing.assert_array_equal(result['pending'], 0.)
        self.audit(initial, result)
        self.assertLess(abs(result['photon_error']).max(), 1e-14)

    def test_local_ots_retains_zero_opacity_photons(self):
        initial = np.zeros((2, 5))
        photons = np.arange(16, dtype=float).reshape(2, 8)
        result = evolve_local(initial, [1e3, 2e3], 1e15, pending=photons)
        np.testing.assert_array_equal(result['pending'], photons)
        np.testing.assert_array_equal(result['diffuse_absorbed'], 0.)
        np.testing.assert_array_equal(result['captures'], 0.)
        np.testing.assert_array_equal(result['photon_error'], 0.)

    def test_local_ots_pending_does_not_move_between_cells(self):
        initial = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
        photons = np.zeros((2, 8)); photons[1, 0] = .25
        result = evolve_local(initial, 1e3, 0., pending=photons)
        self.assertEqual(result['gas'][0, 0], 1.)
        self.assertEqual(result['pending'][1, 0], .25)
        self.assertEqual(result['diffuse_absorbed'].sum(), 0.)

    def test_local_ots_helium_photons_can_ionize_hydrogen(self):
        # No stellar photons: helium recombination radiation is responsible
        # for every new HII atom in this cell.
        initial = np.array([[1., 0., 0., .08, 0.]])
        result = evolve_local(initial, 1e3, 1e14,
                              absorption_donor_fraction=.2)
        self.assertGreater(result['gas'][0, 1], 0.)
        self.assertGreater(result['diffuse_emitted'][0, 2], 0.)
        self.assertGreater(result['diffuse_absorbed'][0, 0], 0.)
        self.audit(initial, result)
        self.assertLess(abs(result['photon_error']).max(), 1e-13)

    def test_local_ots_physical_substep_refinement(self):
        initial = np.array([[.1, .9, .005, .02, .055]])
        runs = [evolve_local(initial, 1e3, 8e14, max_fraction=h,
                             absorption_donor_fraction=.2)
                for h in (.2, .1, .05)]
        differences = [np.abs(a['gas']-b['gas']).sum()
                       for a, b in zip(runs, runs[1:])]
        # Recombination followed by instantaneous OTS is a first-order
        # physical splitting, even though its bare recombination map is
        # second-order. Test its stated order instead of claiming Strang.
        self.assertGreater(differences[0]/differences[1], 1.6)
        self.assertLess(differences[0]/differences[1], 2.4)
        for result in runs:
            self.audit(initial, result)
            self.assertLess(abs(result['photon_error']).max(), 1e-13)


if __name__ == '__main__':
    unittest.main()
