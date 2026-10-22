"""Periodic geometry, causal inventory and shared-chemistry regression checks.

Build the existing native geometry library first, or set ES_ANGULAR_LIBRARY.
No periodic escape, recombination, or emission history is invented by a restart.
"""
import json
import math
import unittest

import numpy as np

from angular_front import AngularFront
from angular_hhe import illuminate_channels
from periodic_front import PeriodicFront


class PeriodicFrontTests(unittest.TestCase):
    def model(self, shape=(3, 3, 3), neutral=0., **options):
        gas = np.zeros((math.prod(shape), 5))
        gas[:, 0] = neutral
        return PeriodicFront(shape, gas, [20.], **options)

    def pulse(self, model, source, amount):
        photons = np.zeros(model.ncell)
        photons[source] = amount
        return photons

    def audit(self, model, result):
        self.assertEqual(result['audits']['escaped_photons'], 0.)
        self.assertLess(result['audits']['resource_ledger_max_relative'], 5e-10)
        self.assertTrue(np.all(model.gas >= 0))
        pending = {}
        for key, amount in model.entries.items():
            self.assertGreater(amount, 0.)
            self.assertGreaterEqual(key[-1], 0.)
            pending.setdefault(key[:3], []).append(amount)
        for key, emission in model.emitted.items():
            error = model.absorbed.get(key, 0.)+math.fsum(pending.get(key, []))-emission
            self.assertLessEqual(abs(error), 5e-10*max(emission, np.finfo(float).tiny))

    def checkpoint_with_light(self, model, rows):
        state = model.export_state()
        state['entries'] = rows
        resources = {}
        for row in rows:
            key = tuple(row[:3])
            resources[key] = resources.get(key, 0.) + row[-1]
        state['resource_ledgers'] = [list(key)+[amount, 0.] for key, amount in resources.items()]
        return PeriodicFront.from_state(state)

    def test_periodic_boundary_target(self):
        model = self.model()
        source, target = 22, 4  # (2,1,1) crosses +x into (0,1,1).
        model.gas[target, 0] = 1.
        result = model.advance(self.pulse(model, source, .6), [1.], 1.)
        self.audit(model, result)
        self.assertAlmostEqual(result['absorbed'][target, 0], .1, 11)
        self.assertAlmostEqual(result['absorbed'].sum(), .1, 11)
        self.assertTrue(all(key[0] == source for key in model.entries))

    def test_distinct_images_share_one_physical_capacity(self):
        model = self.model()
        model.gas[9, 0] = 1.
        # Both images look up x=1 in the same physical grid. Bin 4 has +x support.
        model = self.checkpoint_with_light(model, [[0, 4, 0, 1, 0, 0, 0., 1.],
                                                  [0, 4, 0, 4, 0, 0, 0., 1.]])
        result = model.advance(np.zeros(model.ncell), [1.], 0.)
        self.audit(model, result)
        self.assertAlmostEqual(result['absorbed'].sum(), 1-model.transparency, 11)
        self.assertAlmostEqual(result['photon_inventory'], 1+model.transparency, 11)
        self.assertTrue(any(key[3] > model.shape[0] for key in model.entries))

    def test_vacuum_has_finite_flight_and_no_periodic_escape(self):
        model = self.model((1, 1, 1))
        first = model.advance([1.], [1.], .25, dt_s=.25)
        self.assertEqual(first['absorbed'].sum(), 0.)
        self.assertTrue(all(key[-1] == .75 for key in model.entries))
        result = model.advance([0.], [1.], 3.5, dt_s=3.5)
        self.audit(model, result)
        self.assertAlmostEqual(result['photon_inventory'], 1., 14)
        self.assertTrue(all(math.sqrt(sum(x*x for x in key[3:6])) > 3.75
                            for key in model.entries))
        self.assertLess(result['audits']['routed_nodes'], 10000)

    def test_restart_preserves_photon_images_edges_and_clock(self):
        model = self.model()
        model.advance(self.pulse(model, 13, 1.1), [1.], .35, dt_s=11.)
        restored = PeriodicFront.from_state(json.loads(json.dumps(model.export_state())))
        for target in (model, restored):
            target.gas[0, 0] = .3
            target.advance(self.pulse(target, 4, .2), [1.], 1.35, dt_s=7.)
        self.assertEqual(restored.export_state(), model.export_state())
        self.assertEqual(restored.time_s, 18.)
        self.assertEqual(restored.step_count, 2)

    def test_new_source_keeps_old_emission_origin(self):
        model = self.model()
        model.advance(self.pulse(model, 0, .3), [1.], .2)
        model.advance(self.pulse(model, 26, .7), [1.], .2)
        self.assertEqual({key[0] for key in model.entries}, {0, 26})
        for source, expected in ((0, .3), (26, .7)):
            self.assertAlmostEqual(math.fsum(value for key, value in model.entries.items()
                                             if key[0] == source), expected, 14)

    def test_tiny_positive_photons_survive_without_a_floor(self):
        model = self.model((1, 1, 1))
        result = model.advance([1e-300], [1.], 1.25)
        self.audit(model, result)
        self.assertAlmostEqual(result['photon_inventory']/1e-300, 1., 13)
        # Even the smallest representable cohort is retained, although its
        # angular split necessarily reaches floating-point resolution.
        tiny = np.nextafter(0., 1.)
        model = self.model((1, 1, 1))
        result = model.advance([tiny], [1.], .2)
        self.assertEqual(result['photon_inventory'], tiny)
        model = self.model((1, 1, 1), neutral=1.)
        result = model.advance([tiny], [1.], .2)
        self.assertEqual(result['photon_inventory'], tiny)

    def test_release_before_recombination_prevents_source_trap(self):
        model = self.model(neutral=1.)
        first = model.advance(self.pulse(model, 13, 2.), [1.], 10.)
        self.assertGreater(first['photon_inventory'], 1.)
        self.assertTrue(all(key[3:6] != (0, 0, 0) and key[-1] > 0
                            for key in model.entries))
        # Simulate a following recombination half-step at the old source.
        model.gas[13] = [1., 0., 0., 0., 0.]
        result = model.advance(np.zeros(model.ncell), [1.], 1.)
        self.audit(model, result)
        self.assertEqual(result['absorbed'][13, 0], 0.)
        self.assertGreater(result['absorbed'].sum(), .99)

    def test_bands_share_gas_and_stop_at_their_thresholds(self):
        gas = np.array([[1., 0., .2, 0., 0.]])
        model = PeriodicFront((1, 1, 1), gas, [20., 35., 70.])
        result = model.advance([3.], [.3, .3, .4], 0.)
        self.audit(model, result)
        self.assertLessEqual(result['absorbed'].sum(), 1.4)
        self.assertGreater(model.gas[0, 4], .19)
        self.assertGreater(model.gas[0, 0]+model.gas[0, 2]+model.gas[0, 3], 0.)
        np.testing.assert_allclose(model.gas[:, :2].sum(1), [1.], rtol=2e-12)
        np.testing.assert_allclose(model.gas[:, 2:].sum(1), [.2], rtol=2e-12)

    def test_monochromatic_hydrogen_over_wide_density_range(self):
        # The previous common exposure step reaches 40000 iterations for
        # this neutral-count range, despite every cell needing only 1% HI.
        gas = np.zeros((64, 5))
        gas[:, 0] = np.geomspace(1e-8, 1e8, 64)
        gas[:, 2] = .08*gas[:, 0]
        source = .01*gas[:, 0]
        model = PeriodicFront((4, 4, 4), gas, [18.])
        result = model.advance(source, [1.], 0.)
        self.audit(model, result)
        self.assertEqual(result['audits']['chemistry_method'], 'analytic_h')
        self.assertEqual(result['audits']['chemistry_substeps'], 1)
        np.testing.assert_allclose(result['absorbed'][:, 0], source, rtol=3e-15)
        np.testing.assert_allclose(model.gas[:, 0], .99*gas[:, 0], rtol=3e-15)
        np.testing.assert_array_equal(model.gas[:, 2:], gas[:, 2:])
        self.assertLess(result['photon_inventory'], 3e-15*source.sum())

    def test_identical_soft_groups_share_capacity_and_keep_source_ledgers(self):
        gas = np.zeros((27, 5))
        gas[9] = [.6, .4, .1, .05, .02]
        model = PeriodicFront((3, 3, 3), gas, [18., 18., 60.], transparency=.1)
        # Two sources and two periodic images illuminate the same gas cell.
        # The unused hard group must not disable the H-only specialization.
        model = self.checkpoint_with_light(model, [[0, 4, 0, 1, 0, 0, 0., .2],
                                                  [18, 4, 1, 2, 0, 0, 0., 1.8]])
        result = model.advance(np.zeros(27), [.5, .5, 0.], 0.)
        self.audit(model, result)
        self.assertEqual(result['audits']['chemistry_method'], 'analytic_h')
        np.testing.assert_allclose(model.gas[9], [.1, .9, .1, .05, .02], atol=2e-16)
        self.assertAlmostEqual(result['source_absorbed'][0], .05, 14)
        self.assertAlmostEqual(result['source_absorbed'][18], .45, 14)
        self.assertAlmostEqual(result['photon_inventory'], 1.5, 14)
        self.assertTrue(all(key[-1] > 0 for key in model.entries))

    def test_sparse_spectral_solve_matches_full_grid_chemistry(self):
        gas = np.zeros((27, 5))
        gas[:, 0], gas[:, 2] = 1., .08
        gas[1, 0] = 1e8  # No photons here: must not enter the active solve.
        source = np.zeros(27)
        source[0], source[26] = .2, .3
        weights = np.array([.5, .3, .2])
        model = PeriodicFront((3, 3, 3), gas, [18., 30., 60.])
        reference = illuminate_channels(gas, weights[:, None]*source, model.cross,
                                        epsilon=model.transparency, max_iterations=40000)
        self.assertEqual(reference['stop'], 'photon_remainder')
        result = model.advance(source, weights, 0.)
        self.audit(model, result)
        self.assertEqual(result['audits']['chemistry_method'], 'spectral_midpoint')
        self.assertEqual(result['audits']['chemistry_active_cells'], 2)
        np.testing.assert_allclose(result['gas'], reference['gas'], rtol=3e-12, atol=1e-14)
        np.testing.assert_allclose(result['absorbed'], reference['absorbed'], rtol=3e-12,
                                   atol=1e-14)

    def test_different_soft_cross_sections_use_spectral_solver(self):
        gas = np.array([[1., 0., .08, 0., 0.]])
        model = PeriodicFront((1, 1, 1), gas, [18., 20.])
        result = model.advance([.1], [.3, .7], 0.)
        self.audit(model, result)
        self.assertEqual(result['audits']['chemistry_method'], 'spectral_midpoint')
        self.assertAlmostEqual(result['absorbed'].sum(), .1, 12)

    def test_es_reweight_keeps_flight_boundary_share(self):
        model = self.model()
        key = (0, 4, 0, 0, 0, 0, 0.)
        edges = model.geometry.outgoing(4, (0, 0, 0))
        # Two first neighbors are opaque, and the third remains in flight.
        # ES may rebalance opaque targets but cannot borrow boundary light.
        targets = [model._cell(0, edge[0]) for edge in edges[:2]]
        model.gas[targets, 0] = 1.
        opaque = model._opaque(model.gas)
        scores = np.ones(model.ncell)
        scores[targets[0]] = 1e100
        scores[targets[1]] = 0.
        weighted, _ = model._route(key, 1., 1., opaque, scores)
        model.epsilon = 0.
        plain, _ = model._route(key, 1., 1., opaque, scores)
        self.assertAlmostEqual(math.fsum(v for k, v in weighted.items() if k[-1] > 0),
                               math.fsum(v for k, v in plain.items() if k[-1] > 0), 14)
        first_key = (0, 4, 0, *edges[0][0], 0.)
        self.assertGreater(weighted[first_key], plain[first_key])

    def test_old_target_cohorts_cannot_borrow_each_others_inventory(self):
        model = self.model(neutral=10.)
        model = self.checkpoint_with_light(model, [[0, 4, 0, 1, 0, 0, 0., .2],
                                                  [0, 4, 0, 0, -1, 0, 0., .8]])
        scores = np.zeros(model.ncell)
        scores[9] = 1e100
        result = model.advance(np.zeros(model.ncell), [1.], 100., es_scores=scores)
        self.audit(model, result)
        self.assertAlmostEqual(result['absorbed'][9, 0], .2, 11)
        self.assertAlmostEqual(result['absorbed'][6, 0], .8, 11)

    def test_failure_does_not_mutate_state(self):
        model = self.model(max_route_nodes=1)
        state = model.export_state()
        with self.assertRaisesRegex(RuntimeError, 'node limit'):
            model.advance(self.pulse(model, 13, 1.), [1.], 3., dt_s=1.)
        self.assertEqual(model.export_state(), state)
        model = self.model(max_entries=2)
        state = model.export_state()
        with self.assertRaisesRegex(RuntimeError, 'inventory limit'):
            model.advance(self.pulse(model, 13, 1.), [1.], 3., dt_s=1.)
        self.assertEqual(model.export_state(), state)

    def test_invalid_checkpoint_budget_and_edge_rejected(self):
        model = self.model()
        model.advance(self.pulse(model, 0, 1.), [1.], .2)
        state = model.export_state()
        state['entries'][0][-1] *= 2
        with self.assertRaisesRegex(ValueError, 'ledger mismatch'):
            PeriodicFront.from_state(state)
        state = model.export_state()
        state['entries'][0][-2] = 100.
        with self.assertRaisesRegex(ValueError, 'incoming edge'):
            PeriodicFront.from_state(state)

    def test_step_refinement_toward_existing_hydrogen_limit(self):
        # The front remains far from the box boundary, so the periodic and
        # open spatial problems agree. Large travel approximates infinite c.
        shape = (5, 5, 5)
        with AngularFront(np.ones(shape), np.zeros(shape), 1., [[2, 2, 2]], [1.]) as old:
            reference = old.advance(7.3)['mass'].ravel()
        errors = []
        for steps in (4, 32):
            model = self.model(shape, neutral=1., epsilon=0., transparency=1e-8)
            for _ in range(steps):
                result = model.advance(self.pulse(model, 62, 7.3/steps), [1.], 100.)
            self.audit(model, result)
            errors.append(float(np.abs(model.gas[:, 1]-reference).sum()))
        self.assertGreater(errors[0], .2)
        self.assertLess(errors[1], errors[0]/10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
