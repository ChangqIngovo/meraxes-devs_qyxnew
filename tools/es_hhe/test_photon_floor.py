#!/usr/bin/env python3
"""Regressions for dormant photon accounts and the L10 underflow stall."""
import unittest

import numpy as np

from es_hhe_reference import MYR, SpectralES, joint_exposure_update
from snapshot_worker import SnapshotModel


FLOOR = 1e-30


def nuclei(gas):
    return np.column_stack((gas[:, :2].sum(axis=1), gas[:, 2:].sum(axis=1)))


def electrons(gas):
    return float((gas[:, 1] + gas[:, 3] + 2 * gas[:, 4]).sum())


def snapshot_model():
    metadata = dict(grid_dim=1, helium_number_ratio=.081, radius_ratio=1.2,
        max_radius_cells=0., source_energies_eV=[18.], source_weights=[1.],
        mean_h_atoms_per_cell=1e60, temperature_K=10000., max_step_s=MYR)
    model = SnapshotModel(metadata)
    model.set_initial_density(np.ones(1), metadata["helium_number_ratio"])
    return model


class PhotonFloorTests(unittest.TestCase):
    def assert_joint_conserves(self, gas, photons, result):
        after = result["species"]
        remaining = result["photons"]
        debit = photons - remaining
        captures = float(result["absorbed"].sum())
        self.assertTrue(np.all(after >= 0))
        self.assertTrue(np.all(remaining >= 0))
        self.assertTrue(np.all(remaining <= photons))
        np.testing.assert_allclose(nuclei(after), nuclei(gas), rtol=3e-13, atol=0.)
        # Check returned state and budgets independently of the reported audit.
        scale = max(float(photons.sum()), np.finfo(float).tiny)
        self.assertLessEqual(abs(float(debit.sum()) - captures), 3e-12 * scale)
        charge_scale = max(electrons(gas), electrons(after), captures,
                           np.finfo(float).tiny)
        self.assertLessEqual(abs(electrons(after) - electrons(gas) - captures),
                             3e-12 * charge_scale)

    def assert_model_conserves(self, model):
        emitted = model.emitted + model.diffuse_emitted
        scale = max(emitted, np.finfo(float).tiny)
        self.assertLessEqual(abs(emitted - model.absorbed - float(model.photons.sum())),
                             3e-12 * scale)
        charge = electrons(model.gas) - model.initial_electrons
        charge_scale = max(electrons(model.gas), abs(model.initial_electrons),
                           model.absorbed, model.captures, np.finfo(float).tiny)
        self.assertLessEqual(abs(charge - model.absorbed + model.captures),
                             3e-12 * charge_scale)
        np.testing.assert_allclose(nuclei(model.gas), model.initial_nuclei,
                                   rtol=3e-13, atol=0.)
        self.assertTrue(np.all(model.gas >= 0))
        self.assertTrue(np.all(model.photons >= 0))

    def test_reference_opt_in_and_snapshot_default(self):
        self.assertEqual(SpectralES(side=1).photon_floor, 0.)
        self.assertEqual(snapshot_model().photon_floor, FLOOR)

    def test_invalid_floor(self):
        gas = np.array([[1., 0., 0., 0., 0.]])
        photons = np.array([[.1]])
        for floor in (-1., np.nan, np.inf):
            with self.subTest(floor=floor):
                with self.assertRaises(ValueError):
                    SpectralES(side=1, photon_floor=floor)
                with self.assertRaises(ValueError):
                    joint_exposure_update(gas, photons, [18.], [[True]],
                                          photon_floor=floor)

    def test_dormant_sources_accumulate_and_reactivate(self):
        model = SpectralES(side=1, helium=0., radii2=[0.], photon_floor=FLOOR)
        # Two individually dormant bands must stay dormant even though their
        # combined photon count exceeds the per-account floor.
        model.inject([((0, 0, 0), [.75 * FLOOR, .75 * FLOOR, 0.])])
        before = model.gas.copy()
        model.illuminate()
        np.testing.assert_array_equal(model.gas, before)
        self.assertEqual(model.photons[0, 0], .75 * FLOOR)
        self.assertEqual(model.absorbed, 0.)
        self.assert_model_conserves(model)

        model.inject([((0, 0, 0), [.75 * FLOOR, 0., 0.])])
        model.illuminate()
        self.assertGreater(model.gas[0, 1], 0.)
        self.assertGreater(model.absorbed, 0.)
        self.assertLess(model.photons[0, 0], 1.5 * FLOOR)
        self.assertEqual(model.photons[0, 1], .75 * FLOOR)
        self.assert_model_conserves(model)

    def test_floor_is_applied_after_each_debit_and_on_reentry(self):
        gas = np.array([[100., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]])
        photons = np.array([[1.1 * FLOOR], [1e-5]])
        membership = np.eye(2, dtype=bool)
        first = joint_exposure_update(gas, photons, [18.], membership,
                                      photon_floor=FLOOR, max_iterations=4000)
        retained = first["photons"][0, 0]
        # A midpoint step debits at most 2%: crossing the floor must leave
        # the inactive account intact, rather than run it down to zero.
        self.assertGreaterEqual(retained, .97 * FLOOR)
        self.assertLessEqual(retained, FLOOR)
        self.assert_joint_conserves(gas, photons, first)
        second = joint_exposure_update(first["species"], first["photons"], [18.],
                                       membership, photon_floor=FLOOR,
                                       max_iterations=4000)
        self.assertEqual(second["photons"][0, 0], retained)
        self.assert_joint_conserves(first["species"], first["photons"], second)

    def test_subnormal_account_does_not_stall_other_cells(self):
        # Without the floor, rate*exposure underflows in the dense cell.
        # Its unchanged photon account then constrains the other cell's step.
        gas = np.array([[100., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]])
        photons = np.array([[5e-322], [1e-5]])
        result = joint_exposure_update(gas, photons, [18.], np.eye(2, dtype=bool),
                                       photon_floor=FLOOR, max_iterations=4000)
        self.assertEqual(result["photons"][0, 0], photons[0, 0])
        np.testing.assert_array_equal(result["species"][0], gas[0])
        self.assertLess(result["photons"][1, 0], 2e-18)
        self.assertGreater(result["species"][1, 1], .999999 * photons[1, 0])
        self.assert_joint_conserves(gas, photons, result)

    def test_nearly_ionized_cell_releases_its_excess_photons(self):
        # Reduced snapshot-29 failure: a cell has already crossed its ES
        # threshold, while a remote subnormal account has a much larger hazard.
        gas = np.array([[1.78, 0., 0., 0., 0.],
                        [1e-5, 1., 0., 0., 0.]])
        photons = np.array([[3.66e-322], [.00119]])
        result = joint_exposure_update(gas, photons, [18.], np.eye(2, dtype=bool),
                                       photon_floor=FLOOR, max_iterations=4000)
        self.assertEqual(result["photons"][0, 0], photons[0, 0])
        self.assertLess(result["species"][1, 0], 1e-12)
        self.assertAlmostEqual(result["photons"][1, 0], .00118, delta=1e-12)
        self.assert_joint_conserves(gas, photons, result)

    def test_local_diffuse_accounts_are_retained_and_reactivated(self):
        model = snapshot_model()
        channel = model.primary_count  # H ground-state recombination photons.
        model.photons[0, channel] = .75 * FLOOR
        model.diffuse_emitted = .75 * FLOOR
        before = model.gas.copy()
        model.react(.01 * MYR, cell_volume=1e4)
        np.testing.assert_array_equal(model.gas, before)
        self.assertEqual(model.photons[0, channel], .75 * FLOOR)
        self.assertEqual(model.absorbed, 0.)
        self.assert_model_conserves(model)

        model.photons[0, channel] += .75 * FLOOR
        model.diffuse_emitted += .75 * FLOOR
        model.react(.01 * MYR, cell_volume=1e4)
        self.assertGreater(model.gas[0, 1], 0.)
        self.assertGreater(model.absorbed, 0.)
        self.assertLess(model.photons[0, channel], 1.5 * FLOOR)
        self.assert_model_conserves(model)

    def test_recombination_reactivates_dormant_diffuse_account(self):
        control, retained = snapshot_model(), snapshot_model()
        channel = retained.primary_count
        # A larger test floor makes the dormant contribution independently
        # measurable when recombination emission lifts the account above it.
        floor = 1e-4
        for model in (control, retained):
            model.photon_floor = floor
            model.gas[0, :2] = [.9, .1]
            model.initial_electrons = .1
        retained.photons[0, channel] = .75 * floor
        retained.diffuse_emitted = .75 * floor
        for model in (control, retained):
            model.react(10 * MYR, cell_volume=1e4)
            self.assert_model_conserves(model)
        self.assertGreater(control.diffuse_emitted, 0.)
        self.assertLess(control.diffuse_emitted, floor)
        self.assertEqual(control.absorbed, 0.)
        self.assertGreater(retained.absorbed, 0.)
        self.assertGreater(retained.gas[0, 1], control.gas[0, 1])
        self.assertLess(retained.photons[0, channel],
                        control.photons[0, channel] + .75 * floor)

    def test_density_conversion_preserves_subnormal_diffuse_credit(self):
        model = snapshot_model()
        model.gas[0, :2] = [.9, .1]
        model.initial_electrons = .1
        # H recombines, so this cell really enters the chemistry routine.
        # Its dormant HeIII-ground band has no new emission in this setup.
        channel = model.primary_count + 3
        model.photons[0, channel] = 3.66e-322
        model.diffuse_emitted = 3.66e-322
        self.assertEqual(model.photons[0, channel] / 1e4, 0.)
        model.react(.01 * MYR, cell_volume=1e4)
        self.assertGreater(model.captures, 0.)
        self.assertEqual(model.photons[0, channel], 3.66e-322)
        self.assert_model_conserves(model)

    def test_reference_react_preserves_subnormal_diffuse_credit(self):
        model = SpectralES(side=1, photon_floor=FLOOR)
        model.gas[0, :2] = [.9, .1]
        model.initial_electrons = .1
        channel = 3 + 3  # Three primary bands, then HeIII ground emission.
        model.photons[0, channel] = 3.66e-322
        model.diffuse_emitted = 3.66e-322
        self.assertEqual(model.photons[0, channel] / 1e4, 0.)
        model.react(.01 * MYR, cell_volume=1e4)
        self.assertGreater(model.captures, 0.)
        self.assertEqual(model.photons[0, channel], 3.66e-322)
        self.assert_model_conserves(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
