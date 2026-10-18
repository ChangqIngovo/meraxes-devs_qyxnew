#!/usr/bin/env python3
"""Small integration tests for the online state/accounting contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from es_hhe_reference import MYR
from snapshot_worker import REFERENCE_SHA256, SnapshotModel, load_checkpoint, read_request, run


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="es_snapshot_test_")
        self.directory = Path(self.temporary.name)
        self.request = self.directory / "request.h5"
        self.checkpoint = self.directory / "checkpoint.h5"
        self.normalization = 1e60

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, snapshot, photons=0., density=1., dt=0., **attrs):
        defaults = dict(snapshot=snapshot, grid_dim=1, dt_s=dt,
            cell_volume_cm3=1e64, mean_h_atoms_per_cell=self.normalization,
            helium_number_ratio=.081, radius_ratio=1.2, max_radius_cells=0.,
            run_identity="unittest", temperature_K=10000., max_step_s=MYR,
            redshift=10., source_energy_eV=60.)
        defaults.update(attrs)
        with h5py.File(self.request, "w") as handle:
            handle.attrs.update(defaults)
            handle["density"] = np.full(defaults["grid_dim"]**3, density)
            handle["source_photons"] = np.full(defaults["grid_dim"]**3, photons * self.normalization)
        return self.directory / f"response_{snapshot}.h5"

    def advance(self, snapshot, **kwargs):
        response = self.write(snapshot, **kwargs)
        diagnostics = run(self.request, response, self.checkpoint)
        metadata, _, _ = read_request(self.request)
        model, _, _, _ = load_checkpoint(self.checkpoint, metadata)
        return model, response, diagnostics

    def test_reference_frozen(self):
        path = Path(__file__).with_name("es_hhe_reference.py")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), REFERENCE_SHA256)

    def test_default_radius_spans_periodic_box(self):
        self.write(0, grid_dim=4, max_radius_cells=0.)
        metadata, _, _ = read_request(self.request)
        model = SnapshotModel(metadata)
        self.assertGreater(len(model.radii2), 1)
        self.assertEqual(model.radii2[-1], 12.)

    def test_density_remap_retains_minority_species(self):
        self.write(0)
        metadata, _, _ = read_request(self.request)
        model = SnapshotModel(metadata)
        model.gas[0, :2] = [1e-30, 1.]
        model.initial_electrons = model.electrons()
        model.remap_density(np.array([2.]), .081)
        self.assertEqual(model.gas[0, 0], 2e-30)
        self.assertEqual(model.gas[0, 1], 2.)

    def test_continuity_replay_and_source_off(self):
        initial, _, _ = self.advance(0)
        self.assertEqual(initial.gas[0, 0], 1.)
        illuminated, response, diagnostic = self.advance(1, photons=.6, dt=.01 * MYR)
        self.assertGreater(illuminated.gas[0, 1], 0.)
        self.assertGreater(illuminated.gas[0, 4], 0.)
        replay = run(self.request, response, self.checkpoint)
        self.assertEqual(diagnostic, replay)
        with h5py.File(self.checkpoint, "r") as handle:
            np.testing.assert_array_equal(handle["gas"], illuminated.gas)
            np.testing.assert_array_equal(handle["photons"], illuminated.photons)
        dark, response, diagnostic = self.advance(2, dt=10 * MYR)
        # He recombination photons can increase HII after primary sources stop;
        # this coupled case loses total electrons and HeIII, not necessarily HII.
        self.assertLess(dark.electrons(), illuminated.electrons())
        self.assertLess(dark.gas[0, 4], illuminated.gas[0, 4])
        self.assertGreater(dark.capture_counts[0, 0], illuminated.capture_counts[0, 0])
        self.assertGreater(dark.capture_counts[0, 2], illuminated.capture_counts[0, 2])
        self.assertAlmostEqual(dark.emitted, .6, places=14)
        for key in ("photon_error", "electron_error", "nuclei_error", "matter_transfer_nuclei_error"):
            self.assertLess(abs(diagnostic[key]), 1e-10, key)
        with h5py.File(response, "r") as handle:
            self.assertEqual(handle["xH"].shape, (1,))
            self.assertAlmostEqual(float(handle["HII_recombinations"][0]) / self.normalization,
                                   dark.capture_counts[0, 0], places=14)
            self.assertAlmostEqual(sum(float(handle[key][0]) for key in ("xHeI", "xHeII", "xHeIII")), 1.)

    def test_density_remap_with_retained_photons(self):
        self.advance(0)
        before, _, _ = self.advance(1, photons=2., dt=.001 * MYR)
        self.assertGreater(before.photons.sum(), 0.)
        after, _, diagnostic = self.advance(2, density=.2)
        np.testing.assert_allclose(after.gas, before.gas * .2, rtol=2e-14)
        np.testing.assert_array_equal(after.photons, before.photons)
        np.testing.assert_allclose(after.matter_exchange, before.gas * -.8, rtol=2e-14)
        self.assertLess(abs(diagnostic["electron_error"]), 1e-10)
        vacuum, _, _ = self.advance(3, density=0.)
        np.testing.assert_array_equal(vacuum.gas, 0.)
        refilled, _, _ = self.advance(4, density=.4)
        np.testing.assert_allclose(refilled.gas, [[.4, 0., .4 * .081, 0., 0.]])
        np.testing.assert_array_equal(refilled.photons, before.photons)

    def test_arbitrary_primary_spectrum(self):
        spectrum = self.directory / "spectrum.txt"
        spectrum.write_text("# photon number weights\n18 1\n30 2\n60 3\n60 4\n")
        self.advance(0, source_spectrum_file=str(spectrum))
        model, _, diagnostic = self.advance(1, source_spectrum_file=str(spectrum), photons=.1, dt=.001 * MYR)
        self.assertEqual(model.primary_count, 4)
        self.assertEqual(model.photons.shape, (1, 12))
        np.testing.assert_allclose(model.source_emitted_by_group, [.01, .02, .03, .04], rtol=2e-14)
        self.assertGreater(model.capture_counts.sum(), 0.)
        self.assertLess(abs(diagnostic["photon_error"]), 1e-10)

    def test_order_identity_and_configuration(self):
        self.advance(0)
        for snapshot, attrs, expected in ((2, {}, "does not follow"),
                (1, {"run_identity": "other"}, "another run"),
                (1, {"source_energy_eV": 30.}, "configuration")):
            response = self.write(snapshot, **attrs)
            with self.assertRaisesRegex(ValueError, expected):
                run(self.request, response, self.checkpoint)
        with h5py.File(self.checkpoint, "r") as handle:
            self.assertEqual(handle.attrs["snapshot"], 0)

    def test_invalid_input_transaction(self):
        response = self.write(0, dt=MYR, photons=.1)
        with self.assertRaisesRegex(ValueError, "First ES snapshot"):
            run(self.request, response, self.checkpoint)
        self.assertFalse(self.checkpoint.exists())
        self.advance(0)
        checksum = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        for changes in ({"density": float("nan")}, {"photons": -1.},
                        {"temperature_K": 20000.}, {"radius_ratio": 1.}):
            response = self.write(1, **changes)
            with self.assertRaises(ValueError):
                run(self.request, response, self.checkpoint)
            self.assertFalse(response.exists())
            self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), checksum)

    def test_fraction_and_count_normalization(self):
        # Change physical normalization and volume together: physical density
        # and all dimensionless species/photon counts must remain identical.
        self.advance(0)
        first, _, _ = self.advance(1, photons=.3, dt=.01 * MYR)
        other = self.directory / "other_checkpoint.h5"
        self.normalization *= 10
        for snapshot, photons, dt in ((0, 0., 0.), (1, .3, .01 * MYR)):
            response = self.write(snapshot, photons=photons, dt=dt, cell_volume_cm3=1e65,
                                  run_identity="scaled")
            response = response.with_name("scaled_" + response.name)
            run(self.request, response, other)
        metadata, _, _ = read_request(self.request)
        second, _, _, _ = load_checkpoint(other, metadata)
        np.testing.assert_allclose(first.gas, second.gas, rtol=3e-13, atol=2e-15)
        np.testing.assert_allclose(first.photons, second.photons, rtol=3e-13, atol=2e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
