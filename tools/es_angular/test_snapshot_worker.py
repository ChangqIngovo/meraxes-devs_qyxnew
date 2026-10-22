"""Real-HDF5 snapshot, photon-flight, replay and conservation integration tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import h5py
import numpy as np

from snapshot_worker import (SCHEMA, SOLVER, ARRAY_COLUMNS, SnapshotModel, read_request,
                             load_checkpoint, evolve_snapshot, run)


class SnapshotWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_directory = tempfile.TemporaryDirectory(prefix="angular-worker-library-")
        supplied = os.environ.get("ES_ANGULAR_LIBRARY")
        if supplied:
            cls.library = Path(supplied).resolve()
        else:
            cls.library = Path(cls.build_directory.name)/"libangular_front.so"
            subprocess.run([sys.executable, str(Path(__file__).with_name("build.py")),
                            "--output", str(cls.library)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.build_directory.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="angular-worker-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root/"checkpoint.h5"
        self.counter = 0
        self.base = dict(schema_version=SCHEMA, solver=SOLVER, snapshot=0, start_snapshot=0,
                         grid_dim=2, dt_s=0., cell_volume_cm3=1e51,
                         mean_h_atoms_per_cell=1e48, helium_number_ratio=.08,
                         temperature_K=1e4, redshift=15., max_step_s=1e13,
                         source_energy_eV=18., angular_level=0, epsilon=.1,
                         speed_fraction=1e-7, source_spectrum_file="", run_identity="unit-run",
                         library_path=str(self.library))

    def request(self, *, density=None, source=None, **changes):
        attrs = dict(self.base, **changes)
        n = attrs["grid_dim"]**3
        density = np.ones(n) if density is None else np.asarray(density)
        source = np.zeros(n) if source is None else np.asarray(source)
        self.counter += 1
        path = self.root/f"request_{self.counter}.h5"
        with h5py.File(path, "w") as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
            handle.create_dataset("density", data=density)
            handle.create_dataset("source_photons", data=source)
        return path

    def advance(self, request, response=None):
        response = response or self.root/f"response_{request.stem}.h5"
        diagnostics = run(request, response, self.checkpoint)
        return response, diagnostics

    def assert_ledgers(self, diagnostics):
        for name in ("photon_error", "electron_error", "nucleus_error",
                     "diffuse_channel_error", "primary_resource_error"):
            self.assertLess(diagnostics[name], 2e-9, (name, diagnostics))
        self.assertAlmostEqual(diagnostics["stellar_emitted"]+diagnostics["diffuse_emitted"],
                               diagnostics["primary_absorbed"]+diagnostics["diffuse_absorbed"]
                               +diagnostics["primary_inventory"]+diagnostics["diffuse_pending"], 9)
        self.assertAlmostEqual(diagnostics["electrons"]-diagnostics["initial_electrons"],
                               diagnostics["primary_absorbed"]+diagnostics["diffuse_absorbed"]
                               -diagnostics["gross_recombinations"]+diagnostics["external_electrons"], 9)

    def test_neutral_initialization_and_response_schema(self):
        density = np.array([0., .25, .5, 1., 1.5, 2., 3., 4.])
        request = self.request(density=density, start_snapshot=7, snapshot=7)
        response, diagnostics = self.advance(request)
        self.assert_ledgers(diagnostics)
        self.assertEqual(diagnostics["substeps"], 0)
        with h5py.File(response, "r") as handle:
            self.assertEqual(handle.attrs["schema_version"], 2)
            self.assertEqual(handle.attrs["solver"], SOLVER)
            self.assertEqual(handle.attrs["snapshot"], 7)
            self.assertEqual(handle.attrs["run_identity"], "unit-run")
            self.assertEqual(handle.attrs["mean_electron_per_nucleus"], 0.)
            for name in ("xH", "xHeI"):
                np.testing.assert_array_equal(handle[name][...], np.ones(8))
            for name in ("xHeII", "xHeIII", "recombinations_HII"):
                np.testing.assert_array_equal(handle[name][...], np.zeros(8))
            np.testing.assert_array_equal(handle["gas"][:, 0], density)
            np.testing.assert_allclose(handle["gas"][:, 2], .08*density)
            engine = json.loads(handle["engine_state"][()])
            self.assertEqual(engine["entries"], [])
            self.assertEqual(engine["resource_ledgers"], [])

    def test_initialization_and_request_guards(self):
        source = np.zeros(8); source[0] = 1e48
        for changes in (dict(snapshot=1), dict(dt_s=1.), dict(source=source),
                        dict(solver="old-allocation"), dict(temperature_K=2e4),
                        dict(speed_fraction=0.), dict(speed_fraction=1.1),
                        dict(epsilon=-.1), dict(schema_version=1)):
            with self.subTest(changes=changes):
                request = self.request(**changes)
                with self.assertRaises(ValueError):
                    self.advance(request)
                self.assertFalse(self.checkpoint.exists())
        request = self.request(density=np.ones((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.advance(request)
        request = self.request()
        with h5py.File(request, "r+") as handle:
            del handle.attrs["start_snapshot"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.advance(request)

    def test_c_bridge_one_element_numeric_attributes(self):
        request = self.request()
        reference, _, reference_digest = read_request(request)
        with h5py.File(request, "r+") as handle:
            for name in list(handle.attrs):
                value = handle.attrs[name]
                if np.asarray(value).dtype.kind in "iuf":
                    del handle.attrs[name]
                    handle.attrs[name] = np.array([value])
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            metadata, _, digest = read_request(request)
            _, diagnostics = self.advance(request)
        self.assertEqual(metadata, reference)
        self.assertEqual(digest, reference_digest)
        self.assert_ledgers(diagnostics)
        for invalid in (np.array([2, 2]), np.array([[2]])):
            with h5py.File(request, "r+") as handle:
                del handle.attrs['grid_dim']
                handle.attrs['grid_dim'] = invalid
            with self.assertRaisesRegex(ValueError, 'numeric scalar'):
                read_request(request)

    def test_density_remap_preserves_fractions_and_photon_locations(self):
        density = np.ones(8); density[1] = 0.
        init = self.request(density=density)
        metadata, arrays, _ = read_request(init)
        model = SnapshotModel(metadata, density)
        source = np.zeros(8); source[0] = 3e48
        active = self.request(snapshot=1, dt_s=1e13, density=density, source=source)
        metadata1, arrays1, _ = read_request(active)
        self.assert_ledgers(evolve_snapshot(model, metadata1, arrays1))
        before_gas = model.gas.copy()
        before_flight = copy.deepcopy(model.front.export_state()["entries"])
        before_diffuse = model.diffuse_pending.copy()
        self.assertTrue(before_flight)
        self.assertGreater(model.photo_counts[0, 0], .99)
        new_density = density.copy(); new_density[0] = 2.; new_density[1] = 3.
        model.remap_density(new_density, .08)
        np.testing.assert_allclose(model.gas[0], 2*before_gas[0], rtol=1e-15)
        np.testing.assert_array_equal(model.gas[1], [3., 0., .24, 0., 0.])
        np.testing.assert_array_equal(model.matter_exchange, model.gas-before_gas)
        self.assertEqual(model.front.export_state()["entries"], before_flight)
        np.testing.assert_array_equal(model.diffuse_pending, before_diffuse)
        self.assert_ledgers(model.audit())
        self.assertGreater(model.audit()["external_electrons"], 0.)

    def test_three_snapshots_hhe_gas_ledgers_and_response_count_units(self):
        spectrum = self.root/"spectrum.txt"
        spectrum.write_text("18 0.5\n30 0.25\n60 0.25\n")
        self.base["source_spectrum_file"] = str(spectrum)
        self.advance(self.request())
        source = np.zeros(8); source[0] = .65e48; source[7] = .4e48
        first = self.request(snapshot=1, dt_s=1.5e13, source=source)
        _, first_audit = self.advance(first)
        self.assertEqual(first_audit["substeps"], 2)
        density = np.ones(8); density[0] = 1.2; density[3] = .7
        source2 = np.zeros(8); source2[1] = .3e48
        second = self.request(snapshot=2, dt_s=1e13, source=source2, density=density,
                              redshift=14., cell_volume_cm3=1.2e51)
        response, diagnostics = self.advance(second)
        self.assert_ledgers(diagnostics)
        self.assertAlmostEqual(diagnostics["stellar_emitted"], 1.35, 12)
        self.assertGreater(diagnostics["gross_recombinations"], 0.)
        with h5py.File(response, "r") as handle:
            gas = handle["gas"][...]
            np.testing.assert_allclose(gas[:, :2].sum(1), density, rtol=1e-12)
            np.testing.assert_allclose(gas[:, 2:].sum(1), .08*density, rtol=1e-12)
            self.assertGreater(gas[:, 3:].sum(), 0.)
            np.testing.assert_allclose(handle["recombinations_HII"][...],
                                       handle["capture_counts"][:, 0]*1e48, rtol=1e-15)
            np.testing.assert_allclose(handle["snapshot_recombinations_HII"][...],
                                       handle["interval_capture_counts"][:, 0]*1e48, rtol=1e-15)
            self.assertGreater(handle["recombinations_HII"][...].sum(),
                               handle["snapshot_recombinations_HII"][...].sum())
            electron_fraction = np.sum(gas[:, 1]+gas[:, 3]+2*gas[:, 4])/gas.sum()
            self.assertAlmostEqual(handle.attrs["mean_electron_per_nucleus"], electron_fraction, 14)
            np.testing.assert_allclose(handle["xHeI"][...]+handle["xHeII"][...]
                                       +handle["xHeIII"][...], 1., rtol=1e-14)

    def test_inflight_checkpoint_matches_uninterrupted_and_never_reemits(self):
        init = self.request()
        metadata0, arrays0, _ = read_request(init)
        uninterrupted = SnapshotModel(metadata0, arrays0["density"])
        evolve_snapshot(uninterrupted, metadata0, arrays0)
        self.advance(init)
        source = np.zeros(8); source[0] = 3e48
        first = self.request(snapshot=1, dt_s=1e13, source=source)
        meta1, arrays1, _ = read_request(first)
        evolve_snapshot(uninterrupted, meta1, arrays1)
        _, audit1 = self.advance(first)
        loaded, _, _, _ = load_checkpoint(self.checkpoint, meta1)
        flight1 = loaded.front.export_state()["entries"]
        self.assertTrue(flight1)
        self.assertTrue(all(row[-2] > 0 for row in flight1))
        self.assertTrue(any(any(value != 0 for value in row[3:6]) for row in flight1))
        self.assertEqual(loaded.front.export_state(), uninterrupted.front.export_state())
        primary1 = loaded.photo_counts.copy()
        second = self.request(snapshot=2, dt_s=1e13)
        meta2, arrays2, _ = read_request(second)
        evolve_snapshot(uninterrupted, meta2, arrays2)
        _, audit2 = self.advance(second)
        restored, previous, _, _ = load_checkpoint(self.checkpoint, meta2)
        self.assertEqual(previous, 2)
        self.assertEqual(restored.front.export_state(), uninterrupted.front.export_state())
        for name in ARRAY_COLUMNS:
            np.testing.assert_array_equal(getattr(restored, name), getattr(uninterrupted, name))
        self.assertEqual(audit2["stellar_emitted"], audit1["stellar_emitted"])
        self.assertAlmostEqual(audit2["primary_inventory"], audit1["primary_inventory"], 13)
        np.testing.assert_array_equal(restored.photo_counts, primary1)
        self.assertGreater(restored.gas[0, 0], loaded.gas[0, 0])
        self.assert_ledgers(audit2)

    def test_replay_is_idempotent_and_regenerates_response(self):
        self.advance(self.request())
        source = np.zeros(8); source[0] = .5e48
        request = self.request(snapshot=1, dt_s=1e13, source=source)
        response, audit1 = self.advance(request)
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        audit2 = run(request, response, self.checkpoint)
        self.assertEqual(audit1, audit2)
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before)
        with h5py.File(response, "r") as handle:
            gas = handle["gas"][...]
        response.unlink()
        run(request, response, self.checkpoint)
        with h5py.File(response, "r") as handle:
            np.testing.assert_array_equal(handle["gas"][...], gas)
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before)

    def test_response_commit_crash_retries_from_previous_checkpoint(self):
        self.advance(self.request())
        checkpoint_before = self.checkpoint.read_bytes()
        source = np.zeros(8); source[0] = 3e48
        request = self.request(snapshot=1, dt_s=1e13, source=source)
        response = self.root/"crash_response.h5"
        replace = os.replace

        def fail_checkpoint(source_path, destination):
            if Path(destination) == self.checkpoint:
                raise OSError("simulated checkpoint commit interruption")
            replace(source_path, destination)

        with patch("snapshot_worker.os.replace", side_effect=fail_checkpoint):
            with self.assertRaisesRegex(OSError, "interruption"):
                run(request, response, self.checkpoint)
        self.assertTrue(response.is_file())
        self.assertEqual(self.checkpoint.read_bytes(), checkpoint_before)
        with h5py.File(response, "r") as handle:
            first_state = json.loads(handle["engine_state"][()])
            first_gas = handle["gas"][...]
        self.assertFalse(list(self.root.glob("*.pending.*")))
        diagnostics = run(request, response, self.checkpoint)
        self.assert_ledgers(diagnostics)
        with h5py.File(response, "r") as handle:
            self.assertEqual(json.loads(handle["engine_state"][()]), first_state)
            np.testing.assert_array_equal(handle["gas"][...], first_gas)
        with h5py.File(self.checkpoint, "r") as handle:
            self.assertEqual(handle.attrs["snapshot"], 1)
            self.assertEqual(json.loads(handle["engine_state"][()]), first_state)

    def test_reject_changed_replay_run_configuration_and_discontinuity(self):
        self.advance(self.request())
        for changes, error in ((dict(density=np.arange(8)), "digest"),
                               (dict(snapshot=2), "does not follow"),
                               (dict(snapshot=1, epsilon=.2), "configuration"),
                               (dict(snapshot=1, run_identity="different"), "another run")):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, error):
                    self.advance(self.request(**changes))
        changed_library = self.root/"modified_library.so"
        shutil.copyfile(self.library, changed_library)
        with changed_library.open("ab") as handle:
            handle.write(b"different numerical library identity")
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.advance(self.request(snapshot=1, library_path=str(changed_library)))
        request = self.request(snapshot=1)
        wrong_response = self.root/"unrelated.h5"
        with h5py.File(wrong_response, "w") as handle:
            handle.attrs["run_identity"] = "another"
        with self.assertRaisesRegex(ValueError, "another snapshot/run"):
            self.advance(request, wrong_response)

    def test_reject_tampered_engine_configuration_and_ledger(self):
        self.advance(self.request())
        meta, _, _ = read_request(self.request())
        original = self.checkpoint.read_bytes()
        for modification in ("epsilon", "gas"):
            with self.subTest(modification=modification):
                self.checkpoint.write_bytes(original)
                with h5py.File(self.checkpoint, "r+") as handle:
                    state = json.loads(handle["engine_state"][()])
                    if modification == "epsilon":
                        state["epsilon"] += .1
                    else:
                        state["gas"][0][1] += .2
                    handle["engine_state"][()] = json.dumps(state)
                with self.assertRaises(ValueError):
                    load_checkpoint(self.checkpoint, meta)

    def test_zero_epsilon_skips_prior_and_spectrum_identity_changes_rejected(self):
        self.base["epsilon"] = 0.
        spectrum = self.root/"spectrum.txt"
        spectrum.write_text("18 1\n60 0\n")
        self.base["source_spectrum_file"] = str(spectrum)
        self.advance(self.request())
        source = np.zeros(8); source[0] = .25e48
        _, diagnostics = self.advance(self.request(snapshot=1, dt_s=1e13, source=source))
        self.assertIsNone(diagnostics["es_prior"])
        self.assert_ledgers(diagnostics)
        spectrum.write_text("18 1\n60 1\n")
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.advance(self.request(snapshot=2))


if __name__ == "__main__":
    unittest.main()
