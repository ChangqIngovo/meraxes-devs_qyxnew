#!/usr/bin/env python3
"""Scientific and file-format regression checks for compare_pc_es.py.

Run with ``python tools/es_hhe/test_compare_pc_es.py``. Set
COMPARE_TEST_EVIDENCE to retain the complete toy CLI inputs and figures.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pc_es as comparison


COSMOLOGY = dict(BoxSize=10., Hubble_h=.6751, OmegaM=.3121,
                 OmegaLambda=.6879, OmegaR=0., OmegaK=0.)


def make_run(root, n=4):
    (root/'output').mkdir(parents=True)
    (root/'provenance').mkdir()
    parameters = dict(COSMOLOGY, FileNameGalaxies='meraxes', ReionGridDim=n)
    (root/'input.par').write_text(''.join(f'{key} : {value}\n' for key, value in parameters.items()))
    return root/'output'


def write_grid(path, field, redshift=None, density=None):
    with h5py.File(path, 'w') as f:
        f['xH'] = field
        if redshift is not None:
            f.attrs['Redshift'] = redshift
        if density is not None:
            f['deltax'] = density - 1


def history(values, z=None, params=None):
    if z is None:
        z = np.arange(len(values), 0, -1) + 6
    return SimpleNamespace(history=[dict(redshift=float(a), xHI_volume=float(b))
                                    for a, b in zip(z, values)],
                           params=COSMOLOGY if params is None else params)


class PhysicsChecks(unittest.TestCase):
    def test_periodic_opposite_faces_are_connected_on_each_axis(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                mask = np.zeros((5, 5, 5), dtype=bool)
                p, q = [2, 2, 2], [2, 2, 2]
                p[axis], q[axis] = 0, 4
                mask[tuple(p)] = mask[tuple(q)] = True
                np.testing.assert_array_equal(comparison.periodic_components(mask), [2])

    def test_periodic_diagonal_and_corner_contacts_do_not_connect(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        for point in [(0, 0, 0), (4, 4, 0), (4, 0, 4), (0, 4, 4)]:
            mask[point] = True
        np.testing.assert_array_equal(comparison.periodic_components(mask), [1, 1, 1, 1])
        np.testing.assert_array_equal(comparison.periodic_components(np.zeros_like(mask)), [])
        np.testing.assert_array_equal(comparison.periodic_components(np.ones_like(mask)), [125])

    def test_unique_crossing_interpolates_inside_bracket(self):
        value = comparison.crossing([12, 10, 8], [1, .8, .5], .1)
        self.assertEqual(value['status'], 'measured')
        self.assertAlmostEqual(value['redshift'], 11.)
        self.assertEqual(value['bracket_redshifts'], [12., 10.])

    def test_right_and_left_censoring_do_not_extrapolate(self):
        right = comparison.crossing([12, 10, 8], [1, .96, .91], .1)
        left = comparison.crossing([12, 10, 8], [.8, .6, .4], .1)
        self.assertEqual(right, dict(status='not_reached', redshift=None))
        self.assertEqual(left, dict(status='already_passed_at_first_sample', redshift=None))
        value = comparison.duration(history([1, .96, .91]))['durations']['10-90']
        self.assertEqual(value['status'], 'not_measurable_in_available_history')
        self.assertIsNone(value['delta_z'])
        self.assertIsNone(value['delta_t_myr'])

    def test_recrossing_cannot_be_reported_as_unique_eor_time(self):
        value = comparison.crossing([14, 12, 10, 8], [1, .8, .95, .7], .1)
        self.assertEqual(value, dict(status='multiple_crossings', redshift=None))
        # A rebound which remains below the requested threshold is still a
        # unique crossing, and must not cause blanket monotonicity rejection.
        unique = comparison.crossing([14, 12, 10, 8], [1, .8, .85, .7], .1)
        self.assertEqual(unique['status'], 'measured')
        self.assertAlmostEqual(unique['redshift'], 13.)

    def test_measured_duration_and_cosmic_time_sign(self):
        value = comparison.duration(history([1, 0], z=[10, 8]))
        self.assertAlmostEqual(value['thresholds']['10']['redshift'], 9.8)
        self.assertAlmostEqual(value['thresholds']['90']['redshift'], 8.2)
        self.assertAlmostEqual(value['durations']['10-90']['delta_z'], 1.6)
        self.assertGreater(value['durations']['10-90']['delta_t_myr'], 0)
        # An Einstein-de Sitter universe has an analytic cosmic age.
        z, h = 8., .7
        expected = 2/(3*(100*h/3.0856775814913673e19)) / (1+z)**1.5 / (365.25*86400*1e6)
        actual = comparison.cosmic_age_myr(z, dict(Hubble_h=h, OmegaM=1, OmegaLambda=0))
        self.assertAlmostEqual(actual, expected, places=7)


class InputChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='compare-pc-es-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_root_grids_use_alist_and_overdensity_mass_weights(self):
        run_root = self.root/'run'
        output = make_run(run_root)
        np.savetxt(run_root/'provenance/a_list.txt', [1/13, 1/11])
        field = np.arange(64, dtype=float).reshape(4, 4, 4)/63
        density = 1 + np.arange(64).reshape(field.shape)/10
        write_grid(output/'meraxes_grids_1.hdf5', field, density=density)
        for location in [run_root, output, output/'meraxes_grids_1.hdf5']:
            with self.subTest(location=location):
                run = comparison.Run('test', location)
                self.assertAlmostEqual(run.rows[1]['redshift'], 10)
                self.assertAlmostEqual(run.rows[1]['xHI_volume'], field.mean())
                self.assertAlmostEqual(run.rows[1]['xHI_mass'], np.sum(field*density)/density.sum())
                np.testing.assert_array_equal(run.field(1), field)

    def test_master_external_links_and_redshift_cosmology_attributes(self):
        output = make_run(self.root/'run')
        field = np.full((4, 4, 4), .85)
        # A nonstandard grid filename ensures discovery really follows the
        # external link, rather than finding the same grid by a glob.
        write_grid(output/'linked-field.hdf5', field)
        master = output/'meraxes.hdf5'
        with h5py.File(master, 'w') as f:
            params = f.create_group('InputParams')
            for key, value in COSMOLOGY.items():
                params.attrs[key] = [value]
            group = f.create_group('Snap007')
            group.attrs['Redshift'] = [9.25]
            group['Grids'] = h5py.ExternalLink('linked-field.hdf5', '/')
        run = comparison.Run('test', master)
        self.assertEqual(run.rows[7]['redshift'], 9.25)
        self.assertEqual(run.params['BoxSize'], 10.)
        np.testing.assert_array_equal(run.field(7), field)
        (output/'linked-field.hdf5').unlink()
        with self.assertRaisesRegex(RuntimeError, 'external grid files'):
            comparison.Run('broken', master)

    def test_pc_state_flat_grid_and_hydrogen_weights(self):
        output = make_run(self.root/'run')
        field = np.linspace(.2, 1, 64).reshape(4, 4, 4)
        weights = np.arange(1, 65, dtype=float)
        gas = np.zeros((64, 5))
        gas[:, 0] = weights*field.ravel()
        gas[:, 1] = weights*(1-field.ravel())
        gas[:, 2] = 1000  # Helium must not enter the hydrogen mass weighting.
        with h5py.File(output/'es_state_003.hdf5', 'w') as f:
            f.attrs.update(grid_dim=4, snapshot=3, redshift=10.)
            f['xH'], f['gas'] = field.ravel(), gas
        run = comparison.Run('test', output)
        np.testing.assert_array_equal(run.field(3), field)
        self.assertAlmostEqual(run.rows[3]['xHI_mass'], np.average(field.ravel(), weights=weights))
        self.assertNotAlmostEqual(run.rows[3]['xHI_mass'], field.mean())

    def test_empty_placeholder_supplies_history_but_not_spatial_grid(self):
        output = make_run(self.root/'run')
        with h5py.File(output/'meraxes_grids_0.hdf5', 'w') as f:
            f.attrs['Redshift'] = 20.
            ds = f.create_dataset('xH', shape=(0,), dtype='f4')
            ds.attrs['volume_weighted_global_xH'] = .99
            ds.attrs['mass_weighted_global_xH'] = .98
        run = comparison.Run('test', output)
        self.assertAlmostEqual(run.rows[0]['xHI_volume'], .99)
        self.assertAlmostEqual(run.rows[0]['xHI_mass'], .98)
        self.assertEqual(run.rows[0]['source'], 'grid_attributes')
        self.assertNotIn(0, run.grids)
        with self.assertRaises(ValueError):
            run.field(0)

    def test_unmeasured_log_snapshots_are_skipped_without_fabricated_neutrality(self):
        log = self.root/'run.log'
        text = ''.join(f'Snapshot {snap} (z = {30-snap:.3f})\nNo sources\n' for snap in range(7))
        text += ('Snapshot 7 (z = 23.000)\n xH = 0.99999 VS 0.99998\n'
                 'Snapshot 8 (z = 22.000)\n'
                 'ES/HHe snapshot 8 complete: volume xHI=0.99876543, iterations=1\n'
                 ' xH = 0.99877 VS 0.99870\nSnapshot 9 (z = 21.000)\n')
        log.write_text(text)
        rows, _ = comparison.read_log(log)
        self.assertEqual(set(rows), {7, 8})
        self.assertEqual(rows[8]['xHI_volume'], .99876543)
        self.assertEqual(rows[8]['xHI_mass'], .99870)
        run = comparison.Run('log only', log=log)
        self.assertEqual([r['snapshot'] for r in run.history], [7, 8])
        self.assertFalse(run.grids)

    def test_fixed_slice_preserves_xyz_axis_order_and_coordinate(self):
        field = np.arange(64, dtype=float).reshape(4, 4, 4)/63
        pc = SimpleNamespace(label='PC', rows={1: dict(redshift=9.)}, field=lambda s: field)
        es = SimpleNamespace(label='ES', rows={1: dict(redshift=9.)}, field=lambda s: 1-field)
        for axis in range(3):
            with self.subTest(axis=axis):
                with patch.object(comparison, 'save_plot') as save:
                    records = comparison.morphology_plot(pc, es, [(1, 1, None)], self.root,
                                                         'morphology_same_z', axis, 1, 10., 1)
                    figure = save.call_args.args[0]
                    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(),
                                                  np.take(field, 1, axis=axis).T)
                    self.assertIn(f'Fixed {"xyz"[axis]} plane at 3.750', figure._suptitle.get_text())
                    self.assertGreater(records[0]['rms_difference'], 0)
                    comparison.plt.close(figure)


class EndToEndChecks(unittest.TestCase):
    def test_cli_produces_history_duration_and_morphology_without_input_mutation(self):
        evidence = os.environ.get('COMPARE_TEST_EVIDENCE')
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix='toy-', dir=evidence))
        else:
            temporary = tempfile.TemporaryDirectory(prefix='compare-pc-es-cli-')
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        for label, offset in [('pc', 0), ('es', 1)]:
            output = make_run(root/label, n=8)
            for snap, (z, count) in enumerate([(12., 32), (10., 80), (8., 150)]):
                field = np.ones((8, 8, 8))
                field.ravel()[:count] = .05
                field = np.roll(field, offset, axis=0)
                write_grid(output/f'meraxes_grids_{snap}.hdf5', field, redshift=z,
                           density=np.ones_like(field))
        inputs = list((root/'pc').rglob('*')) + list((root/'es').rglob('*'))
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.is_file()}
        out = root/'figures'
        command = [sys.executable, str(Path(comparison.__file__).resolve()),
                   '--pc-run', str(root/'pc'), '--es-run', str(root/'es'),
                   '--output-dir', str(out), '--snapshots', '0', '1', '2',
                   '--match-xhi', str(1-.95*80/512), '--match-tolerance', '.001',
                   '--slice-axis', 'y', '--slice-index', '1', '--min-component-cells', '1']
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        report = json.loads((out/'comparison.json').read_text())
        self.assertEqual(report['morphology_status'], 'completed')
        self.assertEqual(len(report['same_z']), 3)
        self.assertEqual(len(report['matched_xhi']), 1)
        self.assertGreater(report['same_z'][1]['rms_difference'], 0)
        self.assertAlmostEqual(report['same_z'][1]['pc_mean_xHI'], report['same_z'][1]['es_mean_xHI'])
        self.assertEqual(report['same_z'][1]['pc_components']['largest_cells'], 80)
        duration = json.loads((out/'eor_duration.json').read_text())
        self.assertIsNone(duration['PC']['durations']['10-90']['delta_z'])
        for label in ['pc', 'es']:
            with (out/f'history_{label}.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r['snapshot']) for r in rows], [0, 1, 2])
        for name in ['ionization_history', 'morphology_same_z', 'morphology_matched_xhi',
                     'component_sizes_same_z', 'component_sizes_matched_xhi']:
            for suffix in ['png', 'pdf']:
                self.assertGreater((out/f'{name}.{suffix}').stat().st_size, 1000)
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        if evidence:
            print(f'\nRetained toy CLI evidence: {root}', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
