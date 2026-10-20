#!/usr/bin/env python3
"""Conservative scale screening preserves ES events and partial retries."""
import unittest
from unittest import mock

import numpy as np

from es_hhe_reference import RadiusScreen, SpectralES


class RadiusCertificates(unittest.TestCase):
    def hydrogen(self, neutral=1., photons=.1):
        return np.array([[neutral, 1.-neutral, .08, 0., 0., photons]])

    def certificate(self, aggregate, cross=None, reserve=None):
        if cross is None:
            cross = np.array([[1., 0., 0.]])
        if reserve is None:
            reserve = np.zeros(aggregate.shape[1])
        screen = RadiusScreen(2)
        screen.record(1, aggregate, cross, 1e-5, reserve)
        return screen

    def test_capture_bound_preserves_then_expires_certificate(self):
        aggregate = self.hydrogen(photons=.4)
        screen = self.certificate(aggregate)
        before = aggregate[:, :5].copy()
        photons = aggregate[:, 5:].copy()
        self.assertTrue(screen.can_skip(1))
        after = before.copy(); after[0, 0] -= .1; after[0, 1] += .1
        screen.advance(before, after, photons, photons, 1e-5)
        self.assertTrue(screen.can_skip(1))
        last = after.copy(); last[0, 0] -= .7; last[0, 1] += .7
        screen.advance(after, last, photons, photons, 1e-5)
        self.assertFalse(screen.can_skip(1))

    def test_hydrogen_capture_does_not_spend_helium_margin(self):
        aggregate = self.hydrogen(photons=.01)
        screen = self.certificate(aggregate)
        before = aggregate[:, :5].copy(); after = before.copy()
        after[0, 0] -= .5; after[0, 1] += .5
        screen.advance(before, after, aggregate[:, 5:], aggregate[:, 5:], 1e-5)
        self.assertTrue(screen.can_skip(1))
        self.assertGreater(screen.margins[1, 1], .079)

    def test_previously_unusable_helium_target_is_included(self):
        # Photons can ionize HeII but initially no HeII exists. A later
        # HeI->HeII update makes absorption usable without changing HeI+HeII.
        aggregate = np.array([[0., 0., 1., 0., 0., .2]])
        screen = self.certificate(aggregate, np.array([[0., 0., 1.]]))
        self.assertTrue(screen.can_skip(1))
        before = aggregate[:, :5].copy(); after = before.copy()
        after[0, 2] -= .3; after[0, 3] += .3
        screen.advance(before, after, aggregate[:, 5:], aggregate[:, 5:], 1e-5)
        self.assertTrue(screen.can_skip(1))
        self.assertGreater(screen.margins[1, 2], .799)

    def test_ineligible_target_conservatively_disables_screen(self):
        self.assertFalse(self.certificate(self.hydrogen(neutral=1e-6)).can_skip(1))

    def test_photon_increase_invalidates_certificate(self):
        aggregate = self.hydrogen()
        screen = self.certificate(aggregate)
        gas = aggregate[:, :5]
        photons = aggregate[:, 5:]
        screen.advance(gas, gas, photons, photons + 1., 1e-5)
        self.assertFalse(screen.can_skip(1))

    def test_two_sided_aggregate_roundoff_reserve_rejects_boundary(self):
        aggregate = self.hydrogen(photons=1.-1e-5-1e-10)
        # Perturbations much smaller than a physical budget cannot justify
        # skipping a radius whose candidate decision can change sign.
        reserve = np.full(6, 2e-10)
        self.assertFalse(self.certificate(aggregate, reserve=reserve).can_skip(1))

    def test_screen_retains_only_three_margins_per_radius(self):
        screen = RadiusScreen(37)
        self.assertEqual(screen.margins.shape, (37, 3))
        self.assertEqual(screen.margins.nbytes, 37*3*8)


class ScreenedSelection(unittest.TestCase):
    def model(self, enabled, sources=None):
        model = SpectralES(side=3, radii2=[0., 1., 2., 3.],
                           radius_screening=enabled, donor_fraction=.2)
        if sources is not None:
            model.inject(sources)
        return model

    def compare(self, left, right):
        np.testing.assert_array_equal(left.gas, right.gas)
        np.testing.assert_array_equal(left.photons, right.photons)
        self.assertEqual(left.events, right.events)
        self.assertEqual(left.audit(), right.audit())

    def test_full_illumination_preserves_hydrogen_and_mixed_helium_events(self):
        for sources in ([((0, 0, 0), [2.2, 0., 0.]), ((2, 2, 2), [.3, 0., 0.])],
                        [((0, 0, 0), [.3, .4, .2]), ((1, 1, 1), [.2, .1, .3])]):
            with self.subTest(sources=sources):
                old = self.model(False, sources)
                new = self.model(True, sources)
                old.illuminate(); new.illuminate()
                self.compare(old, new)
                self.assertIsNone(new._radius_screen)

    def test_partial_retry_after_skips_keeps_smallest_remaining_radius(self):
        source = [((0, 0, 0), [.1, 0., 0.])]
        old = self.model(False, source)
        new = self.model(True, source)
        new._radius_screen = RadiusScreen(len(new.radii2))
        excluded = set()
        # A failed/no-progress commit leaves the state unchanged. Every next
        # selection must find the same next partial radius as a fresh scan.
        for expected in range(len(old.radii2)):
            old._partial_retry_cache = None
            new._partial_retry_cache = None
            kind_a, a = old._select(excluded)
            kind_b, b = new._select(excluded)
            self.assertEqual((kind_a, kind_b), ('partial', 'partial'))
            self.assertEqual(a.radius_index, expected)
            self.assertEqual(a.radius_index, b.radius_index)
            np.testing.assert_array_equal(a.centres, b.centres)
            excluded.add(a.radius_index)
        self.assertGreater(new.counters['radius_scans_skipped'], 0)
        new._partial_retry_cache = None
        self.assertEqual(new._select(excluded), (None, []))

    def test_no_absorbable_support_revalidates_skipped_radii(self):
        new = self.model(True)
        new._radius_screen = RadiusScreen(len(new.radii2))
        self.assertEqual(new._select(), (None, []))
        self.assertEqual(new._select(), (None, []))
        self.assertGreater(new.counters['radius_scans_skipped'], 0)

    def test_each_illumination_resets_after_new_source_injection(self):
        old = self.model(False, [((0, 0, 0), [.1, 0., 0.])])
        new = self.model(True, [((0, 0, 0), [.1, 0., 0.])])
        old.illuminate(); new.illuminate()
        self.assertIsNone(new._radius_screen)
        source = [((0, 0, 0), [1.7, 0., 0.])]
        old.inject(source); new.inject(source)
        old.illuminate(); new.illuminate()
        self.compare(old, new)
        self.assertIsNone(new._radius_screen)

    def test_exception_clears_certificate(self):
        new = self.model(True, [((0, 0, 0), [.1, 0., 0.])])
        with mock.patch.object(new, '_select', side_effect=RuntimeError('stop')):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                new.illuminate()
        self.assertIsNone(new._radius_screen)


if __name__ == '__main__':
    unittest.main()
