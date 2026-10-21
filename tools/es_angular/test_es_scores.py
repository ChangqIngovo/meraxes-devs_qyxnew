"""Independent periodic direct sums and limiting cases for the ES prior."""

from itertools import product
import unittest
import numpy as np

from es_scores import default_radii, periodic_es_scores


def direct_score(photons, hydrogen, radii):
    """Independent reference over physical cells, with minimum-image distances."""
    shape = photons.shape
    result = np.zeros(shape)
    maximum = np.finfo(float).max
    points = list(product(*(range(n) for n in shape)))
    for centre in points:
        for radius in radii:
            p = np.longdouble(0)
            h = np.longdouble(0)
            for target in points:
                distance2 = sum(min(abs(a-b), size-abs(a-b))**2
                                for a, b, size in zip(centre, target, shape))
                if distance2 <= radius*radius:
                    p += np.longdouble(photons[target])
                    h += np.longdouble(hydrogen[target])
            ratio = min(p/h, maximum) if h else (maximum if p else 0.)
            result[centre] = max(result[centre], ratio)
    return result


class PeriodicESScoreTests(unittest.TestCase):
    def test_default_scales_include_local_and_half_box(self):
        self.assertEqual(default_radii((10, 12, 14)), (0., 1., 2., 4., 5.))
        self.assertEqual(default_radii((3, 3, 3)), (0., 1., 1.5))
        self.assertEqual(default_radii((1, 1, 1)), (0., .5))

    def test_uniform_fields_and_discrete_kernel_count(self):
        photons = np.full((4, 4, 4), 6.)
        hydrogen = np.full_like(photons, 3.)
        score, audit = periodic_es_scores(photons, hydrogen, [0, 1],
                                           return_diagnostics=True)
        np.testing.assert_array_equal(score, 2.)
        self.assertEqual([x['kernel_cells'] for x in audit['levels']], [1, 7])
        isolated = np.zeros_like(photons)
        isolated[0, 0, 0] = 1.
        result = periodic_es_scores(isolated, np.ones_like(isolated), [1])
        self.assertAlmostEqual(result[0, 0, 0], 1/7, 14)
        self.assertAlmostEqual(result[-1, 0, 0], 1/7, 14)
        self.assertEqual(result[2, 2, 2], 0.)

    def test_periodic_translation_equivariance(self):
        rng = np.random.default_rng(173)
        photons = rng.random((5, 6, 4))
        hydrogen = .1+rng.random(photons.shape)
        shift = (2, -1, 3)
        a = periodic_es_scores(photons, hydrogen)
        b = periodic_es_scores(np.roll(photons, shift, (0, 1, 2)),
                               np.roll(hydrogen, shift, (0, 1, 2)))
        np.testing.assert_allclose(b, np.roll(a, shift, (0, 1, 2)),
                                   rtol=2e-14, atol=1e-15)

    def test_nonuniform_against_independent_direct_sum(self):
        rng = np.random.default_rng(826)
        photons = rng.random((4, 5, 4))
        hydrogen = rng.random(photons.shape)
        photons[photons < .7] = 0.
        hydrogen[hydrogen < .6] = 0.
        radii = [0., 1., 1.5, 2.]
        expected = direct_score(photons, hydrogen, radii)
        actual = periodic_es_scores(photons, hydrogen, radii)
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)

    def test_zero_density_and_empty_support(self):
        photons = np.zeros((6, 6, 6))
        photons[0, 0, 0] = 2.
        score = periodic_es_scores(photons, np.zeros_like(photons), [1.])
        self.assertEqual(score[0, 0, 0], np.finfo(float).max)
        self.assertEqual(score[-1, 0, 0], np.finfo(float).max)
        self.assertEqual(score[3, 3, 3], 0.)
        self.assertTrue(np.all(np.isfinite(score)))
        np.testing.assert_array_equal(periodic_es_scores(photons*0, photons*0), 0.)

    def test_remote_large_source_does_not_erase_tiny_positive(self):
        photons = np.zeros((8, 8, 8))
        hydrogen = np.zeros_like(photons)
        photons[0, 0, 0] = 1e200
        hydrogen[0, 0, 0] = 2e200
        photons[4, 4, 4] = 1e-200
        hydrogen[4, 4, 4] = 2e-200
        score, audit = periodic_es_scores(photons, hydrogen, [1.],
                                           return_diagnostics=True)
        self.assertEqual(score[4, 4, 4], .5)
        self.assertEqual(score[3, 4, 4], .5)
        self.assertEqual(score[2, 2, 2], 0.)
        self.assertGreater(audit['levels'][0]['photons']['direct_cells'], 0)

    def test_large_finite_counts_and_saturated_ratios(self):
        maximum = np.finfo(float).max
        photons = np.full((4, 4, 4), maximum)
        hydrogen = photons.copy()
        np.testing.assert_allclose(periodic_es_scores(photons, hydrogen), 1.)
        np.testing.assert_array_equal(periodic_es_scores(photons, hydrogen*0+1e-200),
                                      maximum)

    def test_inputs_preserved_and_invalid_inputs_rejected(self):
        p = np.ones((3, 3, 3))
        h = p.copy()
        periodic_es_scores(p, h)
        np.testing.assert_array_equal(p, 1.)
        np.testing.assert_array_equal(h, 1.)
        for bad in ([1.], np.full((3, 3, 3), -1.), np.full((3, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                periodic_es_scores(bad, h)
        for radii in ([], [-1], [2], [np.inf]):
            with self.assertRaises(ValueError):
                periodic_es_scores(p, h, radii)


if __name__ == '__main__':
    unittest.main()
