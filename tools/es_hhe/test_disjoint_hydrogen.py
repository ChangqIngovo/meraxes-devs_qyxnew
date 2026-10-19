#!/usr/bin/env python3
"""Independent H endpoints retain photon credits and the ES crossing rule."""
import unittest
from unittest import mock

import numpy as np

import es_hhe_reference as reference


class DisjointHydrogenEndpoint(unittest.TestCase):
    def operator(self, size):
        return reference._CompactSphereOperator(np.arange(size), np.arange(size+1))

    def endpoint(self, gas, photons, energies, *, epsilon=1e-5, floor=1e-30,
                 residual=None, operator=None):
        gas=np.asarray(gas,dtype=float)
        photons=np.asarray(photons,dtype=float)
        cross=reference.cross_sections(np.asarray(energies,dtype=float))
        if cross.size and np.any(cross):
            cross/=cross.max()
        if residual is None:
            residual=np.full_like(photons,floor)
        if operator is None:
            operator=self.operator(len(gas))
        before_gas,before_photons=gas.copy(),photons.copy()
        result=reference._disjoint_hydrogen_endpoint(
            gas,photons,cross,operator,epsilon,residual)
        np.testing.assert_array_equal(gas,before_gas)
        np.testing.assert_array_equal(photons,before_photons)
        return result

    def assert_conserves(self, gas, photons, result):
        self.assertIsNotNone(result)
        n,p,c,d=(result[key] for key in ('species','photons','absorbed','debit'))
        for value in (n,p,c,d):
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertTrue(np.all(value>=0))
        self.assertTrue(np.all(d<=photons))
        np.testing.assert_allclose(n[:,:2].sum(1),gas[:,:2].sum(1),rtol=8e-15)
        np.testing.assert_array_equal(n[:,2:],gas[:,2:])
        np.testing.assert_allclose(p+d,photons,rtol=8e-15,atol=0.)
        np.testing.assert_allclose(d.sum(1),c.sum(1),rtol=8e-15,atol=0.)
        np.testing.assert_allclose(n[:,1]-gas[:,1],c[:,0],rtol=2e-9,atol=8e-16)
        np.testing.assert_array_equal(c[:,1:],0.)

    def test_photon_and_atom_limited_cells_close_independently(self):
        gas=np.array([[1.,0.,.08,0.,0.],[1e-6,1.,.08,0.,0.]])
        photons=np.array([[.3],[.5]])
        result=self.endpoint(gas,photons,[18.])
        self.assert_conserves(gas,photons,result)
        np.testing.assert_array_equal(result['species'][:,0],[.7,0.])
        np.testing.assert_array_equal(result['photons'][:,0],[0.,.5-1e-6])
        self.assertEqual(result['stop'],'absorber_saturation')

    def test_duplicate_bands_preserve_partition_and_dormant_accounts(self):
        gas=np.array([[1.,0.,.08,0.,0.],[1e-6,1.,.08,0.,0.]])
        photons=np.array([[.2,.3,7e-31],[.2,.3,3e-31]])
        result=self.endpoint(gas,photons,[18.,18.,18.])
        self.assert_conserves(gas,photons,result)
        np.testing.assert_array_equal(result['photons'][:,2],photons[:,2])
        np.testing.assert_allclose(result['debit'][1,:2],[.4e-6,.6e-6],rtol=8e-15)

    def test_supplied_per_account_relative_residual_is_preserved(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.array([[.2,1e-14],[.3,2e-14]])
        residual=np.array([[1e-13,1e-13],[1e-13,1e-13]])
        result=self.endpoint(gas,photons,[18.,60.],residual=residual)
        self.assert_conserves(gas,photons,result)
        np.testing.assert_array_equal(result['photons'][:,1],photons[:,1])
        np.testing.assert_array_equal(result['debit'][:,1],0.)

    def test_active_helium_opacity_uses_general_solver(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.array([[.2,.01],[.3,0.]])
        self.assertIsNone(self.endpoint(gas,photons,[18.,60.]))

    def test_distinct_atom_limited_hydrogen_opacities_use_general_solver(self):
        gas=np.tile([1e-6,1.,.08,0.,0.],(2,1))
        photons=np.array([[.2,.01],[.3,0.]])
        self.assertIsNone(self.endpoint(gas,photons,[18.,13.6]))

    def test_unequal_photon_limited_opacities_consume_all_active_accounts(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.array([[.2,.01],[.3,1e-16]])
        result=self.endpoint(gas,photons,[18.,13.6])
        self.assert_conserves(gas,photons,result)
        np.testing.assert_array_equal(result['photons'],0.)
        np.testing.assert_array_equal(result['debit'],photons)

    def test_extreme_atom_to_photon_ratio_keeps_representable_capture(self):
        gas=np.array([[1e-320,1.,.08,0.,0.],[1.,0.,.08,0.,0.]])
        photons=np.array([[1e300],[.2]])
        result=self.endpoint(gas,photons,[18.],floor=0.)
        self.assert_conserves(gas,photons,result)
        self.assertEqual(result['absorbed'][0,0],1e-320)
        self.assertEqual(result['debit'][0,0],1e-320)
        self.assertEqual(result['species'][0,0],0.)
        self.assertEqual(result['photons'][0,0],1e300)

    def test_underflowing_helium_hazard_still_uses_general_solver(self):
        gas=np.array([[0.,1.,0.,np.nextafter(0.,1.),.08],
                      [1.,0.,.08,0.,0.]])
        photons=np.array([[1.,0.],[0.,.2]])
        self.assertIsNone(self.endpoint(gas,photons,[1e9,18.]))

    def test_different_cells_can_use_different_single_opacities(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.array([[.2,0.],[0.,.3]])
        result=self.endpoint(gas,photons,[18.,13.6])
        self.assert_conserves(gas,photons,result)
        np.testing.assert_array_equal(result['photons'],0.)

    def test_crossing_and_roundoff_boundary_use_general_solver(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        for available in (.9,.9-1e-15,1.1):
            with self.subTest(available=available):
                self.assertIsNone(self.endpoint(gas,[[available],[.1]],[18.],epsilon=.1))
        self.assertIsNotNone(self.endpoint(gas,[[.9-1e-10],[.1]],[18.],epsilon=.1))

    def test_overlapping_or_multicell_supports_use_general_solver(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.full((2,1),.1)
        for operator in (
                reference._CompactSphereOperator(np.array([0,0,1]),np.array([0,1,2,3])),
                reference._CompactSphereOperator(np.array([0,1]),np.array([0,2]))):
            self.assertIsNone(self.endpoint(gas,photons,[18.],operator=operator))

    def test_no_active_accounts_remain_unchanged(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.full((2,1),7e-31)
        self.assertIsNone(self.endpoint(gas,photons,[18.]))

    def test_integrated_endpoint_matches_legacy_within_retained_tail_tolerance(self):
        gas=np.array([[1.,0.,.08,0.,0.],[1e-6,1.,.08,0.,0.],
                      [2.,.1,.08,0.,0.]])
        photons=np.array([[.2,.3],[.2,.3],[.1,.2]])
        options=dict(epsilon=1e-5,donor_fraction=.2,count_tolerance=1e-13,
                     photon_floor=1e-30)
        operator=self.operator(len(gas))
        with mock.patch.object(reference,'get_backend',return_value=None):
            actual=reference.joint_exposure_update(gas,photons,[18.,18.],operator,**options)
            with mock.patch.object(reference,'_disjoint_hydrogen_endpoint',return_value=None):
                legacy=reference.joint_exposure_update(gas,photons,[18.,18.],operator,**options)
        self.assert_conserves(gas,photons,actual)
        self.assertEqual(actual['exposure_kind'],'independent_hydrogen_endpoint')
        self.assertIsNone(actual['exposure'])
        self.assertEqual(actual['events'],[])
        for name in ('species','photons','absorbed','debit'):
            np.testing.assert_allclose(actual[name],legacy[name],rtol=4e-12,atol=8e-13)
        self.assertLess(actual['iterations'],legacy['iterations'])

    def test_integrated_crossing_keeps_earliest_common_event(self):
        gas=np.tile([1.,0.,.08,0.,0.],(2,1))
        photons=np.array([[2.],[.2]])
        operator=self.operator(len(gas))
        with mock.patch.object(reference,'get_backend',return_value=None):
            result=reference.joint_exposure_update(gas,photons,[18.],operator,
                epsilon=.1,donor_fraction=.2,photon_floor=1e-30)
        self.assertEqual(result['stop'],'crossing')
        self.assertNotEqual(result['exposure_kind'],'independent_hydrogen_endpoint')
        self.assertEqual(result['events'],[[0,0]])
        self.assertGreater(result['photons'][1,0],0.)


if __name__=='__main__':
    unittest.main()
