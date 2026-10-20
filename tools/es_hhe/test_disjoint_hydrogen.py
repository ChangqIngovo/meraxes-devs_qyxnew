#!/usr/bin/env python3
"""Independent H endpoints retain photon credits and the ES crossing rule."""
import math
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

    def test_distinct_atom_limited_hydrogen_opacities_close_accounts(self):
        gas=np.tile([1e-6,1.,.08,0.,0.],(2,1))
        photons=np.array([[.2,.01],[.3,0.]])
        result=self.endpoint(gas,photons,[18.,13.6])
        self.assert_conserves(gas,photons,result)
        np.testing.assert_array_equal(result['species'][:,0],0.)
        cross=reference.cross_sections(np.array([18.,13.6]))[:,0]
        exposure=-np.log1p(-result['debit'][0]/photons[0])/cross
        np.testing.assert_allclose(exposure,exposure[0],rtol=8e-15)

    def test_snapshot65_mixed_hydrogen_tail_regression(self):
        # Two actual independent cells from snapshot 65's failing 1104-cell
        # batch: an active 13.6 eV recombination tail prevents equal-opacity
        # dispatch despite the dominant 18 eV primary account.
        gas=np.array([
            [2.0849700704491452e-5,2.281709206108317,.1801365833533438,0.,0.],
            [2.1517001390457134e-5,2.1516786220443227,.16987106360887225,0.,0.]])
        photons=np.array([
            [.0026644121651093135,7.273597238049652e-16,7e-31],
            [.05712788891626835,9.265885417669346e-16,3e-31]])
        result=reference.joint_exposure_update(gas,photons,[18.,13.6,60.],
            self.operator(2),epsilon=1e-5,photon_floor=1e-30)
        self.assert_conserves(gas,photons,result)
        self.assertEqual(result['exposure_kind'],'independent_hydrogen_endpoint')
        self.assertEqual(result['substeps'],1)
        np.testing.assert_array_equal(result['species'][:,0],0.)
        np.testing.assert_array_equal(result['photons'][:,2],photons[:,2])
        self.assertTrue(np.all(result['debit'][:,1]>0))
        self.assertTrue(np.all(result['debit'][:,1]<photons[:,1]))
        self.assertEqual(result['events'],[])

    def test_unequal_scalar_endpoint_matches_quadratic_solution(self):
        light=np.array([.3,.2]);opacity=np.array([1.,.5]);amount=.1
        debit=reference._hydrogen_account_debits(light,opacity,amount)
        root=(-.2+math.sqrt(.04+1.2*(.5-amount)))/.6
        expected=light*np.array([1-root*root,1-root])
        np.testing.assert_allclose(debit,expected,rtol=8e-15,atol=0.)
        self.assertEqual(math.fsum(debit),amount)

    def test_unequal_scalar_endpoint_preserves_duplicate_band_partition(self):
        light=np.array([.2,.1,.3]);opacity=np.array([1.,1.,.5]);amount=.1
        debit=reference._hydrogen_account_debits(light,opacity,amount)
        self.assertAlmostEqual(debit[0]/debit[1],2.,places=14)
        permutation=np.array([2,0,1])
        permuted=reference._hydrogen_account_debits(
            light[permutation],opacity[permutation],amount)
        np.testing.assert_allclose(permuted,debit[permutation],rtol=8e-15)
        self.assertEqual(math.fsum(debit),amount)

    def test_unequal_scalar_endpoint_retains_extreme_opacity_channel(self):
        debit=reference._hydrogen_account_debits([1.,1.],[1.,1e-300],1e-6)
        self.assertIsNotNone(debit)
        self.assertGreater(debit[1],0.)
        self.assertEqual(math.fsum(debit),1e-6)
        np.testing.assert_allclose(debit[1],-math.log1p(-debit[0])*1e-300,
                                   rtol=8e-15,atol=0.)

    def test_unequal_scalar_endpoint_preserves_subnormal_capture(self):
        amount=1e-320
        debit=reference._hydrogen_account_debits(
            [1e-310,2e-310],[1.,.5],amount)
        self.assertIsNotNone(debit)
        self.assertTrue(np.all(debit>0))
        self.assertEqual(math.fsum(debit),amount)

    def test_unequal_unrepresentable_exposure_uses_general_solver(self):
        # HI/rate underflows, even though the debit itself is representable.
        self.assertIsNone(reference._hydrogen_account_debits(
            [1e300,1e300],[1.,.5],1e-320))
        gas=np.array([[1e-320,1.,.08,0.,0.],[1.,0.,.08,0.,0.]])
        photons=np.array([[1e300,1e300],[.2,0.]])
        self.assertIsNone(self.endpoint(gas,photons,[18.,13.6],floor=0.))

    def test_unequal_unresolved_optical_depth_uses_general_solver(self):
        # The weak optical depth is ~1e-600, below binary64, but its
        # capture (~1e-300) is representable because the account is large.
        # Returning a zero debit would silently lose this spectral channel.
        self.assertIsNone(reference._hydrogen_account_debits(
            [1e300,1e300],[1.,1e-300],1.))

    def test_unequal_scalar_endpoint_rejects_unresolved_opacity_or_count_scale(self):
        self.assertIsNone(reference._hydrogen_account_debits(
            [1.,1.],[1e300,1e-300],.1))
        self.assertIsNone(reference._hydrogen_account_debits(
            [1e308,1e308],[1.,.5],.1))

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
