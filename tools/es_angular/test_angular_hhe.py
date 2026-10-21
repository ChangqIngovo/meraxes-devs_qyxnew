"""Independent ledger and shared-species checks for frozen-front chemistry."""
import unittest
import numpy as np
from angular_hhe import (band_capacities, illuminate_channels,
                         illuminate_frozen_front, split_front_credits, _reference)


class FrozenFrontChemistryTests(unittest.TestCase):
    def audit(self, initial, credits, result, tolerance=3e-11):
        np.testing.assert_allclose(result['resource_debit'] + result['resource_retained'],
                                   credits, rtol=tolerance, atol=0.)
        np.testing.assert_allclose(result['channel_debit'] + result['target_credits'],
                                   result['target_credits_initial'], rtol=tolerance, atol=0.)
        self.assertTrue(np.all(result['gas'] >= 0))
        self.assertTrue(np.all(result['target_credits'] >= 0))
        self.assertTrue(np.all(result['boundary_credits'] >= 0))
        np.testing.assert_allclose(result['gas'][:, :2].sum(1), initial[:, :2].sum(1),
                                   rtol=tolerance, atol=0.)
        np.testing.assert_allclose(result['gas'][:, 2:].sum(1), initial[:, 2:].sum(1),
                                   rtol=tolerance, atol=0.)
        np.testing.assert_allclose(result['resource_debit'].sum(), result['absorbed'].sum(),
                                   rtol=tolerance, atol=0.)

    def test_boundary_share_never_spent(self):
        gas=np.array([[10.,0,0,0,0]])
        result=illuminate_frozen_front(gas,[1.],[[1.,0,0]],[[.5]],stop_at_event=False)
        self.audit(gas,[1.],result)
        self.assertEqual(result['boundary_credits'][0],.5)
        self.assertAlmostEqual(result['resource_debit'][0],.5,12)
        self.assertGreaterEqual(result['resource_retained'][0],.5)

    def test_target_share_not_recycled_into_another_target(self):
        gas=np.array([[.1,0,0,0,0],[10.,0,0,0,0]])
        result=illuminate_frozen_front(gas,[1.],[[1.,0,0]],[[.5,.5]],stop_at_event=False)
        self.audit(gas,[1.],result)
        self.assertAlmostEqual(result['channel_debit'][0,0],.1,11)
        self.assertAlmostEqual(result['channel_debit'][0,1],.5,11)
        self.assertAlmostEqual(result['target_credits'][0,0],.4,11)

    def test_many_resources_share_one_physical_cell(self):
        gas=np.array([[.4,0,0,0,0]])
        q=np.array([.3,.3,.3])
        result=illuminate_frozen_front(gas,q,[[1,0,0]]*3,[[1.]]*3,
                                      epsilon=1e-6)
        self.audit(gas,q,result)
        self.assertEqual(result['stop'],'transparency_event')
        self.assertLessEqual(result['resource_debit'].sum(),.4)
        np.testing.assert_allclose(result['resource_debit'],np.repeat(.4*(1-1e-6)/3,3),rtol=2e-12)

    def test_hard_band_can_ionize_helium_twice(self):
        gas=np.array([[0.,0.,1.,0.,0.]])
        result=illuminate_frozen_front(gas,[3.],[[1.,2.,3.]],[[1.]],epsilon=1e-6)
        self.audit(gas,[3.],result)
        self.assertEqual(result['events'],[[0,2]])
        self.assertAlmostEqual(result['resource_debit'][0],2*(1-1e-6),11)
        self.assertGreater(result['gas'][0,4],.99999)
        self.assertGreater(result['gas'][0,2]+result['gas'][0,3],0.)

    def test_soft_hard_compete_for_same_hydrogen(self):
        gas=np.array([[1.,0.,.2,0.,0.]])
        q=np.array([.7,.7])
        result=illuminate_frozen_front(gas,q,[[1,0,0],[.1,.7,.4]],[[1.],[1.]],
                                      epsilon=1e-6)
        self.audit(gas,q,result)
        self.assertLessEqual(result['absorbed'][0,0],1.)
        self.assertGreater(result['channel_debit'][0,0],0.)
        self.assertGreater(result['channel_debit'][1,0],0.)
        self.assertLessEqual(result['absorbed'].sum(),1.4)

    def test_intermediate_band_cannot_make_heiii(self):
        gas=np.array([[0.,0.,1.,0.,0.]])
        result=illuminate_frozen_front(gas,[2.],[[1.,2.,0.]],[[1.]],epsilon=1e-6)
        self.audit(gas,[2.],result)
        self.assertEqual(result['events'],[[0,1]])
        self.assertEqual(result['gas'][0,4],0.)
        self.assertAlmostEqual(result['gas'][0,3],1-1e-6,11)

    def test_first_global_event_stops_other_cells(self):
        gas=np.array([[.1,0.,0.,0.,0.],[100.,0.,0.,0.,0.]])
        result=illuminate_frozen_front(gas,[2.],[[1.,0,0]],[[.5,.5]],epsilon=1e-6)
        self.audit(gas,[2.],result)
        self.assertEqual(result['events'],[[0,0]])
        self.assertLess(result['channel_debit'][0,1],1.)
        self.assertGreater(result['target_credits'][0,1],0.)

    def test_equal_opacity_accounts_remain_separate(self):
        gas=np.array([[5.,0.,.5,0.,0.]])
        q=np.array([.1,.3,.6])
        result=illuminate_frozen_front(gas,q,[[1.,.5,.2]]*3,[[1.]]*3,stop_at_event=False)
        self.audit(gas,q,result)
        np.testing.assert_allclose(result['resource_debit']/q,np.ones(3),rtol=2e-12)
        self.assertEqual(result['channel_debit'].shape,(3,1))

    def test_tiny_photon_counts_are_retained_or_spent(self):
        gas=np.array([[1e-200,0.,0.,0.,0.]])
        q=np.array([.1e-200])
        result=illuminate_frozen_front(gas,q,[[1.,0,0]],[[.75]],stop_at_event=False)
        self.audit(gas,q,result)
        np.testing.assert_allclose(result['resource_debit'],q*.75,rtol=3e-12)
        np.testing.assert_allclose(result['boundary_credits'],q*.25,rtol=3e-15)

    def test_floor_retains_dormant_accounts(self):
        gas=np.array([[1.,0.,0.,0.,0.]])
        q=np.array([1e-31])
        result=illuminate_frozen_front(gas,q,[[1.,0,0]],[[1.]],photon_floor=1e-30)
        self.audit(gas,q,result)
        self.assertEqual(result['resource_retained'][0],q[0])
        self.assertEqual(result['resource_debit'][0],0.)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            split_front_credits([1.],[[.6,.6]])
        with self.assertRaises(ValueError):
            illuminate_channels([[1.,0,0,0,0]],[[1.]],[[0.,1,0]])

    def test_roundoff_allocation_correction_is_reported(self):
        gas=np.array([[1.,0.,0.,0.,0.],[1.,0.,0.,0.,0.]])
        result=illuminate_frozen_front(gas,[.2],[[1.,0,0]],[[.2,.8]],stop_at_event=False)
        self.audit(gas,[.2],result)
        self.assertLessEqual(result['split_roundoff_l1'][0],32*np.finfo(float).eps*.2)
        self.assertLessEqual(result['target_credits_initial'].sum(),.2)

    def test_physical_cross_section_thresholds(self):
        energy=np.array([13.6,24.59-1e-8,24.59,54.4-1e-8,54.4])
        cross=_reference().cross_sections(energy)
        np.testing.assert_array_equal((cross>0).sum(axis=1)-1,[0,0,1,1,2])

    def test_band_capacities_monotone_during_photoionization(self):
        gas=np.array([[1.,0.,.2,0.,0.]])
        result=illuminate_frozen_front(gas,[.5,.3,.2],[[1.,0,0],[.5,1.,0],[.2,.4,.6]],
                                      [[1.]]*3,stop_at_event=False)
        self.audit(gas,[.5,.3,.2],result)
        self.assertTrue(np.all(band_capacities(result['gas']) <= band_capacities(gas)))

    def test_local_map_equal_rate_helium_solution(self):
        gas=np.array([[1.,0.,.2,0.,0.]])
        rate=np.array([[.3,.7,.7]])
        state,counts=_reference().upward_cells(gas,rate,2.)
        h=np.exp(-.6); he=np.exp(-1.4)
        expected=np.array([[h,1-h,.2*he,.2*1.4*he,.2*(1-2.4*he)]])
        np.testing.assert_allclose(state,expected,rtol=3e-15,atol=1e-16)
        np.testing.assert_allclose(counts,[[1-h,.2*(1-he),expected[0,4]]],
                                   rtol=3e-15,atol=1e-16)

    def test_frozen_map_semigroup_and_capture_additivity(self):
        rng=np.random.default_rng(71021)
        gas=rng.uniform(.01,2.,(24,5)); rates=rng.uniform(0.,3.,(24,3))
        rates[::3,2]=rates[::3,1]
        rates[::4,1]=0.; rates[::5,2]=0.
        direct,captures=_reference().upward_cells(gas,rates,.9)
        first,first_counts=_reference().upward_cells(gas,rates,.2)
        split,second_counts=_reference().upward_cells(first,rates,.7)
        np.testing.assert_allclose(split,direct,rtol=5e-15,atol=1e-15)
        np.testing.assert_allclose(first_counts+second_counts,captures,
                                   rtol=5e-15,atol=1e-15)

    def test_scalar_and_vector_maps_agree_at_limiting_rates(self):
        rates=np.array([[0.,0.,0.],[1.,1.,1.],[.3,.7,.7*(1+1e-12)],
                        [1.,1.,0.],[1.,0.,1.],[1e-20,1e-20,2.],
                        [1.,2.,1e-20],[1.,1e3,1e3]])
        gas=np.broadcast_to(np.array([1.,.2,.3,.1,.05]),(len(rates),5)).copy()
        state,counts=_reference().upward_cells(gas,rates,1.)
        for j,rate in enumerate(rates):
            mapping,response=_reference().upward_map(rate,1.)
            np.testing.assert_allclose(state[j],mapping@gas[j],rtol=3e-14,atol=1e-16)
            np.testing.assert_allclose(counts[j],response@gas[j],rtol=3e-14,atol=1e-16)


if __name__=='__main__':
    unittest.main(verbosity=2)
