#!/usr/bin/env python3
"""Sparse exact geometry: tails, periodic supports, adjoints and bounded storage."""
import math
import unittest

import numpy as np

from es_hhe_reference import PeriodicSphereGeometry, SpectralES, geometric_radii2, joint_exposure_update


def accurate_sum(values):
    values=np.asarray(values)
    if values.ndim==1:return math.fsum(values.tolist())
    return np.array([math.fsum(column.tolist()) for column in values.T])


class SparseGeometry(unittest.TestCase):
    def test_sparse_fallback_retains_subnormal_remote_signal(self):
        geometry=PeriodicSphereGeometry(9,geometric_radii2(9,1.2),direct_limit=0)
        field=np.zeros(geometry.size)
        field[0]=1.
        remote=(4*9+4)*9+4
        field[remote]=np.nextafter(0.,1.)*67
        field[remote+1]=1.e-200
        prepared=geometry.prepare(field)
        result=geometry.all_sums(prepared,1)
        expected=np.array([accurate_sum(field[geometry.cells(c,1)])
                           for c in range(geometry.size)])
        np.testing.assert_allclose(result,expected,rtol=1.e-12,atol=0.)
        isolated=(4*9+4)*9+3
        self.assertEqual(result[isolated],field[remote])
        self.assertGreater(geometry.counters['dynamic_range_fallbacks'],0)
        self.assertEqual(len(prepared.nonzero_indices[0]),3)

    def test_sparse_membership_matches_direct_signed_and_subnormal_sums(self):
        rng=np.random.default_rng(11)
        geometry=PeriodicSphereGeometry(9,geometric_radii2(9,1.2),direct_limit=0)
        field=np.zeros((geometry.size,3))
        indices=rng.choice(geometry.size,29,replace=False)
        field[indices,0]=10.**rng.uniform(-310,4,len(indices))
        field[indices,1]=rng.normal(size=len(indices))*10.**rng.uniform(-300,5,len(indices))
        field[indices,2]=np.nextafter(0.,1.)*rng.integers(1,20,len(indices))
        rows=np.arange(geometry.size)
        for k in range(len(geometry.radii2)):
            expected=np.array([accurate_sum(field[geometry.cells(c,k)]) for c in rows])
            actual=np.column_stack([geometry.sparse_sums(field[:,j],
                np.flatnonzero(field[:,j]!=0),rows,k) for j in range(3)])
            np.testing.assert_array_equal(actual,expected)

    def test_compact_periodic_overlaps_preserve_aggregate_scatter_and_adjoint(self):
        geometry=PeriodicSphereGeometry(16,np.array([0.,1.,2.,4.]),direct_limit=0)
        operator=geometry.operator([0,1,15,16],[1,2,0,3])
        cells,compact=operator.compact()
        rng=np.random.default_rng(7)
        field=rng.normal(size=(geometry.size,3))
        weights=rng.normal(size=(len(operator),3))
        supports=[geometry.cells(c,k) for c,k in zip(operator.centres,operator.radius_indices)]
        expected_aggregate=np.array([accurate_sum(field[s]) for s in supports])
        expected_scatter=np.array([accurate_sum(weights[[c in s for s in supports]])
                                   for c in range(geometry.size)])
        np.testing.assert_array_equal(operator.aggregate(field),expected_aggregate)
        np.testing.assert_array_equal(operator.scatter(weights),expected_scatter)
        np.testing.assert_array_equal(compact.aggregate(field[cells]),expected_aggregate)
        np.testing.assert_array_equal(compact.scatter(weights),expected_scatter[cells])
        np.testing.assert_array_equal(compact.volumes,operator.volumes)
        np.testing.assert_array_equal(compact.multiplicity,operator.multiplicity[cells])
        lhs=math.fsum((operator.aggregate(field)*weights).ravel().tolist())
        rhs=math.fsum((field*operator.scatter(weights)).ravel().tolist())
        self.assertLessEqual(abs(lhs-rhs),1.e-12*max(abs(lhs),abs(rhs),1.))
        # Weak terms must survive exactly where they do not share a bright term.
        tiny_weights=np.array([1.,1.e-200,np.nextafter(0.,1.)*67,0.])
        expected=np.array([accurate_sum(tiny_weights[[c in s for s in supports]])
                           for c in range(geometry.size)])
        np.testing.assert_array_equal(operator.scatter(tiny_weights),expected)

    def test_radius_zero_compaction_is_exact_and_keeps_uncovered_cells_zero(self):
        geometry=PeriodicSphereGeometry(16,[0.,1.],direct_limit=0)
        operator=geometry.operator([0,17,255],[0,0,0])
        cells,compact=operator.compact()
        np.testing.assert_array_equal(cells,[0,17,255])
        values=np.array([[1.,1.e-300],[2.,3.],[np.nextafter(0.,1.),4.]])
        np.testing.assert_array_equal(compact.aggregate(values),values)
        np.testing.assert_array_equal(compact.scatter(values),values)
        full=operator.scatter(values)
        np.testing.assert_array_equal(full[cells],values)
        self.assertEqual(np.count_nonzero(full[np.setdiff1d(np.arange(geometry.size),cells)]),0)

    def test_large_supports_keep_implicit_storage(self):
        geometry=PeriodicSphereGeometry(16,[0.,1.,192.],direct_limit=0)
        full=geometry.operator([0],[2])
        self.assertIsNone(full.compact())
        self.assertIsNone(full._compact_operator)
        crowded=geometry.operator(np.arange(geometry.size),np.ones(geometry.size,dtype=int))
        self.assertGreater(crowded.volumes.sum(),geometry.size)
        self.assertIsNone(crowded.compact())
        self.assertIsNone(crowded._compact_operator)
        np.testing.assert_array_equal(crowded.multiplicity,np.full(geometry.size,7.))
        # Even within the allocation bound, broad overlapping unions use FFTs
        # to avoid a Python summation cost proportional to most grid cells.
        broad=geometry.operator(np.arange(512),np.ones(512,dtype=int))
        self.assertLess(broad.volumes.sum(),geometry.size)
        self.assertIsNone(broad.compact())
        self.assertIsNone(broad._compact_operator)

    def test_compact_joint_update_matches_dense_and_validates_uncovered_cells(self):
        geometry=PeriodicSphereGeometry(8,[0.,1.],direct_limit=0)
        operator=geometry.operator([0,1],[1,1])
        cells,_=operator.compact()
        membership=np.zeros(operator.shape,dtype=bool)
        for row,centre in enumerate(operator.centres):
            membership[row,geometry.cells(centre,1)]=True
        uncovered=np.flatnonzero(operator.multiplicity==0)
        gas=np.zeros((geometry.size,5))
        gas[:,0]=1.;gas[:,2]=.08
        gas[uncovered,0]=.7;gas[uncovered,1]=.3
        for amount in [.4,25.]:
            with self.subTest(source_photons=amount):
                photons=np.zeros((geometry.size,1))
                photons[0,0]=amount;photons[1,0]=.2
                # A large unrelated reservoir must neither illuminate these
                # supports nor influence the compact chemistry tolerances.
                photons[uncovered[0],0]=1.e4
                options=dict(donor_fraction=.2,count_tolerance=1.e-8)
                dense=joint_exposure_update(gas,photons,[18.],membership,**options)
                compact=joint_exposure_update(gas,photons,[18.],operator,**options)
                self.assertEqual(compact['stop'],dense['stop'])
                self.assertEqual(compact['events'],dense['events'])
                for key in ('species','photons','absorbed','debit'):
                    np.testing.assert_allclose(compact[key],dense[key],rtol=5.e-12,atol=2.e-14)
                np.testing.assert_array_equal(compact['species'][uncovered],gas[uncovered])
                np.testing.assert_array_equal(compact['photons'][uncovered],photons[uncovered])
                self.assertEqual(np.count_nonzero(compact['absorbed'][uncovered]),0)
                self.assertEqual(np.count_nonzero(compact['debit'][uncovered]),0)
                self.assertLess(abs(compact['photon_error']),1.e-12*compact['absorbed_count'])
                self.assertGreater(compact['absorbed'][cells].sum(),0.)
        for field,value in [('gas',np.nan),('photons',np.inf),('photons',-1.)]:
            with self.subTest(invalid_field=field,invalid_value=value):
                bad_gas=gas.copy();bad_photons=photons.copy()
                target=bad_gas if field=='gas' else bad_photons
                target[uncovered[0],0]=value
                with self.assertRaises(ValueError):
                    joint_exposure_update(bad_gas,bad_photons,[18.],operator)

    def test_partial_retry_cache_matches_fresh_scans_and_resets_after_progress(self):
        def make_model():
            model=SpectralES(side=3,radii2=[0.,1.,2.,3.])
            model.inject([((0,0,0),[.1,0.,0.])])
            return model
        model=make_model()
        kind,proposal=model._select()
        self.assertEqual(kind,'partial')
        self.assertEqual(proposal.radius_index,0)
        scanned=model.counters['candidate_checks']
        excluded={proposal.radius_index}
        for _ in range(len(model.radii2)):
            cached_kind,cached=model._select(excluded)
            fresh_kind,fresh=make_model()._select(excluded)
            self.assertEqual(cached_kind,fresh_kind)
            self.assertEqual(model.counters['candidate_checks'],scanned)
            if cached_kind is None:
                break
            self.assertEqual(cached.radius_index,fresh.radius_index)
            np.testing.assert_array_equal(cached.centres,fresh.centres)
            excluded.add(cached.radius_index)
        self.assertIsNone(cached_kind)
        # illuminate() clears exclusions after progress and on every entry.
        # New source photons must force a scan capable of selecting a crossing.
        model.inject([((0,0,0),[100.,0.,0.])])
        kind,proposal=model._select()
        self.assertEqual(kind,'crossing')
        self.assertEqual(proposal.radius_index,len(model.radii2)-1)
        self.assertGreater(model.counters['candidate_checks'],scanned)


if __name__=='__main__':
    unittest.main()
