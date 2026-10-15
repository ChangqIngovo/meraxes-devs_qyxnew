#!/usr/bin/env python3
"""Independent small explicit-edge QP checks of the native unique-map stage.

Requires NumPy/SciPy and a C compiler. Builds a temporary shared library; does
not need the Meraxes data files or Python bindings. No production speed claim.
"""
import argparse
import ctypes as ct
import json
import pathlib
import subprocess
import tempfile
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

SZ, D, I = ct.c_size_t, ct.c_double, ct.c_int
PD, PI, PB = ct.POINTER(D), ct.POINTER(I), ct.POINTER(ct.c_ubyte)
class Problem(ct.Structure):
    _fields_ = [('dim', SZ*3), ('ncell', SZ)] + [(x, PD) for x in ('source','capacity','baseline','lower','upper','radius')] + [('priority', PI)]
class FlowOptions(ct.Structure):
    _fields_ = [(x,SZ) for x in ('max_edges','max_bytes','seed_neighbors','cut_neighbors')] + [(x,D) for x in ('absolute_tolerance','relative_tolerance','flow_tolerance')]
class Priority(ct.Structure):
    _fields_ = [('priority',I)] + [(x,D) for x in ('prefix_flow','retained_in_group','baseline_retained_in_group')]
class FlowMetrics(ct.Structure):
    _fields_ = [(x,I) for x in ('lower_feasible','full_source_budget','full_graph_cut_separated')] + [(x,SZ) for x in ('sources','targets','source_target_edges','positive_edges','separation_rounds','priority_groups','bytes_peak')] + [(x,D) for x in ('photons','lower_demand','absorbed','source_residual','mincut_capacity','source_cut_budget','target_cut_capacity','hall_source_excess','lower_cut_deficit','lower_stage_flow','retention_stage_flow','final_stage_flow','morphology_l1','retained_overlap','max_source_overspend','max_target_excess','max_lower_deficit','max_source_relative_residual','max_edge_balance_error','max_geometry_excess','ledger_error','cut_flow_error')]
Visitor = ct.CFUNCTYPE(None,SZ,SZ,D,ct.c_void_p)
class Result(ct.Structure):
    _fields_ = [('y',PD),('source_residual',PD),('source_spent',PD),('cut_source',PB),('cut_target',PB),('priority_diagnostics',ct.POINTER(Priority)),('priority_diagnostics_capacity',SZ),('visit_edge',Visitor),('visitor_user',ct.c_void_p),('metrics',FlowMetrics),('status',I)]
class ShapeOptions(ct.Structure):
    _fields_ = [(x,SZ) for x in ('max_hall_cuts','max_master_sweeps','max_coordinate_iterations','max_waterfill_iterations','max_evaluations','max_bytes')] + [(x,D) for x in ('absolute_tolerance','relative_tolerance','maximum_multiplier')]
class ShapeMetrics(ct.Structure):
    _fields_ = [('status',I),('last_oracle_status',I)] + [(x,SZ) for x in ('hall_cuts','master_sweeps','master_evaluations','oracle_calls','bytes_peak')] + [(x,D) for x in ('source_budget','target_total','unused_budget','objective','evaluated_dual_lower','signed_primal_dual_gap','equality_error','max_hall_violation','max_priority_violation','max_priority_equality_error','max_bound_violation','max_complementarity','max_oracle_target_error','oracle_target_l1_error','max_final_target_correction','final_target_correction_l1','tolerance')] + [('full_graph_certificate',I)]

def ptr(a,kind=PD): return a.ctypes.data_as(kind)
def make_result(n, edges=None):
    r = Result(); arrays = [np.zeros(n) for _ in range(3)]
    r.y,r.source_residual,r.source_spent = map(ptr,arrays)
    diag = (Priority*n)(); r.priority_diagnostics=diag; r.priority_diagnostics_capacity=n
    cb = Visitor((lambda i,j,v,u: edges.append((i,j,v))) if edges is not None else (lambda *a: None))
    r.visit_edge=cb
    return r, (arrays,diag,cb)

def load_library(path):
    lib=ct.CDLL(str(path))
    lib.pc_allocate.argtypes=[ct.POINTER(Problem),ct.POINTER(FlowOptions),ct.POINTER(Result)]
    lib.pc_shape_refine.argtypes=[ct.POINTER(Problem),ct.POINTER(FlowOptions),ct.POINTER(Result),PD,ct.POINTER(ShapeOptions),ct.POINTER(Result),ct.POINTER(ShapeMetrics)]
    return lib

def solve_case(lib, case, independent=True):
    arrays=[np.ascontiguousarray(case[x],dtype=float) for x in ('S','C','b','L','U','R')]
    S,C,b,L,U,R=arrays; n=len(S); rank=np.ascontiguousarray(case['rank'],dtype=np.int32)
    dim=tuple(case.get('dim',(n,1,1)))
    p=Problem((SZ*3)(*dim),n,*map(ptr,arrays),ptr(rank,PI))
    fo=FlowOptions(); lib.pc_default_options(ct.byref(fo)); fo.absolute_tolerance=1e-12;fo.relative_tolerance=1e-12
    so=ShapeOptions(); lib.pc_shape_default_options(ct.byref(so)); so.absolute_tolerance=1e-12;so.relative_tolerance=1e-11;so.max_hall_cuts=128
    edges=[]; final_edges=[]; ref,keep=make_result(n,edges); out,keep_out=make_result(n,final_edges)
    st=lib.pc_allocate(ct.byref(p),ct.byref(fo),ct.byref(ref))
    if st not in (0,1): raise AssertionError(('reference',st))
    m=ShapeMetrics(); prior=np.ascontiguousarray(case.get('prior',b),dtype=float)
    ss=lib.pc_shape_refine(ct.byref(p),ct.byref(fo),ct.byref(ref),ptr(prior),ct.byref(so),ct.byref(out),ct.byref(m))
    rec={x:getattr(m,x) for x,_ in ShapeMetrics._fields_};rec['name']=case.get('name','random')
    if ss: raise AssertionError(rec)
    y=keep_out[0][0]; F=ref.metrics.absorbed
    assert np.max(np.maximum(L-y,0))<=1e-8
    assert np.max(np.maximum(y-U,0))<=1e-8
    rows=np.zeros(n);cols=np.zeros(n)
    for i,j,v in final_edges:
        assert v>=0
        a=np.array(np.unravel_index(i,dim));z=np.array(np.unravel_index(j,dim))
        delta=np.minimum(np.abs(a-z),np.array(dim)-np.abs(a-z))
        assert np.dot(delta,delta)<=R[j]**2+1e-10
        rows[i]+=v;cols[j]+=v
    assert np.max(np.abs(cols-y))<1e-10
    assert np.max(np.maximum(rows-S,0))<1e-10
    assert np.max(np.abs(rows+keep_out[0][1]-S))<1e-10
    assert abs(np.sum(y)-F)<=128*np.finfo(float).eps*max(F,np.finfo(float).tiny)
    direct_objective=.5*np.sum(np.divide((y-prior)**2,C,out=np.zeros(n),where=C>0))
    assert abs(direct_objective-m.objective)<1e-12
    if independent and F>0:
        coord=np.array(np.unravel_index(np.arange(n),dim)).T
        delta=np.abs(coord[:,None]-coord[None,:]);delta=np.minimum(delta,np.array(dim)-delta)
        legal=np.sum(delta*delta,axis=2)<=R[None,:]**2+1e-12
        ii,jj=np.where(legal & (S[:,None]>0) & (U[None,:]>0));ne=len(ii)
        row=np.zeros((n,ne));col=np.zeros((n,ne));row[ii,np.arange(ne)]=1;col[jj,np.arange(ne)]=1
        # Exhaustively enumerate the exact bipartite Hall inequalities and
        # the affine pieces of each retained-overlap constraint. This is an
        # independent target QP for the complete explicit small graph.
        rows=[]; rhs=[]
        active_sources=np.flatnonzero(S>0)
        for bits in range(1,1<<len(active_sources)):
            Q=active_sources[[bool(bits & (1<<i)) for i in range(len(active_sources))]]
            neighbor=np.any(legal[Q],axis=0)
            rows.append(neighbor.astype(float));rhs.append(np.sum(S[Q])-(np.sum(S)-F))
        for g in range(ref.metrics.priority_groups):
            d=ref.priority_diagnostics[g];cells=np.flatnonzero(rank==d.priority)
            for bits in range(1<<len(cells)):
                A=cells[[bool(bits & (1<<i)) for i in range(len(cells))]]
                B=cells[[not bool(bits & (1<<i)) for i in range(len(cells))]]
                row=np.zeros(n);row[A]=1
                rows.append(row);rhs.append(d.baseline_retained_in_group-np.sum(b[B]))
        mat=np.array(rows); rhs=np.array(rhs)
        nonzero=np.any(mat!=0,axis=1) & ~np.all(mat==1,axis=1)
        mat=mat[nonzero];rhs=rhs[nonzero]
        unique={}
        for row,r in zip(mat,rhs): unique[tuple(row)]=max(r,unique.get(tuple(row),-np.inf))
        mat=np.array(list(unique));rhs=np.array(list(unique.values()))
        cons=[LinearConstraint(np.ones((1,n)),F,F),LinearConstraint(mat,rhs,np.inf)]
        inv=np.divide(1,C,out=np.zeros(n),where=C>0)
        def obj(x):d=x-prior;return .5*np.dot(d*d,inv)
        def jac(x):return (x-prior)*inv
        opt=minimize(obj,keep[0][0].copy(),jac=jac,bounds=Bounds(L,U),constraints=cons,method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
        violation=max(np.max(np.maximum(rhs-mat@opt.x,0)),abs(np.sum(opt.x)-F))
        if not opt.success or violation>1e-8:raise AssertionError(('QP failed',case['name'],opt.message,violation))
        rec['independent_objective']=float(opt.fun);rec['objective_difference']=float(m.objective-opt.fun)
        rec['max_map_difference']=float(np.max(np.abs(y-opt.x)))
        assert abs(rec['objective_difference'])<2e-8,rec
        assert rec['max_map_difference']<3e-5,rec
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');ap.add_argument('--cases',type=int,default=40);a=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        libpath=pathlib.Path(td)/'shape.so'
        subprocess.run(['cc','-O2','-std=c99','-Wall','-Wextra','-Werror','-shared','-fPIC',str(root/'src/physics/photon_conservation.c'),str(root/'src/physics/photon_conservation_shape.c'),'-lm','-o',str(libpath)],check=True)
        lib=load_library(libpath);rng=np.random.default_rng(87214);records=[]
        for k in range(a.cases):
            n=3+k%3;C=rng.uniform(.4,1.3,n);S=rng.uniform(0,1.2,n);S[rng.random(n)<.35]=0
            b=rng.uniform(0,1,n)*C;L=np.zeros(n);U=C.copy();R=rng.choice([0.,1.,3.],n);rank=rng.integers(0,3,n,dtype=np.int32)
            # Positive L is guaranteed locally feasible using a fraction of each local source.
            if k%3==0:L=np.minimum(.2*S,.3*C)
            case=dict(S=S,C=C,b=b,L=L,U=U,R=R,rank=rank,name=f'random_{k}')
            if k%4==0:case['prior']=rng.uniform(0,1,n)*C
            records.append(solve_case(lib,case))
        result={'cases':len(records),'max_abs_objective_difference':max(abs(x.get('objective_difference',0)) for x in records),'max_map_difference':max(x.get('max_map_difference',0) for x in records),'results':records}
        if a.output:pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
if __name__=='__main__':main()
