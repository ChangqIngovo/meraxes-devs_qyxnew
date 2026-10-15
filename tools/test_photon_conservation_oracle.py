#!/usr/bin/env python3
"""Independent explicit-edge LP validation of the standalone C99 allocator."""
import argparse, ctypes as ct, json, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack, eye

PD=ct.POINTER(ct.c_double); PI=ct.POINTER(ct.c_int); PB=ct.POINTER(ct.c_ubyte)
class Problem(ct.Structure):
    _fields_=[('dim',ct.c_size_t*3),('ncell',ct.c_size_t)]+[(k,PD) for k in ['source','capacity','baseline','lower','upper','radius']]+[('priority',PI)]
class Options(ct.Structure):
    _fields_=[(k,ct.c_size_t) for k in ['max_edges','max_bytes','seed_neighbors','cut_neighbors']]+[(k,ct.c_double) for k in ['absolute_tolerance','relative_tolerance','flow_tolerance']]
class Diagnostic(ct.Structure):
    _fields_=[('priority',ct.c_int)]+[(k,ct.c_double) for k in ['prefix_flow','retained_in_group','baseline_retained_in_group']]
class Metrics(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ['lower_feasible','full_source_budget','full_graph_cut_separated']]+[(k,ct.c_size_t) for k in ['sources','targets','source_target_edges','positive_edges','separation_rounds','priority_groups','bytes_peak']]+[(k,ct.c_double) for k in ['photons','lower_demand','absorbed','source_residual','mincut_capacity','source_cut_budget','target_cut_capacity','hall_source_excess','lower_cut_deficit','lower_stage_flow','retention_stage_flow','final_stage_flow','morphology_l1','retained_overlap','max_source_overspend','max_target_excess','max_lower_deficit','max_source_relative_residual','max_edge_balance_error','max_geometry_excess','ledger_error','cut_flow_error']]
Visitor=ct.CFUNCTYPE(None,ct.c_size_t,ct.c_size_t,ct.c_double,ct.c_void_p)
class Result(ct.Structure):
    _fields_=[('y',PD),('source_residual',PD),('source_spent',PD),('cut_source',PB),('cut_target',PB),('priority_diagnostics',ct.POINTER(Diagnostic)),('priority_diagnostics_capacity',ct.c_size_t),('visit_edge',Visitor),('visitor_user',ct.c_void_p),('metrics',Metrics),('status',ct.c_int)]

def load_library(path=None):
    if path is None:
        here=Path(__file__).resolve()
        candidates=[here.parents[1]/'meraxes_pc/src/physics',here.parents[1]/'src/physics']
        source=next(p for p in candidates if (p/'photon_conservation.c').exists())
        path=Path(tempfile.mkdtemp(prefix='pc_oracle_'))/'libpc.so'
        subprocess.run(['cc','-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic','-fPIC','-shared',str(source/'photon_conservation.c'),'-lm','-o',str(path)],check=True)
    lib=ct.CDLL(str(path));lib.pc_default_options.argtypes=[ct.POINTER(Options)];lib.pc_allocate.argtypes=[ct.POINTER(Problem),ct.POINTER(Options),ct.POINTER(Result)];lib.pc_allocate.restype=ct.c_int
    return lib

def allocate(lib,source,capacity,baseline,lower,upper,radius,priority=None,options=None):
    shape=source.shape;arrays=[np.ascontiguousarray(a,dtype=np.float64).ravel() for a in [source,capacity,baseline,lower,upper,radius]]
    rank=np.zeros(source.size,np.int32) if priority is None else np.ascontiguousarray(priority,dtype=np.int32).ravel()
    p=Problem((ct.c_size_t*3)(*shape),source.size,*[a.ctypes.data_as(PD) for a in arrays],rank.ctypes.data_as(PI))
    y=np.zeros(source.size);res=y.copy();spent=y.copy();cs=np.zeros(source.size,np.uint8);ctar=cs.copy();diag=(Diagnostic*max(1,source.size))();edges=[]
    @Visitor
    def visit(i,j,v,user):edges.append((i,j,v))
    r=Result(y.ctypes.data_as(PD),res.ctypes.data_as(PD),spent.ctypes.data_as(PD),cs.ctypes.data_as(PB),ctar.ctypes.data_as(PB),diag,len(diag),visit,None,Metrics(),0)
    if options is None:options=Options();lib.pc_default_options(ct.byref(options))
    status=lib.pc_allocate(ct.byref(p),ct.byref(options),ct.byref(r))
    result={k:getattr(r.metrics,k) for k,_ in Metrics._fields_};result.update(status=status,y=y,residual=res,spent=spent,cut_source=cs,cut_target=ctar,edges=edges,diagnostics=[{k:getattr(diag[i],k) for k,_ in Diagnostic._fields_} for i in range(min(r.metrics.priority_groups,len(diag)))])
    return result

def run(trials=40,library=None):
    lib=load_library(library);rows=[]
    for trial in range(trials):
        rng=np.random.default_rng(42000+trial);n=4;shape=(n,n,n);N=n**3
        source=np.zeros(N);source[rng.choice(N,12,replace=False)]=rng.uniform(.2,2.0,12)
        cap=rng.uniform(.2,1.8,N);cap[rng.choice(N,8,replace=False)]=0
        up=cap*rng.uniform(.6,1.,N);base=cap*rng.uniform(0,1,N);lo=up*rng.choice([0.,.2,.8],N) if trial%3 else np.zeros(N)
        rad=rng.uniform(.4,2.8,N);rank=rng.integers(0,3,N,dtype=np.int32)
        sp=np.flatnonzero(source);tp=np.flatnonzero(up);coords=np.array(np.unravel_index(np.arange(N),shape)).T
        d=abs(coords[sp,None]-coords[tp][None]);d=np.minimum(d,n-d);d2=(d*d).sum(2);ii,jj=np.where(d2<=rad[tp]**2+1e-10);ne=len(ii);kk=np.arange(ne)
        rs=coo_matrix((np.ones(ne),(ii,kk)),shape=(len(sp),ne)).tocsr();rt=coo_matrix((np.ones(ne),(jj,kk)),shape=(len(tp),ne)).tocsr()
        lp0=linprog(-np.ones(ne),A_ub=vstack([rs,rt]),b_ub=np.r_[source[sp],lo[tp]],bounds=(0,None),method='highs');assert lp0.success
        result=allocate(lib,*[a.reshape(shape) for a in [source,cap,base,lo,up,rad]],rank.reshape(shape))
        feasible=bool(abs(-lp0.fun-lo.sum())<1e-8)
        assert result['status']>=0,(trial,result)
        assert result['lower_feasible']==feasible,(trial,result)
        assert abs(result['lower_stage_flow']+lp0.fun)<1e-8
        assert result['full_graph_cut_separated']
        qs=result['cut_source'][sp].astype(bool);qt=result['cut_target'][tp].astype(bool)
        assert not np.any((d2<=rad[tp]**2+1e-10)&qs[:,None]&~qt[None,:])
        all_delta=abs(coords[sp,None]-coords[None]);all_delta=np.minimum(all_delta,n-all_delta)
        full_neighborhood=np.any(((all_delta*all_delta).sum(2)<=rad**2+1e-10)&qs[:,None],axis=0)
        assert np.array_equal(result['cut_target'].astype(bool),full_neighborhood)
        assert result['max_edge_balance_error']<1e-8 and result['cut_flow_error']<1e-8
        row=dict(trial=trial,lower_feasible=feasible,status=result['status'],edge_count=result['source_target_edges'],cut_error=result['cut_flow_error'])
        if feasible:
            current=lo[tp].copy();middle=np.clip(base[tp],lo[tp],up[tp]);lp_prefix=[]
            for diag in result['diagnostics']:
                current[rank[tp]==diag['priority']]=middle[rank[tp]==diag['priority']]
                lp=linprog(-np.ones(ne),A_ub=vstack([rs,rt,-rt]),b_ub=np.r_[source[sp],current,-lo[tp]],bounds=(0,None),method='highs');assert lp.success
                assert abs(diag['prefix_flow']+lp.fun)<1e-8
                lp_prefix.append(-lp.fun)
            au=vstack([rs,rt,-rt]);bu=np.r_[source[sp],up[tp],-lo[tp]]
            lp=linprog(-np.ones(ne),A_ub=au,b_ub=bu,bounds=(0,None),method='highs');assert lp.success;maximum=-lp.fun
            assert abs(result['absorbed']-maximum)<1e-8
            # z_j <= y_j and z_j <= baseline_j. Sequentially maximize groups,
            # holding the final total absorption fixed and prior optima exact.
            ext=vstack([hstack([au,csr_matrix((au.shape[0],len(tp)))]),hstack([-rt,eye(len(tp))]),hstack([csr_matrix((len(tp),ne)),eye(len(tp))])])
            eb=np.r_[bu,np.zeros(len(tp)),base[tp]]
            eq=[np.r_[np.ones(ne),np.zeros(len(tp))]];rhs=[maximum]
            for diag in result['diagnostics']:
                group=(rank[tp]==diag['priority']).astype(float);objective=np.r_[np.zeros(ne),-group]
                optimum=linprog(objective,A_ub=ext,b_ub=eb,A_eq=csr_matrix(np.array(eq)),b_eq=rhs,bounds=(0,None),method='highs');assert optimum.success
                retained=-optimum.fun
                actual=np.minimum(result['y'][tp],base[tp])[rank[tp]==diag['priority']].sum()
                assert abs(actual-retained)<1e-8,(trial,diag,actual,retained)
                assert abs(diag['baseline_retained_in_group']-retained)<1e-8
                eq.append(np.r_[np.zeros(ne),group]);rhs.append(retained)
            ed=np.asarray(result['edges']).reshape(-1,3);ei=ed[:,0].astype(int);ej=ed[:,1].astype(int);ef=ed[:,2]
            assert np.max(abs(np.bincount(ei,weights=ef,minlength=N)-result['spent']))<1e-8
            assert np.max(abs(np.bincount(ej,weights=ef,minlength=N)-result['y']))<1e-8
            assert np.max(result['y']-up)<1e-8 and np.max(lo-result['y'])<1e-8
            row['maxflow_error']=abs(result['absorbed']-maximum)
        else:assert result['status']==2
        rows.append(row)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library');p.add_argument('--trials',type=int,default=40);p.add_argument('--output');a=p.parse_args();rows=run(a.trials,a.library)
    if a.output:Path(a.output).write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(passed=True,trials=len(rows),lower_infeasible=sum(not r['lower_feasible'] for r in rows),max_maxflow_error=max(r.get('maxflow_error',0) for r in rows)),indent=2))
