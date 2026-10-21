"""Reproducible angular-face graph, event, and ES validation fixtures.

Run: python benchmarks.py frozen|dynamic|checks|all --output-dir PATH. JSON files contain raw
numerical diagnostics, not only pass/fail flags. No ray tracing is used by the
solver; the rectangular-shadow classification is an independent geometric test.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from angular_front import AngularFront

FOLDER=Path.cwd()

def grid(n,ns=1,span=4.5,separation=1.5,radius=.6):
    dx=span/n
    offset=round(separation/dx) if ns==2 else 0
    assert ns==1 or abs(offset*dx-separation)<1e-12
    shape=(n+offset,n,n)
    src=np.array([[n//2,n//2,n//2]]+([[n//2+offset,n//2,n//2]] if ns==2 else []))
    coords=np.indices(shape).transpose(1,2,3,0)
    dist2=np.stack([np.sum(((coords-s)*dx)**2,axis=-1) for s in src])
    spheres=dist2<=radius**2+1e-14
    cap=np.full(shape,dx**3)
    initial=cap*np.any(spheres,axis=0)
    return dict(dx=dx,src=src,cap=cap,initial=initial,dist2=dist2,
                initial_source_photons=spheres.sum(axis=(1,2,3))*dx**3)

def solver(c,level,epsilon=0,uniform_density=0):
    return AngularFront(c['cap'],c['initial'],c['dx'],c['src'],
        np.full(len(c['src']),1/len(c['src'])),level=level,epsilon=epsilon,
        initial_source_photons=c['initial_source_photons'],uniform_density=uniform_density)

def audit(c,s):
    mass=s['mass'];time_=s['diagnostics']['time']
    ledger=s['resource_absorbed']+s['resource_escaped']-time_*s['resource_rate']
    result=dict(resource_ledger_max_abs=float(abs(ledger).max()),
        cellwise_source_ledger_L1=float(abs(mass-c['initial']-s['source_mass'].sum(axis=0)).sum()),
        total_count_error=float((mass-c['initial']).sum()+s['resource_escaped'].sum()-time_),
        min_fraction=float((mass/c['cap']).min()),max_fraction=float((mass/c['cap']).max()),
        escaped=float(s['resource_escaped'].sum()),diagnostics=s['diagnostics'])
    assert result['resource_ledger_max_abs']<1e-10,result
    assert result['cellwise_source_ledger_L1']<1e-10,result
    assert abs(result['total_count_error'])<1e-10,result
    assert result['min_fraction']>=0 and result['max_fraction']<=1,result
    return result

def frozen():
    records=[]
    for n,separation in [(32,1.5),(48,1.5),(64,1.5),(96,1.5),(30,2.4)]:
        c=grid(n,2,span=3,separation=separation,radius=1.)
        for level in ([0] if n in [64,96] else [0,1,2]):
            started=time.perf_counter()
            with solver(c,level) as model:
                s=model.snapshot()
                cross=[float(s['source_rate'][i][c['dist2'][1-i]<c['dist2'][i]].sum()/.5) for i in [0,1]]
                r=dict(case='frozen_spheres',n=n,dx=c['dx'],source_indices=c['src'].tolist(),source_positions=((c['src']-c['src'][0])*c['dx']).tolist(),separation=separation,level=level,bins=model.nbins,
                       cross_fraction=cross,analytic_cross_fraction=.125 if separation==1.5 else 0.,
                       elapsed_seconds=time.perf_counter()-started,**audit(c,s))
                records.append(r);print(json.dumps(r),flush=True)
    for n in [32,48]:
        c=grid(n,span=3.2,radius=1.2)
        coords=(np.indices(c['cap'].shape).transpose(1,2,3,0)-c['src'][0])*c['dx']
        # A one-cell-thick screen with an explicitly known rectangular front.
        sx=round(.4/c['dx']); halfcells=round(.2/c['dx'])
        ind=np.indices(c['cap'].shape).transpose(1,2,3,0)-c['src'][0]
        screen=(ind[...,0]==sx)&(abs(ind[...,1])<=halfcells)&(abs(ind[...,2])<=halfcells)
        c['initial'][screen]=0
        front=(sx-.5)*c['dx']; halfwidth=(halfcells+.5)*c['dx']
        x,y,z=np.moveaxis(coords,-1,0);half=c['dx']/2
        # Every point in these target voxels lies behind the screen rectangle.
        core=(x-half>front+c['dx'])&(np.sqrt(c['dist2'][0])>.9)&(c['initial']==0)&(~screen)
        core&=(abs(y)+half)*front < .8*halfwidth*(x-half)
        core&=(abs(z)+half)*front < .8*halfwidth*(x-half)
        exact_screen_omega=4*np.arctan(halfwidth**2/(front*np.sqrt(front**2+2*halfwidth**2)))
        for level in [0,1,2]:
            started=time.perf_counter()
            with solver(c,level) as model:
                s=model.snapshot();rates=s['source_rate'][0]
                r=dict(case='opaque_screen',n=n,dx=c['dx'],source_indices=c['src'].tolist(),level=level,bins=model.nbins,
                       screen_front=front,screen_halfwidth=halfwidth,
                       screen_fraction=float(rates[screen].sum()),
                       analytic_screen_fraction=float(exact_screen_omega/(4*np.pi)),
                       strictly_shadowed_core_leak=float(rates[core].sum()),
                       analytic_core_leak=0.,core_cells=int(core.sum()),
                       elapsed_seconds=time.perf_counter()-started,**audit(c,s))
                assert abs(r['screen_fraction']-r['analytic_screen_fraction'])<1e-12
                records.append(r);print(json.dumps(r),flush=True)
    (FOLDER/'benchmarks_frozen.json').write_text(json.dumps(records,indent=2)+'\n')
    return records

def sphere_fraction(c,radius,order=10):
    dx=c['dx'];coords=(np.indices(c['cap'].shape).transpose(1,2,3,0)-c['src'][0])*dx
    rr=np.sqrt(np.sum(coords**2,axis=-1));halfdiag=np.sqrt(3)*dx/2
    fraction=(rr+halfdiag<=radius).astype(float)
    boundary=(rr-halfdiag<radius)&(rr+halfdiag>radius);centres=coords[boundary]
    nodes,w=leggauss(order);nodes=nodes*dx/2;w=w/2;out=np.zeros(len(centres))
    for a,wa in zip(nodes,w):
        for b,wb in zip(nodes,w):
            d=(centres[:,0]+a)**2+(centres[:,1]+b)**2
            out+=wa*wb*((d[:,None]+(centres[:,2,None]+nodes)**2<=radius**2)@w)
    fraction[boundary]=out
    return fraction

def dynamic():
    records=[]
    for n in [24,30,36]:
        for ns in [1,2]:
            c=grid(n,ns);initial_volume=float(c['initial'].sum())
            for level in [0,1,2]:
                started=time.perf_counter();outputs=[];masses=[]
                with solver(c,level) as model:
                    for t in [0.,.5,1.,1.5,2.,3.,4.]:
                        s=model.advance(t);out=dict(time=t,**audit(c,s));masses.append(s['mass'])
                        if ns==1:
                            radius=(3*(initial_volume+t)/(4*np.pi))**(1/3)
                            ref=sphere_fraction(c,radius)
                            out['sphere_L1_over_total']=float(abs(s['mass']-ref*c['cap']).sum()/(initial_volume+t))
                            out['quadrature_volume_error']=float((ref*c['cap']).sum()-initial_volume-t)
                        else:
                            a,b=s['source_mass'];mirror=np.roll(b[::-1],1,axis=0)
                            out['source_mirror_L1_over_budget']=float(abs(a-mirror).sum()/(t/2)) if t>0 else 0.
                        outputs.append(out)
                    r=dict(case='single' if ns==1 else 'double',n=n,dx=c['dx'],source_indices=c['src'].tolist(),source_positions=((c['src']-c['src'][0])*c['dx']).tolist(),level=level,bins=model.nbins,
                        initial_volume=initial_volume,elapsed_seconds=time.perf_counter()-started,records=outputs)
                    if ns==2:r['equal_volume_spherical_contact_time']=2*(4*np.pi/3*.75**3-c['initial_source_photons'][0])
                    fieldfile=f'angular_{r["case"]}_n{n}_level{level}_fields.npz'
                    np.savez_compressed(FOLDER/fieldfile,mass=np.stack(masses),initial=c['initial'],dx=c['dx'],sources=c['src'],times=np.array([o['time'] for o in outputs]))
                    r['field_file']=fieldfile;records.append(r)
                    print(json.dumps(dict(case=r['case'],n=n,level=level,seconds=r['elapsed_seconds'],last=outputs[-1])),flush=True)
                    (FOLDER/'benchmarks_dynamic.json').write_text(json.dumps(records,indent=2)+'\n')
    return records

def checks():
    records=[]
    c=grid(24,2)
    eps=.1;L=1+eps
    times=[.5,1.,2.,4.];refs={}
    with solver(c,1) as model:
        for t in sorted(set([v for x in times for v in [x/L,x,x*L]])):
            refs[t]=model.advance(t)['mass']
    with solver(c,1,epsilon=eps,uniform_density=1) as model:
        for t in times:
            s=model.advance(t);u=s['mass'];lo=refs[t/L];hi=refs[t*L]
            r=dict(case='ES_comparison',time=t,epsilon=eps,
                lower_violation=float(np.maximum(lo-u,0).max()),upper_violation=float(np.maximum(u-hi,0).max()),
                L1_over_emitted=float(abs(u-refs[t]).sum()/t),L1_bound=2*eps/L,**audit(c,s))
            assert r['lower_violation']<1e-12 and r['upper_violation']<1e-12
            assert r['L1_over_emitted']<=r['L1_bound']+1e-12
            records.append(r)
    with solver(c,1,epsilon=eps,uniform_density=1) as a,solver(c,1,epsilon=eps,uniform_density=1) as b:
        one=a.advance(4)
        for t in np.linspace(0,4,41):split=b.advance(t)
        r=dict(case='output_partition',mass_max_abs=float(abs(one['mass']-split['mass']).max()),
               source_mass_max_abs=float(abs(one['source_mass']-split['source_mass']).max()))
        assert r['mass_max_abs']<1e-12 and r['source_mass_max_abs']<1e-12
        records.append(r)
    # An initially neutral source cell must first consume its own capacity.
    c=grid(5,span=5,radius=.1);c['initial'][:]=0
    with solver(c,1) as model:
        a=model.advance(.25);r=dict(case='neutral_source',source_mass=float(a['mass'][tuple(c['src'][0])]),**audit(c,a))
        assert abs(r['source_mass']-.25)<1e-12
        model.advance(20);records.append(r)
    # An entirely transparent box sends all output to escape, including ES mode.
    c=grid(5,span=5,radius=10)
    with solver(c,1,epsilon=.1,uniform_density=1) as model:
        a=model.advance(3);r=dict(case='fully_transparent_escape',**audit(c,a))
        assert abs(r['escaped']-3)<1e-12
        records.append(r)
    # A dormant zero-budget source must not change the ES score field.
    cap=np.ones((3,3,3));initial=np.zeros_like(cap);initial[1,1,1]=1
    with AngularFront(cap,initial,1,[[1,1,1]],[1],level=1,epsilon=2,uniform_density=1) as single,AngularFront(cap,initial,1,[[1,1,1],[0,1,1]],[1,0],level=1,epsilon=2,uniform_density=1) as ghost:
        a=single.advance(.1);b=ghost.advance(.1)
        difference=float(abs(a['mass']-b['mass']).max())
        assert difference<1e-14
        records.append(dict(case='dormant_source_ES',mass_max_abs=difference))
    with AngularFront(np.ones((1,1,1)),np.zeros((1,1,1)),1,[[0,0,0]],[1],level=0,epsilon=1e17) as model:
        model.set_es_scores(np.zeros((1,1,1)));a=model.advance(.25)
        assert abs(a['mass'].sum()-.25)<1e-14
        records.append(dict(case='large_epsilon_ES',epsilon=1e17,mass=float(a['mass'].sum())))
    # ES redistributes absorbed resource rates only; open-boundary escape is fixed.
    cap=np.ones((5,5,5));initial=cap.copy();initial[4,3,3]=initial[4,4,2]=0
    with AngularFront(cap,initial,1,[[2,2,2]],[1],level=1) as base,AngularFront(cap,initial,1,[[2,2,2]],[1],level=1,epsilon=.1) as bias:
        scores=np.arange(125,dtype=float).reshape(cap.shape);bias.set_es_scores(scores)
        raw=base.advance(.5);modified=bias.advance(.5)
        r=dict(case='ES_mixed_escape',escape_max_change=float(abs(raw['resource_escaped']-modified['resource_escaped']).max()),
               resource_ledger_max_abs=float(abs(modified['resource_absorbed']+modified['resource_escaped']-.5*modified['resource_rate']).max()),
               escaped=float(modified['resource_escaped'].sum()))
        assert r['escape_max_change']<1e-14 and r['resource_ledger_max_abs']<1e-14 and 0<r['escaped']<.5
        records.append(r)
    # A cut one ulp before a capacity event must not leave stale opaque rates.
    with AngularFront(np.ones((1,1,1)),np.full((1,1,1),.9),1,[[0,0,0]],[1],level=0) as model:
        cut=model.advance(np.nextafter(1.-.9,-np.inf));final=model.advance(.2)
        r=dict(case='rounding_before_event',cut_events=cut['diagnostics']['events'],
               final_mass=float(final['mass'].sum()),absorbed=float(final['resource_absorbed'].sum()),
               escaped=float(final['resource_escaped'].sum()),source_absorbed=float(final['source_mass'].sum()),
               clamp_absolute_mass=final['diagnostics']['clamp_absolute_mass'])
        assert abs(r['absorbed']-.1)<1e-14 and abs(r['escaped']-.1)<1e-14
        assert abs(r['source_absorbed']-.1)<1e-14
        records.append(r)
    closed=solver(c,0);closed.close()
    for method in [closed.snapshot,closed.diagnostics,lambda:closed.advance(1),lambda:closed.resource_rate(0,0),lambda:closed.set_es_scores(c['initial'])]:
        try:method()
        except RuntimeError:continue
        raise AssertionError('closed solver method did not fail safely')
    records.append(dict(case='closed_handle',methods_checked=5,passed=True))
    (FOLDER/'benchmarks_checks.json').write_text(json.dumps(records,indent=2)+'\n')
    for r in records:print(json.dumps(r),flush=True)
    return records

def main():
    global FOLDER
    parser=argparse.ArgumentParser()
    parser.add_argument('suite',choices=['frozen','dynamic','checks','all'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    FOLDER=args.output_dir.resolve()
    FOLDER.mkdir(parents=True,exist_ok=True)
    for name,call in [('frozen',frozen),('dynamic',dynamic),('checks',checks)]:
        if args.suite in [name,'all']:call()

if __name__=='__main__':main()
