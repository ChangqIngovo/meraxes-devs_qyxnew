"""Closed, explicitly approximate low-density H/He reaction/emission kernel.

State: HI,HII,HeI,HeII,HeIII physical cm^-3; u is thermal eV cm^-3.
Applicable temperature domain: 5,000--20,000 K. No thermostat/floor.
All preterminal/high-n radiation omitted from the resolved spectrum is assigned
DECOUPLED SUBIONIZING RADIATION, recorded by low_energy_escape. This removes
subsequent interaction with the active ionizing subsystem; it does NOT assert
instantaneous travel across the box. This is the declared soft-UV/high-Lyman-
cascade approximation, not a pending energy reservoir. Ground free-bound photons use
absolute Milne rates/shapes; HeI terminals use Wood; HeIII n2 Kramers is explicit.
Photoelectron kinetic energy thermalizes locally; no secondary ionization.
Terminal two-photon groups preserve actual band counts and energy moments.
Resonance photons are TAGGED and require the chosen closed line treatment in
transport; this module never silently treats them as continuum photons.
"""
from pathlib import Path
import sys
import numpy as np
OLD=Path(__file__).resolve().parents[1]/'reference'/'atomic'
if str(OLD) not in sys.path: sys.path.insert(0,str(OLD))
from atomic_hhe import sigma,recombination_coefficients,electron_density,CHI_EV,ABSORBER,ION,C_LIGHT
from recombination_spectra import (ground_quadrature,absolute_milne_ground,
    helium_n2_kramers_quadrature,hei_coefficients_wood2004,effective_capture_moments,
    terminal_two_photon_quadrature,KB_EV_K)

T_MIN=5000.;T_MAX=20000.
BINDING_EV=np.array([0.,13.6,0.,24.59,24.59+54.4])
V_TWO_PHOTON=.285  # explicit constant-branching approximation over stated domain


def particle_density(n):
    n=np.asarray(n,dtype=float)
    return n[...,0]+n[...,1]+n[...,2]+n[...,3]+n[...,4]+electron_density(n)


def thermal_energy(n,T):
    return 1.5*KB_EV_K*np.asarray(T)*particle_density(n)


def temperature(n,u):
    ntot=particle_density(n)
    return np.divide(np.asarray(u),1.5*KB_EV_K*ntot,out=np.zeros_like(ntot),where=ntot>0)


def _bands(species):
    q=terminal_two_photon_quadrature(species,32)
    e=q['energies_eV'];w=q['weights']
    masks=[e<13.6,(e>=13.6)&(e<24.59),e>=24.59]
    counts=np.array([w[m].sum() for m in masks])
    energies=np.array([(e[m]*w[m]).sum() for m in masks])
    means=np.divide(energies,counts,out=np.zeros(3),where=counts>0)
    return counts,energies,means


HEI_BANDS=_bands('HeI')
HEII_BANDS=_bands('HeII')


def kernel(T=1e4,ground_order=2):
    """Vectorized frozen-temperature coefficients and positive spectral groups.

    Returns energies (...,Gnew), photon_alpha (...,Gnew) cm^3/s, parent(Gnew),
    kind(Gnew):0 continuum,1 HeI resonance,2 HeII resonance.
    alpha_down (...,3), cooling_alpha/low_energy_escape_alpha in eV cm^3/s.
    The TWO-PHOTON spectra are gray WITHIN each absorber-threshold band;
    actual band counts/energies are retained, opacity variation is approximated.
    Ground/n2 continuum quadrature order controls their spectral approximation.
    """
    t=np.asarray(T,dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t<T_MIN) or np.any(t>T_MAX):
        raise ValueError('Closed model domain is 5000<=T<=20000 K; no artificial temperature floor')
    coef=recombination_coefficients(t);q=ground_quadrature(t,ground_order)
    ga=absolute_milne_ground(t,max(8,ground_order))['alpha_absolute_milne_cm3_s']
    wood=hei_coefficients_wood2004(t);qn=helium_n2_kramers_quadrature(t,ground_order)
    moments=effective_capture_moments(t)
    parent=[];kind=[];names=[];energies=[];alpha=[]
    def add(name,a,k,e,r):
        energies.append(np.broadcast_to(e,t.shape));alpha.append(np.broadcast_to(r,t.shape))
        parent.append(a);kind.append(k);names.append(name)
    for a,label in enumerate(['HII_ground','HeII_ground','HeIII_ground']):
        for g in range(ground_order):
            add(label,a,0,q['energies_eV'][...,a,g],ga[...,a]*q['weights'][...,a,g])
    for g in range(ground_order):
        add('HeIII_n2_freebound',2,0,qn['energies_eV'][...,g],coef['HeIII_n2']*qn['weights'][...,g])
    ss=wood['two_s_singlet_cm3_s'];sp=wood['two_p_singlet_cm3_s'];st=wood['two_s_triplet_cm3_s']
    nhei,_,ehei=HEI_BANDS
    add('HeI_2gamma_H_band',1,0,ehei[1],(ss+sp/915)*nhei[1])
    add('HeI_triplet',1,0,19.8,st)
    add('HeI_resonance',1,1,21.2,sp*(914/915))
    nheii,_,eheii=HEII_BANDS
    add('HeII_2gamma_H_band',2,0,eheii[1],V_TWO_PHOTON*coef['B'][...,2]*nheii[1])
    add('HeII_2gamma_HeI_band',2,0,eheii[2],V_TWO_PHOTON*coef['B'][...,2]*nheii[2])
    add('HeII_resonance',2,2,40.8,(1-V_TWO_PHOTON)*coef['B'][...,2])
    energies=np.stack(energies,axis=-1);alpha=np.stack(alpha,axis=-1)
    parent=np.array(parent);kind=np.array(kind)
    down=ga.copy();down[...,0]+=coef['B'][...,0]
    down[...,1]+=ss+sp+st;down[...,2]+=coef['B'][...,2]
    cooling=ga*q['mean_captured_eV']
    cooling[...,0]+=coef['B'][...,0]*moments['HG_B_mean_captured_eV_HII_HeIII'][...,0]
    wa=np.stack([ss,sp,st],axis=-1)
    cooling[...,1]+=(wa*moments['Wood_mean_captured_eV']).sum(axis=-1)
    cooling[...,2]+=coef['B'][...,2]*moments['HG_B_mean_captured_eV_HII_HeIII'][...,1]
    emitted=np.stack([(alpha[...,parent==a]*energies[...,parent==a]).sum(axis=-1) for a in range(3)],axis=-1)
    escape=down*CHI_EV+cooling-emitted
    if np.any(escape<0): raise RuntimeError('Negative low-energy escape: invalid atomic assembly')
    return {'T_K':t,'energies_eV':energies,'photon_alpha_cm3_s':alpha,
            'group_parent':parent,'group_kind':kind,'group_names':names,
            'alpha_down_cm3_s':down,'alpha_ground_cm3_s':ga,
            'alpha_excited_cm3_s':down-ga,
            'captured_cooling_alpha_eV_cm3_s':cooling,
            'low_energy_escape_alpha_eV_cm3_s':escape,
            'potential_HeI_preterminal_ionizing_event_tail_bound':moments['Wood_potential_ionizing_preterminal_event_fraction_upper_bound']}


def _thermal_module():
    import importlib.util
    if not hasattr(_thermal_module,'cached'):
        p=Path(__file__).resolve().parent.parent/'thermal'/'thermal_closure.py'
        spec=importlib.util.spec_from_file_location('closed_thermal_closure',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        _thermal_module.cached=m
    return _thermal_module.cached


def reaction_rates(n,p,energies_eV,T=1e4,ground_order=2,include_collisional=True):
    """Instantaneous coupled RHS with known emitted photons and escaped energy.

    Public for independent fixed-T analytic rate tests. Rates cm^-3 s^-1;
    energy rates eV cm^-3 s^-1. Does not mutate state or insert diffuse photons.
    """
    k=kernel(T,ground_order)
    return _rates(np.asarray(n,dtype=float),np.asarray(p,dtype=float),
                  np.asarray(energies_eV,dtype=float),k,include_collisional)


def _rates(n,p,e,k,collisional):
    cross=sigma(e).T
    photo=C_LIGHT*n[...,ABSORBER,None]*cross*p[...,None,:]
    factors=electron_density(n)[...,None]*n[...,ION]
    down=factors*k['alpha_down_cm3_s']
    if collisional:
        tm=_thermal_module()
        ci=electron_density(n)[...,None]*n[...,ABSORBER]*tm.collisional_ionization_coefficients(k['T_K'])
        extra_cooling=tm.thermal_cooling_rate_eV_cm3_s(n,k['T_K'])
    else:
        ci=np.zeros_like(down);extra_cooling=np.zeros(n.shape[:-1])
    up=photo.sum(axis=-1)+ci;net=up-down
    dn=np.zeros_like(n)
    dn[...,0]=-net[...,0];dn[...,1]=net[...,0]
    dn[...,2]=-net[...,1];dn[...,3]=net[...,1]-net[...,2];dn[...,4]=net[...,2]
    emitted=factors[...,k['group_parent']]*k['photon_alpha_cm3_s']
    heat=(photo*np.maximum(e[None,:]-CHI_EV[:,None],0)).sum(axis=(-2,-1))
    rec_cooling=(factors*k['captured_cooling_alpha_eV_cm3_s']).sum(axis=-1)
    collisional_ion_cooling=(ci*CHI_EV).sum(axis=-1)
    low_escape=(factors*k['low_energy_escape_alpha_eV_cm3_s']).sum(axis=-1)+extra_cooling
    return {'dn':dn,'photon_absorption_rate':photo.sum(axis=-2),
            'primary_absorptions':photo,'recombination_down':down,
            'collisional_ionizations':ci,'emitted':emitted,
            'photoheat':heat,'recombination_cooling':rec_cooling,
            'collisional_ionization_cooling':collisional_ion_cooling,
            'extra_radiative_cooling':extra_cooling,'low_energy_escape':low_escape,
            'du':heat-rec_cooling-collisional_ion_cooling-extra_cooling}



def advance(n_species,n_photon,energies_eV,dt_s,thermal_energy_eV_cm3,
            donor_fraction=.1,ground_order=2,include_collisional=True,
            max_substeps=100000,line_kind=None,line_sigma_cm2=None):
    """Positive reactions with exponential photon survival and diffuse births.

    Rates/spectral nodes freeze at the global-step initial T; n,p,u evolve.
    No photon-loss step limit: frozen-hazard photoabsorption and line conversion
    are integrated exponentially, while material/thermal donor changes limit h.
    Global dt refinement controls frozen-T, frozen-opacity and endpoint births.
    line_kind(G):0 continuum,1 HeI21.2,2 HeII40.8. Independent elastic scattering
    belongs to transport. HeI conversion competes with continuum absorption.
    line_sigma_cm2(...,2) may supply exactly the transport line-center cross
    sections; otherwise they are imported from the shared theory module.
    """
    n=np.asarray(n_species,dtype=float).copy();p=np.asarray(n_photon,dtype=float).copy()
    e=np.asarray(energies_eV,dtype=float);u=np.asarray(thermal_energy_eV_cm3,dtype=float).copy()
    if n.shape[-1]!=5 or p.shape[:-1]!=n.shape[:-1] or p.shape[-1]!=len(e):
        raise ValueError('Expected n(...,5),p(...,G),energy(G)')
    if np.any(n<0) or np.any(p<0) or np.any(u<=0) or dt_s<0 or not 0<donor_fraction<=.5:
        raise ValueError('Require nonnegative state, positive thermal u, dt>=0, donor_fraction in(0,.5]')
    if not all(np.all(np.isfinite(a)) for a in (n,p,u,e)) or not np.isfinite(dt_s):
        raise ValueError('Nonfinite input')
    u=np.broadcast_to(u,n.shape[:-1]).copy();t=temperature(n,u);k=kernel(t,ground_order)
    lk=np.zeros(len(e),dtype=int) if line_kind is None else np.asarray(line_kind,dtype=int)
    if lk.shape!=e.shape or np.any(~np.isin(lk,[0,1,2])): raise ValueError('Invalid line_kind')
    if np.any(abs(e[lk==1]-21.2)>1e-12) or np.any(abs(e[lk==2]-40.8)>1e-12):
        raise ValueError('Tagged line bins must have energies21.2/40.8eV')
    if np.any(lk==1):
        if line_sigma_cm2 is None:
            import importlib.util
            path=Path(__file__).resolve().parent.parent/'theory'/'scatter_drift.py'
            spec=importlib.util.spec_from_file_location('closed_scatter_drift',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            line_sigma_cm2=module.linecenter_sigma(t)
        line_sigma=np.asarray(line_sigma_cm2,dtype=float)[...,0]
    else: line_sigma=np.zeros_like(t)
    initial_n=n.copy();initial_p=p.copy();initial_u=u.copy();shape=n.shape[:-1]
    sums={'primary_absorptions':np.zeros(shape+(3,len(e))),
          'recombination_down':np.zeros(shape+(3,)),
          'collisional_ionizations':np.zeros(shape+(3,)),
          'emitted':np.zeros_like(k['energies_eV'])}
    for key in ['photoheat','recombination_cooling','collisional_ionization_cooling',
                'extra_radiative_cooling','low_energy_escape','line_conversions']:
        sums[key]=np.zeros(shape)
    cross=sigma(e).T
    conversion_group=k['group_names'].index('HeI_2gamma_H_band')
    elapsed=0.;steps=0
    while elapsed<dt_s:
        if steps>=max_substeps: raise RuntimeError('Reaction substep limit reached')
        r=_rates(n,p,e,k,include_collisional)
        h=float(dt_s-elapsed)
        hazard=C_LIGHT*n[...,ABSORBER,None]*cross
        conv_hazard=C_LIGHT*n[...,2,None]*line_sigma[...,None]/915*(lk==1)
        total=hazard.sum(axis=-2)+conv_hazard
        shares=np.divide(hazard,total[...,None,:],out=np.zeros_like(hazard),where=total[...,None,:]>0)
        cshare=np.divide(conv_hazard,total,out=np.zeros_like(total),where=total>0)
        for reduction in range(100):
            newp=p*np.exp(-total*h)
            lost=p-newp
            photo=lost[...,None,:]*shares
            conversions=(lost*cshare).sum(axis=-1)
            rec=h*r['recombination_down'];ci=h*r['collisional_ionizations']
            net=photo.sum(axis=-1)+ci-rec
            dn=np.zeros_like(n)
            dn[...,0]=-net[...,0];dn[...,1]=net[...,0]
            dn[...,2]=-net[...,1];dn[...,3]=net[...,1]-net[...,2];dn[...,4]=net[...,2]
            heat=(photo*np.maximum(e[None,:]-CHI_EV[:,None],0)).sum(axis=(-2,-1))
            du=heat-h*(r['recombination_cooling']+r['collisional_ionization_cooling']+r['extra_radiative_cooling'])
            acceptable=np.all(-dn<=donor_fraction*n+1e-300) and np.all(-du<=donor_fraction*u)
            if acceptable: break
            h*=.5
        else: raise RuntimeError('Could not find positive material/thermal step')
        if h<=0 or not np.isfinite(h): raise RuntimeError('Invalid reaction step')
        n+=dn;p=newp;u+=du
        if np.any(n<0) or np.any(p<0) or np.any(u<=0): raise RuntimeError('Positivity failure')
        sums['primary_absorptions']+=photo;sums['recombination_down']+=rec
        sums['collisional_ionizations']+=ci;sums['photoheat']+=heat
        sums['emitted']+=h*r['emitted']
        sums['emitted'][...,conversion_group]+=conversions*HEI_BANDS[0][1]
        sums['line_conversions']+=conversions
        sums['low_energy_escape']+=conversions*(21.2-HEI_BANDS[1][1])
        for key in ['recombination_cooling','collisional_ionization_cooling','extra_radiative_cooling','low_energy_escape']:
            sums[key]+=h*r[key]
        elapsed+=h;steps+=1
    final_t=temperature(n,u)
    incoming_energy=initial_u+initial_n@BINDING_EV+(initial_p*e).sum(axis=-1)
    final_energy=u+n@BINDING_EV+(p*e).sum(axis=-1)
    emitted_energy=(sums['emitted']*k['energies_eV']).sum(axis=-1)
    energy_error=final_energy+emitted_energy+sums['low_energy_escape']-incoming_energy
    electron_error=electron_density(n)-electron_density(initial_n)-(
        sums['primary_absorptions'].sum(axis=(-2,-1))+sums['collisional_ionizations'].sum(axis=-1)-sums['recombination_down'].sum(axis=-1))
    photon_error=initial_p.sum(axis=-1)-p.sum(axis=-1)-sums['primary_absorptions'].sum(axis=(-2,-1))-sums['line_conversions']
    return {'n_species':n,'n_photon':p,'thermal_energy_eV_cm3':u,'temperature_K':final_t,
            'emitted_energy_eV':k['energies_eV'],'emitted_counts':sums.pop('emitted'),
            'emitted_kind':k['group_kind'],'emitted_line_kind':k['group_kind'],'emitted_group_names':k['group_names'],
            'substeps':steps,'initial_temperature_K':t,
            'within_temperature_domain':(final_t>=T_MIN)&(final_t<=T_MAX),
            'energy_ledger_error_eV_cm3':energy_error,'electron_ledger_error_cm3':electron_error,
            'photon_absorption_ledger_error_cm3':photon_error,**sums}
