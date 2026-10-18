"""Milne ground-state free-bound spectra, independent of frozen OTS module.

Maxwellian free-bound photon shape q(E) proportional to
E^2 sigma(E) exp[-(E-chi)/(k_B T)]. Each ground-state recombination emits one
photon. The integrated count is normalized to HG97 alpha_A-alpha_B; this is
explicitly a shape plus fitted-rate construction, not a claim that the absolute
Milne integral and the independently fitted rates coincide exactly.
"""
import numpy as np
from numpy.polynomial.laguerre import laggauss
from atomic_hhe import sigma, recombination_coefficients, electron_density, ION, CHI_EV

KB_EV_K=8.617333262145e-5
EV_ERG=1.602176634e-12


def ground_quadrature(T=1e4,order=8):
    """Positive spectral nodes for three ground recombination continua.

    Scalar T produces energies_eV and weights of shape (3,order), ordered
    HII->HI, HeII->HeI, HeIII->HeII. Array T produces T.shape+(3,order).
    weights sum to one per row; alpha_ground_cm3_s supplies the absolute rate.
    Nodes derive from Gauss-Laguerre in x=(E-chi)/(k_B T). Spectral convergence
    must be checked for a given absorption column; fixed quadrature order is
    not an accuracy guarantee for arbitrary frequency-dependent response.
    """
    if int(order)!=order or order<1 or order>128:
        raise ValueError('Require integer quadrature order 1..128')
    t=np.asarray(T,dtype=float)
    coef=recombination_coefficients(t)
    x,w=laggauss(int(order))
    es=[];ps=[]; zs=[]
    for a in range(3):
        energy=CHI_EV[a]+KB_EV_K*t[...,None]*x
        raw=w*energy**2*sigma(energy)[...,a]
        norm=raw.sum(axis=-1,keepdims=True)
        if np.any(norm<=0): raise RuntimeError('Invalid Milne normalization')
        es.append(energy);ps.append(raw/norm)
        zs.append(KB_EV_K*t*norm[...,0])
    e=np.stack(es,axis=-2);p=np.stack(ps,axis=-2)
    mean=(e*p).sum(axis=-1)
    excess=((e-CHI_EV[...,None])*p).sum(axis=-1)
    return {'energies_eV':e,'weights':p,'alpha_ground_cm3_s':coef['ground'],
            'mean_photon_eV':mean,'mean_captured_eV':excess,
            'shape_normalization_eV3_cm2':np.stack(zs,axis=-1),
            'quadrature_order':int(order)}


def ground_photon_pdf(energy_eV,absorber,T=1e4,normalization_order=64):
    """Normalized photon probability per eV, for a single absorber (0,1,2)."""
    if absorber not in (0,1,2): raise ValueError('absorber must be 0,1,2')
    if np.ndim(T)!=0: raise ValueError('PDF evaluation expects scalar T')
    e=np.asarray(energy_eV,dtype=float)
    excess=np.maximum(e-CHI_EV[absorber],0)
    z=ground_quadrature(T,normalization_order)['shape_normalization_eV3_cm2'][absorber]
    return np.where(e>=CHI_EV[absorber],e*e*sigma(e)[...,absorber]*np.exp(-excess/(KB_EV_K*T))/z,0.)


def ground_emission(n_species_cm3,T=1e4,dt_s=1.,order=8):
    """Ground free-bound photons and captured-electron cooling per volume.

    n_species shape (...,5), units cm^-3. photon_counts_cm3 has (...,3,order).
    Returned thermal_cooling_erg_cm3 removes captured electron kinetic energy
    only. Ionization binding energy is reduced by chi per downward event and
    must not be subtracted again from thermal energy.
    """
    n=np.asarray(n_species_cm3,dtype=float)
    if n.shape[-1]!=5 or np.any(n<0) or dt_s<0:
        raise ValueError('Require nonnegative densities (...,5) and dt_s')
    q=ground_quadrature(T,order)
    events=electron_density(n)[...,None]*n[...,ION]*q['alpha_ground_cm3_s']*dt_s
    photons=events[...,None]*q['weights']
    emitted=(photons*q['energies_eV']).sum(axis=(-2,-1))*EV_ERG
    binding=(events*CHI_EV).sum(axis=-1)*EV_ERG
    cooling=(events*q['mean_captured_eV']).sum(axis=-1)*EV_ERG
    return {**q,'ground_events_cm3':events,'photon_counts_cm3':photons,
            'emitted_energy_erg_cm3':emitted,'binding_energy_removed_erg_cm3':binding,
            'thermal_cooling_erg_cm3':cooling,
            'energy_ledger_error_erg_cm3':emitted-binding-cooling}


def helium_n2_kramers_quadrature(T=1e4,order=8):
    """Explicit optional approximation, not an exact excited-state spectrum.

    For hydrogenic HeII n=2, chi=13.6 eV. Kramers sigma_n(E) proportional to
    E^-3 with unit bound-free Gaunt factor gives Milne q(E) proportional to
    E^-1 exp[-(E-chi)/kT]. Rate normalized to Friedrich2012 alpha_HeIII,n2.
    This excludes terminal Ly-alpha/two-photon cascade, which is additional.
    """
    t=np.asarray(T,dtype=float)
    coef=recombination_coefficients(t)
    x,w=laggauss(int(order))
    e=13.6+KB_EV_K*t[...,None]*x
    raw=w/e;p=raw/raw.sum(axis=-1,keepdims=True)
    return {'energies_eV':e,'weights':p,'alpha_n2_cm3_s':coef['HeIII_n2'],
            'mean_photon_eV':(e*p).sum(axis=-1),
            'mean_captured_eV':((e-13.6)*p).sum(axis=-1),
            'approximation':'hydrogenic Kramers, Gaunt factor=1'}


def hei_coefficients_wood2004(T=1e4):
    """Wood, Mathis & Ercolano 2004 Eq25, mutually consistent HeI branches.

    Do not combine this branch sum with an independent HG97 HeII alpha_A.
    alpha_A is the sum of ground plus three effective terminal-state rates.
    Validated here at T=1e4 K; temperature-dependent fit is exposed verbatim,
    without claiming an unverified wide temperature validity interval.
    """
    t=np.asarray(T,dtype=float)/1e4
    if np.any(~np.isfinite(t)) or np.any(t<=0): raise ValueError('T must be positive finite')
    ground=1.54e-13*t**-.486
    two_s_singlet=2.06e-14*t**-.676
    two_p_singlet=4.17e-14*t**-.861
    two_s_triplet=2.10e-13*t**-.778
    return {'ground_cm3_s':ground,'two_s_singlet_cm3_s':two_s_singlet,
            'two_p_singlet_cm3_s':two_p_singlet,'two_s_triplet_cm3_s':two_s_triplet,
            'alpha_A_cm3_s':ground+two_s_singlet+two_p_singlet+two_s_triplet,
            'alpha_B_cm3_s':two_s_singlet+two_p_singlet+two_s_triplet,
            'triplet_line_eV':19.8,'singlet_line_eV':21.2,
            'singlet_two_photon_total_eV':20.6,
            'singlet_two_photon_H_ionizing_yield':.56}


def absolute_milne_ground(T=1e4,order=64):
    """Absolute ground alpha from detailed balance and the SAME bound-free sigma.

    Ercolano & Storey 2006 Eq.1, after dividing energy emissivity by h*nu.
    Statistical weight ratios bound/ion are 2, 1/2, 2 for HI,HeI,HeII.
    This is an explicit alternative to HG A-B, never a silent rate replacement.
    Units: alpha cm^3/s; epsilon and photon energy eV.
    """
    t=np.asarray(T,dtype=float)
    q=ground_quadrature(t,order)
    h=6.62607015e-27; c=2.99792458e10; me=9.1093837139e-28
    kb=1.380649e-16
    pref=4*np.pi/c**2*(h*h/(2*np.pi*me*kb*t))**1.5*(EV_ERG/h)**3
    alpha=pref[...,None]*np.array([2.,.5,2.])*q['shape_normalization_eV3_cm2']
    return {**q,'alpha_absolute_milne_cm3_s':alpha,
            'ratio_to_HG_A_minus_B':alpha/q['alpha_ground_cm3_s']}


def effective_capture_moments(T=1e4):
    """Cooling moments inferred from the SAME effective rate fits.

    For a Maxwell average alpha(T)=C*T^-3/2 integral K(epsilon)
    exp[-epsilon/kT] d epsilon with T-independent capture/cascade kernel,
    <epsilon>=kT*[3/2+d ln(alpha)/d ln(T)]. This is a derived effective
    approximation when alpha is an empirical fit, not an independently
    calibrated level-resolved capture spectrum or cooling table.
    Wood power laws have positive gamma-kernel shape 1.5-beta. The returned
    tail probability bounds possible HI-ionizing preterminal photons: if the
    TOTAL preterminal energy is <13.6 eV, no individual photon can ionize HI.
    """
    from scipy.special import gammaincc
    t=np.asarray(T,dtype=float)
    recombination_coefficients(t)  # range validation
    lam=2*np.array([157807.,631515.])/t[...,None]
    y=(lam/2.740)**.407
    hg_b_mean=KB_EV_K*t[...,None]*(2.242*.407*y/(1+y))
    beta=np.array([.676,.861,.778])  # HeI 2s singlet,2p singlet,2s triplet
    shape=1.5-beta
    terminal=np.array([20.6,21.2,19.8])
    threshold=13.6-CHI_EV[1]+terminal
    z=threshold/(KB_EV_K*t[...,None])
    return {'HG_B_mean_captured_eV_HII_HeIII':hg_b_mean,
            'Wood_mean_captured_eV':KB_EV_K*t[...,None]*shape,
            'Wood_effective_gamma_shape':shape,
            'Wood_potential_ionizing_preterminal_event_fraction_upper_bound':gammaincc(shape,z),
            'Wood_potential_ionizing_preterminal_energy_eV_per_event_upper_bound':
                (CHI_EV[1]-terminal)*gammaincc(shape,z)+KB_EV_K*t[...,None]*shape*gammaincc(shape+1,z),
            'assumption':'T-independent low-density effective cascade kernel; finite-range rate-fit inversion'}


def _thermal_module(name):
    """Load sibling research thermal modules without modifying sys.path."""
    import importlib.util
    from pathlib import Path
    if not hasattr(_thermal_module,'cache'): _thermal_module.cache={}
    if name not in _thermal_module.cache:
        path=Path(__file__).resolve().parent.parent/'thermal'/(name+'.py')
        spec=importlib.util.spec_from_file_location('_hhe_research_'+name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _thermal_module.cache[name]=module
    return _thermal_module.cache[name]


def terminal_two_photon_quadrature(species='HeI',order=16):
    """True terminal two-photon shapes; weights count BOTH emitted photons.

    HeI: Chluba & Ali-Haimoud 2016 Eq2; HeII: Nussbaumer & Schmutz shape
    reproduced by Chluba & Sunyaev 2005 Eq4. Symmetric threshold splitting
    preserves total transition energy and resolves absorption discontinuities.
    """
    if species=='HeI':
        return _thermal_module('hei_two_photon').hei_two_photon_quadrature(order)
    if species!='HeII': raise ValueError("species must be HeI or HeII")
    from numpy.polynomial.legendre import leggauss
    m=_thermal_module('two_photon_energy'); e0=40.8
    edges=np.unique([0.,13.6,24.59,e0-13.6,e0-24.59,e0])
    x,w=leggauss(order); es=[];ws=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        e=lo+(x+1)*.5*(hi-lo)
        es.append(e);ws.append(w*.5*(hi-lo)*m.photon_spectrum(e,e0))
    e=np.concatenate(es);p=np.concatenate(ws);p*=2/p.sum()
    return {'energies_eV':e,'weights':p,'ionizing':e>=13.6,
            'transition_energy_eV':e0,'mean_energy_eV_per_photon':float(e@p/2),
            'ionizing_yield_per_decay':float(p[e>=13.6].sum()),
            'HeI_ionizing_yield_per_decay':float(p[e>=24.59].sum())}


def recombination_channel_table(T=1e4,order=8,two_photon_order=16,
                                ground_normalization='hg',v_two_photon=None):
    """Computable emission channels with explicit model boundaries.

    This is a NEW rate assembly, not the frozen Friedrich OTS solver.
    Ground normalization is 'hg' (HG A-B), 'milne' (absolute detailed balance),
    or 'wood_hei' (HG for H/HeIII, Wood alpha1 for HeI). In ALL assemblies
    HeI excited channels use the Wood internally consistent terminal rates;
    therefore its total alphaA is recomputed from ground+Wood branches, not
    silently fixed to HG alphaA. HeIII B uses HG and v=.285 only at 1e4 K.

    Higher Lyman photons are assumed to cascade to n=2 terminal states.
    Resonant HeI21.2 and HeII40.8 photons are tagged for line interactions,
    never declared straight continuum-only photons. Excited free-bound and
    higher-cascade spectra not specified by these rates are retained in an
    UNRESOLVED RADIATIVE ENERGY reservoir. It is not thermalized or deleted.
    Thus the resolved photon+species+energy ledgers close, while that reservoir
    openly marks missing spectral physics rather than claiming full closure.
    """
    if np.ndim(T)!=0: raise ValueError('Channel table requires scalar T')
    t=float(T);coef=recombination_coefficients(t)
    ground=ground_quadrature(t,order);ga=ground['alpha_ground_cm3_s'].copy()
    wood=hei_coefficients_wood2004(t)
    if ground_normalization=='milne': ga=absolute_milne_ground(t,64)['alpha_absolute_milne_cm3_s']
    elif ground_normalization=='wood_hei': ga[1]=wood['ground_cm3_s']
    elif ground_normalization!='hg': raise ValueError('ground_normalization must be hg, milne, or wood_hei')
    if v_two_photon is None:
        if t!=1e4: raise ValueError('v=.285 is validated here only at 1e4 K; supply a validated v(T)')
        v=.285
    else:
        v=float(v_two_photon)
        if not 0<=v<=1: raise ValueError('Require 0<=v<=1')
    channels=[]
    def add(name,parent,alpha,energies,weights,down,kind,assumptions):
        channels.append({'name':name,'parent_transition':parent,
                         'alpha_cm3_s':float(alpha),
                         'energies_eV':np.atleast_1d(np.asarray(energies,dtype=float)),
                         'weights_photons_per_event':np.atleast_1d(np.asarray(weights,dtype=float)),
                         'downward_events_per_channel_event':float(down),
                         'transport_kind':kind,'assumptions':assumptions})
    for a,name in enumerate(['HII_ground','HeII_ground','HeIII_ground']):
        add(name,a,ga[a],ground['energies_eV'][a],ground['weights'][a],1.,'continuum',
            'Milne shape; stated ground rate normalization; isotropic emission')
    add('HII_excited_unresolved',0,coef['B'][0],[],[],1.,'energy_reservoir',
        'HG B down; unspecified excited capture/cascade spectrum retained as radiative energy')
    qi=terminal_two_photon_quadrature('HeI',two_photon_order)
    add('HeI_2s_singlet',1,wood['two_s_singlet_cm3_s'],qi['energies_eV'],qi['weights'],1.,'continuum',
        'Wood effective terminal rate; CosmoSpec two-photon shape; low density, higher Lyman reprocessing')
    add('HeI_2p_singlet_line',1,wood['two_p_singlet_cm3_s']*(914/915),[21.2],[1.],1.,'HeI_resonance',
        'Initial 2p decay branching 914/915; line requires resonance scattering/conversion')
    add('HeI_2p_singlet_conversion',1,wood['two_p_singlet_cm3_s']/915,
        np.r_[.6,qi['energies_eV']],np.r_[1.,qi['weights']],1.,'continuum',
        'Initial 2p->2s branch 1/915: 0.6eV IR plus true20.6eV two-photon decay')
    add('HeI_2s_triplet',1,wood['two_s_triplet_cm3_s'],[19.8],[1.],1.,'continuum',
        'Low-density triplets reach 2^3S; neglect excited-state line opacity and collisional quenching')
    qn=helium_n2_kramers_quadrature(t,order)
    add('HeIII_n2_freebound',2,coef['HeIII_n2'],qn['energies_eV'],qn['weights'],0.,'continuum',
        'Additional photon, not another recombination; hydrogenic Kramers shape with fitted alpha_n2')
    add('HeII_Lya',2,(1-v)*coef['B'][2],[40.8],[1.],1.,'HeII_resonance',
        'HG B and low-density v=.285; high Lyman reprocessing; requires line scattering, no HeI-like conversion')
    qii=terminal_two_photon_quadrature('HeII',two_photon_order)
    add('HeII_2gamma',2,v*coef['B'][2],qii['energies_eV'],qii['weights'],1.,'continuum',
        'Hydrogenic spontaneous two-photon shape, low density; high Lyman reprocessing')
    moments=effective_capture_moments(t)
    cooling=ga*ground['mean_captured_eV']
    cooling[0]+=coef['B'][0]*moments['HG_B_mean_captured_eV_HII_HeIII'][0]
    wa=np.array([wood['two_s_singlet_cm3_s'],wood['two_p_singlet_cm3_s'],wood['two_s_triplet_cm3_s']])
    cooling[1]+=wa@moments['Wood_mean_captured_eV']
    cooling[2]+=coef['B'][2]*moments['HG_B_mean_captured_eV_HII_HeIII'][1]
    down=np.zeros(3); emitted=np.zeros(3);count=np.zeros(3)
    for ch in channels:
        a=ch['parent_transition'];alpha=ch['alpha_cm3_s']
        down[a]+=alpha*ch['downward_events_per_channel_event']
        emitted[a]+=alpha*(ch['energies_eV']@ch['weights_photons_per_event'])
        count[a]+=alpha*ch['weights_photons_per_event'].sum()
    reservoir=down*CHI_EV+cooling-emitted
    if np.any(reservoir < -1e-25): raise RuntimeError('Negative missing-spectrum energy: rate/spectrum assembly inconsistent')
    return {'channels':channels,'T_K':t,'ground_normalization':ground_normalization,
            'alpha_down_cm3_s':down,'resolved_photon_yield_rate_cm3_s':count,
            'resolved_emitted_energy_eV_cm3_s':emitted,
            'captured_electron_cooling_eV_cm3_s':cooling,
            'unresolved_radiative_energy_eV_cm3_s':reservoir,
            'energy_ledger_error_eV_cm3_s':emitted+reservoir-down*CHI_EV-cooling,
            'effective_capture_moments':moments,
            'HeI_total_rate_differs_from_HG':float(down[1]/coef['A'][1]-1),
            'full_atomic_spectral_closure':False,
            'unresolved_scope':'Excited free-bound/higher cascade spectral distribution and line spatial/frequency redistribution'}


def channel_emission(n_species_cm3,dt_s=1.,table=None):
    """Frozen-state emissivity operator, not a positivity/stiff chemistry solver.

    All counts are cm^-3, energy erg cm^-3. Returns an exactly consistent
    species delta for recombination down events. Caller must integrate this
    RHS with donor limits/coupled ionization; do not blindly apply arbitrary dt.
    """
    n=np.asarray(n_species_cm3,dtype=float)
    if n.shape[-1]!=5 or np.any(~np.isfinite(n)) or np.any(n<0) or dt_s<0:
        raise ValueError('Require finite nonnegative (...,5) densities and dt')
    if table is None: table=recombination_channel_table()
    factors=electron_density(n)[...,None]*n[...,ION]*dt_s
    down=factors*table['alpha_down_cm3_s'];dn=np.zeros_like(n)
    dn[...,0]=down[...,0];dn[...,1]=-down[...,0]
    dn[...,2]=down[...,1];dn[...,3]=-down[...,1]+down[...,2];dn[...,4]=-down[...,2]
    counts=[]
    for ch in table['channels']:
        events=factors[...,ch['parent_transition']]*ch['alpha_cm3_s']
        counts.append(events[...,None]*ch['weights_photons_per_event'])
    emitted=(factors*table['resolved_emitted_energy_eV_cm3_s']).sum(axis=-1)*EV_ERG
    unresolved=(factors*table['unresolved_radiative_energy_eV_cm3_s']).sum(axis=-1)*EV_ERG
    cooling=(factors*table['captured_electron_cooling_eV_cm3_s']).sum(axis=-1)*EV_ERG
    binding=(down*CHI_EV).sum(axis=-1)*EV_ERG
    return {'down_events_cm3':down,'delta_species_cm3':dn,'photon_counts_by_channel_cm3':counts,
            'resolved_radiation_energy_erg_cm3':emitted,
            'unresolved_radiation_energy_erg_cm3':unresolved,
            'captured_electron_cooling_erg_cm3':cooling,
            'binding_energy_released_erg_cm3':binding,
            'energy_ledger_error_erg_cm3':emitted+unresolved-binding-cooling,
            'electron_ledger_error_cm3':electron_density(dn)+down.sum(axis=-1)}


def resonance_line_event(species='HeI',photon_count=1.,two_photon_order=16):
    """Expected daughters AFTER a bound-bound resonance absorption event.

    HeI 21.2eV: A(2p->1s)/A(2p->2s)=914 (Wood2004), hence conversion
    branching probability 1/(914+1), not 1/914. Conversion emits a 0.6eV
    IR photon and the real 20.6eV two-photon spectrum. Otherwise scatter the
    line photon. HeII40.8eV has NO corresponding HeI conversion branch.
    This event changes no charge or nuclei. Directions/frequency redistribution
    and the competition with continuum opacity must be supplied by transport.
    The fixed-energy kernel neglects recoil/Doppler energy exchange; it is not
    an implementation of 3D resonance transfer or arbitrary escape probability.
    """
    p=float(photon_count)
    if not np.isfinite(p) or p<0: raise ValueError('photon_count must be finite nonnegative')
    if species=='HeI':
        energy=21.2;b=1/915.;q=terminal_two_photon_quadrature('HeI',two_photon_order)
        daughters_e=np.r_[energy,energy-q['transition_energy_eV'],q['energies_eV']]
        daughters_n=np.r_[p*(1-b),p*b,p*b*q['weights']]
        kinds=['resonance','subionizing_IR']+['two_photon']*len(q['weights'])
    elif species=='HeII':
        energy=40.8;b=0.;daughters_e=np.array([energy]);daughters_n=np.array([p]);kinds=['resonance']
    else: raise ValueError('species must be HeI or HeII')
    return {'input_photons':p,'input_energy_eV':p*energy,
            'conversion_probability':b,'converted_events':p*b,
            'energies_eV':daughters_e,'photon_counts':daughters_n,'daughter_kinds':kinds,
            'delta_species':np.zeros(5),'delta_electron_number':0.,
            'created_minus_destroyed_photons':2*p*b,
            'photon_ledger_error':daughters_n.sum()-p-2*p*b,
            'energy_ledger_error_eV':daughters_e@daughters_n-p*energy,
            'transport_not_implemented':'Spatial propagation, Voigt frequency/angular redistribution, continuum competition'}
