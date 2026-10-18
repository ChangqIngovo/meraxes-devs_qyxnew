"""Closed static, low-temperature H/He thermal model, with eV/cm^3 units.

Model choices: immediate local photoelectron thermalization; no nonthermal
secondaries; monochromatic/coherent resonance scattering exchanges no heat;
unresolved preterminal *spectral detail* is replaced by specified subionizing
escape radiation, meaning inactive radiation removed from the interacting
ionizing subsystem. It need not have crossed the box boundary yet. Its energy
remains in the cumulative global radiation ledger; no low-energy trajectory
calculation is required. This is not a universal photoelectron or atomic model.

Species order HI,HII,HeI,HeII,HeIII. Absorber order HI,HeI,HeII.
Calibrated prototype temperature domain 5e3--2e4 K. No thermal floor/ceiling.
"""
import numpy as np
from scipy.special import gammaincc

KB_EV_K=8.617333262145e-5
EV_ERG=1.602176634e-12
CHI_EV=np.array([13.6,24.59,54.4])
ABS=np.array([0,2,3])
T_MIN=5000.
T_MAX=20000.


def check_temperature(T):
    t=np.asarray(T,dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t<T_MIN) or np.any(t>T_MAX):
        raise ValueError('Low-temperature closure requires 5000 <= T <= 20000 K; no temperature clipping is permitted')
    return t


def _species(n):
    n=np.asarray(n,dtype=float)
    if n.shape[-1]!=5 or np.any(~np.isfinite(n)) or np.any(n<0):
        raise ValueError('Require finite nonnegative species array (...,5)')
    return n


def electron_density(n):
    n=np.asarray(n,dtype=float)
    return n[...,1]+n[...,3]+2*n[...,4]


def nuclei_density(n):
    return np.sum(_species(n),axis=-1)


def particle_density(n):
    return nuclei_density(n)+electron_density(n)


def binding_energy_eV_cm3(n):
    n=_species(n)
    return 13.6*n[...,1]+24.59*n[...,3]+(24.59+54.4)*n[...,4]


def thermal_energy_eV_cm3(n,T):
    """u=(3/2) k_B T(n_H+n_He+n_e); binding energy is separate."""
    t=np.asarray(T,dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t<0): raise ValueError('Invalid T')
    return 1.5*KB_EV_K*t*particle_density(n)


def temperature_from_energy(n,u_eV_cm3):
    """No temperature floor, no thermostat; vacuum u=0 returns T=0."""
    u=np.asarray(u_eV_cm3,dtype=float); npart=particle_density(n)
    if np.any(~np.isfinite(u)) or np.any(u<0): raise ValueError('Invalid thermal energy')
    if np.any((npart==0)&(u!=0)): raise ValueError('Nonzero matter energy in vacuum')
    return np.divide(u,1.5*KB_EV_K*npart,out=np.zeros(np.broadcast_shapes(u.shape,npart.shape)),where=npart>0)


def absorption_energy_ledger(energies_eV,absorbed_counts_cm3):
    """Counts shape (...,frequency,3); no subthreshold absorption allowed.

    All photoelectron kinetic energy is deposited as thermal energy in THIS
    model. The diagnostic high-energy fraction marks where neglect of
    nonthermal excitation/secondary ionization needs physical validation.
    """
    e=np.asarray(energies_eV,dtype=float)
    a=np.asarray(absorbed_counts_cm3,dtype=float)
    if e.ndim!=1 or a.shape[-2:]!=(len(e),3) or np.any(a<0) or np.any(~np.isfinite(a)):
        raise ValueError('Expected positive (...,frequency,3) absorption counts')
    excess=e[:,None]-CHI_EV
    if np.any((excess<0)&(a>0)): raise ValueError('Subthreshold photoionization')
    incoming=np.sum(a*e[:,None],axis=(-2,-1))
    binding=np.sum(a*CHI_EV,axis=(-2,-1))
    heat=np.sum(a*np.maximum(excess,0),axis=(-2,-1))
    high=np.sum(a*np.maximum(excess,0)*(excess>=10.2),axis=(-2,-1))
    return {'absorbed_radiation_eV_cm3':incoming,'binding_gain_eV_cm3':binding,
            'photoheating_eV_cm3':heat,'photoelectron_energy_above_HI_excitation_eV_cm3':high,
            'energy_error_eV_cm3':incoming-binding-heat}


def collisional_ionization_coefficients(T):
    """HG97 Appendix Lotz fits, cm^3/s, absorber order HI,HeI,HeII.

    Published fit accuracy ~3% for 1e4--1e9 K. Below 1e4 K the same expression
    is an explicit exponential-tail extrapolation, not a newly calibrated fit.
    Thermal energy cost per event is the model's matching CHI_EV, not a
    separately rounded threshold. The atomic solver must apply species events.
    """
    t=check_temperature(T)[...,None]
    lam=2*np.array([157807.,285335.,631515.])/t
    amplitude=np.array([21.11,32.38,19.95])
    power=np.array([1.089,1.146,1.089])
    scale=np.array([.354,.416,.553])
    exponent=np.array([.874,.987,.735])
    denom_power=np.array([1.101,1.056,1.275])
    return amplitude*t**-1.5*np.exp(-lam/2)*lam**-power/(1+(lam/scale)**exponent)**denom_power


def collisional_ionization_rates(n,T):
    n=_species(n)
    rates=electron_density(n)[...,None]*n[...,ABS]*collisional_ionization_coefficients(T)
    dn=np.zeros_like(n)
    dn[...,0]=-rates[...,0];dn[...,1]=rates[...,0]
    dn[...,2]=-rates[...,1];dn[...,3]=rates[...,1]-rates[...,2];dn[...,4]=rates[...,2]
    return {'events_cm3_s':rates,'delta_species_cm3_s':dn,
            'thermal_loss_eV_cm3_s':np.sum(rates*CHI_EV,axis=-1)}


def cooling_channels_eV_cm3_s(n,T):
    """Additional inactive-radiation cooling, excluding recombination/CI.

    HI collisional excitation: HG97/Black81/Cen92 fit; 10.2 eV line is below
    ionization threshold. H line transfer/recoil is neglected in this model.
    Free-free: Theuns98 Appendix B fit with gff=1.1+.34 exp(-(5.5-logT)^2/3).
    ALL this radiation is assigned subionizing escape from the active ionizing
    subsystem, not instantaneous geometric escape from the box. The free-free high-E
    energy-tail proxy assumes frequency-independent Gaunt factor. It is tiny
    in the adopted low-T domain but is not identically zero in real plasma.
    """
    n=_species(n);t=check_temperature(T);ne=electron_density(n)
    exc_coeff=7.5e-19*np.exp(-.75*157807./t)/(1+np.sqrt(t/1e5))/EV_ERG
    excitation=ne*n[...,0]*exc_coeff
    gaunt=1.1+.34*np.exp(-(5.5-np.log10(t))**2/3)
    ff_coeff=1.42e-27*gaunt*np.sqrt(t)/EV_ERG
    freefree=ne*(n[...,1]+n[...,3]+4*n[...,4])*ff_coeff
    return {'HI_excitation_eV_cm3_s':excitation,'freefree_eV_cm3_s':freefree,
            'total_eV_cm3_s':excitation+freefree,
            'subionizing_escape_eV_cm3_s':excitation+freefree,
            'freefree_energy_fraction_above_HI_proxy':np.exp(-13.6/(KB_EV_K*t))}


def thermal_cooling_rate_eV_cm3_s(n,T):
    return cooling_channels_eV_cm3_s(n,T)['total_eV_cm3_s']


def thermal_update(n_before,n_after,u_eV_cm3,photoheating_eV_cm3,
                   capture_cooling_eV_cm3,extra_cooling_eV_cm3=0.,
                   collisional_ionization_cooling_eV_cm3=0.,enforce_domain=True):
    """Apply already integrated events; caller owns stiff/substep integration.

    Never deduct binding energy twice. Recombination removes captured electron
    kinetic energy only; emitted photons also carry released binding energy.
    CI removes CHI per upward event. n_after supplies the changed heat capacity.
    """
    nb=_species(n_before);na=_species(n_after)
    for idx in ([0,1],[2,3,4]):
        before=nb[...,idx].sum(axis=-1);after=na[...,idx].sum(axis=-1)
        if np.any(np.abs(before-after)>2e-13*np.maximum(before,1e-300)):
            raise ValueError('Nuclei were not conserved by supplied events')
    u=np.asarray(u_eV_cm3,dtype=float)
    h=np.asarray(photoheating_eV_cm3,dtype=float)
    rec=np.asarray(capture_cooling_eV_cm3,dtype=float)
    extra=np.asarray(extra_cooling_eV_cm3,dtype=float)
    ci=np.asarray(collisional_ionization_cooling_eV_cm3,dtype=float)
    for x in (u,h,rec,extra,ci):
        if np.any(~np.isfinite(x)) or np.any(x<0): raise ValueError('Invalid energy increment')
    unew=u+h-rec-extra-ci
    if np.any(unew<0): raise ValueError('Thermal donor exhausted: reduce timestep, do not clip')
    tnew=temperature_from_energy(na,unew)
    if enforce_domain: check_temperature(tnew)
    return {'u_eV_cm3':unew,'T_K':tnew,
            'energy_update_error_eV_cm3':unew-u-h+rec+extra+ci,
            'inside_temperature_domain':(tnew>=T_MIN)&(tnew<=T_MAX)}


def soft_cascade_tail_diagnostics(T):
    """Controlled effective-model tail diagnostics, not exact atomic errors.

    Wood HeI capture kernel from its power-law alpha is Gamma(1.5-beta,kT).
    Preterminal energy = chi_HeI - E_terminal + epsilon. The returned energy
    bound/13.6 bounds possible missing ionizing photon count in THAT kernel.
    No interacting in-flight reservoir is kept: this energy is assigned to
    inactive subthreshold radiation, possibly still inside the box, in the
    cumulative global ledger. These values quantify an approximation error.
    """
    t=check_temperature(T);kt=KB_EV_K*t[...,None]
    terminal=np.array([20.6,21.2,19.8]);shape=1.5-np.array([.676,.861,.778])
    offset=24.59-terminal;x=(13.6-offset)/kt
    p=gammaincc(shape,x)
    tail_energy=offset*p+kt*shape*gammaincc(shape+1,x)
    return {'HeI_channel_order':['singlet_2s','singlet_2p','triplet_2s'],
            'HeI_capture_mean_eV':shape*kt,
            'HeI_possible_ionizing_preterminal_event_fraction':p,
            'HeI_missing_ionizing_count_per_event_bound':tail_energy/13.6,
            'HeI_missing_ionizing_energy_eV_per_event_bound':tail_energy,
            'freefree_high_energy_fraction_proxy':np.exp(-13.6/(KB_EV_K*t))}


def subionizing_escape_packet(energy_eV_cm3,representative_energy_eV=10.2):
    """An explicit optional grey escape representation, not a true spectrum.

    Photon count is a model choice at the representative energy. This energy
    is removed from the active ionizing subsystem, never deposited as heat or
    held as a waiting unresolved process. It is still accounted as inactive
    radiation, inside or outside the box; geometric escape is not instantaneous.
    """
    energy=np.asarray(energy_eV_cm3,dtype=float); e=float(representative_energy_eV)
    if not 0<e<13.6 or np.any(energy<0) or np.any(~np.isfinite(energy)):
        raise ValueError('Require positive subionizing representative energy')
    return {'escaped_energy_eV_cm3':energy,'representative_energy_eV':e,
            'model_photon_count_cm3':energy/e,'energy_error_eV_cm3':energy/e*e-energy}


def excited_freebound_tail_bounds(T):
    """Chernoff bounds from the HG97 B-rate Maxwell kernel, per ALL B events.

    This avoids inventing a level-resolved free-bound spectrum. Z(T) is
    proportional to T^1.5 alpha_B(T). For any T'>T,
    P(eps>=d) <= Z(T')/Z(T) exp[-d(1/T-1/T')/k_B].
    Optimizing T' gives a conservative bound. HI excited free-bound needs
    eps>=13.6-3.4; HeII n>=3 free-bound needs eps>=13.6-54.4/9. All B captures
    bound the n>=3 subset, so the latter is per total HeIII B event. Higher
    Lyman cascade-reprocessing errors are separate and NOT bounded here.
    The result is conditional on the effective fitted Maxwell kernel.
    """
    from scipy.optimize import minimize_scalar
    t=float(check_temperature(T)); rows=[]
    for name,tr,d in [('HI_excited',157807.,10.2),('HeII_n_ge_3',631515.,13.6-54.4/9)]:
        def logz(temp):
            return -2.242*np.log1p((2*tr/temp/2.740)**.407)
        def objective(r):
            tp=t/(1-r)
            return logz(tp)-logz(t)-r*d/(KB_EV_K*t)
        opt=minimize_scalar(objective,bounds=(0,1-t/5e5),method='bounded')
        rows.append({'channel':name,'capture_threshold_eV':d,
                     'missing_ionizing_freebound_photons_per_B_event_bound':float(min(1,np.exp(opt.fun))),
                     'auxiliary_temperature_K':float(t/(1-opt.x))})
    return {'T_K':t,'rows':rows}
