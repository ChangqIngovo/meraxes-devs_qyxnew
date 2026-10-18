"""Vectorized H/He atomic counts and Friedrich+2012 coupled OTS closure.

Species: HI,HII,HeI,HeII,HeIII. Absorbers/transitions: HI,HeI,HeII.
All number densities are physical cm^-3, times s, cross sections cm^2.
The coupled OTS default is the low-density, T=10^4 K count closure.
It does not assign a fabricated energy spectrum to integrated cascade yields.
"""
import numpy as np

C_LIGHT = 2.99792458e10
CHI_EV = np.array([13.6, 24.59, 54.4])
ABSORBER = np.array([0, 2, 3])
ION = np.array([1, 3, 4])
CHANNEL_NAMES = ('H_ground', 'HeII_ground', 'HeII_excited_H', 'HeIII_ground',
                 'HeIII_n2', 'HeIII_Lya', 'HeIII_2gamma_low', 'HeIII_2gamma_high')
# This records only threshold intervals, not monochromatic emission energies.
CHANNEL_BIN = np.array([0, 1, 0, 2, 0, 1, 0, 1])
P_HEI = 0.96
L_TWO_PHOTON = 1.425
M_TWO_PHOTON = 0.737
V_TWO_PHOTON_1E4K = 0.285


def sigma(energy_eV):
    """Return shape energy.shape+(3,); HI/HeI/HeII in cm^2.

    Hydrogenic HI/HeII expression and Verner+1996 HeI fit, as in Meraxes.
    Thresholds are 13.6,24.59,54.4 eV; radiation below threshold has zero opacity.
    """
    e=np.asarray(energy_eV,dtype=float)
    if np.any(~np.isfinite(e)) or np.any(e<0):
        raise ValueError('Photon energies must be finite and nonnegative')
    result=np.zeros(e.shape+(3,))
    for a,z in [(0,1),(2,2)]:
        ratio=np.maximum(e/CHI_EV[a],1.0)
        eps=np.sqrt(ratio-1.0)
        safe=np.maximum(eps,1e-8)
        shape=np.exp(4-4*np.arctan(safe)/safe)/(-np.expm1(-2*np.pi/safe))
        result[...,a]=np.where(e>=CHI_EV[a],6.3e-18/z**2*ratio**-4*shape,0.0)
    x=e/13.61-.4434
    y=np.sqrt(x*x+2.136**2)
    he=9.492e-16*((x-1)**2+2.039**2)*y**(.5*3.188-5.5)*(1+np.sqrt(y/1.469))**-3.188
    result[...,1]=np.where(e>=CHI_EV[1],he,0.0)
    return result


def recombination_coefficients(T):
    """Hui & Gnedin 1997 radiative alpha_A/B, shape T.shape+(3,), cm^3/s.

    Common H+He fit range is 5e3--5e5 K; no dielectronic contribution here.
    Coupled OTS default below is more narrowly fixed at 1e4 K.
    """
    t=np.asarray(T,dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t<5e3) or np.any(t>5e5):
        raise ValueError('Common radiative H/He fit range: 5e3 <= T <= 5e5 K')
    lam=2*np.array([157807.,285335.,631515.])/t[...,None]
    fa=lambda x:1.269e-13*x**1.503/(1+(x/.522)**.470)**1.923
    fb=lambda x:2.753e-14*x**1.5/(1+(x/2.740)**.407)**2.242
    aa=np.stack([fa(lam[...,0]),3e-14*lam[...,1]**.654,2*fa(lam[...,2])],axis=-1)
    ab=np.stack([fb(lam[...,0]),1.26e-14*lam[...,1]**.750,2*fb(lam[...,2])],axis=-1)
    return {'A':aa,'B':ab,'ground':aa-ab,
            'HeIII_n2':3.4e-13*(t/1e4)**-.6}


def electron_density(n):
    n=np.asarray(n,dtype=float)
    return n[...,1]+n[...,3]+2*n[...,4]


def _route(n_abs,energy):
    opacity=n_abs*sigma(energy)
    total=opacity.sum(axis=-1,keepdims=True)
    return np.divide(opacity,total,out=np.zeros_like(opacity),where=total>0)


def recombination_rates(n_species_cm3,T=1e4,mode='coupled_ots',f_lya=1.0,v_two_photon=None):
    """Return number-event rates and net species derivatives (cm^-3 s^-1).

    coupled_ots implements Friedrich+2012 Eq13/Tables1,2, f=1 by default.
    Rates A,B follow HG97; alpha_HeIII,n2 follows Friedrich Appendix G.
    The v=.285 default is established only at 10^4 K. Other temperatures need
    an explicitly supplied v_two_photon, from a separately validated table.
    p=.96 is the low-density limit. No collisional/dielectronic terms are added.

    At exactly zero relevant opacity, no absorption is invented: that channel's
    photons appear in unresolved_channels and must be propagated/accounted for.
    This is also where nonabsorbed (1-f_lya) line photons are recorded.

    independent_B is the uncoupled OTS comparison: B down, no explicit photons.
    It is an effective net-event ledger, unlike the A+diffuse coupled ledger.
    """
    n=np.asarray(n_species_cm3,dtype=float)
    if n.shape[-1]!=5 or np.any(~np.isfinite(n)) or np.any(n<0):
        raise ValueError('n_species must be finite nonnegative shape (...,5)')
    f=np.asarray(f_lya,dtype=float)
    if np.any(f<0) or np.any(f>1): raise ValueError('Require 0<=f_lya<=1')
    coef=recombination_coefficients(T)
    ne=electron_density(n)
    ions=n[...,ION]
    down_a=ne[...,None]*ions*coef['A']
    down_b=ne[...,None]*ions*coef['B']
    shape=np.broadcast_shapes(n.shape[:-1],np.shape(T),np.shape(f))
    emitted=np.zeros(shape+(8,))
    absorbed=np.zeros(shape+(8,3))
    unresolved=np.zeros_like(emitted)
    if mode=='independent_B':
        down=down_b
    elif mode=='coupled_ots':
        if v_two_photon is None:
            if np.any(np.asarray(T)!=1e4):
                raise ValueError('v=.285 is restricted to T=1e4; supply validated v_two_photon(T)')
            v=V_TWO_PHOTON_1E4K
        else:
            v=np.asarray(v_two_photon,dtype=float)
            if np.any(v<0) or np.any(v>1): raise ValueError('Require 0<=v_two_photon<=1')
        down=down_a
        ground=ne[...,None]*ions*coef['ground']
        emitted[...,0]=ground[...,0]
        emitted[...,1]=ground[...,1]
        emitted[...,2]=P_HEI*down_b[...,1]
        emitted[...,3]=ground[...,2]
        emitted[...,4]=ne*ions[...,2]*coef['HeIII_n2']
        emitted[...,5]=(1-v)*down_b[...,2]
        emitted[...,6]=v*(L_TWO_PHOTON-M_TWO_PHOTON)*down_b[...,2]
        emitted[...,7]=v*M_TWO_PHOTON*down_b[...,2]
        nabs=n[...,ABSORBER]
        h_only=np.zeros(shape+(3,))
        h_only[...,0]=n[...,0]>0
        r24=_route(nabs,CHI_EV[1])
        r54=_route(nabs,CHI_EV[2])
        r40=_route(nabs,40.8)
        routes=np.stack([h_only,r24,h_only,r54,h_only,f[...,None]*r40,h_only,r24],axis=-2)
        absorbed=emitted[...,None]*routes
        # Evaluate physical survival separately from sums of floating fractions:
        # normalization roundoff must not look like a diffuse photon reservoir.
        h_present=np.broadcast_to(n[...,0]>0,shape)
        p24=np.broadcast_to(np.any(r24>0,axis=-1),shape)
        p54=np.broadcast_to(np.any(r54>0,axis=-1),shape)
        p40=np.broadcast_to(np.any(r40>0,axis=-1),shape)
        fraction=np.stack([h_present,p24,h_present,p54,h_present,f*p40,h_present,p24],axis=-1)
        unresolved=emitted*(1-fraction)
    else:
        raise ValueError("mode must be 'coupled_ots' or 'independent_B'")
    up=absorbed.sum(axis=-2)
    net=up-down
    dn=np.zeros(shape+(5,))
    dn[...,0]=-net[...,0]; dn[...,1]=net[...,0]
    dn[...,2]=-net[...,1]; dn[...,3]=net[...,1]-net[...,2]; dn[...,4]=net[...,2]
    return {'down':down,'up':up,'dn':dn,'case_a_down':down_a,'case_b_down':down_b,
            'emitted_channels':emitted,'absorbed_channels_species':absorbed,
            'unresolved_channels':unresolved,'electron_density':ne}


def advance(n_species,n_photon,energies_eV,dt_s,T=1e4,mode='coupled_ots',f_lya=1.0,
            donor_fraction=.2,max_substeps=100000,v_two_photon=None):
    """Batched synchronous, positivity-preserving explicit reference chemistry.

    Inputs cm^-3; energies shape(G,), dt_s seconds. No transport here. Returned
    integrated reactions have cm^-3 units. Unresolved diffuse photons are
    returned as channel counts and are NOT added into the primary photon bins:
    their spectrum/directions cannot be inferred from this count closure.

    An adaptive bound on net depletion ensures nonnegative species/photons;
    convergence is first order. Stiff cases require many steps; not a production
    integrator. Optional max_dt is not implicit: caller time refinement remains
    needed for accuracy when dt is shorter than the depletion bound.
    """
    n=np.array(n_species,dtype=float,copy=True)
    p=np.array(n_photon,dtype=float,copy=True)
    e=np.asarray(energies_eV,dtype=float)
    if n.shape[-1]!=5 or p.shape[:-1]!=n.shape[:-1] or p.shape[-1]!=e.size:
        raise ValueError('Shapes require (...,5), (...,G), (G,)')
    if np.any(n<0) or np.any(p<0) or dt_s<0 or not 0<donor_fraction<1:
        raise ValueError('Nonnegative inputs, dt, and 0<donor_fraction<1 required')
    cross=sigma(e).T
    shape=n.shape[:-1]
    acc={'primary_absorptions':np.zeros(shape+(3,e.size)),
         'recombination_down':np.zeros(shape+(3,)),
         'recombination_up':np.zeros(shape+(3,)),
         'emitted_channels':np.zeros(shape+(8,)),
         'absorbed_channels_species':np.zeros(shape+(8,3)),
         'unresolved_channels':np.zeros(shape+(8,))}
    elapsed=0.; steps=0
    while elapsed<dt_s:
        rec=recombination_rates(n,T,mode,f_lya,v_two_photon)
        photo=C_LIGHT*cross*n[...,ABSORBER,None]*p[...,None,:]
        pup=photo.sum(axis=-1)
        dn=rec['dn'].copy()
        dn[...,0]-=pup[...,0]; dn[...,1]+=pup[...,0]
        dn[...,2]-=pup[...,1]; dn[...,3]+=pup[...,1]-pup[...,2]; dn[...,4]+=pup[...,2]
        dp=-photo.sum(axis=-2)
        losses=np.concatenate([np.maximum(-dn,0.0),-dp],axis=-1)
        state=np.concatenate([n,p],axis=-1)
        pos=losses>0
        bound=donor_fraction*np.min(state[pos]/losses[pos]) if pos.any() else np.inf
        step=min(dt_s-elapsed,bound)
        if step<=0 or elapsed+step==elapsed:
            raise RuntimeError('Local time step underflow; inspect zero-opacity discontinuity')
        n+=step*dn; p+=step*dp
        if np.any(n<0) or np.any(p<0): raise RuntimeError('Positivity failure')
        acc['primary_absorptions']+=step*photo
        for output,key in [('recombination_down','down'),('recombination_up','up'),
                           ('emitted_channels','emitted_channels'),
                           ('absorbed_channels_species','absorbed_channels_species'),
                           ('unresolved_channels','unresolved_channels')]:
            acc[output]+=step*rec[key]
        elapsed+=step; steps+=1
        if steps>max_substeps: raise RuntimeError('max_substeps exceeded')
    return {'species':n,'photons':p,'substeps':steps,'mode':mode,**acc}
