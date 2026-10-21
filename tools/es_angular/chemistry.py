"""Local five-species photoionization chemistry for angular allocation.

Species are HI, HII, HeI, HeII, HeIII; the three capture channels are
HI -> HII, HeI -> HeII, and HeII -> HeIII. Cross sections are in cm^2.
The exact frozen-rate maps return species and integrated capture counts.

The numerical formulas were extracted without arithmetic changes from the
project's former es_hhe_reference.py, SHA-256
4075d324a8e965ecc317bb4a4c85033b5d41cedb1a8385ca52b52c270d3cd542.
HI/HeII use hydrogenic cross sections and HeI uses Verner et al. (1996).
This module contains no excursion-set allocation, sphere selection,
transport, recombination, or native-backend dependency.
"""

import numpy as np

def cross_sections(e):
    """Hydrogenic HI/HeII and Verner et al. 1996 HeI, cm^2."""
    e = np.asarray(e, dtype=float)
    chi = [13.6, 24.59, 54.4]
    result = np.zeros(e.shape+(3,))
    for a, z in [(0, 1), (2, 2)]:
        ratio = np.maximum(e/chi[a], 1.)
        eps = np.maximum(np.sqrt(ratio-1), 1e-8)
        shape = np.exp(4-4*np.arctan(eps)/eps)/(-np.expm1(-2*np.pi/eps))
        result[..., a] = np.where(e >= chi[a], 6.3e-18/z**2*ratio**-4*shape, 0.)
    x = e/13.61-.4434
    y = np.sqrt(x*x+2.136**2)
    he = 9.492e-16*((x-1)**2+2.039**2)*y**(.5*3.188-5.5)*(1+np.sqrt(y/1.469))**-3.188
    result[..., 1] = np.where(e >= chi[1], he, 0.)
    return result

def cascade_probability(x, y):
    """Stable probability of completing two successive exponential waits.

    For x=a*t and y=b*t, the usual expression loss(x)-P(HeII) loses
    the entire two-photon term when one rate is much smaller than the other.
    Write s=min(x,y), d=abs(x-y), and use the nonnegative identity
      C = [1-(1+s)exp(-s)] + s*exp(-s)*[1-(1-exp(-d))/d].
    The square brackets are evaluated by series near zero, preserving C=0
    exactly when either rate is zero. This also preserves the xy/2 limit.
    """
    s, d = min(x,y), abs(x-y)
    if s == 0:
        return 0.
    if s < .5:
        term = .5*s*s
        first = term
        for k in range(3,20):
            term *= -s*(k-1)/(k*(k-2))
            first += term
            if abs(term) <= np.finfo(float).eps*first:
                break
    else:
        first = -np.expm1(-s)-s*np.exp(-s) if s < 700 else 1.
    if d == 0:
        second = 0.
    elif d < .5:
        term = .5*d
        second = term
        for k in range(2,20):
            term *= -d/(k+1)
            second += term
            if abs(term) <= np.finfo(float).eps*second:
                break
    else:
        second = 1.+np.expm1(-d)/d
    return first+(s*np.exp(-s)*second if s < 700 else 0.)

def upward_map(rate, exposure):
    """Exact positive HI->HII and HeI->HeII->HeIII map at frozen rates.

    Also returns integrated transition counts per initial species. No bound
    on an absent species' decay rate is necessary; newly produced HeII may
    be ionized within the same analytic step.
    """
    h,a,b = rate
    h_loss = -np.expm1(-h*exposure)
    i_loss = -np.expm1(-a*exposure)
    ii_loss = -np.expm1(-b*exposure)
    if a==b:
        z = a*exposure
        to_ii = z*np.exp(-z) if z<700 else 0.
    elif b>a:
        to_ii = a*np.exp(-a*exposure)*(-np.expm1(-(b-a)*exposure))/(b-a)
    else:
        to_ii = a*np.exp(-b*exposure)*(-np.expm1(-(a-b)*exposure))/(a-b)
    # Evaluate both helium branches independently and preserve the smaller
    # tail; subtracting a nearly complete HeII branch from total HeI loss can
    # otherwise invent HeIII even when its ionizing flux is zero or tiny.
    to_ii = min(max(float(to_ii),0.),float(i_loss))
    to_iii = min(max(cascade_probability(a*exposure,b*exposure),0.),float(i_loss))
    if to_iii <= to_ii:
        to_ii = i_loss-to_iii
    else:
        to_iii = i_loss-to_ii
    m = np.eye(5)
    m[0,0],m[1,0] = 1-h_loss,h_loss
    m[2,2],m[3,2],m[4,2] = 1-i_loss,to_ii,to_iii
    m[3,3],m[4,3] = 1-ii_loss,ii_loss
    capture = np.zeros((3,5))
    capture[0,0],capture[1,2] = h_loss,i_loss
    capture[2,2],capture[2,3] = to_iii,ii_loss
    return m,capture

def upward_cells(n,rate,exposure):
    """Vector exact upward_map; exposure is scalar or one value per row."""
    t=np.asarray(exposure)
    x=rate*(t[:,None] if t.ndim else t)
    loss=-np.expm1(-x);a,b=x[:,1],x[:,2]
    d=np.abs(a-b);s=np.minimum(a,b)
    ratio=np.divide(-np.expm1(-d),d,out=np.ones_like(d),where=d>0)
    to_ii=a*np.exp(-s)*ratio
    first=-np.expm1(-s)-s*np.exp(-s)
    mask=(s>0)&(s<.5);z=s[mask];term=.5*z*z;total=term.copy()
    for k in range(3,20):
        term*=-z*(k-1)/(k*(k-2));total+=term
        if np.all(abs(term)<=np.finfo(float).eps*abs(total)):break
    first[mask]=total
    second=1.-ratio;mask=(d>0)&(d<.5);z=d[mask];term=.5*z;total=term.copy()
    for k in range(2,20):
        term*=-z/(k+1);total+=term
        if np.all(abs(term)<=np.finfo(float).eps*abs(total)):break
    second[mask]=total
    to_iii=first+s*np.exp(-s)*second
    to_ii=np.minimum(np.maximum(to_ii,0.),loss[:,1])
    to_iii=np.minimum(np.maximum(to_iii,0.),loss[:,1])
    smaller=to_iii<=to_ii
    to_ii=np.where(smaller,loss[:,1]-to_iii,to_ii)
    to_iii=np.where(smaller,to_iii,loss[:,1]-to_ii)
    counts=np.column_stack((n[:,0]*loss[:,0],n[:,2]*loss[:,1],
                           n[:,2]*to_iii+n[:,3]*loss[:,2]))
    new=np.empty_like(n)
    new[:,0]=n[:,0]*(1.-loss[:,0]);new[:,1]=n[:,1]+counts[:,0]
    new[:,2]=n[:,2]*(1.-loss[:,1])
    new[:,3]=n[:,3]*(1.-loss[:,2])+n[:,2]*to_ii
    new[:,4]=n[:,4]+counts[:,2]
    return new,counts

