#!/usr/bin/env python3
"""Photon-conserving, multi-frequency H/He excursion-set reference.

Fixed grid-centred spherical ES candidates; shared budgets and local coupled
photo-chemistry; case-A recombination plus local coupled OTS at fixed 10000 K.
No ray tracing or spatial/angular radiation-transfer equation is solved.

Optimizations preserve the specified ES model: necessary-count rejection,
bounded batched candidate chemistry, implicit top-hat operators, cached FFTs,
three-reaction-channel contractions and exact single-band single-sphere maps.
The optional radius catalogue remains an explicit resolution parameter.

Requires Python 3 and NumPy. Run --validate-physics, --validate-history,
or --validate --validation-workers 5. ES_performance.py compares the same
fixed-centre, photon-depleting model with optimization switches disabled.
Independent ODE checks additionally use SciPy.

Primary 18/30/60 eV groups are test inputs. Sphere mixing, donor partition,
fixed temperature and unresolved-cell chemistry are model assumptions.
This research code is not the deployed Meraxes/Gadi executable.
"""

# Original research module: helium/es_ots_counts.py
"""Isothermal H/He count closure for an ES allocator; no spatial transport.

Species densities: HI,HII,HeI,HeII,HeIII in physical cm^-3.  This implements
positive case-A recombination followed by local coupled-OTS reabsorption.
Eight recombination channels follow Friedrich et al. (2012), Tables 1/2,
with p=0.96, v=0.285, l=1.425, m=0.737 and f_Lya=1 at T=10^4 K.
The continuum channels are threshold-count approximations, NOT full spectra
or energy moments. The model is isothermal, not thermodynamically closed.

Unabsorbed diffuse counts persist by channel. The ES layer may pool them with
other photons in the same accepted region; no photon is discarded here.
"""
from dataclasses import dataclass
import time
import numpy as np
from native_backend import get_backend, rfftn, irfftn

MYR = 365.25 * 86400 * 1e6
CHANNEL_NAMES = ('H_ground', 'HeII_ground', 'HeII_excited_H',
                 'HeIII_ground', 'HeIII_n2', 'HeIII_Lya',
                 'HeIII_2gamma_low', 'HeIII_2gamma_high')
CHANNEL_BIN = np.array([0, 1, 0, 2, 0, 1, 0, 1])
# These are opacity evaluation energies, not emitted mean photon energies.
OPACITY_E_EV = np.array([13.6, 24.59, 13.6, 54.4, 13.6, 40.8, 13.6, 24.59])
ABSORBER = np.array([0, 2, 3])
ION = np.array([1, 3, 4])


def coefficients_1e4():
    """Hui & Gnedin 1997 A/B fits, Friedrich+2012 HeIII n=2 fit."""
    lam = 2 * np.array([157807., 285335., 631515.]) / 1e4
    fa = lambda x: 1.269e-13*x**1.503/(1+(x/.522)**.470)**1.923
    fb = lambda x: 2.753e-14*x**1.5/(1+(x/2.740)**.407)**2.242
    a = np.array([fa(lam[0]), 3e-14*lam[1]**.654, 2*fa(lam[2])])
    b = np.array([fb(lam[0]), 1.26e-14*lam[1]**.750, 2*fb(lam[2])])
    return a, b, 3.4e-13


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


def electron_density(n):
    return n[1]+n[3]+2*n[4]


def nuclei(n):
    return np.array([n[0]+n[1], n[2]+n[3]+n[4]])


def capacities(n):
    """Currently available HI, HeI and HeII targets, all nonnegative.

    Absorbing one photon in HeI moves that atom to HeII and hence unlocks
    one further HeII target. These are NOT three independently duplicable
    inventories and do NOT require all helium to become HeIII.
    """
    return np.asarray(n)[ABSORBER].copy()


def apply_ionizations(n, absorbed):
    a, b, c = absorbed
    n += np.array([-a, a, -b, b-c, c])


def emit_from_captures(capture):
    a, b, n2 = coefficients_1e4()
    ground = capture*(a-b)/a
    excited = capture*b/a
    p, v, l, m = .96, .285, 1.425, .737
    return np.array([ground[0], ground[1], p*excited[1], ground[2],
                     capture[2]*n2/a[2], (1-v)*excited[2],
                     v*(l-m)*excited[2], v*m*excited[2]])


def _two_stage_decay_probabilities(x, y):
    """Positive survival/intermediate/completion for two exponential waits.

    x and y are nonnegative rate-times. Evaluate the completion probability
    without subtracting the intermediate population from the first decay:
      C = [1-(1+s) exp(-s)] + s exp(-s) [1-(1-exp(-d))/d],
    where s=min(x,y), d=abs(x-y). Series preserve the xy/2 limit even when
    both rates are tiny. Infinite rate-times represent analytic saturation.
    """
    x, y = float(x), float(y)
    if min(x, y) < 0 or np.isnan(x) or np.isnan(y):
        raise ValueError('Nonnegative rate-times required')
    if x == 0:
        return 1., 0., 0.
    survival = np.exp(-x)
    loss = -np.expm1(-x)
    if y == 0:
        return survival, loss, 0.
    if np.isinf(x):
        return 0., np.exp(-y), -np.expm1(-y)
    if np.isinf(y):
        return survival, 0., loss
    s, d = min(x, y), abs(x-y)
    decay = np.exp(-s)
    if s > 500:
        log_middle = np.log(x)-s
        if d:
            log_middle += np.log(-np.expm1(-d))-np.log(d)
        middle = np.exp(log_middle)
    else:
        middle = x*decay*(-np.expm1(-d)/d if d else 1.)
    if s < .5:
        term = .5*s*s
        first = term
        for k in range(3, 20):
            term *= -s*(k-1)/(k*(k-2))
            first += term
            if abs(term) <= np.finfo(float).eps*first:
                break
    else:
        first = -np.expm1(-s)-s*decay
    if d == 0:
        second = 0.
    elif d < .5:
        term = .5*d
        second = term
        for k in range(2, 20):
            term *= -d/(k+1)
            second += term
            if abs(term) <= np.finfo(float).eps*second:
                break
    else:
        second = 1.+np.expm1(-d)/d
    complete = first+s*decay*second
    middle = min(max(float(middle), 0.), float(loss))
    complete = min(max(float(complete), 0.), float(loss))
    # Preserve the smaller branch directly; derive the larger from total loss.
    if complete <= middle:
        middle = loss-complete
    else:
        complete = loss-middle
    return survival, middle, complete


def recombine(n, dt):
    """Exact positive downward chains with frozen electron density.

    Recombination events are evaluated directly from positive transition
    probabilities, including the two-capture HeIII -> HeII -> HeI branch.
    """
    n = np.asarray(n, dtype=float)
    if n.shape != (5,) or np.any(~np.isfinite(n)) or np.any(n < 0):
        raise ValueError('Five finite nonnegative species counts required')
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('Finite nonnegative time required')
    a, _, _ = coefficients_1e4()
    lam = a*electron_density(n)
    with np.errstate(over='ignore'):
        h, y, x = lam*dt
    out = n.copy()
    capture = np.zeros(3)
    capture[0] = n[1]*(-np.expm1(-h))
    out[0] += capture[0]
    out[1] = n[1]*np.exp(-h)
    survival, middle, complete = _two_stage_decay_probabilities(x, y)
    capture[2] = n[4]*(-np.expm1(-x))
    capture[1] = n[3]*(-np.expm1(-y))+n[4]*complete
    out[2] += capture[1]
    out[3] = n[3]*np.exp(-y)+n[4]*middle
    out[4] = n[4]*survival
    return out, capture


def absorb_local(n, photons, tolerance=1e-12, max_iter=20000, donor_fraction=.005,
                 photon_floor=0.):
    """Infinite-fast local absorption closure, limited by available atoms.

    Simultaneous competing hazards prevent different spectral channels from
    using an atom twice. Relative opacities evolve after each positive step.
    tolerance stops iteration but RETAINS every residual photon in photons.
    No distances, directions or intercell transfer enter this routine.
    """
    n = np.array(n, dtype=float, copy=True)
    photons = np.array(photons, dtype=float, copy=True)
    if (n.shape != (5,) or photons.shape != (8,)
            or np.any(~np.isfinite(n)) or np.any(~np.isfinite(photons))
            or np.any(n < 0) or np.any(photons < 0)):
        raise ValueError('Finite nonnegative species and diffuse counts required')
    if (not np.isfinite(tolerance) or not 0 < tolerance < 1
            or not np.isfinite(donor_fraction) or not 0 < donor_fraction <= 1
            or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0
            or not np.isfinite(photon_floor) or photon_floor < 0):
        raise ValueError('Invalid local absorption convergence controls')
    absorbed = np.zeros((8, 3))
    cross = cross_sections(OPACITY_E_EV)
    # Residual cutoffs are local to each photon group / transition. Unusable
    # photons or fully ionized spectator nuclei cannot relax another band's
    # accuracy. Include HeI in the reachable HeII capacity because this
    # upward chain creates new HeII while absorbing photons.
    reachable = np.array([n[0], n[2], n[2]+n[3]])
    usable = cross > 0
    photon_tolerance = np.maximum(photon_floor, tolerance*np.minimum(photons, usable@reachable))
    atom_tolerance = tolerance*np.minimum(reachable, photons@usable)
    for iteration in range(max_iter):
        # Keep negligible residual atoms and photons in the returned state,
        # but omit them from numerical hazards. Otherwise an arbitrarily
        # small neutral tail in a huge soft field can impose tiny steps on
        # an unrelated finite HeII reaction indefinitely.
        relevant_atoms = n[ABSORBER] > atom_tolerance
        relevant_photons = photons > photon_tolerance
        op = n[ABSORBER][None, :]*cross*relevant_atoms[None, :]
        hazard = op.sum(axis=1)
        active = (hazard > 0)&relevant_photons
        if not np.any(active):
            return n, photons, absorbed, iteration
        op = op*relevant_photons[:, None]
        hazard = op.sum(axis=1)
        rates = photons[:, None]*op
        lossrate = rates.sum(axis=0)
        donors = lossrate > 0
        exposure = min(100/hazard.max(), np.min(donor_fraction*n[ABSORBER][donors]/lossrate[donors]))
        lost = photons*(-np.expm1(-hazard*exposure))
        share = np.divide(op, hazard[:, None], out=np.zeros_like(op), where=hazard[:, None] > 0)
        absorbed_step = lost[:, None]*share
        apply_ionizations(n, absorbed_step.sum(axis=0))
        photons -= lost
        absorbed += absorbed_step
        if np.any(n < -1e-30) or np.any(photons < -1e-30):
            raise RuntimeError('Positive update failed')
    raise RuntimeError('Local OTS absorption did not converge; no counts discarded')


@dataclass
class Ledger:
    captures: np.ndarray
    emitted: np.ndarray
    absorbed: np.ndarray
    absorption_iterations: int = 0
    steps: int = 0


def evolve_dark(n, duration, max_step, pending=None, absorption_donor_fraction=.005,
                photon_floor=0.):
    """Source-off evolution at fixed density, fixed T=10^4 K.

    Returns species, retained diffuse photons, ledger. Lie splitting error
    is controlled by max_step; no common H/He ionized fraction is imposed.
    """
    n = np.array(n, dtype=float)
    pending = np.zeros(8) if pending is None else np.array(pending, dtype=float)
    if (n.shape != (5,) or pending.shape != (8,)
            or np.any(~np.isfinite(n)) or np.any(~np.isfinite(pending))
            or np.any(n < 0) or np.any(pending < 0)):
        raise ValueError('Finite nonnegative species and diffuse counts required')
    if (not np.isfinite(duration) or not np.isfinite(max_step)
            or duration < 0 or max_step <= 0):
        raise ValueError('Invalid time')
    if (not np.isfinite(absorption_donor_fraction) or not 0 < absorption_donor_fraction <= 1
            or not np.isfinite(photon_floor) or photon_floor < 0):
        raise ValueError('Invalid absorption donor fraction')
    ledger = Ledger(np.zeros(3), np.zeros(8), np.zeros((8, 3)))
    elapsed = 0.
    while elapsed < duration:
        dt = min(max_step, duration-elapsed)
        n, rec = recombine(n, dt)
        emission = emit_from_captures(rec)
        n, pending, absorbed, iterations = absorb_local(n, pending+emission,
            donor_fraction=absorption_donor_fraction, photon_floor=photon_floor)
        ledger.captures += rec
        ledger.emitted += emission
        ledger.absorbed += absorbed
        ledger.absorption_iterations += iterations
        ledger.steps += 1
        elapsed += dt
    return n, pending, ledger


# Original research module: helium/es_spectral_barrier.py
"""Reaction-consistent, event-stopped H/He excursion-set candidate barrier.

The sphere is perfectly mixed in photon density. Chemical absorption competes
between all absorbers; no spatial radiation propagation is solved. The trial
stops when an initially unmet H/He target first reaches the declared ionized
fraction threshold, preserving all unused photons for the next ES operation.
"""
from dataclasses import dataclass
import numpy as np


def spectral_spend(demand, budget, tolerance=1e-12):
    """Feasible count allocation, highest threshold first, lowest usable band.

    demand=(HI,HeI,HeII transitions); budget=(13.6--24.59,24.59--54.4,>=54.4).
    This helper is bookkeeping only; it is NOT the competing-opacity model.
    Returns A[band,species] and unspent counts, or None when infeasible.
    """
    demand = np.asarray(demand, dtype=float)
    p = np.array(budget, dtype=float, copy=True)
    if demand.shape != (3,) or p.shape != (3,) or np.any(demand < 0) or np.any(p < 0):
        raise ValueError('Three nonnegative transition demands and band budgets required')
    # A large budget below a target's threshold cannot relax that target's
    # feasibility condition. Compare each spectral suffix to its own demand.
    if any(p[i:].sum()+tolerance*demand[i:].sum() < demand[i:].sum()
           for i in range(3)):
        return None
    a = np.zeros((3,3))
    for target in (2,1,0):
        remaining = demand[target]
        for band in range(target, 3):
            use = min(remaining, p[band])
            a[band,target] += use
            p[band] -= use
            remaining -= use
        if remaining > tolerance*demand[target]:
            return None
    return a, p


def target_remaining(n):
    n = np.asarray(n)
    total = n.sum(axis=0) if n.ndim == 2 else n
    return np.array([total[0], total[2], total[2]+total[3]])


def target_normalization(n):
    n = np.asarray(n)
    total = n.sum(axis=0) if n.ndim == 2 else n
    h, he = total[:2].sum(), total[2:].sum()
    return np.array([h, he, he])


def update_species(n, absorbed_per_cell):
    a, b, c = absorbed_per_cell.T
    return n+np.stack([-a,a,-b,b-c,c],axis=1)


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


@dataclass
class Trial:
    species: np.ndarray
    photons: np.ndarray
    absorbed_group_cell_species: np.ndarray
    events: tuple
    initially_eligible: np.ndarray
    exposure: float
    iterations: int
    passed: bool
    transfer_matrix: np.ndarray = None
    absorption_response: np.ndarray = None
    monochromatic_fluence: float = None
    exposure_kind: str = "midpoint_coordinate"


def apply_aggregate_trial(trial, original_cells):
    """Commit a sphere-total trial to its cells without rerunning chemistry.

    Returns per-cell species and the group/cell/transition absorption ledger.
    The transfer matrix and response are defined by the common-photon-density
    closure. Their application costs O(number of accepted cells).
    """
    original_cells = np.asarray(original_cells,dtype=float)
    return (original_cells@trial.transfer_matrix.T,
            np.einsum('gsa,ja->gjs',trial.absorption_response,original_cells))


@dataclass
class Crossing:
    radius: float
    cells: np.ndarray
    trial: Trial


def first_crossing(species, photons_per_cell, positions, center, radii,
                   box_length=None, **trial_options):
    """Actual descending-radius top-hat ES crossing with coupled chemistry.

    All radius trials read the same unspent inventories; rejected trials make
    no mutation. Commit the returned allocation exactly once. The caller must
    resolve overlapping accepted regions with a unique shared budget.
    """
    positions = np.asarray(positions,dtype=float)
    displacement = positions-positions[center]
    if box_length is not None:
        displacement -= box_length*np.rint(displacement/box_length)
    distance2 = np.einsum('ij,ij->i',displacement,displacement)
    for radius in sorted(radii,reverse=True):
        cells = np.flatnonzero(distance2<=float(radius)**2+1e-14)
        trial = mixed_es_trial(np.asarray(species)[cells],
                               np.asarray(photons_per_cell)[cells].sum(axis=0),
                               **trial_options)
        if trial.passed:
            return Crossing(float(radius),cells,trial)
    return None


def _single_band_endpoint(total, budget, rate, epsilon=1e-6,
                          enabled_targets=(True,True,True)):
    """Exact monochromatic trajectory, stopped by photons or an ES event.

    A single usable band fixes all H/He rate ratios. Photon depletion changes
    only the common exposure coordinate, so a scalar monotone root gives the
    same endpoint as the coupled gas/photon ODE without depletion substeps.
    Returns (mapping,capture,events,exposure,evaluations,status).
    """
    total=np.asarray(total,dtype=float)
    rate=np.asarray(rate,dtype=float)
    normal=target_normalization(total)
    threshold=epsilon*normal
    eligible=(np.asarray(enabled_targets,dtype=bool)&(normal>0)&
              (target_remaining(total)>threshold+64*np.finfo(float).eps*normal))
    identity=np.eye(5);zero=np.zeros((3,5))
    hazard=float(rate@total[ABSORBER])
    if budget<=0 or hazard<=0:
        return identity,zero,(),0.,0,'no_absorption'
    # Explicit asymptotic map avoids infinite floating-point exposure.
    limit=np.eye(5); cap=np.zeros((3,5))
    if rate[0]>0:
        limit[:,0]=[0,1,0,0,0];cap[0,0]=1.
    if rate[1]>0:
        limit[:,2]=[0,0,0,0,1] if rate[2]>0 else [0,0,0,1,0]
        cap[1,2]=1.;cap[2,2]=float(rate[2]>0)
    if rate[2]>0:
        limit[:,3]=[0,0,0,0,1];cap[2,3]=1.
    can_hit=eligible&(target_remaining(limit@total)<threshold)
    if float((cap@total).sum())<=budget and not np.any(can_hit):
        return limit,cap,(),None,1,'absorber_saturation'
    evaluations=0
    def evaluate(t):
        nonlocal evaluations
        evaluations+=1
        mapping,capture=upward_map(rate,t)
        hit=eligible&(target_remaining(mapping@total)<=threshold)
        return mapping,capture,float((capture@total).sum()),hit
    reachable=(total[ABSORBER]>0)|np.array([False,False,total[2]>0 and rate[1]>0])
    high=min(1./float(np.max(rate[reachable])),budget/hazard)
    if high<=0 or not np.isfinite(high):
        return identity,zero,(),0.,0,'numerical_remainder'
    low=0.;answer=evaluate(high)
    for _ in range(1024):
        if answer[2]>=budget or np.any(answer[3]):
            break
        low=high;high*=2.
        if not np.isfinite(high):
            raise RuntimeError('Monochromatic event bracket overflow')
        answer=evaluate(high)
    else:
        raise RuntimeError('Monochromatic event bracket failed')
    for _ in range(80):
        middle=low+.5*(high-low)
        if middle==low or middle==high:
            break
        test=evaluate(middle)
        if test[2]>=budget or np.any(test[3]):
            high=middle;answer=test
        else:
            low=middle
    if answer[2]<=budget:
        exposure=high
    else:
        exposure=low;answer=evaluate(low)
    mapping,capture,spent,hit=answer
    return mapping,capture,tuple(np.flatnonzero(hit)),exposure,evaluations,(
        'crossing' if np.any(hit) else 'photon_exhaustion')


def mixed_es_trial(species, photons, energies_eV=(18.,30.,60.), epsilon=1e-6,
                   enabled_targets=(True,True,True), donor_fraction=.02,
                   count_tolerance=1e-13, max_iterations=20000,
                   monochromatic_fastpath=True):
    """Positive midpoint gas/photon chemistry on one perfectly mixed ES sphere.

    The common local field permits solving five aggregate species while
    accumulating their linear transfer map; per-cell endpoints and absorption
    ledgers are constructed only once. Positive exact exponential chemistry
    at the predicted midpoint spectrum is second order in donor_fraction.
    Identical active cross-section vectors have an exact scalar exposure solution;
    original spectral accounts remain separate.
    No physical time or recombination is advanced by a radius trial.
    """
    original=np.array(species,dtype=float,copy=True)
    p=np.array(photons,dtype=float,copy=True)
    e=np.asarray(energies_eV,dtype=float)
    if original.ndim==1:
        original=original[None,:]
    enabled=np.asarray(enabled_targets,dtype=bool)
    if (original.ndim!=2 or original.shape[1]!=5 or p.ndim!=1 or e.shape!=p.shape
            or enabled.shape!=(3,) or np.any(original<0) or np.any(p<0)):
        raise ValueError('Nonnegative atoms(cells,5), photons(groups), and three targets required')
    if (not all(np.all(np.isfinite(a)) for a in (original,p,e)) or not 0<epsilon<1
            or not 0<donor_fraction<1 or not 0<count_tolerance<1
            or not isinstance(max_iterations,(int,np.integer)) or max_iterations<1):
        raise ValueError('Finite data and valid solver controls required')
    n=original.sum(axis=0)
    if not np.all(np.isfinite(n)):
        raise ValueError("Aggregate atom counts overflow binary64")
    normal=target_normalization(n)
    if not np.all(np.isfinite(normal)):
        raise ValueError("Atom normalization overflow binary64")
    threshold=epsilon*normal
    eligible=enabled&(normal>0)&(target_remaining(n)>threshold+64*np.finfo(float).eps*normal)
    transfer=np.eye(5);response=np.zeros((len(p),3,5));exposure_total=0.
    iterations=0;monochromatic_fluence=None;used_exact=False
    def finish(events=()):
        events=tuple(events)
        output=original@transfer.T
        record=np.einsum('gsa,ja->gjs',response,original)
        return Trial(output,p,record,tuple(events),eligible,exposure_total,
                     iterations,bool(events),transfer,response,
                     monochromatic_fluence,"normalized_monochromatic_fluence" if used_exact else "midpoint_coordinate")
    if not np.any(eligible):
        return finish()
    cross=cross_sections(e)
    if not np.any(cross>0):
        return finish()
    cross/=cross.max()
    capacity=(cross>0)@np.array([n[0],n[2],n[2]+n[3]])
    residual=count_tolerance*np.minimum(p,capacity)
    smallest=np.nextafter(0.,1.)
    for iteration in range(max_iterations):
        hazard=cross@n[ABSORBER]
        if not np.all(np.isfinite(hazard)):
            raise ValueError("Aggregate absorption hazard overflow binary64")
        photon_rates=p[:,None]*cross
        # Exact exposure uses count budgets directly and can resolve a
        # photon inventory even when its artificial half-step or p*sigma
        # rate underflows. Those two floors apply only to midpoint stepping.
        physical=(hazard>0)&(p>residual)
        physical_groups=np.flatnonzero(physical)
        exact_available=(monochromatic_fastpath and len(physical_groups)>0 and
                         np.all(cross[physical_groups]==cross[physical_groups[0]]))
        active=(physical if exact_available else
                physical&(.5*donor_fraction*p>=smallest)&np.any(photon_rates>0,axis=1))
        if not np.any(active):
            return finish()
        groups=np.flatnonzero(active)
        if monochromatic_fastpath and np.all(cross[groups]==cross[groups[0]]):
            # Equal opacity vectors have identical depletion fractions, even
            # when distinct primary/diffuse channels retain separate accounts.
            light=np.where(active,p,0.)
            available=float(light.sum())
            if not np.isfinite(available):
                raise ValueError('Usable photon total overflow binary64')
            mapping,capture,events,exposure,evals,status=_single_band_endpoint(
                n,available,cross[groups[0]],epsilon,enabled)
            spent=float((capture@n).sum())
            if spent>available:
                raise RuntimeError('Monochromatic trial exceeded photon budget')
            response+=(light/available)[:,None,None]*(capture@transfer)[None,:,:]
            transfer=mapping@transfer
            p-=light*(spent/available)
            iterations+=evals
            # The exact root is normalized fluence, not the midpoint
            # integration coordinate. Do not add unlike debug coordinates.
            used_exact=True;monochromatic_fluence=exposure;exposure_total=None
            return finish(events)
        photon_rates[~active]=0.
        rate=photon_rates.sum(axis=0)
        route0=np.divide(photon_rates,rate[None,:],out=np.zeros_like(cross),where=rate[None,:]>0)
        step=donor_fraction/float(np.max(hazard[active]))
        if not np.isfinite(step) or step<=0:
            return finish()
        def attempt(dt):
            _,half_capture=upward_map(rate,.5*dt)
            half_lost=route0@(half_capture@n)
            if np.any(half_lost>.5*donor_fraction*p):
                return None
            middle_p=np.where(active,p-half_lost,0.)
            middle_rates=middle_p[:,None]*cross
            middle_rate=middle_rates.sum(axis=0)
            route=np.divide(middle_rates,middle_rate[None,:],out=np.zeros_like(cross),where=middle_rate[None,:]>0)
            mapping,capture=upward_map(middle_rate,dt)
            counts=capture@n
            lost=route@counts
            if np.any(lost>donor_fraction*p):
                return None
            return mapping@n,lost,mapping,capture,route
        for _ in range(100):
            result=attempt(step)
            if result is not None:
                break
            step*=.5
        else:
            raise RuntimeError('No positive photon-limited midpoint step found')
        next_n,lost,mapping,capture,route=result
        if not np.any(lost>0) and np.array_equal(next_n,n):
            return finish()
        hit=eligible&(target_remaining(next_n)<=threshold)
        if np.any(hit):
            low,high=0.,step
            for _ in range(64):
                middle=low+.5*(high-low)
                if middle==low or middle==high:
                    break
                trial=attempt(middle)
                if trial is None:
                    raise RuntimeError('Midpoint event root lost feasibility')
                if np.any(eligible&(target_remaining(trial[0])<=threshold)):
                    high=middle
                else:
                    low=middle
            step=high
            next_n,lost,mapping,capture,route=attempt(step)
            hit=eligible&(target_remaining(next_n)<=threshold)
        response+=route[:,:,None]*(capture@transfer)[None,:,:]
        transfer=mapping@transfer
        n=next_n;p-=lost;exposure_total+=step;iterations+=1
        if np.any(n<0) or np.any(p<0) or not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
            raise RuntimeError('Nonpositive or nonfinite sphere chemistry update')
        if np.any(hit):
            return finish(np.flatnonzero(hit))
    raise RuntimeError('Sphere chemistry failed to converge; state not committed')

# Original research module: coupled/es_sphere_operator.py
"""Implicit periodic spherical ES operators with bounded linear storage.

Small grids use reproducible direct sums; larger grids use grouped FFTs.
Explicit prepared fields reuse forward transforms across filter radii. No
source-target distance or proposal-cell membership matrix is constructed.
High-contrast low-signal supports fall back to direct summation: this preserves
small signals but can increase run time. It does not increase storage order.
"""
from collections import OrderedDict
import math
import numpy as np


def _column_sum(values):
    x=np.asarray(values)
    if x.ndim==1:
        return float(math.fsum(x.tolist()))
    return np.array([math.fsum(x[:,j].tolist()) for j in range(x.shape[1])])


def geometric_radii2(side, ratio=1.2, max_radius=None):
    """Explicit logarithmic ES filter catalogue; changes require convergence tests.

    Zero, one cell (if reachable), and the requested maximum are included.
    This helper computes O(log(side)) numbers without allocating a grid.
    """
    if int(side)!=side or side<1 or not np.isfinite(ratio) or ratio<=1:
        raise ValueError('Positive integer side and finite ratio>1 required')
    furthest=np.sqrt(3.)*(int(side)//2)
    maximum=furthest if max_radius is None else min(float(max_radius),furthest)
    if not np.isfinite(maximum) or maximum<0:
        raise ValueError('Finite nonnegative maximum radius required')
    if maximum==0:
        return np.array([0.])
    values=[0.]
    radius=min(1.,maximum)
    values.append(radius)
    while radius*ratio<maximum:
        radius*=ratio
        values.append(radius)
    values.append(maximum)
    squared=np.asarray(values)**2
    if max_radius is None:
        squared[-1]=3*(int(side)//2)**2
    return np.unique(squared)


class PreparedSphereFields:
    """Forward transforms for one unchanged field snapshot, not a global cache."""
    def __init__(self, geometry, values):
        x=np.asarray(values,dtype=float)
        self.scalar=x.ndim==1
        self.x=x[:,None] if self.scalar else x
        if self.x.ndim!=2 or self.x.shape[0]!=geometry.size:
            raise ValueError('Expected one row per grid cell')
        if not np.all(np.isfinite(self.x)):
            raise ValueError('Sphere fields must be finite')
        self.geometry=geometry
        self.transforms=[]
        self.occupancy=[]
        self.constants=[]
        self.nonnegative=[]
        self.error_bounds=[]
        self.nonzero_indices=[]
        geometry.counters['field_preparations']+=1
        shape=(geometry.side,)*3
        for column in self.x.T:
            constant=bool(np.all(column==column[0]))
            nonnegative=bool(np.all(column>=0))
            self.constants.append(constant)
            self.nonnegative.append(nonnegative)
            # Exact zeros carry no signal. Retain every other value, including
            # subnormals, for sparse high-dynamic-range support sums.
            sparse=(not constant and np.count_nonzero(column)<=
                    min(65536,geometry.size//4))
            self.nonzero_indices.append(np.flatnonzero(column!=0) if sparse else None)
            self.error_bounds.append(64*np.finfo(float).eps*max(1.,np.log2(geometry.size))
                                     *float(np.max(np.abs(column))))
            if geometry.direct or constant:
                self.transforms.append(None)
                self.occupancy.append(None)
            else:
                self.transforms.append(rfftn(column.reshape(shape)))
                geometry.counters['forward_ffts']+=1
                present=column>0
                if nonnegative and not np.all(present):
                    self.occupancy.append(rfftn(present.reshape(shape)))
                    geometry.counters['forward_ffts']+=1
                else:
                    self.occupancy.append(None)


class PeriodicSphereGeometry:
    def __init__(self,side,radii2=None,direct_limit=512):
        self.side=int(side)
        if self.side<1 or self.side!=side:
            raise ValueError('Positive integer grid side required')
        self.size=self.side**3
        self.xyz=np.ascontiguousarray(np.indices((self.side,)*3).reshape(3,-1).T)
        delta=np.minimum(self.xyz,self.side-self.xyz)
        self.origin_distance2=np.sum(delta*delta,axis=1)
        self.radii2=(np.unique(self.origin_distance2) if radii2 is None
                     else np.unique(np.asarray(radii2,dtype=float)))
        if (not len(self.radii2) or np.any(~np.isfinite(self.radii2)) or
                np.any(self.radii2<0) or self.radii2[0]!=0):
            raise ValueError('Nonnegative radii-squared catalogue including zero required')
        self.direct=self.size<=direct_limit
        self._kernels=OrderedDict()
        self._offsets=OrderedDict()
        self.counters=dict(field_preparations=0,forward_ffts=0,inverse_ffts=0,
                           kernel_ffts=0,direct_sums=0,dynamic_range_fallbacks=0)

    def offsets(self,radius_index):
        k=int(radius_index)
        if k not in self._offsets:
            self._offsets[k]=self.xyz[self.origin_distance2<=self.radii2[k]]
            if len(self._offsets)>2:
                self._offsets.popitem(last=False)
        else:
            self._offsets.move_to_end(k)
        return self._offsets[k]

    def cells(self,center,radius_index):
        points=(self.offsets(radius_index)+self.xyz[int(center)])%self.side
        return (points[:,0]*self.side+points[:,1])*self.side+points[:,2]

    def kernel(self,radius_index):
        k=int(radius_index)
        if k in self._kernels:
            self._kernels.move_to_end(k)
            return self._kernels[k]
        mask=(self.origin_distance2<=self.radii2[k]).reshape((self.side,)*3)
        self._kernels[k]=(rfftn(mask),int(np.count_nonzero(mask)))
        self.counters['kernel_ffts']+=1
        if len(self._kernels)>2:
            self._kernels.popitem(last=False)
        return self._kernels[k]

    def prepare(self,values):
        return PreparedSphereFields(self,values)

    def sparse_sums(self,column,indices,rows,radius_index):
        """Exact weak-support sums over occupied cells, with bounded workspace.

        This is the same minimum-image membership and math.fsum as cells().
        It skips exact zeros, never a small nonzero photon account. Chunking
        avoids retaining a centre-by-source distance or membership matrix.
        """
        native=get_backend()
        if native is not None:
            return native.sparse_sums(self.side,self.xyz,indices,column,rows,self.radii2[radius_index])
        result=np.zeros(len(rows),dtype=float)
        if not len(indices):return result
        sources=self.xyz[indices]
        values=column[indices]
        chunk=max(1,65536//len(indices))
        for lo in range(0,len(rows),chunk):
            centres=self.xyz[rows[lo:lo+chunk]]
            delta=np.abs(centres[:,None,:]-sources[None,:,:])
            delta=np.minimum(delta,self.side-delta)
            inside=np.sum(delta*delta,axis=2)<=self.radii2[radius_index]
            result[lo:lo+len(centres)]=[math.fsum(values[hit].tolist())
                                      for hit in inside]
        return result

    def all_sums(self,values,radius_index):
        if self.radii2[radius_index]==0 and not isinstance(values,PreparedSphereFields):
            x=np.asarray(values,dtype=float)
            if x.ndim not in (1,2) or x.shape[0]!=self.size or not np.all(np.isfinite(x)):
                raise ValueError('Finite field with one row per grid cell required')
            return x.copy()
        prepared=values if isinstance(values,PreparedSphereFields) else self.prepare(values)
        if prepared.geometry is not self:
            raise ValueError('Prepared fields belong to a different geometry')
        x=prepared.x
        if self.radii2[radius_index]==0:
            result=x.copy()
        elif self.direct:
            result=np.array([_column_sum(x[self.cells(c,radius_index)])
                             for c in range(self.size)])
            self.counters['direct_sums']+=self.size*x.shape[1]
        else:
            kernel,volume=self.kernel(radius_index)
            if volume==self.size:
                result=np.broadcast_to(_column_sum(x),x.shape).copy()
            else:
                result=np.empty_like(x)
                shape=(self.side,)*3
                for j,column in enumerate(x.T):
                    if prepared.constants[j]:
                        result[:,j]=column[0]*volume
                        continue
                    result[:,j]=irfftn(prepared.transforms[j]*kernel,shape).ravel()
                    self.counters['inverse_ffts']+=1
                    occupied=None
                    if prepared.nonnegative[j]:
                        result[:,j]=np.maximum(result[:,j],0.)
                        if prepared.occupancy[j] is not None:
                            occupied=np.rint(irfftn(
                                prepared.occupancy[j]*kernel,shape).ravel())>0
                            self.counters['inverse_ffts']+=1
                            result[~occupied,j]=0.
                    # Conservative signal-to-roundoff guard. Directly sum
                    # weak supports rather than let a remote bright cell set
                    # their accuracy. Kernel support offsets cost O(volume).
                    bound=max(prepared.error_bounds[j],
                        4*np.finfo(float).eps*max(1.,np.log2(self.size))*
                        float(np.max(np.abs(result[:,j]))))
                    weak=np.abs(result[:,j])<=bound/1e-12
                    if occupied is not None:
                        weak &= occupied
                    rows=np.flatnonzero(weak)
                    indices=prepared.nonzero_indices[j]
                    if indices is not None and len(indices)<volume:
                        result[rows,j]=self.sparse_sums(column,indices,rows,radius_index)
                    elif len(rows) and get_backend() is not None:
                        result[rows,j]=get_backend().periodic_offset_sums(self.side,self.xyz,
                            column,rows,self.offsets(radius_index))
                    else:
                        for c in rows:
                            result[c,j]=_column_sum(column[self.cells(c,radius_index)])
                    self.counters['direct_sums']+=len(rows)
                    self.counters['dynamic_range_fallbacks']+=len(rows)
        return result[:,0] if prepared.scalar else result

    def operator(self,centres,radius_indices):
        return SphereOperator(self,centres,radius_indices)


class _SphereTranspose:
    def __init__(self,operator):
        self.operator=operator
        self.shape=operator.shape[::-1]
    def dot(self,values):
        return self.operator.scatter(values)
    def __matmul__(self,values):
        return self.dot(values)


class _CompactSphereOperator:
    """Exact sparse membership on the union of a small selected support set.

    Storage follows the number of actual memberships, with no sphere-by-cell
    dense matrix. Aggregate and scatter use the same memberships and accurate
    sums, so their adjoint relationship does not depend on FFT cancellation.
    """
    def __init__(self,cells,offsets):
        self.cell_indices,self.indices=np.unique(cells,return_inverse=True)
        self.offsets=np.asarray(offsets,dtype=np.int64)
        sizes=np.diff(self.offsets)
        self.shape=(len(sizes),len(self.cell_indices))
        self.volumes=sizes.astype(float)
        self.multiplicity=np.bincount(self.indices,minlength=self.shape[1]).astype(float)
        self.owners=np.repeat(np.arange(self.shape[0]),sizes)
        order=np.argsort(self.indices,kind='stable')
        self.scatter_owners=self.owners[order]
        self.scatter_offsets=np.r_[0,np.cumsum(self.multiplicity,dtype=np.int64)]
        self.single_cells=bool(np.all(sizes==1))
        self.disjoint=bool(np.all(self.multiplicity==1))
        self.T=_SphereTranspose(self)

    def __len__(self):return self.shape[0]

    def aggregate(self,values):
        x=np.asarray(values,dtype=float)
        if x.ndim not in (1,2) or x.shape[0]!=self.shape[1]:
            raise ValueError('Cell count does not match compact operator')
        if self.single_cells:return x[self.indices].copy()
        native=get_backend()
        if native is not None:
            return native.indexed_sum(x,self.indices,self.offsets)
        out=np.empty((len(self),)+x.shape[1:])
        for row,(start,end) in enumerate(zip(self.offsets[:-1],self.offsets[1:])):
            out[row]=_column_sum(x[self.indices[start:end]])
        return out

    def scatter(self,values):
        y=np.asarray(values,dtype=float)
        if y.ndim not in (1,2) or y.shape[0]!=len(self):
            raise ValueError('Sphere count does not match compact operator')
        out=np.empty((self.shape[1],)+y.shape[1:])
        if self.disjoint:
            out[self.indices]=y[self.owners]
            return out
        native=get_backend()
        if native is not None:
            return native.indexed_sum(y,self.scatter_owners,self.scatter_offsets)
        for cell,(start,end) in enumerate(zip(self.scatter_offsets[:-1],self.scatter_offsets[1:])):
            out[cell]=_column_sum(y[self.scatter_owners[start:end]])
        return out

    def sum(self,axis):
        if axis==0:return self.multiplicity
        if axis==1:return self.volumes
        raise ValueError('Specify sphere sum axis 0 or 1')
    def dot(self,values):return self.aggregate(values)
    def __matmul__(self,values):return self.aggregate(values)


class SphereOperator:
    """U@X and U.T@Y, with O(N+K) retained arrays instead of K-by-N storage."""
    is_sphere_operator=True
    def __init__(self,geometry,centres,radius_indices):
        self.geometry=geometry
        c=np.asarray(centres,dtype=int);r=np.asarray(radius_indices,dtype=int)
        if c.ndim!=1 or c.shape!=r.shape:
            raise ValueError('One centre/radius index per sphere required')
        if np.any((c<0)|(c>=geometry.size)) or np.any((r<0)|(r>=len(geometry.radii2))):
            raise ValueError('Sphere centre or radius out of range')
        maximum_distance2=float(geometry.origin_distance2.max())
        full=geometry.radii2[r]>=maximum_distance2
        if np.any(full):
            keep=~full
            keep[np.flatnonzero(full)[0]]=True
            c,r=c[keep],r[keep]
        pairs=np.column_stack((c,r))
        # Driver proposals are already unique and sorted. Avoid a global sort
        # in this normal all-grid-centre path; retain general small-test input.
        sorted_unique=(len(c)<2 or np.all(c[1:]>c[:-1]))
        if len(pairs) and not sorted_unique:
            _,keep=np.unique(pairs,axis=0,return_index=True)
            pairs=pairs[np.sort(keep)]
        self.centres=pairs[:,0];self.radius_indices=pairs[:,1]
        self.shape=(len(pairs),geometry.size)
        self.T=_SphereTranspose(self)
        if len(self) and np.all(self.radius_indices==self.radius_indices[0]):
            volume=np.count_nonzero(geometry.origin_distance2<=geometry.radii2[self.radius_indices[0]])
            self.volumes=np.full(len(self),volume,dtype=float)
        else:
            radii,inverse=np.unique(self.radius_indices,return_inverse=True)
            volume=np.array([np.count_nonzero(geometry.origin_distance2<=geometry.radii2[k])
                             for k in radii],dtype=float)
            self.volumes=volume[inverse]
        self._compact_operator=None
        self._compact_cells=None
        # Selected early-time supports occupy a small part of the volume.
        # Retain their exact index lists only while storage is bounded both
        # absolutely and by the already allocated grid. Large supports keep
        # the implicit FFT implementation and its linear storage bound.
        entries=int(self.volumes.sum())
        if (len(self) and entries<=min(1000000,geometry.size) and
                not (len(self)==1 and self.volumes[0]==geometry.size)):
            offsets=np.r_[0,np.cumsum(self.volumes,dtype=np.int64)]
            cells=np.empty(entries,dtype=np.int64)
            for row,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
                cells[start:end]=geometry.cells(self.centres[row],self.radius_indices[row])
            compact=_CompactSphereOperator(cells,offsets)
            # Many overlapping cells are faster with grouped FFTs than with
            # Python accurate sums. Disjoint supports scatter by indexing and
            # remain efficient even when their union covers most of the grid.
            if compact.disjoint or len(compact.cell_indices)<=geometry.size//16:
                self._compact_operator=compact
                self._compact_cells=compact.cell_indices
                self.multiplicity=np.zeros(self.shape[1])
                self.multiplicity[self._compact_cells]=compact.multiplicity
        if self._compact_operator is None:
            self.multiplicity=np.rint(self.scatter(np.ones(len(self))))

    def __len__(self):
        return self.shape[0]

    def compact(self):
        """Return covered grid indices and an exact operator on those cells."""
        if self._compact_operator is None or len(self._compact_cells)==self.shape[1]:
            return None
        return self._compact_cells,self._compact_operator

    def aggregate(self,values):
        x=np.asarray(values,dtype=float)
        if x.shape[0]!=self.shape[1]:
            raise ValueError('Cell count does not match operator')
        out=np.empty((len(self),)+x.shape[1:])
        if np.all(self.geometry.radii2[self.radius_indices]==0):
            return x[self.centres].copy()
        if len(self)==1 and self.volumes[0]==self.shape[1]:
            return np.asarray(_column_sum(x))[None,...]
        if self._compact_operator is not None:
            return self._compact_operator.aggregate(x[self._compact_cells])
        prepared=None if self.geometry.direct else self.geometry.prepare(x)
        for k in np.unique(self.radius_indices):
            rows=np.flatnonzero(self.radius_indices==k)
            if self.geometry.direct:
                for row in rows:
                    out[row]=_column_sum(x[self.geometry.cells(self.centres[row],int(k))])
                self.geometry.counters['direct_sums']+=len(rows)*(1 if x.ndim==1 else x.shape[1])
            else:
                out[rows]=self.geometry.all_sums(prepared,int(k))[self.centres[rows]]
        return out

    def scatter(self,values):
        y=np.asarray(values,dtype=float)
        if y.shape[0]!=len(self):
            raise ValueError('Sphere count does not match operator')
        out=np.zeros((self.shape[1],)+y.shape[1:])
        if np.all(self.geometry.radii2[self.radius_indices]==0):
            out[self.centres]=y
            return out
        if len(self)==1 and self.volumes[0]==self.shape[1]:
            return np.broadcast_to(y[0],out.shape).copy()
        if self._compact_operator is not None:
            out[self._compact_cells]=self._compact_operator.scatter(y)
            return out
        if self.geometry.direct:
            for cell in range(self.shape[1]):
                delta=np.abs(self.geometry.xyz[self.centres]-self.geometry.xyz[cell])
                delta=np.minimum(delta,self.geometry.side-delta)
                hit=np.sum(delta*delta,axis=1)<=self.geometry.radii2[self.radius_indices]
                out[cell]=_column_sum(y[hit])
            self.geometry.counters['direct_sums']+=self.shape[1]*(1 if y.ndim==1 else y.shape[1])
            return out
        for k in np.unique(self.radius_indices):
            rows=np.flatnonzero(self.radius_indices==k)
            seeds=np.zeros_like(out)
            seeds[self.centres[rows]]=y[rows]
            out+=self.geometry.all_sums(seeds,int(k))
        return out

    def sum(self,axis):
        if axis==0:return self.multiplicity
        if axis==1:return self.volumes
        raise ValueError('Specify sphere sum axis 0 or 1')
    def dot(self,values):return self.aggregate(values)
    def __matmul__(self,values):return self.aggregate(values)

# Original research module: coupled/es_joint_exposure.py
"""Positive photon-depleting chemistry on fixed excursion-set supports.

Only local chemical ODEs are integrated. Accepted spherical supports define
an averaging operator; no spatial or angular radiation equation is solved.
"""
from pathlib import Path
import sys
import numpy as np


class _DenseSphereOperator:
    """Small-test adapter; large grids supply an implicit sphere operator."""
    def __init__(self,membership):
        u=np.asarray(membership)
        if u.ndim!=2 or not np.all((u==0)|(u==1)):
            raise ValueError('Sphere membership must be Boolean')
        if len(u):
            _,unique=np.unique(u,axis=0,return_index=True)
            u=u[np.sort(unique)];u=u[u.sum(axis=1)>0]
        self.u=np.asarray(u,dtype=float);self.shape=self.u.shape
        self.volumes=self.u.sum(axis=1);self.multiplicity=self.u.sum(axis=0)
    def aggregate(self,x):
        return self.u@x
    def scatter(self,x):
        return self.u.T@x


def _upward_cells(n,rate,exposure):
    native=get_backend()
    if native is not None:
        return native.upward_cells(n,rate,exposure)
    return _upward_cells_numpy(n,rate,exposure)


def _upward_cells_numpy(n,rate,exposure):
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


def _disjoint_hydrogen_endpoint(initial,p0,cross,op,epsilon,residual):
    """Exact no-crossing H endpoint for independent one-cell supports.

    Photon-limited cells consume every active account regardless of opacity.
    Equal H opacities preserve account ratios in atom-limited cells. The
    photon total and HI have the same loss, so either asymptotic endpoint
    spends min(HI, photons). This closes the existing reaction ODE;
    it does not impose a floor on a species or discard a photon account.
    Different cells may use different opacities because they do not overlap.
    """
    if not (getattr(op,'single_cells',False) and getattr(op,'disjoint',False)):
        return None
    covered=np.asarray(op.multiplicity)>0
    potential=(p0>residual)&covered[:,None]
    # Preserve the common H/He path whenever a physically active channel can
    # ionize He. An inactive or completely nonabsorbing account stays intact.
    if np.any(potential&np.any(cross[:,1:]>0,axis=1)[None,:]):
        return None
    active=potential&(initial[:,0,None]>0)&(cross[:,0]>0)[None,:]
    live=np.any(active,axis=1)
    if not np.any(live):
        return None
    light=np.where(active,p0,0.)
    available=np.array([math.fsum(row.tolist()) for row in light])
    normal=initial[:,:2].sum(axis=1)
    if np.any(~np.isfinite(available)) or np.any(~np.isfinite(normal)):
        return None
    neutral=initial[:,0]
    atom_limited=live&(available>neutral)
    first=np.argmax(active,axis=1)
    opacity=cross[first,0]
    if np.any(atom_limited[:,None]&active&(cross[:,0][None,:]!=opacity[:,None])):
        return None
    threshold=epsilon*normal
    eligible=(normal>0)&(neutral>threshold+64*np.finfo(float).eps*normal)
    needed=neutral-threshold
    margin=64*np.finfo(float).eps*np.maximum.reduce((available,neutral,normal))
    # Every initially unmet H target must remain strictly unreachable, with
    # the same roundoff allowance as the ES candidate decision. Otherwise
    # the earliest event must still stop the shared integration.
    if np.any(live&eligible&~(available+margin<needed)):
        return None
    amount=np.minimum(neutral,available)
    amount[~live]=0.
    fraction=np.divide(amount,available,out=np.zeros_like(amount),where=available>0)
    debit=light.copy()
    debit[atom_limited]=light[atom_limited]*fraction[atom_limited,None]
    # Dividing HI by a vastly larger photon reservoir can underflow even
    # though the individual capture is representable. Form the normalized
    # account share first in just those cells.
    underflow=atom_limited&(fraction==0)&(amount>0)
    debit[underflow]=(light[underflow]/available[underflow,None])*amount[underflow,None]
    state=initial.copy()
    state[:,0]-=amount
    state[:,1]+=amount
    photons=p0-debit
    absorbed=np.zeros((len(initial),3));absorbed[:,0]=amount
    # All changes share the same capture count. The public finish() audits
    # the small floating-point partition error when duplicate bands exist.
    reason='absorber_saturation' if np.any(atom_limited) else 'photon_exhaustion'
    return dict(species=state,photons=photons,absorbed=absorbed,debit=debit,
                stop=reason)


def joint_exposure_update(gas,photons,energies,membership,epsilon=1e-5,
                          donor_fraction=.02,count_tolerance=1e-13,
                          max_iterations=20000,monochromatic_fastpath=True,
                          photon_floor=0.):
    """Common ES field with positive exponential-midpoint photon depletion.

    membership is Boolean(sphere,cell), or has shape,volumes,multiplicity,
    aggregate(X)=U@X,scatter(Y)=U.T@Y. Counts are per cell. Each donor is shared
    equally among selected supports containing it. The same actual captures
    update atoms and debit photon accounts. All unspent counts are retained.

    Spectral contraction precedes the spatial operator: each aggregate/scatter
    carries three reaction channels, independent of the number of photon bands.
    Backtracking reuses the current-rate field. A single support illuminated
    by identical active opacity vectors has an exact exposure endpoint;
    individual photon channels and donor accounts remain separate.
    """
    initial=np.array(gas,dtype=float,copy=True)
    p0=np.array(photons,dtype=float,copy=True);e=np.asarray(energies,dtype=float)
    op=membership if hasattr(membership,'aggregate') and hasattr(membership,'scatter') else _DenseSphereOperator(membership)
    if (initial.ndim!=2 or initial.shape[1]!=5 or p0.ndim!=2 or p0.shape[0]!=len(initial)
            or e.shape!=(p0.shape[1],) or len(op.shape)!=2 or op.shape[1]!=len(initial)):
        raise ValueError('Invalid gas/photon/sphere shapes')
    if (not all(np.all(np.isfinite(a)) for a in (initial,p0,e)) or np.any(initial<0)
            or np.any(p0<0) or not 0<epsilon<1 or not 0<donor_fraction<1
            or not 0<count_tolerance<1 or not isinstance(max_iterations,(int,np.integer))
            or max_iterations<1 or not np.isfinite(photon_floor) or photon_floor<0):
        raise ValueError('Finite nonnegative states and valid controls required')
    n=initial.copy();p=p0.copy();absorbed=np.zeros((len(n),3));debit_total=np.zeros_like(p)
    volume=np.asarray(op.volumes,dtype=float);multiplicity=np.asarray(op.multiplicity,dtype=float)
    if (volume.shape!=(op.shape[0],) or multiplicity.shape!=(len(n),) or
            np.any(volume<=0) or np.any(multiplicity<0) or
            not np.all(np.isfinite(volume)) or not np.all(np.isfinite(multiplicity))):
        raise ValueError('Invalid sphere volumes or multiplicities')
    # Small selected supports touch only a small fraction of the box. Evolve
    # exactly those cells, using the same membership and donor multiplicities.
    # All input cells were validated above; uncovered accounts stay unchanged.
    compact=op.compact() if hasattr(op,'compact') else None
    if compact is not None:
        cells,restricted=compact
        if len(cells)<len(n):
            result=joint_exposure_update(initial[cells],p0[cells],e,restricted,
                epsilon=epsilon,donor_fraction=donor_fraction,count_tolerance=count_tolerance,
                max_iterations=max_iterations,monochromatic_fastpath=monochromatic_fastpath,
                photon_floor=photon_floor)
            n[cells]=result['species'];p[cells]=result['photons']
            absorbed[cells]=result['absorbed'];debit_total[cells]=result['debit']
            result.update(species=n,photons=p,absorbed=absorbed,debit=debit_total)
            result['absorbed_count']=float(absorbed.sum())
            result['photon_error']=float(debit_total.sum()-absorbed.sum())
            return result
    inverse=np.divide(1.,multiplicity,out=np.zeros_like(multiplicity),where=multiplicity>0)
    cross=cross_sections(e);scale=float(cross.max()) if cross.size else 0.
    if scale:cross/=scale
    sphere_initial=op.aggregate(n)
    normal=np.column_stack((sphere_initial[:,:2].sum(1),sphere_initial[:,2:].sum(1),sphere_initial[:,2:].sum(1)))
    if not np.all(np.isfinite(normal)):
        raise ValueError("Sphere atom totals overflow binary64")
    threshold=epsilon*normal
    initial_remaining=np.column_stack((sphere_initial[:,0],sphere_initial[:,2],sphere_initial[:,2]+sphere_initial[:,3]))
    eligible=(normal>0)&(initial_remaining>threshold+64*np.finfo(float).eps*normal)
    reachable=np.column_stack((n[:,0],n[:,2],n[:,2]+n[:,3]))
    # Conservative reachable counts set only per-account numerical tolerances.
    donor_capacity=(op.scatter(op.aggregate(reachable))@(cross>0).T)*inverse[:,None]
    # Absolute activity floor is in the same count units as each account.
    # Dormant photons remain in p and the ledger; they never constrain steps.
    residual=np.maximum(photon_floor,count_tolerance*np.minimum(p0,donor_capacity))
    total_exposure=0.;evaluations=0;accepted_steps=0;events=[];reason='no_absorption'
    monochromatic_fluence=None;used_exact=False;endpoint_kind=None
    smallest=np.nextafter(0.,1.)

    def finish():
        amount=float(absorbed.sum());photon_error=float(debit_total.sum()-amount)
        if abs(photon_error)>4e-12*max(amount,np.finfo(float).tiny):
            raise RuntimeError('Joint photon accounting failed')
        before=np.column_stack((initial[:,:2].sum(1),initial[:,2:].sum(1)))
        after=np.column_stack((n[:,:2].sum(1),n[:,2:].sum(1)))
        if np.any(abs(after-before)>4e-12*np.maximum(before,np.finfo(float).tiny)):
            raise RuntimeError('Joint nucleus accounting failed')
        electrons=(n[:,1]+n[:,3]+2*n[:,4]-initial[:,1]-initial[:,3]-2*initial[:,4]).sum()
        if abs(electrons-amount)>4e-12*max(amount,float(initial.sum()),np.finfo(float).tiny):
            raise RuntimeError('Joint charge accounting failed')
        if any(np.any(a<0) or not np.all(np.isfinite(a)) for a in (n,p,absorbed,debit_total)):
            raise RuntimeError('Nonpositive or nonfinite common-field update')
        return dict(species=n,photons=p,absorbed=absorbed,debit=debit_total,
                    stop=reason,exposure=total_exposure,iterations=evaluations,
                    monochromatic_fluence=monochromatic_fluence,
                    exposure_kind=endpoint_kind or ("normalized_monochromatic_fluence" if used_exact else "midpoint_coordinate"),
                    substeps=accepted_steps,events=events,proposals=op.shape[0],
                    absorbed_count=amount,photon_error=photon_error,
                    max_debit_fraction=float(np.max(np.divide(debit_total,p0,
                        out=np.zeros_like(p0),where=p0>0))) if p0.size else 0.)

    if not op.shape[0] or not scale or not np.any(p>0):return finish()
    if monochromatic_fastpath and op.shape[0]>1:
        endpoint=_disjoint_hydrogen_endpoint(initial,p0,cross,op,epsilon,residual)
        if endpoint is not None:
            n,p,absorbed,debit_total=(endpoint[key] for key in ('species','photons','absorbed','debit'))
            reason=endpoint['stop'];total_exposure=None;accepted_steps=1
            endpoint_kind='independent_hydrogen_endpoint'
            return finish()
    native=get_backend()
    if native is not None and op.shape[0]>1 and isinstance(op,_CompactSphereOperator):
        result=native.joint_compact(initial,p0,cross,op.indices,op.offsets,
            epsilon,donor_fraction,count_tolerance,photon_floor,max_iterations)
        if result is not None:
            n,p,absorbed,debit_total,reason,total_exposure,evaluations,accepted_steps,events=result
            return finish()
    def fields(light):
        return op.scatter(op.aggregate((light*inverse[:,None])@cross)/volume[:,None])
    def accounting(light,rate,counts):
        response=np.divide(counts,rate,out=np.zeros_like(counts),where=rate>0)
        response=op.scatter(op.aggregate(response)/volume[:,None])
        return (light*inverse[:,None])*(response@cross.T)
    def capture(state,light,rate,dt):
        nonlocal evaluations
        evaluations+=1
        new,counts=_upward_cells(state,rate,dt)
        return new,counts,accounting(light,rate,counts)
    def remaining_from_absorbers(total):
        return np.column_stack((total[:,0],total[:,1],total[:,1]+total[:,2]))

    sphere_absorbers=sphere_initial[:,[0,2,3]]
    for _ in range(max_iterations):
        hazard=(op.scatter(sphere_absorbers/volume[:,None])@cross.T)*inverse[:,None]
        if not np.all(np.isfinite(hazard)):
            raise ValueError("Sphere absorption hazards overflow binary64")
        # max sigma per band suffices to test whether any rate is representable;
        # avoid constructing a cell x band x reaction temporary.
        representable=(p*cross.max(axis=1)[None,:])>0
        physical=(p>residual)&(hazard>0)
        physical_groups=np.flatnonzero(np.any(physical,axis=0))
        exact_available=(monochromatic_fastpath and op.shape[0]==1 and
                         len(physical_groups)>0 and
                         np.all(cross[physical_groups]==cross[physical_groups[0]]))
        # Count-based exact exposure needs neither a representable artificial
        # half-step nor a representable p*sigma product.
        active=(physical if exact_available else
                physical&(.5*donor_fraction*p>=smallest)&representable)
        if not np.any(active):
            reason=('absorber_saturation' if not np.any(hazard>0) else 'photon_exhaustion') if absorbed.sum()>0 else 'no_absorption'
            return finish()
        groups=np.flatnonzero(np.any(active,axis=0))
        if monochromatic_fastpath and op.shape[0]==1 and np.all(cross[groups]==cross[groups[0]]):
            g=int(groups[0])
            covered=multiplicity>0
            light=np.where(active,p,0.)
            available=float(light.sum())
            if not np.isfinite(available):
                raise ValueError('Usable photon total overflow binary64')
            total=op.aggregate(n)[0]
            mapping,cap,hit,exposure,evals,status=_single_band_endpoint(total,available,cross[g],epsilon)
            # Construct local counts before the donor debit. If distributed
            # summation moves the endpoint a few ulps past budget, retreat
            # the exposure itself; never scale a chemical endpoint.
            for _retry in range(64):
                local=n[covered]@mapping.T;counts=n[covered]@cap.T
                spent=float(counts.sum())
                if spent<=available:break
                if exposure is None or exposure<=0:
                    raise RuntimeError('Monochromatic count endpoint infeasible')
                exposure=np.nextafter(exposure,0.)
                mapping,cap=upward_map(cross[g],exposure)
            else:raise RuntimeError('Could not resolve monochromatic budget roundoff')
            # Form the common fraction first: (p/P)*P can round above p.
            debit=light*(spent/available)
            if np.any(debit>p):raise RuntimeError('Monochromatic donor overdraw')
            n[covered]=local;absorbed[covered]+=counts;p-=debit;debit_total+=debit
            evaluations+=evals;accepted_steps+=1
            # Fluence has normalized, dimensionless cross-section units;
            # it is not the artificial midpoint integration coordinate.
            used_exact=True;monochromatic_fluence=exposure;total_exposure=None
            reason=status;events=[[0,int(h)] for h in hit]
            return finish()
        light=np.where(active,p,0.)
        rate=fields(light)  # Reused across every backtrack and event evaluation.
        step=donor_fraction/float(np.max(hazard[active]))
        if not np.isfinite(step) or step<=0:
            reason='numerical_remainder';return finish()
        def attempt(dt):
            _,_,half_debit=capture(n,light,rate,.5*dt)
            if np.any(half_debit>.5*donor_fraction*p):return None
            middle=light-half_debit
            if np.any(middle<0):return None
            new,counts,debit=capture(n,middle,fields(middle),dt)
            return None if np.any(debit>donor_fraction*p) else (new,counts,debit)
        for _backtrack in range(100):
            result=attempt(step)
            if result is not None:break
            step*=.5
        else:raise RuntimeError('No positive common-field midpoint step found')
        next_n,counts,debit=result
        if not np.any(debit>0) and np.array_equal(next_n,n):
            reason='numerical_remainder';return finish()
        next_sphere_absorbers=op.aggregate(next_n[:,[0,2,3]])
        hit=eligible&(remaining_from_absorbers(next_sphere_absorbers)<=threshold)
        if np.any(hit):
            low,high=0.,step
            for _bisect in range(64):
                middle=low+.5*(high-low)
                if middle==low or middle==high:break
                trial=attempt(middle)
                if trial is None:raise RuntimeError('Common-field event bracket lost feasibility')
                total=op.aggregate(trial[0][:,[0,2,3]])
                if np.any(eligible&(remaining_from_absorbers(total)<=threshold)):high=middle
                else:low=middle
            step=high;next_n,counts,debit=attempt(step)
            next_sphere_absorbers=op.aggregate(next_n[:,[0,2,3]])
            hit=eligible&(remaining_from_absorbers(next_sphere_absorbers)<=threshold)
        n=next_n;p-=debit;absorbed+=counts;debit_total+=debit
        total_exposure+=step;accepted_steps+=1;sphere_absorbers=next_sphere_absorbers
        if np.any(p<0) or np.any(n<0):raise RuntimeError('Nonpositive common-field update')
        if np.any(hit):
            reason='crossing';events=np.argwhere(hit).tolist();return finish()
    raise RuntimeError('Common-field chemistry failed to converge; state not committed')

# Original research module: coupled/es_batch_barrier.py
"""Bounded-memory, pass-only batch H/He excursion-set candidate chemistry.

Each row is an independent aggregate sphere. The chemistry and donor limits
match ``mixed_es_trial``; rows have independent adaptive exposure steps. A
candidate needs only its crossing Boolean, so rejected sphere scans do not
build transfer/absorption-response tensors or bisect an already-detected
crossing. No spatial transport or change of the ES barrier is introduced.
"""
from pathlib import Path
import sys
import numpy as np



def mixed_es_pass_batch(gas, photons, energies=(18., 30., 60.), *,
                        epsilon=1e-6, enabled_targets=(True, True, True),
                        donor_fraction=.02, count_tolerance=1e-13,
                        max_iterations=20000, batch_size=256):
    """Return ``passed``, ``absorbed_positive``, and numerical-work counters.

    Inputs have shape (candidates, 5), (candidates, groups), and (groups,).
    Working storage is O(batch_size*(5+groups)); there is no sphere-by-cell
    array. The input arrays and Boolean results scale linearly with candidate
    count. Identical-opacity active groups use the scalar solver's exact limit path;
    all rows with multiple usable bands are integrated together.
    """
    # Delayed imports avoid a dependency cycle in assembled reference builds.

    n_all = np.asarray(gas, dtype=float)
    p_all = np.asarray(photons, dtype=float)
    e = np.asarray(energies, dtype=float)
    enabled = np.asarray(enabled_targets, dtype=bool)
    if (n_all.ndim != 2 or n_all.shape[1] != 5 or p_all.ndim != 2 or
            len(p_all) != len(n_all) or e.shape != (p_all.shape[1],) or
            enabled.shape != (3,)):
        raise ValueError('Expected gas(K,5), photons(K,G), energies(G), targets(3)')
    if (not all(np.all(np.isfinite(x)) for x in (n_all, p_all, e)) or
            np.any(n_all < 0) or np.any(p_all < 0) or
            not np.isfinite(epsilon) or not 0 < epsilon < 1 or
            not np.isfinite(donor_fraction) or not 0 < donor_fraction < 1 or
            not np.isfinite(count_tolerance) or not 0 < count_tolerance < 1 or
            not isinstance(batch_size, (int, np.integer)) or batch_size < 1 or
            not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError('Finite nonnegative states and valid integration options required')
    count = len(n_all)
    passed = np.zeros(count, dtype=bool)
    absorbed_positive = np.zeros(count, dtype=bool)
    iterations = np.zeros(count, dtype=np.int64)
    counters = dict(candidate_rows=count, batch_size=int(batch_size),
                    workspace_rows=min(count, int(batch_size)), batches=0,
                    vector_candidate_steps=0, vector_attempts=0,
                    single_band_exact_rows=0, duplicate_opacity_exact_rows=0, fallback_rows=0,
                    stalled_rows=0, exhausted_rows=0)
    cross = cross_sections(e)
    scale = float(cross.max()) if cross.size else 0.
    if not np.isfinite(scale):
        raise ValueError('Nonfinite photoionization cross sections')
    if scale == 0 or count == 0:
        return dict(passed=passed, absorbed_positive=absorbed_positive,
                    iterations=iterations, counters=counters)
    cross = cross / scale
    _, opacity_class = np.unique(cross, axis=0, return_inverse=True)
    tiny = np.nextafter(0., 1.)
    eps = np.finfo(float).eps

    def rem(n):
        return np.column_stack((n[:, 0], n[:, 2], n[:, 2] + n[:, 3]))

    def identical_opacity(active):
        first = np.argmax(active, axis=1)
        same = opacity_class[None, :] == opacity_class[first, None]
        return active.any(axis=1) & np.all(~active | same, axis=1)

    def route_rates(p):
        products = p[:, :, None] * cross[None, :, :]
        rates = products.sum(axis=1)
        if not np.all(np.isfinite(rates)):
            raise ValueError('Nonfinite aggregate photoionization rates')
        route = np.divide(products, rates[:, None, :], out=np.zeros_like(products),
                          where=rates[:, None, :] > 0)
        return rates, route

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        counters['batches'] += 1
        n = n_all[start:stop].copy()
        p = p_all[start:stop].copy()
        normal = np.column_stack((n[:, :2].sum(axis=1), n[:, 2:].sum(axis=1),
                                  n[:, 2:].sum(axis=1)))
        threshold = epsilon * normal
        eligible = enabled[None, :] & (normal > 0) & (
            rem(n) > threshold + 64 * eps * normal)
        reachable = rem(n)
        capacity = reachable @ (cross > 0).T
        if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(capacity)):
            raise ValueError('Nonfinite aggregate candidate atom counts')
        residual = count_tolerance * np.minimum(p, capacity)
        potential = (p > residual) & (capacity > 0)
        live = eligible.any(axis=1) & potential.any(axis=1)
        single = live & identical_opacity(potential)
        for row in np.flatnonzero(single):
            trial = mixed_es_trial(n[row], p[row], e, epsilon=epsilon,
                                   enabled_targets=enabled, donor_fraction=donor_fraction,
                                   count_tolerance=count_tolerance,
                                   max_iterations=max_iterations)
            index = start + row
            passed[index] = trial.passed
            absorbed_positive[index] = np.any(trial.absorbed_group_cell_species > 0)
            iterations[index] = trial.iterations
            counters['single_band_exact_rows'] += 1
            counters['duplicate_opacity_exact_rows'] += int(potential[row].sum() > 1)
        live[single] = False

        while np.any(live):
            rows = np.flatnonzero(live)
            nn, pp = n[rows], p[rows]
            hazard = nn[:, ABSORBER] @ cross.T
            if not np.all(np.isfinite(hazard)):
                raise ValueError('Nonfinite candidate absorption hazard')
            physical = (hazard > 0) & (pp > residual[rows])
            exact_available = identical_opacity(physical)
            # Direct count roots need neither a fractional half-step nor
            # representable p*sigma. Other rows use the midpoint rate floors.
            midpoint_ready = ((.5 * donor_fraction * pp >= tiny) &
                              np.any(pp[:, :, None] * cross[None, :, :] > 0, axis=2))
            contributing = physical & (exact_available[:, None] | midpoint_ready)
            participating = contributing.any(axis=1)
            exhausted = rows[~participating]
            live[exhausted] = False
            counters['exhausted_rows'] += int(len(exhausted))
            if not np.any(participating):
                continue
            rows = rows[participating]
            nn, pp = nn[participating], pp[participating]
            hazard = hazard[participating]
            contributing = contributing[participating]
            # Identical opacity vectors keep their common rate ratios and
            # fractional depletion, even when separate channels own photons.
            one_band = identical_opacity(contributing)
            for j in np.flatnonzero(one_band):
                row = rows[j]
                index = start + row
                band = int(np.flatnonzero(contributing[j])[0])
                available = float(pp[j, contributing[j]].sum())
                if not np.isfinite(available):
                    raise ValueError('Nonfinite usable photon total')
                _, capture, events, _, evaluations, _ = _single_band_endpoint(
                    nn[j], available, cross[band], epsilon, enabled)
                passed[index] = bool(events)
                absorbed_positive[index] |= np.any(capture @ nn[j] > 0)
                iterations[index] += evaluations
                live[row] = False
                counters['single_band_exact_rows'] += 1
                counters['duplicate_opacity_exact_rows'] += int(contributing[j].sum() > 1)
            keep = ~one_band
            if not np.any(keep):
                continue
            rows, nn, pp = rows[keep], nn[keep], pp[keep]
            hazard, contributing = hazard[keep], contributing[keep]
            active_p = np.where(contributing, pp, 0.)
            rate, route = route_rates(active_p)
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                exposure = donor_fraction / np.max(np.where(contributing, hazard, 0.), axis=1)
            finite_step = np.isfinite(exposure) & (exposure > 0)
            # Extremely small/large count scales can exceed the exposure
            # coordinate's representable range; retain exact scalar behavior.
            for j in np.flatnonzero(~finite_step):
                row = rows[j]
                trial = mixed_es_trial(n[row], p[row], e, epsilon=epsilon,
                                       enabled_targets=enabled, donor_fraction=donor_fraction,
                                       count_tolerance=count_tolerance,
                                       max_iterations=max_iterations)
                index = start + row
                passed[index] = trial.passed
                absorbed_positive[index] |= np.any(trial.absorbed_group_cell_species > 0)
                iterations[index] += trial.iterations
                live[row] = False
                counters['fallback_rows'] += 1
            if not np.any(finite_step):
                continue
            rows, nn, pp = rows[finite_step], nn[finite_step], pp[finite_step]
            active_p, rate, route = active_p[finite_step], rate[finite_step], route[finite_step]
            contributing, exposure = contributing[finite_step], exposure[finite_step]
            next_n = np.empty_like(nn)
            lost = np.empty_like(pp)
            pending = np.ones(len(rows), dtype=bool)
            for _ in range(100):
                take = np.flatnonzero(pending)
                if len(take) == 0:
                    break
                counters['vector_attempts'] += int(len(take))
                _, half_counts = _upward_cells(nn[take], rate[take], exposure[take] / 2.)
                half_lost = np.einsum('kgs,ks->kg', route[take], half_counts)
                half_ok = np.all(half_lost <= .5 * donor_fraction * pp[take], axis=1)
                exposure[take[~half_ok]] *= .5
                good = take[half_ok]
                if len(good) == 0:
                    continue
                mid_p = np.where(contributing[good], active_p[good] - half_lost[half_ok], 0.)
                mid_rate, mid_route = route_rates(mid_p)
                proposed, captures = _upward_cells(nn[good], mid_rate, exposure[good])
                debit = np.einsum('kgs,ks->kg', mid_route, captures)
                ok = np.all(debit <= donor_fraction * pp[good], axis=1)
                accepted = good[ok]
                next_n[accepted] = proposed[ok]
                lost[accepted] = debit[ok]
                pending[accepted] = False
                exposure[good[~ok]] *= .5
            if np.any(pending):
                raise RuntimeError('Batch candidate found no positive photon-limited step')
            remaining_p = pp - lost
            if (not all(np.all(np.isfinite(x)) for x in (next_n, remaining_p, lost)) or
                    np.any(next_n < 0) or np.any(remaining_p < 0) or np.any(lost < 0)):
                raise RuntimeError('Nonfinite or negative batch candidate update')
            index = start + rows
            iterations[index] += 1
            counters['vector_candidate_steps'] += len(rows)
            absorbed_positive[index] |= np.any(lost > 0, axis=1)
            hit = np.any(eligible[rows] & (rem(next_n) <= threshold[rows]), axis=1)
            passed[index[hit]] = True
            stalled = np.all(lost == 0, axis=1) & np.all(next_n == nn, axis=1)
            counters['stalled_rows'] += int(stalled.sum())
            n[rows], p[rows] = next_n, remaining_p
            live[rows[hit | stalled]] = False
            if np.any(live[rows] & (iterations[index] >= max_iterations)):
                raise RuntimeError('Batch sphere reaction trial did not converge')
    return dict(passed=passed, absorbed_positive=absorbed_positive,
                iterations=iterations, counters=counters)

# Original research module: coupled/spectral_es_reference.py
"""Executable ES + competing H/He photo-chemistry + local OTS reference.

Research specification, not production Meraxes code. Every grid cell is a
fixed sphere centre, independent of source catalogue or spectral budgets. ES
spheres contain all active source credits, irrespective of centre. No permanent bubble mixing, source-target
edges, rays or spatial radiation equation. Each trial uses the same snapshot
of the state; selected spheres drive one common local chemical exposure.
"""
from pathlib import Path
import sys
import numpy as np


class ProposalBatch:
    """Compact same-radius candidate set; no Python dictionary per sphere."""
    def __init__(self,centres,radius_index):
        self.centres=np.asarray(centres,dtype=np.int64)
        self.radius_index=int(radius_index)
    def __len__(self):
        return len(self.centres)


class SpectralES:
    def __init__(self, side=3, helium=.081, epsilon=1e-5, donor_fraction=.02,
                 radii2=None, count_tolerance=1e-12, count_pruning=True,
                 trial_cache=True, radius_ratio=None, batch_candidates=True,
                 candidate_batch_size=256, photon_floor=0.):
        self.side=side
        if radius_ratio is not None:
            if radii2 is not None:
                raise ValueError('Specify radii2 or radius_ratio, not both')
            radii2=geometric_radii2(side,ratio=radius_ratio)
        self.geometry=PeriodicSphereGeometry(side,radii2=radii2)
        self.xyz=self.geometry.xyz
        self.size=self.geometry.size
        self.radii2=self.geometry.radii2
        self.count_tolerance=count_tolerance
        if not np.isfinite(photon_floor) or photon_floor<0:
            raise ValueError('Finite nonnegative photon activity floor required')
        self.photon_floor=float(photon_floor)
        self.count_pruning=count_pruning
        self.trial_cache=trial_cache
        self.batch_candidates=batch_candidates
        self.candidate_batch_size=int(candidate_batch_size)
        if self.candidate_batch_size<1 or self.candidate_batch_size!=candidate_batch_size:
            raise ValueError("Positive integer candidate batch size required")
        if not 0<count_tolerance<1:
            raise ValueError('Relative count tolerance must lie in (0,1)')
        self.counters=dict(candidate_checks=0,candidate_integrations=0,
                           count_bound_pruned=0,trial_cache_hits=0,
                           empty_or_ineligible=0,batch_calls=0,inactive_support_retries=0)
        self.gas = np.zeros((self.size,5))
        self.gas[:,0], self.gas[:,2] = 1., helium
        self.initial_nuclei = self.nuclei()
        self.initial_electrons = self.electrons()
        self.energies = np.r_[18.,30.,60.,OPACITY_E_EV]
        self.photons = np.zeros((self.size,len(self.energies)))
        # Diagnostic only: history never alters the fixed candidate catalogue.
        self.source_history = np.zeros(self.size)
        self.epsilon, self.donor_fraction = epsilon, donor_fraction
        self.emitted = 0.
        self.diffuse_emitted = 0.
        self.absorbed = 0.
        self.captures = 0.
        self.events = []

    def nuclei(self):
        return np.column_stack([self.gas[:,:2].sum(axis=1), self.gas[:,2:].sum(axis=1)])

    def electrons(self):
        return float((self.gas[:,1]+self.gas[:,3]+2*self.gas[:,4]).sum())

    def inject(self, sources):
        additions = np.zeros_like(self.photons)
        for position, groups in sources:
            i = np.ravel_multi_index(tuple(position),(self.side,)*3)
            additions[i,:3] += np.asarray(groups)
        if np.any(additions < 0) or not np.all(np.isfinite(additions)):
            raise ValueError('Emission must be finite and nonnegative')
        self.photons += additions
        self.source_history += additions.sum(axis=1)
        self.emitted += float(additions.sum())

    def react(self, dt, max_step=None, cell_volume=1e4):
        """Fixed T=1e4K and homogeneous cell chemistry; volume sets density.

        This advances physical time ONCE. ES radius trials never recombine.
        Residual diffuse photons are retained as separate channels.
        """
        for i in range(self.size):
            previous=self.photons[i,3:].copy()
            pending=previous/cell_volume
            n,p,l = evolve_dark(self.gas[i]/cell_volume,dt,dt if max_step is None else max_step,
                               pending=pending,
                               absorption_donor_fraction=self.donor_fraction,
                               photon_floor=self.photon_floor/cell_volume)
            self.gas[i] = n*cell_volume
            restored=p*cell_volume
            # Retain dormant credits in count units even when converting
            # them to a density would underflow to zero.
            unspent=l.absorbed.sum(axis=1)==0
            restored[unspent]=previous[unspent]+l.emitted[unspent]*cell_volume
            underflow=(~unspent)&(previous>0)&(pending==0)
            restored[underflow]+=previous[underflow]
            self.photons[i,3:] = restored
            self.diffuse_emitted += float(l.emitted.sum()*cell_volume)
            self.absorbed += float(l.absorbed.sum()*cell_volume)
            self.captures += float(l.captures.sum()*cell_volume)
        self.audit()

    def _candidate_decisions(self, gas, photons, cache):
        """Resolve candidates in bounded chunks; cache at most 4096 decisions."""
        active=np.zeros(len(gas),dtype=bool)
        passed=np.zeros(len(gas),dtype=bool)
        for start in range(0,len(gas),self.candidate_batch_size):
            stop=min(start+self.candidate_batch_size,len(gas))
            pending={}
            for j in range(start,stop):
                key=tuple(gas[j])+tuple(photons[j])
                if cache is not None and key in cache:
                    active[j],passed[j]=cache[key]
                    self.counters['trial_cache_hits']+=1
                elif cache is not None:
                    pending.setdefault(key,[]).append(j)
                else:
                    pending[(j,key)]=[j]
            if not pending:
                continue
            first=np.array([indices[0] for indices in pending.values()],dtype=int)
            self.counters['candidate_integrations']+=len(first)
            if self.batch_candidates and len(first)>=8:
                decisions=mixed_es_pass_batch(gas[first],photons[first],self.energies,
                    epsilon=self.epsilon,donor_fraction=self.donor_fraction,
                    batch_size=self.candidate_batch_size)
                flags=zip(decisions['absorbed_positive'],decisions['passed'])
                self.counters['batch_calls']+=1
            else:
                flags=[]
                for j in first:
                    trial=mixed_es_trial(gas[j],photons[j],self.energies,
                        epsilon=self.epsilon,donor_fraction=self.donor_fraction)
                    flags.append((bool(np.any(trial.absorbed_group_cell_species>0)),trial.passed))
            for (key,indices),(absorbs,crosses) in zip(pending.items(),flags):
                active[indices]=absorbs
                passed[indices]=crosses
                if cache is not None:
                    self.counters['trial_cache_hits']+=len(indices)-1
                    if len(cache)<4096:
                        cache[key]=(absorbs,crosses)
        return active,passed

    def _select(self, excluded_partial_radii=()):
        # A failed partial commit changes neither gas nor photons. Reuse its
        # completed scan instead of repeating every FFT for each trial radius.
        # A successful commit clears exclusions, which starts a fresh scan.
        if excluded_partial_radii and getattr(self,'_partial_retry_cache',None) is not None:
            for proposal in reversed(self._partial_retry_cache):
                if proposal.radius_index not in excluded_partial_radii:
                    return 'partial',proposal
            return None,[]
        self._partial_retry_cache=None
        active=getattr(self,'_active_photons',self.photons)
        if self.photon_floor:
            active=np.where(active>self.photon_floor,active,0.)
        fields=self.geometry.prepare(np.column_stack((self.gas,active)))
        cross=cross_sections(self.energies)
        if np.any(cross):
            cross=cross/np.max(cross)
        channels=cross>0
        cache={} if self.trial_cache else None
        partial=[]
        partial_candidates=[]
        cached_centres=0
        for k in reversed(range(len(self.radii2))):
            aggregate=self.geometry.all_sums(fields,k)
            n,p=aggregate[:,:5],aggregate[:,5:]
            native=get_backend()
            flags=(native.candidate_flags(aggregate,cross,self.epsilon,self.count_pruning)
                   if native is not None else None)
            if flags is not None:
                valid,proven_partial=flags
            else:
                normal=np.column_stack((n[:,:2].sum(1),n[:,2:].sum(1),n[:,2:].sum(1)))
                remaining=np.column_stack((n[:,0],n[:,2],n[:,2]+n[:,3]))
                needed=remaining-self.epsilon*normal
                eligible=(normal>0)&(remaining>self.epsilon*normal+64*np.finfo(float).eps*normal)
                # HeII can only be generated after a real initial HeI absorption.
                # Match the chemistry's representable rates, not merely its
                # Boolean transition graph: min-subnormal atoms times sigma can
                # underflow to zero and must not create a stalled partial front.
                hazard=n[:,[0,2,3]]@cross.T
                # A monochromatic endpoint can consume a photon account smaller
                # than a representable midpoint half-step. Leave that decision to
                # the actual selected-support chemistry.
                usable=np.any((p>0)&(hazard>0),axis=1)
                valid=np.any(eligible,axis=1)&usable
                proven_partial=np.zeros(self.size,dtype=bool)
                if self.count_pruning:
                    # Every transition is bounded by its ionizing-band photons,
                    # even before competitors are paid. Below this necessary bound
                    # a full crossing is impossible; no chemical endpoint is needed.
                    upper=p@channels
                    margin=64*np.finfo(float).eps*np.maximum(upper,remaining)
                    possible=np.any(eligible&(upper+margin>=needed),axis=1)
                    proven_partial=valid&~possible
            self.counters['candidate_checks']+=self.size
            self.counters['empty_or_ineligible']+=int(np.count_nonzero(~valid))
            self.counters['count_bound_pruned']+=int(np.count_nonzero(proven_partial))
            unresolved=np.flatnonzero(valid&~proven_partial)
            passing=np.zeros(self.size,dtype=bool)
            absorbable=proven_partial.copy()
            if len(unresolved):
                absorbs,crosses=self._candidate_decisions(n[unresolved],p[unresolved],cache)
                absorbable[unresolved]=absorbs
                passing[unresolved]=crosses
            if np.any(passing):
                return 'crossing',ProposalBatch(np.flatnonzero(passing),k)
            if np.any(absorbable) and k not in excluded_partial_radii:
                partial=ProposalBatch(np.flatnonzero(absorbable),k)
                if partial_candidates is not None:
                    cached_centres+=len(partial)
                    if cached_centres<=1000000:  # At most 8 MB of cached indices.
                        partial_candidates.append(partial)
                    else:
                        partial_candidates=None
        # All partial fronts at the single smallest absorbable radius. An
        # empty distant centre must not grow a partial sphere to a remote source.
        self._partial_retry_cache=partial_candidates
        return ('partial',partial) if partial else (None,[])

    def _commit(self, proposals, kind):
        return self._commit_group(proposals,kind)

    def _commit_group(self, proposals, kind):
        membership=self.geometry.operator(proposals.centres,
            np.full(len(proposals),proposals.radius_index,dtype=int))
        retained=self.photons-self._active_photons
        result=joint_exposure_update(self.gas,self._active_photons,self.energies,
                                    membership,epsilon=self.epsilon,
                                    donor_fraction=self.donor_fraction,
                                    photon_floor=self.photon_floor)
        absorbed=float(result['absorbed'].sum())
        if (absorbed==0 and np.array_equal(result['species'],self.gas) and
                np.array_equal(result['photons'],self._active_photons)):
            if kind=='partial' and result['stop'] in ('no_absorption','numerical_remainder'):
                # Candidate sums can be representable while their actual
                # volume/multiplicity-weighted common field underflows. This
                # exact no-change result identifies a support this numerical
                # solver cannot advance (including an unrepresentable step).
                # Keep all counts and try the next partial radius.
                self.counters['inactive_support_retries']+=1
                return False
            raise RuntimeError('Selected ES support made no numerical progress: '
                f'kind={kind}, radius_squared={self.radii2[proposals.radius_index]}, '
                f'stop={result["stop"]}; state and photon credits retained')
        self.gas=result['species']
        self.photons=result['photons']+retained
        self.absorbed+=absorbed
        self.events.append(dict(kind=kind,proposals=len(membership),
                                exposure=result['exposure'],stop=result['stop'],
                                iterations=result['iterations'],substeps=result['substeps'],
                                exposure_kind=result['exposure_kind'],
                                absorbed=absorbed,
                                radius=float(np.sqrt(self.radii2[proposals.radius_index]))))
        return True

    def illuminate(self, max_batches=500, progress=None):
        """Conservative geometry/chemistry stage; max_batches is a fail-loud guard.

        Unlike scalar H, no general H/He finite-event performance bound is claimed.
        """
        self.audit()
        baseline=self.photons.copy()
        committed=0
        excluded_partial_radii=set()
        while committed<max_batches:
            # Each entry account sets its own residual floor; inactive counts
            # stay in the photon ledger and never set another band's tolerance.
            self._active_photons=np.where(
                self.photons>np.maximum(self.photon_floor,self.count_tolerance*baseline),
                self.photons,0.)
            if not np.any(self._active_photons):
                self.audit()
                return
            started=time.monotonic()
            if progress:
                progress(dict(phase='select',batch=committed+1,state='start'))
            kind, proposals=self._select(excluded_partial_radii)
            if progress:
                progress(dict(phase='select',batch=committed+1,state='done',
                    seconds=round(time.monotonic()-started,3),kind=kind,
                    supports=len(proposals),radius_squared=(float(self.radii2[proposals.radius_index])
                        if proposals else None)))
            if not proposals:
                self.audit()
                return
            started=time.monotonic()
            if progress:
                progress(dict(phase='absorb',batch=committed+1,state='start'))
            accepted=self._commit(proposals,kind)
            if progress:
                details={key:self.events[-1][key] for key in ('stop','absorbed','iterations','substeps','exposure_kind')} if accepted else {}
                progress(dict(phase='absorb',batch=committed+1,state='done',
                    seconds=round(time.monotonic()-started,3),accepted=accepted,**details))
            if accepted:
                committed+=1
                excluded_partial_radii.clear()
            else:
                # At most one retry per catalogue radius for this unchanged
                # state. Successful physical progress starts a fresh scan.
                excluded_partial_radii.add(proposals.radius_index)
            self.audit()
        raise RuntimeError('H/He ES did not finish; state and all credits retained')

    def audit(self):
        photon_error=self.emitted+self.diffuse_emitted-self.absorbed-float(self.photons.sum())
        electron_error=self.electrons()-self.initial_electrons-self.absorbed+self.captures
        nuclei_error=float(np.max(np.abs(self.nuclei()-self.initial_nuclei)))
        tiny=np.finfo(float).tiny
        photon_scale=max(tiny,self.emitted+self.diffuse_emitted)
        electron_scale=max(tiny,abs(self.initial_electrons),self.absorbed,
                           self.captures,float(self.initial_nuclei.sum()))
        nuclei_relative=float(np.max(np.abs(self.nuclei()-self.initial_nuclei)/
                                      np.maximum(self.initial_nuclei,tiny)))
        if (not all(np.isfinite(x) for x in (photon_error,electron_error,nuclei_error)) or
                np.any(~np.isfinite(self.gas)) or np.any(~np.isfinite(self.photons)) or
                np.any(self.gas<0) or np.any(self.photons<0) or
                abs(photon_error)>1e-9*photon_scale or
                abs(electron_error)>1e-9*electron_scale or nuclei_relative>1e-9):
            raise AssertionError((photon_error,electron_error,nuclei_error))
        return dict(photon_error=photon_error,electron_error=electron_error,nuclei_error=nuclei_error,
                    emitted=self.emitted,diffuse_emitted=self.diffuse_emitted,absorbed=self.absorbed,
                    captures=self.captures,unabsorbed=float(self.photons.sum()),
                    hydrogen_ionized=float(self.gas[:,1].sum()),
                    helium_II=float(self.gas[:,3].sum()),helium_III=float(self.gas[:,4].sum()),
                    min_species=float(self.gas.min()),min_photons=float(self.photons.min()),batches=len(self.events))


def validate_history_independence(model_class=SpectralES):
    """Past emission metadata must not change an identical current state.

    Includes the two previously failing geometries, mixed H/He spectra, and
    residual primary/diffuse credits without a recorded stellar emitter.
    No radiation, atom or global accounting variables are changed between
    matched runs: only the diagnostic source_history is perturbed.
    """
    from copy import deepcopy

    results = {}

    def compare(name, initial, histories):
        baseline = deepcopy(initial)
        baseline.illuminate()
        gas_error = photon_error = recombination_proxy_error = 0.
        for history in histories:
            other = deepcopy(initial)
            other.source_history[:] = history
            other.illuminate()
            gas_error = max(gas_error, float(np.max(np.abs(other.gas-baseline.gas))))
            photon_error = max(photon_error, float(np.max(np.abs(other.photons-baseline.photons))))
            recombination_proxy_error = max(recombination_proxy_error,
                abs(float(other.gas[:,1]@other.gas[:,1]-baseline.gas[:,1]@baseline.gas[:,1])))
            # The same physical inputs must execute the same allocation, not
            # merely conserve the same globally integrated photon count.
            assert np.array_equal(other.gas, baseline.gas), name
            assert np.array_equal(other.photons, baseline.photons), name
            assert other.events == baseline.events, name
            other.audit()
        results[name] = dict(history_variants=len(histories),
            max_species_difference=gas_error, max_photon_difference=photon_error,
            hydrogen_sum_x2_difference=recombination_proxy_error, **baseline.audit())
        return baseline

    paired = model_class(side=3, helium=0.)
    paired.inject([((0,0,0),(4.,0.,0.)), ((1,0,0),(4.,0.,0.))])
    histories = []
    for delta in (1e-10, 1e-6, 1., 100.):
        history = paired.source_history.copy()
        history[0] += delta
        histories.append(history)
    # Preserve even the total historical count while changing its locations.
    history = np.zeros(paired.size)
    history[-1] = paired.source_history.sum()
    histories.append(history)
    compare('near_tie_two_sources', paired, histories)

    expired = model_class(side=3, helium=0.)
    expired.gas[0,:2] = [0.,1.]
    expired.initial_electrons = expired.electrons()
    expired.inject([((0,0,0),(.5,0.,0.))])
    expired.source_history[0] = 1.5
    histories = []
    for old in (3.,100.):
        history = expired.source_history.copy()
        history[9] = old
        histories.append(history)
    compare('extinguished_centre', expired, histories)

    mixed = model_class(side=3)
    mixed.inject([((0,1,1),(3.,.6,.3)), ((2,1,1),(3.,.6,.3))])
    rng = np.random.default_rng(20261018)
    compare('mixed_spectrum_HHe', mixed,
            [np.zeros(mixed.size), rng.uniform(0.,100.,mixed.size)])

    residual = model_class(side=3, helium=0.)
    residual.inject([((1,1,1),(.25,0.,0.))])
    remaining = compare('retained_primary_without_history', residual,
                        [np.zeros(residual.size)])
    assert abs(remaining.absorbed-.25)<1e-10

    diffuse = model_class(side=3, helium=0.)
    diffuse.photons[13,3] = .25
    diffuse.diffuse_emitted = .25
    remaining = compare('diffuse_without_stellar_history', diffuse,
                        [rng.uniform(0.,100.,diffuse.size)])
    assert abs(remaining.absorbed-.25)<1e-10

    empty = model_class(side=3)
    unchanged = compare('history_without_photons', empty,
                        [np.full(empty.size,100.)])
    assert np.array_equal(unchanged.gas, empty.gas) and not unchanged.events
    return results


def validate_physics_regressions():
    """The three demonstrated defects, exercised through the delivered code."""
    results = {}
    models = []
    for delta in (0.,1e-6,1e-9):
        model = SpectralES(side=3,helium=0.)
        model.inject([((0,0,0),(4.+delta,0.,0.)),
                      ((1,0,0),(4.-delta,0.,0.))])
        model.illuminate()
        models.append(model)
    differences = [float(np.max(abs(m.gas-models[0].gas))) for m in models]
    assert max(differences)<1e-10
    results['current_budget_near_tie'] = dict(
        source_perturbations=[0.,1e-6,1e-9],max_species_differences=differences,
        total_photons=[m.emitted for m in models],
        audits=[m.audit() for m in models])

    mono = SpectralES(side=3)
    mono.inject([((0,0,0),(0.,4.,0.)),((0,0,1),(0.,4.,0.))])
    mono.illuminate()
    cross = cross_sections(np.array([30.]))[0]
    ratio = float(cross[1]/cross[0])
    required = mono.gas[:,0]**ratio
    actual = mono.gas[:,2]/.081
    error = float(np.max(abs(actual-required)))
    assert error<1e-12
    results['monochromatic_overlap'] = dict(cross_section_ratio=ratio,
        max_neutral_fraction_relation_error=error,audit=mono.audit())

    initial = np.array([0.,1.,0.,1.,0.])
    reference = mixed_es_trial(initial,[0.,0.,1.])
    local_photons = np.zeros(8)
    local_photons[3] = 1.
    local_reference = absorb_local(initial,local_photons)[0]
    inactive = []
    for soft in (0.,1e10,1e13):
        trial = mixed_es_trial(initial,[soft,0.,1.])
        local_photons[0] = soft
        local_n,left,absorbed,iterations = absorb_local(initial,local_photons)
        difference = float(np.max(abs(trial.species-reference.species)))
        local_difference = float(np.max(abs(local_n-local_reference)))
        assert difference==0 and local_difference==0
        assert trial.species[0,4]>.99999 and local_n[4]>.99999
        assert trial.photons[0]==soft and left[0]==soft
        inactive.append(dict(soft_photons=soft,
            trial_HeIII=float(trial.species[0,4]),trial_difference=difference,
            local_OTS_HeIII=float(local_n[4]),local_OTS_difference=local_difference))
    results['inactive_spectral_groups'] = inactive
    # Tiny HeII-ionizing rates must not acquire an O(eps) spurious cascade.
    a,b,t=1650.8995065753666,1.6508995065753667e-27,3.976560122061689e-10
    _,capture=upward_map(np.array([0.,a,b]),t)
    expected=2.154895879560921e-43  # Independent high-precision analytic value.
    relative_error=abs(float(capture[2,2])/expected-1.)
    _,zero_capture=upward_map(np.array([0.,a,0.]),t)
    assert relative_error<1e-12 and zero_capture[2,2]==0.
    # Captured candidate from the formerly failing ninth step of a 32-step run.
    initial=np.array([.12827731100856773,.871722688991432,
                      .004906020388436399,.04897673058323916,.027117249028324453])
    photons=np.array([.09375,.01875,.009375,0.,0.,0.,0.,0.,
                      1.377288031643469e-20,0.,0.])
    captured=mixed_es_trial(initial,photons,np.r_[18.,30.,60.,OPACITY_E_EV],
                            epsilon=1e-5,donor_fraction=.08)
    ledger_error=abs(float((photons-captured.photons).sum()
                          -captured.absorbed_group_cell_species.sum()))
    assert ledger_error<1e-12
    results['tiny_helium_cascade'] = dict(
        probability=float(capture[2,2]),high_precision_reference=expected,
        relative_error=relative_error,zero_hard_rate_probability=float(zero_capture[2,2]),
        captured_time32_trial_iterations=captured.iterations,
        captured_time32_trial_ledger_error=ledger_error)
    return results


def _validation_evolution_run(steps):
    """Independent time-step case, also usable by validation worker processes."""
    s=SpectralES(side=3)
    sources=[((0,1,1),(3.,.6,.3)),((2,1,1),(3.,.6,.3))]
    for step_index in range(steps):
        s.react(MYR/steps)
        s.inject([(p,np.asarray(g)/steps) for p,g in sources])
        s.illuminate()
        if (step_index+1)%8==0 or step_index+1==steps:
            print(f'Validation: evolution {steps}, completed {step_index+1}',
                  file=sys.stderr, flush=True)
    return s


def validate_reference(validation_workers=1):
    """Independent numerical checks of the assembled, spatial ES/HHe reference."""
    from pathlib import Path
    import json, time
    import numpy as np

    start=time.monotonic()
    import hashlib
    results={'reference_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'history_independence': validate_history_independence(),
             'physics_regressions': validate_physics_regressions()}
    print('Validation: history and three physics regressions passed',file=sys.stderr,flush=True)
    sources=[((0,1,1),(3.,.6,.3)),((2,1,1),(3.,.6,.3))]

    def run(src=sources,steps=1,chemistry=False):
        s=SpectralES(side=3)
        for _ in range(steps):
            if chemistry:
                s.react(MYR/steps)
            s.inject([(p,np.asarray(g)/steps) for p,g in src])
            s.illuminate()
        return s

    s=run()
    results['joint_overlap']=s.audit()
    assert s.gas[:,3].sum()>0 and s.gas[:,4].sum()>0
    assert len(s.events)>0

    symmetry={}
    for name,transform in [('translation',lambda p: (np.asarray(p)+[1,1,0])%3),
                           ('rotation',lambda p: np.array([p[1],(-p[0])%3,p[2]])),
                           ('reflection',lambda p: np.array([(-p[0])%3,p[1],p[2]]))]:
        x=run([(tuple(transform(p)),g) for p,g in sources])
        mapped=np.ravel_multi_index(np.asarray([transform(p) for p in s.xyz]).T,(3,)*3)
        error=float(np.max(np.abs(x.gas[mapped]-s.gas)))
        symmetry[name]=error
        assert error<1e-10
    reversed_s=run(list(reversed(sources)))
    symmetry['source_permutation']=float(np.max(np.abs(reversed_s.gas-s.gas)))
    assert symmetry['source_permutation']<1e-12
    results['joint_symmetry']=symmetry
    print('Validation: assembled overlap and symmetry passed',file=sys.stderr,flush=True)

    soft=run([((1,1,1),(8.,0.,0.))])
    assert np.max(soft.gas[:,3:])==0
    results['soft_spectrum']=soft.audit()

    results['time_refinement']=[]
    previous=None
    step_counts=(4,8,16,32,64)
    if validation_workers < 1:
        raise ValueError('Validation workers must be positive')
    pool=None
    if validation_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool=ProcessPoolExecutor(max_workers=min(validation_workers,len(step_counts)))
        time_runs=pool.map(_validation_evolution_run,step_counts)
    else:
        time_runs=map(_validation_evolution_run,step_counts)
    try:
        for steps,x in zip(step_counts,time_runs):
            difference=None if previous is None else float(np.max(np.abs(x.gas-previous)))
            results['time_refinement'].append(dict(steps=steps,max_species_change=difference,**x.audit()))
            previous=x.gas.copy()
            print(f'Validation: time evolution with {steps} substeps passed',file=sys.stderr,flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)

    # After the source switches off, geometry cannot undo historical ions; chemistry does.
    before=x.gas.copy()
    x.react(20*MYR,max_step=.5*MYR)
    x.illuminate()
    results['source_off_20Myr']=dict(before_HII=float(before[:,1].sum()),
                                   before_HeII=float(before[:,3].sum()),
                                   before_HeIII=float(before[:,4].sum()),**x.audit())
    assert x.gas[:,1].sum()<before[:,1].sum()
    assert x.gas[:,4].sum()<before[:,4].sum()

    results['runtime_seconds']=time.monotonic()-start
    results['validation_workers']=validation_workers
    path=Path(__file__).with_name('es_hhe_reference_results.json')
    path.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__ == '__main__':
    if '--validate-physics' in sys.argv:
        import json
        print(json.dumps(validate_physics_regressions(), indent=2))
    elif '--validate-history' in sys.argv:
        import json
        print(json.dumps(validate_history_independence(), indent=2))
    elif '--validate' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--validate',action='store_true')
        parser.add_argument('--validation-workers',type=int,default=1)
        args=parser.parse_args()
        validate_reference(args.validation_workers)
    else:
        model = SpectralES(side=3)
        model.inject([((0,1,1),(3.,.6,.3)), ((2,1,1),(3.,.6,.3))])
        model.illuminate()
        import json
        print(json.dumps(model.audit(), indent=2))
