"""Positive, isothermal H/He recombination for the angular allocation model.

The input contains atom counts, not densities. ``volume_cm3`` converts their
electron count to a physical density. Species order is HI,HII,HeI,HeII,HeIII;
capture order is HII->HI, HeII->HeI, HeIII->HeII.

This module retains the T=10^4 K case-A coefficients and eight diffuse-count
channels of the former es_hhe_reference.py (SHA-256
4075d324a8e965ecc317bb4a4c85033b5d41cedb1a8385ca52b52c270d3cd542).
Coefficients use Hui & Gnedin (1997); the diffuse branching approximation
uses Friedrich et al. (2012), Tables 1/2. The output photons must subsequently
be absorbed locally or retained with their physical location. They must not
be discarded or credited to the original stellar sources. No diffuse photon
transport, thermal evolution or MHR clumping is done here. ``recombine``
returns gross captures and emission; ``evolve_local`` also applies the local
coupled-OTS absorption closure and retains every remaining diffuse photon.
"""

import numpy as np

if __package__:
    from .chemistry import upward_cells, cross_sections
    from .angular_hhe import illuminate_channels
else:
    from chemistry import upward_cells, cross_sections
    from angular_hhe import illuminate_channels


CHANNEL_NAMES = (
    "H_ground", "HeII_ground", "HeII_excited_H", "HeIII_ground",
    "HeIII_n2", "HeIII_Lya", "HeIII_2gamma_low", "HeIII_2gamma_high",
)
# Opacity evaluation energies: these are not emitted mean photon energies.
OPACITY_E_EV = np.array([13.6, 24.59, 13.6, 54.4, 13.6, 40.8, 13.6, 24.59])


def coefficients_1e4():
    """Return case-A, case-B and HeIII n=2 coefficients in cm^3 s^-1."""
    lam = 2 * np.array([157807., 285335., 631515.]) / 1e4
    fa = lambda x: 1.269e-13*x**1.503/(1+(x/.522)**.470)**1.923
    fb = lambda x: 2.753e-14*x**1.5/(1+(x/2.740)**.407)**2.242
    a = np.array([fa(lam[0]), 3e-14*lam[1]**.654, 2*fa(lam[2])])
    b = np.array([fb(lam[0]), 1.26e-14*lam[1]**.750, 2*fb(lam[2])])
    return a, b, 3.4e-13


def emit_from_captures(captures):
    """Return the eight retained diffuse photon counts per physical cell.

    This is the existing fixed-temperature count closure, not a full emitted
    spectrum. A helium capture can produce more than one H-ionizing photon.
    These counts therefore need a separate photon-emission ledger.
    """
    captures = np.asarray(captures, dtype=float)
    if (captures.ndim != 2 or captures.shape[1] != 3
            or np.any(captures < 0) or not np.all(np.isfinite(captures))):
        raise ValueError("captures must have shape (cells,3), finite and nonnegative")
    a, b, n2 = coefficients_1e4()
    ground = captures * (a-b)/a
    excited = captures * b/a
    p, v, l, m = .96, .285, 1.425, .737
    result = np.column_stack((
        ground[:, 0], ground[:, 1], p*excited[:, 1], ground[:, 2],
        captures[:, 2]*n2/a[2], (1-v)*excited[:, 2],
        v*(l-m)*excited[:, 2], v*m*excited[:, 2],
    ))
    if not np.all(np.isfinite(result)):
        raise ValueError("Diffuse recombination count overflow")
    return result


def _electrons(gas):
    return gas[:, 1] + gas[:, 3] + 2*gas[:, 4]


def _downward(gas, ne, step, alpha):
    """Exact positive downward map for a frozen electron density."""
    reverse = gas[:, [1, 0, 4, 3, 2]]
    rates = ne[:, None] * alpha[[0, 2, 1]]
    evolved, transitions = upward_cells(reverse, rates, step)
    return evolved[:, [1, 0, 4, 3, 2]], transitions[:, [0, 2, 1]]


def recombine(gas, volume_cm3, dt_s, *, temperature=1e4,
              max_fraction=.05, max_substeps=100000):
    """Advance local case-A recombination without mutating the inputs.

    ``gas`` is (cells,5) atom counts; ``volume_cm3`` is a positive scalar or
    one physical volume per cell; ``dt_s`` is the physical duration in seconds.
    Temperature is fixed to 10000 K to retain the previous spectral closure.

    Pure-H electron populations (possibly with neutral-He spectators) use the
    exact nonlinear HII solution. Other cells use an exponential midpoint
    map, with evolving electron density and bounded rate-times. Reducing
    ``max_fraction`` controls this second-order local integration error.
    Count and nucleus ledgers close at floating-point precision independently
    of the integration error. The returned ``diffuse_emitted`` must be added
    to the caller's local photon inventory before photoionization/transport.
    """
    initial = np.array(gas, dtype=float, copy=True)
    if (initial.ndim != 2 or initial.shape[1] != 5
            or np.any(initial < 0) or not np.all(np.isfinite(initial))):
        raise ValueError("gas must have shape (cells,5), finite and nonnegative")
    volume = np.asarray(volume_cm3, dtype=float)
    if volume.ndim == 0:
        volume = np.full(len(initial), float(volume))
    if (volume.shape != (len(initial),) or np.any(volume <= 0)
            or not np.all(np.isfinite(volume))):
        raise ValueError("volume_cm3 must be positive, finite, scalar or one per cell")
    if (not np.isfinite(dt_s) or dt_s < 0 or temperature != 1e4
            or not np.isfinite(max_fraction) or not 0 < max_fraction <= .5
            or not isinstance(max_substeps, (int, np.integer)) or max_substeps < 1):
        raise ValueError("Require dt_s>=0, T=10000 K, 0<max_fraction<=0.5 and positive max_substeps")
    alpha, _, _ = coefficients_1e4()
    n = initial.copy()
    captures = np.zeros((len(n), 3))
    pure_h = (n[:, 3] == 0) & (n[:, 4] == 0)
    # z/(1+z) preserves tiny captures; the reciprocal preserves a tiny HII
    # tail at very long times. Neither is computed by subtracting the other.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = alpha[0] * (n[pure_h, 1]/volume[pure_h]) * dt_s
        survival = 1/(1+z)
        loss = np.divide(z, 1+z, out=np.ones_like(z), where=np.isfinite(z))
    captures[pure_h, 0] = n[pure_h, 1]*loss
    n[pure_h, 0] += captures[pure_h, 0]
    n[pure_h, 1] *= survival

    active_rows = np.flatnonzero(~pure_h)
    elapsed, steps = 0., 0
    while len(active_rows) and elapsed < dt_s:
        if steps >= max_substeps:
            raise RuntimeError("Recombination substep limit; no caller state was mutated")
        state = n[active_rows]
        ne = _electrons(state)/volume[active_rows]
        if not np.all(np.isfinite(ne)):
            raise ValueError("Electron density overflow")
        max_rate = float(np.max(ne))*float(alpha.max())
        if max_rate == 0:
            break
        step = min(float(dt_s)-elapsed, max_fraction/max_rate)
        if not np.isfinite(step) or step <= 0 or elapsed+step == elapsed:
            raise RuntimeError("Recombination time resolution exhausted")
        midpoint, _ = _downward(state, ne, .5*step, alpha)
        midpoint_ne = _electrons(midpoint)/volume[active_rows]
        next_state, count = _downward(state, midpoint_ne, step, alpha)
        n[active_rows] = next_state
        captures[active_rows] += count
        elapsed += step
        steps += 1

    before = np.column_stack((initial[:, :2].sum(1), initial[:, 2:].sum(1)))
    after = np.column_stack((n[:, :2].sum(1), n[:, 2:].sum(1)))
    charge_loss = _electrons(initial)-_electrons(n)
    charge_error = charge_loss-captures.sum(1)
    nucleus_error = after-before
    scale = np.maximum(initial.sum(1), np.finfo(float).tiny)
    if (not np.all(np.isfinite(n)) or np.any(n < 0)
            or not np.all(np.isfinite(captures)) or np.any(captures < 0)):
        raise RuntimeError("Invalid recombination state")
    if (np.any(abs(charge_error) > 1e-10*scale)
            or np.any(abs(nucleus_error) > 1e-10*scale[:, None])):
        raise RuntimeError("Recombination nucleus/electron ledger mismatch")
    return dict(gas=n, captures=captures,
                diffuse_emitted=emit_from_captures(captures),
                charge_error=charge_error, nucleus_error=nucleus_error,
                substeps=steps, duration_s=float(dt_s), temperature_K=float(temperature),
                closure="case-A captures; diffuse photons returned for local accounting")


def _absorb_local(gas, photons, donor_fraction, count_tolerance):
    """Consume only photons already assigned to each physical cell."""
    gas, photons = gas.copy(), photons.copy()
    absorbed = np.zeros((len(gas), 3))
    channel_debit = np.zeros_like(photons)
    # Exact, common production case: H-ground photons cannot ionize helium.
    soft = np.all(photons[:, 1:] == 0, axis=1)
    amount = np.minimum(gas[soft, 0], photons[soft, 0])
    gas[soft, 0] -= amount
    gas[soft, 1] += amount
    photons[soft, 0] -= amount
    absorbed[soft, 0] = amount
    channel_debit[soft, 0] = amount
    rows = np.flatnonzero(~soft)
    if len(rows):
        result = illuminate_channels(
            gas[rows], photons[rows].T, cross_sections(OPACITY_E_EV),
            stop_at_event=False, donor_fraction=donor_fraction,
            count_tolerance=count_tolerance,
        )
        gas[rows] = result['gas']
        photons[rows] = result['target_credits'].T
        absorbed[rows] = result['absorbed']
        channel_debit[rows] = result['channel_debit'].T
    return gas, photons, absorbed, channel_debit


def evolve_local(gas, volume_cm3, dt_s, *, pending=None, temperature=1e4,
                 max_fraction=.05, max_step_s=None, max_substeps=100000,
                 absorption_donor_fraction=.02, count_tolerance=1e-12):
    """Advance case-A recombination plus the existing local coupled-OTS model.

    ``pending`` is (cells,8) diffuse photon counts from earlier steps. All
    unabsorbed counts are returned at the same physical cells and channel;
    zero opacity never discards them. Each physical substep recombines once,
    emits the eight channels and performs competing local photoabsorption.
    The general OTS/recombination splitting is first-order in physical step
    size; it is controlled by ``max_fraction`` and optional ``max_step_s``.
    Calling this function for half steps does not remove its internal error.

    Where hydrogen supplies all electrons and no photons remain after the
    initial local absorption, the instantaneous H-ground OTS limit is solved
    analytically with case B. Case-A captures and their ground emission are
    still recorded separately, so this optimization preserves all ledgers.

    Counts and volumes may share any constant normalization: replacing both
    atom counts and physical volumes by counts/A and volume/A leaves physical
    number densities and the evolution unchanged. Returned counts then use
    the same normalization A. This permits a mean-cell-H-atom normalization.
    """
    # Validate species, physical controls and shape using the no-time map.
    checked = recombine(gas, volume_cm3, 0., temperature=temperature,
                        max_fraction=max_fraction, max_substeps=max_substeps)
    initial = checked['gas']
    if not np.isfinite(dt_s) or dt_s < 0:
        raise ValueError("dt_s must be finite and nonnegative")
    if max_step_s is not None and (not np.isfinite(max_step_s) or max_step_s <= 0):
        raise ValueError("max_step_s must be positive and finite")
    if (not np.isfinite(absorption_donor_fraction)
            or not 0 < absorption_donor_fraction < 1
            or not np.isfinite(count_tolerance) or not 0 < count_tolerance < 1):
        raise ValueError("Invalid local photoabsorption controls")
    photons0 = (np.zeros((len(initial), 8)) if pending is None
                else np.array(pending, dtype=float, copy=True))
    if (photons0.shape != (len(initial), 8) or np.any(photons0 < 0)
            or not np.all(np.isfinite(photons0))):
        raise ValueError("pending must be finite nonnegative (cells,8) counts")
    volume = np.broadcast_to(np.asarray(volume_cm3, dtype=float), (len(initial),))
    gas, photons, absorbed, debit = _absorb_local(
        initial, photons0, absorption_donor_fraction, count_tolerance)
    captures = np.zeros((len(initial), 3))
    emitted = np.zeros_like(photons)
    alpha_a, alpha_b, _ = coefficients_1e4()
    exact_h = ((gas[:, 3] == 0) & (gas[:, 4] == 0)
               & np.all(photons == 0, axis=1))
    with np.errstate(over='ignore', invalid='ignore'):
        z = alpha_b[0]*(gas[exact_h, 1]/volume[exact_h])*dt_s
        loss = np.divide(z, 1+z, out=np.ones_like(z), where=np.isfinite(z))
    net_h = gas[exact_h, 1]*loss
    gas[exact_h, 0] += net_h
    gas[exact_h, 1] /= 1+z
    captures[exact_h, 0] = net_h*(alpha_a[0]/alpha_b[0])
    emitted[exact_h] = emit_from_captures(captures[exact_h])
    absorbed[exact_h, 0] += emitted[exact_h, 0]
    debit[exact_h, 0] += emitted[exact_h, 0]

    rows = np.flatnonzero(~exact_h)
    elapsed, steps = 0., 0
    while len(rows) and elapsed < dt_s:
        if steps >= max_substeps:
            raise RuntimeError("Local OTS substep limit; no caller state was mutated")
        ne = _electrons(gas[rows])/volume[rows]
        max_rate = float(ne.max())*float(alpha_a.max())
        step = min(float(dt_s)-elapsed,
                   max_fraction/max_rate if max_rate > 0 else float(dt_s)-elapsed,
                   max_step_s if max_step_s is not None else float(dt_s)-elapsed)
        if not np.isfinite(step) or step <= 0 or elapsed+step == elapsed:
            raise RuntimeError("Local OTS time resolution exhausted")
        reaction = recombine(gas[rows], volume[rows], step,
                             temperature=temperature, max_fraction=max_fraction,
                             max_substeps=max_substeps)
        gas[rows] = reaction['gas']
        captures[rows] += reaction['captures']
        emitted[rows] += reaction['diffuse_emitted']
        photons[rows] += reaction['diffuse_emitted']
        new_gas, new_photons, new_absorbed, new_debit = _absorb_local(
            gas[rows], photons[rows], absorption_donor_fraction, count_tolerance)
        gas[rows], photons[rows] = new_gas, new_photons
        absorbed[rows] += new_absorbed
        debit[rows] += new_debit
        elapsed += step
        steps += 1

    photon_error = photons0.sum(1)+emitted.sum(1)-debit.sum(1)-photons.sum(1)
    charge_error = (_electrons(gas)-_electrons(initial)
                    -absorbed.sum(1)+captures.sum(1))
    nuclei0 = np.column_stack((initial[:, :2].sum(1), initial[:, 2:].sum(1)))
    nuclei1 = np.column_stack((gas[:, :2].sum(1), gas[:, 2:].sum(1)))
    nucleus_error = nuclei1-nuclei0
    gas_scale = np.maximum(initial.sum(1), np.finfo(float).tiny)
    electron_scale = np.maximum(gas_scale, np.maximum(captures.sum(1), absorbed.sum(1)))
    photon_scale = np.maximum(photons0.sum(1)+emitted.sum(1), np.finfo(float).tiny)
    if (np.any(abs(photon_error) > 1e-10*photon_scale)
            or np.any(abs(charge_error) > 1e-10*electron_scale)
            or np.any(abs(nucleus_error) > 1e-10*gas_scale[:, None])
            or np.any(abs(debit.sum(1)-absorbed.sum(1)) > 1e-10*photon_scale)):
        raise RuntimeError("Local OTS photon/nucleus/electron ledger mismatch")
    if any(np.any(x < 0) or not np.all(np.isfinite(x))
           for x in (gas, photons, captures, emitted, absorbed, debit)):
        raise RuntimeError("Invalid local OTS state")
    return dict(gas=gas, pending=photons, captures=captures,
                diffuse_emitted=emitted, diffuse_absorbed=absorbed,
                channel_debit=debit, photon_error=photon_error,
                charge_error=charge_error, nucleus_error=nucleus_error,
                substeps=steps, exact_h_cells=int(exact_h.sum()),
                duration_s=float(dt_s), temperature_K=float(temperature),
                closure="case-A plus local coupled OTS; unabsorbed photons retained")
