"""Positive H/He chemistry for a *frozen* angular-front allocation.

This module does not route photons.  A routing sweep must first split each
source/angle/energy credit into separate target-cell credits and a boundary
credit.  Unspent target credits retain both their resource and target identity.
After a transparency event they must be continued from that target, rather
than redistributed from the original source.  This contract prevents a fixed
route from spending a boundary allocation or stealing another target's share.

The closure within one cell is a common field with cross-section-weighted
competition between all photon channels.  Positive exponential midpoint
steps use the local H/He upward reaction map.  This is a chemistry
approximation, not the constant-arrival-rate H-only event solver, and carries
no V20 global time-sandwich claim.
"""
import numpy as np

if __package__:
    from . import chemistry as _chemistry
else:
    import chemistry as _chemistry


def _reference():
    """Return this method's local chemistry module (NumPy only)."""
    return _chemistry


def band_capacities(gas):
    """Remaining soft, intermediate, and hard photoelectron capacities."""
    n = np.asarray(gas, dtype=float)
    return np.column_stack((n[:, 0], n[:, 0] + n[:, 2],
                            n[:, 0] + 2*n[:, 2] + n[:, 3]))


def band_normalizations(gas):
    n = np.asarray(gas, dtype=float)
    hydrogen = n[:, :2].sum(axis=1)
    helium = n[:, 2:].sum(axis=1)
    return np.column_stack((hydrogen, hydrogen + helium, hydrogen + 2*helium))


def _array(value, name, ndim):
    out = np.array(value, dtype=float, copy=True)
    if out.ndim != ndim or np.any(out < 0) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite, nonnegative and {ndim}-dimensional")
    return out


def split_front_credits(credits, route):
    """Split once; return target(resources,cells) and boundary(resources).

    Sparse matrices exposing ``toarray`` are accepted for this research
    implementation.  It intentionally materializes the target-channel
    inventory; a production version should preserve only nonzero channels.
    Rows whose sums exceed one beyond roundoff are rejected.  A roundoff-only
    overdraw is removed from the largest target credit, not from the gas.
    ``illuminate_frozen_front`` reports the resulting allocation correction.
    """
    p = _array(credits, "credits", 1)
    a = _array(route.toarray() if hasattr(route, "toarray") else route, "route", 2)
    if a.shape[0] != len(p):
        raise ValueError("One route row is required per resource")
    rows = np.array([sum(map(float, row)) for row in a])
    if np.any(rows > 1+16*np.finfo(float).eps):
        raise ValueError("Route target fractions exceed a resource budget")
    targets = p[:, None] * a
    # Sum products rather than independently computing p*(1-sum(a)); the
    # subtraction closes the floating-point split ledger without altering
    # any target share except an explicitly bounded rounding correction.
    target_total = targets.sum(axis=1)
    for row in np.flatnonzero(target_total > p):
        excess = target_total[row]-p[row]
        if excess > 32*np.finfo(float).eps*p[row]:
            raise ValueError("Target credits overdraw their resource")
        largest = int(np.argmax(targets[row]))
        targets[row, largest] -= excess
        for _ in range(8):
            target_total[row] = targets[row].sum()
            if target_total[row] <= p[row]:
                break
            targets[row, largest] = np.nextafter(targets[row, largest], 0.)
        else:
            raise ValueError("Could not resolve target split roundoff")
    boundary = p - target_total
    return targets, boundary


def illuminate_channels(gas, target_credits, cross, *, epsilon=1e-5,
                        donor_fraction=.02, count_tolerance=1e-13,
                        photon_floor=0., max_iterations=20000,
                        stop_at_event=True):
    """Consume fixed per-target spectral credits without moving any photon.

    ``cross`` has shape (resources, 3), for HI, HeI, HeII.  Its zeros must
    describe physical nested bands: HI; HI+HeI; or HI+HeI+HeII.  A completely
    nonabsorbing resource is allowed and is retained.  Each energy group may
    have arbitrarily many source/angle resources with the same cross vector.

    Return gas, remaining target credits, species capture counts, per-channel
    debits, transparency events and conservation diagnostics.  Events are
    (cell, band_class) pairs.  The threshold changes only routing eligibility;
    it never deletes residual neutral gas.  ``stop_at_event=False`` is useful
    for isolated chemistry tests, not for evolving a spatial front.
    """
    initial = _array(gas, "gas", 2)
    p0 = _array(target_credits, "target_credits", 2)
    sigma = _array(cross, "cross", 2)
    if (initial.shape[1] != 5 or sigma.shape != (p0.shape[0], 3)
            or p0.shape[1] != len(initial)):
        raise ValueError("Expected gas(cells,5), credits(resources,cells), cross(resources,3)")
    if (not np.isfinite(epsilon) or not 0 < epsilon < 1
            or not np.isfinite(donor_fraction) or not 0 < donor_fraction < 1
            or not np.isfinite(count_tolerance) or not 0 < count_tolerance < 1
            or not np.isfinite(photon_floor) or photon_floor < 0
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("Invalid chemistry controls")
    pattern = sigma > 0
    if np.any(pattern[:, 1] & ~pattern[:, 0]) or np.any(pattern[:, 2] & ~pattern[:, 1]):
        raise ValueError("Cross sections must describe nested H/He threshold bands")
    classes = pattern.sum(axis=1) - 1
    n, p = initial.copy(), p0.copy()
    absorbed = np.zeros((len(n), 3))
    debit_total = np.zeros_like(p)
    scale = float(sigma.max()) if sigma.size else 0.
    if scale:
        sigma /= scale
    normal = band_normalizations(initial)
    threshold = epsilon * normal
    available = band_capacities(initial)
    enabled = np.zeros_like(available, dtype=bool)
    for k in range(3):
        enabled[:, k] = np.any(p0[classes == k] > 0, axis=0)
    eligible = enabled & (normal > 0) & (
        available > threshold + 64*np.finfo(float).eps*normal)
    resource_capacity = np.zeros_like(p)
    for k in range(3):
        resource_capacity[classes == k] = available[:, k]
    residual = np.maximum(photon_floor,
                          count_tolerance*np.minimum(p0, resource_capacity))
    evaluations = 0
    accepted_steps = 0
    exposure = 0.
    events = []
    reason = "no_absorption"
    chemistry = _chemistry.upward_cells

    def finish():
        amount = float(absorbed.sum())
        debit_amount = float(debit_total.sum())
        scale_count = max(float(initial.sum()), amount, np.finfo(float).tiny)
        error = debit_amount - amount
        nuclei_before = np.column_stack((initial[:, :2].sum(1), initial[:, 2:].sum(1)))
        nuclei_after = np.column_stack((n[:, :2].sum(1), n[:, 2:].sum(1)))
        charge = float((n[:, 1] + n[:, 3] + 2*n[:, 4]
                        - initial[:, 1] - initial[:, 3] - 2*initial[:, 4]).sum())
        if abs(error) > 8e-12*max(amount, np.finfo(float).tiny):
            raise RuntimeError("Angular chemistry photon/capture ledger mismatch")
        if np.any(abs(nuclei_after-nuclei_before) >
                  8e-12*np.maximum(nuclei_before, np.finfo(float).tiny)):
            raise RuntimeError("Angular chemistry nucleus ledger mismatch")
        if abs(charge-amount) > 8e-12*scale_count:
            raise RuntimeError("Angular chemistry electron ledger mismatch")
        if any(np.any(a < 0) or not np.all(np.isfinite(a))
               for a in (n, p, absorbed, debit_total)):
            raise RuntimeError("Angular chemistry produced invalid state")
        return dict(gas=n, target_credits=p, absorbed=absorbed,
                    channel_debit=debit_total,
                    resource_debit=debit_total.sum(axis=1),
                    events=events, stop=reason, exposure=exposure,
                    evaluations=evaluations, substeps=accepted_steps,
                    photon_error=error, charge_error=charge-amount)

    if not scale or not np.any(p > 0):
        return finish()

    def field(light):
        return light.T @ sigma

    def capture(state, light, rates, step):
        nonlocal evaluations
        evaluations += 1
        new, counts = chemistry(state, rates, step)
        response = np.divide(counts, rates, out=np.zeros_like(counts), where=rates > 0)
        debit = light * (sigma @ response.T)
        return new, counts, debit

    for _ in range(max_iterations):
        hazard = sigma @ n[:, [0, 2, 3]].T
        active = (p > residual) & (hazard > 0)
        active &= .5*donor_fraction*p >= np.nextafter(0., 1.)
        if not np.any(active):
            reason = "absorber_saturation" if not np.any(hazard > 0) else "photon_remainder"
            return finish()
        light = np.where(active, p, 0.)
        rates = field(light)
        step = donor_fraction / float(np.max(hazard[active]))
        largest_rate = float(rates.max())
        if largest_rate > 0:
            step = min(step, 10./largest_rate)
        if not np.isfinite(step) or step <= 0:
            reason = "numerical_remainder"
            return finish()

        def attempt(dt):
            _, _, half_debit = capture(n, light, rates, .5*dt)
            if np.any(half_debit > .5*donor_fraction*p):
                return None
            middle = light-half_debit
            if np.any(middle < 0):
                return None
            new, counts, debit = capture(n, middle, field(middle), dt)
            if np.any(debit > donor_fraction*p):
                return None
            return new, counts, debit

        for _backtrack in range(100):
            result = attempt(step)
            if result is not None:
                break
            step *= .5
        else:
            raise RuntimeError("No positive angular chemistry midpoint step")
        next_n, counts, debit = result
        if not np.any(debit > 0) and np.array_equal(next_n, n):
            reason = "numerical_remainder"
            return finish()
        hit = eligible & (band_capacities(next_n) <= threshold)
        if stop_at_event and np.any(hit):
            low, high = 0., step
            for _bisect in range(64):
                middle = low + .5*(high-low)
                if middle == low or middle == high:
                    break
                trial = attempt(middle)
                if trial is None:
                    raise RuntimeError("Angular chemistry event lost feasibility")
                if np.any(eligible & (band_capacities(trial[0]) <= threshold)):
                    high = middle
                else:
                    low = middle
            step = high
            next_n, counts, debit = attempt(step)
            hit = eligible & (band_capacities(next_n) <= threshold)
        n = next_n
        p -= debit
        absorbed += counts
        debit_total += debit
        accepted_steps += 1
        exposure += step
        if stop_at_event and np.any(hit):
            events = np.argwhere(hit).tolist()
            reason = "transparency_event"
            return finish()
    raise RuntimeError("Angular chemistry iteration limit; no caller state was mutated")


def illuminate_frozen_front(gas, credits, cross, route, **options):
    """One route split followed by local chemistry; boundary never participates."""
    targets, boundary = split_front_credits(credits, route)
    result = illuminate_channels(gas, targets, cross, **options)
    result["boundary_credits"] = boundary
    result["target_credits_initial"] = targets
    raw_route = route.toarray() if hasattr(route, "toarray") else route
    ideal_targets = np.asarray(credits, dtype=float)[:, None]*np.asarray(raw_route, dtype=float)
    result["split_roundoff_l1"] = np.abs(targets-ideal_targets).sum(axis=1)
    result["resource_retained"] = result["target_credits"].sum(axis=1)+boundary
    original = np.asarray(credits, dtype=float)
    result["resource_budget_error"] = (
        result["resource_retained"]+result["resource_debit"]-original)
    return result
