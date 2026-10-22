"""Finite-speed periodic continuation of the angular-face allocation model.

Geometry is the existing C99 face/cone intersection. Each inventory entry
keeps its emission cell, angular bin, energy and unwrapped current location.
Wrapping is used only to look up the one shared physical gas cell.

``advance`` is a first-order physical-time split: all new photons are injected
at the beginning, transported to the old opaque front or the causal horizon,
then absorbed together. Newly transparent cells release their remaining light
onto outgoing edges, without further travel in this step. Recombination and
density updates belong to the caller. The open-front global time-sandwich
theorem is not claimed for this time-dependent extension.
"""
from collections import defaultdict
import ctypes as ct
from functools import lru_cache
import heapq
import math
import os
from pathlib import Path

import numpy as np

if __package__:
    from .angular_hhe import band_capacities, band_normalizations, illuminate_channels
    from .chemistry import cross_sections
else:
    from angular_hhe import band_capacities, band_normalizations, illuminate_channels
    from chemistry import cross_sections


class _Bin(ct.Structure):
    _fields_ = [("vertices", ct.c_double * 9), ("inward_normals", ct.c_double * 9),
                ("solid_angle", ct.c_double)]


def _finite(value, name, ndim=None):
    result = np.array(value, dtype=float, copy=True)
    if (ndim is not None and result.ndim != ndim) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid dimensions or nonfinite values")
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _split(total, weights):
    """Nonnegative split with one explicitly balanced rounding remainder."""
    if not weights:
        if total != 0:
            raise RuntimeError("Cannot allocate positive light to no outgoing face")
        return []
    denominator = math.fsum(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Invalid angular allocation weights")
    largest = max(range(len(weights)), key=weights.__getitem__)
    parts = [total * (weight / denominator) for weight in weights]
    # Balance the largest share; placing the residual on a tiny share can
    # create a negative component even for harmless roundoff.
    parts[largest] = total - math.fsum(part for i, part in enumerate(parts) if i != largest)
    if parts[largest] < 0 or not all(math.isfinite(x) and x >= 0 for x in parts):
        raise RuntimeError("Nonpositive angular split after roundoff balancing")
    return parts


class _Geometry:
    def __init__(self, level, library_path):
        path = Path(library_path or os.environ.get("ES_ANGULAR_LIBRARY", "") or
                    Path(__file__).with_name("libangular_front.so")).resolve()
        if not path.is_file():
            raise RuntimeError("Build tools/es_angular/build.py before using periodic transport")
        self.lib = ct.CDLL(str(path))
        self.lib.ag_bin_count.argtypes = [ct.c_uint]
        self.lib.ag_bin_count.restype = ct.c_size_t
        self.lib.ag_make_bins.argtypes = [ct.c_uint, ct.POINTER(_Bin), ct.c_size_t]
        self.lib.ag_make_bins.restype = ct.c_int
        self.lib.ag_face_bin_solid_angle.argtypes = [ct.POINTER(ct.c_double), ct.c_int,
                                                    ct.c_double, ct.POINTER(_Bin)]
        self.lib.ag_face_bin_solid_angle.restype = ct.c_double
        count = self.lib.ag_bin_count(level)
        if not count:
            raise ValueError("Invalid angular level")
        self.bins = (_Bin * count)()
        if self.lib.ag_make_bins(level, self.bins, count):
            raise RuntimeError("Cannot initialize angular bins")
        self.fractions = np.array([b.solid_angle / (4 * math.pi) for b in self.bins])

    @lru_cache(maxsize=65536)
    def outgoing(self, angular_bin, offset):
        """Use half-integer coordinates, as in the validated open solver."""
        result = []
        radius = math.sqrt(sum(x * x for x in offset))
        for axis in range(3):
            for sign in (-1, 1):
                if offset[axis] * sign < 0:
                    continue
                face = list(offset)
                face[axis] += .5 * sign
                omega = self.lib.ag_face_bin_solid_angle(
                    (ct.c_double * 3)(*face), axis, .5, ct.byref(self.bins[angular_bin]))
                if not math.isfinite(omega) or omega < 0:
                    raise RuntimeError("Invalid angular-face solid angle")
                if omega == 0:
                    continue
                target = list(offset)
                target[axis] += sign
                target = tuple(target)
                travel = math.sqrt(sum(x * x for x in target)) - radius
                if travel <= 0 or not math.isfinite(travel):
                    raise RuntimeError("Unresolved radial edge length")
                result.append((target, travel, omega))
        if not result:
            raise RuntimeError("Positive photon inventory has no outgoing angular support")
        return tuple(result)


class PeriodicFront:
    """Sparse source/angle/energy inventory coupled to shared H/He gas.

    Gas is ``(prod(shape),5)`` in HI,HII,HeI,HeII,HeIII order. Counts may
    share one fixed normalization with photons. ``travel_cells`` is the
    proper propagation distance divided by the current proper cell size.
    ``dt_s`` records the caller's physical clock; no recombination occurs here.
    """
    def __init__(self, shape, gas, energies_eV, angular_level=0, epsilon=.1,
                 transparency=1e-5, library_path=None, max_entries=2000000,
                 max_route_nodes=1000000):
        raw_shape = np.asarray(shape)
        if raw_shape.shape != (3,) or raw_shape.dtype.kind not in "iu" or np.any(raw_shape < 1):
            raise ValueError("shape must contain three positive integers")
        self.shape = tuple(int(x) for x in raw_shape)
        self.ncell = math.prod(self.shape)
        if (not isinstance(angular_level, (int, np.integer)) or isinstance(angular_level, bool)
                or not 0 <= angular_level <= 8):
            raise ValueError("angular_level must be an integer in [0,8]")
        self.angular_level = int(angular_level)
        self.epsilon, self.transparency = float(epsilon), float(transparency)
        if (not math.isfinite(self.epsilon) or self.epsilon < 0
                or not math.isfinite(self.transparency) or not 0 < self.transparency < 1):
            raise ValueError("Invalid ES or transparency controls")
        for name, value in (("max_entries", max_entries), ("max_route_nodes", max_route_nodes)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        self.energies_eV = _finite(energies_eV, "energies_eV", 1)
        if not len(self.energies_eV) or np.any(self.energies_eV < 13.6):
            raise ValueError("Every primary photon group must be ionizing (>=13.6 eV)")
        self.cross = cross_sections(self.energies_eV)
        self.classes = np.sum(self.cross > 0, axis=1) - 1
        self.geometry = _Geometry(self.angular_level, library_path)
        self.nbins = len(self.geometry.bins)
        self._source_xyz = np.array(np.unravel_index(np.arange(self.ncell), self.shape)).T
        self.set_gas(gas)
        # key: source, bin, energy, offset x/y/z, remaining edge distance.
        self.entries = {}
        self.emitted = {}
        self.absorbed = {}
        self.time_s = 0.
        self.step_count = 0

    def set_gas(self, gas):
        value = _finite(gas, "gas", 2)
        if value.shape != (self.ncell, 5):
            raise ValueError("gas must have shape (prod(shape),5)")
        self.gas = value

    def _cell(self, source, offset):
        p = self._source_xyz[source]
        x, y, z = ((int(p[k]) + offset[k]) % self.shape[k] for k in range(3))
        return (x * self.shape[1] + y) * self.shape[2] + z

    def _opaque(self, gas):
        normal = band_normalizations(gas)
        # Match the fixed-front kernel's roundoff band at transparency.
        return band_capacities(gas) > (self.transparency + 64*np.finfo(float).eps) * normal

    def _absorb_front(self, gas, pool):
        """Relax fixed target credits; no photon travels during this operation.

        Identical H-only cross sections share one depletion factor per cell.
        The endpoint is therefore an exact capacity-limited proportional
        debit. Different cross sections or He-active light use the coupled
        spectral kernel, restricted to cells that actually receive photons.
        """
        if not np.all(np.isfinite(pool)):
            raise RuntimeError("Nonfinite angular target photon budget")
        captures = np.zeros((self.ncell, 3))
        debits = np.zeros_like(pool)
        cells = np.flatnonzero(np.any(pool > 0, axis=0))
        diagnostics = dict(chemistry_method="none", chemistry_active_cells=len(cells),
                           chemistry_substeps=0, transparency_events=0)
        if not len(cells):
            return gas, captures, debits, diagnostics
        local = gas[cells].copy()
        credits = pool[:, cells].copy()
        credits = np.where(self._opaque(local)[:, self.classes].T, credits, 0.)
        active_bands = np.flatnonzero(np.any(credits > 0, axis=1))
        if not len(active_bands):
            return gas, captures, debits, diagnostics
        active_cross = self.cross[active_bands]
        if (np.all(active_cross[:, 0] > 0) and not np.any(active_cross[:, 1:])
                and np.all(active_cross[:, 0] == active_cross[0, 0])):
            diagnostics["chemistry_method"] = "analytic_h"
            diagnostics["chemistry_substeps"] = 1
            for index, cell in enumerate(cells):
                # Match the iterative kernel's representability guard. Keep
                # unresolved subnormal light in the inventory, never drop it.
                bands = np.flatnonzero(.01*credits[:, index] >= np.nextafter(0., 1.))
                if not len(bands):
                    continue
                amounts = credits[bands, index].tolist()
                available = math.fsum(amounts)
                normal = math.fsum(local[index, :2])
                capacity = max(0., local[index, 0] - self.transparency*normal)
                capture = min(available, capacity)
                pieces = amounts if capture == available else _split(capture, amounts)
                debits[bands, cell] = pieces
                captures[cell, 0] = capture
                local[index, 0] -= capture
                local[index, 1] += capture
                diagnostics["transparency_events"] += int(capture == capacity)
        else:
            diagnostics["chemistry_method"] = "spectral_midpoint"
            for _ in range(3*len(cells) + 1):
                mask = self._opaque(local)[:, self.classes].T
                credits = np.where(mask, credits, 0.)
                if not np.any(credits > 0):
                    break
                reaction = illuminate_channels(local, credits, self.cross,
                                               epsilon=self.transparency,
                                               stop_at_event=True, max_iterations=40000)
                local = reaction['gas']
                captures[cells] += reaction['absorbed']
                debits[:, cells] += reaction['channel_debit']
                credits = reaction['target_credits']
                diagnostics['chemistry_substeps'] += reaction['substeps']
                if reaction['stop'] != 'transparency_event':
                    break
                diagnostics['transparency_events'] += len(reaction['events'])
            else:
                raise RuntimeError("Too many monotone angular transparency events")
        gas[cells] = local
        return gas, captures, debits, diagnostics

    def _route(self, key, amount, travel, opaque, scores, node_budget=None):
        source, angular_bin, energy, x, y, z, remaining = key
        offset = (x, y, z)
        if remaining > travel:
            return {(*key[:6], remaining - travel): amount}, 0
        travel -= remaining
        horizon = math.sqrt(sum(x*x for x in offset)) + travel
        flow = {offset: amount}
        queue = [(math.sqrt(sum(x*x for x in offset)), offset)]
        frontier, flying = defaultdict(float), defaultdict(float)
        visits = 0
        while queue:
            radius, location = heapq.heappop(queue)
            value = flow.pop(location)
            visits += 1
            if visits > (self.max_route_nodes if node_budget is None else node_budget):
                raise RuntimeError("Angular routing node limit exceeded; increase max_route_nodes")
            cell = self._cell(source, location)
            if opaque[cell, self.classes[energy]]:
                frontier[location] += value
                continue
            edges = self.geometry.outgoing(angular_bin, location)
            for (target, distance, weight), part in zip(edges, _split(value, [e[2] for e in edges])):
                if part == 0:
                    continue
                target_radius = math.sqrt(sum(x*x for x in target))
                if target_radius > horizon:
                    flying[(*target, target_radius-horizon)] += part
                else:
                    if target not in flow:
                        heapq.heappush(queue, (target_radius, target))
                        flow[target] = part
                    else:
                        flow[target] += part
            if len(flow) + len(frontier) + len(flying) > self.max_entries:
                raise RuntimeError("Angular routing inventory limit exceeded")
        if self.epsilon and frontier:
            locations = list(frontier)
            total = math.fsum(frontier.values())
            eta = self.epsilon / (2 + self.epsilon)
            base = 2 / (2 + self.epsilon)
            weights = []
            maximum = max(frontier.values())
            for location in locations:
                score = float(scores[self._cell(source, location)])
                modulation = base + 2*eta*(score/(1+score))
                # Normalize before modulation, retaining even subnormal
                # cohorts rather than underflowing all their ES weights.
                weights.append((frontier[location]/maximum) * modulation)
            parts = _split(total, weights)
            frontier = dict(zip(locations, parts))
        result = { (source, angular_bin, energy, *location, 0.): value
                   for location, value in frontier.items() if value > 0 }
        result.update({(source, angular_bin, energy, *location): value
                       for location, value in flying.items() if value > 0})
        before, after = amount, math.fsum(result.values())
        if abs(after-before) > 2e-11*max(before, np.finfo(float).tiny):
            raise RuntimeError("Angular routing failed photon-count audit")
        return result, visits

    def advance(self, source_photons, spectrum_weights, travel_cells, es_scores=None, *, dt_s=0.):
        """Advance atomically; an exception leaves gas, light and ledgers intact."""
        gas = _finite(self.gas, "gas", 2)
        if gas.shape != (self.ncell, 5):
            raise ValueError("Gas shape changed")
        photons = _finite(source_photons, "source_photons", 1)
        weights = _finite(spectrum_weights, "spectrum_weights", 1)
        if photons.shape != (self.ncell,) or weights.shape != self.energies_eV.shape:
            raise ValueError("Source or spectrum dimensions do not match")
        if not math.isclose(math.fsum(weights), 1., rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("spectrum_weights must sum to one")
        travel_cells, dt_s = float(travel_cells), float(dt_s)
        if not all(math.isfinite(v) and v >= 0 for v in (travel_cells, dt_s)):
            raise ValueError("Travel distance and physical step must be finite and nonnegative")
        if not math.isfinite(self.time_s + dt_s):
            raise ValueError("Physical clock overflow")
        scores = np.ones(self.ncell) if es_scores is None else _finite(es_scores, "es_scores", 1)
        if scores.shape != (self.ncell,):
            raise ValueError("One ES score per physical cell is required")
        pending = dict(self.entries)
        emitted, absorbed_resources = dict(self.emitted), dict(self.absorbed)
        resource_weights = [float(f*w) for f in self.geometry.fractions for w in weights]
        for source in np.flatnonzero(photons):
            source = int(source)
            parts = _split(float(photons[source]), resource_weights)
            for g in range(self.nbins):
                for energy in range(len(weights)):
                    part = parts[g*len(weights)+energy]
                    if part == 0:
                        continue
                    resource = (source, g, energy)
                    key = (*resource, 0, 0, 0, 0.)
                    pending[key] = pending.get(key, 0.) + part
                    emitted[resource] = emitted.get(resource, 0.) + part
        if len(pending) > self.max_entries:
            raise RuntimeError("Angular source inventory limit exceeded")
        opaque = self._opaque(gas)
        routed = defaultdict(float)
        visits = 0
        for key, amount in sorted(pending.items()):
            result, count = self._route(key, amount, travel_cells, opaque, scores,
                                        self.max_route_nodes-visits)
            visits += count
            for destination, share in result.items():
                routed[destination] += share
            if len(routed) > self.max_entries:
                raise RuntimeError("Angular transported inventory limit exceeded")
        groups = defaultdict(list)
        pool = np.zeros((len(weights), self.ncell))
        for key, amount in sorted(routed.items()):
            if key[-1] != 0:
                continue
            cell = self._cell(key[0], key[3:6])
            groups[(key[2], cell)].append((key, amount))
            pool[key[2], cell] += amount
        gas, captures, debits, chemistry_audit = self._absorb_front(gas, pool)
        source_absorbed = np.zeros(self.ncell)
        split_debit = 0.
        for (energy, cell), entries in groups.items():
            debit = float(debits[energy, cell])
            if debit == 0:
                continue
            amounts = [entry[1] for entry in entries]
            total = math.fsum(amounts)
            if debit > total*(1+2e-12):
                raise RuntimeError("Chemistry overdraws a physical-cell photon budget")
            debit = min(debit, total)
            pieces = _split(debit, amounts)
            for (key, amount), used in zip(entries, pieces):
                if used > amount:
                    if used-amount > 16*np.finfo(float).eps*amount:
                        raise RuntimeError("Chemistry overdraws a source-image photon channel")
                    used = amount
                routed[key] = amount-used
                resource = key[:3]
                absorbed_resources[resource] = absorbed_resources.get(resource, 0.)+used
                source_absorbed[key[0]] += used
                split_debit += used
        # Release before the caller's second recombination half-step. The
        # photon now belongs to an outgoing edge and cannot be recaptured in
        # its old cell merely because that cell subsequently recombines.
        final = defaultdict(float)
        opaque = self._opaque(gas)
        for key, amount in sorted(routed.items()):
            if amount == 0:
                continue
            if key[-1] > 0 or opaque[self._cell(key[0], key[3:6]), self.classes[key[2]]]:
                final[key] += amount
                continue
            edges = self.geometry.outgoing(key[1], key[3:6])
            for (target, distance, weight), share in zip(edges, _split(amount, [e[2] for e in edges])):
                if share > 0:
                    final[(*key[:3], *target, distance)] += share
        if len(final) > self.max_entries:
            raise RuntimeError("Angular outgoing inventory limit exceeded")
        pending_resources = defaultdict(float)
        for key, amount in final.items():
            pending_resources[key[:3]] += amount
        absolute = relative = 0.
        for resource, emission in emitted.items():
            error = absorbed_resources.get(resource, 0.) + pending_resources.get(resource, 0.) - emission
            if not math.isfinite(error):
                raise RuntimeError("Nonfinite periodic source/angle/energy photon ledger")
            absolute = max(absolute, abs(error))
            relative = max(relative, abs(error)/max(emission, np.finfo(float).tiny))
        if relative > 5e-10:
            raise RuntimeError("Periodic source/angle/energy photon ledger failed")
        total_capture = float(captures.sum())
        count_error = split_debit-total_capture
        if abs(count_error) > 2e-11*max(total_capture, np.finfo(float).tiny):
            raise RuntimeError("Periodic source debit and species capture ledger failed")
        inventory = math.fsum(final.values())
        audits = dict(resource_ledger_max_abs=absolute, resource_ledger_max_relative=relative,
                      photon_capture_error=count_error, routed_nodes=visits,
                      inventory_entries=len(final), **chemistry_audit,
                      emitted_this_step=float(photons.sum()),
                      absorbed_this_step=total_capture, retained_photons=inventory,
                      escaped_photons=0.)
        self.gas, self.entries = gas, dict(final)
        self.emitted, self.absorbed = emitted, absorbed_resources
        self.time_s += dt_s
        self.step_count += 1
        return dict(gas=gas.copy(), absorbed=captures, source_absorbed=source_absorbed,
                    photon_inventory=inventory, audits=audits)

    def export_state(self):
        """JSON-compatible checkpoint; includes all photon image/edge state."""
        return dict(version=1, shape=list(self.shape), gas=self.gas.tolist(),
                    energies_eV=self.energies_eV.tolist(), angular_level=self.angular_level,
                    epsilon=self.epsilon, transparency=self.transparency,
                    max_entries=self.max_entries, max_route_nodes=self.max_route_nodes,
                    time_s=self.time_s, step_count=self.step_count,
                    entries=[list(key)+[value] for key, value in sorted(self.entries.items())],
                    resource_ledgers=[list(key)+[value, self.absorbed.get(key, 0.)]
                                      for key, value in sorted(self.emitted.items())])

    @classmethod
    def from_state(cls, state, library_path=None):
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError("Unsupported periodic-front checkpoint")
        model = cls(state['shape'], state['gas'], state['energies_eV'],
                    angular_level=state['angular_level'], epsilon=state['epsilon'],
                    transparency=state['transparency'], library_path=library_path,
                    max_entries=state['max_entries'], max_route_nodes=state['max_route_nodes'])
        model.time_s = float(state['time_s'])
        model.step_count = state['step_count']
        if (not math.isfinite(model.time_s) or model.time_s < 0
                or not isinstance(model.step_count, int) or model.step_count < 0):
            raise ValueError("Invalid checkpoint clock")
        def indices(values):
            if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
                raise ValueError("Checkpoint indices must be integers")
            return tuple(map(int, values))
        def resource_ok(resource):
            source, angular_bin, energy = resource
            return (0 <= source < model.ncell and 0 <= angular_bin < model.nbins
                    and 0 <= energy < len(model.energies_eV))
        for row in state['entries']:
            if len(row) != 8:
                raise ValueError("Invalid checkpoint photon row")
            integer = indices(row[:6])
            remaining, amount = map(float, row[6:])
            if (not resource_ok(integer[:3]) or not math.isfinite(remaining) or remaining < 0
                    or not math.isfinite(amount) or amount <= 0):
                raise ValueError("Invalid checkpoint photon value")
            if remaining:
                location = integer[3:6]
                incoming = []
                for axis, coordinate in enumerate(location):
                    if coordinate == 0:
                        continue
                    previous = list(location)
                    previous[axis] -= 1 if coordinate > 0 else -1
                    incoming.extend(distance for target, distance, _ in
                                    model.geometry.outgoing(integer[1], tuple(previous))
                                    if target == location)
                if not incoming or remaining > max(incoming)*(1+64*np.finfo(float).eps):
                    raise ValueError("Checkpoint photon is not on a valid incoming edge")
            key = (*integer, remaining)
            if key in model.entries:
                raise ValueError("Duplicate checkpoint photon location")
            model.entries[key] = amount
        if len(model.entries) > model.max_entries:
            raise ValueError("Checkpoint exceeds inventory limit")
        for row in state['resource_ledgers']:
            if len(row) != 5:
                raise ValueError("Invalid checkpoint ledger row")
            key = indices(row[:3])
            emission, absorption = map(float, row[3:])
            if (not resource_ok(key) or key in model.emitted
                    or not all(math.isfinite(x) and x >= 0 for x in (emission, absorption))):
                raise ValueError("Invalid checkpoint resource ledger")
            model.emitted[key], model.absorbed[key] = emission, absorption
        pending = defaultdict(float)
        for key, value in model.entries.items():
            pending[key[:3]] += value
        if set(pending)-set(model.emitted):
            raise ValueError("Checkpoint contains light with no emission ledger")
        for key, emission in model.emitted.items():
            error = model.absorbed[key]+pending.get(key, 0.)-emission
            if abs(error) > 5e-10*max(emission, np.finfo(float).tiny):
                raise ValueError("Checkpoint photon ledger mismatch")
        return model
