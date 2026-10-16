#!/usr/bin/env python3
"""Replay a Debug=2 photon_inputs_*.hdf5 checkpoint without MPI or SciPy.

Requires Python 3, NumPy, h5py and a C99 compiler (CC, or --cc). The three
native solver sources in this checkout are compiled in a temporary directory.
Recorded tolerances and resource limits are preserved; --max-evaluations and
--max-cuts explicitly override the two shape limits. A failed solve or failed
edge audit returns a nonzero exit status and never substitutes the staged map.

Reproduce the recorded budget, from the repository root:
  python3 tools/replay_photon_conservation.py photon_inputs_052.hdf5 --output replay_052.json

For an explicitly larger diagnostic budget, add --max-evaluations 5000000.
Use --compare-certificate photon_conservation_052.hdf5 to report numerical
differences from an existing allocation with identical problem inputs, or
--save-allocation replay_052.npz to save the audited final arrays.

The independent audit checks transport feasibility and photon inventories.
It does not independently prove the native solver's quadratic optimality.
"""

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np


SZ, D, I = ct.c_size_t, ct.c_double, ct.c_int
PD, PI, PB = ct.POINTER(D), ct.POINTER(I), ct.POINTER(ct.c_ubyte)
Visitor = ct.CFUNCTYPE(None, SZ, SZ, D, ct.c_void_p)


def fields(names, kind):
    return [(name, kind) for name in names.split()]


class Problem(ct.Structure):
    _fields_ = ([('dim', SZ * 3), ('ncell', SZ)] +
                fields('source capacity baseline lower upper radius', PD) +
                [('priority', PI)])


class FlowOptions(ct.Structure):
    _fields_ = (fields('max_edges max_bytes seed_neighbors cut_neighbors', SZ) +
                fields('absolute_tolerance relative_tolerance flow_tolerance', D))


class Priority(ct.Structure):
    _fields_ = ([('priority', I)] +
                fields('prefix_flow retained_in_group baseline_retained_in_group', D))


class FlowMetrics(ct.Structure):
    _fields_ = (
        fields('lower_feasible full_source_budget full_graph_cut_separated', I) +
        fields('sources targets source_target_edges positive_edges separation_rounds '
               'priority_groups bytes_peak', SZ) +
        fields('photons lower_demand absorbed source_residual mincut_capacity '
               'source_cut_budget target_cut_capacity hall_source_excess lower_cut_deficit '
               'lower_stage_flow retention_stage_flow final_stage_flow morphology_l1 '
               'retained_overlap max_source_overspend max_target_excess max_lower_deficit '
               'max_source_relative_residual max_edge_balance_error max_geometry_excess '
               'ledger_error cut_flow_error', D))


class Result(ct.Structure):
    _fields_ = (fields('y source_residual source_spent', PD) +
                fields('cut_source cut_target', PB) +
                [('priority_diagnostics', ct.POINTER(Priority)),
                 ('priority_diagnostics_capacity', SZ), ('visit_edge', Visitor),
                 ('visitor_user', ct.c_void_p), ('metrics', FlowMetrics), ('status', I)])


class ESProblem(ct.Structure):
    _fields_ = [('grid', Problem), ('radii_desc', PD), ('radius_count', SZ),
                ('first_crossing', PI)]


class ESMetrics(ct.Structure):
    _fields_ = (fields('attempts front_seed_cells promotion', SZ) +
                fields('band_width empty_front terminal_full_source_budget', I) +
                fields('terminal_maximum_absorption total_capacity', D))


class ESResult(ct.Structure):
    _fields_ = ([('allocation', Result)] + fields('lower upper radius', PD) +
                [('priority', PI), ('metrics', ESMetrics),
                 ('visit_attempt', ct.c_void_p), ('attempt_user', ct.c_void_p)])


class ShapeOptions(ct.Structure):
    _fields_ = (fields('max_hall_cuts max_master_sweeps max_coordinate_iterations '
                      'max_waterfill_iterations max_evaluations max_bytes', SZ) +
                fields('absolute_tolerance relative_tolerance maximum_multiplier', D))


class ShapeMetrics(ct.Structure):
    _fields_ = (
        fields('status last_oracle_status', I) +
        fields('hall_cuts master_sweeps master_evaluations oracle_calls bytes_peak', SZ) +
        fields('source_budget target_total unused_budget objective evaluated_dual_lower '
               'signed_primal_dual_gap equality_error max_hall_violation '
               'max_priority_violation max_priority_equality_error max_bound_violation '
               'max_complementarity max_oracle_target_error oracle_target_l1_error '
               'max_final_target_correction final_target_correction_l1 tolerance', D) +
        [('full_graph_certificate', I)])


class AdapterContext(ct.Structure):
    # Matches the adapter's reservation, not Python's actual resident memory.
    _fields_ = (fields('snapshot nlocal nradii', I) + fields('ncell workspace_bytes', SZ) +
                fields('started particle_normalization', D) + [('input', PD * 10),
                ('first', PI), ('old_zre', ct.c_void_p), ('old_j21', ct.c_void_p),
                ('radii', D * 4096)])


ABI_TYPES = {
    'pc_problem': Problem, 'pc_options': FlowOptions,
    'pc_priority_diagnostic': Priority, 'pc_metrics': FlowMetrics, 'pc_result': Result,
    'pc_es_problem': ESProblem, 'pc_es_metrics': ESMetrics, 'pc_es_result': ESResult,
    'pc_shape_options': ShapeOptions, 'pc_shape_metrics': ShapeMetrics,
}


def struct_dict(value):
    return {name: getattr(value, name) for name, _ in value._fields_}


def ptr(array, kind=PD):
    return array.ctypes.data_as(kind)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_library(directory, compiler, report):
    physics = Path(__file__).resolve().parents[1] / 'src' / 'physics'
    sources = [physics / (name + '.c') for name in
               ('photon_conservation', 'photon_conservation_geometry', 'photon_conservation_es', 'photon_conservation_shape')]
    # Refuse stale ctypes layouts rather than silently corrupting memory.
    expressions, expected = [], []
    for name, binding in ABI_TYPES.items():
        expressions.append(f'sizeof({name})')
        expected.append(ct.sizeof(binding))
        for field, _ in binding._fields_:
            expressions.append(f'offsetof({name}, {field})')
            expected.append(getattr(binding, field).offset)
    probe = directory / 'abi.c'
    probe.write_text('#include "photon_conservation_es.h"\n'
                     '#include "photon_conservation_shape.h"\n'
                     'size_t pc_replay_abi(size_t i) {\n'
                     'static const size_t a[] = {' + ','.join(expressions) + '};\n'
                     'return i < sizeof(a)/sizeof(a[0]) ? a[i] : (size_t)-1;\n}\n')
    library = directory / 'photon_replay.so'
    command = shlex.split(compiler) + [
        '-std=c99', '-O2', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra',
        '-Werror', '-pedantic', '-shared', '-fPIC', '-I', str(physics),
        *map(str, sources), str(probe), '-lm', '-o', str(library)]
    report['build'] = {'command': command, 'sources_sha256': {
        str(path.relative_to(physics.parent.parent)): file_sha256(path)
        for path in sources + [physics / (name + '.h') for name in
                              ('photon_conservation', 'photon_conservation_es',
                               'photon_conservation_shape')]}}
    start = time.monotonic()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    report['build']['seconds'] = time.monotonic() - start
    if completed.returncode:
        raise RuntimeError('C compilation failed:\n' + completed.stderr)
    lib = ct.CDLL(str(library))
    lib.pc_replay_abi.argtypes = [SZ]
    lib.pc_replay_abi.restype = SZ
    if any(lib.pc_replay_abi(k) != value for k, value in enumerate(expected)):
        raise RuntimeError('ctypes layout differs from the current C headers; update replay bindings')
    lib.pc_default_options.argtypes = [ct.POINTER(FlowOptions)]
    lib.pc_default_options.restype = None
    lib.pc_shape_default_options.argtypes = [ct.POINTER(ShapeOptions)]
    lib.pc_shape_default_options.restype = None
    lib.pc_allocate.argtypes = [ct.POINTER(Problem), ct.POINTER(FlowOptions), ct.POINTER(Result)]
    lib.pc_allocate.restype = I
    lib.pc_es_complete.argtypes = [ct.POINTER(ESProblem), ct.POINTER(FlowOptions),
                                  ct.POINTER(ESResult)]
    lib.pc_es_complete.restype = I
    lib.pc_shape_refine.argtypes = [ct.POINTER(Problem), ct.POINTER(FlowOptions),
                                   ct.POINTER(Result), PD, ct.POINTER(ShapeOptions),
                                   ct.POINTER(Result), ct.POINTER(ShapeMetrics)]
    lib.pc_shape_refine.restype = I
    for name in ('pc_status_string', 'pc_shape_status_string'):
        getattr(lib, name).argtypes = [I]
        getattr(lib, name).restype = ct.c_char_p
    return lib


def integer(value, name, minimum=1):
    number = float(value)
    if not math.isfinite(number) or number != int(number) or not minimum <= number <= sys.maxsize:
        raise ValueError(f'{name} must be an integer in [{minimum}, {sys.maxsize}]')
    return int(number)


def read_checkpoint(path):
    with h5py.File(path, 'r') as handle:
        attrs = {}
        for name, value in handle.attrs.items():
            item = np.asarray(value)
            if item.size != 1:
                raise ValueError(f'checkpoint attribute {name} is not scalar')
            attrs[name] = item.reshape(-1)[0].item()
        required = ('grid_dim', 'shape_enabled', 'tolerance', 'absolute_tolerance',
                    'max_edges', 'memory_mb', 'max_iterations', 'max_cuts')
        for name in required:
            if name not in attrs:
                raise ValueError(f'checkpoint lacks attribute {name}')
        dim = integer(attrs['grid_dim'], 'grid_dim')
        n = dim ** 3
        if n > np.iinfo(np.int32).max:
            raise ValueError('grid exceeds the adapter int-count limit')
        data = {}
        for name in ('source', 'capacity', 'baseline', 'radii_cells', 'first_crossing'):
            if name not in handle:
                raise ValueError(f'missing {name}: replay needs a PhotonConservationDebug=2 checkpoint')
            array = handle[name][:]
            if array.ndim != 1 or (name != 'radii_cells' and len(array) != n):
                raise ValueError(f'invalid {name} dimensions')
            if not np.all(np.isfinite(array)):
                raise ValueError(f'{name} contains a nonfinite value')
            if name == 'first_crossing':
                if np.any(array != np.floor(array)) or np.any(array < -1) or np.any(array > 4095):
                    raise ValueError('invalid first_crossing index')
                data[name] = np.ascontiguousarray(array, dtype=np.int32)
            else:
                if np.any(array < 0):
                    raise ValueError(f'{name} contains a negative value')
                data[name] = np.ascontiguousarray(array, dtype=np.float64)
        radii = data['radii_cells']
        if not 1 <= len(radii) <= 4096 or radii[-1] != 0 or np.any(radii[1:] >= radii[:-1]):
            raise ValueError('radii_cells must strictly decrease and end with the identity radius 0')
        if np.any(data['first_crossing'] >= len(radii)):
            raise ValueError('first_crossing exceeds the recorded filter steps')
    return attrs, dim, data


def read_comparison(path, dim, data):
    """Verify problem identity before using a certificate as a map reference."""
    with h5py.File(path, 'r') as handle:
        if 'grid_dim' not in handle.attrs:
            raise ValueError('comparison certificate lacks grid_dim')
        grid_dim = np.asarray(handle.attrs['grid_dim'])
        if grid_dim.size != 1 or integer(grid_dim.reshape(-1)[0], 'certificate grid_dim') != dim:
            raise ValueError('comparison certificate has a different grid')
        for name in ('source', 'capacity', 'baseline', 'radii_cells', 'first_crossing'):
            if name not in handle or not np.array_equal(handle[name][:], data[name]):
                raise ValueError(f'comparison certificate has missing or different {name}')
        if 'allocation' not in handle:
            raise ValueError('comparison certificate lacks the final allocation')
        allocation = np.asarray(handle['allocation'][:], dtype=np.float64)
        if allocation.shape != data['source'].shape or not np.all(np.isfinite(allocation)):
            raise ValueError('comparison certificate has an invalid allocation')
        attrs = {}
        for name in ('snapshot', 'shape_enabled', 'shape_objective', 'shape_gap',
                     'optimization_tolerance', 'elapsed_seconds', 'shape_evaluations',
                     'shape_iterations', 'shape_cuts', 'shape_oracle_calls'):
            if name in handle.attrs:
                item = np.asarray(handle.attrs[name])
                if item.size != 1:
                    raise ValueError(f'comparison attribute {name} is not scalar')
                attrs[name] = item.reshape(-1)[0].item()
        constraints = {name: handle[name][:] for name in ('lower', 'upper', 'radius_cells', 'priority')
                       if name in handle}
    return allocation, attrs, constraints


def compare_allocation(path, comparison, data, allocation, lower, upper, radius, priority):
    """Report differences without imposing a bitwise or arbitrary map threshold."""
    reference, attrs, constraints = comparison
    a, b = allocation.astype(np.longdouble), reference.astype(np.longdouble)
    capacity = data['capacity'].astype(np.longdouble)
    baseline = data['baseline'].astype(np.longdouble)
    active = capacity > 0
    delta = a - b
    fraction_delta = np.divide(delta, capacity, out=np.zeros_like(delta), where=active)

    def objective(y):
        return float(np.sum(.5 * (y[active] - baseline[active]) ** 2 / capacity[active],
                            dtype=np.longdouble))

    actual_objective, reference_objective = objective(a), objective(b)
    total_capacity = np.sum(capacity, dtype=np.longdouble)
    return {
        'certificate': str(path.resolve()), 'certificate_sha256': file_sha256(path),
        'identical_problem_inputs': True,
        'scope': 'Numerical comparison of audited replay to the stored map; no bitwise '
                 'identity or map-accuracy threshold is required. Allocation/capacity '
                 'is the fraction of the recorded photon capacity filled.',
        'reference_attributes': attrs,
        'identical_constraints': {name: bool(np.array_equal(constraints[name], values))
                                 if name in constraints else None
                                 for name, values in (('lower', lower), ('upper', upper),
                                                      ('radius_cells', radius), ('priority', priority))},
        'allocation_max_abs_difference': float(np.max(np.abs(delta), initial=0)),
        'allocation_l1_difference': float(np.sum(np.abs(delta), dtype=np.longdouble)),
        'allocation_total_difference': float(np.sum(delta, dtype=np.longdouble)),
        'capacity_fraction_max_abs_difference': float(np.max(np.abs(fraction_delta), initial=0)),
        'capacity_fraction_rms_difference': float(np.sqrt(np.mean(fraction_delta[active] ** 2)))
                                              if np.any(active) else 0.,
        'capacity_fraction_capacity_weighted_l1_difference':
            float(np.sum(np.abs(delta), dtype=np.longdouble) / total_capacity)
            if total_capacity > 0 else 0.,
        'positive_capacity_cells': int(np.count_nonzero(active)),
        'replay_objective_recomputed': actual_objective,
        'reference_objective_recomputed': reference_objective,
        'objective_difference': actual_objective - reference_objective,
    }


def make_result(n, groups, visitor=None):
    arrays = [np.zeros(n) for _ in range(3)]
    masks = [np.zeros(n, dtype=np.uint8) for _ in range(2)]
    diagnostics = (Priority * groups)()
    result = Result()
    result.y, result.source_residual, result.source_spent = map(ptr, arrays)
    result.cut_source, result.cut_target = (ptr(mask, PB) for mask in masks)
    result.priority_diagnostics = diagnostics
    result.priority_diagnostics_capacity = groups
    result.visit_edge = visitor if visitor is not None else Visitor()
    return result, (arrays, masks, diagnostics, visitor)


class EdgeAudit:
    """Stream edges into extended-precision ledgers; do not store the edge list."""

    def __init__(self, dim, radius):
        self.dim, self.radius = dim, radius
        self.row = np.zeros(len(radius), dtype=np.longdouble)
        self.column = np.zeros(len(radius), dtype=np.longdouble)
        self.edges = 0
        self.error = None
        self.geometry_excess = 0.0
        self.visitor = Visitor(self.add)

    def add(self, source, target, amount, _user):
        # Exceptions must not escape a ctypes callback: retain the failure for audit.
        try:
            if not (0 <= source < len(self.row) and 0 <= target < len(self.row)):
                raise ValueError('edge index outside the grid')
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError('emitted edge is not positive and finite')
            self.edges += 1
            self.row[source] += np.longdouble(amount)
            self.column[target] += np.longdouble(amount)
            dim = self.dim
            a, b = source, target
            distance2 = 0
            for _ in range(3):
                delta = abs(a % dim - b % dim)
                delta = min(delta, dim - delta)
                distance2 += delta * delta
                a //= dim
                b //= dim
            excess = distance2 - float(self.radius[target]) ** 2
            self.geometry_excess = max(self.geometry_excess, excess)
            if excess > 1e-10:
                raise ValueError('edge outside the declared periodic target radius')
        except Exception as error:
            if self.error is None:
                self.error = str(error)

    def finish(self, data, lower, upper, priority, reference, result, keep, options,
               optimization_tolerance):
        y, residual, spent = (a.astype(np.longdouble) for a in keep[0])
        source = data['source'].astype(np.longdouble)
        capacity = data['capacity'].astype(np.longdouble)
        baseline = data['baseline'].astype(np.longdouble)
        lower, upper = lower.astype(np.longdouble), upper.astype(np.longdouble)
        eps = np.longdouble(256 * np.finfo(np.float64).eps)
        relative = np.longdouble(options.relative_tolerance)
        failures = []

        def check(condition, message):
            if not bool(np.all(condition)):
                failures.append(message)

        def maximum(value):
            return float(np.max(value, initial=0))

        check(self.error is None, self.error or 'callback failure')
        check(np.isfinite(y) & np.isfinite(residual) & np.isfinite(spent), 'nonfinite result')
        check((y >= 0) & (residual >= 0) & (spent >= 0), 'negative inventory')
        # No absolute/global floor: an entirely lost faint source must fail.
        check(np.abs(self.row + residual - source) <= eps * source, 'per-source edge ledger')
        check(np.abs(spent + residual - source) <= eps * source, 'per-source reported ledger')
        check(np.abs(self.row - spent) <= eps * source, 'edge/reported source spending')
        check(self.row <= source * (1 + eps), 'source overspend')
        check(np.abs(self.column - y) <= eps * np.maximum(y, self.column), 'edge/target marginal')
        check(y <= upper * (1 + eps), 'upper capacity')
        check(y <= capacity * (1 + eps), 'physical capacity')
        check(y >= lower * (1 - relative - eps), 'lower bound')
        total = np.sum(y, dtype=np.longdouble)
        source_total = np.sum(source, dtype=np.longdouble)
        residual_total = np.sum(residual, dtype=np.longdouble)
        check(abs(total - np.longdouble(reference.metrics.absorbed)) <=
              eps * abs(np.longdouble(reference.metrics.absorbed)), 'staged maximum total changed')
        check(abs(total + residual_total - source_total) <= eps * source_total, 'global ledger')
        check(abs(total - result.metrics.absorbed) <= eps * total, 'reported absorbed total')
        check(self.edges == result.metrics.positive_edges, 'positive-edge count')
        check(result.metrics.full_graph_cut_separated == 1, 'native final graph certificate absent')
        positive = source > 0
        ratio = np.divide(residual, source, out=np.zeros_like(source), where=positive)
        full = bool(np.all(ratio[positive] <= relative))
        check(full == bool(result.metrics.full_source_budget), 'full-source flag mismatch')
        check(result.status == (0 if full else 1), 'flow status inconsistent with source inventory')
        check(math.isfinite(optimization_tolerance) and optimization_tolerance >= 0,
              'invalid native optimization tolerance')
        quota_error, prefix_error, prefix_actual, prefix_expected = 0.0, 0.0, 0., 0.
        quota_allowance, prefix_allowance = 0.0, 0.0
        for index in range(reference.metrics.priority_groups):
            diagnostic = reference.priority_diagnostics[index]
            cells = priority == diagnostic.priority
            retained = np.sum(np.minimum(y[cells], baseline[cells]), dtype=np.longdouble)
            quota_error = max(quota_error, float(abs(retained - diagnostic.baseline_retained_in_group)))
            allowed = options.absolute_tolerance + optimization_tolerance + relative * max(
                abs(retained), abs(diagnostic.baseline_retained_in_group))
            quota_allowance = max(quota_allowance, float(allowed))
            check(abs(retained - diagnostic.baseline_retained_in_group) <= allowed,
                  f'baseline retained quota {diagnostic.priority}')
            prefix_actual += retained
            prefix_expected += np.longdouble(diagnostic.baseline_retained_in_group)
            prefix_error = max(prefix_error, float(abs(prefix_actual - prefix_expected)))
            allowed = (options.absolute_tolerance + (index + 1) * optimization_tolerance +
                       relative * max(abs(prefix_actual), abs(prefix_expected)))
            prefix_allowance = max(prefix_allowance, float(allowed))
            check(abs(prefix_actual - prefix_expected) <= allowed,
                  f'baseline retained prefix {diagnostic.priority}')
        l1 = np.sum(np.abs(y - baseline), dtype=np.longdouble)
        l1_optimization = 2 * reference.metrics.priority_groups * optimization_tolerance
        l1_roundoff = np.longdouble(64 * np.finfo(np.float64).eps) * (
            total + np.sum(baseline, dtype=np.longdouble) +
            2 * abs(reference.metrics.retention_stage_flow) + max(source_total, total, residual_total))
        allowed_l1 = options.absolute_tolerance + l1_optimization + l1_roundoff
        check(abs(l1 - reference.metrics.morphology_l1) <= allowed_l1, 'staged ES L1 changed')
        return {
            'passed': not failures, 'failures': failures,
            'scope': 'Independent emitted-edge feasibility and inventory audit; '
                     'native maximum-flow and quadratic-optimality claims are not independently re-solved.',
            'positive_edges': self.edges, 'source_total': float(source_total),
            'absorbed_total': float(total), 'residual_total': float(residual_total),
            'full_source_budget': full, 'max_source_relative_residual': maximum(ratio),
            'max_source_ledger_error': maximum(np.abs(self.row + residual - source)),
            'max_source_relative_ledger_error': maximum(np.divide(
                np.abs(self.row + residual - source), source,
                out=np.zeros_like(source), where=positive)),
            'max_column_error': maximum(np.abs(self.column - y)),
            'max_source_overspend': maximum(self.row - source),
            'max_target_excess': maximum(y - upper),
            'max_lower_deficit': maximum(lower - y),
            'max_geometry_excess': self.geometry_excess,
            'geometry_distance_squared_margin': 1e-10,
            'max_retained_quota_error': quota_error,
            'max_retained_prefix_error': prefix_error,
            'optimization_tolerance': optimization_tolerance,
            'max_quota_check_allowance': quota_allowance,
            'max_prefix_check_allowance': prefix_allowance,
            'l1_optimization_allowance': l1_optimization,
            'l1_roundoff_allowance': float(l1_roundoff),
            'l1_total_check_allowance': float(allowed_l1),
            'morphology_l1': float(l1), 'inventory_relative_roundoff_bound': float(eps),
        }


def replay(args, report):
    attrs, dim, data = read_checkpoint(args.checkpoint)
    comparison = read_comparison(args.compare_certificate, dim, data) if args.compare_certificate else None
    n = dim ** 3
    report['checkpoint_attributes'] = attrs
    report['checkpoint_sha256'] = file_sha256(args.checkpoint)
    with tempfile.TemporaryDirectory(prefix='photon-replay-') as directory:
        lib = build_library(Path(directory), args.cc, report)
        fo, so = FlowOptions(), ShapeOptions()
        lib.pc_default_options(ct.byref(fo))
        lib.pc_shape_default_options(ct.byref(so))
        fo.absolute_tolerance = float(attrs['absolute_tolerance'])
        fo.relative_tolerance = float(attrs['tolerance'])
        if (not math.isfinite(fo.absolute_tolerance) or fo.absolute_tolerance < 0 or
                not math.isfinite(fo.relative_tolerance) or not 0 <= fo.relative_tolerance < 1):
            raise ValueError('invalid recorded solver tolerance')
        fo.max_edges = integer(attrs['max_edges'], 'max_edges')
        memory = integer(attrs['memory_mb'], 'memory_mb') * 1024 * 1024
        reserve = n * 512 + ct.sizeof(AdapterContext)
        if memory > sys.maxsize or memory <= reserve:
            raise ValueError('recorded memory budget does not cover adapter reservations')
        fo.max_bytes = (memory - reserve) // 2
        so.max_bytes = fo.max_bytes
        so.absolute_tolerance, so.relative_tolerance = fo.absolute_tolerance, fo.relative_tolerance
        so.max_master_sweeps = integer(attrs['max_iterations'], 'max_iterations')
        so.max_hall_cuts = integer(args.max_cuts if args.max_cuts is not None else
                                   attrs['max_cuts'], 'max_cuts')
        so.max_evaluations = integer(args.max_evaluations if args.max_evaluations is not None else
                                     attrs.get('max_evaluations', so.max_evaluations), 'max_evaluations')
        shape_enabled = integer(attrs['shape_enabled'], 'shape_enabled', 0)
        if shape_enabled not in (0, 1):
            raise ValueError('shape_enabled must be 0 or 1')
        if args.shape != 'checkpoint':
            shape_enabled = int(args.shape == 'on')
        report['effective_options'] = {
            'flow': struct_dict(fo), 'shape': struct_dict(so), 'shape_enabled': bool(shape_enabled),
            'adapter_reserved_bytes': reserve, 'adapter_context_bytes': ct.sizeof(AdapterContext),
            'memory_limit_bytes': memory,
            'memory_scope': 'Reproduces native adapter workspace limits; Python RSS is not bounded.',
            'max_evaluations_recorded': 'max_evaluations' in attrs,
            'overrides': {name: getattr(args, name) for name in ('max_cuts', 'max_evaluations')
                          if getattr(args, name) is not None},
        }
        if args.shape != 'checkpoint':
            report['effective_options']['overrides']['shape'] = args.shape
        lower, upper, radius = (np.zeros(n) for _ in range(3))
        priority = np.zeros(n, dtype=np.int32)
        p = Problem((SZ * 3)(dim, dim, dim), n,
                    *map(ptr, (data['source'], data['capacity'], data['baseline'], lower, upper, radius)),
                    ptr(priority, PI))
        radii = data['radii_cells']
        ep = ESProblem(p, ptr(radii), len(radii), ptr(data['first_crossing'], PI))
        reference, reference_keep = make_result(n, len(radii) + 1)
        es = ESResult(reference, ptr(lower), ptr(upper), ptr(radius), ptr(priority, PI))
        started = time.monotonic()
        print('ES: starting completion', file=sys.stderr, flush=True)
        status = lib.pc_es_complete(ct.byref(ep), ct.byref(fo), ct.byref(es))
        report['es'] = {'status': status, 'status_text': lib.pc_status_string(status).decode(),
                        'seconds': time.monotonic() - started, 'completion': struct_dict(es.metrics),
                        'flow': struct_dict(es.allocation.metrics)}
        print(f"ES: {report['es']['status_text']} ({report['es']['seconds']:.3f}s)", file=sys.stderr, flush=True)
        if status not in (0, 1):
            raise RuntimeError('ES completion failed; no successful final allocation')
        audit = EdgeAudit(dim, radius)
        final, final_keep = make_result(n, len(radii) + 1, audit.visitor)
        started = time.monotonic()
        if shape_enabled:
            metrics = ShapeMetrics()
            print('Shape: starting refinement', file=sys.stderr, flush=True)
            status = lib.pc_shape_refine(ct.byref(p), ct.byref(fo), ct.byref(es.allocation), None,
                                         ct.byref(so), ct.byref(final), ct.byref(metrics))
            report['shape'] = {'status': status, 'status_text': lib.pc_shape_status_string(status).decode(),
                               'seconds': time.monotonic() - started, 'metrics': struct_dict(metrics)}
            print(f"Shape: {report['shape']['status_text']} ({report['shape']['seconds']:.3f}s)",
                  file=sys.stderr, flush=True)
            if status != 0:
                raise RuntimeError('unique-shape solve failed; staged map is not a unique-shape result')
        else:
            fixed = Problem.from_buffer_copy(p)
            fixed.lower = fixed.upper = es.allocation.y
            status = lib.pc_allocate(ct.byref(fixed), ct.byref(fo), ct.byref(final))
            report['shape'] = {'enabled': False, 'fixed_marginal_status': status,
                               'seconds': time.monotonic() - started}
            if status not in (0, 1):
                raise RuntimeError('final fixed-marginal certificate failed')
        report['final_flow'] = struct_dict(final.metrics)
        report['edge_audit'] = audit.finish(data, lower, upper, priority, es.allocation,
                                             final, final_keep, fo,
                                             metrics.tolerance if shape_enabled else 0.0)
        # Keep all caller-owned buffers alive through the final native call/audit.
        del reference_keep
        if not report['edge_audit']['passed']:
            raise RuntimeError('independent edge audit failed: ' + '; '.join(report['edge_audit']['failures']))
        if comparison is not None:
            report['certificate_comparison'] = compare_allocation(
                args.compare_certificate, comparison, data, final_keep[0][0],
                lower, upper, radius, priority)
        if args.save_allocation:
            np.savez_compressed(args.save_allocation, grid_dim=dim,
                                allocation=final_keep[0][0], source_residual=final_keep[0][1],
                                source_spent=final_keep[0][2], lower=lower, upper=upper,
                                radius_cells=radius, priority=priority, **data)
            report['saved_allocation'] = str(args.save_allocation.resolve())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, bytes):
        return value.decode(errors='replace')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='Debug=2 photon_inputs_*.hdf5')
    parser.add_argument('--max-evaluations', type=int, help='override recorded shape evaluation limit')
    parser.add_argument('--max-cuts', type=int, help='override recorded Hall-cut limit')
    parser.add_argument('--shape', choices=('checkpoint', 'on', 'off'), default='checkpoint',
                        help='default: use the checkpoint shape_enabled flag')
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'), help='C99 compiler command (default: CC or cc)')
    parser.add_argument('--output', type=Path, help='also write the full JSON report to this file')
    parser.add_argument('--compare-certificate', type=Path,
                        help='compare the final map with a photon_conservation_*.hdf5 for identical inputs')
    parser.add_argument('--save-allocation', type=Path,
                        help='save audited final arrays and problem inputs to a .npz file')
    args = parser.parse_args()
    inputs = {args.checkpoint.resolve()}
    if args.compare_certificate:
        inputs.add(args.compare_certificate.resolve())
    outputs = [path.resolve() for path in (args.output, args.save_allocation) if path is not None]
    if any(path in inputs for path in outputs):
        parser.error('output paths must not overwrite input checkpoint or comparison certificate')
    if len(outputs) != len(set(outputs)):
        parser.error('--output and --save-allocation must use different paths')
    if args.save_allocation and args.save_allocation.suffix != '.npz':
        parser.error('--save-allocation must end in .npz')
    report = {'checkpoint': str(args.checkpoint.resolve()), 'success': False}
    started = time.monotonic()
    try:
        replay(args, report)
        report['success'] = True
    except (OSError, ValueError, RuntimeError, OverflowError, MemoryError) as error:
        report['error'] = str(error)
    report['elapsed_seconds'] = time.monotonic() - started
    text = json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n'
    if args.output:
        try:
            args.output.write_text(text)
        except OSError as error:
            report['success'] = False
            report['output_error'] = str(error)
            text = json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n'
    print(text, end='', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
