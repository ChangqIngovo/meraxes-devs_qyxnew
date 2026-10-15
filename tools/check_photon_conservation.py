#!/usr/bin/env python3
"""Independently audit a Meraxes photon-conservation snapshot certificate.

Requires numpy and h5py. Debug=1 files permit a scalar ledger check. Debug=2
files also contain cell inventories and a positive-edge CSV; those permit
independent source, target, geometry, ES-support and float-write checks.
Recorded L1 optima and priority quotas are checked for consistency, not
independently re-optimized by this script.
Input checkpoints (photon_inputs_NNN.hdf5) are not completed certificates.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np


class Audit:
    def __init__(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol
        self.checks = []

    def close(self, name, actual, expected, scale=None, optimization_tolerance=0.0,
              roundoff_tolerance=0.0):
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        if scale is None:
            scale = np.maximum(np.abs(actual), np.abs(expected))
        scale = np.asarray(scale, dtype=float)
        error = np.abs(actual - expected)
        limit = self.atol + self.rtol * scale + optimization_tolerance + roundoff_tolerance
        passed = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                      and np.all(error <= limit))
        self.checks.append({"name": name, "passed": passed,
                            "max_absolute_error": float(np.max(error, initial=0.0)),
                            "optimization_tolerance": float(np.max(np.asarray(optimization_tolerance), initial=0.0)),
                            "roundoff_tolerance": float(np.max(np.asarray(roundoff_tolerance), initial=0.0))})
        return passed

    def require(self, name, condition, **measurements):
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, **measurements})
        return passed

    def leq(self, name, actual, limit, scale=None):
        actual = np.asarray(actual, dtype=float)
        limit = np.asarray(limit, dtype=float)
        if scale is None:
            scale = np.maximum(np.abs(actual), np.abs(limit))
        excess = actual - limit
        passed = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(limit))
                      and np.all(excess <= self.atol + self.rtol * np.asarray(scale)))
        self.checks.append({"name": name, "passed": passed,
                            "max_excess": float(np.max(excess, initial=0.0))})
        return passed


def cell_array(handle, name, ncell):
    value = np.asarray(handle[name][...]).reshape(-1)
    if value.size != ncell:
        raise ValueError(f"{name}: expected {ncell} entries, found {value.size}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: non-finite entries")
    return value


def audit_edges(path, source, y, radius, dim, audit):
    """Stream a CSV in chunks; source and target indices are parsed as integers."""
    ncell = source.size
    spent = np.zeros(ncell, dtype=np.longdouble)
    received = np.zeros(ncell, dtype=np.longdouble)
    row_terms = np.zeros(ncell, dtype=np.int64)
    count = 0
    geometry_excess = 0.0
    total = 0.0
    geometric_failure = False
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["source", "target", "photons"]:
            raise ValueError(f"{path}: expected source,target,photons header")
        chunk = []
        for row in reader:
            item = (int(row["source"]), int(row["target"]), float(row["photons"]))
            if not (0 <= item[0] < ncell and 0 <= item[1] < ncell):
                raise ValueError(f"{path}: edge index outside the grid")
            if not np.isfinite(item[2]) or item[2] <= 0:
                raise ValueError(f"{path}: edges must carry finite positive counts")
            chunk.append(item)
            if len(chunk) == 65536:
                yield_chunk = chunk
                chunk = []
            else:
                continue
            ii, jj, flow, excess, bad = add_edge_chunk(
                yield_chunk, spent, received, row_terms, radius, dim, audit)
            count += len(ii)
            total += float(np.sum(flow))
            geometry_excess = max(geometry_excess, excess)
            geometric_failure |= bad
        if chunk:
            ii, jj, flow, excess, bad = add_edge_chunk(
                chunk, spent, received, row_terms, radius, dim, audit)
            count += len(ii)
            total += float(np.sum(flow))
            geometry_excess = max(geometry_excess, excess)
            geometric_failure |= bad
    audit.require("positive edges obey periodic target-radius support",
                  not geometric_failure, max_geometry_excess=geometry_excess)
    audit.close("edge arrivals reproduce target allocation", received, y)
    audit.leq("edge departures do not exceed source inventories", spent, source)
    return spent, received, count, total, row_terms


def add_edge_chunk(chunk, spent, received, row_terms, radius, dim, audit):
    ii = np.fromiter((item[0] for item in chunk), dtype=np.int64)
    jj = np.fromiter((item[1] for item in chunk), dtype=np.int64)
    flow = np.fromiter((item[2] for item in chunk), dtype=float)
    np.add.at(spent, ii, flow)
    np.add.at(received, jj, flow)
    np.add.at(row_terms, ii, 1)
    xsource = np.asarray(np.unravel_index(ii, (dim, dim, dim))).T
    xtarget = np.asarray(np.unravel_index(jj, (dim, dim, dim))).T
    distance = np.abs(xsource - xtarget)
    distance = np.minimum(distance, dim - distance)
    distance_squared = np.sum(distance * distance, axis=1)
    excess = distance_squared - radius[jj] ** 2
    return ii, jj, flow, float(np.max(excess, initial=0.0)), bool(np.any(excess > 1e-10))


def audit_source_edge_ledger(audit, source, spent, stored_spent, residual, row_terms, relative_tolerance):
    """A source-normalized audit with no absolute count floor for faint donors."""
    if not np.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError("source relative tolerance must lie in [0,1)")
    positive = source > 0
    inventory = source[positive].astype(np.longdouble)
    rows = spent[positive].astype(np.longdouble)
    recorded = stored_spent[positive].astype(np.longdouble)
    remainder = residual[positive].astype(np.longdouble)
    # Positive edge sums use long double. gamma_n bounds ordinary summation
    # roundoff; binary64 marginal arithmetic receives a separate small allowance.
    accumulation = row_terms[positive].astype(np.longdouble) * np.finfo(np.longdouble).eps
    if np.any(accumulation >= 1):
        raise ValueError("too many source-edge terms for a finite accumulation bound")
    floating_allowance = 8 * np.finfo(float).eps + accumulation / (1 - accumulation)
    limit = relative_tolerance + floating_allowance
    errors = (
        ("source-normalized edge departures respect every source budget", (rows - inventory) / inventory),
        ("source-normalized edge rows reproduce stored spending", np.abs(rows - recorded) / inventory),
        ("source-normalized edge rows plus residual reproduce inventory",
         np.abs(rows + remainder - inventory) / inventory),
    )
    for name, error in errors:
        audit.require(name, np.all(np.isfinite(error)) and np.all(error <= limit),
                      max_relative_error=float(np.max(error, initial=0.0)),
                      source_relative_tolerance=relative_tolerance,
                      max_floating_allowance=float(np.max(floating_allowance, initial=0.0)))
    zero = source == 0
    audit.require("zero-inventory sources have exactly zero edge spending",
                  np.all(spent[zero] == 0))
    audit.require("zero-inventory sources have exactly zero recorded spending and residual",
                  np.all(stored_spent[zero] == 0) and np.all(residual[zero] == 0))


def es_band_mask(b, capacity, dim, width):
    """Rebuild the declared six-neighbour front without using solver masks."""
    if width == -1:
        return np.zeros(b.size, dtype=bool)
    if width == -2:
        return np.ones(b.size, dtype=bool)
    if width < 0:
        raise ValueError(f"invalid band width {width}")
    shape = (dim, dim, dim)
    positive = (capacity > 0).reshape(shape)
    full = ((b == capacity) & (capacity > 0)).reshape(shape)
    reached = ((b > 0) & (b < capacity)).reshape(shape)
    for axis in range(3):
        for shift in (-1, 1):
            reached |= positive & np.roll(positive, shift, axis) & (
                full != np.roll(full, shift, axis))
    for _ in range(width):
        previous = reached.copy()
        for axis in range(3):
            reached |= np.roll(previous, -1, axis) | np.roll(previous, 1, axis)
        if np.array_equal(previous, reached):
            break
    return reached.reshape(-1)


def audit_staged_cut(handle, attrs, source, upper, radius, dim, audit):
    """Check a min-cut against all legal edges, without the solver's sparse graph.

    Zero-upper-bound targets cannot affect cut capacity, so closure is required
    only at positive upper bounds. Bounded NumPy chunks avoid a dense NxN array.
    """
    ncell = source.size
    source_mask = cell_array(handle, "staged_cut_source", ncell)
    target_mask = cell_array(handle, "staged_cut_target", ncell)
    if not (np.all((source_mask == 0) | (source_mask == 1))
            and np.all((target_mask == 0) | (target_mask == 1))):
        raise ValueError("staged cut masks must contain only zero and one")
    source_mask = source_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    cut_capacity = math.fsum(source[~source_mask]) + math.fsum(upper[target_mask])
    audit.close("staged cut capacity from source and target inventories", cut_capacity,
                float(attrs["staged_mincut_capacity"]))
    audit.close("staged cut capacity equals staged absorption", cut_capacity,
                float(attrs["staged_absorbed_total"]))
    audit.require("staged solver reports full-geometry cut certificate",
                  int(attrs["staged_full_graph_certificate"]) == 1)
    source_indices = np.flatnonzero(source_mask & (source > 0))
    target_indices = np.flatnonzero(~target_mask & (upper > 0))
    source_xyz = np.asarray(np.unravel_index(source_indices, (dim, dim, dim))).T
    missing_targets = 0
    for start in range(0, target_indices.size, 32):
        selected = target_indices[start:start + 32]
        target_xyz = np.asarray(np.unravel_index(selected, (dim, dim, dim))).T
        linked = np.zeros(selected.size, dtype=bool)
        # The C allocator declares a squared-distance geometry tolerance of 1e-10.
        radius_squared = radius[selected] ** 2 + 1e-10
        for offset in range(0, source_indices.size, 2048):
            difference = np.abs(target_xyz[:, None, :] - source_xyz[None, offset:offset + 2048, :])
            difference = np.minimum(difference, dim - difference)
            distance_squared = np.sum(difference * difference, axis=2)
            linked |= np.any(distance_squared <= radius_squared[:, None], axis=1)
            if np.all(linked):
                break
        missing_targets += int(np.count_nonzero(linked))
    audit.require("staged cut closes every legal positive-capacity target edge",
                  missing_targets == 0, targets_missing_from_cut=missing_targets)


def inspect(path, edges_path, atol, rtol, require_cells=False, require_full=False):
    audit = Audit(atol, rtol)
    with h5py.File(path, "r") as handle:
        attrs = {}
        for name, value in handle.attrs.items():
            value = np.asarray(value)
            if value.size != 1:
                raise ValueError(f"attribute {name}: expected one scalar value")
            attrs[name] = value.item()
            if isinstance(attrs[name], (int, float)) and not np.isfinite(attrs[name]):
                raise ValueError(f"attribute {name}: non-finite scalar value")
        if "certificate_complete" in attrs and int(attrs["certificate_complete"]) != 1:
            raise ValueError("incomplete photon-conservation checkpoint: inputs were saved, "
                             "but no completed allocation certificate was produced")
        mandatory = ["schema_version", "snapshot", "grid_dim", "solver_status",
                     "source_total", "capacity_total", "absorbed_total", "residual_total",
                     "ledger_error", "float_grid_ledger_error", "float_grid_roundoff_bound"]
        missing = [name for name in mandatory if name not in attrs]
        if missing:
            raise ValueError("missing scalar attributes: " + ", ".join(missing))
        if int(attrs["schema_version"]) != 1:
            raise ValueError(f"unsupported schema_version={attrs['schema_version']}")
        dim = int(attrs["grid_dim"])
        if dim <= 0:
            raise ValueError("grid_dim must be positive")
        ncell = dim ** 3
        photons = float(attrs["source_total"])
        absorbed = float(attrs["absorbed_total"])
        residual = float(attrs["residual_total"])
        capacity_total = float(attrs["capacity_total"])
        status = int(attrs["solver_status"])
        optimization_tolerance = float(attrs.get("optimization_tolerance", 0.0))
        if not np.isfinite(optimization_tolerance) or optimization_tolerance < 0:
            raise ValueError("optimization_tolerance must be finite and nonnegative")
        staged_names = ("staged_absorbed_total", "staged_mincut_capacity",
                        "staged_full_graph_certificate", "terminal_maximum_absorption",
                        "terminal_full_source_budget")
        staged_present = any(name in attrs for name in staged_names)
        if staged_present and not all(name in attrs for name in staged_names):
            raise ValueError("incomplete staged-cut scalar metadata")
        audit.require("completed solver status", status in (0, 1), solver_status=status)
        if "full_source_budget" in attrs:
            flag = attrs["full_source_budget"]
            audit.require("full-source flag agrees with solver status",
                          flag in (0, 1) and bool(flag) == (status == 0))
        for name in ("max_source_relative_residual", "max_edge_balance_error"):
            if name in attrs:
                audit.leq("nonnegative " + name, 0.0, float(attrs[name]))
        for name in ("shape_evaluations", "shape_oracle_calls"):
            if name in attrs:
                value = float(attrs[name])
                audit.require("nonnegative integer " + name, value >= 0 and value == int(value))
        if "max_evaluations" in attrs:
            evaluation_budget = float(attrs["max_evaluations"])
            audit.require("positive integer evaluation budget",
                          evaluation_budget > 0 and evaluation_budget == int(evaluation_budget))
            if "shape_evaluations" in attrs:
                audit.require("completed solve stays within evaluation budget",
                              float(attrs["shape_evaluations"]) <= evaluation_budget)
        audit.leq("nonnegative scalar source inventory", 0.0, photons)
        audit.leq("nonnegative scalar absorption", 0.0, absorbed)
        audit.leq("nonnegative scalar residual", 0.0, residual)
        audit.close("scalar source = absorbed + residual", photons, absorbed + residual)
        audit.close("reported scalar ledger error", abs(float(attrs["ledger_error"])),
                    abs(photons - absorbed - residual), scale=max(photons, absorbed))
        audit.leq("scalar absorption within capacity", absorbed, capacity_total)
        if status == 0 or require_full:
            audit.close("full source-budget absorption", residual, 0.0, scale=photons)
        float_error = abs(float(attrs["float_grid_ledger_error"]))
        float_bound = float(attrs["float_grid_roundoff_bound"])
        reduction_roundoff = 64.0 * np.finfo(float).eps * max(photons, capacity_total)
        audit.leq("nonnegative float-grid roundoff bound", 0.0, float_bound)
        audit.leq("float-grid imbalance within quantization plus ledger error",
                  float_error, float_bound + abs(photons - absorbed - residual) + reduction_roundoff,
                  scale=float_bound)
        if int(attrs.get("shape_enabled", 0)):
            audit.require("unique-map solve completed", int(attrs.get("shape_status", -1)) == 0)
            if "optimization_tolerance" in attrs:
                declared_tolerance = float(attrs["absolute_tolerance"]) + float(attrs["tolerance"]) * max(1.0, photons)
                audit.close("optimization tolerance matches configured solver tolerance",
                            optimization_tolerance, declared_tolerance)
            correction_names = ("shape_oracle_target_l1_error", "shape_max_final_correction",
                                "shape_final_correction_l1")
            if any(name in attrs for name in correction_names):
                if not all(name in attrs for name in correction_names):
                    raise ValueError("incomplete final-correction diagnostic metadata")
                for name in correction_names:
                    audit.leq("nonnegative " + name, 0.0, float(attrs[name]))
                audit.leq("reported maximum correction within total correction",
                          float(attrs["shape_max_final_correction"]), float(attrs["shape_final_correction_l1"]))
                audit.leq("reported total correction within cellwise maximum bound",
                          float(attrs["shape_final_correction_l1"]), ncell * float(attrs["shape_max_final_correction"]))
        else:
            audit.close("shape-disabled optimization tolerance is zero", optimization_tolerance, 0.0)
        if staged_present:
            audit.close("staged absorption reaches recorded terminal maximum",
                        float(attrs["staged_absorbed_total"]), float(attrs["terminal_maximum_absorption"]))
            ledger_roundoff = 64.0 * np.finfo(float).eps * max(photons, absorbed,
                                                              float(attrs["staged_absorbed_total"]))
            audit.close("final map preserves staged absorption", absorbed,
                        float(attrs["staged_absorbed_total"]), scale=0.0,
                        roundoff_tolerance=ledger_roundoff)
        else:
            ledger_roundoff = 64.0 * np.finfo(float).eps * max(photons, absorbed)

        level = "scalar ledger only"
        cells_present = "source" in handle
        if require_cells and not cells_present:
            audit.require("Debug=2 cell certificate available", False)
        if cells_present:
            level = "cell ledger and recorded optimization quotas"
            names = ["source", "source_II", "source_BH", "source_III", "capacity",
                     "density", "neutral_pre", "recombinations_pre", "baseline", "allocation",
                     "source_spent", "source_residual", "xH_ES", "xH", "xH_double",
                     "lower", "upper", "radius_cells", "first_crossing", "priority"]
            arrays = {name: cell_array(handle, name, ncell) for name in names}
            source, cap, b, y = (arrays[name].astype(float) for name in
                                 ("source", "capacity", "baseline", "allocation"))
            lower, upper = arrays["lower"], arrays["upper"]
            q = arrays["neutral_pre"]
            audit.leq("nonnegative cell source inventory", 0.0, source)
            audit.leq("nonnegative cell capacity", 0.0, cap)
            audit.leq("preionization neutral fraction lower bound", 0.0, q)
            audit.leq("preionization neutral fraction upper bound", q, 1.0)
            audit.close("source components sum to source inventory", source,
                        arrays["source_II"] + arrays["source_BH"] + arrays["source_III"])
            audit.close("effective atom capacity definition", cap,
                        q * (arrays["density"] + arrays["recombinations_pre"]))
            expected_b = np.zeros(ncell)
            active_q = q > 0
            expected_b[active_q] = cap[active_q] * (
                1.0 - arrays["xH_ES"][active_q] / q[active_q])
            expected_b = np.clip(expected_b, 0.0, cap)
            audit.close("projected original ES reference", b, expected_b)
            audit.leq("nonnegative lower bound", 0.0, lower)
            audit.leq("ordered target bounds", lower, upper)
            audit.leq("target upper bound within atom capacity", upper, cap)
            audit.leq("allocation satisfies lower bound", lower, y)
            audit.leq("allocation satisfies upper bound", y, upper)
            audit.leq("nonnegative source residual", 0.0, arrays["source_residual"])
            audit.close("per-source spent + residual = source", source,
                        arrays["source_spent"] + arrays["source_residual"])
            positive_source = source > 0
            relative_residual = arrays["source_residual"][positive_source] / source[positive_source]
            maximum_relative_residual = float(np.max(relative_residual, initial=0.0))
            if "max_source_relative_residual" in attrs:
                audit.close("reported maximum source-relative residual", maximum_relative_residual,
                            float(attrs["max_source_relative_residual"]))
            if status == 0 or require_full:
                source_tolerance = float(attrs.get("tolerance", rtol))
                if not 0 <= source_tolerance < 1:
                    raise ValueError("source relative tolerance must lie in [0,1)")
                audit.require("every positive source meets full-absorption tolerance",
                              maximum_relative_residual <= source_tolerance + 64 * np.finfo(float).eps,
                              max_source_relative_residual=maximum_relative_residual,
                              source_relative_tolerance=source_tolerance)
            audit.close("source total from cells", np.sum(source), photons)
            audit.close("capacity total from cells", np.sum(cap), capacity_total)
            audit.close("absorption total from cells", np.sum(y), absorbed)
            audit.close("residual total from cells", np.sum(arrays["source_residual"]), residual)
            audit.close("spent total = absorption", np.sum(arrays["source_spent"]), np.sum(y))

            radii = np.asarray(handle["radii_cells"][...], dtype=float).reshape(-1)
            if not radii.size or not np.all(np.isfinite(radii)):
                raise ValueError("radii_cells must be a nonempty finite ladder")
            audit.require("nonnegative descending ES radius ladder with identity last",
                          bool(np.all(radii >= 0) and np.all(np.diff(radii) <= 0)
                               and radii[-1] == 0.0))
            crossing = arrays["first_crossing"]
            priority = arrays["priority"]
            valid_crossings = bool(np.all(crossing == crossing.astype(np.int64))
                                   and np.all(crossing >= -1) and np.all(crossing < radii.size))
            if not valid_crossings:
                raise ValueError("first_crossing contains invalid radius indices")
            crossing = crossing.astype(np.int64)
            expected_priority = np.where(crossing >= 0, crossing, radii.size)
            audit.close("priority retains original first-crossing order", priority, expected_priority)
            promotion = int(attrs["promotion"])
            if promotion < 0:
                raise ValueError("promotion must be nonnegative")
            base = np.where(crossing >= 0, crossing, radii.size - 1)
            expected_radius = radii[np.maximum(base - promotion, 0)]
            audit.close("selected support follows ES radius promotion", arrays["radius_cells"], expected_radius)
            editable = es_band_mask(b, cap, dim, int(attrs["band_width"]))
            audit.close("lower bounds reproduce selected ES band", lower, np.where(editable, 0.0, b))
            audit.close("upper bounds reproduce selected ES band", upper, np.where(editable, cap, b))
            if "front_seed_cells" in attrs:
                audit.close("reported front seed count", float(attrs["front_seed_cells"]),
                            np.count_nonzero(es_band_mask(b, cap, dim, 0)))
            editable_count = int(np.count_nonzero(editable & (cap > 0)))
            if "editable_cells" in attrs:
                audit.close("reported editable cell count", float(attrs["editable_cells"]), editable_count)
            if "editable_fraction" in attrs:
                audit.close("reported editable volume fraction", float(attrs["editable_fraction"]),
                            editable_count / ncell)
            if staged_present:
                audit_staged_cut(handle, attrs, source, upper, arrays["radius_cells"], dim, audit)

            target = np.clip(b, lower, upper)
            retained = np.minimum(y, target)
            quota = np.asarray(handle["retained_quota"][...], dtype=float).reshape(-1)
            if quota.size != radii.size + 1 or not np.all(np.isfinite(quota)):
                raise ValueError("retained_quota must contain one finite entry per ES priority group")
            measured_quota = np.bincount(expected_priority, weights=retained,
                                        minlength=quota.size)
            active_groups = np.bincount(expected_priority, weights=(upper > 0),
                                        minlength=quota.size) > 0
            active_group_count = int(np.count_nonzero(active_groups))
            audit.close("retention in every original ES priority group", measured_quota, quota,
                        optimization_tolerance=optimization_tolerance)
            audit.close("retention in every original ES priority prefix",
                        np.cumsum(measured_quota), np.cumsum(quota),
                        optimization_tolerance=np.cumsum(active_groups) * optimization_tolerance)
            retention_flow = float(attrs["retention_stage_flow"])
            audit.close("retained quota sum equals T-stage optimum", np.sum(quota), retention_flow)
            l1 = float(np.sum(np.abs(y - b)))
            l1_formula = float(np.sum(y) + np.sum(b) - 2.0 * (
                retention_flow + np.sum(np.minimum(lower, b) - lower)))
            # L1 = F + sum(b) - 2 sum(retained): G group errors bounded by
            # tau imply a 2*G*tau L1 allowance, separately from count roundoff.
            l1_optimization_allowance = 2.0 * active_group_count * optimization_tolerance
            l1_roundoff = 64.0 * np.finfo(float).eps * (
                float(np.sum(np.abs(y))) + float(np.sum(np.abs(b))) + 2.0 * abs(retention_flow))
            audit.close("count L1 equals recorded final L1", l1, float(attrs["final_l1"]))
            audit.close("count L1 equals recorded minimum", l1, float(attrs["min_l1"]),
                        scale=0.0, optimization_tolerance=l1_optimization_allowance,
                        roundoff_tolerance=l1_roundoff + ledger_roundoff)
            audit.close("staged-flow formula for minimum count L1", l1, l1_formula,
                        scale=0.0, optimization_tolerance=l1_optimization_allowance,
                        roundoff_tolerance=l1_roundoff)

            expected_xh = q.copy()
            positive_cap = cap > 0
            expected_xh[positive_cap] = q[positive_cap] * (1.0 - y[positive_cap] / cap[positive_cap])
            expected_xh = np.clip(expected_xh, 0.0, q)
            audit.close("double neutral grid represents allocation", arrays["xH_double"], expected_xh)
            audit.require("stored xH equals float32 rounding of double grid",
                          bool(np.array_equal(arrays["xH"].astype(np.float32),
                                              arrays["xH_double"].astype(np.float32))))
            y_float = np.zeros(ncell)
            y_float[active_q] = cap[active_q] * (1.0 - arrays["xH"][active_q] / q[active_q])
            cell_quantization = float(np.sum(np.abs(y_float - y)))
            grid_imbalance = abs(float(np.sum(y_float)) - absorbed)
            audit.leq("cellwise float count error within reported roundoff bound",
                      cell_quantization, float_bound + reduction_roundoff, scale=float_bound)
            audit.leq("reported float-grid ledger error agrees within reduction roundoff",
                      abs(grid_imbalance - float_error), reduction_roundoff, scale=float_bound)
            if int(attrs.get("shape_enabled", 0)):
                objective = 0.5 * float(np.sum((y[positive_cap] - b[positive_cap]) ** 2
                                               / cap[positive_cap]))
                audit.close("quadratic objective about original ES reference", objective,
                            float(attrs["shape_objective"]))

            if edges_path.exists():
                spent, received, edge_count, edge_total, row_terms = audit_edges(
                    edges_path, source, y, arrays["radius_cells"], dim, audit)
                audit_source_edge_ledger(audit, source, spent, arrays["source_spent"],
                                         arrays["source_residual"], row_terms,
                                         float(attrs.get("tolerance", rtol)))
                audit.close("edge departures reproduce stored source spending", spent, arrays["source_spent"])
                audit.close("independent per-source edge ledger", spent + arrays["source_residual"], source)
                audit.close("positive-edge total equals allocation", edge_total, np.sum(y))
                if "max_edge_balance_error" in attrs:
                    edge_balance = max(float(np.max(np.abs(spent - arrays["source_spent"]), initial=0.0)),
                                       float(np.max(np.abs(received - y), initial=0.0)))
                    audit.close("reported maximum edge balance agrees within summation roundoff",
                                edge_balance, float(attrs["max_edge_balance_error"]), scale=0.0,
                                roundoff_tolerance=ledger_roundoff)
                audit.require("positive-edge count equals recorded count",
                              edge_count == int(attrs["positive_edges"]), positive_edges=edge_count)
                level = "explicit source-target allocation certificate"
            else:
                audit.require("Debug=2 positive-edge CSV available", False)
        report = {
            "file": str(path), "snapshot": int(attrs["snapshot"]), "grid_dim": dim,
            "audit_level": level, "full_source_budget": status == 0,
            "source_total": photons, "absorbed_total": absorbed, "residual_total": residual,
            "optimization_tolerance": optimization_tolerance,
            "passed": all(check["passed"] for check in audit.checks), "checks": audit.checks,
            "optimization_scope": (("Checks selected-geometry maximum absorption using the full-geometry staged cut. "
                                    if staged_present and cells_present else "")
                                  + "Checks recorded L1 and priority quotas; does not independently "
                                  "prove their optima, minimum band/promotion, or quadratic uniqueness."),
        }
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="photon_conservation_NNN.hdf5 snapshot certificate")
    parser.add_argument("--edges", type=Path, help="positive-edge CSV; default: <file stem>.edges.csv")
    parser.add_argument("--atol", type=float, default=1e-11, help="absolute count tolerance (default: 1e-11)")
    parser.add_argument("--rtol", type=float, default=1e-8, help="relative comparison tolerance (default: 1e-8)")
    parser.add_argument("--require-cells", action="store_true", help="fail if Debug=2 arrays/edges are absent")
    parser.add_argument("--require-full", action="store_true", help="also require all source photons to be absorbed")
    parser.add_argument("--json", action="store_true", help="print machine-readable detailed results")
    args = parser.parse_args(argv)
    if not (np.isfinite(args.atol) and np.isfinite(args.rtol) and args.atol >= 0 and args.rtol >= 0):
        parser.error("tolerances must be finite and nonnegative")
    edge_path = args.edges if args.edges is not None else args.file.with_suffix(".edges.csv")
    try:
        report = inspect(args.file, edge_path, args.atol, args.rtol,
                         args.require_cells, args.require_full)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        if args.json:
            print(json.dumps({"file": str(args.file), "passed": False, "error": str(error)}, indent=2))
        else:
            print(f"FAIL {args.file}: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        verdict = "PASS" if report["passed"] else "FAIL"
        print(f"{verdict} {args.file}: {report['audit_level']}")
        print(f"source={report['source_total']:.12g} absorbed={report['absorbed_total']:.12g} "
              f"residual={report['residual_total']:.12g}")
        for check in report["checks"]:
            if not check["passed"]:
                print(f"  FAIL: {check['name']}")
        print(report["optimization_scope"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
