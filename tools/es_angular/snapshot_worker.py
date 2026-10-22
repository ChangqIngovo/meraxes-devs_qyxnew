#!/usr/bin/env python3
"""Stateful, periodic angular-face H/He adapter for the Meraxes snapshot loop.

The request supplies physical escaped stellar photon counts and a prescribed
Eulerian density field. Internal atom/photon counts and cell volumes share
one constant mean-H-cell normalization. Density updates preserve local H/He
fractions and explicitly book external matter exchange; photons never scale
with density. Geometry is the angular-face approximation, with a finite travel
clock and retained periodic images. Pulse emission, front release and local
OTS splitting make the complete physical-time scheme first order.

A checkpoint supports consecutive calls and exact-request replay in one run.
It is not a complete restart of the Meraxes galaxy and feedback state.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np

if __package__:
    from .periodic_front import PeriodicFront
    from .recombination import evolve_local, CHANNEL_NAMES, OPACITY_E_EV
    from .es_scores import periodic_es_scores
else:
    from periodic_front import PeriodicFront
    from recombination import evolve_local, CHANNEL_NAMES, OPACITY_E_EV
    from es_scores import periodic_es_scores

SCHEMA = 2
SOLVER = "angular-face-periodic-hhe-v1"
C_CM_S = 2.99792458e10
TRANSPARENCY = 1e-5
AUDIT_RTOL = 5e-9
MAX_ENTRIES = 2000000
MAX_ROUTE_NODES = 1000000
MAX_SNAPSHOT_STEPS = 1000000
ARRAY_COLUMNS = {
    "initial_gas": 5, "source_history": None, "capture_counts": 3,
    "photo_counts": 3, "diffuse_absorbed_counts": 3,
    "diffuse_emitted_counts": 8, "diffuse_channel_debits": 8,
    "diffuse_pending": 8, "matter_exchange": 5,
    "interval_capture_counts": 3,
}
NUMERICAL_FILES = (
    "snapshot_worker.py", "periodic_front.py", "recombination.py",
    "chemistry.py", "angular_hhe.py", "es_scores.py",
    "angular_faces.c", "angular_faces.h", "angular_front.c", "angular_front.h",
)


def _text(value):
    return value.decode("utf8") if isinstance(value, bytes) else str(value)


def _numeric_scalar(value, name):
    # H5LTset_attribute_* in the C bridge writes one-element arrays, whereas
    # h5py usually writes scalar attributes. Accept both representations.
    scalar = np.asarray(value)
    if scalar.shape not in ((), (1,)) or scalar.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric scalar or one-element attribute")
    return float(scalar.item())


def _finite_positive(value, name, allow_zero=False):
    value = _numeric_scalar(value, name)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _integer(value, name, minimum=0):
    numeric = _numeric_scalar(value, name)
    if not math.isfinite(numeric) or int(numeric) != numeric or numeric < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(numeric)


def _array(value, shape, name, *, signed=False):
    values = np.array(value, dtype=np.float64, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.abs(values).sum())
    if (values.shape != shape or not np.all(np.isfinite(values))
            or not math.isfinite(total) or (not signed and np.any(values < 0))):
        raise ValueError(f"{name} must have shape {shape} and finite {'signed' if signed else 'nonnegative'} counts")
    return values


def _nuclei(gas):
    return np.column_stack((gas[:, :2].sum(1), gas[:, 2:].sum(1)))


def _electrons(gas):
    return gas[:, 1] + gas[:, 3] + 2*gas[:, 4]


def _check_close(residual, scale, label):
    residual, scale = np.asarray(residual), np.asarray(scale)
    if (not np.all(np.isfinite(residual)) or not np.all(np.isfinite(scale))
            or np.any(abs(residual) > AUDIT_RTOL*np.maximum(scale, np.finfo(float).tiny))):
        raise RuntimeError(f"{label} conservation ledger failed")
    return float(np.max(abs(residual), initial=0.))


def read_spectrum(filename, energy):
    """Read separate primary photon-number bands, including zero-weight bands."""
    if filename:
        table = np.loadtxt(filename, comments="#", ndmin=2)
        if table.ndim != 2 or table.shape[1] != 2 or not len(table):
            raise ValueError("Source spectrum must contain energy_eV photon_weight columns")
        energies, weights = table.T
    else:
        energies, weights = np.array([float(energy)]), np.ones(1)
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(weights.sum())
    if (np.any(~np.isfinite(energies)) or np.any(energies < 13.6)
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not math.isfinite(total) or total <= 0):
        raise ValueError("Primary spectrum needs finite energies >=13.6 eV and nonnegative weights")
    return energies.copy(), weights/total


def read_request(filename, spectrum_override=None):
    with h5py.File(filename, "r") as handle:
        attrs = dict(handle.attrs)
        required = ("schema_version", "solver", "snapshot", "start_snapshot", "grid_dim",
                    "dt_s", "cell_volume_cm3", "mean_h_atoms_per_cell", "helium_number_ratio",
                    "temperature_K", "redshift", "max_step_s", "source_energy_eV",
                    "angular_level", "epsilon", "speed_fraction", "source_spectrum_file",
                    "run_identity", "library_path")
        missing = [name for name in required if name not in attrs]
        if missing:
            raise ValueError("Missing request attributes: " + ", ".join(missing))
        if _integer(attrs["schema_version"], "schema_version") != SCHEMA:
            raise ValueError("Unsupported request schema_version")
        if _text(attrs["solver"]) != SOLVER:
            raise ValueError("Unsupported request solver")
        side = _integer(attrs["grid_dim"], "grid_dim", 1)
        arrays = {name: _array(handle[name][...], (side**3,), name)
                  for name in ("density", "source_photons")}
    metadata = {"schema_version": SCHEMA, "solver": SOLVER, "grid_dim": side}
    for name in ("snapshot", "start_snapshot", "angular_level"):
        metadata[name] = _integer(attrs[name], name)
    for name in ("dt_s", "helium_number_ratio", "redshift", "epsilon"):
        metadata[name] = _finite_positive(attrs[name], name, allow_zero=True)
    for name in ("cell_volume_cm3", "mean_h_atoms_per_cell", "temperature_K", "max_step_s",
                 "source_energy_eV", "speed_fraction"):
        metadata[name] = _finite_positive(attrs[name], name)
    for name in ("source_spectrum_file", "run_identity", "library_path"):
        metadata[name] = _text(attrs[name])
    if spectrum_override is not None:
        metadata["source_spectrum_file"] = str(spectrum_override)
    if metadata["temperature_K"] != 1e4:
        raise ValueError("This H/He closure requires temperature_K=10000")
    if metadata["speed_fraction"] > 1:
        raise ValueError("Require 0<speed_fraction<=1")
    if metadata["angular_level"] > 8:
        raise ValueError("angular_level must be <=8")
    if metadata["source_energy_eV"] < 13.6:
        raise ValueError("source_energy_eV must be >=13.6")
    if metadata["snapshot"] < metadata["start_snapshot"] or not metadata["run_identity"]:
        raise ValueError("Require snapshot>=start_snapshot and a nonempty run_identity")
    if not metadata["library_path"] or not Path(metadata["library_path"]).is_file():
        raise ValueError("library_path must name an explicitly built angular library")
    metadata["library_path"] = str(Path(metadata["library_path"]).resolve())
    energies, weights = read_spectrum(metadata["source_spectrum_file"], metadata["source_energy_eV"])
    metadata["source_energies_eV"], metadata["source_weights"] = energies.tolist(), weights.tolist()
    if metadata["dt_s"] == 0 and np.any(arrays["source_photons"]):
        raise ValueError("Zero-duration snapshot cannot emit photons")
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name in ("density", "source_photons"):
        digest.update(arrays[name].astype("<f8", copy=False).tobytes())
    return metadata, arrays, digest.hexdigest()


def model_configuration(metadata):
    """All time-independent controls and numerical code identities for a run."""
    names = ("grid_dim", "start_snapshot", "mean_h_atoms_per_cell", "helium_number_ratio",
             "temperature_K", "max_step_s", "source_energies_eV", "source_weights",
             "angular_level", "epsilon", "speed_fraction")
    folder = Path(__file__).resolve().parent
    return dict({name: metadata[name] for name in names}, solver=SOLVER,
                transparency=TRANSPARENCY, max_entries=MAX_ENTRIES,
                max_route_nodes=MAX_ROUTE_NODES, audit_rtol=AUDIT_RTOL,
                library_sha256=hashlib.sha256(Path(metadata["library_path"]).read_bytes()).hexdigest(),
                numerical_sources={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                   for name in NUMERICAL_FILES})


class SnapshotModel:
    """Cumulative stellar, diffuse, recombination and matter-transfer ledgers."""
    def __init__(self, metadata, density):
        self.shape = (metadata["grid_dim"],)*3
        self.size = math.prod(self.shape)
        density = _array(density, (self.size,), "density")
        gas = np.zeros((self.size, 5))
        gas[:, 0] = density
        gas[:, 2] = density*metadata["helium_number_ratio"]
        _array(gas, (self.size, 5), "initial gas")
        self.front = PeriodicFront(
            self.shape, gas, metadata["source_energies_eV"],
            angular_level=metadata["angular_level"], epsilon=metadata["epsilon"],
            transparency=TRANSPARENCY, library_path=metadata["library_path"],
            max_entries=MAX_ENTRIES, max_route_nodes=MAX_ROUTE_NODES,
        )
        self.source_weights = np.asarray(metadata["source_weights"], dtype=float)
        for name, columns in ARRAY_COLUMNS.items():
            setattr(self, name, np.zeros((self.size,) if columns is None else (self.size, columns)))
        self.initial_gas = gas.copy()

    @property
    def gas(self):
        return self.front.gas

    def remap_density(self, density, helium):
        """Hold each element's local fractions; empty-cell arrivals are neutral.

        This prescribed Eulerian remapping is not a material-advection solve.
        Primary and diffuse photon inventories keep their counts and locations.
        """
        density = _array(density, (self.size,), "density")
        helium = _finite_positive(helium, "helium_number_ratio", True)
        old = self.gas.copy()
        totals = _nuclei(old)
        gas = np.zeros_like(old)
        for start, stop, column, abundance in ((0, 2, 0, 1.), (2, 5, 1, helium)):
            fraction = np.divide(old[:, start:stop], totals[:, column, None],
                                 out=np.zeros((self.size, stop-start)),
                                 where=totals[:, column, None] > 0)
            fraction[totals[:, column] == 0, 0] = 1.
            gas[:, start:stop] = density[:, None]*abundance*fraction
        self.front.set_gas(gas)
        self.matter_exchange += gas-old
        self.audit()

    def react(self, volume, duration):
        result = evolve_local(self.gas, volume, duration, pending=self.diffuse_pending,
                              temperature=1e4)
        self.front.set_gas(result["gas"])
        self.diffuse_pending = result["pending"]
        self.capture_counts += result["captures"]
        self.interval_capture_counts += result["captures"]
        self.diffuse_emitted_counts += result["diffuse_emitted"]
        self.diffuse_absorbed_counts += result["diffuse_absorbed"]
        self.diffuse_channel_debits += result["channel_debit"]

    def audit(self):
        """Check cell-wise chemical ledgers and each source/bin/energy account."""
        _array(self.gas, (self.size, 5), "gas")
        for name, columns in ARRAY_COLUMNS.items():
            _array(getattr(self, name), (self.size,) if columns is None else (self.size, columns),
                   name, signed=(name == "matter_exchange"))
        initial_nuclei = _nuclei(self.initial_gas)
        matter_nuclei = _nuclei(self.matter_exchange)
        nuclei = _nuclei(self.gas)
        nucleus_error = _check_close(nuclei-initial_nuclei-matter_nuclei,
                                    initial_nuclei+abs(matter_nuclei)+nuclei,
                                    "H/He nuclei with matter exchange")
        initial_e = _electrons(self.initial_gas)
        exchange_e = _electrons(self.matter_exchange)
        primary = self.photo_counts.sum(1)
        diffuse = self.diffuse_absorbed_counts.sum(1)
        recombinations = self.capture_counts.sum(1)
        electron_error = _check_close(_electrons(self.gas)-initial_e-exchange_e
                                     -primary-diffuse+recombinations,
                                     self.initial_gas.sum(1)+self.gas.sum(1)
                                     +abs(self.matter_exchange).sum(1)+primary+diffuse+recombinations,
                                     "Electron")
        channel_error = _check_close(self.diffuse_emitted_counts-self.diffuse_channel_debits
                                     -self.diffuse_pending, self.diffuse_emitted_counts,
                                     "Local diffuse channel")
        _check_close(self.diffuse_channel_debits.sum(1)-diffuse,
                     self.diffuse_channel_debits.sum(1)+diffuse, "Diffuse species debit")
        state = self.front.export_state()
        resources = {}
        for row in state["resource_ledgers"]:
            key, emitted, absorbed = tuple(row[:3]), float(row[3]), float(row[4])
            if key in resources:
                raise RuntimeError("Duplicate primary resource ledger")
            resources[key] = [emitted, absorbed, 0.]
        for row in state["entries"]:
            key = tuple(row[:3])
            if key not in resources:
                raise RuntimeError("Photon inventory lacks a resource ledger")
            resources[key][2] += float(row[-1])
        emitted_by_source = np.zeros(self.size)
        absorbed_by_source = np.zeros(self.size)
        primary_inventory, resource_error = 0., 0.
        for key, (emitted, absorbed, retained) in resources.items():
            resource_error = max(resource_error, _check_close(
                emitted-absorbed-retained, emitted, "Primary source/bin/energy"))
            emitted_by_source[int(key[0])] += emitted
            absorbed_by_source[int(key[0])] += absorbed
            primary_inventory += retained
        _check_close(emitted_by_source-self.source_history, self.source_history,
                     "Stellar source history")
        _check_close(absorbed_by_source.sum()-primary.sum(), primary.sum(), "Primary species debit")
        emitted = float(self.source_history.sum())
        emitted_diffuse = float(self.diffuse_emitted_counts.sum())
        pending_diffuse = float(self.diffuse_pending.sum())
        photon_error = _check_close(emitted+emitted_diffuse-float(primary.sum())
                                   -float(diffuse.sum())-primary_inventory-pending_diffuse,
                                   emitted+emitted_diffuse, "Global photon")
        return dict(stellar_emitted=emitted, diffuse_emitted=emitted_diffuse,
                    primary_absorbed=float(primary.sum()), diffuse_absorbed=float(diffuse.sum()),
                    gross_recombinations=float(recombinations.sum()),
                    primary_inventory=primary_inventory, diffuse_pending=pending_diffuse,
                    electrons=float(_electrons(self.gas).sum()),
                    initial_electrons=float(initial_e.sum()), external_electrons=float(exchange_e.sum()),
                    hydrogen_nuclei=float(nuclei[:, 0].sum()), helium_nuclei=float(nuclei[:, 1].sum()),
                    photon_error=photon_error, electron_error=electron_error,
                    nucleus_error=nucleus_error, diffuse_channel_error=channel_error,
                    primary_resource_error=resource_error, active_primary_entries=len(state["entries"]),
                    primary_resource_accounts=len(resources), time_s=float(state["time_s"]))


def evolve_snapshot(model, metadata, arrays, progress=False):
    model.interval_capture_counts.fill(0.)
    model.remap_density(arrays["density"], metadata["helium_number_ratio"])
    duration = metadata["dt_s"]
    normalization = metadata["mean_h_atoms_per_cell"]
    volume = _finite_positive(metadata["cell_volume_cm3"]/normalization, "normalized cell volume")
    with np.errstate(over="ignore", invalid="ignore"):
        source = arrays["source_photons"]/normalization
    source = _array(source, (model.size,), "normalized source counts")
    if np.any((arrays["source_photons"] > 0) & (source == 0)):
        raise ArithmeticError("Photon normalization underflows a nonzero source budget")
    if duration == 0 and np.any(source):
        raise ValueError("Zero-duration snapshot cannot emit photons")
    quotient = duration/metadata["max_step_s"]
    if not math.isfinite(quotient) or quotient > MAX_SNAPSHOT_STEPS:
        raise ValueError("Snapshot physical substep limit exceeded")
    steps = max(1, math.ceil(quotient)) if duration else 0
    es_audit = None
    if steps:
        step = duration/steps
        travel = C_CM_S*metadata["speed_fraction"]*step / math.cbrt(metadata["cell_volume_cm3"])
        _finite_positive(travel, "travel_cells", allow_zero=True)
        emitted_so_far = np.zeros_like(source)
        for index in range(steps):
            started = time.monotonic()
            if progress:
                print(f"ANGULAR snapshot={metadata['snapshot']} step={index+1}/{steps} "
                      f"dt_s={step:.8g} travel_cells={travel:.8g} start", flush=True)
            model.react(volume, step/2)
            remaining = source-emitted_so_far
            addition = remaining if index == steps-1 else np.minimum(source/steps, remaining)
            if np.any(addition < 0):
                raise ArithmeticError("Time splitting produced a negative source remainder")
            model.source_history += addition
            emitted_so_far += addition
            es_score = None
            if metadata["epsilon"]:
                es_score, es_audit = periodic_es_scores(
                    model.source_history.reshape(model.shape),
                    model.gas[:, :2].sum(1).reshape(model.shape), return_diagnostics=True)
                es_score = es_score.reshape(-1)
            result = model.front.advance(addition, model.source_weights, travel,
                                         es_scores=es_score, dt_s=step)
            model.photo_counts += result["absorbed"]
            # advance has released any residual from newly transparent cells
            # onto outward edges before the right-half local recombination.
            model.react(volume, step/2)
            diagnostics = model.audit()
            if progress:
                print(f"ANGULAR snapshot={metadata['snapshot']} step={index+1}/{steps} "
                      f"seconds={time.monotonic()-started:.3f} "
                      f"primary_inventory={diagnostics['primary_inventory']:.8g} "
                      f"local_diffuse={diagnostics['diffuse_pending']:.8g} "
                      f"chemistry={result['audits']['chemistry_method']} "
                      f"active_cells={result['audits']['chemistry_active_cells']} "
                      f"chemistry_substeps={result['audits']['chemistry_substeps']} done", flush=True)
    diagnostics = model.audit()
    diagnostics.update(substeps=steps, normalization_atoms=normalization,
                       es_prior=es_audit, temporal_order=1,
                       density_closure="right-endpoint Eulerian fractions with explicit matter exchange",
                       radiation_closure="periodic angular-face allocation, pulse emission, finite travel clock",
                       recombination_closure="T=10000 K case-A captures and retained local coupled-OTS photons")
    return diagnostics


def _write_json(handle, name, value):
    handle.create_dataset(name, data=json.dumps(value, sort_keys=True, allow_nan=False),
                          dtype=h5py.string_dtype(encoding="utf-8"))


def _read_json(handle, name):
    return json.loads(_text(handle[name][()]))


def _write_state(handle, model):
    handle.create_dataset("gas", data=model.gas)
    for name in ARRAY_COLUMNS:
        handle.create_dataset(name, data=getattr(model, name))
    _write_json(handle, "engine_state", model.front.export_state())


def load_checkpoint(path, metadata):
    with h5py.File(path, "r") as handle:
        if int(handle.attrs["schema_version"]) != SCHEMA or _text(handle.attrs["solver"]) != SOLVER:
            raise ValueError("Unsupported checkpoint schema or solver")
        if _text(handle.attrs["run_identity"]) != metadata["run_identity"]:
            raise ValueError("Checkpoint belongs to another run; use a fresh output directory")
        if _read_json(handle, "configuration") != model_configuration(metadata):
            raise ValueError("Angular configuration, spectrum, numerical code or library changed within a run")
        model = SnapshotModel(metadata, np.zeros(metadata["grid_dim"]**3))
        engine_state = _read_json(handle, "engine_state")
        expected = model.front.export_state()
        for key in ("shape", "energies_eV", "angular_level", "epsilon", "transparency",
                    "max_entries", "max_route_nodes"):
            if engine_state[key] != expected[key]:
                raise ValueError(f"Checkpoint engine configuration differs: {key}")
        model.front = PeriodicFront.from_state(engine_state,
                                               library_path=metadata["library_path"])
        if model.front.gas.shape != (model.size, 5):
            raise ValueError("Checkpoint engine grid is incompatible")
        for name, columns in ARRAY_COLUMNS.items():
            shape = (model.size,) if columns is None else (model.size, columns)
            setattr(model, name, _array(handle[name][...], shape, name,
                                       signed=(name == "matter_exchange")))
        if not np.array_equal(handle["gas"][...], model.gas):
            raise ValueError("Checkpoint gas differs from retained engine state")
        previous = _integer(handle.attrs["snapshot"], "checkpoint snapshot")
        digest = _text(handle.attrs["request_digest"])
        diagnostics = _read_json(handle, "diagnostics")
    model.audit()
    return model, previous, digest, diagnostics


def _common_attributes(handle, metadata, digest):
    handle.attrs["schema_version"] = SCHEMA
    handle.attrs["solver"] = SOLVER
    for key in ("snapshot", "grid_dim", "run_identity", "dt_s", "redshift", "temperature_K"):
        handle.attrs[key] = metadata[key]
    handle.attrs["request_digest"] = digest
    handle.attrs["count_normalization_atoms"] = metadata["mean_h_atoms_per_cell"]
    handle.attrs["cell_volume_cm3"] = metadata["cell_volume_cm3"]
    handle.attrs["count_units"] = "internal arrays: count / count_normalization_atoms"


def write_response(handle, model, metadata, digest, diagnostics):
    _common_attributes(handle, metadata, digest)
    totals = _nuclei(model.gas)
    xh = np.divide(model.gas[:, 0], totals[:, 0], out=np.ones(model.size), where=totals[:, 0] > 0)
    he = np.divide(model.gas[:, 2:], totals[:, 1, None], out=np.zeros((model.size, 3)),
                   where=totals[:, 1, None] > 0)
    he[totals[:, 1] == 0, 0] = 1.
    for name, values in (("xH", xh), ("xHeI", he[:, 0]), ("xHeII", he[:, 1]), ("xHeIII", he[:, 2])):
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError(f"Invalid response fraction {name}")
        handle.create_dataset(name, data=values)
    for prefix, counts in (("recombinations_", model.capture_counts),
                           ("snapshot_recombinations_", model.interval_capture_counts)):
        with np.errstate(over="ignore", invalid="ignore"):
            physical = counts*metadata["mean_h_atoms_per_cell"]
        if not np.all(np.isfinite(physical)):
            raise ArithmeticError("Physical recombination event counts overflow")
        for i, species in enumerate(("HII", "HeII", "HeIII")):
            handle.create_dataset(prefix+species, data=physical[:, i])
    handle.attrs["recombination_units"] = "recombinations_*: cumulative physical gross events; snapshot_recombinations_*: interval physical gross events"
    handle.attrs["mean_electron_per_nucleus"] = (float(_electrons(model.gas).sum()/totals.sum())
                                                 if totals.sum() else 0.)
    handle.create_dataset("source_energies_eV", data=metadata["source_energies_eV"])
    handle.create_dataset("source_weights", data=model.source_weights)
    handle.create_dataset("diffuse_opacity_energies_eV", data=OPACITY_E_EV)
    _write_json(handle, "diffuse_channel_names", CHANNEL_NAMES)
    _write_json(handle, "diagnostics", diagnostics)
    _write_json(handle, "configuration", model_configuration(metadata))
    _write_state(handle, model)


def write_checkpoint(handle, model, metadata, digest, diagnostics):
    _common_attributes(handle, metadata, digest)
    _write_json(handle, "configuration", model_configuration(metadata))
    _write_json(handle, "diagnostics", diagnostics)
    _write_state(handle, model)


def _atomic_file(destination, writer):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name+".pending.", dir=destination.parent)
    os.close(descriptor)
    path = Path(name)
    try:
        with h5py.File(path, "w") as handle:
            writer(handle)
            handle.flush()
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sync_directory(path):
    descriptor = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(request, response, checkpoint, spectrum=None, progress=False):
    """Advance under an exclusive lock, committing response before checkpoint."""
    metadata, arrays, digest = read_request(request, spectrum)
    configuration = model_configuration(metadata)
    request, response, checkpoint = Path(request), Path(response), Path(checkpoint)
    if len({path.resolve() for path in (request, response, checkpoint)}) != 3:
        raise ValueError("Request, response and checkpoint paths must differ")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(checkpoint)+".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        replay = False
        if checkpoint.exists():
            model, previous, previous_digest, diagnostics = load_checkpoint(checkpoint, metadata)
            if metadata["snapshot"] == previous:
                if digest != previous_digest:
                    raise ValueError("Snapshot replay request digest differs from checkpoint")
                replay = True
            elif metadata["snapshot"] != previous+1:
                raise ValueError(f"Snapshot {metadata['snapshot']} does not follow checkpoint {previous}")
        else:
            if (metadata["snapshot"] != metadata["start_snapshot"]
                    or metadata["dt_s"] != 0 or np.any(arrays["source_photons"])):
                raise ValueError("Fresh checkpoint requires snapshot=start_snapshot, dt_s=0 and no sources")
            model = SnapshotModel(metadata, arrays["density"])
        if response.exists():
            with h5py.File(response, "r") as existing:
                if (_text(existing.attrs.get("run_identity", "")) != metadata["run_identity"]
                        or _text(existing.attrs.get("request_digest", "")) != digest
                        or _text(existing.attrs.get("solver", "")) != SOLVER):
                    raise ValueError("Response path already contains another snapshot/run")
        if not replay:
            diagnostics = evolve_snapshot(model, metadata, arrays, progress=progress)
        if model_configuration(metadata) != configuration:
            raise RuntimeError("Numerical code or library changed while processing the request")
        staged_response = _atomic_file(response,
            lambda handle: write_response(handle, model, metadata, digest, diagnostics))
        staged_checkpoint = None
        try:
            if not replay:
                staged_checkpoint = _atomic_file(checkpoint,
                    lambda handle: write_checkpoint(handle, model, metadata, digest, diagnostics))
            if model_configuration(metadata) != configuration:
                raise RuntimeError("Numerical code or library changed while staging the request")
            # A crash before checkpoint replacement leaves the previous state;
            # the same request deterministically regenerates this response. A
            # crash after replacement is an exact-digest replay with no advance.
            os.replace(staged_response, response)
            _sync_directory(response)
            if staged_checkpoint is not None:
                os.replace(staged_checkpoint, checkpoint)
                _sync_directory(checkpoint)
        finally:
            staged_response.unlink(missing_ok=True)
            if staged_checkpoint is not None:
                staged_checkpoint.unlink(missing_ok=True)
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--spectrum", type=Path, help="Override source spectrum: energy_eV photon_weight")
    args = parser.parse_args(argv)
    started = time.monotonic()
    diagnostics = run(args.request, args.response, args.checkpoint, args.spectrum, progress=True)
    print(json.dumps(dict(status="ok", solver=SOLVER, seconds=time.monotonic()-started,
                          **diagnostics), sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
