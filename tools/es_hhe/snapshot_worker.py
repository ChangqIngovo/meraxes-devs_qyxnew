#!/usr/bin/env python3
"""Online, stateful Meraxes adapter for the fixed-centre H/He ES solver.

The native caller supplies escaped photon *counts* for the interval, density
relative to the cosmic mean, and the physical cell volume. This worker is
called before Meraxes continues its feedback/output, not as postprocessing.
Chemistry is isothermal at 10^4 K. No spatial/angular RT equation is solved.
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

from es_hhe_reference import (MYR, OPACITY_E_EV, SpectralES, evolve_dark,
                              geometric_radii2)
from native_backend import execution_metadata, numerical_identity

SCHEMA = 1
PHOTON_ACTIVITY_FLOOR = 1e-30  # Per cell/band, count / mean_h_atoms_per_cell.
REFERENCE_SHA256 = "86590596b2f365773d8c601971551d1b99161195d3f3f3393d429f0d8ff41d7b"
SCALARS = ("emitted", "diffuse_emitted", "absorbed", "captures", "initial_electrons")
ARRAYS = ("gas", "photons", "source_history", "initial_nuclei", "capture_counts",
          "baseline_nuclei", "matter_exchange", "source_emitted_by_group")


def _text(value):
    return value.decode("utf8") if isinstance(value, bytes) else str(value)


def _finite_positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _integer(value, name, minimum=0):
    numeric = float(value)
    if not math.isfinite(numeric) or int(numeric) != numeric or numeric < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(numeric)


def read_spectrum(filename, energy):
    """Read explicit energies and photon-number weights; retain separate bands."""
    if filename:
        table = np.loadtxt(filename, comments="#", ndmin=2)
        if table.ndim != 2 or table.shape[1] != 2 or not len(table):
            raise ValueError("Source spectrum must contain energy_eV photon_weight columns")
        energies, weights = table.T
    else:
        energies, weights = np.array([float(energy)]), np.ones(1)
    if (np.any(~np.isfinite(energies)) or np.any(energies < 13.6) or
            np.any(~np.isfinite(weights)) or np.any(weights < 0) or
            not math.isfinite(float(weights.sum())) or weights.sum() <= 0):
        raise ValueError("Primary spectrum needs finite energies >= 13.6 eV and nonnegative weights")
    # Keep zero-weight/duplicate bands: accounts are explicit, never merged.
    return energies.copy(), weights / weights.sum()


def read_request(filename, spectrum_override=None):
    with h5py.File(filename, "r") as handle:
        attrs = dict(handle.attrs)
        if _integer(attrs.get("schema_version", SCHEMA), "schema_version") != SCHEMA:
            raise ValueError("Unsupported request schema_version")
        side = _integer(attrs["grid_dim"], "grid_dim", 1)
        size = side**3
        arrays = {}
        for name in ("density", "source_photons"):
            values = np.asarray(handle[name], dtype=np.float64).reshape(-1)
            with np.errstate(over="ignore", invalid="ignore"):
                total = float(values.sum())
            if (values.size != size or np.any(~np.isfinite(values)) or np.any(values < 0)
                    or not math.isfinite(total)):
                raise ValueError(f"{name} must have grid_dim^3 finite nonnegative entries")
            arrays[name] = values
    required = ("snapshot", "dt_s", "cell_volume_cm3", "mean_h_atoms_per_cell",
                "helium_number_ratio", "radius_ratio", "max_radius_cells", "run_identity")
    missing = [name for name in required if name not in attrs]
    if missing:
        raise ValueError("Missing request attributes: " + ", ".join(missing))
    metadata = dict(grid_dim=side, snapshot=_integer(attrs["snapshot"], "snapshot"),
        dt_s=_finite_positive(attrs["dt_s"], "dt_s", True),
        cell_volume_cm3=_finite_positive(attrs["cell_volume_cm3"], "cell_volume_cm3"),
        mean_h_atoms_per_cell=_finite_positive(attrs["mean_h_atoms_per_cell"], "mean_h_atoms_per_cell"),
        helium_number_ratio=_finite_positive(attrs["helium_number_ratio"], "helium_number_ratio", True),
        radius_ratio=_finite_positive(attrs["radius_ratio"], "radius_ratio"),
        max_radius_cells=_finite_positive(attrs["max_radius_cells"], "max_radius_cells", True),
        temperature_K=_finite_positive(attrs.get("temperature_K", 10000.), "temperature_K"),
        max_step_s=_finite_positive(attrs.get("max_step_s", MYR), "max_step_s"),
        run_identity=_text(attrs["run_identity"]),
        redshift=float(attrs.get("redshift", 0.)),
        source_spectrum_file=_text(spectrum_override or attrs.get("source_spectrum_file", "")),
        source_energy_eV=float(attrs.get("source_energy_eV", 18.)))
    if metadata["radius_ratio"] <= 1:
        raise ValueError("radius_ratio must be greater than one")
    if metadata["temperature_K"] != 10000.:
        raise ValueError("This ES closure requires temperature_K=10000")
    if not metadata["run_identity"] or not math.isfinite(metadata["redshift"]) or metadata["redshift"] < 0:
        raise ValueError("A run identity and finite nonnegative redshift are required")
    energies, weights = read_spectrum(metadata["source_spectrum_file"], metadata["source_energy_eV"])
    metadata["source_energies_eV"] = energies.tolist()
    metadata["source_weights"] = weights.tolist()
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for name in ("density", "source_photons"):
        digest.update(arrays[name].astype("<f8", copy=False).tobytes())
    return metadata, arrays, digest.hexdigest()


def model_configuration(metadata):
    names = ("grid_dim", "mean_h_atoms_per_cell", "helium_number_ratio", "radius_ratio",
             "max_radius_cells", "temperature_K", "max_step_s", "source_energies_eV", "source_weights")
    return dict({key: metadata[key] for key in names}, photon_activity_floor=PHOTON_ACTIVITY_FLOOR,
                numerical_backend=numerical_identity())


def verify_reference():
    path = Path(__file__).with_name("es_hhe_reference.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("ES reference differs from the validated version; update its validation before running")


class SnapshotModel(SpectralES):
    """Keep the validated reference intact; generalize its primary-band indexing."""
    def __init__(self, metadata):
        # Parameter zero means the complete periodic minimum-image range,
        # not a zero-radius-only ES catalogue.
        radii = geometric_radii2(metadata["grid_dim"], metadata["radius_ratio"],
                                metadata["max_radius_cells"] or None)
        super().__init__(side=metadata["grid_dim"], helium=metadata["helium_number_ratio"],
                         radii2=radii, photon_floor=PHOTON_ACTIVITY_FLOOR)
        self.primary_count = len(metadata["source_energies_eV"])
        self.energies = np.r_[metadata["source_energies_eV"], OPACITY_E_EV]
        self.source_weights = np.asarray(metadata["source_weights"])
        self.photons = np.zeros((self.size, self.primary_count + 8))
        self.capture_counts = np.zeros((self.size, 3))
        self.source_emitted_by_group = np.zeros(self.primary_count)
        self.baseline_nuclei = self.initial_nuclei.copy()
        self.matter_exchange = np.zeros((self.size, 5))

    def inject_counts(self, counts):
        additions = counts[:, None] * self.source_weights[None, :]
        if np.any(~np.isfinite(additions)) or np.any(additions < 0):
            raise ValueError("Emission must be finite and nonnegative")
        self.photons[:, :self.primary_count] += additions
        self.source_history += additions.sum(axis=1)
        self.source_emitted_by_group += additions.sum(axis=0)
        self.emitted += float(additions.sum())
        self.audit()

    def react(self, dt, max_step=None, cell_volume=1e4):
        if dt == 0:
            return
        # Exactly neutral, photon-free cells have exactly zero local update.
        active = np.flatnonzero(np.any(self.gas[:, [1, 3, 4]] > 0, axis=1) |
                                np.any(self.photons[:, self.primary_count:] > self.photon_floor, axis=1))
        for i in active:
            previous = self.photons[i, self.primary_count:].copy()
            pending = previous / cell_volume
            species, photons, ledger = evolve_dark(self.gas[i] / cell_volume, dt,
                dt if max_step is None else max_step,
                pending=pending, absorption_donor_fraction=self.donor_fraction,
                photon_floor=self.photon_floor / cell_volume)
            self.gas[i] = species * cell_volume
            restored = photons * cell_volume
            # Preserve inactive credits in their original count units. A
            # density conversion must not round a retained subnormal to zero.
            unspent = ledger.absorbed.sum(axis=1) == 0
            restored[unspent] = previous[unspent] + ledger.emitted[unspent] * cell_volume
            underflow = (~unspent) & (previous > 0) & (pending == 0)
            restored[underflow] += previous[underflow]
            self.photons[i, self.primary_count:] = restored
            captures = ledger.captures * cell_volume
            self.capture_counts[i] += captures
            self.diffuse_emitted += float(ledger.emitted.sum() * cell_volume)
            self.absorbed += float(ledger.absorbed.sum() * cell_volume)
            self.captures += float(captures.sum())
        self.audit()

    def set_initial_density(self, density, helium):
        self.gas[:] = 0.
        self.gas[:, 0] = density
        self.gas[:, 2] = density * helium
        self.initial_nuclei = self.nuclei()
        self.baseline_nuclei = self.initial_nuclei.copy()
        self.initial_electrons = self.electrons()

    def remap_density(self, density, helium):
        """Prescribed local fractions retained; record external matter transfer.

        This is an Eulerian density closure, not inferred material advection.
        Matter entering a previously empty cell is neutral. Photon counts stay.
        """
        old = self.gas.copy()
        totals = self.nuclei()
        h_fraction = np.divide(old[:, :2], totals[:, 0, None],
            out=np.zeros((self.size, 2)), where=totals[:, 0, None] > 0)
        h_fraction[totals[:, 0] == 0, 0] = 1.
        he_fraction = np.divide(old[:, 2:], totals[:, 1, None],
            out=np.zeros((self.size, 3)), where=totals[:, 1, None] > 0)
        he_fraction[totals[:, 1] == 0, 0] = 1.
        # Preserve tiny neutral/ionized tails directly; 1 - x_ion would erase
        # a representable minority species next to a rounded unit fraction.
        self.gas[:, :2] = density[:, None] * h_fraction
        self.gas[:, 2:] = density[:, None] * helium * he_fraction
        exchange = self.gas - old
        self.matter_exchange += exchange
        self.initial_nuclei = self.nuclei()
        self.initial_electrons += float((exchange[:, 1] + exchange[:, 3] + 2 * exchange[:, 4]).sum())
        self.audit()

    def audit_exchange(self):
        result = self.audit()
        for name in SCALARS:
            if not math.isfinite(getattr(self, name)):
                raise AssertionError(f"Nonfinite ledger scalar {name}")
        with np.errstate(over="ignore", invalid="ignore"):
            total_nuclei = float(self.initial_nuclei.sum())
        if not math.isfinite(total_nuclei):
            raise AssertionError("Total nuclei overflow")
        transferred_nuclei = np.column_stack((self.matter_exchange[:, :2].sum(axis=1),
                                               self.matter_exchange[:, 2:].sum(axis=1)))
        residual = self.nuclei() - self.baseline_nuclei - transferred_nuclei
        scale = np.maximum(self.baseline_nuclei + abs(transferred_nuclei), np.finfo(float).tiny)
        if np.any(abs(residual) > 2e-9 * scale):
            raise AssertionError("Matter-transfer nucleus ledger failed")
        result["matter_transfer_nuclei_error"] = float(abs(residual).max())
        result["external_electrons"] = float((self.matter_exchange[:, 1] +
            self.matter_exchange[:, 3] + 2 * self.matter_exchange[:, 4]).sum())
        result["source_group_error"] = float(self.source_emitted_by_group.sum() - self.emitted)
        if abs(result["source_group_error"]) > 2e-9 * max(self.emitted, np.finfo(float).tiny):
            raise AssertionError("Primary source group ledger failed")
        capture_error = float(self.capture_counts.sum()) - self.captures
        if abs(capture_error) > 2e-9 * max(self.captures, np.finfo(float).tiny):
            raise AssertionError("Per-cell recombination ledger failed")
        result["capture_count_error"] = capture_error
        dormant=(self.photons>0)&(self.photons<=self.photon_floor)
        result["photon_activity_floor"] = self.photon_floor
        result["deferred_photon_accounts"] = int(np.count_nonzero(dormant))
        result["deferred_photons"] = float(self.photons[dormant].sum())
        return result


def load_checkpoint(path, metadata):
    model = SnapshotModel(metadata)
    with h5py.File(path, "r") as handle:
        if int(handle.attrs["schema"]) != SCHEMA:
            raise ValueError("Unsupported checkpoint schema")
        if _text(handle.attrs["reference_sha256"]) != REFERENCE_SHA256:
            raise ValueError("Checkpoint uses a different ES reference")
        if _text(handle.attrs["run_identity"]) != metadata["run_identity"]:
            raise ValueError("Checkpoint belongs to another run; use a fresh output directory")
        if json.loads(_text(handle.attrs["configuration"])) != model_configuration(metadata):
            raise ValueError("ES configuration or primary spectrum changed within a run")
        for name in ARRAYS:
            expected = getattr(model, name).shape
            value = np.asarray(handle[name], dtype=np.float64)
            if value.shape != expected or np.any(~np.isfinite(value)):
                raise ValueError(f"Invalid checkpoint array {name}")
            if name != "matter_exchange" and np.any(value < 0):
                raise ValueError(f"Negative checkpoint array {name}")
            setattr(model, name, value)
        for name in SCALARS:
            setattr(model, name, float(handle.attrs[name]))
        last_snapshot = int(handle.attrs["snapshot"])
        last_digest = _text(handle.attrs["request_digest"])
        diagnostics = json.loads(_text(handle.attrs["diagnostics"]))
    model.audit_exchange()
    return model, last_snapshot, last_digest, diagnostics


def evolve_snapshot(model, metadata, arrays, progress=False):
    model.remap_density(arrays["density"], metadata["helium_number_ratio"])
    dt = metadata["dt_s"]
    normalization = metadata["mean_h_atoms_per_cell"]
    volume = metadata["cell_volume_cm3"] / normalization
    _finite_positive(volume, "normalized cell volume")
    source = arrays["source_photons"] / normalization
    if np.any(~np.isfinite(source)):
        raise ValueError("Normalized source counts overflow")
    steps = max(1, math.ceil(dt / metadata["max_step_s"]))
    if dt == 0 and np.any(source):
        raise ValueError("Zero-duration snapshot cannot emit photons")
    if dt > 0:
        step = dt / steps
        emitted_so_far = np.zeros_like(source)
        for index in range(steps):
            started = time.monotonic()
            def report(info):
                fields = ' '.join(f'{key}={value}' for key, value in info.items())
                print(f"ES snapshot={metadata['snapshot']} substep={index + 1}/{steps} {fields}", flush=True)
            if progress:
                print(f"ES snapshot={metadata['snapshot']} substep={index + 1}/{steps} "
                      f"dt_Myr={step / MYR:.8g} retained={model.photons.sum():.8g} start", flush=True)
                report(dict(phase='recombine_before',state='start'))
            model.react(step / 2, cell_volume=volume)
            if progress:
                report(dict(phase='recombine_before',state='done'))
            # The final remainder closes the prescribed total despite rounding.
            addition = source - emitted_so_far if index == steps - 1 else source / steps
            if np.any(addition < 0):
                raise ArithmeticError("Time splitting produced a negative source remainder")
            model.inject_counts(addition)
            emitted_so_far += addition
            model.illuminate(progress=report if progress else None)
            if progress:
                report(dict(phase='recombine_after',state='start'))
            model.react(step / 2, cell_volume=volume)
            if progress:
                report(dict(phase='recombine_after',state='done'))
                print(f"ES snapshot={metadata['snapshot']} substep={index + 1}/{steps} "
                      f"seconds={time.monotonic() - started:.3f} "
                      f"retained={model.photons.sum():.8g} absorbed_total={model.absorbed:.8g} done", flush=True)
    diagnostics = model.audit_exchange()
    diagnostics.update(substeps=steps if dt else 0, es_counters=model.counters,
                       normalization_atoms=normalization)
    return diagnostics


def _atomic_file(destination, writer):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + ".pending.", dir=destination.parent)
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


def write_response(handle, model, metadata, digest, diagnostics):
    handle.attrs["schema"] = SCHEMA
    handle.attrs["schema_version"] = SCHEMA
    for key in ("snapshot", "grid_dim", "dt_s", "run_identity", "redshift", "temperature_K"):
        handle.attrs[key] = metadata[key]
    handle.attrs["request_digest"] = digest
    handle.attrs["reference_sha256"] = REFERENCE_SHA256
    handle.attrs["photon_activity_floor"] = model.photon_floor
    handle.attrs["execution_backend"] = json.dumps(execution_metadata(), sort_keys=True)
    handle.attrs["diagnostics"] = json.dumps(diagnostics, sort_keys=True, allow_nan=False)
    handle.attrs["count_normalization_atoms"] = metadata["mean_h_atoms_per_cell"]
    handle.attrs["cell_volume_cm3"] = metadata["cell_volume_cm3"]
    handle.attrs["primary_band_count"] = model.primary_count
    handle.attrs["count_units"] = "gas, photons, capture_counts, matter_exchange: count / count_normalization_atoms"
    handle.attrs["recombination_units"] = "HII_recombinations, HeII_recombinations, HeIII_recombinations: actual event count"
    totals = model.nuclei()
    xh = np.divide(model.gas[:, 0], totals[:, 0], out=np.ones(model.size), where=totals[:, 0] > 0)
    he = np.divide(model.gas[:, 2:], totals[:, 1, None], out=np.zeros((model.size, 3)),
                   where=totals[:, 1, None] > 0)
    he[totals[:, 1] == 0, 0] = 1.
    for name, values in (("xH", xh), ("xHeI", he[:, 0]), ("xHeII", he[:, 1]), ("xHeIII", he[:, 2])):
        if np.any(~np.isfinite(values)) or np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
            raise AssertionError(f"Invalid output fraction {name}")
        handle.create_dataset(name, data=values)
    with np.errstate(over="ignore", invalid="ignore"):
        physical_captures = model.capture_counts * metadata["mean_h_atoms_per_cell"]
    if np.any(~np.isfinite(physical_captures)):
        raise ArithmeticError("Physical recombination event counts overflow")
    for index, name in enumerate(("HII_recombinations", "HeII_recombinations", "HeIII_recombinations")):
        handle.create_dataset(name, data=physical_captures[:, index])
    handle.create_dataset("source_energies_eV", data=model.energies[:model.primary_count])
    handle.create_dataset("source_weights", data=model.source_weights)
    handle.create_dataset("photon_energies_eV", data=model.energies)
    for name in ARRAYS:
        handle.create_dataset(name, data=getattr(model, name))
    for name in SCALARS:
        handle.attrs[name] = getattr(model, name)
    denominator = float(totals.sum())
    handle.attrs["mean_electron_per_nucleus"] = model.electrons() / denominator if denominator else 0.


def write_checkpoint(handle, model, metadata, digest, diagnostics):
    handle.attrs["schema"] = SCHEMA
    handle.attrs["reference_sha256"] = REFERENCE_SHA256
    handle.attrs["photon_activity_floor"] = model.photon_floor
    handle.attrs["configuration"] = json.dumps(model_configuration(metadata), sort_keys=True)
    handle.attrs["execution_backend"] = json.dumps(execution_metadata(), sort_keys=True)
    for key in ("snapshot", "run_identity"):
        handle.attrs[key] = metadata[key]
    handle.attrs["request_digest"] = digest
    handle.attrs["diagnostics"] = json.dumps(diagnostics, sort_keys=True, allow_nan=False)
    for name in ARRAYS:
        handle.create_dataset(name, data=getattr(model, name))
    for name in SCALARS:
        handle.attrs[name] = getattr(model, name)


def run(request, response, checkpoint, spectrum=None, progress=False):
    """Advance one interval under an exclusive lock; repeated requests are idempotent."""
    verify_reference()
    metadata, arrays, digest = read_request(request, spectrum)
    response, checkpoint = Path(response), Path(checkpoint)
    if response.resolve() == checkpoint.resolve() or Path(request).resolve() in (response.resolve(), checkpoint.resolve()):
        raise ValueError("Request, response and checkpoint paths must differ")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(checkpoint) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        replay = False
        if checkpoint.exists():
            model, previous, previous_digest, diagnostics = load_checkpoint(checkpoint, metadata)
            if metadata["snapshot"] == previous and digest == previous_digest:
                replay = True
            elif metadata["snapshot"] != previous + 1:
                raise ValueError(f"Snapshot {metadata['snapshot']} does not follow checkpoint {previous}")
        else:
            if metadata["dt_s"] != 0 or np.any(arrays["source_photons"]):
                raise ValueError("First ES snapshot must initialize neutral gas with dt_s=0 and no sources")
            model = SnapshotModel(metadata)
            model.set_initial_density(arrays["density"], metadata["helium_number_ratio"])
        if response.exists():
            with h5py.File(response, "r") as existing:
                if (_text(existing.attrs.get("run_identity", "")) != metadata["run_identity"] or
                        _text(existing.attrs.get("request_digest", "")) != digest):
                    raise ValueError("Response path already contains another snapshot/run")
        if not replay:
            diagnostics = evolve_snapshot(model, metadata, arrays, progress=progress)
        staged_response = _atomic_file(response,
            lambda handle: write_response(handle, model, metadata, digest, diagnostics))
        staged_checkpoint = None
        try:
            if not replay:
                staged_checkpoint = _atomic_file(checkpoint,
                    lambda handle: write_checkpoint(handle, model, metadata, digest, diagnostics))
            # Response is complete before committing state. An interrupted pair
            # is safely regenerated from the old checkpoint or replayed from the
            # new one; Meraxes only consumes output after worker success.
            os.replace(staged_response, response)
            if staged_checkpoint is not None:
                os.replace(staged_checkpoint, checkpoint)
        finally:
            staged_response.unlink(missing_ok=True)
            if staged_checkpoint is not None:
                staged_checkpoint.unlink(missing_ok=True)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--spectrum", type=Path, help="Override primary spectrum: energy_eV photon_weight")
    args = parser.parse_args()
    start = time.monotonic()
    print("ES execution:", json.dumps(execution_metadata(), sort_keys=True), flush=True)
    diagnostics = run(args.request, args.response, args.checkpoint, args.spectrum, progress=True)
    print(json.dumps(dict(status="ok", seconds=time.monotonic() - start, **diagnostics),
                     sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
