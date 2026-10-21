#!/usr/bin/env python3
"""Run the open-boundary, pure-hydrogen angular-front research solver.

No recombination, helium, periodic transport, or production worker is enabled.
Build explicitly with build.py first. Input NPZ files are loaded without pickle.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REQUIRED = {"capacity", "initial_mass", "source_indices", "source_rates"}
OPTIONAL = {"initial_source_photons", "es_score"}


def _array(name, value):
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite real numbers")
    result = np.ascontiguousarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} cannot be represented as finite binary64 numbers")
    return result


def read_input(path):
    with np.load(path, allow_pickle=False) as handle:
        names = set(handle.files)
        missing, extra = REQUIRED - names, names - REQUIRED - OPTIONAL
        if missing or extra:
            raise ValueError(f"Invalid NPZ fields: missing={sorted(missing)}, unexpected={sorted(extra)}")
        data = {key: _array(key, handle[key]) for key in names}
    cap, initial = data["capacity"], data["initial_mass"]
    if cap.ndim != 3 or min(cap.shape) < 1 or max(cap.shape) >= np.iinfo(np.int32).max:
        raise ValueError("capacity must be a nonempty three-dimensional grid")
    if initial.shape != cap.shape or np.any(cap < 0) or np.any(initial < 0) or np.any(initial > cap):
        raise ValueError("initial_mass must match capacity and satisfy 0 <= initial_mass <= capacity")
    rates, src = data["source_rates"], data["source_indices"]
    if rates.ndim != 1 or not len(rates) or np.any(rates < 0):
        raise ValueError("source_rates must be a nonempty nonnegative vector")
    if src.shape != (len(rates), 3) or np.any(src != np.floor(src)):
        raise ValueError("source_indices must have shape (sources, 3) and contain integer coordinates")
    if np.any(src < 0) or np.any(src >= np.asarray(cap.shape)):
        raise ValueError("source_indices must lie inside the grid")
    data["source_indices"] = src.astype(np.int32)
    history = data.get("initial_source_photons", np.zeros_like(rates))
    if history.shape != rates.shape or np.any(history < 0):
        raise ValueError("initial_source_photons must be a nonnegative vector matching source_rates")
    data["initial_source_photons"] = history
    if "es_score" in data and (data["es_score"].shape != cap.shape or np.any(data["es_score"] < 0)):
        raise ValueError("es_score must match the capacity grid and be finite and nonnegative")
    return data


def write_npz(path, arrays):
    path = path.expanduser().resolve()
    if path.suffix != ".npz":
        raise ValueError("NPZ output path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def example(path):
    """A neutral two-source problem; no free pre-ionized seed cells."""
    shape = (9, 9, 9)
    path = write_npz(path, dict(capacity=np.ones(shape), initial_mass=np.zeros(shape),
        source_indices=np.array([[3, 4, 4], [5, 4, 4]], dtype=np.int32),
        source_rates=np.array([4., 4.]), initial_source_photons=np.zeros(2)))
    print(json.dumps({"example": str(path), "boundary": "open", "recombination": False,
                      "run": f"python3 tools/es_angular/run.py --input {path} --output result.npz --times 0 1 4 12"}))


def main(argv=None):
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="NPZ containing capacity, initial_mass, source_indices, source_rates")
    mode.add_argument("--write-example", type=Path, help="Write a reproducible pure-H two-source NPZ and exit")
    parser.add_argument("--output", type=Path, help="Result NPZ; a readable JSON sidecar is also written")
    parser.add_argument("--library", type=Path, help="Existing library; otherwise use ES_ANGULAR_LIBRARY or adjacent libangular_front.so")
    parser.add_argument("--dx", type=float, default=1.)
    parser.add_argument("--level", type=int, default=0, help="Angular subdivision level, 0 through 8")
    parser.add_argument("--epsilon", type=float, default=0., help="Nonnegative ES modulation bound parameter")
    parser.add_argument("--uniform-density", type=float, default=0., help="Enable the analytic uniform-medium score; capacity must equal density*dx^3")
    parser.add_argument("--times", nargs="+", default=["1"], help="Ordered nonnegative output times; spaces or commas accepted")
    parser.add_argument("--max-events", type=int, default=1000000, help="Maximum events per requested output interval")
    args = parser.parse_args(argv)
    if args.write_example:
        if args.output or args.library:
            parser.error("--write-example does not use --output or --library")
        example(args.write_example)
        return 0
    if args.output is None:
        parser.error("--output is required with --input")
    if not math.isfinite(args.dx) or args.dx <= 0:
        parser.error("--dx must be finite and positive")
    if not 0 <= args.level <= 8 or args.max_events < 1:
        parser.error("--level must lie in [0, 8] and --max-events must be positive")
    if not math.isfinite(args.epsilon) or args.epsilon < 0 or not math.isfinite(args.uniform_density) or args.uniform_density < 0:
        parser.error("--epsilon and --uniform-density must be finite and nonnegative")
    times = np.array([float(item) for token in args.times for item in token.split(",")])
    if not len(times) or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
        parser.error("--times must be finite, nonnegative, and strictly increasing")
    source_path, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if output == source_path or output.suffix != ".npz":
        parser.error("--output must be a different .npz path from --input")
    data = read_input(source_path)
    if args.uniform_density > 0:
        if "es_score" in data:
            parser.error("Choose either es_score or --uniform-density, not both")
        expected = args.uniform_density * args.dx ** 3
        if not math.isfinite(expected) or not np.allclose(data["capacity"], expected, rtol=1e-12, atol=0.):
            parser.error("Uniform analytic score requires capacity = uniform_density * dx^3 in every cell")
    if args.epsilon > 0 and args.uniform_density == 0 and "es_score" not in data:
        parser.error("Positive --epsilon requires es_score or a positive --uniform-density")
    library = (args.library or Path(os.environ.get("ES_ANGULAR_LIBRARY", str(folder / "libangular_front.so")))).expanduser().resolve()
    if not library.is_file():
        parser.error(f"Missing native library: {library}. Run tools/es_angular/build.py explicitly first")
    os.environ["ES_ANGULAR_LIBRARY"] = str(library)
    from angular_front import AngularFront
    records = []
    with AngularFront(data["capacity"], data["initial_mass"], args.dx,
                      data["source_indices"], data["source_rates"], level=args.level,
                      epsilon=args.epsilon, initial_source_photons=data["initial_source_photons"],
                      uniform_density=args.uniform_density) as solver:
        if "es_score" in data:
            solver.set_es_scores(data["es_score"])
        for time in times:
            snapshot = solver.advance(float(time), max_events=args.max_events)
            resource_error = snapshot["resource_absorbed"] + snapshot["resource_escaped"] - time * snapshot["resource_rate"]
            absorbed = float(snapshot["resource_absorbed"].sum())
            increment = float((snapshot["mass"] - data["initial_mass"]).sum())
            diagnostics = snapshot["diagnostics"]
            if not all(math.isfinite(value) for value in diagnostics.values()):
                raise RuntimeError("Angular-front diagnostics contain nonfinite values")
            diagnostics["max_resource_ledger_error"] = float(np.max(np.abs(resource_error)))
            diagnostics["mass_ledger_error"] = increment - absorbed
            diagnostics["mass_ledger_error_after_recorded_clamp"] = increment - absorbed - diagnostics["clamp_signed_mass"]
            scale = max(1., float(time * data["source_rates"].sum()), absorbed)
            if (not all(np.all(np.isfinite(value)) for key, value in snapshot.items() if key != "diagnostics") or
                    diagnostics["max_resource_ledger_error"] > 1e-9 * scale or
                    abs(diagnostics["mass_ledger_error_after_recorded_clamp"]) > 1e-9 * scale):
                raise RuntimeError("Angular-front result failed its photon ledger audit")
            snapshot["resource_balance_error"] = resource_error
            records.append(snapshot)
    metadata = {
        "schema": 1, "solver": "angular-front-c99", "species": "hydrogen only",
        "boundary": "open", "recombination": False, "production_mode3": "retired",
        "closure": "opaque-until-full cells; finite angular-bin mixing",
        "time": "allocation time; no finite-light-speed interpretation",
        "initial_source_photons": "ES score history only; not newly emitted photons in output ledgers",
        "source_mass": "absorption since time zero, excluding initial_mass",
        "units": "capacity and mass: atom counts; rates: photon counts per chosen time unit; dx: chosen length unit",
        "input": str(source_path), "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "shape": list(data["capacity"].shape), "sources": len(data["source_rates"]),
        "angular_bins": int(records[0]["resource_absorbed"].shape[1]),
        "dx": args.dx, "level": args.level, "epsilon": args.epsilon,
        "uniform_density": args.uniform_density, "external_es_score": "es_score" in data,
        "times": times.tolist(), "max_events_per_interval": args.max_events,
    }
    arrays = {key: np.stack([row[key] for row in records]) for key in records[0] if key != "diagnostics"}
    arrays.update(times=times, capacity=data["capacity"], initial_mass=data["initial_mass"],
                  source_indices=data["source_indices"], source_rates=data["source_rates"],
                  metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                  diagnostics_json=np.array([json.dumps(row["diagnostics"], sort_keys=True) for row in records]))
    output = write_npz(output, arrays)
    summary = {"metadata": metadata, "diagnostics": [row["diagnostics"] for row in records]}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "metadata": str(output.with_suffix(".json")),
                      "final_diagnostics": records[-1]["diagnostics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, OverflowError) as exc:
        raise SystemExit(f"Angular-front run failed: {exc}") from exc
