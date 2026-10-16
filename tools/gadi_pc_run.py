#!/usr/bin/env python3
"""Run an isolated, recorded Meraxes PC case inside a Gadi PBS allocation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter(text, key):
    """Read an unambiguous scalar setting without rewriting physics blocks."""
    values = re.findall(r"(?m)^\s*" + re.escape(key) + r"\s*:\s*([^\r\n#{}]*)", text)
    if len(values) > 1:
        raise ValueError(f"Duplicate parameter {key}")
    return values[0].strip() if values else None


def replace_parameter(text, key, value):
    parameter(text, key)  # Reject ambiguous input before modifying it.
    pattern = r"(?m)^[ \t]*" + re.escape(key) + r"[ \t]*:[^\r\n]*$"
    replacement = f"{key} : {value}"
    if re.search(pattern, text):
        return re.sub(pattern, lambda _: replacement, text)
    return text.rstrip() + "\n" + replacement + "\n"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-cwd", type=Path,
                        help="Original input working directory, default: input file's directory")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--memory-mb", type=int, default=700000,
                        help="PC solver guard in MiB; not a whole-job RSS limit")
    parser.add_argument("--debug", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--max-edges", type=int,
                        help="Stored source-target edge cap (0 disables it); omit to preserve the input/default")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    if "PBS_JOBID" not in os.environ or "PBS_NCPUS" not in os.environ:
        parser.error("Launch through PBS; this program does not run simulations on login nodes")
    if not 1 <= args.ranks <= int(os.environ["PBS_NCPUS"]):
        parser.error("--ranks must fit PBS_NCPUS")
    if not 1 <= args.memory_mb <= 2147483647:
        parser.error("--memory-mb must fit a positive C int")
    if args.max_edges is not None and not 0 <= args.max_edges <= 2147483647:
        parser.error("--max-edges must fit a nonnegative C int (0 disables the edge cap)")
    for path in (args.input, args.binary, args.manifest):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    input_path = args.input.resolve()
    working = (args.input_cwd or input_path.parent).resolve()
    if not working.is_dir():
        parser.error(f"Missing original working directory: {working}")
    build = json.loads(args.manifest.read_text())
    if sha256(args.binary) != build["binary_sha256"]:
        parser.error("Binary differs from build manifest; rebuild and repeat")
    launcher = shutil.which("mpirun")
    if not launcher:
        parser.error("mpirun is unavailable; load the matching Gadi OpenMPI environment")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", args.label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    case = args.run_root.resolve() / f"{stamp}_{build['commit'][:12]}_{label}_r{args.ranks}"
    case.mkdir(parents=True, exist_ok=False)
    output = case / "output"
    output.mkdir()
    original_bytes = input_path.read_bytes()
    original = original_bytes.decode()
    (case / "original_input.par").write_bytes(original_bytes)
    effective = original
    inputs = [{"role": "InputFile", "path": str(input_path),
               "sha256": sha256(case / "original_input.par")}]
    for key, name in (("SimParamsFile", "simulation.par"), ("DefaultsFile", "defaults.par")):
        value = parameter(original, key)
        if not value:
            if key == "DefaultsFile":
                raise ValueError("Input must explicitly identify DefaultsFile")
            continue
        source = Path(value)
        source = (working / source).resolve() if not source.is_absolute() else source.resolve()
        destination = case / name
        shutil.copyfile(source, destination)
        inputs.append({"role": key, "path": str(source), "sha256": sha256(destination)})
        effective = replace_parameter(effective, key, destination)
    overrides = {
        "OutputDir": str(output),
        "Flag_PhotonConservation": "1",
        "PhotonConservationShape": "1",
        "PhotonConservationDebug": str(args.debug),
        "PhotonConservationMemoryMB": str(args.memory_mb),
    }
    if args.max_edges is not None:
        overrides["PhotonConservationMaxEdges"] = str(args.max_edges)
    for key, value in overrides.items():
        effective = replace_parameter(effective, key, value)
    prepared = case / "input.par"
    prepared.write_text(effective)
    binary = case / "meraxes"
    shutil.copy2(args.binary, binary)
    if sha256(binary) != build["binary_sha256"]:
        raise RuntimeError("Copied binary does not match build manifest")
    shutil.copy2(args.manifest, case / "build_manifest.json")
    command = [launcher, "-np", str(args.ranks), "--bind-to", "core", "--map-by", "slot",
               str(binary), str(prepared)]
    record = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": build["commit"], "binary_sha256": build["binary_sha256"],
        "ranks": args.ranks, "working_directory": str(working), "command": command,
        "input_files": inputs, "prepared_input_sha256": sha256(prepared),
        "overrides": overrides, "pbs_jobid": os.environ["PBS_JOBID"],
        "pbs_ncpus": int(os.environ["PBS_NCPUS"]),
        "pbs_resources": {key: os.environ[key] for key in ("PBS_MEM", "PBS_NNODES", "PBS_JOBFS")
                          if key in os.environ},
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith("PC_MPI_") or key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    write_json(case / "run.json", record)
    print(f"Run directory: {case}", flush=True)
    print("Command: " + " ".join(command), flush=True)
    start = time.monotonic()
    with (case / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=working, stdout=log, stderr=subprocess.STDOUT, check=False)
    record["elapsed_seconds"] = time.monotonic() - start
    record["exit_code"] = result.returncode
    record["status"] = "complete" if result.returncode == 0 else "failed"
    times = []
    mpi_times = []
    # PC timing is separated from input I/O, galaxy evolution and MPI startup.
    pattern = re.compile(r"PC snapshot=(\d+).*?\btime=([0-9.eE+-]+)s "
                         r"completion=([0-9.eE+-]+)s refinement=([0-9.eE+-]+)s")
    mpi_pattern = re.compile(
        r"PC MPI snapshot=(\d+) ranks=(\d+) accepted=(\d+) declined=(\d+) "
        r"batches=(\d+) targets=(\d+) setup=([0-9.eE+-]+)s query=([0-9.eE+-]+)s "
        r"worker_peak=(\d+) worker_limit_mb=(\d+) min_targets=(\d+)")
    with (case / "run.log").open(errors="replace") as log:
        for line in log:
            match = pattern.search(line)
            if match:
                times.append(dict(zip(("snapshot", "pc_seconds", "completion_seconds", "refinement_seconds"),
                                      (int(match[1]), *(float(match[i]) for i in (2, 3, 4))))))
            mpi_match = mpi_pattern.search(line)
            if mpi_match:
                mpi_times.append(dict(zip(
                    ("snapshot", "ranks", "accepted", "declined", "batches", "targets",
                     "setup_seconds", "query_seconds", "worker_peak_bytes", "worker_limit_mb", "min_targets"),
                    (int(mpi_match[1]), int(mpi_match[2]), int(mpi_match[3]), int(mpi_match[4]),
                     int(mpi_match[5]), int(mpi_match[6]), float(mpi_match[7]), float(mpi_match[8]),
                     int(mpi_match[9]), int(mpi_match[10]), int(mpi_match[11])))))
    record["snapshots"] = times
    record["pc_seconds"] = sum(row["pc_seconds"] for row in times)
    record["completion_seconds"] = sum(row["completion_seconds"] for row in times)
    record["refinement_seconds"] = sum(row["refinement_seconds"] for row in times)
    record["geometry_mpi_snapshots"] = mpi_times
    record["geometry_mpi_diagnostics_present"] = bool(mpi_times)
    record["geometry_mpi_used"] = any(row["batches"] > 0 for row in mpi_times)
    record["geometry_mpi"] = {
        key: sum(row[key] for row in mpi_times)
        for key in ("accepted", "declined", "batches", "targets", "setup_seconds", "query_seconds")
    }
    record["geometry_mpi"]["worker_peak_bytes"] = max(
        (row["worker_peak_bytes"] for row in mpi_times), default=0)
    write_json(case / "run.json", record)
    print(f"Exit={result.returncode}, elapsed={record['elapsed_seconds']:.3f}s, "
          f"PC={record['pc_seconds']:.3f}s, refinement={record['refinement_seconds']:.3f}s, "
          f"MPI geometry batches={record['geometry_mpi']['batches']}", flush=True)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
