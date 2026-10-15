#!/usr/bin/env python3
"""Run the small L10_N512 photon-conservation integration validations.

Requires a built Meraxes executable, MPI launcher, unpacked L10_N512 data,
NumPy and h5py. This script does not build, download data, or install anything.
It runs jobs sequentially, saves Debug=2 certificates, audits each certificate,
and reports differences between MPI decompositions without assuming bitwise
identical galaxy histories or source deposition.

Example (execute from the repository root):
  python tools/run_photon_conservation_validation.py \
    --binary build/bin/meraxes --data /data/L10_N512 --work-dir /tmp/pc-check \
    --configs default igm xray --ranks 1 2 --snapshots 48 63 75

The chosen executable must support the requested number of MPI ranks. A
RelWithDebInfo build avoids the original code's Debug galaxy-pointer checks.
Use --audit-only to inspect existing outputs without running the simulation.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("default", "igm", "xray")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_checker():
    path = Path(__file__).with_name("check_photon_conservation.py")
    spec = importlib.util.spec_from_file_location("pc_certificate_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_input(args, configuration, folder):
    template = (REPOSITORY / "src/tests/regression/setups" / f"{configuration}.par.in").read_text()
    template = "\n".join(line for line in template.splitlines() if not line.lstrip().startswith("#"))
    template = f"# Photon-conservation validation: {configuration} physics, Debug=2 certificates.\n" + template.lstrip()
    # SimulationDir is assigned directly so --data can name either the dataset
    # directory itself or its parent without relying on its local basename.
    template = template.replace("@INPUT_FILE_DIR@", str(args.input_dir))
    template = template.replace("@MERAXES_TESTDATA_DIR@/L10_N512", str(args.data))
    template = re.sub(r"(?m)^OutputSnapshots\s*:.*$", "OutputSnapshots : " +
                      ", ".join(map(str, args.snapshots)), template)
    template += ("\nFlag_PhotonConservation : 1\n"
                 "PhotonConservationDebug : 2\n"
                 "PhotonConservationShape : 1\n"
                 f"PhotonConservationMaxIterations : {args.max_iterations}\n"
                 f"PhotonConservationMaxCuts : {args.max_cuts}\n"
                 f"PhotonConservationMaxEvaluations : {args.max_evaluations}\n"
                 f"PhotonConservationTolerance : {args.tolerance:.17g}\n")
    (folder / "input.par").write_text(template)


def run_case(args, configuration, ranks, folder):
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError(f"refusing to overwrite nonempty {folder}; use a new --work-dir or --audit-only")
    folder.mkdir(parents=True, exist_ok=True)
    prepare_input(args, configuration, folder)
    temporary = folder / "mpi_tmp"
    temporary.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(TMPDIR=str(temporary), OMPI_MCA_orte_tmpdir_base=str(temporary))
    command = [args.mpiexec, *args.mpi_arg, "-np", str(ranks), str(args.binary), "input.par"]
    print(f"Running {configuration}, {ranks} MPI ranks: {folder}", flush=True)
    binary_digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    start = time.monotonic()
    with (folder / "run.log").open("w", buffering=1) as stream:
        with subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                if "PC snapshot=" in line or "PC shape failure:" in line:
                    print("  " + line.strip(), flush=True)
            exit_code = process.wait()
    status = {"command": command, "working_directory": str(folder), "binary_sha256": binary_digest,
              "exit_code": exit_code, "elapsed_seconds": time.monotonic() - start}
    write_json(folder / "run_status.json", status)
    return status


def audit_case(args, checker, folder):
    reports = []
    found = set()
    for path in sorted((folder / "output").glob("photon_conservation_*.hdf5")):
        try:
            report = checker.inspect(path, path.with_suffix(".edges.csv"), args.atol, args.rtol,
                                     require_cells=True, require_full=args.require_full)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            report = {"file": str(path), "passed": False, "error": str(error)}
        expected_snapshot = int(path.stem.rsplit("_", 1)[1])
        if report.get("snapshot") == expected_snapshot:
            found.add(expected_snapshot)
        elif "error" not in report:
            report["passed"] = False
            report["error"] = (f"recorded snapshot {report.get('snapshot')} does not match "
                               f"filename snapshot {expected_snapshot}")
        reports.append(report)
    missing = sorted(set(args.snapshots) - found)
    missing_all = sorted(set(range(max(args.snapshots) + 1)) - found)
    result = {"files": len(reports), "missing_requested_snapshots": missing,
              "missing_snapshots": missing_all,
              "passed": bool(reports) and not missing_all and all(report["passed"] for report in reports),
              "reports": reports}
    write_json(folder / "certificate_audit.json", result)
    failures = [report for report in reports if not report["passed"]]
    print(f"  Certificate audit: {len(reports) - len(failures)}/{len(reports)} pass; "
          f"missing snapshots: {missing_all}", flush=True)
    return result


def compare_ranks(first, second):
    reports = []
    for path in sorted((first / "output").glob("photon_conservation_*.hdf5")):
        other = second / "output" / path.name
        if not other.exists():
            continue
        with h5py.File(path, "r") as a, h5py.File(other, "r") as b:
            report = {"snapshot": int(np.asarray(a.attrs["snapshot"]).item())}
            complete = all(int(np.asarray(handle.attrs.get("certificate_complete", 0)).item()) == 1
                           for handle in (a, b))
            report["both_certificates_complete"] = complete
            if not complete:
                reports.append(report)
                continue
            same_inputs = True
            for name in ("source", "capacity", "baseline", "first_crossing", "lower", "upper",
                         "radius_cells", "priority", "allocation", "xH"):
                if name not in a or name not in b:
                    same_inputs = False
                    continue
                av = np.asarray(a[name][...], dtype=float).reshape(-1)
                bv = np.asarray(b[name][...], dtype=float).reshape(-1)
                if av.shape != bv.shape:
                    report[name] = {"same_shape": False}
                    same_inputs = False
                    continue
                difference = np.abs(av - bv)
                same = bool(np.array_equal(av, bv))
                report[name] = {"same_shape": True, "exactly_equal_values": same,
                                "different_cells": int(np.count_nonzero(difference)),
                                "max_absolute_difference": float(np.max(difference, initial=0.0)),
                                "sum_absolute_difference": float(np.sum(difference)),
                                "root_mean_square_difference": float(np.sqrt(np.mean(difference ** 2))),
                                "first_sum": float(np.sum(av)), "second_sum": float(np.sum(bv))}
                if name not in ("allocation", "xH"):
                    same_inputs &= same
            report["identical_recorded_solver_inputs"] = same_inputs
            if (report.get("allocation", {}).get("same_shape", False)
                    and report.get("source", {}).get("same_shape", False)):
                scale = max(abs(report["source"]["first_sum"]), abs(report["source"]["second_sum"]))
                report["allocation_L1_over_source_budget"] = (
                    report["allocation"]["sum_absolute_difference"] / scale if scale else 0.0)
            reports.append(report)
    return {"first": str(first), "second": str(second), "common_snapshots": len(reports),
            "interpretation": "Differences are measurements, not an assertion of MPI invariance. "
                              "Nonidentical source or ES inputs can produce different optimal maps.",
            "reports": reports}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, help="Meraxes executable; required unless --audit-only")
    parser.add_argument("--data", type=Path, help="L10_N512 directory or its parent; required unless --audit-only")
    parser.add_argument("--work-dir", type=Path, required=True, help="destination for new case directories and JSON reports")
    parser.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--snapshots", nargs="+", type=int, default=[48, 63, 75])
    parser.add_argument("--input-dir", type=Path, default=REPOSITORY / "input")
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--mpi-arg", action="append", default=[], help="extra launcher argument; repeat as --mpi-arg=VALUE")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="native solver relative tolerance")
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--max-cuts", type=int, default=512, help="native Hall-cut storage cap")
    parser.add_argument("--max-evaluations", type=int, default=1000000, help="native quadratic-master evaluation cap")
    parser.add_argument("--atol", type=float, default=1e-11, help="independent certificate audit absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-8, help="independent certificate audit relative tolerance")
    parser.add_argument("--require-full", action="store_true", help="require full source absorption within the count tolerance")
    parser.add_argument("--audit-only", action="store_true", help="audit and compare existing case directories without running MPI")
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.input_dir = args.input_dir.resolve()
    args.ranks = sorted(set(args.ranks))
    args.snapshots = sorted(set(args.snapshots))
    args.configs = list(dict.fromkeys(args.configs))
    if any(rank < 1 for rank in args.ranks) or any(snap < 0 or snap > 75 for snap in args.snapshots):
        parser.error("ranks must be positive and L10_N512 snapshots must be between 0 and 75")
    if min(args.max_iterations, args.max_cuts, args.max_evaluations) < 1 or not all(np.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("iteration, cut and evaluation caps must be positive and audit tolerances finite and nonnegative")
    if not np.isfinite(args.tolerance) or not 0 < args.tolerance <= 1e-5:
        parser.error("solver tolerance must satisfy 0 < tolerance <= 1e-5")
    if not args.audit_only:
        if args.binary is None or args.data is None:
            parser.error("--binary and --data are required unless --audit-only")
        args.binary = args.binary.resolve()
        args.data = args.data.resolve()
        if (args.data / "L10_N512").is_dir():
            args.data /= "L10_N512"
        if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
            parser.error(f"binary is not executable: {args.binary}")
        if not args.data.is_dir() or not (args.data / "a_list.txt").exists():
            parser.error(f"dataset directory does not contain a_list.txt: {args.data}")
        if not (args.input_dir / "params/defaults.par").exists():
            parser.error(f"invalid input directory: {args.input_dir}")
        launcher = shutil.which(args.mpiexec)
        if launcher is None:
            parser.error(f"MPI launcher unavailable: {args.mpiexec}")
        args.mpiexec = str(Path(launcher).resolve())
        # Detect every collision before running the first job.
        for config, rank in itertools.product(args.configs, args.ranks):
            folder = args.work_dir / f"{config}_r{rank}"
            if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
                parser.error(f"refusing to overwrite nonempty {folder}; use a new --work-dir or --audit-only")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    checker = load_checker()
    result = {"configs": args.configs, "ranks": args.ranks, "snapshots": args.snapshots,
              "audit_only": args.audit_only, "passed": True, "cases": [], "rank_comparisons": []}
    try:
        for configuration, ranks in itertools.product(args.configs, args.ranks):
            folder = args.work_dir / f"{configuration}_r{ranks}"
            if args.audit_only and not folder.is_dir():
                raise ValueError(f"missing case directory: {folder}")
            if args.audit_only:
                saved_status = folder / "run_status.json"
                status = json.loads(saved_status.read_text()) if saved_status.exists() else None
            else:
                status = run_case(args, configuration, ranks, folder)
            audited = audit_case(args, checker, folder)
            passed = audited["passed"] and (status is None or status["exit_code"] == 0)
            result["cases"].append({"configuration": configuration, "ranks": ranks,
                                    "directory": str(folder), "run": status,
                                    "certificate_files": audited["files"], "passed": passed})
            result["passed"] &= passed
            if not passed and not args.audit_only:
                break
        for configuration in args.configs:
            for first_rank, second_rank in itertools.combinations(args.ranks, 2):
                comparison = compare_ranks(args.work_dir / f"{configuration}_r{first_rank}",
                                           args.work_dir / f"{configuration}_r{second_rank}")
                if comparison["common_snapshots"]:
                    path = args.work_dir / f"{configuration}_r{first_rank}_vs_r{second_rank}.json"
                    write_json(path, comparison)
                    result["rank_comparisons"].append(str(path))
    except (OSError, ValueError, KeyError, TypeError, OverflowError, subprocess.SubprocessError) as error:
        result["passed"] = False
        result["error"] = str(error)
        print(str(error), file=sys.stderr)
    write_json(args.work_dir / "validation_summary.json", result)
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {args.work_dir / 'validation_summary.json'}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
