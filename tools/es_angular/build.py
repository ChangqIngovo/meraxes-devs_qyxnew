#!/usr/bin/env python3
"""Explicit standalone C99 build; requires a C compiler, not MPI or Meraxes.

This script is never invoked automatically by the production H/He worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main(argv=None):
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"),
                        help="Compiler command; defaults to CC, then cc")
    parser.add_argument("--output", type=Path, default=folder / "libangular_front.so")
    args = parser.parse_args(argv)
    compiler = shlex.split(args.cc)
    if not compiler:
        parser.error("--cc must contain a compiler command")
    files = [folder / name for name in
             ("angular_front.c", "angular_faces.c", "angular_front.h", "angular_faces.h")]
    for path in files:
        if not path.is_file():
            parser.error(f"Missing C99 source: {path}")
    output = args.output.expanduser().resolve()
    if output in files or output.suffix not in (".so", ".dylib"):
        parser.error("--output must be a .so or .dylib library, separate from source files")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    command = compiler + ["-O3", "-std=c99", "-fPIC",
                          "-dynamiclib" if sys.platform == "darwin" else "-shared",
                          "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra",
                          str(files[0]), str(files[1]), "-lm", "-o", temporary]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    manifest = {
        "library": str(output),
        "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "compiler": compiler,
        "sources_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "model": "pure hydrogen; open boundary; no recombination",
        "production_mode3": "retired",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Angular-front build failed: {exc}") from exc
