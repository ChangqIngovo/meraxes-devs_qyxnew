#!/usr/bin/env bash
# Build and validate the recorded angular-face Meraxes revision on Gadi.
set -euo pipefail
angular_repo=${ANGULAR_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
angular_repo=$(git -C "$angular_repo" rev-parse --show-toplevel)
angular_build=${ANGULAR_BUILD_DIR:-$angular_repo/build-angular-mpi}
[[ $angular_build = /* ]] || angular_build=$angular_repo/$angular_build
angular_python=${ANGULAR_PYTHON:-/g/data/qp00/sz4640/venvs/tauefix/bin/python}
[[ $angular_python = /* && -x $angular_python ]] || {
    echo 'ANGULAR_PYTHON must be an absolute, executable Python path.' >&2; exit 2;
}
git -C "$angular_repo" diff --quiet HEAD -- || {
    echo 'Tracked files differ from HEAD; build in a clean detached worktree.' >&2; exit 2;
}
if [[ -n $(git -C "$angular_repo" submodule status --recursive | sed -n '/^[+-U]/p') ]]; then
    echo 'Initialize recorded submodule revisions: git submodule update --init --recursive' >&2
    exit 2
fi
module --force purge
module load gcc/12.2.0 openmpi/4.1.4 cmake/3.24.2 hdf5/1.12.2p \
    gsl/2.7.1 fftw3/3.3.10-nci1 python3/3.11.7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$angular_python" -c 'import numpy, h5py; print("NumPy", numpy.__version__, "h5py", h5py.__version__)'
cmake -S "$angular_repo" -B "$angular_build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO='-O3 -g -DNDEBUG -march=cascadelake' \
    -DHDF5_ROOT=/apps/hdf5/1.12.2p -DHDF5_PREFER_PARALLEL=ON \
    -DFFTW_INCLUDE_DIR=/apps/fftw3/3.3.10-nci1/include \
    -DFFTW_LIBRARY=/apps/fftw3/3.3.10-nci1/lib/libfftw3f.so \
    -DFFTW_MPI_LIBRARY=/apps/fftw3/3.3.10-nci1/lib/ompi3/libfftw3f_mpi.so \
    -DBUILD_TESTS=ON -DBUILD_ES_ANGULAR=ON -DUSE_CUDA=OFF \
    -DUSE_MINI_HALOS=OFF -DCALC_MAGS=OFF -DUSE_STOCHASTICITY=OFF \
    "-DPython3_EXECUTABLE=$angular_python" "$@"
# Keep compilation modest on login nodes; override inside a compute allocation.
cmake --build "$angular_build" --parallel "${ANGULAR_BUILD_JOBS:-2}"
ctest --test-dir "$angular_build" --output-on-failure --no-tests=error \
    -R '^(test_angular_|test_retired_modes$)'
module -t list > "$angular_build/angular_loaded_modules.txt" 2>&1 || true

"$angular_python" - "$angular_repo" "$angular_build" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
repo, build = (Path(p).resolve() for p in sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--'], check=True)
binary = build / 'bin/meraxes'
library = build / 'lib/libangular_front.so'
data = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'repository': str(repo),
    'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'submodules': subprocess.check_output(['git', '-C', str(repo), 'submodule', 'status', '--recursive'], text=True),
    'binary_sha256': digest(binary), 'angular_library_sha256': digest(library),
    'cmake_cache_sha256': digest(build / 'CMakeCache.txt'),
    'local_cmake_sha256': digest(repo / 'local.cmake') if (repo / 'local.cmake').is_file() else None,
    'mpicc_version': subprocess.check_output(['mpicc', '--version'], text=True),
    'loaded_modules': (build / 'angular_loaded_modules.txt').read_text(),
    'linked_libraries': subprocess.check_output(['ldd', str(binary)], text=True),
    'tests': 'ctest ^(test_angular_|test_retired_modes$), passed',
}
path = build / 'angular_build_manifest.json'
path.write_text(json.dumps(data, indent=2) + '\n')
print(f'Build manifest: {path}')
PY
