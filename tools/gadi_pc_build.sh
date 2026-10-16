#!/usr/bin/env bash
# Build the checked-out Photon_conservation commit for Gadi Cascade Lake nodes.
set -euo pipefail

repo=${PC_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
repo=$(git -C "$repo" rev-parse --show-toplevel)
build=${PC_BUILD_DIR:-$repo/build-pc-mpi}
[[ $build = /* ]] || build=$repo/$build
[[ $(git -C "$repo" branch --show-current) == Photon_conservation ]] || {
    echo 'Build from the Photon_conservation branch.' >&2; exit 2;
}
git -C "$repo" diff --quiet HEAD -- || {
    echo 'Tracked files differ from HEAD; commit or restore them before a provenance build.' >&2; exit 2;
}
if [[ -n $(git -C "$repo" submodule status --recursive | sed -n '/^[+-U]/p') ]]; then
    echo 'Initialize the recorded submodule revisions before building: git submodule update --init --recursive' >&2
    exit 2
fi

pc_environment=${PC_ENV_FILE:-$repo/pc_env.sh}
if [[ -r $pc_environment ]]; then
    # Existing user environment files need not be written for nounset shells.
    set +u
    source "$pc_environment"
    set -u
else
    module load gcc/12.2.0 openmpi/4.1.4 cmake/3.24.2 hdf5/1.12.2p \
        gsl/2.7.1 fftw3/3.3.10-nci1 python3/3.11.7
fi

cmake -S "$repo" -B "$build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO='-O3 -g -DNDEBUG -march=cascadelake' \
    -DHDF5_ROOT=/apps/hdf5/1.12.2p -DHDF5_PREFER_PARALLEL=ON \
    -DFFTW_INCLUDE_DIR=/apps/fftw3/3.3.10-nci1/include \
    -DFFTW_LIBRARY=/apps/fftw3/3.3.10-nci1/lib/libfftw3f.so \
    -DFFTW_MPI_LIBRARY=/apps/fftw3/3.3.10-nci1/lib/ompi3/libfftw3f_mpi.so \
    -DBUILD_TESTS=ON -DUSE_CUDA=OFF "$@"
# Two build workers are intentional on a login node; use PC_BUILD_JOBS in a job.
cmake --build "$build" --parallel "${PC_BUILD_JOBS:-2}"
ctest --test-dir "$build" --output-on-failure \
    -R '^test_photon_conservation(_es|_shape|_index)?$'
module -t list > "$build/pc_loaded_modules.txt" 2>&1 || true

python3 - "$repo" "$build" "$pc_environment" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

repo, build, environment = (pathlib.Path(p).resolve() for p in sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

binary = build / 'bin/meraxes'
data = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'repository': str(repo),
    'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'submodules': subprocess.check_output(['git', '-C', str(repo), 'submodule', 'status', '--recursive'], text=True),
    'binary': str(binary),
    'binary_sha256': digest(binary),
    'cmake_cache_sha256': digest(build / 'CMakeCache.txt'),
    'pc_env_file': str(environment) if environment.is_file() else None,
    'pc_env_sha256': digest(environment) if environment.is_file() else None,
    'local_cmake_sha256': digest(repo / 'local.cmake') if (repo / 'local.cmake').is_file() else None,
    'mpicc_version': subprocess.check_output(['mpicc', '--version'], text=True),
    'loaded_modules': (build / 'pc_loaded_modules.txt').read_text(),
    'linked_libraries': subprocess.check_output(['ldd', str(binary)], text=True),
}
path = build / 'pc_build_manifest.json'
path.write_text(json.dumps(data, indent=2) + '\n')
print(f'Build manifest: {path}')
PY
