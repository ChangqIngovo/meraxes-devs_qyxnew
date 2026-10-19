"""Optional deterministic C/OpenMP kernels; the Gadi runner requires them.

Parallelism is over independent output rows, never over competing updates to
the photon ledger. Each row retains its input summation order. ctypes.CDLL
releases the GIL while the C/OpenMP work executes.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np


def _flag(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


class NativeBackend:
    def __init__(self, filename, threads):
        self.filename = Path(filename).resolve(strict=True)
        self.threads = int(threads)
        allocation = os.environ.get("ES_MPI_CPUSET")
        if allocation is not None:
            fields = allocation.split(",")
            if not fields or any(not value.isdecimal() for value in fields):
                raise ValueError("ES_MPI_CPUSET must contain the MPI ranks' allocated CPU indices")
            allowed = {int(value) for value in fields}
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("This platform cannot set the ES worker's allocated CPU affinity")
            # The native bridge supplies the union of its ranks' original
            # bindings. Expand only this child; MPI parents retain their cores.
            os.sched_setaffinity(0, allowed)
        self.affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        if self.threads < 1 or self.threads > self.affinity:
            raise ValueError(f"ES_WORKER_THREADS={threads} exceeds CPU affinity ({self.affinity}) or is invalid; "
                             "check MPI CPU allocation and ES_MPI_CPUSET")
        self.library = ctypes.CDLL(str(self.filename))
        lib = self.library
        for name in ("es_kernel_abi_version", "es_kernel_openmp_version"):
            function = getattr(lib, name)
            function.argtypes = []
            function.restype = ctypes.c_int
        if lib.es_kernel_abi_version() != 2 or lib.es_kernel_openmp_version() <= 0:
            raise RuntimeError("ES native kernel ABI/OpenMP mismatch; rebuild with BUILD_ES_NATIVE=ON")
        lib.es_kernel_team_size.argtypes = [ctypes.c_int]
        lib.es_kernel_team_size.restype = ctypes.c_int
        self.team_size = lib.es_kernel_team_size(self.threads)
        if self.team_size != self.threads:
            raise RuntimeError(f"OpenMP created {self.team_size} threads instead of {self.threads}; check thread limits")
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags=("C_CONTIGUOUS", "ALIGNED"))
        ip = np.ctypeslib.ndpointer(dtype=np.int64, flags=("C_CONTIGUOUS", "ALIGNED"))
        bp = np.ctypeslib.ndpointer(dtype=np.uint8, flags=("C_CONTIGUOUS", "ALIGNED"))
        i64 = ctypes.c_int64
        lib.es_sparse_sums.argtypes = [i64, ip, ip, i64, fp, ip, i64, ctypes.c_double, fp, ctypes.c_int]
        lib.es_sparse_sums.restype = ctypes.c_int
        lib.es_periodic_offset_sums.argtypes = [i64, ip, fp, ip, i64, ip, i64, fp, ctypes.c_int]
        lib.es_periodic_offset_sums.restype = ctypes.c_int
        lib.es_indexed_sum.argtypes = [fp, i64, i64, ip, i64, ip, i64, fp, ctypes.c_int]
        lib.es_indexed_sum.restype = ctypes.c_int
        lib.es_upward_cells.argtypes = [ctypes.c_size_t, fp, fp, fp, ctypes.c_int, fp, fp, ctypes.c_int]
        lib.es_upward_cells.restype = ctypes.c_int
        lib.es_joint_compact.argtypes = [i64, i64, i64, i64, fp, fp, fp, ip, ip,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            i64, ctypes.c_int, fp, fp, fp, fp, ip, ip, fp]
        lib.es_joint_compact.restype = ctypes.c_int
        lib.es_candidate_flags.argtypes = [i64, i64, fp, fp, ctypes.c_double,
                                           ctypes.c_int, bp, bp, bp, ctypes.c_int]
        lib.es_candidate_flags.restype = ctypes.c_int
        self.sha256 = hashlib.sha256(self.filename.read_bytes()).hexdigest()

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f"ES native {operation} failed with status {status}; state not committed")

    def sparse_sums(self, side, xyz, indices, column, rows, radius2):
        xyz = np.ascontiguousarray(xyz, dtype=np.int64)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        column = np.ascontiguousarray(column, dtype=np.float64)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if xyz.shape != (side**3, 3) or column.shape != (side**3,) or indices.ndim != 1 or rows.ndim != 1:
            raise ValueError("Invalid sparse sphere array shapes")
        out = np.empty(len(rows))
        self._check(self.library.es_sparse_sums(side, xyz, indices, len(indices), column,
                    rows, len(rows), radius2, out, self.threads), "sparse sums")
        return out

    def indexed_sum(self, values, indices, offsets):
        x = np.ascontiguousarray(values, dtype=np.float64)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        if x.ndim not in (1, 2) or indices.ndim != 1 or offsets.ndim != 1 or len(offsets) < 1:
            raise ValueError("Invalid indexed sum array shapes")
        columns = 1 if x.ndim == 1 else x.shape[1]
        out = np.empty((len(offsets)-1,) + x.shape[1:])
        self._check(self.library.es_indexed_sum(x, len(x), columns, indices, len(indices),
                    offsets, len(offsets)-1, out, self.threads), "indexed sums")
        return out

    def periodic_offset_sums(self, side, xyz, column, rows, offsets):
        xyz = np.ascontiguousarray(xyz, dtype=np.int64)
        column = np.ascontiguousarray(column, dtype=np.float64)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        if (xyz.shape != (side**3, 3) or column.shape != (side**3,)
                or rows.ndim != 1 or offsets.ndim != 2 or offsets.shape[1] != 3):
            raise ValueError("Invalid periodic support array shapes")
        out = np.empty(len(rows))
        self._check(self.library.es_periodic_offset_sums(side, xyz, column, rows,
            len(rows), offsets, len(offsets), out, self.threads), "periodic support sums")
        return out

    def upward_cells(self, species, rates, exposure):
        n = np.ascontiguousarray(species, dtype=np.float64)
        r = np.ascontiguousarray(rates, dtype=np.float64)
        t = np.asarray(exposure, dtype=np.float64)
        vector = int(t.ndim == 1)
        if (n.ndim != 2 or n.shape[1] != 5 or r.shape != (len(n), 3)
                or t.ndim > 1 or (vector and t.shape != (len(n),))):
            raise ValueError("Invalid upward chemistry array shapes")
        t = np.ascontiguousarray(t.reshape(-1))
        updated = np.empty_like(n)
        counts = np.empty_like(r)
        self._check(self.library.es_upward_cells(len(n), n, r, t, vector, updated, counts,
                    self.threads), "H/He chemistry")
        return updated, counts

    def metadata(self):
        return dict(backend="C/OpenMP", abi=2, library_sha256=self.sha256,
                    threads=self.threads, openmp_team_size=self.team_size,
                    affinity_cpus=self.affinity)

    def candidate_flags(self, aggregate, cross, epsilon, count_pruning):
        x = np.ascontiguousarray(aggregate, dtype=np.float64)
        cross = np.ascontiguousarray(cross, dtype=np.float64)
        if (x.ndim != 2 or cross.ndim != 2 or cross.shape[1] != 3
                or x.shape[1] != 5+len(cross)):
            raise ValueError("Invalid candidate field array shapes")
        valid = np.empty(len(x), dtype=np.uint8)
        partial = np.empty(len(x), dtype=np.uint8)
        ambiguous = np.empty(len(x), dtype=np.uint8)
        self._check(self.library.es_candidate_flags(len(x), len(cross), x, cross,
            epsilon, int(count_pruning), valid, partial, ambiguous, self.threads), "candidate flags")
        # Near a pruning boundary, retain the reference BLAS summation and
        # full-array shape. Do not allow a changed contraction order to choose
        # a different candidate. The caller repeats this rare scan in NumPy.
        if np.any(ambiguous):
            return None
        return valid.view(np.bool_), partial.view(np.bool_)

    def joint_compact(self, species, photons, cross, indices, offsets,
                      epsilon, donor_fraction, count_tolerance, photon_floor,
                      max_iterations):
        n = np.ascontiguousarray(species, dtype=np.float64)
        p = np.ascontiguousarray(photons, dtype=np.float64)
        cross = np.ascontiguousarray(cross, dtype=np.float64)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        if (n.ndim != 2 or n.shape[1] != 5 or p.ndim != 2 or len(p) != len(n)
                or cross.shape != (p.shape[1], 3) or indices.ndim != 1
                or offsets.ndim != 1 or len(offsets) < 1):
            raise ValueError("Invalid compact common-field array shapes")
        updated = np.empty_like(n)
        remaining = np.empty_like(p)
        absorbed = np.empty((len(n), 3))
        debit = np.empty_like(p)
        events = np.empty((len(offsets)-1, 3), dtype=np.int64)
        info_i = np.empty(3, dtype=np.int64)
        info_f = np.empty(1)
        status = self.library.es_joint_compact(len(n), p.shape[1], len(offsets)-1,
            len(indices), n, p, cross, indices, offsets, epsilon, donor_fraction,
            count_tolerance, photon_floor, max_iterations, self.threads,
            updated, remaining, absorbed, debit, events, info_i, info_f)
        if status == 100:
            return None
        self._check(status, "compact common-field update")
        stops = ("no_absorption", "photon_exhaustion", "absorber_saturation",
                 "numerical_remainder", "crossing")
        return (updated, remaining, absorbed, debit, stops[int(info_i[0])],
                float(info_f[0]), int(info_i[1]), int(info_i[2]),
                np.argwhere(events).tolist())


_UNINITIALIZED = object()
_backend = _UNINITIALIZED


def get_backend():
    global _backend
    if _backend is _UNINITIALIZED:
        required = _flag("ES_NATIVE_REQUIRED")
        disabled = _flag("ES_NATIVE_DISABLE")
        if required and disabled:
            raise ValueError("ES_NATIVE_REQUIRED and ES_NATIVE_DISABLE cannot both be 1")
        filename = Path(os.environ.get("ES_NATIVE_LIBRARY", Path(__file__).with_name("libes_hhe_kernels.so")))
        if disabled:
            _backend = None
        elif filename.is_file():
            raw_threads = os.environ.get("ES_WORKER_THREADS", "1")
            if not raw_threads.isdecimal():
                raise ValueError("ES_WORKER_THREADS must be a positive integer")
            _backend = NativeBackend(filename, int(raw_threads))
        elif required or "ES_NATIVE_LIBRARY" in os.environ:
            raise RuntimeError(f"Missing ES native kernels: {filename}; rebuild and prepare a new run")
        else:
            _backend = None
    return _backend


def execution_metadata():
    backend = get_backend()
    return backend.metadata() if backend is not None else dict(backend="NumPy", threads=1)


def numerical_identity():
    """Checkpoint identity excludes thread count: per-row arithmetic is fixed."""
    result = execution_metadata()
    identity = {key: result[key] for key in ("backend", "abi", "library_sha256") if key in result}
    identity["dispatch_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return identity


def _fft_workers(cells):
    backend = get_backend()
    if backend is None or cells < 1048576:
        return 1
    return min(backend.threads, max(1, cells // 262144))


def rfftn(values):
    """Thread large transforms; small L10 transforms avoid team overhead."""
    workers = _fft_workers(values.size)
    if get_backend() is not None and values.size >= 1048576:
        from scipy import fft
        return fft.rfftn(values, workers=workers)
    return np.fft.rfftn(values)


def irfftn(values, shape):
    cells = int(np.prod(shape))
    workers = _fft_workers(cells)
    axes = tuple(range(len(shape)))
    if get_backend() is not None and cells >= 1048576:
        from scipy import fft
        return fft.irfftn(values, s=shape, axes=axes, workers=workers)
    return np.fft.irfftn(values, s=shape, axes=axes)
