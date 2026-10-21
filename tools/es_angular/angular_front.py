"""Owned NumPy interface to the angular-face front allocation library.

The model has open boundaries, fixed sources at cell centres and one conserved
ionization capacity per cell. It does not advance H/He chemistry, recombination
or a finite-speed radiation field. Build explicitly with ``build.py`` first.
"""
from pathlib import Path
import ctypes as ct
import os

import numpy as np


_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)
_LIB = None
_FOLDER = Path(__file__).resolve().parent


class Diagnostics(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in (
        "time", "first_connection_time", "max_face_balance_absolute",
        "max_vacuum_flux_absolute", "max_resource_rate_error",
        "clamp_signed_mass", "clamp_absolute_mass", "min_bias_ratio",
        "max_bias_ratio",
    )] + [(name, ct.c_size_t) for name in (
        "events", "steps", "graph_nodes", "graph_bytes",
    )]


def library():
    """Load an explicitly built library; never invoke a compiler implicitly."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = Path(os.environ.get("ES_ANGULAR_LIBRARY", _FOLDER / "libangular_front.so")).resolve()
    if not path.is_file():
        raise RuntimeError(
            "Angular library is absent. Run tools/es_angular/build.py, or set "
            "ES_ANGULAR_LIBRARY to an explicitly built library."
        )
    lib = ct.CDLL(str(path))
    lib.af_abi_version.argtypes = []
    lib.af_abi_version.restype = ct.c_int
    if lib.af_abi_version() != 1:
        raise RuntimeError("Unsupported angular library ABI")
    lib.af_create.restype = ct.c_void_p
    lib.af_create.argtypes = [ct.c_int] * 3 + [
        ct.c_double, ct.c_uint, ct.c_int, _IP, _DP, _DP, _DP,
        ct.c_double, _DP, ct.c_double, ct.c_char_p, ct.c_size_t,
    ]
    lib.af_destroy.argtypes = [ct.c_void_p]
    lib.af_destroy.restype = None
    lib.af_advance.argtypes = [ct.c_void_p, ct.c_double, ct.c_size_t]
    lib.af_advance.restype = ct.c_int
    lib.af_set_es_scores.argtypes = [ct.c_void_p, _DP]
    lib.af_set_es_scores.restype = ct.c_int
    lib.af_error.argtypes = [ct.c_void_p]
    lib.af_error.restype = ct.c_char_p
    lib.af_bin_count.argtypes = [ct.c_void_p]
    lib.af_bin_count.restype = ct.c_size_t
    lib.af_get_diagnostics.argtypes = [ct.c_void_p, ct.POINTER(Diagnostics)]
    lib.af_get_diagnostics.restype = None
    for name in ("af_copy_mass", "af_copy_source_absorption", "af_copy_source_rates"):
        getattr(lib, name).argtypes = [ct.c_void_p, _DP]
        getattr(lib, name).restype = None
    lib.af_copy_resource_ledgers.argtypes = [ct.c_void_p, _DP, _DP, _DP]
    lib.af_copy_resource_ledgers.restype = None
    lib.af_resource_rate.argtypes = [ct.c_void_p, ct.c_size_t, _DP, _DP]
    lib.af_resource_rate.restype = ct.c_int
    _LIB = lib
    return lib


def _pointer(array):
    return array.ctypes.data_as(_DP)


def _array(value, label):
    result = np.ascontiguousarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value, label, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return int(value)


class AngularFront:
    """An event solver with separate ledgers for every source and angular bin.

    Inputs are copied. ``source_mass`` counts only absorption after construction;
    the supplied initial mass is not attributed to any particular source.
    ``advance`` uses allocation time, with no finite-light-speed interpretation.
    Output truncations retain the current frozen ES weights and event rates.
    """

    def __init__(self, capacity, initial_mass, dx, source_indices, source_rates,
                 level=0, epsilon=0, initial_source_photons=None, uniform_density=0):
        self.handle = None
        cap = _array(capacity, "capacity")
        initial = _array(initial_mass, "initial_mass")
        q = _array(source_rates, "source_rates")
        if cap.ndim != 3 or initial.shape != cap.shape or min(cap.shape) <= 0:
            raise ValueError("Matched nonempty three-dimensional fields required")
        if max(cap.shape) > np.iinfo(np.int32).max:
            raise ValueError("Grid dimension exceeds the C interface")
        if q.ndim != 1 or not len(q) or len(q) > np.iinfo(np.int32).max:
            raise ValueError("A nonempty one-dimensional source rate array is required")
        raw_sources = np.asarray(source_indices)
        if raw_sources.shape != (len(q), 3) or raw_sources.dtype.kind not in "iu":
            raise ValueError("Integer source indices must have shape (sources, 3)")
        if np.any(raw_sources < 0) or np.any(raw_sources >= np.array(cap.shape)):
            raise ValueError("Source index outside grid")
        src = np.ascontiguousarray(raw_sources, dtype=np.int32)
        history = np.zeros_like(q) if initial_source_photons is None else _array(
            initial_source_photons, "initial_source_photons")
        if history.shape != q.shape:
            raise ValueError("One initial history budget per source required")
        if (np.any(cap < 0) or np.any(initial < 0) or np.any(initial > cap)
                or np.any(q < 0) or np.any(history < 0)):
            raise ValueError("Nonnegative counts and 0 <= initial_mass <= capacity required")
        dx, epsilon, uniform_density = float(dx), float(epsilon), float(uniform_density)
        if (not np.isfinite(dx) or dx <= 0 or not np.isfinite(epsilon) or epsilon < 0
                or not np.isfinite(uniform_density) or uniform_density < 0):
            raise ValueError("Invalid spacing, ES epsilon or uniform density")
        level = _integer(level, "level", 0, 8)
        self.lib = library()
        error = ct.create_string_buffer(512)
        self.handle = self.lib.af_create(
            *cap.shape, dx, level, len(q), src.ctypes.data_as(_IP), _pointer(q),
            _pointer(cap), _pointer(initial), epsilon, _pointer(history),
            uniform_density, error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode())
        self.shape = cap.shape
        self.ns = len(q)
        self.nbins = int(self.lib.af_bin_count(self.handle))

    def close(self):
        if self.handle:
            self.lib.af_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def _require_open(self):
        if not self.handle:
            raise RuntimeError("AngularFront is closed")

    def _check(self, result):
        if result < 0:
            raise RuntimeError(self.lib.af_error(self.handle).decode())
        return result

    def advance(self, target, max_events=1000000):
        self._require_open()
        max_events = _integer(max_events, "max_events", 1, ct.c_size_t(-1).value)
        result = self._check(self.lib.af_advance(self.handle, float(target), max_events))
        if result:
            raise RuntimeError("Event limit reached; state is retained at the last completed event")
        return self.snapshot()

    def set_es_scores(self, score):
        """Update an external ES field explicitly, independent of output requests."""
        self._require_open()
        score = _array(score, "ES score")
        if score.shape != self.shape or np.any(score < 0):
            raise ValueError("Nonnegative ES score with the grid shape required")
        self._check(self.lib.af_set_es_scores(self.handle, _pointer(score)))

    def diagnostics(self):
        self._require_open()
        result = Diagnostics()
        self.lib.af_get_diagnostics(self.handle, ct.byref(result))
        return {name: getattr(result, name) for name, _ in result._fields_}

    def snapshot(self):
        self._require_open()
        mass = np.empty(self.shape)
        source_mass = np.empty((self.ns,) + self.shape)
        rate = np.empty_like(source_mass)
        self.lib.af_copy_mass(self.handle, _pointer(mass))
        self.lib.af_copy_source_absorption(self.handle, _pointer(source_mass))
        self.lib.af_copy_source_rates(self.handle, _pointer(rate))
        absorbed = np.empty((self.ns, self.nbins))
        escaped, emission = np.empty_like(absorbed), np.empty_like(absorbed)
        self.lib.af_copy_resource_ledgers(
            self.handle, _pointer(absorbed), _pointer(escaped), _pointer(emission))
        return dict(mass=mass, source_mass=source_mass, source_rate=rate,
                    resource_absorbed=absorbed, resource_escaped=escaped,
                    resource_rate=emission, diagnostics=self.diagnostics())

    def resource_rate(self, source, bin_index):
        self._require_open()
        source = _integer(source, "source", 0, self.ns - 1)
        bin_index = _integer(bin_index, "bin_index", 0, self.nbins - 1)
        field, escaped = np.empty(self.shape), ct.c_double()
        self._check(self.lib.af_resource_rate(
            self.handle, source * self.nbins + bin_index, _pointer(field), ct.byref(escaped)))
        return field, escaped.value
