"""Hydrogen excursion-set prior on a nonuniform periodic grid.

For every cell, the prior is the largest ratio of cumulative escaped stellar
photon counts to hydrogen-nucleus counts inside a discrete periodic top hat.
Both fields contain counts per cell in the same normalization. The local-cell
ratio is included; the other default radii double from one cell to half the
shortest box side, including that final radius. Each physical cell occurs at
most once in a periodic kernel.

This is a hydrogen prior for bounded angular allocation, not a spectral H/He
ionization barrier. It neither consumes photons nor changes gas populations.
Recombinations and the actual species budgets are handled by the spatial and
chemical solver. Density updates may replace the denominator independently of
the cumulative escaped-photon history. The same weak prior can be used for
all spectral groups; epsilon=0 in the spatial solver disables its influence.

FFT sums are evaluated after scaling each input by its largest value. Values
within a conservative roundoff scale of zero are recomputed directly from
the original nonnegative field, including small *positive* FFT values. This
distinguishes an exact empty region from a real tiny source without deleting
positive input values. Long-double accumulation preserves such values when
normalization by a much larger remote cell underflows float64.
"""

import numpy as np


def _shape(value):
    result = tuple(value)
    if (len(result) != 3 or any(not isinstance(n, (int, np.integer))
                              or isinstance(n, (bool, np.bool_)) or n < 1
                              for n in result)):
        raise ValueError("Expected three positive integer grid dimensions")
    return tuple(int(n) for n in result)


def default_radii(shape):
    """Return radii in cell-width units, including local and half-box scales."""
    largest = min(_shape(shape)) / 2.
    radii = [0.]
    radius = 1.
    while radius < largest:
        radii.append(radius)
        radius *= 2.
    radii.append(largest)
    return tuple(radii)


def _kernel(shape, radius):
    distance2 = np.zeros(shape, dtype=float)
    for axis, size in enumerate(shape):
        coordinate = np.arange(size)
        distance = np.minimum(coordinate, size-coordinate).astype(float)
        view = [1, 1, 1]
        view[axis] = size
        distance2 += (distance*distance).reshape(view)
    return (distance2 <= radius*radius).astype(float)


def _direct_at(field, kernel, selected):
    """Accumulate only selected centres; no signed cancellation is possible."""
    centres = np.column_stack(np.nonzero(selected))
    result = np.zeros(len(centres), dtype=np.longdouble)
    shape = np.asarray(field.shape)
    support = np.column_stack(np.nonzero(field))
    if len(support) < np.count_nonzero(kernel):
        # Early snapshots may contain only a few sources. Summing their
        # contributions avoids scanning a large, mostly empty spherical
        # support for every centre that needs an exact zero check.
        for source in support:
            inside = kernel[tuple(((centres-source) % shape).T)] != 0
            result[inside] += np.longdouble(field[tuple(source)])
        return result
    # A kernel is symmetric, so either offset sign gives the same top hat.
    for offset in np.column_stack(np.nonzero(kernel)):
        indices = (centres + offset) % shape
        result += field[tuple(indices.T)].astype(np.longdouble)
    return result


def _convolve(field, kernel, kernel_fft):
    scale = float(field.max())
    diagnostics = {"direct_cells": 0, "roundoff_scale_normalized": 0.,
                   "max_correction_normalized": 0.}
    if scale == 0:
        return np.zeros(field.shape, dtype=np.longdouble), diagnostics
    normalized = field / scale
    estimate = np.fft.irfftn(np.fft.rfftn(normalized) * kernel_fft,
                            s=field.shape, axes=(0, 1, 2)).real
    kernel_count = float(kernel.sum())
    # The norm product bounds the scale of the convolution. The prefactor is
    # a conservative trigger for direct reevaluation, not a photon floor.
    tolerance = (128*np.finfo(float).eps*(1+np.log2(field.size))
                 * float(np.linalg.norm(normalized.ravel()))
                 * np.sqrt(kernel_count))
    if (not np.all(np.isfinite(estimate)) or np.any(estimate < -tolerance)
            or np.any(estimate > kernel_count+tolerance)):
        raise FloatingPointError("Periodic FFT convolution exceeds its roundoff bounds")
    result = estimate.astype(np.longdouble) * np.longdouble(scale)
    # Near-zero *positive* estimates also need reevaluation: otherwise FFT
    # noise could invent a photon budget in a genuinely empty support.
    uncertain = (estimate <= tolerance) | (estimate > kernel_count)
    diagnostics["roundoff_scale_normalized"] = float(tolerance)
    if np.any(uncertain):
        exact = _direct_at(field, kernel, uncertain)
        correction = abs(exact/np.longdouble(scale)
                         - estimate[uncertain].astype(np.longdouble))
        if np.any(correction > 2*tolerance):
            raise FloatingPointError("Direct top-hat sum disagrees with FFT beyond roundoff")
        result[uncertain] = exact
        diagnostics["direct_cells"] = int(np.count_nonzero(uncertain))
        diagnostics["max_correction_normalized"] = float(correction.max())
    if np.any(result < 0):
        raise FloatingPointError("Periodic top-hat produced a negative count")
    return result, diagnostics


def periodic_es_scores(photon_history, hydrogen_nuclei, radii=None, *,
                       return_diagnostics=False):
    """Return max_R N_gamma(<R)/N_H(<R), using periodic FFT top hats.

    Input arrays are finite, nonnegative 3-D count fields of equal shape.
    Explicit radii are nonnegative cell-width units and must not exceed half
    the shortest box side. At a zero denominator, 0/0 maps to zero and a
    positive photon count maps to the largest finite float64 value. A ratio
    beyond float64 range is saturated at that same value. Inputs are copied
    and never mutated. With ``return_diagnostics``, also return per-radius
    kernel counts and roundoff reevaluation statistics.
    """
    photons = np.array(photon_history, dtype=float, copy=True)
    hydrogen = np.array(hydrogen_nuclei, dtype=float, copy=True)
    if (photons.ndim != 3 or hydrogen.shape != photons.shape
            or any(n < 1 for n in photons.shape)
            or not np.all(np.isfinite(photons))
            or not np.all(np.isfinite(hydrogen))
            or np.any(photons < 0) or np.any(hydrogen < 0)):
        raise ValueError("Photon history and hydrogen must be equal-shape finite nonnegative 3-D fields")
    shape = photons.shape
    if radii is None:
        selected_radii = default_radii(shape)
    else:
        selected_radii = tuple(float(x) for x in radii)
        if (not selected_radii or not np.all(np.isfinite(selected_radii))
                or any(x < 0 or x > min(shape)/2 for x in selected_radii)):
            raise ValueError("Top-hat radii must lie between zero and half the shortest box side")
        selected_radii = tuple(sorted(set(selected_radii)))
    score = np.zeros(shape, dtype=float)
    maximum = np.finfo(float).max
    records = []
    for radius in selected_radii:
        kernel = _kernel(shape, radius)
        count = int(kernel.sum())
        if count == 1:
            numerator = photons.astype(np.longdouble)
            denominator = hydrogen.astype(np.longdouble)
            stats = [{"direct_cells": int(photons.size),
                      "roundoff_scale_normalized": 0.,
                      "max_correction_normalized": 0.} for _ in range(2)]
        else:
            transform = np.fft.rfftn(kernel)
            numerator, first = _convolve(photons, kernel, transform)
            denominator, second = _convolve(hydrogen, kernel, transform)
            stats = [first, second]
        ratio = np.zeros(shape, dtype=np.longdouble)
        np.divide(numerator, denominator, out=ratio, where=denominator > 0)
        ratio[(denominator == 0) & (numerator > 0)] = maximum
        level = np.minimum(ratio, np.longdouble(maximum)).astype(float)
        np.maximum(score, level, out=score)
        records.append({"radius_cells": radius, "kernel_cells": count,
                        "photons": stats[0], "hydrogen": stats[1]})
    if return_diagnostics:
        return score, {"scope": "periodic hydrogen ES prior; no photon debit",
                       "levels": records}
    return score
