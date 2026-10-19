/* Shared-memory exact support sums for the ES reference algorithm.
 *
 * A non-overlapping expansion records rounding remainders, including
 * subnormal values. Collapse uses round-to-nearest, ties-to-even. Each output
 * follows the original source/index order; OpenMP distributes independent
 * outputs and performs no floating-point parallel reduction. Compile without
 * fast-math, reassociation, contraction or flush-to-zero.
 */
#include "native_geometry.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Finite binary64 numbers occupy 2098 binary places from 2^-1074 through
 * 2^1023. This bounds the number of non-overlapping nonzero partials. */
#define ES_SUM_CAPACITY (DBL_MAX_EXP - DBL_MIN_EXP + DBL_MANT_DIG + 1)

static int expansion_add(double value, double *partials, int *count)
{
    if (!isfinite(value)) return 2;
    int retained = 0;
    for (int j = 0; j < *count; ++j) {
        double other = partials[j];
        if (fabs(value) < fabs(other)) {
            double swap = value;
            value = other;
            other = swap;
        }
        double sum = value + other;
        if (!isfinite(sum)) return 2;
        double residual = other - (sum - value);
        if (residual != 0.0) partials[retained++] = residual;
        value = sum;
    }
    if (value != 0.0) {
        if (retained == ES_SUM_CAPACITY) return 3;
        partials[retained++] = value;
    }
    *count = retained;
    return 0;
}

static double expansion_total(const double *partials, int count)
{
    if (count == 0) return 0.0;
    double value = partials[--count];
    double residual = 0.0;
    while (count > 0) {
        double previous = value;
        double other = partials[--count];
        value = previous + other;
        residual = other - (value - previous);
        if (residual != 0.0) break;
    }
    /* Remaining partials may decide which side of a half-ulp tie contains
     * the exact result. Do not let their sign disappear when rounding. */
    if (count > 0 && ((residual < 0.0 && partials[count - 1] < 0.0) ||
                      (residual > 0.0 && partials[count - 1] > 0.0))) {
        double correction = 2.0 * residual;
        double rounded = value + correction;
        if (rounded - value == correction) value = rounded;
    }
    return value;
}

static int thread_count(int requested, int64_t outputs, long double work)
{
#ifdef _OPENMP
    /* Small compact supports are faster serially. No omp_set_num_threads:
     * the explicit request is private to this kernel's parallel region. */
    int useful = work >= 32768.0L * requested ? requested : (int)(work / 32768.0L);
    if (useful < 1) useful = 1;
    if ((int64_t)useful > outputs) useful = (int)outputs;
    return useful > 0 ? useful : 1;
#else
    (void)requested;
    (void)outputs;
    (void)work;
    return 1;
#endif
}

int es_sparse_sums(int64_t side, const int64_t *xyz,
                   const int64_t *occupied, int64_t noccupied,
                   const double *column, const int64_t *rows, int64_t nrows,
                   double radius2, double *out, int threads)
{
    if (side < 1 || side > 2097151 || noccupied < 0 || nrows < 0 ||
        threads < 1 || threads > 1024 || !isfinite(radius2) || radius2 < 0.0)
        return 1;
    int64_t size = side * side * side;
    if (size > INT64_MAX / 3 || !xyz || !column ||
        (noccupied && !occupied) || (nrows && (!rows || !out))) return 1;
    for (int64_t i = 0; i < noccupied; ++i) {
        int64_t index = occupied[i];
        if (index < 0 || index >= size) return 1;
        if (!isfinite(column[index])) return 2;
        for (int axis = 0; axis < 3; ++axis)
            if (xyz[3 * index + axis] < 0 || xyz[3 * index + axis] >= side) return 1;
    }
    for (int64_t i = 0; i < nrows; ++i) {
        if (rows[i] < 0 || rows[i] >= size) return 1;
        for (int axis = 0; axis < 3; ++axis)
            if (xyz[3 * rows[i] + axis] < 0 || xyz[3 * rows[i] + axis] >= side) return 1;
    }
    int count = thread_count(threads, nrows, (long double)nrows * noccupied);
    int status = 0;
#pragma omp parallel num_threads(count) if(count > 1) reduction(max:status)
    {
        double partials[ES_SUM_CAPACITY];
#pragma omp for schedule(static)
        for (int64_t row = 0; row < nrows; ++row) {
            int n = 0;
            const int64_t *center = xyz + 3 * rows[row];
            for (int64_t j = 0; j < noccupied; ++j) {
                const int64_t *source = xyz + 3 * occupied[j];
                int64_t distance2 = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    int64_t delta = center[axis] - source[axis];
                    if (delta < 0) delta = -delta;
                    if (side - delta < delta) delta = side - delta;
                    distance2 += delta * delta;
                }
                if ((double)distance2 <= radius2) {
                    int error = expansion_add(column[occupied[j]], partials, &n);
                    if (error) {
                        if (error > status) status = error;
                        break;
                    }
                }
            }
            out[row] = expansion_total(partials, n);
            if (!isfinite(out[row]) && status < 2) status = 2;
        }
    }
    return status;
}

int es_indexed_sum(const double *values, int64_t value_rows, int64_t columns,
                   const int64_t *indices, int64_t nindices,
                   const int64_t *offsets, int64_t nrows,
                   double *out, int threads)
{
    if (value_rows < 0 || columns < 1 || nindices < 0 || nrows < 0 ||
        nrows > INT64_MAX / columns || value_rows > INT64_MAX / columns ||
        threads < 1 || threads > 1024 || !offsets ||
        (value_rows && !values) || (nindices && !indices) || (nrows && !out)) return 1;
    if (offsets[0] != 0 || offsets[nrows] != nindices) return 1;
    for (int64_t row = 0; row < nrows; ++row)
        if (offsets[row] < 0 || offsets[row] > offsets[row + 1]) return 1;
    for (int64_t j = 0; j < nindices; ++j)
        if (indices[j] < 0 || indices[j] >= value_rows) return 1;
    int64_t outputs = nrows * columns;
    int count = thread_count(threads, outputs, (long double)nindices * columns);
    int status = 0;
#pragma omp parallel num_threads(count) if(count > 1) reduction(max:status)
    {
        double partials[ES_SUM_CAPACITY];
#pragma omp for schedule(static)
        for (int64_t output = 0; output < outputs; ++output) {
            int64_t row = output / columns, column = output % columns;
            int n = 0;
            for (int64_t j = offsets[row]; j < offsets[row + 1]; ++j) {
                int error = expansion_add(values[indices[j] * columns + column], partials, &n);
                if (error) {
                    if (error > status) status = error;
                    break;
                }
            }
            out[output] = expansion_total(partials, n);
            if (!isfinite(out[output]) && status < 2) status = 2;
        }
    }
    return status;
}
