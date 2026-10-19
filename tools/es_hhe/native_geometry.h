#ifndef MERAXES_ES_NATIVE_GEOMETRY_H
#define MERAXES_ES_NATIVE_GEOMETRY_H

#include <stdint.h>

/* All arrays are contiguous and row-major. Status 0 is success; 1 denotes
 * invalid dimensions/indices, 2 a non-finite input or overflowing sum, and
 * 3 an exhausted summation expansion. Output must be discarded on error.
 * The caller owns array-length validation and must not alias input/output.
 * Thread count affects independent rows only, never summation order. */
int es_sparse_sums(int64_t side, const int64_t *xyz,
                   const int64_t *occupied, int64_t noccupied,
                   const double *column, const int64_t *rows, int64_t nrows,
                   double radius2, double *out, int threads);

/* Sum each translated periodic support in the supplied offset order.
 * xyz is [side^3,3]; offsets is [noffsets,3] with entries in [0,side).
 * No centre-by-support membership array is allocated. */
int es_periodic_offset_sums(int64_t side, const int64_t *xyz,
                           const double *column, const int64_t *rows,
                           int64_t nrows, const int64_t *offsets,
                           int64_t noffsets, double *out, int threads);

int es_indexed_sum(const double *values, int64_t value_rows, int64_t columns,
                   const int64_t *indices, int64_t nindices,
                   const int64_t *offsets, int64_t nrows,
                   double *out, int threads);

#endif
