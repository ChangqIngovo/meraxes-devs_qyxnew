#ifndef MERAXES_ES_NATIVE_CHEMISTRY_H
#define MERAXES_ES_NATIVE_CHEMISTRY_H

#include <stddef.h>

/* Exact fixed-rate H/He upward chemistry for independent cells.
 * n/new are row-major [rows,5], rate/counts are [rows,3]. Exposure contains
 * either one scalar or one value per row, selected by vector_exposure.
 * Returns 0 on success, 1 for invalid arguments, 2 for a nonfinite/negative
 * input or an unrepresentable rate*exposure, and 3 for an invalid output.
 * The caller must discard output arrays when the return value is nonzero.
 * Compile without fast-math and with floating-point contraction disabled.
 */
int es_upward_cells(size_t rows, const double *n, const double *rate,
                    const double *exposure, int vector_exposure,
                    double *new_state, double *counts, int threads);

#endif
