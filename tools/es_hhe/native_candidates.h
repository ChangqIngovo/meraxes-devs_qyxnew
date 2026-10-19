#ifndef MERAXES_ES_NATIVE_CANDIDATES_H
#define MERAXES_ES_NATIVE_CANDIDATES_H

#include <stdint.h>

/* Independent classification of aggregated ES candidates. aggregate is
 * contiguous [rows,5+groups], with five species followed by photon accounts;
 * cross is [groups,3]. Outputs are uint8[rows]. Status 0 means success,
 * 1 invalid arguments, and 2 nonfinite/negative input or arithmetic.
 * An ambiguous row is close enough to the count-pruning boundary that the
 * caller must repeat the original NumPy classification for the full array,
 * preserving its BLAS shape and arithmetic order. No arrays are modified
 * except the three output masks. Discard outputs on nonzero status. */
int es_candidate_flags(int64_t rows, int64_t groups, const double *aggregate,
    const double *cross, double epsilon, int count_pruning,
    uint8_t *valid, uint8_t *proven_partial, uint8_t *ambiguous, int threads);

#endif
