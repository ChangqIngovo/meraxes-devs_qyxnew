#ifndef MERAXES_ES_NATIVE_JOINT_H
#define MERAXES_ES_NATIVE_JOINT_H

#include <stdint.h>

/* Shared midpoint update on a Boolean compact sphere/cell CSR operator.
 * cross is the normalized [groups,3] photoionization cross-section array.
 * Inputs and outputs are distinct contiguous row-major arrays. Input arrays
 * are never modified; all outputs must be discarded on a nonzero status.
 * K<2 returns 100 (unsupported, no outputs touched); one support retains the
 * reference solver's exact monochromatic endpoint.
 *
 * status: 0 success, 1 invalid arguments/CSR, 2 nonfinite arithmetic,
 * 3 allocation failure, 4 iteration limit, 5 midpoint backtracking failure,
 * 6 crossing-bracket feasibility failure, 7 subordinate kernel failure.
 * info_i[3] = stop code, evaluation count, accepted-step count.
 * stop: 0 no_absorption, 1 photon_exhaustion, 2 absorber_saturation,
 *       3 numerical_remainder, 4 crossing.
 * info_f[1] = total exposure. event_mask is int64[K,3].
 */
int es_joint_compact(int64_t cells, int64_t groups, int64_t spheres,
    int64_t memberships, const double *initial, const double *photons,
    const double *cross, const int64_t *indices, const int64_t *offsets,
    double epsilon, double donor_fraction, double count_tolerance,
    double photon_floor, int64_t max_iterations, int threads,
    double *new_state, double *remaining_photons, double *absorbed,
    double *debit, int64_t *event_mask, int64_t *info_i, double *info_f);

#endif
