#ifndef ANGULAR_FRONT_H
#define ANGULAR_FRONT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int af_abi_version(void);

typedef struct af_solver af_solver;

typedef struct {
  double time, first_connection_time;
  double max_face_balance_absolute, max_vacuum_flux_absolute;
  double max_resource_rate_error, clamp_signed_mass, clamp_absolute_mass;
  double min_bias_ratio, max_bias_ratio;
  size_t events, steps, graph_nodes, graph_bytes;
} af_diagnostics;

/* Open-boundary, source-at-cell-centre, opaque-until-full model.
 * Arrays are copied. Cell indexing is ((x*ny)+y)*nz+z.
 * epsilon controls ES rate modulation. With epsilon>0 and uniform_density>0,
 * initial_source_photons supplies the history for the exact uniform-medium,
 * target-centred spherical top-hat score. No recombination is included.
 * Passing uniform_density=0 leaves ES disabled until af_set_es_scores().
 *
 * Input arrays must contain at least 3*ns source indices, ns source rates,
 * and nx*ny*nz entries each for capacity and initial_mass. Optional non-NULL
 * initial_source_photons must contain ns entries; a non-NULL error buffer
 * must contain error_size writable bytes. */
af_solver* af_create(int nx, int ny, int nz, double dx, unsigned angular_level,
                     int ns, const int* source_indices, const double* source_rates,
                     const double* capacity, const double* initial_mass, double epsilon,
                     const double* initial_source_photons, double uniform_density,
                     char* error, size_t error_size);

/* Every non-NULL solver argument must refer to a live object returned by
 * af_create(). No function accepts a pointer after af_destroy() has freed it. */
void af_destroy(af_solver* solver);

/* Explicitly replace an external nonnegative ES score, held until replaced.
 * This is a physical control update, not an output operation.
 * score must contain at least af_cell_count(solver) entries. */
int af_set_es_scores(af_solver* solver, const double* score);

/* Return 0 on reaching target, 1 if max_events was reached, -1 on error.
 * An output-only truncation retains the current frozen event rates. */
int af_advance(af_solver* solver, double target_time, size_t max_events);

size_t af_cell_count(const af_solver* solver);
size_t af_bin_count(const af_solver* solver);
int af_source_count(const af_solver* solver);
const char* af_error(const af_solver* solver);
void af_get_diagnostics(const af_solver* solver, af_diagnostics* out);

/* Copy into caller-owned writable arrays. Required lengths are ncell for
 * af_copy_mass() and nsource*ncell for the two source-resolved copies, where
 * ncell=af_cell_count(solver) and nsource=af_source_count(solver).
 * A NULL solver or output pointer makes these three functions no-ops;
 * non-NULL pointers must be live and arrays must have the required length. */
void af_copy_mass(const af_solver* solver, double* mass);
void af_copy_source_absorption(const af_solver* solver, double* mass);
void af_copy_source_rates(const af_solver* solver, double* rate);

/* Each non-NULL output must contain nsource*af_bin_count(solver) writable
 * entries. Individual NULL outputs are skipped; a NULL solver is a no-op. */
void af_copy_resource_ledgers(const af_solver* solver, double* absorbed,
                             double* escaped, double* emission_rate);

/* Recompute one source-bin absorption rate with current frozen ES weights.
 * rate must contain af_cell_count(solver) writable entries; escape must point
 * to one writable double. No physical state changes. */
int af_resource_rate(af_solver* solver, size_t resource, double* rate, double* escape);

#ifdef __cplusplus
}
#endif

#endif
