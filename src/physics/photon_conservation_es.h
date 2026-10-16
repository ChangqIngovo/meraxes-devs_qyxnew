#ifndef MERAXES_PHOTON_CONSERVATION_ES_H
#define MERAXES_PHOTON_CONSERVATION_ES_H

#include "photon_conservation.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /* Explicit ES completion on a finite periodic grid. No Meraxes state or MPI.
   * The grid source/capacity/baseline and dimensions are used; its lower, upper,
   * radius and priority pointers are ignored and constructed by this driver.
   * radii_desc contains ACTUAL filter steps in decreasing grid-cell units. Its
   * final entry is zero for the unfiltered identity step, not the nominal cell
   * radius printed by the ES loop. first_crossing is the original descending
   * step index; -1 means no crossing. Binary64 doubles are required for the
   * exact comparison of cut capacities as sums of input counts. */
  typedef struct
  {
    pc_problem grid;
    const double* radii_desc;
    size_t radius_count;
    const int* first_crossing;
  } pc_es_problem;

/* Empty band is -1. An absent seed front can only be unlocked by the explicitly
 * declared full-domain band -2. Other nonnegative values are periodic
 * six-neighbour distances from the seed front, in cell steps. */
#define PC_ES_BAND_EMPTY (-1)
#define PC_ES_BAND_ALL (-2)

  typedef struct
  {
    size_t attempts, front_seed_cells, promotion;
    int band_width, empty_front, terminal_full_source_budget;
    double terminal_maximum_absorption, total_capacity;
  } pc_es_metrics;

  typedef void (
    *pc_es_attempt_visitor)(int band_width, size_t promotion, pc_status status, const pc_metrics* metrics, void* user);

  typedef struct
  {
    /* All arrays remain caller-owned. allocation.y/source_residual and the four
     * arrays below are required and have grid.ncell entries. At success they
     * describe the selected final graph/band and its exact staged-flow problem.
     * Writable count/priority arrays must be distinct and cannot alias inputs.
     * Positive-edge visitation and priority diagnostics are emitted only for
     * the final solve, never for search attempts. */
    pc_result allocation;
    double *lower, *upper, *radius;
    int* priority;
    pc_es_metrics metrics;
    pc_es_attempt_visitor visit_attempt;
    void* attempt_user;
  } pc_es_result;

  /* First finds the terminal full-geometry maximum Fmax (which need not be
   * min(sum S,sum C)). It then minimizes editable-band width and, at that band,
   * radius promotion, subject to attaining Fmax and satisfying all lower bounds.
   * Terminal/trial cut INPUT sums are compared exactly, so a lost faint source
   * cannot disappear in a rounded aggregate Fmax. Flow and marginal audits are
   * floating-point checks and remain subject to the allocator tolerances.
   * flow_tolerance must be zero and relative_tolerance must lie in [0,1).
   * PC_OK means all source photons are assigned; PC_PARTIAL is a valid terminal
   * geometry maximum with an explicit residual. A negative status is unresolved
   * and never an infeasibility or maximum-flow certificate. */
  pc_status pc_es_complete(const pc_es_problem* problem, const pc_options* options, pc_es_result* result);
  pc_status pc_es_complete_with_geometry(const pc_es_problem* problem,
                                         const pc_options* options,
                                         pc_es_result* result,
                                         const pc_geometry_executor* geometry);

#ifdef __cplusplus
}
#endif
#endif
