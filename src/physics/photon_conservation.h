#ifndef MERAXES_PHOTON_CONSERVATION_H
#define MERAXES_PHOTON_CONSERVATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* A serial reference allocator. Counts are extensive photon/atom counts;
   * radii and periodic distances use grid-cell units. The allowed-edge test is
   * minimum-image distance squared <= radius squared + 1e-10 (roundoff margin).
   * Cell order is
   * ((x * dim[1]) + y) * dim[2] + z. No MPI or Meraxes global state is used. */
  typedef enum
  {
    PC_OK = 0,
    PC_PARTIAL = 1,          /* Valid maximal allocation; photons remain. */
    PC_LOWER_INFEASIBLE = 2, /* Diagnostic maximum for lower capacities. */
    PC_ERR_INVALID = -1,
    PC_ERR_MEMORY_LIMIT = -2,
    PC_ERR_EDGE_LIMIT = -3,
    PC_ERR_NUMERICAL = -4
  } pc_status;

  typedef struct
  {
    size_t dim[3], ncell;
    const double *source, *capacity, *baseline, *lower, *upper, *radius;
    /* Optional nonnegative rank; smaller ranks have earlier T-stage priority.
     * NULL gives one group. L is solved first, T=clip(b,L,U) group by group,
     * and U is finally unlocked for every target together. */
    const int* priority;
  } pc_problem;

  typedef struct
  {
    size_t max_edges; /* Source-target links; excludes terminal arcs. */
    size_t max_bytes; /* All live internal allocations, not caller arrays. */
    size_t seed_neighbors, cut_neighbors;
    double absolute_tolerance, relative_tolerance;
    /* Residual arcs <= this count are ignored. Default zero avoids a global
     * epsilon silently discarding individually faint source inventories.
     * Positive values make flow/cut optimality tolerance-dependent; callers
     * requiring unmodified source inventories must retain the zero default. */
    double flow_tolerance;
  } pc_options;

  typedef struct
  {
    int priority;
    double prefix_flow;                /* Maximum after this T group is unlocked. */
    double retained_in_group;          /* Sum min(y_j, T_j), T=clip(b_j,L_j,U_j). */
    double baseline_retained_in_group; /* Sum min(y_j, b_j); includes fixed L>b offset. */
  } pc_priority_diagnostic;

  typedef struct
  {
    int lower_feasible, full_source_budget, full_graph_cut_separated;
    size_t sources, targets, source_target_edges, positive_edges;
    size_t separation_rounds, priority_groups, bytes_peak;
    double photons, lower_demand, absorbed, source_residual, mincut_capacity;
    double source_cut_budget, target_cut_capacity, hall_source_excess;
    double lower_cut_deficit;
    double lower_stage_flow, retention_stage_flow, final_stage_flow;
    double morphology_l1, retained_overlap;
    double max_source_overspend, max_target_excess, max_lower_deficit;
    double max_source_relative_residual, max_edge_balance_error;
    double max_geometry_excess, ledger_error, cut_flow_error;
  } pc_metrics;

  typedef void (*pc_edge_visitor)(size_t source_cell, size_t target_cell, double photons, void* user);

  typedef struct
  {
    /* Required caller-owned ncell arrays; initialized even on a guarded failure. */
    double *y, *source_residual;
    double* source_spent; /* Optional ncell consumed-source counts. */
    /* Optional ncell masks for the final complete residual cut. cut_target also
     * includes geometrically reachable U=0 cells, so it represents N(Q) in the
     * full fixed-radius graph for subsequent changed-capacity shape proposals. */
    unsigned char *cut_source, *cut_target;
    /* Optional caller-owned diagnostics buffer. The required count is always
     * reported in metrics.priority_groups; a short buffer is not an error. */
    pc_priority_diagnostic* priority_diagnostics;
    size_t priority_diagnostics_capacity;
    /* Optional explicit positive-edge certificate; invoked after verification. */
    pc_edge_visitor visit_edge;
    void* visitor_user;
    pc_metrics metrics;
    pc_status status;
  } pc_result;

  void pc_default_options(pc_options* options);
  const char* pc_status_string(pc_status status);
  pc_status pc_allocate(const pc_problem* problem, const pc_options* options, pc_result* result);

  /* A fixed-marginal oracle is pc_allocate with lower=upper=desired_y.
   * PC_LOWER_INFEASIBLE still returns a valid first-stage maximum and a complete
   * geometric Hall cut; it is not a maximum for untested larger upper bounds.
   * PC_PARTIAL is a valid maximum under U, with every source remainder explicit.
   * Resource-limit statuses never assert complete-graph optimality. */

#ifdef __cplusplus
}
#endif
#endif
