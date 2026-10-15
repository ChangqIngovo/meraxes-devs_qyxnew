#ifndef MERAXES_PHOTON_CONSERVATION_SHAPE_H
#define MERAXES_PHOTON_CONSERVATION_SHAPE_H

#include "photon_conservation.h"

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum
  {
    PC_SHAPE_OK = 0,
    PC_SHAPE_ERR_INVALID = -1,
    PC_SHAPE_ERR_MEMORY = -2,
    PC_SHAPE_ERR_MASTER_LIMIT = -3,
    PC_SHAPE_ERR_CUT_LIMIT = -4,
    PC_SHAPE_ERR_ORACLE = -5,
    PC_SHAPE_ERR_NUMERICAL = -6
  } pc_shape_status;

  typedef struct
  {
    size_t max_hall_cuts, max_master_sweeps, max_coordinate_iterations;
    size_t max_waterfill_iterations, max_evaluations, max_bytes;
    double absolute_tolerance, relative_tolerance, maximum_multiplier;
  } pc_shape_options;

  typedef struct
  {
    pc_shape_status status;
    pc_status last_oracle_status;
    size_t hall_cuts, master_sweeps, master_evaluations, oracle_calls, bytes_peak;
    double source_budget, target_total, unused_budget;
    double objective, evaluated_dual_lower, signed_primal_dual_gap;
    double equality_error, max_hall_violation, max_priority_violation;
    double max_priority_equality_error, max_bound_violation;
    double max_complementarity, max_oracle_target_error;
    double oracle_target_l1_error, max_final_target_correction, final_target_correction_l1;
    double tolerance;
    int full_graph_certificate;
  } pc_shape_metrics;

  void pc_shape_default_options(pc_shape_options* options);
  const char* pc_shape_status_string(pc_shape_status status);

  /* Minimize 1/2 sum (y-prior)^2 / capacity on the maximum-absorption,
   * minimum-ES-L1, first-crossing-priority face established by reference.
   * prior==NULL selects problem->baseline. Positive-capacity cells carry the
   * strictly convex objective; zero-capacity targets are fixed at zero.
   *
   * reference must be a successful PC_OK/PC_PARTIAL result for the SAME problem,
   * with all priority diagnostics supplied. Its baseline_retained_in_group values refer
   * to min(y, baseline), not to the optional shape prior.
   *
   * Fixed-marginal pc_allocate calls certify transport. For partial absorption
   * F<P, Hall cuts use rhs S(Q)-(P-F); source inventories are never rescaled.
   * On success result contains a final allocation on the original graph and U.
   * A final residual augmentation restores any rounded-away source inventory;
   * its change to the master target field is measured and its objective, total,
   * bounds and retained quotas are checked independently.
   * Its priority diagnostic buffer is copied from reference (original optimal
   * quotas), while its metrics describe the final fixed-marginal oracle.
   *
   * A limit/numerical/oracle failure is explicit; it is NOT permission to label
   * the staged reference map as the unique quadratic optimum. Only caller-owned
   * result arrays are used; reference and result structs must be distinct.
   * The mathematical optimum is unique. Metrics report floating-point
   * residuals, not an interval-arithmetic proof or an always-nonnegative gap.
   */
  pc_shape_status pc_shape_refine(const pc_problem* problem,
                                  const pc_options* flow_options,
                                  const pc_result* reference,
                                  const double* prior,
                                  const pc_shape_options* options,
                                  pc_result* result,
                                  pc_shape_metrics* metrics);

#ifdef __cplusplus
}
#endif
#endif
