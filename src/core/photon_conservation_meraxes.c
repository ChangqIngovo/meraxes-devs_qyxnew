#include "photon_conservation_meraxes.h"

#include "meraxes.h"
#include "misc_tools.h"
#include "photon_conservation_feedback.h"
#include "physics/photon_conservation_es.h"
#include "physics/photon_conservation_shape.h"
#include "reionization.h"

#include <float.h>
#include <hdf5_hl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PC_MAX_RADII 4096
enum
{
  SOURCE,
  SOURCE_II,
  SOURCE_BH,
  SOURCE_III,
  CAPACITY,
  DENSITY,
  NEUTRAL,
  RECOMBINATIONS,
  BASELINE,
  ES_XH,
  NINPUT
};

struct photon_conservation_context
{
  int snapshot, nlocal, nradii;
  size_t ncell, workspace_bytes;
  double started, particle_normalization;
  double* input[NINPUT];
  int* first;
  float *old_zre, *old_j21;
  double radii[PC_MAX_RADII];
};

static void pc_fail(const char* message)
{
  mlog_error("Photon conservation: %s", message);
  ABORT(EXIT_FAILURE);
}

static void* pc_calloc(size_t n, size_t size)
{
  if (n && size > SIZE_MAX / n)
    pc_fail("allocation size overflow");
  void* p = calloc(n ? n : 1, size);
  if (!p)
    pc_fail("allocation failed");
  return p;
}

photon_conservation_context* photon_conservation_begin(int snapshot)
{
  const int n = run_globals.params.ReionGridDim;
  if (n < 1 || (size_t)n > (size_t)INT_MAX / n / n)
    pc_fail("grid exceeds the MPI_Gatherv int-count limit");
  photon_conservation_context* c = pc_calloc(1, sizeof(*c));
  c->snapshot = snapshot;
  c->started = MPI_Wtime();
  c->ncell = (size_t)n * n * n;
  c->nlocal = (int)run_globals.reion_grids.slab_nix[run_globals.mpi_rank] * n * n;
  const size_t limit = (size_t)run_globals.params.PhotonConservationMemoryMB * 1024 * 1024;
  /* Reserve caller arrays, MPI buffers and completion-driver temporaries before
   * granting separate bounded workspaces to the flow and shape solvers. */
  if (c->ncell > (SIZE_MAX - sizeof(*c)) / 512 || c->ncell * 512 + sizeof(*c) >= limit)
    pc_fail("grid exceeds PhotonConservationMemoryMB; this implementation solves on rank zero");
  c->workspace_bytes = (limit - c->ncell * 512 - sizeof(*c)) / 2;
  for (int f = 0; f < NINPUT; f++)
    c->input[f] = pc_calloc((size_t)c->nlocal, sizeof(double));
  c->first = pc_calloc((size_t)c->nlocal, sizeof(int));
  c->old_zre = pc_calloc((size_t)c->nlocal, sizeof(float));
  c->old_j21 = pc_calloc((size_t)c->nlocal, sizeof(float));
  const double cell_mass = run_globals.params.OmegaM * run_globals.RhoCrit * pow(run_globals.params.BoxSize / n, 3);
  const double factor = run_globals.params.physics.ReionEfficiency / cell_mass;
  c->particle_normalization = run_globals.params.BaryonFrac * cell_mass * run_globals.units.UnitMass_in_g /
                              run_globals.params.Hubble_h * (1.0 - 0.75 * run_globals.params.physics.Y_He) / PROTONMASS;
  if (!isfinite(factor) || factor < 0 || !isfinite(c->particle_normalization) || c->particle_normalization <= 0)
    pc_fail("invalid source normalization");
  reion_grids_t* g = &run_globals.reion_grids;
  for (int ix = 0; ix < c->nlocal / (n * n); ix++)
    for (int iy = 0; iy < n; iy++)
      for (int iz = 0; iz < n; iz++) {
        const int j = grid_index(ix, iy, iz, n, INDEX_REAL);
        const int p = grid_index(ix, iy, iz, n, INDEX_PADDED);
        const double d = 1.0 + (double)g->deltax[p];
        const double xe = run_globals.params.Flag_IncludeSpinTemp ? g->x_e_box[p] : 0.0;
        const double rec = run_globals.params.Flag_IncludeRecombinations ? g->N_rec[p] : 0.0;
        const double s2 = factor * g->stars[p];
        const double sbh = run_globals.params.physics.Flag_BHFeedback ? factor * g->effective_bhm[p] : 0.0;
        double s3 = 0;
#if USE_MINI_HALOS
        s3 = run_globals.params.physics.ReionEfficiencyIII / cell_mass * g->starsIII[p];
#endif
        if (!isfinite(d) || d < 0 || !isfinite(xe) || xe < 0 || xe > (double)0.999f || !isfinite(rec) || rec < 0 ||
            !isfinite(s2) || s2 < 0 || !isfinite(sbh) || sbh < 0 || !isfinite(s3) || s3 < 0)
          pc_fail("invalid real-space source/density/recombination input, or xe outside [0,0.999f]");
        c->input[SOURCE][j] = s2 + sbh + s3;
        c->input[SOURCE_II][j] = s2;
        c->input[SOURCE_BH][j] = sbh;
        c->input[SOURCE_III][j] = s3;
        c->input[DENSITY][j] = d;
        c->input[NEUTRAL][j] = 1.0 - xe;
        c->input[RECOMBINATIONS][j] = rec;
        c->input[CAPACITY][j] = (1.0 - xe) * (d + rec);
        c->first[j] = -1;
        c->old_zre[j] = g->z_at_ionization[j];
        c->old_j21[j] = run_globals.params.ReionUVBFlag ? g->J_21_at_ionization[j] : 0.0f;
      }
  return c;
}

void photon_conservation_radius(photon_conservation_context* c, double radius, int identity)
{
  if (c->nradii == PC_MAX_RADII || !isfinite(radius) || radius <= 0)
    pc_fail("invalid or excessively long ES radius ladder");
  c->radii[c->nradii++] = identity ? 0.0 : radius;
}

void photon_conservation_crossing(photon_conservation_context* c, int cell)
{
  if (c->first[cell] < 0)
    c->first[cell] = c->nradii - 1;
}

static void h5_attribute(hid_t file, const char* name, double value)
{
  if (H5LTset_attribute_double(file, "/", name, &value, 1) < 0)
    pc_fail("cannot write audit attribute");
}

static void h5_double(hid_t file, const char* name, size_t n, const double* values)
{
  const hsize_t dims[1] = { (hsize_t)n };
  if (H5LTmake_dataset_double(file, name, 1, dims, values) < 0)
    pc_fail("cannot write audit double dataset");
}

static void h5_int(hid_t file, const char* name, size_t n, const int* values)
{
  const hsize_t dims[1] = { (hsize_t)n };
  if (H5LTmake_dataset_int(file, name, 1, dims, values) < 0)
    pc_fail("cannot write audit integer dataset");
}

static void edge_csv(size_t source, size_t target, double photons, void* user)
{
  if (fprintf((FILE*)user, "%zu,%zu,%.17g\n", source, target, photons) < 0)
    pc_fail("cannot write positive-edge certificate");
}

static void attempt_csv(int band, size_t promotion, pc_status status, const pc_metrics* m, void* user)
{
  if (fprintf((FILE*)user,
              "%d,%zu,%d,%.17g,%.17g,%.17g\n",
              band,
              promotion,
              status,
              m->absorbed,
              m->lower_cut_deficit,
              m->source_residual) < 0)
    pc_fail("cannot write completion-attempt diagnostic");
}

static FILE* diagnostic_file(const char* name, const char* header)
{
  FILE* f = fopen(name, "w");
  if (!f || fputs(header, f) == EOF)
    pc_fail("cannot create audit file in OutputDir");
  return f;
}

/* Keep replayable inputs even if a limit or numerical failure aborts the solve.
 * The final certificate has a separate name and is published by atomic rename. */
static void checkpoint(photon_conservation_context* c,
                       double* const input[NINPUT],
                       const int* first,
                       const double* radii)
{
  if (!run_globals.params.PhotonConservationDebug)
    return;
  char path[STRLEN + 128];
  snprintf(path, sizeof(path), "%sphoton_inputs_%03d.hdf5", run_globals.params.OutputDir, c->snapshot);
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0)
    pc_fail("cannot create input checkpoint");
  h5_attribute(file, "schema_version", 1);
  h5_attribute(file, "snapshot", c->snapshot);
  h5_attribute(file, "grid_dim", run_globals.params.ReionGridDim);
  h5_attribute(file, "certificate_complete", 0);
  h5_attribute(file, "shape_enabled", run_globals.params.PhotonConservationShape);
  h5_attribute(file, "tolerance", run_globals.params.PhotonConservationTolerance);
  h5_attribute(file, "absolute_tolerance", 1e-12);
  h5_attribute(file, "max_edges", run_globals.params.PhotonConservationMaxEdges);
  h5_attribute(file, "memory_mb", run_globals.params.PhotonConservationMemoryMB);
  h5_attribute(file, "max_iterations", run_globals.params.PhotonConservationMaxIterations);
  h5_attribute(file, "max_evaluations", run_globals.params.PhotonConservationMaxEvaluations);
  h5_attribute(file, "max_cuts", run_globals.params.PhotonConservationMaxCuts);
  if (run_globals.params.PhotonConservationDebug == 2) {
    const char* names[NINPUT] = { "source",  "source_II",   "source_BH",          "source_III", "capacity",
                                  "density", "neutral_pre", "recombinations_pre", "baseline",   "xH_ES" };
    for (int f = 0; f < NINPUT; f++)
      h5_double(file, names[f], c->ncell, input[f]);
    h5_double(file, "radii_cells", (size_t)c->nradii, radii);
    h5_int(file, "first_crossing", c->ncell, first);
  }
  if (H5Fclose(file) < 0)
    pc_fail("cannot close input checkpoint");
}

void photon_conservation_finish(photon_conservation_context* c)
{
  const int rank = run_globals.mpi_rank, nranks = run_globals.mpi_size;
  const int dim = run_globals.params.ReionGridDim;
  const size_t n = c->ncell;
  const int debug = run_globals.params.PhotonConservationDebug;
  int *counts = pc_calloc((size_t)nranks, sizeof(int)), *offsets = pc_calloc((size_t)nranks, sizeof(int));
  int* local_slot = pc_calloc((size_t)c->nlocal, sizeof(int));
  for (int r = 0; r < nranks; r++) {
    counts[r] = (int)run_globals.reion_grids.slab_nix[r] * dim * dim;
    offsets[r] = (int)run_globals.reion_grids.slab_ix_start[r] * dim * dim;
  }
  for (int j = 0; j < c->nlocal; j++) {
    const double xh = run_globals.reion_grids.xH[j];
    const double q = c->input[NEUTRAL][j], cap = c->input[CAPACITY][j];
    if (!isfinite(xh) || xh < 0 || xh > 1)
      pc_fail("ES returned an invalid neutral fraction");
    c->input[ES_XH][j] = xh;
    c->input[BASELINE][j] = q > 0 ? fmax(0, fmin(cap, cap * (1.0 - xh / q))) : 0;
  }
  double* input[NINPUT] = { 0 };
  int* first = rank == 0 ? pc_calloc(n, sizeof(int)) : NULL;
  for (int f = 0; f < NINPUT; f++) {
    if (rank == 0)
      input[f] = pc_calloc(n, sizeof(double));
    MPI_Gatherv(c->input[f], c->nlocal, MPI_DOUBLE, input[f], counts, offsets, MPI_DOUBLE, 0, run_globals.mpi_comm);
  }
  MPI_Gatherv(c->first, c->nlocal, MPI_INT, first, counts, offsets, MPI_INT, 0, run_globals.mpi_comm);

  float* corrected = NULL;
  int* feedback_slot = NULL;
  if (rank == 0) {
    double* radii = pc_calloc((size_t)c->nradii, sizeof(double));
    for (int k = 0; k < c->nradii; k++)
      radii[k] = c->radii[k] * dim / run_globals.params.BoxSize;
    checkpoint(c, input, first, radii);
    pc_options options;
    pc_default_options(&options);
    options.max_edges = (size_t)run_globals.params.PhotonConservationMaxEdges;
    options.max_bytes = c->workspace_bytes;
    options.absolute_tolerance = 1e-12;
    options.relative_tolerance = run_globals.params.PhotonConservationTolerance;
    pc_es_problem problem = { 0 };
    problem.grid.dim[0] = problem.grid.dim[1] = problem.grid.dim[2] = (size_t)dim;
    problem.grid.ncell = n;
    problem.grid.source = input[SOURCE];
    problem.grid.capacity = input[CAPACITY];
    problem.grid.baseline = input[BASELINE];
    problem.first_crossing = first;
    problem.radii_desc = radii;
    problem.radius_count = (size_t)c->nradii;
    pc_es_result es = { 0 };
    es.lower = pc_calloc(n, sizeof(double));
    es.upper = pc_calloc(n, sizeof(double));
    es.radius = pc_calloc(n, sizeof(double));
    es.priority = pc_calloc(n, sizeof(int));
    es.allocation.y = pc_calloc(n, sizeof(double));
    es.allocation.source_residual = pc_calloc(n, sizeof(double));
    es.allocation.cut_source = pc_calloc(n, 1);
    es.allocation.cut_target = pc_calloc(n, 1);
    es.allocation.priority_diagnostics_capacity = (size_t)c->nradii + 1;
    es.allocation.priority_diagnostics = pc_calloc((size_t)c->nradii + 1, sizeof(pc_priority_diagnostic));
    char path[STRLEN + 128];
    FILE* attempts = NULL;
    if (debug == 2) {
      snprintf(
        path, sizeof(path), "%sphoton_conservation_%03d.attempts.csv", run_globals.params.OutputDir, c->snapshot);
      attempts = diagnostic_file(path, "band,promotion,status,absorbed,lower_deficit,residual\n");
      es.visit_attempt = attempt_csv;
      es.attempt_user = attempts;
    }
    const double completion_started = MPI_Wtime();
    pc_status status = pc_es_complete(&problem, &options, &es);
    const double completion_seconds = MPI_Wtime() - completion_started;
    if (attempts && fclose(attempts) != 0)
      pc_fail("cannot close completion diagnostic");
    if (status != PC_OK && status != PC_PARTIAL)
      pc_fail(pc_status_string(status));
    const pc_metrics staged = es.allocation.metrics;
    pc_problem selected = problem.grid;
    selected.lower = es.lower;
    selected.upper = es.upper;
    selected.radius = es.radius;
    selected.priority = es.priority;
    pc_result final = { 0 };
    final.y = pc_calloc(n, sizeof(double));
    final.source_residual = pc_calloc(n, sizeof(double));
    final.source_spent = pc_calloc(n, sizeof(double));
    final.cut_source = pc_calloc(n, 1);
    final.cut_target = pc_calloc(n, 1);
    final.priority_diagnostics_capacity = (size_t)c->nradii + 1;
    final.priority_diagnostics = pc_calloc((size_t)c->nradii + 1, sizeof(pc_priority_diagnostic));
    FILE* edges = NULL;
    char edge_path[STRLEN + 128], edge_temporary[STRLEN + 128];
    if (debug == 2) {
      snprintf(edge_path,
               sizeof(edge_path),
               "%sphoton_conservation_%03d.edges.csv",
               run_globals.params.OutputDir,
               c->snapshot);
      snprintf(edge_temporary,
               sizeof(edge_temporary),
               "%sphoton_conservation_%03d.edges.csv.tmp",
               run_globals.params.OutputDir,
               c->snapshot);
      edges = diagnostic_file(edge_temporary, "source,target,photons\n");
      final.visit_edge = edge_csv;
      final.visitor_user = edges;
    }
    pc_shape_metrics shape = { 0 };
    const double refinement_started = MPI_Wtime();
    if (run_globals.params.PhotonConservationShape) {
      pc_shape_options shape_options;
      pc_shape_default_options(&shape_options);
      shape_options.max_bytes = c->workspace_bytes;
      shape_options.max_master_sweeps = (size_t)run_globals.params.PhotonConservationMaxIterations;
      shape_options.max_evaluations = (size_t)run_globals.params.PhotonConservationMaxEvaluations;
      shape_options.max_hall_cuts = (size_t)run_globals.params.PhotonConservationMaxCuts;
      shape_options.absolute_tolerance = 1e-12;
      shape_options.relative_tolerance = run_globals.params.PhotonConservationTolerance;
      pc_shape_status shape_status =
        pc_shape_refine(&selected, &options, &es.allocation, NULL, &shape_options, &final, &shape);
      if (shape_status != PC_SHAPE_OK) {
        fprintf(stderr,
                "PC shape failure: %s, sweeps=%zu cuts=%zu evaluations=%zu gap=%.17g equality=%.17g "
                "hall=%.17g priority=%.17g priority_equality=%.17g bounds=%.17g tolerance=%.17g "
                "oracle=%s target_error=%.17g correction_l1=%.17g\n",
                pc_shape_status_string(shape_status),
                shape.master_sweeps,
                shape.hall_cuts,
                shape.master_evaluations,
                shape.signed_primal_dual_gap,
                shape.equality_error,
                shape.max_hall_violation,
                shape.max_priority_violation,
                shape.max_priority_equality_error,
                shape.max_bound_violation,
                shape.tolerance,
                pc_status_string(shape.last_oracle_status),
                shape.max_oracle_target_error,
                shape.final_target_correction_l1);
        fflush(stderr);
        pc_fail("unique-map solve did not converge; see shape diagnostics above");
      }
    } else {
      pc_problem fixed = selected;
      fixed.lower = fixed.upper = es.allocation.y;
      status = pc_allocate(&fixed, &options, &final);
      if (status != PC_OK && status != PC_PARTIAL)
        pc_fail("final fixed-marginal certificate failed");
    }
    const double refinement_seconds = MPI_Wtime() - refinement_started;
    if (edges) {
      if (fclose(edges) != 0)
        pc_fail("cannot close edge certificate");
      if (rename(edge_temporary, edge_path) != 0)
        pc_fail("cannot publish closed edge certificate by atomic rename");
    }
    corrected = pc_calloc(n, sizeof(float));
    feedback_slot = pc_calloc(n, sizeof(int));
    double *xh_double = pc_calloc(n, sizeof(double)), *xh_float = pc_calloc(n, sizeof(double));
    double* quota = pc_calloc((size_t)c->nradii + 1, sizeof(double));
    for (size_t k = 0; k < staged.priority_groups; k++) {
      const pc_priority_diagnostic d = es.allocation.priority_diagnostics[k];
      if (d.priority < 0 || d.priority > c->nradii)
        pc_fail("invalid priority group");
      quota[d.priority] = d.retained_in_group;
    }
    long double grid_absorbed = 0, roundoff_bound = 0, projection = 0, final_l1 = 0;
    for (size_t j = 0; j < n; j++) {
      const double cap = input[CAPACITY][j], q = input[NEUTRAL][j];
      const double raw_xh = cap > 0 ? q * (1.0 - final.y[j] / cap) : q;
      xh_double[j] = fmax(0, fmin(q, raw_xh));
      corrected[j] = (float)xh_double[j];
      xh_float[j] = corrected[j];
      if (q > 0) {
        const long double written = (long double)cap * (1.0L - (long double)corrected[j] / q);
        grid_absorbed += written;
        const double spacing = fmax(fabs((double)nextafterf(corrected[j], INFINITY) - corrected[j]),
                                    fabs((double)nextafterf(corrected[j], -INFINITY) - corrected[j]));
        roundoff_bound += (long double)cap / q * (0.5 * spacing + 8 * DBL_EPSILON * q);
        projection += fabs(input[BASELINE][j] - cap * (1.0 - input[ES_XH][j] / q));
      }
      final_l1 += fabs(final.y[j] - input[BASELINE][j]);
      feedback_slot[j] = c->nradii - 1;
      for (int k = 0; k < c->nradii; k++)
        if (es.radius[j] == radii[k]) {
          feedback_slot[j] = k;
          break;
        }
      // Original crossed targets retain the ES scale for the UVB closure.
      if (first[j] >= 0)
        feedback_slot[j] = first[j];
    }
    const double grid_error = (double)(grid_absorbed - final.metrics.absorbed);
    if (fabs(grid_error) >
        (double)roundoff_bound + options.absolute_tolerance + 64 * DBL_EPSILON * final.metrics.photons)
      pc_fail("float xH writeback exceeds its calculated rounding bound");
    mlog("PC snapshot=%d status=%s S=%.17g absorbed=%.17g residual=%.17g ledger=%.3g "
         "float_error=%.3g band=%d promotion=%zu L1=%.9g time=%.3fs completion=%.3fs refinement=%.3fs",
         MLOG_MESG,
         c->snapshot,
         pc_status_string(final.status),
         final.metrics.photons,
         final.metrics.absorbed,
         final.metrics.source_residual,
         final.metrics.ledger_error,
         grid_error,
         es.metrics.band_width,
         es.metrics.promotion,
         (double)final_l1,
         MPI_Wtime() - c->started,
         completion_seconds,
         refinement_seconds);
    if (debug) {
      char final_path[STRLEN + 128];
      snprintf(
        final_path, sizeof(final_path), "%sphoton_conservation_%03d.hdf5", run_globals.params.OutputDir, c->snapshot);
      snprintf(path, sizeof(path), "%sphoton_conservation_%03d.hdf5.tmp", run_globals.params.OutputDir, c->snapshot);
      hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      if (file < 0)
        pc_fail("cannot create HDF5 photon ledger");
#define ATTR(name, value) h5_attribute(file, name, (double)(value))
      ATTR("schema_version", 1);
      ATTR("certificate_complete", 0);
      ATTR("snapshot", c->snapshot);
      ATTR("grid_dim", dim);
      ATTR("mpi_ranks", nranks);
      ATTR("solver_status", final.status);
      ATTR("shape_enabled", run_globals.params.PhotonConservationShape);
      ATTR("source_total", final.metrics.photons);
      ATTR("capacity_total", es.metrics.total_capacity);
      ATTR("absorbed_total", final.metrics.absorbed);
      ATTR("residual_total", final.metrics.source_residual);
      ATTR("ledger_error", final.metrics.ledger_error);
      ATTR("full_source_budget", final.metrics.full_source_budget);
      ATTR("max_source_relative_residual", final.metrics.max_source_relative_residual);
      ATTR("lower_stage_flow", staged.lower_stage_flow);
      ATTR("retention_stage_flow", staged.retention_stage_flow);
      ATTR("staged_absorbed_total", staged.absorbed);
      ATTR("staged_mincut_capacity", staged.mincut_capacity);
      ATTR("staged_full_graph_certificate", staged.full_graph_cut_separated);
      ATTR("terminal_maximum_absorption", es.metrics.terminal_maximum_absorption);
      ATTR("terminal_full_source_budget", es.metrics.terminal_full_source_budget);
      ATTR("min_l1", staged.morphology_l1);
      ATTR("final_l1", final_l1);
      ATTR("band_width", es.metrics.band_width);
      ATTR("promotion", es.metrics.promotion);
      ATTR("attempts", es.metrics.attempts);
      ATTR("front_seed_cells", es.metrics.front_seed_cells);
      size_t editable_cells = 0;
      for (size_t j = 0; j < n; j++)
        if (es.upper[j] > es.lower[j])
          editable_cells++;
      ATTR("editable_cells", editable_cells);
      ATTR("editable_fraction", (double)editable_cells / n);
      ATTR("source_count", final.metrics.sources);
      ATTR("target_count", final.metrics.targets);
      ATTR("edge_count", final.metrics.source_target_edges);
      ATTR("positive_edges", final.metrics.positive_edges);
      ATTR("geometry_excess", final.metrics.max_geometry_excess);
      ATTR("max_source_overspend", final.metrics.max_source_overspend);
      ATTR("max_target_excess", final.metrics.max_target_excess);
      ATTR("max_edge_balance_error", final.metrics.max_edge_balance_error);
      ATTR("float_grid_ledger_error", grid_error);
      ATTR("float_grid_roundoff_bound", roundoff_bound);
      ATTR("reference_projection_l1", projection);
      ATTR("normalization_particles_per_cell", c->particle_normalization);
      ATTR("tolerance", options.relative_tolerance);
      ATTR("absolute_tolerance", options.absolute_tolerance);
      ATTR("elapsed_seconds", MPI_Wtime() - c->started);
      ATTR("completion_seconds", completion_seconds);
      ATTR("refinement_seconds", refinement_seconds);
      ATTR("shape_status", shape.status);
      ATTR("shape_iterations", shape.master_sweeps);
      ATTR("shape_evaluations", shape.master_evaluations);
      ATTR("max_evaluations", run_globals.params.PhotonConservationMaxEvaluations);
      ATTR("shape_oracle_calls", shape.oracle_calls);
      ATTR("shape_cuts", shape.hall_cuts);
      ATTR("shape_gap", shape.signed_primal_dual_gap);
      ATTR("shape_objective", shape.objective);
      ATTR("shape_equality_error", shape.equality_error);
      ATTR("shape_priority_error", shape.max_priority_equality_error);
      ATTR("shape_oracle_target_l1_error", shape.oracle_target_l1_error);
      ATTR("shape_max_final_correction", shape.max_final_target_correction);
      ATTR("shape_final_correction_l1", shape.final_target_correction_l1);
      ATTR("optimization_tolerance", shape.tolerance);
      ATTR("full_graph_certificate", final.metrics.full_graph_cut_separated);
#undef ATTR
      if (debug == 2) {
        const char* names[NINPUT] = { "source",  "source_II",   "source_BH",          "source_III", "capacity",
                                      "density", "neutral_pre", "recombinations_pre", "baseline",   "xH_ES" };
        for (int f = 0; f < NINPUT; f++)
          h5_double(file, names[f], n, input[f]);
        h5_double(file, "allocation", n, final.y);
        h5_double(file, "source_spent", n, final.source_spent);
        h5_double(file, "source_residual", n, final.source_residual);
        h5_double(file, "xH", n, xh_float);
        h5_double(file, "xH_double", n, xh_double);
        h5_double(file, "lower", n, es.lower);
        h5_double(file, "upper", n, es.upper);
        h5_double(file, "radius_cells", n, es.radius);
        h5_double(file, "radii_cells", (size_t)c->nradii, radii);
        h5_double(file, "retained_quota", (size_t)c->nradii + 1, quota);
        h5_int(file, "first_crossing", n, first);
        h5_int(file, "priority", n, es.priority);
        int* mask = pc_calloc(n, sizeof(int));
        for (size_t j = 0; j < n; j++)
          mask[j] = es.allocation.cut_source[j];
        h5_int(file, "staged_cut_source", n, mask);
        for (size_t j = 0; j < n; j++)
          mask[j] = es.allocation.cut_target[j];
        h5_int(file, "staged_cut_target", n, mask);
        for (size_t j = 0; j < n; j++)
          mask[j] = final.cut_source[j];
        h5_int(file, "cut_source", n, mask);
        for (size_t j = 0; j < n; j++)
          mask[j] = final.cut_target[j];
        h5_int(file, "cut_target", n, mask);
        free(mask);
      }
      h5_attribute(file, "certificate_complete", 1);
      if (H5Fclose(file) < 0)
        pc_fail("cannot close photon ledger");
      if (rename(path, final_path) != 0)
        pc_fail("cannot publish closed photon ledger by atomic rename");
    }
    if (run_globals.params.PhotonConservationRequireFull && !final.metrics.full_source_budget)
      pc_fail("unabsorbed photons remain; see the ledger and complete geometric cut");
    free(es.lower);
    free(es.upper);
    free(es.radius);
    free(es.priority);
    free(es.allocation.y);
    free(es.allocation.source_residual);
    free(es.allocation.cut_source);
    free(es.allocation.cut_target);
    free(es.allocation.priority_diagnostics);
    free(final.y);
    free(final.source_residual);
    free(final.source_spent);
    free(final.cut_source);
    free(final.cut_target);
    free(final.priority_diagnostics);
    free(radii);
    free(quota);
    free(xh_double);
    free(xh_float);
  }
  MPI_Scatterv(
    corrected, counts, offsets, MPI_FLOAT, run_globals.reion_grids.xH, c->nlocal, MPI_FLOAT, 0, run_globals.mpi_comm);
  MPI_Scatterv(feedback_slot, counts, offsets, MPI_INT, local_slot, c->nlocal, MPI_INT, 0, run_globals.mpi_comm);
  photon_conservation_update_feedback(c->snapshot, c->radii, c->nradii, local_slot, c->old_zre, c->old_j21);
  for (int f = 0; f < NINPUT; f++) {
    free(input[f]);
    free(c->input[f]);
  }
  free(first);
  free(corrected);
  free(feedback_slot);
  free(local_slot);
  free(counts);
  free(offsets);
  free(c->first);
  free(c->old_zre);
  free(c->old_j21);
  free(c);
}
