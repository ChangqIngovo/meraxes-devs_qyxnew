/* Deterministic synthetic benchmark of the complete allocation/refinement path.
 * This is not a cosmological simulation and is not a scalability forecast.
 *
 * Build (from repository root):
 *   cc -O2 -std=c99 -Isrc/physics tools/benchmark_photon_conservation_shape.c \
 *      src/physics/photon_conservation.c src/physics/photon_conservation_shape.c \
 *      src/physics/photon_conservation_geometry.c -lm -o benchmark_pc_shape
 * MPI: replace cc with mpicc, add -DPC_BENCHMARK_MPI and
 *      src/core/photon_conservation_mpi.c to the command above.
 * Run with mpiexec -n RANKS ./benchmark_pc_shape --n 12 --groups 4
 *      --scenario patchy --repeat 3 --output field.bin
 * --mpi-min-targets defaults to 1 so MPI is exercised even on small graphs;
 * production uses its own threshold to avoid unprofitable tiny collectives.
 * Run rank counts sequentially; compare complete output files with cmp.
 * Add -pg to all compilation/link steps for a gprof profile.
 */
#define _POSIX_C_SOURCE 200809L
#include "photon_conservation_shape.h"
#ifdef PC_BENCHMARK_MPI
#include "../src/core/photon_conservation_mpi.h"
static int benchmark_mpi_initialized = 0;
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t random_state = UINT64_C(987654321);

static double random_unit(void)
{
  random_state ^= random_state >> 12;
  random_state ^= random_state << 25;
  random_state ^= random_state >> 27;
  return (double)((random_state * UINT64_C(2685821657736338717)) >> 11) / 9007199254740992.0;
}

static double now_seconds(void)
{
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value)) {
    perror("clock_gettime");
#ifdef PC_BENCHMARK_MPI
    if (benchmark_mpi_initialized) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    exit(2);
  }
  return (double)value.tv_sec + 1e-9 * value.tv_nsec;
}

static size_t integer_argument(const char* text)
{
  char* end = NULL;
  unsigned long long value;
  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno || !end || *end || !value || value > SIZE_MAX || *text == '-') {
    fprintf(stderr, "Invalid positive integer: %s\n", text);
    exit(2);
  }
  return (size_t)value;
}

static void* allocate(size_t count, size_t size)
{
  void* result;
  if (size && count > SIZE_MAX / size) {
    fprintf(stderr, "Allocation overflow\n");
#ifdef PC_BENCHMARK_MPI
    if (benchmark_mpi_initialized) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    exit(2);
  }
  result = calloc(count, size);
  if (!result) {
    fprintf(stderr, "Allocation failed\n");
#ifdef PC_BENCHMARK_MPI
    if (benchmark_mpi_initialized) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    exit(2);
  }
  return result;
}

int main(int argc, char** argv)
{
  size_t dim = 8, repeats = 1, groups = 1, max_evaluations = 1000000, max_cuts = 512;
  size_t mpi_min_targets = 1;
  size_t n, j, repeat;
  const char *scenario = "bottleneck", *output = NULL;
  double radius = 1.01;
  double *arrays, *source, *capacity, *baseline, *lower, *upper, *radii;
  double *reference_y, *reference_residual, *result_y, *result_residual, *spent, *saved;
  int* priority;
  pc_priority_diagnostic *reference_quota, *result_quota;
  pc_problem problem;
  pc_options flow_options;
  pc_shape_options shape_options;
  pc_result reference, result;
  pc_status allocation_status;
  double allocation_started, allocation_seconds;
  int failed = 0;
#ifdef PC_BENCHMARK_MPI
  int mpi_rank, context_ok, all_contexts_ok;
  pc_mpi_context* mpi_context = NULL;
  pc_geometry_executor* geometry = NULL;
#endif
  for (int a = 1; a < argc; ++a) {
    const char* option = argv[a];
    if (!strcmp(option, "--help")) {
      puts("Options: --n DIM --repeat COUNT --groups COUNT --radius CELLS "
           "--scenario bottleneck|smooth|patchy --seed INTEGER "
           "--max-evaluations COUNT --max-cuts COUNT --output FIELD.bin "
           "--mpi-min-targets COUNT (MPI build only)");
      return 0;
    }
    if (++a == argc) {
      fprintf(stderr, "Missing value after %s\n", option);
      return 2;
    }
    if (!strcmp(option, "--n")) dim = integer_argument(argv[a]);
    else if (!strcmp(option, "--repeat")) repeats = integer_argument(argv[a]);
    else if (!strcmp(option, "--groups")) groups = integer_argument(argv[a]);
    else if (!strcmp(option, "--seed")) random_state = integer_argument(argv[a]);
    else if (!strcmp(option, "--max-evaluations")) max_evaluations = integer_argument(argv[a]);
    else if (!strcmp(option, "--max-cuts")) max_cuts = integer_argument(argv[a]);
    else if (!strcmp(option, "--mpi-min-targets")) mpi_min_targets = integer_argument(argv[a]);
    else if (!strcmp(option, "--radius")) {
      char* end = NULL;
      radius = strtod(argv[a], &end);
      if (!end || *end || !isfinite(radius) || radius < 1.) {
        fprintf(stderr, "Radius must be finite and at least one cell\n");
        return 2;
      }
    } else if (!strcmp(option, "--scenario")) scenario = argv[a];
    else if (!strcmp(option, "--output")) output = argv[a];
    else {
      fprintf(stderr, "Unknown option: %s\n", option);
      return 2;
    }
  }
  if (dim < 2 || dim > 256 || groups > 32 ||
      (strcmp(scenario, "bottleneck") && strcmp(scenario, "smooth") && strcmp(scenario, "patchy"))) {
    fprintf(stderr, "Require 2 <= n <= 256, groups <= 32, and a documented scenario\n");
    return 2;
  }
#ifdef PC_BENCHMARK_MPI
  MPI_Init(&argc, &argv);
  benchmark_mpi_initialized = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  mpi_context = pc_mpi_create(MPI_COMM_WORLD, (size_t)2048 * 1024 * 1024, mpi_min_targets);
  context_ok = mpi_context != NULL;
  MPI_Allreduce(&context_ok, &all_contexts_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_contexts_ok) {
    fprintf(stderr, "MPI benchmark context allocation failed\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (mpi_rank) {
    pc_mpi_worker(mpi_context);
    pc_mpi_destroy(mpi_context);
    MPI_Finalize();
    return 0;
  }
  geometry = pc_mpi_executor(mpi_context);
#else
  (void)mpi_min_targets;
#endif
  n = dim * dim * dim;
  arrays = allocate(12 * n, sizeof(double));
  priority = allocate(n, sizeof(int));
  reference_quota = allocate(groups, sizeof(*reference_quota));
  result_quota = allocate(groups, sizeof(*result_quota));
  source = arrays;
  capacity = source + n;
  baseline = capacity + n;
  lower = baseline + n;
  upper = lower + n;
  radii = upper + n;
  reference_y = radii + n;
  reference_residual = reference_y + n;
  result_y = reference_residual + n;
  result_residual = result_y + n;
  spent = result_residual + n;
  saved = spent + n;
  for (j = 0; j < n; ++j) {
    size_t x = j / (dim * dim), y = (j / dim) % dim, z = j % dim;
    double modulation;
    capacity[j] = .8 + .4 * random_unit();
    baseline[j] = .02 * capacity[j] * (.8 + .4 * random_unit());
    if (!strcmp(scenario, "smooth"))
      modulation = .20 + .10 * random_unit();
    else if (!strcmp(scenario, "patchy"))
      modulation = ((x / 3 + y / 3 + z / 3) % 2) ? .47 : .03;
    else
      modulation = x < dim / 2 ? .03 : .47;
    source[j] = capacity[j] * modulation * (.95 + .10 * random_unit());
    upper[j] = capacity[j];
    radii[j] = radius;
    priority[j] = (int)((x + 2 * y + 3 * z) % groups);
  }
  memset(&problem, 0, sizeof(problem));
  problem.dim[0] = problem.dim[1] = problem.dim[2] = dim;
  problem.ncell = n;
  problem.source = source;
  problem.capacity = capacity;
  problem.baseline = baseline;
  problem.lower = lower;
  problem.upper = upper;
  problem.radius = radii;
  problem.priority = priority;
  memset(&reference, 0, sizeof(reference));
  memset(&result, 0, sizeof(result));
  reference.y = reference_y;
  reference.source_residual = reference_residual;
  reference.priority_diagnostics = reference_quota;
  reference.priority_diagnostics_capacity = groups;
  result.y = result_y;
  result.source_residual = result_residual;
  result.source_spent = spent;
  result.priority_diagnostics = result_quota;
  result.priority_diagnostics_capacity = groups;
  pc_default_options(&flow_options);
  flow_options.max_bytes = (size_t)2048 * 1024 * 1024;
  flow_options.max_edges = 20000000;
  flow_options.absolute_tolerance = 1e-12;
  flow_options.relative_tolerance = 1e-9;
  pc_shape_default_options(&shape_options);
  shape_options.max_bytes = (size_t)2048 * 1024 * 1024;
  shape_options.max_hall_cuts = max_cuts;
  shape_options.max_evaluations = max_evaluations;
  shape_options.max_master_sweeps = 10000;
  shape_options.absolute_tolerance = 1e-12;
  shape_options.relative_tolerance = 1e-9;
  allocation_started = now_seconds();
#ifdef PC_BENCHMARK_MPI
  allocation_status = pc_allocate_with_geometry(&problem, &flow_options, &reference, geometry);
#else
  allocation_status = pc_allocate(&problem, &flow_options, &reference);
#endif
  allocation_seconds = now_seconds() - allocation_started;
  if (allocation_status != PC_OK && allocation_status != PC_PARTIAL) {
    fprintf(stderr, "Reference allocation failed: %s\n", pc_status_string(allocation_status));
    failed = 1;
    goto cleanup;
  }
  puts("scenario,dim,cells,groups,repeat,allocation_seconds,shape_seconds,status,cuts,sweeps,evaluations,oracles,objective,gap,tolerance,repeat_max_abs,ledger_error,mpi_ranks,mpi_query_seconds,mpi_begin_seconds,mpi_queries,mpi_targets");
  for (repeat = 0; repeat < repeats; ++repeat) {
    pc_shape_metrics metrics;
    pc_shape_status status;
    double max_difference = 0., started = now_seconds(), elapsed;
    double mpi_query_seconds = 0., mpi_begin_seconds = 0.;
    size_t mpi_queries = 0, mpi_targets = 0;
    int mpi_ranks = 1;
#ifdef PC_BENCHMARK_MPI
    pc_mpi_stats before, after;
    pc_mpi_get_stats(mpi_context, &before);
    status = pc_shape_refine_with_geometry(&problem, &flow_options, &reference, NULL,
                                          &shape_options, &result, &metrics, geometry);
    pc_mpi_get_stats(mpi_context, &after);
    mpi_query_seconds = after.query_seconds - before.query_seconds;
    mpi_begin_seconds = after.begin_seconds - before.begin_seconds;
    mpi_queries = after.query_calls - before.query_calls;
    mpi_targets = after.targets - before.targets;
    mpi_ranks = after.ranks;
#else
    status = pc_shape_refine(&problem, &flow_options, &reference, NULL, &shape_options, &result, &metrics);
#endif
    elapsed = now_seconds() - started;
    if (!repeat)
      memcpy(saved, result_y, n * sizeof(double));
    for (j = 0; j < n; ++j)
      max_difference = fmax(max_difference, fabs(result_y[j] - saved[j]));
    printf("%s,%zu,%zu,%zu,%zu,%.9g,%.9g,%d,%zu,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.9g,%.9g,%zu,%zu\n",
           scenario, dim, n, groups, repeat, allocation_seconds, elapsed, status,
           metrics.hall_cuts, metrics.master_sweeps, metrics.master_evaluations, metrics.oracle_calls,
           metrics.objective, metrics.signed_primal_dual_gap, metrics.tolerance,
           max_difference, result.metrics.ledger_error, mpi_ranks, mpi_query_seconds,
           mpi_begin_seconds, mpi_queries, mpi_targets);
    fflush(stdout);
    if (status != PC_SHAPE_OK || !metrics.full_graph_certificate ||
        memcmp(saved, result_y, n * sizeof(double)) != 0 || !isfinite(max_difference) || max_difference != 0.) {
      fprintf(stderr, "Shape status: %s; complete certificate=%d\n", pc_shape_status_string(status), metrics.full_graph_certificate);
      failed = 1;
      break;
    }
  }
  if (!failed && output) {
    FILE* stream = fopen(output, "wb");
    int wrote;
    if (!stream) {
      perror(output);
      failed = 1;
      goto cleanup;
    }
    wrote = fwrite(saved, sizeof(double), n, stream) == n;
    if (fclose(stream) || !wrote) {
      fprintf(stderr, "Output write failed: %s\n", output);
      failed = 1;
    }
  }
cleanup:
#ifdef PC_BENCHMARK_MPI
  pc_mpi_stop(mpi_context);
  pc_mpi_destroy(mpi_context);
#endif
  free(result_quota);
  free(reference_quota);
  free(priority);
  free(arrays);
#ifdef PC_BENCHMARK_MPI
  MPI_Finalize();
#endif
  return failed;
}
