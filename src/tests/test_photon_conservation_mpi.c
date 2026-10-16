/* MPI accelerates only frozen-tree queries. These regressions require exact
 * ordered queries and exact solver/certificate replay, not just similar maps.
 * Run with one, two and four ranks. No Meraxes simulation data are needed. */
#include "../core/photon_conservation_mpi.h"
#include "../physics/photon_conservation_es.h"
#include "../physics/photon_conservation_shape.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CELLS 64
#define EDGES 4096
#define CHECK(c) do { if (!(c)) { \
  fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #c); \
  MPI_Abort(MPI_COMM_WORLD, 1); abort(); } } while (0)

typedef struct { size_t source, target; double amount; } saved_edge;
typedef struct {
  pc_result result;
  double y[CELLS], residual[CELLS], spent[CELLS];
  unsigned char cut_source[CELLS], cut_target[CELLS];
  pc_priority_diagnostic priority[CELLS];
  saved_edge edge[EDGES];
  size_t edge_count;
} saved_result;

static void save_edge(size_t source, size_t target, double amount, void* user)
{
  saved_result* r = user;
  CHECK(r->edge_count < EDGES);
  r->edge[r->edge_count++] = (saved_edge){ source, target, amount };
}

static void initialize_result(saved_result* r)
{
  memset(r, 0, sizeof(*r));
  r->result.y = r->y;
  r->result.source_residual = r->residual;
  r->result.source_spent = r->spent;
  r->result.cut_source = r->cut_source;
  r->result.cut_target = r->cut_target;
  r->result.priority_diagnostics = r->priority;
  r->result.priority_diagnostics_capacity = CELLS;
  r->result.visit_edge = save_edge;
  r->result.visitor_user = r;
}

static void compare_flow_metrics(const pc_metrics* a, const pc_metrics* b)
{
#define SAME(f) CHECK(a->f == b->f)
  SAME(lower_feasible); SAME(full_source_budget); SAME(full_graph_cut_separated);
  SAME(sources); SAME(targets); SAME(source_target_edges); SAME(positive_edges);
  SAME(separation_rounds); SAME(priority_groups);
  /* bytes_peak includes scratch needed by the selected execution path. */
  SAME(photons); SAME(lower_demand); SAME(absorbed); SAME(source_residual);
  SAME(mincut_capacity); SAME(source_cut_budget); SAME(target_cut_capacity);
  SAME(hall_source_excess); SAME(lower_cut_deficit); SAME(lower_stage_flow);
  SAME(retention_stage_flow); SAME(final_stage_flow); SAME(morphology_l1);
  SAME(retained_overlap); SAME(max_source_overspend); SAME(max_target_excess);
  SAME(max_lower_deficit); SAME(max_source_relative_residual);
  SAME(max_edge_balance_error); SAME(max_geometry_excess); SAME(ledger_error);
  SAME(cut_flow_error);
#undef SAME
}

static void compare_results(const saved_result* a, const saved_result* b, size_t n)
{
  CHECK(a->result.status == b->result.status);
  CHECK(memcmp(a->y, b->y, n * sizeof(double)) == 0);
  CHECK(memcmp(a->residual, b->residual, n * sizeof(double)) == 0);
  CHECK(memcmp(a->spent, b->spent, n * sizeof(double)) == 0);
  CHECK(memcmp(a->cut_source, b->cut_source, n) == 0);
  CHECK(memcmp(a->cut_target, b->cut_target, n) == 0);
  compare_flow_metrics(&a->result.metrics, &b->result.metrics);
  CHECK(a->edge_count == b->edge_count);
  for (size_t i = 0; i < a->edge_count; ++i) {
    CHECK(a->edge[i].source == b->edge[i].source);
    CHECK(a->edge[i].target == b->edge[i].target);
    CHECK(memcmp(&a->edge[i].amount, &b->edge[i].amount, sizeof(double)) == 0);
  }
  CHECK(a->result.metrics.priority_groups <= CELLS);
  for (size_t i = 0; i < a->result.metrics.priority_groups; ++i) {
    CHECK(a->priority[i].priority == b->priority[i].priority);
    CHECK(a->priority[i].prefix_flow == b->priority[i].prefix_flow);
    CHECK(a->priority[i].retained_in_group == b->priority[i].retained_in_group);
    CHECK(a->priority[i].baseline_retained_in_group == b->priority[i].baseline_retained_in_group);
  }
}

/* Independent literal pre-optimization bound, including all three images.
 * This does not call the production geometry implementation. */
static double original_bound(const size_t dim[3], const pc_geometry_node* node, const int pos[3])
{
  double total = 0.;
  for (int axis = 0; axis < 3; ++axis) {
    double best = DBL_MAX;
    for (int shift = -1; shift <= 1; ++shift) {
      double p = pos[axis] + shift * (double)dim[axis];
      double d = fmax(fmax(node->lo[axis] - p, p - node->hi[axis]), 0.);
      best = fmin(best, d);
    }
    total += best * best;
  }
  return total;
}

static void original_collect(const size_t dim[3], const pc_geometry_node* tree, int node,
                             const pc_geometry_query* query, int limit, int* out, int* count)
{
  if (*count == limit || original_bound(dim, tree + node, query->pos) >
                            query->radius * query->radius + 1e-10)
    return;
  if (tree[node].leaf >= 0) {
    out[(*count)++] = tree[node].leaf;
    return;
  }
  int a = tree[node].left, b = tree[node].right;
  if (original_bound(dim, tree + a, query->pos) > original_bound(dim, tree + b, query->pos)) {
    int t = a; a = b; b = t;
  }
  original_collect(dim, tree, a, query, limit, out, count);
  original_collect(dim, tree, b, query, limit, out, count);
}

static int build_tree(pc_geometry_node* tree, int* next, const int points[][3], int first, int end, int revision)
{
  int node = (*next)++;
  pc_geometry_node* t = tree + node;
  t->leaf = -1;
  for (int axis = 0; axis < 3; ++axis) {
    t->lo[axis] = t->hi[axis] = points[first][axis];
    for (int i = first + 1; i < end; ++i) {
      if (points[i][axis] < t->lo[axis]) t->lo[axis] = points[i][axis];
      if (points[i][axis] > t->hi[axis]) t->hi[axis] = points[i][axis];
    }
  }
  if (end == first + 1) {
    t->leaf = (first * 5 + revision) % 13;
    t->left = t->right = -1;
  } else {
    int middle = first + (end - first) / 2;
    t->left = build_tree(tree, next, points, first, middle, revision);
    t->right = build_tree(tree, next, points, middle, end, revision);
  }
  return node;
}

static void geometry_regression(pc_geometry_executor* executor)
{
  enum { QUERIES = 253, STRIDE = 16 };
  pc_geometry_node tree[25];
  pc_geometry_query queries[QUERIES];
  int expected[QUERIES * STRIDE], result[QUERIES * STRIDE];
  int expected_count[QUERIES], result_count[QUERIES];
  const int limits[] = { 1, 4, 8, STRIDE };
  /* Reuse the tree address and node count with different contents. A cached
   * tree must not survive begin: real residual cuts change eligible sources. */
  for (int variant = 0; variant < 3; ++variant) {
    size_t dim[3] = { 7, 5, 3 };
    if (variant == 2) { dim[0] = INT_MAX; dim[1] = dim[2] = 1; }
    int points[13][3], nodes = 0;
    for (int i = 0; i < 13; ++i) {
      points[i][0] = variant == 2 ? (int)((int64_t)i * (INT_MAX - 1) / 12) : (i * 3 + variant) % 7;
      points[i][1] = variant == 2 ? 0 : (i * 2 + variant) % 5;
      points[i][2] = variant == 2 ? 0 : (i + variant) % 3;
    }
    CHECK(build_tree(tree, &nodes, points, 0, 13, variant) == 0);
    CHECK(nodes == 25);
    for (int i = 0; i < QUERIES; ++i) {
      for (int axis = 0; axis < 3; ++axis)
        queries[i].pos[axis] = i < 13 ? points[i][axis] :
          (int)(((uint64_t)i * (axis + 17)) % dim[axis]);
      switch (i % 7) {
        case 0: queries[i].radius = 0.; break;
        case 1: queries[i].radius = 1.; break;
        case 2: queries[i].radius = sqrt(2.); break;
        case 3: queries[i].radius = nextafter(1., 0.); break;
        case 4: queries[i].radius = nextafter(1., INFINITY); break;
        case 5: queries[i].radius = .25; break;
        default: queries[i].radius = (double)dim[0]; break;
      }
    }
    for (size_t lim = 0; lim < sizeof(limits) / sizeof(limits[0]); ++lim) {
      int limit = limits[lim];
      for (int i = 0; i < QUERIES; ++i) {
        expected_count[i] = 0;
        original_collect(dim, tree, 0, queries + i, limit, expected + i * limit, expected_count + i);
        pc_geometry_collect(dim, tree, queries + i, limit, result + i * limit, result_count + i);
        CHECK(expected_count[i] == result_count[i]);
        CHECK(memcmp(expected + i * limit, result + i * limit, (size_t)expected_count[i] * sizeof(int)) == 0);
      }
      int accepted = executor->begin(executor->user, dim, tree, (size_t)nodes, (size_t)limit, QUERIES);
      CHECK(accepted >= 0);
      if (accepted) {
        CHECK(executor->batch_targets > 0);
        for (size_t offset = 0; offset < QUERIES;) {
          /* Seven targets gives uneven chunks and fewer queries than ranks
           * on the final call. Respect any smaller backend batch capacity. */
          size_t count = QUERIES - offset;
          if (count > 7) count = 7;
          if (count > executor->batch_targets) count = executor->batch_targets;
          CHECK(executor->query(executor->user, queries + offset, count,
                                result_count + offset, result + offset * (size_t)limit));
          offset += count;
        }
        executor->end(executor->user);
        for (int i = 0; i < QUERIES; ++i) {
          CHECK(expected_count[i] == result_count[i]);
          CHECK(memcmp(expected + i * limit, result + i * limit, (size_t)expected_count[i] * sizeof(int)) == 0);
        }
      }
    }
  }
}

static pc_status compare_allocation(const pc_problem* p, const pc_options* options,
                                    pc_geometry_executor* executor, saved_result* mpi_result)
{
  saved_result serial;
  initialize_result(&serial);
  initialize_result(mpi_result);
  pc_status a = pc_allocate(p, options, &serial.result);
  pc_status b = pc_allocate_with_geometry(p, options, &mpi_result->result, executor);
  CHECK(a == b);
  compare_results(&serial, mpi_result, p->ncell);
  return b;
}

static void allocation_regression(pc_geometry_executor* executor)
{
  double source[48], capacity[48], baseline[48], lower[48], radius[48];
  int priority[48];
  pc_problem p = { { 6, 4, 2 }, 48, source, capacity, baseline, lower, capacity, radius, priority };
  pc_options options;
  saved_result result;
  pc_default_options(&options);
  for (size_t i = 0; i < 48; ++i) {
    source[i] = i % 7 == 0 ? .6 : 0.;
    capacity[i] = .7 + (i % 5) * .13;
    baseline[i] = .13 * capacity[i];
    lower[i] = 0.;
    radius[i] = 1.01;
    priority[i] = (int)(i % 3);
  }
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_OK);
  options.max_edges = 1;
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_ERR_EDGE_LIMIT);
  pc_default_options(&options);
  options.max_bytes = 1;
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_ERR_MEMORY_LIMIT);
  pc_default_options(&options);
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_OK);
  for (size_t i = 0; i < 48; ++i) lower[i] = .9 * capacity[i];
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_LOWER_INFEASIBLE);
  for (size_t i = 0; i < 48; ++i) { lower[i] = 0.; source[i] = 1.5; }
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_PARTIAL);
  for (size_t i = 0; i < 48; ++i) source[i] = 0.;
  CHECK(compare_allocation(&p, &options, executor, &result) == PC_OK);
  {
    /* Zero-U neighbours are absent from the target flow vertices, but are
     * essential to the full geometric Hall cut consumed by shape refinement. */
    double s[8] = { 10 }, c[8] = { 1,1,1,1,1,1,1,1 }, z[8] = { 0 };
    double upper[8] = { 0,0,1,0,0,0,0,0 }, r[8] = { 2.01,2.01,2.01,2.01,2.01,2.01,2.01,2.01 };
    pc_problem zero = { { 8,1,1 }, 8, s,c,z,z,upper,r,NULL };
    CHECK(compare_allocation(&zero, &options, executor, &result) == PC_PARTIAL);
    CHECK(result.cut_source[0] && result.cut_target[7]);
  }
}

static void es_regression(pc_geometry_executor* executor)
{
  const double ladder[] = { 2., 1., 0. };
  uint32_t rng = 19790317;
  for (int sample = 0; sample < 12; ++sample) {
    double source[8], capacity[8], baseline[8];
    int first[8], ap[8], bp[8];
    double al[8], au[8], ar[8], bl[8], bu[8], br[8];
    for (int i = 0; i < 8; ++i) {
      rng = 1664525u * rng + 1013904223u;
      source[i] = (double)((rng >> 16) % 9) * .25;
      rng = 1664525u * rng + 1013904223u;
      capacity[i] = (rng >> 16) % 5 ? 1. : 0.;
      baseline[i] = capacity[i] * (double)((rng >> 24) % 5) * .25;
      rng = 1664525u * rng + 1013904223u;
      first[i] = (int)((rng >> 16) % 4) - 1;
    }
    pc_es_problem p = { { { 2,2,2 }, 8, source,capacity,baseline,NULL,NULL,NULL,NULL }, ladder,3,first };
    saved_result a, b;
    initialize_result(&a); initialize_result(&b);
    pc_es_result ea = { 0 }, eb = { 0 };
    ea.allocation = a.result; ea.lower = al; ea.upper = au; ea.radius = ar; ea.priority = ap;
    eb.allocation = b.result; eb.lower = bl; eb.upper = bu; eb.radius = br; eb.priority = bp;
    pc_options options;
    pc_default_options(&options);
    pc_status sa = pc_es_complete(&p, &options, &ea);
    pc_status sb = pc_es_complete_with_geometry(&p, &options, &eb, executor);
    CHECK(sa == sb && (sa == PC_OK || sa == PC_PARTIAL));
    a.result = ea.allocation; b.result = eb.allocation;
    compare_results(&a, &b, 8);
    CHECK(memcmp(al, bl, sizeof(al)) == 0 && memcmp(au, bu, sizeof(au)) == 0);
    CHECK(memcmp(ar, br, sizeof(ar)) == 0 && memcmp(ap, bp, sizeof(ap)) == 0);
#define SAME(f) CHECK(ea.metrics.f == eb.metrics.f)
    SAME(attempts); SAME(front_seed_cells); SAME(promotion); SAME(band_width);
    SAME(empty_front); SAME(terminal_full_source_budget);
    SAME(terminal_maximum_absorption); SAME(total_capacity);
#undef SAME
  }
}

static void compare_shape_metrics(const pc_shape_metrics* a, const pc_shape_metrics* b)
{
#define SAME(f) CHECK(a->f == b->f)
  SAME(status); SAME(last_oracle_status); SAME(hall_cuts); SAME(master_sweeps);
  SAME(master_evaluations); SAME(oracle_calls); SAME(source_budget);
  SAME(target_total); SAME(unused_budget); SAME(objective); SAME(evaluated_dual_lower);
  SAME(signed_primal_dual_gap); SAME(equality_error); SAME(max_hall_violation);
  SAME(max_priority_violation); SAME(max_priority_equality_error); SAME(max_bound_violation);
  SAME(max_complementarity); SAME(max_oracle_target_error); SAME(oracle_target_l1_error);
  SAME(max_final_target_correction); SAME(final_target_correction_l1);
  SAME(tolerance); SAME(full_graph_certificate);
#undef SAME
}

static void shape_case(const pc_problem* p, const double* prior, const double* expected,
                       pc_geometry_executor* executor)
{
  saved_result ref, a, b;
  initialize_result(&ref);
  pc_options flow;
  pc_shape_options options;
  pc_default_options(&flow);
  flow.absolute_tolerance = flow.relative_tolerance = 1e-12;
  pc_status st = pc_allocate(p, &flow, &ref.result);
  CHECK(st == PC_OK || st == PC_PARTIAL);
  pc_shape_default_options(&options);
  options.absolute_tolerance = 1e-12;
  options.relative_tolerance = 1e-11;
  options.max_hall_cuts = 128;
  /* A memory-limit failure must leave the MPI service usable for the solve. */
  for (int limited = 1; limited >= 0; --limited) {
    initialize_result(&a); initialize_result(&b);
    pc_shape_metrics ma, mb;
    pc_shape_options local = options;
    if (limited) local.max_bytes = 1;
    pc_shape_status sa = pc_shape_refine(p, &flow, &ref.result, prior, &local, &a.result, &ma);
    pc_shape_status sb = pc_shape_refine_with_geometry(p, &flow, &ref.result, prior, &local,
                                                      &b.result, &mb, executor);
    CHECK(sa == sb);
    CHECK(sa == (limited ? PC_SHAPE_ERR_MEMORY : PC_SHAPE_OK));
    compare_results(&a, &b, p->ncell);
    compare_shape_metrics(&ma, &mb);
    if (!limited) {
      CHECK(mb.full_graph_certificate);
      for (size_t i = 0; i < p->ncell; ++i) {
        CHECK(fabs(b.y[i] - expected[i]) < 2e-7);
        CHECK(b.y[i] >= p->lower[i] - 1e-10 && b.y[i] <= p->upper[i] + 1e-10);
        CHECK(fabs(b.spent[i] + b.residual[i] - p->source[i]) < 1e-10);
      }
    }
  }
}

static void shape_regression(pc_geometry_executor* executor)
{
  {
    double s[] = { 1.5,0,.5 }, c[] = { 1,1,2 }, b[] = { .3,1,.7 }, z[] = { 0,0,0 };
    double r[] = { 0,0,3 }, expected[] = { .3 + 1./3.,0,.7 + 2./3. };
    pc_problem p = { { 3,1,1 },3,s,c,b,z,c,r,NULL };
    shape_case(&p, NULL, expected, executor);
  }
  {
    double s[] = { 3,0,0,0,.8,0,0,0 }, c[] = { 1,1,0,0,1,1,0,0 }, z[8] = { 0 };
    double r[] = { 1,1,0,0,1,1,0,0 }, expected[] = { 1,1,0,0,.4,.4,0,0 };
    pc_problem p = { { 8,1,1 },8,s,c,z,z,c,r,NULL };
    shape_case(&p, NULL, expected, executor);
  }
  {
    double s[] = { 1,0,0,0,1.5,0,0,0 }, c[] = { 1,1,0,0,1,1,0,0 };
    double b[] = { .8,.8,0,0,.2,.6,0,0 }, z[8] = { 0 }, r[] = { 1,1,0,0,1,1,0,0 };
    double prior[] = { 0,1,0,0,.9,.1,0,0 }, expected[] = { .8,.2,0,0,.9,.6,0,0 };
    int priority[] = { 0,1,9,9,0,1,9,9 };
    pc_problem p = { { 8,1,1 },8,s,c,b,z,c,r,priority };
    shape_case(&p, prior, expected, executor);
  }
}

/* Exercise the allocator's accepted-executor path even in a one-rank run.
 * This fixture is not a substitute for the real multi-rank protocol tests. */
typedef struct {
  pc_geometry_executor executor;
  const pc_geometry_node* tree;
  size_t dim[3], stride, begins, ends, queries;
  int active, fail_next_query;
} local_executor;

static int local_begin(void* user, const size_t dim[3], const pc_geometry_node* tree,
                       size_t nodes, size_t max_neighbors, size_t targets)
{
  local_executor* local = user;
  CHECK(!local->active && nodes > 0 && targets > 0);
  local->tree = tree;
  memcpy(local->dim, dim, sizeof(local->dim));
  local->stride = max_neighbors;
  local->executor.batch_targets = 1 + 2 * (local->begins % 3);
  ++local->begins;
  local->active = 1;
  return 1;
}

static int local_query(void* user, const pc_geometry_query* queries, size_t count,
                       int* counts, int* neighbours)
{
  local_executor* local = user;
  CHECK(local->active && count <= local->executor.batch_targets);
  ++local->queries;
  if (local->fail_next_query) {
    local->fail_next_query = 0;
    return 0;
  }
  for (size_t i = 0; i < count; ++i)
    pc_geometry_collect(local->dim, local->tree, queries + i, (int)local->stride,
                         neighbours + i * local->stride, counts + i);
  return 1;
}

static void local_end(void* user)
{
  local_executor* local = user;
  CHECK(local->active);
  local->active = 0;
  ++local->ends;
}

static void local_executor_regression(void)
{
  local_executor local = { 0 };
  local.executor = (pc_geometry_executor){ &local, 1, local_begin, local_query, local_end };
  geometry_regression(&local.executor);
  allocation_regression(&local.executor);
  es_regression(&local.executor);
  shape_regression(&local.executor);
  {
    double s[] = { 1,0 }, c[] = { 1,1 }, z[] = { 0,0 }, r[] = { 1,1 };
    pc_problem p = { { 2,1,1 },2,s,c,z,z,c,r,NULL };
    pc_options options;
    saved_result result;
    pc_default_options(&options);
    initialize_result(&result);
    local.fail_next_query = 1;
    CHECK(pc_allocate_with_geometry(&p, &options, &result.result, &local.executor) == PC_ERR_MEMORY_LIMIT);
    CHECK(!local.active && local.begins == local.ends);
    CHECK(compare_allocation(&p, &options, &local.executor, &result) == PC_OK);
  }
  CHECK(!local.active && local.begins == local.ends && local.queries > 0);
  printf("PASS accepted local executor lifecycle begins=%zu queries=%zu\n", local.begins, local.queries);
}

static void service_test(int rank, int ranks, size_t memory_limit, int worker_declines)
{
  /* Different limits are deliberate: rank one declines after receiving
   * BEGIN, testing the collective allocation-failure protocol, whereas a
   * uniform one-byte limit is rejected locally before sending BEGIN. */
  size_t local_limit = worker_declines && rank == 1 ? 1 : memory_limit;
  pc_mpi_context* context = pc_mpi_create(MPI_COMM_WORLD, local_limit, 0);
  int created = context != NULL, all_created = 0;
  MPI_Allreduce(&created, &all_created, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  CHECK(all_created);
  if (rank == 0) {
    pc_geometry_executor* executor = pc_mpi_executor(context);
    CHECK(executor != NULL);
    geometry_regression(executor);
    allocation_regression(executor);
    es_regression(executor);
    shape_regression(executor);
    pc_mpi_stats stats;
    pc_mpi_get_stats(context, &stats);
    CHECK(stats.ranks == ranks && stats.begin_calls > 0);
    if (ranks > 1 && memory_limit != 1 && !worker_declines) {
      CHECK(stats.accepted_begins > 0 && stats.query_calls > 0 && stats.targets > 0);
    } else {
      CHECK(stats.declined_begins > 0 && stats.accepted_begins == 0);
      CHECK(stats.query_calls == 0 && stats.targets == 0);
    }
    printf("PASS MPI geometry/allocator/ES/shape ranks=%d memory=%zu worker_decline=%d begins=%zu accepted=%zu queries=%zu targets=%zu\n",
           ranks, memory_limit, worker_declines, stats.begin_calls, stats.accepted_begins, stats.query_calls, stats.targets);
    pc_mpi_stop(context);
  } else {
    pc_mpi_worker(context);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  pc_mpi_destroy(context);
}

int main(int argc, char** argv)
{
  int rank, ranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (rank == 0) local_executor_regression();
  service_test(rank, ranks, 64u * 1024u * 1024u, 0);
  service_test(rank, ranks, 1, 0);
  service_test(rank, ranks, 64u * 1024u * 1024u, 1);
  if (rank == 0) puts("All MPI photon-conservation equivalence cases passed.");
  MPI_Finalize();
  return 0;
}
