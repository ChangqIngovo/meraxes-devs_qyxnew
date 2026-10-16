#include "photon_conservation.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                                             \
      failures++;                                                                                                      \
    }                                                                                                                  \
  } while (0)
#define CLOSE(a, b) (fabs((a) - (b)) < 1e-9)

typedef struct
{
  double source[8], capacity[8], baseline[8], lower[8], upper[8], radius[8];
  int priority[8];
  double y[8], residual[8], spent[8], rows[8], columns[8];
  unsigned char cut_source[8], cut_target[8];
  pc_priority_diagnostic diagnostics[8];
  size_t edges;
  pc_problem p;
  pc_options o;
  pc_result r;
} fixture;

static void visit(size_t i, size_t j, double photons, void* user)
{
  fixture* f = user;
  CHECK(i < f->p.ncell && j < f->p.ncell && photons > 0);
  if (i < 8 && j < 8) {
    f->rows[i] += photons;
    f->columns[j] += photons;
  }
  f->edges++;
}

static void initialize(fixture* f, size_t n)
{
  memset(f, 0, sizeof(*f));
  f->p.dim[0] = n;
  f->p.dim[1] = f->p.dim[2] = 1;
  f->p.ncell = n;
  f->p.source = f->source;
  f->p.capacity = f->capacity;
  f->p.baseline = f->baseline;
  f->p.lower = f->lower;
  f->p.upper = f->upper;
  f->p.radius = f->radius;
  f->p.priority = f->priority;
  f->r.y = f->y;
  f->r.source_residual = f->residual;
  f->r.source_spent = f->spent;
  f->r.cut_source = f->cut_source;
  f->r.cut_target = f->cut_target;
  f->r.priority_diagnostics = f->diagnostics;
  f->r.priority_diagnostics_capacity = 8;
  f->r.visit_edge = visit;
  f->r.visitor_user = f;
  pc_default_options(&f->o);
}

static pc_status solve(fixture* f)
{
  pc_status status = pc_allocate(&f->p, &f->o, &f->r);
  size_t i;
  if (status == PC_OK || status == PC_PARTIAL) {
    for (i = 0; i < f->p.ncell; i++) {
      CHECK(CLOSE(f->rows[i], f->spent[i]));
      CHECK(CLOSE(f->columns[i], f->y[i]));
      CHECK(CLOSE(f->spent[i] + f->residual[i], f->source[i]));
      CHECK(f->y[i] <= f->upper[i] + 1e-9 && f->y[i] >= f->lower[i] - 1e-9);
    }
    CHECK(f->r.metrics.full_graph_cut_separated);
    CHECK(f->r.metrics.cut_flow_error < 1e-9);
    CHECK(f->r.metrics.ledger_error < 1e-9);
  }
  return status;
}

static void analytic_cases(void)
{
  fixture f;
  initialize(&f, 1);
  f.source[0] = 3;
  f.capacity[0] = f.upper[0] = 2;
  f.baseline[0] = 1;
  CHECK(solve(&f) == PC_PARTIAL);
  CHECK(CLOSE(f.y[0], 2));
  CHECK(CLOSE(f.residual[0], 1));
  CHECK(f.edges == 1);

  /* A zero-capacity cell is still an emitting source. */
  initialize(&f, 2);
  f.source[0] = 1;
  f.capacity[1] = f.upper[1] = 1;
  f.radius[1] = 1;
  CHECK(solve(&f) == PC_OK);
  CHECK(CLOSE(f.y[1], 1));
  CHECK(CLOSE(f.spent[0], 1));
  CHECK(CLOSE(f.y[0], 0));

  /* Minimum-image periodic distance connects cells 0 and 3. */
  initialize(&f, 4);
  f.source[0] = 1;
  f.capacity[3] = f.upper[3] = 1;
  f.radius[3] = 1;
  CHECK(solve(&f) == PC_OK);
  CHECK(CLOSE(f.y[3], 1));

  /* Global capacity suffices but every local support is disconnected. */
  initialize(&f, 4);
  f.source[0] = f.source[2] = 1;
  f.capacity[1] = f.upper[1] = 1;
  f.capacity[3] = f.upper[3] = 1;
  CHECK(solve(&f) == PC_PARTIAL);
  CHECK(CLOSE(f.r.metrics.absorbed, 0));
  CHECK(CLOSE(f.r.metrics.hall_source_excess, 2));
  CHECK(f.cut_source[0] && f.cut_source[2] && !f.cut_target[1] && !f.cut_target[3]);

  initialize(&f, 2);
  f.source[0] = 1;
  f.capacity[1] = f.upper[1] = f.lower[1] = 1;
  CHECK(solve(&f) == PC_LOWER_INFEASIBLE);
  CHECK(!f.r.metrics.lower_feasible);
  CHECK(CLOSE(f.r.metrics.lower_cut_deficit, 1));
  CHECK(f.edges == 0);

  /* A zero current marginal remains in N(Q) if a later proposal can use it. */
  initialize(&f, 3);
  f.source[0] = 1;
  f.capacity[1] = 1;
  f.radius[1] = 1;
  CHECK(solve(&f) == PC_PARTIAL);
  CHECK(f.cut_source[0]);
  CHECK(f.cut_target[1]);

  /* Largest-crossing rank gets its target overlap first. */
  initialize(&f, 3);
  f.source[0] = 1;
  f.capacity[1] = f.upper[1] = f.capacity[2] = f.upper[2] = 1;
  f.radius[1] = f.radius[2] = 1;
  f.baseline[1] = f.baseline[2] = .8;
  f.priority[1] = 0;
  f.priority[2] = 1;
  CHECK(solve(&f) == PC_OK);
  CHECK(CLOSE(f.y[1], .8));
  CHECK(CLOSE(f.y[2], .2));
  CHECK(f.r.metrics.priority_groups == 2);
  CHECK(CLOSE(f.diagnostics[0].prefix_flow, .8));
  CHECK(CLOSE(f.diagnostics[1].baseline_retained_in_group, .2));

  /* Non-monotone baseline with a floor above its own baseline kink. */
  initialize(&f, 3);
  f.source[0] = 1;
  f.capacity[1] = f.upper[1] = f.capacity[2] = f.upper[2] = 1;
  f.radius[1] = f.radius[2] = 1;
  f.lower[1] = .7;
  f.baseline[1] = .2;
  f.baseline[2] = .8;
  f.priority[1] = 1;
  f.priority[2] = 0;
  CHECK(solve(&f) == PC_OK);
  CHECK(CLOSE(f.y[1], .7));
  CHECK(CLOSE(f.y[2], .3));
  CHECK(CLOSE(f.diagnostics[1].retained_in_group, .7));
  CHECK(CLOSE(f.diagnostics[1].baseline_retained_in_group, .2));

  /* An absolute tolerance never hides an entirely stranded faint source. */
  initialize(&f, 2);
  f.source[0] = 1;
  f.source[1] = 1e-15;
  f.capacity[0] = f.upper[0] = 1;
  CHECK(solve(&f) == PC_PARTIAL);
  CHECK(f.residual[1] == 1e-15);
  CHECK(f.r.metrics.max_source_relative_residual == 1);

  initialize(&f, 2);
  f.source[0] = 1;
  f.capacity[1] = f.upper[1] = f.lower[1] = 1e-15;
  CHECK(solve(&f) == PC_LOWER_INFEASIBLE);
  CHECK(!f.r.metrics.lower_feasible);

  initialize(&f, 1);
  f.capacity[0] = f.upper[0] = 1;
  CHECK(solve(&f) == PC_OK);
  CHECK(CLOSE(f.r.metrics.absorbed, 0));
  initialize(&f, 1);
  f.source[0] = 1;
  CHECK(solve(&f) == PC_PARTIAL);
  CHECK(CLOSE(f.residual[0], 1));
}

static void invalid_and_limits(void)
{
  fixture f;
  initialize(&f, 1);
  f.source[0] = NAN;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.source[0] = INFINITY;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.source[0] = -1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.radius[0] = NAN;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.priority[0] = -1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.upper[0] = 1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.lower[0] = 1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.p.dim[0] = SIZE_MAX;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.o.relative_tolerance = 1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  initialize(&f, 1);
  f.source[0] = 1;
  f.r.y = f.source;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_INVALID);
  CHECK(f.source[0] == 1);

  initialize(&f, 2);
  f.source[0] = f.capacity[0] = f.upper[0] = 1;
  f.source[1] = f.capacity[1] = f.upper[1] = 1;
  f.o.max_edges = 1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_EDGE_LIMIT);
  CHECK(!f.r.metrics.full_graph_cut_separated);
  CHECK(f.edges == 0);
  initialize(&f, 1);
  f.source[0] = f.capacity[0] = f.upper[0] = 1;
  f.o.max_edges = 0;
  f.o.max_bytes = 1;
  CHECK(pc_allocate(&f.p, &f.o, &f.r) == PC_ERR_MEMORY_LIMIT);
  CHECK(!f.r.metrics.full_graph_cut_separated);
}

static void unlimited_edges(void)
{
  fixture reference, f;
  const size_t limits[] = { 0, SIZE_MAX };
  size_t i, j;
  initialize(&reference, 8);
  CHECK(reference.o.max_edges == 0);
  for (i = 0; i < 8; i++) {
    reference.source[i] = i % 3 == 0 ? 1.25 : 0;
    reference.capacity[i] = reference.upper[i] = 1;
    reference.radius[i] = 2.01;
    reference.baseline[i] = .25;
    reference.priority[i] = (int)(i % 2);
  }
  reference.o.max_edges = 64;
  CHECK(solve(&reference) == PC_OK);
  CHECK(reference.r.metrics.source_target_edges > 1);
  for (j = 0; j < sizeof(limits) / sizeof(limits[0]); j++) {
    initialize(&f, 8);
    memcpy(f.source, reference.source, sizeof(f.source));
    memcpy(f.capacity, reference.capacity, sizeof(f.capacity));
    memcpy(f.upper, reference.upper, sizeof(f.upper));
    memcpy(f.radius, reference.radius, sizeof(f.radius));
    memcpy(f.baseline, reference.baseline, sizeof(f.baseline));
    memcpy(f.priority, reference.priority, sizeof(f.priority));
    f.o.max_edges = limits[j];
    /* A large count ceiling is not a request to allocate that many edges. */
    f.o.max_bytes = 1024 * 1024;
    CHECK(solve(&f) == PC_OK);
    CHECK(f.r.metrics.bytes_peak <= f.o.max_bytes);
    CHECK(f.r.metrics.source_target_edges == reference.r.metrics.source_target_edges);
    CHECK(f.r.metrics.positive_edges == reference.r.metrics.positive_edges);
    CHECK(f.r.metrics.full_source_budget == reference.r.metrics.full_source_budget);
    CHECK(f.r.metrics.absorbed == reference.r.metrics.absorbed);
    CHECK(f.r.metrics.source_residual == reference.r.metrics.source_residual);
    CHECK(f.r.metrics.mincut_capacity == reference.r.metrics.mincut_capacity);
    CHECK(f.r.metrics.priority_groups == reference.r.metrics.priority_groups);
    for (i = 0; i < 8; i++) {
      CHECK(f.y[i] == reference.y[i]);
      CHECK(f.spent[i] == reference.spent[i]);
      CHECK(f.residual[i] == reference.residual[i]);
      CHECK(f.cut_source[i] == reference.cut_source[i]);
      CHECK(f.cut_target[i] == reference.cut_target[i]);
    }
    for (i = 0; i < f.r.metrics.priority_groups; i++) {
      CHECK(f.diagnostics[i].priority == reference.diagnostics[i].priority);
      CHECK(f.diagnostics[i].prefix_flow == reference.diagnostics[i].prefix_flow);
      CHECK(f.diagnostics[i].retained_in_group == reference.diagnostics[i].retained_in_group);
      CHECK(f.diagnostics[i].baseline_retained_in_group == reference.diagnostics[i].baseline_retained_in_group);
    }
  }
}

static void chunks_and_memory(void)
{
  const size_t n = 4097;
  double *source = calloc(n, sizeof(double)), *cap = calloc(n, sizeof(double));
  double *zero = calloc(n, sizeof(double)), *y = calloc(n, sizeof(double)), *res = calloc(n, sizeof(double));
  pc_problem p;
  pc_result r;
  pc_options o;
  size_t i, peak;
  CHECK(source && cap && zero && y && res);
  if (!source || !cap || !zero || !y || !res)
    exit(2);
  memset(&p, 0, sizeof(p));
  memset(&r, 0, sizeof(r));
  p.dim[0] = n;
  p.dim[1] = p.dim[2] = 1;
  p.ncell = n;
  p.source = source;
  p.capacity = p.upper = cap;
  p.lower = p.baseline = p.radius = zero;
  r.y = y;
  r.source_residual = res;
  /* One local source-target pair per cell crosses several edge chunks. */
  for (i = 0; i < n; i++)
    source[i] = cap[i] = 1;
  pc_default_options(&o);
  o.max_edges = 0;
  CHECK(pc_allocate(&p, &o, &r) == PC_OK);
  CHECK(r.metrics.source_target_edges == n);
  CHECK(r.metrics.absorbed == n && r.metrics.source_residual == 0);
  CHECK(r.metrics.full_graph_cut_separated && r.metrics.full_source_budget);
  for (i = 0; i < n; i++)
    CHECK(y[i] == 1 && res[i] == 0);
  peak = r.metrics.bytes_peak;
  o.max_bytes = peak;
  CHECK(pc_allocate(&p, &o, &r) == PC_OK);
  CHECK(r.metrics.bytes_peak <= o.max_bytes);
  /* Metadata size can depend on the budget, so peak - 1 need not fail. */
  o.max_bytes = peak / 2;
  CHECK(pc_allocate(&p, &o, &r) == PC_ERR_MEMORY_LIMIT);
  CHECK(r.metrics.bytes_peak <= o.max_bytes);
  free(source);
  free(cap);
  free(zero);
  free(y);
  free(res);
}

int main(void)
{
  analytic_cases();
  invalid_and_limits();
  unlimited_edges();
  chunks_and_memory();
  if (failures) {
    fprintf(stderr, "%d photon-conservation checks failed\n", failures);
    return 1;
  }
  puts("photon-conservation standalone checks passed");
  return 0;
}
