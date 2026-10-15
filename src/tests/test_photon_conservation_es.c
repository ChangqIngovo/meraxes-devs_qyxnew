/* Standalone tests: link photon_conservation.c and photon_conservation_es.c.
 * These are finite-model tests, not a Meraxes chemistry validation. */
#include "photon_conservation_es.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REF_N 8
#define REF_V (2 * REF_N + 2)

/* Independent dense Edmonds--Karp reference. Dyadic capacities make these
 * small cases exact in binary arithmetic; it enumerates every family member
 * instead of using the production driver's binary searches or KD tree. */
static double dense_augment(double residual[REF_V][REF_V], int nv)
{
  double added = 0.0;
  for (;;) {
    int parent[REF_V], queue[REF_V], head = 0, tail = 1;
    for (int i = 0; i < nv; ++i)
      parent[i] = -1;
    queue[0] = 0;
    parent[0] = 0;
    while (head < tail && parent[1] < 0) {
      int a = queue[head++];
      for (int b = 0; b < nv; ++b)
        if (parent[b] < 0 && residual[a][b] > 0.0) {
          parent[b] = a;
          queue[tail++] = b;
        }
    }
    if (parent[1] < 0)
      return added;
    double amount = INFINITY;
    for (int v = 1; v; v = parent[v])
      amount = fmin(amount, residual[parent[v]][v]);
    for (int v = 1; v; v = parent[v]) {
      residual[parent[v]][v] -= amount;
      residual[v][parent[v]] += amount;
    }
    added += amount;
  }
}

static int periodic_distance(int a, int b, const int dim[3], int squared)
{
  int distance = 0;
  for (int axis = 2; axis >= 0; --axis) {
    int delta = abs(a % dim[axis] - b % dim[axis]);
    if (dim[axis] - delta < delta)
      delta = dim[axis] - delta;
    distance += squared ? delta * delta : delta;
    a /= dim[axis];
    b /= dim[axis];
  }
  return distance;
}

static double dense_flow(int n,
                         const int dim[3],
                         const double* source,
                         const double* lower,
                         const double* upper,
                         const double* radius,
                         int* lower_feasible)
{
  double residual[REF_V][REF_V] = { { 0 } }, demand = 0.0;
  for (int j = 0; j < n; ++j) {
    residual[0][2 + j] = source[j];
    residual[2 + n + j][1] = lower[j];
    demand += lower[j];
    for (int i = 0; i < n; ++i)
      if ((double)periodic_distance(i, j, dim, 1) <= radius[j] * radius[j])
        residual[2 + i][2 + n + j] = 1024.0;
  }
  double total = dense_augment(residual, 2 * n + 2);
  *lower_feasible = total == demand;
  if (!*lower_feasible)
    return total;
  for (int j = 0; j < n; ++j)
    residual[2 + n + j][1] += upper[j] - lower[j];
  return total + dense_augment(residual, 2 * n + 2);
}

static void exhaustive_reference(void)
{
  const int dim[3] = { 2, 2, 2 };
  const double ladder[3] = { 2.0, 1.0, 0.0 };
  uint32_t rng = 19790317;
  for (int sample = 0; sample < 40; ++sample) {
    double source[REF_N], capacity[REF_N], baseline[REF_N];
    double lower[REF_N], upper[REF_N], radius[REF_N], y[REF_N], unused[REF_N];
    int first[REF_N], priority[REF_N], seed[REF_N] = { 0 }, distance[REF_N];
    for (int j = 0; j < REF_N; ++j) {
      rng = 1664525u * rng + 1013904223u;
      source[j] = (double)((rng >> 16) % 9) * 0.25;
      rng = 1664525u * rng + 1013904223u;
      capacity[j] = (rng >> 16) % 5 ? 1.0 : 0.0;
      baseline[j] = capacity[j] * (double)((rng >> 24) % 5) * 0.25;
      rng = 1664525u * rng + 1013904223u;
      first[j] = (int)((rng >> 16) % 4) - 1;
      seed[j] = baseline[j] > 0.0 && baseline[j] < capacity[j];
    }
    for (int j = 0; j < REF_N; ++j)
      for (int k = 0; k < REF_N; ++k)
        if (capacity[j] > 0.0 && capacity[k] > 0.0 && periodic_distance(j, k, dim, 0) == 1 &&
            ((baseline[j] == capacity[j]) != (baseline[k] == capacity[k])))
          seed[j] = seed[k] = 1;
    int maxdist = -1, seed_count = 0;
    for (int j = 0; j < REF_N; ++j) {
      distance[j] = -1;
      seed_count += seed[j];
      for (int k = 0; k < REF_N; ++k)
        if (seed[k]) {
          int d = periodic_distance(j, k, dim, 0);
          if (distance[j] < 0 || d < distance[j])
            distance[j] = d;
        }
      if (distance[j] > maxdist)
        maxdist = distance[j];
      lower[j] = 0.0;
      upper[j] = capacity[j];
      radius[j] = ladder[0];
    }
    int feasible, expected_rank = -1, expected_promotion = -1;
    double maximum = dense_flow(REF_N, dim, source, lower, upper, radius, &feasible);
    assert(feasible);
    const int terminal_rank = maxdist + 2;
    for (int rank = 0; rank <= terminal_rank && expected_rank < 0; ++rank) {
      for (int j = 0; j < REF_N; ++j) {
        int editable = rank == terminal_rank || (rank > 0 && distance[j] >= 0 && distance[j] < rank);
        lower[j] = editable ? 0.0 : baseline[j];
        upper[j] = editable ? capacity[j] : baseline[j];
        radius[j] = ladder[0];
      }
      double total = dense_flow(REF_N, dim, source, lower, upper, radius, &feasible);
      if (feasible && total == maximum)
        expected_rank = rank;
    }
    assert(expected_rank >= 0);
    for (int promotion = 0; promotion < 3 && expected_promotion < 0; ++promotion) {
      for (int j = 0; j < REF_N; ++j) {
        int original = first[j] < 0 ? 2 : first[j];
        radius[j] = ladder[original > promotion ? original - promotion : 0];
      }
      double total = dense_flow(REF_N, dim, source, lower, upper, radius, &feasible);
      if (feasible && total == maximum)
        expected_promotion = promotion;
    }
    pc_es_problem problem = { 0 };
    pc_es_result result = { 0 };
    for (int axis = 0; axis < 3; ++axis)
      problem.grid.dim[axis] = (size_t)dim[axis];
    problem.grid.ncell = REF_N;
    problem.grid.source = source;
    problem.grid.capacity = capacity;
    problem.grid.baseline = baseline;
    problem.radii_desc = ladder;
    problem.radius_count = 3;
    problem.first_crossing = first;
    result.allocation.y = y;
    result.allocation.source_residual = unused;
    result.lower = lower;
    result.upper = upper;
    result.radius = radius;
    result.priority = priority;
    pc_status status = pc_es_complete(&problem, NULL, &result);
    assert(status == PC_OK || status == PC_PARTIAL);
    int expected_band = expected_rank == terminal_rank ? PC_ES_BAND_ALL
                        : expected_rank == 0           ? PC_ES_BAND_EMPTY
                                                       : expected_rank - 1;
    assert(result.metrics.band_width == expected_band);
    assert(result.metrics.promotion == (size_t)expected_promotion);
    assert(result.metrics.front_seed_cells == (size_t)seed_count);
    assert(result.metrics.terminal_maximum_absorption == maximum);
    assert(result.allocation.metrics.absorbed == maximum);
  }
}

static void faint_source_and_periodic_geometry(void)
{
  const double ladder[2] = { 1.0, 0.0 };
  double source[4] = { 0.0, 0.0, 0.5, 0.0 };
  const double capacity[4] = { 1.0, 0.0, 0.0, 1.0 };
  double baseline[4] = { 0.0, 0.0, 0.0, 0.5 };
  int first[4] = { -1, -1, -1, 0 }, priority[4];
  double y[4], unused[4], lower[4], upper[4], radius[4];
  pc_es_problem problem = { 0 };
  pc_es_result result = { 0 };
  problem.grid.dim[0] = 4;
  problem.grid.dim[1] = problem.grid.dim[2] = 1;
  problem.grid.ncell = 4;
  problem.grid.source = source;
  problem.grid.capacity = capacity;
  problem.grid.baseline = baseline;
  problem.radii_desc = ladder;
  problem.radius_count = 2;
  problem.first_crossing = first;
  result.allocation.y = y;
  result.allocation.source_residual = unused;
  result.lower = lower;
  result.upper = upper;
  result.radius = radius;
  result.priority = priority;
  /* Original nonlocal first crossing already permits donor 2 to target 3. */
  assert(pc_es_complete(&problem, NULL, &result) == PC_OK);
  assert(result.metrics.band_width == PC_ES_BAND_EMPTY);
  assert(result.metrics.promotion == 0);
  assert(y[3] == 0.5 && radius[3] == 1.0);
  source[2] = 0.0;
  source[0] = 0.5;
  assert(pc_es_complete(&problem, NULL, &result) == PC_OK);
  assert(result.metrics.band_width == PC_ES_BAND_EMPTY);
  assert(result.metrics.promotion == 0);
  assert(y[3] == 0.5); /* Periodic wrap from cell 0 to target 3. */

  {
    double s[5] = { 2.0, 0.0, 0x1p-80, 0.0, 0.0 };
    const double c[5] = { 1.0, 0.0, 0.0, 1.0, 0.0 };
    const double b[5] = { 1.0, 0.0, 0.0, 0.0, 0.0 };
    const int f[5] = { 1, -1, -1, -1, -1 };
    double yy[5], rr[5], ll[5], uu[5], rad[5];
    int pr[5];
    problem.grid.dim[0] = problem.grid.ncell = 5;
    problem.grid.source = s;
    problem.grid.capacity = c;
    problem.grid.baseline = b;
    problem.first_crossing = f;
    result.allocation.y = yy;
    result.allocation.source_residual = rr;
    result.lower = ll;
    result.upper = uu;
    result.radius = rad;
    result.priority = pr;
    /* No positive-C face interface exists, so the declared all-domain escape
     * is needed. 1+2^-80 rounds to 1 even in usual extended precision: exact
     * input-cut comparison must still reject empty band and local radius. */
    assert(pc_es_complete(&problem, NULL, &result) == PC_PARTIAL);
    assert(result.metrics.band_width == PC_ES_BAND_ALL);
    assert(result.metrics.promotion == 1);
    assert(yy[0] == 1.0 && yy[3] == 0x1p-80 && rr[2] == 0.0);
    pc_options options;
    pc_default_options(&options);
    options.max_bytes = 0;
    assert(pc_es_complete(&problem, &options, &result) == PC_ERR_INVALID);
    options.max_bytes = 1;
    assert(pc_es_complete(&problem, &options, &result) == PC_ERR_MEMORY_LIMIT);
  }
}

static void close_count(double actual, double expected)
{
  assert(fabs(actual - expected) < 1e-10);
}

static void check_case(const double* source,
                       const double* capacity,
                       const double* baseline,
                       const int* first,
                       const double* ladder,
                       size_t nladder,
                       pc_status expected_status,
                       int expected_band,
                       size_t expected_promotion,
                       const double* expected_y,
                       double expected_maximum)
{
  double y[2], residual[2], lower[2], upper[2], radius[2];
  int priority[2];
  pc_priority_diagnostic groups[4];
  pc_options options;
  pc_es_problem problem;
  pc_es_result result;
  memset(&problem, 0, sizeof(problem));
  memset(&result, 0, sizeof(result));
  problem.grid.dim[0] = 2;
  problem.grid.dim[1] = problem.grid.dim[2] = 1;
  problem.grid.ncell = 2;
  problem.grid.source = source;
  problem.grid.capacity = capacity;
  problem.grid.baseline = baseline;
  problem.radii_desc = ladder;
  problem.radius_count = nladder;
  problem.first_crossing = first;
  result.allocation.y = y;
  result.allocation.source_residual = residual;
  result.allocation.priority_diagnostics = groups;
  result.allocation.priority_diagnostics_capacity = 4;
  result.lower = lower;
  result.upper = upper;
  result.radius = radius;
  result.priority = priority;
  pc_default_options(&options);
  pc_status status = pc_es_complete(&problem, &options, &result);
  if (status != expected_status)
    fprintf(
      stderr, "Unexpected ES status %s (expected %s)\n", pc_status_string(status), pc_status_string(expected_status));
  assert(status == expected_status);
  assert(result.metrics.band_width == expected_band);
  assert(result.metrics.promotion == expected_promotion);
  close_count(result.metrics.terminal_maximum_absorption, expected_maximum);
  for (size_t j = 0; j < 2; ++j) {
    close_count(y[j], expected_y[j]);
    assert(y[j] >= lower[j] - 1e-10 && y[j] <= upper[j] + 1e-10);
    assert(residual[j] >= -1e-10);
    assert(priority[j] == (first[j] >= 0 ? first[j] : (int)nladder));
  }
  close_count(y[0] + y[1] + residual[0] + residual[1], source[0] + source[1]);
}

int main(void)
{
  const double ladder[] = { 1.0, 0.0 }, self_only[] = { 0.0 };
  const double unit_capacity[] = { 1.0, 1.0 };
  {
    /* The larger sphere fails its barrier (1.5<2), but the fractional front
     * must still be permitted to absorb the remaining half photon. */
    const double s[] = { 1.5, 0.0 }, b[] = { 1.0, 0.0 }, want[] = { 1.0, 0.5 };
    const int first[] = { 1, -1 };
    check_case(s, unit_capacity, b, first, ladder, 2, PC_OK, 0, 1, want, 1.5);
  }
  {
    /* A feasible original local tail needs neither a band nor promotion. */
    const double s[] = { 0.2, 0.3 }, b[] = { 0.2, 0.3 };
    const int first[] = { -1, -1 };
    check_case(s, unit_capacity, b, first, ladder, 2, PC_OK, PC_ES_BAND_EMPTY, 0, b, 0.5);
  }
  {
    /* Global capacity exceeds the budget, but the terminal reach is local. */
    const double s[] = { 1.5, 0.0 }, b[] = { 1.0, 0.0 };
    const int first[] = { 0, -1 };
    check_case(s, unit_capacity, b, first, self_only, 1, PC_PARTIAL, PC_ES_BAND_EMPTY, 0, b, 1.0);
  }
  {
    /* A fully marked reference has no interface: recession is an explicit
     * full-domain relaxation, not an invented thin front. */
    const double s[] = { 1.0, 0.0 }, b[] = { 1.0, 1.0 };
    const int first[] = { 1, 1 };
    const double want[] = { 1.0, 0.0 };
    check_case(s, unit_capacity, b, first, ladder, 2, PC_OK, PC_ES_BAND_ALL, 0, want, 1.0);
  }
  {
    /* Removing a zero-capacity target must not remove its source inventory. */
    const double s[] = { 0.5, 0.0 }, c[] = { 0.0, 1.0 }, b[] = { 0.0, 0.0 };
    const int first[] = { -1, -1 };
    const double want[] = { 0.0, 0.5 };
    check_case(s, c, b, first, ladder, 2, PC_OK, PC_ES_BAND_ALL, 1, want, 0.5);
  }
  {
    /* Excess total sources do not imply every target can be reached. */
    const double s[] = { 3.0, 0.0 }, b[] = { 1.0, 0.0 };
    const int first[] = { 0, -1 };
    check_case(s, unit_capacity, b, first, self_only, 1, PC_PARTIAL, PC_ES_BAND_EMPTY, 0, b, 1.0);
  }
  exhaustive_reference();
  faint_source_and_periodic_geometry();
  puts("ES completion: nine hand cases, resource guards and 40 exhaustive-family references passed");
  return 0;
}
