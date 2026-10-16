#include "photon_conservation_es.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* A positive binary64 sum uses at most 2098 bits plus log2(n) carry bits.
 * 68 words also cover the largest possible size_t-length input on this API.
 * Compare cut capacities as exact INPUT sums: a faint source must not vanish
 * merely because two large rounded objective totals compare within absTol. */
#define ES_SUM_WORDS 68
typedef struct
{
  uint32_t word[ES_SUM_WORDS];
} exact_sum;

static void add_word(exact_sum* sum, size_t word, uint32_t value)
{
  uint64_t carry = value;
  while (carry && word < ES_SUM_WORDS) {
    carry += sum->word[word];
    sum->word[word++] = (uint32_t)carry;
    carry >>= 32;
  }
}

static void add_double(exact_sum* sum, double value)
{
  int exponent, shift;
  uint64_t mantissa, low, high;
  size_t word;
  if (value == 0.0)
    return;
  mantissa = (uint64_t)ldexp(frexp(value, &exponent), 53);
  shift = exponent - 53 + 1074;
  if (shift < 0) {
    mantissa >>= -shift;
    shift = 0;
  }
  word = (size_t)shift / 32;
  shift %= 32;
  low = (mantissa & UINT32_MAX) << shift;
  high = (mantissa >> 32) << shift;
  add_word(sum, word, (uint32_t)low);
  add_word(sum, word + 1, (uint32_t)(low >> 32));
  add_word(sum, word + 1, (uint32_t)high);
  add_word(sum, word + 2, (uint32_t)(high >> 32));
}

static int compare_sum(const exact_sum* a, const exact_sum* b)
{
  for (size_t k = ES_SUM_WORDS; k-- > 0;)
    if (a->word[k] != b->word[k])
      return a->word[k] > b->word[k] ? 1 : -1;
  return 0;
}

typedef struct
{
  const pc_es_problem* input;
  pc_es_result* output;
  pc_options options;
  const pc_geometry_executor* geometry;
  int* distance;
  size_t* queue;
  unsigned char *cut_source, *cut_target;
  int maximum_distance, terminal_band_rank;
  size_t maximum_promotion;
  double maximum_absorption;
  int terminal_full;
  exact_sum maximum_cut, current_cut;
} es_context;

static size_t neighbour(const pc_problem* p, size_t cell, int axis, int forward)
{
  size_t stride = axis == 0 ? p->dim[1] * p->dim[2] : axis == 1 ? p->dim[2] : 1;
  size_t coordinate = (cell / stride) % p->dim[axis];
  if (forward)
    return coordinate + 1 < p->dim[axis] ? cell + stride : cell - (p->dim[axis] - 1) * stride;
  return coordinate ? cell - stride : cell + (p->dim[axis] - 1) * stride;
}

static int overlaps(const void* a, size_t na, const void* b, size_t nb)
{
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  if (!a || !b)
    return 0;
  return x <= y ? y - x < na : x - y < nb;
}

static int valid_input(const pc_es_problem* in, const pc_es_result* out)
{
  size_t product = 1, j;
  if (FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || sizeof(size_t) > 8)
    return 0;
  if (!in || !out || !in->grid.source || !in->grid.capacity || !in->grid.baseline || !in->first_crossing ||
      !in->radii_desc || !in->radius_count || in->radius_count > (size_t)INT_MAX || !out->allocation.y ||
      !out->allocation.source_residual || !out->lower || !out->upper || !out->radius || !out->priority)
    return 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (!in->grid.dim[axis] || in->grid.dim[axis] > (size_t)(INT_MAX / 3) || product > SIZE_MAX / in->grid.dim[axis])
      return 0;
    product *= in->grid.dim[axis];
  }
  if (product != in->grid.ncell || !product || product > SIZE_MAX / sizeof(size_t) ||
      product > SIZE_MAX / sizeof(int) || product > SIZE_MAX / sizeof(double) ||
      in->radius_count > SIZE_MAX / sizeof(double))
    return 0;
  {
    const void* read[5] = { in->grid.source, in->grid.capacity, in->grid.baseline, in->first_crossing, in->radii_desc };
    size_t read_bytes[5] = { product * sizeof(double),
                             product * sizeof(double),
                             product * sizeof(double),
                             product * sizeof(int),
                             in->radius_count * sizeof(double) };
    void* write[7] = { out->lower,
                       out->upper,
                       out->radius,
                       out->priority,
                       out->allocation.y,
                       out->allocation.source_residual,
                       out->allocation.source_spent };
    size_t write_bytes[7] = { product * sizeof(double), product * sizeof(double), product * sizeof(double),
                              product * sizeof(int),    product * sizeof(double), product * sizeof(double),
                              product * sizeof(double) };
    for (size_t a = 0; a < 7; ++a) {
      for (size_t b = 0; b < 5; ++b)
        if (overlaps(write[a], write_bytes[a], read[b], read_bytes[b]))
          return 0;
      for (size_t b = 0; b < a; ++b)
        if (overlaps(write[a], write_bytes[a], write[b], write_bytes[b]))
          return 0;
    }
  }
  if (in->radii_desc[in->radius_count - 1] != 0.0)
    return 0;
  for (j = 0; j < in->radius_count; ++j) {
    if (!isfinite(in->radii_desc[j]) || in->radii_desc[j] < 0.0 || (j && !(in->radii_desc[j - 1] > in->radii_desc[j])))
      return 0;
  }
  for (j = 0; j < product; ++j) {
    double s = in->grid.source[j], c = in->grid.capacity[j];
    double b = in->grid.baseline[j];
    if (!isfinite(s) || !isfinite(c) || !isfinite(b) || s < 0.0 || c < 0.0 || b < 0.0 || b > c ||
        in->first_crossing[j] < -1 || in->first_crossing[j] >= (int)in->radius_count)
      return 0;
  }
  return 1;
}

static void seed_front(es_context* ctx)
{
  const pc_problem* p = &ctx->input->grid;
  size_t head = 0, tail = 0;
  for (size_t j = 0; j < p->ncell; ++j) {
    ctx->distance[j] = -1;
    if (p->baseline[j] > 0.0 && p->baseline[j] < p->capacity[j])
      ctx->distance[j] = 0;
  }
  /* Zero-capacity cells do not manufacture a neutral/ionized interface. */
  for (size_t j = 0; j < p->ncell; ++j) {
    if (!(p->capacity[j] > 0.0))
      continue;
    for (int axis = 0; axis < 3; ++axis) {
      size_t k = neighbour(p, j, axis, 1);
      if (p->capacity[k] > 0.0 && ((p->baseline[j] == p->capacity[j]) != (p->baseline[k] == p->capacity[k]))) {
        ctx->distance[j] = 0;
        ctx->distance[k] = 0;
      }
    }
  }
  for (size_t j = 0; j < p->ncell; ++j)
    if (ctx->distance[j] == 0)
      ctx->queue[tail++] = j;
  ctx->output->metrics.front_seed_cells = tail;
  ctx->output->metrics.empty_front = tail == 0;
  ctx->maximum_distance = tail ? 0 : -1;
  /* Distances traverse the whole periodic grid, including zero capacities. */
  while (head < tail) {
    size_t j = ctx->queue[head++];
    for (int axis = 0; axis < 3; ++axis)
      for (int forward = 0; forward < 2; ++forward) {
        size_t k = neighbour(p, j, axis, forward);
        if (ctx->distance[k] < 0) {
          ctx->distance[k] = ctx->distance[j] + 1;
          if (ctx->distance[k] > ctx->maximum_distance)
            ctx->maximum_distance = ctx->distance[k];
          ctx->queue[tail++] = k;
        }
      }
  }
  /* rank 0: frozen; ranks 1..D+1: widths 0..D; last: explicit all-grid. */
  ctx->terminal_band_rank = ctx->maximum_distance + 2;
}

static int band_width(const es_context* ctx, int rank)
{
  if (rank == ctx->terminal_band_rank)
    return PC_ES_BAND_ALL;
  return rank == 0 ? PC_ES_BAND_EMPTY : rank - 1;
}

static void make_problem(es_context* ctx, int rank, size_t promotion, pc_problem* problem)
{
  const pc_es_problem* in = ctx->input;
  pc_es_result* out = ctx->output;
  int width = band_width(ctx, rank);
  *problem = in->grid;
  problem->lower = out->lower;
  problem->upper = out->upper;
  problem->radius = out->radius;
  problem->priority = out->priority;
  for (size_t j = 0; j < problem->ncell; ++j) {
    int edit = width == PC_ES_BAND_ALL || (width >= 0 && ctx->distance[j] >= 0 && ctx->distance[j] <= width);
    size_t original = in->first_crossing[j] >= 0 ? (size_t)in->first_crossing[j] : in->radius_count - 1;
    size_t selected = original > promotion ? original - promotion : 0;
    out->lower[j] = edit ? 0.0 : problem->baseline[j];
    out->upper[j] = edit ? problem->capacity[j] : problem->baseline[j];
    out->radius[j] = in->radii_desc[selected];
    out->priority[j] = in->first_crossing[j] >= 0 ? in->first_crossing[j] : (int)in->radius_count;
  }
}

static pc_status solve(es_context* ctx, int rank, size_t promotion, int final, pc_metrics* metrics)
{
  pc_problem problem;
  pc_result trial = ctx->output->allocation;
  make_problem(ctx, rank, promotion, &problem);
  trial.cut_source = ctx->cut_source;
  trial.cut_target = ctx->cut_target;
  if (!final) {
    /* Feasibility searches need no priority optimization or edge callbacks. */
    problem.priority = NULL;
    trial.visit_edge = NULL;
    trial.visitor_user = NULL;
    trial.priority_diagnostics = NULL;
    trial.priority_diagnostics_capacity = 0;
  }
  pc_status status = pc_allocate_with_geometry(&problem, &ctx->options, &trial, ctx->geometry);
  /* A resource or numerical failure must not become a false infeasible
   * predicate in either monotone search. Every nonnegative allocator status
   * must include its complete-geometry cut audit. */
  if (status >= 0 && !trial.metrics.full_graph_cut_separated)
    status = PC_ERR_NUMERICAL;
  memset(&ctx->current_cut, 0, sizeof(ctx->current_cut));
  if (status == PC_OK || status == PC_PARTIAL)
    for (size_t j = 0; j < problem.ncell; ++j) {
      if (!ctx->cut_source[j])
        add_double(&ctx->current_cut, problem.source[j]);
      if (ctx->cut_target[j])
        add_double(&ctx->current_cut, problem.upper[j]);
    }
  ++ctx->output->metrics.attempts;
  if (metrics)
    *metrics = trial.metrics;
  ctx->output->allocation.metrics = trial.metrics;
  ctx->output->allocation.status = status;
  if (final) {
    if (ctx->output->allocation.cut_source)
      memcpy(ctx->output->allocation.cut_source, ctx->cut_source, problem.ncell);
    if (ctx->output->allocation.cut_target)
      memcpy(ctx->output->allocation.cut_target, ctx->cut_target, problem.ncell);
    trial.cut_source = ctx->output->allocation.cut_source;
    trial.cut_target = ctx->output->allocation.cut_target;
    ctx->output->allocation = trial;
  }
  if (ctx->output->visit_attempt)
    ctx->output->visit_attempt(band_width(ctx, rank), promotion, status, &trial.metrics, ctx->output->attempt_user);
  return status;
}

static int attains_maximum(const es_context* ctx, pc_status status, const pc_metrics* metrics)
{
  double tolerance;
  if (status == PC_LOWER_INFEASIBLE || !metrics->lower_feasible)
    return 0;
  {
    int cut_order = compare_sum(&ctx->current_cut, &ctx->maximum_cut);
    if (cut_order < 0)
      return 0;
    if (cut_order > 0)
      return -1; /* A certified subset maximum cannot exceed its superset. */
  }
  /* A full-source terminal maximum additionally requires the allocator's
   * individual-source certificate; a small global count deficit is not enough. */
  if (ctx->terminal_full && !metrics->full_source_budget)
    return -1;
  tolerance = ctx->options.absolute_tolerance + ctx->options.relative_tolerance * fabs(ctx->maximum_absorption);
  return fabs(metrics->absorbed - ctx->maximum_absorption) <= tolerance ? 1 : -1;
}

pc_status pc_es_complete_with_geometry(const pc_es_problem* input, const pc_options* options,
                                      pc_es_result* output, const pc_geometry_executor* geometry)
{
  es_context ctx;
  pc_metrics metrics;
  pc_status status;
  size_t n, driver_bytes;
  int band_lo, band_hi;
  size_t promotion_lo, promotion_hi;
  if (!valid_input(input, output))
    return PC_ERR_INVALID;
  memset(&ctx, 0, sizeof(ctx));
  memset(&output->metrics, 0, sizeof(output->metrics));
  memset(&output->allocation.metrics, 0, sizeof(output->allocation.metrics));
  output->allocation.status = PC_ERR_INVALID;
  ctx.input = input;
  ctx.output = output;
  ctx.geometry = geometry;
  if (options)
    ctx.options = *options;
  else
    pc_default_options(&ctx.options);
  if (!ctx.options.max_bytes || ctx.options.flow_tolerance != 0.0 || !isfinite(ctx.options.relative_tolerance) ||
      ctx.options.relative_tolerance < 0.0 || ctx.options.relative_tolerance >= 1.0)
    return PC_ERR_INVALID;
  n = input->grid.ncell;
  if (n > SIZE_MAX / (sizeof(int) + sizeof(size_t) + 2))
    return PC_ERR_MEMORY_LIMIT;
  driver_bytes = n * (sizeof(int) + sizeof(size_t) + 2);
  if (driver_bytes >= ctx.options.max_bytes) {
    output->allocation.status = PC_ERR_MEMORY_LIMIT;
    return PC_ERR_MEMORY_LIMIT;
  }
  ctx.options.max_bytes -= driver_bytes;
  ctx.distance = malloc(n * sizeof(*ctx.distance));
  ctx.queue = malloc(n * sizeof(*ctx.queue));
  ctx.cut_source = malloc(n);
  ctx.cut_target = malloc(n);
  if (!ctx.distance || !ctx.queue || !ctx.cut_source || !ctx.cut_target) {
    free(ctx.distance);
    free(ctx.queue);
    free(ctx.cut_source);
    free(ctx.cut_target);
    output->allocation.status = PC_ERR_MEMORY_LIMIT;
    return PC_ERR_MEMORY_LIMIT;
  }
  seed_front(&ctx);
  ctx.maximum_promotion = input->radius_count - 1;
  {
    long double total_capacity = 0.0L;
    for (size_t j = 0; j < n; ++j)
      total_capacity += (long double)input->grid.capacity[j];
    output->metrics.total_capacity = (double)total_capacity;
  }

  /* The terminal graph defines the attainable target, including excess sources
   * or finite-reach bottlenecks. It is not replaced by min(P,total capacity). */
  status = solve(&ctx, ctx.terminal_band_rank, ctx.maximum_promotion, 0, &metrics);
  if (status < 0 || status == PC_LOWER_INFEASIBLE) {
    if (status >= 0)
      status = PC_ERR_NUMERICAL;
    goto done;
  }
  ctx.maximum_absorption = metrics.absorbed;
  ctx.maximum_cut = ctx.current_cut;
  ctx.terminal_full = metrics.full_source_budget;
  output->metrics.terminal_maximum_absorption = ctx.maximum_absorption;
  output->metrics.terminal_full_source_budget = ctx.terminal_full;

  /* Monotone lower-floor feasibility AND attainment of the terminal maximum. */
  band_lo = 0;
  band_hi = ctx.terminal_band_rank;
  while (band_lo < band_hi) {
    int mid = band_lo + (band_hi - band_lo) / 2;
    status = solve(&ctx, mid, ctx.maximum_promotion, 0, &metrics);
    if (status < 0)
      goto done;
    int attained = attains_maximum(&ctx, status, &metrics);
    if (attained < 0) {
      status = PC_ERR_NUMERICAL;
      goto done;
    }
    if (attained)
      band_hi = mid;
    else
      band_lo = mid + 1;
  }
  promotion_lo = 0;
  promotion_hi = ctx.maximum_promotion;
  while (promotion_lo < promotion_hi) {
    size_t mid = promotion_lo + (promotion_hi - promotion_lo) / 2;
    status = solve(&ctx, band_lo, mid, 0, &metrics);
    if (status < 0)
      goto done;
    int attained = attains_maximum(&ctx, status, &metrics);
    if (attained < 0) {
      status = PC_ERR_NUMERICAL;
      goto done;
    }
    if (attained)
      promotion_hi = mid;
    else
      promotion_lo = mid + 1;
  }
  output->metrics.band_width = band_width(&ctx, band_lo);
  output->metrics.promotion = promotion_lo;
  status = solve(&ctx, band_lo, promotion_lo, 1, &metrics);
  if (status >= 0 && attains_maximum(&ctx, status, &metrics) != 1)
    status = PC_ERR_NUMERICAL;

done:
  output->allocation.metrics.bytes_peak += driver_bytes;
  output->allocation.status = status;
  free(ctx.distance);
  free(ctx.queue);
  free(ctx.cut_source);
  free(ctx.cut_target);
  return status;
}

pc_status pc_es_complete(const pc_es_problem* input, const pc_options* options, pc_es_result* output)
{
  return pc_es_complete_with_geometry(input, options, output, NULL);
}
