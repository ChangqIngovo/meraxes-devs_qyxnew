#include "photon_conservation_shape.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
  const pc_problem* p;
  const double* prior;
  pc_shape_options o;
  pc_shape_metrics* m;
  size_t n, groups, cuts, bitbytes, bytes, nactive;
  size_t* active;
  size_t newton_size;
  double *newton_matrix, *newton_factor, *newton_work;
  size_t* newton_index;
  long double source_sum, target_sum;
  double total, photons, tolerance, master_feasibility;
  long double lambda;
  double *y, *bias, *flow_y, *remainder, *spent, *master_upper, *desired;
  double *quota, *nu, *rhs, *mu;
  int* group;
  unsigned char *cut_source, *cut_target;
  unsigned char** mask;
  pc_shape_status error;
} shape_context;

void pc_shape_default_options(pc_shape_options* o)
{
  if (!o)
    return;
  o->max_hall_cuts = 32;
  o->max_master_sweeps = 500;
  o->max_coordinate_iterations = 80;
  o->max_waterfill_iterations = 100;
  o->max_evaluations = 200000;
  o->max_bytes = (size_t)1024 * 1024 * 1024;
  o->absolute_tolerance = 1e-12;
  o->relative_tolerance = 1e-10;
  o->maximum_multiplier = 1e12;
}

const char* pc_shape_status_string(pc_shape_status s)
{
  switch (s) {
    case PC_SHAPE_OK:
      return "unique shape converged and transport certified";
    case PC_SHAPE_ERR_INVALID:
      return "invalid shape problem or reference certificate";
    case PC_SHAPE_ERR_MEMORY:
      return "shape memory limit or allocation failure";
    case PC_SHAPE_ERR_MASTER_LIMIT:
      return "shape dual iteration/evaluation limit";
    case PC_SHAPE_ERR_CUT_LIMIT:
      return "shape Hall-cut limit";
    case PC_SHAPE_ERR_ORACLE:
      return "fixed-marginal transport oracle failed";
    case PC_SHAPE_ERR_NUMERICAL:
      return "shape numerical certificate failed";
    default:
      return "unknown shape status";
  }
}

static void* shape_alloc(shape_context* c, size_t count, size_t size)
{
  size_t bytes;
  void* p;
  if (size && count > SIZE_MAX / size)
    goto fail;
  bytes = count * size;
  if (bytes > SIZE_MAX - c->bytes || (c->o.max_bytes && bytes > c->o.max_bytes - c->bytes))
    goto fail;
  p = calloc(count ? count : 1, size ? size : 1);
  if (!p)
    goto fail;
  c->bytes += bytes;
  if (c->bytes > c->m->bytes_peak)
    c->m->bytes_peak = c->bytes;
  return p;
fail:
  c->error = PC_SHAPE_ERR_MEMORY;
  return NULL;
}

static int bit_get(const unsigned char* mask, size_t j)
{
  return (mask[j >> 3] >> (j & 7)) & 1;
}

static double clip(double y, double l, double u)
{
  return y < l ? l : (y > u ? u : y);
}

/* Exact scalar minimizer of f_j(y)+c_j*y-nu_g*min(y,b_j). */
static int shape_map(shape_context* c, long double lambda, long double* sum, long double* slope)
{
  size_t j, a;
  *sum = *slope = 0.;
  if (++c->m->master_evaluations > c->o.max_evaluations) {
    c->error = PC_SHAPE_ERR_MASTER_LIMIT;
    return 0;
  }
  for (a = 0; a < c->nactive; ++a) {
    j = c->active[a];
    double C = c->p->capacity[j], y = 0.;
    if (C > 0.) {
      long double q = (long double)c->prior[j] + (long double)C * ((long double)c->bias[j] - lambda);
      long double upper_root = q + (long double)C * c->nu[c->group[j]];
      double b = c->p->baseline[j];
      int moving = 1;
      if (q > b)
        y = q;
      else if (upper_root < b)
        y = upper_root;
      else {
        y = b;
        moving = 0;
      }
      y = clip(y, c->p->lower[j], c->master_upper[j]);
      if (!isfinite(y)) {
        c->error = PC_SHAPE_ERR_NUMERICAL;
        return 0;
      }
      if (moving && y > c->p->lower[j] && y < c->master_upper[j])
        *slope += C;
    }
    c->y[j] = y;
    *sum += y;
  }
  return 1;
}

/* Eliminate the total multiplier with safeguarded segment-wise Newton. */
static int waterfill(shape_context* c)
{
  size_t j, a, it;
  long double lo = LDBL_MAX, hi = -LDBL_MAX, lambda;
  double tol = 8. * DBL_EPSILON * c->total;
  long double sum = 0., slope = 0.;
  for (a = 0; a < c->nactive; ++a) {
    j = c->active[a];
    double C = c->p->capacity[j];
    long double a = ((long double)c->prior[j] - c->master_upper[j]) / C + c->bias[j];
    long double b = ((long double)c->prior[j] - c->p->lower[j]) / C + c->bias[j] + c->nu[c->group[j]];
    if (a < lo)
      lo = a;
    if (b > hi)
      hi = b;
  }
  if (lo == LDBL_MAX) {
    memset(c->y, 0, c->n * sizeof(double));
    c->lambda = 0.;
    return fabs(c->total) <= c->tolerance;
  }
  lambda = fminl(hi, fmaxl(lo, c->lambda));
  for (it = 0; it < c->o.max_waterfill_iterations; ++it) {
    long double error;
    long double next;
    if (!shape_map(c, lambda, &sum, &slope))
      return 0;
    error = sum - c->target_sum;
    if (fabsl(error) <= tol) {
      c->lambda = lambda;
      return 1;
    }
    if (error > 0.)
      lo = lambda;
    else
      hi = lambda;
    next = slope > 0. ? lambda + error / slope : .5 * (lo + hi);
    if (!(next > lo && next < hi) || !isfinite(next))
      next = lo + .5 * (hi - lo);
    if (next == lambda) {
      c->lambda = lambda;
      if (fabsl(error) <= 32. * DBL_EPSILON * c->total)
        return 1;
      c->error = PC_SHAPE_ERR_NUMERICAL;
      return 0;
    }
    lambda = next;
  }
  c->error = PC_SHAPE_ERR_MASTER_LIMIT;
  return 0;
}

static double multiplier(const shape_context* c, size_t k)
{
  return k < c->cuts ? c->mu[k] : c->nu[k - c->cuts];
}

static void set_multiplier(shape_context* c, size_t k, double value)
{
  size_t j, a;
  if (k < c->cuts) {
    double delta = value - c->mu[k];
    for (a = 0; a < c->nactive; ++a) {
      j = c->active[a];
      if (bit_get(c->mask[k], j))
        c->bias[j] += delta;
    }
    c->mu[k] = value;
  } else
    c->nu[k - c->cuts] = value;
}

static double constraint_slack(const shape_context* c, size_t k)
{
  size_t j, a;
  long double value = 0.;
  if (k < c->cuts) {
    for (a = 0; a < c->nactive; ++a) {
      j = c->active[a];
      if (bit_get(c->mask[k], j))
        value += c->y[j];
    }
    value -= c->rhs[k];
  } else {
    size_t g = k - c->cuts;
    for (a = 0; a < c->nactive; ++a) {
      j = c->active[a];
      if ((size_t)c->group[j] == g)
        value += fmin(c->y[j], c->p->baseline[j]);
    }
    value -= c->quota[g];
  }
  return (double)value;
}

/* Exact one-coordinate convex minimization. Both Hall and retention slacks
 * are monotone in their own nonnegative multiplier after total elimination. */
static int optimize_coordinate(shape_context* c, size_t k)
{
  double old = multiplier(c, k), lo = 0., hi, slack;
  size_t it;
  set_multiplier(c, k, 0.);
  if (!waterfill(c))
    return 0;
  slack = constraint_slack(c, k);
  if (slack >= -c->tolerance * .1)
    return 1;
  hi = fmax(1., old);
  for (;;) {
    if (hi > c->o.maximum_multiplier) {
      c->error = PC_SHAPE_ERR_MASTER_LIMIT;
      return 0;
    }
    set_multiplier(c, k, hi);
    if (!waterfill(c))
      return 0;
    slack = constraint_slack(c, k);
    if (fabs(slack) <= c->tolerance * .05)
      return 1;
    if (slack > 0.)
      break;
    lo = hi;
    hi *= 2.;
  }
  for (it = 0; it < c->o.max_coordinate_iterations; ++it) {
    double mid = lo + .5 * (hi - lo);
    set_multiplier(c, k, mid);
    if (!waterfill(c))
      return 0;
    slack = constraint_slack(c, k);
    if (fabs(slack) <= c->tolerance * .1)
      return 1;
    if (slack < 0.)
      lo = mid;
    else
      hi = mid;
    if (hi - lo <= 16. * DBL_EPSILON * fmax(1., fabs(mid))) {
      if (fabs(slack) <= c->tolerance)
        return 1;
      c->error = PC_SHAPE_ERR_NUMERICAL;
      return 0;
    }
  }
  c->error = PC_SHAPE_ERR_MASTER_LIMIT;
  return 0;
}

static int master_certificate(shape_context* c)
{
  size_t j, k, a;
  long double total = 0., objective = 0., gap = 0.;
  double kkt = 0., hall = 0., priority = 0., complementarity = 0., bounds = 0.;
  /* Rebuild the exact multiplier sum before evaluating the dual infimum. */
  for (a = 0; a < c->nactive; ++a) {
    j = c->active[a];
    c->bias[j] = 0.;
    for (k = 0; k < c->cuts; ++k)
      if (bit_get(c->mask[k], j))
        c->bias[j] += c->mu[k];
  }
  if (!waterfill(c))
    return 0;
  for (j = 0; j < c->n; ++j) {
    double d = c->y[j] - c->prior[j];
    total += c->y[j];
    if (c->p->capacity[j] > 0.)
      objective += .5L * d * d / c->p->capacity[j];
    bounds = fmax(bounds, fmax(c->p->lower[j] - c->y[j], c->y[j] - c->p->upper[j]));
  }
  for (k = 0; k < c->cuts + c->groups; ++k) {
    double s = constraint_slack(c, k), theta = multiplier(c, k);
    double projected = theta > 0. ? fabs(s) : fmax(-s, 0.);
    kkt = fmax(kkt, projected);
    if (k < c->cuts)
      hall = fmax(hall, -s);
    else
      priority = fmax(priority, -s);
    complementarity = fmax(complementarity, fabs(theta * s));
    gap += (long double)theta * s;
  }
  gap -= (long double)c->lambda * (total - c->target_sum);
  c->m->objective = (double)objective;
  c->m->signed_primal_dual_gap = (double)gap;
  c->m->evaluated_dual_lower = (double)(objective - gap);
  c->m->equality_error = (double)fabsl(total - c->target_sum);
  c->m->max_hall_violation = hall;
  c->m->max_priority_violation = priority;
  c->m->max_complementarity = complementarity;
  c->m->max_bound_violation = bounds;
  /* Final acceptance uses the transported primal against this dual bound.
   * Solving all individual dual KKT equations more tightly is unnecessary. */
  return (kkt <= c->tolerance || (hall <= c->master_feasibility && priority <= c->master_feasibility)) &&
         c->m->equality_error <= c->tolerance && bounds <= c->tolerance && fabsl(gap) <= c->tolerance;
}

/* Semismooth Newton on the small active dual. The total multiplier has
 * already been eliminated: H = B' diag(C_free) B - vv'/sum(C_free).
 * A tiny diagonal regularizer handles dependent cuts. Every step is checked
 * against the actual separable dual; coordinate minimization remains fallback. */
static int try_newton(shape_context* c)
{
  size_t count = 0, k, l, a, j, it, dim = c->newton_size;
  double *H = c->newton_matrix, *work = c->newton_work;
  double *old = work, *gradient = old + dim, *step = gradient + dim;
  double *center = step + dim, *feature = center + dim, *rhs = feature + dim;
  double* factor = c->newton_factor;
  double slope = 0., trace = 0., direction = 0., alpha = 1.;
  double old_kkt = 0.;
  long double old_lambda = c->lambda, old_equality = -c->target_sum;
  if (!dim)
    return 0;
  memcpy(c->desired, c->y, c->n * sizeof(double));
  for (a = 0; a < c->nactive; ++a)
    old_equality += c->y[c->active[a]];
  for (k = 0; k < c->cuts + c->groups; ++k) {
    double slack = constraint_slack(c, k);
    if (multiplier(c, k) > 0. || slack < -c->tolerance) {
      /* Stored inactive cuts do not consume the bounded Newton workspace. */
      if (count == dim)
        return 0;
      c->newton_index[count] = k;
      old[count] = multiplier(c, k);
      gradient[count] = slack;
      old_kkt = fmax(old_kkt, fabs(slack));
      ++count;
    }
  }
  if (!count)
    return 0;
  memset(H, 0, count * count * sizeof(double));
  memset(center, 0, count * sizeof(double));
  for (a = 0; a < c->nactive; ++a) {
    double C, y;
    j = c->active[a];
    C = c->p->capacity[j];
    y = c->y[j];
    if (!(y > c->p->lower[j] && y < c->master_upper[j]) || (y == c->p->baseline[j] && c->nu[c->group[j]] > 0.))
      continue;
    slope += C;
    for (k = 0; k < count; ++k) {
      size_t v = c->newton_index[k];
      feature[k] = v < c->cuts ? bit_get(c->mask[v], j) : ((size_t)c->group[j] == v - c->cuts && y < c->p->baseline[j]);
      center[k] += C * feature[k];
    }
  }
  if (!(slope > 0.))
    return 0;
  for (k = 0; k < count; ++k)
    center[k] /= slope;
  /* Center first, then form the Gram matrix: subtracting two large raw
   * moments loses the curvature of almost-full, nearly dependent masks. */
  for (a = 0; a < c->nactive; ++a) {
    double C, y;
    j = c->active[a];
    C = c->p->capacity[j];
    y = c->y[j];
    if (!(y > c->p->lower[j] && y < c->master_upper[j]) || (y == c->p->baseline[j] && c->nu[c->group[j]] > 0.))
      continue;
    for (k = 0; k < count; ++k) {
      size_t v = c->newton_index[k];
      feature[k] =
        (v < c->cuts ? bit_get(c->mask[v], j) : ((size_t)c->group[j] == v - c->cuts && y < c->p->baseline[j])) -
        center[k];
    }
    for (k = 0; k < count; ++k)
      for (l = 0; l <= k; ++l)
        H[k * count + l] += C * feature[k] * feature[l];
  }
  for (k = 0; k < count; ++k)
    for (l = 0; l <= k; ++l) {
      H[l * count + k] = H[k * count + l];
      if (k == l)
        trace += fmax(0., H[k * count + l]);
    }
  if (!(trace > 0.))
    return 0;
  for (k = 0; k < count; ++k) {
    H[k * count + k] += fmax(trace * 1e-12 / count, DBL_MIN);
    feature[k] = 1.;
  }
  /* Freeze outward directions at multiplier bounds and re-solve the
   * remaining principal system. Never project during back-substitution. */
  for (;;) {
    size_t free_count = 0;
    int removed = 0;
    for (k = 0; k < count; ++k) {
      step[k] = 0.;
      if (feature[k] != 0.)
        center[free_count++] = (double)k;
    }
    if (!free_count)
      return 0;
    for (k = 0; k < free_count; ++k) {
      rhs[k] = -gradient[(size_t)center[k]];
      for (l = 0; l < free_count; ++l)
        factor[k * free_count + l] = H[(size_t)center[k] * count + (size_t)center[l]];
    }
    for (k = 0; k < free_count; ++k) {
      size_t pivot = k;
      for (l = k + 1; l < free_count; ++l)
        if (fabs(factor[l * free_count + k]) > fabs(factor[pivot * free_count + k]))
          pivot = l;
      if (!(fabs(factor[pivot * free_count + k]) > DBL_MIN))
        return 0;
      if (pivot != k) {
        double v = rhs[k];
        rhs[k] = rhs[pivot];
        rhs[pivot] = v;
        for (l = k; l < free_count; ++l) {
          v = factor[k * free_count + l];
          factor[k * free_count + l] = factor[pivot * free_count + l];
          factor[pivot * free_count + l] = v;
        }
      }
      for (a = k + 1; a < free_count; ++a) {
        double ratio = factor[a * free_count + k] / factor[k * free_count + k];
        for (l = k + 1; l < free_count; ++l)
          factor[a * free_count + l] -= ratio * factor[k * free_count + l];
        rhs[a] -= ratio * rhs[k];
      }
    }
    for (k = free_count; k-- > 0;) {
      for (l = k + 1; l < free_count; ++l)
        rhs[k] -= factor[k * free_count + l] * rhs[l];
      rhs[k] /= factor[k * free_count + k];
      if (!isfinite(rhs[k]))
        return 0;
      step[(size_t)center[k]] = rhs[k];
    }
    for (k = 0; k < count; ++k) {
      if ((old[k] == 0. && step[k] < 0.) || (old[k] == c->o.maximum_multiplier && step[k] > 0.)) {
        feature[k] = 0.;
        removed = 1;
      }
    }
    if (!removed)
      break;
  }
  for (k = 0; k < count; ++k) {
    if (step[k] < 0.)
      alpha = fmin(alpha, -old[k] / step[k]);
    if (step[k] > 0.)
      alpha = fmin(alpha, (c->o.maximum_multiplier - old[k]) / step[k]);
    direction -= gradient[k] * step[k];
  }
  if (!(direction > 0. && alpha > 0.))
    return 0;
  for (it = 0; it < 24; ++it) {
    double new_kkt = 0.;
    long double change = 0., scale = 0., noise;
    int converged;
    for (k = 0; k < count; ++k)
      set_multiplier(c, c->newton_index[k], fmin(c->o.maximum_multiplier, fmax(0., old[k] + alpha * step[k])));
    converged = master_certificate(c);
    if (c->error)
      return -1;
    for (k = 0; k < c->cuts + c->groups; ++k) {
      double slack = constraint_slack(c, k);
      new_kkt = fmax(new_kkt, multiplier(c, k) > 0. ? fabs(slack) : fmax(-slack, 0.));
    }
    /* Evaluate the dual increment directly. Subtracting rounded O(Q)
     * totals hides meaningful progress close to a nearly saturated face. */
    for (k = 0; k < count; ++k) {
      long double term = -((long double)multiplier(c, c->newton_index[k]) - old[k]) * gradient[k];
      change += term;
      scale += fabsl(term);
    }
    {
      long double term = (c->lambda - old_lambda) * old_equality;
      change += term;
      scale += fabsl(term);
    }
    for (a = 0; a < c->nactive; ++a) {
      long double yn, yo, dy, term;
      j = c->active[a];
      yn = c->y[j];
      yo = c->desired[j];
      dy = yn - yo;
      term = dy * ((yn + yo - 2.L * c->prior[j]) / (2.L * c->p->capacity[j]) + c->lambda - c->bias[j]) -
             (long double)c->nu[c->group[j]] * (fminl(yn, c->p->baseline[j]) - fminl(yo, c->p->baseline[j]));
      change += term;
      scale += fabsl(term);
    }
    noise = 128.L * DBL_EPSILON * scale + DBL_MIN;
    if (converged || (change >= 1e-4L * alpha * direction - noise &&
                      ((long double)alpha * direction > noise || new_kkt < old_kkt * (1. - 1e-4))))
      return 1;
    alpha *= .5;
  }
  for (k = 0; k < count; ++k)
    set_multiplier(c, c->newton_index[k], old[k]);
  (void)master_certificate(c);
  return c->error ? -1 : 0;
}

/* A rejected feasible-flow polish may expose only an already-known cut.
 * Tighten the same master and retry; public acceptance gates never change. */
static int tighten_master(shape_context* c)
{
  double floor = 32. * DBL_EPSILON * c->total;
  double tolerance = fmax(floor, .25 * c->tolerance);
  double feasibility = fmax(floor, .25 * c->master_feasibility);
  if (!(tolerance < c->tolerance || feasibility < c->master_feasibility)) {
    c->error = PC_SHAPE_ERR_NUMERICAL;
    return 0;
  }
  c->tolerance = tolerance;
  c->master_feasibility = feasibility;
  return 1;
}

static int solve_master(shape_context* c)
{
  size_t sweep, k;
  if (!waterfill(c))
    return 0;
  for (sweep = 0; sweep <= c->o.max_master_sweeps; ++sweep) {
    if (master_certificate(c))
      return 1;
    if (c->error)
      return 0;
    if (sweep == c->o.max_master_sweeps)
      break;
    ++c->m->master_sweeps;
    if (sweep > 0) {
      int newton = try_newton(c);
      if (newton < 0)
        return 0;
      if (newton > 0)
        continue;
    }
    for (k = 0; k < c->cuts + c->groups; ++k)
      if (!optimize_coordinate(c, k))
        return 0;
  }
  c->error = PC_SHAPE_ERR_MASTER_LIMIT;
  return 0;
}

/* The returned primal is the independently accumulated transport field. */
static int returned_certificate(shape_context* c)
{
  size_t j, k;
  long double objective = 0., total = 0., correction = 0.;
  double lower = c->m->evaluated_dual_lower;
  c->m->max_hall_violation = c->m->max_priority_violation = 0.;
  c->m->max_priority_equality_error = c->m->max_bound_violation = 0.;
  c->m->max_final_target_correction = 0.;
  memcpy(c->y, c->flow_y, c->n * sizeof(double));
  for (j = 0; j < c->n; ++j) {
    double d = c->y[j] - c->prior[j];
    if (c->p->capacity[j] > 0.)
      objective += .5L * d * d / c->p->capacity[j];
    total += c->y[j];
    correction += fabs(c->y[j] - c->desired[j]);
    c->m->max_final_target_correction = fmax(c->m->max_final_target_correction, fabs(c->y[j] - c->desired[j]));
    c->m->max_bound_violation =
      fmax(c->m->max_bound_violation, fmax(c->p->lower[j] - c->y[j], c->y[j] - c->p->upper[j]));
  }
  for (k = 0; k < c->cuts + c->groups; ++k) {
    double slack = constraint_slack(c, k);
    if (k < c->cuts)
      c->m->max_hall_violation = fmax(c->m->max_hall_violation, -slack);
    else {
      c->m->max_priority_violation = fmax(c->m->max_priority_violation, -slack);
      c->m->max_priority_equality_error = fmax(c->m->max_priority_equality_error, fabs(slack));
    }
  }
  c->m->objective = (double)objective;
  c->m->final_target_correction_l1 = (double)correction;
  c->m->signed_primal_dual_gap = (double)(objective - (long double)lower);
  c->m->equality_error = (double)fabsl(total - c->target_sum);
  return c->m->equality_error <= 64. * DBL_EPSILON * c->total && c->m->max_hall_violation <= c->m->tolerance &&
         c->m->max_priority_equality_error <= c->m->tolerance && c->m->max_bound_violation <= c->m->tolerance &&
         fabs(c->m->signed_primal_dual_gap) <= c->m->tolerance;
}

static void copy_allocation(const pc_problem* p, const pc_result* from, const pc_result* reference, pc_result* to)
{
  size_t count;
  memcpy(to->y, from->y, p->ncell * sizeof(double));
  memcpy(to->source_residual, from->source_residual, p->ncell * sizeof(double));
  if (to->source_spent) {
    size_t j;
    for (j = 0; j < p->ncell; ++j)
      to->source_spent[j] = from->source_spent ? from->source_spent[j] : p->source[j] - from->source_residual[j];
  }
  if (to->cut_source && from->cut_source)
    memcpy(to->cut_source, from->cut_source, p->ncell);
  if (to->cut_target && from->cut_target)
    memcpy(to->cut_target, from->cut_target, p->ncell);
  to->metrics = from->metrics;
  to->status = from->status;
  count = reference->metrics.priority_groups;
  if (count > to->priority_diagnostics_capacity)
    count = to->priority_diagnostics_capacity;
  if (to->priority_diagnostics && count)
    memcpy(to->priority_diagnostics, reference->priority_diagnostics, count * sizeof(pc_priority_diagnostic));
  to->metrics.priority_groups = reference->metrics.priority_groups;
}

pc_shape_status pc_shape_refine(const pc_problem* p,
                                const pc_options* flow_options,
                                const pc_result* reference,
                                const double* prior,
                                const pc_shape_options* options,
                                pc_result* result,
                                pc_shape_metrics* metrics)
{
  shape_context c;
  pc_shape_metrics local_metrics;
  pc_options fo;
  pc_problem fixed;
  pc_result allocation;
  size_t j, g;
  long double source_total = 0., low_total = 0., up_total = 0.;
  pc_shape_status status = PC_SHAPE_ERR_INVALID;
  memset(&c, 0, sizeof(c));
  if (!metrics)
    metrics = &local_metrics;
  memset(metrics, 0, sizeof(*metrics));
  metrics->status = status;
  if (!p || !reference || !result || result == reference || !result->y || !result->source_residual || !reference->y ||
      !reference->source_residual || (reference->status != PC_OK && reference->status != PC_PARTIAL) ||
      !reference->metrics.lower_feasible || !reference->metrics.full_graph_cut_separated || !p->ncell || !p->source ||
      !p->capacity || !p->baseline || !p->lower || !p->upper || !p->radius)
    return status;
  c.p = p;
  c.prior = prior ? prior : p->baseline;
  c.n = p->ncell;
  c.m = metrics;
  c.groups = reference->metrics.priority_groups;
  pc_shape_default_options(&c.o);
  if (options)
    c.o = *options;
  pc_default_options(&fo);
  if (flow_options)
    fo = *flow_options;
  if (!c.o.max_master_sweeps || !c.o.max_coordinate_iterations || !c.o.max_waterfill_iterations ||
      !c.o.max_evaluations || !isfinite(c.o.absolute_tolerance) || c.o.absolute_tolerance < 0. ||
      !isfinite(c.o.relative_tolerance) || c.o.relative_tolerance < 0. || !isfinite(c.o.maximum_multiplier) ||
      c.o.maximum_multiplier <= 0. || c.n > SIZE_MAX - 7)
    return status;
  for (j = 0; j < c.n; ++j) {
    if (!isfinite(p->source[j]) || p->source[j] < 0. || !isfinite(p->capacity[j]) || p->capacity[j] < 0. ||
        !isfinite(p->baseline[j]) || p->baseline[j] < 0. || !isfinite(c.prior[j]) || !isfinite(p->lower[j]) ||
        !isfinite(p->upper[j]) || p->lower[j] < 0. || p->upper[j] < p->lower[j] || p->upper[j] > p->capacity[j] ||
        (p->capacity[j] == 0. && (p->lower[j] != 0. || p->upper[j] != 0.)))
      return status;
    source_total += p->source[j];
    low_total += p->lower[j];
    up_total += p->upper[j];
  }
  c.source_sum = source_total;
  for (j = 0; j < c.n; ++j)
    c.target_sum += reference->y[j];
  c.photons = (double)source_total;
  c.total = (double)c.target_sum;
  c.tolerance = c.o.absolute_tolerance + c.o.relative_tolerance * fmax(1., c.photons);
  if (!isfinite(c.total) || !isfinite(c.photons) || c.total < 0. || c.total > c.photons + c.tolerance ||
      c.total < low_total - c.tolerance || c.total > up_total + c.tolerance)
    return status;
  metrics->source_budget = c.photons;
  metrics->target_total = c.total;
  metrics->unused_budget = fmax(0., c.photons - c.total);
  metrics->tolerance = c.tolerance;
  c.master_feasibility = 4. * c.tolerance;
  /* Solve the master more tightly than the transport acceptance threshold;
   * otherwise a known Hall cut can pass the master and fail the oracle. */
  c.tolerance = fmax(32. * DBL_EPSILON * c.total, .25 * c.tolerance);
  if (c.total > 0.)
    c.tolerance = fmin(c.tolerance, 1e-8 * c.total);
  if (c.total == 0.) {
    long double objective = 0.;
    /* Nonnegative target counts with zero total have the unique map y=0. */
    for (j = 0; j < c.n; ++j) {
      if (p->lower[j] != 0. || reference->y[j] != 0.)
        return status;
      if (p->capacity[j] > 0.)
        objective += .5L * c.prior[j] * c.prior[j] / p->capacity[j];
    }
    copy_allocation(p, reference, reference, result);
    metrics->objective = metrics->evaluated_dual_lower = (double)objective;
    metrics->last_oracle_status = reference->status;
    metrics->full_graph_certificate = 1;
    metrics->status = PC_SHAPE_OK;
    return PC_SHAPE_OK;
  }
  if (!c.groups || !reference->priority_diagnostics || reference->priority_diagnostics_capacity < c.groups)
    return status;
  c.bitbytes = (c.n + 7) / 8;
#define SHAPE_ALLOC(member, count, type)                                                                               \
  do {                                                                                                                 \
    c.member = shape_alloc(&c, count, sizeof(type));                                                                   \
    if (!c.member)                                                                                                     \
      goto done;                                                                                                       \
  } while (0)
  SHAPE_ALLOC(y, c.n, double);
  SHAPE_ALLOC(bias, c.n, double);
  SHAPE_ALLOC(flow_y, c.n, double);
  SHAPE_ALLOC(remainder, c.n, double);
  SHAPE_ALLOC(spent, c.n, double);
  SHAPE_ALLOC(active, c.n, size_t);
  SHAPE_ALLOC(master_upper, c.n, double);
  SHAPE_ALLOC(desired, c.n, double);
  for (j = 0; j < c.n; ++j) {
    c.master_upper[j] = p->upper[j];
    /* Every distinct periodic lattice cell is at least distance one away.
     * The 1e-10 geometric slack matches the allocator's sphere predicate. */
    if (p->radius[j] * p->radius[j] + 1e-10 < 1.)
      c.master_upper[j] = fmin(c.master_upper[j], p->source[j]);
    if (p->lower[j] > c.master_upper[j]) {
      c.error = PC_SHAPE_ERR_INVALID;
      goto done;
    }
    if (c.master_upper[j] > 0.)
      c.active[c.nactive++] = j;
  }
  SHAPE_ALLOC(group, c.n, int);
  SHAPE_ALLOC(cut_source, c.n, unsigned char);
  SHAPE_ALLOC(cut_target, c.n, unsigned char);
  SHAPE_ALLOC(quota, c.groups, double);
  SHAPE_ALLOC(nu, c.groups, double);
  SHAPE_ALLOC(rhs, c.o.max_hall_cuts, double);
  SHAPE_ALLOC(mu, c.o.max_hall_cuts, double);
  SHAPE_ALLOC(mask, c.o.max_hall_cuts, unsigned char*);
  if (c.groups <= 256) {
    c.newton_size = c.o.max_hall_cuts > 256 - c.groups ? 256 : c.groups + c.o.max_hall_cuts;
    SHAPE_ALLOC(newton_matrix, c.newton_size * c.newton_size, double);
    SHAPE_ALLOC(newton_factor, c.newton_size * c.newton_size, double);
    SHAPE_ALLOC(newton_work, 6 * c.newton_size, double);
    SHAPE_ALLOC(newton_index, c.newton_size, size_t);
  }
#undef SHAPE_ALLOC
  for (g = 0; g < c.groups; ++g) {
    c.quota[g] = reference->priority_diagnostics[g].baseline_retained_in_group;
    if (!isfinite(c.quota[g]) || c.quota[g] < 0.) {
      c.error = PC_SHAPE_ERR_INVALID;
      goto done;
    }
  }
  for (j = 0; j < c.n; ++j) {
    int rank = p->priority ? p->priority[j] : 0;
    for (g = 0; g < c.groups && reference->priority_diagnostics[g].priority != rank; ++g) {
    }
    if (g == c.groups && p->upper[j] == 0.)
      g = 0;
    if (g == c.groups) {
      c.error = PC_SHAPE_ERR_INVALID;
      goto done;
    }
    c.group[j] = (int)g;
  }
  /* Verify supplied quotas against the staged field before relying on them. */
  memcpy(c.y, reference->y, c.n * sizeof(double));
  for (g = 0; g < c.groups; ++g)
    if (fabs(constraint_slack(&c, g)) > c.tolerance) {
      c.error = PC_SHAPE_ERR_INVALID;
      goto done;
    }
  memset(&allocation, 0, sizeof(allocation));
  allocation.y = c.flow_y;
  allocation.source_residual = c.remainder;
  allocation.source_spent = c.spent;
  allocation.cut_source = c.cut_source;
  allocation.cut_target = c.cut_target;
  fixed = *p;
  fixed.lower = c.y;
  fixed.upper = c.y;
  for (;;) {
    pc_status oracle_status;
    double max_target_error = 0.;
    if (!solve_master(&c))
      goto done;
    ++metrics->oracle_calls;
    oracle_status = pc_allocate(&fixed, &fo, &allocation);
    metrics->last_oracle_status = oracle_status;
    if (oracle_status < 0 || !allocation.metrics.full_graph_cut_separated) {
      c.error = PC_SHAPE_ERR_ORACLE;
      goto done;
    }
    metrics->oracle_target_l1_error = 0.;
    for (j = 0; j < c.n; ++j) {
      double error = fabs(c.flow_y[j] - c.y[j]);
      max_target_error = fmax(max_target_error, error);
      metrics->oracle_target_l1_error += error;
    }
    metrics->max_oracle_target_error = max_target_error;
    {
      /* Every valid fixed-flow field supplies a candidate floor for a primal
       * repair. Its discrepancy is diagnostic, never an acceptance gate. */
      double max_group_error = 0.;
      for (g = 0; g < c.groups; ++g)
        max_group_error = fmax(max_group_error, fabs(constraint_slack(&c, c.cuts + g)));
      metrics->max_priority_equality_error = max_group_error;
      if (allocation.metrics.max_source_overspend > metrics->tolerance ||
          allocation.metrics.max_target_excess > metrics->tolerance ||
          allocation.metrics.max_edge_balance_error > metrics->tolerance) {
        c.error = PC_SHAPE_ERR_NUMERICAL;
        goto done;
      }
      /* Restore any rounded-away source inventory on the SAME original
       * graph and upper bounds. Increasing target marginals cannot reduce
       * retained overlap. The final objective and optimal-face constraints
       * are checked on the actual transported field below. */
      memcpy(c.desired, c.y, c.n * sizeof(double));
      for (j = 0; j < c.n; ++j)
        c.y[j] = clip(c.flow_y[j], p->lower[j], p->upper[j]);
      fixed.upper = p->upper;
      ++metrics->oracle_calls;
      oracle_status = pc_allocate(&fixed, &fo, &allocation);
      metrics->last_oracle_status = oracle_status;
      if (oracle_status < 0 || !allocation.metrics.full_graph_cut_separated) {
        c.error = PC_SHAPE_ERR_ORACLE;
        goto done;
      }
      if (!allocation.metrics.lower_feasible ||
          (reference->metrics.full_source_budget && !allocation.metrics.full_source_budget) ||
          !returned_certificate(&c)) {
        /* A nearby feasible flow need not preserve the priority face to the
         * requested tolerance. Reject that repair and continue separation. */
        memcpy(c.y, c.desired, c.n * sizeof(double));
        fixed.upper = c.y;
        ++metrics->oracle_calls;
        oracle_status = pc_allocate(&fixed, &fo, &allocation);
        metrics->last_oracle_status = oracle_status;
        if (oracle_status < 0 || !allocation.metrics.full_graph_cut_separated) {
          c.error = PC_SHAPE_ERR_ORACLE;
          goto done;
        }
        goto separate_cut;
      }
      /* A caller's edge visitor sees only the final checked allocation. */
      if (result->visit_edge) {
        /* Re-accumulated flow may exceed a bound by one rounding unit.
         * Clamp only the next proposal; never alter the certified ledger. */
        for (j = 0; j < c.n; ++j)
          c.y[j] = clip(c.y[j], p->lower[j], p->upper[j]);
        allocation.visit_edge = result->visit_edge;
        allocation.visitor_user = result->visitor_user;
        ++metrics->oracle_calls;
        oracle_status = pc_allocate(&fixed, &fo, &allocation);
        if (oracle_status < 0 || !allocation.metrics.lower_feasible || !allocation.metrics.full_graph_cut_separated ||
            (reference->metrics.full_source_budget && !allocation.metrics.full_source_budget) ||
            !returned_certificate(&c)) {
          c.error = PC_SHAPE_ERR_ORACLE;
          goto done;
        }
      }
      copy_allocation(p, &allocation, reference, result);
      metrics->full_graph_certificate = 1;
      c.error = PC_SHAPE_OK;
      goto done;
    }
    {
    separate_cut:;
      long double budget = 0., demand = 0.;
      double rhs;
      size_t k, duplicate = SIZE_MAX;
      for (j = 0; j < c.n; ++j) {
        if (c.cut_source[j])
          budget += p->source[j];
        if (c.cut_target[j])
          demand += c.y[j];
      }
      rhs = (double)(budget - (c.source_sum - c.target_sum));
      if ((long double)rhs - demand <= c.tolerance * .1) {
        if (!tighten_master(&c))
          goto done;
        continue;
      }
      for (k = 0; k < c.cuts; ++k) {
        int same = 1;
        for (j = 0; j < c.n; ++j)
          if (bit_get(c.mask[k], j) != (c.cut_target[j] != 0)) {
            same = 0;
            break;
          }
        if (same) {
          duplicate = k;
          break;
        }
      }
      if (duplicate != SIZE_MAX) {
        if (rhs <= c.rhs[duplicate] + c.tolerance * .1) {
          if (!tighten_master(&c))
            goto done;
          continue;
        }
        c.rhs[duplicate] = rhs;
      } else {
        if (c.cuts == c.o.max_hall_cuts) {
          c.error = PC_SHAPE_ERR_CUT_LIMIT;
          goto done;
        }
        c.mask[c.cuts] = shape_alloc(&c, c.bitbytes, 1);
        if (!c.mask[c.cuts])
          goto done;
        for (j = 0; j < c.n; ++j)
          if (c.cut_target[j])
            c.mask[c.cuts][j >> 3] |= (unsigned char)(1u << (j & 7));
        c.rhs[c.cuts] = rhs;
        ++c.cuts;
      }
      metrics->hall_cuts = c.cuts;
    }
  }
  c.error = PC_SHAPE_ERR_CUT_LIMIT;
done:
  status = c.error;
  if (c.mask)
    for (j = 0; j < c.cuts; ++j)
      free(c.mask[j]);
  free(c.y);
  free(c.bias);
  free(c.flow_y);
  free(c.remainder);
  free(c.spent);
  free(c.active);
  free(c.master_upper);
  free(c.desired);
  free(c.group);
  free(c.cut_source);
  free(c.cut_target);
  free(c.quota);
  free(c.nu);
  free(c.rhs);
  free(c.mu);
  free(c.mask);
  free(c.newton_matrix);
  free(c.newton_factor);
  free(c.newton_work);
  free(c.newton_index);
  metrics->status = status;
  return status;
}
