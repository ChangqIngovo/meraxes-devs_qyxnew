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

/* Reduced one-coordinate curvature after eliminating the total multiplier.
 * Accumulate the two binary-feature classes separately to avoid cancellation
 * for a Hall mask covering almost all of the free target capacity. */
static double coordinate_curvature(const shape_context* c, size_t k)
{
  size_t a;
  long double w0 = 0., w1 = 0.;
  for (a = 0; a < c->nactive; ++a) {
    size_t j = c->active[a];
    double y = c->y[j];
    int feature;
    if (!(y > c->p->lower[j] && y < c->master_upper[j]) || (y == c->p->baseline[j] && c->nu[c->group[j]] > 0.))
      continue;
    feature = k < c->cuts ? bit_get(c->mask[k], j) : ((size_t)c->group[j] == k - c->cuts && y < c->p->baseline[j]);
    if (feature)
      w1 += c->p->capacity[j];
    else
      w0 += c->p->capacity[j];
  }
  return w0 > 0. && w1 > 0. ? (double)(w0 * (w1 / (w0 + w1))) : 0.;
}

/* Warm-start the monotone coordinate solve. Newton and secant steps are
 * proposals only: actual waterfilled slacks maintain the sign bracket. */
static int optimize_coordinate(shape_context* c, size_t k)
{
  double theta = multiplier(c, k), slack = constraint_slack(c, k);
  double lo = 0., hi = c->o.maximum_multiplier, slo = 0., shi = 0.;
  int have_lo = 0, have_hi = 0;
  size_t it;
  for (it = 0; it < c->o.max_coordinate_iterations; ++it) {
    double slope, next;
    if (fabs(slack) <= c->tolerance * .1 || (theta == 0. && slack >= -c->tolerance * .1))
      return 1;
    if (slack < 0.) {
      lo = theta;
      slo = slack;
      have_lo = 1;
      if (theta == c->o.maximum_multiplier) {
        c->error = PC_SHAPE_ERR_MASTER_LIMIT;
        return 0;
      }
    } else {
      hi = theta;
      shi = slack;
      have_hi = 1;
    }
    slope = coordinate_curvature(c, k);
    next = slope > 0. ? theta - slack / slope : NAN;
    if (have_lo && have_hi) {
      double margin = .01 * (hi - lo);
      if (!(next > lo + margin && next < hi - margin))
        next = lo - slo * (hi - lo) / (shi - slo);
      /* Interleave bisection so a flat slack cannot stall false position. */
      if ((it & 1) || !(next > lo + margin && next < hi - margin))
        next = lo + .5 * (hi - lo);
    } else if (slack > 0.) {
      if (!(next >= 0. && next < theta))
        next = 0.;
    } else {
      if (!(next > theta) || !isfinite(next))
        next = fmax(1., 2. * theta);
      /* Before a bracket exists, avoid a speculative huge Newton jump
       * across a nearly flat segment: retain bounded directional growth. */
      next = fmin(c->o.maximum_multiplier, fmin(fmax(1., 2. * theta), next));
    }
    if (next == theta) {
      if (fabs(slack) <= c->tolerance)
        return 1;
      c->error = PC_SHAPE_ERR_NUMERICAL;
      return 0;
    }
    set_multiplier(c, k, next);
    if (!waterfill(c))
      return 0;
    theta = next;
    slack = constraint_slack(c, k);
  }
  if (fabs(slack) <= c->tolerance * .1 || (theta == 0. && slack >= -c->tolerance * .1))
    return 1;
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
    c->lambda = old_lambda;
    converged = master_certificate(c);
    if (c->error == PC_SHAPE_ERR_NUMERICAL) {
      /* A speculative Newton step may lose scalar resolution far from the
       * current iterate. Reject that trial; retain the previous valid state. */
      c->error = PC_SHAPE_OK;
      alpha *= .5;
      continue;
    }
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
  c->lambda = old_lambda;
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

typedef struct
{
  double *coefficient, *nu;
  long double cut_constant, cut_multiplier, lambda;
} shape_dual;

static int shape_components(shape_context* c, const pc_options* fo, const pc_result* reference, pc_result* result);

static pc_shape_status pc_shape_refine_core(const pc_problem* p,
                                            const pc_options* flow_options,
                                            const pc_result* reference,
                                            const double* prior,
                                            const pc_shape_options* options,
                                            pc_result* result,
                                            pc_shape_metrics* metrics,
                                            int split_components,
                                            shape_dual* dual_export)
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
  if (split_components && shape_components(&c, &fo, reference, result))
    return metrics->status;
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
  if (status == PC_SHAPE_OK && dual_export) {
    size_t k;
    dual_export->lambda = c.lambda;
    dual_export->cut_constant = dual_export->cut_multiplier = 0.;
    for (j = 0; j < c.n; ++j)
      dual_export->coefficient[j] = (double)(c.lambda - c.bias[j]);
    memcpy(dual_export->nu, c.nu, c.groups * sizeof(double));
    for (k = 0; k < c.cuts; ++k) {
      dual_export->cut_constant += (long double)c.mu[k] * c.rhs[k];
      dual_export->cut_multiplier += c.mu[k];
    }
  }
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

static size_t component_root(size_t* parent, size_t k)
{
  size_t root = k;
  while (parent[root] != root)
    root = parent[root];
  while (parent[k] != k) {
    size_t next = parent[k];
    parent[k] = root;
    k = next;
  }
  return root;
}

static void component_join(size_t* parent, size_t a, size_t b)
{
  a = component_root(parent, a);
  b = component_root(parent, b);
  if (a != b)
    parent[b] = a;
}

static void component_position(const pc_problem* p, size_t cell, size_t pos[3])
{
  pos[2] = cell % p->dim[2];
  cell /= p->dim[2];
  pos[1] = cell % p->dim[1];
  pos[0] = cell / p->dim[1];
}

static double component_distance(const pc_problem* p, const size_t a[3], const size_t b[3])
{
  size_t k;
  double d2 = 0.;
  for (k = 0; k < 3; ++k) {
    size_t d = a[k] > b[k] ? a[k] - b[k] : b[k] - a[k];
    if (d > p->dim[k] - d)
      d = p->dim[k] - d;
    d2 += (double)d * d;
  }
  return d2;
}

/* Re-evaluate the actual returned field, including a visitor replay, against
 * the original global face and the lifted global dual lower bound. */
static int component_certificate(shape_context* c,
                                 const pc_result* reference,
                                 const pc_result* result,
                                 const double* proposal,
                                 pc_priority_diagnostic* quota,
                                 long double dual,
                                 pc_shape_metrics* m)
{
  const pc_problem* p = c->p;
  size_t j, k;
  long double objective = 0., total = 0.;
  m->max_bound_violation = m->max_priority_violation = m->max_priority_equality_error = 0.;
  m->max_final_target_correction = m->final_target_correction_l1 = 0.;
  for (k = 0; k < c->groups; ++k)
    quota[k].baseline_retained_in_group = 0.;
  for (j = 0; j < c->n; ++j) {
    long double d = (long double)result->y[j] - c->prior[j];
    double correction = fabs(result->y[j] - proposal[j]);
    int rank = p->priority ? p->priority[j] : 0;
    total += result->y[j];
    if (p->capacity[j] > 0.)
      objective += .5L * d * d / p->capacity[j];
    m->max_bound_violation = fmax(m->max_bound_violation, fmax(p->lower[j] - result->y[j], result->y[j] - p->upper[j]));
    m->max_final_target_correction = fmax(m->max_final_target_correction, correction);
    m->final_target_correction_l1 += correction;
    for (k = 0; k < c->groups && quota[k].priority != rank; ++k) {
    }
    if (k < c->groups)
      quota[k].baseline_retained_in_group += fmin(result->y[j], p->baseline[j]);
  }
  for (k = 0; k < c->groups; ++k) {
    double slack = quota[k].baseline_retained_in_group - reference->priority_diagnostics[k].baseline_retained_in_group;
    m->max_priority_violation = fmax(m->max_priority_violation, -slack);
    m->max_priority_equality_error = fmax(m->max_priority_equality_error, fabs(slack));
  }
  m->objective = (double)objective;
  m->evaluated_dual_lower = (double)dual;
  m->signed_primal_dual_gap = (double)(objective - dual);
  m->equality_error = (double)fabsl(total - c->target_sum);
  return isfinite(m->objective) && isfinite(m->evaluated_dual_lower) && isfinite(m->signed_primal_dual_gap) &&
         m->equality_error <= 64. * DBL_EPSILON * c->total && m->max_hall_violation <= m->tolerance &&
         m->max_priority_equality_error <= m->tolerance && m->max_bound_violation <= m->tolerance &&
         fabsl(objective - dual) <= m->tolerance;
}

/* The full transport graph, rather than the currently materialized flow
 * edges, determines independent components. A target-centred lattice scan
 * is used when cheaper than scanning the occupied sources. No adjacency
 * matrix or source-target edge list is stored. */
static int shape_components(shape_context* c, const pc_options* fo, const pc_result* reference, pc_result* result)
{
  const pc_problem* p = c->p;
  size_t n = c->n, j, k, q, ns = 0, nc = 0, bytes, peak = 0;
  size_t *parent = NULL, *sources = NULL, *roots = NULL, *counts = NULL, *single = NULL;
  double *arrays = NULL, *source, *baseline, *lower, *upper, *prior, *ref_y, *ref_residual, *out_y, *out_residual;
  double *proposal, *coefficient, *comp_lambda, *comp_budget, *shared_nu = NULL, *local_nu = NULL;
  pc_priority_diagnostic* quota = NULL;
  pc_shape_metrics sum;
  pc_shape_status status = PC_SHAPE_OK;
  int handled = 0, fallback = 0;
  long double dual = 0., cut_constant = 0., global_lambda = -LDBL_MAX;
  if (!p->dim[0] || !p->dim[1] || !p->dim[2] || p->dim[0] > SIZE_MAX / p->dim[1] ||
      p->dim[0] * p->dim[1] > SIZE_MAX / p->dim[2] || p->dim[0] * p->dim[1] * p->dim[2] != n ||
      n > SIZE_MAX / (6 * sizeof(size_t) + 13 * sizeof(double)))
    return 0;
  bytes = n * (6 * sizeof(size_t) + 13 * sizeof(double));
  if (c->groups > (SIZE_MAX - bytes) / (sizeof(*quota) + 2 * sizeof(double)))
    return 0;
  bytes += c->groups * (sizeof(*quota) + 2 * sizeof(double));
  /* This optimization is optional; do not consume the whole solver budget
   * merely to discover that the graph is connected. */
  if (c->o.max_bytes && bytes > c->o.max_bytes / 2)
    return 0;
  parent = malloc(2 * n * sizeof(*parent));
  sources = malloc(n * sizeof(*sources));
  roots = malloc(n * sizeof(*roots));
  counts = calloc(n, sizeof(*counts));
  single = malloc(n * sizeof(*single));
  if (!parent || !sources || !roots || !counts || !single)
    goto cleanup;
  for (j = 0; j < 2 * n; ++j)
    parent[j] = j;
  for (j = 0; j < n; ++j)
    if (p->source[j] > 0.)
      sources[ns++] = j;
  for (j = 0; j < n; ++j) {
    size_t pos[3], half, widths[3], starts[3], volume = 1;
    double r2;
    if (!(p->upper[j] > 0.))
      continue;
    if (!isfinite(p->radius[j]) || p->radius[j] < 0.)
      goto cleanup;
    component_position(p, j, pos);
    r2 = p->radius[j] * p->radius[j] + 1e-10;
    half = p->radius[j] >= (double)n ? n : (size_t)floor(sqrt(r2));
    for (k = 0; k < 3; ++k) {
      if (half >= p->dim[k] / 2) {
        widths[k] = p->dim[k];
        starts[k] = 0;
      } else {
        widths[k] = 2 * half + 1;
        starts[k] = pos[k] >= half ? pos[k] - half : p->dim[k] - (half - pos[k]);
      }
      if (volume > ns / widths[k])
        volume = ns + 1;
      else
        volume *= widths[k];
    }
    if (volume < ns) {
      size_t x, y, z;
      for (x = 0; x < widths[0]; ++x)
        for (y = 0; y < widths[1]; ++y)
          for (z = 0; z < widths[2]; ++z) {
            size_t sp[3] = { (starts[0] + x) % p->dim[0], (starts[1] + y) % p->dim[1], (starts[2] + z) % p->dim[2] };
            size_t i = (sp[0] * p->dim[1] + sp[1]) * p->dim[2] + sp[2];
            if (p->source[i] > 0. && component_distance(p, pos, sp) <= r2)
              component_join(parent, i, n + j);
          }
    } else {
      for (q = 0; q < ns; ++q) {
        size_t sp[3], i = sources[q];
        component_position(p, i, sp);
        if (component_distance(p, pos, sp) <= r2)
          component_join(parent, i, n + j);
      }
    }
  }
  /* Compress all labels before reusing counts as the component lookup. */
  for (j = 0; j < 2 * n; ++j)
    parent[j] = component_root(parent, j);
  for (j = 0; j < n; ++j) {
    size_t root;
    if (!(p->upper[j] > 0.))
      continue;
    root = parent[n + j];
    /* A target with no eligible source is fixed at zero and contributes
     * only a constant to the original quadratic objective. */
    if (root >= n)
      continue;
    if (!counts[root])
      roots[nc++] = root;
    ++counts[root];
    single[root] = j;
  }
  if (nc < 2)
    goto cleanup;
  arrays = calloc(13 * n, sizeof(double));
  quota = calloc(c->groups, sizeof(*quota));
  shared_nu = calloc(c->groups, sizeof(double));
  local_nu = calloc(c->groups, sizeof(double));
  if (!arrays || !quota || !shared_nu || !local_nu)
    goto cleanup;
  for (k = 0; k < c->groups; ++k)
    quota[k] = reference->priority_diagnostics[k];
  source = arrays;
  baseline = source + n;
  lower = baseline + n;
  upper = lower + n;
  prior = upper + n;
  ref_y = prior + n;
  ref_residual = ref_y + n;
  out_y = ref_residual + n;
  out_residual = out_y + n;
  proposal = out_residual + n;
  coefficient = proposal + n;
  comp_lambda = coefficient + n;
  comp_budget = comp_lambda + n;
  for (j = 0; j < n; ++j)
    if (p->source[j] > 0.)
      comp_budget[parent[j]] += p->source[j];
  memset(&sum, 0, sizeof(sum));
  sum.source_budget = c->m->source_budget;
  sum.target_total = c->m->target_total;
  sum.unused_budget = c->m->unused_budget;
  sum.tolerance = c->m->tolerance;
  peak = bytes;
  handled = 1;
  for (q = 0; q < nc; ++q) {
    size_t root = roots[q], targets = counts[root];
    long double budget = 0., reference_total = 0.;
    pc_problem sub = *p;
    pc_result ref = *reference, out;
    pc_shape_options options = c->o;
    pc_shape_metrics m;
    shape_dual local_dual;
    /* Avoid a full-grid clear or scan for the common isolated pair. */
    if (targets == 1) {
      j = single[root];
      proposal[j] = reference->y[j];
      continue;
    }
    memset(arrays, 0, 9 * n * sizeof(double));
    for (k = 0; k < c->groups; ++k) {
      quota[k] = reference->priority_diagnostics[k];
      quota[k].baseline_retained_in_group = 0.;
    }
    for (j = 0; j < n; ++j) {
      if (p->source[j] > 0. && parent[j] == root) {
        source[j] = p->source[j];
        ref_residual[j] = reference->source_residual[j];
        budget += source[j];
      }
      if (p->upper[j] > 0. && parent[n + j] == root) {
        int rank = p->priority ? p->priority[j] : 0;
        baseline[j] = p->baseline[j];
        lower[j] = p->lower[j];
        upper[j] = p->upper[j];
        prior[j] = c->prior[j];
        ref_y[j] = reference->y[j];
        reference_total += ref_y[j];
        for (k = 0; k < c->groups && quota[k].priority != rank; ++k) {
        }
        if (k == c->groups) {
          status = PC_SHAPE_ERR_INVALID;
          goto cleanup;
        }
        quota[k].baseline_retained_in_group += fmin(ref_y[j], baseline[j]);
      }
    }
    comp_budget[roots[q]] = (double)budget;
    /* A numerically empty local maximum has no dual state to export.
     * Let the global solver handle its exact rounded target-total face. */
    if (!(reference_total > 0.)) {
      fallback = 1;
      goto cleanup;
    }
    if (sum.master_evaluations >= options.max_evaluations || sum.hall_cuts > options.max_hall_cuts) {
      status = sum.master_evaluations >= options.max_evaluations ? PC_SHAPE_ERR_MASTER_LIMIT : PC_SHAPE_ERR_CUT_LIMIT;
      goto cleanup;
    }
    options.max_evaluations -= sum.master_evaluations;
    options.max_hall_cuts -= sum.hall_cuts;
    if (options.max_bytes)
      options.max_bytes -= bytes;
    /* Allocate the same global error budget across components. In
     * particular, do not apply max(1,S_component) independently. */
    options.absolute_tolerance = 0.;
    options.relative_tolerance = sum.tolerance * (double)(budget / c->source_sum) / fmax(1., (double)budget);
    sub.source = source;
    sub.baseline = baseline;
    sub.lower = lower;
    sub.upper = upper;
    ref.y = ref_y;
    ref.source_residual = ref_residual;
    ref.priority_diagnostics = quota;
    ref.priority_diagnostics_capacity = c->groups;
    memset(&out, 0, sizeof(out));
    out.y = out_y;
    out.source_residual = out_residual;
    local_dual.coefficient = out_residual;
    local_dual.nu = local_nu;
    status = pc_shape_refine_core(&sub, fo, &ref, prior, &options, &out, &m, 0, &local_dual);
    sum.hall_cuts += m.hall_cuts;
    sum.master_sweeps += m.master_sweeps;
    sum.master_evaluations += m.master_evaluations;
    sum.oracle_calls += m.oracle_calls;
    if (bytes + m.bytes_peak > peak)
      peak = bytes + m.bytes_peak;
    sum.max_hall_violation = fmax(sum.max_hall_violation, m.max_hall_violation);
    sum.max_complementarity = fmax(sum.max_complementarity, m.max_complementarity);
    sum.max_oracle_target_error = fmax(sum.max_oracle_target_error, m.max_oracle_target_error);
    sum.oracle_target_l1_error += m.oracle_target_l1_error;
    if (status != PC_SHAPE_OK) {
      fallback = 1;
      goto cleanup;
    }
    comp_lambda[roots[q]] = (double)local_dual.lambda;
    global_lambda = fmaxl(global_lambda, comp_lambda[roots[q]]);
    {
      long double local_total = 0.;
      for (j = 0; j < n; ++j)
        local_total += ref_y[j];
      cut_constant += local_dual.cut_constant -
                      local_dual.cut_multiplier * ((c->source_sum - c->target_sum) - (budget - local_total));
    }
    for (k = 0; k < c->groups; ++k)
      shared_nu[k] = fmax(shared_nu[k], local_nu[k]);
    for (j = 0; j < n; ++j)
      if (upper[j] > 0.) {
        proposal[j] = out_y[j];
        coefficient[j] = local_dual.coefficient[j];
      }
  }
  /* Lift every local Hall inequality back to the original global
   * transport polytope. A common priority multiplier is essential: rounded
   * per-component quotas must not silently define a tighter QP.
   * With delta=P-F, a local cut S(Q)-(P_c-F_c) becomes S(Q)-delta.
   * Taking lambda=max(lambda_c) and adding each valid component Hall cut
   * sum(y_c)>=P_c-delta with multiplier lambda-lambda_c leaves the local
   * linear coefficients unchanged. The scalar infima below, using common
   * nonnegative nu_g, therefore give a lower bound for the ORIGINAL QP,
   * even if rounded local quotas do not factor its feasible priority face. */
  for (q = 0; q < nc; ++q)
    if (counts[roots[q]] == 1) {
      int rank;
      j = single[roots[q]];
      rank = p->priority ? p->priority[j] : 0;
      for (k = 0; k < c->groups && reference->priority_diagnostics[k].priority != rank; ++k) {
      }
      if (k == c->groups) {
        status = PC_SHAPE_ERR_INVALID;
        goto cleanup;
      }
      comp_lambda[roots[q]] =
        (c->prior[j] - proposal[j]) / p->capacity[j] + (proposal[j] < p->baseline[j] ? shared_nu[k] : 0.);
      coefficient[j] = comp_lambda[roots[q]];
      global_lambda = fmaxl(global_lambda, comp_lambda[roots[q]]);
    }
  dual = cut_constant - global_lambda * c->target_sum;
  for (q = 0; q < nc; ++q)
    dual +=
      (global_lambda - comp_lambda[roots[q]]) * ((long double)comp_budget[roots[q]] - (c->source_sum - c->target_sum));
  for (k = 0; k < c->groups; ++k)
    dual += (long double)shared_nu[k] * reference->priority_diagnostics[k].baseline_retained_in_group;
  for (j = 0; j < n; ++j) {
    if (p->capacity[j] > 0.) {
      int rank = p->priority ? p->priority[j] : 0;
      long double nu = 0., a, value, low_root, high_root, d;
      for (k = 0; k < c->groups && reference->priority_diagnostics[k].priority != rank; ++k) {
      }
      if (k < c->groups)
        nu = shared_nu[k];
      a = p->upper[j] > 0. && parent[n + j] < n ? coefficient[j] : global_lambda;
      low_root = (long double)c->prior[j] - (long double)p->capacity[j] * a;
      high_root = low_root + (long double)p->capacity[j] * nu;
      value = low_root > p->baseline[j] ? low_root : (high_root < p->baseline[j] ? high_root : p->baseline[j]);
      value = fmaxl(p->lower[j], fminl(p->upper[j], value));
      d = value - c->prior[j];
      dual += .5L * d * d / p->capacity[j] + a * value - nu * fminl(value, p->baseline[j]);
    }
    lower[j] = clip(proposal[j], p->lower[j], p->upper[j]);
  }
  {
    pc_problem fixed = *p;
    fixed.lower = lower;
    ++sum.oracle_calls;
    {
      pc_edge_visitor visitor = result->visit_edge;
      result->visit_edge = NULL;
      sum.last_oracle_status = pc_allocate(&fixed, fo, result);
      result->visit_edge = visitor;
    }
    if (sum.last_oracle_status < 0 || !result->metrics.lower_feasible || !result->metrics.full_graph_cut_separated ||
        (reference->metrics.full_source_budget && !result->metrics.full_source_budget)) {
      status = PC_SHAPE_ERR_ORACLE;
      goto cleanup;
    }
    if (!component_certificate(c, reference, result, proposal, quota, dual, &sum)) {
      /* The lifted dual can be loose when independent components prefer
       * very different priority multipliers. Retain the original global
       * solve as the exact fallback, using only the remaining work budget. */
      fallback = 1;
      goto cleanup;
    }
    if (result->visit_edge) {
      ++sum.oracle_calls;
      sum.last_oracle_status = pc_allocate(&fixed, fo, result);
      if (sum.last_oracle_status < 0 || !result->metrics.lower_feasible || !result->metrics.full_graph_cut_separated ||
          (reference->metrics.full_source_budget && !result->metrics.full_source_budget) ||
          !component_certificate(c, reference, result, proposal, quota, dual, &sum)) {
        status = PC_SHAPE_ERR_ORACLE;
        goto cleanup;
      }
    }
    sum.full_graph_certificate = 1;
    if (result->priority_diagnostics) {
      size_t count =
        c->groups < result->priority_diagnostics_capacity ? c->groups : result->priority_diagnostics_capacity;
      memcpy(result->priority_diagnostics, reference->priority_diagnostics, count * sizeof(*quota));
    }
    result->metrics.priority_groups = c->groups;
  }
cleanup:
  free(parent);
  free(sources);
  free(roots);
  free(counts);
  free(single);
  free(arrays);
  free(quota);
  free(shared_nu);
  free(local_nu);
  if (fallback) {
    pc_shape_options options = c->o;
    pc_shape_metrics m;
    if (sum.master_evaluations >= options.max_evaluations) {
      status = PC_SHAPE_ERR_MASTER_LIMIT;
    } else if (sum.hall_cuts > options.max_hall_cuts) {
      status = PC_SHAPE_ERR_CUT_LIMIT;
    } else {
      options.max_evaluations -= sum.master_evaluations;
      options.max_hall_cuts -= sum.hall_cuts;
      status = pc_shape_refine_core(p, fo, reference, c->prior, &options, result, &m, 0, NULL);
      m.master_evaluations += sum.master_evaluations;
      m.master_sweeps += sum.master_sweeps;
      m.hall_cuts += sum.hall_cuts;
      m.oracle_calls += sum.oracle_calls;
      m.bytes_peak = m.bytes_peak > peak ? m.bytes_peak : peak;
      *c->m = m;
      return 1;
    }
  }
  if (handled) {
    sum.bytes_peak = peak;
    sum.status = status;
    *c->m = sum;
  }
  return handled;
}

pc_shape_status pc_shape_refine(const pc_problem* p,
                                const pc_options* flow_options,
                                const pc_result* reference,
                                const double* prior,
                                const pc_shape_options* options,
                                pc_result* result,
                                pc_shape_metrics* metrics)
{
  return pc_shape_refine_core(p, flow_options, reference, prior, options, result, metrics, 1, NULL);
}
