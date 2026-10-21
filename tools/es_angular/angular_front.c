#include "angular_front.h"
#include "angular_faces.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int af_abi_version(void)
{
  return 1;
}

typedef struct {
  size_t cell;
  int next[6];
  double p[6], geometric_in;
} node;

typedef struct {
  node* nodes;
  size_t count;
  int injection;
  double q;
} resource;

struct af_solver {
  int dims[3], ns;
  size_t cells, nb, nr, max_nodes;
  double dx, epsilon, eta, density;
  int external_score;
  int* sources;
  double *q, *history, *cap, *u, *source_acc, *source_rate, *total_rate, *w;
  unsigned char* paid;
  resource* resources;
  ag_bin* bins;
  double *flow, *absorb, *res_abs_rate, *res_escape_rate, *res_abs, *res_escape;
  double *sort_r, *sort_q;
  size_t* queue;
  unsigned char* seen;
  af_diagnostics d;
  char error[256];
};

static int fail(af_solver* s, const char* message)
{
  snprintf(s->error, sizeof(s->error), "%s", message);
  return -1;
}

static void* alloc_array(size_t n, size_t z)
{
  if (z && n > SIZE_MAX / z)
    return NULL;
  return calloc(n ? n : 1, z);
}

static size_t cell_id(const af_solver* s, int x, int y, int z)
{
  return ((size_t)x * s->dims[1] + y) * s->dims[2] + z;
}

static void cell_xyz(const af_solver* s, size_t j, int p[3])
{
  p[2] = (int)(j % s->dims[2]);
  j /= s->dims[2];
  p[1] = (int)(j % s->dims[1]);
  p[0] = (int)(j / s->dims[1]);
}

static size_t face_id(const af_solver* s, int axis, const int p[3], int plus)
{
  int x = p[0], y = p[1], z = p[2];
  if (axis == 0)
    x += plus;
  else if (axis == 1)
    y += plus;
  else
    z += plus;
  if (axis == 0)
    return ((size_t)x * s->dims[1] + y) * s->dims[2] + z;
  if (axis == 1)
    return ((size_t)x * (s->dims[1] + 1) + y) * s->dims[2] + z;
  return ((size_t)x * s->dims[1] + y) * (s->dims[2] + 1) + z;
}

static int is_outgoing(int offset, int sign)
{
  return offset * sign >= 0;
}

static int source_graphs(af_solver* s, int si)
{
  const int* src = s->sources + 3 * si;
  size_t nf[3] = {
    (size_t)(s->dims[0] + 1) * s->dims[1] * s->dims[2],
    (size_t)s->dims[0] * (s->dims[1] + 1) * s->dims[2],
    (size_t)s->dims[0] * s->dims[1] * (s->dims[2] + 1)
  };
  double* faces[3] = { NULL, NULL, NULL };
  int* map = NULL;
  size_t *order = NULL, *bucket = NULL;
  int status = -1, maxd = s->dims[0] + s->dims[1] + s->dims[2];

  for (int k = 0; k < 3; k++) {
    faces[k] = alloc_array(nf[k], sizeof(double));
    if (!faces[k])
      goto cleanup;
  }
  map = alloc_array(s->cells, sizeof(int));
  order = alloc_array(s->cells, sizeof(size_t));
  bucket = alloc_array((size_t)maxd + 2, sizeof(size_t));
  if (!map || !order || !bucket)
    goto cleanup;

  for (size_t j = 0; j < s->cells; j++) {
    int p[3];
    cell_xyz(s, j, p);
    int d = abs(p[0] - src[0]) + abs(p[1] - src[1]) + abs(p[2] - src[2]);
    bucket[d + 1]++;
  }
  for (int k = 1; k <= maxd + 1; k++)
    bucket[k] += bucket[k - 1];
  for (size_t j = 0; j < s->cells; j++) {
    int p[3];
    cell_xyz(s, j, p);
    int d = abs(p[0] - src[0]) + abs(p[1] - src[1]) + abs(p[2] - src[2]);
    order[bucket[d]++] = j;
  }

  const double pi = acos(-1.0);
  for (size_t g = 0; g < s->nb; g++) {
    resource* r = s->resources + (size_t)si * s->nb + g;
    r->q = s->q[si] * (s->bins[g].solid_angle / (4 * pi));
    for (int axis = 0; axis < 3; axis++) {
      int lim[3] = { s->dims[0], s->dims[1], s->dims[2] };
      lim[axis]++;
      for (int x = 0; x < lim[0]; x++)
        for (int y = 0; y < lim[1]; y++)
          for (int z = 0; z < lim[2]; z++) {
            int p[3] = { x, y, z };
            double c[3];
            /* Solid angles are scale invariant: exact half-integer grid coordinates
             * avoid introducing rounded, non-collinear contacts through dx scaling. */
            for (int k = 0; k < 3; k++)
              c[k] = p[k] - src[k] - (k == axis ? .5 : 0);
            double omega = ag_face_bin_solid_angle(c, axis, .5, s->bins + g);
            if (!isfinite(omega) || omega < 0) {
              fail(s, "invalid angular face solid angle");
              goto cleanup;
            }
            faces[axis][face_id(s, axis, p, 0)] = omega / (4 * pi);
          }
    }
    for (size_t j = 0; j < s->cells; j++)
      map[j] = -1;

    size_t count = 0;
    for (size_t k = 0; k < s->cells; k++) {
      size_t j = order[k];
      int p[3];
      cell_xyz(s, j, p);
      double out = 0, in = 0;
      for (int a = 0; a < 3; a++)
        for (int side = 0; side < 2; side++) {
          double phi = faces[a][face_id(s, a, p, side)];
          if (is_outgoing(p[a] - src[a], side ? 1 : -1))
            out += phi;
          else
            in += phi;
        }
      double expected =
        (j == cell_id(s, src[0], src[1], src[2])) ? s->bins[g].solid_angle / (4 * pi) : 0;
      double residual = fabs(out - in - expected);
      if (residual > s->d.max_face_balance_absolute)
        s->d.max_face_balance_absolute = residual;
      if (out > 0) {
        if (count > INT_MAX) {
          fail(s, "angular graph index overflow");
          goto cleanup;
        }
        map[j] = (int)count++;
      } else if (in > 1e-13) {
        fail(s, "zero angular outgoing flux with positive incoming geometry");
        goto cleanup;
      }
    }

    r->nodes = alloc_array(count, sizeof(node));
    if (!r->nodes)
      goto cleanup;
    r->count = count;
    r->injection = map[cell_id(s, src[0], src[1], src[2])];
    if (r->injection < 0) {
      fail(s, "source angular bin has no outgoing node");
      goto cleanup;
    }
    for (size_t k = 0; k < s->cells; k++) {
      size_t j = order[k];
      if (map[j] < 0)
        continue;
      node* n = r->nodes + map[j];
      int p[3];
      cell_xyz(s, j, p);
      n->cell = j;
      double out = 0;
      for (int a = 0; a < 3; a++)
        for (int side = 0; side < 2; side++)
          if (is_outgoing(p[a] - src[a], side ? 1 : -1))
            out += faces[a][face_id(s, a, p, side)];
      n->geometric_in = out;
      for (int a = 0; a < 3; a++)
        for (int side = 0; side < 2; side++) {
          int e = 2 * a + side;
          n->next[e] = -2;
          n->p[e] = 0;
          if (!is_outgoing(p[a] - src[a], side ? 1 : -1))
            continue;
          double phi = faces[a][face_id(s, a, p, side)];
          if (phi == 0)
            continue;
          n->p[e] = phi / out;
          int other[3] = { p[0], p[1], p[2] };
          other[a] += side ? 1 : -1;
          if (other[a] < 0 || other[a] >= s->dims[a])
            n->next[e] = -1;
          else {
            n->next[e] = map[cell_id(s, other[0], other[1], other[2])];
            if (n->next[e] < 0) {
              snprintf(s->error, sizeof(s->error),
                       "positive shared face has no angular target node: src=%d bin=%zu cell=(%d,%d,%d) axis=%d side=%d phi=%.17g",
                       si, g, p[0], p[1], p[2], a, side, phi);
              goto cleanup;
            }
          }
        }
    }
    if (count > s->max_nodes)
      s->max_nodes = count;
    s->d.graph_nodes += count;
  }
  status = 0;

cleanup:
  for (int k = 0; k < 3; k++)
    free(faces[k]);
  free(map);
  free(order);
  free(bucket);
  if (status && !s->error[0])
    fail(s, "angular graph allocation failed");
  return status;
}

static int connected(af_solver* s)
{
  size_t begin = cell_id(s, s->sources[0], s->sources[1], s->sources[2]);
  if (!s->paid[begin])
    return 0;
  if (s->ns < 2)
    return 1;
  memset(s->seen, 0, s->cells);
  size_t a = 0, b = 0;
  s->queue[b++] = begin;
  s->seen[begin] = 1;
  while (a < b) {
    size_t j = s->queue[a++];
    int p[3];
    cell_xyz(s, j, p);
    for (int k = 0; k < 3; k++)
      for (int sign = -1; sign <= 1; sign += 2) {
        int v[3] = { p[0], p[1], p[2] };
        v[k] += sign;
        if (v[k] < 0 || v[k] >= s->dims[k])
          continue;
        size_t z = cell_id(s, v[0], v[1], v[2]);
        if (s->paid[z] && !s->seen[z]) {
          s->seen[z] = 1;
          s->queue[b++] = z;
        }
      }
  }
  for (int i = 1; i < s->ns; i++)
    if (!s->seen[cell_id(s, s->sources[3 * i], s->sources[3 * i + 1], s->sources[3 * i + 2])])
      return 0;
  return 1;
}

static double es_weight(const af_solver* s, double score)
{
  /* Algebraically 1+eta*(score-1)/(score+1), without cancellation at score=0. */
  double lower = 2 / (2 + s->epsilon);
  return lower + 2 * s->eta * (isinf(score) ? 1 : score / (1 + score));
}

static void update_es(af_solver* s)
{
  if (s->external_score || s->epsilon == 0 || s->density == 0)
    return;
  const double prefactor = 4 * acos(-1.) * s->density / 3;
  for (size_t j = 0; j < s->cells; j++) {
    if (s->paid[j]) {
      s->w[j] = 1;
      continue;
    }
    int p[3];
    cell_xyz(s, j, p);
    for (int i = 0; i < s->ns; i++) {
      double r2 = 0;
      for (int a = 0; a < 3; a++) {
        double d = (p[a] - s->sources[3 * i + a]) * s->dx;
        r2 += d * d;
      }
      s->sort_r[i] = r2;
      s->sort_q[i] = s->history[i] + s->q[i] * s->d.time;
      for (int k = i; k > 0 && s->sort_r[k] < s->sort_r[k - 1]; k--) {
        double x = s->sort_r[k];
        s->sort_r[k] = s->sort_r[k - 1];
        s->sort_r[k - 1] = x;
        x = s->sort_q[k];
        s->sort_q[k] = s->sort_q[k - 1];
        s->sort_q[k - 1] = x;
      }
    }
    double budget = 0, score = 0;
    for (int i = 0; i < s->ns; i++) {
      budget += s->sort_q[i];
      if (i + 1 < s->ns && s->sort_r[i + 1] == s->sort_r[i])
        continue;
      if (s->sort_r[i] == 0) {
        if (budget > 0) {
          score = INFINITY;
          break;
        }
        continue;
      }
      double z = budget / (prefactor * s->sort_r[i] * sqrt(s->sort_r[i]));
      if (z > score)
        score = z;
    }
    s->w[j] = es_weight(s, score);
  }
}

static int resource_sweep(af_solver* s, size_t ri, double* escape, double* absorbed, int vacuum)
{
  resource* r = s->resources + ri;
  memset(s->flow, 0, r->count * sizeof(double));
  memset(s->absorb, 0, r->count * sizeof(double));
  s->flow[r->injection] = r->q;
  *escape = 0;
  *absorbed = 0;
  for (size_t k = 0; k < r->count; k++) {
    node* n = r->nodes + k;
    double f = s->flow[k];
    if (vacuum) {
      double expected = s->q[ri / s->nb] * n->geometric_in;
      double err = fabs(f - expected);
      if (err > s->d.max_vacuum_flux_absolute)
        s->d.max_vacuum_flux_absolute = err;
    }
    if (!vacuum && !s->paid[n->cell]) {
      s->absorb[k] = f;
      *absorbed += f;
      continue;
    }
    for (int e = 0; e < 6; e++)
      if (n->next[e] >= 0)
        s->flow[n->next[e]] += f * n->p[e];
      else if (n->next[e] == -1)
        *escape += f * n->p[e];
  }
  if (!vacuum && s->epsilon > 0 && *absorbed > 0) {
    double weighted = 0;
    for (size_t k = 0; k < r->count; k++)
      weighted += (s->absorb[k] / *absorbed) * s->w[r->nodes[k].cell];
    if (!(weighted > 0))
      return fail(s, "invalid ES angular normalization");
    double factor = 1 / weighted;
    for (size_t k = 0; k < r->count; k++)
      if (s->absorb[k] > 0) {
        double ratio = factor * s->w[r->nodes[k].cell];
        s->absorb[k] *= ratio;
        if (ratio < s->d.min_bias_ratio)
          s->d.min_bias_ratio = ratio;
        if (ratio > s->d.max_bias_ratio)
          s->d.max_bias_ratio = ratio;
      }
    *absorbed = 0;
    for (size_t k = 0; k < r->count; k++)
      *absorbed += s->absorb[k];
  }
  return 0;
}

static int sweep(af_solver* s)
{
  update_es(s);
  memset(s->source_rate, 0, (size_t)s->ns * s->cells * sizeof(double));
  memset(s->total_rate, 0, s->cells * sizeof(double));
  for (size_t ri = 0; ri < s->nr; ri++) {
    double escaped, absorbed;
    if (resource_sweep(s, ri, &escaped, &absorbed, 0))
      return -1;
    if (!isfinite(absorbed) || !isfinite(escaped))
      return fail(s, "resource rate overflow");
    s->res_abs_rate[ri] = absorbed;
    s->res_escape_rate[ri] = escaped;
    resource* r = s->resources + ri;
    double err = fabs(absorbed + escaped - r->q);
    if (err > s->d.max_resource_rate_error)
      s->d.max_resource_rate_error = err;
    size_t base = (ri / s->nb) * s->cells;
    for (size_t k = 0; k < r->count; k++) {
      size_t j = r->nodes[k].cell;
      s->source_rate[base + j] += s->absorb[k];
      s->total_rate[j] += s->absorb[k];
      if (!isfinite(s->total_rate[j]) || !isfinite(s->source_rate[base + j]))
        return fail(s, "cell absorption rate overflow");
    }
  }
  return 0;
}

af_solver* af_create(int nx, int ny, int nz, double dx, unsigned level, int ns,
                     const int* src, const double* q, const double* cap, const double* initial,
                     double epsilon, const double* history, double density, char* error, size_t errsize)
{
  af_solver* s = calloc(1, sizeof(*s));
  if (!s) {
    if (error && errsize)
      snprintf(error, errsize, "allocation failed");
    return NULL;
  }
  s->dims[0] = nx;
  s->dims[1] = ny;
  s->dims[2] = nz;
  s->ns = ns;
  s->dx = dx;
  s->epsilon = epsilon;
  s->eta = epsilon / (2 + epsilon);
  s->density = density;
  s->d.first_connection_time = -1;
  s->d.min_bias_ratio = s->d.max_bias_ratio = 1;
  if (nx <= 0 || ny <= 0 || nz <= 0 || ns <= 0 || ns > INT_MAX / 3 || !src || !q || !cap || !initial ||
      !isfinite(dx) || dx <= 0 || !isfinite(epsilon) || epsilon < 0 || !isfinite(density) || density < 0) {
    fail(s, "invalid angular solver input");
    goto bad;
  }
  if ((size_t)nx > SIZE_MAX / (size_t)ny / (size_t)nz) {
    fail(s, "grid size overflow");
    goto bad;
  }
  s->cells = (size_t)nx * ny * nz;
  if (s->cells > SIZE_MAX / 4) {
    fail(s, "face grid size overflow");
    goto bad;
  }
  if (nx == INT_MAX || ny == INT_MAX || nz == INT_MAX || (long long)nx + ny + nz > INT_MAX - 2 ||
      (size_t)ns > SIZE_MAX / s->cells || (size_t)ns > SIZE_MAX / 3) {
    fail(s, "grid or source index overflow");
    goto bad;
  }
  s->nb = ag_bin_count(level);
  if (!s->nb || s->nb > SIZE_MAX / (size_t)ns) {
    fail(s, "invalid angular level");
    goto bad;
  }
  s->nr = (size_t)ns * s->nb;

#define ALLOC(field, count, type)                         \
  do {                                                    \
    s->field = alloc_array((count), sizeof(type));          \
    if (!s->field) {                                       \
      fail(s, "allocation failed");                        \
      goto bad;                                           \
    }                                                     \
  } while (0)

  ALLOC(sources, 3 * (size_t)ns, int);
  ALLOC(q, ns, double);
  ALLOC(history, ns, double);
  ALLOC(cap, s->cells, double);
  ALLOC(u, s->cells, double);
  ALLOC(paid, s->cells, unsigned char);
  ALLOC(w, s->cells, double);
  ALLOC(source_acc, (size_t)ns * s->cells, double);
  ALLOC(source_rate, (size_t)ns * s->cells, double);
  ALLOC(total_rate, s->cells, double);
  ALLOC(resources, s->nr, resource);
  ALLOC(bins, s->nb, ag_bin);
  ALLOC(res_abs_rate, s->nr, double);
  ALLOC(res_escape_rate, s->nr, double);
  ALLOC(res_abs, s->nr, double);
  ALLOC(res_escape, s->nr, double);
  ALLOC(sort_r, ns, double);
  ALLOC(sort_q, ns, double);
  ALLOC(queue, s->cells, size_t);
  ALLOC(seen, s->cells, unsigned char);

  memcpy(s->sources, src, 3 * (size_t)ns * sizeof(int));
  memcpy(s->q, q, (size_t)ns * sizeof(double));
  memcpy(s->cap, cap, s->cells * sizeof(double));
  memcpy(s->u, initial, s->cells * sizeof(double));
  if (history)
    memcpy(s->history, history, (size_t)ns * sizeof(double));
  for (int i = 0; i < ns; i++) {
    if (!isfinite(q[i]) || q[i] < 0 || !isfinite(s->history[i]) || s->history[i] < 0) {
      fail(s, "invalid source rate or history");
      goto bad;
    }
    for (int a = 0; a < 3; a++)
      if (src[3 * i + a] < 0 || src[3 * i + a] >= s->dims[a]) {
        fail(s, "source index outside grid");
        goto bad;
      }
  }
  for (size_t j = 0; j < s->cells; j++) {
    if (!isfinite(cap[j]) || cap[j] < 0 || !isfinite(initial[j]) || initial[j] < 0 || initial[j] > cap[j]) {
      fail(s, "invalid atom capacity or ionized mass");
      goto bad;
    }
    s->paid[j] = initial[j] == cap[j];
    s->w[j] = 1;
  }
  if (ag_make_bins(level, s->bins, s->nb)) {
    fail(s, "angular bin construction failed");
    goto bad;
  }
  for (int i = 0; i < ns; i++)
    if (source_graphs(s, i))
      goto bad;
  ALLOC(flow, s->max_nodes, double);
  ALLOC(absorb, s->max_nodes, double);
#undef ALLOC

  s->d.graph_bytes = s->d.graph_nodes * sizeof(node) + s->nr * sizeof(resource);
  for (size_t ri = 0; ri < s->nr; ri++) {
    double e, a;
    if (resource_sweep(s, ri, &e, &a, 1))
      goto bad;
  }
  if (connected(s))
    s->d.first_connection_time = 0;
  if (sweep(s))
    goto bad;
  return s;

bad:
  if (error && errsize)
    snprintf(error, errsize, "%s", s->error);
  af_destroy(s);
  return NULL;
}

void af_destroy(af_solver* s)
{
  if (!s)
    return;
  if (s->resources)
    for (size_t i = 0; i < s->nr; i++)
      free(s->resources[i].nodes);
  free(s->sources);
  free(s->q);
  free(s->history);
  free(s->cap);
  free(s->u);
  free(s->paid);
  free(s->source_acc);
  free(s->source_rate);
  free(s->total_rate);
  free(s->w);
  free(s->resources);
  free(s->bins);
  free(s->flow);
  free(s->absorb);
  free(s->res_abs_rate);
  free(s->res_escape_rate);
  free(s->res_abs);
  free(s->res_escape);
  free(s->sort_r);
  free(s->sort_q);
  free(s->queue);
  free(s->seen);
  free(s);
}

int af_set_es_scores(af_solver* s, const double* score)
{
  if (!s || !score)
    return -1;
  for (size_t j = 0; j < s->cells; j++)
    if (!isfinite(score[j]) || score[j] < 0)
      return fail(s, "external ES score must be finite and nonnegative");
  for (size_t j = 0; j < s->cells; j++)
    s->w[j] = es_weight(s, score[j]);
  s->external_score = 1;
  return sweep(s);
}

int af_advance(af_solver* s, double target, size_t max_events)
{
  if (!s || !isfinite(target) || target < s->d.time)
    return s ? fail(s, "invalid target time") : -1;
  long double emitted = 0;
  for (int i = 0; i < s->ns; i++)
    emitted += (long double)s->q[i] * target;
  if (emitted > DBL_MAX)
    return fail(s, "requested photon budget exceeds double precision range");
  size_t start_events = s->d.events;
  while (s->d.time < target) {
    if (s->d.events - start_events >= max_events)
      return 1;
    double dt_event = INFINITY;
    size_t winner = SIZE_MAX;
    for (size_t j = 0; j < s->cells; j++)
      if (!s->paid[j] && s->total_rate[j] > 0) {
        double dt = (s->cap[j] - s->u[j]) / s->total_rate[j];
        if (dt < dt_event) {
          dt_event = dt;
          winner = j;
        }
      }
    double left = target - s->d.time;
    int event = dt_event <= left;
    double dt = event ? dt_event : left;
    if (!(dt > 0))
      return fail(s, "nonpositive event step");
    for (size_t j = 0; j < s->cells; j++)
      if (!s->paid[j]) {
        s->u[j] += dt * s->total_rate[j];
        for (int i = 0; i < s->ns; i++)
          s->source_acc[(size_t)i * s->cells + j] += dt * s->source_rate[(size_t)i * s->cells + j];
        if (event && (j == winner || s->u[j] >= s->cap[j])) {
          double correction = s->cap[j] - s->u[j];
          s->d.clamp_signed_mass += correction;
          s->d.clamp_absolute_mass += fabs(correction);
          s->u[j] = s->cap[j];
          s->paid[j] = 1;
        } else if (!event && s->u[j] >= s->cap[j]) {
          /* A diagnostic time cut may round up to capacity while still preceding
           * the physical event. Keep it opaque and retain frozen ES/routing rates.
           * Explicitly record this at-most-one-ulp storage correction. */
          double below = nextafter(s->cap[j], 0.0), correction = below - s->u[j];
          s->d.clamp_signed_mass += correction;
          s->d.clamp_absolute_mass += fabs(correction);
          s->u[j] = below;
        }
      }
    for (size_t ri = 0; ri < s->nr; ri++) {
      s->res_abs[ri] += dt * s->res_abs_rate[ri];
      s->res_escape[ri] += dt * s->res_escape_rate[ri];
    }
    s->d.time = event ? s->d.time + dt : target;
    s->d.steps++;
    if (event) {
      s->d.events++;
      if (s->d.first_connection_time < 0 && connected(s))
        s->d.first_connection_time = s->d.time;
      if (sweep(s))
        return -1;
    }
  }
  return 0;
}

size_t af_cell_count(const af_solver* s)
{
  return s ? s->cells : 0;
}

size_t af_bin_count(const af_solver* s)
{
  return s ? s->nb : 0;
}

int af_source_count(const af_solver* s)
{
  return s ? s->ns : 0;
}

const char* af_error(const af_solver* s)
{
  return s ? s->error : "null solver";
}

void af_get_diagnostics(const af_solver* s, af_diagnostics* out)
{
  if (s && out)
    *out = s->d;
}

void af_copy_mass(const af_solver* s, double* out)
{
  if (!s || !out)
    return;
  memcpy(out, s->u, s->cells * sizeof(double));
}

void af_copy_source_absorption(const af_solver* s, double* out)
{
  if (!s || !out)
    return;
  memcpy(out, s->source_acc, (size_t)s->ns * s->cells * sizeof(double));
}

void af_copy_source_rates(const af_solver* s, double* out)
{
  if (!s || !out)
    return;
  memcpy(out, s->source_rate, (size_t)s->ns * s->cells * sizeof(double));
}

void af_copy_resource_ledgers(const af_solver* s, double* a, double* e, double* q)
{
  if (!s)
    return;
  for (size_t k = 0; k < s->nr; k++) {
    if (a)
      a[k] = s->res_abs[k];
    if (e)
      e[k] = s->res_escape[k];
    if (q)
      q[k] = s->resources[k].q;
  }
}

int af_resource_rate(af_solver* s, size_t ri, double* out, double* escape)
{
  if (!s || ri >= s->nr || !out || !escape)
    return -1;
  double a;
  if (resource_sweep(s, ri, escape, &a, 0))
    return -1;
  memset(out, 0, s->cells * sizeof(double));
  resource* r = s->resources + ri;
  for (size_t k = 0; k < r->count; k++)
    out[r->nodes[k].cell] = s->absorb[k];
  return 0;
}
