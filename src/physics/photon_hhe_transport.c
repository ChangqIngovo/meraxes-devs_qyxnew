#include "photon_hhe_transport.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HHE_PI 3.141592653589793238462643383279502884

static int fail(hhe_transport* t, int code, const char* message)
{
  if (t)
    snprintf(t->error, sizeof(t->error), "%s", message);
  return code;
}

static int valid_mesh(const hhe_mesh* m)
{
  int a;
  if (!m)
    return 0;
  for (a = 0; a < 3; ++a)
    if (m->dimensions[a] < 1 || !isfinite(m->box_cm[a]) || m->box_cm[a] <= 0)
      return 0;
  if (m->slab_x_start < 0 || m->slab_x_count < 0 || m->slab_x_count > m->dimensions[0] ||
      m->slab_x_start > m->dimensions[0] - m->slab_x_count || (m->periodic != 0 && m->periodic != 1))
    return 0;
  return 1;
}

size_t hhe_mesh_local_cells(const hhe_mesh* m)
{
  size_t n;
  if (!valid_mesh(m))
    return 0;
  n = (size_t)m->slab_x_count;
  if ((size_t)m->dimensions[1] > SIZE_MAX / (n ? n : 1))
    return 0;
  n *= (size_t)m->dimensions[1];
  if ((size_t)m->dimensions[2] > SIZE_MAX / (n ? n : 1))
    return 0;
  return n * (size_t)m->dimensions[2];
}

double hhe_mesh_cell_volume(const hhe_mesh* m)
{
  double v = 1;
  int a;
  if (!valid_mesh(m))
    return NAN;
  for (a = 0; a < 3; ++a)
    v *= m->box_cm[a] / m->dimensions[a];
  return v;
}

double hhe_transport_cfl_step(const hhe_mesh* m, double cfl)
{
  double dx = DBL_MAX;
  int a;
  if (!valid_mesh(m) || !isfinite(cfl) || cfl <= 0 || cfl > 1)
    return NAN;
  for (a = 0; a < 3; ++a)
    dx = fmin(dx, m->box_cm[a] / m->dimensions[a]);
  return cfl * dx / HHE_C_LIGHT;
}

void hhe_transport_init(hhe_transport* t, uint64_t seed, size_t bytes)
{
  if (!t)
    return;
  memset(t, 0, sizeof(*t));
  t->seed = seed;
  t->memory_limit_bytes = bytes;
  t->error_cell = SIZE_MAX;
}

void hhe_transport_free(hhe_transport* t)
{
  if (!t)
    return;
  free(t->packets);
  free(t->pending_emission);
  free(t->cell_recombinations);
  memset(t, 0, sizeof(*t));
}

static int memory_available(const hhe_transport* t,
                            size_t packet_capacity,
                            size_t pending_cells,
                            size_t temporary_bytes)
{
  size_t n, p;
  if (packet_capacity > SIZE_MAX / sizeof(hhe_packet) ||
      pending_cells > SIZE_MAX / ((HHE_NGROUPS + 3) * sizeof(double)))
    return 0;
  n = packet_capacity * sizeof(hhe_packet);
  p = pending_cells * (HHE_NGROUPS + 3) * sizeof(double);
  if (n > SIZE_MAX - p || n + p > SIZE_MAX - temporary_bytes)
    return 0;
  n += p + temporary_bytes;
  return !t->memory_limit_bytes || n <= t->memory_limit_bytes;
}

static int reserve(hhe_transport* t, size_t count)
{
  size_t capacity;
  hhe_packet* p;
  if (count <= t->capacity)
    return HHE_TRANSPORT_OK;
  capacity = t->capacity ? t->capacity : 64;
  while (capacity < count) {
    if (capacity > SIZE_MAX / 2) {
      capacity = count;
      break;
    }
    capacity *= 2;
  }
  if (!memory_available(t, capacity, t->pending_cells, 0))
    capacity = count;
  if (!memory_available(t, capacity, t->pending_cells, 0))
    return fail(t, HHE_TRANSPORT_MEMORY, "H/He packet memory exhausted; no photons discarded");
  p = (hhe_packet*)realloc(t->packets, capacity * sizeof(hhe_packet));
  if (!p)
    return fail(t, HHE_TRANSPORT_MEMORY, "H/He packet allocation failed; no photons discarded");
  t->packets = p;
  t->capacity = capacity;
  return HHE_TRANSPORT_OK;
}

int hhe_transport_append(hhe_transport* t, const hhe_packet* packets, size_t count)
{
  size_t i;
  int status;
  if (!t || (count && !packets) || count > SIZE_MAX - t->size)
    return fail(t, HHE_TRANSPORT_INPUT, "Invalid incoming H/He packet array");
  if ((status = reserve(t, t->size + count)))
    return status;
  if (count)
    memcpy(t->packets + t->size, packets, count * sizeof(*packets));
  for (i = 0; i < count; ++i)
    t->packets[t->size + i].location = HHE_PACKET_LOCAL;
  t->size += count;
  if (t->size > t->ledger.peak_packets)
    t->ledger.peak_packets = (uint64_t)t->size;
  return HHE_TRANSPORT_OK;
}

void hhe_transport_compact(hhe_transport* t)
{
  size_t i, n = 0;
  if (!t)
    return;
  for (i = 0; i < t->size; ++i)
    if (t->packets[i].location == HHE_PACKET_LOCAL && t->packets[i].count != 0)
      t->packets[n++] = t->packets[i];
  t->size = n;
}

static uint64_t mix64(uint64_t z)
{
  z += UINT64_C(0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static double uniform(hhe_packet* p)
{
  uint64_t r;
  p->rng_state ^= p->rng_state >> 12;
  p->rng_state ^= p->rng_state << 25;
  p->rng_state ^= p->rng_state >> 27;
  r = p->rng_state * UINT64_C(2685821657736338717);
  return ((double)(r >> 11) + 0.5) * 0x1.0p-53;
}

static size_t local_cell(const hhe_mesh* m, const int cell[3])
{
  return ((size_t)(cell[0] - m->slab_x_start) * m->dimensions[1] + (size_t)cell[1]) * m->dimensions[2] +
         (size_t)cell[2];
}

static int cell_owned(const hhe_mesh* m, const int cell[3])
{
  return cell[0] >= m->slab_x_start && cell[0] < m->slab_x_start + m->slab_x_count;
}

static int cell_valid(const hhe_mesh* m, const int cell[3])
{
  int a;
  for (a = 0; a < 3; ++a)
    if (cell[a] < 0 || cell[a] >= m->dimensions[a])
      return 0;
  return 1;
}

static void cell_coordinates(const hhe_mesh* m, size_t index, int cell[3])
{
  cell[2] = (int)(index % (size_t)m->dimensions[2]);
  index /= (size_t)m->dimensions[2];
  cell[1] = (int)(index % (size_t)m->dimensions[1]);
  cell[0] = (int)(index / (size_t)m->dimensions[1]) + m->slab_x_start;
}

static int emit(hhe_transport* t,
                const hhe_mesh* m,
                const double* counts,
                double multiplier,
                double birth,
                unsigned int ndir,
                int origin)
{
  size_t c, ncell = hhe_mesh_local_cells(m), new_count = 0;
  unsigned int a;
  int g, axis, status;
  uint64_t time_bits;
  if (!t || !valid_mesh(m) || (ncell && !counts) || !isfinite(multiplier) || multiplier < 0 || !isfinite(birth) ||
      ndir == 0)
    return fail(t, HHE_TRANSPORT_INPUT, "Invalid H/He photon emission input");
  for (c = 0; c < ncell; ++c)
    for (g = 0; g < HHE_NGROUPS; ++g) {
      double rate = counts[c * HHE_NGROUPS + g];
      double n = rate * multiplier;
      if (!isfinite(rate) || rate < 0 || !isfinite(n) || n < 0)
        return fail(t, HHE_TRANSPORT_INPUT, "Nonfinite or negative emitted photons");
      if (n > 0) {
        if (n / ndir == 0)
          return fail(t, HHE_TRANSPORT_INPUT, "Angular photon weight underflows floating-point range");
        if (new_count > SIZE_MAX - ndir)
          return fail(t, HHE_TRANSPORT_MEMORY, "H/He emission packet count overflow");
        new_count += ndir;
      }
    }
  if (new_count > SIZE_MAX - t->size)
    return fail(t, HHE_TRANSPORT_MEMORY, "H/He packet size overflow");
  if ((status = reserve(t, t->size + new_count)))
    return status;
  memcpy(&time_bits, &birth, sizeof(time_bits));
  for (c = 0; c < ncell; ++c) {
    int cell[3];
    uint64_t id;
    cell_coordinates(m, c, cell);
    id = ((uint64_t)cell[0] * (uint64_t)m->dimensions[1] + (uint64_t)cell[1]) * (uint64_t)m->dimensions[2] +
         (uint64_t)cell[2];
    for (g = 0; g < HHE_NGROUPS; ++g) {
      double n = counts[c * HHE_NGROUPS + g] * multiplier;
      if (n == 0)
        continue;
      t->ledger.emitted[origin][g] += n;
      for (a = 0; a < ndir; ++a) {
        hhe_packet* p = &t->packets[t->size++];
        double z = 1 - 2 * ((double)a + 0.5) / ndir;
        double phi = a * HHE_PI * (3 - sqrt(5.0));
        double r = sqrt(fmax(0, 1 - z * z));
        memset(p, 0, sizeof(*p));
        for (axis = 0; axis < 3; ++axis) {
          p->cell[axis] = cell[axis];
          p->position_cm[axis] = ((double)cell[axis] + 0.5) * m->box_cm[axis] / m->dimensions[axis];
        }
        p->direction[0] = r * cos(phi);
        p->direction[1] = r * sin(phi);
        p->direction[2] = z;
        p->count = n / ndir;
        p->energy_eV = hhe_group_energy_eV[g];
        p->birth_s = p->clock_s = birth;
        p->emitter_id = id;
        p->group = g;
        p->origin = origin;
        p->rng_state =
          mix64(t->seed ^ mix64(id) ^ mix64(time_bits) ^ mix64((uint64_t)g * ndir + a) ^ mix64((uint64_t)origin + 1));
        if (!p->rng_state)
          p->rng_state = 1;
        p->scatter_tau = -log(uniform(p));
      }
    }
  }
  t->ledger.created_packets += (uint64_t)new_count;
  if (t->size > t->ledger.peak_packets)
    t->ledger.peak_packets = (uint64_t)t->size;
  return HHE_TRANSPORT_OK;
}

int hhe_transport_emit_sources(hhe_transport* t,
                               const hhe_mesh* m,
                               const double* rates,
                               double dt,
                               double birth,
                               unsigned int directions)
{
  return emit(t, m, rates, dt, birth, directions, 0);
}

int hhe_transport_begin_drift(hhe_transport* t, double dt)
{
  size_t i;
  if (!t || !isfinite(dt) || dt < 0)
    return fail(t, HHE_TRANSPORT_INPUT, "Invalid H/He drift interval");
  for (i = 0; i < t->size; ++i)
    if (t->packets[i].location != HHE_PACKET_LOCAL || t->packets[i].remaining_s != 0)
      return fail(t, HHE_TRANSPORT_PHASE, "Unfinished H/He packet exchange before new drift");
  for (i = 0; i < t->size; ++i)
    t->packets[i].remaining_s = dt;
  return HHE_TRANSPORT_OK;
}

static int packet_valid(const hhe_packet* p, const hhe_mesh* m)
{
  int a;
  double norm = 0;
  if (p->group < 0 || p->group >= HHE_NGROUPS || p->origin < 0 || p->origin > 1 || !isfinite(p->count) ||
      p->count < 0 || !isfinite(p->remaining_s) || p->remaining_s < 0 || !isfinite(p->birth_s) ||
      !isfinite(p->clock_s) || !isfinite(p->scatter_tau) || p->scatter_tau < 0 ||
      p->energy_eV != hhe_group_energy_eV[p->group] || !cell_valid(m, p->cell))
    return 0;
  for (a = 0; a < 3; ++a) {
    double low = p->cell[a] * (m->box_cm[a] / m->dimensions[a]);
    double high = (p->cell[a] + 1.) * (m->box_cm[a] / m->dimensions[a]);
    double eps = 32 * DBL_EPSILON * m->box_cm[a];
    if (!isfinite(p->position_cm[a]) || !isfinite(p->direction[a]) || p->position_cm[a] < low - eps ||
        p->position_cm[a] > high + eps)
      return 0;
    norm += p->direction[a] * p->direction[a];
  }
  return fabs(norm - 1) <= 1.e-10;
}

static int drift_packet(hhe_transport* t, const hhe_mesh* m, const double* opacity, hhe_packet* p)
{
  long double left = (long double)HHE_C_LIGHT * p->remaining_s;
  const double original_remaining = p->remaining_s;
  int a, kind = hhe_group_kind[p->group];
  while (left > 0 && p->location == HHE_PACKET_LOCAL) {
    double bound = INFINITY, face_dist[3], kappa = 0, scatter, step;
    int crossed = 0;
    if (!cell_owned(m, p->cell)) {
      p->location = HHE_PACKET_ROUTE;
      break;
    }
    for (a = 0; a < 3; ++a) {
      if (p->direction[a] == 0)
        face_dist[a] = INFINITY;
      else {
        int face = p->cell[a] + (p->direction[a] > 0);
        double coordinate = face * (m->box_cm[a] / m->dimensions[a]);
        face_dist[a] = fmax(0, (coordinate - p->position_cm[a]) / p->direction[a]);
      }
      bound = fmin(bound, face_dist[a]);
    }
    if (kind)
      kappa = opacity[2 * local_cell(m, p->cell) + kind - 1];
    scatter = kappa > 0 ? p->scatter_tau / kappa : INFINITY;
    step = fmin((double)left, fmin(bound, scatter));
    if (!isfinite(step) || step < 0)
      return fail(t, HHE_TRANSPORT_GEOMETRY, "Nonfinite H/He photon flight distance");
    if (step == 0 && bound > 0 && scatter > 0)
      return fail(t, HHE_TRANSPORT_GEOMETRY, "H/He remaining flight below floating-point precision");
    for (a = 0; a < 3; ++a)
      p->position_cm[a] += step * p->direction[a];
    if ((long double)step >= left)
      left = 0;
    else {
      long double next = left - step;
      if (step > 0 && next == left)
        return fail(t, HHE_TRANSPORT_GEOMETRY, "H/He scattering flight below time precision");
      left = next;
    }
    if (step == bound) {
      p->scatter_tau = fmax(0, p->scatter_tau - kappa * step);
      for (a = 0; a < 3; ++a) {
        if (face_dist[a] <= bound * (1 + 8 * DBL_EPSILON)) {
          int direction = p->direction[a] > 0 ? 1 : -1;
          int face = p->cell[a] + (direction > 0);
          p->position_cm[a] = face * (m->box_cm[a] / m->dimensions[a]);
          p->cell[a] += direction;
          ++t->ledger.cell_crossings;
          crossed = 1;
          if (p->cell[a] < 0 || p->cell[a] >= m->dimensions[a]) {
            if (m->periodic) {
              p->cell[a] = direction > 0 ? 0 : m->dimensions[a] - 1;
              p->position_cm[a] = direction > 0 ? 0 : m->box_cm[a];
            } else
              p->location = HHE_PACKET_ESCAPED;
          }
        }
      }
      if (!crossed)
        return fail(t, HHE_TRANSPORT_GEOMETRY, "H/He zero-progress face crossing");
      if (p->location == HHE_PACKET_LOCAL && !cell_owned(m, p->cell))
        p->location = HHE_PACKET_ROUTE;
    } else if (step == scatter && left > 0) {
      double mu = 2 * uniform(p) - 1;
      double phi = 2 * HHE_PI * uniform(p);
      double r = sqrt(fmax(0, 1 - mu * mu));
      p->direction[0] = r * cos(phi);
      p->direction[1] = r * sin(phi);
      p->direction[2] = mu;
      p->scatter_tau = -log(uniform(p));
      ++t->ledger.scatter_events;
    } else {
      p->scatter_tau = fmax(0, p->scatter_tau - kappa * step);
    }
  }
  p->remaining_s = (double)(left / HHE_C_LIGHT);
  p->clock_s += original_remaining - p->remaining_s;
  if (p->location == HHE_PACKET_ESCAPED) {
    t->ledger.escaped[p->origin][p->group] += p->count;
    p->remaining_s = 0;
  }
  return HHE_TRANSPORT_OK;
}

int hhe_transport_drift_local(hhe_transport* t, const hhe_mesh* m, const hhe_state* gas)
{
  size_t i, c, ncell = hhe_mesh_local_cells(m);
  double* opacity = NULL;
  int have_line = 0, status = HHE_TRANSPORT_OK;
  if (!t || !valid_mesh(m) || (ncell && !gas))
    return fail(t, HHE_TRANSPORT_INPUT, "Invalid H/He local drift mesh");
  for (i = 0; i < t->size; ++i) {
    hhe_packet* p = &t->packets[i];
    if (p->location != HHE_PACKET_LOCAL)
      continue;
    if (!packet_valid(p, m))
      return fail(t, HHE_TRANSPORT_INPUT, "Invalid H/He packet phase state");
    if (hhe_group_kind[p->group] && p->remaining_s > 0 && cell_owned(m, p->cell))
      have_line = 1;
  }
  if (have_line) {
    if (ncell > SIZE_MAX / (2 * sizeof(double)) ||
        !memory_available(t, t->capacity, t->pending_cells, ncell * 2 * sizeof(double)))
      return fail(t, HHE_TRANSPORT_MEMORY, "H/He opacity workspace exceeds memory budget");
    opacity = (double*)calloc(ncell * 2, sizeof(double));
    if (ncell && !opacity)
      return fail(t, HHE_TRANSPORT_MEMORY, "H/He opacity allocation failed");
    for (c = 0; c < ncell; ++c) {
      double sigma[2];
      if (gas[c].species[2] == 0 && gas[c].species[3] == 0)
        continue;
      if (hhe_line_cross_sections(hhe_temperature(&gas[c]), sigma) != HHE_OK) {
        free(opacity);
        t->error_cell = c;
        return fail(t, HHE_TRANSPORT_CHEMISTRY, "H/He line opacity temperature outside model domain");
      }
      opacity[2 * c] = gas[c].species[2] * sigma[0] * (914. / 915.);
      opacity[2 * c + 1] = gas[c].species[3] * sigma[1];
    }
  }
  for (i = 0; i < t->size; ++i) {
    hhe_packet* p = &t->packets[i];
    if (p->location != HHE_PACKET_LOCAL)
      continue;
    if (!cell_owned(m, p->cell)) {
      p->location = HHE_PACKET_ROUTE;
      continue;
    }
    if (p->remaining_s == 0)
      continue;
    status = drift_packet(t, m, opacity, p);
    if (status)
      break;
  }
  free(opacity);
  return status;
}

static void add_ledger(hhe_ledger* dst, const hhe_ledger* src, double v)
{
  int s, g;
  for (s = 0; s < 3; ++s) {
    for (g = 0; g < HHE_NGROUPS; ++g)
      dst->absorbed[s][g] += src->absorbed[s][g] * v;
    dst->recombinations[s] += src->recombinations[s] * v;
    dst->collisional_ionizations[s] += src->collisional_ionizations[s] * v;
  }
#define ADD(field) dst->field += src->field * v
  ADD(line_conversions);
  ADD(photoheat);
  ADD(capture_cooling);
  ADD(collisional_ionization_cooling);
  ADD(extra_radiative_cooling);
  ADD(inactive_radiation);
  ADD(high_energy_photoheat);
  ADD(energy_error);
  ADD(electron_error);
  ADD(photon_error);
#undef ADD
  dst->substeps += src->substeps;
}

int hhe_transport_chemistry(hhe_transport* t, const hhe_mesh* m, hhe_state* gas, double dt, const hhe_options* options)
{
  size_t ncell = hhe_mesh_local_cells(m), c, i, temporary_bytes;
  size_t *heads = NULL, *next = NULL;
  double volume = hhe_mesh_cell_volume(m);
  int g, status = HHE_TRANSPORT_OK;
  if (!t || !valid_mesh(m) || (ncell && !gas) || !isfinite(volume) || volume <= 0 || !isfinite(dt) || dt < 0)
    return fail(t, HHE_TRANSPORT_INPUT, "Invalid H/He shared chemistry input");
  if (t->has_pending)
    return fail(t, HHE_TRANSPORT_PHASE, "Unemitted H/He endpoint photons before chemistry");
  for (i = 0; i < t->size; ++i)
    if (t->packets[i].location != HHE_PACKET_LOCAL || t->packets[i].remaining_s != 0 ||
        !packet_valid(&t->packets[i], m) || !cell_owned(m, t->packets[i].cell))
      return fail(t, HHE_TRANSPORT_PHASE, "H/He chemistry requires completed local packet drift");
  if (ncell > SIZE_MAX / sizeof(size_t) || t->size > SIZE_MAX / sizeof(size_t) - ncell)
    return fail(t, HHE_TRANSPORT_MEMORY, "H/He chemistry workspace size overflow");
  temporary_bytes = (ncell + t->size) * sizeof(size_t);
  if (!memory_available(t, t->capacity, ncell, temporary_bytes))
    return fail(t, HHE_TRANSPORT_MEMORY, "H/He chemistry workspace exceeds memory budget");
  if (ncell != t->pending_cells) {
    double* rec;
    if (t->pending_cells != 0)
      return fail(t, HHE_TRANSPORT_PHASE, "Changing H/He slab size requires explicit state repartition");
    double* p = (double*)realloc(t->pending_emission, ncell * HHE_NGROUPS * sizeof(double));
    if (ncell && !p)
      return fail(t, HHE_TRANSPORT_MEMORY, "H/He endpoint emission allocation failed");
    t->pending_emission = p;
    rec = (double*)calloc(ncell * 3, sizeof(double));
    if (ncell && !rec)
      return fail(t, HHE_TRANSPORT_MEMORY, "H/He cell recombination allocation failed");
    free(t->cell_recombinations);
    t->cell_recombinations = rec;
    t->pending_cells = ncell;
  }
  if (ncell)
    memset(t->pending_emission, 0, ncell * HHE_NGROUPS * sizeof(double));
  heads = (size_t*)malloc(ncell * sizeof(size_t));
  next = (size_t*)malloc(t->size * sizeof(size_t));
  if ((ncell && !heads) || (t->size && !next)) {
    free(heads);
    free(next);
    return fail(t, HHE_TRANSPORT_MEMORY, "H/He cell packet index allocation failed");
  }
  for (c = 0; c < ncell; ++c)
    heads[c] = SIZE_MAX;
  for (i = 0; i < t->size; ++i) {
    c = local_cell(m, t->packets[i].cell);
    next[i] = heads[c];
    heads[c] = i;
  }
  for (c = 0; c < ncell; ++c) {
    double photons[HHE_NGROUPS] = { 0 }, original[HHE_NGROUPS];
    double compensation[HHE_NGROUPS] = { 0 }, emission[HHE_NGROUPS] = { 0 };
    double factor[HHE_NGROUPS], photo_fraction[HHE_NGROUPS];
    hhe_ledger ledger;
    double density = 0;
    int s;
    for (s = 0; s < HHE_NSPECIES; ++s) {
      if (!isfinite(gas[c].species[s]) || gas[c].species[s] < 0) {
        status = fail(t, HHE_TRANSPORT_INPUT, "Invalid gas species in H/He transport cell");
        break;
      }
      density += gas[c].species[s];
    }
    if (status)
      break;
    if (density == 0 && gas[c].thermal_eV_cm3 == 0)
      continue; /* true vacuum */
    for (i = heads[c]; i != SIZE_MAX; i = next[i]) {
      double y, sum;
      g = t->packets[i].group;
      y = t->packets[i].count - compensation[g];
      sum = photons[g] + y;
      compensation[g] = (sum - photons[g]) - y;
      photons[g] = sum;
    }
    for (g = 0; g < HHE_NGROUPS; ++g)
      original[g] = photons[g] /= volume;
    status = hhe_advance(&gas[c], photons, dt, options, emission, &ledger);
    if (status != HHE_OK) {
      t->error_cell = c;
      t->chemistry_status = status;
      status = fail(t, HHE_TRANSPORT_CHEMISTRY, hhe_status_message(status));
      break;
    }
    for (g = 0; g < HHE_NGROUPS; ++g) {
      double photo = ledger.absorbed[0][g] + ledger.absorbed[1][g] + ledger.absorbed[2][g];
      double removed = original[g] - photons[g];
      factor[g] = original[g] > 0 ? photons[g] / original[g] : 1;
      if (!isfinite(factor[g]) || factor[g] < -64 * DBL_EPSILON || factor[g] > 1 + 64 * DBL_EPSILON) {
        status = fail(t, HHE_TRANSPORT_CHEMISTRY, "Invalid H/He photon survival fraction");
        break;
      }
      factor[g] = fmax(0, fmin(1, factor[g]));
      photo_fraction[g] = hhe_group_kind[g] == 1 && removed > 0 ? fmax(0, fmin(1, photo / removed)) : 1;
      t->pending_emission[c * HHE_NGROUPS + g] = emission[g] * volume;
    }
    if (status)
      break;
    for (i = heads[c]; i != SIZE_MAX; i = next[i]) {
      hhe_packet* p = &t->packets[i];
      double old = p->count, lost, photo;
      g = p->group;
      p->count *= factor[g];
      lost = old - p->count;
      photo = lost * photo_fraction[g];
      t->ledger.photoabsorbed[p->origin][g] += photo;
      t->ledger.converted[p->origin][g] += lost - photo;
    }
    add_ledger(&t->ledger.chemistry, &ledger, volume);
    for (s = 0; s < 3; ++s)
      t->cell_recombinations[3 * c + s] += ledger.recombinations[s] * volume;
  }
  free(heads);
  free(next);
  if (status)
    return status;
  t->has_pending = 1;
  hhe_transport_compact(t);
  return HHE_TRANSPORT_OK;
}

int hhe_transport_emit_pending(hhe_transport* t, const hhe_mesh* m, double birth, unsigned int directions)
{
  int status;
  if (!t || !t->has_pending || t->pending_cells != hhe_mesh_local_cells(m))
    return fail(t, HHE_TRANSPORT_PHASE, "No matching H/He endpoint emission buffer");
  status = emit(t, m, t->pending_emission, 1, birth, directions, 1);
  if (!status) {
    t->has_pending = 0;
    if (t->pending_cells)
      memset(t->pending_emission, 0, t->pending_cells * HHE_NGROUPS * sizeof(double));
  }
  return status;
}

int hhe_transport_rescale_positions(hhe_transport* t, const double old_box[3], const double new_box[3])
{
  size_t i;
  int a;
  if (!t || !old_box || !new_box || t->has_pending)
    return fail(t, HHE_TRANSPORT_PHASE, "Geometry remap requires complete H/He step");
  for (a = 0; a < 3; ++a)
    if (!isfinite(old_box[a]) || !isfinite(new_box[a]) || old_box[a] <= 0 || new_box[a] <= 0)
      return fail(t, HHE_TRANSPORT_INPUT, "Invalid H/He geometry remap");
  for (i = 0; i < t->size; ++i)
    if (t->packets[i].remaining_s != 0 || t->packets[i].location != HHE_PACKET_LOCAL)
      return fail(t, HHE_TRANSPORT_PHASE, "Geometry remap encountered unfinished packet drift");
  for (i = 0; i < t->size; ++i)
    for (a = 0; a < 3; ++a)
      t->packets[i].position_cm[a] *= new_box[a] / old_box[a];
  return HHE_TRANSPORT_OK;
}

void hhe_transport_live_counts(const hhe_transport* t, double counts[2][HHE_NGROUPS])
{
  size_t i;
  memset(counts, 0, 2 * HHE_NGROUPS * sizeof(double));
  if (!t)
    return;
  for (i = 0; i < t->size; ++i) {
    const hhe_packet* p = &t->packets[i];
    if (p->location == HHE_PACKET_LOCAL)
      counts[p->origin][p->group] += p->count;
  }
}
