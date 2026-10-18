/* Physical-phase and decomposition checks using the real transport core. */
#include "photon_hhe_transport.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ok(int status, const hhe_transport* t)
{
  if (status) {
    fprintf(stderr, "transport status %d: %s\n", status, t->error);
    assert(0);
  }
}

static void near(double a, double b, double relative, double absolute)
{
  if (!isfinite(a) || fabs(a - b) > absolute + relative * fmax(fabs(a), fabs(b))) {
    fprintf(stderr, "%.17g != %.17g\n", a, b);
    assert(0);
  }
}

static void temperature(hhe_state* s, double T)
{
  double n = 0.;
  for (int a = 0; a < 5; ++a)
    n += s->species[a];
  s->thermal_eV_cm3 = 1.5 * HHE_KB_EV_K * T * (n + hhe_electron_density(s));
}

static int group(int kind)
{
  for (int g = 0; g < HHE_NGROUPS; ++g)
    if (hhe_group_kind[g] == kind)
      return g;
  assert(0);
  return -1;
}

static hhe_packet make_packet(int kind)
{
  hhe_packet p;
  memset(&p, 0, sizeof(p));
  p.position_cm[0] = .5e19;
  p.position_cm[1] = .5e19;
  p.position_cm[2] = .5e19;
  p.direction[0] = 1.;
  p.count = 3.;
  p.group = group(kind);
  p.energy_eV = hhe_group_energy_eV[p.group];
  p.scatter_tau = .8;
  p.rng_state = UINT64_C(719360193);
  return p;
}

static hhe_mesh mesh(int ranks, int rank, int periodic)
{
  hhe_mesh m = { { 4, 2, 2 }, rank * (4 / ranks), 4 / ranks, { 4e19, 2e19, 2e19 }, periodic };
  return m;
}

/* Simulate packet exchange across the same exact slab boundaries used by MPI.
 * Every packet carries its RNG and unused scattering optical depth. */
static hhe_packet run_partition(int ranks, int kind, uint64_t* scatters)
{
  hhe_transport t[4];
  hhe_mesh m[4];
  hhe_state gas[4][16];
  for (int r = 0; r < ranks; ++r) {
    hhe_transport_init(t + r, 123, 0);
    m[r] = mesh(ranks, r, 1);
    memset(gas[r], 0, sizeof(gas[r]));
    if (kind)
      for (size_t i = 0; i < hhe_mesh_local_cells(m + r); ++i) {
        gas[r][i].species[0] = 1e-4;
        gas[r][i].species[2] = 8e-6;
        temperature(&gas[r][i], 10000.);
      }
  }
  hhe_packet p = make_packet(kind);
  ok(hhe_transport_append(t, &p, 1), t);
  for (int r = 0; r < ranks; ++r)
    ok(hhe_transport_begin_drift(t + r, 10.e19 / HHE_C_LIGHT), t + r);
  int cycles = 0;
  for (;;) {
    int routed = 0;
    for (int r = 0; r < ranks; ++r)
      ok(hhe_transport_drift_local(t + r, m + r, gas[r]), t + r);
    for (int r = 0; r < ranks; ++r) {
      for (size_t i = 0; i < t[r].size; ++i)
        if (t[r].packets[i].location == HHE_PACKET_ROUTE) {
          hhe_packet moved = t[r].packets[i];
          const int owner = moved.cell[0] / (4 / ranks);
          assert(owner >= 0 && owner < ranks && owner != r);
          ok(hhe_transport_append(t + owner, &moved, 1), t + owner);
          ++routed;
        }
      hhe_transport_compact(t + r);
    }
    if (!routed)
      break;
    assert(++cycles < 1000);
  }
  size_t live = 0;
  *scatters = 0;
  for (int r = 0; r < ranks; ++r) {
    live += t[r].size;
    *scatters += t[r].ledger.scatter_events;
    if (t[r].size)
      p = t[r].packets[0];
    hhe_transport_free(t + r);
  }
  assert(live == 1 && p.count == 3. && p.remaining_s == 0.);
  near(p.clock_s, 10.e19 / HHE_C_LIGHT, 2e-15, 0.);
  return p;
}

static void decomposition_and_periodic_flight(void)
{
  for (int kind = 0; kind <= 1; ++kind) {
    uint64_t n0;
    hhe_packet reference = run_partition(1, kind, &n0);
    if (kind)
      assert(n0 > 0);
    else
      near(reference.position_cm[0], 2.5e19, 4e-15, 0.);
    for (int ranks = 2; ranks <= 4; ranks *= 2) {
      uint64_t events;
      hhe_packet trial = run_partition(ranks, kind, &events);
      assert(events == n0 && trial.rng_state == reference.rng_state);
      assert(trial.group == reference.group && trial.count == reference.count);
      near(trial.scatter_tau, reference.scatter_tau, 5e-13, 1e-14);
      for (int a = 0; a < 3; ++a) {
        assert(trial.cell[a] == reference.cell[a]);
        near(trial.position_cm[a], reference.position_cm[a], 5e-13, 1e5);
        near(trial.direction[a], reference.direction[a], 5e-13, 1e-14);
      }
    }
  }
}

static void source_off_open_escape(void)
{
  hhe_mesh m = mesh(1, 0, 0);
  hhe_state gas[16] = { { { 0 }, 0 } };
  hhe_transport t;
  hhe_transport_init(&t, 7, 0);
  hhe_packet p = make_packet(0);
  ok(hhe_transport_append(&t, &p, 1), &t);
  const double dt = 1e19 / HHE_C_LIGHT;
  for (int k = 0; k < 3; ++k) {
    ok(hhe_transport_begin_drift(&t, dt), &t);
    ok(hhe_transport_drift_local(&t, &m, gas), &t);
    assert(t.size == 1 && t.packets[0].location == HHE_PACKET_LOCAL && t.packets[0].count == 3.);
    near(t.packets[0].position_cm[0], (.5 + k + 1) * 1e19, 3e-15, 0.);
  }
  ok(hhe_transport_begin_drift(&t, dt), &t);
  ok(hhe_transport_drift_local(&t, &m, gas), &t);
  assert(t.packets[0].location == HHE_PACKET_ESCAPED);
  near(t.packets[0].clock_s, 3.5 * dt, 3e-15, 0.);
  near(t.ledger.escaped[0][p.group], 3., 0., 0.);
  hhe_transport_compact(&t);
  assert(t.size == 0);
  hhe_transport_free(&t);
}

static double material_energy(const hhe_state* s, double volume)
{
  return volume * (s->thermal_eV_cm3 + 13.6 * s->species[1] + 24.59 * s->species[3] + 78.99 * s->species[4]);
}

static void dynamic_source_off_and_energy(void)
{
  hhe_mesh m = { { 1, 1, 1 }, 0, 1, { 1e19, 1e19, 1e19 }, 1 };
  hhe_state gas = { { 2e-5, 8e-5, 3e-6, 3e-6, 2e-6 }, 0 };
  temperature(&gas, 10000.);
  const double volume = hhe_mesh_cell_volume(&m), initial_energy = material_energy(&gas, volume);
  const double dt = 1e8;
  hhe_options options = { .05, 0, 1 };
  double rates[HHE_NGROUPS] = { 0 };
  assert(hhe_project_photons(18., 1e-6 * volume / dt, 0, rates) == HHE_OK);
  hhe_transport t;
  hhe_transport_init(&t, 721, 0);
  double first_diffuse = 0.;
  for (int step = 0; step < 4; ++step) {
    if (step == 1)
      memset(rates, 0, sizeof(rates));
    ok(hhe_transport_emit_sources(&t, &m, rates, dt, step * dt, 6), &t);
    ok(hhe_transport_begin_drift(&t, .5 * dt), &t);
    ok(hhe_transport_drift_local(&t, &m, &gas), &t);
    hhe_transport_compact(&t);
    ok(hhe_transport_chemistry(&t, &m, &gas, dt, &options), &t);
    const size_t before_emission = t.size;
    ok(hhe_transport_begin_drift(&t, .5 * dt), &t);
    ok(hhe_transport_drift_local(&t, &m, &gas), &t);
    hhe_transport_compact(&t);
    ok(hhe_transport_emit_pending(&t, &m, (step + 1) * dt, 6), &t);
    for (size_t i = before_emission; i < t.size; ++i) {
      assert(t.packets[i].origin == 1);
      assert(t.packets[i].birth_s == (step + 1) * dt && t.packets[i].clock_s == (step + 1) * dt);
    }
    if (!step)
      for (int g = 0; g < HHE_NGROUPS; ++g)
        first_diffuse += t.ledger.emitted[1][g];
  }
  double live[2][HHE_NGROUPS], input = 0., radiation = 0., diffuse = 0., diffuse_absorbed = 0.;
  hhe_transport_live_counts(&t, live);
  for (int g = 0; g < HHE_NGROUPS; ++g) {
    input += t.ledger.emitted[0][g] * hhe_group_energy_eV[g];
    diffuse += t.ledger.emitted[1][g];
    diffuse_absorbed += t.ledger.photoabsorbed[1][g];
    for (int origin = 0; origin < 2; ++origin) {
      const double used = live[origin][g] + t.ledger.photoabsorbed[origin][g] + t.ledger.converted[origin][g] +
                          t.ledger.escaped[origin][g];
      near(used, t.ledger.emitted[origin][g], 3e-13, 1e-50);
      radiation += (live[origin][g] + t.ledger.escaped[origin][g]) * hhe_group_energy_eV[g];
    }
  }
  assert(diffuse > first_diffuse && first_diffuse > 0. && diffuse_absorbed > 0.);
  near(initial_energy + input,
       material_energy(&gas, volume) + radiation + t.ledger.chemistry.inactive_radiation,
       3e-14,
       0.);
  near(gas.species[0] + gas.species[1], 1e-4, 3e-15, 0.);
  near(gas.species[2] + gas.species[3] + gas.species[4], 8e-6, 3e-15, 0.);
  printf("source-off: diffuse births %.9g, diffuse reabsorptions %.9g, scatter events %llu\n",
         diffuse,
         diffuse_absorbed,
         (unsigned long long)t.ledger.scatter_events);
  hhe_transport_free(&t);
}

int main(void)
{
  decomposition_and_periodic_flight();
  source_off_open_escape();
  dynamic_source_off_and_energy();
  puts("H/He transport: 1/2/4 slab trajectories, finite-c escape, state carry, source-off recycling and energy passed");
  return 0;
}
