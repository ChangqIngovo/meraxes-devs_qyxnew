/* Real MPI 1/2/4-rank transport and chemistry against a serial reference. */
#include "core/photon_hhe_mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int rank_id, ranks;
static ptrdiff_t starts[4], counts[4];
static void check(int truth, const char* message)
{
  if (!truth) {
    fprintf(stderr, "rank %d: %s\n", rank_id, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
static void ok(int status, const hhe_transport* t)
{
  if (status) {
    fprintf(stderr, "rank %d: status %d, %s\n", rank_id, status, t->error);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
static void near(double a, double b, double rtol, double atol)
{
  if (!isfinite(a) || fabs(a - b) > atol + rtol * fmax(fabs(a), fabs(b))) {
    fprintf(stderr, "rank %d: %.17g != %.17g\n", rank_id, a, b);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
static hhe_mesh mesh(int whole)
{
  hhe_mesh m = {
    { 4, 1, 1 }, whole ? 0 : (int)starts[rank_id], whole ? 4 : (int)counts[rank_id], { 4e19, 1e19, 1e19 }, 1
  };
  return m;
}
static void fill_gas(hhe_state* gas, int n, int offset)
{
  for (int c = 0; c < n; ++c) {
    const double f = 1. + .05 * (c + offset);
    hhe_state s = { { 2e-5 * f, 8e-5 * f, 3e-6 * f, 3e-6 * f, 2e-6 * f }, 0 };
    double particles =
      s.species[0] + s.species[1] + s.species[2] + s.species[3] + s.species[4] + hhe_electron_density(&s);
    s.thermal_eV_cm3 = 1.5 * HHE_KB_EV_K * 10000. * particles;
    gas[c] = s;
  }
}
static int group(int kind)
{
  for (int g = 0; g < HHE_NGROUPS; ++g)
    if (hhe_group_kind[g] == kind)
      return g;
  return -1;
}
static void trajectory(void)
{
  for (int kind = 0; kind < 3; ++kind) {
    hhe_mesh local = mesh(0), whole = mesh(1);
    hhe_state gas[4], serial_gas[4];
    fill_gas(gas, (int)counts[rank_id], (int)starts[rank_id]);
    fill_gas(serial_gas, 4, 0);
    hhe_transport t, serial;
    hhe_transport_init(&t, 99, 0);
    hhe_transport_init(&serial, 99, 0);
    hhe_packet p;
    memset(&p, 0, sizeof(p));
    for (int a = 0; a < 3; ++a)
      p.position_cm[a] = .5e19;
    p.direction[0] = 1.;
    p.group = group(kind);
    p.energy_eV = hhe_group_energy_eV[p.group];
    p.count = 7.;
    p.scatter_tau = .8;
    p.rng_state = UINT64_C(719360193);
    if (!rank_id) {
      ok(hhe_transport_append(&t, &p, 1), &t);
      ok(hhe_transport_append(&serial, &p, 1), &serial);
      ok(hhe_transport_begin_drift(&serial, 10.e19 / HHE_C_LIGHT), &serial);
      ok(hhe_transport_drift_local(&serial, &whole, serial_gas), &serial);
      p = serial.packets[0];
    }
    MPI_Bcast(&p, (int)sizeof(p), MPI_BYTE, 0, MPI_COMM_WORLD);
    ok(photon_hhe_mpi_drift(&t, &local, gas, 10.e19 / HHE_C_LIGHT, starts, counts, MPI_COMM_WORLD), &t);
    unsigned long long local_count = (unsigned long long)t.size, total_count;
    MPI_Allreduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    check(total_count == 1, "MPI duplicated/discarded a packet");
    if (t.size) {
      const hhe_packet q = t.packets[0];
      check(q.rng_state == p.rng_state, "MPI changed random stream");
      check(q.count == p.count && q.group == p.group && q.remaining_s == 0., "MPI changed photon phase");
      for (int a = 0; a < 3; ++a) {
        check(q.cell[a] == p.cell[a], "MPI changed cell ownership");
        near(q.position_cm[a], p.position_cm[a], 2e-12, 1e5);
        near(q.direction[a], p.direction[a], 2e-12, 1e-14);
      }
      near(q.clock_s, p.clock_s, 2e-13, 1e-10);
      near(q.scatter_tau, p.scatter_tau, 2e-12, 1e-14);
    }
    unsigned long long events = t.ledger.scatter_events, total_events;
    MPI_Reduce(&events, &total_events, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank_id)
      check(total_events == serial.ledger.scatter_events, "MPI changed scattering events");
    hhe_transport_free(&t);
    hhe_transport_free(&serial);
  }
}
static void fill_rates(double* rates, const hhe_mesh* m, double dt, int active)
{
  memset(rates, 0, hhe_mesh_local_cells(m) * HHE_NGROUPS * sizeof(double));
  if (!active)
    return;
  for (int c = 0; c < m->slab_x_count; ++c) {
    int global = c + m->slab_x_start;
    if (global == 1 || global == 2)
      check(hhe_project_photons(
              global == 1 ? 18. : 35., 1e-7 * hhe_mesh_cell_volume(m) / dt, 0, rates + c * HHE_NGROUPS) == HHE_OK,
            "source projection");
  }
}
static void step(hhe_transport* t, const hhe_mesh* m, hhe_state* gas, double dt, int index, int distributed)
{
  double rates[4 * HHE_NGROUPS];
  fill_rates(rates, m, dt, index == 0);
  hhe_options options = { .05, 0, 1 };
  ok(hhe_transport_emit_sources(t, m, rates, dt, index * dt, 6), t);
  if (distributed)
    ok(photon_hhe_mpi_drift(t, m, gas, .5 * dt, starts, counts, MPI_COMM_WORLD), t);
  else {
    ok(hhe_transport_begin_drift(t, .5 * dt), t);
    ok(hhe_transport_drift_local(t, m, gas), t);
    hhe_transport_compact(t);
  }
  ok(hhe_transport_chemistry(t, m, gas, dt, &options), t);
  if (distributed)
    ok(photon_hhe_mpi_drift(t, m, gas, .5 * dt, starts, counts, MPI_COMM_WORLD), t);
  else {
    ok(hhe_transport_begin_drift(t, .5 * dt), t);
    ok(hhe_transport_drift_local(t, m, gas), t);
    hhe_transport_compact(t);
  }
  ok(hhe_transport_emit_pending(t, m, (index + 1) * dt, 6), t);
}
static void dynamic_shared_chemistry(void)
{
  hhe_mesh local = mesh(0), whole = mesh(1);
  hhe_state gas[4], reference[4];
  fill_gas(gas, (int)counts[rank_id], (int)starts[rank_id]);
  fill_gas(reference, 4, 0);
  hhe_transport t, serial;
  hhe_transport_init(&t, 721, 0);
  hhe_transport_init(&serial, 721, 0);
  for (int i = 0; i < 4; ++i) {
    if (!rank_id)
      step(&serial, &whole, reference, 1e8, i, 0);
    step(&t, &local, gas, 1e8, i, 1);
  }
  MPI_Bcast(reference, (int)sizeof(reference), MPI_BYTE, 0, MPI_COMM_WORLD);
  for (int c = 0; c < (int)counts[rank_id]; ++c) {
    for (int s = 0; s < 5; ++s)
      near(gas[c].species[s], reference[c + starts[rank_id]].species[s], 2e-13, 1e-20);
    near(gas[c].thermal_eV_cm3, reference[c + starts[rank_id]].thermal_eV_cm3, 2e-13, 1e-20);
  }
  double budget[10 * HHE_NGROUPS + 2] = { 0 }, total[10 * HHE_NGROUPS + 2], expected[10 * HHE_NGROUPS + 2] = { 0 };
  double live[2][HHE_NGROUPS], serial_live[2][HHE_NGROUPS];
  hhe_transport_live_counts(&t, live);
  hhe_transport_live_counts(&serial, serial_live);
  for (int origin = 0; origin < 2; ++origin)
    for (int g = 0; g < HHE_NGROUPS; ++g) {
      int j = origin * HHE_NGROUPS + g;
      budget[j] = live[origin][g];
      expected[j] = serial_live[origin][g];
      budget[2 * HHE_NGROUPS + j] = t.ledger.emitted[origin][g];
      expected[2 * HHE_NGROUPS + j] = serial.ledger.emitted[origin][g];
      budget[4 * HHE_NGROUPS + j] = t.ledger.photoabsorbed[origin][g];
      expected[4 * HHE_NGROUPS + j] = serial.ledger.photoabsorbed[origin][g];
      budget[6 * HHE_NGROUPS + j] = t.ledger.converted[origin][g];
      expected[6 * HHE_NGROUPS + j] = serial.ledger.converted[origin][g];
      budget[8 * HHE_NGROUPS + j] = t.ledger.escaped[origin][g];
      expected[8 * HHE_NGROUPS + j] = serial.ledger.escaped[origin][g];
    }
  budget[10 * HHE_NGROUPS] = t.ledger.chemistry.inactive_radiation;
  expected[10 * HHE_NGROUPS] = serial.ledger.chemistry.inactive_radiation;
  for (int s = 0; s < 3; ++s) {
    budget[10 * HHE_NGROUPS + 1] += t.ledger.chemistry.recombinations[s];
    expected[10 * HHE_NGROUPS + 1] += serial.ledger.chemistry.recombinations[s];
  }
  MPI_Reduce(budget, total, 10 * HHE_NGROUPS + 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (!rank_id)
    for (int j = 0; j < 10 * HHE_NGROUPS + 2; ++j)
      near(total[j], expected[j], 3e-11, 1e-50);
  hhe_transport_free(&t);
  hhe_transport_free(&serial);
}
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  check(ranks == 1 || ranks == 2 || ranks == 4, "test requires 1, 2 or 4 ranks");
  for (int r = 0; r < ranks; ++r) {
    starts[r] = r * (4 / ranks);
    counts[r] = 4 / ranks;
  }
  trajectory();
  dynamic_shared_chemistry();
  if (!rank_id)
    printf("H/He MPI %d ranks: continuum/HeI/HeII phase and dynamic gas/photon budgets match serial\n", ranks);
  MPI_Finalize();
  return 0;
}
