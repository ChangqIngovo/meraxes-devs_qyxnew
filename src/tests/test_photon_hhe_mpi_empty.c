/* MPI ownership/phase regression. Run with 1, 2 and 3 ranks; the three-rank
 * case intentionally contains an empty x slab. No Meraxes global state. */
#include "core/photon_hhe_mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int condition, const char* message, MPI_Comm comm)
{
  int local = condition ? 0 : 1, failed, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Allreduce(&local, &failed, 1, MPI_INT, MPI_MAX, comm);
  if (failed) {
    if (!condition)
      fprintf(stderr, "rank %d: %s\n", rank, message);
    MPI_Abort(comm, 1);
  }
}

static void seed_packets(hhe_transport* t, const hhe_mesh* mesh, MPI_Comm comm)
{
  hhe_transport_init(t, 17, 1024 * 1024);
  for (int id = 0; id < 2; id++) {
    if (id < mesh->slab_x_start || id >= mesh->slab_x_start + mesh->slab_x_count)
      continue;
    hhe_packet packet;
    memset(&packet, 0, sizeof(packet));
    packet.position_cm[0] = (id ? 1.75 : 0.25) * HHE_C_LIGHT;
    packet.position_cm[1] = packet.position_cm[2] = HHE_C_LIGHT * 0.5;
    packet.direction[0] = id ? -1 : 1;
    packet.cell[0] = id;
    packet.count = id ? 7 : 3;
    packet.group = 0;
    packet.energy_eV = hhe_group_energy_eV[0];
    packet.scatter_tau = 1.25;
    packet.rng_state = 31 + (unsigned)id;
    packet.emitter_id = (unsigned)id;
    if (hhe_transport_append(t, &packet, 1))
      MPI_Abort(comm, 1);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  ptrdiff_t* start = calloc((size_t)ranks, sizeof(*start));
  ptrdiff_t* count = calloc((size_t)ranks, sizeof(*count));
  check(start && count, "test allocation", MPI_COMM_WORLD);
  for (int r = 0; r < ranks; r++) {
    if (ranks == 1) {
      start[r] = 0;
      count[r] = 2;
    } else {
      start[r] = r < 2 ? r : 2;
      count[r] = r < 2 ? 1 : 0;
    }
  }
  hhe_mesh mesh = { { 2, 1, 1 }, (int)start[rank], (int)count[rank], { 2 * HHE_C_LIGHT, HHE_C_LIGHT, HHE_C_LIGHT }, 1 };
  hhe_state gas[2] = { { { 0, 0, 0, 0, 0 }, 0 }, { { 0, 0, 0, 0, 0 }, 0 } };
  hhe_transport t;

  /* Landing exactly on the slab boundary must use the outgoing direction,
   * even when no flight time remains to expose the wrong owner later. */
  seed_packets(&t, &mesh, MPI_COMM_WORLD);
  int status = photon_hhe_mpi_drift(&t, &mesh, gas, 0.75, start, count, MPI_COMM_WORLD);
  check(status == 0, "exact-face drift status", MPI_COMM_WORLD);
  int correct = 1;
  double weight = 0;
  for (size_t i = 0; i < t.size; i++) {
    const hhe_packet* p = t.packets + i;
    const int expected_cell = p->emitter_id ? 0 : 1;
    correct &= p->cell[0] == expected_cell && p->cell[0] >= mesh.slab_x_start &&
               p->cell[0] < mesh.slab_x_start + mesh.slab_x_count;
    correct &= fabs(p->position_cm[0] / HHE_C_LIGHT - 1) < 1e-14;
    correct &= p->remaining_s == 0 && fabs(p->clock_s - 0.75) < 1e-14;
    correct &= p->scatter_tau == 1.25 && p->rng_state == 31 + p->emitter_id;
    weight += p->count;
  }
  MPI_Allreduce(MPI_IN_PLACE, &weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  check(correct && fabs(weight - 10) < 1e-14, "exact-face phase/ownership/count", MPI_COMM_WORLD);
  hhe_transport_free(&t);

  seed_packets(&t, &mesh, MPI_COMM_WORLD);
  status = photon_hhe_mpi_drift(&t, &mesh, gas, 3.5, start, count, MPI_COMM_WORLD);
  check(status == 0, "multiple-handoff periodic drift status", MPI_COMM_WORLD);
  correct = 1;
  weight = 0;
  for (size_t i = 0; i < t.size; i++) {
    const hhe_packet* p = t.packets + i;
    const double expected = p->emitter_id ? 0.25 : 1.75;
    correct &= fabs(p->position_cm[0] / HHE_C_LIGHT - expected) < 2e-14;
    correct &= p->remaining_s == 0 && fabs(p->clock_s - 3.5) < 2e-14;
    correct &= p->scatter_tau == 1.25 && p->rng_state == 31 + p->emitter_id;
    weight += p->count;
  }
  MPI_Allreduce(MPI_IN_PLACE, &weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  check(correct && fabs(weight - 10) < 1e-14, "periodic phase/count conservation", MPI_COMM_WORLD);
  hhe_transport_free(&t);

  mesh.periodic = 0;
  seed_packets(&t, &mesh, MPI_COMM_WORLD);
  status = photon_hhe_mpi_drift(&t, &mesh, gas, 3.5, start, count, MPI_COMM_WORLD);
  check(status == 0, "open-boundary drift status", MPI_COMM_WORLD);
  weight = t.ledger.escaped[0][0];
  MPI_Allreduce(MPI_IN_PLACE, &weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  check(t.size == 0 && fabs(weight - 10) < 1e-14, "open-boundary escape counted exactly once", MPI_COMM_WORLD);
  hhe_transport_free(&t);

  free(start);
  free(count);
  if (!rank)
    printf("HHe MPI phase/ownership tests passed (%d ranks)\n", ranks);
  MPI_Finalize();
  return 0;
}
