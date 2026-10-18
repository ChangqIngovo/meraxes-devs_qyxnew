#include "photon_hhe_mpi.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int collective_status(int local, MPI_Comm comm)
{
  int global;
  if (MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
    MPI_Abort(comm, EXIT_FAILURE);
  return global;
}

static int packet_owner(int x, int ranks, const ptrdiff_t* start, const ptrdiff_t* count)
{
  for (int r = 0; r < ranks; r++)
    if (x >= start[r] && x < start[r] + count[r])
      return r;
  return -1;
}

int photon_hhe_mpi_drift(hhe_transport* t,
                         const hhe_mesh* mesh,
                         const hhe_state* gas,
                         double dt_s,
                         const ptrdiff_t* start,
                         const ptrdiff_t* count,
                         MPI_Comm comm)
{
  int rank, ranks, status;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  status = collective_status(hhe_transport_begin_drift(t, dt_s), comm);
  if (status)
    return status;
  int* send_n = calloc((size_t)ranks * 5, sizeof(*send_n));
  status = collective_status(send_n ? 0 : HHE_TRANSPORT_MEMORY, comm);
  if (status) {
    free(send_n);
    return status;
  }
  int *recv_n = send_n + ranks, *send_d = recv_n + ranks;
  int *recv_d = send_d + ranks, *cursor = recv_d + ranks;

  for (;;) {
    status = collective_status(hhe_transport_drift_local(t, mesh, gas), comm);
    if (status)
      break;
    memset(send_n, 0, (size_t)ranks * 5 * sizeof(*send_n));
    unsigned long long outgoing = 0, global_outgoing = 0;
    for (size_t p = 0; p < t->size; p++) {
      if (t->packets[p].location != HHE_PACKET_ROUTE)
        continue;
      int dest = packet_owner(t->packets[p].cell[0], ranks, start, count);
      if (dest < 0 || dest == rank) {
        status = HHE_TRANSPORT_GEOMETRY;
        break;
      }
      if (send_n[dest] > INT_MAX - (int)sizeof(hhe_packet)) {
        snprintf(t->error, sizeof(t->error), "MPI byte count overflow; reduce the transport step");
        status = HHE_TRANSPORT_MEMORY;
        break;
      }
      send_n[dest] += (int)sizeof(hhe_packet);
      outgoing++;
    }
    status = collective_status(status, comm);
    if (status)
      break;
    MPI_Allreduce(&outgoing, &global_outgoing, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    if (!global_outgoing) {
      hhe_transport_compact(t);
      break;
    }
    if (MPI_Alltoall(send_n, 1, MPI_INT, recv_n, 1, MPI_INT, comm) != MPI_SUCCESS)
      MPI_Abort(comm, EXIT_FAILURE);
    size_t send_bytes = 0, recv_bytes = 0;
    for (int r = 0; r < ranks; r++) {
      if (send_bytes > INT_MAX || recv_bytes > INT_MAX || recv_n[r] < 0 || recv_n[r] % (int)sizeof(hhe_packet)) {
        status = HHE_TRANSPORT_MEMORY;
        break;
      }
      send_d[r] = cursor[r] = (int)send_bytes;
      recv_d[r] = (int)recv_bytes;
      send_bytes += (size_t)send_n[r];
      recv_bytes += (size_t)recv_n[r];
    }
    if (send_bytes > INT_MAX || recv_bytes > INT_MAX)
      status = HHE_TRANSPORT_MEMORY;
    unsigned char *send_buf = NULL, *recv_buf = NULL;
    if (!status) {
      send_buf = malloc(send_bytes ? send_bytes : 1);
      recv_buf = malloc(recv_bytes ? recv_bytes : 1);
      if (!send_buf || !recv_buf)
        status = HHE_TRANSPORT_MEMORY;
    }
    status = collective_status(status, comm);
    if (status) {
      free(send_buf);
      free(recv_buf);
      break;
    }
    for (size_t p = 0; p < t->size; p++) {
      if (t->packets[p].location != HHE_PACKET_ROUTE)
        continue;
      int dest = packet_owner(t->packets[p].cell[0], ranks, start, count);
      memcpy(send_buf + cursor[dest], t->packets + p, sizeof(hhe_packet));
      cursor[dest] += (int)sizeof(hhe_packet);
    }
    if (MPI_Alltoallv(send_buf, send_n, send_d, MPI_BYTE, recv_buf, recv_n, recv_d, MPI_BYTE, comm) != MPI_SUCCESS)
      MPI_Abort(comm, EXIT_FAILURE);
    /* The outgoing copy is now on its owning rank; its complete phase and RNG
     * state are retained. No count threshold or packet roulette is applied. */
    hhe_transport_compact(t);
    status = hhe_transport_append(t, (const hhe_packet*)recv_buf, recv_bytes / sizeof(hhe_packet));
    free(send_buf);
    free(recv_buf);
    status = collective_status(status, comm);
    if (status)
      break;
  }
  free(send_n);
  if (status && !t->error[0])
    snprintf(t->error, sizeof(t->error), "collective MPI drift failed (status %d)", status);
  return status;
}
