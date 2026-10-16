#ifndef MERAXES_PHOTON_CONSERVATION_MPI_H
#define MERAXES_PHOTON_CONSERVATION_MPI_H

#include "../physics/photon_conservation_geometry.h"
#include <mpi.h>
#include <stddef.h>

typedef struct pc_mpi_context pc_mpi_context;

typedef struct
{
  size_t begin_calls, accepted_begins, declined_begins;
  size_t query_calls, targets, bytes_peak;
  int ranks;
  double begin_seconds, query_seconds;
} pc_mpi_stats;

/* Creation and destruction are local operations; comm is borrowed and must
 * remain valid. All ranks must successfully create their context before any
 * rank enters the service phase. A zero byte limit means no explicit limit.
 * min_targets selects when distributed spatial queries are worthwhile.
 *
 * During the service phase rank zero alone uses the executor and eventually
 * calls stop. Every other rank calls worker exactly once; it returns after
 * stop. Commands are collective on comm, so no other collectives may overlap
 * this phase. MPI failures abort comm; allocation failure during begin is a
 * clean, collective decline, allowing the original scalar solver to run.
 */
pc_mpi_context* pc_mpi_create(MPI_Comm comm, size_t max_worker_bytes, size_t min_targets);
pc_geometry_executor* pc_mpi_executor(pc_mpi_context* context);
void pc_mpi_worker(pc_mpi_context* context);
void pc_mpi_stop(pc_mpi_context* context);
void pc_mpi_destroy(pc_mpi_context* context);
/* Rank-zero totals; bytes_peak is the largest executor payload per worker,
 * including its replicated tree, not the total memory used by Meraxes. */
void pc_mpi_get_stats(const pc_mpi_context* context, pc_mpi_stats* stats);

#endif
