#include "photon_conservation_mpi.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum
{
  PC_MPI_BEGIN = 1,
  PC_MPI_QUERY,
  PC_MPI_END,
  PC_MPI_STOP
};

struct pc_mpi_context
{
  MPI_Comm comm;
  MPI_Datatype node_type, query_type;
  int rank, ranks, active, stopped;
  size_t max_bytes, min_targets, dim[3], nodes, max_neighbors, batch;
  const pc_geometry_node* tree;
  pc_geometry_node* owned_tree;
  pc_geometry_query* queries;
  int *counts, *neighbours, *row_counts, *row_offsets;
  int *neighbour_counts, *neighbour_offsets;
  pc_geometry_executor executor;
  pc_mpi_stats stats;
};

static void checked(pc_mpi_context* c, int error)
{
  if (error != MPI_SUCCESS)
    MPI_Abort(c->comm, error);
}

static int product(size_t a, size_t b, size_t* result)
{
  if (a && b > SIZE_MAX / a)
    return 0;
  *result = a * b;
  return 1;
}

static int sum(size_t a, size_t b, size_t* result)
{
  if (b > SIZE_MAX - a)
    return 0;
  *result = a + b;
  return 1;
}

static void command(pc_mpi_context* c, int value)
{
  checked(c, MPI_Bcast(&value, 1, MPI_INT, 0, c->comm));
}

static void release(pc_mpi_context* c)
{
  free(c->owned_tree);
  free(c->queries);
  free(c->counts);
  free(c->neighbours);
  free(c->row_counts);
  free(c->row_offsets);
  free(c->neighbour_counts);
  free(c->neighbour_offsets);
  c->owned_tree = NULL;
  c->tree = NULL;
  c->queries = NULL;
  c->counts = NULL;
  c->neighbours = NULL;
  c->row_counts = c->row_offsets = NULL;
  c->neighbour_counts = c->neighbour_offsets = NULL;
  c->active = 0;
}

/* Include the replicated tree when accounting for each worker's memory.
 * Rank zero borrows the original tree and therefore uses less memory. */
static int required_bytes(const pc_mpi_context* c, size_t nodes, size_t neighbours, size_t batch, size_t* bytes)
{
  size_t tree_bytes, per_row, row_bytes, metadata;
  const size_t rows = batch / (size_t)c->ranks + (batch % (size_t)c->ranks != 0);
  return product(nodes, sizeof(pc_geometry_node), &tree_bytes) && product(neighbours, sizeof(int), &per_row) &&
         sum(per_row, sizeof(pc_geometry_query) + sizeof(int), &per_row) && product(rows, per_row, &row_bytes) &&
         product((size_t)c->ranks, 4 * sizeof(int), &metadata) && sum(tree_bytes, row_bytes, bytes) &&
         sum(*bytes, metadata, bytes);
}

static int prepare(pc_mpi_context* c, const unsigned long long header[7], const pc_geometry_node* tree)
{
  size_t bytes, tree_bytes, cells;
  int valid = 1, accepted;
  for (size_t i = 0; i < 7; ++i)
    if (header[i] > SIZE_MAX)
      valid = 0;
  c->dim[0] = (size_t)header[0];
  c->dim[1] = (size_t)header[1];
  c->dim[2] = (size_t)header[2];
  c->nodes = (size_t)header[3];
  c->max_neighbors = (size_t)header[4];
  c->batch = (size_t)header[5];
  valid = valid && c->dim[0] && c->dim[1] && c->dim[2] && c->nodes && c->max_neighbors && c->max_neighbors <= INT_MAX &&
          c->batch && c->batch <= INT_MAX / c->max_neighbors &&
          required_bytes(c, c->nodes, c->max_neighbors, c->batch, &bytes) && (!c->max_bytes || bytes <= c->max_bytes);
  if (valid) {
    const size_t rows = c->batch / (size_t)c->ranks + (c->batch % (size_t)c->ranks != 0);
    valid = product(c->nodes, sizeof(pc_geometry_node), &tree_bytes) && product(rows, c->max_neighbors, &cells);
    if (valid) {
      if (c->rank)
        c->owned_tree = malloc(tree_bytes);
      c->tree = c->rank ? c->owned_tree : tree;
      c->queries = malloc(rows * sizeof(*c->queries));
      c->counts = malloc(rows * sizeof(*c->counts));
      c->neighbours = calloc(cells, sizeof(*c->neighbours));
      c->row_counts = malloc((size_t)c->ranks * sizeof(int));
      c->row_offsets = malloc((size_t)c->ranks * sizeof(int));
      c->neighbour_counts = malloc((size_t)c->ranks * sizeof(int));
      c->neighbour_offsets = malloc((size_t)c->ranks * sizeof(int));
      valid = c->tree && c->queries && c->counts && c->neighbours && c->row_counts && c->row_offsets &&
              c->neighbour_counts && c->neighbour_offsets;
    }
  }
  /* No tree payload or query is sent before every rank owns its buffers. */
  checked(c, MPI_Allreduce(&valid, &accepted, 1, MPI_INT, MPI_MIN, c->comm));
  if (!accepted) {
    release(c);
    return 0;
  }
  for (size_t first = 0; first < c->nodes;) {
    const size_t remaining = c->nodes - first;
    const int chunk = remaining > INT_MAX ? INT_MAX : (int)remaining;
    checked(c, MPI_Bcast((void*)(c->tree + first), chunk, c->node_type, 0, c->comm));
    first += (size_t)chunk;
  }
  c->active = 1;
  c->executor.batch_targets = c->batch;
  if (bytes > c->stats.bytes_peak)
    c->stats.bytes_peak = bytes;
  return 1;
}

static int begin(void* user,
                 const size_t dim[3],
                 const pc_geometry_node* tree,
                 size_t nodes,
                 size_t max_neighbors,
                 size_t targets)
{
  pc_mpi_context* c = user;
  unsigned long long header[7];
  size_t batch, bytes;
  const double start = MPI_Wtime();
  if (c->rank || c->active || c->stopped)
    return -1;
  ++c->stats.begin_calls;
  if (c->ranks < 2 || targets < c->min_targets || !targets || !nodes || !max_neighbors)
    goto decline;
  if (!tree || !dim || !dim[0] || !dim[1] || !dim[2] || max_neighbors > INT_MAX)
    return -1;
  batch = targets < 4096 ? targets : 4096;
  if (batch > INT_MAX / max_neighbors)
    batch = INT_MAX / max_neighbors;
  /* Keep the worker limit a hard bound even when neighbour lists grow. */
  while (batch && (!required_bytes(c, nodes, max_neighbors, batch, &bytes) || (c->max_bytes && bytes > c->max_bytes)))
    batch /= 2;
  if (!batch)
    goto decline;
  for (size_t i = 0; i < 3; ++i)
    header[i] = (unsigned long long)dim[i];
  header[3] = (unsigned long long)nodes;
  header[4] = (unsigned long long)max_neighbors;
  header[5] = (unsigned long long)batch;
  header[6] = (unsigned long long)targets;
  command(c, PC_MPI_BEGIN);
  checked(c, MPI_Bcast(header, 7, MPI_UNSIGNED_LONG_LONG, 0, c->comm));
  if (!prepare(c, header, tree))
    goto decline;
  ++c->stats.accepted_begins;
  c->stats.begin_seconds += MPI_Wtime() - start;
  return 1;

decline:
  ++c->stats.declined_begins;
  c->stats.begin_seconds += MPI_Wtime() - start;
  return 0;
}

static void execute(pc_mpi_context* c, const pc_geometry_query* queries, size_t count, int* counts, int* neighbours)
{
  size_t offset = 0;
  for (int rank = 0; rank < c->ranks; ++rank) {
    const size_t rows = count / (size_t)c->ranks + ((size_t)rank < count % (size_t)c->ranks);
    c->row_counts[rank] = (int)rows;
    c->row_offsets[rank] = (int)offset;
    c->neighbour_counts[rank] = (int)(rows * c->max_neighbors);
    c->neighbour_offsets[rank] = (int)(offset * c->max_neighbors);
    offset += rows;
  }
  const int local = c->row_counts[c->rank];
  checked(
    c,
    MPI_Scatterv(queries, c->row_counts, c->row_offsets, c->query_type, c->queries, local, c->query_type, 0, c->comm));
  for (int i = 0; i < local; ++i) {
    c->counts[i] = 0;
    pc_geometry_collect(c->dim,
                        c->tree,
                        c->queries + i,
                        (int)c->max_neighbors,
                        c->neighbours + (size_t)i * c->max_neighbors,
                        c->counts + i);
  }
  checked(c, MPI_Gatherv(c->counts, local, MPI_INT, counts, c->row_counts, c->row_offsets, MPI_INT, 0, c->comm));
  checked(c,
          MPI_Gatherv(c->neighbours,
                      c->neighbour_counts[c->rank],
                      MPI_INT,
                      neighbours,
                      c->neighbour_counts,
                      c->neighbour_offsets,
                      MPI_INT,
                      0,
                      c->comm));
}

static int query(void* user, const pc_geometry_query* queries, size_t count, int* counts, int* neighbours)
{
  pc_mpi_context* c = user;
  int n;
  const double start = MPI_Wtime();
  if (c->rank || !c->active || count > c->batch || (count && (!queries || !counts || !neighbours)))
    return 0;
  if (!count)
    return 1;
  n = (int)count;
  command(c, PC_MPI_QUERY);
  checked(c, MPI_Bcast(&n, 1, MPI_INT, 0, c->comm));
  execute(c, queries, count, counts, neighbours);
  ++c->stats.query_calls;
  c->stats.targets += count;
  c->stats.query_seconds += MPI_Wtime() - start;
  return 1;
}

static void end(void* user)
{
  pc_mpi_context* c = user;
  if (!c->rank && c->active) {
    command(c, PC_MPI_END);
    release(c);
  }
}

static MPI_Datatype datatype(pc_mpi_context* c,
                             int count,
                             const int* lengths,
                             const MPI_Aint* displacements,
                             const MPI_Datatype* types,
                             MPI_Aint extent)
{
  MPI_Datatype packed, resized;
  checked(c, MPI_Type_create_struct(count, lengths, displacements, types, &packed));
  checked(c, MPI_Type_create_resized(packed, 0, extent, &resized));
  checked(c, MPI_Type_commit(&resized));
  checked(c, MPI_Type_free(&packed));
  return resized;
}

pc_mpi_context* pc_mpi_create(MPI_Comm comm, size_t max_worker_bytes, size_t min_targets)
{
  pc_mpi_context* c = calloc(1, sizeof(*c));
  if (!c)
    return NULL;
  c->comm = comm;
  c->max_bytes = max_worker_bytes;
  c->min_targets = min_targets;
  checked(c, MPI_Comm_rank(comm, &c->rank));
  checked(c, MPI_Comm_size(comm, &c->ranks));
  const int node_lengths[] = { 3, 3, 1, 1, 1 };
  const MPI_Aint node_offsets[] = { offsetof(pc_geometry_node, lo),
                                    offsetof(pc_geometry_node, hi),
                                    offsetof(pc_geometry_node, left),
                                    offsetof(pc_geometry_node, right),
                                    offsetof(pc_geometry_node, leaf) };
  const MPI_Datatype node_types[] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT };
  const int query_lengths[] = { 3, 1 };
  const MPI_Aint query_offsets[] = { offsetof(pc_geometry_query, pos), offsetof(pc_geometry_query, radius) };
  const MPI_Datatype query_types[] = { MPI_INT, MPI_DOUBLE };
  c->node_type = datatype(c, 5, node_lengths, node_offsets, node_types, (MPI_Aint)sizeof(pc_geometry_node));
  c->query_type = datatype(c, 2, query_lengths, query_offsets, query_types, (MPI_Aint)sizeof(pc_geometry_query));
  c->stats.ranks = c->ranks;
  c->executor.user = c;
  c->executor.batch_targets = 4096;
  c->executor.begin = begin;
  c->executor.query = query;
  c->executor.end = end;
  return c;
}

pc_geometry_executor* pc_mpi_executor(pc_mpi_context* context)
{
  return context ? &context->executor : NULL;
}

void pc_mpi_worker(pc_mpi_context* c)
{
  if (!c || !c->rank || c->stopped)
    return;
  for (;;) {
    int value;
    checked(c, MPI_Bcast(&value, 1, MPI_INT, 0, c->comm));
    switch (value) {
      case PC_MPI_BEGIN: {
        unsigned long long header[7];
        if (c->active)
          MPI_Abort(c->comm, MPI_ERR_OTHER);
        checked(c, MPI_Bcast(header, 7, MPI_UNSIGNED_LONG_LONG, 0, c->comm));
        prepare(c, header, NULL);
        break;
      }
      case PC_MPI_QUERY: {
        int count;
        checked(c, MPI_Bcast(&count, 1, MPI_INT, 0, c->comm));
        if (!c->active || count < 0 || (size_t)count > c->batch)
          MPI_Abort(c->comm, MPI_ERR_OTHER);
        execute(c, NULL, (size_t)count, NULL, NULL);
        break;
      }
      case PC_MPI_END:
        release(c);
        break;
      case PC_MPI_STOP:
        release(c);
        c->stopped = 1;
        return;
      default:
        MPI_Abort(c->comm, MPI_ERR_OTHER);
    }
  }
}

void pc_mpi_stop(pc_mpi_context* c)
{
  if (c && !c->rank && !c->stopped) {
    end(c);
    if (c->ranks > 1)
      command(c, PC_MPI_STOP);
    c->stopped = 1;
  }
}

void pc_mpi_destroy(pc_mpi_context* c)
{
  if (c) {
    release(c);
    checked(c, MPI_Type_free(&c->node_type));
    checked(c, MPI_Type_free(&c->query_type));
    free(c);
  }
}

void pc_mpi_get_stats(const pc_mpi_context* context, pc_mpi_stats* stats)
{
  if (stats) {
    memset(stats, 0, sizeof(*stats));
    if (context)
      *stats = context->stats;
  }
}
