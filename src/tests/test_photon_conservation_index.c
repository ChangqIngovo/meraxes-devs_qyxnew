/* Exercise the production private edge-index operations with sparse chunk
 * tables. Five actual edges straddle each former 32-bit boundary; no billion-
 * edge graph is allocated. Including the implementation keeps this test seam
 * out of the public solver API. Do not also link photon_conservation.c. */
#include "../physics/photon_conservation.c"

#include <stdio.h>

static int failures = 0;
#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                                             \
      failures++;                                                                                                      \
    }                                                                                                                  \
  } while (0)

static void sparse_high_index_flow(pc_arc_index base)
{
  pc_state s;
  pc_arc_index head[4] = { -1, -1, -1, -1 }, current[4], path_arc[4], edge[5];
  int level[4], queue[4], path_node[4];
  double path_min[4];
  const size_t first_chunk = (size_t)base >> PC_CHUNK_SHIFT;
  size_t i;
  memset(&s, 0, sizeof(s));
  pc_default_options(&s.o);
  s.o.max_edges = 0;
  s.o.max_bytes = 16 * 1024 * 1024;
  s.nv = 4;
  s.photons = 2;
  s.head = head;
  s.current = current;
  s.level = level;
  s.queue = queue;
  s.path_node = path_node;
  s.path_arc = path_arc;
  s.path_min = path_min;
  s.arc_count = (size_t)base;
  s.chunk_count = first_chunk + 1;
  s.chunk_capacity = first_chunk + 2;
  s.allocation_capacity = 8;
  s.allocations = calloc(s.allocation_capacity, sizeof(*s.allocations));
  s.chunks = calloc(s.chunk_capacity, sizeof(*s.chunks));
  CHECK(s.allocations && s.chunks);
  if (!s.allocations || !s.chunks)
    exit(2);
  s.chunks[first_chunk] = calloc(PC_CHUNK_SIZE, sizeof(pc_arc));
  CHECK(s.chunks[first_chunk]);
  if (!s.chunks[first_chunk])
    exit(2);
  s.bytes = s.allocation_capacity * sizeof(*s.allocations) + s.chunk_capacity * sizeof(*s.chunks) +
            PC_CHUNK_SIZE * sizeof(pc_arc);
  s.bytes_peak = s.bytes;

  edge[0] = pc_add_edge(&s, 0, 2, 2, 0);
  edge[1] = pc_add_edge(&s, 2, 3, 2, 1);
  edge[2] = pc_add_edge(&s, 3, 1, 2, 0);
  /* Zero-capacity links force next-pointer traversal above the boundary. */
  edge[3] = pc_add_edge(&s, 0, 2, 0, 1);
  edge[4] = pc_add_edge(&s, 0, 3, 0, 1);
  CHECK(s.error == PC_OK);
  CHECK(s.arc_count == (size_t)base + 10);
  CHECK(s.chunk_count == first_chunk + 2);
  CHECK(s.pair_count == 3);
  for (i = 0; i < 5; i++) {
    CHECK(edge[i] == base + (pc_arc_index)(2 * i));
    CHECK(pc_get_arc(&s, edge[i]) == &s.chunks[(size_t)edge[i] >> PC_CHUNK_SHIFT][(size_t)edge[i] & PC_CHUNK_MASK]);
  }
  CHECK(pc_get_arc(&s, edge[4])->next == edge[3]);
  CHECK(pc_get_arc(&s, edge[3])->next == edge[0]);
  pc_maxflow(&s);
  for (i = 0; i < 3; i++) {
    CHECK(pc_get_arc(&s, edge[i])->cap == 0);
    CHECK(pc_get_arc(&s, edge[i] ^ 1)->cap == 2);
  }
  CHECK(!pc_bfs(&s));
  CHECK(level[0] == 0 && level[1] == -1);
  CHECK(s.bytes_peak <= s.o.max_bytes);

  /* The new representational limit must reject before pointer indexing. */
  s.arc_count = (size_t)INT64_MAX - 1;
  CHECK(pc_add_edge(&s, 0, 1, 1, 0) == -1);
  CHECK(s.error == PC_ERR_EDGE_LIMIT);
  for (i = 0; i < s.allocation_count; i++)
    free(s.allocations[i].ptr);
  free(s.chunks[first_chunk]);
  free(s.chunks);
  free(s.allocations);
}

int main(void)
{
  CHECK(sizeof(pc_arc_index) >= 8);
  sparse_high_index_flow((pc_arc_index)INT_MAX - 3);
  sparse_high_index_flow((pc_arc_index)UINT32_MAX - 3);
  if (failures) {
    fprintf(stderr, "%d photon-conservation index checks failed\n", failures);
    return 1;
  }
  puts("photon-conservation 64-bit index checks passed");
  return 0;
}
