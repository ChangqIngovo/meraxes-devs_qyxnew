#ifndef MERAXES_PHOTON_CONSERVATION_GEOMETRY_H
#define MERAXES_PHOTON_CONSERVATION_GEOMETRY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* Read-only periodic spatial queries. Leaf indices refer to the caller's
   * ordered source list, not to a private rank-local source numbering. */
  typedef struct
  {
    int lo[3], hi[3], left, right, leaf;
  } pc_geometry_node;

  typedef struct
  {
    int pos[3];
    double radius;
  } pc_geometry_query;

  /* Optional executor. begin returns 1 to accept, 0 for the serial path, or -1
   * on failure. query returns nonzero on success. end follows every accepted
   * begin, including failure paths. Queries and their neighbours must retain
   * the exact serial tree traversal order. No executor may change the tree,
   * residual capacities, source inventory, or flow/certificate decisions.
   * Output neighbours use a fixed stride of max_neighbors; only counts[i]
   * entries in each row are meaningful. The executor owns its own memory.
   */
  typedef struct
  {
    void* user;
    size_t batch_targets;
    int (*begin)(void* user,
                 const size_t dim[3],
                 const pc_geometry_node* tree,
                 size_t nodes,
                 size_t max_neighbors,
                 size_t targets);
    int (*query)(void* user, const pc_geometry_query* queries, size_t count, int* counts, int* neighbours);
    void (*end)(void* user);
  } pc_geometry_executor;

  /* Shared scalar implementation used by both the solver and MPI workers. */
  void pc_geometry_collect(const size_t dim[3],
                           const pc_geometry_node* tree,
                           const pc_geometry_query* query,
                           int limit,
                           int* neighbours,
                           int* count);

#ifdef __cplusplus
}
#endif

#endif
