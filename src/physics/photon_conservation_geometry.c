#include "photon_conservation_geometry.h"

/* Coordinates and box lengths are nonnegative integers <= INT_MAX. Each
 * difference below stays in that range; conversion precedes multiplication.
 * This is exactly the minimum over the three periodic images of an interval,
 * without repeated fmax calls or integer expressions such as 2*box_length. */
static double bound(const size_t dim[3], const pc_geometry_node* node, const int pos[3])
{
  double total = 0.;
  for (int q = 0; q < 3; ++q) {
    int d = 0;
    if (pos[q] < node->lo[q]) {
      int direct = node->lo[q] - pos[q];
      int wrapped = (int)dim[q] - (node->hi[q] - pos[q]);
      d = direct < wrapped ? direct : wrapped;
    } else if (pos[q] > node->hi[q]) {
      int direct = pos[q] - node->hi[q];
      int wrapped = (int)dim[q] - (pos[q] - node->lo[q]);
      d = direct < wrapped ? direct : wrapped;
    }
    total += (double)d * (double)d;
  }
  return total;
}

static void collect(const size_t dim[3],
                    const pc_geometry_node* tree,
                    int index,
                    const pc_geometry_query* query,
                    double radius2,
                    double distance2,
                    int limit,
                    int* neighbours,
                    int* count)
{
  const pc_geometry_node* node = &tree[index];
  if (*count == limit || distance2 > radius2)
    return;
  if (node->leaf >= 0) {
    neighbours[(*count)++] = node->leaf;
    return;
  }
  double a = bound(dim, &tree[node->left], query->pos);
  double b = bound(dim, &tree[node->right], query->pos);
  if (a <= b) {
    collect(dim, tree, node->left, query, radius2, a, limit, neighbours, count);
    collect(dim, tree, node->right, query, radius2, b, limit, neighbours, count);
  } else {
    collect(dim, tree, node->right, query, radius2, b, limit, neighbours, count);
    collect(dim, tree, node->left, query, radius2, a, limit, neighbours, count);
  }
}

void pc_geometry_collect(const size_t dim[3],
                         const pc_geometry_node* tree,
                         const pc_geometry_query* query,
                         int limit,
                         int* neighbours,
                         int* count)
{
  *count = 0;
  if (limit <= 0)
    return;
  collect(dim,
          tree,
          0,
          query,
          query->radius * query->radius + 1e-10,
          bound(dim, &tree[0], query->pos),
          limit,
          neighbours,
          count);
}
