#include "photon_conservation.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Fixed-size edge chunks avoid unaccounted realloc peaks. Both directions of
 * an edge are adjacent; their indices differ by one. */
#define PC_CHUNK_SHIFT 12
#define PC_CHUNK_SIZE (1u << PC_CHUNK_SHIFT)
#define PC_CHUNK_MASK (PC_CHUNK_SIZE - 1u)
#define PC_GEOMETRY_EPS 1e-10

typedef struct
{
  int to, next;
  double cap;
} pc_arc;
typedef struct
{
  size_t cell;
  int pos[3], handle, priority;
} pc_cell;
typedef pc_geometry_node pc_kd_node;
typedef struct
{
  void* ptr;
  size_t bytes;
} pc_allocation;

typedef struct
{
  const pc_problem* p;
  pc_options o;
  pc_result* r;
  pc_status error;
  pc_allocation* allocations;
  size_t allocation_count, allocation_capacity, bytes, bytes_peak;
  pc_arc** chunks;
  size_t chunk_capacity, chunk_count, arc_count, pair_count;
  int nv, ns, nt, source_offset, target_offset;
  int *head, *level, *current, *queue, *path_node, *path_arc;
  double* path_min;
  pc_cell *sources, *targets;
  int *kd_order, *target_order, *found;
  pc_kd_node* kd;
  int kd_count;
  double* sink_caps;
  long double photons;
  int final_cut_complete;
  const pc_geometry_executor* geometry;
  pc_geometry_query* queries;
  int *query_targets, *query_counts, *query_neighbours;
  size_t query_capacity, query_stride;
} pc_state;

static int pc_size_product(size_t a, size_t b, size_t* out)
{
  if (a && b > SIZE_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}

static int pc_overlap(const void* a, const void* b, size_t bytes)
{
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  if (!a || !b)
    return 0;
  return x <= y ? y - x < bytes : x - y < bytes;
}

/* The allocation-record table is allocated once and counted explicitly. */
static void* pc_memory(pc_state* s, size_t count, size_t size)
{
  size_t bytes;
  void* ptr;
  if (!pc_size_product(count, size, &bytes) || bytes > s->o.max_bytes - s->bytes ||
      s->allocation_count == s->allocation_capacity) {
    s->error = PC_ERR_MEMORY_LIMIT;
    return NULL;
  }
  if (bytes == 0)
    bytes = 1;
  if (bytes > s->o.max_bytes - s->bytes) {
    s->error = PC_ERR_MEMORY_LIMIT;
    return NULL;
  }
  ptr = calloc(1, bytes);
  if (!ptr) {
    s->error = PC_ERR_MEMORY_LIMIT;
    return NULL;
  }
  s->allocations[s->allocation_count].ptr = ptr;
  s->allocations[s->allocation_count++].bytes = bytes;
  s->bytes += bytes;
  if (s->bytes > s->bytes_peak)
    s->bytes_peak = s->bytes;
  return ptr;
}

static pc_arc* pc_get_arc(pc_state* s, int index)
{
  return &s->chunks[(unsigned)index >> PC_CHUNK_SHIFT][(unsigned)index & PC_CHUNK_MASK];
}

static int pc_add_edge(pc_state* s, int from, int to, double cap, int pair)
{
  int handle;
  pc_arc *a, *b;
  if (pair && s->pair_count >= s->o.max_edges) {
    s->error = PC_ERR_EDGE_LIMIT;
    return -1;
  }
  if (s->arc_count > (size_t)INT_MAX - 2) {
    s->error = PC_ERR_EDGE_LIMIT;
    return -1;
  }
  if ((s->arc_count >> PC_CHUNK_SHIFT) == s->chunk_count) {
    if (s->chunk_count == s->chunk_capacity) {
      s->error = PC_ERR_EDGE_LIMIT;
      return -1;
    }
    s->chunks[s->chunk_count] = pc_memory(s, PC_CHUNK_SIZE, sizeof(pc_arc));
    if (!s->chunks[s->chunk_count])
      return -1;
    s->chunk_count++;
  }
  handle = (int)s->arc_count;
  a = pc_get_arc(s, handle);
  b = pc_get_arc(s, handle ^ 1);
  a->to = to;
  a->cap = cap;
  a->next = s->head[from];
  s->head[from] = handle;
  b->to = from;
  b->cap = 0;
  b->next = s->head[to];
  s->head[to] = handle ^ 1;
  s->arc_count += 2;
  if (pair)
    s->pair_count++;
  return handle;
}

static int pc_bfs(pc_state* s)
{
  int begin = 0, end = 0, u, e;
  for (u = 0; u < s->nv; u++)
    s->level[u] = -1;
  s->queue[end++] = 0;
  s->level[0] = 0;
  while (begin != end) {
    u = s->queue[begin++];
    for (e = s->head[u]; e >= 0; e = pc_get_arc(s, e)->next) {
      pc_arc* a = pc_get_arc(s, e);
      if (a->cap > s->o.flow_tolerance && s->level[a->to] < 0) {
        s->level[a->to] = s->level[u] + 1;
        s->queue[end++] = a->to;
      }
    }
  }
  return s->level[1] >= 0;
}

/* Iterative blocking-flow DFS: no graph-size-dependent C recursion stack. */
static double pc_augment(pc_state* s)
{
  int depth = 0;
  s->path_node[0] = 0;
  s->path_min[0] = (double)s->photons;
  for (;;) {
    int u = s->path_node[depth], e;
    if (u == 1) {
      double amount = s->path_min[depth];
      int k;
      for (k = 0; k < depth; k++) {
        pc_arc* a = pc_get_arc(s, s->path_arc[k]);
        pc_arc* b = pc_get_arc(s, s->path_arc[k] ^ 1);
        a->cap -= amount;
        b->cap += amount;
      }
      return amount;
    }
    e = s->current[u];
    while (e >= 0) {
      pc_arc* a = pc_get_arc(s, e);
      if (a->cap > s->o.flow_tolerance && s->level[a->to] == s->level[u] + 1)
        break;
      e = a->next;
    }
    s->current[u] = e;
    if (e < 0) {
      s->level[u] = -1;
      if (depth == 0)
        return 0;
      depth--;
      s->current[s->path_node[depth]] = pc_get_arc(s, s->path_arc[depth])->next;
    } else {
      pc_arc* a = pc_get_arc(s, e);
      s->path_arc[depth] = e;
      s->path_node[depth + 1] = a->to;
      s->path_min[depth + 1] = fmin(s->path_min[depth], a->cap);
      depth++;
    }
  }
}

static void pc_maxflow(pc_state* s)
{
  while (pc_bfs(s)) {
    int i;
    for (i = 0; i < s->nv; i++)
      s->current[i] = s->head[i];
    while (pc_augment(s) > s->o.flow_tolerance) {
    }
  }
}

static void pc_coordinates(const pc_problem* p, size_t cell, int pos[3])
{
  pos[2] = (int)(cell % p->dim[2]);
  cell /= p->dim[2];
  pos[1] = (int)(cell % p->dim[1]);
  cell /= p->dim[1];
  pos[0] = (int)cell;
}

static double pc_distance2(const pc_state* s, const int a[3], const int b[3])
{
  double result = 0;
  int q;
  for (q = 0; q < 3; q++) {
    double d = fabs((double)a[q] - b[q]);
    d = fmin(d, (double)s->p->dim[q] - d);
    result += d * d;
  }
  return result;
}

static int pc_source_before(const pc_state* s, int a, int b, int axis)
{
  int av = s->sources[a].pos[axis], bv = s->sources[b].pos[axis];
  return av < bv || (av == bv && s->sources[a].cell < s->sources[b].cell);
}

static void pc_select_median(pc_state* s, int begin, int end, int median, int axis)
{
  while (end - begin > 1) {
    int pivot = s->kd_order[begin + (end - begin) / 2], a = begin, b = end - 1;
    while (a <= b) {
      while (pc_source_before(s, s->kd_order[a], pivot, axis))
        a++;
      while (pc_source_before(s, pivot, s->kd_order[b], axis))
        b--;
      if (a <= b) {
        int tmp = s->kd_order[a];
        s->kd_order[a++] = s->kd_order[b];
        s->kd_order[b--] = tmp;
      }
    }
    if (median <= b)
      end = b + 1;
    else if (median >= a)
      begin = a;
    else
      return;
  }
}

static int pc_build_kd(pc_state* s, int begin, int end)
{
  int index = s->kd_count++, q, k, axis = 0;
  pc_kd_node* node = &s->kd[index];
  node->left = node->right = node->leaf = -1;
  for (q = 0; q < 3; q++) {
    node->lo[q] = INT_MAX;
    node->hi[q] = INT_MIN;
    for (k = begin; k < end; k++) {
      int p = s->sources[s->kd_order[k]].pos[q];
      if (p < node->lo[q])
        node->lo[q] = p;
      if (p > node->hi[q])
        node->hi[q] = p;
    }
    if (node->hi[q] - node->lo[q] > node->hi[axis] - node->lo[axis])
      axis = q;
  }
  if (end - begin == 1)
    node->leaf = s->kd_order[begin];
  else {
    int mid = begin + (end - begin) / 2;
    pc_select_median(s, begin, end, mid, axis);
    node->left = pc_build_kd(s, begin, mid);
    node->right = pc_build_kd(s, mid, end);
  }
  return index;
}

static double pc_kd_bound(const pc_state* s, const pc_kd_node* node, const int pos[3])
{
  double total = 0;
  int q;
  for (q = 0; q < 3; q++) {
    double best = 0.;
    /* Integer grid coordinates are in [0,dim). The periodic distance to an
     * interval is the smaller of its direct and opposite-face distances.
     * This exactly replaces the minimum over three translated intervals. */
    if (pos[q] < node->lo[q]) {
      int direct = node->lo[q] - pos[q];
      int wrapped = (int)s->p->dim[q] - (node->hi[q] - pos[q]);
      best = direct < wrapped ? direct : wrapped;
    } else if (pos[q] > node->hi[q]) {
      int direct = pos[q] - node->hi[q];
      int wrapped = (int)s->p->dim[q] - (pos[q] - node->lo[q]);
      best = direct < wrapped ? direct : wrapped;
    }
    total += best * best;
  }
  return total;
}

static int pc_has_neighbor(const pc_state* s, int node_index, const int pos[3], double radius)
{
  const pc_kd_node* node = &s->kd[node_index];
  if (pc_kd_bound(s, node, pos) > radius * radius + PC_GEOMETRY_EPS)
    return 0;
  if (node->leaf >= 0)
    return 1;
  return pc_has_neighbor(s, node->left, pos, radius) || pc_has_neighbor(s, node->right, pos, radius);
}

/* An accepted executor sees one immutable KD tree per residual cut. Buffer
 * storage is charged to the ordinary solver memory limit and reused across
 * all cut rounds. Worker storage has a separate explicit backend budget. */
static int pc_geometry_begin(pc_state* s, size_t targets, size_t limit)
{
  int accepted;
  size_t count, entries;
  if (!s->geometry || !targets)
    return 0;
  accepted = s->geometry->begin(s->geometry->user, s->p->dim, s->kd, (size_t)s->kd_count, limit, targets);
  if (accepted <= 0) {
    if (accepted < 0)
      s->error = PC_ERR_MEMORY_LIMIT;
    return accepted;
  }
  if (!s->geometry->batch_targets) {
    s->error = PC_ERR_INVALID;
    goto fail;
  }
  if (s->query_capacity)
    return 1;
  count = s->geometry->batch_targets;
  if (count > s->p->ncell)
    count = s->p->ncell;
  if (count > 65536)
    count = 65536;
  s->query_stride = s->o.seed_neighbors > s->o.cut_neighbors ? s->o.seed_neighbors : s->o.cut_neighbors;
  if (s->query_stride > (size_t)s->ns)
    s->query_stride = (size_t)s->ns;
  if (!count || !pc_size_product(count, s->query_stride, &entries)) {
    s->error = PC_ERR_MEMORY_LIMIT;
    goto fail;
  }
  s->queries = pc_memory(s, count, sizeof(*s->queries));
  s->query_targets = pc_memory(s, count, sizeof(int));
  s->query_counts = pc_memory(s, count, sizeof(int));
  s->query_neighbours = pc_memory(s, entries, sizeof(int));
  if (!s->queries || !s->query_targets || !s->query_counts || !s->query_neighbours)
    goto fail;
  s->query_capacity = count;
  return 1;
fail:
  s->geometry->end(s->geometry->user);
  return -1;
}

static size_t pc_geometry_batch_size(const pc_state* s)
{
  return s->query_capacity < s->geometry->batch_targets ? s->query_capacity : s->geometry->batch_targets;
}

/* Commit responses in exactly the serial target/traversal order. Geometry
 * queries cannot mutate the flow network, so batching changes no decision. */
static int pc_geometry_commit(pc_state* s, size_t count, size_t limit, int audit)
{
  int added = 0;
  if (!s->geometry->query(s->geometry->user, s->queries, count, s->query_counts, s->query_neighbours)) {
    s->error = PC_ERR_MEMORY_LIMIT;
    return -1;
  }
  for (size_t a = 0; a < count; ++a) {
    int j = s->query_targets[a], found = s->query_counts[a];
    if (found < 0 || (size_t)found > limit) {
      s->error = PC_ERR_NUMERICAL;
      return -1;
    }
    for (int k = 0; k < found; ++k) {
      int source = s->query_neighbours[a * limit + (size_t)k];
      if (source < 0 || source >= s->ns) {
        s->error = PC_ERR_NUMERICAL;
        return -1;
      }
      if (!audit &&
          pc_add_edge(s, s->source_offset + source, s->target_offset + j, (double)(2 * s->photons + 1), 1) < 0)
        return -1;
      ++added;
    }
    if (audit && found)
      s->r->cut_target[j] = 1;
  }
  return added;
}

static int pc_insert_geometry(pc_state* s, int initial)
{
  int i, j, count_sources = 0, added = 0, parallel;
  size_t targets = 0, batch = 0;
  size_t limit = initial ? s->o.seed_neighbors : s->o.cut_neighbors;
  if (limit > (size_t)s->ns)
    limit = (size_t)s->ns;
  for (i = 0; i < s->ns; i++)
    if (initial || s->level[s->source_offset + i] >= 0)
      s->kd_order[count_sources++] = i;
  if (count_sources == 0)
    return 0;
  s->kd_count = 0;
  pc_build_kd(s, 0, count_sources);
  for (j = 0; j < s->nt; ++j)
    targets += initial || s->level[s->target_offset + j] < 0;
  parallel = pc_geometry_begin(s, targets, limit);
  if (parallel < 0)
    return -1;
  for (j = 0; j < s->nt; j++)
    if (initial || s->level[s->target_offset + j] < 0) {
      int count = 0, k;
      pc_geometry_query query;
      memcpy(query.pos, s->targets[j].pos, sizeof(query.pos));
      query.radius = s->p->radius[s->targets[j].cell];
      if (parallel) {
        s->queries[batch] = query;
        s->query_targets[batch++] = j;
        if (batch == pc_geometry_batch_size(s)) {
          int next = pc_geometry_commit(s, batch, limit, 0);
          if (next < 0)
            goto parallel_fail;
          added += next;
          batch = 0;
        }
        continue;
      }
      pc_geometry_collect(s->p->dim, s->kd, &query, (int)limit, s->found, &count);
      for (k = 0; k < count; k++) {
        if (pc_add_edge(s, s->source_offset + s->found[k], s->target_offset + j, (double)(2 * s->photons + 1), 1) < 0)
          return -1;
        added++;
      }
    }
  if (parallel) {
    if (batch) {
      int next = pc_geometry_commit(s, batch, limit, 0);
      if (next < 0)
        goto parallel_fail;
      added += next;
    }
    s->geometry->end(s->geometry->user);
  }
  return added;
parallel_fail:
  s->geometry->end(s->geometry->user);
  return -1;
}

static int pc_complete_stage(pc_state* s)
{
  int added;
  s->final_cut_complete = 0;
  do {
    pc_maxflow(s);
    s->r->metrics.separation_rounds++;
    added = pc_insert_geometry(s, 0);
    if (added < 0)
      return 0;
  } while (added);
  s->final_cut_complete = 1;
  return 1;
}

static long double pc_flow_total(pc_state* s)
{
  long double total = 0;
  int j;
  for (j = 0; j < s->nt; j++)
    total += pc_get_arc(s, s->targets[j].handle ^ 1)->cap;
  return total;
}

static int pc_unlock(pc_state* s, int j, double cap)
{
  if (cap == s->sink_caps[j])
    return 0;
  pc_get_arc(s, s->targets[j].handle)->cap += cap - s->sink_caps[j];
  s->sink_caps[j] = cap;
  return 1;
}

static double pc_middle(const pc_state* s, size_t cell)
{
  return fmax(s->p->lower[cell], fmin(s->p->upper[cell], s->p->baseline[cell]));
}

static int pc_target_before(const pc_state* s, int a, int b)
{
  return s->targets[a].priority < s->targets[b].priority ||
         (s->targets[a].priority == s->targets[b].priority && s->targets[a].cell < s->targets[b].cell);
}

static void pc_sort_targets(pc_state* s, int begin, int end)
{
  if (end - begin > 1) {
    int pivot = s->target_order[begin + (end - begin) / 2], a = begin, b = end - 1;
    while (a <= b) {
      while (pc_target_before(s, s->target_order[a], pivot))
        a++;
      while (pc_target_before(s, pivot, s->target_order[b]))
        b--;
      if (a <= b) {
        int tmp = s->target_order[a];
        s->target_order[a++] = s->target_order[b];
        s->target_order[b--] = tmp;
      }
    }
    if (begin < b + 1)
      pc_sort_targets(s, begin, b + 1);
    if (a < end)
      pc_sort_targets(s, a, end);
  }
}

static int pc_close(double a, double b, const pc_options* o)
{
  return fabs(a - b) <= o->absolute_tolerance + o->relative_tolerance * fmax(fabs(a), fabs(b));
}

static int pc_initialize(pc_state* s)
{
  size_t max_arcs, table_bytes, chunk_count, i;
  int ns = 0, nt = 0;
  for (i = 0; i < s->p->ncell; i++) {
    if (s->p->source[i] > 0)
      ns++;
    if (s->p->upper[i] > 0)
      nt++;
  }
  if ((size_t)ns + nt > (size_t)INT_MAX / 2 || s->o.max_edges > (size_t)INT_MAX / 2 - (size_t)ns - nt) {
    s->error = PC_ERR_EDGE_LIMIT;
    return 0;
  }
  s->ns = ns;
  s->nt = nt;
  s->source_offset = 2;
  s->target_offset = 2 + ns;
  s->nv = 2 + ns + nt;
  max_arcs = 2 * ((size_t)ns + nt + s->o.max_edges);
  chunk_count = (max_arcs + PC_CHUNK_SIZE - 1) / PC_CHUNK_SIZE;
  s->chunk_capacity = chunk_count;
  s->allocation_capacity = chunk_count + 24;
  if (!pc_size_product(s->allocation_capacity, sizeof(pc_allocation), &table_bytes) || table_bytes > s->o.max_bytes) {
    s->error = PC_ERR_MEMORY_LIMIT;
    return 0;
  }
  s->allocations = calloc(s->allocation_capacity, sizeof(pc_allocation));
  if (!s->allocations) {
    s->error = PC_ERR_MEMORY_LIMIT;
    return 0;
  }
  s->bytes = s->bytes_peak = table_bytes;
#define PC_ALLOC(field, count, type)                                                                                   \
  do {                                                                                                                 \
    s->field = pc_memory(s, (count), sizeof(type));                                                                    \
    if (!s->field)                                                                                                     \
      return 0;                                                                                                        \
  } while (0)
  PC_ALLOC(chunks, chunk_count, pc_arc*);
  PC_ALLOC(head, s->nv, int);
  PC_ALLOC(level, s->nv, int);
  PC_ALLOC(current, s->nv, int);
  PC_ALLOC(queue, s->nv, int);
  PC_ALLOC(path_node, s->nv, int);
  PC_ALLOC(path_arc, s->nv, int);
  PC_ALLOC(path_min, s->nv, double);
  PC_ALLOC(sources, ns, pc_cell);
  PC_ALLOC(targets, nt, pc_cell);
  PC_ALLOC(kd_order, ns, int);
  PC_ALLOC(kd, (size_t)ns * 2, pc_kd_node);
  PC_ALLOC(target_order, nt, int);
  PC_ALLOC(sink_caps, nt, double);
  PC_ALLOC(found, fmin(fmax(s->o.seed_neighbors, s->o.cut_neighbors), ns), int);
#undef PC_ALLOC
  for (i = 0; i < (size_t)s->nv; i++) {
    s->head[i] = -1;
    s->level[i] = -1;
  }
  ns = nt = 0;
  for (i = 0; i < s->p->ncell; i++) {
    if (s->p->source[i] > 0) {
      pc_cell* c = &s->sources[ns];
      c->cell = i;
      pc_coordinates(s->p, i, c->pos);
      c->handle = pc_add_edge(s, 0, s->source_offset + ns, s->p->source[i], 0);
      if (c->handle < 0)
        return 0;
      ns++;
    }
    if (s->p->upper[i] > 0) {
      pc_cell* c = &s->targets[nt];
      c->cell = i;
      pc_coordinates(s->p, i, c->pos);
      c->priority = s->p->priority ? s->p->priority[i] : 0;
      c->handle = pc_add_edge(s, s->target_offset + nt, 1, s->p->lower[i], 0);
      if (c->handle < 0)
        return 0;
      s->sink_caps[nt] = s->p->lower[i];
      s->target_order[nt] = nt;
      nt++;
    }
  }
  pc_sort_targets(s, 0, s->nt);
  return 1;
}

static int pc_audit(pc_state* s)
{
  pc_metrics* m = &s->r->metrics;
  const pc_problem* p = s->p;
  long double absorbed = 0, residual = 0, lower = 0, qs = 0, qt = 0, l1 = 0, overlap = 0;
  long double* columns;
  int i, j, valid = 1;
  columns = pc_memory(s, s->nt, sizeof(long double));
  if (!columns)
    return 0;
  m->full_source_budget = 1;
  for (i = 0; i < s->ns; i++) {
    pc_cell* c = &s->sources[i];
    double spent = pc_get_arc(s, c->handle ^ 1)->cap;
    double unused = pc_get_arc(s, c->handle)->cap;
    long double row = 0;
    int e;
    s->r->source_residual[c->cell] = unused;
    if (s->r->source_spent)
      s->r->source_spent[c->cell] = spent;
    residual += unused;
    /* A global/absolute tolerance must not erase an individually faint source. */
    if (unused / p->source[c->cell] > s->o.relative_tolerance)
      m->full_source_budget = 0;
    if (unused / p->source[c->cell] > m->max_source_relative_residual)
      m->max_source_relative_residual = unused / p->source[c->cell];
    if (spent - p->source[c->cell] > m->max_source_overspend)
      m->max_source_overspend = spent - p->source[c->cell];
    if (s->level[s->source_offset + i] >= 0) {
      qs += p->source[c->cell];
      if (s->r->cut_source)
        s->r->cut_source[c->cell] = 1;
    }
    for (e = s->head[s->source_offset + i]; e >= 0; e = pc_get_arc(s, e)->next) {
      pc_arc* a = pc_get_arc(s, e);
      if (a->to >= s->target_offset && a->to < s->nv) {
        double amount = pc_get_arc(s, e ^ 1)->cap;
        if (amount > 0) {
          double radius, distance2, excess;
          j = a->to - s->target_offset;
          row += amount;
          columns[j] += amount;
          m->positive_edges++;
          radius = p->radius[s->targets[j].cell];
          distance2 = pc_distance2(s, c->pos, s->targets[j].pos);
          excess = distance2 - radius * radius;
          if (excess > m->max_geometry_excess)
            m->max_geometry_excess = excess;
          if (excess > PC_GEOMETRY_EPS)
            valid = 0;
        }
      }
    }
    if (fabsl(row - spent) > m->max_edge_balance_error)
      m->max_edge_balance_error = (double)fabsl(row - spent);
    if (!pc_close((double)row, spent, &s->o) || !pc_close(spent + unused, p->source[c->cell], &s->o))
      valid = 0;
  }
  for (j = 0; j < s->nt; j++) {
    pc_cell* c = &s->targets[j];
    double y = pc_get_arc(s, c->handle ^ 1)->cap;
    s->r->y[c->cell] = y;
    absorbed += y;
    if (y - p->upper[c->cell] > m->max_target_excess)
      m->max_target_excess = y - p->upper[c->cell];
    if (p->lower[c->cell] - y > m->max_lower_deficit)
      m->max_lower_deficit = p->lower[c->cell] - y;
    if (fabsl(columns[j] - y) > m->max_edge_balance_error)
      m->max_edge_balance_error = (double)fabsl(columns[j] - y);
    if (!pc_close((double)columns[j], y, &s->o) ||
        y > p->upper[c->cell] + s->o.absolute_tolerance + s->o.relative_tolerance * p->upper[c->cell])
      valid = 0;
    if (s->level[s->target_offset + j] >= 0) {
      qt += s->sink_caps[j];
      if (s->r->cut_target)
        s->r->cut_target[c->cell] = 1;
    }
  }
  /* Zero-capacity targets are omitted from the sparse network, but belong to
   * the declared fixed-radius graph. Include their Q-neighbours in the cut
   * mask: their present cut cost is zero, and a later shape iteration may
   * increase their capacity. This makes the exported mask the full N(Q). */
  if (s->r->cut_target) {
    int count = 0;
    for (i = 0; i < s->ns; i++)
      if (s->level[s->source_offset + i] >= 0)
        s->kd_order[count++] = i;
    if (count) {
      int parallel;
      size_t batch = 0;
      s->kd_count = 0;
      pc_build_kd(s, 0, count);
      parallel = pc_geometry_begin(s, p->ncell - (size_t)s->nt, 1);
      if (parallel < 0)
        return 0;
      for (i = 0; (size_t)i < p->ncell; i++)
        if (p->upper[i] == 0) {
          int pos[3];
          pc_coordinates(p, (size_t)i, pos);
          if (parallel) {
            memcpy(s->queries[batch].pos, pos, sizeof(pos));
            s->queries[batch].radius = p->radius[i];
            s->query_targets[batch++] = i;
            if (batch == pc_geometry_batch_size(s)) {
              if (pc_geometry_commit(s, batch, 1, 1) < 0) {
                s->geometry->end(s->geometry->user);
                return 0;
              }
              batch = 0;
            }
            continue;
          }
          if (pc_has_neighbor(s, 0, pos, p->radius[i]))
            s->r->cut_target[i] = 1;
        }
      if (parallel) {
        int ok = !batch || pc_geometry_commit(s, batch, 1, 1) >= 0;
        s->geometry->end(s->geometry->user);
        if (!ok)
          return 0;
      }
    }
  }
  for (i = 0; (size_t)i < p->ncell; i++) {
    lower += p->lower[i];
    l1 += fabs(s->r->y[i] - p->baseline[i]);
    overlap += fmin(s->r->y[i], p->baseline[i]);
  }
  m->photons = (double)s->photons;
  m->absorbed = (double)absorbed;
  m->source_residual = (double)residual;
  m->lower_demand = (double)lower;
  m->source_cut_budget = (double)qs;
  m->target_cut_capacity = (double)qt;
  m->mincut_capacity = (double)(s->photons - qs + qt);
  m->hall_source_excess = (double)(qs - qt);
  m->lower_cut_deficit = m->lower_feasible ? 0 : (double)(lower - absorbed);
  m->morphology_l1 = (double)l1;
  m->retained_overlap = (double)overlap;
  m->ledger_error = (double)fabsl(absorbed + residual - s->photons);
  m->cut_flow_error = (double)fabsl(s->photons - qs + qt - absorbed);
  m->sources = s->ns;
  m->targets = s->nt;
  m->source_target_edges = s->pair_count;
  m->full_graph_cut_separated = s->final_cut_complete;
  if (!pc_close((double)(absorbed + residual), (double)s->photons, &s->o) ||
      (s->final_cut_complete && !pc_close(m->mincut_capacity, m->absorbed, &s->o)))
    valid = 0;
  return valid;
}

void pc_default_options(pc_options* o)
{
  if (!o)
    return;
  o->max_edges = 4000000;
  o->max_bytes = (size_t)512 * 1024 * 1024;
  o->seed_neighbors = 4;
  o->cut_neighbors = 8;
  o->absolute_tolerance = 1e-10;
  o->relative_tolerance = 1e-10;
  o->flow_tolerance = 0.;
}

const char* pc_status_string(pc_status status)
{
  switch (status) {
    case PC_OK:
      return "complete source-budget allocation";
    case PC_PARTIAL:
      return "maximum allocation with explicit source remainder";
    case PC_LOWER_INFEASIBLE:
      return "incompatible target lower bounds";
    case PC_ERR_INVALID:
      return "invalid input";
    case PC_ERR_MEMORY_LIMIT:
      return "internal memory limit reached";
    case PC_ERR_EDGE_LIMIT:
      return "source-target edge limit reached";
    default:
      return "numerical or certificate verification failure";
  }
}

pc_status pc_allocate_with_geometry(const pc_problem* p,
                                    const pc_options* options,
                                    pc_result* r,
                                    const pc_geometry_executor* geometry)
{
  pc_state s;
  pc_options defaults;
  size_t product = 1, k;
  long double lower = 0, upper_total = 0, baseline_total = 0;
  int initialized = 0, j;
  pc_status status = PC_ERR_INVALID;
  if (!r)
    return PC_ERR_INVALID;
  memset(&r->metrics, 0, sizeof(r->metrics));
  r->status = PC_ERR_INVALID;
  if (!p || !r->y || !r->source_residual || !p->source || !p->capacity || !p->baseline || !p->lower || !p->upper ||
      !p->radius)
    return PC_ERR_INVALID;
  for (k = 0; k < 3; k++)
    if (!p->dim[k] || p->dim[k] > INT_MAX || !pc_size_product(product, p->dim[k], &product))
      return PC_ERR_INVALID;
  if (product != p->ncell || p->ncell > (size_t)INT_MAX - 2)
    return PC_ERR_INVALID;
  {
    const double* input[6] = { p->source, p->capacity, p->baseline, p->lower, p->upper, p->radius };
    double* output[3] = { r->y, r->source_residual, r->source_spent };
    size_t a, b, bytes = p->ncell * sizeof(double);
    for (a = 0; a < 3; a++) {
      for (b = 0; b < 6; b++)
        if (pc_overlap(output[a], input[b], bytes))
          return PC_ERR_INVALID;
      for (b = 0; b < a; b++)
        if (pc_overlap(output[a], output[b], bytes))
          return PC_ERR_INVALID;
    }
  }
  pc_default_options(&defaults);
  memset(&s, 0, sizeof(s));
  s.p = p;
  s.r = r;
  s.geometry = geometry;
  if (geometry && (!geometry->begin || !geometry->query || !geometry->end || !geometry->batch_targets))
    return PC_ERR_INVALID;
  s.o = options ? *options : defaults;
  if (!s.o.max_bytes || !s.o.seed_neighbors || !s.o.cut_neighbors || s.o.seed_neighbors > INT_MAX ||
      s.o.cut_neighbors > INT_MAX || !isfinite(s.o.absolute_tolerance) || !isfinite(s.o.relative_tolerance) ||
      !isfinite(s.o.flow_tolerance) || s.o.absolute_tolerance < 0 || s.o.relative_tolerance < 0 ||
      s.o.relative_tolerance >= 1 || s.o.flow_tolerance < 0)
    return PC_ERR_INVALID;
  for (k = 0; k < p->ncell; k++) {
    r->y[k] = 0;
    r->source_residual[k] = p->source[k];
    if (r->source_spent)
      r->source_spent[k] = 0;
    if (r->cut_source)
      r->cut_source[k] = 0;
    if (r->cut_target)
      r->cut_target[k] = 0;
    if (!isfinite(p->source[k]) || !isfinite(p->capacity[k]) || !isfinite(p->baseline[k]) || !isfinite(p->lower[k]) ||
        !isfinite(p->upper[k]) || !isfinite(p->radius[k]) || p->source[k] < 0 || p->capacity[k] < 0 ||
        p->baseline[k] < 0 || p->baseline[k] > p->capacity[k] || p->lower[k] < 0 || p->upper[k] < p->lower[k] ||
        p->upper[k] > p->capacity[k] || p->radius[k] < 0 || (p->priority && p->priority[k] < 0))
      return PC_ERR_INVALID;
    s.photons += p->source[k];
    lower += p->lower[k];
    upper_total += p->upper[k];
    baseline_total += p->baseline[k];
  }
  if (s.photons > DBL_MAX / 4 || lower > DBL_MAX / 4 || upper_total > DBL_MAX / 4 || baseline_total > DBL_MAX / 4)
    return PC_ERR_INVALID;
  if (!pc_initialize(&s)) {
    status = s.error;
    goto cleanup;
  }
  initialized = 1;
  if (pc_insert_geometry(&s, 1) < 0 || !pc_complete_stage(&s)) {
    status = s.error;
    goto audit;
  }
  r->metrics.lower_stage_flow = (double)pc_flow_total(&s);
  r->metrics.lower_feasible = 1;
  for (j = 0; j < s.nt; j++) {
    double requested = p->lower[s.targets[j].cell];
    if (requested > 0 && pc_get_arc(&s, s.targets[j].handle)->cap / requested > s.o.relative_tolerance)
      r->metrics.lower_feasible = 0;
  }
  if (!r->metrics.lower_feasible) {
    status = PC_LOWER_INFEASIBLE;
    goto audit;
  }
  for (j = 0; j < s.nt;) {
    int begin = j, rank = s.targets[s.target_order[j]].priority, changed = 0;
    long double retained = 0, baseline_retained = 0;
    size_t group;
    while (j < s.nt && s.targets[s.target_order[j]].priority == rank) {
      int target = s.target_order[j++];
      changed |= pc_unlock(&s, target, pc_middle(&s, s.targets[target].cell));
    }
    /* Unchanged capacities preserve the residual graph and its separated
     * cut. Fixed-marginal oracles otherwise repeat these identical solves. */
    if (changed && !pc_complete_stage(&s)) {
      status = s.error;
      goto audit;
    }
    for (; begin < j; begin++) {
      int target = s.target_order[begin];
      double flow = pc_get_arc(&s, s.targets[target].handle ^ 1)->cap;
      retained += fmin(flow, pc_middle(&s, s.targets[target].cell));
      baseline_retained += fmin(flow, p->baseline[s.targets[target].cell]);
    }
    group = r->metrics.priority_groups++;
    if (r->priority_diagnostics && group < r->priority_diagnostics_capacity) {
      r->priority_diagnostics[group].priority = rank;
      r->priority_diagnostics[group].prefix_flow = (double)pc_flow_total(&s);
      r->priority_diagnostics[group].retained_in_group = (double)retained;
      r->priority_diagnostics[group].baseline_retained_in_group = (double)baseline_retained;
    }
  }
  r->metrics.retention_stage_flow = (double)pc_flow_total(&s);
  {
    int changed = 0;
    for (j = 0; j < s.nt; j++)
      changed |= pc_unlock(&s, j, p->upper[s.targets[j].cell]);
    if (changed && !pc_complete_stage(&s)) {
      status = s.error;
      goto audit;
    }
  }
  r->metrics.final_stage_flow = (double)pc_flow_total(&s);
  status = PC_OK;
audit:
  if (!pc_audit(&s))
    status = s.error < 0 ? s.error : PC_ERR_NUMERICAL;
  if (status == PC_OK && !r->metrics.full_source_budget)
    status = PC_PARTIAL;
  if ((status == PC_OK || status == PC_PARTIAL) && r->visit_edge) {
    int i;
    for (i = 0; i < s.ns; i++) {
      int e;
      for (e = s.head[s.source_offset + i]; e >= 0; e = pc_get_arc(&s, e)->next) {
        pc_arc* a = pc_get_arc(&s, e);
        if (a->to >= s.target_offset && a->to < s.nv) {
          double amount = pc_get_arc(&s, e ^ 1)->cap;
          if (amount > 0)
            r->visit_edge(s.sources[i].cell, s.targets[a->to - s.target_offset].cell, amount, r->visitor_user);
        }
      }
    }
  }
cleanup:
  if (!initialized)
    r->metrics.full_graph_cut_separated = 0;
  r->metrics.bytes_peak = s.bytes_peak;
  for (k = 0; k < s.allocation_count; k++)
    free(s.allocations[k].ptr);
  free(s.allocations);
  r->status = status;
  return status;
}

pc_status pc_allocate(const pc_problem* p, const pc_options* options, pc_result* r)
{
  return pc_allocate_with_geometry(p, options, r, NULL);
}
