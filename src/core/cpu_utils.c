#include "cpu_utils.h"
#include "meraxes.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void* cpu_checked_alloc(size_t count, size_t size)
{
  if (size && count > SIZE_MAX / size) {
    mlog_error("CPU workspace allocation overflows size_t.");
    ABORT(EXIT_FAILURE);
  }
  void* p = calloc(count ? count : 1, size ? size : 1);
  if (!p) {
    mlog_error("Cannot allocate CPU workspace (%zu records of %zu bytes).", count, size);
    ABORT(EXIT_FAILURE);
  }
  return p;
}

void filter_fields(const fftwf_complex* const* input,
                   fftwf_complex* const* output,
                   int n_fields,
                   int local_ix_start,
                   int slab_nx,
                   int dim,
                   float radius,
                   int filter_type,
                   int apply_filter)
{
  const int middle = dim / 2;
  const float box_size = (float)run_globals.params.BoxSize;
  const float delta_k = (float)(2.0 * M_PI / box_size);
  if (!apply_filter) {
    size_t bytes = (size_t)slab_nx * dim * (middle + 1) * sizeof(fftwf_complex);
    for (int f = 0; f < n_fields; ++f)
      if (input[f] != output[f])
        memcpy(output[f], input[f], bytes);
    return;
  }
  if (filter_type < 0 || filter_type > 2) {
    mlog_error("Unknown filter type %d.", filter_type);
    ABORT(EXIT_FAILURE);
  }
  for (int x = 0; x < slab_nx; ++x) {
    int gx = x + local_ix_start;
    float kx = (gx > middle ? gx - dim : gx) * delta_k;
    for (int y = 0; y < dim; ++y) {
      float ky = (y > middle ? y - dim : y) * delta_k;
      for (int z = 0; z <= middle; ++z) {
        float kz = z * delta_k;
        float kR = sqrtf(kx * kx + ky * ky + kz * kz) * radius;
        size_t i = z + (size_t)(middle + 1) * (y + (size_t)dim * x);
        fftwf_complex w = 1.0f;
        int multiply = 1, zero = 0;
        switch (filter_type) {
          case 0:
            if (kR > 1e-4)
              w = (fftwf_complex)(3.0 * (sinf(kR) / powf(kR, 3) - cosf(kR) / powf(kR, 2)));
            else
              multiply = 0;
            break;
          case 1:
            kR *= 0.413566994;
            zero = kR > 1;
            multiply = 0;
            break;
          case 2:
            kR *= 0.643;
            w = (fftwf_complex)powf((float)M_E, (float)(-kR * kR / 2.0));
            break;
        }
        for (int f = 0; f < n_fields; ++f) {
          fftwf_complex v = input[f][i];
          if (zero)
            v = 0.0f;
          else if (multiply)
            v *= w;
          output[f][i] = v;
        }
      }
    }
  }
}

void* exchange_records(const void* send,
                       const size_t* send_count,
                       size_t record_size,
                       size_t** recv_count,
                       size_t** recv_offset,
                       MPI_Comm comm)
{
  int ranks, rank;
  MPI_Comm_size(comm, &ranks);
  MPI_Comm_rank(comm, &rank);
  unsigned long long* counts = cpu_checked_alloc((size_t)ranks * 2, sizeof(*counts));
  for (int p = 0; p < ranks; ++p)
    counts[p] = send_count[p];
  MPI_Alltoall(counts, 1, MPI_UNSIGNED_LONG_LONG, counts + ranks, 1, MPI_UNSIGNED_LONG_LONG, comm);
  *recv_count = cpu_checked_alloc(ranks, sizeof(size_t));
  *recv_offset = cpu_checked_alloc((size_t)ranks + 1, sizeof(size_t));
  size_t* send_offset = cpu_checked_alloc((size_t)ranks + 1, sizeof(size_t));
  int small = 1;
  for (int p = 0; p < ranks; ++p) {
    (*recv_count)[p] = (size_t)counts[ranks + p];
    if ((*recv_count)[p] > SIZE_MAX - (*recv_offset)[p] || send_count[p] > SIZE_MAX - send_offset[p])
      ABORT(EXIT_FAILURE);
    (*recv_offset)[p + 1] = (*recv_offset)[p] + (*recv_count)[p];
    send_offset[p + 1] = send_offset[p] + send_count[p];
  }
  free(counts);
  if (!record_size || record_size > INT_MAX)
    ABORT(EXIT_FAILURE);
  if (send_offset[ranks] > SIZE_MAX / record_size)
    ABORT(EXIT_FAILURE);
  size_t limit = (size_t)INT_MAX / record_size;
  if (send_offset[ranks] > limit || (*recv_offset)[ranks] > limit)
    small = 0;
  MPI_Allreduce(MPI_IN_PLACE, &small, 1, MPI_INT, MPI_LAND, comm);
  char* recv = cpu_checked_alloc((*recv_offset)[ranks], record_size);
  char dummy = 0;
  if (!send)
    send = &dummy;
  if (small) {
    int* v = cpu_checked_alloc((size_t)ranks * 4, sizeof(int));
    for (int p = 0; p < ranks; ++p) {
      v[p] = (int)(send_count[p] * record_size);
      v[ranks + p] = (int)(send_offset[p] * record_size);
      v[2 * ranks + p] = (int)((*recv_count)[p] * record_size);
      v[3 * ranks + p] = (int)((*recv_offset)[p] * record_size);
    }
    MPI_Alltoallv(send, v, v + ranks, MPI_BYTE, recv, v + 2 * ranks, v + 3 * ranks, MPI_BYTE, comm);
    free(v);
  } else {
    /* MPI-3 uses int byte counts. Large jobs use bounded rounds; every rank
     * takes the same number of rounds so an empty peer cannot deadlock one. */
    for (int step = 0; step < ranks; ++step) {
      int to = (rank + step) % ranks;
      int from = (rank - step + ranks) % ranks;
      size_t ns = send_count[to], nr = (*recv_count)[from];
      unsigned long long rounds = (ns > nr ? ns : nr) / limit + 1;
      MPI_Allreduce(MPI_IN_PLACE, &rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
      for (unsigned long long r = 0; r < rounds; ++r) {
        size_t off = (size_t)r * limit;
        size_t sc = off < ns ? (ns - off < limit ? ns - off : limit) : 0;
        size_t rc = off < nr ? (nr - off < limit ? nr - off : limit) : 0;
        const char* sp = sc ? (const char*)send + (send_offset[to] + off) * record_size : &dummy;
        char* rp = rc ? recv + ((*recv_offset)[from] + off) * record_size : &dummy;
        MPI_Sendrecv(sp,
                     (int)(sc * record_size),
                     MPI_BYTE,
                     to,
                     29117,
                     rp,
                     (int)(rc * record_size),
                     MPI_BYTE,
                     from,
                     29117,
                     comm,
                     MPI_STATUS_IGNORE);
      }
    }
  }
  free(send_offset);
  return recv;
}
