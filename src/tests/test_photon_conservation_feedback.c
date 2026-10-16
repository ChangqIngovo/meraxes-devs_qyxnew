#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define _MAIN
#include "core/photon_conservation_feedback.h"
#include "core/reionization.h"
#include "meraxes.h"

/* This test links the production helper, filter, hubble normalization and
 * MPI/FFTW backend. Analytic constant-plus-cosine fields provide independent
 * expected values. Spin-temperature mode supplies a prescribed neutral-gas
 * temperature, so no external RECFAST table or simulation data is required.
 */
static void check(double a, double b, int* bad)
{
  if (fabs(a - b) > 3e-5 * fmax(1.0, fabs(b))) {
    fprintf(stderr, "rank%d mismatch %.12g != %.12g\n", run_globals.mpi_rank, a, b);
    *bad = 1;
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  fftwf_mpi_init();
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  init_mlog(MPI_COMM_WORLD, stdout, stderr, stderr);
  const int dim = 4, rank = run_globals.mpi_rank, ranks = run_globals.mpi_size;
  if (ranks < 1 || ranks > dim || dim % ranks) {
    if (!rank)
      fprintf(stderr, "Use 1, 2 or 4 MPI ranks.\n");
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return 77;
  }
  ptrdiff_t nx, start;
  ptrdiff_t nc = fftwf_mpi_local_size_3d(dim, dim, dim / 2 + 1, MPI_COMM_WORLD, &nx, &start);
  ptrdiff_t *allnx = calloc((size_t)ranks, sizeof(*allnx)), *allstart = calloc((size_t)ranks, sizeof(*allstart)),
            *allnc = calloc((size_t)ranks, sizeof(*allnc));
  MPI_Allgather(&nx, sizeof(nx), MPI_BYTE, allnx, sizeof(nx), MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(&start, sizeof(start), MPI_BYTE, allstart, sizeof(start), MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(&nc, sizeof(nc), MPI_BYTE, allnc, sizeof(nc), MPI_BYTE, MPI_COMM_WORLD);
  reion_grids_t* g = &run_globals.reion_grids;
  g->slab_nix = allnx;
  g->slab_ix_start = allstart;
  g->slab_n_complex = allnc;
  size_t nr = (size_t)nx * dim * dim;
  float *xh = calloc(nr, sizeof(float)), *oldz = calloc(nr, sizeof(float)), *oldj = calloc(nr, sizeof(float));
  int* slot = calloc(nr, sizeof(int));
  g->xH = xh;
  g->z_at_ionization = calloc(nr, sizeof(float));
  g->J_21_at_ionization = calloc(nr, sizeof(float));
  g->Gamma12 = calloc(nr, sizeof(float));
  g->r_bubble = calloc(nr, sizeof(float));
  g->temp_kinetic_all_gas = calloc(nr, sizeof(float));
  g->Tk_box = calloc(nr, sizeof(float));
  g->deltax = calloc((size_t)2 * nc, sizeof(float));
  for (size_t j = 0; j < nr; ++j) {
    xh[j] = 1;
    oldz[j] = -1;
    slot[j] = -1;
    g->Tk_box[j] = 100;
  }
  for (int x = 0; x < nx; ++x) {
    size_t i = (size_t)x * dim * dim;
    int gx = (int)start + x;
    if (gx == 0) {
      xh[i] = 0;
      slot[i] = 0;
    }
    if (gx == 1) {
      xh[i] = .2f;
      oldj[i] = 7;
    }
    if (gx == 2) {
      xh[i] = 0;
      slot[i] = 1;
      oldz[i] = 8;
      oldj[i] = 5;
    }
  }
  const float constant[6] = { 2, 3, 5, 7, 11, 13 }, amplitude[6] = { .1f, .2f, .3f, .5f, .4f, .6f };
  fftwf_complex *input[6], *output[6];
  fftwf_plan inverse[6];
  for (int f = 0; f < 6; ++f) {
    float* real = fftwf_alloc_real((size_t)2 * nc);
    input[f] = fftwf_alloc_complex((size_t)nc);
    output[f] = fftwf_alloc_complex((size_t)nc);
    fftwf_plan forward = fftwf_mpi_plan_dft_r2c_3d(dim, dim, dim, real, input[f], MPI_COMM_WORLD, FFTW_ESTIMATE);
    inverse[f] = fftwf_mpi_plan_dft_c2r_3d(dim, dim, dim, output[f], (float*)output[f], MPI_COMM_WORLD, FFTW_ESTIMATE);
    memset(real, 0, (size_t)2 * nc * sizeof(float));
    for (int x = 0; x < nx; ++x)
      for (int y = 0; y < dim; ++y)
        for (int z = 0; z < dim; ++z)
          real[z + 2 * (dim / 2 + 1) * (y + dim * x)] =
            constant[f] + amplitude[f] * (float)cos(2 * M_PI * (start + x) / dim);
    fftwf_execute(forward);
    for (ptrdiff_t j = 0; j < nc; ++j)
      input[f][j] /= (float)(dim * dim * dim);
    fftwf_destroy_plan(forward);
    fftwf_free(real);
  }
#define ASSIGN(x, f)                                                                                                   \
  do {                                                                                                                 \
    g->x##_unfiltered = input[f];                                                                                      \
    g->x##_filtered = output[f];                                                                                       \
    g->x##_filtered_reverse_plan = inverse[f];                                                                         \
  } while (0)
  ASSIGN(stars, 0);
  ASSIGN(weighted_sfr, 1);
  ASSIGN(effective_bhm, 2);
  ASSIGN(effective_bhar, 3);
#if USE_MINI_HALOS
  ASSIGN(starsIII, 4);
  ASSIGN(weighted_sfrIII, 5);
#endif
#undef ASSIGN
  double zz[1] = { 7.0 }, radii[2] = { 3, 0 };
  run_globals.ZZ = zz;
  run_globals.units.UnitLength_in_cm = run_globals.units.UnitMass_in_g = run_globals.units.UnitTime_in_s = 1;
  run_globals.params.ReionGridDim = dim;
  run_globals.params.BoxSize = 8;
  run_globals.params.SnaplistLength = 1;
  run_globals.params.Flag_IncludeSpinTemp = 1;
  run_globals.params.OmegaLambda = 1;
  run_globals.Hubble = 0.5;
  run_globals.params.ReionSfrTimescale = 3;
  run_globals.params.ReionUVBFlag = 1;
  run_globals.params.Flag_IncludeRecombinations = 1;
  run_globals.params.ReionFilterType = 0;
  run_globals.params.physics.Flag_BHFeedback = 1;
  run_globals.params.physics.ReionAlphaUV = 2;
  run_globals.params.physics.ReionAlphaUVBH = 3;
  run_globals.params.physics.ReionNionPhotPerBary = 4;
  run_globals.params.physics.ReionNionPhotPerBaryIII = 8;
  run_globals.params.physics.ReionGammaHaloBias = 2;
  photon_conservation_update_feedback(0, radii, 2, slot, oldz, oldj);
  int bad = 0;
  for (int x = 0; x < nx; ++x) {
    size_t i = (size_t)x * dim * dim;
    int gx = (int)start + x;
    if (gx == 1) {
      check(g->Gamma12[i], 0, &bad);
      check(g->r_bubble[i], 0, &bad);
      check(g->temp_kinetic_all_gas[i], 8020, &bad);
      check(g->J_21_at_ionization[i], 7, &bad);
    }
    if (gx != 0 && gx != 2)
      continue;
    double r = gx == 0 ? 3.0 : 2.0 * L_FACTOR, kr = 2 * M_PI / 8.0 * r;
    double w = gx == 0 ? 3 * (sin(kr) / (kr * kr * kr) - cos(kr) / (kr * kr)) : 1.0;
    double mu = cos(2 * M_PI * gx / dim), v[6];
    for (int f = 0; f < 6; ++f)
      v[f] = constant[f] + amplitude[f] * mu * w;
    double rate = 2.0 / (2.0 + 2.75) * v[1] + 3.0 / (3.0 + 2.75) * v[3], mass = 2.0 * v[0] + 3.0 * v[2];
#if USE_MINI_HALOS
    rate += 2.0 / (2.0 + 2.75) * 2.0 * v[5];
    mass += 2.0 * 2.0 * v[4];
#endif
    double base = 64 * r / 8.0 / PROTONMASS * 4;
    check(g->Gamma12[i], base * SIGMA_HI * 1e12 * rate, &bad);
    check(g->r_bubble[i], r, &bad);
    check(g->z_at_ionization[i], gx == 0 ? 7 : 8, &bad);
    check(g->J_21_at_ionization[i], gx == 0 ? base * PLANCK * 1e21 / (4 * M_PI) * 2 / 6 * mass : 5, &bad);
  }
  run_globals.params.ReionUVBFlag = 2;
  photon_conservation_update_feedback(0, radii, 2, slot, oldz, oldj);
  for (int x = 0; x < nx; ++x) {
    size_t i = (size_t)x * dim * dim;
    int gx = (int)start + x;
    if (gx == 1) {
      check(g->J_21_at_ionization[i], 7, &bad);
      check(g->Gamma12[i], 0, &bad);
    }
    if (gx == 2) {
      check(g->z_at_ionization[i], 8, &bad);
      if (g->J_21_at_ionization[i] <= 5)
        bad = 1;
    }
  }
  run_globals.params.ReionUVBFlag = 0;
  run_globals.params.Flag_IncludeRecombinations = 0;
  photon_conservation_update_feedback(0, radii, 2, slot, oldz, NULL);
  for (int x = 0; x < nx; ++x) {
    size_t i = (size_t)x * dim * dim;
    int gx = (int)start + x;
    if (gx == 0)
      check(g->z_at_ionization[i], 7, &bad);
    if (gx == 2)
      check(g->r_bubble[i], 2.0 * L_FACTOR, &bad);
  }
  MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (!rank)
    printf("production MPI/FFTW feedback ranks=%d status=%d\n", ranks, bad);
  for (int f = 0; f < 6; ++f) {
    fftwf_destroy_plan(inverse[f]);
    fftwf_free(input[f]);
    fftwf_free(output[f]);
  }
  free(allnx);
  free(allstart);
  free(allnc);
  free(xh);
  free(oldz);
  free(oldj);
  free(slot);
  free(g->z_at_ionization);
  free(g->J_21_at_ionization);
  free(g->Gamma12);
  free(g->r_bubble);
  free(g->temp_kinetic_all_gas);
  free(g->Tk_box);
  free(g->deltax);
  fftwf_mpi_cleanup();
  MPI_Finalize();
  return bad;
}
