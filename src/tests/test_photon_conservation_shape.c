#include "../physics/photon_conservation_shape.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8
static void check_case(const char* name,
                       size_t n,
                       const double* S,
                       const double* C,
                       const double* b,
                       const double* L,
                       const double* U,
                       const double* R,
                       const int* rank,
                       const double* expected,
                       const double* prior)
{
  pc_problem p = { { n, 1, 1 }, n, S, C, b, L, U, R, rank };
  pc_options fo;
  pc_shape_options so;
  pc_result ref, out;
  pc_shape_metrics m;
  pc_priority_diagnostic rd[N], od[N];
  double ry[N], rr[N], y[N], rem[N], spent[N];
  size_t j;
  pc_status st;
  pc_shape_status ss;
  memset(&ref, 0, sizeof(ref));
  memset(&out, 0, sizeof(out));
  ref.y = ry;
  ref.source_residual = rr;
  ref.priority_diagnostics = rd;
  ref.priority_diagnostics_capacity = N;
  out.y = y;
  out.source_residual = rem;
  out.source_spent = spent;
  out.priority_diagnostics = od;
  out.priority_diagnostics_capacity = N;
  pc_default_options(&fo);
  fo.absolute_tolerance = 1e-12;
  fo.relative_tolerance = 1e-12;
  pc_shape_default_options(&so);
  so.absolute_tolerance = 1e-12;
  so.relative_tolerance = 1e-11;
  so.max_hall_cuts = 128;
  st = pc_allocate(&p, &fo, &ref);
  assert(st == PC_OK || st == PC_PARTIAL);
  ss = pc_shape_refine(&p, &fo, &ref, prior, &so, &out, &m);
  if (ss != PC_SHAPE_OK)
    fprintf(stderr,
            "%s: %s cuts=%zu sweeps=%zu eval=%zu hall=%.3g pri=%.3g gap=%.3g\n",
            name,
            pc_shape_status_string(ss),
            m.hall_cuts,
            m.master_sweeps,
            m.master_evaluations,
            m.max_hall_violation,
            m.max_priority_violation,
            m.signed_primal_dual_gap);
  assert(ss == PC_SHAPE_OK);
  assert(m.full_graph_certificate);
  for (j = 0; j < n; j++) {
    assert(fabs(y[j] - expected[j]) < (expected[j] > 0. ? 2e-7 * expected[j] : 2e-8));
    assert(y[j] >= L[j] - 1e-10 && y[j] <= U[j] + 1e-10);
    assert(spent[j] >= -1e-10 && spent[j] <= S[j] + 1e-10);
    assert(fabs(spent[j] + rem[j] - S[j]) < 1e-10);
  }
  if (m.target_total > 0) {
    so.max_bytes = 1;
    assert(pc_shape_refine(&p, &fo, &ref, prior, &so, &out, &m) == PC_SHAPE_ERR_MEMORY);
  }
  printf("PASS %s\n", name);
}
/* Sparse deterministic replay of the first promoted-sphere regression.
 * All unlisted cells have zero source and zero target bounds. This retains
 * the 32^3 geometry in a compact fixture without committing simulation data. */
static void check_promoted_priority_regression(void)
{
  static const double cells[][8] = {
    { 4004,
      0.048104238585484405,
      1.7334089279174805,
      0.048104190583899253,
      0,
      1.7334089279174805,
      1.5142047679068233,
      29 },
    { 4182,
      0.18628884270008214,
      2.0173184871673584,
      0.18628879334607973,
      0,
      2.0173184871673584,
      1.5142047679068233,
      29 },
    { 9056,
      0.22040885766496701,
      2.3064892292022705,
      0.22040883099253961,
      0,
      2.3064892292022705,
      1.5142047679068233,
      29 },
    { 13340,
      0.063520651580424803,
      1.8407604694366455,
      0.06352061450350277,
      0,
      1.8407604694366455,
      1.5142047679068233,
      29 },
    { 18263,
      0.17272301710600707,
      1.2918959856033325,
      0.17272296813936805,
      0,
      1.2918959856033325,
      1.5142047679068233,
      29 },
    { 18325,
      0.059068988960723878,
      2.8159701824188232,
      0.05906898512947123,
      0,
      2.8159701824188232,
      1.5142047679068233,
      29 },
    { 18326,
      0.70562905605039095,
      2.5096638202667236,
      0.70562903915268294,
      0,
      2.5096638202667236,
      1.5142047679068233,
      29 },
    { 18429,
      0.97676454091446574,
      1.5967099666595459,
      0.97676439647254654,
      0,
      1.5967099666595459,
      1.5142047679068233,
      29 },
    { 19224,
      0.29741486726221461,
      1.9797728061676025,
      0.29741487763037355,
      0,
      1.9797728061676025,
      1.5142047679068233,
      29 },
    { 21113,
      0.52939743897498259,
      1.9303079843521118,
      0.52939737859584568,
      0,
      1.9303079843521118,
      1.5142047679068233,
      29 },
    { 21176, 1.7931462065782835, 2.8425421714782715, 1.793146109054419, 0, 2.8425421714782715, 1.5142047679068233, 29 },
    { 22184,
      0.23162033077686348,
      1.2170200347900391,
      0.23162036961821286,
      0,
      1.2170200347900391,
      1.5142047679068233,
      29 },
    { 22336,
      0.90912334195030864,
      2.2416765689849854,
      0.90912329661144042,
      0,
      2.2416765689849854,
      1.5142047679068233,
      29 },
    { 23361,
      1.9359394501698397,
      2.6265790462493896,
      1.9359396473108177,
      0,
      2.6265790462493896,
      1.5142047679068233,
      29 },
    { 23497,
      0.028034613550573909,
      2.1179149150848389,
      0.028034586221707514,
      0,
      2.1179149150848389,
      1.5142047679068233,
      29 },
    { 24215,
      0.057919193723524158,
      2.6049652099609375,
      0.057919165931707539,
      0,
      2.6049652099609375,
      1.5142047679068233,
      29 },
    { 25434, 0, 1.4220728874206543, 0, 0, 1.4220728874206543, 1.5142047679068233, 29 },
    { 26426, 0, 1.27715003490448, 0, 0, 1.27715003490448, 1.5142047679068233, 29 },
    { 26457, 0, 1.4053134918212891, 0, 0, 1.4053134918212891, 1.5142047679068233, 29 },
    { 26458, 1.774446476130098, 1.6841491460800171, 1.6841491460800171, 0, 1.6841491460800171, 1.5142047679068233, 28 },
    { 26459, 0, 2.6317934989929199, 0, 0, 2.6317934989929199, 1.5142047679068233, 29 },
    { 26490, 0, 1.6964895725250244, 0, 0, 1.6964895725250244, 1.5142047679068233, 29 },
    { 26920,
      0.053319235132854678,
      1.4961155652999878,
      0.053319242126391941,
      0,
      1.4961155652999878,
      1.5142047679068233,
      29 },
    { 27339,
      0.53414873067869029,
      1.8869491815567017,
      0.53414865477602547,
      0,
      1.8869491815567017,
      1.5142047679068233,
      29 },
    { 27482, 0, 1.4926151037216187, 0, 0, 1.4926151037216187, 1.5142047679068233, 29 },
    { 27515,
      0.095439666375017773,
      2.3336267471313477,
      0.095439612368579674,
      0,
      2.3336267471313477,
      1.5142047679068233,
      29 },
    { 27559,
      0.38319307766190469,
      1.881417989730835,
      0.38319304655406938,
      0,
      1.881417989730835,
      1.5142047679068233,
      29 },
    { 27768,
      0.074783615931717728,
      2.7653803825378418,
      0.074783645611915972,
      0,
      2.7653803825378418,
      1.5142047679068233,
      29 },
    { 32355,
      0.23710296629094635,
      1.1752121448516846,
      0.23710294994178582,
      0,
      1.1752121448516846,
      1.5142047679068233,
      29 },
    { 32384,
      0.16010786370499949,
      2.7277264595031738,
      0.16010785647867465,
      0,
      2.7277264595031738,
      1.5142047679068233,
      29 },
  };
  size_t n = 32 * 32 * 32, j, k;
  double* data = calloc(10 * n, sizeof(double));
  int* rank = calloc(n, sizeof(int));
  double *S = data, *C = data + n, *b = data + 2 * n, *L = data + 3 * n, *U = data + 4 * n, *R = data + 5 * n;
  pc_problem p = { { 32, 32, 32 }, n, S, C, b, L, U, R, rank };
  pc_result ref = { 0 }, out = { 0 };
  pc_priority_diagnostic diagnostic[32];
  pc_options fo;
  pc_shape_options so;
  pc_shape_metrics m;
  assert(data && rank);
  for (k = 0; k < sizeof(cells) / sizeof(cells[0]); k++) {
    j = (size_t)cells[k][0];
    S[j] = cells[k][1];
    C[j] = cells[k][2];
    b[j] = cells[k][3];
    L[j] = cells[k][4];
    U[j] = cells[k][5];
    R[j] = cells[k][6];
    rank[j] = (int)cells[k][7];
  }
  ref.y = data + 6 * n;
  ref.source_residual = data + 7 * n;
  ref.priority_diagnostics = diagnostic;
  ref.priority_diagnostics_capacity = 32;
  out.y = data + 8 * n;
  out.source_residual = data + 9 * n;
  pc_default_options(&fo);
  fo.absolute_tolerance = 1e-12;
  fo.relative_tolerance = 1e-9;
  pc_shape_default_options(&so);
  so.max_hall_cuts = 128;
  so.max_master_sweeps = 10000;
  so.relative_tolerance = 1e-9;
  assert(pc_allocate(&p, &fo, &ref) == PC_OK);
  assert(pc_shape_refine(&p, &fo, &ref, NULL, &so, &out, &m) == PC_SHAPE_OK);
  assert(m.full_graph_certificate && out.metrics.full_source_budget);
  assert(m.max_priority_equality_error <= m.tolerance);
  assert(fabs(m.signed_primal_dual_gap) <= m.tolerance);
  assert(m.equality_error <= 64 * 2.220446049250313e-16 * ref.metrics.absorbed);
  free(data);
  free(rank);
  puts("PASS promoted sparse priority regression");
}

int main(void)
{
  {
    double S[] = { 1, 0 }, C[] = { 1, 1 }, b[] = { 0, 0 }, L[] = { 0, 0 }, U[] = { 1, 1 }, R[] = { 1, 1 },
           e[] = { .5, .5 };
    check_case("symmetric complete", 2, S, C, b, L, U, R, NULL, e, NULL);
  }
  {
    double S[] = { 3, 0 }, C[] = { 1, 1 }, b[] = { 0, 0 }, L[] = { 0, 0 }, U[] = { 1, 1 }, R[] = { 1, 1 },
           e[] = { 1, 1 };
    check_case("partial absorption", 2, S, C, b, L, U, R, NULL, e, NULL);
  }
  {
    double S[] = { .9, 0, 0 }, C[] = { 1, 1, 1 }, b[] = { .8, .8, .8 }, L[] = { 0, 0, 0 }, U[] = { 1, 1, 1 },
           R[] = { 3, 3, 3 }, e[] = { .8, .05, .05 };
    int rank[] = { 0, 1, 1 };
    check_case("priority retained face", 3, S, C, b, L, U, R, rank, e, NULL);
  }
  {
    double S[] = { 0, 0 }, C[] = { 1, 0 }, b[] = { .8, 0 }, L[] = { 0, 0 }, U[] = { 1, 0 }, R[] = { 1, 1 },
           e[] = { 0, 0 };
    int rank[] = { 0, 12 };
    check_case("zero source absent rank", 2, S, C, b, L, U, R, rank, e, NULL);
  }
  {
    double S[] = { 1, 0 }, C[] = { 0, 0 }, b[] = { 0, 0 }, L[] = { 0, 0 }, U[] = { 0, 0 }, R[] = { 1, 1 },
           e[] = { 0, 0 };
    int rank[] = { 0, 12 };
    check_case("zero target capacity", 2, S, C, b, L, U, R, rank, e, NULL);
  }
  {
    double S[] = { 1.5, 0, .5 }, C[] = { 1, 1, 2 }, b[] = { .3, 1, .7 }, L[] = { 0, 0, 0 }, U[] = { 1, 1, 2 },
           R[] = { 0, 0, 3 }, e[] = { .3 + 1. / 3., 0, .7 + 2. / 3. };
    check_case("mixed expansion contraction Hall repair", 3, S, C, b, L, U, R, NULL, e, NULL);
  }
  {
    double S[] = { 1.2, 0, 0 }, C[] = { 1, 1, 0 }, b[] = { .2, .7, 0 }, L[] = { .5, 0, 0 }, U[] = { 1, 1, 0 },
           R[] = { 3, 3, 3 }, e[] = { .5, .7, 0 };
    int rank[] = { 0, 1, 12 };
    check_case("lower above baseline absent fixed rank", 3, S, C, b, L, U, R, rank, e, NULL);
  }
  {
    double S[] = { 1e-15, 0 }, C[] = { 1, 1 }, b[] = { 0, 0 }, L[] = { 0, 0 }, U[] = { 1, 1 }, R[] = { 1, 1 },
           e[] = { 5e-16, 5e-16 };
    check_case("faint inventory preserved", 2, S, C, b, L, U, R, NULL, e, NULL);
  }
  {
    double S[] = { 1, 0 }, C[] = { 1, 1 }, b[] = { 0, 0 }, L[] = { 0, 0 }, U[] = { 1, 1 }, R[] = { 0, 0 },
           e[] = { 1, 0 }, prior[] = { 0, 1 };
    check_case("zero candidate original neighbor", 2, S, C, b, L, U, R, NULL, e, prior);
  }
  check_promoted_priority_regression();
  puts("All native unique-shape cases passed.");
  return 0;
}
