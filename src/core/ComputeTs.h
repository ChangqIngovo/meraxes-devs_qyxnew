#ifndef COMPUTE_TS_H
#define COMPUTE_TS_H

#include "utils.h"

#define R_XLy_MAX (float)(500)
// Tolerance for "spectral index == 1", where the power-law luminosity
// conversion switches to its logarithmic special case (GAL/III/AGN soft/hard).
#define SPEC_INDEX_UNITY_TOL (1e-6)

#ifdef __cplusplus
extern "C"
{
#endif

  void ComputeTs(int snapshot, timer_info* timer_total);

#ifdef __cplusplus
}
#endif

#endif
