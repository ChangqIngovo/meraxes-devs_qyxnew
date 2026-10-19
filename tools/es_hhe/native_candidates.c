#include "native_candidates.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

int es_candidate_flags(int64_t rows, int64_t groups, const double *aggregate,
    const double *cross, double epsilon, int count_pruning,
    uint8_t *valid, uint8_t *proven_partial, uint8_t *ambiguous, int threads)
{
    if (rows < 0 || groups < 0 || groups > INT64_MAX-5 ||
        threads < 1 || threads > 1024 ||
        !isfinite(epsilon) || epsilon <= 0. || epsilon >= 1. ||
        (count_pruning != 0 && count_pruning != 1) ||
        (rows && (!aggregate || !valid || !proven_partial || !ambiguous)) ||
        (groups && !cross)) return 1;
    if ((uint64_t)rows > SIZE_MAX / sizeof(double) / (uint64_t)(groups+5) ||
        (uint64_t)groups > SIZE_MAX / (3*sizeof(double))) return 1;
    for (int64_t j=0; j<groups*3; ++j)
        if (!isfinite(cross[j]) || cross[j]<0.) return 2;

    /* The NumPy path forms exactly these two/three-term species sums. Its
     * photon-band dot products may use a different BLAS reduction order.
     * Bound both reductions' rounding error near the decision boundary and
     * explicitly delegate those rows back to that original implementation. */
    const double roundoff = 64. * DBL_EPSILON;
    const double count_roundoff = 8. * DBL_EPSILON * ((double)groups + 3.);
    int useful = rows/2048 < threads ? (int)(rows/2048) : threads;
    if (useful < 1) useful = 1;
    int status=0;
    #pragma omp parallel for num_threads(useful) if(useful > 1) schedule(static) reduction(max:status)
    for (int64_t row=0; row<rows; ++row) {
        const double *x=aggregate+row*(groups+5);
        valid[row]=0;proven_partial[row]=0;ambiguous[row]=0;
        int invalid=0;
        for (int64_t j=0; j<groups+5; ++j)
            if (!isfinite(x[j]) || x[j]<0.) invalid=1;
        if (invalid) {status=2;continue;}
        const double hydrogen=x[0]+x[1];
        const double helium=(x[2]+x[3])+x[4];
        if (!isfinite(hydrogen) || !isfinite(helium)) {status=2;continue;}
        const double normal[3]={hydrogen,helium,helium};
        const double remaining[3]={x[0],x[2],x[2]+x[3]};
        double needed[3];
        int eligible[3],any_eligible=0;
        for (int s=0;s<3;++s) {
            double threshold=epsilon*normal[s];
            needed[s]=remaining[s]-threshold;
            eligible[s]=normal[s]>0. && remaining[s]>threshold+roundoff*normal[s];
            if (eligible[s]) any_eligible=1;
        }
        if (!any_eligible) continue;
        int usable=0;
        for (int64_t g=0;g<groups;++g) {
            if (x[5+g]<=0.) continue;
            /* All terms are nonnegative. Positive hazard requires a
             * representable product; Boolean opacity reachability would
             * incorrectly activate underflowed subnormal absorber counts. */
            double hazard=(x[0]*cross[3*g]+x[2]*cross[3*g+1])+x[3]*cross[3*g+2];
            if (!isfinite(hazard)) {invalid=1;break;}
            if (hazard>0.) usable=1;
        }
        if (invalid) {status=2;continue;}
        if (!usable) continue;
        valid[row]=1;
        if (!count_pruning) continue;
        double upper[3]={0.,0.,0.};
        for (int64_t g=0;g<groups;++g)
            for (int s=0;s<3;++s)
                if (cross[3*g+s]>0.) upper[s]+=x[5+g];
        int possible=0;
        for (int s=0;s<3;++s) {
            if (!isfinite(upper[s])) {invalid=1;break;}
            if (!eligible[s]) continue;
            double scale=fmax(upper[s],remaining[s]);
            double left=upper[s]+roundoff*scale;
            if (!isfinite(left)) {invalid=1;break;}
            if (left>=needed[s]) possible=1;
            if (fabs(left-needed[s])<=count_roundoff*scale) ambiguous[row]=1;
        }
        if (invalid) {status=2;continue;}
        proven_partial[row]=!possible;
    }
    return status;
}
