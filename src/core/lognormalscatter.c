#include <math.h>
#include <stdint.h>
#include <gsl/gsl_cdf.h>

#include "lognormalscatter.h"

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

static inline double u01_from_u64(uint64_t x)
{
    return ((x >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

double apply_lognormal_scatter_from_mean_id(double mean_esc,
                                            double scatter_dex,
                                            unsigned long galaxy_id,
                                            uint64_t salt)
{
    double sigma_ln, zeta;
    double u, g, val;
    uint64_t h;

    if (scatter_dex <= 0.0)
        return mean_esc;

    if (mean_esc <= 0.0)
        return mean_esc;

    sigma_ln = log(10.0) * scatter_dex;
    zeta     = log(mean_esc) - 0.5 * sigma_ln * sigma_ln;

    h = splitmix64(((uint64_t)galaxy_id) ^ salt);
    u = u01_from_u64(h);
    g = gsl_cdf_ugaussian_Pinv(u);

    val = exp(zeta + sigma_ln * g);
    return val;
}