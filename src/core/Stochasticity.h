#ifndef STOCHASTICITY_H
#define STOCHASTICITY_H

#include <stddef.h>

#include "meraxes.h"
#if !USE_SFR_INTEGRATION
#define NO_SFR_GSM_MIN_COUNT 1
#define NO_SHMR_LOG10_MSTAR_FLOOR (-10.0)
#endif
#define NO_SFR_SFR_MIN_COUNT 1
#define NO_SHMR_LOG10_SFR_FLOOR (-30.0)

// Parameters for removing SFR--halo scatter.
// Source tables use this halo-mass grid; GSM is integrated from SFR when enabled.
#define SFR_NTYPES 3
#define SFR_NX     376
#define SFR_XMIN   (-3.50)
#define SFR_XMAX   (4.00)
#define SFR_DX     ((SFR_XMAX - SFR_XMIN) / ((double)(SFR_NX - 1)))

#define SFR_INDEX(t,i) \
  ((size_t)(t) * (size_t)SFR_NX + (size_t)(i))

#if USE_SFR_INTEGRATION
/* Integrate an FFTW-padded real slab using a right-endpoint snapshot rate.
 * cumulative is double-precision persistent history; stars is the float grid
 * consumed by the ionization solver. Only nx*dim*dim real cells are touched.
 * totals = {previous mass, current rate, actual mass increment, new mass}.
 * All quantities use Meraxes internal units. No fesc or photon-yield factor
 * is applied here: rate is already the final escaped, recalibrated source.
 * Returns 0 on success, -1 on invalid geometry/input, -2 on float overflow.
 * On error the caller must abort the run; updates are not transactional.
 */
int integrate_sfr_source_slab(double* cumulative,
                              const float* rate,
                              float* stars,
                              size_t nx,
                              int dim,
                              double dt,
                              double totals[4]);
#endif

void compute_fesc_recalibration_factors(void);
void fesc_recalibration(void);
double compute_xray_recalibration_factor(double local_xray_raw, double local_xray_target);

void build_no_sfr_tables(int population);
void apply_no_sfr_treatment(int snapshot);
void compute_no_sfr_recalibration_factors(int population);

double extract_recalibration_factors(galaxy_t* gal, int population, bool use_gsm);

void no_sfr_sources_init(void);
void no_sfr_sources_free(void);

#endif