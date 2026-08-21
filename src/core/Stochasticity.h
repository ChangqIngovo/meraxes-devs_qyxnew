#ifndef STOCHASTICITY_H
#define STOCHASTICITY_H

#include "meraxes.h"
#define NO_SHMR_SHMR_MIN_COUNT 3
#define NO_SHMR_SFR_MIN_COUNT 10
#define NO_SHMR_LOG10_MSTAR_FLOOR (-10.0)
#define NO_SHMR_LOG10_SFR_FLOOR (-30.0)

// Parameters for removing stellar-source--halo scatter.
// Both the stellar-mass and SFR tables use this halo-mass grid.
#define SHMR_NTYPES 3
#define SHMR_NX     376
#define SHMR_XMIN   (-3.50)
#define SHMR_XMAX   (4.00)
#define SHMR_DX     ((SHMR_XMAX - SHMR_XMIN) / ((double)(SHMR_NX - 1)))

#define SHMR_INDEX(t,i) \
  ((size_t)(t) * (size_t)SHMR_NX + (size_t)(i))

void compute_fesc_recalibration_factors(void);
void fesc_recalibration(void);

void build_no_shmr_tables(int population);
void apply_no_shmr_treatment(void);
void compute_no_shmr_recalibration_factors(int population);

double extract_recalibration_factors(galaxy_t* gal, int population, bool GSM);

void no_shmr_sources_init(void);
void no_shmr_sources_free(void);

#endif
