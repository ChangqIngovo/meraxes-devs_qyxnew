#ifndef STOCHASTICITY_H
#define STOCHASTICITY_H

struct galaxy_t;

#if USE_STOCHASTICITY
void fesc_recalibration(int snapshot);

void no_shmr_sources_prepare(int snapshot);
double no_shmr_sources_grid_gsm(const struct galaxy_t* gal);
double no_shmr_sources_grid_sfr(const struct galaxy_t* gal);
double no_shmr_sources_grid_sfr_source(const struct galaxy_t* gal);
#if USE_MINI_HALOS
double no_shmr_sources_grid_gsm_popIII(const struct galaxy_t* gal);
double no_shmr_sources_grid_sfr_popIII(const struct galaxy_t* gal);
double no_shmr_sources_grid_sfr_source_popIII(const struct galaxy_t* gal);
#endif
void no_shmr_sources_free(void);

void init_reion_source_tables(void);
#endif

#endif
