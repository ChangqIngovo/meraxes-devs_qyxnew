#ifndef FESC_RECALIBRATION_H
#define FESC_RECALIBRATION_H

struct galaxy_t;

#if USE_SCATTERS
void fesc_recalibration_init(void);

void fesc_accumulate_target_popII(struct galaxy_t* gal,
                                  double new_stars,
                                  double sfr,
                                  double target_fesc);

#if USE_MINI_HALOS
void fesc_accumulate_target_popIII(struct galaxy_t* gal,
                                   double new_stars,
                                   double sfr,
                                   double target_fesc);
#endif

void fesc_recalibration_prepare(int snapshot);

double fesc_recalibration_grid_gsm(const struct galaxy_t* gal);
double fesc_recalibration_grid_sfr(const struct galaxy_t* gal);

#if USE_MINI_HALOS
double fesc_recalibration_grid_gsm_popIII(const struct galaxy_t* gal);
double fesc_recalibration_grid_sfr_popIII(const struct galaxy_t* gal);
#endif

void fesc_recalibration_free(void);
#endif

#endif