#ifndef FESC_RECALIBRATION_H
#define FESC_RECALIBRATION_H

#include <stdint.h>

#include "meraxes.h"

#ifdef __cplusplus
extern "C" {
#endif

void fesc_recalibration_init(void);

double fesc_recalibration_apply_scatter(double target_fesc,
                                        unsigned long galaxy_id,
                                        uint64_t salt);

void fesc_recalibration_accumulate_popII(galaxy_t* gal,
                                         double new_stars,
                                         double sfr,
                                         double target_fesc);

void fesc_recalibration_prepare(int snapshot);

double fesc_recalibration_grid_gsm(const galaxy_t* gal);
double fesc_recalibration_grid_sfr(const galaxy_t* gal);

void fesc_recalibration_free(void);

#ifdef __cplusplus
}
#endif

#endif