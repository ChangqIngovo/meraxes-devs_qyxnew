#ifndef LOGNORMALSCATTER_H
#define LOGNORMALSCATTER_H

#include <stdint.h>

double apply_lognormal_scatter_from_mean_id(double mean_esc,
                                            double scatter_dex,
                                            unsigned long galaxy_id,
                                            uint64_t salt);

#endif