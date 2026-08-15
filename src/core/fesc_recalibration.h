#ifndef FESC_RECALIBRATION_H
#define FESC_RECALIBRATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void fesc_recalibration_init(void);

  double fesc_recalibration_apply_scatter(
      double target_fesc,
      unsigned long galaxy_id,
      uint64_t salt);

  void fesc_recalibration_free(void);

#ifdef __cplusplus
}
#endif

#endif