#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "fesc_recalibration.h"
#include "lognormalscatter.h"
#include "meraxes.h"
#include "misc_tools.h"

#define FESC_SCATTER_LOOKUP_SIZE 30001

static double
    fesc_scatter_mu_lookup[FESC_SCATTER_LOOKUP_SIZE];

static double fesc_scatter_lookup_sigma_dex = -1.0;


static int fesc_mean_preserving_enabled(void)
{
  physics_params_t* params =
      &(run_globals.params.physics);

  return params->Flag_SourceRecalibration &&
         !params->Flag_RemoveSHMRScatter &&
         params->EscapeFracScatterDex > 0.0;
}


static double normal_cdf(double value)
{
  return 0.5 * erfc(-value / sqrt(2.0));
}


// X = exp(mu + sigma_ln * Z)
// Return the mean after ckipping.
static double clipped_lognormal_mean(
    double mu,
    double sigma_ln)
{
  double below_clip =
      exp(mu + 0.5 * sigma_ln * sigma_ln) *
      normal_cdf(
          (-mu - sigma_ln * sigma_ln) /
          sigma_ln
      );

  double clipped_tail =
      normal_cdf(mu / sigma_ln);

  return below_clip + clipped_tail;
}


// Find mu such that E[min(1, exp(mu + sigma_ln * Z))] = target.
static double solve_clipped_lognormal_mu(
    double target,
    double sigma_ln)
{
  double lower =
      log(target) -
      0.5 * sigma_ln * sigma_ln;

  double upper = log(target);
  double step = fmax(1.0, sigma_ln);

  while (clipped_lognormal_mean(
             upper,
             sigma_ln
         ) < target) {
    upper += step;
  }

  for (int iteration = 0;
       iteration < 100;
       iteration++) {
    double middle =
        0.5 * (lower + upper);

    if (clipped_lognormal_mean(
            middle,
            sigma_ln
        ) < target) {
      lower = middle;
    } else {
      upper = middle;
    }
  }

  return 0.5 * (lower + upper);
}

// Build a uniformly spaced target fesc -> lognormal mu lookup.
static void build_fesc_scatter_lookup(
    double sigma_dex)
{
  double sigma_ln =
      log(10.0) * sigma_dex;

  fesc_scatter_mu_lookup[0] =
      -INFINITY;

  for (int ii = 1;
       ii < FESC_SCATTER_LOOKUP_SIZE - 1;
       ii++) {
    double target =
        (double)ii /
        (double)(FESC_SCATTER_LOOKUP_SIZE - 1);

    fesc_scatter_mu_lookup[ii] =
        solve_clipped_lognormal_mu(
            target,
            sigma_ln
        );
  }

  fesc_scatter_mu_lookup[
      FESC_SCATTER_LOOKUP_SIZE - 1
  ] = INFINITY;

  fesc_scatter_lookup_sigma_dex =
      sigma_dex;

  if (run_globals.mpi_rank == 0) {
    mlog(
        "Built uniform mean-preserving "
        "clipped-lognormal fesc lookup "
        "(sigma=%.6g dex, entries=%d).",
        MLOG_MESG,
        sigma_dex,
        FESC_SCATTER_LOOKUP_SIZE
    );
  }
}


// Return the lognormal centre corresponding to target fesc.
static double get_mean_preserving_scatter_centre(
    double target,
    double sigma_dex)
{
  if (fesc_scatter_lookup_sigma_dex !=sigma_dex) {
    build_fesc_scatter_lookup(
        sigma_dex
    );
  }

  double table_position =
      target *
      (double)(FESC_SCATTER_LOOKUP_SIZE - 1);

  int left =
      (int)floor(table_position);

  double sigma_ln =
      log(10.0) * sigma_dex;

  // The endpoint values have infinite mu.
  // Directly solve inside the first and last intervals.
  if (left <= 0 ||
      left >= FESC_SCATTER_LOOKUP_SIZE - 2) {
    return exp(
        solve_clipped_lognormal_mu(
            target,
            sigma_ln
        )
    );
  }
  //interpolate
  int right =
      left + 1;

  double fraction =
      table_position -
      (double)left;

  double mu =
      fesc_scatter_mu_lookup[left] +
      fraction *
      (
          fesc_scatter_mu_lookup[right] -
          fesc_scatter_mu_lookup[left]
      );

  return exp(mu);
}


void fesc_recalibration_init(void)
{
  physics_params_t* params =
      &(run_globals.params.physics);

  if (!fesc_mean_preserving_enabled())
    return;

  if (fesc_scatter_lookup_sigma_dex !=
      params->EscapeFracScatterDex) {
    build_fesc_scatter_lookup(
        params->EscapeFracScatterDex
    );
  }
}


double fesc_recalibration_apply_scatter(
    double target_fesc,
    unsigned long galaxy_id,
    uint64_t salt)
{
  physics_params_t* params =
      &(run_globals.params.physics);

  double scatter_centre =
      target_fesc;
  //If enabled recalibration, make sure upper and lower bound are 0 or 1, since they are undefined in the table
  if (fesc_mean_preserving_enabled()) {
    
    if (target_fesc == 0.0 ||target_fesc == 1.0)
    return target_fesc;

    scatter_centre =
        get_mean_preserving_scatter_centre(
            target_fesc,
            params->EscapeFracScatterDex
        );
  }
  //If no recalibration then scatter centre=target fesc
  double scattered_fesc =
      apply_lognormal_scatter_from_mean_id(
          scatter_centre,
          params->EscapeFracScatterDex,
          galaxy_id,
          salt
      );
  //clip
  if (scattered_fesc > 1.0)
    scattered_fesc = 1.0;

  return scattered_fesc;
}


void fesc_recalibration_free(void)
{
  fesc_scatter_lookup_sigma_dex = -1.0;
}