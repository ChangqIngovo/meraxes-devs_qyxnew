#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "fesc_recalibration.h"
#include "lognormalscatter.h"
#include "misc_tools.h"

#define FESC_RECALIBRATION_EPS 1.0e-300

enum fesc_global_sum_index
{
  FESC_GSM_RAW = 0,
  FESC_GSM_TARGET,
  FESC_SFR_RAW,
  FESC_SFR_TARGET,
  FESC_GLOBAL_NSUM
};

static double fesc_correction_gsm = 1.0;
static double fesc_correction_sfr = 1.0;
static int fesc_recalibration_initialized = 0;
static int fesc_recalibration_active = 0;
static int fesc_recalibration_snapshot = -1;

static int fesc_recalibration_enabled(void)
{
  physics_params_t* params = &(run_globals.params.physics);

  return params->Flag_SourceRecalibration &&
         !params->Flag_RemoveSHMRScatter &&
         params->EscapeFracScatterDex > 0.0;
}

static int fesc_recalibration_source_eligible(const galaxy_t* gal)
{
  return gal->Type >= 0 && gal->Type <= 2;
}

void fesc_recalibration_init(void)
{
  if (fesc_recalibration_initialized)
    return;

  fesc_correction_gsm = 1.0;
  fesc_correction_sfr = 1.0;
  fesc_recalibration_active = 0;
  fesc_recalibration_snapshot = -1;
  fesc_recalibration_initialized = 1;

  if (run_globals.mpi_rank == 0) {
    mlog("Global fesc source recalibration is %s.", MLOG_MESG,
         fesc_recalibration_enabled() ? "active" : "inactive");
  }
}

double fesc_recalibration_apply_scatter(double target_fesc,
                                        unsigned long galaxy_id,
                                        uint64_t salt)
{
  physics_params_t* params = &(run_globals.params.physics);

  if (params->EscapeFracScatterDex <= 0.0)
    return target_fesc;

  double scattered_fesc = apply_lognormal_scatter_from_mean_id(
      target_fesc,
      params->EscapeFracScatterDex,
      galaxy_id,
      salt
  );

  if (scattered_fesc > 1.0)
    scattered_fesc = 1.0;

  return scattered_fesc;
}

void fesc_recalibration_accumulate_popII(galaxy_t* gal,
                                         double new_stars,
                                         double sfr,
                                         double target_fesc)
{
  if (!fesc_recalibration_enabled())
    return;

  gal->TargetFescWeightedGSM += new_stars * target_fesc;
  gal->TargetFescWeightedSfr += sfr * target_fesc;
}

static double fesc_get_global_correction(double target,
                                         double source,
                                         const char* name,
                                         int snapshot)
{
  if (!isfinite(target) || target < 0.0 ||
      !isfinite(source) || source < 0.0) {
    mlog_error(
        "Invalid global fesc budget at snapshot %d: "
        "%s target=%g source=%g.",
        snapshot,
        name,
        target,
        source
    );
    ABORT(EXIT_FAILURE);
  }

  if (source <= FESC_RECALIBRATION_EPS) {
    if (target <= FESC_RECALIBRATION_EPS)
      return 1.0;

    mlog_error(
        "Cannot recalibrate %s at snapshot %d: "
        "target=%g but source=%g.",
        name,
        snapshot,
        target,
        source
    );
    ABORT(EXIT_FAILURE);
  }

  double correction = target / source;

  if (!isfinite(correction) || correction < 0.0) {
    mlog_error(
        "Invalid global fesc correction at snapshot %d: "
        "%s C=%g.",
        snapshot,
        name,
        correction
    );
    ABORT(EXIT_FAILURE);
  }

  return correction;
}

void fesc_recalibration_prepare(int snapshot)
{
  if (!fesc_recalibration_initialized)
    fesc_recalibration_init();

  fesc_recalibration_snapshot = snapshot;
  fesc_recalibration_active = 0;
  fesc_correction_gsm = 1.0;
  fesc_correction_sfr = 1.0;

  if (!fesc_recalibration_enabled())
    return;

  double local[FESC_GLOBAL_NSUM] = {0.0};
  double global[FESC_GLOBAL_NSUM] = {0.0};
  galaxy_t* gal = run_globals.FirstGal;

  while (gal != NULL) {
    if (fesc_recalibration_source_eligible(gal)) {
      local[FESC_GSM_RAW] += gal->FescWeightedGSM;
      local[FESC_GSM_TARGET] += gal->TargetFescWeightedGSM;
      local[FESC_SFR_RAW] += gal->FescWeightedSfr;
      local[FESC_SFR_TARGET] += gal->TargetFescWeightedSfr;
    }

    gal = gal->Next;
  }

  MPI_Allreduce(
      local,
      global,
      FESC_GLOBAL_NSUM,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

  fesc_correction_gsm = fesc_get_global_correction(
      global[FESC_GSM_TARGET],
      global[FESC_GSM_RAW],
      "FescWeightedGSM",
      snapshot
  );

  fesc_correction_sfr = fesc_get_global_correction(
      global[FESC_SFR_TARGET],
      global[FESC_SFR_RAW],
      "FescWeightedSfr",
      snapshot
  );

  fesc_recalibration_active = 1;

  if (run_globals.mpi_rank == 0) {
    mlog(
        "Global fesc recalibration snapshot=%d: "
        "C_GSM=%.12g C_SFR=%.12g.",
        MLOG_MESG,
        snapshot,
        fesc_correction_gsm,
        fesc_correction_sfr
    );
  }
}

double fesc_recalibration_grid_gsm(const galaxy_t* gal)
{
  if (fesc_recalibration_active &&
      fesc_recalibration_source_eligible(gal)) {
    return fesc_correction_gsm * gal->FescWeightedGSM;
  }

  return gal->FescWeightedGSM;
}

double fesc_recalibration_grid_sfr(const galaxy_t* gal)
{
  if (fesc_recalibration_active &&
      fesc_recalibration_source_eligible(gal)) {
    return fesc_correction_sfr * gal->FescWeightedSfr;
  }

  return gal->FescWeightedSfr;
}

void fesc_recalibration_free(void)
{
  fesc_correction_gsm = 1.0;
  fesc_correction_sfr = 1.0;
  fesc_recalibration_initialized = 0;
  fesc_recalibration_active = 0;
  fesc_recalibration_snapshot = -1;
}