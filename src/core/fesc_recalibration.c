#include <math.h>
#include <stdlib.h>

#include "fesc_recalibration.h"
#include "meraxes.h"
#include "misc_tools.h"

#if USE_SCATTERS

#define FESC_RECALIBRATION_EPS 1.0e-300

enum fesc_global_sum_index
{
  FESC_POPII_GSM_RAW = 0,
  FESC_POPII_GSM_TARGET,
  FESC_POPII_SFR_RAW,
  FESC_POPII_SFR_TARGET,
#if USE_MINI_HALOS
  FESC_POPIII_GSM_RAW,
  FESC_POPIII_GSM_TARGET,
  FESC_POPIII_SFR_RAW,
  FESC_POPIII_SFR_TARGET,
#endif
  FESC_GLOBAL_NSUM
};

enum fesc_correction_index
{
  FESC_CORRECTION_POPII_GSM = 0,
  FESC_CORRECTION_POPII_SFR,
#if USE_MINI_HALOS
  FESC_CORRECTION_POPIII_GSM,
  FESC_CORRECTION_POPIII_SFR,
#endif
  FESC_CORRECTION_N
};

static double fesc_correction_popII_gsm = 1.0;
static double fesc_correction_popII_sfr = 1.0;
#if USE_MINI_HALOS
static double fesc_correction_popIII_gsm = 1.0;
static double fesc_correction_popIII_sfr = 1.0;
#endif
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

  fesc_correction_popII_gsm = 1.0;
  fesc_correction_popII_sfr = 1.0;
#if USE_MINI_HALOS
  fesc_correction_popIII_gsm = 1.0;
  fesc_correction_popIII_sfr = 1.0;
#endif
  fesc_recalibration_active = 0;
  fesc_recalibration_snapshot = -1;
  fesc_recalibration_initialized = 1;

  if (run_globals.mpi_rank == 0) {
    mlog("Global fesc source recalibration is %s.", MLOG_MESG,
         fesc_recalibration_enabled() ? "active" : "inactive");
  }
}

void fesc_accumulate_target_popII(galaxy_t* gal,
                                  double new_stars,
                                  double sfr,
                                  double target_fesc)
{
  if (!fesc_recalibration_enabled())
    return;

  gal->TargetFescWeightedGSM += new_stars * target_fesc;
  gal->TargetFescWeightedSfr += sfr * target_fesc;
}

#if USE_MINI_HALOS
void fesc_accumulate_target_popIII(galaxy_t* gal,
                                  double new_stars,
                                  double sfr,
                                  double target_fesc)
{
  if (!fesc_recalibration_enabled())
    return;

  gal->TargetFescIIIWeightedGSM += new_stars * target_fesc;
  gal->TargetFescIIIWeightedSfr += sfr * target_fesc;
}
#endif

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
  fesc_correction_popII_gsm = 1.0;
  fesc_correction_popII_sfr = 1.0;
#if USE_MINI_HALOS
  fesc_correction_popIII_gsm = 1.0;
  fesc_correction_popIII_sfr = 1.0;
#endif

  if (!fesc_recalibration_enabled())
    return;

  double local[FESC_GLOBAL_NSUM] = {0.0};
  double global[FESC_GLOBAL_NSUM] = {0.0};
  double corrections[FESC_CORRECTION_N];
  galaxy_t* gal = run_globals.FirstGal;

  for (int ii = 0; ii < FESC_CORRECTION_N; ii++)
    corrections[ii] = 1.0;

  while (gal != NULL) {
    //Galaxies retain cumulative Pop III source history after transitioning 
    // to Pop II, and mergers can transfer that history to a Pop II parent.
    if (fesc_recalibration_source_eligible(gal)) {
      local[FESC_POPII_GSM_RAW] += gal->FescWeightedGSM;
      local[FESC_POPII_GSM_TARGET] += gal->TargetFescWeightedGSM;
      local[FESC_POPII_SFR_RAW] += gal->FescWeightedSfr;
      local[FESC_POPII_SFR_TARGET] += gal->TargetFescWeightedSfr;
#if USE_MINI_HALOS
      local[FESC_POPIII_GSM_RAW] += gal->FescIIIWeightedGSM;
      local[FESC_POPIII_GSM_TARGET] += gal->TargetFescIIIWeightedGSM;
      local[FESC_POPIII_SFR_RAW] += gal->FescIIIWeightedSfr;
      local[FESC_POPIII_SFR_TARGET] += gal->TargetFescIIIWeightedSfr;
#endif
    }

    gal = gal->Next;
  }

  MPI_Reduce(
      local,
      global,
      FESC_GLOBAL_NSUM,
      MPI_DOUBLE,
      MPI_SUM,
      0,
      run_globals.mpi_comm
  );

  if (run_globals.mpi_rank == 0) {
    corrections[FESC_CORRECTION_POPII_GSM] = fesc_get_global_correction(
        global[FESC_POPII_GSM_TARGET],
        global[FESC_POPII_GSM_RAW],
        "PopII FescWeightedGSM",
        snapshot
    );

    corrections[FESC_CORRECTION_POPII_SFR] = fesc_get_global_correction(
        global[FESC_POPII_SFR_TARGET],
        global[FESC_POPII_SFR_RAW],
        "PopII FescWeightedSfr",
        snapshot
    );
#if USE_MINI_HALOS
    corrections[FESC_CORRECTION_POPIII_GSM] = fesc_get_global_correction(
        global[FESC_POPIII_GSM_TARGET],
        global[FESC_POPIII_GSM_RAW],
        "PopIII FescWeightedGSM",
        snapshot
    );

    corrections[FESC_CORRECTION_POPIII_SFR] = fesc_get_global_correction(
        global[FESC_POPIII_SFR_TARGET],
        global[FESC_POPIII_SFR_RAW],
        "PopIII FescWeightedSfr",
        snapshot
    );
#endif
  }

  MPI_Bcast(
      corrections,
      FESC_CORRECTION_N,
      MPI_DOUBLE,
      0,
      run_globals.mpi_comm
  );

  fesc_correction_popII_gsm = corrections[FESC_CORRECTION_POPII_GSM];
  fesc_correction_popII_sfr = corrections[FESC_CORRECTION_POPII_SFR];
#if USE_MINI_HALOS
  fesc_correction_popIII_gsm = corrections[FESC_CORRECTION_POPIII_GSM];
  fesc_correction_popIII_sfr = corrections[FESC_CORRECTION_POPIII_SFR];
#endif

  fesc_recalibration_active = 1;

  if (run_globals.mpi_rank == 0) {
#if USE_MINI_HALOS
    mlog(
        "Global fesc recalibration snapshot=%d: "
        "C_GSM_II=%.12g C_SFR_II=%.12g "
        "C_GSM_III=%.12g C_SFR_III=%.12g.",
        MLOG_MESG,
        snapshot,
        fesc_correction_popII_gsm,
        fesc_correction_popII_sfr,
        fesc_correction_popIII_gsm,
        fesc_correction_popIII_sfr
    );
#else
    mlog(
        "Global fesc recalibration snapshot=%d: "
        "C_GSM=%.12g C_SFR=%.12g.",
        MLOG_MESG,
        snapshot,
        fesc_correction_popII_gsm,
        fesc_correction_popII_sfr
    );
#endif
  }
}

double fesc_recalibration_grid_gsm(const galaxy_t* gal)
{
  if (fesc_recalibration_active &&
      fesc_recalibration_source_eligible(gal)) {
    return fesc_correction_popII_gsm * gal->FescWeightedGSM;
  }

  return gal->FescWeightedGSM;
}

double fesc_recalibration_grid_sfr(const galaxy_t* gal)
{
  if (fesc_recalibration_active &&
      fesc_recalibration_source_eligible(gal)) {
    return fesc_correction_popII_sfr * gal->FescWeightedSfr;
  }

  return gal->FescWeightedSfr;
}

#if USE_MINI_HALOS
double fesc_recalibration_grid_gsm_popIII(const galaxy_t* gal)
{
  if (fesc_recalibration_active &&
      fesc_recalibration_source_eligible(gal)) {
    return fesc_correction_popIII_gsm * gal->FescIIIWeightedGSM;
  }

  return gal->FescIIIWeightedGSM;
}

double fesc_recalibration_grid_sfr_popIII(const galaxy_t* gal)
{
  if (fesc_recalibration_active &&
      fesc_recalibration_source_eligible(gal)) {
    return fesc_correction_popIII_sfr * gal->FescIIIWeightedSfr;
  }

  return gal->FescIIIWeightedSfr;
}
#endif

void fesc_recalibration_free(void)
{
  fesc_correction_popII_gsm = 1.0;
  fesc_correction_popII_sfr = 1.0;
#if USE_MINI_HALOS
  fesc_correction_popIII_gsm = 1.0;
  fesc_correction_popIII_sfr = 1.0;
#endif
  fesc_recalibration_initialized = 0;
  fesc_recalibration_active = 0;
  fesc_recalibration_snapshot = -1;
}
#endif