#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "fesc_recalibration.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "no_shmr_sources.h"
#include "reionization.h"

#define NO_SHMR_SHMR_MIN_COUNT 3
#define NO_SHMR_SFR_MIN_COUNT 10
#define NO_SHMR_LOG10_MSTAR_FLOOR (-10.0)
#define NO_SHMR_LOG10_SFR_FLOOR (-30.0)
#define NO_SHMR_RECALIBRATION_EPS 1.0e-300

typedef struct no_shmr_source_record_t
{
  galaxy_t* gal;
  int mvir_bin;
  double raw_gsm;
  double raw_sfr;
  double det_gsm;
  double det_sfr;
  double grid_gsm;
  double grid_sfr;
} no_shmr_source_record_t;

static no_shmr_source_record_t* no_shmr_records = NULL;
static size_t no_shmr_record_count = 0;
static int no_shmr_prepared_snapshot = -1;
static int no_shmr_initialized = 0;
static int no_shmr_owns_tables = 0;
static int no_shmr_applied = 0;

static int no_shmr_enabled(void)
{
  return run_globals.params.physics.Flag_RemoveSHMRScatter != 0;
}

static void* no_shmr_calloc(size_t count, size_t size)
{
  void* allocation = calloc(count, size);

  if (allocation == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
  }

  return allocation;
}

static int no_shmr_source_eligible(const galaxy_t* gal)
{
#if USE_MINI_HALOS
  if (gal->Galaxy_Population != 2)
    return 0;
#endif

  return gal->Type >= 0 && gal->Type <= 2;
}

static int no_shmr_mvir_bin_clamped(const galaxy_t* gal)
{
  double log10_mvir;
  int bin;

  if (!(gal->Mvir > 0.0)) {
    mlog_error("Cannot bin a noSHMR source with Mvir=%g.", gal->Mvir);
    ABORT(EXIT_FAILURE);
  }

  log10_mvir = log10(gal->Mvir);
  bin = (int)floor((log10_mvir - SHMR_XMIN) / SHMR_DX);

  if (bin < 0)
    bin = 0;

  if (bin >= SHMR_NX)
    bin = SHMR_NX - 1;

  return bin;
}

static double no_shmr_log10_mstar_to_linear(double value)
{
  if (value <= NO_SHMR_LOG10_MSTAR_FLOOR)
    return 0.0;

  return pow(10.0, value);
}

static double no_shmr_log10_sfr_to_linear(double value)
{
  if (value <= NO_SHMR_LOG10_SFR_FLOOR)
    return 0.0;

  return pow(10.0, value);
}

static int no_shmr_count_finite_entries(const double* values,
                                        int n_values)
{
  int count = 0;

  for (int ii = 0; ii < n_values; ii++) {
    if (isfinite(values[ii]))
      count++;
  }

  return count;
}

static void no_shmr_fill_inside_only_with_floor(double* values,
                                                int n_values,
                                                double floor_value)
{
  int first = -1;
  int last = -1;

  for (int ii = 0; ii < n_values; ii++) {
    if (isfinite(values[ii])) {
      first = ii;
      break;
    }
  }

  for (int ii = n_values - 1; ii >= 0; ii--) {
    if (isfinite(values[ii])) {
      last = ii;
      break;
    }
  }

  for (int ii = 0; ii < first; ii++)
    values[ii] = floor_value;

  for (int ii = last + 1; ii < n_values; ii++)
    values[ii] = floor_value;

  int left = first;

  while (left < last) {
    int right = left + 1;

    while (right <= last && !isfinite(values[right]))
      right++;

    for (int ii = left + 1; ii < right; ii++) {
      double fraction =
          (double)(ii - left) / (double)(right - left);

      values[ii] =
          values[left] +
          fraction * (values[right] - values[left]);
    }

    left = right;
  }
}

static double no_shmr_get_log10_mstar(const galaxy_t* gal,
                                      int snapshot)
{
  double log10_mvir;
  double y0;
  double y1;
  double position;
  double fraction;
  int index_left;
  int index_right;

  log10_mvir = log10(gal->Mvir);

  if (log10_mvir <= SHMR_XMIN)
    return run_globals.SHMRs[
        SHMR_INDEX(snapshot, gal->Type, 0)
    ];

  if (log10_mvir >= SHMR_XMAX)
    return run_globals.SHMRs[
        SHMR_INDEX(snapshot, gal->Type, SHMR_NX - 1)
    ];

  position = (log10_mvir - SHMR_XMIN) / SHMR_DX;
  index_right = (int)ceil(position);
  index_left = index_right - 1;
  fraction = position - (double)index_left;
  y0 = run_globals.SHMRs[
      SHMR_INDEX(snapshot, gal->Type, index_left)
  ];
  y1 = run_globals.SHMRs[
      SHMR_INDEX(snapshot, gal->Type, index_right)
  ];

  return y0 + fraction * (y1 - y0);
}

static double no_shmr_get_sfr(double log10_mstar,
                              int type,
                              int snapshot)
{
  double y0;
  double y1;
  double position;
  double fraction;
  int index_left;
  int index_right;

  if (log10_mstar <= SFR_XMIN) {
    return no_shmr_log10_sfr_to_linear(
        run_globals.SFRs[SFR_INDEX(snapshot, type, 0)]
    );
  }

  if (log10_mstar >= SFR_XMAX) {
    return no_shmr_log10_sfr_to_linear(
        run_globals.SFRs[SFR_INDEX(snapshot, type, SFR_NX - 1)]
    );
  }

  position = (log10_mstar - SFR_XMIN) / SFR_DX;
  index_right = (int)ceil(position);
  index_left = index_right - 1;
  fraction = position - (double)index_left;
  y0 = run_globals.SFRs[
      SFR_INDEX(snapshot, type, index_left)
  ];
  y1 = run_globals.SFRs[
      SFR_INDEX(snapshot, type, index_right)
  ];

  return no_shmr_log10_sfr_to_linear(
      y0 + fraction * (y1 - y0)
  );
}

static void no_shmr_build_shmr_table(int snapshot)
{
  size_t n_cells = (size_t)SHMR_NTYPES * (size_t)SHMR_NX;
  double* local_sum_log = no_shmr_calloc(n_cells, sizeof(double));
  double* global_sum_log = no_shmr_calloc(n_cells, sizeof(double));
  long long* local_count = no_shmr_calloc(n_cells, sizeof(long long));
  long long* global_count = no_shmr_calloc(n_cells, sizeof(long long));

  galaxy_t* gal = run_globals.FirstGal;

  while (gal != NULL) {
    if (no_shmr_source_eligible(gal) &&
        gal->Mvir > 0.0 &&
        gal->GrossStellarMass > 0.0) {
      double log10_mvir = log10(gal->Mvir);
      double log10_mstar = log10(gal->GrossStellarMass);
      int bin =
          (int)floor((log10_mvir - SHMR_XMIN) / SHMR_DX);

      if (bin >= 0 && bin < SHMR_NX) {
        size_t index =
            (size_t)gal->Type * (size_t)SHMR_NX +
            (size_t)bin;

        local_sum_log[index] += log10_mstar;
        local_count[index]++;
      }
    }

    gal = gal->Next;
  }

  MPI_Allreduce(local_sum_log,
                global_sum_log,
                (int)n_cells,
                MPI_DOUBLE,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_count,
                global_count,
                (int)n_cells,
                MPI_LONG_LONG_INT,
                MPI_SUM,
                run_globals.mpi_comm);

  {
    double values[SHMR_NX];

    for (int bin = 0; bin < SHMR_NX; bin++) {
      size_t index = (size_t)bin;

      if (global_count[index] >= NO_SHMR_SHMR_MIN_COUNT) {
        values[bin] =
            global_sum_log[index] /
            (double)global_count[index];
      } else {
        values[bin] = NAN;
      }
    }

    no_shmr_fill_inside_only_with_floor(
        values,
        SHMR_NX,
        NO_SHMR_LOG10_MSTAR_FLOOR
    );

    for (int bin = 0; bin < SHMR_NX; bin++) {
      run_globals.SHMRs[
          SHMR_INDEX(snapshot, 0, bin)
      ] = (float)values[bin];
    }
  }

  for (int type = 1; type < SHMR_NTYPES; type++) {
    double values[SHMR_NX];

    for (int bin = 0; bin < SHMR_NX; bin++) {
      size_t index =
          (size_t)type * (size_t)SHMR_NX +
          (size_t)bin;

      if (global_count[index] >= NO_SHMR_SHMR_MIN_COUNT) {
        values[bin] =
            global_sum_log[index] /
            (double)global_count[index];
      } else {
        values[bin] = NAN;
      }
    }

    if (no_shmr_count_finite_entries(values, SHMR_NX) < 2) {
      for (int bin = 0; bin < SHMR_NX; bin++) {
        values[bin] = run_globals.SHMRs[
            SHMR_INDEX(snapshot, 0, bin)
        ];
      }
    } else {
      no_shmr_fill_inside_only_with_floor(
          values,
          SHMR_NX,
          NO_SHMR_LOG10_MSTAR_FLOOR
      );
    }

    for (int bin = 0; bin < SHMR_NX; bin++) {
      run_globals.SHMRs[
          SHMR_INDEX(snapshot, type, bin)
      ] = (float)values[bin];
    }
  }

  free(local_sum_log);
  free(global_sum_log);
  free(local_count);
  free(global_count);
}

static void no_shmr_build_sfr_table(int snapshot)
{
  size_t n_cells = (size_t)SFR_NTYPES * (size_t)SFR_NX;
  double* local_sum_log = no_shmr_calloc(n_cells, sizeof(double));
  double* global_sum_log = no_shmr_calloc(n_cells, sizeof(double));
  long long* local_count = no_shmr_calloc(n_cells, sizeof(long long));
  long long* global_count = no_shmr_calloc(n_cells, sizeof(long long));

  galaxy_t* gal = run_globals.FirstGal;

  while (gal != NULL) {
    if (no_shmr_source_eligible(gal) &&
        gal->Mvir > 0.0 &&
        gal->Sfr > 0.0) {
      double log10_mstar =
          no_shmr_get_log10_mstar(gal, snapshot);
      double mstar =
          no_shmr_log10_mstar_to_linear(log10_mstar);

      if (mstar > 0.0) {
        double log10_sfr = log10(gal->Sfr);
        int bin =
            (int)floor((log10_mstar - SFR_XMIN) / SFR_DX);

        if (bin >= 0 && bin < SFR_NX) {
          size_t index =
              (size_t)gal->Type * (size_t)SFR_NX +
              (size_t)bin;

          local_sum_log[index] += log10_sfr;
          local_count[index]++;
        }
      }
    }

    gal = gal->Next;
  }

  MPI_Allreduce(local_sum_log,
                global_sum_log,
                (int)n_cells,
                MPI_DOUBLE,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_count,
                global_count,
                (int)n_cells,
                MPI_LONG_LONG_INT,
                MPI_SUM,
                run_globals.mpi_comm);

  {
    double values[SFR_NX];

    for (int bin = 0; bin < SFR_NX; bin++) {
      size_t index = (size_t)bin;

      if (global_count[index] >= NO_SHMR_SFR_MIN_COUNT) {
        values[bin] =
            global_sum_log[index] /
            (double)global_count[index];
      } else {
        values[bin] = NAN;
      }
    }

    no_shmr_fill_inside_only_with_floor(
        values,
        SFR_NX,
        NO_SHMR_LOG10_SFR_FLOOR
    );

    for (int bin = 0; bin < SFR_NX; bin++) {
      run_globals.SFRs[
          SFR_INDEX(snapshot, 0, bin)
      ] = (float)values[bin];
    }
  }

  for (int type = 1; type < SFR_NTYPES; type++) {
    double values[SFR_NX];

    for (int bin = 0; bin < SFR_NX; bin++) {
      size_t index =
          (size_t)type * (size_t)SFR_NX +
          (size_t)bin;

      if (global_count[index] >= NO_SHMR_SFR_MIN_COUNT) {
        values[bin] =
            global_sum_log[index] /
            (double)global_count[index];
      } else {
        values[bin] = NAN;
      }
    }

    if (no_shmr_count_finite_entries(values, SFR_NX) < 2) {
      for (int bin = 0; bin < SFR_NX; bin++) {
        values[bin] = run_globals.SFRs[
            SFR_INDEX(snapshot, 0, bin)
        ];
      }
    } else {
      no_shmr_fill_inside_only_with_floor(
          values,
          SFR_NX,
          NO_SHMR_LOG10_SFR_FLOOR
      );
    }

    for (int bin = 0; bin < SFR_NX; bin++) {
      run_globals.SFRs[
          SFR_INDEX(snapshot, type, bin)
      ] = (float)values[bin];
    }
  }

  free(local_sum_log);
  free(global_sum_log);
  free(local_count);
  free(global_count);
}

// Evaluate the deterministic source on a stack copy. The real galaxy keeps
// its raw fields; only the deterministic history fields are advanced.
static void no_shmr_evaluate_deterministic_source(
    galaxy_t* gal,
    double mstar_source,
    double sfr_source,
    int snapshot,
    double* det_gsm,
    double* det_sfr)
{
  galaxy_t source_view;
  double previous_mstar;
  double previous_weighted_gsm;
  double new_stars_source;

  previous_mstar = gal->SourceGrossStellarMass;
  previous_weighted_gsm = gal->SourceFescWeightedGSM;

  if (mstar_source < previous_mstar)
    mstar_source = previous_mstar;

  new_stars_source = mstar_source - previous_mstar;

  source_view = *gal;
  source_view.GrossStellarMass = mstar_source;
  source_view.StellarMass = mstar_source;
  source_view.Sfr = sfr_source;
  source_view.FescWeightedGSM = previous_weighted_gsm;
  source_view.FescWeightedSfr = 0.0;

  update_galaxy_fesc_vals(
      &source_view,
      new_stars_source,
      snapshot
  );

  gal->SourceGrossStellarMass = mstar_source;
  gal->SourceFescWeightedGSM = source_view.FescWeightedGSM;

  *det_gsm = source_view.FescWeightedGSM;
  *det_sfr = source_view.FescWeightedSfr;
}

static void no_shmr_build_records(int snapshot)
{
  galaxy_t* gal;
  size_t record_index;

  free(no_shmr_records);
  no_shmr_records = NULL;
  no_shmr_record_count = 0;

  gal = run_globals.FirstGal;

  while (gal != NULL) {
    if (no_shmr_source_eligible(gal))
      no_shmr_record_count++;

    gal = gal->Next;
  }

  if (no_shmr_record_count > 0) {
    no_shmr_records = no_shmr_calloc(no_shmr_record_count,
                                     sizeof(no_shmr_source_record_t));
  }

  record_index = 0;
  gal = run_globals.FirstGal;

  while (gal != NULL) {
    if (no_shmr_source_eligible(gal)) {
      no_shmr_source_record_t* record =
          &no_shmr_records[record_index++];

      record->gal = gal;
      record->mvir_bin = no_shmr_mvir_bin_clamped(gal);
      record->raw_gsm = gal->FescWeightedGSM;
      record->raw_sfr = gal->FescWeightedSfr;
    }

    gal = gal->Next;
  }

  for (size_t ii = 0; ii < no_shmr_record_count; ii++) {
    no_shmr_source_record_t* record = &no_shmr_records[ii];
    double log10_mstar_source;
    double mstar_source = 0.0;
    double sfr_source = 0.0;
    int occupied;

    gal = record->gal;

    occupied =
        gal->GrossStellarMass > 0.0 ||
        gal->Sfr > 0.0;

    log10_mstar_source =
        no_shmr_get_log10_mstar(gal, snapshot);

    if (occupied) {
      mstar_source =
          no_shmr_log10_mstar_to_linear(log10_mstar_source);
    }

    if (gal->Sfr > 0.0 && mstar_source > 0.0) {
      sfr_source = no_shmr_get_sfr(
          log10_mstar_source,
          gal->Type,
          snapshot
      );
    }

    no_shmr_evaluate_deterministic_source(
        gal,
        mstar_source,
        sfr_source,
        snapshot,
        &record->det_gsm,
        &record->det_sfr
    );

    record->grid_gsm = record->det_gsm;
    record->grid_sfr = record->det_sfr;
  }
}

static void no_shmr_apply_fixed_bin_recalibration(void)
{
  size_t n_bins = (size_t)SHMR_NTYPES * (size_t)SHMR_NX;
  double* local_target_gsm = no_shmr_calloc(n_bins, sizeof(double));
  double* global_target_gsm = no_shmr_calloc(n_bins, sizeof(double));
  double* local_target_sfr = no_shmr_calloc(n_bins, sizeof(double));
  double* global_target_sfr = no_shmr_calloc(n_bins, sizeof(double));
  double* local_source_gsm = no_shmr_calloc(n_bins, sizeof(double));
  double* global_source_gsm = no_shmr_calloc(n_bins, sizeof(double));
  double* local_source_sfr = no_shmr_calloc(n_bins, sizeof(double));
  double* global_source_sfr = no_shmr_calloc(n_bins, sizeof(double));
  long long* local_target_gsm_count = no_shmr_calloc(n_bins, sizeof(long long));
  long long* global_target_gsm_count = no_shmr_calloc(n_bins, sizeof(long long));
  long long* local_target_sfr_count = no_shmr_calloc(n_bins, sizeof(long long));
  long long* global_target_sfr_count = no_shmr_calloc(n_bins, sizeof(long long));

  for (size_t ii = 0; ii < no_shmr_record_count; ii++) {
    const no_shmr_source_record_t* record = &no_shmr_records[ii];
    size_t index =
        (size_t)record->gal->Type * (size_t)SHMR_NX +
        (size_t)record->mvir_bin;

    local_target_gsm[index] += record->raw_gsm;
    local_target_sfr[index] += record->raw_sfr;
    local_source_gsm[index] += record->det_gsm;
    local_source_sfr[index] += record->det_sfr;

    if (record->raw_gsm > 0.0)
      local_target_gsm_count[index]++;

    if (record->raw_sfr > 0.0)
      local_target_sfr_count[index]++;
  }

  MPI_Allreduce(local_target_gsm,
                global_target_gsm,
                (int)n_bins,
                MPI_DOUBLE,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_target_sfr,
                global_target_sfr,
                (int)n_bins,
                MPI_DOUBLE,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_source_gsm,
                global_source_gsm,
                (int)n_bins,
                MPI_DOUBLE,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_source_sfr,
                global_source_sfr,
                (int)n_bins,
                MPI_DOUBLE,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_target_gsm_count,
                global_target_gsm_count,
                (int)n_bins,
                MPI_LONG_LONG_INT,
                MPI_SUM,
                run_globals.mpi_comm);

  MPI_Allreduce(local_target_sfr_count,
                global_target_sfr_count,
                (int)n_bins,
                MPI_LONG_LONG_INT,
                MPI_SUM,
                run_globals.mpi_comm);

  for (size_t ii = 0; ii < no_shmr_record_count; ii++) {
    no_shmr_source_record_t* record = &no_shmr_records[ii];
    size_t index =
        (size_t)record->gal->Type * (size_t)SHMR_NX +
        (size_t)record->mvir_bin;

    // A populated bin can have zero deterministic source after edge filling.
    // In that case preserve the positive-source mask and assign its mean.
    if (global_source_gsm[index] <=
            NO_SHMR_RECALIBRATION_EPS &&
        global_target_gsm[index] >
            NO_SHMR_RECALIBRATION_EPS) {
      record->grid_gsm =
          record->raw_gsm > 0.0
              ? global_target_gsm[index] /
                    (double)global_target_gsm_count[index]
              : 0.0;
    } else {
      record->grid_gsm = global_source_gsm[index] > NO_SHMR_RECALIBRATION_EPS
          ? record->det_gsm * (global_target_gsm[index] /
                global_source_gsm[index])
          : record->det_gsm;
    }

    if (global_source_sfr[index] <=
            NO_SHMR_RECALIBRATION_EPS &&
        global_target_sfr[index] >
            NO_SHMR_RECALIBRATION_EPS) {
      record->grid_sfr =
          record->raw_sfr > 0.0
              ? global_target_sfr[index] /
                    (double)global_target_sfr_count[index]
              : 0.0;
    } else {
      record->grid_sfr = global_source_sfr[index] > NO_SHMR_RECALIBRATION_EPS
          ? record->det_sfr * (global_target_sfr[index] /
                global_source_sfr[index])
          : record->det_sfr;
    }
  }

  free(local_target_gsm);
  free(global_target_gsm);
  free(local_target_sfr);
  free(global_target_sfr);
  free(local_source_gsm);
  free(global_source_gsm);
  free(local_source_sfr);
  free(global_source_sfr);
  free(local_target_gsm_count);
  free(global_target_gsm_count);
  free(local_target_sfr_count);
  free(global_target_sfr_count);
}

static void no_shmr_sources_init(void)
{
  size_t n_shmr;
  size_t n_sfr;

  if (no_shmr_initialized)
    return;

  no_shmr_initialized = 1;

  if (!no_shmr_enabled())
    return;

  if (run_globals.SourceTableNSnaps <= 0) {
    mlog_error(
        "Cannot initialise noSHMR source tables with "
        "SourceTableNSnaps=%d.",
        run_globals.SourceTableNSnaps
    );
    ABORT(EXIT_FAILURE);
  }

  n_shmr =
      (size_t)run_globals.SourceTableNSnaps *
      (size_t)SHMR_NTYPES *
      (size_t)SHMR_NX;

  n_sfr =
      (size_t)run_globals.SourceTableNSnaps *
      (size_t)SFR_NTYPES *
      (size_t)SFR_NX;

  run_globals.SHMRs = malloc(sizeof(float) * n_shmr);
  run_globals.SFRs = malloc(sizeof(float) * n_sfr);

  if (run_globals.SHMRs == NULL || run_globals.SFRs == NULL) {
    free(run_globals.SHMRs);
    free(run_globals.SFRs);
    run_globals.SHMRs = NULL;
    run_globals.SFRs = NULL;

    mlog_error("Failed to allocate fixed-bin noSHMR source tables.");
    ABORT(EXIT_FAILURE);
  }

  no_shmr_owns_tables = 1;

  for (size_t ii = 0; ii < n_shmr; ii++)
    run_globals.SHMRs[ii] = NO_SHMR_LOG10_MSTAR_FLOOR;

  for (size_t ii = 0; ii < n_sfr; ii++)
    run_globals.SFRs[ii] = NO_SHMR_LOG10_SFR_FLOOR;

  if (run_globals.mpi_rank == 0) {
    mlog("noSHMR recalibration is %s.", MLOG_MESG,
         run_globals.params.physics.Flag_SourceRecalibration
             ? "active" : "inactive");
  }
}

static void no_shmr_sources_prepare(int snapshot)
{
  if (!no_shmr_initialized)
    no_shmr_sources_init();

  if (snapshot < 0 ||
      snapshot >= run_globals.SourceTableNSnaps) {
    mlog_error(
        "noSHMR snapshot %d is outside [0, %d).",
        snapshot,
        run_globals.SourceTableNSnaps
    );
    ABORT(EXIT_FAILURE);
  }

  if (no_shmr_prepared_snapshot == snapshot)
    return;

  if (no_shmr_applied) {
    mlog_error(
        "Cannot prepare noSHMR snapshot %d while a source override "
        "is applied.",
        snapshot
    );
    ABORT(EXIT_FAILURE);
  }

  no_shmr_build_shmr_table(snapshot);
  no_shmr_build_sfr_table(snapshot);
  no_shmr_build_records(snapshot);

  if (run_globals.params.physics.Flag_SourceRecalibration)
    no_shmr_apply_fixed_bin_recalibration();

  no_shmr_prepared_snapshot = snapshot;
}

void no_shmr_sources_apply(int snapshot)
{
  if (!no_shmr_enabled())
    return;

  no_shmr_sources_prepare(snapshot);

  if (no_shmr_applied) {
    mlog_error("A noSHMR source override is already applied.");
    ABORT(EXIT_FAILURE);
  }

  for (size_t ii = 0; ii < no_shmr_record_count; ii++) {
    no_shmr_source_record_t* record = &no_shmr_records[ii];

    record->gal->FescWeightedGSM = record->grid_gsm;
    record->gal->FescWeightedSfr = record->grid_sfr;
  }

  no_shmr_applied = 1;
}

void no_shmr_sources_restore(void)
{
  if (!no_shmr_applied)
    return;

  for (size_t ii = 0; ii < no_shmr_record_count; ii++) {
    no_shmr_source_record_t* record = &no_shmr_records[ii];

    record->gal->FescWeightedGSM = record->raw_gsm;
    record->gal->FescWeightedSfr = record->raw_sfr;
  }

  no_shmr_applied = 0;
}

void no_shmr_sources_free(void)
{
  no_shmr_sources_restore();

  free(no_shmr_records);
  no_shmr_records = NULL;
  no_shmr_record_count = 0;
  no_shmr_prepared_snapshot = -1;

  if (no_shmr_owns_tables) {
    free(run_globals.SHMRs);
    free(run_globals.SFRs);
    run_globals.SHMRs = NULL;
    run_globals.SFRs = NULL;
  }

  no_shmr_owns_tables = 0;
  no_shmr_initialized = 0;
}

void init_reion_source_tables(void)
{
  no_shmr_sources_init();
  fesc_recalibration_init();
}