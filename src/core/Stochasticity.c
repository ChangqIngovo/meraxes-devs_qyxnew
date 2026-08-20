#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "Stochasticity.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "reionization.h"

#if USE_STOCHASTICITY

// fesc stochasticity related 
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

static int stochasticity_source_eligible(const galaxy_t* gal)
{
  return gal->Type >= 0 && gal->Type <= 2;
}

static double fesc_get_global_correction(double target,
                                         double source,
                                         const char* name)
{
  if (!isfinite(target) || target < 0.0 ||
      !isfinite(source) || source < 0.0) {
    mlog_error(
        "Invalid global fesc budget: "
        "%s target=%g source=%g.",
        name,
        target,
        source
    );
    ABORT(EXIT_FAILURE);
  }

  if (source <= ABS_TOL) {
    if (target <= ABS_TOL)
      return 1.0;

    mlog_error(
        "Cannot recalibrate %s: "
        "target=%g but source=%g.",
        name,
        target,
        source
    );
    ABORT(EXIT_FAILURE);
  }

  double correction = target / source;

  if (!isfinite(correction) || correction < 0.0) {
    mlog_error(
        "Invalid global fesc correction: "
        "%s C=%g.",
        name,
        correction
    );
    ABORT(EXIT_FAILURE);
  }

  return correction;
}

void fesc_recalibration(void)
{

  double local[FESC_GLOBAL_NSUM] = {0.0};
  double global[FESC_GLOBAL_NSUM] = {0.0};
  double corrections[FESC_CORRECTION_N];
  galaxy_t* gal = run_globals.FirstGal;

  for (int ii = 0; ii < FESC_CORRECTION_N; ii++)
    corrections[ii] = 1.0;

  while (gal != NULL) {
    //Galaxies retain cumulative Pop III source history after transitioning 
    // to Pop II, and mergers can transfer that history to a Pop II parent.
    if (stochasticity_source_eligible(gal)) {
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
        "PopII FescWeightedGSM"
    );

    corrections[FESC_CORRECTION_POPII_SFR] = fesc_get_global_correction(
        global[FESC_POPII_SFR_TARGET],
        global[FESC_POPII_SFR_RAW],
        "PopII FescWeightedSfr"
    );
#if USE_MINI_HALOS
    corrections[FESC_CORRECTION_POPIII_GSM] = fesc_get_global_correction(
        global[FESC_POPIII_GSM_TARGET],
        global[FESC_POPIII_GSM_RAW],
        "PopIII FescWeightedGSM"
    );

    corrections[FESC_CORRECTION_POPIII_SFR] = fesc_get_global_correction(
        global[FESC_POPIII_SFR_TARGET],
        global[FESC_POPIII_SFR_RAW],
        "PopIII FescWeightedSfr"
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

  if (run_globals.mpi_rank == 0) {
#if USE_MINI_HALOS
    mlog(
        "Global fesc recalibration: "
        "C_GSM_II=%.12g C_SFR_II=%.12g "
        "C_GSM_III=%.12g C_SFR_III=%.12g.",
        MLOG_MESG,
        corrections[FESC_CORRECTION_POPII_GSM],
        corrections[FESC_CORRECTION_POPII_SFR],
        corrections[FESC_CORRECTION_POPIII_GSM],
        corrections[FESC_CORRECTION_POPIII_SFR]
    );
#else
    mlog(
        "Global fesc recalibration: "
        "C_GSM=%.12g C_SFR=%.12g.",
        MLOG_MESG,
        corrections[FESC_CORRECTION_POPII_GSM],
        corrections[FESC_CORRECTION_POPII_SFR]
    );
#endif
  }

  // Apply the global corrections to all galaxies in this snapshot.
  gal = run_globals.FirstGal;
  while (gal != NULL) {
    if (stochasticity_source_eligible(gal)) {
      gal->FescWeightedGSM *= corrections[FESC_CORRECTION_POPII_GSM];
      gal->FescWeightedSfr *= corrections[FESC_CORRECTION_POPII_SFR];
#if USE_MINI_HALOS
      gal->FescIIIWeightedGSM *= corrections[FESC_CORRECTION_POPIII_GSM];
      gal->FescIIIWeightedSfr *= corrections[FESC_CORRECTION_POPIII_SFR];
#endif  
    }

    gal = gal->Next;
  }
  
}

// SHMR stochasticity related. Population is the raw Galaxy_Population
static void no_shmr_fill_inside_only_with_floor(double* values,
                                                const unsigned char* valid,
                                                int n_values,
                                                double floor_value)
{
  int first = -1;
  int last = -1;

  for (int ii = 0; ii < n_values; ii++) {
    if (valid[ii]) {
      first = ii;
      break;
    }
  }

  for (int ii = n_values - 1; ii >= 0; ii--) {
    if (valid[ii]) {
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

    while (right <= last && !valid[right])
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

static double no_shmr_get_table_value(const float* table,
                                      const galaxy_t* gal,
						    		  double floor_value)
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
    return table[SHMR_INDEX(gal->Type, 0)];

  if (log10_mvir >= SHMR_XMAX)
    return table[SHMR_INDEX(gal->Type, SHMR_NX - 1)];

  position = (log10_mvir - SHMR_XMIN) / SHMR_DX;
  index_left = (int)floor(position);
  index_right = index_left + 1;
  fraction = position - (double)index_left;

  y0 = table[SHMR_INDEX(gal->Type, index_left)];
  y1 = table[SHMR_INDEX(gal->Type, index_right)];

  y0 += fraction * (y1 - y0);
  if (y0 < floor_value)
	y0 = floor_value;

  return pow(10.0, y0);
}


// Build per-cell global medians by gathering all rank-local samples, then
// sorting each cell's combined values on rank 0. Only cells meeting
// min_count are marked valid and assigned a median.
static void no_shmr_global_bin_medians(
    const double* local_values,
    const size_t* local_offsets,
    size_t n_cells,
    int min_count,
    double* medians,
    unsigned char* valid)
{
  size_t local_total = local_offsets[n_cells];
  long long global_total = 0;
  int* local_counts;
  int* rank_counts = NULL;
  int* receive_counts = NULL;
  int* displacements = NULL;
  double* global_values = NULL;

  local_counts = calloc(n_cells, sizeof(*local_counts));
  if (local_counts == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
  }

  for (size_t cell = 0; cell < n_cells; cell++) {
    local_counts[cell] = (int)(
        local_offsets[cell + 1] - local_offsets[cell]
    );
  }

  {
    long long local_total_mpi = (long long)local_total;

    MPI_Allreduce(
        &local_total_mpi,
        &global_total,
        1,
        MPI_LONG_LONG_INT,
        MPI_SUM,
        run_globals.mpi_comm
    );
  }

  if (global_total > INT_MAX) {
    mlog_error(
        "Global noSHMR median sample count %lld exceeds MPI_Gatherv limits.",
        global_total
    );
    ABORT(EXIT_FAILURE);
  }

  if (run_globals.mpi_rank == 0) {
    rank_counts = calloc(
        (size_t)run_globals.mpi_size * n_cells,
        sizeof(*rank_counts)
    );
    receive_counts = calloc(
        (size_t)run_globals.mpi_size,
        sizeof(*receive_counts)
    );
    displacements = calloc(
        (size_t)run_globals.mpi_size,
        sizeof(*displacements)
    );
    if (rank_counts == NULL || receive_counts == NULL || displacements == NULL) {
      mlog_error("Failed to allocate noSHMR memory.");
      ABORT(EXIT_FAILURE);
    }
  }

  MPI_Gather(
      local_counts,
      (int)n_cells,
      MPI_INT,
      rank_counts,
      (int)n_cells,
      MPI_INT,
      0,
      run_globals.mpi_comm
  );

  if (run_globals.mpi_rank == 0) {
    int displacement = 0;

    for (int rank = 0; rank < run_globals.mpi_size; rank++) {
      int rank_total = 0;

      for (size_t cell = 0; cell < n_cells; cell++)
        rank_total += rank_counts[(size_t)rank * n_cells + cell];

      receive_counts[rank] = rank_total;
      displacements[rank] = displacement;
      displacement += rank_total;
    }

    if (global_total > 0) {
      global_values = calloc(
          (size_t)global_total,
          sizeof(*global_values)
      );
      if (global_values == NULL) {
        mlog_error("Failed to allocate noSHMR memory.");
        ABORT(EXIT_FAILURE);
      }
    }
  }

  {
    double dummy = 0.0;

    MPI_Gatherv(
        local_total > 0 ? local_values : &dummy,
        (int)local_total,
        MPI_DOUBLE,
        run_globals.mpi_rank == 0 && global_total > 0
            ? global_values
            : &dummy,
        receive_counts,
        displacements,
        MPI_DOUBLE,
        0,
        run_globals.mpi_comm
    );
  }

  if (run_globals.mpi_rank == 0) {
    int* rank_cursor = calloc(
        (size_t)run_globals.mpi_size,
        sizeof(*rank_cursor)
    );
    if (rank_cursor == NULL) {
      mlog_error("Failed to allocate noSHMR memory.");
      ABORT(EXIT_FAILURE);
    }

    int max_cell_count = 0;
    double* scratch = NULL;

    for (int rank = 0; rank < run_globals.mpi_size; rank++)
      rank_cursor[rank] = displacements[rank];

    for (size_t cell = 0; cell < n_cells; cell++) {
      int cell_count = 0;

      for (int rank = 0; rank < run_globals.mpi_size; rank++) {
        cell_count += rank_counts[
            (size_t)rank * n_cells + cell
        ];
      }

      if (cell_count > max_cell_count)
        max_cell_count = cell_count;
    }

    if (max_cell_count > 0) {
      scratch = calloc(
          (size_t)max_cell_count,
          sizeof(*scratch)
      );
      if (scratch == NULL) {
        mlog_error("Failed to allocate noSHMR memory.");
        ABORT(EXIT_FAILURE);
      }
    }

    for (size_t cell = 0; cell < n_cells; cell++) {
      int cell_count = 0;

      for (int rank = 0; rank < run_globals.mpi_size; rank++) {
        int count = rank_counts[(size_t)rank * n_cells + cell];

        for (int ii = 0; ii < count; ii++) {
          scratch[cell_count++] = global_values[
              rank_cursor[rank] + ii
          ];
        }

        rank_cursor[rank] += count;
      }

      if (cell_count >= min_count) {
        qsort(
            scratch,
            (size_t)cell_count,
            sizeof(*scratch),
            compare_doubles
        );

        if (cell_count % 2 == 0) {
          medians[cell] = 0.5 * (
              scratch[cell_count / 2 - 1] +
              scratch[cell_count / 2]
          );
        } else {
          medians[cell] = scratch[cell_count / 2];
        }

        valid[cell] = 1;
      }
    }

    free(rank_cursor);
    free(scratch);
  }

  free(local_counts);
  free(rank_counts);
  free(receive_counts);
  free(displacements);
  free(global_values);
}

// Write one population/type median table for a snapshot. Sparse non-central
// types fall back to type 0, and interior gaps are interpolated while edges
// are filled with the supplied floor value.
static void no_shmr_store_median_table(
    float* table,
    int population,
    const double* medians,
    const unsigned char* valid,
    double floor_value)
{
  int count;
  for (int type = 0; type < SHMR_NTYPES; type++) {
    double values[SHMR_NX];
    unsigned char type_valid[SHMR_NX];
	count = 0;

    for (int bin = 0; bin < SHMR_NX; bin++) {
      size_t cell = (size_t)type * (size_t)SHMR_NX + (size_t)bin;

      values[bin] = medians[cell];
      type_valid[bin] = valid[cell];
	  if (type_valid[bin])
	    count++;
    }

    if (type > 0 && count < 2) {
      for (int bin = 0; bin < SHMR_NX; bin++) {
        values[bin] = table[
            SHMR_INDEX(0, bin)
        ];
      }
    } else {
      no_shmr_fill_inside_only_with_floor(
          values,
          type_valid,
          SHMR_NX,
          floor_value
      );
    }

    for (int bin = 0; bin < SHMR_NX; bin++) {
      table[
          SHMR_INDEX(type, bin)
      ] = (float)values[bin];
    }
  }
}

// Build the SHMR/SFR source tables for one population's [type x halo-mass
// bin] grid, using only galaxies belonging to that population.
void no_shmr_build_source_tables(int population)
{
  size_t n_cells =
      (size_t)SHMR_NTYPES *
      (size_t)SHMR_NX;

  size_t* gsm_offsets = calloc(
      n_cells + 1,
      sizeof(*gsm_offsets)
  );

  size_t* sfr_offsets = calloc(
      n_cells + 1,
      sizeof(*sfr_offsets)
  );

  size_t* gsm_cursors;
  size_t* sfr_cursors;
  double* gsm_values = NULL;
  double* sfr_values = NULL;

  double* gsm_medians = calloc(
      n_cells,
      sizeof(*gsm_medians)
  );

  double* sfr_medians = calloc(
      n_cells,
      sizeof(*sfr_medians)
  );

  unsigned char* gsm_valid = calloc(
      n_cells,
      sizeof(*gsm_valid)
  );

  unsigned char* sfr_valid = calloc(
      n_cells,
      sizeof(*sfr_valid)
  );

  if (gsm_offsets == NULL || sfr_offsets == NULL ||
      gsm_medians == NULL || sfr_medians == NULL ||
      gsm_valid == NULL || sfr_valid == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
  }

  galaxy_t* gal = run_globals.FirstGal;
  double log10_mvir, mstar, sfr;
  int bin;
  size_t cell;
  while (gal != NULL) {
    if (stochasticity_source_eligible(gal) && gal->Galaxy_Population == population) {
	  log10_mvir = log10(gal->Mvir);
      bin = log10_mvir < SHMR_XMIN ? 0 : log10_mvir > SHMR_XMAX ? SHMR_NX - 1 : (int)floor((log10_mvir - SHMR_XMIN) / SHMR_DX);
	  cell = (size_t) ( gal->Type * SHMR_NX + bin);

#if USE_MINI_HALOS
	  mstar = gal->Galaxy_Population == 3 ? gal->GrossStellarMassIII : gal->GrossStellarMass;
      sfr = gal->Galaxy_Population == 3 ? gal->SfrIII : gal->Sfr;
#else
      mstar = gal->GrossStellarMass;
      sfr = gal->Sfr;
#endif

	  if (mstar > 0.0 && mstar <= DBL_MAX)
		gsm_offsets[cell + 1]++;

	  if (sfr > 0.0 && sfr <= DBL_MAX)
		sfr_offsets[cell + 1]++;
    }

    gal = gal->Next;
  }

  for (size_t cell = 0; cell < n_cells; cell++) {
    gsm_offsets[cell + 1] += gsm_offsets[cell];
    sfr_offsets[cell + 1] += sfr_offsets[cell];
  }

  if (gsm_offsets[n_cells] > INT_MAX ||
      sfr_offsets[n_cells] > INT_MAX) {
    mlog_error("Too many local noSHMR median samples for MPI_Gatherv.");
    ABORT(EXIT_FAILURE);
  }

  if (gsm_offsets[n_cells] > 0) {
    gsm_values = calloc(
        gsm_offsets[n_cells],
        sizeof(*gsm_values)
    );
    if (gsm_values == NULL) {
      mlog_error("Failed to allocate noSHMR memory.");
      ABORT(EXIT_FAILURE);
    }
  }

  if (sfr_offsets[n_cells] > 0) {
    sfr_values = calloc(
        sfr_offsets[n_cells],
        sizeof(*sfr_values)
    );
    if (sfr_values == NULL) {
      mlog_error("Failed to allocate noSHMR memory.");
      ABORT(EXIT_FAILURE);
    }
  }

  gsm_cursors = calloc(n_cells, sizeof(*gsm_cursors));
  sfr_cursors = calloc(n_cells, sizeof(*sfr_cursors));
  if (gsm_cursors == NULL || sfr_cursors == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
  }

  for (size_t cell = 0; cell < n_cells; cell++) {
    gsm_cursors[cell] = gsm_offsets[cell];
    sfr_cursors[cell] = sfr_offsets[cell];
  }

  gal = run_globals.FirstGal;
  
  while (gal != NULL) {
    if (stochasticity_source_eligible(gal) && gal->Galaxy_Population == population) {
      log10_mvir = log10(gal->Mvir);
      bin = log10_mvir < SHMR_XMIN ? 0 : log10_mvir > SHMR_XMAX ? SHMR_NX - 1 : (int)floor((log10_mvir - SHMR_XMIN) / SHMR_DX);
      cell = (size_t) ( gal->Type * SHMR_NX + bin);

#if USE_MINI_HALOS
	  mstar = gal->Galaxy_Population == 3 ? gal->GrossStellarMassIII : gal->GrossStellarMass;
      sfr = gal->Galaxy_Population == 3 ? gal->SfrIII : gal->Sfr;
#else
      mstar = gal->GrossStellarMass;
      sfr = gal->Sfr;
#endif

	  if (mstar > 0.0 && mstar <= DBL_MAX)
		gsm_values[gsm_cursors[cell]++] = log10(mstar);

	  if (sfr > 0.0 && sfr <= DBL_MAX)
		sfr_values[sfr_cursors[cell]++] = log10(sfr);
    }

    gal = gal->Next;
  }

  free(gsm_cursors);
  free(sfr_cursors);

  no_shmr_global_bin_medians(
      gsm_values,
      gsm_offsets,
      n_cells,
      NO_SHMR_SHMR_MIN_COUNT,
      gsm_medians,
      gsm_valid
  );

  no_shmr_global_bin_medians(
      sfr_values,
      sfr_offsets,
      n_cells,
      NO_SHMR_SFR_MIN_COUNT,
      sfr_medians,
      sfr_valid
  );

  if (run_globals.mpi_rank == 0) {
    no_shmr_store_median_table(
#if USE_MINI_HALOS
        population == 3 ? run_globals.SHMRsIII : run_globals.SHMRs,
#else
        run_globals.SHMRs,
#endif
        population,
        gsm_medians,
        gsm_valid,
        NO_SHMR_LOG10_MSTAR_FLOOR
    );

    no_shmr_store_median_table(
#if USE_MINI_HALOS
        population == 3 ? run_globals.SFRsIII : run_globals.SFRs,
#else
        run_globals.SFRs,
#endif
        population,
        sfr_medians,
        sfr_valid,
        NO_SHMR_LOG10_SFR_FLOOR
    );
  }

  // Broadcast the SHMR and SFR tables for one population from rank 0 so all
  // ranks use identical noSHMR source grids in later steps.
  int n_values = SHMR_NTYPES * SHMR_NX;
#if USE_MINI_HALOS
  float* shmr_table = population == 3 ? run_globals.SHMRsIII : run_globals.SHMRs;
  float* sfr_table = population == 3 ? run_globals.SFRsIII : run_globals.SFRs;
#else
  float* shmr_table = run_globals.SHMRs;
  float* sfr_table = run_globals.SFRs;
#endif

  MPI_Bcast(
      shmr_table,
      n_values,
      MPI_FLOAT,
      0,
      run_globals.mpi_comm
  );

  MPI_Bcast(
      sfr_table,
      n_values,
      MPI_FLOAT,
      0,
      run_globals.mpi_comm
  );

  free(gsm_values);
  free(sfr_values);
  free(gsm_offsets);
  free(sfr_offsets);
  free(gsm_medians);
  free(sfr_medians);
  free(gsm_valid);
  free(sfr_valid);
}


// Evaluate one galaxy against the selected source model values, run the
// existing fesc update path on a temporary source view, then commit the
// resulting no-scatter and target weighted fields back to the real galaxy.
static void no_shmr_evaluate_active_source(
    galaxy_t* gal,
    double mstar_source,
    double sfr_source,
    int snapshot)
{
  galaxy_t source_view = *gal;

#if USE_MINI_HALOS
  double previous_mstar =
      gal->Galaxy_Population == 3 ? gal->GrossStellarMassIIINoScatter : gal->GrossStellarMassNoScatter;

  double previous_weighted_gsm =
      gal->Galaxy_Population == 3 ? gal->FescIIIWeightedGSMNoScatter : gal->FescWeightedGSMNoScatter;
#else
  double previous_mstar = gal->GrossStellarMassNoScatter;

  double previous_weighted_gsm = gal->FescWeightedGSMNoScatter;
#endif

  if (mstar_source < previous_mstar)
    mstar_source = previous_mstar;

  double new_stars_source = mstar_source - previous_mstar;

  source_view.StellarMass = mstar_source;

#if USE_MINI_HALOS
  if (gal->Galaxy_Population == 3) {
    gal->SfrIIINoScatter = sfr_source;
    source_view.GrossStellarMassIII = mstar_source;
    source_view.StellarMass_III = mstar_source;
    source_view.SfrIII = sfr_source;
    source_view.FescIIIWeightedGSM = previous_weighted_gsm;
    source_view.FescIIIWeightedSfr = 0.0;
  } else {
    gal->SfrNoScatter = sfr_source;
    source_view.GrossStellarMass = mstar_source;
    source_view.StellarMass_II = mstar_source;
    source_view.Sfr = sfr_source;
    source_view.FescWeightedGSM = previous_weighted_gsm;
    source_view.FescWeightedSfr = 0.0;
  }
#else
  gal->SfrNoScatter = sfr_source;
  source_view.GrossStellarMass = mstar_source;
  source_view.Sfr = sfr_source;
  source_view.FescWeightedGSM = previous_weighted_gsm;
  source_view.FescWeightedSfr = 0.0;
#endif

  update_galaxy_fesc_vals(
      &source_view,
      new_stars_source,
      snapshot
  );

#if USE_MINI_HALOS
  if (gal->Galaxy_Population == 3) {
    gal->GrossStellarMassIIINoScatter = mstar_source;
    gal->FescIIIWeightedGSMNoScatter =
        source_view.FescIIIWeightedGSM;
    gal->TargetFescIIIWeightedGSM =
        source_view.FescIIIWeightedGSM;
    gal->TargetFescIIIWeightedSfr =
        source_view.FescIIIWeightedSfr;
    return;
  }
#endif

  gal->GrossStellarMassNoScatter = mstar_source;
  gal->FescWeightedGSMNoScatter =
      source_view.FescWeightedGSM;
  gal->TargetFescWeightedGSM =
      source_view.FescWeightedGSM;
  gal->TargetFescWeightedSfr =
      source_view.FescWeightedSfr;
}

// Initialize per-galaxy noSHMR source targets from the snapshot tables,
// resetting target accumulators first and only evaluating galaxies with
// active stellar-mass or SFR source content.
void no_shmr_prepare_sources(int snapshot)
{
  galaxy_t* gal = run_globals.FirstGal;

  while (gal != NULL) {
    if (stochasticity_source_eligible(gal)) {
#if USE_MINI_HALOS
      double raw_mstar =
          gal->Galaxy_Population == 3 ? gal->GrossStellarMassIII : gal->GrossStellarMass;

      double raw_sfr =
          gal->Galaxy_Population == 3 ? gal->SfrIII : gal->Sfr;
#else
      double raw_mstar = gal->GrossStellarMass;

      double raw_sfr = gal->Sfr;
#endif

      gal->TargetFescWeightedGSM =
          gal->FescWeightedGSMNoScatter;
      gal->TargetFescWeightedSfr = 0.0;
      gal->SfrNoScatter = 0.0;

#if USE_MINI_HALOS
      gal->TargetFescIIIWeightedGSM =
          gal->FescIIIWeightedGSMNoScatter;
      gal->TargetFescIIIWeightedSfr = 0.0;
      gal->SfrIIINoScatter = 0.0;
#endif

      if (raw_mstar > 0.0 || raw_sfr > 0.0) {
        double mstar_source = no_shmr_get_table_value(
#if USE_MINI_HALOS
            gal->Galaxy_Population == 3 ? run_globals.SHMRsIII : run_globals.SHMRs,
#else
            run_globals.SHMRs,
#endif
            gal,
			NO_SHMR_LOG10_MSTAR_FLOOR
        );

        double sfr_source = raw_sfr > 0.0
            ? no_shmr_get_table_value(
#if USE_MINI_HALOS
                  gal->Galaxy_Population == 3 ? run_globals.SFRsIII : run_globals.SFRs,
#else
                  run_globals.SFRs,
#endif
                  gal,
				  NO_SHMR_LOG10_SFR_FLOOR
              )
            : 0.0;

        no_shmr_evaluate_active_source(
            gal,
            mstar_source,
            sfr_source,
            snapshot
        );
      }
    }

    gal = gal->Next;
  }
}

static int no_shmr_has_gsm_recalibration_source(
    const galaxy_t* gal,
    int population)
{
#if USE_MINI_HALOS
  double raw_gsm = population == 3 ? gal->FescIIIWeightedGSM : gal->FescWeightedGSM;
  double grid_gsm = population == 3 ? gal->TargetFescIIIWeightedGSM : gal->TargetFescWeightedGSM;
#else
  double raw_gsm = gal->FescWeightedGSM;
  double grid_gsm = gal->TargetFescWeightedGSM;
#endif

  return stochasticity_source_eligible(gal) &&
         (raw_gsm > 0.0 || grid_gsm > 0.0);
}

static int no_shmr_has_sfr_recalibration_source(
    const galaxy_t* gal,
    int population)
{
#if USE_MINI_HALOS
  double raw_sfr = population == 3 ? gal->FescIIIWeightedSfr : gal->FescWeightedSfr;
  double grid_sfr = population == 3 ? gal->TargetFescIIIWeightedSfr : gal->TargetFescWeightedSfr;
#else
  double raw_sfr = gal->FescWeightedSfr;
  double grid_sfr = gal->TargetFescWeightedSfr;
#endif

  return stochasticity_source_eligible(gal) &&
         gal->Galaxy_Population == population &&
         (raw_sfr > 0.0 || grid_sfr > 0.0);
}

// Apply fixed-bin global recalibration factors: reduce per-bin raw/source
// budgets across MPI ranks, compute correction ratios for bins with support,
// and scale galaxy target weighted GSM/SFR in the corresponding bin.
void no_shmr_apply_fixed_bin_recalibration(int population)
{
  size_t n_bins =
      (size_t)SHMR_NTYPES * (size_t)SHMR_NX;

    double* local_target_gsm = calloc(n_bins, sizeof(double));
    double* global_target_gsm = calloc(n_bins, sizeof(double));
    double* local_target_sfr = calloc(n_bins, sizeof(double));
    double* global_target_sfr = calloc(n_bins, sizeof(double));
    double* local_source_gsm = calloc(n_bins, sizeof(double));
    double* global_source_gsm = calloc(n_bins, sizeof(double));
    double* local_source_sfr = calloc(n_bins, sizeof(double));
    double* global_source_sfr = calloc(n_bins, sizeof(double));

    long long* local_target_gsm_count = calloc(n_bins, sizeof(long long));
    long long* global_target_gsm_count = calloc(n_bins, sizeof(long long));
    long long* local_target_sfr_count = calloc(n_bins, sizeof(long long));
    long long* global_target_sfr_count = calloc(n_bins, sizeof(long long));

    if (local_target_gsm == NULL || global_target_gsm == NULL ||
      local_target_sfr == NULL || global_target_sfr == NULL ||
      local_source_gsm == NULL || global_source_gsm == NULL ||
      local_source_sfr == NULL || global_source_sfr == NULL ||
      local_target_gsm_count == NULL || global_target_gsm_count == NULL ||
      local_target_sfr_count == NULL || global_target_sfr_count == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
    }

  galaxy_t* gal = run_globals.FirstGal;
  size_t index; 

  double log10_mvir, raw_gsm, grid_gsm, raw_sfr, grid_sfr, new_grid_gsm, new_grid_sfr;
  int bin, has_gsm, has_sfr;
  while (gal != NULL) {
#if USE_MINI_HALOS
    raw_gsm = population == 3 ? gal->FescIIIWeightedGSM : gal->FescWeightedGSM;
    grid_gsm = population == 3 ? gal->TargetFescIIIWeightedGSM : gal->TargetFescWeightedGSM;
	raw_sfr = population == 3 ? gal->FescIIIWeightedSfr : gal->FescWeightedSfr;
	grid_sfr = population == 3 ? gal->TargetFescIIIWeightedSfr : gal->TargetFescWeightedSfr;
#else
    raw_gsm = gal->FescWeightedGSM;
    grid_gsm = gal->TargetFescWeightedGSM;
	raw_sfr = gal->FescWeightedSfr;
	grid_sfr = gal->TargetFescWeightedSfr;
#endif    
    has_gsm = stochasticity_source_eligible(gal) && (raw_gsm > 0.0 || grid_gsm > 0.0);
	has_sfr = stochasticity_source_eligible(gal) && (raw_sfr > 0.0 || grid_sfr > 0.0) && gal->Galaxy_Population == population;

    if (has_gsm || has_sfr) {
	  log10_mvir = log10(gal->Mvir);
      bin = log10_mvir < SHMR_XMIN ? 0 : log10_mvir > SHMR_XMAX ? SHMR_NX - 1 : (int)floor((log10_mvir - SHMR_XMIN) / SHMR_DX);
      index = (size_t) (gal->Type * SHMR_NX + bin);

      if (has_gsm) {
        local_target_gsm[index] += raw_gsm;
        local_source_gsm[index] += grid_gsm;
        if (raw_gsm > 0.0)
          local_target_gsm_count[index]++;
      }

      if (has_sfr) {
        local_target_sfr[index] += raw_sfr;
        local_source_sfr[index] += grid_sfr;
        if (raw_sfr > 0.0)
          local_target_sfr_count[index]++;
      }
    }

    gal = gal->Next;
  }

  MPI_Allreduce(
      local_target_gsm,
      global_target_gsm,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

  MPI_Allreduce(
      local_target_sfr,
      global_target_sfr,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

  MPI_Allreduce(
      local_source_gsm,
      global_source_gsm,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

  MPI_Allreduce(
      local_source_sfr,
      global_source_sfr,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

  MPI_Allreduce(
      local_target_gsm_count,
      global_target_gsm_count,
      (int)n_bins,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      run_globals.mpi_comm
  );

  MPI_Allreduce(
      local_target_sfr_count,
      global_target_sfr_count,
      (int)n_bins,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      run_globals.mpi_comm
  );

  gal = run_globals.FirstGal;
  while (gal != NULL) {
#if USE_MINI_HALOS
    raw_gsm = population == 3 ? gal->FescIIIWeightedGSM : gal->FescWeightedGSM;
    grid_gsm = population == 3 ? gal->TargetFescIIIWeightedGSM : gal->TargetFescWeightedGSM;
	raw_sfr = population == 3 ? gal->FescIIIWeightedSfr : gal->FescWeightedSfr;
	grid_sfr = population == 3 ? gal->TargetFescIIIWeightedSfr : gal->TargetFescWeightedSfr;
#else
    raw_gsm = gal->FescWeightedGSM;
    grid_gsm = gal->TargetFescWeightedGSM;
	raw_sfr = gal->FescWeightedSfr;
	grid_sfr = gal->TargetFescWeightedSfr;
#endif    
    has_gsm = stochasticity_source_eligible(gal) && (raw_gsm > 0.0 || grid_gsm > 0.0);
	has_sfr = stochasticity_source_eligible(gal) && (raw_sfr > 0.0 || grid_sfr > 0.0) && gal->Galaxy_Population == population;

    if (!has_gsm && !has_sfr) {
      gal = gal->Next;
      continue;
    }

	log10_mvir = log10(gal->Mvir);
    bin = log10_mvir < SHMR_XMIN ? 0 : log10_mvir > SHMR_XMAX ? SHMR_NX - 1 : (int)floor((log10_mvir - SHMR_XMIN) / SHMR_DX);
    index = (size_t) (gal->Type * SHMR_NX + bin);

    if (has_gsm) {
      if (global_source_gsm[index] <=
              ABS_TOL &&
          global_target_gsm[index] >
              ABS_TOL) {
        new_grid_gsm =
            raw_gsm > 0.0
                ? global_target_gsm[index] /
                      (double)global_target_gsm_count[index]
                : 0.0;
      } else {
        new_grid_gsm =
            global_source_gsm[index] >
                    ABS_TOL
                ? grid_gsm *
                      (global_target_gsm[index] /
                       global_source_gsm[index])
                : grid_gsm;
      }

#if USE_MINI_HALOS
      if (population == 3)
        gal->TargetFescIIIWeightedGSM = new_grid_gsm;
      else
        gal->TargetFescWeightedGSM = new_grid_gsm;
#else
      gal->TargetFescWeightedGSM = new_grid_gsm;
#endif
    }

    if (has_sfr) {
      if (global_source_sfr[index] <=
              ABS_TOL &&
          global_target_sfr[index] >
              ABS_TOL) {
        new_grid_sfr =
            raw_sfr > 0.0
                ? global_target_sfr[index] /
                      (double)global_target_sfr_count[index]
                : 0.0;
      } else {
        new_grid_sfr =
            global_source_sfr[index] >
                    ABS_TOL
                ? grid_sfr *
                      (global_target_sfr[index] /
                       global_source_sfr[index])
                : grid_sfr;
      }

#if USE_MINI_HALOS
      if (population == 3)
        gal->TargetFescIIIWeightedSfr = new_grid_sfr;
      else
        gal->TargetFescWeightedSfr = new_grid_sfr;
#else
      gal->TargetFescWeightedSfr = new_grid_sfr;
#endif
    }

    gal = gal->Next;
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

void no_shmr_sources_init(void)
{
  size_t n_shmr;

  // Initialise runtime SHMR/SFR source tables.
  run_globals.SHMRs = NULL;
  run_globals.SFRs = NULL;
#if USE_MINI_HALOS
  run_globals.SHMRsIII = NULL;
  run_globals.SFRsIII = NULL;
#endif

  n_shmr =
      (size_t)SHMR_NTYPES *
      (size_t)SHMR_NX;

  run_globals.SHMRs =
      calloc(n_shmr, sizeof(float));

  run_globals.SFRs =
      calloc(n_shmr, sizeof(float));

  if (run_globals.SHMRs == NULL || run_globals.SFRs == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
  }

#if USE_MINI_HALOS
  run_globals.SHMRsIII =
      calloc(n_shmr, sizeof(float));

  run_globals.SFRsIII =
      calloc(n_shmr, sizeof(float));

  if (run_globals.SHMRsIII == NULL || run_globals.SFRsIII == NULL) {
    mlog_error("Failed to allocate noSHMR memory.");
    ABORT(EXIT_FAILURE);
  }
#endif
}

void no_shmr_sources_free(void)
{

  free(run_globals.SHMRs);
  free(run_globals.SFRs);

#if USE_MINI_HALOS
  free(run_globals.SHMRsIII);
  free(run_globals.SFRsIII);
#endif

  run_globals.SHMRs = NULL;
  run_globals.SFRs = NULL;

#if USE_MINI_HALOS
  run_globals.SHMRsIII = NULL;
  run_globals.SFRsIII = NULL;
#endif

}

#endif
