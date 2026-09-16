#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "Stochasticity.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "reionization.h"

#if USE_SFR_INTEGRATION
int integrate_sfr_source_slab(double* cumulative,
                              const float* rate,
                              float* stars,
                              size_t nx,
                              int dim,
                              double dt,
                              double totals[4])
{
  if (dim <= 0 || !isfinite(dt) || dt < 0.0 || totals == NULL)
    return -1;
  if (nx > 0 && (cumulative == NULL || rate == NULL || stars == NULL))
    return -1;
  const size_t row_size = 2 * ((size_t)dim / 2 + 1);
  if (nx > SIZE_MAX / (size_t)dim / row_size)
    return -1;
  for (int k = 0; k < 4; ++k)
    totals[k] = 0.0;

  for (size_t ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < dim; ++iy)
      for (int iz = 0; iz < dim; ++iz) {
        const size_t cell = (ix * (size_t)dim + (size_t)iy) * row_size + (size_t)iz;
        const double old_mass = cumulative[cell];
        const double source_rate = rate[cell];
        if (!isfinite(old_mass) || old_mass < 0.0 || !isfinite(source_rate) || source_rate < 0.0)
          return -1;
        const double new_mass = old_mass + source_rate * dt;
        if (!isfinite(new_mass) || new_mass > FLT_MAX)
          return -2;
        cumulative[cell] = new_mass;
        stars[cell] = (float)new_mass;
        totals[0] += old_mass;
        totals[1] += source_rate;
        totals[2] += new_mass - old_mass;
        totals[3] += new_mass;
      }
  for (int k = 0; k < 4; ++k)
    if (!isfinite(totals[k]))
      return -2;
  return 0;
}
#endif

#if USE_STOCHASTICITY

// Galaxy_Population only exists on galaxy_t when USE_MINI_HALOS is on; without
// mini-halos every galaxy is implicitly population 2.
static inline bool galaxy_in_population(const galaxy_t* gal, int population)
{
#if USE_MINI_HALOS
  return gal->Galaxy_Population == population;
#else
  (void)gal;
  return population == 2;
#endif
}

// fesc stochasticity related
enum fesc_global_sum_index
{
#if USE_SFR_INTEGRATION
  FESC_POPII_SFR_RAW = 0,
#else
  FESC_POPII_GSM_RAW = 0,
  FESC_POPII_GSM_TARGET,
  FESC_POPII_SFR_RAW,
#endif
  FESC_POPII_SFR_TARGET,
#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  FESC_POPIII_GSM_RAW,
  FESC_POPIII_GSM_TARGET,
#endif
  FESC_POPIII_SFR_RAW,
  FESC_POPIII_SFR_TARGET,
#endif
  FESC_GLOBAL_NSUM
};

// X-ray stochasticity related
enum xray_global_sum_index
{
  XRAY_LUMINOSITY_RAW = 0,
  XRAY_LUMINOSITY_TARGET,
  XRAY_GLOBAL_NSUM
};

enum stochasticity_calibration_index
{
#if USE_SFR_INTEGRATION
  SFR = 0,
#else
  GSM = 0,
  SFR,
#endif
#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  POPIII_GSM,
#endif
  POPIII_SFR,
#endif
  CAL_N
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

void compute_fesc_recalibration_factors(void)
{

  double local[FESC_GLOBAL_NSUM] = {0.0};
  double global[FESC_GLOBAL_NSUM] = {0.0};
  galaxy_t* gal = run_globals.FirstGal;

  run_globals.fesc_stochasticity_calibrations = 
      calloc(CAL_N, sizeof(double));
  for (int ii = 0; ii < CAL_N; ii++)
    run_globals.fesc_stochasticity_calibrations[ii] = 1.0;

  while (gal != NULL) {
#if USE_SFR_INTEGRATION
    // Recalibrate the current escaped SFR before integrating the source grid.
#else
    //Galaxies retain cumulative Pop III source history after transitioning
    // to Pop II, and mergers can transfer that history to a Pop II parent.
#endif
    if (stochasticity_source_eligible(gal)) {
#if !USE_SFR_INTEGRATION
      local[FESC_POPII_GSM_RAW] += gal->StochasticityTreatedFescWeightedGSM;
      local[FESC_POPII_GSM_TARGET] += gal->FescWeightedGSM;
#endif
      local[FESC_POPII_SFR_RAW] += gal->StochasticityTreatedFescWeightedSfr;
      local[FESC_POPII_SFR_TARGET] += gal->FescWeightedSfr;
#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
      local[FESC_POPIII_GSM_RAW] += gal->StochasticityTreatedFescIIIWeightedGSM;
      local[FESC_POPIII_GSM_TARGET] += gal->FescIIIWeightedGSM;
#endif
      local[FESC_POPIII_SFR_RAW] += gal->StochasticityTreatedFescIIIWeightedSfr;
      local[FESC_POPIII_SFR_TARGET] += gal->FescIIIWeightedSfr;
#endif
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

#if !USE_SFR_INTEGRATION
  run_globals.fesc_stochasticity_calibrations[GSM] = fesc_get_global_correction(
      global[FESC_POPII_GSM_TARGET],
      global[FESC_POPII_GSM_RAW],
      "PopII FescWeightedGSM"
  );

#endif
  run_globals.fesc_stochasticity_calibrations[SFR] = fesc_get_global_correction(
      global[FESC_POPII_SFR_TARGET],
      global[FESC_POPII_SFR_RAW],
      "PopII FescWeightedSfr"
  );
#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  run_globals.fesc_stochasticity_calibrations[POPIII_GSM] = fesc_get_global_correction(
      global[FESC_POPIII_GSM_TARGET],
      global[FESC_POPIII_GSM_RAW],
      "PopIII FescWeightedGSM"
  );

#endif
  run_globals.fesc_stochasticity_calibrations[POPIII_SFR] = fesc_get_global_correction(
      global[FESC_POPIII_SFR_TARGET],
      global[FESC_POPIII_SFR_RAW],
      "PopIII FescWeightedSfr"
  );
#endif
  

  if (run_globals.mpi_rank == 0) {
#if USE_MINI_HALOS
    mlog(
        "Global fesc recalibration: "
#if USE_SFR_INTEGRATION
        "C_SFR_II=%.12g C_SFR_III=%.12g.",
#else
        "C_GSM_II=%.12g C_SFR_II=%.12g "
        "C_GSM_III=%.12g C_SFR_III=%.12g.",
#endif
        MLOG_MESG,
#if !USE_SFR_INTEGRATION
        run_globals.fesc_stochasticity_calibrations[GSM],
#endif
        run_globals.fesc_stochasticity_calibrations[SFR],
#if !USE_SFR_INTEGRATION
        run_globals.fesc_stochasticity_calibrations[POPIII_GSM],
#endif
        run_globals.fesc_stochasticity_calibrations[POPIII_SFR]
    );
#else
    mlog(
        "Global fesc recalibration: "
#if USE_SFR_INTEGRATION
        "C_SFR=%.12g.",
#else
        "C_GSM=%.12g C_SFR=%.12g.",
#endif
        MLOG_MESG,
#if !USE_SFR_INTEGRATION
        run_globals.fesc_stochasticity_calibrations[GSM],
#endif
        run_globals.fesc_stochasticity_calibrations[SFR]
    );
#endif
  }  
}
// Essentially the same but for xray
double compute_xray_recalibration_factor(double local_xray_raw,
                                         double local_xray_target)
{
  double local[XRAY_GLOBAL_NSUM] = {0.0};
  double global[XRAY_GLOBAL_NSUM] = {0.0};

  local[XRAY_LUMINOSITY_RAW] = local_xray_raw;
  local[XRAY_LUMINOSITY_TARGET] = local_xray_target;

  MPI_Allreduce(
      local,
      global,
      XRAY_GLOBAL_NSUM,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

  double correction = 1.0;

  if (global[XRAY_LUMINOSITY_RAW] > 0.0) {
    correction = global[XRAY_LUMINOSITY_TARGET] /
                 global[XRAY_LUMINOSITY_RAW];
  } else if (global[XRAY_LUMINOSITY_TARGET] > 0.0) {
    mlog_error(
        "Cannot recalibrate X-ray luminosity: target=%g raw=%g.",
        global[XRAY_LUMINOSITY_TARGET],
        global[XRAY_LUMINOSITY_RAW]
    );
    ABORT(EXIT_FAILURE);
  }

  if (run_globals.mpi_rank == 0) {
    mlog(
        "Global X-ray recalibration: C_X=%.12g.",
        MLOG_MESG,
        correction
    );
  }

  return correction;
}
// SFR stochasticity related. Population is the raw Galaxy_Population

// Fill missing values in a one-dimensional binned table.
// Bins between valid entries are linearly interpolated using the nearest
// valid values on either side. Bins before the first valid entry and after
// the last valid entry are set to floor_value. Valid entries are unchanged.
static void no_sfr_fill_inside_only(double* values,
                                    const unsigned char* valid,
                                    int n_values,
                                    double floor_value)
{
  if (n_values <= 0)
    return;

  int first = 0;

  while (first < n_values && !valid[first])
    first++;

  /* No valid bins: fill the entire table with the floor. */
  if (first == n_values) {
    for (int ii = 0; ii < n_values; ii++)
      values[ii] = floor_value;

    return;
  }

  int last = n_values - 1;

  while (!valid[last])
    last--;

  for (int ii = 0; ii < first; ii++)
    values[ii] = floor_value;

  for (int ii = last + 1; ii < n_values; ii++)
    values[ii] = floor_value;

  int left = first;

  double fraction;

  while (left < last) {
    int right = left + 1;

    while (right <= last && !valid[right])
      right++;

    for (int ii = left + 1; ii < right; ii++) {
      fraction = (double)(ii - left) / (double)(right - left);

      values[ii] =
          values[left] +
          fraction * (values[right] - values[left]);
    }

    left = right;
  }
}

static double no_sfr_get_table_value(const float* table,
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

#if USE_SFR_INTEGRATION
  if (log10_mvir <= SFR_XMIN) {
    y0 = table[SFR_INDEX(gal->Type, 0)];
    return pow(10.0, fmax(y0, floor_value));
  }
#else
  if (log10_mvir <= SFR_XMIN)
    return pow(10.0, table[SFR_INDEX(gal->Type, 0)]);
#endif

#if USE_SFR_INTEGRATION
  if (log10_mvir >= SFR_XMAX) {
    y0 = table[SFR_INDEX(gal->Type, SFR_NX - 1)];
    return pow(10.0, fmax(y0, floor_value));
  }
#else
  if (log10_mvir >= SFR_XMAX)
    return pow(10.0, table[SFR_INDEX(gal->Type, SFR_NX - 1)]);
#endif

  position = (log10_mvir - SFR_XMIN) / SFR_DX;
  index_left = (int)floor(position);
  index_right = index_left + 1;
  fraction = position - (double)index_left;

  y0 = table[SFR_INDEX(gal->Type, index_left)];
  y1 = table[SFR_INDEX(gal->Type, index_right)];

  y0 += fraction * (y1 - y0);
  if (y0 < floor_value)
    y0 = floor_value;

  return pow(10.0, y0);
}


// Build per-cell global medians by gathering all rank-local samples, then
// sorting each cell's combined values on rank 0. Only cells meeting
// min_count are marked valid and assigned a median.
static void no_sfr_global_bin_medians(
    const double* local_values,
    const size_t* local_offsets,
    size_t n_cells,
    float* table,
    double floor_value,
    int min_count)
{
  double* medians = calloc(n_cells, sizeof(*medians));
  unsigned char* valid = calloc(n_cells, sizeof(*valid));
  if (medians == NULL || valid == NULL) {
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
  }

  long long local_total = (long long) local_offsets[n_cells];
  long long global_total = 0;
  int* local_counts;
  int* rank_counts = NULL;
  int* receive_counts = NULL;
  int* displacements = NULL;
  double* global_values = NULL;

  local_counts = calloc(n_cells, sizeof(*local_counts));
  if (local_counts == NULL) {
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
  }

  for (size_t cell = 0; cell < n_cells; cell++) {
    local_counts[cell] = (int)(
        local_offsets[cell + 1] - local_offsets[cell]
    );
  }
  
  MPI_Allreduce(
	&local_total,
	&global_total,
	1,
	MPI_LONG_LONG_INT,
	MPI_SUM,
	run_globals.mpi_comm
  );
  

  if (global_total > INT_MAX) {
    mlog_error(
        "Global noSFR median sample count %lld exceeds MPI_Gatherv limits.",
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
      mlog_error("Failed to allocate noSFR memory.");
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
        mlog_error("Failed to allocate noSFR memory.");
        ABORT(EXIT_FAILURE);
      }
    }
  }

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

  if (run_globals.mpi_rank == 0) {
    int* rank_cursor = calloc(
        (size_t)run_globals.mpi_size,
        sizeof(*rank_cursor)
    );
    if (rank_cursor == NULL) {
      mlog_error("Failed to allocate noSFR memory.");
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
        mlog_error("Failed to allocate noSFR memory.");
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

	    // Write one median table for each galaxy type independently.
       // Interior gaps are interpolated, while values outside the valid
       // range are filled with the supplied floor value.
       for (int type = 0; type < SFR_NTYPES; type++) {
         double values[SFR_NX];
         unsigned char type_valid[SFR_NX];

         for (int bin = 0; bin < SFR_NX; bin++) {
           size_t cell = (size_t)type * (size_t)SFR_NX + (size_t)bin;

           values[bin] = medians[cell];
           type_valid[bin] = valid[cell];
         }

         no_sfr_fill_inside_only(
             values,
             type_valid,
             SFR_NX,
             floor_value
         );

         for (int bin = 0; bin < SFR_NX; bin++) {
           table[SFR_INDEX(type, bin)] = (float)values[bin];
         }
       }
    free(rank_cursor);
    free(scratch);
  }

  MPI_Bcast(
      table,
      (int)n_cells,
      MPI_FLOAT,
      0,
      run_globals.mpi_comm
  );

  free(local_counts);
  free(rank_counts);
  free(receive_counts);
  free(displacements);
  free(global_values);
  free(medians);
  free(valid);

}

// Build the source tables for one population's [type x halo-mass
// bin] grid, using only galaxies belonging to that population.
void build_no_sfr_tables(int population)
{
  size_t n_cells =
      (size_t)SFR_NTYPES *
      (size_t)SFR_NX;

#if !USE_SFR_INTEGRATION
  size_t* gsm_offsets = calloc(
      n_cells + 1,
      sizeof(*gsm_offsets)
  );

#endif
  size_t* sfr_offsets = calloc(
      n_cells + 1,
      sizeof(*sfr_offsets)
  );

#if !USE_SFR_INTEGRATION
  size_t* gsm_cursors;
#endif
  size_t* sfr_cursors;
#if !USE_SFR_INTEGRATION
  double* gsm_values = NULL;
#endif
  double* sfr_values = NULL;

#if USE_SFR_INTEGRATION
  if (sfr_offsets == NULL) {
#else
  if (gsm_offsets == NULL || sfr_offsets == NULL) {
#endif
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
  }

  galaxy_t* gal = run_globals.FirstGal;
#if USE_SFR_INTEGRATION
  double log10_mvir, sfr;
#else
  double log10_mvir, mstar, sfr;
#endif
  int bin;
  size_t cell;
  while (gal != NULL) {
    if (stochasticity_source_eligible(gal) && galaxy_in_population(gal, population)) {
	  log10_mvir = log10(gal->Mvir);
      bin = log10_mvir < SFR_XMIN ? 0 : log10_mvir > SFR_XMAX ? SFR_NX - 1 : (int)floor((log10_mvir - SFR_XMIN) / SFR_DX);
	  cell = (size_t) ( gal->Type * SFR_NX + bin);

#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
	  mstar = gal->Galaxy_Population == 3 ? gal->GrossStellarMassIII : gal->GrossStellarMass;
#endif
      sfr = gal->Galaxy_Population == 3 ? gal->SfrIII : gal->Sfr;
#else
#if !USE_SFR_INTEGRATION
      mstar = gal->GrossStellarMass;
#endif
      sfr = gal->Sfr;
#endif
#if !USE_SFR_INTEGRATION

	  if (mstar > 0.0 && mstar <= DBL_MAX)
		gsm_offsets[cell + 1]++;
#endif

	  if (sfr > 0.0 && sfr <= DBL_MAX)
		sfr_offsets[cell + 1]++;
    }

    gal = gal->Next;
  }

  for (size_t cell = 0; cell < n_cells; cell++) {
#if !USE_SFR_INTEGRATION
    gsm_offsets[cell + 1] += gsm_offsets[cell];
#endif
    sfr_offsets[cell + 1] += sfr_offsets[cell];
  }

#if USE_SFR_INTEGRATION
  if (sfr_offsets[n_cells] > INT_MAX) {
#else
  if (gsm_offsets[n_cells] > INT_MAX ||
      sfr_offsets[n_cells] > INT_MAX) {
#endif
    mlog_error("Too many local noSFR median samples for MPI_Gatherv.");
    ABORT(EXIT_FAILURE);
  }

#if !USE_SFR_INTEGRATION
  if (gsm_offsets[n_cells] > 0) {
    gsm_values = calloc(
        gsm_offsets[n_cells],
        sizeof(*gsm_values)
    );
    if (gsm_values == NULL) {
      mlog_error("Failed to allocate noSFR memory.");
      ABORT(EXIT_FAILURE);
    }
  }
#endif

  if (sfr_offsets[n_cells] > 0) {
    sfr_values = calloc(
        sfr_offsets[n_cells],
        sizeof(*sfr_values)
    );
    if (sfr_values == NULL) {
      mlog_error("Failed to allocate noSFR memory.");
      ABORT(EXIT_FAILURE);
    }
  }

#if !USE_SFR_INTEGRATION
  gsm_cursors = calloc(n_cells, sizeof(*gsm_cursors));
#endif
  sfr_cursors = calloc(n_cells, sizeof(*sfr_cursors));
#if USE_SFR_INTEGRATION
  if (sfr_cursors == NULL) {
#else
  if (gsm_cursors == NULL || sfr_cursors == NULL) {
#endif
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
  }

  for (size_t cell = 0; cell < n_cells; cell++) {
#if !USE_SFR_INTEGRATION
    gsm_cursors[cell] = gsm_offsets[cell];
#endif
    sfr_cursors[cell] = sfr_offsets[cell];
  }

  gal = run_globals.FirstGal;
  
  while (gal != NULL) {
    if (stochasticity_source_eligible(gal) && galaxy_in_population(gal, population)) {
      log10_mvir = log10(gal->Mvir);
      bin = log10_mvir < SFR_XMIN ? 0 : log10_mvir > SFR_XMAX ? SFR_NX - 1 : (int)floor((log10_mvir - SFR_XMIN) / SFR_DX);
      cell = (size_t) ( gal->Type * SFR_NX + bin);

#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
	  mstar = gal->Galaxy_Population == 3 ? gal->GrossStellarMassIII : gal->GrossStellarMass;
#endif
      sfr = gal->Galaxy_Population == 3 ? gal->SfrIII : gal->Sfr;
#else
#if !USE_SFR_INTEGRATION
      mstar = gal->GrossStellarMass;
#endif
      sfr = gal->Sfr;
#endif
#if !USE_SFR_INTEGRATION

	  if (mstar > 0.0 && mstar <= DBL_MAX)
		gsm_values[gsm_cursors[cell]++] = log10(mstar);
#endif

	  if (sfr > 0.0 && sfr <= DBL_MAX)
		sfr_values[sfr_cursors[cell]++] = log10(sfr);
    }

    gal = gal->Next;
  }

#if !USE_SFR_INTEGRATION
  free(gsm_cursors);
#endif
  free(sfr_cursors);
#if !USE_SFR_INTEGRATION

  no_sfr_global_bin_medians(
      gsm_values,
      gsm_offsets,
      n_cells,
#if USE_MINI_HALOS
      population == 3 ? run_globals.SHMRsIII : run_globals.SHMRs,
#else
      run_globals.SHMRs,
#endif
      NO_SHMR_LOG10_MSTAR_FLOOR,
      NO_SFR_GSM_MIN_COUNT
  );
  free(gsm_values);
  free(gsm_offsets);
#endif

  no_sfr_global_bin_medians(
      sfr_values,
      sfr_offsets,
      n_cells,
#if USE_MINI_HALOS
      population == 3 ? run_globals.SFRsIII : run_globals.SFRs,
#else
      run_globals.SFRs,
#endif
      NO_SHMR_LOG10_SFR_FLOOR,
      NO_SFR_SFR_MIN_COUNT
  );
  free(sfr_values);
  free(sfr_offsets);

}

// Initialize per-galaxy source targets, using integrated GSM when enabled,
// resetting target accumulators first and only evaluating galaxies with
// active stellar-mass or SFR source content.
void apply_no_sfr_treatment(int snapshot)
{
  galaxy_t* gal = run_globals.FirstGal;
  galaxy_t source_view;
  double raw_mstar, raw_sfr, mstar_source, sfr_source;
  double previous_mstar, new_stars_source;
#if USE_SFR_INTEGRATION
  double dt = snapshot == 0 ? 0.0 : run_globals.LTTime[snapshot - 1] - run_globals.LTTime[snapshot];

  if (!isfinite(dt) || dt < 0.0 || (snapshot > 0 && dt == 0.0)) {
    mlog_error("Invalid adjacent-snapshot interval for noSFR source integration at snapshot %d: %g.", snapshot, dt);
    ABORT(EXIT_FAILURE);
  }
#endif

  while (gal != NULL) {
    if (stochasticity_source_eligible(gal)) {
#if USE_MINI_HALOS
      raw_mstar = gal->Galaxy_Population == 3 ? gal->GrossStellarMassIII : gal->GrossStellarMass;
      raw_sfr = gal->Galaxy_Population == 3 ? gal->SfrIII : gal->Sfr;
#else
      raw_mstar = gal->GrossStellarMass;
      raw_sfr = gal->Sfr;
#endif

      previous_mstar =
#if USE_MINI_HALOS
        gal->Galaxy_Population == 3 ? gal->GrossStellarMassIIINoScatter : gal->GrossStellarMassNoScatter;
#else
        gal->GrossStellarMassNoScatter;
#endif

#if !USE_SFR_INTEGRATION
      mstar_source = previous_mstar;
#endif
      sfr_source = 0.0;
#if !USE_SFR_INTEGRATION

      if (raw_mstar > 0.0)
        mstar_source = no_sfr_get_table_value(
#if USE_MINI_HALOS
            gal->Galaxy_Population == 3 ? run_globals.SHMRsIII : run_globals.SHMRs,
#else
            run_globals.SHMRs,
#endif
            gal,
            NO_SHMR_LOG10_MSTAR_FLOOR
        );
#endif

      if (raw_sfr > 0.0)
        sfr_source = no_sfr_get_table_value(
#if USE_MINI_HALOS
            gal->Galaxy_Population == 3 ? run_globals.SFRsIII : run_globals.SFRs,
#else
            run_globals.SFRs,
#endif
            gal,
            NO_SHMR_LOG10_SFR_FLOOR
        );
#if USE_SFR_INTEGRATION

      // SFR and adjacent-snapshot time are both in internal units.
      new_stars_source = sfr_source * dt;
      mstar_source = previous_mstar + new_stars_source;
#endif

      if (raw_mstar > 0.0 || raw_sfr > 0.0) {
        source_view = *gal;
        source_view.StellarMass = mstar_source;

#if USE_MINI_HALOS
        if (gal->Galaxy_Population == 3) {
          source_view.GrossStellarMassIII = mstar_source;
          source_view.StellarMass_III = mstar_source;
          source_view.SfrIII = sfr_source;
          source_view.FescIIIWeightedGSM = gal->StochasticityTreatedFescIIIWeightedGSM;
          source_view.FescIIIWeightedSfr = 0.0;
        }
        else {
          source_view.GrossStellarMass = mstar_source;
          source_view.StellarMass_II = mstar_source;
          source_view.Sfr = sfr_source;
          source_view.FescWeightedGSM = gal->StochasticityTreatedFescWeightedGSM;
          source_view.FescWeightedSfr = 0.0;
        }
#else
        source_view.GrossStellarMass = mstar_source;
        source_view.Sfr = sfr_source;
        source_view.FescWeightedGSM = gal->StochasticityTreatedFescWeightedGSM;
        source_view.FescWeightedSfr = 0.0;
#endif
#if !USE_SFR_INTEGRATION

        new_stars_source = 0.0;
        if (mstar_source > previous_mstar)
          new_stars_source = mstar_source - previous_mstar;
#endif

        update_galaxy_fesc_vals(&source_view, new_stars_source, snapshot);

#if USE_MINI_HALOS
if (gal->Galaxy_Population == 3) {
  gal->GrossStellarMassIIINoScatter = source_view.GrossStellarMassIII;
  gal->SfrIIINoScatter = source_view.SfrIII;

  gal->StochasticityTreatedFescIIIWeightedGSM = source_view.FescIIIWeightedGSM;
  gal->StochasticityTreatedFescIIIWeightedSfr = source_view.FescIIIWeightedSfr;
} else {
  gal->GrossStellarMassNoScatter = source_view.GrossStellarMass;
  gal->SfrNoScatter = source_view.Sfr;

  gal->StochasticityTreatedFescWeightedGSM = source_view.FescWeightedGSM;
  gal->StochasticityTreatedFescWeightedSfr = source_view.FescWeightedSfr;
}
#else
gal->GrossStellarMassNoScatter = source_view.GrossStellarMass;
gal->SfrNoScatter = source_view.Sfr;

gal->StochasticityTreatedFescWeightedGSM = source_view.FescWeightedGSM;
gal->StochasticityTreatedFescWeightedSfr = source_view.FescWeightedSfr;
#endif
      }
    }
    gal = gal->Next;
  }
}

// Apply fixed-bin global recalibration factors: reduce per-bin raw/source
// budgets across MPI ranks, compute correction ratios for bins with support,
// and scale galaxy target weighted sources in the corresponding bin.
void compute_no_sfr_recalibration_factors(int population)
{
  size_t n_bins =
      (size_t)SFR_NTYPES * (size_t)SFR_NX;

#if !USE_SFR_INTEGRATION
    double* local_target_gsm = calloc(n_bins, sizeof(double));
    double* global_target_gsm = calloc(n_bins, sizeof(double));
#endif
    double* local_target_sfr = calloc(n_bins, sizeof(double));
    double* global_target_sfr = calloc(n_bins, sizeof(double));
#if !USE_SFR_INTEGRATION
    double* local_raw_gsm = calloc(n_bins, sizeof(double));
    double* global_raw_gsm = calloc(n_bins, sizeof(double));
#endif
    double* local_raw_sfr = calloc(n_bins, sizeof(double));
    double* global_raw_sfr = calloc(n_bins, sizeof(double));
#if !USE_SFR_INTEGRATION
double* no_sfr_gsm_stochasticity_calibrations;
#endif
double* no_sfr_sfr_stochasticity_calibrations;

#if USE_MINI_HALOS
if (population == 3) {
#if !USE_SFR_INTEGRATION
  no_sfr_gsm_stochasticity_calibrations =
      run_globals.no_sfr_gsm_stochasticity_calibrations_iii;
#endif
  no_sfr_sfr_stochasticity_calibrations =
      run_globals.no_sfr_sfr_stochasticity_calibrations_iii;
} else {
#if !USE_SFR_INTEGRATION
  no_sfr_gsm_stochasticity_calibrations =
      run_globals.no_sfr_gsm_stochasticity_calibrations;
#endif
  no_sfr_sfr_stochasticity_calibrations =
      run_globals.no_sfr_sfr_stochasticity_calibrations;
}
#else
#if !USE_SFR_INTEGRATION
no_sfr_gsm_stochasticity_calibrations =
    run_globals.no_sfr_gsm_stochasticity_calibrations;
#endif
no_sfr_sfr_stochasticity_calibrations =
    run_globals.no_sfr_sfr_stochasticity_calibrations;
#endif

#if !USE_SFR_INTEGRATION
    long long* local_target_gsm_count = calloc(n_bins, sizeof(long long));
    long long* global_target_gsm_count = calloc(n_bins, sizeof(long long));
#endif
    long long* local_target_sfr_count = calloc(n_bins, sizeof(long long));
    long long* global_target_sfr_count = calloc(n_bins, sizeof(long long));

#if USE_SFR_INTEGRATION
    if (local_target_sfr == NULL || global_target_sfr == NULL ||
#else
    if (local_target_gsm == NULL || global_target_gsm == NULL ||
      local_target_sfr == NULL || global_target_sfr == NULL ||
      local_raw_gsm == NULL || global_raw_gsm == NULL ||
#endif
      local_raw_sfr == NULL || global_raw_sfr == NULL ||
#if !USE_SFR_INTEGRATION
      local_target_gsm_count == NULL || global_target_gsm_count == NULL ||
#endif
      local_target_sfr_count == NULL || global_target_sfr_count == NULL) {
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
    }

  galaxy_t* gal = run_globals.FirstGal;
  size_t index; 

#if USE_SFR_INTEGRATION
  double log10_mvir, raw_sfr, target_sfr;
  int bin, has_sfr;
#else
  double log10_mvir, raw_gsm, target_gsm, raw_sfr, target_sfr, new_target_gsm, new_target_sfr;
  int bin, has_gsm, has_sfr;
#endif
  while (gal != NULL) {
#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
    target_gsm = population == 3 ? gal->FescIIIWeightedGSM : gal->FescWeightedGSM;
    raw_gsm = population == 3 ? gal->StochasticityTreatedFescIIIWeightedGSM : gal->StochasticityTreatedFescWeightedGSM;
#endif
	target_sfr = population == 3 ? gal->FescIIIWeightedSfr : gal->FescWeightedSfr;
	raw_sfr = population == 3 ? gal->StochasticityTreatedFescIIIWeightedSfr : gal->StochasticityTreatedFescWeightedSfr;
#else
#if !USE_SFR_INTEGRATION
    target_gsm = gal->FescWeightedGSM; // with SHMR scatter
    raw_gsm = gal->StochasticityTreatedFescWeightedGSM; // no SHMR scatter
#endif
	target_sfr = gal->FescWeightedSfr;
	raw_sfr = gal->StochasticityTreatedFescWeightedSfr;
#endif    
#if !USE_SFR_INTEGRATION
    has_gsm = stochasticity_source_eligible(gal) && (raw_gsm > 0.0 || target_gsm > 0.0);
#endif
	has_sfr = stochasticity_source_eligible(gal) && (raw_sfr > 0.0 || target_sfr > 0.0) && galaxy_in_population(gal, population);

#if USE_SFR_INTEGRATION
    if (has_sfr) {
#else
    if (has_gsm || has_sfr) {
#endif
	    log10_mvir = log10(gal->Mvir);
      bin = log10_mvir < SFR_XMIN ? 0 : log10_mvir > SFR_XMAX ? SFR_NX - 1 : (int)floor((log10_mvir - SFR_XMIN) / SFR_DX);
      index = (size_t) (gal->Type * SFR_NX + bin);
#if !USE_SFR_INTEGRATION

      if (has_gsm) {
		// the naming is confusing, but local_target is the raw data with SHMR scatter while local_raw is the target data without SHMR scatter
		// in other words, the source is the target and the target is the raw data for recalibration purposes
        local_target_gsm[index] += target_gsm;
        local_raw_gsm[index] += raw_gsm;
        if (target_gsm > 0.0)
          local_target_gsm_count[index]++;
      }
#endif

      if (has_sfr) {
        local_target_sfr[index] += target_sfr;
        local_raw_sfr[index] += raw_sfr;
        if (target_sfr > 0.0)
          local_target_sfr_count[index]++;
      }
    }

    gal = gal->Next;
  }

#if !USE_SFR_INTEGRATION
  MPI_Allreduce(
      local_target_gsm,
      global_target_gsm,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );
#endif

  MPI_Allreduce(
      local_target_sfr,
      global_target_sfr,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

#if !USE_SFR_INTEGRATION
  MPI_Allreduce(
      local_raw_gsm,
      global_raw_gsm,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );
#endif

  MPI_Allreduce(
      local_raw_sfr,
      global_raw_sfr,
      (int)n_bins,
      MPI_DOUBLE,
      MPI_SUM,
      run_globals.mpi_comm
  );

#if !USE_SFR_INTEGRATION
  MPI_Allreduce(
      local_target_gsm_count,
      global_target_gsm_count,
      (int)n_bins,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      run_globals.mpi_comm
  );
#endif

  MPI_Allreduce(
      local_target_sfr_count,
      global_target_sfr_count,
      (int)n_bins,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      run_globals.mpi_comm
  );

  for (index=0; index<n_bins; index++) {
#if !USE_SFR_INTEGRATION
    if (global_raw_gsm[index] <= ABS_TOL && global_target_gsm[index] > ABS_TOL) {
      no_sfr_gsm_stochasticity_calibrations[index] =
          global_target_gsm[index] /
          (double)global_target_gsm_count[index];
    } else {
      no_sfr_gsm_stochasticity_calibrations[index] =
          global_raw_gsm[index] > ABS_TOL
              ? global_target_gsm[index] /
                global_raw_gsm[index]
              : 1.0;
    }

#endif
    if (global_raw_sfr[index] <= ABS_TOL && global_target_sfr[index] > ABS_TOL) {
      no_sfr_sfr_stochasticity_calibrations[index] =
          global_target_sfr[index] /
          (double)global_target_sfr_count[index];
    } else {
      no_sfr_sfr_stochasticity_calibrations[index] =
          global_raw_sfr[index] > ABS_TOL
              ? global_target_sfr[index] /
                global_raw_sfr[index]
              : 1.0;
    }
  }

#if !USE_SFR_INTEGRATION
  free(local_target_gsm);
  free(global_target_gsm);
#endif
  free(local_target_sfr);
  free(global_target_sfr);
#if !USE_SFR_INTEGRATION
  free(local_raw_gsm);
  free(global_raw_gsm);
#endif
  free(local_raw_sfr);
  free(global_raw_sfr);
#if !USE_SFR_INTEGRATION
  free(local_target_gsm_count);
  free(global_target_gsm_count);
#endif
  free(local_target_sfr_count);
  free(global_target_sfr_count);
}

void no_sfr_sources_init(void)
{
  size_t n_sfr;

  // Initialise runtime source tables.
#if !USE_SFR_INTEGRATION
  run_globals.SHMRs = NULL;
#endif
  run_globals.SFRs = NULL;
#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  run_globals.SHMRsIII = NULL;
#endif
  run_globals.SFRsIII = NULL;
#endif

  n_sfr =
      (size_t)SFR_NTYPES *
      (size_t)SFR_NX;

#if !USE_SFR_INTEGRATION
  run_globals.SHMRs =
      calloc(n_sfr, sizeof(float));

#endif
  run_globals.SFRs =
      calloc(n_sfr, sizeof(float));

#if !USE_SFR_INTEGRATION
  run_globals.no_sfr_gsm_stochasticity_calibrations =
      calloc(n_sfr, sizeof(double));
#endif
  run_globals.no_sfr_sfr_stochasticity_calibrations =
      calloc(n_sfr, sizeof(double));

#if USE_SFR_INTEGRATION
  if (run_globals.SFRs == NULL || run_globals.no_sfr_sfr_stochasticity_calibrations == NULL) {
#else
  if (run_globals.SHMRs == NULL || run_globals.SFRs == NULL || run_globals.no_sfr_gsm_stochasticity_calibrations == NULL || run_globals.no_sfr_sfr_stochasticity_calibrations == NULL) {
#endif
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
  }

#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  run_globals.SHMRsIII =
      calloc(n_sfr, sizeof(float));

#endif
  run_globals.SFRsIII =
      calloc(n_sfr, sizeof(float));

#if !USE_SFR_INTEGRATION
  run_globals.no_sfr_gsm_stochasticity_calibrations_iii =
      calloc(n_sfr, sizeof(double));
#endif
  run_globals.no_sfr_sfr_stochasticity_calibrations_iii =
      calloc(n_sfr, sizeof(double));

#if USE_SFR_INTEGRATION
  if (run_globals.SFRsIII == NULL || run_globals.no_sfr_sfr_stochasticity_calibrations_iii == NULL) {
#else
  if (run_globals.SHMRsIII == NULL || run_globals.SFRsIII == NULL || run_globals.no_sfr_gsm_stochasticity_calibrations_iii == NULL || run_globals.no_sfr_sfr_stochasticity_calibrations_iii == NULL) {
#endif
    mlog_error("Failed to allocate noSFR memory.");
    ABORT(EXIT_FAILURE);
  }
#endif

}


void no_sfr_sources_free(void)
{

#if !USE_SFR_INTEGRATION
  free(run_globals.SHMRs);
#endif
  free(run_globals.SFRs);
  free(run_globals.fesc_stochasticity_calibrations);
#if !USE_SFR_INTEGRATION
  free(run_globals.no_sfr_gsm_stochasticity_calibrations);
#endif
  free(run_globals.no_sfr_sfr_stochasticity_calibrations);

#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  free(run_globals.SHMRsIII);
#endif
  free(run_globals.SFRsIII);
#if !USE_SFR_INTEGRATION
  free(run_globals.no_sfr_gsm_stochasticity_calibrations_iii);
#endif
  free(run_globals.no_sfr_sfr_stochasticity_calibrations_iii);
#endif

#if !USE_SFR_INTEGRATION
  run_globals.SHMRs = NULL;
#endif
  run_globals.SFRs = NULL;
  run_globals.fesc_stochasticity_calibrations = NULL;
#if !USE_SFR_INTEGRATION
  run_globals.no_sfr_gsm_stochasticity_calibrations = NULL;
#endif
  run_globals.no_sfr_sfr_stochasticity_calibrations = NULL;

#if USE_MINI_HALOS
#if !USE_SFR_INTEGRATION
  run_globals.SHMRsIII = NULL;
#endif
  run_globals.SFRsIII = NULL;
#if !USE_SFR_INTEGRATION
  run_globals.no_sfr_gsm_stochasticity_calibrations_iii = NULL;
#endif
  run_globals.no_sfr_sfr_stochasticity_calibrations_iii = NULL;
#endif

}


double extract_recalibration_factors(galaxy_t* gal, int population, bool use_gsm){
#if USE_SFR_INTEGRATION
  (void)use_gsm;
#endif
  
  if (run_globals.params.physics.Flag_RemoveSFRScatter == 0){
#if USE_MINI_HALOS
    if (population == 3)
#if USE_SFR_INTEGRATION
      return run_globals.fesc_stochasticity_calibrations[POPIII_SFR];
#else
      return use_gsm ? run_globals.fesc_stochasticity_calibrations[POPIII_GSM] : run_globals.fesc_stochasticity_calibrations[POPIII_SFR];
#endif
#endif
#if USE_SFR_INTEGRATION
    return run_globals.fesc_stochasticity_calibrations[SFR];
#else
    return use_gsm ? run_globals.fesc_stochasticity_calibrations[GSM] : run_globals.fesc_stochasticity_calibrations[SFR];
#endif
  }

  double log10_mvir = log10(gal->Mvir);
  int bin = log10_mvir < SFR_XMIN ? 0 : log10_mvir > SFR_XMAX ? SFR_NX - 1 : (int)floor((log10_mvir - SFR_XMIN) / SFR_DX);
  size_t index = (size_t) (gal->Type * SFR_NX + bin);

#if USE_MINI_HALOS
  if (population == 3)
#if USE_SFR_INTEGRATION
    return run_globals.no_sfr_sfr_stochasticity_calibrations_iii[index];
#else
    return use_gsm ? run_globals.no_sfr_gsm_stochasticity_calibrations_iii[index] : run_globals.no_sfr_sfr_stochasticity_calibrations_iii[index];
#endif
#endif
#if USE_SFR_INTEGRATION
    return run_globals.no_sfr_sfr_stochasticity_calibrations[index];
#else
    return use_gsm ? run_globals.no_sfr_gsm_stochasticity_calibrations[index] : run_globals.no_sfr_sfr_stochasticity_calibrations[index];
#endif
}

#endif