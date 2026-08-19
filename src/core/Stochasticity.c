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

static int fesc_recalibration_source_eligible(const galaxy_t* gal)
{
	return gal->Type >= 0 && gal->Type <= 2;
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

	if (source <= ABS_TOL) {
		if (target <= ABS_TOL)
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

void fesc_recalibration(int snapshot)
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

	if (run_globals.mpi_rank == 0) {
#if USE_MINI_HALOS
		mlog(
				"Global fesc recalibration snapshot=%d: "
				"C_GSM_II=%.12g C_SFR_II=%.12g "
				"C_GSM_III=%.12g C_SFR_III=%.12g.",
				MLOG_MESG,
				snapshot,
				corrections[FESC_CORRECTION_POPII_GSM],
				corrections[FESC_CORRECTION_POPII_SFR],
				corrections[FESC_CORRECTION_POPIII_GSM],
				corrections[FESC_CORRECTION_POPIII_SFR]
		);
#else
		mlog(
				"Global fesc recalibration snapshot=%d: "
				"C_GSM=%.12g C_SFR=%.12g.",
				MLOG_MESG,
				snapshot,
				corrections[FESC_CORRECTION_POPII_GSM],
				corrections[FESC_CORRECTION_POPII_SFR]
		);
#endif
	}

	// Apply the global corrections to all galaxies in this snapshot.
	gal = run_globals.FirstGal;
	while (gal != NULL) {
		if (fesc_recalibration_source_eligible(gal)) {
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

#define NO_SHMR_SHMR_MIN_COUNT 3
#define NO_SHMR_SFR_MIN_COUNT 10
#define NO_SHMR_LOG10_MSTAR_FLOOR (-10.0)
#define NO_SHMR_LOG10_SFR_FLOOR (-30.0)
#define NO_SHMR_RECALIBRATION_EPS 1.0e-300

typedef enum no_shmr_population_t
{
	NO_SHMR_POPII = 2,
#if USE_MINI_HALOS
	NO_SHMR_POPIII = 3,
#endif
} no_shmr_population_t;

#if USE_MINI_HALOS
#define NO_SHMR_NPOPULATIONS 2
#else
#define NO_SHMR_NPOPULATIONS 1
#endif

static int no_shmr_prepared_snapshot = -1;
static int no_shmr_initialized = 0;

static void* no_shmr_calloc(size_t count, size_t size)
{
	void* allocation = calloc(count, size);

	if (allocation == NULL) {
		mlog_error("Failed to allocate noSHMR memory.");
		ABORT(EXIT_FAILURE);
	}

	return allocation;
}

static size_t no_shmr_population_slot(no_shmr_population_t population)
{
	return (size_t)(population - NO_SHMR_POPII);
}

static int no_shmr_type_eligible(const galaxy_t* gal)
{
	return gal->Type >= 0 && gal->Type <= 2;
}

static no_shmr_population_t no_shmr_current_population(const galaxy_t* gal)
{
#if USE_MINI_HALOS
	return (no_shmr_population_t)gal->Galaxy_Population;
#else
	(void)gal;
	return NO_SHMR_POPII;
#endif
}

static float* no_shmr_shmr_table(no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return run_globals.SHMRsIII;
#else
	(void)population;
#endif

	return run_globals.SHMRs;
}

static float* no_shmr_sfr_table(no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return run_globals.SFRsIII;
#else
	(void)population;
#endif

	return run_globals.SFRs;
}

static double no_shmr_gross_stellar_mass(const galaxy_t* gal,
																				 no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->GrossStellarMassIII;
#else
	(void)population;
#endif

	return gal->GrossStellarMass;
}

static double no_shmr_sfr(const galaxy_t* gal,
													no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->SfrIII;
#else
	(void)population;
#endif

	return gal->Sfr;
}

static double no_shmr_raw_gsm(const galaxy_t* gal,
															no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->FescIIIWeightedGSM;
#else
	(void)population;
#endif

	return gal->FescWeightedGSM;
}

static double no_shmr_raw_sfr(const galaxy_t* gal,
															no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->FescIIIWeightedSfr;
#else
	(void)population;
#endif

	return gal->FescWeightedSfr;
}

static double no_shmr_gross_stellar_mass_no_scatter(
		const galaxy_t* gal,
		no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->GrossStellarMassIIINoScatter;
#else
	(void)population;
#endif

	return gal->GrossStellarMassNoScatter;
}

static double no_shmr_weighted_gsm_no_scatter(
		const galaxy_t* gal,
		no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->FescIIIWeightedGSMNoScatter;
#else
	(void)population;
#endif

	return gal->FescWeightedGSMNoScatter;
}

static double no_shmr_grid_gsm(const galaxy_t* gal,
															 no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->TargetFescIIIWeightedGSM;
#else
	(void)population;
#endif

	return gal->TargetFescWeightedGSM;
}

static double no_shmr_grid_sfr(const galaxy_t* gal,
															 no_shmr_population_t population)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII)
		return gal->TargetFescIIIWeightedSfr;
#else
	(void)population;
#endif

	return gal->TargetFescWeightedSfr;
}

static void no_shmr_set_grid_gsm(galaxy_t* gal,
																 no_shmr_population_t population,
																 double value)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII) {
		gal->TargetFescIIIWeightedGSM = value;
		return;
	}
#else
	(void)population;
#endif

	gal->TargetFescWeightedGSM = value;
}

static void no_shmr_set_grid_sfr(galaxy_t* gal,
																 no_shmr_population_t population,
																 double value)
{
#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII) {
		gal->TargetFescIIIWeightedSfr = value;
		return;
	}
#else
	(void)population;
#endif

	gal->TargetFescWeightedSfr = value;
}

static int no_shmr_mvir_bin_clamped(const galaxy_t* gal)
{
	double log10_mvir;
	int bin;

	if (!(gal->Mvir > 0.0) || gal->Mvir > DBL_MAX) {
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

static int no_shmr_count_valid_entries(const unsigned char* valid,
																			 int n_values)
{
	int count = 0;

	for (int ii = 0; ii < n_values; ii++) {
		if (valid[ii])
			count++;
	}

	return count;
}

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

static double no_shmr_get_log10_table_value(const float* table,
																						const galaxy_t* gal,
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
		return table[SHMR_INDEX(snapshot, gal->Type, 0)];

	if (log10_mvir >= SHMR_XMAX)
		return table[SHMR_INDEX(snapshot, gal->Type, SHMR_NX - 1)];

	position = (log10_mvir - SHMR_XMIN) / SHMR_DX;
	index_left = (int)floor(position);
	index_right = index_left + 1;
	fraction = position - (double)index_left;

	y0 = table[SHMR_INDEX(snapshot, gal->Type, index_left)];
	y1 = table[SHMR_INDEX(snapshot, gal->Type, index_right)];

	return y0 + fraction * (y1 - y0);
}

static double no_shmr_get_log10_mstar(const galaxy_t* gal,
																			int snapshot,
																			no_shmr_population_t population)
{
	return no_shmr_get_log10_table_value(
			no_shmr_shmr_table(population),
			gal,
			snapshot
	);
}

static double no_shmr_get_sfr(const galaxy_t* gal,
															int snapshot,
															no_shmr_population_t population)
{
	return no_shmr_log10_sfr_to_linear(
			no_shmr_get_log10_table_value(
					no_shmr_sfr_table(population),
					gal,
					snapshot
			)
	);
}

static size_t no_shmr_sample_cell(no_shmr_population_t population,
																	int type,
																	int bin)
{
	return
			((no_shmr_population_slot(population) * (size_t)SHMR_NTYPES) +
			 (size_t)type) * (size_t)SHMR_NX +
			(size_t)bin;
}

static void no_shmr_count_population_samples(
		size_t* gsm_offsets,
		size_t* sfr_offsets,
		const galaxy_t* gal,
		no_shmr_population_t population,
		int bin)
{
	size_t cell = no_shmr_sample_cell(
			population,
			gal->Type,
			bin
	);

	double mstar = no_shmr_gross_stellar_mass(gal, population);
	double sfr = no_shmr_sfr(gal, population);

	if (mstar > 0.0 && mstar <= DBL_MAX)
		gsm_offsets[cell + 1]++;

	if (sfr > 0.0 && sfr <= DBL_MAX)
		sfr_offsets[cell + 1]++;
}

static void no_shmr_store_population_samples(
		double* gsm_values,
		double* sfr_values,
		size_t* gsm_cursors,
		size_t* sfr_cursors,
		const galaxy_t* gal,
		no_shmr_population_t population,
		int bin)
{
	size_t cell = no_shmr_sample_cell(
			population,
			gal->Type,
			bin
	);

	double mstar = no_shmr_gross_stellar_mass(gal, population);
	double sfr = no_shmr_sfr(gal, population);

	if (mstar > 0.0 && mstar <= DBL_MAX)
		gsm_values[gsm_cursors[cell]++] = log10(mstar);

	if (sfr > 0.0 && sfr <= DBL_MAX)
		sfr_values[sfr_cursors[cell]++] = log10(sfr);
}

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

	local_counts = no_shmr_calloc(n_cells, sizeof(*local_counts));

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
		rank_counts = no_shmr_calloc(
				(size_t)run_globals.mpi_size * n_cells,
				sizeof(*rank_counts)
		);
		receive_counts = no_shmr_calloc(
				(size_t)run_globals.mpi_size,
				sizeof(*receive_counts)
		);
		displacements = no_shmr_calloc(
				(size_t)run_globals.mpi_size,
				sizeof(*displacements)
		);
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
			global_values = no_shmr_calloc(
					(size_t)global_total,
					sizeof(*global_values)
			);
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
		int* rank_cursor = no_shmr_calloc(
				(size_t)run_globals.mpi_size,
				sizeof(*rank_cursor)
		);

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
			scratch = no_shmr_calloc(
					(size_t)max_cell_count,
					sizeof(*scratch)
			);
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

static void no_shmr_store_median_table(
		float* table,
		int snapshot,
		no_shmr_population_t population,
		const double* medians,
		const unsigned char* valid,
		double floor_value)
{
	for (int type = 0; type < SHMR_NTYPES; type++) {
		double values[SHMR_NX];
		unsigned char type_valid[SHMR_NX];

		for (int bin = 0; bin < SHMR_NX; bin++) {
			size_t cell = no_shmr_sample_cell(
					population,
					type,
					bin
			);

			values[bin] = medians[cell];
			type_valid[bin] = valid[cell];
		}

		if (type > 0 &&
				no_shmr_count_valid_entries(type_valid, SHMR_NX) < 2) {
			for (int bin = 0; bin < SHMR_NX; bin++) {
				values[bin] = table[
						SHMR_INDEX(snapshot, 0, bin)
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
					SHMR_INDEX(snapshot, type, bin)
			] = (float)values[bin];
		}
	}
}

static void no_shmr_broadcast_population_tables(
		int snapshot,
		no_shmr_population_t population)
{
	int n_values = SHMR_NTYPES * SHMR_NX;

	MPI_Bcast(
			no_shmr_shmr_table(population) + SHMR_INDEX(snapshot, 0, 0),
			n_values,
			MPI_FLOAT,
			0,
			run_globals.mpi_comm
	);

	MPI_Bcast(
			no_shmr_sfr_table(population) + SHMR_INDEX(snapshot, 0, 0),
			n_values,
			MPI_FLOAT,
			0,
			run_globals.mpi_comm
	);
}

static void no_shmr_build_source_tables(int snapshot)
{
	size_t n_cells =
			(size_t)NO_SHMR_NPOPULATIONS *
			(size_t)SHMR_NTYPES *
			(size_t)SHMR_NX;

	size_t* gsm_offsets = no_shmr_calloc(
			n_cells + 1,
			sizeof(*gsm_offsets)
	);

	size_t* sfr_offsets = no_shmr_calloc(
			n_cells + 1,
			sizeof(*sfr_offsets)
	);

	size_t* gsm_cursors;
	size_t* sfr_cursors;
	double* gsm_values = NULL;
	double* sfr_values = NULL;

	double* gsm_medians = no_shmr_calloc(
			n_cells,
			sizeof(*gsm_medians)
	);

	double* sfr_medians = no_shmr_calloc(
			n_cells,
			sizeof(*sfr_medians)
	);

	unsigned char* gsm_valid = no_shmr_calloc(
			n_cells,
			sizeof(*gsm_valid)
	);

	unsigned char* sfr_valid = no_shmr_calloc(
			n_cells,
			sizeof(*sfr_valid)
	);

	galaxy_t* gal = run_globals.FirstGal;

	while (gal != NULL) {
		if (no_shmr_type_eligible(gal)) {
			int bin = no_shmr_mvir_bin_clamped(gal);
			no_shmr_population_t population =
					no_shmr_current_population(gal);

			no_shmr_count_population_samples(
					gsm_offsets,
					sfr_offsets,
					gal,
					population,
					bin
			);
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
		gsm_values = no_shmr_calloc(
				gsm_offsets[n_cells],
				sizeof(*gsm_values)
		);
	}

	if (sfr_offsets[n_cells] > 0) {
		sfr_values = no_shmr_calloc(
				sfr_offsets[n_cells],
				sizeof(*sfr_values)
		);
	}

	gsm_cursors = no_shmr_calloc(n_cells, sizeof(*gsm_cursors));
	sfr_cursors = no_shmr_calloc(n_cells, sizeof(*sfr_cursors));

	for (size_t cell = 0; cell < n_cells; cell++) {
		gsm_cursors[cell] = gsm_offsets[cell];
		sfr_cursors[cell] = sfr_offsets[cell];
	}

	gal = run_globals.FirstGal;

	while (gal != NULL) {
		if (no_shmr_type_eligible(gal)) {
			int bin = no_shmr_mvir_bin_clamped(gal);
			no_shmr_population_t population =
					no_shmr_current_population(gal);

			no_shmr_store_population_samples(
					gsm_values,
					sfr_values,
					gsm_cursors,
					sfr_cursors,
					gal,
					population,
					bin
			);
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
				run_globals.SHMRs,
				snapshot,
				NO_SHMR_POPII,
				gsm_medians,
				gsm_valid,
				NO_SHMR_LOG10_MSTAR_FLOOR
		);

		no_shmr_store_median_table(
				run_globals.SFRs,
				snapshot,
				NO_SHMR_POPII,
				sfr_medians,
				sfr_valid,
				NO_SHMR_LOG10_SFR_FLOOR
		);

#if USE_MINI_HALOS
		no_shmr_store_median_table(
				run_globals.SHMRsIII,
				snapshot,
				NO_SHMR_POPIII,
				gsm_medians,
				gsm_valid,
				NO_SHMR_LOG10_MSTAR_FLOOR
		);

		no_shmr_store_median_table(
				run_globals.SFRsIII,
				snapshot,
				NO_SHMR_POPIII,
				sfr_medians,
				sfr_valid,
				NO_SHMR_LOG10_SFR_FLOOR
		);
#endif
	}

	no_shmr_broadcast_population_tables(snapshot, NO_SHMR_POPII);

#if USE_MINI_HALOS
	no_shmr_broadcast_population_tables(snapshot, NO_SHMR_POPIII);
#endif

	free(gsm_values);
	free(sfr_values);
	free(gsm_offsets);
	free(sfr_offsets);
	free(gsm_medians);
	free(sfr_medians);
	free(gsm_valid);
	free(sfr_valid);
}

static void no_shmr_evaluate_active_source(
		galaxy_t* gal,
		no_shmr_population_t population,
		double mstar_source,
		double sfr_source,
		int snapshot)
{
	galaxy_t source_view = *gal;

	double previous_mstar =
			no_shmr_gross_stellar_mass_no_scatter(gal, population);

	double previous_weighted_gsm =
			no_shmr_weighted_gsm_no_scatter(gal, population);

	if (mstar_source < previous_mstar)
		mstar_source = previous_mstar;

	double new_stars_source = mstar_source - previous_mstar;

	source_view.StellarMass = mstar_source;

#if USE_MINI_HALOS
	if (population == NO_SHMR_POPIII) {
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
	if (population == NO_SHMR_POPIII) {
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

static void no_shmr_prepare_sources(int snapshot)
{
	galaxy_t* gal = run_globals.FirstGal;

	while (gal != NULL) {
		if (no_shmr_type_eligible(gal)) {
			no_shmr_population_t population =
					no_shmr_current_population(gal);

			double raw_mstar =
					no_shmr_gross_stellar_mass(gal, population);

			double raw_sfr =
					no_shmr_sfr(gal, population);

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
				double mstar_source = no_shmr_log10_mstar_to_linear(
						no_shmr_get_log10_mstar(
								gal,
								snapshot,
								population
						)
				);

				double sfr_source = raw_sfr > 0.0
						? no_shmr_get_sfr(gal, snapshot, population)
						: 0.0;

				no_shmr_evaluate_active_source(
						gal,
						population,
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
		no_shmr_population_t population)
{
	return no_shmr_type_eligible(gal) &&
				 (no_shmr_raw_gsm(gal, population) > 0.0 ||
					no_shmr_grid_gsm(gal, population) > 0.0);
}

static int no_shmr_has_sfr_recalibration_source(
		const galaxy_t* gal,
		no_shmr_population_t population)
{
	return no_shmr_type_eligible(gal) &&
				 no_shmr_current_population(gal) == population &&
				 (no_shmr_raw_sfr(gal, population) > 0.0 ||
					no_shmr_grid_sfr(gal, population) > 0.0);
}

static void no_shmr_apply_fixed_bin_recalibration(
		no_shmr_population_t population)
{
	size_t n_bins =
			(size_t)SHMR_NTYPES * (size_t)SHMR_NX;

	double* local_target_gsm =
			no_shmr_calloc(n_bins, sizeof(double));
	double* global_target_gsm =
			no_shmr_calloc(n_bins, sizeof(double));
	double* local_target_sfr =
			no_shmr_calloc(n_bins, sizeof(double));
	double* global_target_sfr =
			no_shmr_calloc(n_bins, sizeof(double));
	double* local_source_gsm =
			no_shmr_calloc(n_bins, sizeof(double));
	double* global_source_gsm =
			no_shmr_calloc(n_bins, sizeof(double));
	double* local_source_sfr =
			no_shmr_calloc(n_bins, sizeof(double));
	double* global_source_sfr =
			no_shmr_calloc(n_bins, sizeof(double));

	long long* local_target_gsm_count =
			no_shmr_calloc(n_bins, sizeof(long long));
	long long* global_target_gsm_count =
			no_shmr_calloc(n_bins, sizeof(long long));
	long long* local_target_sfr_count =
			no_shmr_calloc(n_bins, sizeof(long long));
	long long* global_target_sfr_count =
			no_shmr_calloc(n_bins, sizeof(long long));

	galaxy_t* gal = run_globals.FirstGal;

	while (gal != NULL) {
		int has_gsm =
				no_shmr_has_gsm_recalibration_source(gal, population);

		int has_sfr =
				no_shmr_has_sfr_recalibration_source(gal, population);

		if (has_gsm || has_sfr) {
			size_t index =
					(size_t)gal->Type * (size_t)SHMR_NX +
					(size_t)no_shmr_mvir_bin_clamped(gal);

			if (has_gsm) {
				double raw_gsm =
						no_shmr_raw_gsm(gal, population);

				local_target_gsm[index] += raw_gsm;
				local_source_gsm[index] +=
						no_shmr_grid_gsm(gal, population);

				if (raw_gsm > 0.0)
					local_target_gsm_count[index]++;
			}

			if (has_sfr) {
				double raw_sfr =
						no_shmr_raw_sfr(gal, population);

				local_target_sfr[index] += raw_sfr;
				local_source_sfr[index] +=
						no_shmr_grid_sfr(gal, population);

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
		int has_gsm =
				no_shmr_has_gsm_recalibration_source(gal, population);

		int has_sfr =
				no_shmr_has_sfr_recalibration_source(gal, population);

		if (!has_gsm && !has_sfr) {
			gal = gal->Next;
			continue;
		}

		size_t index =
				(size_t)gal->Type * (size_t)SHMR_NX +
				(size_t)no_shmr_mvir_bin_clamped(gal);

		if (has_gsm) {
			double raw_gsm =
					no_shmr_raw_gsm(gal, population);

			double source_gsm =
					no_shmr_grid_gsm(gal, population);

			if (global_source_gsm[index] <=
							NO_SHMR_RECALIBRATION_EPS &&
					global_target_gsm[index] >
							NO_SHMR_RECALIBRATION_EPS) {
				no_shmr_set_grid_gsm(
						gal,
						population,
						raw_gsm > 0.0
								? global_target_gsm[index] /
											(double)global_target_gsm_count[index]
								: 0.0
				);
			} else {
				no_shmr_set_grid_gsm(
						gal,
						population,
						global_source_gsm[index] >
										NO_SHMR_RECALIBRATION_EPS
								? source_gsm *
											(global_target_gsm[index] /
											 global_source_gsm[index])
								: source_gsm
				);
			}
		}

		if (has_sfr) {
			double raw_sfr =
					no_shmr_raw_sfr(gal, population);

			double source_sfr =
					no_shmr_grid_sfr(gal, population);

			if (global_source_sfr[index] <=
							NO_SHMR_RECALIBRATION_EPS &&
					global_target_sfr[index] >
							NO_SHMR_RECALIBRATION_EPS) {
				no_shmr_set_grid_sfr(
						gal,
						population,
						raw_sfr > 0.0
								? global_target_sfr[index] /
											(double)global_target_sfr_count[index]
								: 0.0
				);
			} else {
				no_shmr_set_grid_sfr(
						gal,
						population,
						global_source_sfr[index] >
										NO_SHMR_RECALIBRATION_EPS
								? source_sfr *
											(global_target_sfr[index] /
											 global_source_sfr[index])
								: source_sfr
				);
			}
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

static void no_shmr_sources_init(void)
{
	size_t n_shmr;

	if (no_shmr_initialized)
		return;

	no_shmr_initialized = 1;

	if (run_globals.params.physics.Flag_RemoveSHMRScatter == 0)
		return;

	if (run_globals.params.SnaplistLength <= 0) {
		mlog_error(
				"Cannot initialise noSHMR source tables with "
				"SourceTableNSnaps=%d.",
				run_globals.params.SnaplistLength
		);
		ABORT(EXIT_FAILURE);
	}

	n_shmr =
			(size_t)run_globals.params.SnaplistLength *
			(size_t)SHMR_NTYPES *
			(size_t)SHMR_NX;

	run_globals.SHMRs =
			no_shmr_calloc(n_shmr, sizeof(float));

	run_globals.SFRs =
			no_shmr_calloc(n_shmr, sizeof(float));

#if USE_MINI_HALOS
	run_globals.SHMRsIII =
			no_shmr_calloc(n_shmr, sizeof(float));

	run_globals.SFRsIII =
			no_shmr_calloc(n_shmr, sizeof(float));
#endif
}

void no_shmr_sources_prepare(int snapshot)
{
	if (run_globals.params.physics.Flag_RemoveSHMRScatter == 0)
		return;

	if (!no_shmr_initialized)
		no_shmr_sources_init();

	if (snapshot < 0 ||
			snapshot >= run_globals.params.SnaplistLength) {
		mlog_error(
				"noSHMR snapshot %d is outside [0, %d).",
				snapshot,
				run_globals.params.SnaplistLength
		);
		ABORT(EXIT_FAILURE);
	}

	if (no_shmr_prepared_snapshot == snapshot)
		return;

	no_shmr_build_source_tables(snapshot);
	no_shmr_prepare_sources(snapshot);

	if (run_globals.params.physics.Flag_SourceRecalibration) {
		no_shmr_apply_fixed_bin_recalibration(NO_SHMR_POPII);

#if USE_MINI_HALOS
		no_shmr_apply_fixed_bin_recalibration(NO_SHMR_POPIII);
#endif
	}

	no_shmr_prepared_snapshot = snapshot;
}

double no_shmr_sources_grid_sfr_source(const galaxy_t* gal)
{
	if (run_globals.params.Flag_InstantaneousSFR) {
		return run_globals.params.physics.Flag_RemoveSHMRScatter == 1
				? gal->SfrNoScatter
				: gal->Sfr;
	}

	return run_globals.params.physics.Flag_RemoveSHMRScatter == 1
			? gal->GrossStellarMassNoScatter
			: gal->GrossStellarMass;
}

#if USE_MINI_HALOS


double no_shmr_sources_grid_sfr_source_popIII(const galaxy_t* gal)
{
	if (run_globals.params.Flag_InstantaneousSFR) {
		return run_globals.params.physics.Flag_RemoveSHMRScatter == 1
				? gal->SfrIIINoScatter
				: gal->SfrIII;
	}

	return run_globals.params.physics.Flag_RemoveSHMRScatter == 1
			? gal->GrossStellarMassIIINoScatter
			: gal->GrossStellarMassIII;
}
#endif

void no_shmr_sources_free(void)
{
	no_shmr_prepared_snapshot = -1;

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

	run_globals.params.SnaplistLength = 0;
	no_shmr_initialized = 0;
}

void init_reion_source_tables(void)
{
	no_shmr_sources_init();
}

#endif
