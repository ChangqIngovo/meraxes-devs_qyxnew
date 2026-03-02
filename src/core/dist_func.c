#include <assert.h>
#include <hdf5_hl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "dist_func.h"
#include "meraxes.h"

//! Extract Mvir (halo virial mass) from galaxy
double df_get_mvir(const galaxy_t* gal)
{
  // Convert to log10(M) in solar masses / h
  double mvir_solar_h = gal->Mvir * 1e10 / run_globals.params.Hubble_h;
  return log10(mvir_solar_h);
}

//! Extract stellar mass from galaxy
double df_get_stellar_mass(const galaxy_t* gal)
{
  // Convert to log10(M_stellar) in solar masses
  double stellar_mass = gal->StellarMass * 1e10 / run_globals.params.Hubble_h;
  if (stellar_mass <= 0.0)
    return -1e10;  // Flag for invalid
  return log10(stellar_mass);
}

void df_init(distribution_function_t* df, double x_min, double x_max, int bins_per_dex, const char* description)
{
  assert(df != NULL);
  assert(x_max > x_min);
  assert(bins_per_dex != 0);

  df->x_min = x_min;
  df->x_max = x_max;

  if (bins_per_dex > 0) {
    // Logarithmic binning: bins per dex
    df->bin_width = 1.0 / bins_per_dex;
    df->n_bins = (int)((x_max - x_min) / df->bin_width);
  } else {
    // Linear binning: -bins_per_dex is total number of bins
    df->n_bins = -bins_per_dex;
    df->bin_width = (x_max - x_min) / df->n_bins;
  }

  // Allocate memory for bins and counts
  df->bins = (distribution_bin_t*)malloc(df->n_bins * sizeof(distribution_bin_t));
  df->bin_counts = (int*)calloc(df->n_bins, sizeof(int));
  assert(df->bins != NULL);
  assert(df->bin_counts != NULL);

  // Store description
  if (description != NULL) {
    strncpy(df->description, description, 255);
    df->description[255] = '\0';
  } else {
    strcpy(df->description, "Distribution Function");
  }

  // Initialize bin centers
  for (int i = 0; i < df->n_bins; i++) {
    double radius = df->bin_width / 2.0;
    df->bins[i].center = x_min + (i + radius) * df->bin_width;
    df->bins[i].number_density = 0.0;
    df->bins[i].uncertainty = 0.0;
  }
}

void df_free(distribution_function_t* df)
{
  assert(df != NULL);
  if (df->bins != NULL) {
    free(df->bins);
    df->bins = NULL;
  }
  if (df->bin_counts != NULL) {
    free(df->bin_counts);
    df->bin_counts = NULL;
  }
}

void df_calculate(distribution_function_t* df, galaxy_t* galaxies,
                  galaxy_property_fn get_property, double hubble_h, double box_size)
{
  assert(df != NULL);
  assert(galaxies != NULL);
  // get_property can be NULL - will use default Mvir extractor

  // Calculate comoving volume in (Mpc/h)^3
  df->volume = (box_size / hubble_h);
  df->volume = df->volume * df->volume * df->volume;

  // Reset bin counts
  memset(df->bin_counts, 0, df->n_bins * sizeof(int));

  // Step 1: Bin the data by iterating through linked list
  for (galaxy_t* gal = galaxies; gal != NULL; gal = gal->Next) {
    // Skip ghost galaxies
    if (gal->ghost_flag) {
      continue;
    }

    // Get property value from galaxy (default to Mvir if no function provided)
    double value;
    if (get_property != NULL) {
      value = get_property(gal);
    } else {
      // Default to Mvir for HMF
      value = df_get_mvir(gal);
    }

    // Check if valid and within range
    if (value < df->x_min || value > df->x_max) {
      continue;
    }

    // Find bin
    int bin_idx = (int)((value - df->x_min) / df->bin_width);

    // Safety check (in case of floating point rounding)
    if (bin_idx >= 0 && bin_idx < df->n_bins) {
      df->bin_counts[bin_idx]++;
    }
  }
}

void df_mpi_reduce(distribution_function_t* df, int mpi_rank, int mpi_size)
{
#ifdef USE_MPI
  assert(df != NULL);

  if (mpi_size > 1) {
    // Sum counts across all processes to rank 0 only (more efficient than Allreduce)
    MPI_Reduce(df->bin_counts, df->bin_counts, df->n_bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#else
  (void)mpi_rank;
  (void)mpi_size;
#endif
}

void df_write_hdf5(hid_t file_id, const char* group_name, const distribution_function_t* df,
                   const char* dataset_prefix, const char* units)
{
  assert(df != NULL);
  assert(df->bins != NULL);

  // Open group
  hid_t group_id = H5Gopen(file_id, group_name, H5P_DEFAULT);
  if (group_id < 0) {
    mlog_error("Failed to open group %s for writing distribution function", group_name);
    return;
  }

  hsize_t dims[1] = {(hsize_t)df->n_bins};

  // Prepare data arrays and compute densities/uncertainties
  double* centers = (double*)malloc(df->n_bins * sizeof(double));
  double* densities = (double*)malloc(df->n_bins * sizeof(double));
  double* uncertainties = (double*)malloc(df->n_bins * sizeof(double));

  assert(centers != NULL);
  assert(densities != NULL);
  assert(uncertainties != NULL);

  for (int i = 0; i < df->n_bins; i++) {
    centers[i] = df->bins[i].center;
    
    // Compute number density and uncertainty from bin counts
    double count = (double)(df->bin_counts[i]);
    double bin_width = df->bin_width;

    // Number density: N / (volume * bin_width)
    densities[i] = count / (df->volume * bin_width);

    // Poisson uncertainty: sqrt(N) / (volume * bin_width)
    if (count > 0) {
      uncertainties[i] = sqrt(count) / (df->volume * bin_width);
    } else {
      uncertainties[i] = 0.0;
    }
  }

  // Create dataset names with prefix
  char centers_name[256], densities_name[256], uncertainties_name[256];
  snprintf(centers_name, 256, "%s_centers", dataset_prefix);
  snprintf(densities_name, 256, "%s_densities", dataset_prefix);
  snprintf(uncertainties_name, 256, "%s_uncertainties", dataset_prefix);

  // Write datasets
  H5LTmake_dataset_double(group_id, centers_name, 1, dims, centers);
  H5LTmake_dataset_double(group_id, densities_name, 1, dims, densities);
  H5LTmake_dataset_double(group_id, uncertainties_name, 1, dims, uncertainties);

  // Write metadata
  H5LTset_attribute_int(group_id, ".", "n_bins", (int*)&df->n_bins, 1);
  H5LTset_attribute_double(group_id, ".", "x_min", (double*)&df->x_min, 1);
  H5LTset_attribute_double(group_id, ".", "x_max", (double*)&df->x_max, 1);
  H5LTset_attribute_double(group_id, ".", "bin_width", (double*)&df->bin_width, 1);
  H5LTset_attribute_double(group_id, ".", "volume", (double*)&df->volume, 1);
  H5LTset_attribute_string(group_id, ".", "description", df->description);
  H5LTset_attribute_string(group_id, ".", "units", (char*)units);

  H5Gclose(group_id);

  free(centers);
  free(densities);
  free(uncertainties);
}
