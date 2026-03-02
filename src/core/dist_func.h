#ifndef DIST_FUNC_H
#define DIST_FUNC_H

#include "meraxes.h"

//! Generic bin structure for distribution functions (HMF, SMF, LF, etc.)
typedef struct distribution_bin_t
{
  double center;      //!< Bin center value (e.g., Log10(M) for HMF)
  double number_density;  //!< Number density [h^3 Mpc^-3]
  double uncertainty;     //!< Poisson uncertainty on number density
} distribution_bin_t;

//! Generic distribution function structure
typedef struct distribution_function_t
{
  int n_bins;         //!< Number of bins
  double x_min;       //!< Minimum value (e.g., Log10(M) minimum)
  double x_max;       //!< Maximum value
  double bin_width;   //!< Bin width in dex or linear units
  distribution_bin_t* bins;    //!< Array of bin data
  double volume;      //!< Comoving volume in (Mpc/h)^3
  int* bin_counts;    //!< Count array for use in MPI reductions
  char description[256];  //!< Description of what this function represents
} distribution_function_t;

// Backward compatibility typedef
typedef distribution_function_t hmf_t;

// Function pointer type for extracting properties from galaxy_t
typedef double (*galaxy_property_fn)(const galaxy_t* gal);

#ifdef __cplusplus
extern "C"
{
#endif

  //! Initialize a generic distribution function with binning
  //! \param[in,out] df Pointer to distribution function structure
  //! \param[in] x_min Lower bin edge
  //! \param[in] x_max Upper bin edge
  //! \param[in] bins_per_dex Number of bins per dex (typically 10, or <=0 for linear)
  //! \param[in] description Human-readable description of function (e.g., "Halo Mass Function")
  void df_init(distribution_function_t* df, double x_min, double x_max, int bins_per_dex, const char* description);

  //! Calculate a generic distribution function from a linked list of galaxies
  //! \param[in,out] df Pointer to distribution function to be filled
  //! \param[in] galaxies Pointer to first galaxy in linked list
  //! \param[in] get_property Function pointer to extract property from galaxy (required)
  //! \param[in] hubble_h Hubble constant (H0 / 100)
  //! \param[in] box_size Box size in Mpc/h
  void df_calculate(distribution_function_t* df, galaxy_t* galaxies, 
                    galaxy_property_fn get_property, double hubble_h, double box_size);

  //! Accumulate counts from local process to global distribution (for MPI)
  //! Must be called on all processes before finalization
  //! \param[in,out] df Pointer to distribution function
  //! \param[in] mpi_rank Current MPI rank
  //! \param[in] mpi_size Total number of MPI processes
  void df_mpi_reduce(distribution_function_t* df, int mpi_rank, int mpi_size);

  //! Free distribution function structure
  //! \param[in,out] df Pointer to distribution function to be freed
  void df_free(distribution_function_t* df);

  //! Write distribution function to HDF5 file
  //! \param[in] file_id HDF5 file identifier
  //! \param[in] group_name Name of group to write to (e.g., "SnapXXX")
  //! \param[in] df Pointer to distribution function structure
  //! \param[in] dataset_prefix Prefix for dataset names (e.g., "HMF" for "HMF_centers")
  //! \param[in] units Unit string for the data (e.g., "per Mpc^3 per dex" for HMF/SMF)
  void df_write_hdf5(hid_t file_id, const char* group_name, const distribution_function_t* df, 
                     const char* dataset_prefix, const char* units);

#ifdef __cplusplus
}
#endif

#endif  // DIST_FUNC_H
