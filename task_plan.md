# HMF Calculation Implementation Task Plan

## Goal
Implement Halo Mass Function (HMF) calculation in the C codebase and output results to HDF5 file, matching the Python script's functionality. Support generic distribution functions for SMF, LF, etc.

## Current Understanding
- **Python script** (`run.py`): Reads galaxies from HDF5, filters by `GhostFlag==0`, creates histogram of log10(Mvir), outputs to binary file
- **C codebase**: Meraxes – galaxy/halo simulation, uses HDF5 for I/O, has structured output system in `save.c`
- **Key Python parameters**: 
  - Mvir in units of 1e10 M_sun/h
  - HMF = number density / (volume × bin_width)
  - Includes Poisson uncertainties
  - Bins = 10 per dex between xmin=3 and xmax (calculated from data)
- **MPI Considerations**: Each MPI process calculates local bin counts, then uses MPI_Allreduce to aggregate globally before computing number densities

## Phases

### Phase 1: ✅ Add HMF Histogram Structure
- [x] Create `hmf.h` header with generic distribution function and histogram structures
- [x] Define bin parameters (min, max, n_bins) for flexible binning (log or linear)
- [x] Include Poisson uncertainty calculations
- [x] Rename structures to be generic: `distribution_function_t`, `distribution_bin_t`
- [x] Keep backward-compatible `hmf_t` typedef

**Status**: complete

### Phase 2: ✅ Implement HMF Calculation Function  
- [x] Create `hmf.c` with generic distribution function API
- [x] `df_init()` - flexible initialization with log/linear binning
- [x] `df_calculate()` - bin galaxies using property extractor function pointer
- [x] `df_calculate()` defaults to Mvir extraction when NULL function pointer passed
- [x] Add local bin counting logic (pre-aggregation)
- [x] Include Poisson uncertainty calculations
- [x] Filter out ghost galaxies
- [x] Keep backward-compatible `hmf_init()` and `hmf_free()` wrappers
- [x] Add `get_mvir()` and `get_stellar_mass()` static property extractors for future SMF

**Status**: complete

### Phase 3: ✅ Integrate with Save System (MPI-aware)
- [x] Update `save.c` to include `#include "hmf.h"`
- [x] Add HMF output block in `write_snapshot()` function
- [x] Collect local galaxies on each MPI process
- [x] Call `df_calculate()` to compute local bin counts
- [x] Call `df_mpi_reduce()` to handle MPI aggregation (MPI_Allreduce)
- [x] Only master process (rank 0) writes HDF5 output
- [x] Proper cleanup with `df_free()`

**Status**: complete

### Phase 4: ✅ Add Configuration Options
- [x] Add HMF parameters to `run_params_t` in `meraxes.h`:
  - `Flag_OutputHMF` - enable/disable HMF output
  - `HMF_MinMass` - log10(M) minimum for bins
  - `HMF_MaxMass` - log10(M) maximum for bins
  - `HMF_BinsPerDex` - number of bins per dex (e.g., 10)
- [x] Parameters can be set in config file/parameter parsing

**Status**: complete

## Key Implementation Details

### MPI Handling
- Each process collects its local galaxies from `run_globals.FirstGal` linked list
- Calls `df_calculate()` to bin only local galaxies into `bin_counts` array
- Calls `df_mpi_reduce()` which:
  - Uses `MPI_Allreduce()` to sum bin counts across all processes
  - Computes global number densities and uncertainties on ALL processes
  - Gracefully handles non-MPI builds (no-op when `USE_MPI` undefined)
- Only process rank 0 writes to HDF5 file to avoid conflicts

### Generic Distribution Function Framework
- `distribution_function_t` structure holds bins, counts, volume, and metadata
- `galaxy_property_fn` function pointer allows extracting any property from galaxy_t
- `df_calculate()` with NULL function pointer defaults to Mvir (HMF mode)
- Supports both logarithmic (bins_per_dex > 0) and linear (bins_per_dex < 0) binning
- Easy to extend for SMF (stellar mass function), LF (luminosity function), etc.

### HDF5 Output Format
- Datasets written to group with configurable prefix (e.g., "HMF"):
  - `HMF_centers` - bin center values
  - `HMF_densities` - number density in each bin
  - `HMF_uncertainties` - Poisson uncertainties
- Attributes for metadata:
  - `n_bins`, `x_min`, `x_max`, `bin_width`, `volume`
  - `description` - human-readable label

## Files Modified
- `/home/563/yq5547/bitbucket/meraxes-devs/src/core/hmf.h` - New generic distribution function header
- `/home/563/yq5547/bitbucket/meraxes-devs/src/core/hmf.c` - New implementation with MPI support
- `/home/563/yq5547/bitbucket/meraxes-devs/src/core/save.c` - Integrated HMF output with MPI aggregation
- `/home/563/yq5547/bitbucket/meraxes-devs/src/meraxes.h` - Added HMF config parameters to `run_params_t`

## Next Steps (Optional Enhancements)
1. Add Stellar Mass Function (SMF) using `get_stellar_mass()` extractor
2. Add Luminosity Function when CALC_MAGS is enabled
3. Add configuration file parsing for HMF parameters
4. Create Python script to read HMF datasets from HDF5 output
5. Add optional adaptive binning based on data distribution
