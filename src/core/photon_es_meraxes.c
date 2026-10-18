#include "photon_es_meraxes.h"
#include "meraxes.h"
#include "misc_tools.h"

#include <errno.h>
#include <float.h>
#include <hdf5_hl.h>
#include <limits.h>
#include <math.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char** environ;

/* This first production adapter runs the optimized reference in one external
 * process. Meraxes still constructs sources and density with MPI, but this is
 * not a distributed ES solver. Only one snapshot's interchange fields reside
 * in C on rank zero; the worker retains the physical state on disk. */
typedef struct
{
  int initialized, last_snapshot, n, local_cells, total_cells;
  int* counts;
  int* offsets;
  double* local;
  double* density;
  double* global;
  double mean_electron_per_nucleus;
  char identity[128];
  char request[STRLEN + 128];
  char checkpoint[STRLEN + 128];
} es_bridge_state;

static es_bridge_state state;

static void fail(const char* reason)
{
  mlog_error("ES/HHe snapshot bridge: %s", reason);
  ABORT(EXIT_FAILURE);
}

static void collective_check(int invalid, const char* stage)
{
  int any_invalid;
  MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, run_globals.mpi_comm);
  if (any_invalid)
    fail(stage);
}

static void* allocate(size_t count, size_t size)
{
  if (size && count > SIZE_MAX / size)
    fail("allocation size overflow");
  void* result = calloc(count ? count : 1, size);
  if (!result)
    fail("allocation failed");
  return result;
}

static void checked_path(char* result, size_t size, const char* suffix)
{
  const char* output = run_globals.params.OutputDir;
  const size_t length = strlen(output);
  const int written = snprintf(result, size, "%s%s%s", output, length && output[length - 1] == '/' ? "" : "/", suffix);
  if (written < 0 || (size_t)written >= size)
    fail("output path is too long");
}

static int exists(const char* path)
{
  struct stat info;
  if (!stat(path, &info))
    return 1;
  if (errno != ENOENT)
    fail("cannot inspect an ES state path");
  return 0;
}

static void h5check(herr_t status)
{
  if (status < 0)
    fail("HDF5 interchange failed");
}

static void attribute(hid_t file, const char* key, double value)
{
  if (!isfinite(value))
    fail("nonfinite request metadata");
  h5check(H5LTset_attribute_double(file, "/", key, &value, 1));
}

static double mean_h_atoms(void)
{
  const double dx = run_globals.params.BoxSize / state.n;
  const double mass = run_globals.params.OmegaM * run_globals.RhoCrit * dx * dx * dx *
                      run_globals.params.BaryonFrac * run_globals.units.UnitMass_in_g / run_globals.params.Hubble_h;
  return mass * (1.0 - run_globals.params.physics.Y_He) / PROTONMASS;
}

static void initialize(int snapshot)
{
  if (snapshot != run_globals.params.HHeStartSnapshot)
    fail("first call must be HHeStartSnapshot; implicit reset/restart is forbidden");
  const int n = run_globals.params.ReionGridDim;
  if (n < 1 || (double)n * n * n > INT_MAX)
    fail("grid exceeds MPI_Gatherv count/displacement range");
  state.n = n;
  state.total_cells = n * n * n;
  state.counts = allocate((size_t)run_globals.mpi_size, sizeof(*state.counts));
  state.offsets = allocate((size_t)run_globals.mpi_size, sizeof(*state.offsets));
  int covered = 0;
  for (int r = 0; r < run_globals.mpi_size; r++) {
    const ptrdiff_t nx = run_globals.reion_grids.slab_nix[r];
    const ptrdiff_t start = run_globals.reion_grids.slab_ix_start[r];
    if (nx < 0 || start < 0 || start > n || nx > n - start || (nx && start != covered))
      fail("invalid FFTW slab layout");
    state.counts[r] = (int)nx * n * n;
    state.offsets[r] = (int)start * n * n;
    covered += (int)nx;
  }
  if (covered != n)
    fail("FFTW slabs do not cover the grid");
  state.local_cells = state.counts[run_globals.mpi_rank];
  state.local = allocate((size_t)state.local_cells, sizeof(*state.local));
  if (run_globals.mpi_rank == 0) {
    state.density = allocate((size_t)state.total_cells, sizeof(*state.density));
    state.global = allocate((size_t)state.total_cells, sizeof(*state.global));
    snprintf(state.identity, sizeof(state.identity), "es-%lld-%ld", (long long)time(NULL), (long)getpid());
    checked_path(state.request, sizeof(state.request), ".es_request.hdf5");
    checked_path(state.checkpoint, sizeof(state.checkpoint), ".es_checkpoint.hdf5");
    if (exists(state.checkpoint))
      fail("ES checkpoint already exists; choose a fresh OutputDir (automatic restart is not supported)");
  }
  state.last_snapshot = snapshot - 1;
  state.initialized = 1;
  mlog("Online ES/HHe initialized at snapshot %d: fixed periodic ES centres, shared spectral photon accounts, "
       "T=10000 K. Snapshot density is prescribed; source rates use a right-endpoint hold. "
       "The Python ES worker executes on one process; no radiation transport equation is solved.",
       MLOG_MESG,
       snapshot);
}

static void gather_input(int snapshot, double dt_s)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  const int n = state.n;
  const int nx = (int)grids->slab_nix[run_globals.mpi_rank];
  int invalid = 0;
  for (int x = 0; x < nx; x++)
    for (int y = 0; y < n; y++)
      for (int z = 0; z < n; z++) {
        const int j = (x * n + y) * n + z;
        const int p = grid_index(x, y, z, n, INDEX_PADDED);
        const double density = 1.0 + grids->deltax[p];
        if (!isfinite(density) || density < 0)
          invalid = 1;
        state.local[j] = density;
      }
  collective_check(invalid, "density is nonfinite or negative");
  MPI_Gatherv(state.local, state.local_cells, MPI_DOUBLE, state.density, state.counts, state.offsets,
              MPI_DOUBLE, 0, run_globals.mpi_comm);
  const double factor = run_globals.units.UnitMass_in_g / run_globals.units.UnitTime_in_s *
                        run_globals.params.physics.ReionNionPhotPerBary / PROTONMASS;
  for (int x = 0; x < nx; x++)
    for (int y = 0; y < n; y++)
      for (int z = 0; z < n; z++) {
        const int j = (x * n + y) * n + z;
        const int p = grid_index(x, y, z, n, INDEX_PADDED);
        /* FescWeightedSfr is an escaped instantaneous SFR, not a cumulative
         * stellar grid; no subtraction across moving source positions. */
        const double photons = grids->weighted_sfr[p] * factor * dt_s;
        if (!isfinite(grids->weighted_sfr[p]) || grids->weighted_sfr[p] < 0 || !isfinite(photons) || photons < 0)
          invalid = 1;
        state.local[j] = photons;
      }
  collective_check(invalid, "escaped stellar source photons are nonfinite or negative");
  MPI_Gatherv(state.local, state.local_cells, MPI_DOUBLE, state.global, state.counts, state.offsets,
              MPI_DOUBLE, 0, run_globals.mpi_comm);
  if (run_globals.mpi_rank != 0)
    return;
  hid_t file = H5Fcreate(state.request, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0)
    fail("cannot create ES request file");
  hsize_t cells = (hsize_t)state.total_cells;
  h5check(H5LTmake_dataset_double(file, "density", 1, &cells, state.density));
  h5check(H5LTmake_dataset_double(file, "source_photons", 1, &cells, state.global));
  const double dx_cm = run_globals.params.BoxSize / n * run_globals.units.UnitLength_in_cm /
                       run_globals.params.Hubble_h / (1.0 + run_globals.ZZ[snapshot]);
  attribute(file, "schema_version", 1);
  attribute(file, "snapshot", snapshot);
  attribute(file, "start_snapshot", run_globals.params.HHeStartSnapshot);
  attribute(file, "grid_dim", n);
  attribute(file, "dt_s", dt_s);
  attribute(file, "cell_volume_cm3", dx_cm * dx_cm * dx_cm);
  attribute(file, "mean_h_atoms_per_cell", mean_h_atoms());
  attribute(file, "helium_number_ratio", run_globals.params.physics.Y_He / (4.0 * (1.0 - run_globals.params.physics.Y_He)));
  attribute(file, "temperature_K", run_globals.params.HHeInitialTemperature);
  attribute(file, "redshift", run_globals.ZZ[snapshot]);
  attribute(file, "max_step_s", run_globals.params.HHeMaxStepMyr * SEC_PER_MEGAYEAR);
  attribute(file, "radius_ratio", run_globals.params.ESRadiusRatio);
  attribute(file, "max_radius_cells", run_globals.params.ESMaxRadiusCells);
  attribute(file, "source_energy_eV", run_globals.params.HHeSourceEnergyEV);
  h5check(H5LTset_attribute_string(file, "/", "source_spectrum_file", run_globals.params.HHeSourceSpectrumFile));
  h5check(H5LTset_attribute_string(file, "/", "run_identity", state.identity));
  h5check(H5Fclose(file));
}

static void run_worker(const char* response)
{
  if (exists(response))
    fail("ES response already exists; refusing to overwrite a previous run");
  char* const args[] = { run_globals.params.ESPythonExecutable, run_globals.params.ESWorkerPath,
                        "--request", state.request, "--response", (char*)response,
                        "--checkpoint", state.checkpoint, NULL };
  pid_t child;
  /* posix_spawnp passes argv directly, without shell expansion, and avoids
   * running application/library code in an MPI process after fork(). */
  int rc = posix_spawnp(&child, args[0], NULL, NULL, args, environ);
  if (rc) {
    mlog_error("ES worker launch failed: %s", strerror(rc));
    fail("could not launch ES Python worker");
  }
  int status;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    fail("ES worker failed; inspect the preceding worker traceback in run.log");
}

static void read_field(hid_t file, const char* name)
{
  int rank;
  hsize_t dimensions[1];
  H5T_class_t type;
  size_t size;
  h5check(H5LTget_dataset_ndims(file, name, &rank));
  if (rank != 1)
    fail("ES response fields must be flattened one-dimensional arrays");
  h5check(H5LTget_dataset_info(file, name, dimensions, &type, &size));
  if (dimensions[0] != (hsize_t)state.total_cells || type != H5T_FLOAT)
    fail("ES response field shape/type mismatch");
  h5check(H5LTread_dataset_double(file, name, state.global));
}

static double read_scalar_attribute(hid_t file, const char* name)
{
  hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
  if (attr < 0)
    fail("ES worker response metadata is missing");
  hid_t space = H5Aget_space(attr);
  hid_t type = H5Aget_type(attr);
  if (space < 0 || type < 0 || H5Sget_simple_extent_npoints(space) != 1 ||
      (H5Tget_class(type) != H5T_FLOAT && H5Tget_class(type) != H5T_INTEGER))
    fail("ES worker response metadata must be a numeric scalar");
  double value;
  h5check(H5Aread(attr, H5T_NATIVE_DOUBLE, &value));
  h5check(H5Tclose(type));
  h5check(H5Sclose(space));
  h5check(H5Aclose(attr));
  if (!isfinite(value))
    fail("ES worker response metadata is nonfinite");
  return value;
}

static void check_identity(hid_t file)
{
  hid_t attr = H5Aopen(file, "run_identity", H5P_DEFAULT);
  if (attr < 0)
    fail("ES worker response identity is missing");
  hid_t type = H5Aget_type(attr);
  hid_t space = H5Aget_space(attr);
  if (type < 0 || space < 0 || H5Tget_class(type) != H5T_STRING || H5Sget_simple_extent_npoints(space) != 1)
    fail("ES worker response identity is malformed");
  if (H5Tis_variable_str(type) > 0) {
    char* identity = NULL;
    h5check(H5Aread(attr, type, &identity));
    if (!identity || strcmp(identity, state.identity))
      fail("ES worker response belongs to another run");
    h5check(H5free_memory(identity));
  } else {
    const size_t size = H5Tget_size(type);
    char identity[sizeof(state.identity) + 1] = { 0 };
    if (size > sizeof(state.identity))
      fail("ES worker response identity is too long");
    h5check(H5Aread(attr, type, identity));
    if (strcmp(identity, state.identity))
      fail("ES worker response belongs to another run");
  }
  h5check(H5Sclose(space));
  h5check(H5Tclose(type));
  h5check(H5Aclose(attr));
}

static void publish(int snapshot, const char* response)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  hid_t file = -1;
  if (run_globals.mpi_rank == 0) {
    file = H5Fopen(response, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
      fail("ES worker did not create a readable response");
    if (read_scalar_attribute(file, "schema_version") != 1)
      fail("unsupported ES worker response schema");
    if (read_scalar_attribute(file, "snapshot") != snapshot)
      fail("ES worker response snapshot mismatch");
    if (read_scalar_attribute(file, "grid_dim") != state.n)
      fail("ES worker response grid mismatch");
    check_identity(file);
  }
  double sum[7] = { 0 }; /* volume xHI, weighted xHI, density, electrons, SFR, Nrec, weighted Nrec */
  double* helium_sum = allocate((size_t)state.local_cells, sizeof(double));
  const double helium_ratio = run_globals.params.physics.Y_He / (4.0 * (1.0 - run_globals.params.physics.Y_He));
  const double mean_h = mean_h_atoms();
  const char* names[] = { "xH", "xHeI", "xHeII", "xHeIII", "HII_recombinations" };
  for (int f = 0; f < 5; f++) {
    if (run_globals.mpi_rank == 0)
      read_field(file, names[f]);
    MPI_Scatterv(state.global, state.counts, state.offsets, MPI_DOUBLE,
                 state.local, state.local_cells, MPI_DOUBLE, 0, run_globals.mpi_comm);
    int invalid = 0;
    const int n = state.n;
    const int nx = (int)grids->slab_nix[run_globals.mpi_rank];
    for (int x = 0; x < nx; x++)
      for (int y = 0; y < n; y++)
        for (int z = 0; z < n; z++) {
          const int j = (x * n + y) * n + z;
          const int p = grid_index(x, y, z, n, INDEX_PADDED);
          const double value = state.local[j];
          const double density = 1.0 + grids->deltax[p];
          if (!isfinite(value) || value < 0 || (f < 4 && value > 1)) {
            invalid = 1;
            continue;
          }
          if (f == 0) {
            grids->xH[j] = (float)value;
            grids->temp_kinetic_all_gas[j] = (float)run_globals.params.HHeInitialTemperature;
            /* Multiple overlapping supports have no unique cell radius. This
             * zero field is an unavailable legacy diagnostic, not a radius
             * inferred from the allocation or a UV intensity estimate. */
            grids->r_bubble[j] = 0;
            if (value < 0.5 && grids->z_at_ionization[j] < 0)
              grids->z_at_ionization[j] = (float)run_globals.ZZ[snapshot];
            if (run_globals.params.Flag_IncludeRecombinations) {
              grids->Gamma12[j] = 0;
              grids->residual_xH[j] = (float)value;
              grids->clumping_factor[j] = 1;
              grids->t_resp[j] = 1e30f;
            }
            sum[0] += value;
            sum[1] += value * density;
            sum[2] += density;
            sum[3] += (1.0 - value) * density;
            sum[4] += grids->weighted_sfr[p];
          } else if (f < 4) {
            helium_sum[j] += value;
            if (f > 1)
              sum[3] += (f - 1) * value * density * helium_ratio;
          } else if (run_globals.params.Flag_IncludeRecombinations) {
            const double nuclei = mean_h * (1.0 + helium_ratio) * density;
            const double per_nucleus = nuclei > 0 ? value / nuclei : 0;
            if (!isfinite(per_nucleus) || per_nucleus > FLT_MAX)
              invalid = 1;
            grids->N_rec[p] = (float)per_nucleus;
            sum[5] += per_nucleus;
            sum[6] += per_nucleus * density;
          }
        }
    collective_check(invalid, "ES response contains invalid fractions or recombination counts");
  }
  int invalid = 0;
  for (int j = 0; j < state.local_cells; j++)
    if (fabs(helium_sum[j] - 1.0) > 1e-10)
      invalid = 1;
  free(helium_sum);
  collective_check(invalid, "ES response helium fractions do not sum to one");
  if (run_globals.mpi_rank == 0)
    h5check(H5Fclose(file));
  MPI_Allreduce(MPI_IN_PLACE, sum, 7, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  const double cells = state.total_cells;
  grids->volume_weighted_global_xH = sum[0] / cells;
  grids->mass_weighted_global_xH = sum[2] > 0 ? sum[1] / sum[2] : 1;
  grids->volume_weighted_global_temp_kinetic_all_gas = run_globals.params.HHeInitialTemperature;
  grids->mass_weighted_global_temp_kinetic_all_gas = run_globals.params.HHeInitialTemperature;
  grids->volume_weighted_global_weighted_sfr = sum[4] / cells * run_globals.units.UnitMass_in_g /
    run_globals.units.UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
  grids->volume_weighted_global_effective_bhar = 0;
  grids->volume_weighted_global_r_bubble = grids->mass_weighted_global_r_bubble = 0;
  grids->volume_weighted_global_Gamma12 = grids->mass_weighted_global_Gamma12 = 0;
  grids->volume_weighted_global_N_rec = sum[5] / cells;
  grids->mass_weighted_global_N_rec = sum[2] > 0 ? sum[6] / sum[2] : 0;
  grids->volume_weighted_global_residual_xH = grids->volume_weighted_global_xH;
  grids->mass_weighted_global_residual_xH = grids->mass_weighted_global_xH;
  grids->volume_weighted_global_clumping_factor = grids->mass_weighted_global_clumping_factor = 1;
  state.mean_electron_per_nucleus = sum[2] > 0 ? sum[3] / (sum[2] * (1.0 + helium_ratio)) : 0;
}

void photon_es_meraxes(int snapshot)
{
  if (snapshot < run_globals.params.HHeStartSnapshot)
    return;
  if (!state.initialized)
    initialize(snapshot);
  if (snapshot != state.last_snapshot + 1)
    fail("ES snapshot state requires consecutive calls");
  const double dt_s = snapshot == run_globals.params.HHeStartSnapshot ? 0 :
    (run_globals.LTTime[state.last_snapshot] - run_globals.LTTime[snapshot]) *
      run_globals.units.UnitTime_in_s / run_globals.params.Hubble_h;
  if (!isfinite(dt_s) || dt_s < 0 || (snapshot != run_globals.params.HHeStartSnapshot && dt_s == 0))
    fail("invalid physical snapshot interval");
  gather_input(snapshot, dt_s);
  char suffix[64], response[STRLEN + 128];
  snprintf(suffix, sizeof(suffix), "es_state_%03d.hdf5", snapshot);
  checked_path(response, sizeof(response), suffix);
  if (run_globals.mpi_rank == 0) {
    mlog("ES/HHe worker: snapshot %d, dt=%.8g Myr, grid=%d^3", MLOG_MESG, snapshot, dt_s / SEC_PER_MEGAYEAR, state.n);
    run_worker(response);
  }
  MPI_Barrier(run_globals.mpi_comm);
  publish(snapshot, response);
  state.last_snapshot = snapshot;
  mlog("ES/HHe snapshot %d complete: volume xHI=%.8g, ne/(nH+nHe)=%.8g. "
       "Full species/photon audit: %s. Legacy Gamma12/r_bubble are unavailable placeholders.",
       MLOG_MESG, snapshot, run_globals.reion_grids.volume_weighted_global_xH,
       state.mean_electron_per_nucleus, response);
}

double photon_es_mean_electron_per_nucleus(void)
{
  return state.mean_electron_per_nucleus;
}

void photon_es_meraxes_free(void)
{
  free(state.counts);
  free(state.offsets);
  free(state.local);
  free(state.density);
  free(state.global);
  memset(&state, 0, sizeof(state));
}
