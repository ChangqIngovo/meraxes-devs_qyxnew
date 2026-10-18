#define _POSIX_C_SOURCE 200809L
#include <complex.h>
#include <fftw3-mpi.h>
#include <hdf5_hl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define _MAIN
#include "core/misc_tools.h"
#include "core/photon_es_meraxes.h"
#include "core/reionization.h"
#include "meraxes.h"

/* Exercise the real MPI bridge, Python worker and HDF5 state. These prescribed
 * 2^3 snapshots test integration and unit/accounting contracts, not an EoR
 * calibration. Four ranks intentionally include two ranks with empty slabs. */
enum { SIDE = 2, CELLS = 8, SPECIES = 5, PRIMARY = 3, GROUPS = PRIMARY + 8 };

static void check(int valid, const char* message)
{
  if (!valid) {
    fprintf(stderr, "rank %d: %s\n", run_globals.mpi_rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

static void near(double actual, double expected, double rtol, double atol)
{
  if (!isfinite(actual) || !isfinite(expected) ||
      fabs(actual - expected) > atol + rtol * fmax(fabs(actual), fabs(expected))) {
    fprintf(stderr, "rank %d: %.17g != %.17g\n", run_globals.mpi_rank, actual, expected);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

static float* field(size_t size)
{
  float* result = calloc(size ? size : 1, sizeof(*result));
  check(result != NULL, "grid allocation");
  return result;
}

static double attribute(hid_t file, const char* name)
{
  double value;
  check(H5LTget_attribute_double(file, "/", name, &value) >= 0, name);
  check(isfinite(value), "nonfinite HDF5 attribute");
  return value;
}

static void values(hid_t file, const char* name, double* result)
{
  check(H5LTread_dataset_double(file, name, result) >= 0, name);
}

static double total(const double* array, size_t size)
{
  double result = 0;
  for (size_t i = 0; i < size; i++) {
    check(isfinite(array[i]), "nonfinite HDF5 state");
    result += array[i];
  }
  return result;
}

static double normalization(void)
{
  const double dx = run_globals.params.BoxSize / SIDE;
  return run_globals.params.OmegaM * run_globals.RhoCrit * dx * dx * dx *
    run_globals.params.BaryonFrac * run_globals.units.UnitMass_in_g /
    run_globals.params.Hubble_h * (1 - run_globals.params.physics.Y_He) / PROTONMASS;
}

typedef struct
{
  double hii, captures, helium_captures, external_electrons;
} audit_result;

static audit_result audit(int snapshot, const char* directory, double expected_emitted,
                          double expected_density)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/.es_checkpoint.hdf5", directory);
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  check(file >= 0, "ES checkpoint missing");
  near(attribute(file, "snapshot"), snapshot, 0, 0);
  double gas[CELLS][SPECIES], photons[CELLS][GROUPS], baseline[CELLS][2];
  double exchange[CELLS][SPECIES], captures[CELLS][3], history[CELLS], source_groups[PRIMARY];
  values(file, "gas", &gas[0][0]);
  values(file, "photons", &photons[0][0]);
  values(file, "baseline_nuclei", &baseline[0][0]);
  values(file, "matter_exchange", &exchange[0][0]);
  values(file, "capture_counts", &captures[0][0]);
  values(file, "source_history", history);
  values(file, "source_emitted_by_group", source_groups);
  const double norm = normalization(), helium = .24 / (4 * .76);
  const double emitted = attribute(file, "emitted");
  const double diffuse = attribute(file, "diffuse_emitted");
  const double absorbed = attribute(file, "absorbed");
  const double capture_total = attribute(file, "captures");
  const double initial_electrons = attribute(file, "initial_electrons");
  near(emitted * norm, expected_emitted, 3e-13, 0);
  near(total(history, CELLS), emitted, 3e-13, 0);
  near(total(source_groups, PRIMARY), emitted, 3e-13, 0);
  const double weights[PRIMARY] = { .7, .2, .1 };
  for (int g = 0; g < PRIMARY; g++)
    near(source_groups[g], emitted * weights[g], 3e-13, 0);
  /* Independent photon and electron equations, not worker diagnostic strings. */
  const double live = total(&photons[0][0], CELLS * GROUPS);
  near(emitted + diffuse, absorbed + live, 2e-10, 0);
  near(total(&captures[0][0], CELLS * 3), capture_total, 2e-12, 0);
  double electrons = 0, external_electrons = 0, nuclei = 0;
  audit_result result = { 0, capture_total, 0, 0 };
  for (int j = 0; j < CELLS; j++) {
    for (int s = 0; s < SPECIES; s++)
      check(isfinite(gas[j][s]) && gas[j][s] >= 0, "invalid gas population");
    for (int g = 0; g < GROUPS; g++)
      check(photons[j][g] >= 0, "negative photon population");
    const double nh = gas[j][0] + gas[j][1];
    const double nhe = gas[j][2] + gas[j][3] + gas[j][4];
    near(nh, expected_density, 2e-12, 0);
    near(nhe, expected_density * helium, 2e-12, 0);
    near(nh, baseline[j][0] + exchange[j][0] + exchange[j][1], 2e-12, 0);
    near(nhe, baseline[j][1] + exchange[j][2] + exchange[j][3] + exchange[j][4], 2e-12, 0);
    electrons += gas[j][1] + gas[j][3] + 2 * gas[j][4];
    external_electrons += exchange[j][1] + exchange[j][3] + 2 * exchange[j][4];
    nuclei += nh + nhe;
    result.hii += gas[j][1];
    result.helium_captures += captures[j][1] + captures[j][2];
  }
  near(initial_electrons, external_electrons, 2e-12, 1e-15);
  near(electrons - external_electrons, absorbed - capture_total, 0, 2e-10 * nuclei);
  result.external_electrons = external_electrons;
  H5Fclose(file);

  snprintf(path, sizeof(path), "%s/es_state_%03d.hdf5", directory, snapshot);
  file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  check(file >= 0, "ES response missing");
  near(attribute(file, "snapshot"), snapshot, 0, 0);
  near(attribute(file, "grid_dim"), SIDE, 0, 0);
  near(attribute(file, "dt_s"), snapshot ? 1e8 : 0, 1e-14, 0);
  near(attribute(file, "mean_electron_per_nucleus"), electrons / nuclei, 2e-12, 0);
  near(photon_es_mean_electron_per_nucleus(), electrons / nuclei, 2e-12, 0);
  double fractions[4][CELLS], recombinations[3][CELLS];
  const char* fraction_names[] = { "xH", "xHeI", "xHeII", "xHeIII" };
  const char* capture_names[] = { "HII_recombinations", "HeII_recombinations", "HeIII_recombinations" };
  for (int f = 0; f < 4; f++)
    values(file, fraction_names[f], fractions[f]);
  for (int c = 0; c < 3; c++)
    values(file, capture_names[c], recombinations[c]);
  for (int j = 0; j < CELLS; j++) {
    near(fractions[0][j], gas[j][0] / expected_density, 2e-12, 0);
    for (int s = 0; s < 3; s++) {
      near(fractions[s + 1][j], gas[j][s + 2] / (expected_density * helium), 2e-12, 0);
      near(recombinations[s][j], captures[j][s] * norm, 2e-12, 0);
    }
  }
  reion_grids_t* grids = &run_globals.reion_grids;
  int rank = run_globals.mpi_rank;
  for (int x = 0; x < grids->slab_nix[rank]; x++)
    for (int y = 0; y < SIDE; y++)
      for (int z = 0; z < SIDE; z++) {
        const int local = (x * SIDE + y) * SIDE + z;
        const int global = ((x + (int)grids->slab_ix_start[rank]) * SIDE + y) * SIDE + z;
        const int padded = grid_index(x, y, z, SIDE, INDEX_PADDED);
        near(grids->xH[local], fractions[0][global], 0, 7e-8);
        near(grids->N_rec[padded], captures[global][0] / (expected_density * (1 + helium)), 1e-7, 0);
        near(grids->temp_kinetic_all_gas[local], 10000, 0, 0);
      }
  near(grids->volume_weighted_global_xH, total(fractions[0], CELLS) / CELLS, 2e-12, 0);
  H5Fclose(file);
  return result;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  init_mlog(MPI_COMM_WORLD, stdout, stderr, stderr);
  check(argc == 3 && argv[1][0] == '/' && argv[2][0] == '/', "usage: test_photon_es_adapter /python /worker.py");
  const int rank = run_globals.mpi_rank, ranks = run_globals.mpi_size;
  check(ranks == 1 || ranks == 2 || ranks == 4, "test requires 1, 2 or 4 MPI ranks");
  ptrdiff_t starts[4], counts[4];
  for (int r = 0; r < ranks; r++) {
    counts[r] = ranks == 1 ? 2 : (r < 2 ? 1 : 0);
    starts[r] = ranks == 1 ? 0 : (r < 2 ? r : 2);
  }
  reion_grids_t* grids = &run_globals.reion_grids;
  grids->slab_ix_start = starts;
  grids->slab_nix = counts;
  const size_t cells = counts[rank] * SIDE * SIDE;
  const size_t padded = counts[rank] * SIDE * (SIDE + 2);
  grids->deltax = field(padded);
  grids->weighted_sfr = field(padded);
  grids->N_rec = field(padded);
  grids->xH = field(cells);
  grids->temp_kinetic_all_gas = field(cells);
  grids->r_bubble = field(cells);
  grids->z_at_ionization = field(cells);
  grids->Gamma12 = field(cells);
  grids->residual_xH = field(cells);
  grids->clumping_factor = field(cells);
  grids->t_resp = field(cells);
  for (size_t j = 0; j < cells; j++)
    grids->z_at_ionization[j] = -1;
  for (size_t j = 0; j < padded; j++)
    grids->N_rec[j] = 9e9f; /* A legacy sink must never enter the ES/HHe worker. */
  run_globals.params.ReionGridDim = SIDE;
  run_globals.params.BoxSize = 2;
  run_globals.params.Hubble_h = .5;
  run_globals.params.OmegaM = .3;
  run_globals.params.BaryonFrac = .16;
  run_globals.params.physics.Y_He = .24;
  run_globals.params.physics.ReionNionPhotPerBary = 6000;
  run_globals.units.UnitMass_in_g = 1e30;
  run_globals.units.UnitTime_in_s = 1e8;
  run_globals.units.UnitLength_in_cm = 1e18;
  run_globals.RhoCrit = 10;
  run_globals.params.HHeInitialTemperature = 10000;
  run_globals.params.HHeStartSnapshot = 0;
  run_globals.params.HHeSourceEnergyEV = 18;
  run_globals.params.HHeMaxStepMyr = 1;
  run_globals.params.ESRadiusRatio = 1.2;
  run_globals.params.ESMaxRadiusCells = 2;
  run_globals.params.Flag_IncludeRecombinations = 1;
  run_globals.params.Flag_PhotonConservation = 3;
  snprintf(run_globals.params.ESPythonExecutable, STRLEN, "%s", argv[1]);
  snprintf(run_globals.params.ESWorkerPath, STRLEN, "%s", argv[2]);
  double redshift[] = { 9, 8.9, 8.8 }, lookback[] = { 3, 2.5, 2 };
  run_globals.ZZ = redshift;
  run_globals.LTTime = lookback;
  run_globals.params.SnaplistLength = 3;

  /* A second star-formation event sees cumulative gal->Sfr, but must add only
   * its own escaped source rate. Exercise the production source mapper. */
  run_globals.params.physics.EscapeFracDependency = 0;
  run_globals.params.physics.EscapeFracNorm = .25;
  galaxy_t galaxy = { 0 };
  galaxy.dt = 1;
#if USE_MINI_HALOS
  galaxy.Galaxy_Population = 2;
#endif
  galaxy.Sfr = 2;
  update_galaxy_fesc_vals(&galaxy, 2, 0);
  galaxy.Sfr = 5;
  update_galaxy_fesc_vals(&galaxy, 3, 0);
  near(galaxy.FescWeightedSfr, 5 * .25, 0, 0);

  char directory[256] = "/tmp/meraxes-es-adapter.XXXXXX";
  if (!rank)
    check(mkdtemp(directory) != NULL, "temporary output directory");
  MPI_Bcast(directory, (int)sizeof(directory), MPI_CHAR, 0, MPI_COMM_WORLD);
  snprintf(run_globals.params.OutputDir, STRLEN, "%s/", directory);
  snprintf(run_globals.params.HHeSourceSpectrumFile, STRLEN, "%s/spectrum.txt", directory);
  if (!rank) {
    FILE* spectrum = fopen(run_globals.params.HHeSourceSpectrumFile, "w");
    check(spectrum != NULL, "test spectrum creation");
    check(fputs("18 .7\n30 .2\n60 .1\n", spectrum) >= 0, "test spectrum write");
    check(fclose(spectrum) == 0, "test spectrum close");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  photon_es_meraxes(0);
  audit_result initial = audit(0, directory, 0, 1);
  near(initial.hii, 0, 0, 0);
  near(initial.captures, 0, 0, 0);
  const float source = 1e-5f;
  if (starts[rank] == 0 && counts[rank] > 0)
    grids->weighted_sfr[0] = source;
  const double expected = (double)source * 1e30 / 1e8 * 6000 / PROTONMASS * 1e8;
  photon_es_meraxes(1);
  audit_result illuminated = audit(1, directory, expected, 1);
  check(illuminated.hii > 0, "escaped photons did not ionize hydrogen");
  check(illuminated.helium_captures > 0, "helium recombination was not exercised");
  const float overdensity = .2f;
  for (size_t j = 0; j < padded; j++) {
    grids->weighted_sfr[j] = 0;
    grids->deltax[j] = overdensity;
    grids->N_rec[j] = 9e9f;
  }
  photon_es_meraxes(2);
  audit_result dark = audit(2, directory, expected, 1 + (double)overdensity);
  check(dark.hii > 0, "ion state was not carried through source-off snapshot");
  check(dark.external_electrons > 0, "prescribed matter remap was not recorded");
  check(dark.captures > illuminated.captures, "source-off recombination did not advance");
  check(dark.helium_captures > illuminated.helium_captures, "source-off helium recombination did not advance");
  photon_es_meraxes_free();
  MPI_Barrier(MPI_COMM_WORLD);
  if (!rank) {
    char path[1024];
    for (int snapshot = 0; snapshot < 3; snapshot++) {
      snprintf(path, sizeof(path), "%s/es_state_%03d.hdf5", directory, snapshot);
      check(unlink(path) == 0, "response cleanup");
    }
    const char* names[] = { ".es_request.hdf5", ".es_checkpoint.hdf5", ".es_checkpoint.hdf5.lock", "spectrum.txt" };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
      snprintf(path, sizeof(path), "%s/%s", directory, names[i]);
      check(unlink(path) == 0, "state cleanup");
    }
    check(rmdir(directory) == 0, "directory cleanup");
    printf("ES/HHe adapter %d ranks: source/time units, state/remap, H/He recombination, "
           "empty slabs, HDF5 fields and independent photon/nucleus/electron ledgers passed\n", ranks);
  }
  free(grids->deltax);
  free(grids->weighted_sfr);
  free(grids->N_rec);
  free(grids->xH);
  free(grids->temp_kinetic_all_gas);
  free(grids->r_bubble);
  free(grids->z_at_ionization);
  free(grids->Gamma12);
  free(grids->residual_xH);
  free(grids->clumping_factor);
  free(grids->t_resp);
  MPI_Finalize();
  return 0;
}
