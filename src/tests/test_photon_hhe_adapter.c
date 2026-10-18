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
#include "core/photon_hhe_meraxes.h"
#include "meraxes.h"

/* Real Meraxes globals + production adapter + HDF5. No merger-tree fixture.
 * Inputs are explicitly prescribed. This tests integration/units, not EoR. */
static void check(int valid, const char* message)
{
  if (!valid) {
    fprintf(stderr, "rank %d: %s\n", run_globals.mpi_rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
static void near(double a, double b, double rtol, double atol)
{
  if (!isfinite(a) || fabs(a - b) > atol + rtol * fmax(fabs(a), fabs(b))) {
    fprintf(stderr, "rank %d: %.17g != %.17g\n", run_globals.mpi_rank, a, b);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
static void read_values(hid_t file, const char* name, double* values)
{
  check(H5LTread_dataset_double(file, name, values) >= 0, name);
}
static double sum(const double* values, size_t n)
{
  double total = 0.;
  for (size_t i = 0; i < n; ++i)
    total += values[i];
  return total;
}
static float* field(size_t n)
{
  float* p = calloc(n ? n : 1, sizeof(*p));
  check(p != NULL, "allocation");
  return p;
}

static double audit_snapshot(int snap,
                             const char* directory,
                             size_t cells,
                             double expected_primary,
                             double* speciesHII,
                             double* exchange_energy)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/hhe_state_%03d.rank%05d.hdf5", directory, snap, run_globals.mpi_rank);
  hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
  check(file >= 0, "adapter HDF5 output missing");
  double emitted[2][HHE_NGROUPS], live[2][HHE_NGROUPS], escaped[2][HHE_NGROUPS];
  double absorbed[2][HHE_NGROUPS], converted[2][HHE_NGROUPS], energies[HHE_NGROUPS], box[3];
  read_values(file, "EmittedPhotons", &emitted[0][0]);
  read_values(file, "LivePhotons", &live[0][0]);
  read_values(file, "EscapedPhotons", &escaped[0][0]);
  read_values(file, "AbsorbedPhotons", &absorbed[0][0]);
  read_values(file, "ConvertedPhotons", &converted[0][0]);
  read_values(file, "GroupEnergy_eV", energies);
  read_values(file, "BoxPhysical_cm", box);
  double local_primary = sum(emitted[0], HHE_NGROUPS), total_primary;
  MPI_Allreduce(&local_primary, &total_primary, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  near(total_primary, expected_primary, 2e-13, 1e-50);
  for (int o = 0; o < 2; ++o)
    for (int g = 0; g < HHE_NGROUPS; ++g)
      near(emitted[o][g], live[o][g] + escaped[o][g] + absorbed[o][g] + converted[o][g], 1e-12, 1e-50);
  double local_energy = 0., inactive = 0., matter = 0., initial = 0.;
  read_values(file, "LocalInactiveRadiation_eV", &inactive);
  read_values(file, "LocalMatterExchange_eV", &matter);
  read_values(file, "LocalInitialEnergy_eV", &initial);
  *exchange_energy = matter;
  double data[8], gas[6][8];
  const char* names[6] = { "nHI_cm-3", "nHII_cm-3", "nHeI_cm-3", "nHeII_cm-3", "nHeIII_cm-3", "ThermalEnergy_eV_cm-3" };
  for (int s = 0; s < 6; ++s)
    read_values(file, names[s], gas[s]);
  const double volume = box[0] * box[1] * box[2] / 8.;
  *speciesHII = sum(gas[1], cells) * volume;
  for (size_t j = 0; j < cells; ++j) {
    local_energy += (gas[5][j] + 13.6 * gas[1][j] + 24.59 * gas[3][j] + 78.99 * gas[4][j]) * volume;
    near(run_globals.reion_grids.xH[j], gas[0][j] / (gas[0][j] + gas[1][j]), 0., 7e-8);
  }
  for (int o = 0; o < 2; ++o)
    for (int g = 0; g < HHE_NGROUPS; ++g)
      local_energy += (live[o][g] + escaped[o][g] - (o == 0 ? emitted[o][g] : 0.)) * energies[g];
  local_energy += inactive - initial - matter;
  double error;
  MPI_Allreduce(&local_energy, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double scale = initial + fabs(matter) + local_primary * 18., global_scale;
  MPI_Allreduce(&scale, &global_scale, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  near(error, 0., 0., 3e-13 * global_scale);
  double seconds;
  read_values(file, "TimeSinceStart_s", &seconds);
  near(seconds, snap * 1e8, 1e-14, 0.);
  read_values(file, "Temperature_K", data);
  for (size_t j = 0; j < cells; ++j)
    check(data[j] >= HHE_T_MIN && data[j] <= HHE_T_MAX, "output temperature domain");
  /* Read the compound packet phase itself, not only its summed ledger. */
  typedef struct
  {
    double birth_s, clock_s, count;
  } phase;
  hid_t dataset = H5Dopen2(file, "Packets", H5P_DEFAULT), space = H5Dget_space(dataset);
  hsize_t number = 0;
  H5Sget_simple_extent_dims(space, &number, NULL);
  phase* packets = calloc(number ? number : 1, sizeof(*packets));
  check(packets != NULL, "packet read allocation");
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(phase));
  H5Tinsert(type, "birth_s", HOFFSET(phase, birth_s), H5T_NATIVE_DOUBLE);
  H5Tinsert(type, "clock_s", HOFFSET(phase, clock_s), H5T_NATIVE_DOUBLE);
  H5Tinsert(type, "count", HOFFSET(phase, count), H5T_NATIVE_DOUBLE);
  check(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, packets) >= 0, "packet phase read");
  for (hsize_t i = 0; i < number; ++i) {
    check(packets[i].birth_s <= packets[i].clock_s && packets[i].count > 0., "invalid packet birth/count");
    near(packets[i].clock_s, seconds, 1e-13, 1e-6);
  }
  free(packets);
  H5Tclose(type);
  H5Sclose(space);
  H5Dclose(dataset);
  H5Fclose(file);
  return total_primary;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  init_mlog(MPI_COMM_WORLD, stdout, stderr, stderr);
  const int rank = run_globals.mpi_rank, ranks = run_globals.mpi_size, n = 2;
  check(ranks >= 1 && ranks <= 4, "adapter test supports 1--4 ranks");
  ptrdiff_t starts[4], counts[4];
  for (int r = 0; r < ranks; ++r) {
    counts[r] = ranks == 1 ? 2 : (r < 2 ? 1 : 0);
    starts[r] = ranks == 1 ? 0 : (r < 2 ? r : 2);
  }
  reion_grids_t* g = &run_globals.reion_grids;
  g->slab_ix_start = starts;
  g->slab_nix = counts;
  const size_t cells = (size_t)counts[rank] * n * n, padded = (size_t)counts[rank] * n * (n + 2);
  g->deltax = field(padded);
  g->weighted_sfr = field(padded);
  g->N_rec = field(padded);
  g->xH = field(cells);
  g->temp_kinetic_all_gas = field(cells);
  g->r_bubble = field(cells);
  g->z_at_ionization = field(cells);
  g->Gamma12 = field(cells);
  g->residual_xH = field(cells);
  g->clumping_factor = field(cells);
  g->t_resp = field(cells);
  for (size_t j = 0; j < cells; ++j)
    g->z_at_ionization[j] = -1.;
  for (size_t j = 0; j < padded; ++j)
    g->N_rec[j] = 9e9f; /* Must never act as an old ES sink. */
  run_globals.params.ReionGridDim = n;
  run_globals.params.BoxSize = 2.;
  run_globals.params.Hubble_h = .5;
  run_globals.params.OmegaM = .3;
  run_globals.params.BaryonFrac = .16;
  run_globals.params.physics.Y_He = .24;
  run_globals.params.physics.ReionNionPhotPerBary = 6000.;
  run_globals.units.UnitMass_in_g = 1e30;
  run_globals.units.UnitTime_in_s = 1e8;
  run_globals.units.UnitLength_in_cm = 1e20;
  run_globals.RhoCrit = 10.;
  run_globals.params.HHeMemoryMB = 64;
  run_globals.params.HHeInitialTemperature = 10000.;
  run_globals.params.HHeStartSnapshot = 0;
  run_globals.params.HHeSourceEnergyEV = 18.;
  run_globals.params.HHeBoundaryPeriodic = 1;
  run_globals.params.HHeEmissionDirections = 6;
  run_globals.params.HHeCellCrossingFraction = .25;
  run_globals.params.HHeMaxStepMyr = 1.;
  run_globals.params.RandomSeed = 721;
  run_globals.params.Flag_IncludeRecombinations = 1;
  run_globals.params.Flag_PhotonConservation = 2;
  double redshift[3] = { 9., 8.9, 8.8 }, lookback[3] = { 3., 2.5, 2. };
  run_globals.ZZ = redshift;
  run_globals.LTTime = lookback;
  run_globals.params.SnaplistLength = 3;
  char directory[256] = "/tmp/meraxes-hhe-adapter.XXXXXX";
  if (!rank)
    check(mkdtemp(directory) != NULL, "temporary output directory");
  MPI_Bcast(directory, (int)sizeof(directory), MPI_CHAR, 0, MPI_COMM_WORLD);
  snprintf(run_globals.params.OutputDir, STRLEN, "%s/", directory);
  double hii0, hii1, hii2, external;
  photon_hhe_meraxes(0);
  audit_snapshot(0, directory, cells, 0., &hii0, &external);
  check(hii0 == 0., "initial warm gas must be neutral");
  float source = 1e-5f;
  if (starts[rank] == 0 && counts[rank] > 0)
    g->weighted_sfr[0] = source;
  const double expected = (double)source * 1e30 / 1e8 * 6000. / PROTONMASS * 1e8;
  photon_hhe_meraxes(1);
  audit_snapshot(1, directory, cells, expected, &hii1, &external);
  for (size_t j = 0; j < padded; ++j) {
    g->weighted_sfr[j] = 0.;
    g->deltax[j] = .2f;
    g->N_rec[j] = 9e9f;
  }
  photon_hhe_meraxes(2);
  audit_snapshot(2, directory, cells, expected, &hii2, &external);
  double global_hii1, global_hii2, global_external;
  MPI_Allreduce(&hii1, &global_hii1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&hii2, &global_hii2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&external, &global_external, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  check(global_hii1 > 0. && global_hii2 > 0., "ion state was not carried through source-off snapshot");
  check(global_external > 0., "prescribed matter remap was not recorded");
  check(photon_hhe_mean_electron_per_baryon() > 0., "electron publication missing");
  photon_hhe_meraxes_free();
  for (int s = 0; s < 3; ++s) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/hhe_state_%03d.rank%05d.hdf5", directory, s, rank);
    unlink(path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (!rank) {
    rmdir(directory);
    printf("H/He adapter %d ranks: SFR/time h-units, state carry, prescribed remap, HDF5 phases and independent "
           "budgets passed\n",
           ranks);
  }
  free(g->deltax);
  free(g->weighted_sfr);
  free(g->N_rec);
  free(g->xH);
  free(g->temp_kinetic_all_gas);
  free(g->r_bubble);
  free(g->z_at_ionization);
  free(g->Gamma12);
  free(g->residual_xH);
  free(g->clumping_factor);
  free(g->t_resp);
  MPI_Finalize();
  return 0;
}
