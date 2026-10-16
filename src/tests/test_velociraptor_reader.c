#define _POSIX_C_SOURCE 200809L
#include <hdf5_hl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define _MAIN
#include "core/galaxies.h"
#include "core/read_halos.h"
#include "meraxes.h"
#include "tree_flags.h"

/* Exercise the production collective reader without downloaded trees. The
 * fixture deliberately has no Tail dataset, like the supported L10 input.
 * Two forests test both unfiltered and rank-local catalogue indexing. */
enum
{
  NHALO = 4
};

static void require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "rank %d: %s\n", run_globals.mpi_rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
}

static void vector(hid_t group, const char* name, hid_t type, const void* values)
{
  const hsize_t size = NHALO;
  require(H5LTmake_dataset(group, name, 1, &size, type, values) >= 0, name);
}

static void create_fixture(const char* directory)
{
  char path[STRLEN + 64];
  snprintf(path, sizeof(path), "%s/trees", directory);
  require(mkdir(path, 0700) == 0, "create fixture trees directory");
  snprintf(path, sizeof(path), "%s/trees/fixture.hdf5", directory);
  hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  require(file >= 0, "create fixture file");
  hid_t header = H5Gcreate(file, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  require(header >= 0, "create Header");
  hid_t units = H5Gcreate(header, "Units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  require(units >= 0, "create Units");
  double mass_unit = 1e10, scale_factor = .25;
  require(H5LTset_attribute_double(file, "Header/Units", "Mass_unit_to_solarmass", &mass_unit, 1) >= 0,
          "write mass unit");
  H5Gclose(units);
  H5Gclose(header);
  hid_t snap = H5Gcreate(file, "Snap_000", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  require(snap >= 0, "create snapshot");
  int count = NHALO;
  require(H5LTset_attribute_int(file, "Snap_000", "NHalos", &count, 1) >= 0, "write halo count");
  require(H5LTset_attribute_double(file, "Snap_000", "scalefactor", &scale_factor, 1) >= 0, "write scale factor");
  long forest[] = { 101, 101, 202, 202 }, host[] = { -1, 1, -1, 3 };
  long head[] = { 1000000000001L, 1000000000002L, 1000000000003L, 1000000000004L };
  unsigned long id[] = { 1, 2, 3, 4 }, particles[] = { 100, 20, 200, 30 };
  float mass200[] = { 2, 1, 0, 1 }, mass[] = { 3, .4f, 4, .5f }, radius[] = { .1f, .1f, .2f, .1f };
  float x[] = { 1, 2, 3, 4 }, y[] = { .25f, .5f, .75f, 1 }, z[] = { 4.5f, 4, 3.5f, 3 };
  float velocity[] = { 10, 20, 30, 40 }, angular_momentum[] = { .1f, .2f, .3f, .4f };
  vector(snap, "ForestID", H5T_NATIVE_LONG, forest);
  vector(snap, "Head", H5T_NATIVE_LONG, head);
  vector(snap, "hostHaloID", H5T_NATIVE_LONG, host);
  vector(snap, "ID", H5T_NATIVE_ULONG, id);
  vector(snap, "npart", H5T_NATIVE_ULONG, particles);
  vector(snap, "Mass_200crit", H5T_NATIVE_FLOAT, mass200);
  vector(snap, "Mass_tot", H5T_NATIVE_FLOAT, mass);
  vector(snap, "R_200crit", H5T_NATIVE_FLOAT, radius);
  vector(snap, "Vmax", H5T_NATIVE_FLOAT, velocity);
  vector(snap, "Xc", H5T_NATIVE_FLOAT, x);
  vector(snap, "Yc", H5T_NATIVE_FLOAT, y);
  vector(snap, "Zc", H5T_NATIVE_FLOAT, z);
  vector(snap, "VXc", H5T_NATIVE_FLOAT, velocity);
  vector(snap, "VYc", H5T_NATIVE_FLOAT, velocity);
  vector(snap, "VZc", H5T_NATIVE_FLOAT, velocity);
  vector(snap, "AngMom", H5T_NATIVE_FLOAT, angular_momentum);
  H5Gclose(snap);
  require(H5Fclose(file) >= 0, "close fixture");
}

static void check_merger(halo_t* halo, int heavy_first)
{
  galaxy_t light = { 0 }, heavy = { 0 };
  int mergers = 0;
  light.Mvir = 2;
  heavy.Mvir = 5;
  light.HaloDescIndex = 7;
  heavy.HaloDescIndex = 19;
  for (int axis = 0; axis < 3; ++axis)
    light.Pos[axis] = heavy.Pos[axis] = NAN;
  halo->Galaxy = NULL;
  connect_galaxy_and_halo(heavy_first ? &heavy : &light, halo, &mergers);
  connect_galaxy_and_halo(heavy_first ? &light : &heavy, halo, &mergers);
  require(mergers == 1 && halo->Galaxy == &heavy, "mass selects merger parent in either arrival order");
  require(heavy.NextGalInHalo == &light && light.NextGalInHalo == NULL && light.Type == 2,
          "merger creates the expected satellite chain");
  require(light.Halo == halo && heavy.Halo == halo && light.FirstGalInHalo == &heavy && heavy.FirstGalInHalo == &heavy,
          "merger updates both galaxy links");
  copy_halo_props_to_galaxy(halo, &heavy);
  for (int axis = 0; axis < 3; ++axis)
    require(isfinite(heavy.Pos[axis]) && heavy.Pos[axis] == halo->Pos[axis] && heavy.Pos[axis] >= 0 &&
              heavy.Pos[axis] < run_globals.params.BoxSize,
            "selected parent receives finite in-box halo position");
  halo->Galaxy = NULL;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  run_globals.mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);
  init_mlog(MPI_COMM_WORLD, stdout, stderr, stderr);
  require(run_globals.mpi_size == 1 || run_globals.mpi_size == 2, "use one or two MPI ranks");
  char directory[STRLEN] = "./velociraptor-reader-XXXXXX";
  if (!run_globals.mpi_rank) {
    require(mkdtemp(directory) != NULL, "create fixture directory");
    create_fixture(directory);
  }
  MPI_Bcast(directory, sizeof(directory), MPI_CHAR, 0, MPI_COMM_WORLD);
  snprintf(run_globals.params.SimulationDir, STRLEN, "%s", directory);
  snprintf(run_globals.params.CatalogFilePrefix, STRLEN, "%s", "fixture.hdf5");
  run_globals.params.TreesID = VELOCIRAPTOR_TREES;
  run_globals.params.Hubble_h = .5;
  run_globals.params.BoxSize = 10;
  run_globals.params.OmegaM = 1;
  run_globals.params.OmegaK = run_globals.params.OmegaLambda = 0;
  run_globals.G = 1;
  double redshift[] = { 3 }, rhocrit[] = { 1 };
  run_globals.ZZ = redshift;
  run_globals.rhocrit = rhocrit;
  long requested_forest = run_globals.mpi_rank ? 202 : 101;
  if (run_globals.mpi_size == 2) {
    run_globals.RequestedForestId = &requested_forest;
    run_globals.NRequestedForests = 1;
  }
  const unsigned char poison[] = { 0x00, 0xa5, 0x5a, 0xff };
  for (int ignore = 0; ignore <= 1; ++ignore)
    for (size_t pattern = 0; pattern < sizeof(poison); ++pattern) {
      halo_t halos[NHALO];
      fof_group_t groups[NHALO];
      int lookup[NHALO], n_halos = -1, n_groups = -1;
      unsigned char value = poison[(pattern + (size_t)run_globals.mpi_rank) % sizeof(poison)];
      memset(halos, value, sizeof(halos));
      memset(groups, value, sizeof(groups));
      run_globals.params.FlagIgnoreProgIndex = ignore;
      read_trees__velociraptor(0, halos, &n_halos, groups, &n_groups, lookup);
      require(n_halos == NHALO / run_globals.mpi_size && n_groups == 2 / run_globals.mpi_size,
              "reader keeps expected forests");
      for (int i = 0; i < n_halos; ++i) {
        halo_t* halo = &halos[i];
        int original = run_globals.mpi_size == 1 ? i : 2 * run_globals.mpi_rank + i;
        int flags = TREE_CASE_NO_PROGENITORS | (original == 2 ? TREE_CASE_BELOW_VIRIAL_THRESHOLD : 0);
        require(lookup[i] == original && halo->ID == (unsigned long)original + 1, "catalogue index preserved");
        require(halo->ProgIndex == -1 && halo->TreeFlags == flags, "reader resets all progenitor state");
        require(halo->DescIndex == original && halo->SnapOffset == 1, "descendant link preserved");
        require(halo->Type == original % 2 && halo->Galaxy == NULL, "halo type and empty galaxy link");
        require(halo->FOFGroup == &groups[i / 2] && groups[i / 2].FirstHalo == &halos[i - i % 2],
                "rank-local FOF connection");
        require(halo->NextHaloInFOFGroup == (original % 2 ? NULL : &halos[i + 1]), "FOF satellite chain");
        require(isfinite(halo->Rvir) && halo->Rvir > 0 && isfinite(halo->Vvir) && halo->Vvir > 0,
                "reader computes finite virial properties");
        require(halo->Pos[0] == 2 * (original + 1) && halo->Pos[1] == .5 * (original + 1) &&
                  halo->Pos[2] == 9 - original,
                "reader converts positions");
        check_merger(halo, 0);
        check_merger(halo, 1);
      }
    }
  MPI_Barrier(MPI_COMM_WORLD);
  if (!run_globals.mpi_rank) {
    char path[STRLEN + 64];
    snprintf(path, sizeof(path), "%s/trees/fixture.hdf5", directory);
    require(unlink(path) == 0, "remove fixture file");
    snprintf(path, sizeof(path), "%s/trees", directory);
    require(rmdir(path) == 0 && rmdir(directory) == 0, "remove fixture directories");
    printf("VELOCIraptor reader and merger fallback passed with %d MPI rank(s).\n", run_globals.mpi_size);
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
