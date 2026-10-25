#include <assert.h>
#include <hdf5_hl.h>
#include <limits.h>
#include <math.h>

#include "cpu_utils.h"

#include "debug.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "modifiers.h"
#include "read_halos.h"
#include "tree_flags.h"
#include "virial_properties.h"

trees_info_t read_trees_info__velociraptor(const int snapshot)
{
  trees_info_t trees_info;

  if (run_globals.mpi_rank == 0) {
    // TODO: This could maybe only ever be done once and stored in run_globals.
    char fname[STRLEN + 34];
    switch (run_globals.params.TreesID) {
      case VELOCIRAPTOR_TREES:
        sprintf(fname, "%s/trees/meraxes_augmented_stats.h5", run_globals.params.SimulationDir);
        break;
      case VELOCIRAPTOR_TREES_AUG:
        sprintf(fname, "%s/augmented_trees/meraxes_augmented_stats.h5", run_globals.params.SimulationDir);
        break;
      default:
        mlog_error("Unrecognised input trees identifier (TreesID).");
        break;
    }

    hid_t fd = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fd < 0) {
      mlog("Failed to open file %s", MLOG_MESG, fname);
      ABORT(EXIT_FAILURE);
    }

    int n_snaps = 0;
    H5LTget_attribute_int(fd, "/", "n_snaps", &n_snaps);
    H5LTget_attribute_int(fd, "/", "n_halos_max", &(trees_info.n_halos_max));
    H5LTget_attribute_int(fd, "/", "n_fof_groups_max", &(trees_info.n_fof_groups_max));

    int* buffer = malloc(sizeof(int) * n_snaps);
    H5LTread_dataset_int(fd, "n_halos", buffer);
    trees_info.n_halos = buffer[snapshot];
    H5LTread_dataset_int(fd, "n_fof_groups", buffer);
    trees_info.n_fof_groups = buffer[snapshot];
    free(buffer);

    H5Fclose(fd);
  }

  // broadcast the snapshot info
  MPI_Bcast(&trees_info, sizeof(trees_info_t), MPI_BYTE, 0, run_globals.mpi_comm);

  return trees_info;
}

static int id_to_ind(long id)
{
  return (int)(((uint64_t)id % (uint64_t)1e12) - 1);
}

static int id_to_snap(long id)
{
  return (int)(id / 1e12l);
}

inline static void convert_input_virial_props(double* Mvir,
                                              double* Rvir,
                                              double* Vvir,
                                              double* FOFMvirModifier,
                                              const int len,
                                              const int snapshot,
                                              const bool fof_flag)
{
  // Update the virial properties for subhalos
  if (*Mvir == -1) {
    assert(len > 0);
    *Mvir = calculate_Mvir(*Mvir, len);
  } else {
    if (fof_flag && (run_globals.RequestedMassRatioModifier == 1)) {
      // Modifier the FoF mass and update the virial radius
      assert(FOFMvirModifier != NULL);
      *FOFMvirModifier =
        interpolate_modifier(run_globals.mass_ratio_modifier, log10(*Mvir / run_globals.params.Hubble_h) + 10.0);
      *Mvir *= *FOFMvirModifier;
    }
  }

  if (*Rvir == -1)
    *Rvir = calculate_Rvir(*Mvir, snapshot);

  if (*Vvir == -1)
    *Vvir = calculate_Vvir(*Mvir, *Rvir);
}

typedef struct
{
  long forest;
  int rank;
} forest_owner_t;

typedef struct
{
  uint64_t row;
  int rank;
} owned_tree_row_t;

static int compare_forest_owner(const void* a, const void* b)
{
  const forest_owner_t *x = a, *y = b;
  if (x->forest != y->forest)
    return (x->forest > y->forest) - (x->forest < y->forest);
  return (x->rank > y->rank) - (x->rank < y->rank);
}

static int compare_tree_row(const void* a, const void* b)
{
  uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
  return (x > y) - (x < y);
}

static void append_tree_row(owned_tree_row_t** rows, size_t* count, size_t* capacity, uint64_t row, int rank)
{
  if (*count == *capacity) {
    size_t next = *capacity ? *capacity * 2 : 1024;
    if (next < *capacity || next > SIZE_MAX / sizeof(**rows))
      ABORT(EXIT_FAILURE);
    void* p = realloc(*rows, next * sizeof(**rows));
    if (!p)
      ABORT(EXIT_FAILURE);
    *rows = p;
    *capacity = next;
  }
  (*rows)[(*count)++] = (owned_tree_row_t){ row, rank };
}

// Scan ForestID once across the communicator and route row numbers to forest
// owners. The larger property arrays are then read only by their destination.
static uint64_t* owned_tree_rows(hid_t group, int n_entries, size_t* n_owned)
{
  int ranks = run_globals.mpi_size, rank = run_globals.mpi_rank;
  int* counts = cpu_checked_alloc(ranks, sizeof(int));
  int* offsets = cpu_checked_alloc((size_t)ranks + 1, sizeof(int));
  int* all_forests = cpu_checked_alloc(ranks, sizeof(int));
  int local_all = run_globals.RequestedForestId == NULL && run_globals.NRequestedForests < 0;
  int local_count = run_globals.RequestedForestId ? run_globals.NRequestedForests : 0;
  if (local_count < 0)
    ABORT(EXIT_FAILURE);
  MPI_Allgather(&local_all, 1, MPI_INT, all_forests, 1, MPI_INT, run_globals.mpi_comm);
  MPI_Allgather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, run_globals.mpi_comm);
  for (int p = 0; p < ranks; ++p) {
    if (counts[p] > INT_MAX - offsets[p]) {
      mlog_error("Forest ownership directory exceeds MPI_Allgatherv integer limits.");
      ABORT(EXIT_FAILURE);
    }
    offsets[p + 1] = offsets[p] + counts[p];
  }
  long* forest_ids = cpu_checked_alloc(offsets[ranks], sizeof(long));
  long dummy = 0;
  MPI_Allgatherv(local_count ? run_globals.RequestedForestId : &dummy,
                 local_count,
                 MPI_LONG,
                 forest_ids,
                 counts,
                 offsets,
                 MPI_LONG,
                 run_globals.mpi_comm);
  forest_owner_t* owners = cpu_checked_alloc(offsets[ranks], sizeof(*owners));
  for (int p = 0; p < ranks; ++p)
    for (int i = offsets[p]; i < offsets[p + 1]; ++i)
      owners[i] = (forest_owner_t){ forest_ids[i], p };
  size_t n_owners = offsets[ranks];
  qsort(owners, n_owners, sizeof(*owners), compare_forest_owner);
  int n_all = 0;
  for (int p = 0; p < ranks; ++p)
    if (all_forests[p])
      all_forests[n_all++] = p;
  free(forest_ids);
  free(counts);
  free(offsets);

  hid_t dataset = H5Dopen(group, "ForestID", H5P_DEFAULT);
  hid_t space = H5Dget_space(dataset);
  hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer, H5FD_MPIO_INDEPENDENT);
  const hsize_t chunk = 65536;
  long* buffer = cpu_checked_alloc(chunk, sizeof(long));
  hsize_t begin = (uint64_t)n_entries * rank / ranks;
  hsize_t end = (uint64_t)n_entries * (rank + 1) / ranks;
  owned_tree_row_t* routes = NULL;
  size_t n_routes = 0, capacity = 0;
  for (hsize_t first = begin; first < end; first += chunk) {
    hsize_t n = end - first < chunk ? end - first : chunk;
    H5Sselect_hyperslab(space, H5S_SELECT_SET, &first, NULL, &n, NULL);
    hid_t memory = H5Screate_simple(1, &n, NULL);
    if (H5Dread(dataset, H5T_NATIVE_LONG, memory, space, xfer, buffer) < 0)
      ABORT(EXIT_FAILURE);
    H5Sclose(memory);
    for (hsize_t i = 0; i < n; ++i) {
      size_t lo = 0, hi = n_owners;
      while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (owners[mid].forest < buffer[i])
          lo = mid + 1;
        else
          hi = mid;
      }
      int previous_rank = -1;
      for (size_t j = lo; j < n_owners && owners[j].forest == buffer[i]; ++j) {
        int p = owners[j].rank;
        if (p != previous_rank)
          append_tree_row(&routes, &n_routes, &capacity, first + i, p);
        previous_rank = p;
      }
      for (int p = 0; p < n_all; ++p)
        append_tree_row(&routes, &n_routes, &capacity, first + i, all_forests[p]);
    }
  }
  H5Pclose(xfer);
  H5Sclose(space);
  H5Dclose(dataset);
  free(buffer);
  free(owners);
  free(all_forests);

  size_t* send_count = cpu_checked_alloc(ranks, sizeof(size_t));
  size_t* cursor = cpu_checked_alloc((size_t)ranks + 1, sizeof(size_t));
  for (size_t i = 0; i < n_routes; ++i)
    ++send_count[routes[i].rank];
  for (int p = 0; p < ranks; ++p)
    cursor[p + 1] = cursor[p] + send_count[p];
  uint64_t* send = cpu_checked_alloc(n_routes, sizeof(uint64_t));
  for (size_t i = 0; i < n_routes; ++i)
    send[cursor[routes[i].rank]++] = routes[i].row;
  free(routes);
  free(cursor);
  size_t *recv_count, *recv_offset;
  uint64_t* rows =
    exchange_records(send, send_count, sizeof(uint64_t), &recv_count, &recv_offset, run_globals.mpi_comm);
  *n_owned = recv_offset[ranks];
  qsort(rows, *n_owned, sizeof(*rows), compare_tree_row);
  free(send);
  free(send_count);
  free(recv_count);
  free(recv_offset);
  return rows;
}

void read_trees__velociraptor(int snapshot,
                              halo_t* halos,
                              int* n_halos,
                              fof_group_t* fof_groups,
                              int* n_fof_groups,
                              int* index_lookup)
{
  switch (run_globals.params.TreesID) {
    case VELOCIRAPTOR_TREES:
      mlog("Reading velociraptor trees for snapshot %d...", MLOG_OPEN, snapshot);
      break;
    case VELOCIRAPTOR_TREES_AUG:
      mlog("Reading velociraptor augmented trees for snapshot %d...", MLOG_OPEN, snapshot);
      break;
    default:
      mlog_error("Unrecognised input trees identifier (TreesID).");
      break;
  }

  int n_tree_entries = 0;
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, run_globals.mpi_comm, MPI_INFO_NULL);

  double mass_unit_to_internal = 1.0;
  double scale_factor = -999.;

  *n_halos = 0;
  *n_fof_groups = 0;

  char fname[STRLEN * 2 + 8];
  switch (run_globals.params.TreesID) {
    case VELOCIRAPTOR_TREES:
      sprintf(fname, "%s/trees/%s", run_globals.params.SimulationDir, run_globals.params.CatalogFilePrefix);
      break;
    case VELOCIRAPTOR_TREES_AUG:
      sprintf(fname, "%s/augmented_trees/%s", run_globals.params.SimulationDir, run_globals.params.CatalogFilePrefix);
      break;
    default:
      mlog_error("Unrecognised input trees identifier (TreesID).");
      break;
  }

  hid_t fd = H5Fopen(fname, H5F_ACC_RDONLY, plist_id);
  if (fd < 0) {
    mlog("Failed to open file %s", MLOG_MESG, fname);
    ABORT(EXIT_FAILURE);
  }
  H5Pclose(plist_id);

  char snap_group_name[9];
  sprintf(snap_group_name, "Snap_%03d", snapshot);
  hid_t snap_group = H5Gopen(fd, snap_group_name, H5P_DEFAULT);

  H5LTget_attribute_int(fd, snap_group_name, "NHalos", &n_tree_entries);

  // check the units
  H5LTget_attribute_double(fd, "Header/Units", "Mass_unit_to_solarmass", &mass_unit_to_internal);
  mass_unit_to_internal /= 1.0e10;
  H5LTget_attribute_double(fd, snap_group_name, "scalefactor", &scale_factor);

  size_t n_owned = 0;
  uint64_t* owned_rows = owned_tree_rows(snap_group, n_tree_entries, &n_owned);
  size_t buffer_size = n_owned < 65536 ? n_owned : 65536;

  long* Head = cpu_checked_alloc(buffer_size, sizeof(long));
  long* hostHaloID = cpu_checked_alloc(buffer_size, sizeof(long));
  float* Mass_200crit = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Mass_tot = cpu_checked_alloc(buffer_size, sizeof(float));
  float* R_200crit = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Vmax = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Xc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Yc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* Zc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* VXc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* VYc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* VZc = cpu_checked_alloc(buffer_size, sizeof(float));
  float* AngMom = cpu_checked_alloc(buffer_size, sizeof(float));
  unsigned long* ID = cpu_checked_alloc(buffer_size, sizeof(unsigned long));
  unsigned long* npart = cpu_checked_alloc(buffer_size, sizeof(unsigned long));

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT); // or H5FD_MPIO_COLLECTIVE?
  hid_t fspace_id = H5Screate_simple(1, (hsize_t[1]){ n_tree_entries }, NULL);

  double hubble_h = run_globals.params.Hubble_h;

  struct
  {
    const char* name;
    hid_t type, dataset;
    void* data;
  } columns[] = {
    { "Head", H5T_NATIVE_LONG, -1, Head },
    { "hostHaloID", H5T_NATIVE_LONG, -1, hostHaloID },
    { "Mass_200crit", H5T_NATIVE_FLOAT, -1, Mass_200crit },
    { "Mass_tot", H5T_NATIVE_FLOAT, -1, Mass_tot },
    { "R_200crit", H5T_NATIVE_FLOAT, -1, R_200crit },
    { "Vmax", H5T_NATIVE_FLOAT, -1, Vmax },
    { "Xc", H5T_NATIVE_FLOAT, -1, Xc },
    { "Yc", H5T_NATIVE_FLOAT, -1, Yc },
    { "Zc", H5T_NATIVE_FLOAT, -1, Zc },
    { "VXc", H5T_NATIVE_FLOAT, -1, VXc },
    { "VYc", H5T_NATIVE_FLOAT, -1, VYc },
    { "VZc", H5T_NATIVE_FLOAT, -1, VZc },
    { "AngMom", H5T_NATIVE_FLOAT, -1, AngMom },
    { "ID", H5T_NATIVE_ULONG, -1, ID },
    { "npart", H5T_NATIVE_ULONG, -1, npart },
  };
  const size_t n_columns = sizeof(columns) / sizeof(columns[0]);
  // All ranks open/close datasets in the same order, including ranks with no
  // selected rows. Only the independent H5Dread calls vary between ranks.
  for (size_t p = 0; p < n_columns; ++p) {
    columns[p].dataset = H5Dopen(snap_group, columns[p].name, H5P_DEFAULT);
    if (columns[p].dataset < 0)
      ABORT(EXIT_FAILURE);
  }
  hsize_t* selection = cpu_checked_alloc(buffer_size, sizeof(hsize_t));
  size_t n_read = 0;
  while (n_read < n_owned) {
    int n_to_read = (int)(n_owned - n_read < buffer_size ? n_owned - n_read : buffer_size);
    for (int i = 0; i < n_to_read; ++i)
      selection[i] = owned_rows[n_read + i];
    if (H5Sselect_elements(fspace_id, H5S_SELECT_SET, n_to_read, selection) < 0)
      ABORT(EXIT_FAILURE);
    hid_t memspace_id = H5Screate_simple(1, (hsize_t[1]){ n_to_read }, NULL);
    for (size_t p = 0; p < n_columns; ++p)
      if (H5Dread(columns[p].dataset, columns[p].type, memspace_id, fspace_id, plist_id, columns[p].data) < 0)
        ABORT(EXIT_FAILURE);
    H5Sclose(memspace_id);

    for (int ii = 0; ii < n_to_read; ++ii) {
      halo_t* halo = &(halos[*n_halos]);

      halo->ID = ID[ii];
      halo->DescIndex = id_to_ind(Head[ii]);

      if (run_globals.params.FlagIgnoreProgIndex)
        halo->ProgIndex = -1;
      // else
      //   halo->ProgIndex = id_to_ind(Tail[ii]);

      halo->NextHaloInFOFGroup = NULL;
      halo->Type = hostHaloID[ii] == -1 ? 0 : 1;
      halo->SnapOffset = id_to_snap(Head[ii]) - snapshot;

      // Any other tree flags need to be set using both the current and
      // progenitor halo information (stored in the galaxy), therefore we
      // need to leave setting those until later...
      if (run_globals.params.FlagIgnoreProgIndex)
        halo->TreeFlags = TREE_CASE_NO_PROGENITORS;
      // else
      //   halo->TreeFlags = (unsigned long)Tail[ii] != ID[ii] ? 0 : TREE_CASE_NO_PROGENITORS;

      // Here we have a cyclic pointer, indicating that this halo's life ends here
      if ((unsigned long)Head[ii] == ID[ii])
        halo->DescIndex = -1;

      if (index_lookup)
        index_lookup[*n_halos] = (int)owned_rows[n_read + ii];

      // TODO: What masses and radii should I use for centrals (inclusive vs. exclusive etc.)?
      if (halo->Type == 0) {
        fof_group_t* fof_group = &fof_groups[*n_fof_groups];

        // This check is to ensure sensible values of mass_200crit and avoid having
        // very weird halos. Put this check back if you feel that the N-body is weird.

        /*if ((Mass_200crit[ii] < 5 * Mass_tot[ii])) {
            fof_group->Mvir = Mass_200crit[ii] * hubble_h * mass_unit_to_internal;;
            fof_group->Rvir = R_200crit[ii] * hubble_h;
        }*/
        // else {
        //  BELOW_VIRIAL_THRESHOLD merger halo swammping
        if (Mass_200crit[ii] <= 0) {
          halo->TreeFlags |= TREE_CASE_BELOW_VIRIAL_THRESHOLD;
          fof_group->Mvir = Mass_tot[ii] * hubble_h * mass_unit_to_internal;
          fof_group->Rvir = -1;
        }

        else {
          fof_group->Mvir = (double)Mass_200crit[ii] * hubble_h * mass_unit_to_internal;
          fof_group->Rvir = (double)R_200crit[ii] * hubble_h;
        }

        fof_group->Vvir = -1;
        fof_group->FOFMvirModifier = 1.0;

        convert_input_virial_props(
          &fof_group->Mvir, &fof_group->Rvir, &fof_group->Vvir, &fof_group->FOFMvirModifier, -1, snapshot, true);

        halo->FOFGroup = &(fof_groups[*n_fof_groups]);
        fof_groups[(*n_fof_groups)++].FirstHalo = halo;
      } else {
        // We can take advantage of the fact that host halos always
        // seem to appear before their subhalos (checked below) in the
        // trees to immediately connect FOF group members.
        int host_index = id_to_ind(hostHaloID[ii]);

        if (index_lookup)
          host_index = find_original_index(host_index, index_lookup, *n_halos);

        assert(host_index > -1);
        assert(host_index < *n_halos);

        halo_t* prev_halo = &halos[host_index];
        halo->FOFGroup = prev_halo->FOFGroup;

        while (prev_halo->NextHaloInFOFGroup != NULL)
          prev_halo = prev_halo->NextHaloInFOFGroup;

        prev_halo->NextHaloInFOFGroup = halo;
      }

      halo->Len = (int)npart[ii];
      halo->Pos[0] = apply_pbc_pos(Xc[ii] * hubble_h / scale_factor);
      halo->Pos[1] = apply_pbc_pos(Yc[ii] * hubble_h / scale_factor);
      halo->Pos[2] = apply_pbc_pos(Zc[ii] * hubble_h / scale_factor);
      halo->Vel[0] = VXc[ii] / scale_factor;
      halo->Vel[1] = VYc[ii] / scale_factor;
      halo->Vel[2] = VZc[ii] / scale_factor;
      halo->Vmax = Vmax[ii];

      // TODO: What masses and radii should I use for satellites (inclusive vs. exclusive etc.)?
      halo->Mvir = (double)Mass_tot[ii] * hubble_h * mass_unit_to_internal;
      halo->Rvir = -1;
      halo->Vvir = -1;
      convert_input_virial_props(&halo->Mvir, &halo->Rvir, &halo->Vvir, NULL, -1, snapshot, false);

      halo->AngMom = AngMom[ii] * hubble_h;
      halo->Galaxy = NULL;

      (*n_halos)++;
    }

    n_read += n_to_read;
  }

  for (size_t p = 0; p < n_columns; ++p)
    H5Dclose(columns[p].dataset);
  free(selection);
  free(owned_rows);
  free(Head);
  free(hostHaloID);
  free(Mass_200crit);
  free(Mass_tot);
  free(R_200crit);
  free(Vmax);
  free(Xc);
  free(Yc);
  free(Zc);
  free(VXc);
  free(VYc);
  free(VZc);
  free(AngMom);
  free(ID);
  free(npart);
  H5Pclose(plist_id);
  H5Sclose(fspace_id);
  H5Gclose(snap_group);
  H5Fclose(fd);

  mlog("...done", MLOG_CLOSE);
}
