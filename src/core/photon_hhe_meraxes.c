#include "photon_hhe_meraxes.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "photon_hhe_mpi.h"

#include <errno.h>
#include <hdf5_hl.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Persistent state belongs to FFTW's x slab. The enclosing Meraxes run owns
 * the prescribed density/source history, while this module owns ions, heat,
 * photons and all recombinations. It never reads the old ES N_rec sink. */
typedef struct
{
  int initialized, last_snapshot;
  hhe_state* gas;
  double* rates;
  hhe_mesh mesh;
  hhe_transport transport;
  double spectrum[HHE_NGROUPS];
  double time_s, initial_energy_eV;
  double matter_exchange_eV, species_exchange[5];
  double mean_electron_per_baryon;
} hhe_meraxes_state;

static hhe_meraxes_state state;

static void fail(const char* reason)
{
  mlog_error("HHe transport: %s", reason);
  ABORT(EXIT_FAILURE);
}

static void require_collective(int local_status, const char* stage)
{
  int global_status;
  MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, run_globals.mpi_comm);
  if (global_status) {
    if (local_status)
      mlog_error("HHe %s failed on rank %d: %s (status=%d, cell=%zu, chemistry=%d)",
                 stage,
                 run_globals.mpi_rank,
                 state.transport.error,
                 local_status,
                 state.transport.error_cell,
                 state.transport.chemistry_status);
    fail(stage);
  }
}

static void* allocate(size_t n, size_t item)
{
  if (item && n > SIZE_MAX / item)
    fail("allocation size overflow");
  void* p = calloc(n ? n : 1, item);
  require_collective(p ? 0 : HHE_TRANSPORT_MEMORY, "state allocation");
  return p;
}

static void load_spectrum(void)
{
  int status = 0;
  if (run_globals.mpi_rank == 0) {
    const char* path = run_globals.params.HHeSourceSpectrumFile;
    if (!path[0]) {
      status = hhe_project_photons(run_globals.params.HHeSourceEnergyEV, 1.0, 0, state.spectrum);
    } else {
      FILE* input = fopen(path, "r");
      if (!input)
        status = HHE_INVALID_INPUT;
      char line[1024];
      double weight_sum = 0;
      if (input) {
        while (fgets(line, sizeof(line), input)) {
          char* p = line;
          while (*p == ' ' || *p == '\t')
            p++;
          if (!*p || *p == '#' || *p == '\n' || *p == '\r')
            continue;
          double energy, weight;
          char extra;
          if (sscanf(p, "%lf %lf %c", &energy, &weight, &extra) != 2 || !isfinite(weight) || weight < 0 ||
              !isfinite(energy)) {
            status = HHE_INVALID_INPUT;
            break;
          }
          status = hhe_project_photons(energy, weight, 0, state.spectrum);
          if (status)
            break;
          weight_sum += weight;
        }
        if (ferror(input) || !(weight_sum > 0) || !isfinite(weight_sum))
          status = HHE_INVALID_INPUT;
        fclose(input);
        if (!status)
          for (int g = 0; g < HHE_NGROUPS; g++)
            state.spectrum[g] /= weight_sum;
      }
    }
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, run_globals.mpi_comm);
  if (status)
    fail("invalid HHe source spectrum; use supported ionizing energies and nonnegative photon weights");
  MPI_Bcast(state.spectrum, HHE_NGROUPS, MPI_DOUBLE, 0, run_globals.mpi_comm);
}

static hhe_mesh mesh_at(int snapshot)
{
  hhe_mesh mesh;
  const int n = run_globals.params.ReionGridDim;
  for (int a = 0; a < 3; a++) {
    mesh.dimensions[a] = n;
    mesh.box_cm[a] = run_globals.params.BoxSize * run_globals.units.UnitLength_in_cm / run_globals.params.Hubble_h /
                     (1.0 + run_globals.ZZ[snapshot]);
  }
  mesh.slab_x_start = (int)run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank];
  mesh.slab_x_count = (int)run_globals.reion_grids.slab_nix[run_globals.mpi_rank];
  mesh.periodic = run_globals.params.HHeBoundaryPeriodic;
  return mesh;
}

static double mean_cell_mass_g(void)
{
  const double dx = run_globals.params.BoxSize / run_globals.params.ReionGridDim;
  return run_globals.params.OmegaM * run_globals.RhoCrit * dx * dx * dx * run_globals.params.BaryonFrac *
         run_globals.units.UnitMass_in_g / run_globals.params.Hubble_h;
}

static void initialize(int snapshot)
{
  state.mesh = mesh_at(snapshot);
  const size_t cells = hhe_mesh_local_cells(&state.mesh);
  const size_t limit = (size_t)run_globals.params.HHeMemoryMB * 1024 * 1024;
  if (cells > SIZE_MAX / (sizeof(hhe_state) + (HHE_NGROUPS + 1) * sizeof(double)))
    fail("local state allocation overflow");
  /* Reserve one per-cell output workspace as well as the persistent fields.
   * MPI exchange buffers and HDF5 internals remain additional process RSS. */
  const size_t base_bytes = cells * (sizeof(hhe_state) + (HHE_NGROUPS + 1) * sizeof(double));
  if (base_bytes >= limit)
    fail("HHeMemoryMB cannot hold the local gas and source grids");
  state.gas = allocate(cells, sizeof(*state.gas));
  state.rates = allocate(cells * HHE_NGROUPS, sizeof(*state.rates));
  hhe_transport_init(&state.transport, (uint64_t)(unsigned int)run_globals.params.RandomSeed, limit - base_bytes);
  load_spectrum();
  const double volume = hhe_mesh_cell_volume(&state.mesh);
  const double mean_rho = mean_cell_mass_g() / volume;
  const double Y = run_globals.params.physics.Y_He;
  int invalid = 0;
  for (int x = 0; x < state.mesh.slab_x_count; x++)
    for (int y = 0; y < state.mesh.dimensions[1]; y++)
      for (int z = 0; z < state.mesh.dimensions[2]; z++) {
        const int n = state.mesh.dimensions[0];
        const size_t j = ((size_t)x * n + y) * n + z;
        const int p = grid_index(x, y, z, n, INDEX_PADDED);
        const double density = 1.0 + run_globals.reion_grids.deltax[p];
        if (!isfinite(density) || density < 0) {
          invalid = 1;
          continue;
        }
        hhe_state* gas = state.gas + j;
        gas->species[0] = mean_rho * density * (1.0 - Y) / PROTONMASS;
        gas->species[2] = mean_rho * density * Y / (4.0 * PROTONMASS);
        gas->thermal_eV_cm3 =
          1.5 * HHE_KB_EV_K * run_globals.params.HHeInitialTemperature * (gas->species[0] + gas->species[2]);
        state.initial_energy_eV += gas->thermal_eV_cm3 * volume;
      }
  require_collective(invalid, "initial density validation");
  state.initialized = 1;
  state.last_snapshot = snapshot;
  mlog("HHe driven-snapshot mode initialized at snapshot %d: T=%.8g K; no photons before this start. "
       "Each interval uses prescribed density; remaps preserve ion fractions/T and book external matter energy. "
       "Radiation redshifting, hydrodynamic advection and cosmological thermal evolution are omitted.",
       MLOG_MESG,
       snapshot,
       run_globals.params.HHeInitialTemperature);
}

static void remap(int snapshot)
{
  const hhe_mesh next_mesh = mesh_at(snapshot);
  const double old_volume = hhe_mesh_cell_volume(&state.mesh);
  const double new_volume = hhe_mesh_cell_volume(&next_mesh);
  const double mean_rho = mean_cell_mass_g() / new_volume;
  const double Y = run_globals.params.physics.Y_He;
  int status = 0;
  for (int x = 0; x < state.mesh.slab_x_count; x++)
    for (int y = 0; y < state.mesh.dimensions[1]; y++)
      for (int z = 0; z < state.mesh.dimensions[2]; z++) {
        const int n = state.mesh.dimensions[0];
        const size_t j = ((size_t)x * n + y) * n + z;
        const int p = grid_index(x, y, z, n, INDEX_PADDED);
        const double density = 1.0 + run_globals.reion_grids.deltax[p];
        if (!isfinite(density) || density < 0) {
          status = HHE_INVALID_INPUT;
          continue;
        }
        int rc = photon_hhe_driven_remap(state.gas + j,
                                         mean_rho * density * (1.0 - Y) / PROTONMASS,
                                         mean_rho * density * Y / (4.0 * PROTONMASS),
                                         run_globals.params.HHeInitialTemperature,
                                         old_volume,
                                         new_volume,
                                         state.species_exchange,
                                         &state.matter_exchange_eV);
        if (rc)
          status = rc;
      }
  require_collective(status, "prescribed density/volume remap");
  require_collective(hhe_transport_rescale_positions(&state.transport, state.mesh.box_cm, next_mesh.box_cm),
                     "prescribed position remap");
  state.mesh = next_mesh;
}

static void prepare_sources(void)
{
  const double factor = run_globals.units.UnitMass_in_g / run_globals.units.UnitTime_in_s *
                        run_globals.params.physics.ReionNionPhotPerBary / PROTONMASS;
  const int n = state.mesh.dimensions[0];
  int invalid = 0;
  for (int x = 0; x < state.mesh.slab_x_count; x++)
    for (int y = 0; y < n; y++)
      for (int z = 0; z < n; z++) {
        const size_t j = ((size_t)x * n + y) * n + z;
        const int p = grid_index(x, y, z, n, INDEX_PADDED);
        /* This field is an escaped instantaneous SFR per cell. Never subtract
         * cumulative stellar grids at changing galaxy positions. */
        const double rate = run_globals.reion_grids.weighted_sfr[p] * factor;
        if (!isfinite(rate) || rate < 0) {
          invalid = 1;
          continue;
        }
        for (int g = 0; g < HHE_NGROUPS; g++)
          state.rates[j * HHE_NGROUPS + g] = rate * state.spectrum[g];
      }
  require_collective(invalid, "escaped instantaneous stellar-source validation");
}

static double gas_total_energy(void)
{
  double sum = 0;
  const double volume = hhe_mesh_cell_volume(&state.mesh);
  for (size_t j = 0; j < hhe_mesh_local_cells(&state.mesh); j++)
    sum += (state.gas[j].thermal_eV_cm3 + hhe_binding_energy(state.gas + j)) * volume;
  return sum;
}

static void publish_grids(int snapshot)
{
  reion_grids_t* g = &run_globals.reion_grids;
  const int n = state.mesh.dimensions[0];
  const double V = hhe_mesh_cell_volume(&state.mesh);
  const double ncells = (double)n * n * n;
  /* volume xHI, number-weighted xHI, nuclei, volume T, weighted T,
   * electrons, escaped SFR, total Gamma, weighted Gamma. */
  double sum[11] = { 0 };
  double* gamma = allocate(hhe_mesh_local_cells(&state.mesh), sizeof(double));
  for (size_t p = 0; p < state.transport.size; p++) {
    const hhe_packet* packet = state.transport.packets + p;
    if (packet->location != HHE_PACKET_LOCAL)
      fail("nonlocal packet at output");
    size_t j = ((size_t)(packet->cell[0] - state.mesh.slab_x_start) * n + packet->cell[1]) * n + packet->cell[2];
    gamma[j] += HHE_C_LIGHT * packet->count / V * hhe_group_cross_section_cm2[packet->group][0];
  }
  for (int x = 0; x < state.mesh.slab_x_count; x++)
    for (int y = 0; y < n; y++)
      for (int z = 0; z < n; z++) {
        const size_t j = ((size_t)x * n + y) * n + z;
        const int p = grid_index(x, y, z, n, INDEX_PADDED);
        const hhe_state* gas = state.gas + j;
        const double H = gas->species[0] + gas->species[1];
        const double He = gas->species[2] + gas->species[3] + gas->species[4];
        const double nuclei = H + He;
        const double neutral = H > 0 ? gas->species[0] / H : 1.0;
        const double T = nuclei > 0 ? hhe_temperature(gas) : 0;
        g->xH[j] = (float)neutral;
        g->temp_kinetic_all_gas[j] = (float)T;
        g->r_bubble[j] = 0; /* A local transport field has no ES filter radius. */
        if (neutral < 0.5 && g->z_at_ionization[j] < 0)
          g->z_at_ionization[j] = (float)run_globals.ZZ[snapshot];
        if (run_globals.params.Flag_IncludeRecombinations) {
          g->Gamma12[j] = (float)(gamma[j] / (1e-12 * run_globals.params.Hubble_h * run_globals.params.Hubble_h));
          g->residual_xH[j] = (float)neutral;
          g->clumping_factor[j] = 1;
          hhe_coefficients coeff;
          double response = gamma[j];
          if (nuclei > 0) {
            if (hhe_coefficients_at(T, &coeff))
              fail("output temperature is outside the HHe model domain");
            response += coeff.alpha_down[0] * hhe_electron_density(gas);
          }
          g->t_resp[j] = response > 0 ? (float)(1.0 / response / SEC_PER_MEGAYEAR) : 1e30f;
          /* Diagnostic gross HI recombinations per current gas nucleus. The
           * quantity is never used as a sink by this transport model. */
          const double rec = state.transport.cell_recombinations ? state.transport.cell_recombinations[j * 3] : 0;
          g->N_rec[p] = nuclei > 0 ? (float)(rec / (nuclei * V)) : 0;
          sum[9] += g->N_rec[p];
          sum[10] += g->N_rec[p] * nuclei;
        }
        sum[0] += neutral;
        sum[1] += neutral * nuclei;
        sum[2] += nuclei;
        sum[3] += T;
        sum[4] += T * nuclei;
        sum[5] += hhe_electron_density(gas);
        sum[6] += g->weighted_sfr[p];
        sum[7] += gamma[j];
        sum[8] += gamma[j] * nuclei;
      }
  free(gamma);
  MPI_Allreduce(MPI_IN_PLACE, sum, 11, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  g->volume_weighted_global_xH = sum[0] / ncells;
  g->mass_weighted_global_xH = sum[2] > 0 ? sum[1] / sum[2] : 1;
  g->volume_weighted_global_temp_kinetic_all_gas = sum[3] / ncells;
  g->mass_weighted_global_temp_kinetic_all_gas = sum[2] > 0 ? sum[4] / sum[2] : 0;
  g->volume_weighted_global_weighted_sfr =
    sum[6] / ncells * run_globals.units.UnitMass_in_g / run_globals.units.UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
  g->volume_weighted_global_effective_bhar = 0;
  g->volume_weighted_global_r_bubble = g->mass_weighted_global_r_bubble = 0;
  const double gamma_unit = 1e-12 * run_globals.params.Hubble_h * run_globals.params.Hubble_h;
  g->volume_weighted_global_Gamma12 = sum[7] / ncells / gamma_unit;
  g->mass_weighted_global_Gamma12 = sum[2] > 0 ? sum[8] / sum[2] / gamma_unit : 0;
  g->volume_weighted_global_N_rec = sum[9] / ncells;
  g->mass_weighted_global_N_rec = sum[2] > 0 ? sum[10] / sum[2] : 0;
  g->volume_weighted_global_residual_xH = g->volume_weighted_global_xH;
  g->mass_weighted_global_residual_xH = g->mass_weighted_global_xH;
  g->volume_weighted_global_clumping_factor = g->mass_weighted_global_clumping_factor = 1;
  state.mean_electron_per_baryon = sum[2] > 0 ? sum[5] / sum[2] : 0;
}

static void h5check(herr_t result)
{
  if (result < 0)
    fail("could not write HHe state output");
}

static void dataset(hid_t file, const char* name, int rank, const hsize_t* dims, hid_t type, const void* data)
{
  h5check(H5LTmake_dataset(file, name, rank, dims, type, data));
}

static void scalar(hid_t file, const char* name, double value)
{
  hsize_t one = 1;
  dataset(file, name, 1, &one, H5T_NATIVE_DOUBLE, &value);
}

static void write_packets(hid_t file)
{
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(hhe_packet));
  hsize_t three = 3;
  hid_t vector_double = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, &three);
  hid_t vector_int = H5Tarray_create2(H5T_NATIVE_INT, 1, &three);
#define MEMBER(field, dtype) h5check(H5Tinsert(type, #field, HOFFSET(hhe_packet, field), dtype))
  MEMBER(position_cm, vector_double);
  MEMBER(direction, vector_double);
  MEMBER(count, H5T_NATIVE_DOUBLE);
  MEMBER(energy_eV, H5T_NATIVE_DOUBLE);
  MEMBER(birth_s, H5T_NATIVE_DOUBLE);
  MEMBER(clock_s, H5T_NATIVE_DOUBLE);
  MEMBER(remaining_s, H5T_NATIVE_DOUBLE);
  MEMBER(scatter_tau, H5T_NATIVE_DOUBLE);
  MEMBER(rng_state, H5T_NATIVE_UINT64);
  MEMBER(emitter_id, H5T_NATIVE_UINT64);
  MEMBER(cell, vector_int);
  MEMBER(group, H5T_NATIVE_INT);
  MEMBER(origin, H5T_NATIVE_INT);
  MEMBER(location, H5T_NATIVE_INT);
#undef MEMBER
  hsize_t count = state.transport.size;
  dataset(file, "Packets", 1, &count, type, state.transport.packets);
  H5Tclose(vector_int);
  H5Tclose(vector_double);
  H5Tclose(type);
}

static void write_output(int snapshot, double elapsed_s)
{
  double live[2][HHE_NGROUPS];
  hhe_transport_live_counts(&state.transport, live);
  double energy[7] = { state.initial_energy_eV,
                       state.matter_exchange_eV,
                       gas_total_energy(),
                       0,
                       0,
                       0,
                       state.transport.ledger.chemistry.inactive_radiation };
  for (int o = 0; o < 2; o++)
    for (int g = 0; g < HHE_NGROUPS; g++) {
      if (!o)
        energy[3] += state.transport.ledger.emitted[o][g] * hhe_group_energy_eV[g];
      energy[4] += live[o][g] * hhe_group_energy_eV[g];
      energy[5] += state.transport.ledger.escaped[o][g] * hhe_group_energy_eV[g];
    }
  double global_energy[7];
  MPI_Allreduce(energy, global_energy, 7, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  const double residual = global_energy[0] + global_energy[1] + global_energy[3] - global_energy[2] - global_energy[4] -
                          global_energy[5] - global_energy[6];
  const double scale = fabs(global_energy[0]) + fabs(global_energy[1]) + fabs(global_energy[3]);
  const double relative_residual = scale > 0 ? residual / scale : residual;
  double photon_residual[2][HHE_NGROUPS];
  for (int o = 0; o < 2; o++)
    for (int g = 0; g < HHE_NGROUPS; g++)
      photon_residual[o][g] = state.transport.ledger.emitted[o][g] - live[o][g] - state.transport.ledger.escaped[o][g] -
                              state.transport.ledger.photoabsorbed[o][g] - state.transport.ledger.converted[o][g];
  MPI_Allreduce(MPI_IN_PLACE, photon_residual, 2 * HHE_NGROUPS, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  unsigned long long packet_count = state.transport.size, global_packets;
  unsigned long long scatters = state.transport.ledger.scatter_events, global_scatters;
  MPI_Allreduce(&packet_count, &global_packets, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(&scatters, &global_scatters, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, run_globals.mpi_comm);
  mlog("HHe snapshot %d: xHI=%.8g, T=%.8g K, photons packets=%llu, scatters=%llu, "
       "relative energy residual=%.8g, external matter energy=%.8g eV, interval=%.8g Myr",
       MLOG_MESG,
       snapshot,
       run_globals.reion_grids.volume_weighted_global_xH,
       run_globals.reion_grids.volume_weighted_global_temp_kinetic_all_gas,
       global_packets,
       global_scatters,
       relative_residual,
       global_energy[1],
       elapsed_s / SEC_PER_MEGAYEAR);

  /* Independent files avoid any global field gather and retain full local
   * phase state. Schema 1 is an audit product, not an implemented restart API. */
  char path[STRLEN + 128], temporary[STRLEN + 132];
  snprintf(
    path, sizeof(path), "%shhe_state_%03d.rank%05d.hdf5", run_globals.params.OutputDir, snapshot, run_globals.mpi_rank);
  snprintf(temporary, sizeof(temporary), "%s.tmp", path);
  hid_t file = H5Fcreate(temporary, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0)
    fail("cannot create HHe per-rank audit file in OutputDir");
  scalar(file, "SchemaVersion", 1);
  scalar(file, "Snapshot", snapshot);
  scalar(file, "StartSnapshot", run_globals.params.HHeStartSnapshot);
  scalar(file, "TimeSinceStart_s", state.time_s);
  scalar(file, "Interval_s", elapsed_s);
  scalar(file, "Redshift", run_globals.ZZ[snapshot]);
  scalar(file, "SlabXStart", state.mesh.slab_x_start);
  scalar(file, "SlabXCount", state.mesh.slab_x_count);
  scalar(file, "BoundaryPeriodic", state.mesh.periodic);
  scalar(file, "SourceRightEndpointHold", 1);
  scalar(file, "RadiationRedshiftIncluded", 0);
  scalar(file, "HydrodynamicAdvectionIncluded", 0);
  scalar(file, "DiskRestartSupported", 0);
  scalar(file, "InitialTemperature_K", run_globals.params.HHeInitialTemperature);
  scalar(file, "GlobalEnergyResidual_eV", residual);
  scalar(file, "GlobalRelativeEnergyResidual", relative_residual);
  scalar(file, "LocalMatterExchange_eV", state.matter_exchange_eV);
  scalar(file, "LocalInitialEnergy_eV", state.initial_energy_eV);
  scalar(file, "LocalInactiveRadiation_eV", energy[6]);
  scalar(file, "LocalScatterEvents", (double)scatters);
  scalar(file, "MeanElectronPerNucleus", state.mean_electron_per_baryon);
  hsize_t three = 3, five = 5, seven = 7, groups = HHE_NGROUPS;
  dataset(file, "BoxPhysical_cm", 1, &three, H5T_NATIVE_DOUBLE, state.mesh.box_cm);
  dataset(file, "GridDimensions", 1, &three, H5T_NATIVE_INT, state.mesh.dimensions);
  dataset(file, "GroupEnergy_eV", 1, &groups, H5T_NATIVE_DOUBLE, hhe_group_energy_eV);
  dataset(file, "GroupKind", 1, &groups, H5T_NATIVE_INT, hhe_group_kind);
  dataset(file, "SourcePhotonFraction", 1, &groups, H5T_NATIVE_DOUBLE, state.spectrum);
  dataset(file, "LocalSpeciesExchange", 1, &five, H5T_NATIVE_DOUBLE, state.species_exchange);
  dataset(file, "GlobalEnergyComponents_eV", 1, &seven, H5T_NATIVE_DOUBLE, global_energy);
  h5check(H5LTset_attribute_string(
    file,
    "GlobalEnergyComponents_eV",
    "Order",
    "initial,external_matter,current_gas,primary_emitted,live_photons,escaped_photons,inactive_radiation"));
  hsize_t specdims[2] = { 2, HHE_NGROUPS };
  dataset(file, "LivePhotons", 2, specdims, H5T_NATIVE_DOUBLE, live);
  dataset(file, "EmittedPhotons", 2, specdims, H5T_NATIVE_DOUBLE, state.transport.ledger.emitted);
  dataset(file, "EscapedPhotons", 2, specdims, H5T_NATIVE_DOUBLE, state.transport.ledger.escaped);
  dataset(file, "AbsorbedPhotons", 2, specdims, H5T_NATIVE_DOUBLE, state.transport.ledger.photoabsorbed);
  dataset(file, "ConvertedPhotons", 2, specdims, H5T_NATIVE_DOUBLE, state.transport.ledger.converted);
  dataset(file, "GlobalPhotonBudgetResidual", 2, specdims, H5T_NATIVE_DOUBLE, photon_residual);
  dataset(file, "RecombinationEvents", 1, &three, H5T_NATIVE_DOUBLE, state.transport.ledger.chemistry.recombinations);
  dataset(file,
          "CollisionalIonizationEvents",
          1,
          &three,
          H5T_NATIVE_DOUBLE,
          state.transport.ledger.chemistry.collisional_ionizations);
  scalar(file, "HeILineConversions", state.transport.ledger.chemistry.line_conversions);
  scalar(file, "Photoheat_eV", state.transport.ledger.chemistry.photoheat);
  scalar(file, "HighEnergyPhotoheat_eV", state.transport.ledger.chemistry.high_energy_photoheat);
  scalar(file, "CaptureCooling_eV", state.transport.ledger.chemistry.capture_cooling);
  scalar(file, "CollisionalIonizationCooling_eV", state.transport.ledger.chemistry.collisional_ionization_cooling);
  scalar(file, "ExtraRadiativeCooling_eV", state.transport.ledger.chemistry.extra_radiative_cooling);
  const size_t cells = hhe_mesh_local_cells(&state.mesh);
  double* field = allocate(cells, sizeof(*field));
  hsize_t shape[3] = { (hsize_t)state.mesh.slab_x_count,
                       (hsize_t)state.mesh.dimensions[1],
                       (hsize_t)state.mesh.dimensions[2] };
  const char* names[5] = { "nHI_cm-3", "nHII_cm-3", "nHeI_cm-3", "nHeII_cm-3", "nHeIII_cm-3" };
  for (int s = 0; s < 5; s++) {
    for (size_t j = 0; j < cells; j++)
      field[j] = state.gas[j].species[s];
    dataset(file, names[s], 3, shape, H5T_NATIVE_DOUBLE, field);
  }
  for (size_t j = 0; j < cells; j++)
    field[j] = state.gas[j].thermal_eV_cm3;
  dataset(file, "ThermalEnergy_eV_cm-3", 3, shape, H5T_NATIVE_DOUBLE, field);
  for (size_t j = 0; j < cells; j++) {
    double nuclei = 0;
    for (int s = 0; s < 5; s++)
      nuclei += state.gas[j].species[s];
    field[j] = nuclei > 0 ? hhe_temperature(state.gas + j) : 0;
  }
  dataset(file, "Temperature_K", 3, shape, H5T_NATIVE_DOUBLE, field);
  const char* rec_names[3] = { "HIIRecombinationEvents", "HeIIRecombinationEvents", "HeIIIRecombinationEvents" };
  for (int tr = 0; tr < 3; tr++) {
    for (size_t j = 0; j < cells; j++)
      field[j] = state.transport.cell_recombinations ? state.transport.cell_recombinations[j * 3 + tr] : 0;
    dataset(file, rec_names[tr], 3, shape, H5T_NATIVE_DOUBLE, field);
  }
  free(field);
  write_packets(file);
  h5check(H5Fclose(file));
  if (rename(temporary, path))
    fail("cannot finalize HHe state output");
}

void photon_hhe_meraxes(int snapshot)
{
  if (snapshot < run_globals.params.HHeStartSnapshot)
    return;
  if (!state.initialized) {
    if (snapshot != run_globals.params.HHeStartSnapshot)
      fail("first HHe call must be HHeStartSnapshot; disk restart is not supported");
    initialize(snapshot);
    publish_grids(snapshot);
    write_output(snapshot, 0);
    return;
  }
  if (snapshot != state.last_snapshot + 1)
    fail("HHe state requires consecutive snapshots; no implicit reset or restart");
  const double interval = (run_globals.LTTime[state.last_snapshot] - run_globals.LTTime[snapshot]) *
                          run_globals.units.UnitTime_in_s / run_globals.params.Hubble_h;
  if (!(interval > 0) || !isfinite(interval))
    fail("invalid physical snapshot interval");
  remap(snapshot);
  prepare_sources();
  double maxstep = hhe_transport_cfl_step(&state.mesh, run_globals.params.HHeCellCrossingFraction);
  maxstep = fmin(maxstep, run_globals.params.HHeMaxStepMyr * SEC_PER_MEGAYEAR);
  if (!(maxstep > 0) || !isfinite(maxstep))
    fail("invalid transport timestep");
  const double raw_steps = ceil(interval / maxstep);
  if (raw_steps >= (double)ULLONG_MAX)
    fail("transport step count overflow");
  const unsigned long long steps = (unsigned long long)raw_steps;
  const double dt = interval / (double)steps;
  const unsigned int angles = (unsigned int)run_globals.params.HHeEmissionDirections;
  const hhe_options options = { .donor_fraction = 0.1, .max_substeps = 0, .include_collisional = 1 };
  mlog("HHe advancing snapshot %d with %llu steps, dt=%.8g yr, %.8g physical cm cells",
       MLOG_MESG,
       snapshot,
       steps,
       dt / SEC_PER_YEAR,
       state.mesh.box_cm[0] / state.mesh.dimensions[0]);
  for (unsigned long long step = 0; step < steps; step++) {
    const double time = state.time_s + (double)step * dt;
    require_collective(hhe_transport_emit_sources(&state.transport, &state.mesh, state.rates, dt, time, angles),
                       "primary photon emission");
    require_collective(photon_hhe_mpi_drift(&state.transport,
                                            &state.mesh,
                                            state.gas,
                                            dt * 0.5,
                                            run_globals.reion_grids.slab_ix_start,
                                            run_globals.reion_grids.slab_nix,
                                            run_globals.mpi_comm),
                       "first drift");
    require_collective(hhe_transport_chemistry(&state.transport, &state.mesh, state.gas, dt, &options),
                       "shared H/He chemistry");
    require_collective(photon_hhe_mpi_drift(&state.transport,
                                            &state.mesh,
                                            state.gas,
                                            dt * 0.5,
                                            run_globals.reion_grids.slab_ix_start,
                                            run_globals.reion_grids.slab_nix,
                                            run_globals.mpi_comm),
                       "second drift");
    require_collective(hhe_transport_emit_pending(&state.transport, &state.mesh, time + dt, angles),
                       "diffuse photon emission");
    if ((step + 1) % (steps / 20 + 1) == 0)
      mlog("HHe snapshot %d: completed transport step %llu/%llu", MLOG_MESG, snapshot, step + 1, steps);
  }
  state.time_s += interval;
  state.last_snapshot = snapshot;
  publish_grids(snapshot);
  write_output(snapshot, interval);
}

double photon_hhe_mean_electron_per_baryon(void)
{
  return state.mean_electron_per_baryon;
}

void photon_hhe_meraxes_free(void)
{
  hhe_transport_free(&state.transport);
  free(state.gas);
  free(state.rates);
  memset(&state, 0, sizeof(state));
}
