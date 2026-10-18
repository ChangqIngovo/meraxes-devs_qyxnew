#ifndef PHOTON_HHE_TRANSPORT_H
#define PHOTON_HHE_TRANSPORT_H

#include "photon_hhe_chemistry.h"
#include <stddef.h>
#include <stdint.h>

/* MPI-independent slab-local transport. All lengths are physical cm, times
 * seconds and packet counts actual photon numbers. The caller exchanges ROUTE
 * packets until no rank has unfinished drift; chemistry is then local.
 * One packet is monochromatic. Its tagged line group is never inferred from
 * its energy: continuum at a line energy remains continuum.
 */
enum hhe_packet_location
{
  HHE_PACKET_LOCAL = 0,
  HHE_PACKET_ROUTE = 1,
  HHE_PACKET_ESCAPED = 2
};
enum hhe_transport_status
{
  HHE_TRANSPORT_OK = 0,
  HHE_TRANSPORT_INPUT = 101,
  HHE_TRANSPORT_MEMORY = 102,
  HHE_TRANSPORT_GEOMETRY = 103,
  HHE_TRANSPORT_PHASE = 104,
  HHE_TRANSPORT_CHEMISTRY = 105
};

typedef struct
{
  double position_cm[3], direction[3];
  double count, energy_eV, birth_s, clock_s, remaining_s;
  double scatter_tau; /* residual exponential optical depth, retained on MPI handoff */
  uint64_t rng_state, emitter_id;
  int cell[3];                 /* global cell, including directional ownership of exact faces */
  int group, origin, location; /* origin 0 primary / 1 diffuse */
} hhe_packet;

typedef struct
{
  int dimensions[3], slab_x_start, slab_x_count;
  double box_cm[3];
  int periodic; /* 0 open, 1 periodic in all three axes */
} hhe_mesh;

typedef struct
{
  double emitted[2][HHE_NGROUPS], escaped[2][HHE_NGROUPS];
  double photoabsorbed[2][HHE_NGROUPS], converted[2][HHE_NGROUPS];
  hhe_ledger chemistry; /* volume-integrated events and energies */
  uint64_t scatter_events, cell_crossings, created_packets, peak_packets;
} hhe_transport_ledger;

typedef struct
{
  hhe_packet* packets;
  size_t size, capacity;
  double* pending_emission;    /* endpoint births, local cell-major actual counts */
  double* cell_recombinations; /* cumulative actual events [local cell][3] */
  size_t pending_cells;
  int has_pending;
  uint64_t seed;
  size_t memory_limit_bytes; /* 0: allocation-limited, never discard photons */
  hhe_transport_ledger ledger;
  size_t error_cell;
  int chemistry_status;
  char error[192];
} hhe_transport;

void hhe_transport_init(hhe_transport* transport, uint64_t seed, size_t memory_limit_bytes);
void hhe_transport_free(hhe_transport* transport);
size_t hhe_mesh_local_cells(const hhe_mesh* mesh);
double hhe_mesh_cell_volume(const hhe_mesh* mesh);
/* Maximum drift/react/drift step for a chosen cell light-crossing CFL <= 1. */
double hhe_transport_cfl_step(const hhe_mesh* mesh, double cfl);
int hhe_transport_append(hhe_transport* transport, const hhe_packet* packets, size_t count);
/* Drop only already exported ROUTE, already ledgered ESCAPED, or exact-zero
 * packets. Caller must copy/export ROUTE packets before calling this. */
void hhe_transport_compact(hhe_transport* transport);
/* rates[cell][group] are actual photons/s in each local cell. Sources enter at
 * the start of dt, with deterministic equal-weight angular quadrature. */
int hhe_transport_emit_sources(hhe_transport* transport,
                               const hhe_mesh* mesh,
                               const double* rates,
                               double dt_s,
                               double birth_s,
                               unsigned int directions);
int hhe_transport_begin_drift(hhe_transport* transport, double dt_s);
/* Walk until dt completes, a nonlocal x slab is reached, or a true open-box
 * face is crossed. Every elastic event and every cell crossing is explicit;
 * there is no scattering event cap, roulette or small-count deletion. */
int hhe_transport_drift_local(hhe_transport* transport, const hhe_mesh* mesh, const hhe_state* gas);
/* Shared-cell absorption, conversion and evolving H/He thermal chemistry.
 * Requires all packets local and drift finished. Diffuse output is held until
 * emit_pending after the second drift. A failure requires caller abort; cells
 * completed before a failure are not rolled back. */
int hhe_transport_chemistry(hhe_transport* transport,
                            const hhe_mesh* mesh,
                            hhe_state* gas,
                            double dt_s,
                            const hhe_options* options);
int hhe_transport_emit_pending(hhe_transport* transport, const hhe_mesh* mesh, double birth_s, unsigned int directions);
/* A prescribed geometry remap, not cosmological radiation evolution. Cell
 * identities, photon energies and counts stay fixed; use only between phases. */
int hhe_transport_rescale_positions(hhe_transport* transport, const double old_box_cm[3], const double new_box_cm[3]);
void hhe_transport_live_counts(const hhe_transport* transport, double counts[2][HHE_NGROUPS]);

#endif
