#ifndef PHOTON_HHE_MPI_H
#define PHOTON_HHE_MPI_H

#include "../physics/photon_hhe_transport.h"
#include <mpi.h>
#include <stddef.h>

/* Every rank calls this with its own FFTW slab and persistent local gas.
 * slab_start/count have communicator-size entries and cover the x dimension.
 * No gas grid or complete packet population is gathered on rank zero. */
int photon_hhe_mpi_drift(hhe_transport* transport,
                         const hhe_mesh* mesh,
                         const hhe_state* gas,
                         double dt_s,
                         const ptrdiff_t* slab_start,
                         const ptrdiff_t* slab_count,
                         MPI_Comm comm);

#endif
