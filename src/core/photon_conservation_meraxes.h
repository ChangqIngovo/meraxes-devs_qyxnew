#ifndef PHOTON_CONSERVATION_MERAXES_H
#define PHOTON_CONSERVATION_MERAXES_H

/* The context captures real-space inventories before any FFT. */
typedef struct photon_conservation_context photon_conservation_context;

photon_conservation_context* photon_conservation_begin(int snapshot);
void photon_conservation_radius(photon_conservation_context* context, double radius, int identity);
void photon_conservation_crossing(photon_conservation_context* context, int cell);
void photon_conservation_finish(photon_conservation_context* context);

#endif
