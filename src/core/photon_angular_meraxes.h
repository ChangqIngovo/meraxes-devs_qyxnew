#ifndef PHOTON_ANGULAR_MERAXES_H
#define PHOTON_ANGULAR_MERAXES_H

/* Periodic angular-face H/He worker. The worker owns the photon inventory,
 * species, recombinations and consecutive-snapshot checkpoint state. */
void photon_angular_meraxes(int snapshot);
void photon_angular_meraxes_free(void);
double photon_angular_mean_electron_per_nucleus(void);

#endif
