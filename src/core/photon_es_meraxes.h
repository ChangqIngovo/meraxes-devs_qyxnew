#ifndef PHOTON_ES_MERAXES_H
#define PHOTON_ES_MERAXES_H

/* Online ES/HHe snapshot bridge. The external worker owns persistent species,
 * photon accounts, and all recombinations. No packet or legacy N_rec solve is
 * invoked by this mode. */
void photon_es_meraxes(int snapshot);
void photon_es_meraxes_free(void);
double photon_es_mean_electron_per_nucleus(void);

#endif
