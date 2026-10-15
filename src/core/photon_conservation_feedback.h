#ifndef PHOTON_CONSERVATION_FEEDBACK_H
#define PHOTON_CONSERVATION_FEEDBACK_H

/* Reconstruct the existing ES-scale UV-background closure after the
 * conservative snapshot correction. All ranks supply the same descending
 * physical-radius ladder (Mpc/h), whose final entry is zero for the identity
 * cell filter. radius_slot, old_zre and old_j21 use the local REAL layout.
 * Slots are required only for corrected fully ionized cells. old_j21 may be
 * NULL when ReionUVBFlag is zero. The normalized, unfiltered source Fourier
 * fields from _find_HII_bubbles must still be available.
 *
 * This updates xH-dependent feedback and the partial-cell temperature. The
 * caller subsequently runs the existing full-cell thermal evolution,
 * recombination update and global summaries exactly once.
 */
void photon_conservation_update_feedback(int snapshot,
                                         const double* radii_mpc_h,
                                         int nradii,
                                         const int* radius_slot,
                                         const float* old_zre,
                                         const float* old_j21);

#endif
