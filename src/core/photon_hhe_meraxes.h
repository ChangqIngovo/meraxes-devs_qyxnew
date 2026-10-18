#ifndef PHOTON_HHE_MERAXES_H
#define PHOTON_HHE_MERAXES_H

#include "../physics/photon_hhe_chemistry.h"
#include <math.h>

/* The mesh is externally prescribed. Retain each element's ion fractions and
 * T during a snapshot remap and explicitly account for matter brought into or
 * out of a fixed comoving cell. Vacuum receiving matter gets neutral gas at
 * the declared initial temperature. This is not a hydrodynamic advection or
 * adiabatic-expansion solve. Species and energy exchanges are actual counts/eV.
 * Exposed as a pure helper for independent tests; outputs accumulate on success. */
static inline int photon_hhe_driven_remap(hhe_state* state,
                                          double nH,
                                          double nHe,
                                          double initial_T,
                                          double old_volume,
                                          double new_volume,
                                          double exchange[5],
                                          double* energy_exchange)
{
  if (!state || !exchange || !energy_exchange || !isfinite(nH) || !isfinite(nHe) || nH < 0 || nHe < 0 ||
      !(old_volume > 0) || !(new_volume > 0) || !isfinite(old_volume) || !isfinite(new_volume) ||
      initial_T < HHE_T_MIN || initial_T > HHE_T_MAX || !isfinite(initial_T))
    return HHE_INVALID_INPUT;
  const hhe_state old = *state;
  for (int s = 0; s < 5; s++)
    if (!isfinite(old.species[s]) || old.species[s] < 0)
      return HHE_INVALID_INPUT;
  if (!isfinite(old.thermal_eV_cm3) || old.thermal_eV_cm3 < 0)
    return HHE_INVALID_INPUT;
  const double oldH = old.species[0] + old.species[1];
  const double oldHe = old.species[2] + old.species[3] + old.species[4];
  if (oldH + oldHe == 0 && old.thermal_eV_cm3 != 0)
    return HHE_INVALID_INPUT;
  double T = oldH + oldHe > 0 ? hhe_temperature(&old) : initial_T;
  if (!isfinite(T) || T < HHE_T_MIN || T > HHE_T_MAX)
    return HHE_TEMPERATURE_DOMAIN;
  hhe_state next = { { 0, 0, 0, 0, 0 }, 0 };
  if (oldH > 0) {
    next.species[0] = nH * old.species[0] / oldH;
    next.species[1] = nH * old.species[1] / oldH;
  } else
    next.species[0] = nH;
  if (oldHe > 0) {
    for (int s = 2; s < 5; s++)
      next.species[s] = nHe * old.species[s] / oldHe;
  } else
    next.species[2] = nHe;
  next.thermal_eV_cm3 = 1.5 * HHE_KB_EV_K * T * (nH + nHe + hhe_electron_density(&next));
  if (!isfinite(next.thermal_eV_cm3))
    return HHE_INVALID_INPUT;
  for (int s = 0; s < 5; s++)
    exchange[s] += next.species[s] * new_volume - old.species[s] * old_volume;
  *energy_exchange += (next.thermal_eV_cm3 + hhe_binding_energy(&next)) * new_volume -
                      (old.thermal_eV_cm3 + hhe_binding_energy(&old)) * old_volume;
  *state = next;
  return HHE_OK;
}

void photon_hhe_meraxes(int snapshot);
void photon_hhe_meraxes_free(void);
/* Number-weighted ne/(nH+nHe), including doubly ionized helium. */
double photon_hhe_mean_electron_per_baryon(void);

#endif
