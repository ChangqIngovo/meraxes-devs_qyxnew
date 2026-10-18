/* Independent event budgets and analytic limits for the native H/He kernel.
 * No Meraxes globals, simulation catalogue or external atomic tables needed.
 */
#include "../core/photon_hhe_meraxes.h"
#include "photon_hhe_chemistry.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static double sum(const double* a, int n)
{
  double result = 0.;
  for (int i = 0; i < n; ++i)
    result += a[i];
  return result;
}

static double photon_energy(const double* a)
{
  double result = 0.;
  for (int g = 0; g < HHE_NGROUPS; ++g)
    result += a[g] * hhe_group_energy_eV[g];
  return result;
}

static void set_temperature(hhe_state* s, double temperature)
{
  s->thermal_eV_cm3 = 1.5 * HHE_KB_EV_K * temperature * (sum(s->species, HHE_NSPECIES) + hhe_electron_density(s));
}

static void close_to(double actual, double expected, double absolute)
{
  if (!isfinite(actual) || fabs(actual - expected) > absolute) {
    fprintf(stderr, "actual %.17g, expected %.17g, tolerance %.3g\n", actual, expected, absolute);
    assert(0);
  }
}

static int band(double energy)
{
  return energy < 24.59 ? 0 : (energy < 54.4 ? 1 : 2);
}

static void projection(void)
{
  const double energies[] = { 13.6, 13.7, 18., 21.2, 24.59, 35., 40.8, 54.4, 70. };
  for (unsigned j = 0; j < sizeof(energies) / sizeof(*energies); ++j) {
    double photons[HHE_NGROUPS] = { 0 };
    assert(hhe_project_photons(energies[j], 0.7, 0, photons) == HHE_OK);
    close_to(sum(photons, HHE_NGROUPS), .7, 4e-16);
    close_to(photon_energy(photons), .7 * energies[j], 3e-14);
    for (int g = 0; g < HHE_NGROUPS; ++g) {
      assert(photons[g] >= 0.);
      if (photons[g] > 0.) {
        assert(hhe_group_kind[g] == 0);
        assert(band(hhe_group_energy_eV[g]) == band(energies[j]));
      }
    }
  }
  for (int kind = 1; kind <= 2; ++kind) {
    double photons[HHE_NGROUPS] = { 0 };
    const double energy = kind == 1 ? 21.2 : 40.8;
    assert(hhe_project_photons(energy, 2., kind, photons) == HHE_OK);
    for (int g = 0; g < HHE_NGROUPS; ++g)
      if (photons[g])
        assert(hhe_group_kind[g] == kind);
    close_to(photon_energy(photons), 2. * energy, 1e-14);
    assert(hhe_project_photons(energy + 1., 1., kind, photons) == HHE_INVALID_INPUT);
  }
}

static void independent_budgets(const hhe_state* before,
                                const double* p0,
                                const hhe_state* after,
                                const double* p1,
                                const double* emitted,
                                const hhe_ledger* l)
{
  close_to(after->species[0] + after->species[1], before->species[0] + before->species[1], 2e-18);
  close_to(sum(after->species + 2, 3), sum(before->species + 2, 3), 2e-19);
  double absorbed = 0.;
  for (int a = 0; a < 3; ++a)
    absorbed += sum(l->absorbed[a], HHE_NGROUPS);
  close_to(sum(p0, HHE_NGROUPS) - sum(p1, HHE_NGROUPS), absorbed + l->line_conversions, 1e-18);
  close_to(hhe_electron_density(after) - hhe_electron_density(before),
           absorbed + sum(l->collisional_ionizations, 3) - sum(l->recombinations, 3),
           2e-18);
  /* Compute independently of the kernel's reported energy_error. */
  const double start = before->thermal_eV_cm3 + 13.6 * before->species[1] + 24.59 * before->species[3] +
                       78.99 * before->species[4] + photon_energy(p0);
  const double end = after->thermal_eV_cm3 + 13.6 * after->species[1] + 24.59 * after->species[3] +
                     78.99 * after->species[4] + photon_energy(p1) + photon_energy(emitted) + l->inactive_radiation;
  close_to(end, start, 4e-15 * fmax(start, 1e-8));
  for (int a = 0; a < HHE_NSPECIES; ++a)
    assert(after->species[a] >= 0.);
  for (int g = 0; g < HHE_NGROUPS; ++g)
    assert(p1[g] >= 0. && emitted[g] >= 0.);
  assert(after->thermal_eV_cm3 > 0.);
}

static void coupled_reactions(void)
{
  hhe_state before = { { 2e-5, 8e-5, 1.5e-6, 3.5e-6, 3e-6 }, 0 };
  set_temperature(&before, 10000.);
  hhe_state after = before;
  double p0[HHE_NGROUPS] = { 0 }, p1[HHE_NGROUPS], emitted[HHE_NGROUPS];
  assert(hhe_project_photons(18., 3e-6, 0, p0) == HHE_OK);
  assert(hhe_project_photons(35., 5e-7, 0, p0) == HHE_OK);
  assert(hhe_project_photons(70., 1e-7, 0, p0) == HHE_OK);
  assert(hhe_project_photons(21.2, 1e-7, 1, p0) == HHE_OK);
  memcpy(p1, p0, sizeof(p0));
  hhe_options options = { .05, 0, 1 };
  hhe_ledger ledger;
  assert(hhe_advance(&after, p1, 2e12, &options, emitted, &ledger) == HHE_OK);
  independent_budgets(&before, p0, &after, p1, emitted, &ledger);
  assert(sum(ledger.recombinations, 3) > 0.);
  assert(sum(ledger.collisional_ionizations, 3) > 0.);
  assert(ledger.line_conversions > 0.);
  assert(sum(emitted, HHE_NGROUPS) > 0.);
}

static void analytic_competing_hazards(int line)
{
  hhe_state before = { { 1e-4, 0., 8e-6, 0., 0. }, 0 };
  set_temperature(&before, 10000.);
  hhe_state after = before;
  double p0[HHE_NGROUPS] = { 0 }, p1[HHE_NGROUPS], emitted[HHE_NGROUPS];
  assert(hhe_project_photons(line ? 21.2 : 35., 1e-14, line, p0) == HHE_OK);
  memcpy(p1, p0, sizeof(p0));
  hhe_coefficients coefficients;
  assert(hhe_coefficients_at(10000., &coefficients) == HHE_OK);
  hhe_options options = { .1, 0, 0 };
  hhe_ledger ledger;
  const double dt = 1e11;
  assert(hhe_advance(&after, p1, dt, &options, emitted, &ledger) == HHE_OK);
  assert(ledger.substeps == 1);
  double expected_conversion = 0.;
  for (int g = 0; g < HHE_NGROUPS; ++g) {
    const double hi = HHE_C_LIGHT * before.species[0] * hhe_group_cross_section_cm2[g][0];
    const double hei = HHE_C_LIGHT * before.species[2] * hhe_group_cross_section_cm2[g][1];
    const double conversion =
      hhe_group_kind[g] == 1 ? HHE_C_LIGHT * before.species[2] * coefficients.line_sigma[0] / 915. : 0.;
    const double total = hi + hei + conversion;
    const double lost = total > 0. ? -p0[g] * expm1(-total * dt) : 0.;
    close_to(p1[g], p0[g] * exp(-total * dt), 1e-28);
    close_to(ledger.absorbed[0][g], total > 0. ? lost * hi / total : 0., 1e-28);
    close_to(ledger.absorbed[1][g], total > 0. ? lost * hei / total : 0., 1e-28);
    if (total > 0.)
      expected_conversion += lost * conversion / total;
  }
  close_to(ledger.line_conversions, expected_conversion, 1e-28);
  if (line) {
    /* Independent quadrature moment for the 20.6 eV HeI two-photon decay. */
    close_to(sum(emitted, HHE_NGROUPS), expected_conversion * .5500167007245592, 1e-28);
    close_to(photon_energy(emitted), expected_conversion * 8.862418007407822, 1e-27);
  }
  independent_budgets(&before, p0, &after, p1, emitted, &ledger);
}

static void rejected_input_is_transactional(void)
{
  hhe_state state = { { 5e-5, 5e-5, 8e-6, 0., 0. }, 0 };
  set_temperature(&state, 4999.);
  hhe_state old = state;
  double p[HHE_NGROUPS] = { 0 }, old_p[HHE_NGROUPS], e[HHE_NGROUPS], old_e[HHE_NGROUPS];
  for (int i = 0; i < HHE_NGROUPS; ++i)
    e[i] = i + 1.;
  memcpy(old_p, p, sizeof(p));
  memcpy(old_e, e, sizeof(e));
  hhe_ledger ledger, old_ledger;
  memset(&ledger, 0xa5, sizeof(ledger));
  memcpy(&old_ledger, &ledger, sizeof(ledger));
  hhe_options options = { .1, 0, 1 };
  assert(hhe_advance(&state, p, 1e10, &options, e, &ledger) == HHE_TEMPERATURE_DOMAIN);
  assert(!memcmp(&state, &old, sizeof(state)) && !memcmp(p, old_p, sizeof(p)) && !memcmp(e, old_e, sizeof(e)) &&
         !memcmp(&ledger, &old_ledger, sizeof(ledger)));
}

static void driven_snapshot_remap(void)
{
  hhe_state state = { { 2e-5, 8e-5, 1.5e-6, 3.5e-6, 3e-6 }, 0 };
  set_temperature(&state, 11000.);
  const hhe_state before = state;
  double exchange[5] = { 0 }, energy_exchange = 0.;
  const double old_volume = 3e60, new_volume = 5e60;
  assert(photon_hhe_driven_remap(&state, 3e-4, 4e-6, 10000., old_volume, new_volume, exchange, &energy_exchange) ==
         HHE_OK);
  close_to(state.species[1] / (state.species[0] + state.species[1]), .8, 3e-16);
  for (int s = 2; s < 5; ++s)
    close_to(state.species[s] / 4e-6, before.species[s] / 8e-6, 2e-16);
  close_to(hhe_temperature(&state), 11000., 4e-12);
  for (int s = 0; s < 5; ++s)
    close_to(state.species[s] * new_volume - before.species[s] * old_volume, exchange[s], 1e40);
  const double total_before =
    (before.thermal_eV_cm3 + 13.6 * before.species[1] + 24.59 * before.species[3] + 78.99 * before.species[4]) *
    old_volume;
  const double total_after =
    (state.thermal_eV_cm3 + 13.6 * state.species[1] + 24.59 * state.species[3] + 78.99 * state.species[4]) * new_volume;
  close_to(total_after - total_before, energy_exchange, 5e43);
  /* Emptying the cell removes exactly its recorded material state. */
  assert(photon_hhe_driven_remap(&state, 0., 0., 10000., new_volume, new_volume, exchange, &energy_exchange) == HHE_OK);
  close_to(sum(state.species, 5), 0., 0.);
  close_to(state.thermal_eV_cm3, 0., 0.);
  close_to(energy_exchange, -total_before, 5e43);
  /* Fresh matter entering vacuum has explicitly neutral composition. */
  assert(photon_hhe_driven_remap(&state, 1e-4, 8e-6, 9000., new_volume, new_volume, exchange, &energy_exchange) ==
         HHE_OK);
  assert(state.species[0] == 1e-4 && state.species[2] == 8e-6);
  assert(state.species[1] == 0. && state.species[3] == 0. && state.species[4] == 0.);
  close_to(hhe_temperature(&state), 9000., 3e-12);
}

int main(void)
{
  projection();
  coupled_reactions();
  analytic_competing_hazards(0);
  analytic_competing_hazards(1);
  rejected_input_is_transactional();
  driven_snapshot_remap();
  puts("H/He kernel: projection, competing hazards, charge/nuclei/energy and transactional failures passed");
  return 0;
}
