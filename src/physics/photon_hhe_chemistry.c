#include "photon_hhe_chemistry.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "photon_hhe_atomic_table.h"

const double hhe_ionization_energy_eV[3] = { 13.6, 24.59, 54.4 };
static const int absorber[3] = { 0, 2, 3 };
static const int ion[3] = { 1, 3, 4 };

double hhe_electron_density(const hhe_state* state)
{
  return state->species[1] + state->species[3] + 2.0 * state->species[4];
}

static double particles(const hhe_state* state)
{
  double count = hhe_electron_density(state);
  int i;
  for (i = 0; i < 5; ++i)
    count += state->species[i];
  return count;
}

double hhe_temperature(const hhe_state* state)
{
  double count = particles(state);
  return count > 0.0 ? state->thermal_eV_cm3 / (1.5 * HHE_KB_EV_K * count) : 0.0;
}

double hhe_binding_energy(const hhe_state* state)
{
  return 13.6 * state->species[1] + 24.59 * state->species[3] + (24.59 + 54.4) * state->species[4];
}

int hhe_coefficients_at(double temperature_K, hhe_coefficients* out)
{
  int row, a, g;
  double position, w;
  const hhe_coefficients *lo, *hi;
  if (out == NULL || !isfinite(temperature_K))
    return HHE_INVALID_INPUT;
  if (temperature_K < HHE_T_MIN || temperature_K > HHE_T_MAX)
    return HHE_TEMPERATURE_DOMAIN;
  position = (temperature_K - HHE_T_MIN) / HHE_TABLE_DT;
  row = (int)floor(position);
  if (row >= HHE_TABLE_ROWS - 1)
    row = HHE_TABLE_ROWS - 2;
  w = position - row;
  lo = &hhe_atomic_table[row];
  hi = &hhe_atomic_table[row + 1];
#define INTERPOLATE(field) out->field = (1.0 - w) * lo->field + w * hi->field
  for (a = 0; a < 3; ++a) {
    INTERPOLATE(alpha_down[a]);
    INTERPOLATE(capture_cooling[a]);
    INTERPOLATE(inactive_radiation[a]);
    INTERPOLATE(collisional_ionization[a]);
    for (g = 0; g < HHE_NGROUPS; ++g)
      INTERPOLATE(emission[a][g]);
  }
  INTERPOLATE(HI_excitation);
  INTERPOLATE(freefree);
  for (a = 0; a < 2; ++a)
    INTERPOLATE(line_sigma[a]);
#undef INTERPOLATE
  /* Evaluate the stiff exponential thermal tails directly: interpolating an
   * exponentially tiny HeII CI coefficient has a large relative error near
   * 5000 K. These are the identical HG97/Black/Cen/Theuns fits used to generate
   * the table and frozen reference, including the declared low-T CI extension. */
  {
    static const double threshold_K[3] = { 157807.0, 285335.0, 631515.0 };
    static const double amplitude[3] = { 21.11, 32.38, 19.95 };
    static const double power[3] = { 1.089, 1.146, 1.089 };
    static const double scale[3] = { 0.354, 0.416, 0.553 };
    static const double exponent[3] = { 0.874, 0.987, 0.735 };
    static const double denominator_power[3] = { 1.101, 1.056, 1.275 };
    double gaunt;
    for (a = 0; a < 3; ++a) {
      double lambda = 2.0 * threshold_K[a] / temperature_K;
      out->collisional_ionization[a] = amplitude[a] * pow(temperature_K, -1.5) * exp(-lambda / 2.0) *
                                       pow(lambda, -power[a]) /
                                       pow(1.0 + pow(lambda / scale[a], exponent[a]), denominator_power[a]);
    }
    out->HI_excitation =
      7.5e-19 * exp(-0.75 * 157807.0 / temperature_K) / (1.0 + sqrt(temperature_K / 1e5)) / 1.602176634e-12;
    gaunt = 1.1 + 0.34 * exp(-pow(5.5 - log10(temperature_K), 2.0) / 3.0);
    out->freefree = 1.42e-27 * gaunt * sqrt(temperature_K) / 1.602176634e-12;
  }
  return HHE_OK;
}

int hhe_line_cross_sections(double temperature_K, double sigma_cm2[2])
{
  double position, w;
  int row, a;
  if (sigma_cm2 == NULL || !isfinite(temperature_K))
    return HHE_INVALID_INPUT;
  if (temperature_K < HHE_T_MIN || temperature_K > HHE_T_MAX)
    return HHE_TEMPERATURE_DOMAIN;
  position = (temperature_K - HHE_T_MIN) / HHE_TABLE_DT;
  row = (int)floor(position);
  if (row >= HHE_TABLE_ROWS - 1)
    row = HHE_TABLE_ROWS - 2;
  w = position - row;
  for (a = 0; a < 2; ++a)
    sigma_cm2[a] = (1.0 - w) * hhe_atomic_table[row].line_sigma[a] + w * hhe_atomic_table[row + 1].line_sigma[a];
  return HHE_OK;
}

int hhe_project_photons(double energy_eV, double count, int kind, double spectrum[HHE_NGROUPS])
{
  int lower, upper;
  double w;
  if (spectrum == NULL || !isfinite(energy_eV) || !isfinite(count) || count < 0.0 || kind < 0 || kind > 2)
    return HHE_INVALID_INPUT;
  if (kind != 0) {
    int group = HHE_NCONTINUUM + kind - 1;
    if (energy_eV != hhe_group_energy_eV[group])
      return HHE_INVALID_INPUT;
    spectrum[group] += count;
    return HHE_OK;
  }
  if (energy_eV < hhe_group_energy_eV[0] || energy_eV > hhe_group_energy_eV[HHE_NCONTINUUM - 1])
    return HHE_INVALID_INPUT;
  upper = 1;
  while (upper < HHE_NCONTINUUM - 1 && hhe_group_energy_eV[upper] <= energy_eV)
    ++upper;
  lower = upper - 1;
  w = (energy_eV - hhe_group_energy_eV[lower]) / (hhe_group_energy_eV[upper] - hhe_group_energy_eV[lower]);
  spectrum[lower] += count * (1.0 - w);
  spectrum[upper] += count * w;
  return HHE_OK;
}

static double radiation_energy(const double photons[HHE_NGROUPS])
{
  double energy = 0.0;
  int g;
  for (g = 0; g < HHE_NGROUPS; ++g)
    energy += photons[g] * hhe_group_energy_eV[g];
  return energy;
}

int hhe_advance(hhe_state* state,
                double photons[HHE_NGROUPS],
                double dt_s,
                const hhe_options* options,
                double emitted[HHE_NGROUPS],
                hhe_ledger* ledger)
{
  hhe_state gas, initial;
  hhe_ledger sums;
  hhe_coefficients coeff;
  hhe_options defaults = { 0.1, 0UL, 1 };
  double live[HHE_NGROUPS], births[HHE_NGROUPS] = { 0.0 };
  double elapsed = 0.0, initial_energy, count;
  int a, g, i, status;
  if (state == NULL || photons == NULL || emitted == NULL || ledger == NULL || !isfinite(dt_s) || dt_s < 0.0)
    return HHE_INVALID_INPUT;
  if (options == NULL)
    options = &defaults;
  if (!isfinite(options->donor_fraction) || options->donor_fraction <= 0.0 || options->donor_fraction > 0.5)
    return HHE_INVALID_INPUT;
  for (i = 0; i < 5; ++i)
    if (!isfinite(state->species[i]) || state->species[i] < 0.0)
      return HHE_INVALID_INPUT;
  for (g = 0; g < HHE_NGROUPS; ++g)
    if (!isfinite(photons[g]) || photons[g] < 0.0)
      return HHE_INVALID_INPUT;
  if (!isfinite(state->thermal_eV_cm3) || state->thermal_eV_cm3 < 0.0)
    return HHE_INVALID_INPUT;
  memset(&sums, 0, sizeof(sums));
  count = particles(state);
  if (!isfinite(count))
    return HHE_INVALID_INPUT;
  /* Physical vacuum does not need a fitted temperature or a density floor. */
  if (count == 0.0) {
    if (state->thermal_eV_cm3 != 0.0)
      return HHE_INVALID_INPUT;
    memcpy(emitted, births, sizeof(births));
    *ledger = sums;
    return HHE_OK;
  }
  status = hhe_coefficients_at(hhe_temperature(state), &coeff);
  if (status != HHE_OK)
    return status;
  initial = gas = *state;
  memcpy(live, photons, sizeof(live));
  initial_energy = gas.thermal_eV_cm3 + hhe_binding_energy(&gas) + radiation_energy(live);
  if (!isfinite(initial_energy))
    return HHE_INVALID_INPUT;

  while (elapsed < dt_s) {
    double factors[3], recombination_rate[3], ci_rate[3];
    double hazard[3][HHE_NGROUPS], total[HHE_NGROUPS], conversion_hazard[HHE_NGROUPS];
    double photo[3][HHE_NGROUPS], lost[HHE_NGROUPS], dn[5];
    double rec_cooling = 0.0, ci_cooling = 0.0, extra_cooling = 0.0;
    double inactive_rate = 0.0, ne = hhe_electron_density(&gas);
    double h = dt_s - elapsed, heat = 0.0, high_heat = 0.0, du = 0.0, conversion = 0.0;
    int acceptable;
    if ((options->max_substeps && sums.substeps >= options->max_substeps) || sums.substeps == ULONG_MAX)
      return HHE_SUBSTEP_LIMIT;
    for (a = 0; a < 3; ++a) {
      factors[a] = ne * gas.species[ion[a]];
      recombination_rate[a] = factors[a] * coeff.alpha_down[a];
      ci_rate[a] = options->include_collisional ? ne * gas.species[absorber[a]] * coeff.collisional_ionization[a] : 0.0;
      rec_cooling += factors[a] * coeff.capture_cooling[a];
      inactive_rate += factors[a] * coeff.inactive_radiation[a];
      ci_cooling += ci_rate[a] * hhe_ionization_energy_eV[a];
    }
    if (options->include_collisional)
      extra_cooling = ne * gas.species[0] * coeff.HI_excitation +
                      ne * (gas.species[1] + gas.species[3] + 4.0 * gas.species[4]) * coeff.freefree;
    if (!isfinite(rec_cooling + inactive_rate + ci_cooling + extra_cooling))
      return HHE_INVALID_INPUT;
    for (g = 0; g < HHE_NGROUPS; ++g) {
      conversion_hazard[g] = hhe_group_kind[g] == 1 ? HHE_C_LIGHT * gas.species[2] * coeff.line_sigma[0] / 915.0 : 0.0;
      total[g] = conversion_hazard[g];
      for (a = 0; a < 3; ++a) {
        hazard[a][g] = HHE_C_LIGHT * gas.species[absorber[a]] * hhe_group_cross_section_cm2[g][a];
        total[g] += hazard[a][g];
      }
      if (!isfinite(total[g]))
        return HHE_INVALID_INPUT;
    }

    /* Exponential frozen-hazard losses avoid a photon-depletion CFL bound.
     * The positive material/thermal donor bound still resolves stiff changes. */
    for (;;) {
      double net[3] = { 0.0, 0.0, 0.0 };
      heat = high_heat = conversion = 0.0;
      for (g = 0; g < HHE_NGROUPS; ++g) {
        lost[g] = live[g] * (-expm1(-total[g] * h));
        for (a = 0; a < 3; ++a) {
          double excess = hhe_group_energy_eV[g] - hhe_ionization_energy_eV[a];
          photo[a][g] = total[g] > 0.0 ? lost[g] * (hazard[a][g] / total[g]) : 0.0;
          net[a] += photo[a][g];
          if (excess > 0.0) {
            heat += photo[a][g] * excess;
            if (excess >= 10.2)
              high_heat += photo[a][g] * excess;
          }
        }
        if (total[g] > 0.0)
          conversion += lost[g] * (conversion_hazard[g] / total[g]);
      }
      for (a = 0; a < 3; ++a)
        net[a] += h * (ci_rate[a] - recombination_rate[a]);
      dn[0] = -net[0];
      dn[1] = net[0];
      dn[2] = -net[1];
      dn[3] = net[1] - net[2];
      dn[4] = net[2];
      du = heat - h * (rec_cooling + ci_cooling + extra_cooling);
      acceptable = -du <= options->donor_fraction * gas.thermal_eV_cm3;
      for (i = 0; i < 5; ++i)
        if (-dn[i] > options->donor_fraction * gas.species[i] + 1e-300)
          acceptable = 0;
      if (acceptable)
        break;
      h *= 0.5;
      if (h <= 0.0 || elapsed + h == elapsed)
        return HHE_STEP_UNDERFLOW;
    }
    if (!isfinite(h) || h <= 0.0 || elapsed + h == elapsed)
      return HHE_STEP_UNDERFLOW;
    for (i = 0; i < 5; ++i) {
      gas.species[i] += dn[i];
      if (!isfinite(gas.species[i]) || gas.species[i] < 0.0)
        return HHE_POSITIVITY_ERROR;
    }
    gas.thermal_eV_cm3 += du;
    if (!isfinite(gas.thermal_eV_cm3) || gas.thermal_eV_cm3 <= 0.0)
      return HHE_POSITIVITY_ERROR;
    for (g = 0; g < HHE_NGROUPS; ++g) {
      live[g] -= lost[g];
      if (live[g] < 0.0)
        return HHE_POSITIVITY_ERROR;
      births[g] += conversion * hhe_conversion_emission[g];
      for (a = 0; a < 3; ++a) {
        births[g] += h * factors[a] * coeff.emission[a][g];
        sums.absorbed[a][g] += photo[a][g];
      }
    }
    for (a = 0; a < 3; ++a) {
      sums.recombinations[a] += h * recombination_rate[a];
      sums.collisional_ionizations[a] += h * ci_rate[a];
    }
    sums.line_conversions += conversion;
    sums.photoheat += heat;
    sums.high_energy_photoheat += high_heat;
    sums.capture_cooling += h * rec_cooling;
    sums.collisional_ionization_cooling += h * ci_cooling;
    sums.extra_radiative_cooling += h * extra_cooling;
    sums.inactive_radiation += h * (inactive_rate + extra_cooling) + conversion * hhe_conversion_inactive;
    elapsed += h;
    ++sums.substeps;
  }

  if (hhe_temperature(&gas) < HHE_T_MIN || hhe_temperature(&gas) > HHE_T_MAX)
    return HHE_TEMPERATURE_DOMAIN;
  sums.energy_error = gas.thermal_eV_cm3 + hhe_binding_energy(&gas) + radiation_energy(live) +
                      radiation_energy(births) + sums.inactive_radiation - initial_energy;
  sums.electron_error = hhe_electron_density(&gas) - hhe_electron_density(&initial);
  sums.photon_error = -sums.line_conversions;
  for (g = 0; g < HHE_NGROUPS; ++g) {
    sums.photon_error += photons[g] - live[g];
    for (a = 0; a < 3; ++a) {
      sums.electron_error -= sums.absorbed[a][g];
      sums.photon_error -= sums.absorbed[a][g];
    }
  }
  for (a = 0; a < 3; ++a)
    sums.electron_error += sums.recombinations[a] - sums.collisional_ionizations[a];
  *state = gas;
  memcpy(photons, live, sizeof(live));
  memcpy(emitted, births, sizeof(births));
  *ledger = sums;
  return HHE_OK;
}

const char* hhe_status_message(int status)
{
  switch (status) {
    case HHE_OK:
      return "success";
    case HHE_INVALID_INPUT:
      return "invalid H/He input";
    case HHE_TEMPERATURE_DOMAIN:
      return "H/He model requires 5000 <= T <= 20000 K; no temperature clipping";
    case HHE_SUBSTEP_LIMIT:
      return "H/He chemistry substep limit reached";
    case HHE_STEP_UNDERFLOW:
      return "H/He chemistry time step underflow";
    case HHE_POSITIVITY_ERROR:
      return "H/He chemistry positivity failure";
    default:
      return "unknown H/He status";
  }
}
