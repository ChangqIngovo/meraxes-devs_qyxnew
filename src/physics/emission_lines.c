#include "emission_lines.h"
#include <math.h>

void set_OIII_coeffs(double T)
{
  if (T <= 0.0)
    T = 1e4;

  double beta_q = sqrt(2.0 * M_PI * pow(HBAR_SI, 4) / (BOLTZMANN_SI * pow(ELECTRON_MASS_SI, 3))) * 1e6;

  run_globals.loiii_params.o30 = 0.243 * pow(T / 1e4, 0.120 + 0.031 * log(T / 1e4));
  run_globals.loiii_params.o40 = 0.0321 * pow(T / 1e4, 0.118 + 0.057 * log(T / 1e4));
  run_globals.loiii_params.k03 = (beta_q / sqrt(T)) * run_globals.loiii_params.o30 * exp(-29169.0 / T);
  run_globals.loiii_params.k04 = (beta_q / sqrt(T)) * run_globals.loiii_params.o40 * exp(-61207.0 / T);
}

void compute_LOIII(galaxy_t* gal, int snapshot)
{
  if (gal == NULL || snapshot < 0 || run_globals.ZZ == NULL)
    return;

  run_units_t* units = &run_globals.units;

  double cs_cms_sq = COLD_GAS_SOUND_SPEED_CMS * COLD_GAS_SOUND_SPEED_CMS;
  double sfr_gs = gal->Sfr * units->UnitMass_in_g / units->UnitTime_in_s;

  if (gal->ColdGas <= 0.0 || gal->Mvir <= 0.0 || gal->DiskScaleLength <= 0.0 || gal->Rvir <= 0.0 || gal->dt <= 0.0 ||
      gal->Vmax <= 0.0 || gal->StellarMass < 0.0 || gal->Mcool < 0.0 || gal->Spin <= 0.0) {
    gal->LOIII = 0.0;
    gal->ionization_param = 0.0;
    return;
  }

  double Rd_cm_sq = gal->DiskScaleLength * units->UnitLength_in_cm;
  Rd_cm_sq = Rd_cm_sq * Rd_cm_sq;
  double frac_md = (gal->StellarMass + gal->ColdGas) / gal->Mvir;
  double Re_kpc = 0.08 * pow(gal->Spin / 0.05, 4.0 / 3.0) * pow(frac_md / 0.17, -2.0 / 3.0) *
                        pow(gal->Mvir / run_globals.params.Hubble_h / 1e-2, -2.0 / 9.0) * // normalized at 1e8 Msol
                        pow((1.0 + run_globals.ZZ[snapshot]) / 10.0, -4.0 / 3.0);

  double Mtot_g = (gal->StellarMass + gal->ColdGas) / run_globals.params.Hubble_h * units->UnitMass_in_g;
  double mass_g_sq = gal->ColdGas / run_globals.params.Hubble_h * units->UnitMass_in_g * Mtot_g;
  double np_prefactor = (GRAVITY / SOLAR_MASS * mass_g_sq) / (8.0 * M_PI * pow(Rd_cm_sq, 2) * 2.71 * 2.71);
  double H_cm = (2.0 * cs_cms_sq * Rd_cm_sq) / (GRAVITY * Mtot_g);
  double Vdisk = M_PI * Rd_cm_sq * H_cm;
  double dsn = (0.156 * sfr_gs) / (SN_PROGENITOR_MASS_MSUN * SOLAR_MASS * Vdisk);
  double vsn_cms = cbrt((2.0 * dsn * 0.03 * 1e51 * H_cm) / np_prefactor * cs_cms_sq ); // np_prefactor / cs_cms_sq is np1 * PROTONMASS
  
  double delta = (gal->Mvir + gal->ColdGas + gal->StellarMass) / gal->ColdGas * gal->DiskScaleLength / gal->Rvir;
  double Q_toomre = 0.7 * sqrt(2.0) * delta * (COLD_GAS_SOUND_SPEED_CMS / ( gal->Vmax * units->UnitVelocity_in_cm_per_s ));
  double M_dot_gs = gal->Mcool * units->UnitMass_in_g / (gal->dt * units->UnitTime_in_s); // little h cancelled out
  double vacc_cms = cbrt((0.6 * GRAVITY * M_dot_gs * Q_toomre * Q_toomre) / (6.0 * 0.7 * 0.7));
  double Vsb = (4.0 / 3.0) * M_PI * pow(Re_kpc * (MPC / 1e3), 3);
  double np2 = np_prefactor / (cs_cms_sq + vsn_cms * vsn_cms + vacc_cms * vacc_cms); // this is actually np2 * PROTONMASS
  double Nbub = Mtot_g / np2 / Vsb;
  double QHI = ((sfr_gs / Nbub / PROTONMASS)) * 4000.0;
  double Rs  = pow(3 * QHI / (4 * M_PI * ALPHA_HII * np2 * np2), 1.0/3.0);   // cm

  gal->ionization_param = 1.5874 * QHI / (4 * M_PI * Rs * Rs * np2); // cm/s
  if (!isfinite(gal->ionization_param) || gal->ionization_param < 0.0)
    gal->ionization_param = 0.0;

  gal->LOIII = ((pow(10.0, 8.69) / 1e12) * (gal->MetalsColdGas / gal->ColdGas / Z_SUN) *
                  (run_globals.loiii_params.k03 + run_globals.loiii_params.k04 * (LOIII_A43 / (LOIII_A43 + LOIII_A41))) *
                  (LOIII_A32 / (LOIII_A32 + LOIII_A31)) *
                  (QHI * Nbub / ALPHA_HII) * PLANCK_SI * nu32 * 0.8) * 1e7;

  if (!isfinite(gal->LOIII) || gal->LOIII < 0.0)
    gal->LOIII = 0.0;
}