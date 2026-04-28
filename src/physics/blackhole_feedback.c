#include <math.h>
#include "blackhole_feedback.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <assert.h>


/* The road map of the x-rays emission by AGN (24 Apr. 2026)
 *   1. xray_transmission_band()  "integrate Morrison & McCammon"
 *   2. _NH_distribution()
 *   3. apply_xray_obscuration()
 */

static double morrison_mccammon_sigma(double E_keV)
{
  /* Table 2 of Morrison & McCammon (1983).
   * Each row: { E_low [keV], E_high [keV], C0, C1, C2 }
   * sigma(E) = (C0 + C1*E + C2*E^2) * 1e-24 cm^2   for E in [E_low, E_high)
   */
  static const double mm83[14][5] = {
    /* E_low    E_high    C0          C1          C2        */
    { 0.030,   0.100,   17.3,       608.1,     -2150.0   },
    { 0.100,   0.284,   34.6,       267.9,      -476.1   },
    { 0.284,   0.400,   78.1,        18.8,         4.3   },
    { 0.400,   0.532,   71.4,        66.8,       -51.4   },
    { 0.532,   0.707,   95.5,       145.8,       -61.1   },
    { 0.707,   0.867,  308.9,      -380.6,       294.0   },
    { 0.867,   1.303,  120.6,       169.3,       -47.7   },
    { 1.303,   1.840,  141.3,        66.3,       -30.5   },
    { 1.840,   2.471,  202.7,        42.7,       -16.7   },
    { 2.471,   3.210,  342.7,         5.7,         0.7   },
    { 3.210,   4.038,  352.2,        11.1,        -3.7   },
    { 4.038,   7.111,  433.9,        -2.4,         0.75  },
    { 7.111,   8.331,  629.0,        30.9,         0.0   },
    { 8.331,  10.000,  701.2,        25.2,         0.0   },
  };

  double C0;
  double C1;
  double C2;
  int    i;

  /* Energy outside the table range */
  if (E_keV > 10.0) return 0.0;

  /* Find the matching energy interval */
  for (i = 0; i < 14; i++) {
    if (E_keV >= mm83[i][0] && E_keV < mm83[i][1]) {
      C0 = mm83[i][2];
      C1 = mm83[i][3];
      C2 = mm83[i][4];
      return (C0 + C1 * E_keV + C2 * E_keV * E_keV) * 1.0e-24; /* cm^2 */
    }
  }

  /* E < 0.030 keV: extrapolate using the first row */
  C0 = mm83[0][2];
  C1 = mm83[0][3];
  C2 = mm83[0][4];
  return (C0 + C1 * E_keV + C2 * E_keV * E_keV) * 1.0e-24;
}

/* --- Model parameters (Ueda+2014 / Shen+2020) -------------------------- */
#define OBS_EPSILON   1.7     /* ratio logNH=23-24 to logNH=22-23 quasars  */
#define OBS_FCTK      1.0     /* CTK fraction relative to absorbed CTN      */
#define OBS_PSI_MIN   0.20    /* minimum absorbed fraction                  */
#define OBS_PSI_MAX   0.84    /* maximum absorbed fraction                  */
#define OBS_LX_REF    43.75   /* reference log10(LX/erg/s) for psi(LX,z)   */


static double xray_transmission_band(double log_NH_min, double log_NH_max,
                                     double E_min_keV,  double E_max_keV,
                                     int    n_NH,       int    n_E)
{
  double T_NH_avg;
  double dlog_NH;
  double dE;
  double log_NH;
  double NH;
  double T_E_sum;
  double E;
  double sigma;
  double tau;
  double T;
  int    i_NH;
  int    i_E;

  const double sigma_T = 6.6524e-25;   /* Thomson cross-section [cm^2] */

  /* --- initialisations --- */
  T_NH_avg = 0.0;
  dlog_NH  = (log_NH_max - log_NH_min) / (double)n_NH;
  dE       = (E_max_keV  - E_min_keV)  / (double)n_E;

  /* --- outer loop: average over NH bin --- */
  for (i_NH = 0; i_NH < n_NH; i_NH++) {

    /* midpoint of each log-NH sub-interval */
    log_NH  = log_NH_min + (i_NH + 0.5) * dlog_NH;
    NH      = pow(10.0, log_NH);
    T_E_sum = 0.0;

    /* --- inner loop: integrate transmission over energy band --- */
    for (i_E = 0; i_E < n_E; i_E++) {

      E     = E_min_keV + (i_E + 0.5) * dE;   /* midpoint */
      sigma = morrison_mccammon_sigma(E);       /* Morrison & McCammon (1983), Table 2 */
      tau   = sigma * NH;
      T     = exp(-tau);

      /* add Compton scattering correction for Compton-thick columns */
      if (log_NH >= 24.0)
        T *= exp(-NH * 1.21 * sigma_T);

      T_E_sum += T;
    }

    /* average over energy band */
    T_NH_avg += T_E_sum / (double)n_E;
  }

  /* average over NH bin */
  return T_NH_avg / (double)n_NH;
}

/*
 * Absorbed CTN fraction at the reference luminosity log10(LX)=43.75.
 * Redshift dependence clamped at z=2 following Ueda+2014.
 */
static double _psi_ref(double z)
{
  double zeff = (z < 2.0) ? z : 2.0;
  return 0.43 * pow(1.0 + zeff, 0.48);
}

/*
 * psi(LX_log, z) — absorbed CTN fraction.
 * LX_log is log10(LX / erg/s).
 * Clamped to [OBS_PSI_MIN, OBS_PSI_MAX].
 */
static double _psi(double LX_log, double z)
{
  double val = _psi_ref(z) - 0.24 * (LX_log - OBS_LX_REF);
  if (val < OBS_PSI_MIN) val = OBS_PSI_MIN;
  if (val > OBS_PSI_MAX) val = OBS_PSI_MAX;
  return val;
}

/*
 * _NH_distribution — NH population fractions.
 * Returns f[0..4] for bins: 20-21, 21-22, 22-23, 23-24, CTK(24-26).
 * f[4] (CTK) is per half-dex; multiply by 2 for full 24-26 range.
 * Mirrors NH_distribution() from Cell 9 (Ueda+2014 Eqs 7-8).
 */
static void _NH_distribution(double LX_log, double z, double f[5])
{
  double p;
  double eps;
  double thr;
  double total;
  int    i;

  p   = _psi(LX_log, z);
  eps = OBS_EPSILON;
  thr = (1.0 + eps) / (3.0 + eps);

  /* --- Raw fractions (Ueda+2014) --- */
  if (p < thr) {
    f[0] = 1.0 - (2.0 + eps) / (1.0 + eps) * p;   /* 20-21 */
    f[1] = 1.0 / (1.0 + eps) * p;                  /* 21-22 */
  } else {
    f[0] = 2.0/3.0 - (3.0 + 2.0*eps) / (3.0 + 3.0*eps) * p;
    f[1] = 1.0/3.0 - eps / (3.0 + 3.0*eps) * p;
  }

  f[2] = 1.0 / (1.0 + eps) * p;                    /* 22-23 */
  f[3] = eps / (1.0 + eps) * p;                    /* 23-24 */
  f[4] = (OBS_FCTK / 2.0) * p;                     /* CTK per half-dex */

  /* --- Normalisation (CTK spans 2 dex → ×2) --- */
  total = f[0] + f[1] + f[2] + f[3] + 2.0 * f[4];

  if (total > 0.0) {
    f[0] /= total;
    f[1] /= total;
    f[2] /= total;
    f[3] /= total;
    f[4] = (2.0 * f[4]) / total;  /* fold the ×2 into f[4] */
  } else {
    /* Fallback: no valid distribution */
    for (i = 0; i < 5; i++) f[i] = 0.0;
  }
}

static void apply_xray_obscuration(double  LX_log_sun,
                                   double  redshift,
                                   double *LX_obs_total,
                                   double *obs_fraction)
{
  double T[5];
  double LX_log_ergs;
  double f[5];
  double LX_lin;
  double contrib_total;

  *LX_obs_total = -99.0;
  *obs_fraction = 0.0;

  if (LX_log_sun <= -90.0) return;

  /* --- Transmission per NH bin --- */
  T[0] = xray_transmission_band(20.0, 21.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[1] = xray_transmission_band(21.0, 22.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[2] = xray_transmission_band(22.0, 23.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[3] = xray_transmission_band(23.0, 24.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[4] = xray_transmission_band(24.0, 26.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);

  /* Convert to erg/s for psi(LX,z) */
  LX_log_ergs = LX_log_sun + 33.583;

  /* --- Get NORMALISED NH fractions --- */
  _NH_distribution(LX_log_ergs, redshift, f);

  /* Intrinsic luminosity */
  LX_lin = pow(10.0, LX_log_sun);

  /* --- Contributions (no total_f, no ×2 anymore!) --- */
  contrib_total =
      LX_lin * (f[0] * T[0]
              + f[1] * T[1]
              + f[2] * T[2]
              + f[3] * T[3]
              + f[4] * T[4]);

  *LX_obs_total = (contrib_total > 0.0) ? log10(contrib_total) : -99.0;

  *obs_fraction =
      f[0] * T[0]
    + f[1] * T[1]
    + f[2] * T[2]
    + f[3] * T[3]
    + f[4] * T[4];
}


void calculate_BHemissivity(double  BlackHoleMass,
                            double  accreted_mass,
                            double *emissivity,
                            double *accretion_time,
                            double *quasar_luv,
                            double *quasar_lx,
                            double *xray_emissivity)
{
  double Lbol;
  double kb_UV;
  double kb_hard;
  double LX_erg_s;
  physics_params_t *physics = &(run_globals.params.physics);

  /* Accretion timescale in internal units */
  *accretion_time = log1p(accreted_mass / BlackHoleMass)
                    * run_globals.EddingtonTimescale * ETA
                    / physics->EddingtonRatio;

  /* Bolometric luminosity in 1e10 Lsun at MIDDLE of accretion time */
  Lbol = sqrt(1.0 + accreted_mass / BlackHoleMass)
         * physics->EddingtonRatio * BlackHoleMass
         / run_globals.params.Hubble_h * LUMINOSITY_CONVERTOR;

  /* UV bolometric correction (Shen+2020) */
  kb_UV  = 1.862 * pow(Lbol, -0.361) + 4.87  * pow(Lbol, -0.0063);

  /* Hard X-ray (2-10 keV) bolometric correction (Shen+2020) */
  kb_hard = 4.073 * pow(Lbol, -0.026) + 12.60 * pow(Lbol,  0.278);

  /* UV luminosity in 1e10 Lsun */
  *quasar_luv = Lbol / kb_UV;

  /* X-ray luminosity in 1e10 Lsun */
  *quasar_lx = Lbol / kb_hard;

  /* UV ionising photon emissivity in 1e60 photons */
  *emissivity = physics->quasar_fobs * *quasar_luv * LB2EMISSIVITY
               * *accretion_time * run_globals.units.UnitTime_in_s
               / run_globals.params.Hubble_h;

  /* X-ray emissivity in erg (convert 1e10 Lsun → erg/s first) */
  LX_erg_s       = *quasar_lx * 1e10 * 3.828e33;
  *xray_emissivity = LX_erg_s
                   * (*accretion_time) * run_globals.units.UnitTime_in_s
                   / run_globals.params.Hubble_h;
  /*obs_fraction (from NH distribution + transmission) is
   * multiplied in previous_merger_driven_BH_growth() after obscuration
   * is applied, replacing the fixed quasar_fobs opening angle. */
}

static double get_vvir(galaxy_t *gal)
{
  /* If this galaxy is the central of its FOF group use the FOF Halo properties */
  return ((gal->Type == 0) && (!gal->ghost_flag))
             ? gal->Halo->FOFGroup->Vvir
             : gal->Vvir;
}

/* quasar feedback suggested by Croton et al. 2016 */
static void update_reservoirs_from_quasar_mode_bh_feedback(galaxy_t *gal,
                                                            double    m_reheat)
{
  double     metallicity;
  galaxy_t  *central;

  if (gal->ghost_flag)
    central = gal;
  else
    central = gal->Halo->FOFGroup->FirstOccupiedHalo->Galaxy;

  if (m_reheat < gal->ColdGas) {
    metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);
    gal->ColdGas        -= m_reheat;
    gal->MetalsColdGas  -= m_reheat * metallicity;
    central->MetalsHotGas += m_reheat * metallicity;
    central->HotGas       += m_reheat;
  } else {
    metallicity = calc_metallicity(central->HotGas, central->MetalsHotGas);
    gal->ColdGas               = 0.0;
    gal->MetalsColdGas         = 0.0;
    central->HotGas           -= m_reheat;
    central->MetalsHotGas     -= m_reheat * metallicity;
    central->EjectedGas       += m_reheat;
    central->MetalsEjectedGas += m_reheat * metallicity;
  }

  /* Check validity of modified reservoir values */
  CLAMP_NEGATIVE(central->HotGas);
  CLAMP_NEGATIVE(central->MetalsHotGas);
  CLAMP_NEGATIVE(gal->ColdGas);
  CLAMP_NEGATIVE(gal->MetalsColdGas);
  CLAMP_NEGATIVE(gal->StellarMass);
  CLAMP_NEGATIVE(central->EjectedGas);
  CLAMP_NEGATIVE(central->MetalsEjectedGas);
}

double radio_mode_BH_heating(galaxy_t *gal, double cooling_mass, double x)
{
  double       heated_mass   = 0.0;
  double       Vvir;
  double       accreted_mass;
  double       eddington_mass;
  double       metallicity;
  run_units_t *units;

  if (gal->HotGas > 0.0) {
    Vvir  = get_vvir(gal);
    units = &(run_globals.units);

    /* Bondi-Hoyle accretion model */
    accreted_mass =
        run_globals.params.physics.RadioModeEff
        * run_globals.G * BONDI_HOYLE_COEFFICIENT
        * x * gal->BlackHoleMass * gal->dt;

    /* Eddington rate */
    eddington_mass = exp(gal->dt / run_globals.EddingtonTimescale
                         / ETA * run_globals.params.physics.EddingtonRatio)
                     * gal->BlackHoleMass;

    /* Limit accretion by the Eddington rate */
    if (accreted_mass > eddington_mass)
      accreted_mass = eddington_mass;

    /* Limit accretion by amount of hot gas available */
    if (accreted_mass > gal->HotGas)
      accreted_mass = gal->HotGas;

    /* Mass heated by AGN following Croton et al. 2006 */
    heated_mass = 2. * ETA * run_globals.Csquare / Vvir / Vvir * accreted_mass;

    /* Limit heating to the amount of cooling */
    if (heated_mass > cooling_mass) {
      accreted_mass = cooling_mass / heated_mass * accreted_mass;
      heated_mass   = cooling_mass;
    }

    gal->BlackHoleAccretedHotMass = accreted_mass;

    metallicity = calc_metallicity(gal->HotGas, gal->MetalsHotGas);

    gal->BlackHoleMass  += accreted_mass * (1. - ETA);
    gal->HotGas         -= accreted_mass;
    gal->MetalsHotGas   -= accreted_mass * metallicity;
  }
  return heated_mass;
}

void merger_driven_BH_growth(galaxy_t *gal, double merger_ratio, int snapshot)
{
  double Vvir;
  double z_scaling;
  double accreting_mass;
  double metallicity;

  if (gal->ColdGas > 0) {

    Vvir = get_vvir(gal);

    /* Suggested by Bonoli et al. 2009 and Wyithe et al. 2003 */
    z_scaling = pow((1 + run_globals.ZZ[snapshot]),
                    run_globals.params.physics.quasar_mode_scaling);

    accreting_mass = run_globals.params.physics.BlackHoleGrowthRate
                     * merger_ratio
                     / (1.0 + pow(VELOCITY_SCALE / Vvir, 2))
                     * gal->ColdGas * z_scaling;

    /* Limit accretion to what is available */
    if (accreting_mass > gal->ColdGas)
      accreting_mass = gal->ColdGas;

    metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);

    gal->BlackHoleAccretingColdMass += accreting_mass;
    gal->ColdGas                    -= accreting_mass;
    gal->MetalsColdGas              -= accreting_mass * metallicity;
  }
}

void previous_merger_driven_BH_growth(galaxy_t *gal, int snapshot)
{
  double       m_reheat;
  double       accreted_mass;
  double       BHemissivity;
  double       accretion_time;
  double       quasar_luv;
  double       quasar_lx;
  double       xray_emissivity;
  double       LX_obs_total;
  double       obs_fraction;
  double       t_off;
  double       t_resp;
  double       dt;
  double       factor;
  double       Vvir;
  run_units_t *units;

  Vvir   = get_vvir(gal);
  units  = &(run_globals.units);
  factor = EMISSIVITY_CONVERTOR * gal->FescBH
           / run_globals.params.physics.ReionNionPhotPerBary;

  /* Use snapshot cadence timestep for ghost galaxies */
  dt = (snapshot > 0)
           ? (run_globals.LTTime[snapshot - 1] - run_globals.LTTime[snapshot])
           : 0.0;

  /* Determine the accretion on-time within this snapshot */
  if (run_globals.params.physics.Flag_BHARExponentialCut) {
    if (gal->BHAccretionOnTime < 0.0)
      gal->BHAccretionOnTime = gsl_rng_uniform(run_globals.random_generator);
    else
      gal->BHAccretionOnTime = 0.0;

    dt *= (1.0 - gal->BHAccretionOnTime);
  } else {
    gal->BHAccretionOnTime = 0.0;
  }

  if (dt > 0.0) {

    /* Eddington rate (using effective timestep) */
    accreted_mass = expm1(dt / run_globals.EddingtonTimescale
                          / ETA * run_globals.params.physics.EddingtonRatio)
                    * gal->BlackHoleMass;

    if (accreted_mass > gal->BlackHoleAccretingColdMass)
      accreted_mass = gal->BlackHoleAccretingColdMass;

    gal->BlackHoleAccretedColdMass  += accreted_mass;
    gal->BlackHoleAccretingColdMass -= accreted_mass;

    if (gal->BlackHoleAccretingColdMass <= 0.0)
      gal->BHAccretionOnTime = -1.0;

    /* Compute intrinsic UV and X-ray luminosities */
    calculate_BHemissivity(gal->BlackHoleMass, accreted_mass,
                           &BHemissivity, &accretion_time,
                           &quasar_luv,   &quasar_lx,
                           &xray_emissivity);

    /* Apply obscuration model */
    apply_xray_obscuration(quasar_lx,
                           run_globals.ZZ[snapshot],
                           &LX_obs_total,
                           &obs_fraction);

    /* Store results in galaxy_t */
    gal->BHemissivity     += BHemissivity;
    gal->QuasarLuv        += quasar_luv;
    gal->QuasarLX         += quasar_lx;
    gal->QuasarLX_obs     += LX_obs_total;

    gal->BHXrayEmissivity += xray_emissivity * obs_fraction;
    gal->DutyCycleAGN      = accretion_time / dt;
    CLAMP_0_1(gal->DutyCycleAGN);

    gal->BlackHoleMass  += (1. - ETA) * accreted_mass;
    gal->EffectiveBHM   += BHemissivity * factor;

    BHemissivity *= factor / accretion_time;

    if (run_globals.params.physics.Flag_BHARExponentialCut) {
      t_off = (1.0 - gal->DutyCycleAGN) * dt;
      if (t_off > 0.0) {
        t_resp = gal->t_resp * run_globals.params.Hubble_h
                 / units->UnitTime_in_Megayears;
        if (t_resp > 0.0)
          BHemissivity *= exp(-t_off / t_resp);
      }
    } else {
      BHemissivity *= gal->DutyCycleAGN;
    }

    gal->EffectiveBHAR += BHemissivity;

    /* Quasar mode feedback */
    m_reheat = run_globals.params.physics.QuasarModeEff
               * 2. * ETA * run_globals.Csquare
               * accreted_mass / Vvir / Vvir;
    update_reservoirs_from_quasar_mode_bh_feedback(gal, m_reheat);
  }
}