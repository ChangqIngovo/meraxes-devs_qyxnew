#include <math.h>
#include "blackhole_feedback.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <assert.h>


/* =========================================================================
 * X-ray obscuration model — Ueda+2014 / Shen+2020
 *
 * Ported directly from obsec_0_14.ipynb (Cells 9, 10, 11).
 *
 * Three steps mirror the Python exactly:
 *   1. xray_transmission_band()  — Cell 10: integrate Morrison & McCammon
 *                                   cross-section over [E_min, E_max] keV
 *                                   and average over a logNH bin.
 *   2. _NH_distribution()        — Cell 9:  NH population fractions
 *                                   following Ueda+2014 Eqs 7-8.
 *   3. apply_xray_obscuration()  — Cell 11: combine fractions × transmission
 *                                   to get observed LX per obscuration class.
 * =========================================================================
 */

/*
 * morrison_mccammon_sigma — Morrison & McCammon (1983) Table 2.
 *
 * Returns the photoelectric absorption cross-section per H atom [cm^2]
 * at energy E_keV, using the piecewise polynomial fit:
 *
 *   sigma(E) = (C0 + C1*E + C2*E^2) * 1e-24  cm^2
 *
 * Valid range: 0.030 – 10.0 keV.
 * Outside range: returns 0.0 for E > 10 keV (optically thin);
 *                extrapolates the first row for E < 0.030 keV.
 */
static double morrison_mccammon_sigma(double E_keV)
{
  /* Table 2 of Morrison & McCammon (1983).
   * Columns: E_low [keV], E_high [keV], C0, C1, C2 */
  static const double mm83[14][5] = {
    { 0.030,  0.100,  17.3,    608.1,  -2150.0 },
    { 0.100,  0.284,  34.6,    267.9,   -476.1 },
    { 0.284,  0.400,  78.1,     18.8,      4.3 },
    { 0.400,  0.532,  71.4,     66.8,    -51.4 },
    { 0.532,  0.707,  95.5,    145.8,    -61.1 },
    { 0.707,  0.867, 308.9,   -380.6,    294.0 },
    { 0.867,  1.303, 120.6,    169.3,    -47.7 },
    { 1.303,  1.840, 141.3,     66.3,    -30.5 },
    { 1.840,  2.471, 202.7,     42.7,    -16.7 },
    { 2.471,  3.210, 342.7,      5.7,      0.7 },
    { 3.210,  4.038, 352.2,     11.1,     -3.7 },
    { 4.038,  7.111, 433.9,     -2.4,      0.75},
    { 7.111,  8.331, 629.0,     30.9,      0.0 },
    { 8.331, 10.000, 701.2,     25.2,      0.0 },
  };

  double C0;
  double C1;
  double C2;
  int    i;

  /* Above table range: IGM optically thin, cross-section negligible */
  if (E_keV > 10.0) return 0.0;

  /* Find the matching energy interval */
  for (i = 0; i < 14; i++) {
    if (E_keV >= mm83[i][0] && E_keV < mm83[i][1]) {
      C0 = mm83[i][2];
      C1 = mm83[i][3];
      C2 = mm83[i][4];
      return (C0 + C1 * E_keV + C2 * E_keV * E_keV) * 1.0e-24;
    }
  }

  /* Below table range (E < 0.030 keV): extrapolate with first row */
  C0 = mm83[0][2];
  C1 = mm83[0][3];
  C2 = mm83[0][4];
  return (C0 + C1 * E_keV + C2 * E_keV * E_keV) * 1.0e-24;
}
#define OBS_EPSILON   1.7     /* ratio logNH=23-24 to logNH=22-23 quasars  */
#define OBS_FCTK      1.0     /* CTK fraction relative to absorbed CTN      */
#define OBS_PSI_MIN   0.20    /* minimum absorbed fraction                  */
#define OBS_PSI_MAX   0.84    /* maximum absorbed fraction                  */
#define OBS_LX_REF    43.75   /* reference log10(LX/erg/s) for psi(LX,z)   */

/*
 * xray_transmission_band — Cell 10 of obsec_0_14.ipynb translated to C.
 *
 * Computes the band-averaged X-ray transmission T, averaged over BOTH
 * the energy band [E_min, E_max] keV AND the NH bin [log_NH_min, log_NH_max].
 *
 * Cross-section: sigma(E) = 2e-22 * E^{-2.6}  cm^2  (Morrison & McCammon)
 * Compton scattering correction applied for logNH >= 24:
 *   T_compton = exp(-NH / 1.5e24)
 *
 * Parameters
 * ----------
 *   log_NH_min : lower bound of NH bin in log10 cm^-2  (e.g. 20)
 *   log_NH_max : upper bound of NH bin in log10 cm^-2  (e.g. 21)
 *   E_min_keV  : lower energy bound in keV             (e.g. 2.0)
 *   E_max_keV  : upper energy bound in keV             (e.g. 10.0)
 *   n_NH       : number of NH samples within bin       (e.g. 20)
 *   n_E        : number of energy samples for integration (e.g. 40)
 *
 * Returns
 * -------
 *   Band- and NH-averaged transmission T in [0, 1].
 */
static double xray_transmission_band(double log_NH_min, double log_NH_max,
                                     double E_min_keV,  double E_max_keV,
                                     int    n_NH,       int    n_E)
{
  /* All declarations at top of block for C89 compliance */
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

  T_NH_avg = 0.0;
  dlog_NH  = (log_NH_max - log_NH_min) / (double)n_NH;
  dE       = (E_max_keV  - E_min_keV)  / (double)n_E;

  for (i_NH = 0; i_NH < n_NH; i_NH++) {
    /* Sample logNH uniformly across the bin — midpoint of each sub-interval */
    log_NH  = log_NH_min + (i_NH + 0.5) * dlog_NH;
    NH      = pow(10.0, log_NH);
    T_E_sum = 0.0;

    for (i_E = 0; i_E < n_E; i_E++) {
      E     = E_min_keV + (i_E + 0.5) * dE;         /* midpoint             */
      sigma = morrison_mccammon_sigma(E);             /* Morrison & McCammon (1983) Table 2 */
      tau   = sigma * NH;
      T     = exp(-tau);

      /* Compton scattering correction for Compton-thick columns (logNH >= 24).
       * Uses the Thomson cross-section sigma_T = 6.6524e-25 cm^2.          */
      if (log_NH >= 24.0)
        T *= exp(-NH * 1.21 * 6.6524e-25);

      T_E_sum += T;
    }

    /* Average over energy band */
    T_NH_avg += T_E_sum / (double)n_E;
  }

  /* Average over NH bin */
  return T_NH_avg / (double)n_NH;
}

/*
 * Absorbed CTN fraction at the reference luminosity log10(LX)=43.75.
 * Redshift dependence clamped at z=2 following Ueda+2014.
 * Mirrors psi_43_75(z) from Cell 9.
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
 * Mirrors psi() from Cell 9.
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
  double p   = _psi(LX_log, z);
  double eps = OBS_EPSILON;
  double thr = (1.0 + eps) / (3.0 + eps);

  if (p < thr) {
    f[0] = 1.0 - (2.0 + eps) / (1.0 + eps) * p;   /* logNH 20-21 */
    f[1] = 1.0 / (1.0 + eps) * p;                  /* logNH 21-22 */
  } else {
    f[0] = 2.0/3.0 - (3.0 + 2.0*eps) / (3.0 + 3.0*eps) * p;
    f[1] = 1.0/3.0 - eps / (3.0 + 3.0*eps) * p;
  }
  f[2] = 1.0 / (1.0 + eps) * p;                    /* logNH 22-23 */
  f[3] = eps / (1.0 + eps) * p;                    /* logNH 23-24 */
  f[4] = (OBS_FCTK / 2.0) * p;                     /* CTK per half-dex */
}

/*
 * apply_xray_obscuration — Cell 11 of obsec_0_14.ipynb translated to C.
 *
 * Given the intrinsic log10(LX / L_sun) and the snapshot redshift,
 * computes the four observed LX contributions by combining NH fractions
 * with energy-band-averaged transmission.
 *
 * Energy band: 2-10 keV (hard X-ray), matching the notebook default.
 * To change band: adjust E_MIN_KEV / E_MAX_KEV below.
 *
 * Outputs are log10(LX_observed / L_sun).
 * Sentinel value -99.0 means zero contribution (mask in Python with lx < -90).
 *
 * Parameters
 * ----------
 *   LX_log_sun    : intrinsic log10(L_X / L_sun)
 *   redshift      : current snapshot redshift
 *   LX_obs_total  : output — total observed (all classes combined)
 */
static void apply_xray_obscuration(double LX_log_sun,
                                   double redshift,
                                   double *LX_obs_total,
                                   double *obs_fraction)
{
  /* All declarations at top of block for C89 compliance */
  /* Energy band parameters — change these to use a different band   */
  const double E_MIN_KEV = 2.0;    /* lower energy bound in keV     */
  const double E_MAX_KEV = 10.0;   /* upper energy bound in keV     */
  const int    N_NH      = 20;     /* NH samples per bin            */
  const int    N_E       = 40;     /* energy samples for integration*/

  double T[5];
  double LX_log_ergs;
  double f[5];
  double total_f;
  double LX_lin;
  double contrib_unabs;
  double contrib_abs;
  double contrib_ctk;
  double contrib_total;

  *LX_obs_total  = -99.0;
  *obs_fraction  = 0.0;

  if (LX_log_sun <= -90.0) return;

  /* Compute transmission for each NH bin by calling xray_transmission_band()
   * exactly as in Cell 10 of obsec_0_14.ipynb:
   *
   *   T_by_bin = {
   *     '20_21': xray_transmission_band(20, 21, 2.0, 10.0),
   *     '21_22': xray_transmission_band(21, 22, 2.0, 10.0),
   *     '22_23': xray_transmission_band(22, 23, 2.0, 10.0),
   *     '23_24': xray_transmission_band(23, 24, 2.0, 10.0),
   *     'CTK'  : xray_transmission_band(24, 26, 2.0, 10.0),
   *   }
   *
   * These are computed at runtime (not pre-baked constants) so the energy
   * band can be changed by editing E_MIN_KEV / E_MAX_KEV above.
   */
  T[0] = xray_transmission_band(20.0, 21.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[1] = xray_transmission_band(21.0, 22.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[2] = xray_transmission_band(22.0, 23.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[3] = xray_transmission_band(23.0, 24.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);
  T[4] = xray_transmission_band(24.0, 26.0, E_MIN_KEV, E_MAX_KEV, N_NH, N_E);

  /* Convert log10(LX/L_sun) → log10(LX/erg/s) for psi(LX,z)
   * log10(L_sun / erg/s) = log10(3.828e33) ≈ 33.583 */
  LX_log_ergs = LX_log_sun + 33.583;

  /* NH fractions from Ueda+2014 Eqs 7-8 */
  _NH_distribution(LX_log_ergs, redshift, f);

  /* Normalise fractions (CTK spans 2 dex so counts x2) */
  total_f = f[0] + f[1] + f[2] + f[3] + 2.0 * f[4];
  if (total_f <= 0.0) return;

  LX_lin = pow(10.0, LX_log_sun);

  contrib_unabs = LX_lin * (f[0]/total_f * T[0] + f[1]/total_f * T[1]);
  contrib_abs   = LX_lin * (f[2]/total_f * T[2] + f[3]/total_f * T[3]);
  contrib_ctk   = LX_lin * (2.0 * f[4]/total_f * T[4]);
  contrib_total = contrib_unabs + contrib_abs + contrib_ctk;

  *LX_obs_total = (contrib_total > 0.0) ? log10(contrib_total) : -99.0;

  /* obs_fraction is a dimensionless number in [0, 1].
   *
   * It is the NH-population-weighted, band-averaged transmission:
   *
   *   obs_fraction = sum_bins [ f(NH_bin)/total_f  *  T(NH_bin) ]
   *
   * Because contrib_total = LX_lin * obs_fraction, the luminosity
   * cancels exactly and we write it directly as the weighted sum:
   *
   *   obs_fraction  =  f[0]/total_f * T[0]   (logNH 20-21)
   *                  + f[1]/total_f * T[1]   (logNH 21-22)
   *                  + f[2]/total_f * T[2]   (logNH 22-23)
   *                  + f[3]/total_f * T[3]   (logNH 23-24)
   *                  + 2*f[4]/total_f * T[4] (logNH 24-26, CTK)
   *
   * No division by LX_lin is needed — units are pure fractions.
   * Compare with quasar_fobs = 1 - cos(open_angle/2) which is
   * a fixed constant (~0.234 for open_angle=80 degrees).
   * obs_fraction varies with LX and z through psi(LX,z). */
  *obs_fraction = f[0]/total_f * T[0]
               + f[1]/total_f * T[1]
               + f[2]/total_f * T[2]
               + f[3]/total_f * T[3]
               + 2.0 * f[4]/total_f * T[4];
}

/* ======================================================================= */
void calculate_BHemissivity(double BlackHoleMass, double accreted_mass,
                            double *emissivity,     double *accretion_time,
                            double *quasar_luv,     double *quasar_lx,
                            double *xray_emissivity)
{
  double Lbol;          /* bolometric luminosity in 1e10 Lsun          */
  double kb;            /* UV bolometric correction (Shen+2020)         */
  double kb_hard;       /* hard X-ray bolometric correction (Shen+2020) */
  double LX_1e10Lsun;  /* hard X-ray luminosity in 1e10 Lsun           */
  double LX_erg_s;      /* hard X-ray luminosity in erg/s               */
  physics_params_t* physics = &(run_globals.params.physics);

  /* Accretion timescale in internal units */
  *accretion_time = log1p(accreted_mass / BlackHoleMass)
                    * run_globals.EddingtonTimescale * ETA
                    / physics->EddingtonRatio;

  /* Bolometric luminosity in 1e10 Lsun at MIDDLE of accretion time */
  Lbol = sqrt(1.0 + accreted_mass / BlackHoleMass)
         * physics->EddingtonRatio * BlackHoleMass
         / run_globals.params.Hubble_h * LUMINOSITY_CONVERTOR;

  /* UV bolometric correction (Richards+2006 / Shen+2020) */
  kb      = 6.25  * pow(Lbol, -0.37)  + 9.0   * pow(Lbol, -0.012);

  /* Hard X-ray (2-10 keV) bolometric correction (Shen+2020) */
  kb_hard = 4.073 * pow(Lbol, -0.026) + 12.60 * pow(Lbol,  0.278);

  /* UV luminosity in 1e10 Lsun */
  *quasar_luv = Lbol / kb;

  /* Hard X-ray intrinsic luminosity in log10(L_X / L_sun)
   * Lbol/kb_hard is in 1e10 Lsun → add 10 to convert to log10(LX/Lsun) */
  LX_1e10Lsun = Lbol / kb_hard;
  *quasar_lx  = (LX_1e10Lsun > 0.0) ? (log10(LX_1e10Lsun) + 10.0) : -99.0;

  /* UV ionising photon emissivity in 1e60 photons */
  *emissivity = physics->quasar_fobs * *quasar_luv * LB2EMISSIVITY
               * *accretion_time * run_globals.units.UnitTime_in_s
               / run_globals.params.Hubble_h;

  LX_erg_s = LX_1e10Lsun * 1e10 * 3.828e33;
  *xray_emissivity = LX_erg_s
                     * (*accretion_time) * run_globals.units.UnitTime_in_s
                     / run_globals.params.Hubble_h / 1e60;
  /* Note: obs_fraction (from NH distribution + transmission) is
   * multiplied in previous_merger_driven_BH_growth() after obscuration
   * is applied, replacing the fixed quasar_fobs opening angle. */
}

static double get_vvir(galaxy_t* gal) {
    // If this galaxy is the central of it's FOF group then use the FOF Halo properties
    // TODO: This needs closer thought as to if this is the best thing to do...
  return ((gal->Type == 0) && (!gal->ghost_flag)) ? gal->Halo->FOFGroup->Vvir : gal->Vvir;
}

// quasar feedback suggested by Croton et al. 2016
static void update_reservoirs_from_quasar_mode_bh_feedback(galaxy_t* gal, double m_reheat)
{
  double metallicity;
  galaxy_t* central;

  if (gal->ghost_flag)
    central = gal;
  else
    central = gal->Halo->FOFGroup->FirstOccupiedHalo->Galaxy;

  if (m_reheat < gal->ColdGas) {
    metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);
    gal->ColdGas -= m_reheat;
    gal->MetalsColdGas -= m_reheat * metallicity;
    central->MetalsHotGas += m_reheat * metallicity;
    central->HotGas += m_reheat;
  } else {
    metallicity = calc_metallicity(central->HotGas, central->MetalsHotGas);
    gal->ColdGas = 0.0;
    gal->MetalsColdGas = 0.0;
    central->HotGas -= m_reheat;
    central->MetalsHotGas -= m_reheat * metallicity;
    central->EjectedGas += m_reheat;
    central->MetalsEjectedGas += m_reheat * metallicity;
  }

  // Check the validity of the modified reservoir values (HotGas can be negative for too strong quasar feedback)
  CLAMP_NEGATIVE(central->HotGas);
  CLAMP_NEGATIVE(central->MetalsHotGas);
  CLAMP_NEGATIVE(gal->ColdGas);
  CLAMP_NEGATIVE(gal->MetalsColdGas);
  CLAMP_NEGATIVE(gal->StellarMass);
  CLAMP_NEGATIVE(central->EjectedGas);
  CLAMP_NEGATIVE(central->MetalsEjectedGas);
}

double radio_mode_BH_heating(galaxy_t* gal, double cooling_mass, double x)
{
  double heated_mass;
  double Vvir;
  double accreted_mass;
  double eddington_mass;
  double metallicity;
  run_units_t* units;

  heated_mass = 0.0;

  if (gal->HotGas > 0.0) {
    Vvir  = get_vvir(gal);
    units = &(run_globals.units);


    /* bondi-hoyle accretion model */
    accreted_mass =
      run_globals.params.physics.RadioModeEff * run_globals.G * BONDI_HOYLE_COEFFICIENT * x * gal->BlackHoleMass * gal->dt;

    /* eddington rate */
    eddington_mass = exp(gal->dt / run_globals.EddingtonTimescale / ETA * run_globals.params.physics.EddingtonRatio) *
                            gal->BlackHoleMass;

    // limit accretion by the eddington rate
    if (accreted_mass > eddington_mass)
      accreted_mass = eddington_mass;

    // limit accretion by amount of hot gas available
    if (accreted_mass > gal->HotGas)
      accreted_mass = gal->HotGas;

    // mass heated by AGN following Croton et al. 2006
    heated_mass = 2. * ETA * run_globals.Csquare / Vvir / Vvir * accreted_mass;

    // limit the amount of heating to the amount of cooling
    if (heated_mass > cooling_mass) {
      accreted_mass = cooling_mass / heated_mass * accreted_mass;
      heated_mass = cooling_mass;
    }

    gal->BlackHoleAccretedHotMass = accreted_mass;

    // add the accreted mass to the black hole from hotgas
    metallicity = calc_metallicity(gal->HotGas, gal->MetalsHotGas);

    // Assuming all energy from radio mode is going to heat the cooling flow
    // So no emissivity from radio mode!
    // TODO: we could add heating effienciency to split the energy into
    // heating and reionization.
    gal->BlackHoleMass += accreted_mass * (1. - ETA);
    gal->HotGas -= accreted_mass;
    gal->MetalsHotGas -= accreted_mass * metallicity;
  }
  return heated_mass;
}

void merger_driven_BH_growth(galaxy_t* gal, double merger_ratio, int snapshot)
{
  double Vvir;
  double z_scaling;
  double accreting_mass;
  double metallicity;

  if (gal->ColdGas > 0) {
    Vvir = get_vvir(gal);

    z_scaling = pow((1 + run_globals.ZZ[snapshot]), run_globals.params.physics.quasar_mode_scaling);

    accreting_mass = run_globals.params.physics.BlackHoleGrowthRate * merger_ratio /
                            (1.0 + pow(VELOCITY_SCALE / Vvir, 2)) * gal->ColdGas * z_scaling;

    // limit accretion to what is available
    if (accreting_mass > gal->ColdGas)
      accreting_mass = gal->ColdGas;

    metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);

    // put the mass onto the accretion disk and let the black hole accrete it in the next snapshot
    // TODO: since the merger is put in the end of galaxy evolution, this is following the
    // inconsistence consistently
    gal->BlackHoleAccretingColdMass += accreting_mass;
    gal->ColdGas -= accreting_mass;
    gal->MetalsColdGas -= accreting_mass * metallicity;
  }
}

void previous_merger_driven_BH_growth(galaxy_t* gal, int snapshot)
{
  // If there is any cold gas to feed the black hole...
  double m_reheat;
  double accreted_mass;
  double BHemissivity, accretion_time, quasar_luv;
  double quasar_lx, xray_emissivity;
  double LX_obs_total;
  double obs_fraction;
  double t_off, t_resp;
  double Vvir;
  double factor;
  double dt;
  run_units_t* units;

  Vvir   = get_vvir(gal);
  units  = &(run_globals.units);
  factor = EMISSIVITY_CONVERTOR * gal->FescBH / run_globals.params.physics.ReionNionPhotPerBary;
  dt     = (snapshot > 0) ? (run_globals.LTTime[snapshot - 1] - run_globals.LTTime[snapshot]) : 0.0;

  // Determine the accretion on-time within this snapshot
  if (run_globals.params.physics.Flag_BHARExponentialCut) {
    if (gal->BHAccretionOnTime < 0.0) {
      // First snapshot of accretion after merger - assign random on-time
      gal->BHAccretionOnTime = gsl_rng_uniform(run_globals.random_generator);
    } else {
      // Accretion was already happening in previous snapshot - start immediately
      gal->BHAccretionOnTime = 0.0;
    }
    // Adjust effective timestep based on when accretion starts
    dt *= (1.0 - gal->BHAccretionOnTime);
  } else {
    // No random on-time when using duty-cycle weighting
    gal->BHAccretionOnTime = 0.0;
  }


  if (dt > 0.0){
    // Eddington rate (using effective timestep)
    accreted_mass = expm1(dt / run_globals.EddingtonTimescale / ETA * run_globals.params.physics.EddingtonRatio) *
                    gal->BlackHoleMass;

    // limit accretion to what is need
    if (accreted_mass > gal->BlackHoleAccretingColdMass)
      accreted_mass = gal->BlackHoleAccretingColdMass;

    gal->BlackHoleAccretedColdMass += accreted_mass;
    gal->BlackHoleAccretingColdMass -= accreted_mass;

    // Reset on-time if accretion is complete
    if (gal->BlackHoleAccretingColdMass <= 0.0)
      gal->BHAccretionOnTime = -1.0;

    /* Step 1: compute intrinsic UV and X-ray luminosities */
    calculate_BHemissivity(gal->BlackHoleMass, accreted_mass,
                           &BHemissivity, &accretion_time,
                           &quasar_luv, &quasar_lx,
                           &xray_emissivity);

    /* Step 2: apply obscuration model
     * Calls xray_transmission_band() for each NH bin (2-10 keV)
     * then weights by NH fractions from Ueda+2014 Eqs 7-8
     * to get the total observed X-ray luminosity.
     * Mirrors Cell 10 + Cell 11 of obsec_0_14.ipynb exactly. */
    apply_xray_obscuration(quasar_lx,
                           run_globals.ZZ[snapshot],
                           &LX_obs_total,
                           &obs_fraction);

    /* Step 3: store results in galaxy_t */
    gal->BHemissivity     += BHemissivity;                /* UV photons (1e60)         */
    gal->QuasarLuv        += quasar_luv;                  /* UV lum  (1e10 Lsun)       */
    gal->QuasarLX          = quasar_lx;                   /* intrinsic log10(LX/Lsun)  */
    gal->QuasarLX_obs      = LX_obs_total;                /* observed  log10(LX/Lsun)  */

    /* X-ray emissivity: scale intrinsic value by obs_fraction
     * obs_fraction comes from apply_xray_obscuration() and encodes
     * the fraction of photons that reach the observer after:
     *   (a) NH population weighting  (Ueda+2014 Eqs 7-8)
     *   (b) 2-10 keV band transmission (Morrison & McCammon)
     * This replaces the fixed quasar_fobs = 1-cos(open_angle/2)
     * which was a geometry-only approximation. */
    gal->BHXrayEmissivity += xray_emissivity * obs_fraction;
    gal->DutyCycleAGN = accretion_time / dt;
    CLAMP_0_1(gal->DutyCycleAGN);
        
    gal->BlackHoleMass += (1. - ETA) * accreted_mass;
    gal->EffectiveBHM += BHemissivity * factor;

    BHemissivity *= factor / accretion_time;

    if (run_globals.params.physics.Flag_BHARExponentialCut) {
      t_off = (1.0 - gal->DutyCycleAGN) * dt;
      if (t_off > 0.0) {
        t_resp = gal->t_resp * run_globals.params.Hubble_h / units->UnitTime_in_Megayears;
        if (t_resp > 0.0)
          BHemissivity *= exp(-t_off / t_resp);
      }
    } else
      BHemissivity *= gal->DutyCycleAGN;

    gal->EffectiveBHAR += BHemissivity;

    // quasar mode feedback
    m_reheat = run_globals.params.physics.QuasarModeEff * 2. * ETA * run_globals.Csquare * accreted_mass / Vvir / Vvir;
    update_reservoirs_from_quasar_mode_bh_feedback(gal, m_reheat);
  }
}