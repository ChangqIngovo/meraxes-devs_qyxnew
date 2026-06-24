#include <math.h>
#include "blackhole_feedback.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <assert.h>



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


  if (E_keV > 10.0) return 0.0;

 
  for (i = 0; i < 14; i++) {
    if (E_keV >= mm83[i][0] && E_keV < mm83[i][1]) {
      C0 = mm83[i][2];
      C1 = mm83[i][3];
      C2 = mm83[i][4];
      return (C0 + C1 * E_keV + C2 * E_keV * E_keV) * 1.0e-24;
    }
  }

 
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

static double xray_transmission_band(double log_NH_min, double log_NH_max,
                                     double E_min_keV,  double E_max_keV)
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

  T_NH_avg = 0.0;
  dlog_NH  = (log_NH_max - log_NH_min) / (double)AGN_XRAY_N_NH;
  dE       = (E_max_keV  - E_min_keV)  / (double)AGN_XRAY_N_E;

  for (i_NH = 0; i_NH < AGN_XRAY_N_NH; i_NH++) {
    log_NH  = log_NH_min + (i_NH + 0.5) * dlog_NH;
    NH      = pow(10.0, log_NH);
    T_E_sum = 0.0;

    for (i_E = 0; i_E < AGN_XRAY_N_E; i_E++) {
      E     = E_min_keV + (i_E + 0.5) * dE;
      sigma = morrison_mccammon_sigma(E);
      tau   = sigma * NH;
      T     = exp(-tau);

      if (log_NH >= 24.0)
        T *= exp(-NH * 1.21 * 6.6524e-25);

      T_E_sum += T;
    }
    T_NH_avg += T_E_sum / (double)AGN_XRAY_N_E;
  }
  return T_NH_avg / (double)AGN_XRAY_N_NH;
}

static double _psi_ref(double z)
{
  double zeff = (z < 2.0) ? z : 2.0;
  return 0.43 * pow(1.0 + zeff, 0.48);
}

static double _psi(double LX_log, double z)
{
  double val = _psi_ref(z) - 0.24 * (LX_log - OBS_LX_REF);
  if (val < OBS_PSI_MIN) val = OBS_PSI_MIN;
  if (val > OBS_PSI_MAX) val = OBS_PSI_MAX;
  return val;
}

static void _NH_distribution(double LX_log, double z, double f[5])
{
  double p     = _psi(LX_log, z);
  double eps   = OBS_EPSILON;
  double thr   = (1.0 + eps) / (3.0 + eps);
  double total_f;
  int    i;

  if (p < thr) {
    f[0] = 1.0 - (2.0 + eps) / (1.0 + eps) * p;
    f[1] = 1.0 / (1.0 + eps) * p;
  } else {
    f[0] = 2.0/3.0 - (3.0 + 2.0*eps) / (3.0 + 3.0*eps) * p;
    f[1] = 1.0/3.0 - eps / (3.0 + 3.0*eps) * p;
  }
  f[2] = 1.0 / (1.0 + eps) * p;
  f[3] = eps / (1.0 + eps) * p;
  f[4] = (OBS_FCTK / 2.0) * p;

  total_f = f[0] + f[1] + f[2] + f[3] + 2.0 * f[4];
  if (total_f <= 0.0) {
    for (i = 0; i < 5; i++) f[i] = 0.0;
    return;
  }
  for (i = 0; i < 5; i++) f[i] /= total_f;
}

static void apply_xray_obscuration(double LX_1e10Lsun,
                                   double redshift,
                                   double *LX_obs_1e10Lsun,
                                   double *obs_fraction,
                                   double *obs_fraction_soft)
{
  double T[5];
  double T_soft[5];
  double LX_log_ergs;
  double f[5];

  *LX_obs_1e10Lsun   = 0.0;
  *obs_fraction      = 0.0;
  *obs_fraction_soft = 0.0;

  if (LX_1e10Lsun <= 0.0) return;

  T[0] = xray_transmission_band(20.0, 21.0, AGN_HARD_E_MIN, AGN_HARD_E_MAX);
  T[1] = xray_transmission_band(21.0, 22.0, AGN_HARD_E_MIN, AGN_HARD_E_MAX);
  T[2] = xray_transmission_band(22.0, 23.0, AGN_HARD_E_MIN, AGN_HARD_E_MAX);
  T[3] = xray_transmission_band(23.0, 24.0, AGN_HARD_E_MIN, AGN_HARD_E_MAX);
  T[4] = xray_transmission_band(24.0, 26.0, AGN_HARD_E_MIN, AGN_HARD_E_MAX);

  T_soft[0] = xray_transmission_band(20.0, 21.0, AGN_SOFT_E_MIN, AGN_SOFT_E_MAX);
  T_soft[1] = xray_transmission_band(21.0, 22.0, AGN_SOFT_E_MIN, AGN_SOFT_E_MAX);
  T_soft[2] = xray_transmission_band(22.0, 23.0, AGN_SOFT_E_MIN, AGN_SOFT_E_MAX);
  T_soft[3] = xray_transmission_band(23.0, 24.0, AGN_SOFT_E_MIN, AGN_SOFT_E_MAX);
  T_soft[4] = xray_transmission_band(24.0, 26.0, AGN_SOFT_E_MIN, AGN_SOFT_E_MAX);

  LX_log_ergs = log10(LX_1e10Lsun) + 10.0 + log10(SOLAR_LUM);
  _NH_distribution(LX_log_ergs, redshift, f);

  *obs_fraction = f[0] * T[0] + f[1] * T[1] + f[2] * T[2]
                + f[3] * T[3] + 2.0 * f[4] * T[4];

  *obs_fraction_soft = f[0] * T_soft[0] + f[1] * T_soft[1] + f[2] * T_soft[2]
                     + f[3] * T_soft[3] + 2.0 * f[4] * T_soft[4];

  *LX_obs_1e10Lsun = LX_1e10Lsun * (*obs_fraction);
}

void calculate_BHemissivity(double BlackHoleMass, double accreted_mass,
                            double *emissivity,     double *accretion_time,
                            double *quasar_luv,     double *quasar_lx,
                            double *xray_emissivity)
{
  double Lbol;
  double kb;
  double kb_hard;
  physics_params_t* physics = &(run_globals.params.physics);

  *accretion_time = log1p(accreted_mass / BlackHoleMass)
                    * run_globals.EddingtonTimescale * ETA
                    / physics->EddingtonRatio;

  Lbol = sqrt(1.0 + accreted_mass / BlackHoleMass)
         * physics->EddingtonRatio * BlackHoleMass
         / run_globals.params.Hubble_h * LUMINOSITY_CONVERTOR;

  kb      = 6.25  * pow(Lbol, -0.37)  + 9.0   * pow(Lbol, -0.012);
  kb_hard = 4.073 * pow(Lbol, -0.026) + 12.60 * pow(Lbol,  0.278);

  *quasar_luv = Lbol / kb;
  *quasar_lx  = Lbol / kb_hard;

  *emissivity = physics->quasar_fobs * *quasar_luv * LB2EMISSIVITY
               * *accretion_time * run_globals.units.UnitTime_in_s
               / run_globals.params.Hubble_h;

  *xray_emissivity = *quasar_lx * 1e10 * SOLAR_LUM;
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
  double m_reheat;
  double accreted_mass;
  double BHemissivity, accretion_time, quasar_luv;
  double quasar_lx, xray_emissivity;
  double LX_obs_1e10Lsun;
  double obs_fraction;
  double obs_fraction_soft;
  double t_off, t_resp;
  double Vvir;
  double factor;
  double dt;
  double LX_intrinsic_erg_s;
  double E_thresh, E_break, E_max;
  double as, ah;
  double L_soft_shape, L_hard_shape, sed_ratio;
  run_units_t* units;

  Vvir   = get_vvir(gal);
  units  = &(run_globals.units);
  factor = EMISSIVITY_CONVERTOR * gal->FescBH / run_globals.params.physics.ReionNionPhotPerBary;
  dt     = (snapshot > 0) ? (run_globals.LTTime[snapshot - 1] - run_globals.LTTime[snapshot]) : 0.0;
  
  if (dt <= 0.0) {
      return;  // No time passed, exit function
  }

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

    calculate_BHemissivity(gal->BlackHoleMass, accreted_mass,
                           &BHemissivity, &accretion_time,
                           &quasar_luv, &quasar_lx,
                           &xray_emissivity);
    apply_xray_obscuration(quasar_lx,
                           run_globals.ZZ[snapshot],
                           &LX_obs_1e10Lsun,
                           &obs_fraction,
                           &obs_fraction_soft);


    gal->DutyCycleAGN = (accretion_time > 1e-30) ? (accretion_time / dt) : 0.0;
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

    gal->BHemissivity     += BHemissivity;
    gal->QuasarLuv        += quasar_luv;
    gal->QuasarLX         += quasar_lx;
    gal->QuasarLX_obs     += LX_obs_1e10Lsun;
    if (quasar_lx > 0.0) {
      LX_intrinsic_erg_s = quasar_lx * 1e10 * SOLAR_LUM;
      E_thresh     = run_globals.params.physics.NuXrayThreshold;
      E_break      = run_globals.params.physics.NuXraySoftCut;
      E_max        = run_globals.params.physics.NuXrayMax;
      as           = run_globals.params.physics.SpecIndexXrayAGNSoft;
      ah           = run_globals.params.physics.SpecIndexXrayAGNHard;
      L_soft_shape = (fabs(1.0-as) < 1e-6)
          ? log(E_break / E_thresh)
          : (pow(E_break, 1.0-as) - pow(E_thresh, 1.0-as)) / (1.0-as);
      L_hard_shape = (fabs(1.0-ah) < 1e-6)
          ? log(E_max / E_break) * pow(E_break, ah - as)
          : (pow(E_max, 1.0-ah) - pow(E_break, 1.0-ah)) / (1.0-ah) * pow(E_break, ah - as);
      sed_ratio    = (L_hard_shape > 0.0) ? (L_soft_shape / L_hard_shape) : 0.0;

      gal->EffectiveXrayBHAR      += LX_intrinsic_erg_s * obs_fraction;
      gal->EffectiveXrayBHAR_soft += LX_intrinsic_erg_s * sed_ratio * obs_fraction_soft;
    }
    gal->EffectiveBHAR += BHemissivity;
    // quasar mode feedback
    m_reheat = run_globals.params.physics.QuasarModeEff * 2. * ETA * run_globals.Csquare * accreted_mass / Vvir / Vvir;
    update_reservoirs_from_quasar_mode_bh_feedback(gal, m_reheat);
  }
}  