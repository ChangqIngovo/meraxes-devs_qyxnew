#include <math.h>
#include "blackhole_feedback.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <assert.h>

void calculate_BHemissivity(double BlackHoleMass, double accreted_mass, double *emissivity, double *accretion_time, double *quasar_mag)
{
  double Lbol; // bolometric luminosity
  double kb;   // bolometric correction
  physics_params_t* physics = &(run_globals.params.physics);

  // Compute accretion_time directly in internal units using pre-computed EddingtonTimescale
  *accretion_time = log1p(accreted_mass / BlackHoleMass) * run_globals.EddingtonTimescale * ETA / physics->EddingtonRatio;

  // Bolometric luminosity in 1e10 Lsun at the MIDDLE of accretion time
  // TODO: this introduce inconsistency compared to the calculation of luminosity.
  // Here we assume Lbol(t) = Lbol(t = accretion_time/2), which is only an approximation!
  Lbol = sqrt(1. + accreted_mass / BlackHoleMass) * physics->EddingtonRatio * BlackHoleMass /
         run_globals.params.Hubble_h * LUMINOSITY_CONVERTOR;
  kb = 6.25 * pow(Lbol, -0.37) + 9.0 * pow(Lbol, -0.012);

  *emissivity = physics->quasar_fobs * Lbol / kb * LB2EMISSIVITY * *accretion_time * run_globals.units.UnitTime_in_s / run_globals.params.Hubble_h; // 1e60 photon numbers

  // Calculate UV magnitude M1450 (AB system)
  // LB = Lbol/kb in 1e10 Lsun
  // MB = 4.74 - 2.5*log10(1e10*LB) = -20.26 - 2.5*log10(LB)
  // MAB,B = MB - 0.09 (Vega to AB)
  // M1450 = MAB,B + 0.524 (B to 1450A, assuming spectral index 0.44)
  // => M1450 = -19.826 - 2.5*log10(LB)
  *quasar_mag = -19.826 - 2.5 * log10(Lbol / kb);
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
  double heated_mass = 0.0;

  // if there is any hot gas
  if (gal->HotGas > 0.0) {
    double Vvir = get_vvir(gal);
    run_units_t* units = &(run_globals.units);


    // bondi-hoyle accretion model
    double accreted_mass =
      run_globals.params.physics.RadioModeEff * run_globals.G * BONDI_HOYLE_COEFFICIENT * x * gal->BlackHoleMass * gal->dt;

    // eddington rate
    double eddington_mass = exp(gal->dt / run_globals.EddingtonTimescale / ETA * run_globals.params.physics.EddingtonRatio) *
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
    double metallicity = calc_metallicity(gal->HotGas, gal->MetalsHotGas);

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
  if (gal->ColdGas > 0) {
    // If there is any cold gas to feed the black hole...
    
    double Vvir = get_vvir(gal);

    // Suggested by Bonoli et al. 2009 and Wyithe et al. 2003
    double z_scaling = pow((1 + run_globals.ZZ[snapshot]), run_globals.params.physics.quasar_mode_scaling);

    double accreting_mass = run_globals.params.physics.BlackHoleGrowthRate * merger_ratio /
                            (1.0 + pow(VELOCITY_SCALE / Vvir, 2)) * gal->ColdGas * z_scaling;

    // limit accretion to what is available
    if (accreting_mass > gal->ColdGas)
      accreting_mass = gal->ColdGas;

    // add the accreting mass to the black hole from coldgas
    double metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);

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
  double BHemissivity, accretion_time, quasar_mag;
  double t_off, t_resp;
  double Vvir = get_vvir(gal);
  run_units_t* units = &(run_globals.units);
  double factor = EMISSIVITY_CONVERTOR * gal->FescBH / run_globals.params.physics.ReionNionPhotPerBary;

  // Use snapshot cadence timestep instead of gal->dt to ensure proper accretion
  // for ghost galaxies that have been in ghost state for multiple snapshots.
  // snapshot is the current snapshot, snapshot+1 is the next snapshot in the past.
  double dt = (snapshot > 0) ? (run_globals.LTTime[snapshot - 1] - run_globals.LTTime[snapshot]) : 0.0;

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

    // N_gamma,q * N_bh; later 1e60*BHemissivity * PROTONMASS/1e10/SOLAR_MASS will be N_gamma,q * M_bh
    calculate_BHemissivity(gal->BlackHoleMass, accreted_mass, &BHemissivity, &accretion_time, &quasar_mag);
    // historical reason for us to store nion rather than the emissivity in BHemissivity...
    gal->BHemissivity += BHemissivity;
    gal->QuasarMag = quasar_mag;
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
