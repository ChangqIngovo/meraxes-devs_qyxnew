#include <math.h>
#include "blackhole_feedback.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <assert.h>

void calculate_BHemissivity(double BlackHoleMass, double accreted_mass, double *emissivity, double *accretion_time)
{
  double Lbol; // bolometric luminotisy
  double kb;   // bolometric correction
  physics_params_t* physics = &(run_globals.params.physics);

  *accretion_time = log1p(accreted_mass / BlackHoleMass) * EDDINGTON_TIME_SCALE * ETA / physics->EddingtonRatio; // Myr

  // Bolometric luminosity in 1e10 Lsun at the MIDDLE of accretion time
  // TODO: this introduce inconsistency compared to the calculation of luminosity.
  // Here we assume Lbol(t) = Lbol(t = accretion_time/2), which is only an approximation!
  Lbol = sqrt(1. + accreted_mass / BlackHoleMass) * physics->EddingtonRatio * BlackHoleMass /
         run_globals.params.Hubble_h * LUMINOSITY_CONVERTOR;
  kb = 6.25 * pow(Lbol, -0.37) + 9.0 * pow(Lbol, -0.012);

  *emissivity = physics->quasar_fobs * Lbol / kb * LB2EMISSIVITY * *accretion_time; // 1e60 photon numbers
  *accretion_time /= (run_globals.units.UnitTime_in_Megayears / run_globals.params.Hubble_h); // internal units
}

static double get_vvir(galaxy_t* gal) {
    // If this galaxy is the central of it's FOF group then use the FOF Halo properties
    // TODO: This needs closer thought as to if this is the best thing to do...
  return (gal->Type == 0) ? gal->Halo->FOFGroup->Vvir : gal->Vvir;
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

  // Check the validity of the modified reservoir values (HotGas can be negtive for too strong quasar feedback)
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
    double eddington_mass = exp(gal->dt * units->UnitTime_in_Megayears / run_globals.params.Hubble_h / EDDINGTON_TIME_SCALE /
                                ETA * run_globals.params.physics.EddingtonRatio) *
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
                            (1.0 + pow(Vvir, -2)) * gal->ColdGas * z_scaling;

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

void previous_merger_driven_BH_growth(galaxy_t* gal)
{
  // If there is any cold gas to feed the black hole...
  double m_reheat;
  double accreted_mass;
  double BHemissivity, accretion_time;
  double Vvir = get_vvir(gal);
  run_units_t* units = &(run_globals.units);
  double factor = EMISSIVITY_CONVERTOR * gal->FescBH / run_globals.params.physics.ReionNionPhotPerBary;

  // Eddington rate
  accreted_mass = expm1(gal->dt * units->UnitTime_in_Megayears / run_globals.params.Hubble_h / EDDINGTON_TIME_SCALE / ETA *
                        run_globals.params.physics.EddingtonRatio) *
                  gal->BlackHoleMass;

  // limit accretion to what is need
  if (accreted_mass > gal->BlackHoleAccretingColdMass)
    accreted_mass = gal->BlackHoleAccretingColdMass;

  gal->BlackHoleAccretedColdMass += accreted_mass;
  gal->BlackHoleAccretingColdMass -= accreted_mass;

  // N_gamma,q * N_bh; later 1e60*BHemissivity * PROTONMASS/1e10/SOLAR_MASS will be N_gamma,q * M_bh
  calculate_BHemissivity(gal->BlackHoleMass, accreted_mass, &BHemissivity, &accretion_time);
  // historical reason for us to store nion rather than the emissivity in BHemissivity...
  gal->BHemissivity += BHemissivity;
  gal->DutyCycleAGN = accretion_time / gal->dt;
  assert(gal->DutyCycleAGN <= 1);
  gal->BlackHoleMass += (1. - ETA) * accreted_mass;
  gal->EffectiveBHM += BHemissivity * factor;
  gal->EffectiveBHAR += BHemissivity / accretion_time * factor;

  // quasar mode feedback
  m_reheat = run_globals.params.physics.QuasarModeEff * 2. * ETA * run_globals.Csquare * accreted_mass / Vvir / Vvir;
  update_reservoirs_from_quasar_mode_bh_feedback(gal, m_reheat);
}
