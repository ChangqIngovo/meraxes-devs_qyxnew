#ifndef BLACKHOLE_FEEDBACK_H
#define BLACKHOLE_FEEDBACK_H

#include "meraxes.h"

#define ETA 0.06
// standard efficiency, 6% accreted mass is radiated

#define BONDI_HOYLE_COEFFICIENT 3.4754
// 15/8*pi*mu=3.4754, with mu=0.59; x=k*m_p*t/lambda
#define VELOCITY_SCALE 280.0

#define EMISSIVITY_CONVERTOR 8.40925088e-8
//=1e60 * PROTONMASS / 1e10 / SOLAR_MASS
// Unit_convertor is to convert emissivity from 1e60 photons to photons per atom

#define LUMINOSITY_CONVERTOR 32886.5934
// = (1 solar mass *(speed of light)^2/450e6year) /3.828e26watt
// where 450e6year is eddington time scale

#define LB2EMISSIVITY 2.127633e-6
// this is a little bit complicated...
// firstly B band luminosity is:   LB = Lbol/kb 1e10Lsun
// then B band magnitude is:       MB = 4.74 - 2.5log10(1e10LB)
// then convert from Vega to AB:   MAB,B = MB-0.09
// then UV mag:                    M1450 = MAB,B+0.524
// then UV lum:                    LUV = 10**((M1450_1-51.594843)/-2.5) #erg/s/Hz
// then 912 lum:                   L912 = LUV*(1200./1450)**0.44*(912/1200.)**1.57  #erg/s/Hz
// then BHemissivity:              L912/Planck constant/1.57 #photons/s
// then total BHemissivity in this step: BHemissivity*accretion_time(seconds)
// BHemissivity = fobs*Lbol/kb*2.1276330276278045e+54*accretion_time(seconds)/1e60; // photon numbers/1e60
// BHemissivity = fobs*Lbol/kb*2.1276330276278045e-6*accretion_time(seconds); // photon numbers/1e60

/* Morrison & McCammon (1983) Table 2: photoelectric cross-section coefficients.
 * σ(E) = (C0 + C1·E + C2·E²) × E^{-3} × 10^{-24} cm²,  E in keV.
 * Columns: E_low [keV], E_high [keV], C0, C1, C2 */
static const double mm83[14][5] = {
  { 0.030,  0.100,  17.3,    608.1,  -2150.0 },
  { 0.100,  0.284,  34.6,    267.9,   -476.1 },
  { 0.284,  0.400,  78.1,     18.8,      4.3 },
  { 0.400,  0.532,  71.4,     66.8,    -51.4 },
  { 0.532,  0.707,  95.5,    145.8,    -61.1 },
  { 0.707,  0.867, 308.9,   -380.6,    294.0 },
  { 0.867,  1.303, 120.6,    169.3,    -47.7 },
  { 1.303,  1.840, 141.3,    146.8,    -31.5 },
  { 1.840,  2.471, 202.7,    104.7,    -17.0 },
  { 2.471,  3.210, 342.7,     18.7,      0.0 },
  { 3.210,  4.038, 352.2,     18.7,      0.0 },
  { 4.038,  7.111, 433.9,     -2.4,      0.75},
  { 7.111,  8.331, 629.0,     30.9,      0.0 },
  { 8.331, 10.000, 701.2,     25.2,      0.0 },
};

#ifdef __cplusplus
extern "C"
{
#endif
  double radio_mode_BH_heating(struct galaxy_t* gal, double cooling_mass, double x);

  /*
   * calculate_BHemissivity
   * Computes UV and intrinsic hard X-ray luminosities and emissivities.
   *
   * Outputs:
   *   emissivity      UV ionising photon emissivity  [1e60 photons]
   *   accretion_time  e-folding accretion timescale  [internal units]
   *   quasar_luv      UV luminosity                  [1e10 Lsun]
   *   quasar_lx       intrinsic log10(LX / Lsun)     hard X-ray
   *   xray_emissivity intrinsic X-ray emissivity     [1e60 erg]
   */
  void calculate_BHemissivity(double BlackHoleMass, double accreted_mass,
                              double *emissivity,     double *accretion_time,
                              double *quasar_luv,     double *quasar_lx,
                              double *quasar_lx_soft, double *xray_emissivity);

  void merger_driven_BH_growth(struct galaxy_t* gal, double merger_ratio, int snapshot);
  void previous_merger_driven_BH_growth(struct galaxy_t* gal, int snapshot);


#ifdef __cplusplus
}
#endif

#endif