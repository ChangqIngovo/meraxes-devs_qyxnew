#ifndef PHOTON_HHE_CHEMISTRY_H
#define PHOTON_HHE_CHEMISTRY_H

/* Closed warm H/He model. Physical cm^-3, seconds, eV and eV cm^-3.
 * Species: HI,HII,HeI,HeII,HeIII. Transition: HI,HeI,HeII.
 * This is opt-in 5000 <= T <= 20000 K chemistry, never a temperature clamp.
 * Emissions are endpoint births; the caller transports them on the next step.
 */
#define HHE_NSPECIES 5
#define HHE_NTRANSITIONS 3
#define HHE_NCONTINUUM 29
#define HHE_NGROUPS 31
#define HHE_T_MIN 5000.0
#define HHE_T_MAX 20000.0
#define HHE_C_LIGHT 2.99792458e10
#define HHE_KB_EV_K 8.617333262145e-5

enum hhe_status
{
  HHE_OK = 0,
  HHE_INVALID_INPUT = 1,
  HHE_TEMPERATURE_DOMAIN = 2,
  HHE_SUBSTEP_LIMIT = 3,
  HHE_STEP_UNDERFLOW = 4,
  HHE_POSITIVITY_ERROR = 5
};

typedef struct
{
  double species[HHE_NSPECIES];
  double thermal_eV_cm3;
} hhe_state;

typedef struct
{
  double donor_fraction;      /* (0,0.5], default 0.1 */
  unsigned long max_substeps; /* 0 means no artificial substep cap */
  int include_collisional;    /* CI + HI excitation + free-free */
} hhe_options;

typedef struct
{
  double alpha_down[3];
  double capture_cooling[3];
  double inactive_radiation[3];
  double emission[3][HHE_NGROUPS]; /* coefficient per parent ion, cm^3 s^-1 */
  double collisional_ionization[3];
  double HI_excitation, freefree; /* eV cm^3 s^-1 */
  double line_sigma[2];
} hhe_coefficients;

typedef struct
{
  double absorbed[3][HHE_NGROUPS];
  double recombinations[3], collisional_ionizations[3];
  double line_conversions;
  double photoheat, capture_cooling, collisional_ionization_cooling;
  double extra_radiative_cooling, inactive_radiation;
  double high_energy_photoheat; /* electron excess >= 10.2 eV */
  double energy_error, electron_error, photon_error;
  unsigned long substeps;
} hhe_ledger;

extern const double hhe_group_energy_eV[HHE_NGROUPS];
extern const int hhe_group_kind[HHE_NGROUPS]; /* 0 continuum, 1 HeI, 2 HeII */
extern const double hhe_group_cross_section_cm2[HHE_NGROUPS][3];
extern const double hhe_ionization_energy_eV[3];

double hhe_electron_density(const hhe_state* state);
double hhe_temperature(const hhe_state* state);
double hhe_binding_energy(const hhe_state* state);
int hhe_coefficients_at(double temperature_K, hhe_coefficients* coefficients);
int hhe_line_cross_sections(double temperature_K, double sigma_cm2[2]);
/* Adds photons to existing spectrum. Kind 1/2 requires exactly 21.2/40.8 eV;
 * continuum projection never crosses an absorber threshold. */
int hhe_project_photons(double energy_eV, double count, int kind, double spectrum[HHE_NGROUPS]);
/* Transactional: state/photons/emitted/ledger are untouched on failure.
 * Frozen initial-temperature coefficients; adaptive material/thermal donors.
 * New diffuse counts are returned separately and never absorbed in this call.
 */
int hhe_advance(hhe_state* state,
                double photons[HHE_NGROUPS],
                double dt_s,
                const hhe_options* options,
                double emitted[HHE_NGROUPS],
                hhe_ledger* ledger);
const char* hhe_status_message(int status);

#endif
