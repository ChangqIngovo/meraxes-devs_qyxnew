#!/usr/bin/env python3
"""Real He II two-photon spectrum and energy accounting, not a full gas solver.

Spectrum: Nussbaumer & Schmutz (1984), as reproduced in Chluba & Sunyaev
(2005), Eq. 4, arXiv:astro-ph/0508144. The hydrogenic spectral shape is
rescaled from H I to He II; nonrelativistic E_2s-1s = 40.8 eV is used.
Cross sections: Verner et al. (1996), parameters in Hui & Gnedin (1997),
Appendix A. All integrals count BOTH photons, integral q(E) dE = 2.

An event test follows a real conditional atomic channel:
He III + electron(epsilon) -> He II(2s) + photon(13.6+epsilon eV), then
He II(2s) -> He II(1s) + two photons with total energy 40.8 eV.
No arbitrary branching ratios or representative photon energies are used.
It is NOT a model for the relative probabilities of all helium captures.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad, quad_vec

KB_EV = 8.617333262145e-5
EV_ERG = 1.602176634e-12
CHI = np.array([13.6, 24.59, 54.4])
LINE_ENERGY_EV = 40.8


def shape(y):
    y = np.asarray(y)
    w = np.maximum(y*(1.0-y), 0.0)
    ans = 46.26*(w*(1.0-(4.0*w)**0.8) +
                 0.88*w**1.53*(4.0*w)**0.8)
    return np.where((y >= 0) & (y <= 1), ans, 0.0)


NORM = quad(shape, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12)[0]


def photon_spectrum(energy_ev, line_energy_ev=LINE_ENERGY_EV):
    """Photons per decay per eV, including both members of each pair."""
    return 2.0*shape(np.asarray(energy_ev)/line_energy_ev)/NORM/line_energy_ev


def band_moments(low_ev, high_ev, line_energy_ev=LINE_ENERGY_EV):
    low = max(0.0, low_ev)
    high = min(line_energy_ev, high_ev)
    if high <= low:
        return {'number': 0.0, 'energy_ev': 0.0, 'mean_energy_ev': None}
    n = quad(lambda e: photon_spectrum(e, line_energy_ev), low, high,
             epsabs=1e-12, epsrel=1e-12)[0]
    en = quad(lambda e: e*photon_spectrum(e, line_energy_ev), low, high,
              epsabs=1e-12, epsrel=1e-12)[0]
    return {'number': n, 'energy_ev': en, 'mean_energy_ev': en/n}


def cross_sections(energy_ev):
    """Verner ground-state photoionization fit, returned in cm^2."""
    e0 = np.array([0.4298, 13.61, 1.720])
    sigma0 = np.array([5.475e-14, 9.492e-16, 1.369e-14])
    p = np.array([2.963, 3.188, 2.963])
    ya = np.array([32.88, 1.469, 32.88])
    yw = np.array([0., 2.039, 0.])
    y0 = np.array([0., .4434, 0.])
    y1 = np.array([0., 2.136, 0.])
    x = energy_ev/e0-y0
    y = np.sqrt(x*x+y1*y1)
    result = sigma0*((x-1)**2+yw*yw)*y**(.5*p-5.5)*(1+np.sqrt(y/ya))**(-p)
    return np.where(energy_ev >= CHI, result, 0.)


def absorption_shares(energy_ev, absorber_densities):
    """Local OTS shares; zero opacity leaves photons for explicit transport."""
    opacity = np.asarray(absorber_densities)*cross_sections(energy_ev)
    total = opacity.sum()
    return opacity/total if total > 0.0 else np.zeros(3)


def two_photon_deposition(absorber_densities):
    """Real spectrum integrated with opacity at EACH photon energy.

    Returns absorbed number and absorbed energy by species, plus unabsorbed
    number/energy. This computes local OTS deposition for infinitesimal
    emission in a frozen mixed medium, not a finite-capacity chemistry step.
    Below-threshold photons are part of the unabsorbed radiation energy.
    """
    def integrand(e):
        share = absorption_shares(e, absorber_densities)
        p = photon_spectrum(e)
        return np.r_[p*share, p*e*share]
    total = np.zeros(6)
    for lo, hi in zip([0., CHI[0], CHI[1]], [CHI[0], CHI[1], LINE_ENERGY_EV]):
        part = quad_vec(integrand, lo, hi, epsabs=1e-11, epsrel=1e-11)[0]
        total += part
    return {'absorbed_number': total[:3], 'absorbed_energy_ev': total[3:],
            'unabsorbed_number': 2.-total[:3].sum(),
            'unabsorbed_energy_ev': LINE_ENERGY_EV-total[3:].sum()}


def conditional_2s_recombination(absorber_densities, captured_electron_ev=.5,
                                 temperature_k=1e4):
    """One conditional He III -> He II(2s) capture plus its 2s cascade.

    The captured electron energy is a specified event property, NOT an
    invented mean over a Maxwellian recombination kernel. The photon-derived
    kinetic pool is audited before any secondary-electron deposition.
    The optional thermostat result additionally assumes complete immediate
    thermalization of that pool; it is not a secondary-electron model.
    """
    dep = two_photon_deposition(absorber_densities)
    e_cont = CHI[2]-LINE_ENERGY_EV+captured_electron_ev
    share = absorption_shares(e_cont, absorber_densities)
    number = dep['absorbed_number']+share
    energy = dep['absorbed_energy_ev']+e_cont*share
    unabs_number = dep['unabsorbed_number']+1-share.sum()
    unabs_energy = dep['unabsorbed_energy_ev']+e_cont*(1-share.sum())
    ion_binding = CHI @ number
    delta_binding = ion_binding-CHI[2]
    photoelectron_pool = energy.sum()-ion_binding
    delta_kinetic = photoelectron_pool-captured_electron_ev
    delta_electrons = number.sum()-1.
    required_delta_thermal = 1.5*KB_EV*temperature_k*delta_electrons
    thermostat_into_gas = required_delta_thermal-delta_kinetic
    return {
        'captured_electron_ev': captured_electron_ev,
        'absorber_densities_arbitrary_common_units': list(absorber_densities),
        'absorbed_number_by_species': number.tolist(),
        'absorbed_energy_ev_by_species': energy.tolist(),
        'unabsorbed_radiation_number': unabs_number,
        'unabsorbed_radiation_energy_ev': unabs_energy,
        'photoelectron_kinetic_pool_ev': photoelectron_pool,
        'delta_material_kinetic_ev': delta_kinetic,
        'delta_binding_ev': delta_binding,
        'delta_electrons': delta_electrons,
        'fixed_T_delta_thermal_ev': required_delta_thermal,
        'thermostat_into_gas_if_full_thermalization_ev': thermostat_into_gas,
        'photon_ledger_error': number.sum()+unabs_number-3.,
        'energy_ledger_error_ev': delta_binding+delta_kinetic+unabs_energy,
        'fixed_T_ledger_error_ev': delta_binding+required_delta_thermal+unabs_energy-thermostat_into_gas,
    }


def hydrogen_rates_and_cooling(temperature_k):
    """Hui & Gnedin 1997 H II fits; cooling is removed electron kinetic energy."""
    t = temperature_k
    lam = 2.*157807./t
    aa = 1.269e-13*lam**1.503/(1+(lam/.522)**.470)**1.923
    ab = 2.753e-14*lam**1.5/(1+(lam/2.740)**.407)**2.242
    ea = 1.778e-29*t*lam**1.965/(1+(lam/.541)**.502)**2.697
    eb = 3.435e-30*t*lam**1.970/(1+(lam/2.250)**.376)**3.720
    return aa, ab, ea, eb


def hydrogen_ots_cancellation(temperature_k):
    """Ground-state cooling reabsorbed locally cancels once, not twice."""
    aa, ab, ea, eb = hydrogen_rates_and_cooling(temperature_k)
    ground_cooling = ea-eb
    ground_mean_electron_ev = ground_cooling/(aa-ab)/EV_ERG
    net_cooling = ea-ground_cooling
    incorrect_B_plus_reheat = eb-ground_cooling
    return {'temperature_k': temperature_k,
            'alpha_A_cm3_s': aa, 'alpha_B_cm3_s': ab,
            'case_A_cooling_erg_cm3_s': ea, 'case_B_cooling_erg_cm3_s': eb,
            'ground_mean_captured_electron_ev': ground_mean_electron_ev,
            'case_A_minus_local_reheat_erg_cm3_s': net_cooling,
            'identity_relative_error': (net_cooling-eb)/eb,
            'incorrect_B_plus_reheat_fraction_of_correct_cooling': incorrect_B_plus_reheat/eb}


def run_tests():
    bands = [band_moments(a,b) for a,b in [(0.,13.6),(13.6,24.59),(24.59,40.8)]]
    assert abs(sum(x['number'] for x in bands)-2) < 1e-12
    assert abs(sum(x['energy_ev'] for x in bands)-40.8) < 1e-10
    moments = {'bands_ev': [[0.,13.6],[13.6,24.59],[24.59,40.8]], 'moments': bands,
               'HI_ionizing_yield': bands[1]['number']+bands[2]['number'],
               'HeI_ionizing_yield': bands[2]['number'],
               'Friedrich_l': 1.425, 'Friedrich_m': .737,
               'total_emitted_photons': sum(x['number'] for x in bands),
               'total_emitted_energy_ev': sum(x['energy_ev'] for x in bands)}
    tests = []
    for density in ([1.,0.,0.], [1.,.08,0.], [.001,.08,0.], [0.,.08,0.], [0.,0.,1.]):
        for epsilon in (0.,.5,2.):
            out = conditional_2s_recombination(density, epsilon)
            assert abs(out['energy_ledger_error_ev']) < 1e-10
            assert abs(out['fixed_T_ledger_error_ev']) < 1e-10
            assert abs(out['photon_ledger_error']) < 1e-12
            assert out['photoelectron_kinetic_pool_ev'] >= 0.
            tests.append(out)
    cancellation = [hydrogen_ots_cancellation(t) for t in (5000.,10000.,20000.)]
    result = {'scope': 'Real HeII 2gamma spectrum and a conditional state-resolved capture channel; not a full thermal evolution.',
              'two_photon_spectrum': moments,
              'conditional_capture_tests': tests,
              'pure_hydrogen_OTS_cooling_cancellation': cancellation,
              'max_abs_energy_ledger_error_ev': max(abs(x['energy_ledger_error_ev']) for x in tests),
              'max_abs_fixed_T_ledger_error_ev': max(abs(x['fixed_T_ledger_error_ev']) for x in tests)}
    dest = Path(__file__).with_name('thermal_results.json')
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'results':str(dest), 'spectrum_moments':moments,
                      'max_abs_energy_ledger_error_ev':result['max_abs_energy_ledger_error_ev'],
                      'max_abs_fixed_T_ledger_error_ev':result['max_abs_fixed_T_ledger_error_ev'],
                      'hydrogen_cancellation_10000K':cancellation[1]}, indent=2))


if __name__ == '__main__':
    run_tests()
