"""He I 2^1S -> 1^1S two-photon spectrum, Chluba & Ali-Haimoud 2016 Eq. 2.

Primary source: https://arxiv.org/pdf/1510.03877
The returned quadrature counts both emitted photons. It deliberately retains
the nonionizing part, so the emitted count and energy have complete ledgers.
Use the neutral-helium shape, not the hydrogenic He II two-photon shape.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

HEI_2S_ENERGY_EV = 20.6
HI_THRESHOLD_EV = 13.6


def unnormalized_shape(y):
    y = np.asarray(y, dtype=float)
    w = np.maximum(y*(1.-y), 0.)
    shape = 19.604*w**3/(w+.03)**2*(1.742-7.2*w+12.8*w*w)
    return np.where((y >= 0) & (y <= 1), shape, 0.)


NORMALIZATION = quad(unnormalized_shape, 0., 1., epsabs=1e-13, epsrel=1e-13)[0]


def spectrum_photons_per_ev(energy_ev, transition_energy_ev=HEI_2S_ENERGY_EV):
    return 2./NORMALIZATION*unnormalized_shape(np.asarray(energy_ev)/transition_energy_ev)/transition_energy_ev


def hei_two_photon_quadrature(order=16, transition_energy_ev=HEI_2S_ENERGY_EV):
    """3*order quadrature points, split at HI threshold and its pair image.

    weights are photon counts per 2s decay, not a probability normalized to 1.
    The symmetric intervals keep each photon and its partner at E0-E paired.
    Positive weights are normalized to count=2; pairing then also preserves
    the total decay energy exactly. The ionizing yield still needs refinement.
    """
    x, w = leggauss(order)
    edges = sorted([0.,transition_energy_ev-HI_THRESHOLD_EV,
                    HI_THRESHOLD_EV,transition_energy_ev])
    energies, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        e = lo + (x+1.)*.5*(hi-lo)
        q = w*.5*(hi-lo)*spectrum_photons_per_ev(e, transition_energy_ev)
        energies.append(e)
        weights.append(q)
    energies = np.concatenate(energies)
    weights = np.concatenate(weights)
    weights *= 2./weights.sum()
    return {'energies_eV': energies, 'weights': weights,
            'ionizing': energies >= HI_THRESHOLD_EV,
            'transition_energy_eV': transition_energy_ev,
            'mean_energy_eV_per_photon': float(energies@weights/2),
            'ionizing_yield_per_decay': float(weights[energies>=HI_THRESHOLD_EV].sum())}


def run_tests():
    exact_yield = quad(lambda e:spectrum_photons_per_ev(e),HI_THRESHOLD_EV,HEI_2S_ENERGY_EV,
                       epsabs=1e-13,epsrel=1e-13)[0]
    exact_energy = quad(lambda e:e*spectrum_photons_per_ev(e),HI_THRESHOLD_EV,HEI_2S_ENERGY_EV,
                        epsabs=1e-13,epsrel=1e-13)[0]
    rows = []
    for order in [4,8,16,32,64]:
        q=hei_two_photon_quadrature(order)
        es=q['energies_eV']; ws=q['weights']; ion=q['ionizing']
        rows.append({'order_per_interval':order,'total_photons':float(ws.sum()),
                     'total_energy_eV':float(es@ws),
                     'energy_error_eV':float(es@ws-HEI_2S_ENERGY_EV),
                     'HI_ionizing_photons':float(ws[ion].sum()),
                     'HI_ionizing_photon_error':float(ws[ion].sum()-exact_yield),
                     'ionizing_energy_eV':float(es[ion]@ws[ion]),
                     'ionizing_energy_error_eV':float(es[ion]@ws[ion]-exact_energy)})
    assert abs(rows[-1]['energy_error_eV'])<1e-12
    assert abs(rows[-1]['HI_ionizing_photon_error'])<1e-12
    result={'source':'Chluba & Ali-Haimoud 2016, arxiv:1510.03877, Eq.2',
            'assumptions':'Spontaneous low-density HeI 2s two-photon decay; no collisional quenching or induced processes',
            'transition_energy_eV':HEI_2S_ENERGY_EV,
            'raw_analytic_fit_integral':NORMALIZATION,
            'HI_ionizing_yield':exact_yield,
            'HI_ionizing_energy_eV':exact_energy,
            'HI_ionizing_mean_photon_energy_eV':exact_energy/exact_yield,
            'nonionizing_energy_eV':HEI_2S_ENERGY_EV-exact_energy,
            'Wood2004_historical_yield':.56,
            'relative_yield_difference_from_Wood2004':exact_yield/.56-1.,
            'quadrature_refinement':rows}
    Path(__file__).with_name('hei_two_photon_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run_tests()
