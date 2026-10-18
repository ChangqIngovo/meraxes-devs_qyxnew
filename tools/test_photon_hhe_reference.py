#!/usr/bin/env python3
"""Compare native H/He reactions with the frozen NumPy/SciPy research solver.

Uses a temporary shared library; no simulation files or installed Meraxes needed.
The reference uses its original temperature-dependent emission nodes. Independent
positive interpolation projects only its returned photons for this comparison.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/hhe_reference/atomic"))
from closed_hhe import advance, kernel, thermal_energy

G = 31
Vec = C.c_double * G


class State(C.Structure):
    _fields_ = [("species", C.c_double * 5), ("thermal", C.c_double)]


class Options(C.Structure):
    _fields_ = [("donor", C.c_double), ("max_steps", C.c_ulong), ("collisional", C.c_int)]


class Coefficients(C.Structure):
    _fields_ = [("down", C.c_double * 3), ("capture", C.c_double * 3),
                ("inactive", C.c_double * 3), ("emission", Vec * 3),
                ("ci", C.c_double * 3), ("excitation", C.c_double),
                ("freefree", C.c_double), ("line_sigma", C.c_double * 2)]


class Ledger(C.Structure):
    _fields_ = [("absorbed", Vec * 3), ("recombinations", C.c_double * 3),
                ("ci", C.c_double * 3)] + [(name, C.c_double) for name in
                ("conversions", "photoheat", "capture", "ci_cooling", "extra_cooling",
                 "inactive", "high_photoheat", "energy_error", "electron_error",
                 "photon_error")] + [("substeps", C.c_ulong)]


def projection(energies, numbers, kinds, grid, grid_kind):
    result = np.zeros(G)
    for energy, number, kind in zip(energies, numbers, kinds):
        if not number:
            continue
        if kind:
            match = np.flatnonzero(grid_kind == kind)
            assert len(match) == 1 and grid[match[0]] == energy
            result[match[0]] += number
            continue
        band = lambda e: 0 if e < 24.59 else (1 if e < 54.4 else 2)
        available = np.array([i for i in range(G) if grid_kind[i] == 0 and
                              band(grid[i]) == band(energy)])
        exact = available[grid[available] == energy]
        if len(exact):
            result[exact[0]] += number
            continue
        lower = available[grid[available] < energy][-1]
        upper = available[grid[available] > energy][0]
        result[lower] += number * (grid[upper]-energy)/(grid[upper]-grid[lower])
        result[upper] += number * (energy-grid[lower])/(grid[upper]-grid[lower])
    np.testing.assert_allclose(result.sum(), np.sum(numbers), rtol=5e-15, atol=1e-35)
    np.testing.assert_allclose(result @ grid, np.sum(np.asarray(energies)*numbers), rtol=5e-15, atol=1e-35)
    return result


def compare(library):
    library.hhe_coefficients_at.argtypes = [C.c_double, C.POINTER(Coefficients)]
    library.hhe_advance.argtypes = [C.POINTER(State), C.POINTER(C.c_double), C.c_double,
                                   C.POINTER(Options), C.POINTER(C.c_double), C.POINTER(Ledger)]
    grid = np.array(Vec.in_dll(library, "hhe_group_energy_eV"))
    kinds = np.array((C.c_int * G).in_dll(library, "hhe_group_kind"))
    coefficient_max = 0.
    for T in [5000., 8750., 10039.0625, 12500., 20000.]:
        coeff = Coefficients()
        assert library.hhe_coefficients_at(T, C.byref(coeff)) == 0
        ref = kernel(T)
        projected = np.stack([projection(ref['energies_eV'],
            np.where(ref['group_parent'] == a, ref['photon_alpha_cm3_s'], 0.),
            ref['group_kind'], grid, kinds) for a in range(3)])
        for actual, expected in [(coeff.down, ref['alpha_down_cm3_s']),
                                 (coeff.capture, ref['captured_cooling_alpha_eV_cm3_s']),
                                 (coeff.inactive, ref['low_energy_escape_alpha_eV_cm3_s']),
                                 (coeff.emission, projected)]:
            actual = np.asarray(actual)
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-30)
            coefficient_max = max(coefficient_max, float(np.max(abs(actual-expected)/np.maximum(abs(expected),1e-25))))
    rows = []
    for label, T, dt, density, initial_photons in [
        ('mixed_continuum_and_lines',10039.0625,2e12,
         [2e-5,8e-5,1.5e-6,3.5e-6,3e-6],
         [(18.,3e-6,0),(35.,5e-7,0),(70.,1e-7,0),(21.2,2e-7,1),(40.8,2e-7,2)]),
        ('source_off_recombination',8750.,1e13,
         [2e-5,8e-5,1.5e-6,3.5e-6,3e-6],[]),
        ('line_conversion',12500.,1e13,
         [1e-4,0.,8e-6,0.,0.],[(21.2,1e-14,1)]),
    ]:
        n=np.array(density); u=float(thermal_energy(n,T)); p=np.zeros(G)
        for e,count,kind in initial_photons:
            p+=projection([e],[count],[kind],grid,kinds)
        coeff=Coefficients()
        assert library.hhe_coefficients_at(T,C.byref(coeff))==0
        ref=advance(n,p,grid,dt,u,donor_fraction=.05,line_kind=kinds,
                    line_sigma_cm2=np.array(coeff.line_sigma))
        state=State((C.c_double*5)(*n),u); photons=Vec(*p); emitted=Vec(); ledger=Ledger()
        status=library.hhe_advance(C.byref(state),photons,dt,C.byref(Options(.05,0,1)),emitted,C.byref(ledger))
        assert status==0, (label,status)
        reference_emitted=projection(ref['emitted_energy_eV'],ref['emitted_counts'],
                                     ref['emitted_kind'],grid,kinds)
        comparisons=[(state.species,ref['n_species']), (state.thermal,ref['thermal_energy_eV_cm3']),
                     (photons,ref['n_photon']), (emitted,reference_emitted),
                     (ledger.absorbed,ref['primary_absorptions']),
                     (ledger.recombinations,ref['recombination_down']),
                     (ledger.ci,ref['collisional_ionizations']),
                     (ledger.inactive,ref['low_energy_escape']),
                     (ledger.conversions,ref['line_conversions'])]
        for actual,expected in comparisons:
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=2e-24,err_msg=label)
        assert ledger.substeps==ref['substeps']
        rows.append({'case':label,'substeps':ledger.substeps,
                     'max_species_absolute_difference_cm3':float(np.max(abs(np.array(state.species)-ref['n_species']))),
                     'thermal_absolute_difference_eV_cm3':float(abs(state.thermal-ref['thermal_energy_eV_cm3'])),
                     'max_emission_difference_cm3':float(np.max(abs(np.array(emitted)-reference_emitted)))})
    return {'reference':'frozen research_20261018_closed Python model',
            'coefficient_table_node_max_relative_error':coefficient_max,'cases':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc',default=os.environ.get('CC','cc'))
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='meraxes_hhe_reference_') as directory:
        library=Path(directory)/'chemistry.so'
        subprocess.run([args.cc,'-std=c99','-O2','-fPIC','-shared',
                        '-I'+str(ROOT/'src/physics'),str(ROOT/'src/physics/photon_hhe_chemistry.c'),
                        '-lm','-o',str(library)],check=True)
        result=compare(C.CDLL(str(library)))
    rendered=json.dumps(result,indent=2)+'\n'
    if args.json:
        args.json.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':
    main()
