"""Independent analytic axial bridge clock for the symmetric two-source fixture.

No graph, event solver, ray tracing, NumPy, or compiled code is used here.
The result is an axial bridge time; arbitrary 3-D earliest connectivity is not
claimed to obey this formula. See axial_contact_theory.txt for the derivation.
"""
from pathlib import Path
import argparse
import json
import math

FOLDER=Path(__file__).resolve().parent

def axial_contact(n,span=4.5,radius=.6,separation=1.5,source_rate=.5):
    h=span/n;m=round(separation/(2*h))
    if abs(m*h-separation/2)>1e-12:raise ValueError('midpoint must be a cell centre')
    k0=math.floor(math.sqrt(radius*radius+1e-14)/h)
    if k0>=m:raise ValueError('initial axis already connected')
    def fraction(k):
        r=k-.5
        return math.atan(.25/(r*math.sqrt(r*r+.5)))/math.pi
    terms=[h**3/(source_rate*fraction(k)) for k in range(k0+1,m)]
    terms.append(h**3/(2*source_rate*fraction(m)))
    clock=math.fsum(terms)
    continuum=4*math.pi/(3*source_rate)*((separation/2)**3-radius**3)
    theta=(radius-k0*h)/h
    leading=(4*math.pi/source_rate)*h*(radius*radius*theta-(separation/2)**2/2)
    return dict(n=n,dx=h,m=m,initial_axis_last_index=k0,initial_axis_radius=k0*h,
        fractional_radius_cells=theta,axial_bridge_time=clock,continuum_time=continuum,
        error=clock-continuum,error_over_dx=(clock-continuum)/h,
        leading_error=leading,remainder_after_leading=clock-continuum-leading,
        remainder_over_dx_squared=(clock-continuum-leading)/(h*h))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    sequence=[axial_contact(n) for n in [24,30,36,48,60,72,96,144,192,288,384,576,768,1152,1536]]
    measured={}
    dynamic=out/'benchmarks_dynamic.json'
    if dynamic.exists():
        for row in json.loads(dynamic.read_text()):
            if row['case']=='double' and row['level']==0:
                measured[row['n']]=row['records'][-1]['diagnostics']['first_connection_time']
    refinement=out/'benchmarks_contact_refinement.json'
    if refinement.exists():
        for row in json.loads(refinement.read_text()):measured[row['n']]=row['contact_time']
    for row in sequence:
        if row['n'] in measured:
            row['simulated_first_6_connection_time']=measured[row['n']]
            row['simulation_minus_axial_formula']=measured[row['n']]-row['axial_bridge_time']
    result=dict(scope='Exact axial bridge clock for this aligned symmetric fixture; earliest 3D connectivity matches in the measured cases, but is not asserted as a general theorem.',
        radius=.6,separation=1.5,source_rate=.5,sequence=sequence)
    (out/'axial_contact.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in sequence:print(json.dumps(row),flush=True)

if __name__=='__main__':main()
