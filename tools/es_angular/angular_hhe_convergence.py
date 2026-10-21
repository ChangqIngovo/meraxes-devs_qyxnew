"""Reproduce the fixed-front spectral midpoint refinement diagnostic."""
import json
import argparse
from pathlib import Path
import numpy as np
from angular_hhe import illuminate_frozen_front


def run():
    gas=np.array([[1.,0.,.2,0.,0.],[.4,.2,.1,0.,0.]])
    credits=np.array([.3,.4,.2])
    route=np.array([[.4,.6],[.7,.3],[.2,.8]])
    cross=np.array([[1.,0,0],[.3,.7,0],[.1,.3,.5]])
    fractions=[.08,.04,.02,.01]
    rows=[illuminate_frozen_front(gas,credits,cross,route,
          donor_fraction=value,stop_at_event=False) for value in fractions]
    differences=[float(np.abs(a['gas']-b['gas']).sum()) for a,b in zip(rows,rows[1:])]
    return dict(scope='two-cell three-spectrum fixed-front chemistry',
        donor_fractions=fractions,
        accepted_substeps=[r['substeps'] for r in rows],
        adjacent_gas_l1=differences,
        refinement_ratios=[a/b for a,b in zip(differences,differences[1:])],
        max_abs_photon_error=max(abs(r['photon_error']) for r in rows),
        initial_split_roundoff_l1=float(rows[0]['split_roundoff_l1'].sum()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='Write the JSON report to this path; default: stdout only')
    args=parser.parse_args()
    report=run()
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
