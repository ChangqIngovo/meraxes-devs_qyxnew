"""Spatial refinement of the earliest six-connected paid path, at 8 bins."""
import json
import time
import argparse
from pathlib import Path
import numpy as np
from benchmarks import grid,solver,audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sizes',type=int,nargs='+',default=[48,60,72,96])
    args=parser.parse_args()
    if any(n<12 or n%12 for n in args.sizes):
        parser.error('sizes must be positive multiples of 12, so source positions match')
    folder=args.output_dir.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for n in args.sizes:
        c=grid(n,2);started=time.perf_counter()
        with solver(c,0) as model:
            s=model.advance(2.5)
            r=dict(n=n,dx=c['dx'],source_indices=c['src'].tolist(),source_positions=((c['src']-c['src'][0])*c['dx']).tolist(),
                level=0,bins=model.nbins,initial_source_volume=float(c['initial_source_photons'][0]),
                contact_time=s['diagnostics']['first_connection_time'],
                equal_volume_spherical_contact_time=float(2*(4*np.pi/3*.75**3-c['initial_source_photons'][0])),
                elapsed_seconds=time.perf_counter()-started,**audit(c,s))
            records.append(r);print(json.dumps(r),flush=True)
            (folder/'benchmarks_contact_refinement.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
