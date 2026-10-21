"""Independent invariants for exact angular-bin/Cartesian-face geometry.

No NumPy dependency. Uses ES_ANGULAR_LIBRARY or explicitly builds in a temporary
directory, then checks partition, divergence and free-graph identities. Prints
JSON; an optional --output saves the report.
"""
from pathlib import Path
import argparse
import ctypes as ct
import itertools
import json
import math
import os
import random
import subprocess
import tempfile


class Bin(ct.Structure):
    _fields_=[('vertices',ct.c_double*9),('inward_normals',ct.c_double*9),
              ('solid_angle',ct.c_double)]


class Geometry:
    def __init__(self,library):
        self.lib=ct.CDLL(str(library))
        self.lib.ag_bin_count.argtypes=[ct.c_uint]
        self.lib.ag_bin_count.restype=ct.c_size_t
        self.lib.ag_make_bins.argtypes=[ct.c_uint,ct.POINTER(Bin),ct.c_size_t]
        self.lib.ag_make_bins.restype=ct.c_int
        self.lib.ag_face_solid_angle.argtypes=[ct.POINTER(ct.c_double),ct.c_int,ct.c_double]
        self.lib.ag_face_solid_angle.restype=ct.c_double
        self.lib.ag_face_bin_solid_angle.argtypes=[ct.POINTER(ct.c_double),ct.c_int,
                                                  ct.c_double,ct.POINTER(Bin)]
        self.lib.ag_face_bin_solid_angle.restype=ct.c_double

    def bins(self,level):
        n=self.lib.ag_bin_count(level)
        result=(Bin*n)()
        assert self.lib.ag_make_bins(level,result,n)==0
        return result

    def face(self,center,axis,halfwidth,bin=None):
        center=(ct.c_double*3)(*center)
        if bin is None:
            return self.lib.ag_face_solid_angle(center,axis,halfwidth)
        return self.lib.ag_face_bin_solid_angle(center,axis,halfwidth,ct.byref(bin))


def identity_checks(geometry,level):
    bins=geometry.bins(level)
    cells=list(itertools.product(range(-3,4),repeat=3))
    cache={}
    def face_values(cell,axis,sign):
        # A globally shared face uses its doubled grid centre as the key.
        key=list(2*x for x in cell);key[axis]+=sign
        key=(axis,*key)
        if key not in cache:
            center=[v/2 for v in key[1:]]
            cache[key]=[geometry.face(center,axis,.5,b) for b in bins]
        return cache[key]
    max_divergence=max_partition=0.
    max_source_error=0.;zero_out_positive_in=0
    out_angles={};face_output={}
    for cell in cells:
        outgoing=[0.]*len(bins);incoming=[0.]*len(bins)
        faces=[]
        for axis in range(3):
            for sign in (-1,1):
                values=face_values(cell,axis,sign)
                center=list(cell);center[axis]+=.5*sign
                total=geometry.face(center,axis,.5)
                assert min(values)>=0 and all(math.isfinite(v) for v in values)
                max_partition=max(max_partition,abs(math.fsum(values)-total))
                dest=outgoing if cell[axis]*sign>=0 else incoming
                for g,value in enumerate(values):dest[g]+=value
                faces.append((axis,sign,values))
        if cell==(0,0,0):
            max_source_error=max(abs(a-b.solid_angle) for a,b in zip(outgoing,bins))
            incoming=[b.solid_angle for b in bins]
        max_divergence=max(max_divergence,max(abs(a-b) for a,b in zip(outgoing,incoming)))
        zero_out_positive_in+=sum(a==0 and b>0 for a,b in zip(outgoing,incoming))
        out_angles[cell]=outgoing
        face_output[cell]=[(ax,sg,v) for ax,sg,v in faces if cell[ax]*sg>=0]
    # Free streaming starts with one resource per angular bin. Check every
    # outgoing face, rather than only the global emitted/escaped total.
    flow={cell:[0.]*len(bins) for cell in cells}
    flow[(0,0,0)]=[b.solid_angle for b in bins]
    escapes=[0.]*len(bins)
    max_flux_absolute=max_flux_relative=0.
    for cell in sorted(cells,key=lambda p:sum(map(abs,p))):
        for g,incoming in enumerate(flow[cell]):
            out=out_angles[cell][g]
            if out==0:
                assert incoming==0,(cell,g,incoming)
                continue
            for axis,sign,values in face_output[cell]:
                value=incoming*values[g]/out
                expected=values[g]
                max_flux_absolute=max(max_flux_absolute,abs(value-expected))
                if expected>0:
                    max_flux_relative=max(max_flux_relative,abs(value/expected-1))
                neighbour=list(cell);neighbour[axis]+=sign;neighbour=tuple(neighbour)
                if neighbour in flow:flow[neighbour][g]+=value
                else:escapes[g]+=value
    angular_ledger=max(abs(a-b.solid_angle) for a,b in zip(escapes,bins))
    assert max_partition<2e-13
    assert max_divergence<2e-13
    assert max_source_error<2e-13
    assert zero_out_positive_in==0
    assert max_flux_absolute<2e-13
    assert angular_ledger<2e-13
    return dict(level=level,bins=len(bins),cells=len(cells),shared_faces=len(cache),
        bin_sphere_partition_error=abs(math.fsum(b.solid_angle for b in bins)-4*math.pi),
        maximum_face_partition_error_sr=max_partition,
        maximum_cell_divergence_error_sr=max_divergence,
        maximum_source_bin_flux_error_sr=max_source_error,
        zero_out_positive_in_cells=zero_out_positive_in,
        free_graph_maximum_face_absolute_error_sr=max_flux_absolute,
        free_graph_maximum_face_relative_error=max_flux_relative,
        free_graph_maximum_angular_ledger_error_sr=angular_ledger)


def additional_checks(geometry):
    rng=random.Random(71021)
    bins=geometry.bins(2)
    max_partition=max_scale=max_reflection=0.
    for _ in range(200):
        axis=rng.randrange(3)
        h=10**rng.uniform(-3,2)
        center=[rng.uniform(-12,12)*h for _ in range(3)]
        center[axis]=rng.choice([-1,1])*rng.uniform(.1,15)*h
        total=geometry.face(center,axis,h)
        values=[geometry.face(center,axis,h,b) for b in bins]
        assert all(0<=v<=total+1e-12 for v in values)
        max_partition=max(max_partition,abs(math.fsum(values)-total))
        factor=10**rng.uniform(-2,2)
        scaled=[geometry.face([factor*x for x in center],axis,h*factor,b) for b in bins]
        max_scale=max(max_scale,max(abs(a-b) for a,b in zip(values,scaled)))
        opposite=geometry.face([-v for v in center],axis,h)
        max_reflection=max(max_reflection,abs(opposite-total))
    assert max_partition<2e-12
    assert max_scale<2e-12
    assert max_reflection<2e-12
    assert geometry.lib.ag_bin_count(9)==0
    assert geometry.lib.ag_make_bins(0,None,0)==-1
    assert math.isnan(geometry.face([0,0,0],0,.5))
    assert math.isnan(geometry.face([.5,0,0],3,.5))
    assert math.isnan(geometry.face([.5,0,0],0,-.5))
    assert geometry.face([0,2,0],0,.5)==0
    # Octant bins must give exactly zero on opposite-octant small faces.
    base=geometry.bins(0)
    zero_checks=0
    for b in base:
        mean=[sum(b.vertices[3*i+k] for i in range(3)) for k in range(3)]
        center=[-2*math.copysign(1,x) for x in mean]
        for axis in range(3):
            assert geometry.face(center,axis,.25,b)==0
            zero_checks+=1
    return dict(random_faces=200,maximum_random_partition_error_sr=max_partition,
        maximum_scale_invariance_error_sr=max_scale,
        maximum_central_reflection_error_sr=max_reflection,
        exact_empty_intersection_checks=zero_checks,invalid_input_checks_passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='ag_geometry_') as temporary:
        library=os.environ.get('ES_ANGULAR_LIBRARY')
        if not library:
            library=Path(temporary)/'angular_faces.so'
            subprocess.run(['cc','-std=c99','-O3','-Wall','-Wextra','-Werror','-fPIC','-shared',
                            str(folder/'angular_faces.c'),'-lm','-o',str(library)],check=True)
        geometry=Geometry(library)
        report=dict(method='Planar convex clipping by angular cone halfspaces; exact spherical polygon area.',
            levels=[identity_checks(geometry,level) for level in range(3)],
            additional=additional_checks(geometry),
            scope='Geometry and transparent fixed-graph identities only; opaque-mask shadow accuracy is tested by the transport solver.')
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
