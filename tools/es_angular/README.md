# Angular-face photon allocation

This directory implements the source-resolved angular-face method developed in
the V20/V21 investigation: shared-face solid angles, one photon ledger per
source and angular bin, capacity events, and bounded excursion-set (ES)
modulation. It is an independently runnable C99 library with a NumPy interface.

The spatial solver currently implements **one ionization capacity per cell,
constant sources, and open boundaries**. The earlier Meraxes PC mode 3 and
its worker have been removed; nonzero legacy PC modes are rejected. The
separate [fixed-front H/He kernel](HHE.md)
implements shared spectral chemistry; periodic spatial continuation and its
checkpoint state are not connected to that kernel. These are material limits
of this implementation, not enabled production features.

## Build and run

From the repository root, with a C99 compiler and Python with NumPy:

```bash
python3 tools/es_angular/build.py
python3 tools/es_angular/run.py --write-example /tmp/angular_input.npz
python3 tools/es_angular/run.py \
  --input /tmp/angular_input.npz --output /tmp/angular_result.npz \
  --level 1 --times 0 1 4 12 --epsilon 0.1 --uniform-density 1
```

The build is explicit. Running the solver never invokes a compiler. Set
`ES_ANGULAR_LIBRARY=/absolute/path/libangular_front.so` or pass `--library` to
use another build. In an existing Meraxes CMake environment,
`-DBUILD_ES_ANGULAR=ON` adds the independent `es_angular_front` target. Its
default is OFF. The standalone build does not require MPI, FFTW, HDF5, or
Meraxes. The original Meraxes ES baseline remains available with
`Flag_PhotonConservation=0`; the angular solver is not selected by that flag.

The CLI accepts an NPZ containing:

| Array | Shape | Meaning |
|---|---|---|
| `capacity` | `(nx,ny,nz)` | Total ionizable atoms in each cell |
| `initial_mass` | grid shape | Initially ionized atoms, between zero and capacity |
| `source_indices` | `(sources,3)` | Integer cell-centre source coordinates |
| `source_rates` | `(sources,)` | Constant emitted photons per allocation-time unit |
| `initial_source_photons` | `(sources,)`, optional | Initial source history for the built-in ES score |
| `es_score` | grid shape, optional | Externally computed nonnegative ES score |

The initial ionized mass is counted once, even in overlapping bubbles.
`initial_source_photons` is score history, not an additional photon injection.
A neutral source cell must consume photons before it becomes transparent.
Zero-capacity cells are transparent.

`--level L` produces `8 * 4**L` spherical triangular bins. Sources are isotropic
at injection. `--dx` sets the cubic-cell spacing. Arbitrary nonnegative
capacity fields are supported. The analytic score additionally requires a
uniform medium; the CLI checks `capacity = uniform_density * dx**3`.
For a nonuniform ES model, provide its score explicitly. That score must be
computed from the intended ES filter/barrier; a generic density field is not
automatically an ES score.

The result NPZ contains output times, ionized mass, per-source absorbed mass
and rates, per-source/per-bin emitted rates, absorbed and escaped ledgers,
and JSON metadata. A readable JSON sidecar records diagnostics and hashes of
the input and native library. Outputs are snapshots, **not restart checkpoints**.
Time is the allocation variable associated with the supplied constant rates;
no finite-light-speed propagation is modeled.

## Geometry and allocation

For source `s`, angular bin `g`, and cell face `f`, the unattenuated flux is

\[
 \Phi_{sgf}=\frac{q_s}{4\pi}\,
       \left|\Omega_s(f)\cap B_g\right|.
\]

The face rectangle is clipped by the three cone halfspaces of the spherical
triangle. Its solid angle is integrated analytically after triangulation.
Each shared face is evaluated once per source and bin. Geometry is evaluated
in integer/half-integer grid coordinates: solid angles are scale invariant,
and this avoids introducing inconsistent shared corners through separately
rounded multiplications by a nonbinary `dx`.

The radial field has no flux through a cone's side walls. Consequently each
source-free cell has equal total incoming and outgoing unattenuated flux in
each bin. Outgoing edges point away from the source along grid axes, forming
a DAG ordered by Manhattan distance. A transparent node divides its incoming
flux in proportion to its outgoing geometric face fluxes. A cell remains
opaque until its remaining capacity is paid. Shared gas capacity is consumed
by the sum of all sources and bins.

For frozen rates `b_sgj`, the next capacity event occurs at

\[
 \Delta t=\min_{j:\sum_{sg}b_{sgj}>0}
  \frac{C_j-u_j}{\sum_{sg}b_{sgj}}.
\]

All cells and source ledgers advance together to that event, then routing is
recomputed. Output requests only truncate a step; they do not rescore ES.
Roundoff corrections at full-capacity events are exposed in the diagnostics.
The outer-box flux is recorded as escaped photons and is never reassigned to
an interior target.

This construction uses fixed face adjacency and cone geometry. It does not
trace individual source-to-target rays or record their propagation paths.
It still contains a spatial visibility model: it is an ES-guided allocation
method, not a purely local ES threshold criterion.

## Bounded ES modulation

Let `a_sgj` be the first-opaque-cell absorption rates of the geometric graph,
and let `A_sg = sum_j a_sgj`. Set

\[
 \eta=\frac{\epsilon}{2+\epsilon},\qquad
 w_j=1+\eta\frac{S_j-1}{S_j+1},\qquad
 b_{sgj}=a_{sgj}w_j
 \frac{A_{sg}}{\sum_k a_{sgk}w_k}.
\]

Zero-absorption resources remain unchanged. This normalization preserves the
absorbed share of every source-bin resource; the escaped share is also
unchanged. It cannot invent a target with zero geometric support. For a
nonzero supported rate,

\[
 \frac{1}{1+\epsilon}\le\frac{b_{sgj}}{a_{sgj}}\le1+\epsilon.
\]

The built-in uniform-medium, target-centred top-hat score is

\[
 S_j=\max_i
 \frac{\sum_{s:\,r_{sj}\le r_{ij}}(Q_{s,0}+q_s t)}
      {(4\pi/3)n\,r_{ij}^3}.
\]

Equal source distances are grouped before evaluating the enclosed count.
The zero-radius point-source limit has saturated score. Weights are frozen
between capacity events. `set_es_scores()` is an explicit model-control
update, separate from output scheduling.

For the fixed nonnegative graph, constant rates, irreversible filling, and
the same initial state, the previously derived comparison applies:

\[
 u_0\!\left(\frac{t}{1+\epsilon}\right)
 \le u_{\rm ES}(t)\le u_0((1+\epsilon)t).
\]

With no escape, the added mass error satisfies
`||u_ES-u_0||_1 / (Qdot*t) <= 2*epsilon/(1+epsilon)`.
This bounds ES modulation relative to the **same discrete geometric model**.
It does not bound the graph's angular mixing, voxel geometry, or physical
radiative-transfer error. No global version is claimed for the separate
recombining H/He system.

## Approximation and the overlapping-bubble case

For two transparent spheres of radius `R` with separation `d`, the directions
from the first centre that stay in the union and exit through the second
sphere satisfy `cos(theta) >= d/(2R)`. Their fraction is

\[
 f=\frac{1-d/(2R)}{2};\qquad d=1.5R\implies f=0.125.
\]

This is not the fraction of all rays that intersect the second sphere after
possibly crossing a neutral gap. The algorithm stops photons at the first
opaque cell, so those latter directions cannot bypass the gap.

The frozen-bubble benchmark measures the corresponding crossing of the
transparent neck. Its total depends on spatial representation of the neck.
Angular refinement improves redistribution beyond an obstruction, but cannot
repair the voxelized neck area. Even with exact unattenuated face fluxes,
mixing within a finite angular bin broadens shadows. Space and angle must be
refined separately. Results and reproducible commands are in
[VALIDATION.md](VALIDATION.md).

The symmetric two-source contact benchmark also has an independently derived
axial filling clock. It agrees with the graph events and converges with
`O(dx)` error; voxel-edge rounding explains its nonmonotonic coarse-grid
sequence. The [derivation](axial_contact_theory.txt) separates this special
case from a general three-dimensional topology claim.

## Tests

```bash
python3 tools/es_angular/test_angular_front.py
python3 tools/es_angular/test_angular_cli.py
python3 tools/es_angular/test_angular_faces.py
python3 tools/es_angular/test_angular_hhe.py
```

The first two tests use `ES_ANGULAR_LIBRARY` if set; otherwise their test
harness explicitly compiles into a temporary directory. The geometry test
also builds its own temporary library. The H/He tests use the local NumPy
chemistry module. No production snapshot or Gadi job is started by these commands.

Further integration must retain a photon's current target/image and angular
identity when it passes a newly paid cell. Periodic boundary credits need
conservative continuation into adjacent images, and recombination must use
the photons' current local inventory. Those states must participate in
checkpoint/replay. Aggregating unused photons back to their original source
would change spatial allocation and is not a valid implementation shortcut.
