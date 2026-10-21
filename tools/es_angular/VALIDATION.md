# V21 validation

These results validate the implemented discrete method and distinguish its
spatial, angular and ES-modulation errors. Raw JSON is in `validation/`.
All quantities below use the dimensionless units defined by `benchmarks.py`.
The scripts regenerate field NPZ files; those reproducible binary fields are
not stored in the repository.

## Overlapping transparent bubbles

Two centre-selected spheres have `R=1`, separation `1.5`, and one isotropic
source of rate `0.5` at each centre. The crossing fraction is normalized by
the corresponding source rate. Here `h=3/N`; the grid has shape
`(N + separation/h, N, N)`, so `N` is the transverse resolution.

| N | Measured crossing fraction | Exact continuous-sphere fraction |
|---:|---:|---:|
| 32 | 11.1334247% | 12.5% |
| 48 | 11.7692136% | 12.5% |
| 64 | 11.9373257% | 12.5% |
| 96 | 12.1478570% | 12.5% |

Both sources agree. At N=32 and 48, 8/32/128 angular bins give the same total
within roundoff. This is expected: the unobstructed upstream neck flux is
fixed by the voxelized cross-section, while the angular partition sums back
to the same exact face flux. Angular refinement changes the downstream
distribution, not the neck area. The N=96 residual is **0.352143 percentage
points**, or about 2.82% relative to the 12.5% reference.

With separation `2.4`, the measured crossing is exactly zero at all three
angular resolutions: photons cannot jump the intervening neutral cells.

## Obstructed visibility

The opaque-screen fixture places a one-cell-thick rectangular screen inside
an otherwise transparent radius-1.2 sphere. The shadow diagnostic includes
only outer target cells whose entire voxel is geometrically behind the
screen, with an additional 20% margin from its projected edge. The exact
direct flux in that set is zero.

At N=48, `h=3.2/48`, screen front `x=0.3666667`, and halfwidth `0.2333333`:

| Angular bins | Photon fraction incorrectly reaching the shadow core |
|---:|---:|
| 8 | 0.8953351% |
| 32 | 0.4874280% |
| 128 | 0.1065062% |

Fractions are relative to the source's total emitted rate, not to the screen
interception rate. The screen itself absorbs 9.30685284%, agreeing with its
independent rectangular solid angle to better than `1e-15` in absolute
source fraction. Residual shadow illumination is angular-bin mixing; exact
unattenuated geometry alone does not make masked visibility exact.

## Evolving fronts and the contact-time discrepancy

For a single source with rate 1, initial radius 0.6, `h=4.5/N`, and final
allocation time 4, the ionized-mass L1 difference from an equal-total-volume
continuous sphere is:

| N | L1 / total ionized mass |
|---:|---:|
| 24 | 7.2794653% |
| 30 | 5.5741121% |
| 36 | 4.2694341% |

The reference uses independent order-10 tensor Gauss quadrature for boundary
voxels. This is a total discrete-versus-sphere discrepancy, including the
voxelized initial condition and reference quadrature. It is not an isolated
estimate of the later transport error. Angular refinement leaves this radial
fixture unchanged within roundoff. The old directional-upwind baseline had
29.5% discrepancy at N=36 under the same comparison.

For equal sources of rate `q=0.5`, separation `d=1.5`, and initial radius
`R0=0.6`, an **independent discrete axial clock** explains the strongly
nonmonotonic coarse-grid contact times. Let `m=d/(2h)` and let `k0` be the
last initially paid axial cell. Then

\[
 T_h=\sum_{k=k_0+1}^{m-1}\frac{h^3}{qf_k}
       +\frac{h^3}{2qf_m},\qquad
 f_k=\frac1\pi\arctan\frac{1/4}
 {(k-1/2)\sqrt{(k-1/2)^2+1/2}}.
\]

The midpoint receives both sources. Axial cells have no incoming transverse
edges, making this a separately solvable chain. With zero initial-mask
tolerance, expansion gives

\[
 T_h=T_{\rm cont}+\frac{4\pi}{q}h
 \left[R_0^2\operatorname{frac}(R_0/h)-\frac{(d/2)^2}{2}\right]+O(h^2),
 \qquad T_{\rm cont}=1.7247343668.
\]

Thus this event has first-order spatial consistency with an oscillating
leading error. The implementation's squared-radius mask tolerance `1e-14`
adds a negligible fixed radius perturbation; it should not be mistaken for
a new physical scale in the asymptotic statement.

The graph gives contact times 1.035368, 0.869414, 1.754841, 1.465136,
1.248486 and 1.639824 at N=24, 30, 36, 48, 60 and 72. The independent axial
formula matches these measurements within `3e-15`. Its analytic refinement
sequence extends to N=1536. The additional N=96 graph run gives **1.733709376**,
matching the axial formula within `1.7e-14`; the equal-initial-volume spherical
reference there is 1.735349169. Its maximum per-resource photon ledger residual
is `1.39e-16`. See [the derivation](axial_contact_theory.txt)
and `validation/axial_contact.json` for the full sequence and any additional
graph measurements. Agreement between the axial clock and the earliest
three-dimensional six-connected path is a measured property of these
symmetric fixtures, not a general topology theorem.

## ES modulation and spectral chemistry

For the N=24 two-source fixture with 32 bins and epsilon=0.1, the componentwise
time sandwich has zero measured violation at times 0.5, 1, 2 and 4. At time 4,
the added mass L1 difference divided by newly emitted photons is 0.133753%,
below the 18.1818% analytical bound. This is the additional ES error relative
to the same angular graph; it is separate from the spatial errors above.

The fixed-front H/He kernel passes all 17 tests, including HI competition,
HeI-to-HeIII double ionization, photon floors, earliest transparency events,
and preservation of each target and boundary credit. In the three-band
midpoint refinement test, consecutive gas L1 differences for donor fractions
0.08/0.04/0.02/0.01 are `8.93e-6`, `2.20e-6`, `5.45e-7`, with ratios
4.065 and 4.032. Photon/capture residuals are below `4.8e-15`. This demonstrates
second-order refinement for that chemistry fixture, not convergence of an
as-yet unconnected periodic H/He transport solver.

## Implementation checks and reproduction

The implementation passed 13 core tests, 5 CLI tests and 17 fixed-front H/He
tests. The 13 core tests also passed an AddressSanitizer/UndefinedBehaviorSanitizer
build. Strict C99 compilation passed with `-Wall -Wextra -Werror -pedantic`.
The geometry suite independently checked 8/32/128-bin partitions, per-cell
divergence, per-face free propagation, 200 random faces, and empty/invalid
inputs. Maximum free-propagation relative face error was `5.56e-16`.

The retirement guard test also passed: it compiles the actual
`check_problem_params` function from Meraxes in a minimal C harness and
checks that mode 0 is accepted while seven nonzero values are explicitly
rejected. This verifies the parameter guard without claiming a full MPI run.

Regressions specifically cover a diagnostic time one ULP before a capacity
event, neutral source cells, zero-budget sources in the ES score, extreme
finite weights, mixed absorption/escape, and closed Python handles.
Changing output schedules agrees within floating-point error.

```bash
python3 tools/es_angular/build.py
python3 tools/es_angular/test_retired_modes.py
python3 tools/es_angular/benchmarks.py all --output-dir /tmp/angular_validation
python3 tools/es_angular/contact_refinement.py --output-dir /tmp/angular_validation
python3 tools/es_angular/axial_contact.py --output-dir /tmp/angular_validation
python3 tools/es_angular/test_angular_faces.py --output /tmp/angular_validation/geometry.json
python3 tools/es_angular/angular_hhe_convergence.py --output /tmp/angular_validation/hhe.json
```

Full Meraxes CMake/MPI compilation and Gadi execution were not performed in
this environment because CMake/MPI are absent. The previous mode-3 worker,
native kernels, launchers and tests have been removed. The new C99 core and
Python tools were compiled and exercised directly. Its shared H/He chemistry
now resides in the local `chemistry.py` module. Periodic H/He continuation,
recombination-aware photon locations and restart serialization remain
necessary before connecting the angular method to production snapshots.
