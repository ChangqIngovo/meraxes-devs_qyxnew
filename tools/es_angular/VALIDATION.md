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

The measurements above concern the open-front geometry and fixed-front
chemistry. The previous mode-3 worker, native kernels, launchers and tests
have been removed. The current H/He chemistry resides in the local
`chemistry.py` module.

## Periodic snapshot integration

Mode 4 now connects periodic finite-speed photon inventory, H/He
recombination, density remapping and worker checkpoint/replay to the Meraxes
snapshot loop. See [PERIODIC.md](PERIODIC.md) for the explicit approximations.
The complete time split is first order; the old global time-sandwich bound
and fixed-front midpoint convergence do not apply to the complete new method.

Local tests cover periodic crossings, repeated images sharing gas capacity,
finite propagation through transparent periodic gas, retained source/angle/
energy/location inventory, release before recombination, and exact worker
checkpoint continuation. The H-only limiting comparison on a 5^3 grid at
time 7.3 reduces the total absolute ionized-count discrepancy from about 0.30
at 4 steps to 1.4e-7 at 32 steps. This is a consistency fixture, not a measured
general convergence order. Real HDF5 tests include three successive snapshots
with mixed H/He spectra and recombination, density matter-exchange accounting,
cumulative physical recombination counts, idempotent replay and configuration
rejection.

The final local combined suite passed **84 tests in 53.646 s** under Python
3.12, NumPy 2.5.4 and h5py 3.16.0. Numerical source hashes and the complete
test summary are recorded in
[`validation/production_integration_checks.json`](validation/production_integration_checks.json).

The new C bridge was syntax-compiled with strict warnings against the real
Meraxes structures and temporary declarations for unavailable external
libraries. This is not a linked MPI/HDF5 build. Production parameter guards
and source-event updates are exercised directly by `test_retired_modes.py`.
The Gadi preparation/PBS preflight was exercised on real HDF5 fixtures with a
stub executable, including file hashes, fresh-run checks and invalid-input
rejection. It does not substitute for the real Meraxes executable.

Full Meraxes MPI/HDF5 compilation and the actual Gadi L10 simulation have not
been performed in this environment. The [Gadi launcher](../../docs/angular_gadi.md)
performs the build and tests before permitting preparation/submission.

## Gadi chemistry failure and monochromatic endpoint correction

The supplied log for job `181886918.gadi-pbs`, running commit `22bb2292`,
confirms a working production MPI executable and successful completion of
snapshots 0–22. At snapshot 23 (z about 21.57), substep 27/34, the primary
photoionization kernel raised its 40000-iteration limit. This traceback is
the reported termination cause. No failing request/checkpoint was supplied,
so the exact snapshot state has not been replayed locally.

A synthetic reproduction with 64 cells, HI counts logarithmically spaced
from `1e-8` to `1e8`, and 18 eV credits equal to 1% of each cell's HI also
hits that limit. The common exposure step follows the largest active opacity;
cells with much smaller opacity make slow progress. The periodic worker now
uses the analytic fixed-target H endpoint described in [PERIODIC.md](PERIODIC.md)
for identical H-only cross sections. This fixture finishes in one update,
with captures equal to the supplied credits at the recorded precision.

For a separate 32^3 synthetic grid with 15 illuminated cells, the former
chemistry call took 1516 iterations and 6.236 s; the endpoint call took
0.000403 s. The maximum absolute difference in gas or captures was `1.01e-15`.
These are single local **chemistry-only** timings, excluding routing, ES FFTs,
recombination, I/O and galaxy evolution. They are not a Gadi simulation
speedup prediction. Mixed spectra retain the coupled iterative solver, now
restricted to illuminated cells; its general stiffness limit is not removed.

The updated combined suite passed **89 tests in 22.653 s**, including the
wide-density failure fixture, proportional debits from multiple sources and
periodic images, mixed-spectrum agreement with the full-grid kernel, and
real HDF5 numeric attributes of shape `(1,)` with deprecation warnings treated
as errors. Existing geometry, recombination, checkpoint and source-ledger
checks also pass. Results and source hashes are in
[`validation/monochromatic_endpoint_checks.json`](validation/monochromatic_endpoint_checks.json).
The corrected full L10 simulation still requires a fresh Gadi run.
