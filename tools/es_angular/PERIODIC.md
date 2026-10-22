# Periodic angular-face H/He snapshots

`Flag_PhotonConservation=4` invokes `src/core/photon_angular_meraxes.c` from
the normal Meraxes reionization call. Meraxes evolves its galaxies and builds
the density and escaped stellar SFR grids with MPI. Rank zero runs
`snapshot_worker.py`, then scatters the returned HI and recombination fields.
The radiation worker itself uses one process. The old PC allocation backends
are not used. Mode 0 remains the original ES comparison.

This is a simulation implementation of an approximate closure, not a claim
that its cosmological predictions have already converged. The Gadi L10 run
is the first full simulation validation. See [the launch guide](../../docs/angular_gadi.md).

## Photon positions and periodic gas

Each primary photon inventory row records its emission cell, angular bin,
energy group, unwrapped offset, remaining edge distance and photon count.
Physical absorption uses `(emission_cell + offset) mod grid_shape`. All images
of a physical cell share the same five species; images do not create capacity.
Previously emitted photons retain their birth cell when later sources move.

Outgoing probabilities use the existing C99 shared-face solid angles without
changing the geometry. A radial outward edge has travel distance

\[
\Delta r=\lVert\boldsymbol k_{\rm next}\rVert_2-
         \lVert\boldsymbol k\rVert_2>0.
\]

These distances telescope to a source-centred radial distance. The per-step
travel allowance is `c * AngularSpeedFraction * dt / proper_cell_size`.
This is a radial light-cone approximation for the face graph; it is not a
straight-ray transport solution. Finite travel time prevents a fully
transparent periodic box from recirculating forever in one call. There is no
escape sink on the torus. Photon redshifting across energy groups is not
modeled; energies are fixed group labels.

A packet stops at its first opaque target or its travel horizon. Only the
opaque-target shares reached from that *current* packet location receive
bounded ES redistribution. In-flight shares are unchanged. Leftovers are
never returned to the original source or combined across different current
locations for redistribution. Identical-energy arrivals at one physical cell
may be combined for chemistry, then debited proportionally back to their
individual resource ledgers.

After photoionization makes a target transparent, its unspent light is split
onto outgoing edges **before** the following recombination update. Further
travel begins in the next substep. Otherwise repeated local recombination
could artificially trap light at a source that had already cleared its cell.
No positive inventory is discarded by an arbitrary photon floor. Resource
limits fail explicitly rather than silently deleting photons.

## Chemistry, sources and time

The gas vector is `(HI,HII,HeI,HeII,HeIII)`. Primary photons compete at fixed
target cells, stopping at class-specific transparency thresholds.
Photons of 13.6–24.59 eV can spend HI capacity, 24.59–54.4 eV can spend HI+HeI,
and harder photons can spend HI+2HeI+HeII. Multiple sources share those same
populations.

For identical H-only cross sections, including the default 18 eV source,
the photoionization endpoint in an initially opaque cell is evaluated directly:

\[
A_j=\min\!\left(P_j,\max[0,N_{{\rm HI},j}-\varepsilon_{\rm tr}N_{{\rm H},j}]\right),
\qquad D_{a,j}=A_j\frac{P_{a,j}}{P_j}.
\]

Here `P` contains only photons that have reached that target, `a` labels
identical-cross-section arrivals, and `epsilon_tr` is the transparency
threshold, distinct from the ES modulation parameter. All these arrivals
have the same fractional depletion, so their proportional debits are the
analytic endpoint of the fixed-target reaction. HI decreases and HII increases
by `A`; helium is unchanged. Unspent photons retain their resource identities
and are released onto outgoing edges by the usual rule. Subnormal credits
that fail the existing representability guard remain in inventory.

This removes a global exposure-step stiffness failure in the monochromatic
case; it does not change the physical travel clock or the first-order time
split. Distinct H cross sections or He-active light still use the coupled
spectral kernel and its transparency events. That solve packs only cells
with arriving photons; dark cells cannot constrain or participate in the
reaction. The general spectral iteration limit remains in place.

`recombination.py` uses case-A H/He recombination and an eight-channel local
on-the-spot (OTS) closure. Recombination captures and diffuse photon emission
are separate counts: a helium capture can produce more than one ionizing
photon. Unabsorbed diffuse light stays at its physical cell and in its channel.
This does not transport diffuse light between cells. Temperature is fixed at
10,000 K; UVB intensity, heating, AGN/Pop III spectra, and radiative feedback
are not supplied by this closure. Unsupported flags are rejected by the
Meraxes parameter guard. Legacy `Gamma12` and `r_bubble` outputs are zero
placeholders, not physical estimates.

Each snapshot uses its right-endpoint density and source field. The start
snapshot initializes neutral gas with zero elapsed time. Later intervals are
split by `AngularMaxStepMyr` (default 0.1 Myr). Every substep performs half a
local recombination update, injects its source share, transports and absorbs
primary photons, and performs the second recombination half-step. Pulsed
emission and deferred continuation make the full scheme **first order in
time**. The fixed-front midpoint chemistry tests do not establish second-order
convergence for the whole simulation. Refine `AngularMaxStepMyr` to assess this
additional error; angular resolution and spatial resolution remain separate.

When snapshot density changes, species fractions are preserved where previous
gas existed; newly appearing gas is neutral. The five-species change is
recorded as matter exchange. Photons are neither scaled nor charged for this
prescribed density remapping. Counts use one fixed mean-H-atoms-per-cell
normalization, with the correspondingly scaled physical volume in chemistry.

The source count is escaped instantaneous SFR times the physical interval and
ionizing yield per stellar baryon. The mode-4 stellar-event update uses only
`new_stars / gal->dt`, avoiding duplicate additions of the accumulated SFR
when one galaxy has several formation events in a snapshot.

## ES prior and scope of bounds

`es_scores.py` computes the periodic discrete spherical top-hat prior

\[
S_j=\max_R\frac{(K_R*Q_{\star,\mathrm{history}})_j}
                    {(K_R*N_H)_j}.
\]

The radii include the local cell, powers of two cells, and half the shortest
box dimension. Every physical cell enters a kernel at most once. This is an
H-based source-history prior, not a spectral H/He barrier or a photon budget
that is spent a second time. The bounded modulation controls allocation only.
Vacuum with zero history gets score zero; positive history over zero H
saturates the finite score. Near-zero FFT results are checked with direct
nonnegative sums to avoid manufacturing support from FFT noise.

Per-cohort geometric support and total count are preserved. The original
irreversible fixed-graph global time-sandwich theorem is **not** asserted for
this periodic, recombining, finite-speed evolution.

## Snapshot contract and audits

The request and response use HDF5 schema 2 and solver identity
`angular-face-periodic-hhe-v1`. The request supplies flattened density and
source-photon grids, physical time/volume/atom normalization, source spectrum,
geometry controls and a run identity. Responses contain HI and all He
fractions, cumulative physical HII recombinations and audit information.
The auxiliary `angular_state_XXX.hdf5` retains the full H/He response even
though the legacy Meraxes grid interface exposes only HI. Optical depth uses
the actual H/He electron count.

The worker keeps stellar, diffuse, absorption, recombination and matter
exchange ledgers. The accounting identities are

\[
Q_\star+Q_{\rm diffuse,emitted}
=Q_{\rm primary,absorbed}+Q_{\rm diffuse,absorbed}
 +Q_{\rm primary,pending}+Q_{\rm diffuse,pending},
\]

\[
\Delta N_e=Q_{\rm primary,absorbed}+Q_{\rm diffuse,absorbed}
 -N_{\rm recombination\ captures}+\Delta N_{e,\rm matter}.
\]

Hydrogen and helium nuclei are conserved after the recorded matter exchange.
The primary engine also retains separate source/bin/energy accounting.
Small floating-point residuals are audited; residual tolerance is not an
extra sink or an excuse to drop small packets.

The checkpoint preserves gas, resource ledgers, unwrapped positions, remaining
flight distances, local diffuse light and source history. It validates
consecutive snapshot numbers, input digests and run configuration. Retrying
the same worker request is supported; silently applying it twice is not.
This **does not implement a full Meraxes galaxy/MPI restart**. The C bridge
therefore requires a fresh output directory and rejects an existing angular
checkpoint at initialization.
