# Time-dependent H/He photon transport

`Flag_PhotonConservation=2` selects the native C H/He transport model. It evolves
the five species HI, HII, HeI, HeII and HeIII, gas thermal energy, and a persistent
population of photon packets. Finite-speed propagation, simultaneous absorption,
recombination emission and subsequent absorption form one time-dependent loop.
The model runs on the existing MPI mesh slabs; crossing packets migrate to the
rank that owns the next slab.

| Mode | Ionization calculation |
|---|---|
| `Flag_PhotonConservation=0` | Existing excursion-set calculation. |
| `Flag_PhotonConservation=1` | Existing ES allocation and maximum-flow completion. |
| `Flag_PhotonConservation=2` | H/He packet transport and thermal chemistry. |

Mode 2 does not use the maximum-flow solver or infer photon trajectories from
its edges. The earlier ES hierarchy acceleration experiment is not connected to
this implementation. The runtime therefore depends on emitted packets, cell
crossings, resonance events and chemistry steps, rather than a few FFTs.

## Physical model

Continuous-spectrum packets move at the physical speed of light between
interactions. HI, HeI and HeII compete for the same photons through their summed
opacity. Each absorbed photon produces one primary photoionization; the excess
energy above the appropriate ionization threshold becomes local thermal energy.
All incident packets in a cell participate in a common chemistry update.

Downward H/He recombination rates produce new continuum, resonance-line and
two-photon radiation. These photons are born at the end of the time step and
can be absorbed in later steps. The explicit recombination calculation does not
subtract the legacy cumulative `Nrec` capacity a second time. Helium
recombination photons can ionize hydrogen.

The thermal state satisfies

\[
u=\frac{3}{2} k_{\rm B}T(n_{\rm H}+n_{\rm He}+n_e).
\]

It includes photoheating, captured-electron recombination cooling, H/He
collisional ionization, HI collisional excitation and free-free cooling.
The atomic coefficients have a declared domain of **5000--20000 K**. Crossing
either limit is an error; the code does not replace the temperature by a floor
or ceiling.

The following approximations define this model:

- The gas density is held fixed during each snapshot interval. There is no
  hydrodynamic advection or cosmological radiation-redshift calculation.
- Photoelectrons thermalize locally. Secondary ionizations and nonthermal
  excitation are omitted; absorbed excess energy above 10.2 eV is diagnosed.
- HeI 21.2 eV and HeII 40.8 eV resonance photons have separate tags. Elastic
  scattering is monochromatic, coherent and isotropic, with no Doppler
  redistribution or recoil heating. HeI also has its specified conversion
  branch into lower-energy radiation.
- Continuum and two-photon spectra use a fixed energy grid, positive
  count-and-energy-preserving projection, and grey groups within absorber
  threshold bands. Continuum photons at a resonance energy do not acquire a
  line tag.
- Unresolved low-energy cascade radiation is assigned to a specified
  noninteracting radiation channel. Its energy remains in the ledger; this
  does not mean that it has instantly crossed the simulation boundary.

These choices define a warm-gas approximation. In particular, initializing a
neutral cell at 10000 K is not a prediction for the cold pre-reionization IGM.
The native coefficients are generated from the retained research reference in
`tools/hhe_reference/`; the generated header records its source fingerprint.

## Sources and snapshot history

The adapter uses the deposited, escape-fraction-weighted instantaneous SFR grid
to obtain an **absolute photon emission rate** in photons per second, with
`ReionNionPhotPerBary` setting its normalization. Escape fractions already
contained in that grid are not applied again. It does not differentiate the
cumulative stellar-mass grid, whose spatial changes can include galaxy motion.

`HHeStartSnapshot` creates neutral H and He at `HHeInitialTemperature` and an
empty radiation population. It does not reconstruct photons or ionization
before that starting snapshot, or advance a source interval at initialization.
Subsequent snapshots must arrive in sequence.
The starting snapshot must be inside the actual run's snapshot range; selecting
the final snapshot produces an initialization-only result.
The adapter retains the species, thermal state and in-flight packets in memory
between snapshots. To advance from snapshot `i-1` to `i`, it holds snapshot
`i`'s supplied SFR constant over that preceding interval. This right-endpoint
source approximation is recorded in the output.

When the supplied density or physical box size changes, an explicitly external
remap preserves each cell's ion fractions and temperature, rescales packet
positions, and records the resulting matter and gas-energy exchange. Packet
counts and energies are unchanged by this remap. There is no implicit
cosmological redshift, expansion cooling or hydrodynamic transport. These
external exchanges must be included when auditing more than one snapshot.
Vacuum cells contain no thermal gas and are skipped by chemistry; their stored
`Temperature_K=0` is an undefined-temperature sentinel, not a cold-gas solution.
If an externally prescribed remap fills a previously empty cell, it creates
neutral gas at the declared initial temperature and records the added matter
and energy.

The default source energy is 18 eV. Such primary photons ionize HI but cannot
photoionize HeI or HeII. A source spectrum is therefore needed to study stellar
helium photoionization. Its optional ASCII file has two columns,
`energy_eV photon_number_weight`; the nonnegative weights are normalized to
unit sum and describe photon number, not energy or luminosity. This
illustrative two-bin spectrum is not a calibrated
stellar population:

```text
# energy_eV  photon_number_fraction
18.0         0.7
35.0         0.3
```

Set `HHeSourceSpectrumFile` to its path. Without that parameter, the single
`HHeSourceEnergyEV` value is used. A hard spectrum can heat the gas beyond the
allowed temperature domain; the solver reports this instead of clipping it.

## Configuration

Copy the overrides in
[`input/params/photon_hhe_transport.par`](../input/params/photon_hhe_transport.par)
into a complete simulation parameter file. Keep the existing simulation and
input-table paths. This fragment does not replace a full `input.par`.

| Parameter | Default | Meaning |
|---|---:|---|
| `Flag_PhotonConservation` | `0` | Set to `2` to select this model. |
| `HHeInitialTemperature` | `10000` | Initial neutral-gas temperature in K. |
| `HHeSourceEnergyEV` | `18` | Monochromatic primary energy when no spectrum file is supplied. |
| `HHeSourceSpectrumFile` | empty | Optional two-column photon-number spectrum. |
| `HHeStartSnapshot` | `0` | Snapshot at which the neutral, empty-radiation state is created. |
| `HHeEmissionDirections` | `12` | Equal-weight emission directions per emitting cell and energy group. |
| `HHeCellCrossingFraction` | `0.25` | Maximum step as a fraction of a cell light-crossing time. |
| `HHeMaxStepMyr` | `0.1` | Additional upper bound on a transport step in Myr. |
| `HHeMemoryMB` | `2048` | Per-rank persistent H/He state and transport memory guard in MB. |
| `HHeBoundaryPeriodic` | `1` | `1`: periodic box; `0`: open boundaries with recorded photon escape. |

The transport step satisfies both the cell-crossing and time limits. Local
chemistry also substeps according to material and thermal changes. Refining
time steps, angles, spatial resolution and energy bins addresses distinct
numerical errors; conservation alone does not measure these errors.

Each step emits the primary interval budget, drifts for half a step, applies
shared-cell chemistry for the full step, drifts for the remaining half, then
creates diffuse endpoint packets. Atomic temperature-dependent coefficients
are held at the chemistry call's initial temperature; thermal energy and
species still evolve during its substeps. Primary start-time injection and
diffuse endpoint injection make time refinement necessary even with the two
half drifts.

Absorption is an operator-split approximation: the packet reacts for the full
chemistry step in the cell it occupies after the first half drift. The code
does **not** integrate continuum absorption along each crossed path segment.
A packet that leaves an open boundary during the first half drift has no
chemistry update in that step. Decreasing `HHeCellCrossingFraction` and
`HHeMaxStepMyr` tests this cell-location and boundary error. The pair of half
drifts does not establish second-order accuracy for the full emission and
chemistry scheme.

Mode 2 requires `Flag_PatchyReion=1`, `Flag_IncludeRecombinations=1`, and a build
with `USE_MINI_HALOS=OFF`.
`FlagInteractive` and `FlagMCMC` must both be zero: repeated-run wrappers do not
yet reset and reconstruct the retained native radiation state.
UV-background feedback, the reionization modifier, CGM escape-fraction
suppression and spin-temperature evolution are currently incompatible. Set
`ReionUVBFlag=0`, `Flag_ReionizationModifier=0`,
`Flag_FescCGMSuppression=0` and `Flag_IncludeSpinTemp=0`.
`EscapeFracBHNorm` must be zero because AGN ionizing sources are not yet
connected to this transport adapter. `Flag_BHFeedback` can remain enabled:
galaxy black-hole feedback and escaped AGN photon emission are separate inputs.
The adapter computes a local HI photoionization-rate diagnostic from the
in-flight packet population. It does not infer a `J_21` field or activate
UV-background feedback from that diagnostic.

The mode-1 `PhotonConservationMaxEdges`, shape, cut and flow-iteration settings
do not control mode 2. The H/He memory allowance first reserves persistent gas
and source arrays plus one per-cell output workspace; the remaining allowance
guards transport allocations.
Allocation failure is reported rather than deleting faint packets. Resonance
scattering has no event-count cap. MPI exchange buffers, HDF5 internals
and the other Meraxes allocations require additional memory. This
parameter is not a scheduler memory request or a bound on total process memory.
All surviving photon cohorts remain in memory: there is no packet merging or
small-count pruning. Full `512^3` memory usage and runtime have not been
validated, and neither the default guard nor the MPI decomposition establishes
production-scale feasibility.

## Build and run

The native solver uses the normal Meraxes C/MPI build and the committed atomic
table; it does not require Python during a simulation. With the usual Meraxes
dependencies available, create a CPU build:

```bash
cmake -S . -B build-hhe \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DUSE_CUDA=OFF -DUSE_MINI_HALOS=OFF -DUSE_STOCHASTICITY=ON \
  -DBUILD_TESTS=ON
cmake --build build-hhe -j 8
```

Run the resulting `build-hhe/bin/meraxes` with the complete parameter file from
the run directory using the site's MPI launcher. Keep the scheduler allocation
and MPI rank count consistent. A new transport state is created when the
process starts; saved diagnostic files are not yet read as a disk restart.
Start with a small grid and inspect the photon, energy and temperature
diagnostics before a larger run.

Regenerating the atomic header requires NumPy and SciPy:

```bash
python3 tools/generate_hhe_atomic_table.py
python3 tools/generate_hhe_atomic_table.py --check
```

The generator retains the coefficient/emission energy identity during
temperature interpolation. Changing the retained atomic reference, quadrature
or energy grid requires regenerating the header and repeating the native
kernel verification.

## Parallel execution and conservation

Each rank owns its gas cells and local photon packets. A drift follows cell
faces until the packet exhausts its travel time, leaves an open boundary or
enters another rank's slab. MPI exchanges the full packet state, including the
remaining flight time and residual scattering optical depth. Chemistry occurs
after all ranks finish that drift phase. Gas grids and the complete packet
population are not assembled on rank zero.

Primary and diffuse photons have separate emitted, absorbed, converted,
escaped and live counts. Conversion and recombination can change the number
of ionizing photons, so the photon ledger includes their actual creation and
destruction terms. The total-energy audit includes gas thermal energy, atomic
ionization energy, active packets, escaped radiation, inactive radiation and
the prescribed external remap exchange. Hydrogen and helium nuclei are
conserved by local chemistry; their externally imposed changes are recorded
separately.

## Output

Every active H/He snapshot writes one file per MPI rank, independently of the
standard selected galaxy snapshots:

```text
<OutputDir>/hhe_state_000.rank00000.hdf5
<OutputDir>/hhe_state_000.rank00001.hdf5
...
```

The flat HDF5 schema has `SchemaVersion=1`. Its principal datasets are:

| Datasets | Contents |
|---|---|
| `nHI_cm-3`, `nHII_cm-3`, `nHeI_cm-3`, `nHeII_cm-3`, `nHeIII_cm-3` | Local three-dimensional slab, physical number densities. |
| `ThermalEnergy_eV_cm-3`, `Temperature_K` | Local thermal state. |
| `Packets` | Compound array containing positions, directions, counts, energy, clocks, tags, random state and cell ownership. |
| `GroupEnergy_eV`, `GroupKind`, `SourcePhotonFraction` | Energy grid, continuum/line tags, and normalized primary spectrum. |
| `LivePhotons`, `EmittedPhotons`, `EscapedPhotons`, `AbsorbedPhotons`, `ConvertedPhotons` | Local photon ledgers, shape `(2, 31)`; origin index 0 is primary and 1 is diffuse. |
| `RecombinationEvents`, `CollisionalIonizationEvents` | Local integrated event counts for the three H/He transitions. |
| `HIIRecombinationEvents`, `HeIIRecombinationEvents`, `HeIIIRecombinationEvents` | Per-cell cumulative recombination events. |
| `Photoheat_eV`, `HighEnergyPhotoheat_eV`, `CaptureCooling_eV`, `CollisionalIonizationCooling_eV`, `ExtraRadiativeCooling_eV` | Local integrated thermal-process ledger. |
| `LocalSpeciesExchange`, `LocalMatterExchange_eV`, `LocalInactiveRadiation_eV` | External species exchange, external gas-energy exchange, and decoupled radiation energy. |
| `GlobalEnergyComponents_eV`, `GlobalEnergyResidual_eV` | Global energy audit, repeated identically in each rank file. |
| `GlobalRelativeEnergyResidual`, `GlobalPhotonBudgetResidual` | Global normalized energy residual and photon residual by origin and group, repeated in each rank file. |
| `SlabXStart`, `SlabXCount`, `GridDimensions`, `BoxPhysical_cm` | Mesh ownership and physical geometry. |

Sum local budgets across ranks to audit a whole box: packets migrate, so a
single rank's births and losses need not balance its surviving population.
Read the **global** audit datasets from one rank only. The energy-component
order is stored in the `Order` attribute, and the balance is

\[
E_{\rm initial}+E_{\rm external\ matter}+E_{\rm primary}
=E_{\rm gas}+E_{\rm live}+E_{\rm escaped}+E_{\rm inactive}.
\]

The log reports the step count and step duration at the start of each advanced
snapshot, progress approximately every 5% of its steps, and the snapshot's
packet/scattering totals and global energy residual at completion.

Standard Meraxes grids receive `xH`, the gas temperature and the local
photoionization-rate diagnostic. `N_rec` is a diagnostic of gross HII
recombinations per current H+He nucleus, not a capacity consumed by the new
solver. No ES filter radius is inferred from the transport solution.

Within the simulated interval, the Thomson-optical-depth calculation uses
the actual number-weighted electron abundance,
`sum(ne)/sum(nH+nHe)`, including the additional electron from HeIII. The
reported total still includes Meraxes' existing post-reionization extrapolation
with ionized hydrogen and its helium transition at redshift 4. That low-redshift
contribution is an imposed continuation, not a prediction of this warm-gas
transport run.

The files retain the complete local species, thermal state and photon
population for independent audits. `DiskRestartSupported=0` records that no
reader currently restores this state into a new process.

## Verification entry points

```bash
ctest --test-dir build-hhe -R photon_hhe --output-on-failure
python3 tools/test_photon_hhe_reference.py
python3 tools/generate_hhe_atomic_table.py --check
```

The native kernel tests independently recompute nuclei, reaction charge,
photons and total energy; they also cover competing absorber/conversion
hazards, spectrum projection, rejected-input rollback and prescribed snapshot
remaps. Native transport tests cover periodic flight, open-boundary escape,
packet state across slab handoffs, persistent state and recombination recycling
after source switch-off. The Python comparison checks native reaction results
against the retained research kernel and requires NumPy/SciPy plus a C
compiler. These focused tests do not substitute for a cosmological simulation
or resolution and time-step comparisons of its output.

## Integration validation (2026-10-18)

The CPU Meraxes executable builds in Debug and Release with stochastic sources
compiled in. The native chemistry and transport tests pass under address and
undefined-behavior instrumentation (LeakSanitizer was disabled because the
validation container cannot inspect process threads). The Release CTest suite
covers the old ES/allocation tests and the new H/He tests: 20 selected tests pass.
Real MPI tests compare 1, 2 and 4 ranks with a serial reference, including
resonance RNG/optical-depth state, shared chemistry and empty slabs. The adapter
fixture also passes with 3 ranks, exercising an empty slab in per-rank HDF5.

The adapter fixture drives three prescribed snapshots through the actual
Meraxes adapter and independently reads back gas, packet phases and budgets.
It checks the source and time units at `h=0.5`, no emission before initialization,
continued evolution after sources turn off, explicit matter-remap energy, and
independence from the legacy cumulative recombination sink. Its independently
recomputed relative energy residual is of order `1e-16`.

The retained Python comparison gives identical species and thermal energies for
mixed continuum/line absorption, source-off recombination and HeI conversion;
the largest emitted-number difference is `1.99e-23 cm^-3`. Atomic-table
regeneration is deterministic. These are integration and small-problem tests,
not a cosmological accuracy benchmark or a Gadi production run.
