# Photon-conserving excursion-set completion

`Flag_PhotonConservation=1` applies a conservative allocation to the ionization
map produced by the CPU excursion-set (ES) loop. The existing barrier, filter
sequence and first-crossing labels provide the reference map and spatial
priorities. The completion assigns source inventories to target capacities and
then writes the corrected neutral fraction back before feedback, recombination
updates and global summaries are finalized.

The default `Flag_PhotonConservation=0` keeps the legacy path. The enabled mode
requires `Flag_PatchyReion=1`, `ReionFilterType=0`, `ReionRtoMFilterType=0` and
`ReionDeltaRFactor>1`. The global flow solver runs on MPI rank zero; optional
MPI workers accelerate its read-only spatial queries. Memory, edge and
iteration limits remain explicit. Its cost is not comparable to a few FFTs.

## Snapshot inventories

The adapter captures unfiltered, real-space inventories before the ES FFTs.
Counts use the mean-cell H+He-nucleus normalization already present in Meraxes.
With `d=1+delta`, `q=1-x_e_box` (or one when spin-temperature evolution is off),
and the incoming cumulative effective recombination field `Nrec_pre`, the
target capacity is

\[
C_j=q_j(d_j+N_{{\rm rec},j}^{\rm pre}).
\]

The source inventory is the deposited cumulative escaped stellar and effective
black-hole mass multiplied by their existing ionizing efficiencies and divided
by the mean total mass per grid cell. Pop. III sources use their own efficiency
when that component is compiled. These grids already include the adopted
escape fractions. The efficiency includes the existing baryon and
`1-3*Y_He/4` conversion; neither escape fractions nor helium factors are applied
a second time. `normalization_particles_per_cell` converts the normalized
counts to the corresponding effective number of nuclei.

The incoming `x_e_box` must be between zero and the existing float-valued
`0.999` ES ceiling. Values outside that range fail explicitly rather than
being clipped into a different inventory. This bound is required by the
existing full-ionization thermal threshold, which would otherwise confuse a
fully X-ray-ionized cell with a UV-completed cell.

The original ES neutral fraction gives the reference absorbed count

\[
b_j=\operatorname{clip}\left[C_j(1-x_{{\rm H},j}^{\rm ES}/q_j),0,C_j\right].
\]

Set `b=0` where `q=0`. The diagnostic `reference_projection_l1` records this
projection because legacy cell-level clipping and density floors need not
match the captured inventories exactly. Multiplication of the local fields
defines the capacity; a product of separately filtered fields is not an
additive atom inventory.

This is conservation of the snapshot's effective UV inventory under the
existing H/He and recombination prescription. It does not introduce separate
H I, He I and He II photon channels, spectral transport, or a time-dependent
radiation reservoir. The next snapshot recomputes its allocation from its own
cumulative inputs. In enabled mode the UV completion runs at every snapshot,
including after overlap and with `ReionUVBFlag=0`; turning off UVB feedback
does not turn off the allocation.

## Spatial completion and solution order

1. **Record the ES hierarchy.** The original descending filter radii and each
   target's first-crossing index are stored. The final unfiltered step has
   radius zero in the allocation geometry. Cells that never cross retain a
   separate, last-priority group.
2. **Find the maximum absorbable inventory.** The terminal problem allows every
   target to receive from sources within the original maximum filter radius,
   with bounds from zero to its capacity. Finite support can leave a source
   group without enough reachable atoms even when the whole box has spare
   capacity. The solver obtains this terminal maximum and records every
   unabsorbed source inventory.
3. **Choose the smallest declared completion.** Editable-front seeds are
   partially ionized reference cells and both sides of a face between fully
   ionized and non-fully-ionized positive-capacity cells. A band is a periodic
   six-neighbour distance from those seeds. Outside the band, the target count
   is fixed at `b`; inside it, the count may range from zero to `C`. Among
   bands that can attain the terminal maximum, choose the smallest width,
   then the smallest common promotion along the original radius ladder. An
   empty band is `-1`; if there is no front, full-domain release is explicitly
   represented by `-2`. The reported `front_seed_cells`, `editable_cells` and
   `editable_fraction` show the actual released volume: partially ionized
   tails can make even width zero include many cells.
4. **Preserve the reference in a defined order.** At the selected band and
   geometry, solve the lower bounds, unlock reference capacities in original
   first-crossing order from large scales to small scales, then unlock the
   remaining upper capacities. Residual reassignments may change which source
   supplies an atom. The resulting map maximizes absorption, minimizes total
   absolute count change from `b`, and preserves each ES-priority prefix as
   much as possible within those preceding optima.
5. **Select a unique target map when `PhotonConservationShape=1`.** On that
   optimal face, minimize one half of the sum of `(y-b)^2/C` over
   positive-capacity cells. This strictly convex objective selects a unique
   target allocation in exact arithmetic; its individual source assignments
   need not be unique. Fixed-marginal flow calls certify source-target
   feasibility, and violated source-capacity cuts refine the quadratic solve.
   Numerical residuals and the signed primal-dual gap are reported. Shape
   mode zero returns the staged map, whose choice among tied maps may depend
   on traversal order.

For a target whose first crossing is index `k`, promotion `p` gives the
target radius at index `max(k-p,0)` in the descending ladder. A never-crossed
target begins at the final identity step. A source-target edge is allowed
exactly when their periodic Euclidean separation is no greater than that
target's radius. Opening a larger radius permits partial absorption; it does
not assert that the original barrier crossed at that larger radius.

The allocator starts from a sparse subset of these edges. After a maximum-flow
solve, it looks for missing legal edges across the residual cut and adds them.
A stage is accepted when its cut is valid for the full declared geometry, or
when its universal absorption upper bound has already been attained. Limits
and numerical failures abort explicitly; they are not reported as proofs of
infeasibility or as a converged unique map.

The completion driver compares terminal and trial cut capacities as exact sums
of the supplied binary64 source and target inventories. This prevents a faint
source from disappearing from the band/promotion decision merely because two
rounded total counts agree. Flow assignments and optimization diagnostics
remain floating-point calculations with the reported tolerances.

## Writeback and feedback

The double-precision allocation `y` determines
`xH_double=q*(1-y/C)`, with `xH_double=q` when `C=0`. It is clamped to `[0,q]`
before conversion to the existing float grid. The double source ledger and
the float neutral grid are therefore audited separately. A float grid is not
expected to preserve a double-precision count identity exactly. The output
includes a cellwise rounding bound and the resulting global count difference.

Feedback is rebuilt using the corrected ionization state. Original crossed
cells retain their original ES scale for the UV-background closure; newly
completed cells use their selected completion scale. The existing source-rate
filters provide `Gamma12`, and cumulative sources provide `J_21`. Newly full
cells acquire a reionization redshift; existing reionization history and
constant-at-ionization UVB values are retained. Partial cells have their
temperature updated from the corrected neutral fraction, and their current
full-bubble `Gamma12` and radius are cleared. The existing full-cell thermal
and recombination update then runs once. This closure does not infer a
radiation intensity from individual allocation edges.

## Parameters

Copy the fragment in `input/params/photon_conservation.par` into a complete
simulation input file. Start with a small grid and debug level two.

| Parameter | Default | Meaning |
|---|---:|---|
| `Flag_PhotonConservation` | 0 | Enable this CPU ES completion. |
| `PhotonConservationShape` | 1 | 1: unique quadratic target-map refinement; 0: staged flow only. |
| `PhotonConservationDebug` | 1 | 0: log; 1: scalar snapshot ledger; 2: cell inventories, positive edges and search attempts. |
| `PhotonConservationRequireFull` | 0 | Abort after diagnostics unless each source meets the full-absorption tolerance. |
| `PhotonConservationMaxEdges` | 0 | 0: no source-target edge-count cap; a positive value sets an explicit cap. |
| `PhotonConservationMemoryMB` | 2048 | Memory guard for the gathered problem and solver workspace. |
| `PhotonConservationMaxIterations` | 10000 | Maximum sweeps per master solve; the limit resets when a new cut/refinement starts another master solve. |
| `PhotonConservationMaxEvaluations` | 1000000 | Global maximum scalar objective/waterfill map evaluations for one quadratic refinement. |
| `PhotonConservationMaxCuts` | 512 | Total generated source-capacity cut budget across component refinement and any global fallback. |
| `PhotonConservationTolerance` | 1e-9 | Relative solver tolerance, required in `(0,1e-5]`; the adapter uses an absolute count tolerance of `1e-12`. |

With `RequireFull=0`, status one is a valid maximum under finite support with
an explicit residual. That residual is a diagnostic unassigned inventory, not
a modeled UV background. An exhausted resource limit is an error in either
mode. Memory guards apply to the reference solver; they do not bound the
memory occupied by the rest of Meraxes.
Stored flow-arc indices are 64 bit and graph storage grows on demand, within
the memory guard. Grid/node indices and MPI message counts retain their
existing integer bounds. Disabling the edge cap does not disable memory,
iteration or evaluation limits.

For an existing run, rebuild Meraxes and explicitly set
`PhotonConservationMaxEdges : 0` in its input. An old input or included defaults
file specifying `20000000` still imposes that cap; omitting the parameter from
the top-level input does not override an included defaults file. Older
executables reject zero. Positive caps remain valid and must fit the input
parameter's signed 32-bit integer type; zero does not impose that edge-count
restriction on the solver.

`MaxIterations` limits each master solve; the reported `shape_iterations`
counts sweeps cumulatively across those solves. `MaxEvaluations` independently
limits scalar objective/waterfill map calls across the entire refinement. Reaching
either limit is an unresolved resource-budget failure, not evidence that the
allocation is infeasible or that the optimizer has stagnated.

## Diagnostics and independent audit

Debug levels one and two write
`OutputDir/photon_conservation_NNN.hdf5` for snapshot `NNN`. Schema version one
stores scalar attributes at the file root. The main groups of attributes are:

| Purpose | Attributes |
|---|---|
| Identity and status | `schema_version`, `certificate_complete`, `snapshot`, `grid_dim`, `mpi_ranks`, `solver_status`, `shape_enabled` |
| Snapshot ledger | `source_total`, `capacity_total`, `absorbed_total`, `residual_total`, `ledger_error`, `full_source_budget`, `max_source_relative_residual` |
| ES completion | `band_width`, `promotion`, `attempts`, `front_seed_cells`, `editable_cells`, `editable_fraction`, `lower_stage_flow`, `retention_stage_flow`, `min_l1`, `final_l1` |
| Selected and terminal maxima | `staged_absorbed_total`, `staged_mincut_capacity`, `staged_full_graph_certificate`, `terminal_maximum_absorption`, `terminal_full_source_budget` |
| Allocation | `source_count`, `target_count`, `edge_count`, `positive_edges`, `geometry_excess`, `max_source_overspend`, `max_target_excess`, `max_edge_balance_error`, `full_graph_certificate` |
| Float output | `float_grid_ledger_error`, `float_grid_roundoff_bound`, `reference_projection_l1`, `normalization_particles_per_cell` |
| Quadratic solve | `shape_status`, `shape_iterations`, `shape_evaluations`, `shape_oracle_calls`, `shape_cuts`, `shape_gap`, `shape_objective`, `shape_equality_error`, `shape_priority_error` |
| Final numerical adjustment | `shape_oracle_target_l1_error`, `shape_max_final_correction`, `shape_final_correction_l1` |
| Cost and tolerances | `elapsed_seconds`, `completion_seconds`, `refinement_seconds`, `tolerance`, `absolute_tolerance`, `optimization_tolerance`, `max_evaluations` |

`solver_status=0` means that every source meets the configured relative
full-absorption tolerance; any remaining numerical residual is still written
to the ledger. Status `1` means partial maximum absorption. The
`full_source_budget` flag is one exactly when status is zero, and
`max_source_relative_residual` records the largest residual-to-inventory ratio
among positive sources. The checker compares both to the status and, at debug
level two, to the individual source arrays. `max_edge_balance_error` measures
the largest difference between summed edge flows and a stored source or target
marginal. `shape_evaluations` and `shape_oracle_calls` expose work beyond the
master-sweep count.
`float_grid_ledger_error` is the signed difference between counts
reconstructed from the written float neutral fraction and the double
allocation. `shape_gap` is a floating-point signed diagnostic; it is not an
interval-arithmetic certificate. `optimization_tolerance` is the absolute
acceptance scale for the quadratic refinement; it is zero when shape mode is
disabled. It is distinct from the cell ledger and float-grid rounding tests.
The three adjustment diagnostics report, respectively, the initial flow
oracle's total target mismatch and the maximum and summed target changes from
the quadratic candidate to the final transported map. The final map is
rechecked for source accounting, bounds, retained quotas and objective gap.
These adjustment norms are diagnostics, not additional permitted photon loss.
On failure, master residuals and the most recent oracle/correction diagnostics
can belong to different attempted stages. They are not a single final-map
certificate; only a successful refinement proceeds to the complete output.

Before solving, the adapter writes a separate input checkpoint,
`OutputDir/photon_inputs_NNN.hdf5`, with `certificate_complete=0`. Debug level
one stores its attributes; debug level two also includes the captured input
arrays.
The checkpoint also records the shape option, tolerances, edge limit, memory
limit, iteration limit, `max_evaluations` and `max_cuts`. It is retained after the solve. The
allocation certificate is first written to
`photon_conservation_NNN.hdf5.tmp`, marked complete after all datasets have
been written, closed, and then atomically renamed to
`photon_conservation_NNN.hdf5`. This avoids truncating the replayable input
checkpoint while publishing the final result. If a solve fails, its checkpoint
can support reproduction but is not a completed certificate. Passing it
directly to the checker produces an explicit incomplete-certificate failure.
The positive-edge sidecar likewise writes to `.edges.csv.tmp`, closes and
renames it before publishing the complete HDF5 certificate.

To replay a debug-level-two input checkpoint without rerunning earlier
snapshots or building the full Meraxes executable:

```sh
python3 tools/replay_photon_conservation.py output/photon_inputs_052.hdf5 \
  --output replay_052.json
```

This requires Python 3, NumPy, h5py and a C compiler. The script builds the
three standalone allocation sources in a temporary directory; it does not
require MPI, FFTW, the simulation data or SciPy. By default it uses the
checkpoint's tolerances and resource limits, including the adapter's memory
reservation. `--max-evaluations` and `--max-cuts` explicitly override those
two limits for a diagnostic comparison. The JSON report distinguishes the ES
completion, quadratic refinement and independent positive-edge accounting
checks. Replay does not run Meraxes feedback or independently optimize the
quadratic objective, and its JSON is not a complete simulation certificate.

To compare a replay with an existing snapshot certificate and save the audited
target map for analysis:

```sh
python3 tools/replay_photon_conservation.py output/photon_inputs_054.hdf5 \
  --compare-certificate output/photon_conservation_054.hdf5 \
  --save-allocation replay_054.npz --output replay_054.json
```

The comparison requires identical source, capacity, baseline and ES hierarchy
inputs. It reports the objective and cellwise allocation differences, including
`allocation/capacity`; different valid source-edge assignments are allowed.
The default tolerances and evaluation limits still come from the checkpoint.

Debug level two also stores flattened arrays in global real-cell order,
`index=((x*grid_dim)+y)*grid_dim+z`:

| Purpose | Datasets |
|---|---|
| Inputs | `source`, `source_II`, `source_BH`, `source_III`, `capacity`, `density`, `neutral_pre`, `recombinations_pre` |
| Reference and result | `baseline`, `allocation`, `source_spent`, `source_residual`, `xH_ES`, `xH_double`, `xH` |
| Geometry and priorities | `lower`, `upper`, `radius_cells`, `first_crossing`, `priority`, `radii_cells`, `retained_quota` |
| Selected upper-capacity cut | `staged_cut_source`, `staged_cut_target` |
| Final fixed-marginal cut | `cut_source`, `cut_target` |

`xH` contains the actual float-rounded values, stored as doubles for the
diagnostic file. `radii_cells` is the radius ladder rather than a grid array.
`retained_quota` has one entry per radius step plus the never-crossed group.
The staged cut uses the selected upper capacities `U`: its capacity is the sum
of source inventories outside its source set plus upper capacities inside its
target set. The checker verifies that capacity equals the staged absorption
and that every geometrically reachable positive-`U` target of its source set
belongs to its target set. Together with the feasible allocation, this checks
maximum absorption in the selected geometry to the reported tolerances. The
final cut belongs to the fixed-marginal feasibility solve for the chosen map;
neither cut by itself certifies the earlier minimum band, minimum count change
or quadratic optimum.

Two sidecars are written at debug level two:

- `photon_conservation_NNN.edges.csv`: `source,target,photons` for every
  positive source-target assignment.
- `photon_conservation_NNN.attempts.csv`:
  `band,promotion,status,absorbed,lower_deficit,residual` for completion trials.

Run the independent checker with Python 3, NumPy and h5py:

```sh
python3 tools/check_photon_conservation.py output/photon_conservation_048.hdf5 --require-cells
python3 tools/check_photon_conservation.py output/photon_conservation_048.hdf5 --require-cells --require-full --json
```

Without `--require-cells`, a debug-level-one file receives a scalar ledger
check only. Debug-level-two checks independently sum positive edges, enforce
each source budget and target capacity, check periodic distances, reconstruct
the selected ES radius and band rules, verify recorded L1 and priority quotas,
check the staged full-geometry cut, and compare float-grid errors with the
rounding bound. Each retained-priority group allows the recorded optimization
tolerance `tau`; a prefix containing `k` active groups allows `k*tau`.
Since `L1=F+sum(b)-2*sum(retained)`, the aggregate L1 comparison allows
`2*G*tau` for `G` active groups, plus explicit floating summation and total-count
roundoff. Here an active group contains at least one positive-upper-bound
target. The report records these allowances separately. Final versus staged
absorption uses a tight total-count roundoff check. Source accounting,
positive-edge sums and float rounding tests receive no optimization-tolerance
allowance. Positive-source edge budgets, edge-versus-recorded spending and
edge-plus-residual ledgers are also checked after normalization by that source's
own inventory, using the declared relative tolerance plus a floating summation
bound, with no global absolute count floor. Zero-inventory sources must have
exactly zero outgoing edge flow, recorded spending and residual. The cut check uses bounded-memory pairwise distances and can be
slow for a large partial-absorption certificate. Older schema-one files that
lack staged-cut fields receive only the earlier consistency checks. The
checker does not independently solve for the L1 or hierarchy optima or prove
quadratic convergence.
It exits nonzero for failed or incomplete requested checks.

`tools/run_photon_conservation_validation.py` runs and audits the configured
L10_N512 cases. Its audit reads only final `photon_conservation_*.hdf5` files,
excluding input checkpoints and temporary files, and requires every snapshot
from zero through the largest requested output snapshot. A missing final
certificate is a failure even when all earlier certificates pass. Use a fresh
work directory for new runs; `--audit-only` examines an existing run without
restarting it.

A failure in `pos_to_ngp` before a snapshot reaches `find_HII_bubbles` is
upstream of photon allocation. One such failure in a 32-rank L10_N512 run
occurred at snapshot 35 after snapshot 34 completed with zero photon
residual. The VELOCIraptor reader previously left `ProgIndex` and `TreeFlags`
uninitialised when `FlagIgnoreProgIndex=0`. These fields now explicitly select
the existing mass-based merger fallback because no progenitor link is loaded;
see [the reader regression tests](testing.md). Invalid position assertions
are retained. This correction requires a fresh evolution because it can
change galaxy histories and therefore the photon sources themselves.

## Validation and reproduction

The checks completed for this implementation are:

| Check | Observed result | What it establishes |
|---|---|---|
| Default and `USE_MINI_HALOS=ON` builds | Both build successfully | Compilation of the adapter and optional Pop. III source path. |
| Five native/feedback CTest entries | 5/5 pass | Allocator, ES completion, quadratic refinement, and feedback reconstruction with one and two MPI ranks. |
| Native shape hand cases | 19/19 pass, including AddressSanitizer/UndefinedBehaviorSanitizer | Symmetry, partial absorption, priority retention, disconnected components, faint inventories, Hall-cut repair and strengthened-certificate regressions. Leak detection is unavailable in the execution sandbox. |
| Independent explicit-edge LP comparisons | 40/40 seeded cases pass; maximum absorption discrepancy about `7.1e-15` | Small `4^3`-cell graphs: lower feasibility, maximum absorption, priority retention, edge ledgers and complete cut geometry. |
| Independent explicit-graph QP comparisons | 40/40 seeded cases pass; objective differences about `1e-13` | Small three-to-five-cell graphs, including nonzero lower bounds and alternate quadratic priors. The reference optimizer is also numerical. |
| Saved real-input replays | 45/45 pass with 512 cuts and 1,000,000 evaluations available | Completion and quadratic refinement on captured simulation inputs; these replays do not constitute a complete evolution. |
| Uploaded Gadi snapshot 52 after the coordinate-search fix | Pass at the original 1,000,000-evaluation limit, using 48,237 evaluations and 224 cuts | Absorbed inventory `1074.309407529475`, zero residual, and independent positive-edge accounting/geometry checks pass. The final gap `6.93e-7` is below the unchanged `1.07431e-6` optimization tolerance. |
| Uploaded Gadi snapshot 62 after the certificate/headroom fix | Pass with 386 cuts, 66,707 evaluations and 904 flow calls | Zero residual and an independent edge audit; the global gap `1.54034e-6` is below the unchanged `2.77153e-6` tolerance. |
| `Flag_PhotonConservation=0` regression summaries | Default, IGM and X-ray configurations all agree in strict comparison mode | The tested original output summaries are unchanged with the option disabled. |
| Full enabled L10_N512 evolution, default configuration, 32 MPI ranks | Snapshots 0--75 complete with exit code zero; 76/76 certificates pass 6,308 independent checks | Original tolerance and work limits, zero source residual throughout, and no missing snapshots. This covers one configuration/rank combination, not the full validation matrix. |

The full-run certificate audit independently checks the explicit source/target
ledgers, allowed edge geometry, selected-geometry maximum absorption and
recorded priority quotas. It does not independently prove priority/L1
optimality, minimum band/promotion, or quadratic uniqueness; the separate
small-graph QP comparisons and native primal-dual acceptance checks address
different parts of the validation.

From the repository root, with the normal Meraxes build dependencies installed
(see `BUILD.md`), build and run only the new tests:

```sh
cmake -S . -B build-pc -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON
cmake --build build-pc --parallel --target meraxes test_photon_conservation test_photon_conservation_es test_photon_conservation_shape test_photon_conservation_feedback
ctest --test-dir build-pc --output-on-failure -R '^test_photon_conservation'
```

The three allocation tests do not require Criterion, external simulation data
or an MPI launch. The two feedback entries use the MPI launcher discovered by
CMake. The 19 native shape cases include faint-source, partial-absorption,
Hall-cut, component and flat-priority Newton-trial regressions. Two analytic
certificate cases must pass with zero generated Hall cuts; they cover full
and partial absorption and fail if the strengthened certificate is disabled.
A second build
directory configured with `-DUSE_MINI_HALOS=ON` checks
that compile-time source variant. `RelWithDebInfo` is also the recommended
configuration for the integration harness: the legacy one-rank Debug galaxy
pointer assertion can stop a run independently of this allocator.

For the tested Gadi environment (GCC 12.2.0 and OpenMPI 4.1.4), the
single-precision FFTW MPI library is in the `ompi3` subdirectory. Passing only
`FFTW_ROOT` does not locate that variant with the current finder. Configure
with the verified paths after loading the usual dependencies:

```sh
cmake -S . -B build-pc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=mpicc \
  -DHDF5_ROOT=/apps/hdf5/1.12.2p -DHDF5_PREFER_PARALLEL=ON \
  -DFFTW_INCLUDE_DIR=/apps/fftw3/3.3.10-nci1/include \
  -DFFTW_LIBRARY=/apps/fftw3/3.3.10-nci1/lib/libfftw3f.so \
  -DFFTW_MPI_LIBRARY=/apps/fftw3/3.3.10-nci1/lib/ompi3/libfftw3f_mpi.so \
  -DBUILD_TESTS=OFF -DUSE_CUDA=OFF
cmake --build build-pc --parallel 2
```

Do not substitute the `ompi5`, Intel MPI or double-precision libraries for
this build. A configured `build-pc` already caches these explicit paths;
subsequent source updates need only the build command in the same dependency
environment.

The independent optimizers require Python 3, NumPy, SciPy and a C compiler.
They compile temporary shared libraries directly from the native sources:

```sh
python3 tools/test_photon_conservation_oracle.py --trials 40 --output build-pc/photon_lp.json
python3 tools/test_photon_conservation_shape.py --cases 40 --output build-pc/photon_qp.json
```

To run the complete integration matrix, supply the unpacked L10_N512 test
data and a new destination directory:

```sh
python3 tools/run_photon_conservation_validation.py \
  --binary build-pc/bin/meraxes --data /path/to/L10_N512 \
  --work-dir /path/to/new-pc-validation \
  --configs default igm xray --ranks 1 2 --snapshots 48 63 75
```

This command is a reproduction procedure, not a statement that the full matrix
has passed. It writes per-run exit status, per-snapshot certificate audits,
MPI input/map comparisons and `validation_summary.json`. MPI differences are
reported with the source and ES inputs: decomposition-dependent input changes
must be distinguished from different numerical solutions to identical inputs.
The small-graph optimization tests and summary regressions do not establish
cosmological accuracy or large-grid performance.

## Scope and computational limits

If the assembled feasible map narrowly fails the global optimality check,
the certificate can use two pointwise bounds implied by the original
transport and retained-priority constraints. A target cannot absorb
more than the total inventory of sources within its permitted radius:
`U'_j = min(U_j, sum_{i -> j} S_i)`. For priority group `g`, retaining its
quota `Q_g` also requires
`L'_j = max(L_j, Q_g - sum_{k in g, k != j} min(U'_k, b_k))`.
Every feasible field satisfies these necessary conditions, so minimizing
the same Lagrangian over their tighter scalar intervals remains a valid
global dual lower bound. Supply sums are enclosed upward and lower bounds
downward, with conservative widening for rounded references. This changes
only the certificate: the local/master iterations, transported map, graph,
inventories, objective and priority quotas retain their original definitions.
Idle component work arrays are reused, and one additional scalar-map
evaluation is charged to the existing limit. Both full and partial
absorption are supported; final feasibility and optimality checks remain
mandatory. The original master domain is retained to preserve its numerical
conditioning.

The shape solver first identifies connected components of the **full allowed
geometric graph**, including legal edges not yet materialized by the flow
solver. Components cannot exchange photons. Their maximum absorption and
lexicographic retained-overlap objectives therefore separate in exact
arithmetic. Independent component solves avoid repeatedly discovering these
budgets through global Hall cuts; components with a single target have a fixed
target count and skip the iterative shape solve.

Floating-point priority quotas need additional care. The implementation does
not accept the sum of local dual bounds as a certificate for the original
global problem. It translates local source-subset cuts to global unused-budget
constraints, uses shared nonnegative priority multipliers, and evaluates a
dual lower bound for the original global quadratic problem. The assembled map
must pass the original global gap, total, capacity and priority checks, followed
by a full-graph flow certificate. If the lifted bound is too loose, the original
global solver runs with the remaining evaluation and cut budgets. Component
tolerances share half the original global error budget, leaving headroom for
the lifted certificate and final transport accumulation. This asks the local
solves for greater accuracy; neither the objective nor the global acceptance
tolerance is relaxed. Component decisions remain serial; their transport
queries may use the optional MPI geometry executor described below.

On the historical uploaded snapshot-54 input (32 cubed cells, normalized
source inventory `1352.0677771`), the full graph contains
167 components and 9,174 eligible links. The recorded global solver used
478,733 evaluations, 491 cuts and 1,476 flow calls. Component refinement uses
16,469 evaluations, 194 cuts and 516 flow calls at the same `1e-9` relative
tolerance, including the caching and local-headroom changes. The global
primal-dual gap is `4.449e-7`, below `1.352e-6`. The independent positive-edge
audit passes with zero residual.
This captured input predates the VELOCIraptor initialization fix; it is
distinct from snapshot 54 of the fresh evolution with that fix.

Development-host replay takes 17.4 seconds in shape refinement under
concurrent load. The final code passes 19
native shape cases, 40 independent small explicit-graph QP comparisons, and
AddressSanitizer/UndefinedBehaviorSanitizer checks (leak detection unavailable
in the execution sandbox). The unchanged baseline was still refining the shape after
582 seconds on the same host and was subsequently interrupted, so its complete
runtime was not measured. The separately supplied Gadi certificate records
3,263 seconds of total PC elapsed time, without a separate refinement time;
it is not a same-machine or same-metric timing comparison. These measurements
apply to this snapshot. Large, connected ionization regions may benefit less.

The production log and certificate now separate `completion_seconds` from
`refinement_seconds` to locate later bottlenecks without inferring their source
from the total time alone.

When a proposed marginal fails the final repair checks, refinement reuses the
cut already separated for that identical marginal. It no longer repeats the
same fixed-marginal transport solve merely to recover the cut masks. This
requires two additional byte arrays, charged to the existing memory budget.
For this caching change alone, a sequential snapshot-54 comparison reduced
flow calls from 687 to 500 and
shape time from 18.19 to 13.87 seconds. The allocation arrays, objective,
gap, cuts and master evaluations were bit-identical; both independent edge
audits passed. These are development-host measurements, not a Gadi speed claim.

For fresh-evolution snapshots 54, 55 and 56 after the VELOCIraptor reader fix,
the input source, capacity, baseline, radius and priority arrays match the
previous implementation bit for bit. With all three changes enabled, flow
calls fall from 450 to 348, 466 to 376 and 516 to 398, respectively (19–23%).
All three global gaps are smaller. These comparisons include the tighter
local tolerances, so they do not imply bit-identical output fields or a
controlled wall-time speedup.

The uploaded snapshot-62 input reproduced a subtler failure: all local solves
completed using 378 cuts and 67,471 master evaluations, but the lifted global
gap was `2.8416902e-6`, above the required `2.7715323e-6`. The global fallback
then restarted with only 134 cuts remaining and exhausted that budget. Its
reported large final gap describes the restarted iterate, not the preceding
component result. Reserving local accuracy headroom avoids that restart:
the final replay passes with 386 cuts, 66,707 evaluations, 904 flow calls,
zero residual and a global gap of `1.5403356e-6`, below the unchanged
`2.7715323e-6` tolerance. Snapshot 43 also passes in 0.47 seconds in an
isolated replay, guarding against an earlier master-conditioning regression.
An independent explicit-edge quadratic solve, followed by native transport
and an independently evaluated global dual bound, gives objective
`102.15797348785863` and gap `4.66e-8`. The final native snapshot-62 objective
is `102.1579746822792`, higher by `1.19442e-6` and within its certified gap;
the largest target-count difference is `1.25304e-6`. Its 287-second shape
time was measured concurrently with a 32-rank evolution and is not a
controlled speed comparison.

The fresh default-configuration evolution completes all snapshots 0--75 with
32 MPI ranks and the original limits. Snapshot 75 uses 242 cuts, 33,850
evaluations and 500 flow calls; its gap `3.88662e-6` is below `9.01239e-6`.
The native run takes 3,420.82 seconds, including 844.79 seconds in the final
snapshot's PC correction. These timings come from an oversubscribed local
container and are not a controlled Gadi comparison. Late snapshots remain
expensive despite the reduction in redundant flow work.

The correction gathers the global problem onto rank zero and scatters the
corrected neutral fraction back to the original slabs. MPI input ordering is
explicit. The global residual-network updates remain serial, while geometric
queries can run across MPI ranks. The number of stored edges,
cut separations and quadratic iterations depends on source clustering and
overlap geometry; there is no near-FFT complexity guarantee. Debug-level-two
files can be large because the positive-edge certificate is explicit.
For example, an isolated snapshot-41 replay exceeded an earlier 200,000-call
evaluation budget but converged with 436,125 evaluations, 2,556 master sweeps
and 84 cuts, taking about 15.3 seconds in the shape solve on the test machine.
This distinguishes a finite evaluation budget from stagnation; it does not
establish the cost of a full evolution or a production-size grid.

The original coordinate search reproduced the Gadi snapshot-52 failure at
1,000,001 attempted evaluations. With a diagnostic five-million limit it
converged after 2,208,080 evaluations; 98.2% of those evaluations belonged to
coordinate searches that repeatedly discarded their previous multiplier.
The updated search starts from that multiplier, uses safeguarded scalar
Newton/secant steps, and interleaves bisection across flat segments. A
numerically unresolvable speculative coupled-Newton step is backtracked;
resource exhaustion and a failed restored state still abort. Neither the
feasible set nor the final acceptance gates change.

The historical snapshot-52 replay of the coordinate-search fix used
48,237 evaluations, 3,793 master sweeps and
675 flow-oracle calls, with a measured shape time of about 61 seconds on the
development machine. The original enlarged-budget comparison took about
122 seconds under different concurrent load; these are diagnostic timings,
not a controlled speed benchmark. Reducing scalar evaluations does not remove
the costs of the coupled solve and flow oracles. This single replay does not
establish completion of later snapshots or large-grid scalability.

Count conservation and optimality relative to the chosen ES reference do not
establish the physical accuracy of bubble connectivity, size distributions or
the 21-cm power spectrum. Those require cosmological convergence and reference
simulation comparisons. The finite radius cap, partial-front completion and
quadratic map selection are declared model choices whose effects should be
measured in those comparisons.
The minimum band, promotion and reference-retention ordering are also choices
among conserving maps, not consequences of photon counting. A large L1 change
from the original ES map is therefore an allocation-change diagnostic, not
by itself a measure of physical morphology error.

## Parallel execution

See [MPI execution and Gadi measurements](photon_conservation_mpi.md) for the
optional distributed geometry search, unchanged global conservation checks,
resource limits, regression tests and reproducible performance comparisons.

