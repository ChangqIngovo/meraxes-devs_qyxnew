# Photon-conserving excursion-set completion

`Flag_PhotonConservation=1` applies a conservative allocation to the ionization
map produced by the CPU excursion-set (ES) loop. The existing barrier, filter
sequence and first-crossing labels provide the reference map and spatial
priorities. The completion assigns source inventories to target capacities and
then writes the corrected neutral fraction back before feedback, recombination
updates and global summaries are finalized.

The default `Flag_PhotonConservation=0` keeps the legacy path. The enabled mode
requires `Flag_PatchyReion=1`, `ReionFilterType=0`, `ReionRtoMFilterType=0` and
`ReionDeltaRFactor>1`. It is a serial reference solver on MPI rank zero, with
explicit memory, edge and iteration limits. It is not a solver with a cost
comparable to a few FFTs.

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
| `PhotonConservationMaxEdges` | 20000000 | Maximum stored source-target links. |
| `PhotonConservationMemoryMB` | 2048 | Memory guard for the gathered problem and solver workspace. |
| `PhotonConservationMaxIterations` | 10000 | Maximum sweeps per master solve; the limit resets when a new cut/refinement starts another master solve. |
| `PhotonConservationMaxEvaluations` | 1000000 | Global maximum scalar objective/waterfill map evaluations for one quadratic refinement. |
| `PhotonConservationMaxCuts` | 512 | Maximum stored source-capacity cuts in the quadratic refinement. |
| `PhotonConservationTolerance` | 1e-9 | Relative solver tolerance, required in `(0,1e-5]`; the adapter uses an absolute count tolerance of `1e-12`. |

With `RequireFull=0`, status one is a valid maximum under finite support with
an explicit residual. That residual is a diagnostic unassigned inventory, not
a modeled UV background. An exhausted resource limit is an error in either
mode. Memory guards apply to the reference solver; they do not bound the
memory occupied by the rest of Meraxes.
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
| Cost and tolerances | `elapsed_seconds`, `tolerance`, `absolute_tolerance`, `optimization_tolerance`, `max_evaluations` |

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

## Validation and reproduction

The checks completed for this implementation are:

| Check | Observed result | What it establishes |
|---|---|---|
| Default and `USE_MINI_HALOS=ON` builds | Both build successfully | Compilation of the adapter and optional Pop. III source path. |
| Five native/feedback CTest entries | 5/5 pass | Allocator, ES completion, quadratic refinement, and feedback reconstruction with one and two MPI ranks. |
| Native shape hand cases | 10/10 pass | Symmetry, partial absorption, priority retention, faint inventories, Hall-cut repair and promoted-priority regression. |
| Independent explicit-edge LP comparisons | 40/40 seeded cases pass; maximum absorption discrepancy about `7.1e-15` | Small `4^3`-cell graphs: lower feasibility, maximum absorption, priority retention, edge ledgers and complete cut geometry. |
| Independent explicit-graph QP comparisons | 40/40 seeded cases pass; objective differences about `1e-13` | Small three-to-five-cell graphs, including nonzero lower bounds and alternate quadratic priors. The reference optimizer is also numerical. |
| Saved real-input replays | 45/45 pass with 512 cuts and 1,000,000 evaluations available | Completion and quadratic refinement on captured simulation inputs; these replays do not constitute a complete evolution. |
| `Flag_PhotonConservation=0` regression summaries | Default, IGM and X-ray configurations all agree in strict comparison mode | The tested original output summaries are unchanged with the option disabled. |
| Full enabled L10_N512 evolution | Local run not completed | The remaining end-to-end validation matrix is to be run on the target server; no complete local evolution or full-matrix pass is claimed. |

From the repository root, with the normal Meraxes build dependencies installed
(see `BUILD.md`), build and run only the new tests:

```sh
cmake -S . -B build-pc -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON
cmake --build build-pc --parallel --target meraxes test_photon_conservation test_photon_conservation_es test_photon_conservation_shape test_photon_conservation_feedback
ctest --test-dir build-pc --output-on-failure -R '^test_photon_conservation'
```

The three allocation tests do not require Criterion, external simulation data
or an MPI launch. The two feedback entries use the MPI launcher discovered by
CMake. The native shape driver contains ten hand cases, including faint-source,
partial-absorption, Hall-cut and promoted-priority regressions. A second build
directory configured with `-DUSE_MINI_HALOS=ON` checks
that compile-time source variant. `RelWithDebInfo` is also the recommended
configuration for the integration harness: the legacy one-rank Debug galaxy
pointer assertion can stop a run independently of this allocator.

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

The correction gathers the global problem onto rank zero and scatters the
corrected neutral fraction back to the original slabs. MPI input ordering is
explicit, but the allocation itself is serial. The number of stored edges,
cut separations and quadratic iterations depends on source clustering and
overlap geometry; there is no near-FFT complexity guarantee. Debug-level-two
files can be large because the positive-edge certificate is explicit.
For example, an isolated snapshot-41 replay exceeded an earlier 200,000-call
evaluation budget but converged with 436,125 evaluations, 2,556 master sweeps
and 84 cuts, taking about 15.3 seconds in the shape solve on the test machine.
This distinguishes a finite evaluation budget from stagnation; it does not
establish the cost of a full evolution or a production-size grid.

Count conservation and optimality relative to the chosen ES reference do not
establish the physical accuracy of bubble connectivity, size distributions or
the 21-cm power spectrum. Those require cosmological convergence and reference
simulation comparisons. The finite radius cap, partial-front completion and
quadratic map selection are declared model choices whose effects should be
measured in those comparisons.
