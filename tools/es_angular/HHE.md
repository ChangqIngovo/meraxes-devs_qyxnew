# Fixed-front H/He chemistry

`angular_hhe.py` implements the chemistry step for a **frozen angular-front
allocation**. Its five-species upward reaction map and cross sections are
self-contained in `chemistry.py` and require only NumPy. The formulas were
extracted unchanged from the earlier chemistry implementation; its source
hash and physical attribution are recorded in that module. No retired
allocation strategy or worker is imported. This file describes the chemistry
kernel; its periodic snapshot caller is documented in [PERIODIC.md](PERIODIC.md).

The inputs are gas `(cells, 5)`, photon credits `(resources,)`, cross sections
`(resources, 3)` for HI/HeI/HeII, and routing fractions `(resources, cells)`.
A resource identifies a source, angular bin and spectral group. All resources
arriving at one physical cell compete for the same gas populations. Periodic
images of a physical cell must share one gas row; copying its gas would create
extra absorption capacity.

## Budgets and chemistry

The routing fractions split each resource once into separate target-cell
credits and a boundary credit. Chemistry consumes only the target-cell
credits. It returns every unspent `(resource, target)` credit and preserves
boundary credits. A fixed route with a 50% target share can therefore spend
at most 50% of its resource at that target. Small floating-point corrections
to the initial split are explicitly reported as `split_roundoff_l1`.

Within each target cell, a positive exponential midpoint update handles
cross-section-weighted spectral competition. The same calculated captures
update the gas and debit individual photon channels. The kernel audits photon,
electron and nucleus counts and rejects negative states or budget overdrafts.
HeI ionization creates HeII, which a hard photon can then ionize again; these
are two absorptions from shared populations, not independent capacity stores.

## Events and continuation

The remaining photoelectron capacities for the three threshold classes are

| Photon threshold class | Remaining capacity |
| --- | --- |
| 13.6–24.59 eV | HI |
| 24.59–54.4 eV | HI + HeI |
| At least 54.4 eV | HI + 2 HeI + HeII |

All three decrease monotonically during photoionization without
recombination. By default, the earliest cell/class transparency threshold
stops the simultaneous chemistry update. The threshold only changes routing
eligibility: it does not delete residual neutral gas or its nuclei.

After an event, a spatial solver must continue each unspent credit **from its
current target**, preserving its resource and position. Summing leftovers
back to the original source and rerouting them changes their allocation.
`periodic_front.py` now preserves those credits and their finite flight times;
`snapshot_worker.py` adds recombination-driven reopening and checkpointing.
No global H/He time-sandwich claim follows from the H-only theorem.

## Verification

Run the fixed-front ledger, spectral competition and event tests with:

```sh
python3 tools/es_angular/test_angular_hhe.py
```

The midpoint refinement diagnostic prints JSON and writes a file only when
an output path is supplied:

```sh
python3 tools/es_angular/angular_hhe_convergence.py
python3 tools/es_angular/angular_hhe_convergence.py --output /tmp/hhe-refinement.json
```

This reference adapter materializes the resource-by-target inventory, even
when passed a sparse routing matrix. Sparse storage is a future optimization.
