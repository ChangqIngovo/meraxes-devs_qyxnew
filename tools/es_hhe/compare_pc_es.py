#!/usr/bin/env python3
"""Compare PC and original ES histories and ionization morphology.

Requires numpy, scipy, h5py and matplotlib. Run directories, output directories
or Meraxes master files are accepted. Logs alone produce history/duration
results; spatial comparisons require the actual xH grids. Inputs are read-only.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import integrate, ndimage

COLORS = ('#17679a', '#c65e27')
NEUTRAL = LinearSegmentedColormap.from_list('neutral', ['#145a86', '#58acd0', '#ffffff'])


def scalar(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError('Expected scalar metadata')
    item = value.reshape(-1)[0]
    return item.decode() if isinstance(item, bytes) else item.item() if hasattr(item, 'item') else item


def read_params(path):
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*(\w+)\s*:\s*(.*?)\s*$', line.split('#', 1)[0])
        if match:
            values[match[1]] = match[2]
    return values


def read_log(path):
    rows, parameters = {}, {}
    snapshot = None
    for line in path.read_text().splitlines():
        match = re.match(r'^\s+(\w+)\t(.*)$', line)
        if match:
            parameters[match[1]] = match[2].split('Meraxes git commit:')[0].strip()
        match = re.match(r'^Snapshot (\d+)\s+\(z = ([\d.]+)\)', line)
        if match:
            snapshot = int(match[1])
            rows[snapshot] = dict(snapshot=snapshot, redshift=float(match[2]), source='log')
        match = re.search(r'ES/HHe snapshot (\d+) complete: volume xHI=([^,]+)', line)
        if match:
            rows[int(match[1])]['xHI_volume'] = float(match[2])
        match = re.match(r'^\s*xH = ([\d.eE+-]+) VS ([\d.eE+-]+)', line)
        if match and snapshot is not None:
            rows[snapshot].setdefault('xHI_volume', float(match[1]))
            rows[snapshot]['xHI_mass'] = float(match[2])
    return {s: r for s, r in rows.items() if 'xHI_volume' in r}, parameters


def periodic_components(mask):
    """Sizes of all 6-connected regions, joining opposite periodic faces."""
    labels, count = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    if not count:
        return np.empty(0, dtype=np.int64)
    parent = np.arange(count + 1)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for axis in range(3):
        lo, hi = np.take(labels, 0, axis), np.take(labels, -1, axis)
        good = (lo > 0) & (hi > 0)
        for a, b in np.unique(np.column_stack((lo[good], hi[good])), axis=0):
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[rb] = ra
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    merged = np.zeros(count + 1, dtype=np.int64)
    for i in range(1, count + 1):
        merged[root(i)] += sizes[i]
    return np.sort(merged[merged > 0])[::-1]


def crossing(z, xhi, ionized_fraction):
    """A unique downward crossing; never extrapolate or monotonize history."""
    z, xhi = np.asarray(z), np.asarray(xhi)
    target = 1 - ionized_fraction
    if len(z) < 2:
        return {'status': 'insufficient_samples', 'redshift': None}
    if xhi[0] < target:
        return {'status': 'already_passed_at_first_sample', 'redshift': None}
    below = xhi <= target
    starts = np.flatnonzero((~below[:-1]) & below[1:])
    upward = np.any(below[:-1] & ~below[1:])
    if len(starts) > 1 or upward:
        return {'status': 'multiple_crossings', 'redshift': None}
    if xhi[0] == target:
        return {'status': 'measured', 'redshift': float(z[0])}
    if not len(starts):
        return {'status': 'not_reached', 'redshift': None}
    i = int(starts[0])
    w = (target - xhi[i]) / (xhi[i+1] - xhi[i])
    return {'status': 'measured', 'redshift': float(z[i] + w*(z[i+1]-z[i])),
            'bracket_redshifts': [float(z[i]), float(z[i+1])]}


def cosmic_age_myr(z, parameters):
    """Integrate the expansion history specified by the simulation parameters."""
    if any(key not in parameters for key in ('Hubble_h', 'OmegaM', 'OmegaLambda')):
        return None
    h, matter, dark = (float(parameters[key]) for key in ('Hubble_h', 'OmegaM', 'OmegaLambda'))
    radiation, curvature = float(parameters.get('OmegaR', 0)), float(parameters.get('OmegaK', 0))
    w = float(parameters.get('wLambda', -1))
    hubble_s = 100*h/3.0856775814913673e19
    def integrand(a):
        if a == 0:
            return 0.
        e2 = radiation/a**4 + matter/a**3 + curvature/a**2 + dark*a**(-3*(1+w))
        if e2 <= 0:
            raise ValueError('Invalid cosmology')
        return 1/(a*np.sqrt(e2))
    return integrate.quad(integrand, 0, 1/(1+z), epsabs=1e-11)[0]/hubble_s/(365.25*86400*1e6)


class Run:
    def __init__(self, label, location=None, log=None):
        self.label, self.location = label, Path(location).resolve() if location else None
        self.rows, self.params, self.grids, self.redshifts = {}, {}, {}, {}
        self.master = None
        self.notes = []
        if self.location:
            if not self.location.exists():
                raise FileNotFoundError(self.location)
            self.output = self.location.parent if self.location.is_file() else self.location
            if (self.output/'output').is_dir():
                self.output = self.output/'output'
            self.root = self.output.parent if self.output.name == 'output' else self.output
            if log is None:
                logs = sorted(self.root.glob('run_*.log'), key=lambda p: p.stat().st_mtime)
                log = logs[-1] if logs else None
            input_file = self.root/'input.par'
            if input_file.is_file():
                overrides = read_params(input_file)
                for key, fallback in [('DefaultsFile', 'defaults.par'),
                                      ('SimParamsFile', 'Genesis_L10_N512.par')]:
                    p = Path(overrides.get(key, ''))
                    if not p.is_absolute():
                        p = self.root/p
                    if not p.is_file():
                        p = self.root/'provenance'/fallback
                    if p.is_file():
                        self.params.update(read_params(p))
                self.params.update(overrides)
        if log:
            self.log = str(Path(log).resolve())
            self.rows, logged = read_log(Path(log))
            self.params.update(logged)
        else:
            self.log = None
        if self.location:
            self.discover_grids()
        if not self.rows:
            raise ValueError(f'{label}: no measured neutral-fraction history found')
        self.history = sorted(self.rows.values(), key=lambda r: -r['redshift'])
        z = np.array([r['redshift'] for r in self.history])
        x = np.array([r['xHI_volume'] for r in self.history])
        if (not np.isfinite(z).all() or np.any(np.diff(z) >= 0) or
                not np.isfinite(x).all() or np.any((x < 0) | (x > 1))):
            raise ValueError(f'{label}: invalid redshifts or neutral fractions')

    def discover_grids(self):
        filename = str(self.params.get('FileNameGalaxies', 'meraxes'))
        candidate = self.location if self.location.is_file() else self.output/(filename+'.hdf5')
        if candidate.is_file():
            with h5py.File(candidate, 'r') as f:
                if 'InputParams' in f:
                    for key, value in f['InputParams'].attrs.items():
                        if np.asarray(value).size == 1:
                            self.params[key] = scalar(value)
                for name in f:
                    match = re.fullmatch(r'Snap(\d+)', name)
                    if not match:
                        continue
                    snap = int(match[1])
                    if 'Redshift' in f[name].attrs:
                        self.redshifts[snap] = float(scalar(f[name].attrs['Redshift']))
                    if 'Grids' in f[name]:
                        self.grids[snap] = (candidate, name+'/Grids')
            self.master = candidate
        alist = self.root/'provenance/a_list.txt'
        if not alist.is_file() and 'SimulationDir' in self.params:
            alist = Path(str(self.params['SimulationDir']))/'a_list.txt'
        if alist.is_file():
            scale = np.loadtxt(alist, ndmin=1)
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f'Invalid scale-factor list: {alist}')
            for snap, a in enumerate(scale):
                self.redshifts.setdefault(snap, 1/float(a)-1)
        for path in sorted(self.output.glob(filename+'_grids_*.hdf5')):
            snap = int(re.search(r'_grids_(\d+)\.hdf5$', path.name)[1])
            self.grids[snap] = (path, '')
        for path in sorted(self.output.glob('es_state_*.hdf5')):
            snap = int(re.search(r'es_state_(\d+)\.hdf5$', path.name)[1])
            existing = self.grids.get(snap)
            use_state = existing is None
            if existing:
                try:
                    with h5py.File(existing[0], 'r') as f:
                        g = f[existing[1]] if existing[1] else f
                        use_state = 'xH' not in g or g['xH'].size == 0
                except (KeyError, OSError):
                    use_state = True
            if use_state:
                self.grids[snap] = (path, '')
        for snap in sorted(self.grids):
            path, group = self.grids[snap]
            try:
                with h5py.File(path, 'r') as f:
                    g = f[group] if group else f
                    if 'xH' not in g:
                        del self.grids[snap]
                        continue
                    z = self.redshifts.get(snap, self.rows.get(snap, {}).get('redshift'))
                    for key in ('redshift', 'Redshift'):
                        if key in g.attrs:
                            z = float(scalar(g.attrs[key]))
                    if z is None:
                        raise ValueError(f'No redshift for snapshot {snap}; include master or a_list.txt')
                    ds = g['xH']
                    if ds.size:
                        x, density = self._arrays(g)
                        mean = float(x.mean())
                        mass = float(np.sum(x*density)/density.sum()) if density is not None else None
                    else:
                        mean = float(scalar(ds.attrs['volume_weighted_global_xH']))
                        mass = float(scalar(ds.attrs['mass_weighted_global_xH'])) if 'mass_weighted_global_xH' in ds.attrs else None
                    self.rows[snap] = dict(snapshot=snap, redshift=float(z), xHI_volume=mean,
                                           xHI_mass=mass, source='grid' if ds.size else 'grid_attributes')
                    if not ds.size:
                        del self.grids[snap]  # Attributes support histories, not spatial maps.
            except (KeyError, OSError) as exc:
                raise RuntimeError(f'Cannot read {path}:{group}; keep external grid files alongside the master') from exc

    @staticmethod
    def _arrays(g):
        x = np.asarray(g['xH'], dtype=float)
        if x.ndim == 1 and 'grid_dim' in g.attrs:
            n = int(scalar(g.attrs['grid_dim']))
            x = x.reshape(n, n, n)
        if x.ndim != 3 or len(set(x.shape)) != 1:
            raise ValueError('Morphology needs a complete cubic xH grid')
        if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError('xH contains nonfinite or out-of-range values')
        density = None
        if 'deltax' in g and g['deltax'].size:
            density = 1 + np.asarray(g['deltax'], dtype=float)
        elif 'gas' in g:
            gas = np.asarray(g['gas'], dtype=float)
            density = (gas[:, 0]+gas[:, 1]).reshape(x.shape)
        if density is not None and (density.shape != x.shape or not np.isfinite(density).all()
                                    or np.any(density < 0) or density.sum() <= 0):
            raise ValueError('Invalid gas-density weights')
        return x, density

    def field(self, snapshot):
        if snapshot not in self.grids:
            raise ValueError(f'{self.label}: missing xH grid for snapshot {snapshot}')
        path, group = self.grids[snapshot]
        with h5py.File(path, 'r') as f:
            return self._arrays(f[group] if group else f)[0]


def duration(run):
    z = [r['redshift'] for r in run.history]
    x = [r['xHI_volume'] for r in run.history]
    result = {'definition': 'volume ionized fraction = 1 - mean(xHI)',
              'interpolation': 'linear xHI(z) between bracketing samples; no extrapolation',
              'thresholds': {}, 'durations': {}}
    for percent in (10, 25, 50, 75, 90):
        value = crossing(z, x, percent/100)
        value['cosmic_age_myr'] = cosmic_age_myr(value['redshift'], run.params) if value['redshift'] is not None else None
        result['thresholds'][str(percent)] = value
    for lo, hi in ((10, 90), (25, 75)):
        a, b = result['thresholds'][str(lo)], result['thresholds'][str(hi)]
        measured = a['status'] == b['status'] == 'measured'
        result['durations'][f'{lo}-{hi}'] = {
            'status': 'measured' if measured else 'not_measurable_in_available_history',
            'delta_z': a['redshift']-b['redshift'] if measured else None,
            'delta_t_myr': b['cosmic_age_myr']-a['cosmic_age_myr'] if measured and
                          a['cosmic_age_myr'] is not None and b['cosmic_age_myr'] is not None else None}
    return result


def save_plot(fig, out, name):
    for extension in ('png', 'pdf'):
        fig.savefig(out/(name+'.'+extension), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def history_plot(runs, durations, out):
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]}, layout='constrained')
    for run, color in zip(runs, COLORS):
        z = [r['redshift'] for r in run.history]
        axes[0].plot(z, [r['xHI_volume'] for r in run.history], color=color, lw=2, label=run.label+' (volume)')
        mass = [r.get('xHI_mass') if r.get('xHI_mass') is not None else np.nan for r in run.history]
        axes[0].plot(z, mass, color=color, ls=':', lw=1.5, label=run.label+' (mass)')
    common = sorted(set(runs[0].rows) & set(runs[1].rows))
    common = [s for s in common if abs(runs[0].rows[s]['redshift']-runs[1].rows[s]['redshift']) < 1e-3]
    if common:
        axes[1].plot([runs[0].rows[s]['redshift'] for s in common],
                     [runs[0].rows[s]['xHI_volume']-runs[1].rows[s]['xHI_volume'] for s in common], color='#34495e')
    axes[1].axhline(0, color='.5', lw=.7)
    axes[0].axhline(.9, color='.65', lw=.8, ls='--', label='10% volume ionized')
    axes[0].set_ylabel(r'Mean neutral fraction, $\langle x_{\rm HI}\rangle$')
    axes[1].set_ylabel(r'$\Delta\langle x_{\rm HI}\rangle_V$'+'\nPC - original ES')
    axes[1].set_xlabel('Redshift, z')
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    high = max(r['redshift'] for run in runs for r in run.history)
    low = min(r['redshift'] for run in runs for r in run.history)
    axes[1].set_xlim(high+.1, low-.1)
    for ax in axes:
        ax.grid(alpha=.18)
    states = [durations[r.label]['durations']['10-90'] for r in runs]
    subtitle = '10-90% EoR duration: outside the sampled range' if all(s['delta_z'] is None for s in states) else 'EoR durations reported in eor_duration.json'
    fig.suptitle('PC and original ES: ionization histories\n'+subtitle, fontsize=13)
    save_plot(fig, out, 'ionization_history')


def select_pairs(pc, es, snapshots, targets, tolerance):
    common = sorted(set(pc.grids) & set(es.grids) & set(pc.rows) & set(es.rows))
    if not snapshots and common:
        snapshots = sorted(set(common[i] for i in np.linspace(max(0, len(common)//2), len(common)-1, 3).astype(int)))
    same = []
    for s in snapshots:
        if s not in common:
            raise ValueError(f'Snapshot {s} is not available in both grid sets')
        if abs(pc.rows[s]['redshift']-es.rows[s]['redshift']) > 1e-3:
            raise ValueError(f'Snapshot {s} has different redshifts between runs')
        same.append((s, s, None))
    if not targets:
        low = max(min(r['xHI_volume'] for r in run.history) for run in (pc, es))
        high = min(max(r['xHI_volume'] for r in run.history) for run in (pc, es))
        targets = [x for x in (.7, .5, .3) if low <= x <= high]
        if not targets:
            targets = [x for x in (.98, .95, .92) if low <= x <= high]
    matched, skipped = [], []
    for target in targets:
        snaps = [min((s for s in run.grids if s in run.rows),
                     key=lambda s: abs(run.rows[s]['xHI_volume']-target)) for run in (pc, es)]
        means = [run.rows[s]['xHI_volume'] for run, s in zip((pc, es), snaps)]
        if max(abs(m-target) for m in means) > tolerance or abs(means[0]-means[1]) > tolerance:
            skipped.append({'target_xHI': target, 'reason': 'no sufficiently close snapshot pair', 'nearest_means': means})
        else:
            matched.append((*snaps, target))
    return same, matched, skipped


def morphology_plot(pc, es, pairs, out, name, axis, index, box_size, min_cells):
    if not pairs:
        return []
    cached = {}
    def get(run, snap):
        key = run.label, snap
        if key not in cached:
            cached[key] = run.field(snap)
        return cached[key]
    differences = [get(pc, a)-get(es, b) for a, b, _ in pairs]
    limit = max(float(np.max(np.abs(d))) for d in differences) or 1.
    fig, axes = plt.subplots(len(pairs), 3, figsize=(11.5, 3.7*len(pairs)), squeeze=False,
                             layout='constrained')
    stats = []
    for row, ((a, b, target), diff) in enumerate(zip(pairs, differences)):
        x, y = get(pc, a), get(es, b)
        if x.shape != y.shape:
            raise ValueError('Morphology grids must have identical shapes')
        n = x.shape[0]
        slice_index = n//2 if index is None else index
        if not 0 <= slice_index < n:
            raise ValueError('Slice index outside the grid')
        for col, (array, run, snap) in enumerate(((x, pc, a), (y, es, b), (diff, None, None))):
            ax = axes[row, col]
            image = ax.imshow(np.take(array, slice_index, axis=axis).T, origin='lower',
                              extent=(0, box_size, 0, box_size), interpolation='nearest',
                              cmap=NEUTRAL if col < 2 else 'RdBu_r',
                              vmin=0 if col < 2 else -limit, vmax=1 if col < 2 else limit,
                              rasterized=True)
            plane_axes = [c for i, c in enumerate('xyz') if i != axis]
            ax.set_xlabel(plane_axes[0]+' [cMpc/h]')
            ax.set_ylabel(plane_axes[1]+' [cMpc/h]')
            if run:
                meta = run.rows[snap]
                ax.set_title(f'{run.label}: snap {snap}, z={meta["redshift"]:.3f}\n'+
                             r'$\langle x_{\rm HI}\rangle_V$'+f'={array.mean():.5f}', fontsize=11)
            else:
                ax.set_title('PC - original ES\n'+(f'target mean xHI={target:.3f}' if target is not None else 'same redshift'), fontsize=11)
            fig.colorbar(image, ax=ax, fraction=.046, pad=.025,
                         label=r'$x_{\rm HI}$' if col < 2 else r'$\Delta x_{\rm HI}$')
        mx, my = x < .5, y < .5
        union = np.count_nonzero(mx | my)
        pair_stats = {'pc_snapshot': a, 'es_snapshot': b, 'target_xHI': target,
                      'pc_mean_xHI': float(x.mean()), 'es_mean_xHI': float(y.mean()),
                      'rms_difference': float(np.sqrt(np.mean(diff**2))),
                      'mean_abs_difference': float(np.mean(abs(diff))),
                      'ionized_mask_jaccard': float(np.count_nonzero(mx & my)/union) if union else None,
                      'pearson_xHI': float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if np.std(x)>0 and np.std(y)>0 else None}
        for label, mask in (('pc', mx), ('es', my)):
            sizes = periodic_components(mask)
            keep = sizes[sizes >= min_cells]
            pair_stats[label+'_components'] = {'all_count': int(len(sizes)),
                'retained_count': int(len(keep)), 'min_cells': min_cells,
                'largest_cells': int(sizes.max()) if len(sizes) else 0,
                'retained_ionized_volume_fraction': float(keep.sum()/sizes.sum()) if sizes.sum() else None,
                'retained_equivalent_radii_cMpc_h': ((3*keep/(4*np.pi))**(1/3)*box_size/n).tolist()}
        stats.append(pair_stats)
    coordinate = (slice_index+.5)*box_size/n
    fig.suptitle(('Same-redshift morphology' if name.endswith('same_z') else 'Nearest-snapshot mean-neutral-fraction matching')+
                 f'\nFixed {"xyz"[axis]} plane at {coordinate:.3f} cMpc/h; one-cell thickness; no smoothing', fontsize=13)
    save_plot(fig, out, name)
    return stats


def sizes_plot(records, out, name):
    if not records:
        return
    fig, axes = plt.subplots(1, len(records), figsize=(4.3*len(records), 3.7), squeeze=False, layout='constrained')
    for ax, record in zip(axes[0], records):
        for label, key, color in [('PC', 'pc_components', COLORS[0]), ('Original ES', 'es_components', COLORS[1])]:
            r = np.sort(record[key]['retained_equivalent_radii_cMpc_h'])
            if len(r):
                ax.step(r, (np.arange(len(r))+1)/len(r), where='post', color=color,
                        marker='.', markersize=5, label=label)
            else:
                ax.plot([], [], color=color, label=label+' (no retained regions)')
        ax.set_title(f'PC S{record["pc_snapshot"]} / ES S{record["es_snapshot"]}', fontsize=11)
        ax.set_xlabel(r'Component volume-equivalent radius [cMpc/h]')
        ax.set_ylabel('Number-weighted cumulative fraction')
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.18)
    minimum = records[0]['pc_components']['min_cells']
    fig.suptitle(f'Periodic 6-connected regions: xHI < 0.5, at least {minimum} cells\nComponent radius is not a mean free path', fontsize=12)
    save_plot(fig, out, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pc', 'es'):
        parser.add_argument('--'+name+'-run', type=Path)
        parser.add_argument('--'+name+'-log', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshots', nargs='*', type=int, default=[])
    parser.add_argument('--match-xhi', nargs='*', type=float, default=[])
    parser.add_argument('--match-tolerance', type=float, default=.005)
    parser.add_argument('--slice-axis', choices=['x', 'y', 'z'], default='z')
    parser.add_argument('--slice-index', type=int)
    parser.add_argument('--min-component-cells', type=int, default=19)
    args = parser.parse_args()
    if args.match_tolerance <= 0 or args.min_component_cells < 1 or any(not 0 <= x <= 1 for x in args.match_xhi):
        parser.error('Invalid matching tolerance, neutral fractions or component-size cut')
    pc, es = Run('PC', args.pc_run, args.pc_log), Run('Original ES', args.es_run, args.es_log)
    for run, mode in ((pc, '3'), (es, '0')):
        if 'Flag_PhotonConservation' in run.params and int(run.params['Flag_PhotonConservation']) != int(mode):
            raise ValueError(f'{run.label}: expected Flag_PhotonConservation={mode}')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new, empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    for run in (pc, es):
        with (out/('history_pc.csv' if run is pc else 'history_es.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['snapshot', 'redshift', 'xHI_volume', 'xHI_mass', 'source'])
            writer.writeheader()
            writer.writerows(run.history)
    durations = {run.label: duration(run) for run in (pc, es)}
    (out/'eor_duration.json').write_text(json.dumps(durations, indent=2, allow_nan=False)+'\n')
    history_plot((pc, es), durations, out)
    difference = {key: [pc.params.get(key), es.params.get(key)] for key in sorted(set(pc.params)|set(es.params))
                  if str(pc.params.get(key)) != str(es.params.get(key))}
    report = {'pc_input': str(pc.location), 'es_input': str(es.location),
              'pc_log': pc.log, 'es_log': es.log, 'parameter_differences': difference,
              'interpretation': 'Complete PC versus original ES models; source integration and recombination closures differ.',
              'morphology_status': 'not_run_missing_grids'}
    if pc.grids and es.grids:
        if ('SimulationDir' in pc.params and 'SimulationDir' in es.params and
                str(pc.params['SimulationDir']).rstrip('/') != str(es.params['SimulationDir']).rstrip('/')):
            raise ValueError('Spatial differences require the same simulation realization; SimulationDir differs')
        for key in ('BoxSize', 'Hubble_h', 'OmegaM', 'OmegaLambda', 'OmegaR', 'OmegaK', 'wLambda'):
            if key in pc.params and key in es.params and not np.isclose(float(pc.params[key]), float(es.params[key]), rtol=1e-8, atol=1e-12):
                raise ValueError(f'Cannot spatially compare different {key}')
        if 'BoxSize' not in pc.params:
            raise ValueError('BoxSize metadata is required for morphology')
        same, matched, skipped = select_pairs(pc, es, args.snapshots, args.match_xhi, args.match_tolerance)
        for name, pairs in [('same_z', same), ('matched_xhi', matched)]:
            records = morphology_plot(pc, es, pairs, out, 'morphology_'+name,
                                      'xyz'.index(args.slice_axis), args.slice_index,
                                      float(pc.params['BoxSize']), args.min_component_cells)
            report[name] = records
            sizes_plot(records, out, 'component_sizes_'+name)
        report.update(morphology_status='completed', skipped_matching_targets=skipped,
                      match_tolerance=args.match_tolerance)
    (out/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for run in (pc, es):
        final = run.history[-1]
        print(f'{run.label}: {len(run.history)} history samples, final z={final["redshift"]:.6g}, '
              f'xHI_V={final["xHI_volume"]:.8f}')
        print('  EoR 10-90%:', durations[run.label]['durations']['10-90'])
    print('Morphology:', report['morphology_status'])
    print('Figures and data:', out)


if __name__ == '__main__':
    main()
