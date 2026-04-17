"""VCell finite-volume 3D reaction-diffusion Process wrapper.

Uses the bridge pattern: each Process instance pre-runs a full spatial
simulation with pyvcell (which in turn shells out to the native
pyvcell-fvsolver PDE solver), caches all time snapshots, then streams
them back one interval at a time through update().

The pre-run approach is appropriate here because VCell's finite-volume
solver is a batch engine — it reads .fvinput + .vcg files, runs the
complete simulation, and writes a zarr datastore. Re-running with
mutated initial conditions on each PBG step would be an order of
magnitude more expensive than solving once and indexing snapshots.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from process_bigraph import Process


def _suppress_jvm_noise():
    """Route libvcell JVM stdout to /dev/null during the pre-run.

    libvcell prints verbose Log4j warnings every time it converts VCML.
    We don't want those polluting PBG runs, so we dup /dev/null over
    fd 1 for the solve, then restore.
    """
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return saved


def _restore_stdout(saved):
    os.dup2(saved, 1)
    os.close(saved)


class VCellFVProcess(Process):
    """Bridge Process wrapping the VCell finite-volume 3D RD solver.

    Builds a spatial biomodel from an Antimony string + a box domain
    geometry, runs the full simulation with `pyvcell.vcml.simulate`
    (which invokes `pyvcell_fvsolver.solve` internally), and emits
    per-species 3D concentration fields snapshot-by-snapshot as the
    composite's global time advances.

    Config:
        antimony:            Antimony source for the reaction network.
        compartment:         Name of the compartment that maps onto the
                             spatial domain. Must exist in the antimony.
        extent:              Physical size of the cubic/rect domain in um.
        mesh_size:           Voxel grid dimensions (nx, ny, nz).
        duration:            Total simulated time to pre-compute.
        output_time_step:    Snapshot interval written by the solver.
        species:             Per-species {init_conc, diff_coef}. Missing
                             species fall back to init_conc=0, diff_coef=0.
        track_species:       Subset of species to emit as fields/means.
                             Empty -> all model species.
        quiet:               Suppress JVM/solver stdout during pre-run.
    """

    config_schema = {
        'antimony': {'_type': 'string', '_default': ''},
        'compartment': {'_type': 'string', '_default': 'cell'},
        'extent_x': {'_type': 'float', '_default': 10.0},
        'extent_y': {'_type': 'float', '_default': 10.0},
        'extent_z': {'_type': 'float', '_default': 10.0},
        'mesh_x': {'_type': 'integer', '_default': 20},
        'mesh_y': {'_type': 'integer', '_default': 20},
        'mesh_z': {'_type': 'integer', '_default': 20},
        'duration': {'_type': 'float', '_default': 1.0},
        'output_time_step': {'_type': 'float', '_default': 0.1},
        'init_concs': {
            '_type': 'map[string]',
            '_default': {},
        },
        'diff_coefs': {
            '_type': 'map[float]',
            '_default': {},
        },
        'track_species': {
            '_type': 'list[string]',
            '_default': [],
        },
        'quiet': {'_type': 'boolean', '_default': True},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._results = None
        self._channel_ids = []
        self._tracked = []
        self._time_points = []
        self._current_time = 0.0
        self._workspace = None

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'fields': 'overwrite[map[list]]',
            'means': 'overwrite[map[float]]',
            'totals': 'overwrite[map[float]]',
            'time': 'overwrite[float]',
            'num_snapshots': 'overwrite[integer]',
            'snapshot_index': 'overwrite[integer]',
        }

    def initial_state(self):
        self._run()
        return self._snapshot_at(0.0)

    def _run(self):
        """Pre-run the full spatial simulation and cache the Result."""
        if self._results is not None:
            return

        import pyvcell.vcml as vc

        self._workspace = Path(tempfile.mkdtemp(prefix='pbg_vcell_'))
        vc.set_workspace_dir(self._workspace)

        saved = _suppress_jvm_noise() if self.config['quiet'] else None
        try:
            biomodel = vc.load_antimony_str(self.config['antimony'])

            ex = (float(self.config['extent_x']),
                  float(self.config['extent_y']),
                  float(self.config['extent_z']))
            geo = vc.Geometry(name='geo', origin=(0.0, 0.0, 0.0),
                              extent=ex, dim=3)
            medium = geo.add_background(name='background')
            app = biomodel.add_application('app1', geometry=geo)

            target_compartment = self.config['compartment']
            comp = biomodel.model.get_compartment(target_compartment)
            if comp is None:
                comps = [c.name for c in biomodel.model.compartments]
                raise ValueError(
                    f"compartment '{target_compartment}' not in antimony. "
                    f"Available: {comps}")
            app.map_compartment(compartment=comp, domain=medium)

            init_concs = self.config['init_concs']
            diff_coefs = self.config['diff_coefs']
            model_species = {s.name for s in biomodel.model.species}
            mappings = []
            for name in model_species:
                mappings.append(vc.SpeciesMapping(
                    species_name=name,
                    init_conc=str(init_concs.get(name, '0.0')),
                    diff_coef=float(diff_coefs.get(name, 0.0)),
                ))
            app.species_mappings = mappings

            mesh = (int(self.config['mesh_x']),
                    int(self.config['mesh_y']),
                    int(self.config['mesh_z']))
            app.add_sim(
                name='sim1',
                duration=float(self.config['duration']),
                output_time_step=float(self.config['output_time_step']),
                mesh_size=mesh,
            )

            self._results = vc.simulate(biomodel=biomodel, simulation='sim1')
        finally:
            if saved is not None:
                _restore_stdout(saved)

        self._channel_ids = list(self._results.get_channel_ids())
        tracked = list(self.config['track_species'])
        if not tracked:
            tracked = [c for c in self._channel_ids
                       if c in {s.name for s in biomodel.model.species}]
        self._tracked = [c for c in tracked if c in self._channel_ids]

        tp = self._results.get_time_axis()
        self._time_points = list(tp) if isinstance(tp, list) else [float(tp)]

    def _nearest_snapshot(self, t):
        if not self._time_points:
            return 0
        arr = np.asarray(self._time_points)
        idx = int(np.argmin(np.abs(arr - t)))
        return idx

    def _snapshot_at(self, t):
        idx = self._nearest_snapshot(t)
        fields = {}
        means = {}
        totals = {}
        for ch in self._tracked:
            arr = self._results.get_slice(channel_id=ch, time_index=idx)
            fields[ch] = arr.astype(float).tolist()
            means[ch] = float(arr.mean())
            totals[ch] = float(arr.sum())
        return {
            'fields': fields,
            'means': means,
            'totals': totals,
            'time': float(self._time_points[idx]),
            'num_snapshots': len(self._time_points),
            'snapshot_index': idx,
        }

    def update(self, state, interval):
        self._run()
        self._current_time += float(interval)
        return self._snapshot_at(self._current_time)

    def get_mesh_shape(self):
        """Return (nx, ny, nz) of the finite-volume grid."""
        return (int(self.config['mesh_x']),
                int(self.config['mesh_y']),
                int(self.config['mesh_z']))

    def get_extent(self):
        """Return (Lx, Ly, Lz) of the spatial domain."""
        return (float(self.config['extent_x']),
                float(self.config['extent_y']),
                float(self.config['extent_z']))

    def get_channel_ids(self):
        """Return all channel ids produced by the solver (species + aux)."""
        self._run()
        return list(self._channel_ids)

    def get_time_points(self):
        """Return the list of absolute solver output times."""
        self._run()
        return list(self._time_points)

    def __del__(self):
        try:
            if self._results is not None:
                self._results.cleanup()
        except Exception:
            pass
