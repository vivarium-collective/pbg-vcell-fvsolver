"""Pre-built composite document factories for VCell FV reaction-diffusion."""


def make_rd_document(
    antimony,
    species=None,
    init_concs=None,
    diff_coefs=None,
    compartment='cell',
    extent=(10.0, 10.0, 10.0),
    mesh_size=(20, 20, 20),
    duration=1.0,
    output_time_step=0.1,
    track_species=None,
    interval=0.1,
    quiet=True,
):
    """Create a composite document for a 3D reaction-diffusion simulation.

    Args:
        antimony: Antimony source defining species, reactions, parameters.
        species: Convenience dict {name: {'init_conc', 'diff_coef'}}. If
                 given, populates init_concs and diff_coefs.
        init_concs: Explicit {name: init_conc_expression_string}.
        diff_coefs: Explicit {name: diffusion_coefficient_float}.
        compartment: Compartment name in the antimony to map onto the domain.
        extent: (Lx, Ly, Lz) in micrometers.
        mesh_size: (nx, ny, nz) voxel grid.
        duration: Total simulated time (seconds) to pre-compute.
        output_time_step: Snapshot cadence.
        track_species: Subset of species to emit. None/empty = all.
        interval: PBG update interval between snapshots.
        quiet: Suppress JVM/solver stdout during pre-run.

    Returns:
        Composite document dict ready to pass to Composite({'state': doc}).
    """
    if track_species is None:
        track_species = []
    if init_concs is None:
        init_concs = {}
    if diff_coefs is None:
        diff_coefs = {}
    if species:
        for name, spec in species.items():
            init_concs.setdefault(name, str(spec.get('init_conc', '0.0')))
            diff_coefs.setdefault(name, float(spec.get('diff_coef', 0.0)))
    init_concs = {k: str(v) for k, v in init_concs.items()}
    diff_coefs = {k: float(v) for k, v in diff_coefs.items()}

    emit_schema = {
        'time': 'float',
        'num_snapshots': 'integer',
        'snapshot_index': 'integer',
        'means': 'map[float]',
        'totals': 'map[float]',
    }

    emit_inputs = {
        'time': ['global_time'],
        'num_snapshots': ['stores', 'num_snapshots'],
        'snapshot_index': ['stores', 'snapshot_index'],
        'means': ['stores', 'means'],
        'totals': ['stores', 'totals'],
    }

    return {
        'rd': {
            '_type': 'process',
            'address': 'local:VCellFVProcess',
            'config': {
                'antimony': antimony,
                'compartment': compartment,
                'extent_x': float(extent[0]),
                'extent_y': float(extent[1]),
                'extent_z': float(extent[2]),
                'mesh_x': int(mesh_size[0]),
                'mesh_y': int(mesh_size[1]),
                'mesh_z': int(mesh_size[2]),
                'duration': float(duration),
                'output_time_step': float(output_time_step),
                'init_concs': init_concs,
                'diff_coefs': diff_coefs,
                'track_species': list(track_species),
                'quiet': quiet,
            },
            'interval': float(interval),
            'inputs': {},
            'outputs': {
                'fields': ['stores', 'fields'],
                'means': ['stores', 'means'],
                'totals': ['stores', 'totals'],
                'time': ['stores', 'last_time'],
                'num_snapshots': ['stores', 'num_snapshots'],
                'snapshot_index': ['stores', 'snapshot_index'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': emit_schema},
            'inputs': emit_inputs,
        },
    }
