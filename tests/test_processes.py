"""Unit tests for VCellFVProcess."""

import numpy as np


def test_process_instantiation(core, antimony_ab):
    from pbg_vcell_fvsolver import VCellFVProcess
    proc = VCellFVProcess(
        config={
            'antimony': antimony_ab,
            'compartment': 'cell',
            'init_concs': {'A': '10.0', 'B': '0.0'},
            'diff_coefs': {'A': 0.0, 'B': 0.0},
            'duration': 0.1,
            'output_time_step': 0.1,
            'extent_x': 5.0, 'extent_y': 5.0, 'extent_z': 5.0,
            'mesh_x': 6, 'mesh_y': 6, 'mesh_z': 6,
            'track_species': ['A', 'B'],
        },
        core=core,
    )
    assert proc.get_mesh_shape() == (6, 6, 6)
    assert proc.get_extent() == (5.0, 5.0, 5.0)


def test_initial_state_shape(core, antimony_ab):
    from pbg_vcell_fvsolver import VCellFVProcess
    proc = VCellFVProcess(
        config={
            'antimony': antimony_ab,
            'compartment': 'cell',
            'init_concs': {'A': '10.0', 'B': '0.0'},
            'diff_coefs': {'A': 0.2, 'B': 0.2},
            'duration': 0.2,
            'output_time_step': 0.1,
            'extent_x': 5.0, 'extent_y': 5.0, 'extent_z': 5.0,
            'mesh_x': 6, 'mesh_y': 6, 'mesh_z': 6,
            'track_species': ['A', 'B'],
        },
        core=core,
    )
    s0 = proc.initial_state()
    assert set(s0.keys()) >= {'fields', 'means', 'totals', 'time',
                              'num_snapshots', 'snapshot_index'}
    arr = np.asarray(s0['fields']['A'])
    assert arr.shape == (6, 6, 6)
    assert s0['snapshot_index'] == 0
    assert s0['time'] == 0.0
    assert np.isclose(s0['means']['A'], 10.0)


def test_reaction_decay(core, antimony_ab):
    """A -> B with k=0.3 and no diffusion: A should decay over time."""
    from pbg_vcell_fvsolver import VCellFVProcess
    proc = VCellFVProcess(
        config={
            'antimony': antimony_ab,
            'compartment': 'cell',
            'init_concs': {'A': '10.0', 'B': '0.0'},
            'diff_coefs': {'A': 0.0, 'B': 0.0},
            'duration': 1.0,
            'output_time_step': 0.5,
            'extent_x': 5.0, 'extent_y': 5.0, 'extent_z': 5.0,
            'mesh_x': 6, 'mesh_y': 6, 'mesh_z': 6,
            'track_species': ['A', 'B'],
        },
        core=core,
    )
    s0 = proc.initial_state()
    s1 = proc.update({}, 1.0)
    assert s1['means']['A'] < s0['means']['A']
    assert s1['means']['B'] > s0['means']['B']


def test_mass_conservation_reaction(core, antimony_ab):
    """A -> B should conserve total A+B within numerical tolerance."""
    from pbg_vcell_fvsolver import VCellFVProcess
    proc = VCellFVProcess(
        config={
            'antimony': antimony_ab,
            'compartment': 'cell',
            'init_concs': {'A': '5.0', 'B': '0.0'},
            'diff_coefs': {'A': 0.0, 'B': 0.0},
            'duration': 1.0,
            'output_time_step': 0.5,
            'extent_x': 4.0, 'extent_y': 4.0, 'extent_z': 4.0,
            'mesh_x': 5, 'mesh_y': 5, 'mesh_z': 5,
            'track_species': ['A', 'B'],
        },
        core=core,
    )
    s0 = proc.initial_state()
    s1 = proc.update({}, 1.0)
    total0 = s0['means']['A'] + s0['means']['B']
    total1 = s1['means']['A'] + s1['means']['B']
    assert abs(total0 - total1) / total0 < 0.02


def test_time_points(core, antimony_ab):
    from pbg_vcell_fvsolver import VCellFVProcess
    proc = VCellFVProcess(
        config={
            'antimony': antimony_ab,
            'compartment': 'cell',
            'init_concs': {'A': '1.0', 'B': '0.0'},
            'diff_coefs': {'A': 0.0, 'B': 0.0},
            'duration': 1.0,
            'output_time_step': 0.25,
            'extent_x': 4.0, 'extent_y': 4.0, 'extent_z': 4.0,
            'mesh_x': 4, 'mesh_y': 4, 'mesh_z': 4,
            'track_species': ['A'],
        },
        core=core,
    )
    proc.initial_state()
    tps = proc.get_time_points()
    assert tps[0] == 0.0
    assert tps[-1] >= 1.0 - 1e-9
    assert len(tps) >= 5


def test_channel_discovery(core, antimony_ab):
    from pbg_vcell_fvsolver import VCellFVProcess
    proc = VCellFVProcess(
        config={
            'antimony': antimony_ab,
            'compartment': 'cell',
            'init_concs': {'A': '1.0', 'B': '0.0'},
            'diff_coefs': {'A': 0.0, 'B': 0.0},
            'duration': 0.1,
            'output_time_step': 0.1,
            'extent_x': 4.0, 'extent_y': 4.0, 'extent_z': 4.0,
            'mesh_x': 4, 'mesh_y': 4, 'mesh_z': 4,
            'track_species': [],  # auto-discover
        },
        core=core,
    )
    proc.initial_state()
    channels = proc.get_channel_ids()
    assert 'A' in channels
    assert 'B' in channels
