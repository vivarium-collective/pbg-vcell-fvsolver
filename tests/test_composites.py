"""Integration tests: full composite assembly and run."""

import numpy as np
from process_bigraph import Composite


def test_composite_runs_and_advances_time(core, antimony_ab):
    from pbg_vcell_fvsolver import make_rd_document
    doc = make_rd_document(
        antimony=antimony_ab,
        species={'A': {'init_conc': '10.0', 'diff_coef': 0.0},
                 'B': {'init_conc': '0.0', 'diff_coef': 0.0}},
        compartment='cell',
        extent=(5.0, 5.0, 5.0),
        mesh_size=(5, 5, 5),
        duration=1.0,
        output_time_step=0.5,
        track_species=['A', 'B'],
        interval=0.5,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(1.0)
    assert sim.state['stores']['last_time'] >= 0.99
    assert sim.state['stores']['means']['A'] < 10.0


def test_composite_field_shape(core, antimony_ab):
    from pbg_vcell_fvsolver import make_rd_document
    doc = make_rd_document(
        antimony=antimony_ab,
        species={'A': {'init_conc': '5.0', 'diff_coef': 0.1},
                 'B': {'init_conc': '0.0', 'diff_coef': 0.1}},
        compartment='cell',
        extent=(4.0, 4.0, 4.0),
        mesh_size=(6, 6, 6),
        duration=0.4,
        output_time_step=0.2,
        track_species=['A'],
        interval=0.2,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(0.2)
    arr = np.asarray(sim.state['stores']['fields']['A'])
    assert arr.shape == (6, 6, 6)


def test_composite_snapshot_advances(core, antimony_ab):
    """Successive updates should land on successive solver snapshots."""
    from pbg_vcell_fvsolver import make_rd_document
    doc = make_rd_document(
        antimony=antimony_ab,
        species={'A': {'init_conc': '10.0', 'diff_coef': 0.0},
                 'B': {'init_conc': '0.0', 'diff_coef': 0.0}},
        compartment='cell',
        extent=(4.0, 4.0, 4.0),
        mesh_size=(4, 4, 4),
        duration=1.0,
        output_time_step=0.25,
        track_species=['A'],
        interval=0.25,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(0.25)
    idx_after_one = sim.state['stores']['snapshot_index']
    sim.run(0.25)
    idx_after_two = sim.state['stores']['snapshot_index']
    assert idx_after_two > idx_after_one
