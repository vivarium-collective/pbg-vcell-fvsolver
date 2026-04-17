# pbg-vcell-fvsolver

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
wrapper around [`pyvcell`](https://github.com/virtualcell/pyvcell) and
[`pyvcell-fvsolver`](https://pypi.org/project/pyvcell-fvsolver/) —
VCell's native finite-volume 3D reaction-diffusion PDE solver as a PBG
`Process`.

**[View Interactive Demo Report](https://vivarium-collective.github.io/pbg-vcell-fvsolver/)** — Gaussian-pulse diffusion, an A→B→C cascade, and a two-source binding reaction, with PyVista vtk.js 3D viewers, Plotly charts, and bigraph architecture diagrams.

## What it does

Accepts an Antimony reaction network, adds a box geometry + diffusion
coefficients + initial-concentration expressions, calls
`pyvcell.simulate` (which compiles to VCML → FV input files and
invokes `pyvcell_fvsolver.solve()`), caches the 4D result, and streams
per-species 3D voxel fields as the composite advances.

## Installation

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[demo,test]"
```

## Quick Start

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_vcell_fvsolver import VCellFVProcess, make_rd_document

ANTIMONY = """
    compartment ec;
    compartment cell;
    species A in cell;
    species B in cell;
    A -> B; k*A
    k = 0.3
    A = 10
"""

core = allocate_core()
core.register_link('VCellFVProcess', VCellFVProcess)
core.register_link('ram-emitter', RAMEmitter)

doc = make_rd_document(
    antimony=ANTIMONY,
    species={'A': {'init_conc': 'exp(-((x-5)^2+(y-5)^2+(z-5)^2))',
                   'diff_coef': 0.5},
             'B': {'init_conc': '0.0', 'diff_coef': 0.5}},
    extent=(10.0, 10.0, 10.0),
    mesh_size=(22, 22, 22),
    duration=2.0,
    output_time_step=0.25,
    track_species=['A', 'B'],
    interval=0.25,
)

sim = Composite({'state': doc}, core=core)
sim.run(2.0)
print(sim.state['stores']['means'])         # per-species mean [µM]
field_A = sim.state['stores']['fields']['A']  # 22×22×22 float grid
```

## API

**`VCellFVProcess`** config: `antimony`, `compartment`, `init_concs`
(map `name → init_conc_expression`), `diff_coefs` (map `name → float`),
`extent_{x,y,z}`, `mesh_{x,y,z}`, `duration`, `output_time_step`,
`track_species`, `quiet`.

**Outputs** (all `overwrite`): `fields` (per-species 3D grid as nested
lists), `means`, `totals`, `time`, `num_snapshots`, `snapshot_index`.

**Helpers**: `get_mesh_shape()`, `get_extent()`, `get_channel_ids()`,
`get_time_points()`.

**`make_rd_document(...)`** — composite factory wiring the process to
stores and a `ram-emitter`.

## Architecture

```
Antimony ─► pyvcell ─► VCML ─► .fvinput/.vcg
                                    │
                                    ▼
              pyvcell-fvsolver.solve()  [native C++]
                                    │
                                    ▼
                         zarr datastore (4D)
                                    │
                                    ▼
                       VCellFVProcess ─► PBG ports
```

`VCellFVProcess` pre-runs the batch solver on the first `update()`
call and then indexes cached snapshots — matching VCell's native
execution model (the FV solver is a batch engine, not a stepper).

## Demo

```bash
python demo/demo_report.py
```

Produces `demo/report.html` with three configurations, interactive
PyVista vtk.js 3D viewers (same rendering stack as pyvcell's trame
widget), Plotly charts, bigraph-viz diagrams, and per-experiment
model/geometry tables.

## Tests

```bash
pytest -q
```

## License

MIT.
