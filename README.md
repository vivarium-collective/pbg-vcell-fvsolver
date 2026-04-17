# pbg-vcell-fvsolver

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
wrapper around [`pyvcell`](https://github.com/virtualcell/pyvcell) and
[`pyvcell-fvsolver`](https://pypi.org/project/pyvcell-fvsolver/) — the
Python bindings to VCell's native **finite-volume PDE solver** — for
3D reaction-diffusion simulation.

Define a reaction network in Antimony, describe diffusion coefficients
and initial concentrations, hand it to `VCellFVProcess`, and the
wrapper drives VCell's finite-volume kernel to produce per-species 3D
concentration fields that compose naturally with other PBG processes.

## What it does

- Accepts Antimony text for the reaction network (species, reactions, rates).
- Builds a VCML model programmatically, adds a cubic/rectangular spatial
  geometry, species-to-compartment mappings with diffusion coefficients,
  and a FV simulation configuration.
- Calls `pyvcell.vcml.simulate`, which converts VCML → FV input files and
  invokes `pyvcell_fvsolver.solve()` (a C++ binding to the same native
  solver VCell ships).
- Caches the full 4D output and streams 3D voxel snapshots on each
  `update(interval)` through the PBG composite.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
# for demo report / tests:
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
    species={
        'A': {'init_conc': 'exp(-((x-5)^2 + (y-5)^2 + (z-5)^2))',
              'diff_coef': 0.5},
        'B': {'init_conc': '0.0', 'diff_coef': 0.5},
    },
    compartment='cell',
    extent=(10.0, 10.0, 10.0),    # µm
    mesh_size=(22, 22, 22),       # voxels
    duration=2.0,
    output_time_step=0.25,
    track_species=['A', 'B'],
    interval=0.25,
)

sim = Composite({'state': doc}, core=core)
sim.run(2.0)

print(sim.state['stores']['means'])          # per-species mean [µM]
field_A = sim.state['stores']['fields']['A'] # 22×22×22 float grid
```

## API Reference

### `VCellFVProcess` (process-bigraph Process)

| Port      | Type                  | Description |
|-----------|-----------------------|-------------|
| `fields`  | `overwrite[map[list]]` | Per-species 3D concentration grid (nested lists, shape `nx × ny × nz`) |
| `means`   | `overwrite[map[float]]`| Domain-mean concentration per species |
| `totals`  | `overwrite[map[float]]`| Sum over all voxels (proxy for total mass) |
| `time`    | `overwrite[float]`    | Current simulated time (seconds) |
| `num_snapshots` | `overwrite[integer]` | Number of solver output times cached |
| `snapshot_index` | `overwrite[integer]` | Index into the cached snapshots |

**Config**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `antimony` | str | `''` | Antimony source for the reaction network. |
| `compartment` | str | `'cell'` | Compartment name to map onto the 3D domain. |
| `init_concs` | `map[string]` | `{}` | Species → init_conc **expression** (may use `x,y,z`). |
| `diff_coefs` | `map[float]` | `{}` | Species → diffusion coefficient (µm²/s). |
| `extent_x/y/z` | float | `10.0` | Physical extents of the cubic domain. |
| `mesh_x/y/z` | int | `20` | Voxel grid dimensions. |
| `duration` | float | `1.0` | Total simulated time to pre-compute. |
| `output_time_step` | float | `0.1` | Solver snapshot interval. |
| `track_species` | `list[string]` | `[]` | Subset to emit; `[]` = auto (all model species). |
| `quiet` | bool | `True` | Suppress JVM/libvcell stdout noise. |

**Helper methods**

- `get_mesh_shape() -> (nx, ny, nz)`
- `get_extent()     -> (Lx, Ly, Lz)`
- `get_time_points() -> list[float]`
- `get_channel_ids() -> list[str]` — all solver channels including
  auxiliary outputs (`region_mask`, `x/y/z`, initial conditions, …).

### `make_rd_document(...)` (factory)

Builds a composite document dict wiring `VCellFVProcess` to a `stores`
tree and a `ram-emitter` for time-series collection. Returns a dict
suitable for `Composite({'state': doc}, core=core)`.

## Architecture

```
Antimony  ──► pyvcell ──► libvcell ──► VCML ──► .fvinput + .vcg
                                                        │
                                                        ▼
                       pyvcell-fvsolver.solve() [C++]
                                   │
                                   ▼
              zarr datastore (4D: t × x × y × z × species)
                                   │
             pyvcell.sim_results.Result ◄─── VCellFVProcess
                                   │
                                   ▼
                        PBG outputs (fields, means, totals)
```

`VCellFVProcess` uses a pre-run bridge pattern: it runs the full
batch simulation once on the first `update()` call, caches the
zarr-backed `Result`, then emits per-snapshot fields as composite
time advances. This matches VCell's native execution model — the
finite-volume solver is a batch engine, not an interactive stepper.

## Demo

```bash
source .venv/bin/activate
python demo/demo_report.py
```

Produces `demo/report.html` (~3.5 MB) with three distinct 3D
reaction-diffusion scenarios:

1. **Gaussian Pulse Diffusion** — pure Fickian diffusion benchmark.
2. **Reaction-Diffusion Cascade** — A → B → C with different D's.
3. **Two-Source Mixing** — A + B → C with opposing diffusion fronts.

Each section includes an interactive Three.js voxel-cloud viewer
(drag/rotate/play), Plotly time-series charts, a bigraph-viz PNG of
the composite architecture, and a collapsible JSON tree of the PBG
document. The report auto-opens in Safari on macOS.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## License

MIT.
