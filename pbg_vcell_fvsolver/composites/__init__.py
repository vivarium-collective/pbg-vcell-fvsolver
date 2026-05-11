"""VCell-FV composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_rd_document(antimony=…, ...)` builds a
   PBG state-dict programmatically for callers that want full control over
   the antimony model, geometry, mesh, and wiring. Used by
   `demo/demo_report.py` for the three RD experiments.

2. **Declarative `*.composite.yaml`** — sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The dashboard's
   composite explorer discovers these automatically once the package is
   installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_vcell_fvsolver.processes import VCellFVProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factory (legacy / programmatic API)
# ---------------------------------------------------------------------------


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


def register_vcell_fvsolver(core=None):
    """Return a core with VCellFVProcess, the RAM emitter, and the
    RDFieldPlots Visualization registered."""
    if core is None:
        core = allocate_core()
    core.register_link('VCellFVProcess', VCellFVProcess)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    try:
        from pbg_vcell_fvsolver.visualizations import RDFieldPlots
        core.register_link('RDFieldPlots', RDFieldPlots)
    except ImportError:
        # pbg-superpowers not installed; Visualization-wired composites
        # won't be buildable but the rest of the package still works.
        pass
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_vcell_fvsolver() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_vcell_fvsolver()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
