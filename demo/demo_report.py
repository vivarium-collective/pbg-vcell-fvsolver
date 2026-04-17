"""Demo: VCell FV solver multi-configuration 3D reaction-diffusion report.

Runs three distinct 3D reaction-diffusion simulations through the
process-bigraph wrapper (pure diffusion of a Gaussian pulse, a linear
A -> B -> C cascade, and a two-source mixing scenario), then emits a
self-contained HTML report with interactive voxel-cloud 3D viewers,
Plotly time-series charts, colored bigraph-viz architecture diagrams,
and a navigatable PBG document tree per configuration.
"""

import json
import os
import base64
import tempfile
import time
import numpy as np

from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_vcell_fvsolver import VCellFVProcess, make_rd_document


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'diffusion',
        'title': 'Gaussian Pulse Diffusion',
        'subtitle': 'Single-species diffusion from a point source',
        'description': (
            'A Gaussian concentration pulse of species A centered at the '
            'middle of the cubic domain diffuses outward under Fick\'s law. '
            'No reactions — a pure diffusion benchmark of the VCell finite '
            'volume PDE solver. The peak concentration decays while the '
            'distribution spreads uniformly over the domain.'
        ),
        'antimony': """
            compartment ec;
            compartment cell;
            species A in cell;
            A = 0
        """,
        'init_concs': {
            'A': '50 * exp(-((x-5)^2 + (y-5)^2 + (z-5)^2) / 1.5)',
        },
        'diff_coefs': {'A': 1.0},
        'track': ['A'],
        'primary': 'A',
        'extent': (10.0, 10.0, 10.0),
        'mesh_size': (22, 22, 22),
        'duration': 8.0,
        'output_time_step': 0.4,
        'n_snapshots': 20,
        'camera': [15.0, 10.0, 15.0],
        'color_scheme': 'indigo',
    },
    {
        'id': 'cascade',
        'title': 'Reaction-Diffusion Cascade',
        'subtitle': 'A → B → C with distinct diffusion coefficients',
        'description': (
            'A linear reaction cascade: A decays into B, which decays into C. '
            'All three species diffuse, but with very different coefficients '
            '(A slow, B medium, C fast). The wrapper drives VCell\'s finite '
            'volume PDE solver to couple first-order kinetics with Fickian '
            'transport across a 3D Cartesian mesh, producing three distinct '
            'spatial profiles that emerge and decay in sequence.'
        ),
        'antimony': """
            compartment ec;
            compartment cell;
            species A in cell;
            species B in cell;
            species C in cell;
            J1: A -> B; k1*A
            J2: B -> C; k2*B
            k1 = 0.6
            k2 = 0.3
            A = 0
            B = 0
            C = 0
        """,
        'init_concs': {
            'A': '30 * exp(-((x-5)^2 + (y-5)^2 + (z-5)^2) / 2.0)',
            'B': '0.0',
            'C': '0.0',
        },
        'diff_coefs': {'A': 0.2, 'B': 0.6, 'C': 1.2},
        'track': ['A', 'B', 'C'],
        'primary': 'B',
        'extent': (10.0, 10.0, 10.0),
        'mesh_size': (22, 22, 22),
        'duration': 6.0,
        'output_time_step': 0.3,
        'n_snapshots': 20,
        'camera': [15.0, 10.0, 15.0],
        'color_scheme': 'emerald',
    },
    {
        'id': 'mixing',
        'title': 'Two-Source Mixing',
        'subtitle': 'Opposing diffusion fronts with a binding reaction',
        'description': (
            'Two Gaussian sources of species A and species B sit on opposite '
            'corners of the cubic domain. As both diffuse inward they collide '
            'in the middle where a bimolecular association reaction '
            'A + B -> C traps product along the interface. The VCell finite '
            'volume solver handles the coupled transport and nonlinear '
            'reaction across the 3D mesh.'
        ),
        'antimony': """
            compartment ec;
            compartment cell;
            species A in cell;
            species B in cell;
            species C in cell;
            J1: A + B -> C; k1*A*B
            k1 = 1.5
            A = 0; B = 0; C = 0
        """,
        'init_concs': {
            'A': '40 * exp(-((x-2)^2 + (y-2)^2 + (z-2)^2) / 1.5)',
            'B': '40 * exp(-((x-8)^2 + (y-8)^2 + (z-8)^2) / 1.5)',
            'C': '0.0',
        },
        'diff_coefs': {'A': 0.6, 'B': 0.6, 'C': 0.1},
        'track': ['A', 'B', 'C'],
        'primary': 'C',
        'extent': (10.0, 10.0, 10.0),
        'mesh_size': (22, 22, 22),
        'duration': 6.0,
        'output_time_step': 0.3,
        'n_snapshots': 20,
        'camera': [15.0, 10.0, 15.0],
        'color_scheme': 'rose',
    },
]


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
}


# ── Simulation runner ───────────────────────────────────────────────


def run_simulation(cfg_entry):
    """Run a single RD simulation, collecting per-snapshot fields and stats."""
    core = allocate_core()
    core.register_link('VCellFVProcess', VCellFVProcess)
    core.register_link('ram-emitter', RAMEmitter)

    t0 = time.perf_counter()
    proc = VCellFVProcess(
        config={
            'antimony': cfg_entry['antimony'],
            'compartment': 'cell',
            'init_concs': cfg_entry['init_concs'],
            'diff_coefs': cfg_entry['diff_coefs'],
            'duration': float(cfg_entry['duration']),
            'output_time_step': float(cfg_entry['output_time_step']),
            'extent_x': float(cfg_entry['extent'][0]),
            'extent_y': float(cfg_entry['extent'][1]),
            'extent_z': float(cfg_entry['extent'][2]),
            'mesh_x': int(cfg_entry['mesh_size'][0]),
            'mesh_y': int(cfg_entry['mesh_size'][1]),
            'mesh_z': int(cfg_entry['mesh_size'][2]),
            'track_species': cfg_entry['track'],
        },
        core=core,
    )
    state0 = proc.initial_state()
    time_points = proc.get_time_points()

    # Subsample to n_snapshots if solver produced more
    n = min(cfg_entry['n_snapshots'], len(time_points))
    indices = np.linspace(0, len(time_points) - 1, n).astype(int).tolist()

    snapshots = []
    for idx in indices:
        # Re-read at each sampled index using internal result cache
        snap_state = proc._snapshot_at(time_points[idx])
        snapshots.append({
            'time': round(snap_state['time'], 4),
            'fields': snap_state['fields'],
            'means': snap_state['means'],
            'totals': snap_state['totals'],
        })

    runtime = time.perf_counter() - t0
    return snapshots, runtime


# ── Voxel point cloud packing ───────────────────────────────────────


def pack_voxel_frames(snapshots, primary, extent, mesh_size, threshold=0.02):
    """Convert per-snapshot 3D fields into a compact per-voxel format for JS.

    Only voxels above `threshold * max(primary_field)` are rendered.
    Returns:
      positions: flat (N, 3) world-space voxel centers (time-invariant)
      concs:     list[N] of normalized [0,1] primary concentrations per snapshot
      frames:    list of {time, mean, max} metadata per snapshot
      max_c:     global max concentration for colorbar
    """
    nx, ny, nz = mesh_size
    Lx, Ly, Lz = extent
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    xs = np.arange(nx) * dx + dx / 2 - Lx / 2
    ys = np.arange(ny) * dy + dy / 2 - Ly / 2
    zs = np.arange(nz) * dz + dz / 2 - Lz / 2
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    all_positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    all_fields = np.stack([
        np.asarray(s['fields'][primary]).ravel() for s in snapshots
    ], axis=0)
    global_max = float(max(all_fields.max(), 1e-9))

    mask = all_fields.max(axis=0) > threshold * global_max
    kept_positions = all_positions[mask]

    frames = []
    concs_per_frame = []
    for i, s in enumerate(snapshots):
        field = all_fields[i][mask]
        normalized = field / global_max
        concs_per_frame.append([float(round(v, 4)) for v in normalized])
        frames.append({
            'time': s['time'],
            'mean': float(round(s['means'][primary], 5)),
            'max': float(round(field.max() if len(field) else 0.0, 5)),
        })

    return {
        'positions': [[float(round(p, 4)) for p in pos] for pos in kept_positions],
        'concs': concs_per_frame,
        'frames': frames,
        'max_c': global_max,
        'n_voxels': int(kept_positions.shape[0]),
        'extent': list(extent),
        'mesh_size': list(mesh_size),
    }


# ── Bigraph diagram ─────────────────────────────────────────────────


def generate_bigraph_image(cfg_entry):
    """Render a simplified colored bigraph PNG for the config."""
    from bigraph_viz import plot_bigraph

    tracked = cfg_entry['track']
    outputs = {
        'means': ['stores', 'means'],
        'fields': ['stores', 'fields'],
        'time': ['stores', 'last_time'],
    }
    emit_inputs = {
        'means': ['stores', 'means'],
        'time': ['global_time'],
    }

    doc = {
        'rd': {
            '_type': 'process',
            'address': 'local:VCellFVProcess',
            'config': {
                'compartment': 'cell',
                'mesh_size': list(cfg_entry['mesh_size']),
                'duration': cfg_entry['duration'],
                'species': tracked,
            },
            'inputs': {},
            'outputs': outputs,
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {'means': 'map[float]', 'time': 'float'}},
            'inputs': emit_inputs,
        },
    }

    node_colors = {
        ('rd',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }
    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    with open(os.path.join(outdir, 'bigraph.png'), 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg_entry):
    return make_rd_document(
        antimony=cfg_entry['antimony'],
        init_concs=dict(cfg_entry['init_concs']),
        diff_coefs=dict(cfg_entry['diff_coefs']),
        compartment='cell',
        extent=cfg_entry['extent'],
        mesh_size=cfg_entry['mesh_size'],
        duration=cfg_entry['duration'],
        output_time_step=cfg_entry['output_time_step'],
        track_species=cfg_entry['track'],
        interval=cfg_entry['output_time_step'],
    )


# ── HTML generation ─────────────────────────────────────────────────


def generate_html(sim_results, output_path):
    sections_html = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        primary = cfg['primary']

        voxel_pack = pack_voxel_frames(
            snapshots, primary, cfg['extent'], cfg['mesh_size'])

        times = [s['time'] for s in snapshots]
        species_timeseries = {
            name: [s['means'].get(name, 0.0) for s in snapshots]
            for name in cfg['track']
        }
        total_timeseries = {
            name: [s['totals'].get(name, 0.0) for s in snapshots]
            for name in cfg['track']
        }
        max_concs = [vp_max for vp_max in
                     [s['means'].get(primary, 0.0) for s in snapshots]]
        total_mass = [sum(s['totals'].values()) for s in snapshots]

        all_js_data[sid] = {
            'voxels': voxel_pack,
            'camera': cfg['camera'],
            'primary': primary,
            'tracked': cfg['track'],
            'charts': {
                'times': times,
                'means': species_timeseries,
                'totals': total_timeseries,
                'primary_mean': max_concs,
                'total_mass': total_mass,
            },
            'scheme': cs['primary'],
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        n_vox = voxel_pack['n_voxels']
        peak = voxel_pack['max_c']
        primary_init = species_timeseries[primary][0]
        primary_final = species_timeseries[primary][-1]

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Mesh</span><span class="metric-value">{cfg['mesh_size'][0]}&times;{cfg['mesh_size'][1]}&times;{cfg['mesh_size'][2]}</span></div>
        <div class="metric"><span class="metric-label">Voxels rendered</span><span class="metric-value">{n_vox:,}</span></div>
        <div class="metric"><span class="metric-label">Peak [{primary}]</span><span class="metric-value">{peak:.3g}</span></div>
        <div class="metric"><span class="metric-label">Mean [{primary}]</span><span class="metric-value">{primary_final:.3g}</span><span class="metric-sub">from {primary_init:.3g}</span></div>
        <div class="metric"><span class="metric-label">Duration</span><span class="metric-value">{cfg['duration']} s</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Voxel Field Viewer &mdash; species {primary}</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_vox:,}</strong> voxels &middot;
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="colorbar-box">
          <div class="cb-title">[{primary}] &micro;M</div>
          <div class="cb-val">{peak:.2g}</div>
          <div class="cb-gradient"></div>
          <div class="cb-val">0</div>
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Species Mean &amp; Totals</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-means-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-totals-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-peak-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-conservation-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VCell FV Solver &mdash; 3D Reaction-Diffusion Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:780px; }}
.page-header code {{ background:#e0e7ff; padding:.1rem .4rem; border-radius:4px;
                     font-size:.82rem; color:#4338ca; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; flex-wrap:wrap; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600; color:#334155;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:900px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.mesh-canvas {{ width:100%; height:500px; display:block; cursor:grab; }}
.mesh-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#1e293b; }}
.colorbar-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(255,255,255,.92);
                 border:1px solid #e2e8f0; border-radius:8px; padding:.6rem;
                 display:flex; flex-direction:column; align-items:center; gap:.2rem;
                 backdrop-filter:blur(4px); }}
.cb-title {{ font-size:.65rem; text-transform:uppercase; letter-spacing:.04em; color:#64748b; }}
.cb-gradient {{ width:16px; height:100px; border-radius:3px;
  background:linear-gradient(to bottom, #e61a0d, #e6c01a, #4dd94d, #12b5c9, #3112cc); }}
.cb-val {{ font-size:.65rem; color:#94a3b8; }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(241,245,249,.97));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#64748b; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#334155; min-width:100px; text-align:right; }}
.play-btn {{ background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>VCell Finite-Volume 3D Reaction-Diffusion Report</h1>
  <p>Three 3D reaction-diffusion simulations wrapped as
  <strong>process-bigraph</strong> Processes, with the native VCell
  <code>pyvcell-fvsolver</code> finite volume PDE kernel producing the
  spatial fields. Drag the 3D viewers to inspect the voxel concentration
  clouds; each simulation is a self-contained PBG composite driven through
  the <code>VCellFVProcess</code> wrapper.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-vcell-fvsolver</strong> &mdash;
  pyvcell + pyvcell-fvsolver + process-bigraph &mdash;
  Finite-volume PDE solver on a 3D Cartesian mesh
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Three.js Voxel Cloud Viewers ───
const viewers = {{}};
const playStates = {{}};

function turboColor(t) {{
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.25) {{
    const s = t / 0.25;
    r = 0.19; g = 0.07 + 0.63*s; b = 0.99 - 0.19*s;
  }} else if (t < 0.5) {{
    const s = (t - 0.25) / 0.25;
    r = 0.19 + 0.11*s; g = 0.70 + 0.15*s; b = 0.80 - 0.55*s;
  }} else if (t < 0.75) {{
    const s = (t - 0.5) / 0.25;
    r = 0.30 + 0.60*s; g = 0.85 - 0.10*s; b = 0.25 - 0.15*s;
  }} else {{
    const s = (t - 0.75) / 0.25;
    r = 0.90 + 0.10*s; g = 0.75 - 0.55*s; b = 0.10 - 0.05*s;
  }}
  return [r, g, b];
}}

function initViewer(sid) {{
  const d = DATA[sid];
  const v = d.voxels;
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 500;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true, alpha:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0xf1f5f9);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(35, W/H, 0.01, 200);
  cam.position.set(...d.camera);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;

  // Bounding box outline
  const [Lx, Ly, Lz] = v.extent;
  const boxGeom = new THREE.BoxGeometry(Lx, Ly, Lz);
  const boxEdges = new THREE.EdgesGeometry(boxGeom);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x64748b, transparent:true, opacity:0.45}}));
  scene.add(boxLine);

  // Voxel point cloud
  const nv = v.positions.length;
  const positions = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {{
    positions[i*3]   = v.positions[i][0];
    positions[i*3+1] = v.positions[i][1];
    positions[i*3+2] = v.positions[i][2];
  }}
  const colors = new Float32Array(nv * 3);
  const alphas = new Float32Array(nv);

  const pGeom = new THREE.BufferGeometry();
  pGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  pGeom.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
  pGeom.setAttribute('alpha',    new THREE.BufferAttribute(alphas, 1));

  const voxelSize = Math.min(Lx / v.mesh_size[0], Ly / v.mesh_size[1], Lz / v.mesh_size[2]);
  const pMat = new THREE.ShaderMaterial({{
    vertexShader: `
      attribute float alpha;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {{
        vColor = color;
        vAlpha = alpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = ${{(voxelSize * 60).toFixed(2)}} / -mv.z;
        gl_Position = projectionMatrix * mv;
      }}
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      void main() {{
        vec2 c = gl_PointCoord - vec2(0.5);
        float d = length(c);
        if (d > 0.5) discard;
        float soft = smoothstep(0.5, 0.3, d);
        gl_FragColor = vec4(vColor, vAlpha * soft);
      }}
    `,
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  }});
  const points = new THREE.Points(pGeom, pMat);
  scene.add(points);

  function updateFrame(idx) {{
    const frame = v.concs[idx];
    for (let i = 0; i < nv; i++) {{
      const c = frame[i];
      const [r, g, b] = turboColor(c);
      colors[i*3]   = r;
      colors[i*3+1] = g;
      colors[i*3+2] = b;
      alphas[i]     = Math.pow(c, 0.6) * 0.85;
    }}
    pGeom.attributes.color.needsUpdate = true;
    pGeom.attributes.alpha.needsUpdate = true;
  }}

  updateFrame(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateFrame(idx);
    tval.textContent = 't = ' + v.frames[idx].time;
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateFrame, slider, tval }};
  playStates[sid] = {{ playing: false, interval: null }};

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const vv = viewers[sid];
  const v = DATA[sid].voxels;
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    vv.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(vv.slider.value) + 1;
      if (idx >= v.frames.length) idx = 0;
      vv.slider.value = idx;
      vv.updateFrame(idx);
      vv.tval.textContent = 't = ' + v.frames[idx].time;
    }}, 350);
  }} else {{
    btn.textContent = 'Play';
    vv.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time (s)', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};
const SPECIES_COLORS = ['#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#8b5cf6'];

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  const c = d.charts;
  const tracked = d.tracked;

  const meanTraces = tracked.map((name, i) => ({{
    x: c.times, y: c.means[name], type:'scatter', mode:'lines+markers',
    line:{{ color:SPECIES_COLORS[i % SPECIES_COLORS.length], width:2 }},
    marker:{{ size:4 }}, name:name,
  }}));
  Plotly.newPlot('chart-means-'+sid, meanTraces,
    {{...pLayout, title:{{ text:'Mean concentration', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'[µM]', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true}},
    pCfg);

  const totalTraces = tracked.map((name, i) => ({{
    x: c.times, y: c.totals[name], type:'scatter', mode:'lines+markers',
    line:{{ color:SPECIES_COLORS[i % SPECIES_COLORS.length], width:1.8 }},
    marker:{{ size:3 }}, name:name,
  }}));
  Plotly.newPlot('chart-totals-'+sid, totalTraces,
    {{...pLayout, title:{{ text:'Total mass (sum over voxels)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Σ conc', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true}},
    pCfg);

  Plotly.newPlot('chart-peak-'+sid, [{{
    x:c.times, y:c.primary_mean, type:'scatter', mode:'lines+markers',
    line:{{ color:d.scheme, width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(99,102,241,0.07)',
  }}],
    {{...pLayout, title:{{ text:'Mean [' + d.primary + ']', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'[µM]', font:{{ size:10 }} }} }}, showlegend:false}},
    pCfg);

  Plotly.newPlot('chart-conservation-'+sid, [{{
    x:c.times, y:c.total_mass, type:'scatter', mode:'lines+markers',
    line:{{ color:'#8b5cf6', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(139,92,246,0.07)',
  }}],
    {{...pLayout, title:{{ text:'Total mass (all species)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Σ conc', font:{{ size:10 }} }} }}, showlegend:false}},
    pCfg);
}});

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s, {len(snapshots)} snapshots')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    try:
        import subprocess
        subprocess.run(['open', '-a', 'Safari', output_path], check=False)
    except Exception:
        pass


if __name__ == '__main__':
    run_demo()
