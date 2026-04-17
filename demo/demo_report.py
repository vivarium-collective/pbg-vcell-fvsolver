"""Demo: VCell FV solver multi-configuration 3D reaction-diffusion report.

Runs three 3D reaction-diffusion simulations through the process-bigraph
wrapper. For each experiment the report bundles:

- the Antimony model and a species/diffusion parameter table,
- three interactive PyVista vtk.js viewers (early / mid / late
  timepoints) showing a clipped volumetric concentration field in the
  same style as pyvcell's built-in trame widget,
- a matplotlib slice strip across all tracked species and time,
- Plotly time-series charts (means, totals, conservation),
- a colored bigraph-viz PNG and a collapsible JSON tree of the
  composite document.
"""

import base64
import io
import json
import os
import shutil
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_vcell_fvsolver import VCellFVProcess, make_rd_document

pv.OFF_SCREEN = True


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'diffusion',
        'title': 'Gaussian Pulse Diffusion',
        'subtitle': 'Single-species diffusion from a point source',
        'description': (
            "A Gaussian concentration pulse of species A centered at the "
            "middle of the cubic domain diffuses outward under Fick's law. "
            "No reactions — a pure diffusion benchmark of the VCell finite "
            "volume PDE solver. The peak concentration decays while the "
            "distribution spreads uniformly over the domain."
        ),
        'antimony': """
compartment ec;
compartment cell;
species A in cell;
A = 0
""".strip(),
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
        'n_snapshots': 12,
        'color_scheme': 'indigo',
    },
    {
        'id': 'cascade',
        'title': 'Reaction-Diffusion Cascade',
        'subtitle': 'A → B → C with distinct diffusion coefficients',
        'description': (
            'A linear reaction cascade: A decays into B, which decays into '
            'C. All three species diffuse but with very different '
            'coefficients (A slow, B medium, C fast). The wrapper drives '
            "VCell's finite volume PDE solver to couple first-order "
            'kinetics with Fickian transport across a 3D Cartesian mesh, '
            'producing three distinct spatial profiles that emerge and '
            'decay in sequence.'
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
""".strip(),
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
        'n_snapshots': 12,
        'color_scheme': 'emerald',
    },
    {
        'id': 'mixing',
        'title': 'Two-Source Mixing',
        'subtitle': 'Opposing diffusion fronts with a binding reaction',
        'description': (
            'Two Gaussian sources of species A and species B sit on '
            'opposite corners of the cubic domain. As both diffuse '
            'inward they collide in the middle where a bimolecular '
            'association reaction A + B -> C traps product along the '
            'interface. The VCell finite volume solver handles the '
            'coupled transport and nonlinear reaction across the 3D mesh.'
        ),
        'antimony': """
compartment ec;
compartment cell;
species A in cell;
species B in cell;
species C in cell;
J1: A + B -> C; k1*A*B
k1 = 1.5
A = 0
B = 0
C = 0
""".strip(),
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
        'n_snapshots': 12,
        'color_scheme': 'rose',
    },
]


COLOR_SCHEMES = {
    'indigo':  {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
}


# ── Simulation runner ───────────────────────────────────────────────


def run_simulation(cfg_entry):
    """Run a single RD simulation, collecting per-snapshot 3D fields."""
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
    proc.initial_state()
    time_points = proc.get_time_points()

    n = min(cfg_entry['n_snapshots'], len(time_points))
    indices = np.linspace(0, len(time_points) - 1, n).astype(int).tolist()

    snapshots = []
    for idx in indices:
        snap_state = proc._snapshot_at(time_points[idx])
        snapshots.append({
            'time': round(snap_state['time'], 4),
            'fields': {k: np.asarray(v) for k, v in snap_state['fields'].items()},
            'means': snap_state['means'],
            'totals': snap_state['totals'],
        })

    runtime = time.perf_counter() - t0
    return snapshots, runtime


# ── PyVista rendering (trame-widget style) ──────────────────────────


def _build_imagedata(field3d, extent):
    """Wrap a 3D numpy field (nx, ny, nz) as a PyVista ImageData grid."""
    nx, ny, nz = field3d.shape
    Lx, Ly, Lz = extent
    spacing = (Lx / nx, Ly / ny, Lz / nz)
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing,
                        origin=(0.0, 0.0, 0.0))
    grid.point_data['conc'] = field3d.astype(np.float32).ravel(order='F')
    return grid


def render_pyvista_snapshot(field3d, extent, species, vmin, vmax, out_path,
                            clip_frac=0.55):
    """Render a clipped-volume 3D view of `field3d` as interactive vtk.js HTML.

    Matches the visual style of pyvcell.sim_results.widget.App: a clipped
    unstructured mesh with turbo colormap, outline box, and scalar bar.
    """
    grid = _build_imagedata(field3d, extent)
    cx, cy, cz = grid.center
    bounds = grid.bounds
    clip_x = bounds[0] + clip_frac * (bounds[1] - bounds[0])
    clipped = grid.clip_box(
        bounds=(clip_x, bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
        invert=False,
    )

    pl = pv.Plotter(off_screen=True, window_size=(560, 440))
    pl.set_background('#f1f5f9')
    label = f'[{species}] µM'
    if clipped.n_points > 0:
        pl.add_mesh(
            clipped,
            scalars='conc',
            cmap='turbo',
            clim=(float(vmin), float(vmax)),
            show_scalar_bar=True,
            scalar_bar_args={
                'title': label,
                'title_font_size': 12,
                'label_font_size': 10,
                'color': '#334155',
                'shadow': False,
            },
            nan_opacity=0.0,
        )
    pl.add_mesh(grid.outline(), color='#475569', line_width=2)
    pl.show_grid(color='#94a3b8', xtitle='x', ytitle='y', ztitle='z',
                 font_size=10, minor_ticks=False)
    pl.camera_position = 'iso'
    pl.camera.zoom(1.05)
    pl.export_html(out_path)
    pl.close()


def render_pyvista_stills(cfg, snapshots, out_dir):
    """Write three interactive vtk.js HTMLs (early/mid/late) for primary species.

    Returns a list of (label, relative_path, time) tuples.
    """
    primary = cfg['primary']
    all_vals = np.concatenate([
        snap['fields'][primary].ravel() for snap in snapshots])
    vmax = float(max(all_vals.max(), 1e-9))
    vmin = 0.0

    n = len(snapshots)
    picks = [
        ('Early', 0),
        ('Middle', n // 2),
        ('Late', n - 1),
    ]
    results = []
    for label, i in picks:
        snap = snapshots[i]
        rel = f"views/{cfg['id']}_{primary}_{label.lower()}.html"
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        render_pyvista_snapshot(
            snap['fields'][primary],
            cfg['extent'], primary, vmin, vmax, out_path,
        )
        results.append((label, rel, snap['time']))
    return results, vmin, vmax


def render_slice_strip(cfg, snapshots):
    """Build a PNG strip of z-midplane slices per species per timepoint.

    Returns a base64 data URI of the composite image.
    """
    primary = cfg['primary']
    tracked = cfg['track']
    n_species = len(tracked)
    n_times = min(6, len(snapshots))
    picks = np.linspace(0, len(snapshots) - 1, n_times).astype(int)

    fig, axes = plt.subplots(
        n_species, n_times,
        figsize=(1.6 * n_times, 1.7 * n_species),
        squeeze=False,
    )
    fig.patch.set_facecolor('#ffffff')

    nz = snapshots[0]['fields'][primary].shape[2]
    mid = nz // 2

    per_species_max = {
        sp: float(max(
            max(snapshots[i]['fields'][sp][:, :, mid].max() for i in picks),
            1e-12,
        ))
        for sp in tracked
    }

    for r, sp in enumerate(tracked):
        vmax = per_species_max[sp]
        for c, ti in enumerate(picks):
            ax = axes[r][c]
            field = snapshots[ti]['fields'][sp]
            ax.imshow(
                field[:, :, mid].T, origin='lower',
                cmap='turbo', vmin=0.0, vmax=vmax,
                aspect='equal',
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#cbd5e1')
            if r == 0:
                ax.set_title(f"t = {snapshots[ti]['time']:.2f}",
                             fontsize=8, color='#334155')
            if c == 0:
                ax.set_ylabel(sp, fontsize=10, color='#334155',
                              rotation=0, labelpad=12, va='center')

    fig.suptitle(f'Mid-plane z-slice ({mid}) — max scaled per-species',
                 fontsize=9, color='#64748b', y=1.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor='#ffffff')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'data:image/png;base64,{b64}'


# ── Bigraph diagram ─────────────────────────────────────────────────


def generate_bigraph_image(cfg_entry):
    """Render a simplified colored bigraph PNG for the config."""
    from bigraph_viz import plot_bigraph

    doc = {
        'rd': {
            '_type': 'process',
            'address': 'local:VCellFVProcess',
            'config': {
                'compartment': 'cell',
                'mesh_size': list(cfg_entry['mesh_size']),
                'duration': cfg_entry['duration'],
                'species': list(cfg_entry['track']),
            },
            'inputs': {},
            'outputs': {
                'means': ['stores', 'means'],
                'fields': ['stores', 'fields'],
                'time': ['stores', 'last_time'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {'means': 'map[float]', 'time': 'float'}},
            'inputs': {
                'means': ['stores', 'means'],
                'time': ['global_time'],
            },
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


# ── Model / species table ───────────────────────────────────────────


def _esc(s):
    return (str(s)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))


def species_table_html(cfg):
    rows = []
    for sp in cfg['track']:
        ic = cfg['init_concs'].get(sp, '0.0')
        dc = cfg['diff_coefs'].get(sp, 0.0)
        rows.append(
            f'<tr><td class="sp-name">{_esc(sp)}</td>'
            f'<td class="sp-ic"><code>{_esc(ic)}</code></td>'
            f'<td class="sp-dc">{dc:g}</td></tr>'
        )
    return (
        '<table class="species-table"><thead>'
        '<tr><th>Species</th><th>Initial concentration (µM)</th>'
        '<th>Diff. coef. (µm²/s)</th></tr></thead>'
        '<tbody>' + ''.join(rows) + '</tbody></table>'
    )


def geometry_table_html(cfg):
    Lx, Ly, Lz = cfg['extent']
    nx, ny, nz = cfg['mesh_size']
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    return (
        '<table class="geo-table">'
        f'<tr><td>Domain extent</td><td>{Lx:g} × {Ly:g} × {Lz:g} µm</td></tr>'
        f'<tr><td>Voxel grid</td><td>{nx} × {ny} × {nz} '
        f'<span class="muted">({nx*ny*nz:,} cells)</span></td></tr>'
        f'<tr><td>Voxel size</td><td>{dx:.3g} × {dy:.3g} × {dz:.3g} µm</td></tr>'
        f'<tr><td>Duration</td><td>{cfg["duration"]} s</td></tr>'
        f'<tr><td>Output step</td><td>{cfg["output_time_step"]} s</td></tr>'
        f'<tr><td>Compartment</td><td><code>cell</code></td></tr>'
        '</table>'
    )


# ── HTML generation ─────────────────────────────────────────────────


def generate_html(sim_results, output_path, out_dir):
    sections_html = []
    all_chart_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]

        print(f'  Rendering PyVista views for {sid}...')
        stills, vmin, vmax = render_pyvista_stills(cfg, snapshots, out_dir)

        print(f'  Rendering slice strip for {sid}...')
        strip_uri = render_slice_strip(cfg, snapshots)

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        times = [s['time'] for s in snapshots]
        species_timeseries = {
            name: [s['means'].get(name, 0.0) for s in snapshots]
            for name in cfg['track']
        }
        total_timeseries = {
            name: [s['totals'].get(name, 0.0) for s in snapshots]
            for name in cfg['track']
        }
        primary_mean = [s['means'].get(cfg['primary'], 0.0) for s in snapshots]
        total_mass = [sum(s['totals'].values()) for s in snapshots]
        all_chart_data[sid] = {
            'times': times,
            'means': species_timeseries,
            'totals': total_timeseries,
            'primary_mean': primary_mean,
            'total_mass': total_mass,
            'primary': cfg['primary'],
            'tracked': cfg['track'],
            'scheme': cs['primary'],
        }

        primary = cfg['primary']
        primary_init = species_timeseries[primary][0]
        primary_final = species_timeseries[primary][-1]
        nx, ny, nz = cfg['mesh_size']

        still_blocks = ''.join(
            f'''
          <div class="still-wrap">
            <div class="still-label">{label} &mdash; t = {t:g}s</div>
            <iframe class="still-frame" src="{path}" loading="lazy"></iframe>
          </div>'''
            for (label, path, t) in stills
        )

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
        <div class="metric"><span class="metric-label">Mesh</span><span class="metric-value">{nx}&times;{ny}&times;{nz}</span></div>
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{len(cfg['track'])}</span></div>
        <div class="metric"><span class="metric-label">Peak [{primary}]</span><span class="metric-value">{vmax:.3g}</span></div>
        <div class="metric"><span class="metric-label">Final [{primary}]</span><span class="metric-value">{primary_final:.3g}</span><span class="metric-sub">from {primary_init:.3g}</span></div>
        <div class="metric"><span class="metric-label">Duration</span><span class="metric-value">{cfg['duration']} s</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">Model &amp; Geometry</h3>
      <div class="model-row">
        <div class="model-col">
          <div class="col-title">Antimony source</div>
          <pre class="antimony-code">{_esc(cfg['antimony'])}</pre>
        </div>
        <div class="model-col">
          <div class="col-title">Species mapping</div>
          {species_table_html(cfg)}
          <div class="col-title" style="margin-top:1rem;">Geometry &amp; solver</div>
          {geometry_table_html(cfg)}
        </div>
      </div>

      <h3 class="subsection-title">3D Volume Viewer &mdash; species {primary}
        <span class="hint">(early / middle / late &mdash; drag to rotate, scroll to zoom)</span>
      </h3>
      <div class="stills-row">{still_blocks}</div>

      <h3 class="subsection-title">Mid-plane z-slice strip &mdash; all tracked species</h3>
      <div class="slice-strip-wrap">
        <img src="{strip_uri}" alt="mid-plane z-slices" class="slice-strip">
      </div>

      <h3 class="subsection-title">Species Time Series</h3>
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
.page-header p {{ color:#64748b; font-size:.95rem; max-width:820px; }}
.page-header code {{ background:#e0e7ff; padding:.1rem .4rem; border-radius:4px;
                     font-size:.82rem; color:#4338ca; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        flex-wrap:wrap; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             color:#334155; transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:920px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.8rem 0 .8rem; }}
.subsection-title .hint {{ font-weight:400; font-size:.78rem; color:#94a3b8; margin-left:.5rem; }}

.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:.8rem; margin-bottom:1rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}

.model-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:.5rem; }}
.model-col {{ min-width:0; }}
.col-title {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.06em;
              color:#64748b; font-weight:700; margin-bottom:.5rem; }}
.antimony-code {{ background:#0f172a; color:#cbd5e1; border-radius:10px;
                  padding:1rem 1.1rem; font-family:'SF Mono',Menlo,Monaco,monospace;
                  font-size:.8rem; line-height:1.55; overflow:auto;
                  border:1px solid #1e293b; }}
.species-table, .geo-table {{ width:100%; border-collapse:collapse; font-size:.82rem;
                               background:#f8fafc; border:1px solid #e2e8f0;
                               border-radius:10px; overflow:hidden; }}
.species-table th {{ background:#eef2ff; color:#4338ca; text-align:left;
                     padding:.5rem .7rem; font-size:.72rem; text-transform:uppercase;
                     letter-spacing:.05em; font-weight:700; }}
.species-table td, .geo-table td {{ padding:.5rem .7rem; border-top:1px solid #e2e8f0; }}
.species-table td.sp-name {{ font-weight:700; color:#1e293b; }}
.species-table code {{ font-family:'SF Mono',Menlo,monospace; font-size:.78rem;
                       color:#0f766e; }}
.geo-table td:first-child {{ color:#64748b; width:38%; font-size:.78rem; }}
.geo-table code {{ font-family:'SF Mono',Menlo,monospace; color:#4338ca; }}
.muted {{ color:#94a3b8; font-size:.8em; }}

.stills-row {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem;
               margin-bottom:.5rem; }}
.still-wrap {{ background:#f1f5f9; border:1px solid #e2e8f0; border-radius:10px;
               overflow:hidden; }}
.still-label {{ padding:.55rem .8rem; font-size:.75rem; font-weight:600;
                color:#475569; background:#f8fafc; border-bottom:1px solid #e2e8f0; }}
.still-frame {{ width:100%; height:440px; border:none; display:block; background:#f1f5f9; }}

.slice-strip-wrap {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1rem; text-align:center; margin-bottom:.5rem; }}
.slice-strip {{ max-width:100%; height:auto; }}

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
  .charts-row,.pbg-row,.model-row,.stills-row {{ grid-template-columns:1fr; }}
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
  spatial fields. The 3D viewers embedded below are
  <strong>PyVista + vtk.js</strong> scenes — the same rendering stack
  used by pyvcell's built-in trame widget — exported as self-contained
  interactive HTML.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-vcell-fvsolver</strong> &mdash;
  pyvcell + pyvcell-fvsolver + process-bigraph &mdash;
  Finite-volume PDE solver on a 3D Cartesian mesh
</div>

<script>
const CHARTS = {json.dumps(all_chart_data)};
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

Object.keys(CHARTS).forEach(sid => {{
  const d = CHARTS[sid];
  const tracked = d.tracked;

  const meanTraces = tracked.map((name, i) => ({{
    x: d.times, y: d.means[name], type:'scatter', mode:'lines+markers',
    line:{{ color:SPECIES_COLORS[i % SPECIES_COLORS.length], width:2 }},
    marker:{{ size:4 }}, name:name,
  }}));
  Plotly.newPlot('chart-means-'+sid, meanTraces,
    {{...pLayout, title:{{ text:'Mean concentration', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'[µM]', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true}},
    pCfg);

  const totalTraces = tracked.map((name, i) => ({{
    x: d.times, y: d.totals[name], type:'scatter', mode:'lines+markers',
    line:{{ color:SPECIES_COLORS[i % SPECIES_COLORS.length], width:1.8 }},
    marker:{{ size:3 }}, name:name,
  }}));
  Plotly.newPlot('chart-totals-'+sid, totalTraces,
    {{...pLayout, title:{{ text:'Total mass (sum over voxels)', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Σ conc', font:{{ size:10 }} }} }},
      legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true}},
    pCfg);

  Plotly.newPlot('chart-peak-'+sid, [{{
    x:d.times, y:d.primary_mean, type:'scatter', mode:'lines+markers',
    line:{{ color:d.scheme, width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(99,102,241,0.07)',
  }}],
    {{...pLayout, title:{{ text:'Mean [' + d.primary + ']', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'[µM]', font:{{ size:10 }} }} }}, showlegend:false}},
    pCfg);

  Plotly.newPlot('chart-conservation-'+sid, [{{
    x:d.times, y:d.total_mass, type:'scatter', mode:'lines+markers',
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
    views_dir = os.path.join(demo_dir, 'views')
    if os.path.isdir(views_dir):
        shutil.rmtree(views_dir)

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s, {len(snapshots)} snapshots')

    print('Generating HTML report...')
    generate_html(sim_results, output_path, demo_dir)

    try:
        import subprocess
        subprocess.run(['open', '-a', 'Safari', output_path], check=False)
    except Exception:
        pass


if __name__ == '__main__':
    run_demo()
