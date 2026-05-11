"""Visualization Step subclasses for pbg-vcell-fvsolver.

Visualizations follow the pbg-superpowers convention (v0.4.15+):
each subclass overrides `update()` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See pbg_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from pbg_superpowers.visualization import Visualization


class RDFieldPlots(Visualization):
    """Time-series HTML plot of VCell-FV reaction-diffusion mean/total concentrations.

    Consumes the per-snapshot `means` and `totals` maps (species name ->
    float) plus the current `time`, accumulates them across calls, and
    emits a Plotly HTML figure on every update. Downstream consumers
    (dashboards, notebook viewers) read the latest 'html' from the wired
    store.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'VCell FV reaction-diffusion'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times: list[float] = []
        # Per-species accumulators (created on first sighting).
        self.means_history: dict[str, list[float]] = {}
        self.totals_history: dict[str, list[float]] = {}

    def inputs(self):
        return {
            'means': 'map[float]',
            'totals': 'map[float]',
            'time': 'float',
        }

    def _append(self, store, names_seen, value_map):
        # Extend any species missing from this snapshot with their last
        # value (or 0.0 if brand new), then append the incoming values
        # so all series stay length-aligned with self.times.
        n = len(self.times)
        for name in names_seen:
            if name not in store:
                store[name] = [0.0] * (n - 1)
            elif len(store[name]) < n - 1:
                last = store[name][-1] if store[name] else 0.0
                store[name].extend([last] * ((n - 1) - len(store[name])))
        for name, val in value_map.items():
            store.setdefault(name, [0.0] * (n - 1))
            store[name].append(float(val) if val is not None else 0.0)
        # Any species in store but absent from this snapshot: carry last forward.
        for name, series in store.items():
            if len(series) < n:
                last = series[-1] if series else 0.0
                series.append(last)

    def update(self, state, interval=1.0):
        t = state.get('time')
        self.times.append(
            float(t) if t is not None else len(self.times) * (interval or 1.0)
        )

        means = state.get('means') or {}
        totals = state.get('totals') or {}
        all_names = set(self.means_history) | set(self.totals_history) | set(means) | set(totals)
        self._append(self.means_history, all_names, means)
        self._append(self.totals_history, all_names, totals)

        title = (self.config or {}).get('title', 'VCell FV reaction-diffusion')
        mean_traces = []
        for name in sorted(self.means_history):
            ys = self.means_history[name]
            mean_traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines+markers","name":"mean(' + name + ')",'
                '"xaxis":"x","yaxis":"y"}'
            )
        total_traces = []
        for name in sorted(self.totals_history):
            ys = self.totals_history[name]
            total_traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines+markers","name":"total(' + name + ')",'
                '"xaxis":"x2","yaxis":"y2"}'
            )
        all_traces = mean_traces + total_traces
        html = (
            f'<div id="rdfp" style="height:420px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("rdfp",[{",".join(all_traces)}],'
            f'{{title:"{title}",'
            f'grid:{{rows:1,columns:2,pattern:"independent"}},'
            f'margin:{{l:55,r:15,t:45,b:40}},'
            f'xaxis:{{title:"time",domain:[0,0.46]}},'
            f'yaxis:{{title:"mean conc"}},'
            f'xaxis2:{{title:"time",domain:[0.54,1.0]}},'
            f'yaxis2:{{title:"total conc",anchor:"x2"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return {'html': html}
