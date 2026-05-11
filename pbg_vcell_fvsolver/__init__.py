"""process-bigraph wrapper for VCell's finite-volume 3D reaction-diffusion solver.

Drives the native `pyvcell-fvsolver` PDE solver through the high-level
`pyvcell` Antimony -> VCML -> FV-input pipeline.
"""

from pbg_vcell_fvsolver.processes import VCellFVProcess
from pbg_vcell_fvsolver.composites import (
    make_rd_document,
    register_vcell_fvsolver,
    build_composite,
    list_composite_specs,
    load_composite_spec,
)

__all__ = [
    'VCellFVProcess',
    'make_rd_document',
    'register_vcell_fvsolver',
    'build_composite',
    'list_composite_specs',
    'load_composite_spec',
]
