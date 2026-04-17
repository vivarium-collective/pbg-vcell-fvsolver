"""Shared fixtures for the pbg-vcell-fvsolver test suite."""

import pytest
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_vcell_fvsolver import VCellFVProcess


ANTIMONY_AB = """
    compartment ec;
    compartment cell;
    species A in cell;
    species B in cell;
    A -> B; k*A
    k = 0.3
    A = 10
"""


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('VCellFVProcess', VCellFVProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


@pytest.fixture
def antimony_ab():
    return ANTIMONY_AB
