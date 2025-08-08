"""
Simulation Module
================

Synthetic data generators and simulation utilities for DOA estimation.
"""

from .scenarios import SimulationScenario
from .monte_carlo import MonteCarlo

__all__ = ['SimulationScenario', 'MonteCarlo']