"""
Array Processing Module
======================

Core infrastructure for DOA estimation including:
- Uniform Linear Array (ULA) geometry
- Signal models for narrowband sources
- Array manifold and steering vectors
"""

from .ula import UniformLinearArray
from .signal_model import SignalModel

__all__ = ['UniformLinearArray', 'SignalModel']