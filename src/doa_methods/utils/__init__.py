"""
Utilities Module
===============

Common utilities for DOA estimation including:
- Peak finding algorithms
- Visualization tools
- Mathematical utilities
"""

from .peak_finding import find_peaks
from .math_utils import *
from .visualization import DOAPlotter, quick_plot_comparison

__all__ = ['find_peaks', 'DOAPlotter', 'quick_plot_comparison']