"""
Evaluation Module
================

Performance evaluation and comparison tools for DOA estimation methods.
"""

from .metrics import DOAMetrics
from .benchmarks import DOABenchmark
from .comparison import MethodComparison

__all__ = ['DOAMetrics', 'DOABenchmark', 'MethodComparison']