"""
DOA Methods: A Comprehensive Tutorial Repository
==============================================

Educational Python implementation of classical Direction of Arrival (DOA) estimation methods.
Focus on narrowband signals and Uniform Linear Arrays (ULA) for beginner researchers and students.

Modules:
- array_processing: ULA geometry and signal models
- classical: Conventional beamforming, Capon/MVDR
- subspace: MUSIC, Root-MUSIC, ESPRIT, Unitary ESPRIT  
- maximum_likelihood: Stochastic ML, Deterministic ML, WSF
- sparse: L1-SVD, SBL, SPICE
- simulation: Synthetic data generators
- evaluation: Performance metrics and comparison tools
- utils: Visualization and utility functions
"""

__version__ = "1.0.0"
__author__ = "DOA Methods Tutorial"

from . import array_processing
from . import classical
from . import subspace
from . import maximum_likelihood
from . import sparse
from . import simulation
from . import evaluation
from . import utils

__all__ = [
    'array_processing',
    'classical', 
    'subspace',
    'maximum_likelihood',
    'sparse',
    'simulation',
    'evaluation',
    'utils'
]