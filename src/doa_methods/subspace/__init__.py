"""
Subspace Methods
===============

High-resolution DOA estimation methods based on subspace decomposition:
- MUSIC (Multiple Signal Classification)
- Root-MUSIC (Polynomial rooting version)
- ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
- Unitary ESPRIT (Real-valued computations)
"""

from .music import MUSIC
from .root_music import RootMUSIC
from .esprit import ESPRIT
from .unitary_esprit import UnitaryESPRIT

__all__ = ['MUSIC', 'RootMUSIC', 'ESPRIT', 'UnitaryESPRIT']