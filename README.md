# DOA Methods: A Comprehensive Tutorial Repository

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Educational Python implementation of classical Direction of Arrival (DOA) estimation methods for **narrowband signals** and **Uniform Linear Arrays (ULA)**. Designed for beginner researchers and students in array signal processing.

## 🎯 Scope and Objectives

This repository provides:
- **Clean, well-commented implementations** matching academic notation
- **Modular design** with consistent APIs across all methods
- **Comprehensive tutorials** with step-by-step explanations
- **Performance comparison tools** and statistical analysis
- **Synthetic data generators** for controlled testing
- **Visualization utilities** for understanding algorithm behavior

## 📚 Implemented Methods

### Classical Methods
- **Conventional Beamforming** (Delay-and-Sum)
- **Capon Beamforming** (MVDR - Minimum Variance Distortionless Response)

### Subspace Methods  
- **MUSIC** (Multiple Signal Classification)
- **Root-MUSIC** (Polynomial rooting version)
- **ESPRIT** (Estimation of Signal Parameters via Rotational Invariance Techniques)
- **Unitary ESPRIT** (Real-valued computations)

### Maximum Likelihood Methods
- **Stochastic ML** 
- **Deterministic ML**
- **WSF** (Weighted Subspace Fitting)

### Sparse Methods
- **L1-SVD** 
- **SBL** (Sparse Bayesian Learning)
- **SPICE** (Sparse Iterative Covariance-based Estimation)

## 📁 Repository Structure

```
DOA Survey/
├── src/doa_methods/              # Core implementation
│   ├── array_processing/         # ULA geometry, signal models
│   ├── classical/               # Conventional & Capon beamforming
│   ├── subspace/                # MUSIC, ESPRIT variants
│   ├── maximum_likelihood/      # ML-based methods
│   ├── sparse/                  # Sparse reconstruction methods
│   ├── simulation/              # Data generators, scenarios
│   ├── evaluation/              # Performance metrics, comparison
│   └── utils/                   # Common utilities
├── tutorials/                   # Jupyter notebooks
├── examples/                    # Standalone demo scripts
├── tests/                       # Unit tests
└── docs/                        # API documentation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/doa-methods.git
cd doa-methods

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
import numpy as np
from doa_methods.array_processing import UniformLinearArray, SignalModel
from doa_methods.classical import ConventionalBeamforming
from doa_methods.subspace import MUSIC

# Create 16-element ULA
array = UniformLinearArray(M=16, d=0.5)

# Generate synthetic data
signal_model = SignalModel(array)
doas_true = np.deg2rad([-20, 10])  # Two sources at -20° and 10°
X, S, N = signal_model.generate_signals(
    doas=doas_true, 
    N_snapshots=100, 
    snr_db=10
)

# Conventional beamforming
cbf = ConventionalBeamforming(array)
doas_cbf = cbf.estimate(X, K=2)

# MUSIC algorithm
music = MUSIC(array)  
doas_music = music.estimate(X, K=2)

print(f"True DOAs: {np.rad2deg(doas_true):.1f}°")
print(f"CBF estimates: {np.rad2deg(doas_cbf):.1f}°")
print(f"MUSIC estimates: {np.rad2deg(doas_music):.1f}°")
```

## 🔬 Test Scenarios

The repository includes predefined scenarios for systematic evaluation:

- **Two closely spaced sources** (resolution testing)
- **Multiple uncorrelated sources** 
- **Correlated sources** (challenging scenario)
- **Low SNR conditions**
- **Varying source powers**
- **Limited snapshots** (small sample size)

Performance is evaluated across:
- **Number of sources**: 1-4
- **SNR range**: -10 to 30 dB  
- **Array sizes**: 8, 16, 32, 64 elements
- **Snapshot counts**: 10-1000

## 📊 Performance Metrics

- **RMSE** (Root Mean Square Error)
- **Resolution capability** analysis
- **Computational complexity** comparisons
- **Success rate** in challenging conditions
- **Bias and variance** characterization

## 📖 Tutorials

Interactive Jupyter notebooks cover:

1. **Array Processing Fundamentals** - ULA geometry, steering vectors
2. **Classical Methods** - Beamforming principles and trade-offs
3. **Subspace Methods** - High-resolution techniques
4. **Maximum Likelihood** - Optimal estimation theory
5. **Sparse Methods** - Modern compressed sensing approaches
6. **Performance Comparison** - Systematic benchmarking
7. **Real-World Considerations** - Practical implementation issues

## 🔧 Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/ examples/

# Type checking
mypy src/

# Build documentation
cd docs/
make html
```

## 📝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📚 References

Key references for implemented methods:
- Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. IEEE Transactions on Antennas and Propagation.
- Roy, R., & Kailath, T. (1989). ESPRIT-estimation of signal parameters via rotational invariance techniques. IEEE Transactions on Acoustics, Speech, and Signal Processing.
- Stoica, P., & Nehorai, A. (1989). MUSIC, maximum likelihood, and Cramer-Rao bound. IEEE Transactions on Acoustics, Speech, and Signal Processing.

## 🤝 Acknowledgments

This educational repository is designed to help researchers and students understand DOA estimation methods through clean implementations and comprehensive tutorials.