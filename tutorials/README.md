# DOA Methods Tutorials

This directory contains comprehensive Jupyter notebook tutorials covering all aspects of Direction of Arrival (DOA) estimation methods. The tutorials are designed for progressive learning, starting from fundamental concepts and advancing to sophisticated algorithms.

## 📚 Tutorial Sequence

### 1. Fundamentals
- **01_Array_Processing_Fundamentals.ipynb** - ULA geometry, signal models, steering vectors
- **02_Spatial_Sampling_and_Aliasing.ipynb** - Spatial frequency, visible region, aliasing effects
- **03_Covariance_Matrices_and_Eigenanalysis.ipynb** - Sample covariance, eigendecomposition, signal/noise subspaces

### 2. Classical Methods
- **04_Conventional_Beamforming.ipynb** - Delay-and-sum beamforming, beam patterns, limitations
- **05_Capon_Beamforming.ipynb** - MVDR, adaptive beamforming, diagonal loading
- **06_Classical_Methods_Comparison.ipynb** - Performance comparison, resolution analysis

### 3. Subspace Methods
- **07_MUSIC_Algorithm.ipynb** - Multiple Signal Classification, noise subspace, spectrum
- **08_Root_MUSIC.ipynb** - Polynomial rooting, computational advantages
- **09_ESPRIT_Methods.ipynb** - Rotational invariance, TLS, Unitary ESPRIT
- **10_Subspace_Methods_Deep_Dive.ipynb** - Forward-backward averaging, spatial smoothing

### 4. Maximum Likelihood Methods
- **11_Stochastic_ML.ipynb** - Stochastic signal model, ML estimation
- **12_Deterministic_ML.ipynb** - Deterministic signals, concentrated likelihood
- **13_WSF_Method.ipynb** - Weighted Subspace Fitting, MODE algorithm

### 5. Sparse Methods
- **14_Sparse_Signal_Representation.ipynb** - Sparse reconstruction, overcomplete dictionaries
- **15_L1_SVD_Method.ipynb** - L1 regularization, convex optimization
- **16_Bayesian_Methods.ipynb** - SBL, SPICE, hyperparameter learning

### 6. Performance Analysis
- **17_Performance_Metrics.ipynb** - RMSE, bias, variance, CRLB, efficiency
- **18_Resolution_Analysis.ipynb** - Angular resolution, source separation limits
- **19_SNR_Threshold_Effects.ipynb** - Breakdown phenomena, threshold SNR

### 7. Practical Considerations
- **20_Model_Mismatch.ipynb** - Array calibration errors, mutual coupling
- **21_Correlated_Sources.ipynb** - Coherent signals, decorrelation techniques
- **22_Computational_Complexity.ipynb** - Algorithm complexity, real-time considerations

### 8. Advanced Topics
- **23_Wideband_DOA_Estimation.ipynb** - Coherent signal subspace, incoherent methods
- **24_2D_DOA_Estimation.ipynb** - Planar arrays, azimuth-elevation estimation
- **25_Robust_Methods.ipynb** - Outlier rejection, robust statistics

### 9. Applications and Case Studies
- **26_Radar_Applications.ipynb** - Phased array radar, target tracking
- **27_Communications_Applications.ipynb** - Smart antennas, beamforming
- **28_Acoustic_Applications.ipynb** - Speech enhancement, source localization

### 10. Comprehensive Comparison
- **29_Method_Comparison_Study.ipynb** - Side-by-side performance evaluation
- **30_Best_Practices_and_Guidelines.ipynb** - Method selection, parameter tuning

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages (see `requirements.txt`)

### Installation
```bash
# Clone repository and navigate to tutorials
cd "DOA Survey/tutorials"

# Install dependencies
pip install -r ../requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running Tutorials
1. Start with `01_Array_Processing_Fundamentals.ipynb`
2. Follow the sequence for structured learning
3. Each notebook includes:
   - **Theory**: Mathematical background with clear explanations
   - **Code**: Well-commented implementation examples
   - **Visualizations**: Plots and interactive widgets
   - **Exercises**: Hands-on problems to reinforce concepts

## 📊 Interactive Features

The tutorials include interactive elements:
- **Parameter exploration** with sliders and widgets
- **Real-time visualizations** of algorithm behavior
- **Performance comparison** tools
- **Monte Carlo simulations** with progress bars

## 🎯 Learning Objectives

After completing these tutorials, you will:
- Understand fundamental array signal processing concepts
- Be able to implement and modify DOA estimation algorithms
- Know how to evaluate and compare method performance
- Understand practical limitations and solutions
- Be capable of selecting appropriate methods for specific applications

## 📝 Tutorial Features

Each tutorial includes:
- **Clear mathematical notation** consistent with literature
- **Step-by-step derivations** of key algorithms
- **Practical implementation details** and tips
- **Common pitfalls** and how to avoid them
- **References** to seminal papers and books
- **Exercises** with solutions

## 💡 Tips for Learning

1. **Sequential Approach**: Follow tutorials in order for best understanding
2. **Hands-On Practice**: Run and modify code examples
3. **Parameter Exploration**: Change parameters to see effects
4. **Cross-References**: Use tutorials alongside academic papers
5. **Practical Exercises**: Try the end-of-chapter problems

## 🔗 Additional Resources

- **Academic Papers**: Key references provided in each tutorial
- **Books**: Classical texts on array signal processing
- **Online Resources**: Links to useful websites and tools
- **Example Data**: Real and synthetic datasets for practice

## 📞 Support

If you encounter issues or have questions:
1. Check the FAQ section in each notebook
2. Review the troubleshooting guide
3. Open an issue on the project repository
4. Join the community discussion forum

## 🤝 Contributing

Contributions to improve tutorials are welcome:
- Report errors or unclear explanations
- Suggest additional examples or exercises
- Contribute new tutorial topics
- Improve visualizations or interactive elements

---

**Happy Learning!** 🎓

These tutorials represent a comprehensive educational resource for DOA estimation methods. Take your time to understand each concept thoroughly before moving to the next topic.