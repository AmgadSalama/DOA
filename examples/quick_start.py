"""
Quick Start Example
==================

Minimal example demonstrating basic usage of DOA estimation methods.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doa_methods.array_processing import UniformLinearArray, SignalModel
from doa_methods.classical import ConventionalBeamforming
from doa_methods.subspace import MUSIC


def quick_start_example():
    """
    Quick start example showing basic DOA estimation workflow.
    """
    print("DOA Methods - Quick Start Example")
    print("=" * 40)
    
    # Step 1: Create array
    print("\n1. Creating 16-element ULA with half-wavelength spacing...")
    array = UniformLinearArray(M=16, d=0.5)
    print(f"Array info: {array.get_info()}")
    
    # Step 2: Generate synthetic data
    print("\n2. Generating synthetic data...")
    signal_model = SignalModel(array)
    
    # Two sources at -20° and 15°
    doas_true = np.deg2rad([-20, 15])
    print(f"True DOAs: {np.rad2deg(doas_true)} degrees")
    
    # Generate 100 snapshots at 10 dB SNR
    X, S, N = signal_model.generate_signals(
        doas=doas_true,
        N_snapshots=100,
        snr_db=10,
        seed=42  # For reproducible results
    )
    print(f"Data shape: {X.shape} (M x N_snapshots)")
    
    # Step 3: DOA estimation with Conventional Beamforming
    print("\n3. DOA estimation with Conventional Beamforming...")
    cbf = ConventionalBeamforming(array)
    doas_cbf = cbf.estimate(X, K=2)
    
    print(f"CBF estimates: {np.rad2deg(doas_cbf):.2f} degrees")
    
    # Step 4: DOA estimation with MUSIC
    print("\n4. DOA estimation with MUSIC...")
    music = MUSIC(array)
    doas_music = music.estimate(X, K=2)
    
    print(f"MUSIC estimates: {np.rad2deg(doas_music):.2f} degrees")
    
    # Step 5: Error analysis
    print("\n5. Error Analysis...")
    
    # CBF errors
    cbf_errors = np.abs(np.sort(doas_cbf) - np.sort(doas_true))
    cbf_rmse = np.sqrt(np.mean(cbf_errors**2))
    
    # MUSIC errors
    music_errors = np.abs(np.sort(doas_music) - np.sort(doas_true))
    music_rmse = np.sqrt(np.mean(music_errors**2))
    
    print(f"CBF RMSE: {np.rad2deg(cbf_rmse):.2f} degrees")
    print(f"MUSIC RMSE: {np.rad2deg(music_rmse):.2f} degrees")
    
    # Step 6: Show beam pattern (conceptual)
    print("\n6. Computing beam patterns...")
    angle_grid = array.angle_grid(N_grid=181)
    
    cbf_pattern = cbf.beam_pattern(X, angle_grid)
    print(f"CBF beam pattern computed over {len(angle_grid)} angles")
    
    # Find peaks in beam pattern
    from doa_methods.utils.peak_finding import find_peaks
    cbf_peaks = find_peaks(angle_grid, cbf_pattern, method='local_maxima')
    print(f"CBF peaks found at: {np.rad2deg(cbf_peaks[:2]):.2f} degrees")
    
    print("\n" + "=" * 40)
    print("Quick start example completed!")
    print("\nNext steps:")
    print("- Try examples/demo.py for comprehensive demonstrations")
    print("- Explore tutorials/ for detailed explanations")
    print("- See src/doa_methods/ for implementation details")


if __name__ == "__main__":
    quick_start_example()