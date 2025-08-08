"""
DOA Methods Demo Script
======================

Comprehensive demonstration of all implemented DOA estimation methods.
This script showcases the usage of different algorithms and provides
performance comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from typing import Dict, List

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doa_methods.array_processing import UniformLinearArray, SignalModel
from doa_methods.classical import ConventionalBeamforming, CaponBeamforming
from doa_methods.subspace import MUSIC, RootMUSIC, ESPRIT, UnitaryESPRIT
from doa_methods.simulation import SimulationScenario
from doa_methods.utils.peak_finding import find_peaks


class DOADemo:
    """
    Demonstration class for DOA estimation methods.
    """
    
    def __init__(self, M: int = 16, d: float = 0.5):
        """
        Initialize demo with array configuration.
        
        Parameters
        ----------
        M : int
            Number of array elements
        d : float
            Element spacing in wavelengths
        """
        self.array = UniformLinearArray(M=M, d=d)
        self.signal_model = SignalModel(self.array)
        
        # Initialize estimators
        self.estimators = {
            'Conventional BF': ConventionalBeamforming(self.array),
            'Capon': CaponBeamforming(self.array),
            'MUSIC': MUSIC(self.array),
            'Root-MUSIC': RootMUSIC(self.array),
            'ESPRIT': ESPRIT(self.array),
            'Unitary ESPRIT': UnitaryESPRIT(self.array)
        }
        
        print(f"Initialized DOA Demo with {M}-element ULA (d = {d}λ)")
        print(f"Available methods: {list(self.estimators.keys())}")
    
    def basic_example(self):
        """Run basic example with two sources."""
        print("\n" + "="*50)
        print("BASIC EXAMPLE: Two Sources")
        print("="*50)
        
        # Generate data
        doas_true = np.deg2rad([-20, 15])
        N_snapshots = 200
        snr_db = 10
        
        print(f"True DOAs: {np.rad2deg(doas_true)} degrees")
        print(f"SNR: {snr_db} dB, Snapshots: {N_snapshots}")
        
        X, S, N = self.signal_model.generate_signals(
            doas=doas_true,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            seed=42
        )
        
        # Run all estimators
        results = {}
        compute_times = {}
        
        for name, estimator in self.estimators.items():
            try:
                start_time = time.time()
                
                if name in ['Conventional BF', 'Capon']:
                    doas_est = estimator.estimate(X, K=2)
                else:
                    doas_est = estimator.estimate(X, K=2)
                
                compute_time = time.time() - start_time
                
                results[name] = doas_est
                compute_times[name] = compute_time
                
                errors = np.abs(np.sort(doas_est) - np.sort(doas_true))
                rmse = np.sqrt(np.mean(errors**2))
                
                print(f"\n{name}:")
                print(f"  Estimates: {np.rad2deg(doas_est):.2f} degrees")
                print(f"  RMSE: {np.rad2deg(rmse):.2f} degrees")
                print(f"  Time: {compute_time*1000:.2f} ms")
                
            except Exception as e:
                print(f"\n{name}: ERROR - {str(e)}")
                results[name] = None
                compute_times[name] = None
        
        return results, compute_times
    
    def resolution_test(self):
        """Test angular resolution capability."""
        print("\n" + "="*50)
        print("RESOLUTION TEST: Closely Spaced Sources")
        print("="*50)
        
        # Test different separations
        separations_deg = [2, 4, 6, 8, 10, 15, 20]
        N_snapshots = 500
        snr_db = 15
        
        resolution_results = {}
        
        for sep_deg in separations_deg:
            print(f"\nTesting separation: {sep_deg}°")
            
            # Generate data with symmetric sources
            sep_rad = np.deg2rad(sep_deg)
            doas_true = np.array([-sep_rad/2, sep_rad/2])
            
            X, _, _ = self.signal_model.generate_signals(
                doas=doas_true,
                N_snapshots=N_snapshots,
                snr_db=snr_db,
                seed=42
            )
            
            # Test high-resolution methods
            hr_methods = ['MUSIC', 'Root-MUSIC', 'ESPRIT']
            
            for method_name in hr_methods:
                if method_name not in resolution_results:
                    resolution_results[method_name] = {'separations': [], 'resolved': []}
                
                try:
                    estimator = self.estimators[method_name]
                    doas_est = estimator.estimate(X, K=2)
                    
                    # Check if sources are resolved (simple criterion)
                    if len(doas_est) == 2:
                        est_sep = np.abs(np.diff(np.sort(doas_est)))[0]
                        resolved = est_sep > 0.5 * sep_rad  # 50% of true separation
                    else:
                        resolved = False
                    
                    resolution_results[method_name]['separations'].append(sep_deg)
                    resolution_results[method_name]['resolved'].append(resolved)
                    
                    print(f"  {method_name}: {'✓' if resolved else '✗'}")
                    
                except:
                    resolution_results[method_name]['separations'].append(sep_deg)
                    resolution_results[method_name]['resolved'].append(False)
                    print(f"  {method_name}: ✗ (error)")
        
        return resolution_results
    
    def snr_performance(self):
        """Test performance vs SNR."""
        print("\n" + "="*50)
        print("SNR PERFORMANCE TEST")
        print("="*50)
        
        snr_range = np.arange(-5, 21, 5)
        doas_true = np.deg2rad([-25, 10])
        N_snapshots = 200
        N_trials = 20
        
        snr_results = {}
        
        for method_name in ['MUSIC', 'Capon', 'Conventional BF']:
            snr_results[method_name] = {'snr': [], 'rmse': [], 'success_rate': []}
            
            print(f"\nTesting {method_name}:")
            
            for snr_db in snr_range:
                print(f"  SNR = {snr_db} dB", end=' ')
                
                errors = []
                successes = 0
                
                for trial in range(N_trials):
                    try:
                        X, _, _ = self.signal_model.generate_signals(
                            doas=doas_true,
                            N_snapshots=N_snapshots,
                            snr_db=snr_db,
                            seed=trial
                        )
                        
                        estimator = self.estimators[method_name]
                        
                        if method_name in ['Conventional BF', 'Capon']:
                            doas_est = estimator.estimate(X, K=2)
                        else:
                            doas_est = estimator.estimate(X, K=2)
                        
                        if len(doas_est) == 2:
                            trial_errors = np.abs(np.sort(doas_est) - np.sort(doas_true))
                            errors.extend(trial_errors)
                            successes += 1
                        
                    except:
                        pass
                
                if len(errors) > 0:
                    rmse = np.sqrt(np.mean(np.array(errors)**2))
                else:
                    rmse = np.inf
                
                success_rate = successes / N_trials
                
                snr_results[method_name]['snr'].append(snr_db)
                snr_results[method_name]['rmse'].append(np.rad2deg(rmse))
                snr_results[method_name]['success_rate'].append(success_rate)
                
                print(f"RMSE: {np.rad2deg(rmse):.2f}°, Success: {success_rate:.1%}")
        
        return snr_results
    
    def demonstrate_beam_patterns(self):
        """Show beam patterns for classical methods."""
        print("\n" + "="*50)
        print("BEAM PATTERN DEMONSTRATION")
        print("="*50)
        
        # Generate test data
        doas_true = np.deg2rad([-30, 20])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true,
            N_snapshots=200,
            snr_db=15,
            seed=42
        )
        
        # Angle grid for beam patterns
        angle_grid = np.linspace(-np.pi/2, np.pi/2, 181)
        
        # Compute beam patterns
        cbf = self.estimators['Conventional BF']
        capon = self.estimators['Capon']
        music = self.estimators['MUSIC']
        
        cbf_pattern = cbf.beam_pattern(X, angle_grid)
        capon_pattern = capon.beam_pattern(X, angle_grid)
        music_spectrum = music.music_spectrum_vectorized(angle_grid, 
            music.estimate(X, K=2))  # Get noise subspace for spectrum
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(np.rad2deg(angle_grid), 10*np.log10(cbf_pattern))
        plt.axvline(np.rad2deg(doas_true[0]), color='r', linestyle='--', label='True DOA')
        plt.axvline(np.rad2deg(doas_true[1]), color='r', linestyle='--')
        plt.title('Conventional Beamforming')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(np.rad2deg(angle_grid), 10*np.log10(capon_pattern))
        plt.axvline(np.rad2deg(doas_true[0]), color='r', linestyle='--', label='True DOA')
        plt.axvline(np.rad2deg(doas_true[1]), color='r', linestyle='--')
        plt.title('Capon Beamforming')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('beam_patterns_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Beam patterns saved as 'beam_patterns_demo.png'")
    
    def showcase_scenarios(self):
        """Demonstrate predefined simulation scenarios."""
        print("\n" + "="*50)
        print("SIMULATION SCENARIOS SHOWCASE")
        print("="*50)
        
        scenarios = {
            'Two Close Sources': SimulationScenario.two_sources_close,
            'Multiple Uncorrelated': SimulationScenario.multiple_sources_uncorrelated,
            'Correlated Sources': SimulationScenario.correlated_sources,
            'Low SNR': SimulationScenario.low_snr_scenario,
            'Limited Snapshots': SimulationScenario.limited_snapshots
        }
        
        for scenario_name, scenario_func in scenarios.items():
            print(f"\n{scenario_name}:")
            
            # Generate scenario data
            scenario_data = scenario_func(self.array, seed=42)
            X = scenario_data['X']
            doas_true = scenario_data['doas_true']
            params = scenario_data['params']
            
            print(f"  True DOAs: {np.rad2deg(doas_true):.2f} degrees")
            print(f"  Parameters: {params}")
            
            # Test MUSIC on this scenario
            try:
                music = self.estimators['MUSIC']
                doas_est = music.estimate(X, K=len(doas_true))
                
                errors = np.abs(np.sort(doas_est) - np.sort(doas_true))
                rmse = np.sqrt(np.mean(errors**2))
                
                print(f"  MUSIC estimates: {np.rad2deg(doas_est):.2f} degrees")
                print(f"  RMSE: {np.rad2deg(rmse):.2f} degrees")
                
            except Exception as e:
                print(f"  MUSIC failed: {str(e)}")
    
    def computational_complexity(self):
        """Compare computational complexity."""
        print("\n" + "="*50)
        print("COMPUTATIONAL COMPLEXITY COMPARISON")
        print("="*50)
        
        # Test with different array sizes
        array_sizes = [8, 16, 32, 64]
        methods_to_test = ['Conventional BF', 'Capon', 'MUSIC', 'ESPRIT']
        
        for M in array_sizes:
            print(f"\nArray size: {M} elements")
            
            # Create temporary array
            temp_array = UniformLinearArray(M=M, d=0.5)
            temp_signal_model = SignalModel(temp_array)
            
            # Generate test data
            doas_true = np.deg2rad([-20, 15])
            X, _, _ = temp_signal_model.generate_signals(
                doas=doas_true,
                N_snapshots=100,
                snr_db=10,
                seed=42
            )
            
            for method_name in methods_to_test:
                if method_name == 'Conventional BF':
                    estimator = ConventionalBeamforming(temp_array)
                elif method_name == 'Capon':
                    estimator = CaponBeamforming(temp_array)
                elif method_name == 'MUSIC':
                    estimator = MUSIC(temp_array)
                elif method_name == 'ESPRIT':
                    estimator = ESPRIT(temp_array)
                
                # Time the estimation
                start_time = time.time()
                try:
                    doas_est = estimator.estimate(X, K=2)
                    compute_time = time.time() - start_time
                    print(f"  {method_name}: {compute_time*1000:.2f} ms")
                except:
                    print(f"  {method_name}: Failed")


def main():
    """Main demo function."""
    print("DOA Methods Comprehensive Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = DOADemo(M=16, d=0.5)
    
    # Run demonstrations
    try:
        # Basic example
        results, times = demo.basic_example()
        
        # Resolution test
        resolution_results = demo.resolution_test()
        
        # SNR performance
        snr_results = demo.snr_performance()
        
        # Beam patterns (if matplotlib available)
        try:
            demo.demonstrate_beam_patterns()
        except ImportError:
            print("Matplotlib not available - skipping beam pattern plots")
        
        # Simulation scenarios
        demo.showcase_scenarios()
        
        # Computational complexity
        demo.computational_complexity()
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*50)
        print("\nKey Takeaways:")
        print("- High-resolution methods (MUSIC, ESPRIT) provide better angular resolution")
        print("- Classical methods are more robust in low SNR conditions")
        print("- Root-MUSIC avoids spectral search, improving speed")
        print("- Unitary ESPRIT offers numerical advantages")
        print("\nFor detailed tutorials, see the notebooks in tutorials/")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()