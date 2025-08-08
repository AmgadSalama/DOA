"""
Tests for Classical DOA Methods
===============================

Unit tests for Conventional Beamforming and Capon methods.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doa_methods.array_processing import UniformLinearArray, SignalModel
from doa_methods.classical import ConventionalBeamforming, CaponBeamforming


class TestConventionalBeamforming(unittest.TestCase):
    """Test cases for Conventional Beamforming."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=8, d=0.5)
        self.cbf = ConventionalBeamforming(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_initialization(self):
        """Test CBF initialization."""
        self.assertEqual(self.cbf.array, self.array)
        
    def test_beam_pattern_shape(self):
        """Test beam pattern computation shape."""
        # Generate test data
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=100, snr_db=10, seed=42)
        
        # Compute beam pattern
        angle_grid = np.linspace(-np.pi/2, np.pi/2, 181)
        pattern = self.cbf.beam_pattern(X, angle_grid)
        
        # Check shape and properties
        self.assertEqual(len(pattern), len(angle_grid))
        self.assertTrue(np.all(pattern >= 0))  # Power should be non-negative
        self.assertTrue(np.all(np.isreal(pattern)))  # Should be real-valued
        
    def test_broadside_source_detection(self):
        """Test detection of broadside source."""
        # Generate data with source at broadside
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        # Estimate DOAs
        doas_est = self.cbf.estimate(X, K=1)
        
        # Should detect source near broadside
        self.assertEqual(len(doas_est), 1)
        self.assertLess(np.abs(doas_est[0] - 0), np.deg2rad(5))  # Within 5 degrees
        
    def test_two_source_estimation(self):
        """Test estimation with two well-separated sources."""
        doas_true = np.deg2rad([-30, 30])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=10, seed=42)
        
        doas_est = self.cbf.estimate(X, K=2)
        
        # Should estimate two sources
        self.assertEqual(len(doas_est), 2)
        
        # Check if estimates are reasonably close to true values
        doas_est_sorted = np.sort(doas_est)
        doas_true_sorted = np.sort(doas_true)
        
        for i in range(2):
            self.assertLess(np.abs(doas_est_sorted[i] - doas_true_sorted[i]), 
                           np.deg2rad(10))  # Within 10 degrees
            
    def test_directivity_pattern(self):
        """Test directivity pattern computation."""
        theta0 = np.deg2rad(30)  # Steering direction
        angle_grid = np.linspace(-np.pi/2, np.pi/2, 181)
        
        directivity = self.cbf.directivity_pattern(angle_grid, theta0)
        
        # Check shape and properties
        self.assertEqual(len(directivity), len(angle_grid))
        self.assertTrue(np.all(directivity >= 0))
        
        # Maximum should be at or near steering direction
        max_idx = np.argmax(directivity)
        max_angle = angle_grid[max_idx]
        self.assertLess(np.abs(max_angle - theta0), np.deg2rad(2))  # Within 2 degrees
        
    def test_angular_resolution(self):
        """Test angular resolution estimate."""
        resolution = self.cbf.angular_resolution()
        
        # Should be positive and reasonable for this array
        self.assertGreater(resolution, 0)
        self.assertLess(resolution, np.pi)  # Should be less than 180 degrees
        
        # For 8-element array with d=0.5λ, expect resolution ~14 degrees
        expected_resolution = 2 / (self.array.M * self.array.d)
        self.assertAlmostEqual(resolution, expected_resolution, places=3)


class TestCaponBeamforming(unittest.TestCase):
    """Test cases for Capon Beamforming."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=8, d=0.5)
        self.capon = CaponBeamforming(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_initialization(self):
        """Test Capon initialization."""
        self.assertEqual(self.capon.array, self.array)
        self.assertEqual(self.capon.diagonal_loading, 0.0)
        
    def test_initialization_with_loading(self):
        """Test Capon with diagonal loading."""
        loading = 0.01
        capon_loaded = CaponBeamforming(self.array, diagonal_loading=loading)
        self.assertEqual(capon_loaded.diagonal_loading, loading)
        
    def test_beam_pattern_shape(self):
        """Test Capon beam pattern shape."""
        # Generate test data
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=100, snr_db=10, seed=42)
        
        angle_grid = np.linspace(-np.pi/2, np.pi/2, 181)
        pattern = self.capon.beam_pattern(X, angle_grid)
        
        # Check properties
        self.assertEqual(len(pattern), len(angle_grid))
        self.assertTrue(np.all(pattern > 0))  # Should be positive
        self.assertTrue(np.all(np.isreal(pattern)))
        
    def test_better_resolution_than_cbf(self):
        """Test that Capon has better resolution than CBF."""
        # Two closely spaced sources
        doas_true = np.deg2rad([-8, 8])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        # Compare beam patterns
        angle_grid = np.linspace(-np.pi/2, np.pi/2, 361)
        
        cbf = ConventionalBeamforming(self.array)
        cbf_pattern = cbf.beam_pattern(X, angle_grid)
        capon_pattern = self.capon.beam_pattern(X, angle_grid)
        
        # Capon should have sharper peaks
        # Find peak indices
        cbf_peaks = self._find_peaks(cbf_pattern)
        capon_peaks = self._find_peaks(capon_pattern)
        
        # Capon should resolve sources better (more distinct peaks)
        self.assertGreaterEqual(len(capon_peaks), len(cbf_peaks))
        
    def test_weights_computation(self):
        """Test Capon weights computation."""
        # Generate test data
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=100, snr_db=10, seed=42)
        
        # Compute weights for broadside
        theta0 = 0.0
        weights = self.capon.weights(X, theta0)
        
        # Check properties
        self.assertEqual(len(weights), self.array.M)
        self.assertTrue(np.iscomplexobj(weights))
        
        # Weights should satisfy distortionless constraint
        a = self.array.steering_vector(theta0)
        response = np.conj(weights).T @ a
        self.assertAlmostEqual(np.abs(response), 1.0, places=3)
        
    def test_beamform_output(self):
        """Test beamformed output computation."""
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=50, snr_db=10, seed=42)
        
        # Beamform towards source
        theta0 = 0.0
        y = self.capon.beamform_output(X, theta0)
        
        # Check shape and type
        self.assertEqual(y.shape, (1, X.shape[1]))
        self.assertTrue(np.iscomplexobj(y))
        
    def test_output_power(self):
        """Test output power computation."""
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=100, snr_db=10, seed=42)
        
        # Power when steering toward source vs away
        power_toward = self.capon.output_power(X, 0.0)
        power_away = self.capon.output_power(X, np.deg2rad(60))
        
        # Power should be higher when steering toward source
        self.assertGreater(power_toward, power_away)
        
    def test_diagonal_loading_effect(self):
        """Test effect of diagonal loading."""
        # Generate data with low SNR
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=20, snr_db=0, seed=42)
        
        # Compare with and without loading
        capon_no_loading = CaponBeamforming(self.array, diagonal_loading=0.0)
        capon_with_loading = CaponBeamforming(self.array, diagonal_loading=0.1)
        
        try:
            doas_no_loading = capon_no_loading.estimate(X, K=1)
            doas_with_loading = capon_with_loading.estimate(X, K=1)
            
            # Both should produce estimates
            self.assertEqual(len(doas_no_loading), 1)
            self.assertEqual(len(doas_with_loading), 1)
            
        except np.linalg.LinAlgError:
            # Loading should prevent singular matrix errors
            self.fail("Diagonal loading should prevent singular matrix errors")
            
    def test_set_diagonal_loading(self):
        """Test setting diagonal loading parameter."""
        new_loading = 0.05
        self.capon.set_diagonal_loading(new_loading)
        self.assertEqual(self.capon.diagonal_loading, new_loading)
        
    def _find_peaks(self, pattern, min_height=None):
        """Helper function to find peaks in pattern."""
        if min_height is None:
            min_height = 0.1 * np.max(pattern)
            
        peaks = []
        for i in range(1, len(pattern) - 1):
            if (pattern[i] > pattern[i-1] and 
                pattern[i] > pattern[i+1] and 
                pattern[i] > min_height):
                peaks.append(i)
        return peaks


class TestMethodComparison(unittest.TestCase):
    """Test comparison between classical methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=16, d=0.5)
        self.cbf = ConventionalBeamforming(self.array)
        self.capon = CaponBeamforming(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_single_source_comparison(self):
        """Compare methods for single source."""
        doas_true = np.deg2rad([15])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        doas_cbf = self.cbf.estimate(X, K=1)
        doas_capon = self.capon.estimate(X, K=1)
        
        # Both should detect the source
        self.assertEqual(len(doas_cbf), 1)
        self.assertEqual(len(doas_capon), 1)
        
        # Both should be reasonably accurate
        error_cbf = np.abs(doas_cbf[0] - doas_true[0])
        error_capon = np.abs(doas_capon[0] - doas_true[0])
        
        self.assertLess(error_cbf, np.deg2rad(5))
        self.assertLess(error_capon, np.deg2rad(5))
        
    def test_resolution_comparison(self):
        """Compare resolution capabilities."""
        # Closely spaced sources
        doas_true = np.deg2rad([-5, 5])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=300, snr_db=20, seed=42)
        
        # Try to resolve both sources
        doas_cbf = self.cbf.estimate(X, K=2)
        doas_capon = self.capon.estimate(X, K=2)
        
        # Capon should generally perform better at resolving close sources
        # This is a statistical test, so we check that at least one method works
        cbf_resolved = len(doas_cbf) == 2
        capon_resolved = len(doas_capon) == 2
        
        self.assertTrue(cbf_resolved or capon_resolved)
        
        # If both resolve, Capon should typically be more accurate
        if cbf_resolved and capon_resolved:
            cbf_errors = np.abs(np.sort(doas_cbf) - np.sort(doas_true))
            capon_errors = np.abs(np.sort(doas_capon) - np.sort(doas_true))
            
            cbf_rmse = np.sqrt(np.mean(cbf_errors**2))
            capon_rmse = np.sqrt(np.mean(capon_errors**2))
            
            # This might not always be true due to randomness, but often is
            # We just check that both are reasonable
            self.assertLess(cbf_rmse, np.deg2rad(10))
            self.assertLess(capon_rmse, np.deg2rad(10))


if __name__ == '__main__':
    unittest.main()