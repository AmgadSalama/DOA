"""
Tests for Subspace DOA Methods
==============================

Unit tests for MUSIC, Root-MUSIC, ESPRIT, and Unitary ESPRIT methods.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doa_methods.array_processing import UniformLinearArray, SignalModel
from doa_methods.subspace import MUSIC, RootMUSIC, ESPRIT, UnitaryESPRIT


class TestMUSIC(unittest.TestCase):
    """Test cases for MUSIC algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=16, d=0.5)
        self.music = MUSIC(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_initialization(self):
        """Test MUSIC initialization."""
        self.assertEqual(self.music.array, self.array)
        
    def test_single_source_estimation(self):
        """Test MUSIC with single source."""
        doas_true = np.deg2rad([20])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=500, snr_db=20, seed=42)
        
        doas_est = self.music.estimate(X, K=1)
        
        # Should find one source
        self.assertEqual(len(doas_est), 1)
        
        # Should be reasonably accurate
        error = np.abs(doas_est[0] - doas_true[0])
        self.assertLess(error, np.deg2rad(2))
        
    def test_two_source_estimation(self):
        """Test MUSIC with two sources."""
        doas_true = np.deg2rad([-25, 15])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=300, snr_db=15, seed=42)
        
        doas_est = self.music.estimate(X, K=2)
        
        # Should find two sources
        self.assertEqual(len(doas_est), 2)
        
        # Match estimates to true DOAs
        doas_est_sorted = np.sort(doas_est)
        doas_true_sorted = np.sort(doas_true)
        
        for i in range(2):
            error = np.abs(doas_est_sorted[i] - doas_true_sorted[i])
            self.assertLess(error, np.deg2rad(3))
            
    def test_music_spectrum(self):
        """Test MUSIC spectrum computation."""
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        # Get noise subspace
        R = X @ X.conj().T / X.shape[1]
        eigenvals, eigenvecs = np.linalg.eigh(R)
        idx = np.argsort(eigenvals)[::-1]
        U_n = eigenvecs[:, idx[1:]]  # Noise subspace (assuming K=1)
        
        # Compute spectrum
        angle_grid = np.linspace(-np.pi/2, np.pi/2, 181)
        spectrum = self.music.music_spectrum_vectorized(angle_grid, U_n)
        
        # Check properties
        self.assertEqual(len(spectrum), len(angle_grid))
        self.assertTrue(np.all(spectrum > 0))
        self.assertTrue(np.all(np.isfinite(spectrum)))
        
        # Should have peak near true DOA
        peak_idx = np.argmax(spectrum)
        peak_angle = angle_grid[peak_idx]
        self.assertLess(np.abs(peak_angle - 0), np.deg2rad(5))
        
    def test_source_number_estimation(self):
        """Test source number estimation."""
        # Known case: 2 sources
        doas_true = np.deg2rad([-20, 20])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=500, snr_db=20, seed=42)
        
        K_est = self.music.source_number_estimation(X, method='eigenvalue_gap')
        
        # Should estimate close to true number (allowing some tolerance)
        self.assertGreaterEqual(K_est, 1)
        self.assertLessEqual(K_est, 4)  # Reasonable range
        
    def test_forward_backward_averaging(self):
        """Test forward-backward averaging."""
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=100, snr_db=10, seed=42)
        
        # Estimate with and without FB averaging
        doas_no_fb = self.music.estimate(X, K=1, use_fb_averaging=False)
        doas_with_fb = self.music.estimate(X, K=1, use_fb_averaging=True)
        
        # Both should work
        self.assertEqual(len(doas_no_fb), 1)
        self.assertEqual(len(doas_with_fb), 1)
        
        # Both should be reasonably accurate
        self.assertLess(np.abs(doas_no_fb[0] - 0), np.deg2rad(5))
        self.assertLess(np.abs(doas_with_fb[0] - 0), np.deg2rad(5))


class TestRootMUSIC(unittest.TestCase):
    """Test cases for Root-MUSIC algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=16, d=0.5)
        self.root_music = RootMUSIC(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_initialization(self):
        """Test Root-MUSIC initialization."""
        self.assertEqual(self.root_music.array, self.array)
        
    def test_single_source_estimation(self):
        """Test Root-MUSIC with single source."""
        doas_true = np.deg2rad([15])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=300, snr_db=20, seed=42)
        
        doas_est = self.root_music.estimate(X, K=1)
        
        self.assertEqual(len(doas_est), 1)
        error = np.abs(doas_est[0] - doas_true[0])
        self.assertLess(error, np.deg2rad(3))
        
    def test_two_source_estimation(self):
        """Test Root-MUSIC with two sources."""
        doas_true = np.deg2rad([-20, 25])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=400, snr_db=15, seed=42)
        
        doas_est = self.root_music.estimate(X, K=2)
        
        # Should find close to correct number of sources
        self.assertGreaterEqual(len(doas_est), 1)
        self.assertLessEqual(len(doas_est), 3)  # Allow some tolerance
        
    def test_comparison_with_music(self):
        """Test that Root-MUSIC gives similar results to MUSIC."""
        doas_true = np.deg2rad([10])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=500, snr_db=20, seed=42)
        
        # Get results from both methods
        music = MUSIC(self.array)
        doas_music = music.estimate(X, K=1)
        doas_root_music = self.root_music.estimate(X, K=1)
        
        # Both should find one source
        self.assertEqual(len(doas_music), 1)
        self.assertEqual(len(doas_root_music), 1)
        
        # Results should be similar (within a few degrees)
        difference = np.abs(doas_music[0] - doas_root_music[0])
        self.assertLess(difference, np.deg2rad(5))


class TestESPRIT(unittest.TestCase):
    """Test cases for ESPRIT algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=16, d=0.5)
        self.esprit = ESPRIT(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_initialization(self):
        """Test ESPRIT initialization."""
        self.assertEqual(self.esprit.array, self.array)
        
    def test_single_source_estimation(self):
        """Test ESPRIT with single source."""
        doas_true = np.deg2rad([12])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=300, snr_db=20, seed=42)
        
        doas_est = self.esprit.estimate(X, K=1)
        
        self.assertEqual(len(doas_est), 1)
        error = np.abs(doas_est[0] - doas_true[0])
        self.assertLess(error, np.deg2rad(3))
        
    def test_two_source_estimation(self):
        """Test ESPRIT with two sources."""
        doas_true = np.deg2rad([-18, 22])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=400, snr_db=18, seed=42)
        
        doas_est = self.esprit.estimate(X, K=2)
        
        # Should estimate reasonably close to correct number
        self.assertGreaterEqual(len(doas_est), 1)
        self.assertLessEqual(len(doas_est), 3)
        
    def test_ls_vs_tls_methods(self):
        """Test different least squares methods."""
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        # Test both LS and TLS
        doas_ls = self.esprit.estimate(X, K=1, ls_method='ls')
        doas_tls = self.esprit.estimate(X, K=1, ls_method='total')
        
        # Both should work
        self.assertEqual(len(doas_ls), 1)
        self.assertEqual(len(doas_tls), 1)
        
        # Both should be reasonably accurate
        self.assertLess(np.abs(doas_ls[0] - 0), np.deg2rad(5))
        self.assertLess(np.abs(doas_tls[0] - 0), np.deg2rad(5))
        
    def test_estimate_with_pairing(self):
        """Test ESPRIT with pairing information."""
        doas_true = np.deg2rad([15])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        results = self.esprit.estimate_with_pairing(X, K=1, return_eigenvals=True)
        
        # Check return structure
        self.assertIn('doas', results)
        self.assertIn('phi_eigenvalues', results)
        self.assertIn('spatial_frequencies', results)
        self.assertIn('signal_eigenvalues', results)
        
        # Check DOA estimate
        self.assertEqual(len(results['doas']), 1)
        error = np.abs(results['doas'][0] - doas_true[0])
        self.assertLess(error, np.deg2rad(5))


class TestUnitaryESPRIT(unittest.TestCase):
    """Test cases for Unitary ESPRIT algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=16, d=0.5)
        self.unitary_esprit = UnitaryESPRIT(self.array)
        self.signal_model = SignalModel(self.array)
        
    def test_initialization(self):
        """Test Unitary ESPRIT initialization."""
        self.assertEqual(self.unitary_esprit.array, self.array)
        self.assertEqual(self.unitary_esprit.M, 16)
        
    def test_single_source_estimation(self):
        """Test Unitary ESPRIT with single source."""
        doas_true = np.deg2rad([8])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=300, snr_db=20, seed=42)
        
        doas_est = self.unitary_esprit.estimate(X, K=1)
        
        self.assertEqual(len(doas_est), 1)
        error = np.abs(doas_est[0] - doas_true[0])
        self.assertLess(error, np.deg2rad(5))
        
    def test_unitary_transformation(self):
        """Test unitary transformation properties."""
        # Generate test covariance matrix
        doas_true = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=200, snr_db=15, seed=42)
        
        R = X @ X.conj().T / X.shape[1]
        
        # Apply forward-backward averaging
        R_fb = self.unitary_esprit._forward_backward_averaging(R)
        
        # Check centro-Hermitian property
        M = self.array.M
        J = np.eye(M)[::-1]  # Exchange matrix
        
        # R_fb should satisfy: R_fb = J @ R_fb^* @ J
        expected = J @ R_fb.conj() @ J
        np.testing.assert_allclose(R_fb, expected, rtol=1e-10)
        
        # Apply unitary transformation
        R_real = self.unitary_esprit._unitary_transformation(R_fb)
        
        # Should be real
        self.assertTrue(np.allclose(R_real.imag, 0))
        
    def test_computational_savings(self):
        """Test computational savings estimation."""
        savings = self.unitary_esprit.computational_savings()
        
        # Check return structure
        self.assertIn('complex_flops', savings)
        self.assertIn('real_flops', savings)
        self.assertIn('computational_savings', savings)
        self.assertIn('memory_savings', savings)
        
        # Savings should be positive
        self.assertGreater(savings['computational_savings'], 1.0)
        self.assertEqual(savings['memory_savings'], 2.0)
        
    def test_stability_comparison(self):
        """Test stability comparison with standard ESPRIT."""
        doas_true = np.deg2rad([5])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=100, snr_db=10, seed=42)
        
        comparison = self.unitary_esprit.stability_comparison(X, K=1)
        
        # Check return structure
        self.assertIn('doas_unitary', comparison)
        self.assertIn('doas_standard', comparison)
        self.assertIn('condition_number_standard', comparison)
        self.assertIn('condition_number_unitary', comparison)
        self.assertIn('condition_improvement', comparison)
        self.assertIn('doa_differences', comparison)
        
        # Both methods should produce estimates
        self.assertEqual(len(comparison['doas_unitary']), 1)
        self.assertEqual(len(comparison['doas_standard']), 1)
        
        # Condition improvement should be positive
        self.assertGreater(comparison['condition_improvement'], 0)


class TestSubspaceMethodComparison(unittest.TestCase):
    """Test comparison between subspace methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=20, d=0.5)
        self.signal_model = SignalModel(self.array)
        
        self.music = MUSIC(self.array)
        self.root_music = RootMUSIC(self.array)
        self.esprit = ESPRIT(self.array)
        
    def test_single_source_comparison(self):
        """Compare all methods for single source."""
        doas_true = np.deg2rad([18])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=500, snr_db=20, seed=42)
        
        # Get estimates from all methods
        doas_music = self.music.estimate(X, K=1)
        doas_root = self.root_music.estimate(X, K=1)
        doas_esprit = self.esprit.estimate(X, K=1)
        
        # All should find one source
        self.assertEqual(len(doas_music), 1)
        self.assertEqual(len(doas_root), 1)
        self.assertEqual(len(doas_esprit), 1)
        
        # All should be reasonably accurate
        methods = ['MUSIC', 'Root-MUSIC', 'ESPRIT']
        estimates = [doas_music[0], doas_root[0], doas_esprit[0]]
        
        for method, estimate in zip(methods, estimates):
            error = np.abs(estimate - doas_true[0])
            self.assertLess(error, np.deg2rad(3), 
                           f"{method} error too large: {np.rad2deg(error):.2f}°")
            
    def test_resolution_comparison(self):
        """Compare resolution capabilities."""
        # Moderately spaced sources for fair comparison
        doas_true = np.deg2rad([-12, 12])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas_true, N_snapshots=1000, snr_db=20, seed=42)
        
        # Get estimates
        doas_music = self.music.estimate(X, K=2)
        doas_root = self.root_music.estimate(X, K=2)
        doas_esprit = self.esprit.estimate(X, K=2)
        
        # Count how many methods successfully resolve both sources
        successful_methods = 0
        
        for method_name, doas_est in [('MUSIC', doas_music), 
                                     ('Root-MUSIC', doas_root),
                                     ('ESPRIT', doas_esprit)]:
            if len(doas_est) == 2:
                # Check if estimates are close to true values
                doas_est_sorted = np.sort(doas_est)
                doas_true_sorted = np.sort(doas_true)
                
                errors = np.abs(doas_est_sorted - doas_true_sorted)
                if np.all(errors < np.deg2rad(5)):
                    successful_methods += 1
        
        # At least one method should succeed with these conditions
        self.assertGreater(successful_methods, 0)


if __name__ == '__main__':
    unittest.main()