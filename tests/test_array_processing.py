"""
Tests for Array Processing Module
=================================

Unit tests for UniformLinearArray and SignalModel classes.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from doa_methods.array_processing import UniformLinearArray, SignalModel


class TestUniformLinearArray(unittest.TestCase):
    """Test cases for UniformLinearArray class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=8, d=0.5, fc=1e9)
        
    def test_array_initialization(self):
        """Test array initialization parameters."""
        self.assertEqual(self.array.M, 8)
        self.assertEqual(self.array.d, 0.5)
        self.assertEqual(self.array.fc, 1e9)
        self.assertEqual(self.array.wavelength, 0.3)  # c/fc = 3e8/1e9
        
    def test_element_positions(self):
        """Test element position calculation."""
        expected_positions = np.arange(8) * 0.5
        np.testing.assert_array_equal(self.array.element_positions, expected_positions)
        
    def test_steering_vector_single_angle(self):
        """Test steering vector for single angle."""
        theta = 0.0  # Broadside
        a = self.array.steering_vector(theta)
        
        # Check shape and type
        self.assertEqual(a.shape, (8,))
        self.assertTrue(np.iscomplexobj(a))
        
        # For broadside, all elements should have same phase
        expected = np.ones(8, dtype=complex)
        np.testing.assert_array_almost_equal(a, expected)
        
    def test_steering_vector_multiple_angles(self):
        """Test steering vector for multiple angles."""
        thetas = np.array([-np.pi/4, 0, np.pi/4])
        A = self.array.steering_vector(thetas)
        
        # Check shape
        self.assertEqual(A.shape, (8, 3))
        
        # Check broadside column
        np.testing.assert_array_almost_equal(A[:, 1], np.ones(8, dtype=complex))
        
    def test_spatial_frequency(self):
        """Test spatial frequency calculation."""
        theta = np.pi/6  # 30 degrees
        omega = self.array.spatial_frequency(theta)
        
        expected = 2 * np.pi * 0.5 * np.sin(theta)
        self.assertAlmostEqual(omega, expected)
        
    def test_angle_grid(self):
        """Test angle grid generation."""
        grid = self.array.angle_grid(N_grid=181)
        
        self.assertEqual(len(grid), 181)
        self.assertAlmostEqual(grid[0], -np.pi/2)
        self.assertAlmostEqual(grid[-1], np.pi/2)
        
    def test_array_manifold(self):
        """Test array manifold matrix."""
        thetas = np.linspace(-np.pi/2, np.pi/2, 5)
        A = self.array.array_manifold(thetas)
        
        self.assertEqual(A.shape, (8, 5))
        
        # Check that each column is a valid steering vector
        for i, theta in enumerate(thetas):
            expected_col = self.array.steering_vector(theta)
            np.testing.assert_array_almost_equal(A[:, i], expected_col)


class TestSignalModel(unittest.TestCase):
    """Test cases for SignalModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.array = UniformLinearArray(M=8, d=0.5)
        self.signal_model = SignalModel(self.array)
        
    def test_signal_model_initialization(self):
        """Test signal model initialization."""
        self.assertEqual(self.signal_model.array, self.array)
        self.assertEqual(self.signal_model.M, 8)
        
    def test_generate_signals_basic(self):
        """Test basic signal generation."""
        doas = np.array([0.0])  # Single source at broadside
        X, S, N = self.signal_model.generate_signals(
            doas=doas,
            N_snapshots=100,
            snr_db=10,
            seed=42
        )
        
        # Check shapes
        self.assertEqual(X.shape, (8, 100))
        self.assertEqual(S.shape, (1, 100))
        self.assertEqual(N.shape, (8, 100))
        
        # Check data types
        self.assertTrue(np.iscomplexobj(X))
        self.assertTrue(np.iscomplexobj(S))
        self.assertTrue(np.iscomplexobj(N))
        
    def test_generate_signals_multiple_sources(self):
        """Test signal generation with multiple sources."""
        doas = np.deg2rad([-30, 30])  # Two sources
        X, S, N = self.signal_model.generate_signals(
            doas=doas,
            N_snapshots=50,
            snr_db=10,
            seed=42
        )
        
        # Check shapes
        self.assertEqual(X.shape, (8, 50))
        self.assertEqual(S.shape, (2, 50))
        self.assertEqual(N.shape, (8, 50))
        
    def test_sample_covariance(self):
        """Test sample covariance matrix calculation."""
        # Generate test data
        doas = np.deg2rad([0])
        X, _, _ = self.signal_model.generate_signals(
            doas=doas, N_snapshots=1000, snr_db=20, seed=42)
        
        # Compute sample covariance
        R = self.signal_model.sample_covariance(X)
        
        # Check properties
        self.assertEqual(R.shape, (8, 8))
        self.assertTrue(np.allclose(R, R.conj().T))  # Hermitian
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(R)
        self.assertTrue(np.all(np.real(eigenvals) > 0))
        
    def test_theoretical_covariance(self):
        """Test theoretical covariance matrix."""
        doas = np.deg2rad([0])
        powers = np.array([1.0])
        noise_power = 0.1
        
        R = self.signal_model.theoretical_covariance(doas, powers, noise_power)
        
        # Check shape and properties
        self.assertEqual(R.shape, (8, 8))
        self.assertTrue(np.allclose(R, R.conj().T))  # Hermitian
        
        # For single source at broadside with unit power and noise_power 0.1,
        # diagonal elements should be approximately 1.1
        self.assertAlmostEqual(np.real(R[0, 0]), 1.1, places=1)
        
    def test_snr_scaling(self):
        """Test SNR scaling in generated signals."""
        doas = np.deg2rad([0])
        
        # Generate signals with different SNRs
        X_low, S_low, N_low = self.signal_model.generate_signals(
            doas=doas, N_snapshots=1000, snr_db=0, seed=42)
        X_high, S_high, N_high = self.signal_model.generate_signals(
            doas=doas, N_snapshots=1000, snr_db=20, seed=42)
        
        # Measure actual SNRs
        signal_power_low = np.mean(np.abs(S_low)**2)
        noise_power_low = np.mean(np.abs(N_low)**2)
        snr_low = 10 * np.log10(signal_power_low / noise_power_low)
        
        signal_power_high = np.mean(np.abs(S_high)**2)
        noise_power_high = np.mean(np.abs(N_high)**2)
        snr_high = 10 * np.log10(signal_power_high / noise_power_high)
        
        # Check that high SNR is indeed higher
        self.assertGreater(snr_high, snr_low)
        
        # Check approximate values (within 2 dB tolerance)
        self.assertAlmostEqual(snr_low, 0, delta=2)
        self.assertAlmostEqual(snr_high, 20, delta=2)
        
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        doas = np.deg2rad([0])
        
        # Generate same data twice with same seed
        X1, S1, N1 = self.signal_model.generate_signals(
            doas=doas, N_snapshots=50, snr_db=10, seed=42)
        X2, S2, N2 = self.signal_model.generate_signals(
            doas=doas, N_snapshots=50, snr_db=10, seed=42)
        
        # Should be identical
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(N1, N2)
        
    def test_correlation_parameter(self):
        """Test correlated sources generation."""
        doas = np.deg2rad([-10, 10])
        
        # Generate uncorrelated sources
        X_uncorr, S_uncorr, _ = self.signal_model.generate_signals(
            doas=doas, N_snapshots=1000, snr_db=10, correlation=0.0, seed=42)
        
        # Generate correlated sources
        X_corr, S_corr, _ = self.signal_model.generate_signals(
            doas=doas, N_snapshots=1000, snr_db=10, correlation=0.8, seed=42)
        
        # Compute correlation coefficients
        corr_uncorr = np.corrcoef(S_uncorr)[0, 1]
        corr_corr = np.corrcoef(S_corr.real)[0, 1]  # Use real part for simplicity
        
        # Correlated sources should have higher correlation
        self.assertGreater(np.abs(corr_corr), np.abs(corr_uncorr))
        self.assertGreater(np.abs(corr_corr), 0.5)  # Should be significantly correlated


class TestArrayGeometry(unittest.TestCase):
    """Test cases for array geometry calculations."""
    
    def test_half_wavelength_spacing(self):
        """Test properties of half-wavelength spacing."""
        array = UniformLinearArray(M=4, d=0.5)
        
        # Test spatial frequency range
        theta_max = np.pi/2
        omega_max = array.spatial_frequency(theta_max)
        
        self.assertAlmostEqual(omega_max, np.pi)  # Should reach Nyquist limit
        
    def test_spatial_aliasing_detection(self):
        """Test detection of spatial aliasing conditions."""
        # Array with spacing > λ/2
        array_aliased = UniformLinearArray(M=4, d=0.8)
        
        # Endfire direction should exceed Nyquist
        omega_endfire = array_aliased.spatial_frequency(np.pi/2)
        self.assertGreater(np.abs(omega_endfire), np.pi)
        
    def test_different_frequencies(self):
        """Test array behavior at different frequencies."""
        # Same physical array at different frequencies
        array_1ghz = UniformLinearArray(M=8, d=0.5, fc=1e9)
        array_2ghz = UniformLinearArray(M=8, d=0.5, fc=2e9)
        
        # Physical spacing should be different
        self.assertNotEqual(array_1ghz.d_meters, array_2ghz.d_meters)
        
        # But electrical spacing (in wavelengths) should be same
        self.assertEqual(array_1ghz.d, array_2ghz.d)


if __name__ == '__main__':
    unittest.main()