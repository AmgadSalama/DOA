"""
MUSIC (Multiple Signal Classification)
=====================================

Implementation of the MUSIC DOA estimation algorithm.
MUSIC is a high-resolution subspace-based method that exploits
the orthogonality between signal and noise subspaces.
"""

import numpy as np
from typing import Optional, Union
from ..array_processing import UniformLinearArray
from ..utils.peak_finding import find_peaks
from ..utils.math_utils import eigendecomposition, signal_noise_subspaces


class MUSIC:
    """
    MUSIC DOA Estimator.
    
    MUSIC creates a spectrum by projecting steering vectors onto the 
    noise subspace. DOAs correspond to nulls in this projection,
    which appear as peaks in the MUSIC spectrum.
    
    The MUSIC spectrum is:
    P_MUSIC(θ) = 1 / (a^H(θ) U_n U_n^H a(θ))
    
    where U_n is the noise subspace matrix.
    """
    
    def __init__(self, array: UniformLinearArray):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        """
        self.array = array
        
    def estimate(self,
                X: np.ndarray,
                K: int,
                angle_grid: Optional[np.ndarray] = None,
                use_fb_averaging: bool = False,
                eigenvalue_threshold: Optional[float] = None) -> np.ndarray:
        """
        Estimate DOAs using MUSIC.
        
        Parameters
        ----------
        X : np.ndarray
            Received data matrix (M × N)
        K : int
            Number of sources
        angle_grid : np.ndarray, optional
            Grid of angles to search over
        use_fb_averaging : bool
            Use forward-backward averaging
        eigenvalue_threshold : float, optional
            Threshold for eigenvalue-based source detection
            
        Returns
        -------
        np.ndarray
            Estimated DOAs in radians
        """
        if angle_grid is None:
            angle_grid = self.array.angle_grid()
            
        # Sample covariance matrix
        R = X @ X.conj().T / X.shape[1]
        
        # Apply forward-backward averaging if requested
        if use_fb_averaging:
            from ..utils.math_utils import forward_backward_averaging
            R = forward_backward_averaging(R)
        
        # Eigendecomposition
        eigenvals, eigenvecs = eigendecomposition(R, 'descending')
        
        # Automatic source number estimation if threshold provided
        if eigenvalue_threshold is not None and K is None:
            K = self._estimate_source_number(eigenvals, eigenvalue_threshold)
        
        if K >= self.array.M:
            raise ValueError(f"Number of sources ({K}) must be less than array size ({self.array.M})")
        
        # Noise subspace
        U_n = eigenvecs[:, K:]
        
        # Compute MUSIC spectrum
        music_spectrum = self.music_spectrum(angle_grid, U_n)
        
        # Find peaks
        doas = find_peaks(angle_grid, music_spectrum, method='local_maxima')
        
        # Return K largest peaks if more than K found
        if len(doas) > K:
            spectrum_values = np.array([music_spectrum[np.argmin(np.abs(angle_grid - doa))] for doa in doas])
            top_indices = np.argsort(spectrum_values)[::-1][:K]
            doas = doas[top_indices]
            
        return np.sort(doas)
    
    def music_spectrum(self, angle_grid: np.ndarray, U_n: np.ndarray) -> np.ndarray:
        """
        Compute MUSIC spectrum.
        
        Parameters
        ----------
        angle_grid : np.ndarray
            Grid of angles
        U_n : np.ndarray
            Noise subspace matrix (M × (M-K))
            
        Returns
        -------
        np.ndarray
            MUSIC spectrum
        """
        spectrum = np.zeros(len(angle_grid))
        
        for i, theta in enumerate(angle_grid):
            a = self.array.steering_vector(theta).reshape(-1, 1)
            
            # Projection onto noise subspace: a^H U_n U_n^H a
            projection = a.conj().T @ U_n @ U_n.conj().T @ a
            
            # MUSIC spectrum: 1 / projection
            spectrum[i] = 1.0 / (np.real(projection.item()) + 1e-12)
            
        return spectrum
    
    def music_spectrum_vectorized(self, angle_grid: np.ndarray, U_n: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of MUSIC spectrum (faster).
        
        Parameters
        ----------
        angle_grid : np.ndarray
            Grid of angles
        U_n : np.ndarray
            Noise subspace matrix
            
        Returns
        -------
        np.ndarray
            MUSIC spectrum
        """
        # Array manifold matrix
        A = self.array.array_manifold(angle_grid)  # M × N_grid
        
        # Compute projections: A^H U_n U_n^H A
        projections = np.sum(np.abs(A.conj().T @ U_n)**2, axis=1)
        
        # MUSIC spectrum
        spectrum = 1.0 / (projections + 1e-12)
        
        return spectrum
    
    def source_number_estimation(self, X: np.ndarray, method: str = 'aic') -> int:
        """
        Estimate number of sources.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        method : str
            Estimation method ('aic', 'mdl', 'eigenvalue_gap')
            
        Returns
        -------
        int
            Estimated number of sources
        """
        R = X @ X.conj().T / X.shape[1]
        eigenvals, _ = eigendecomposition(R, 'descending')
        
        if method in ['aic', 'mdl']:
            from ..utils.math_utils import source_number_estimation
            return source_number_estimation(eigenvals, method)
        elif method == 'eigenvalue_gap':
            return self._eigenvalue_gap_method(eigenvals)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _estimate_source_number(self, eigenvals: np.ndarray, threshold: float) -> int:
        """Estimate source number using eigenvalue threshold."""
        noise_level = np.mean(eigenvals[-3:])  # Average of smallest eigenvalues
        return np.sum(eigenvals > threshold * noise_level)
    
    def _eigenvalue_gap_method(self, eigenvals: np.ndarray) -> int:
        """Estimate source number using eigenvalue gaps."""
        # Compute ratios of consecutive eigenvalues
        ratios = eigenvals[:-1] / eigenvals[1:]
        
        # Find largest gap (largest ratio)
        max_gap_idx = np.argmax(ratios)
        
        return max_gap_idx + 1
    
    def angular_resolution(self, snr_db: float = 10, N_snapshots: int = 100) -> float:
        """
        Theoretical angular resolution for MUSIC.
        
        MUSIC can resolve sources separated by approximately:
        Δθ ≈ √(6/(SNR × N × M^3)) radians (for high SNR)
        
        Parameters
        ----------
        snr_db : float
            SNR in dB
        N_snapshots : int
            Number of snapshots
            
        Returns
        -------
        float
            Angular resolution in radians
        """
        snr_linear = 10**(snr_db/10)
        resolution = np.sqrt(6 / (snr_linear * N_snapshots * self.array.M**3))
        return resolution
    
    def estimate_doa_errors(self, X: np.ndarray, K: int, doas_true: np.ndarray) -> dict:
        """
        Estimate DOA estimation errors using MUSIC.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        K : int
            Number of sources
        doas_true : np.ndarray
            True DOAs for comparison
            
        Returns
        -------
        dict
            Error analysis results
        """
        # Estimate DOAs
        doas_est = self.estimate(X, K)
        
        # Match estimated to true DOAs (simple nearest neighbor)
        if len(doas_est) == len(doas_true):
            errors = np.abs(doas_est - np.sort(doas_true))
        else:
            # Handle different number of estimates
            errors = np.full_like(doas_true, np.inf)
            for i, true_doa in enumerate(doas_true):
                if len(doas_est) > 0:
                    closest_idx = np.argmin(np.abs(doas_est - true_doa))
                    errors[i] = np.abs(doas_est[closest_idx] - true_doa)
        
        return {
            'doas_estimated': doas_est,
            'doas_true': doas_true,
            'absolute_errors': errors,
            'rmse': np.sqrt(np.mean(errors**2)),
            'mean_error': np.mean(errors),
            'max_error': np.max(errors)
        }
    
    def __call__(self, X: np.ndarray, array: UniformLinearArray, K: int = 2, **kwargs) -> np.ndarray:
        """Callable interface for Monte Carlo framework."""
        return self.estimate(X, K, **kwargs)