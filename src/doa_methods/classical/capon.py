"""
Capon Beamforming (MVDR)
=======================

Implementation of the Capon beamforming DOA estimation method.
Also known as Minimum Variance Distortionless Response (MVDR) beamformer.
"""

import numpy as np
from typing import Union, Optional
from ..array_processing import UniformLinearArray
from ..utils.peak_finding import find_peaks


class CaponBeamforming:
    """
    Capon (MVDR) Beamforming DOA Estimator.
    
    The Capon beamformer minimizes the output power while maintaining
    a distortionless response in the look direction. It has better
    resolution than conventional beamforming.
    
    The beam pattern is given by:
    P_Capon(θ) = 1 / (a^H(θ) R^(-1) a(θ))
    
    where:
    - a(θ): steering vector for angle θ  
    - R: sample covariance matrix
    """
    
    def __init__(self, array: UniformLinearArray, diagonal_loading: float = 0.0):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        diagonal_loading : float
            Diagonal loading factor for regularization (default: 0.0)
        """
        self.array = array
        self.diagonal_loading = diagonal_loading
        
    def estimate(self,
                X: np.ndarray,
                angle_grid: Optional[np.ndarray] = None,
                K: Optional[int] = None,
                peak_finding: str = 'max') -> np.ndarray:
        """
        Estimate DOAs using Capon beamforming.
        
        Parameters
        ----------
        X : np.ndarray
            Received data matrix (M × N)
        angle_grid : np.ndarray, optional
            Grid of angles to search over
        K : int, optional
            Number of sources
        peak_finding : str
            Peak finding method
            
        Returns
        -------
        np.ndarray
            Estimated DOAs in radians
        """
        # Default angle grid
        if angle_grid is None:
            angle_grid = self.array.angle_grid()
            
        # Compute beam pattern
        beam_pattern = self.beam_pattern(X, angle_grid)
        
        # Find peaks
        if K is not None:
            doas = self._find_k_peaks(angle_grid, beam_pattern, K)
        else:
            doas = find_peaks(angle_grid, beam_pattern, method=peak_finding)
            
        return np.array(doas)
    
    def beam_pattern(self, X: np.ndarray, angle_grid: np.ndarray) -> np.ndarray:
        """
        Compute the Capon beam pattern.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (M × N)
        angle_grid : np.ndarray
            Grid of angles
            
        Returns
        -------
        np.ndarray
            Beam pattern values
        """
        # Sample covariance matrix
        R = X @ X.conj().T / X.shape[1]
        
        # Add diagonal loading for regularization
        if self.diagonal_loading > 0:
            R += self.diagonal_loading * np.eye(self.array.M)
        
        # Compute inverse (with numerical stability check)
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if R is singular
            R_inv = np.linalg.pinv(R)
        
        # Array manifold
        A = self.array.array_manifold(angle_grid)
        
        # Compute Capon beam pattern: P(θ) = 1 / (a^H(θ) R^(-1) a(θ))
        beam_pattern = np.zeros(len(angle_grid))
        
        for i in range(len(angle_grid)):
            a = A[:, i:i+1]  # steering vector
            denominator = np.real(a.conj().T @ R_inv @ a)
            beam_pattern[i] = 1.0 / max(denominator, 1e-12)  # Avoid division by zero
            
        return beam_pattern.flatten()
    
    def beam_pattern_theoretical(self,
                               doas_true: np.ndarray,
                               powers: np.ndarray,
                               angle_grid: np.ndarray,
                               noise_power: float = 1.0) -> np.ndarray:
        """
        Compute theoretical Capon beam pattern.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True source DOAs
        powers : np.ndarray
            Source powers
        angle_grid : np.ndarray
            Angle grid
        noise_power : float
            Noise power
            
        Returns
        -------
        np.ndarray
            Theoretical beam pattern
        """
        # Theoretical covariance
        A_true = self.array.array_manifold(doas_true)
        P_s = np.diag(powers)
        R = A_true @ P_s @ A_true.conj().T + noise_power * np.eye(self.array.M)
        
        # Add diagonal loading
        if self.diagonal_loading > 0:
            R += self.diagonal_loading * np.eye(self.array.M)
            
        R_inv = np.linalg.inv(R)
        
        # Array manifold for grid
        A_grid = self.array.array_manifold(angle_grid)
        
        # Beam pattern
        beam_pattern = np.zeros(len(angle_grid))
        for i in range(len(angle_grid)):
            a = A_grid[:, i:i+1]
            denominator = np.real(a.conj().T @ R_inv @ a)
            beam_pattern[i] = 1.0 / max(denominator, 1e-12)
            
        return beam_pattern.flatten()
    
    def weights(self, X: np.ndarray, theta0: float) -> np.ndarray:
        """
        Compute Capon beamforming weights for steering angle theta0.
        
        The optimal weights are:
        w = R^(-1) a(θ0) / (a^H(θ0) R^(-1) a(θ0))
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (M × N)
        theta0 : float
            Steering angle in radians
            
        Returns
        -------
        np.ndarray
            Capon weights (M × 1)
        """
        # Sample covariance
        R = X @ X.conj().T / X.shape[1]
        
        # Add diagonal loading
        if self.diagonal_loading > 0:
            R += self.diagonal_loading * np.eye(self.array.M)
            
        # Steering vector
        a = self.array.steering_vector(theta0).reshape(-1, 1)
        
        # Compute weights
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
            
        numerator = R_inv @ a
        denominator = a.conj().T @ R_inv @ a
        
        weights = numerator / denominator
        
        return weights.flatten()
    
    def beamform_output(self, X: np.ndarray, theta0: float) -> np.ndarray:
        """
        Compute the beamformed output for steering angle theta0.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (M × N)
        theta0 : float
            Steering angle
            
        Returns
        -------
        np.ndarray
            Beamformed output (1 × N)
        """
        w = self.weights(X, theta0)
        return w.conj().T @ X
    
    def output_power(self, X: np.ndarray, theta0: float) -> float:
        """
        Compute output power for steering angle theta0.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        theta0 : float
            Steering angle
            
        Returns
        -------
        float
            Output power
        """
        y = self.beamform_output(X, theta0)
        return np.mean(np.abs(y)**2)
    
    def _find_k_peaks(self, angle_grid: np.ndarray, beam_pattern: np.ndarray, K: int) -> np.ndarray:
        """Find K largest peaks."""
        from scipy.signal import find_peaks as scipy_find_peaks
        
        peaks, _ = scipy_find_peaks(beam_pattern, height=0)
        
        if len(peaks) == 0:
            max_idx = np.argmax(beam_pattern)
            return np.array([angle_grid[max_idx]])
        
        heights = beam_pattern[peaks]
        sorted_indices = np.argsort(heights)[::-1]
        
        K_peaks = peaks[sorted_indices[:min(K, len(peaks))]]
        K_peaks = np.sort(K_peaks)
        
        return angle_grid[K_peaks]
    
    def angular_resolution(self, snr_db: float = 10) -> float:
        """
        Estimate angular resolution of Capon beamforming.
        
        Capon has better resolution than conventional beamforming,
        approximately improved by a factor of sqrt(SNR).
        
        Parameters
        ----------
        snr_db : float
            SNR in dB
            
        Returns
        -------
        float
            Angular resolution in radians
        """
        conventional_resolution = 2 / (self.array.M * self.array.d)
        snr_linear = 10**(snr_db/10)
        improvement_factor = np.sqrt(snr_linear)
        
        return conventional_resolution / improvement_factor
    
    def set_diagonal_loading(self, loading: float):
        """
        Set diagonal loading factor.
        
        Parameters
        ----------
        loading : float
            Diagonal loading factor
        """
        self.diagonal_loading = loading
        
    def __call__(self, X: np.ndarray, array: UniformLinearArray, **kwargs) -> np.ndarray:
        """Callable interface for Monte Carlo framework."""
        return self.estimate(X, **kwargs)