"""
Conventional Beamforming (Delay-and-Sum)
========================================

Implementation of the conventional beamforming DOA estimation method.
Also known as the Delay-and-Sum beamformer.
"""

import numpy as np
from typing import Union, Optional
from ..array_processing import UniformLinearArray
from ..utils.peak_finding import find_peaks


class ConventionalBeamforming:
    """
    Conventional Beamforming DOA Estimator.
    
    The conventional beamformer forms a beam pattern by coherently summing
    the array outputs with appropriate phase shifts. The DOA estimates
    correspond to the peaks in the beam pattern.
    
    The beam pattern is given by:
    P_CBF(θ) = a^H(θ) R a(θ) / M^2
    
    where:
    - a(θ): steering vector for angle θ
    - R: sample covariance matrix
    - M: number of array elements
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
                angle_grid: Optional[np.ndarray] = None,
                K: Optional[int] = None,
                peak_finding: str = 'max') -> np.ndarray:
        """
        Estimate DOAs using conventional beamforming.
        
        Parameters
        ----------
        X : np.ndarray
            Received data matrix (M × N)
        angle_grid : np.ndarray, optional
            Grid of angles to search over (default: 181 points from -π/2 to π/2)
        K : int, optional
            Number of sources (if None, determined by peak finding)
        peak_finding : str
            Peak finding method ('max', 'threshold', 'adaptive')
            
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
            # Find K largest peaks
            doas = self._find_k_peaks(angle_grid, beam_pattern, K)
        else:
            # Use peak finding algorithm
            doas = find_peaks(angle_grid, beam_pattern, method=peak_finding)
            
        return np.array(doas)
    
    def beam_pattern(self, X: np.ndarray, angle_grid: np.ndarray) -> np.ndarray:
        """
        Compute the conventional beamforming beam pattern.
        
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
        
        # Array manifold
        A = self.array.array_manifold(angle_grid)  # M × N_grid
        
        # Compute beam pattern: P(θ) = a^H(θ) R a(θ) / M^2
        beam_pattern = np.zeros(len(angle_grid))
        
        for i, theta in enumerate(angle_grid):
            a = A[:, i:i+1]  # steering vector
            beam_pattern[i] = np.real(a.conj().T @ R @ a) / self.array.M**2
            
        return beam_pattern.flatten()
    
    def beam_pattern_theoretical(self, 
                               doas_true: np.ndarray,
                               powers: np.ndarray,
                               angle_grid: np.ndarray,
                               noise_power: float = 1.0) -> np.ndarray:
        """
        Compute theoretical beam pattern for known sources.
        
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
        
        # Array manifold for grid
        A_grid = self.array.array_manifold(angle_grid)
        
        # Beam pattern
        beam_pattern = np.zeros(len(angle_grid))
        for i in range(len(angle_grid)):
            a = A_grid[:, i:i+1]
            beam_pattern[i] = np.real(a.conj().T @ R @ a) / self.array.M**2
            
        return beam_pattern.flatten()
    
    def _find_k_peaks(self, angle_grid: np.ndarray, beam_pattern: np.ndarray, K: int) -> np.ndarray:
        """Find K largest peaks in beam pattern."""
        # Find local maxima
        from scipy.signal import find_peaks as scipy_find_peaks
        
        peaks, properties = scipy_find_peaks(beam_pattern, height=0)
        
        if len(peaks) == 0:
            # If no peaks found, return global maximum
            max_idx = np.argmax(beam_pattern)
            return np.array([angle_grid[max_idx]])
        
        # Sort by height and take K largest
        heights = beam_pattern[peaks]
        sorted_indices = np.argsort(heights)[::-1]
        
        K_peaks = peaks[sorted_indices[:min(K, len(peaks))]]
        K_peaks = np.sort(K_peaks)  # Sort by angle
        
        return angle_grid[K_peaks]
    
    def directivity_pattern(self, theta: np.ndarray, theta0: float) -> np.ndarray:
        """
        Compute the directivity pattern (array factor) for steering angle theta0.
        
        Parameters
        ----------
        theta : np.ndarray
            Angles to evaluate
        theta0 : float
            Steering angle
            
        Returns
        -------
        np.ndarray
            Directivity pattern
        """
        # Steering vector for theta0
        a0 = self.array.steering_vector(theta0)
        
        # Array manifold for theta
        A = self.array.array_manifold(theta)
        
        # Directivity: |a^H(θ0) a(θ)|^2 / M^2
        directivity = np.abs(a0.conj().T @ A)**2 / self.array.M**2
        
        return directivity.flatten()
    
    def angular_resolution(self) -> float:
        """
        Estimate the angular resolution of conventional beamforming.
        
        The 3dB beamwidth is approximately 2 / (M * d * cos(θ))
        For broadside (θ=0): 2 / (M * d) radians
        
        Returns
        -------
        float
            Angular resolution in radians (3dB beamwidth)
        """
        return 2 / (self.array.M * self.array.d)
    
    def __call__(self, X: np.ndarray, array: UniformLinearArray, **kwargs) -> np.ndarray:
        """
        Callable interface for compatibility with Monte Carlo framework.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        array : UniformLinearArray
            Array (should match self.array)
            
        Returns
        -------
        np.ndarray
            DOA estimates
        """
        return self.estimate(X, **kwargs)