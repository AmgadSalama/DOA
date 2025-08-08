"""
Signal Model for DOA Estimation
===============================

This module implements the narrowband signal model used in DOA estimation.
The model assumes narrowband signals impinging on a uniform linear array.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from .ula import UniformLinearArray


class SignalModel:
    """
    Narrowband signal model for DOA estimation.
    
    The signal model is: x(t) = A(θ)s(t) + n(t)
    where:
    - x(t): M×1 received signal vector
    - A(θ): M×K array manifold matrix
    - s(t): K×1 source signal vector  
    - n(t): M×1 additive noise vector
    
    Parameters
    ----------
    array : UniformLinearArray
        Array geometry object
    """
    
    def __init__(self, array: UniformLinearArray):
        self.array = array
        self.M = array.M
        
    def generate_signals(self, 
                        doas: Union[List[float], np.ndarray],
                        N_snapshots: int,
                        snr_db: Union[float, List[float]] = 10,
                        signal_type: str = 'complex_sinusoid',
                        correlation: float = 0.0,
                        noise_type: str = 'white',
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic array data with known DOAs.
        
        Parameters
        ----------
        doas : array-like
            True DOAs in radians
        N_snapshots : int
            Number of temporal snapshots
        snr_db : float or list
            Signal-to-noise ratio in dB (per source if list)
        signal_type : str
            Type of source signals ('complex_sinusoid', 'random')
        correlation : float
            Source correlation coefficient (0-1)
        noise_type : str
            Noise type ('white', 'colored')
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        X : np.ndarray
            Received data matrix (M × N_snapshots)
        S : np.ndarray
            Source signals (K × N_snapshots)
        N : np.ndarray
            Noise matrix (M × N_snapshots)
        """
        if seed is not None:
            np.random.seed(seed)
            
        doas = np.atleast_1d(doas)
        K = len(doas)  # Number of sources
        
        # Array manifold matrix
        A = self.array.steering_vector(doas)  # M × K
        if A.ndim == 1:
            A = A.reshape(-1, 1)
            
        # Generate source signals
        S = self._generate_source_signals(K, N_snapshots, signal_type, correlation)
        
        # Apply SNR scaling
        snr_db = np.atleast_1d(snr_db)
        if len(snr_db) == 1:
            snr_db = np.repeat(snr_db, K)
        
        for k in range(K):
            snr_linear = 10**(snr_db[k]/10)
            S[k] = S[k] * np.sqrt(snr_linear)
            
        # Generate noise
        N = self._generate_noise(N_snapshots, noise_type)
        
        # Received signal
        X = A @ S + N
        
        return X, S, N
    
    def _generate_source_signals(self, K: int, N: int, signal_type: str, correlation: float) -> np.ndarray:
        """Generate source signals."""
        if signal_type == 'complex_sinusoid':
            # Complex sinusoidal signals with random phases
            phases = np.random.uniform(0, 2*np.pi, K)
            S = np.zeros((K, N), dtype=complex)
            for k in range(K):
                S[k] = np.exp(1j * (2*np.pi*0.1*np.arange(N) + phases[k]))
                
        elif signal_type == 'random':
            # Complex Gaussian signals
            S = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)
            
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
            
        # Add correlation if specified
        if correlation > 0 and K > 1:
            # Simple correlation model: s2 = ρ*s1 + √(1-ρ²)*independent
            rho = correlation
            for k in range(1, K):
                independent = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
                S[k] = rho * S[0] + np.sqrt(1 - rho**2) * independent
                
        return S
    
    def _generate_noise(self, N: int, noise_type: str) -> np.ndarray:
        """Generate noise."""
        if noise_type == 'white':
            # White Gaussian noise
            N_noise = (np.random.randn(self.M, N) + 1j * np.random.randn(self.M, N)) / np.sqrt(2)
            
        elif noise_type == 'colored':
            # Simple colored noise model
            white_noise = (np.random.randn(self.M, N) + 1j * np.random.randn(self.M, N)) / np.sqrt(2)
            # Apply simple coloring filter
            h = np.array([1, 0.5])
            N_noise = np.zeros_like(white_noise)
            for m in range(self.M):
                N_noise[m] = np.convolve(white_noise[m], h, mode='same')
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        return N_noise
    
    def sample_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sample covariance matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (M × N)
            
        Returns
        -------
        np.ndarray
            Sample covariance matrix (M × M)
        """
        return X @ X.conj().T / X.shape[1]
    
    def theoretical_covariance(self, 
                              doas: np.ndarray,
                              powers: np.ndarray,
                              noise_power: float = 1.0) -> np.ndarray:
        """
        Compute theoretical covariance matrix.
        
        Parameters
        ----------
        doas : np.ndarray
            True DOAs in radians
        powers : np.ndarray
            Source powers
        noise_power : float
            Noise power
            
        Returns
        -------
        np.ndarray
            Theoretical covariance matrix
        """
        A = self.array.steering_vector(doas)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
            
        P = np.diag(powers)  # Source power matrix
        R = A @ P @ A.conj().T + noise_power * np.eye(self.M)
        
        return R
    
    def snr_estimate(self, X: np.ndarray, doas: np.ndarray) -> np.ndarray:
        """
        Estimate SNR given data and DOAs.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (M × N)
        doas : np.ndarray
            Estimated DOAs
            
        Returns
        -------
        np.ndarray
            Estimated SNRs in dB
        """
        R = self.sample_covariance(X)
        A = self.array.steering_vector(doas)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
            
        # Estimate signal and noise powers
        # This is a simplified estimation
        eigenvals = np.linalg.eigvals(R)
        eigenvals = np.sort(np.real(eigenvals))[::-1]
        
        K = len(doas)
        signal_power = np.mean(eigenvals[:K])
        noise_power = np.mean(eigenvals[K:])
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return np.repeat(snr_db, K)  # Simplified: same SNR for all sources