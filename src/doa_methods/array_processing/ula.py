"""
Uniform Linear Array (ULA) Implementation
=========================================

This module implements the ULA geometry and array manifold for DOA estimation.
"""

import numpy as np
from typing import Union, List


class UniformLinearArray:
    """
    Uniform Linear Array (ULA) class for DOA estimation.
    
    A ULA consists of M sensors uniformly spaced along a straight line.
    The array manifold relates the spatial frequencies to the directions of arrival.
    
    Parameters
    ----------
    M : int
        Number of array elements
    d : float, optional
        Inter-element spacing in wavelengths (default: 0.5)
    fc : float, optional
        Carrier frequency in Hz (default: 1e9)
    c : float, optional
        Speed of light in m/s (default: 3e8)
    """
    
    def __init__(self, M: int, d: float = 0.5, fc: float = 1e9, c: float = 3e8):
        self.M = M
        self.d = d  # Inter-element spacing in wavelengths
        self.fc = fc  # Carrier frequency
        self.c = c  # Speed of light
        self.wavelength = c / fc
        self.d_meters = d * self.wavelength  # Inter-element spacing in meters
        
        # Element positions (in wavelengths)
        self.element_positions = np.arange(M) * d
        
    def steering_vector(self, theta: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the array steering vector for given DOA(s).
        
        The steering vector for a ULA is:
        a(θ) = [1, e^(-j2π(d/λ)sin(θ)), ..., e^(-j2π(M-1)(d/λ)sin(θ))]^T
        
        Parameters
        ----------
        theta : float or array-like
            Direction of arrival in radians (-π/2 to π/2)
            
        Returns
        -------
        np.ndarray
            Steering vector(s) of shape (M,) for single DOA or (M, N) for N DOAs
        """
        theta = np.atleast_1d(theta)
        
        # Spatial frequency
        # ω = 2π(d/λ)sin(θ) = 2πd sin(θ) when d is in wavelengths
        omega = 2 * np.pi * self.d * np.sin(theta)
        
        # Element positions
        m = np.arange(self.M).reshape(-1, 1)
        
        # Steering vector
        a = np.exp(-1j * m * omega)
        
        return a.squeeze() if a.shape[1] == 1 else a
    
    def array_manifold(self, theta_grid: np.ndarray) -> np.ndarray:
        """
        Compute the array manifold matrix for a grid of angles.
        
        Parameters
        ----------
        theta_grid : np.ndarray
            Grid of angles in radians
            
        Returns
        -------
        np.ndarray
            Array manifold matrix of shape (M, len(theta_grid))
        """
        return self.steering_vector(theta_grid)
    
    def spatial_frequency(self, theta: Union[float, np.ndarray]) -> np.ndarray:
        """
        Convert DOA to spatial frequency.
        
        Parameters
        ----------
        theta : float or array-like
            Direction of arrival in radians
            
        Returns
        -------
        np.ndarray
            Spatial frequency ω = 2π(d/λ)sin(θ)
        """
        return 2 * np.pi * self.d * np.sin(theta)
    
    def theta_from_spatial_freq(self, omega: Union[float, np.ndarray]) -> np.ndarray:
        """
        Convert spatial frequency to DOA.
        
        Parameters
        ----------
        omega : float or array-like
            Spatial frequency
            
        Returns
        -------
        np.ndarray
            Direction of arrival in radians
        """
        return np.arcsin(omega / (2 * np.pi * self.d))
    
    def angle_grid(self, N_grid: int = 181) -> np.ndarray:
        """
        Generate a uniform grid of angles for DOA estimation.
        
        Parameters
        ----------
        N_grid : int, optional
            Number of grid points (default: 181)
            
        Returns
        -------
        np.ndarray
            Angle grid from -π/2 to π/2
        """
        return np.linspace(-np.pi/2, np.pi/2, N_grid)
    
    def get_info(self) -> dict:
        """
        Get array information.
        
        Returns
        -------
        dict
            Dictionary containing array parameters
        """
        return {
            'M': self.M,
            'd_wavelengths': self.d,
            'd_meters': self.d_meters,
            'fc': self.fc,
            'wavelength': self.wavelength,
            'aperture': (self.M - 1) * self.d_meters
        }