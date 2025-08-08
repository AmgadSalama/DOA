"""
Root-MUSIC
==========

Implementation of the Root-MUSIC DOA estimation algorithm.
Root-MUSIC avoids the spectral peak search by finding polynomial roots,
making it computationally more efficient and potentially more accurate
than conventional MUSIC.
"""

import numpy as np
from typing import Optional
from ..array_processing import UniformLinearArray
from ..utils.math_utils import eigendecomposition, polynomial_rooting


class RootMUSIC:
    """
    Root-MUSIC DOA Estimator.
    
    Root-MUSIC formulates the MUSIC spectrum as a polynomial and finds
    its roots. The DOAs correspond to roots closest to the unit circle.
    
    For a ULA with unit spacing, the polynomial is:
    P(z) = a^T(z) U_n U_n^H a*(z)
    
    where a(z) is the steering vector as a function of z = exp(jω).
    """
    
    def __init__(self, array: UniformLinearArray):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry (must be ULA)
        """
        self.array = array
        if self.array.d != 0.5:
            print("Warning: Root-MUSIC is optimized for half-wavelength spacing")
        
    def estimate(self,
                X: np.ndarray,
                K: int,
                use_fb_averaging: bool = False) -> np.ndarray:
        """
        Estimate DOAs using Root-MUSIC.
        
        Parameters
        ----------
        X : np.ndarray
            Received data matrix (M × N)
        K : int
            Number of sources
        use_fb_averaging : bool
            Use forward-backward averaging
            
        Returns
        -------
        np.ndarray
            Estimated DOAs in radians
        """
        # Sample covariance matrix
        R = X @ X.conj().T / X.shape[1]
        
        # Apply forward-backward averaging if requested
        if use_fb_averaging:
            from ..utils.math_utils import forward_backward_averaging
            R = forward_backward_averaging(R)
        
        # Eigendecomposition
        eigenvals, eigenvecs = eigendecomposition(R, 'descending')
        
        if K >= self.array.M:
            raise ValueError(f"Number of sources ({K}) must be less than array size ({self.array.M})")
        
        # Noise subspace
        U_n = eigenvecs[:, K:]
        
        # Compute polynomial coefficients
        coeffs = self._compute_polynomial_coefficients(U_n)
        
        # Find polynomial roots
        roots = np.roots(coeffs)
        
        # Select K roots closest to unit circle (inside)
        doas = self._select_doa_roots(roots, K)
        
        return np.sort(doas)
    
    def _compute_polynomial_coefficients(self, U_n: np.ndarray) -> np.ndarray:
        """
        Compute polynomial coefficients from noise subspace.
        
        The polynomial is formed by:
        P(z) = c_{2M-2} z^{2M-2} + ... + c_1 z + c_0
        
        where coefficients come from the quadratic form:
        P(z) = [1, z, z^2, ..., z^{M-1}] C [1, z^{-1}, z^{-2}, ..., z^{-(M-1)}]^*
        
        and C = U_n U_n^H.
        
        Parameters
        ----------
        U_n : np.ndarray
            Noise subspace matrix (M × (M-K))
            
        Returns
        -------
        np.ndarray
            Polynomial coefficients
        """
        M = self.array.M
        
        # Form matrix C = U_n U_n^H
        C = U_n @ U_n.conj().T
        
        # Compute polynomial coefficients
        # For polynomial of degree 2(M-1), we need coefficients c_{-(M-1)} to c_{M-1}
        coeffs = np.zeros(2*M - 1, dtype=complex)
        
        for k in range(-(M-1), M):
            if k == 0:
                # Diagonal elements
                coeffs[k + M - 1] = np.sum(np.diag(C))
            elif k > 0:
                # Super-diagonal elements
                if k < M:
                    coeffs[k + M - 1] = np.sum(np.diag(C, k))
            else:
                # Sub-diagonal elements  
                if -k < M:
                    coeffs[k + M - 1] = np.sum(np.diag(C, k))
        
        # The polynomial coefficients for numpy.roots (highest degree first)
        # Convert from c_{-(M-1)} ... c_0 ... c_{M-1} to standard form
        poly_coeffs = np.flip(coeffs)
        
        return poly_coeffs.real if np.allclose(poly_coeffs.imag, 0) else poly_coeffs
    
    def _select_doa_roots(self, roots: np.ndarray, K: int) -> np.ndarray:
        """
        Select K roots closest to unit circle that correspond to DOAs.
        
        Parameters
        ----------
        roots : np.ndarray
            Polynomial roots
        K : int
            Number of sources
            
        Returns
        -------
        np.ndarray
            DOA estimates in radians
        """
        # Filter roots inside or on unit circle
        unit_roots = roots[np.abs(roots) <= 1.0 + 1e-6]
        
        if len(unit_roots) == 0:
            # If no roots inside unit circle, take closest ones
            distances = np.abs(np.abs(roots) - 1)
            closest_indices = np.argsort(distances)[:K]
            selected_roots = roots[closest_indices]
        else:
            # Select K roots closest to unit circle
            distances = np.abs(np.abs(unit_roots) - 1)
            if len(unit_roots) <= K:
                selected_roots = unit_roots
            else:
                closest_indices = np.argsort(distances)[:K]
                selected_roots = unit_roots[closest_indices]
        
        # Convert roots to DOAs
        doas = []
        for root in selected_roots:
            # For ULA: z = exp(j*2π*d*sin(θ))
            # With d = 0.5 (half wavelength): z = exp(j*π*sin(θ))
            # Therefore: sin(θ) = angle(z)/π, θ = arcsin(angle(z)/π)
            
            angle = np.angle(root)
            sin_theta = angle / (np.pi * self.array.d * 2)  # Account for spacing
            
            # Ensure |sin(θ)| ≤ 1 for valid DOA
            if np.abs(sin_theta) <= 1:
                theta = np.arcsin(sin_theta)
                doas.append(theta)
        
        return np.array(doas)
    
    def polynomial_spectrum(self, angle_grid: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial spectrum over angle grid.
        
        This can be useful for visualization and comparison with MUSIC spectrum.
        
        Parameters
        ----------
        angle_grid : np.ndarray
            Grid of angles
        coeffs : np.ndarray
            Polynomial coefficients
            
        Returns
        -------
        np.ndarray
            Polynomial values (spectrum)
        """
        spectrum = np.zeros(len(angle_grid))
        
        for i, theta in enumerate(angle_grid):
            # Convert angle to z = exp(j*2π*d*sin(θ))
            z = np.exp(1j * 2 * np.pi * self.array.d * np.sin(theta))
            
            # Evaluate polynomial P(z)
            poly_val = np.polyval(coeffs, z)
            
            # Root-MUSIC spectrum is 1/|P(z)|
            spectrum[i] = 1.0 / (np.abs(poly_val) + 1e-12)
            
        return spectrum
    
    def compare_with_music(self, X: np.ndarray, K: int) -> dict:
        """
        Compare Root-MUSIC with conventional MUSIC.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        K : int
            Number of sources
            
        Returns
        -------
        dict
            Comparison results
        """
        import time
        
        # Root-MUSIC
        start_time = time.time()
        root_doas = self.estimate(X, K)
        root_time = time.time() - start_time
        
        # Conventional MUSIC
        from .music import MUSIC
        music = MUSIC(self.array)
        start_time = time.time()
        music_doas = music.estimate(X, K)
        music_time = time.time() - start_time
        
        return {
            'root_music_doas': root_doas,
            'music_doas': music_doas,
            'root_music_time': root_time,
            'music_time': music_time,
            'doa_differences': np.abs(np.sort(root_doas) - np.sort(music_doas)),
            'speedup_factor': music_time / root_time
        }
    
    def stability_analysis(self, X: np.ndarray, K: int, perturbation_std: float = 0.01) -> dict:
        """
        Analyze stability of Root-MUSIC to small perturbations.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        K : int
            Number of sources
        perturbation_std : float
            Standard deviation of added noise
            
        Returns
        -------
        dict
            Stability analysis results
        """
        reference_doas = self.estimate(X, K)
        
        # Add small perturbation and re-estimate
        noise = (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape)) * perturbation_std
        X_pert = X + noise
        perturbed_doas = self.estimate(X_pert, K)
        
        # Compute sensitivity
        if len(reference_doas) == len(perturbed_doas):
            sensitivity = np.abs(np.sort(perturbed_doas) - np.sort(reference_doas))
        else:
            sensitivity = np.array([np.inf])
        
        return {
            'reference_doas': reference_doas,
            'perturbed_doas': perturbed_doas,
            'sensitivity': sensitivity,
            'max_sensitivity': np.max(sensitivity),
            'mean_sensitivity': np.mean(sensitivity)
        }
    
    def __call__(self, X: np.ndarray, array: UniformLinearArray, K: int = 2, **kwargs) -> np.ndarray:
        """Callable interface for Monte Carlo framework."""
        return self.estimate(X, K, **kwargs)