"""
Unitary ESPRIT
==============

Implementation of Unitary ESPRIT for DOA estimation.
Unitary ESPRIT transforms the complex-valued problem into real-valued
computations, improving numerical stability and computational efficiency.
"""

import numpy as np
from typing import Optional
from ..array_processing import UniformLinearArray
from ..utils.math_utils import eigendecomposition


class UnitaryESPRIT:
    """
    Unitary ESPRIT DOA Estimator.
    
    Unitary ESPRIT exploits the centro-Hermitian structure of ULA
    covariance matrices to work with real-valued matrices, leading to
    improved computational efficiency and numerical stability.
    
    The method uses unitary transformations to convert complex
    eigenvalue problems to real ones.
    """
    
    def __init__(self, array: UniformLinearArray):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry (should have even number of elements for optimal performance)
        """
        self.array = array
        self.M = array.M
        
    def estimate(self,
                X: np.ndarray,
                K: int,
                use_fb_averaging: bool = True) -> np.ndarray:
        """
        Estimate DOAs using Unitary ESPRIT.
        
        Parameters
        ----------
        X : np.ndarray
            Received data matrix (M × N)
        K : int
            Number of sources
        use_fb_averaging : bool
            Use forward-backward averaging (recommended)
            
        Returns
        -------
        np.ndarray
            Estimated DOAs in radians
        """
        # Sample covariance matrix
        R = X @ X.conj().T / X.shape[1]
        
        # Forward-backward averaging to create centro-Hermitian matrix
        if use_fb_averaging:
            R = self._forward_backward_averaging(R)
        
        # Create real-valued covariance using unitary transformation
        R_real = self._unitary_transformation(R)
        
        # Eigendecomposition of real matrix
        eigenvals, eigenvecs = eigendecomposition(R_real, 'descending')
        
        if K >= self.array.M:
            raise ValueError(f"Number of sources ({K}) must be less than array size ({self.array.M})")
        
        # Real signal subspace
        U_s_real = eigenvecs[:, :K]
        
        # Transform back to complex domain for DOA estimation
        U_s = self._inverse_unitary_transformation(U_s_real)
        
        # Apply ESPRIT algorithm
        doas = self._esprit_algorithm(U_s, K)
        
        return np.sort(doas)
    
    def _forward_backward_averaging(self, R: np.ndarray) -> np.ndarray:
        """
        Apply forward-backward averaging.
        
        This creates a centro-Hermitian matrix:
        R_fb = (R + J R^* J) / 2
        
        where J is the exchange matrix.
        """
        J = np.eye(self.M)[::-1]  # Exchange matrix
        R_fb = (R + J @ R.conj() @ J) / 2
        return R_fb
    
    def _unitary_transformation(self, R: np.ndarray) -> np.ndarray:
        """
        Transform centro-Hermitian matrix to real form.
        
        Uses a unitary transformation matrix Q to create:
        R_real = Q^H R Q
        
        where Q transforms complex vectors to real ones.
        """
        # Create unitary transformation matrix Q
        Q = self._create_unitary_matrix()
        
        # Transform: R_real = Q^H R Q
        R_real = Q.conj().T @ R @ Q
        
        # Should be real (up to numerical precision)
        return R_real.real
    
    def _create_unitary_matrix(self) -> np.ndarray:
        """
        Create unitary transformation matrix for real-valued processing.
        
        The matrix Q maps complex M-dimensional vectors to real
        M-dimensional vectors while preserving the signal subspace.
        """
        M = self.M
        Q = np.zeros((M, M), dtype=complex)
        
        if M % 2 == 0:
            # Even array size
            mid = M // 2
            
            # First half: (e_k + j*e_{M-k-1})/sqrt(2) for k = 0, ..., mid-1
            for k in range(mid):
                Q[k, k] = 1/np.sqrt(2)
                Q[M-1-k, k] = 1j/np.sqrt(2)
                
            # Second half: (e_k - j*e_{M-k-1})/sqrt(2) for k = mid, ..., M-1
            for k in range(mid, M):
                idx = k - mid
                Q[idx, k] = 1/np.sqrt(2)
                Q[M-1-idx, k] = -1j/np.sqrt(2)
        else:
            # Odd array size - middle element is real
            mid = M // 2
            
            # Middle element
            Q[mid, mid] = 1
            
            # Other elements
            for k in range(mid):
                Q[k, k] = 1/np.sqrt(2)
                Q[M-1-k, k] = 1j/np.sqrt(2)
                
                Q[k, M-1-k] = 1/np.sqrt(2)
                Q[M-1-k, M-1-k] = -1j/np.sqrt(2)
        
        return Q
    
    def _inverse_unitary_transformation(self, U_s_real: np.ndarray) -> np.ndarray:
        """
        Transform real signal subspace back to complex domain.
        
        Parameters
        ----------
        U_s_real : np.ndarray
            Real signal subspace
            
        Returns
        -------
        np.ndarray
            Complex signal subspace
        """
        Q = self._create_unitary_matrix()
        U_s = Q @ U_s_real
        return U_s
    
    def _esprit_algorithm(self, U_s: np.ndarray, K: int) -> np.ndarray:
        """
        Apply ESPRIT algorithm to signal subspace.
        
        Parameters
        ----------
        U_s : np.ndarray
            Signal subspace matrix
        K : int
            Number of sources
            
        Returns
        -------
        np.ndarray
            DOA estimates
        """
        # Selection matrices for subarrays
        J1 = np.eye(self.M-1, self.M)         # First M-1 elements
        J2 = np.eye(self.M-1, self.M, k=1)   # Last M-1 elements
        
        # Subarray signal subspaces
        U_s1 = J1 @ U_s
        U_s2 = J2 @ U_s
        
        # Solve generalized eigenvalue problem using TLS
        eigenvals_phi = self._total_least_squares_solution(U_s1, U_s2)
        
        # Convert eigenvalues to DOAs
        phases = np.angle(eigenvals_phi)
        spatial_freqs = phases / (2 * np.pi * self.array.d)
        
        # Filter valid DOAs
        valid_mask = np.abs(spatial_freqs) <= 1
        valid_spatial_freqs = spatial_freqs[valid_mask]
        
        doas = np.arcsin(valid_spatial_freqs)
        
        return doas
    
    def _total_least_squares_solution(self, U_s1: np.ndarray, U_s2: np.ndarray) -> np.ndarray:
        """
        Solve using Total Least Squares method.
        
        Same as standard ESPRIT but with improved numerical stability
        due to real-valued computations in earlier stages.
        """
        K = U_s1.shape[1]
        
        # Augmented matrix [U_s1 | U_s2]
        C = np.hstack([U_s1, U_s2])
        
        # SVD
        U, s, Vh = np.linalg.svd(C, full_matrices=True)
        V = Vh.T
        
        # Extract blocks
        V12 = V[:K, K:]
        V22 = V[K:, K:]
        
        # TLS solution
        if np.linalg.cond(V22) < 1e12:
            Psi = -V12 @ np.linalg.inv(V22)
        else:
            # Fallback to pseudoinverse
            Psi = np.linalg.lstsq(U_s1, U_s2, rcond=None)[0]
        
        return np.linalg.eigvals(Psi)
    
    def computational_savings(self) -> dict:
        """
        Estimate computational savings compared to standard ESPRIT.
        
        Returns
        -------
        dict
            Computational complexity comparison
        """
        # Complex operations for standard ESPRIT
        complex_eigendecomp = 8 * self.M**3 / 3  # Complex eigendecomposition
        complex_matrix_ops = 8 * self.M**2        # Other complex operations
        
        # Real operations for Unitary ESPRIT
        real_eigendecomp = self.M**3 / 3          # Real eigendecomposition
        real_matrix_ops = self.M**2               # Real operations
        unitary_transform = 4 * self.M**2         # Transformation overhead
        
        total_complex = complex_eigendecomp + complex_matrix_ops
        total_real = real_eigendecomp + real_matrix_ops + unitary_transform
        
        savings_ratio = total_complex / total_real
        
        return {
            'complex_flops': total_complex,
            'real_flops': total_real,
            'computational_savings': savings_ratio,
            'memory_savings': 2.0  # Factor of 2 for real vs complex storage
        }
    
    def stability_comparison(self, X: np.ndarray, K: int) -> dict:
        """
        Compare numerical stability with standard ESPRIT.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        K : int
            Number of sources
            
        Returns
        -------
        dict
            Stability comparison results
        """
        # Unitary ESPRIT estimate
        doas_unitary = self.estimate(X, K)
        
        # Standard ESPRIT estimate
        from .esprit import ESPRIT
        esprit_std = ESPRIT(self.array)
        doas_standard = esprit_std.estimate(X, K)
        
        # Compare condition numbers of involved matrices
        R = X @ X.conj().T / X.shape[1]
        
        # Standard covariance condition number
        cond_standard = np.linalg.cond(R)
        
        # Real covariance condition number (after unitary transform)
        R_fb = self._forward_backward_averaging(R)
        R_real = self._unitary_transformation(R_fb)
        cond_unitary = np.linalg.cond(R_real)
        
        return {
            'doas_unitary': doas_unitary,
            'doas_standard': doas_standard,
            'condition_number_standard': cond_standard,
            'condition_number_unitary': cond_unitary,
            'condition_improvement': cond_standard / cond_unitary,
            'doa_differences': np.abs(np.sort(doas_unitary) - np.sort(doas_standard))
        }
    
    def __call__(self, X: np.ndarray, array: UniformLinearArray, K: int = 2, **kwargs) -> np.ndarray:
        """Callable interface for Monte Carlo framework."""
        return self.estimate(X, K, **kwargs)