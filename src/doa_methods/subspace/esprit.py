"""
ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
=============================================================================

Implementation of the ESPRIT DOA estimation algorithm.
ESPRIT exploits the rotational invariance property of ULAs to
estimate DOAs without spectral peak searching.
"""

import numpy as np
from typing import Optional
from ..array_processing import UniformLinearArray
from ..utils.math_utils import eigendecomposition


class ESPRIT:
    """
    ESPRIT DOA Estimator.
    
    ESPRIT divides the ULA into two overlapping subarrays and exploits
    the rotational invariance between them to estimate DOAs. This is
    done by solving a generalized eigenvalue problem.
    
    For a ULA, the relationship between subarrays is:
    X_2 = X_1 * Φ
    
    where Φ = diag(exp(jω_1), exp(jω_2), ..., exp(jω_K))
    and ω_i = 2π*d*sin(θ_i) are the spatial frequencies.
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
                use_fb_averaging: bool = False,
                ls_method: str = 'total') -> np.ndarray:
        """
        Estimate DOAs using ESPRIT.
        
        Parameters
        ----------
        X : np.ndarray
            Received data matrix (M × N)
        K : int
            Number of sources
        use_fb_averaging : bool
            Use forward-backward averaging
        ls_method : str
            Least squares method ('ls', 'total')
            
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
        
        # Eigendecomposition to get signal subspace
        eigenvals, eigenvecs = eigendecomposition(R, 'descending')
        
        if K >= self.array.M:
            raise ValueError(f"Number of sources ({K}) must be less than array size ({self.array.M})")
        
        # Signal subspace
        U_s = eigenvecs[:, :K]
        
        # Create subarray selection matrices
        J1, J2 = self._create_selection_matrices()
        
        # Extract subarray signal subspaces
        U_s1 = J1 @ U_s  # First subarray
        U_s2 = J2 @ U_s  # Second subarray
        
        # Solve for eigenvalues of Φ
        if ls_method == 'ls':
            eigenvals_phi = self._least_squares_solution(U_s1, U_s2)
        elif ls_method == 'total':
            eigenvals_phi = self._total_least_squares_solution(U_s1, U_s2)
        else:
            raise ValueError(f"Unknown LS method: {ls_method}")
        
        # Convert eigenvalues to DOAs
        doas = self._eigenvalues_to_doas(eigenvals_phi)
        
        return np.sort(doas)
    
    def _create_selection_matrices(self) -> tuple:
        """
        Create selection matrices for overlapping subarrays.
        
        For M-element array, creates (M-1)×M matrices that select
        the first M-1 and last M-1 elements respectively.
        
        Returns
        -------
        J1, J2 : np.ndarray
            Selection matrices for first and second subarrays
        """
        M = self.array.M
        
        # First subarray: elements 0 to M-2
        J1 = np.eye(M-1, M)
        
        # Second subarray: elements 1 to M-1
        J2 = np.eye(M-1, M, k=1)
        
        return J1, J2
    
    def _least_squares_solution(self, U_s1: np.ndarray, U_s2: np.ndarray) -> np.ndarray:
        """
        Standard least squares solution for ESPRIT.
        
        Solves: U_s2 = U_s1 * Ψ
        where Ψ is related to the rotation matrix Φ.
        
        Parameters
        ----------
        U_s1, U_s2 : np.ndarray
            Subarray signal subspaces
            
        Returns
        -------
        np.ndarray
            Eigenvalues of Φ matrix
        """
        # Solve: Ψ = (U_s1^H * U_s1)^(-1) * U_s1^H * U_s2
        try:
            U1_pinv = np.linalg.pinv(U_s1)
            Psi = U1_pinv @ U_s2
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            Psi = np.linalg.lstsq(U_s1, U_s2, rcond=None)[0]
        
        # Eigenvalues of Ψ are the eigenvalues of Φ
        eigenvals_phi = np.linalg.eigvals(Psi)
        
        return eigenvals_phi
    
    def _total_least_squares_solution(self, U_s1: np.ndarray, U_s2: np.ndarray) -> np.ndarray:
        """
        Total least squares (TLS) solution for ESPRIT.
        
        More robust than standard LS, especially in low SNR conditions.
        
        Parameters
        ----------
        U_s1, U_s2 : np.ndarray
            Subarray signal subspaces
            
        Returns
        -------
        np.ndarray
            Eigenvalues of Φ matrix
        """
        K = U_s1.shape[1]
        
        # Form augmented matrix [U_s1 | U_s2]
        C = np.hstack([U_s1, U_s2])
        
        # SVD of augmented matrix
        U, s, Vh = np.linalg.svd(C, full_matrices=True)
        
        # TLS solution from right singular vectors
        V = Vh.T
        V12 = V[:K, K:]      # Upper right block
        V22 = V[K:, K:]      # Lower right block
        
        # Check if V22 is invertible
        if np.linalg.cond(V22) > 1e12:
            # Fallback to LS solution
            return self._least_squares_solution(U_s1, U_s2)
        
        # TLS solution: Ψ = -V12 * V22^(-1)
        Psi = -V12 @ np.linalg.inv(V22)
        
        # Eigenvalues of Ψ
        eigenvals_phi = np.linalg.eigvals(Psi)
        
        return eigenvals_phi
    
    def _eigenvalues_to_doas(self, eigenvals_phi: np.ndarray) -> np.ndarray:
        """
        Convert eigenvalues of Φ to DOAs.
        
        The eigenvalues are of the form exp(jω_i) where
        ω_i = 2π*d*sin(θ_i) is the spatial frequency.
        
        Parameters
        ----------
        eigenvals_phi : np.ndarray
            Eigenvalues of Φ matrix
            
        Returns
        -------
        np.ndarray
            DOAs in radians
        """
        # Extract phases (spatial frequencies)
        phases = np.angle(eigenvals_phi)
        
        # Convert to DOAs: ω = 2π*d*sin(θ) -> θ = arcsin(ω/(2π*d))
        spatial_freqs = phases / (2 * np.pi * self.array.d)
        
        # Filter valid DOAs (|sin(θ)| ≤ 1)
        valid_mask = np.abs(spatial_freqs) <= 1
        valid_spatial_freqs = spatial_freqs[valid_mask]
        
        doas = np.arcsin(valid_spatial_freqs)
        
        return doas
    
    def estimate_with_pairing(self, 
                             X: np.ndarray, 
                             K: int, 
                             return_eigenvals: bool = False) -> dict:
        """
        ESPRIT estimation with eigenvalue-eigenvector pairing information.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        K : int
            Number of sources
        return_eigenvals : bool
            Return eigenvalue information
            
        Returns
        -------
        dict
            Results including DOAs and pairing information
        """
        # Get signal subspace
        R = X @ X.conj().T / X.shape[1]
        eigenvals, eigenvecs = eigendecomposition(R, 'descending')
        U_s = eigenvecs[:, :K]
        
        # Subarray selection
        J1, J2 = self._create_selection_matrices()
        U_s1, U_s2 = J1 @ U_s, J2 @ U_s
        
        # TLS solution
        eigenvals_phi = self._total_least_squares_solution(U_s1, U_s2)
        
        # Convert to DOAs
        doas = self._eigenvalues_to_doas(eigenvals_phi)
        
        result = {
            'doas': np.sort(doas),
            'phi_eigenvalues': eigenvals_phi,
            'spatial_frequencies': np.angle(eigenvals_phi) / (2 * np.pi * self.array.d)
        }
        
        if return_eigenvals:
            result['signal_eigenvalues'] = eigenvals[:K]
            result['noise_eigenvalues'] = eigenvals[K:]
            
        return result
    
    def resolution_analysis(self, separation_angle: float, snr_db: float, N_snapshots: int) -> dict:
        """
        Analyze resolution capability for given conditions.
        
        Parameters
        ----------
        separation_angle : float
            Angular separation in radians
        snr_db : float
            SNR in dB
        N_snapshots : int
            Number of snapshots
            
        Returns
        -------
        dict
            Resolution analysis results
        """
        # Theoretical resolution limit (approximate)
        snr_linear = 10**(snr_db/10)
        
        # ESPRIT resolution (simplified model)
        resolution_limit = np.sqrt(6 / (snr_linear * N_snapshots * self.array.M**2))
        
        can_resolve = separation_angle > resolution_limit
        
        return {
            'separation_angle_deg': np.rad2deg(separation_angle),
            'resolution_limit_deg': np.rad2deg(resolution_limit),
            'can_resolve': can_resolve,
            'resolution_margin': separation_angle / resolution_limit if resolution_limit > 0 else np.inf
        }
    
    def __call__(self, X: np.ndarray, array: UniformLinearArray, K: int = 2, **kwargs) -> np.ndarray:
        """Callable interface for Monte Carlo framework."""
        return self.estimate(X, K, **kwargs)