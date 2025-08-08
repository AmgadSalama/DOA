"""
Mathematical Utilities
=====================

Common mathematical functions used in DOA estimation.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy.linalg import eigh, svd


def eigendecomposition(R: np.ndarray, sort_order: str = 'descending') -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigendecomposition of covariance matrix.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M × M)
    sort_order : str
        Sort order for eigenvalues ('descending' or 'ascending')
        
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues
    eigenvectors : np.ndarray
        Eigenvectors (columns)
    """
    eigenvals, eigenvecs = eigh(R)
    
    if sort_order == 'descending':
        idx = np.argsort(eigenvals)[::-1]
    else:
        idx = np.argsort(eigenvals)
        
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    return eigenvals, eigenvecs


def signal_noise_subspaces(R: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signal and noise subspaces from covariance matrix.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix
    K : int
        Number of sources
        
    Returns
    -------
    Us : np.ndarray
        Signal subspace (M × K)
    Un : np.ndarray
        Noise subspace (M × (M-K))
    """
    eigenvals, eigenvecs = eigendecomposition(R, 'descending')
    
    Us = eigenvecs[:, :K]
    Un = eigenvecs[:, K:]
    
    return Us, Un


def forward_backward_averaging(R: np.ndarray) -> np.ndarray:
    """
    Apply forward-backward averaging to covariance matrix.
    
    This technique improves performance for correlated sources by
    creating a centro-Hermitian matrix.
    
    Parameters
    ----------
    R : np.ndarray
        Original covariance matrix
        
    Returns
    -------
    np.ndarray
        Forward-backward averaged covariance matrix
    """
    M = R.shape[0]
    J = np.eye(M)[::-1]  # Exchange matrix
    
    # Forward-backward average: (R + J R^* J) / 2
    R_fb = (R + J @ R.conj() @ J) / 2
    
    return R_fb


def spatial_smoothing(X: np.ndarray, L: int) -> np.ndarray:
    """
    Apply spatial smoothing to data matrix.
    
    Spatial smoothing creates multiple subarrays to decorrelate
    coherent sources.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (M × N)
    L : int
        Subarray length
        
    Returns
    -------
    np.ndarray
        Spatially smoothed covariance matrix
    """
    M, N = X.shape
    
    if L > M:
        raise ValueError("Subarray length cannot exceed array size")
        
    num_subarrays = M - L + 1
    R_smooth = np.zeros((L, L), dtype=complex)
    
    for i in range(num_subarrays):
        X_sub = X[i:i+L, :]
        R_sub = X_sub @ X_sub.conj().T / N
        R_smooth += R_sub
        
    R_smooth /= num_subarrays
    
    return R_smooth


def toeplitz_reconstruction(r: np.ndarray) -> np.ndarray:
    """
    Reconstruct Toeplitz matrix from correlation sequence.
    
    Parameters
    ----------
    r : np.ndarray
        Correlation sequence
        
    Returns
    -------
    np.ndarray
        Toeplitz matrix
    """
    from scipy.linalg import toeplitz
    
    return toeplitz(r)


def hankel_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """
    Construct Hankel matrix from signal vector.
    
    Parameters
    ----------
    x : np.ndarray
        Signal vector
    L : int
        Number of rows
        
    Returns
    -------
    np.ndarray
        Hankel matrix
    """
    N = len(x)
    if L > N:
        raise ValueError("L cannot exceed signal length")
        
    K = N - L + 1
    H = np.zeros((L, K), dtype=x.dtype)
    
    for i in range(L):
        H[i, :] = x[i:i+K]
        
    return H


def polynomial_rooting(coeffs: np.ndarray) -> np.ndarray:
    """
    Find roots of polynomial and convert to DOAs.
    
    Parameters
    ----------
    coeffs : np.ndarray
        Polynomial coefficients
        
    Returns
    -------
    np.ndarray
        DOAs in radians
    """
    roots = np.roots(coeffs)
    
    # Filter roots on or inside unit circle
    unit_circle_roots = roots[np.abs(roots) <= 1.01]  # Small tolerance
    
    # Convert to angles
    angles = np.angle(unit_circle_roots)
    
    # Convert from spatial frequency to DOA
    # Assuming unit spacing: θ = arcsin(ω/(2π))
    # For roots: ω = angle of root
    doas = np.arcsin(angles / (2 * np.pi))
    
    # Keep only valid DOAs (real and within [-π/2, π/2])
    valid_mask = np.isreal(doas) & (np.abs(doas.real) <= np.pi/2)
    doas = doas[valid_mask].real
    
    return np.sort(doas)


def matrix_rank_estimation(S: np.ndarray, threshold: Optional[float] = None) -> int:
    """
    Estimate matrix rank using SVD.
    
    Parameters
    ----------
    S : np.ndarray
        Matrix
    threshold : float, optional
        Threshold for rank determination (default: automatic)
        
    Returns
    -------
    int
        Estimated rank
    """
    U, s, Vh = svd(S, full_matrices=False)
    
    if threshold is None:
        # Automatic threshold based on machine precision
        m, n = S.shape
        threshold = max(m, n) * np.finfo(float).eps * s[0]
        
    rank = np.sum(s > threshold)
    
    return rank


def source_number_estimation(eigenvals: np.ndarray, method: str = 'aic') -> int:
    """
    Estimate number of sources from eigenvalues.
    
    Parameters
    ----------
    eigenvals : np.ndarray
        Eigenvalues (sorted in descending order)
    method : str
        Estimation method ('aic', 'mdl', 'gde')
        
    Returns
    -------
    int
        Estimated number of sources
    """
    M = len(eigenvals)
    
    if method == 'aic':
        return _aic_criterion(eigenvals)
    elif method == 'mdl':
        return _mdl_criterion(eigenvals)
    elif method == 'gde':
        return _gde_criterion(eigenvals)
    else:
        raise ValueError(f"Unknown method: {method}")


def _aic_criterion(eigenvals: np.ndarray) -> int:
    """Akaike Information Criterion."""
    M = len(eigenvals)
    N = 100  # Assumed number of snapshots (should be parameter)
    
    aic_values = []
    for k in range(M):
        # Geometric mean of noise eigenvalues
        noise_eigenvals = eigenvals[k:]
        if len(noise_eigenvals) == 0:
            break
            
        geo_mean = np.exp(np.mean(np.log(noise_eigenvals + 1e-12)))
        arith_mean = np.mean(noise_eigenvals)
        
        if arith_mean == 0:
            break
            
        aic = -N * (M - k) * np.log(geo_mean / arith_mean) + k * (2*M - k)
        aic_values.append(aic)
        
    return np.argmin(aic_values) if aic_values else 0


def _mdl_criterion(eigenvals: np.ndarray) -> int:
    """Minimum Description Length."""
    M = len(eigenvals)
    N = 100  # Assumed number of snapshots
    
    mdl_values = []
    for k in range(M):
        noise_eigenvals = eigenvals[k:]
        if len(noise_eigenvals) == 0:
            break
            
        geo_mean = np.exp(np.mean(np.log(noise_eigenvals + 1e-12)))
        arith_mean = np.mean(noise_eigenvals)
        
        if arith_mean == 0:
            break
            
        mdl = -N * (M - k) * np.log(geo_mean / arith_mean) + 0.5 * k * (2*M - k) * np.log(N)
        mdl_values.append(mdl)
        
    return np.argmin(mdl_values) if mdl_values else 0


def _gde_criterion(eigenvals: np.ndarray) -> int:
    """Gerschgorin Disk Estimator."""
    # Simple threshold-based approach
    # More sophisticated GDE implementation would be needed for practical use
    threshold = 0.1 * eigenvals[0]  # 10% of largest eigenvalue
    return np.sum(eigenvals > threshold)


def angle_wrap(angles: np.ndarray) -> np.ndarray:
    """
    Wrap angles to [-π, π] range.
    
    Parameters
    ----------
    angles : np.ndarray
        Angles in radians
        
    Returns
    -------
    np.ndarray
        Wrapped angles
    """
    return np.mod(angles + np.pi, 2*np.pi) - np.pi


def db_to_linear(db_values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert dB to linear scale."""
    return 10**(np.array(db_values) / 10)


def linear_to_db(linear_values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert linear to dB scale."""
    return 10 * np.log10(np.array(linear_values))