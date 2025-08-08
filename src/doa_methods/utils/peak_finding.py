"""
Peak Finding Utilities
=====================

Peak finding algorithms for DOA estimation methods.
"""

import numpy as np
from typing import List, Union, Optional
from scipy.signal import find_peaks as scipy_find_peaks


def find_peaks(angle_grid: np.ndarray, 
               spectrum: np.ndarray,
               method: str = 'max',
               threshold: Optional[float] = None,
               min_separation: Optional[float] = None,
               prominence: Optional[float] = None) -> np.ndarray:
    """
    Find peaks in a DOA spectrum.
    
    Parameters
    ----------
    angle_grid : np.ndarray
        Grid of angles in radians
    spectrum : np.ndarray
        Spectrum values (beam pattern, MUSIC spectrum, etc.)
    method : str
        Peak finding method:
        - 'max': Find global maximum
        - 'threshold': Find peaks above threshold
        - 'adaptive': Adaptive threshold based on spectrum statistics
        - 'local_maxima': All local maxima with prominence
    threshold : float, optional
        Threshold for peak detection (used with 'threshold' method)
    min_separation : float, optional
        Minimum angular separation between peaks in radians
    prominence : float, optional
        Minimum prominence for local maxima
        
    Returns
    -------
    np.ndarray
        Peak angles in radians
    """
    if len(spectrum) != len(angle_grid):
        raise ValueError("angle_grid and spectrum must have same length")
        
    if method == 'max':
        return _find_global_max(angle_grid, spectrum)
    elif method == 'threshold':
        return _find_threshold_peaks(angle_grid, spectrum, threshold, 
                                   min_separation, prominence)
    elif method == 'adaptive':
        return _find_adaptive_peaks(angle_grid, spectrum, min_separation, prominence)
    elif method == 'local_maxima':
        return _find_local_maxima(angle_grid, spectrum, prominence, min_separation)
    else:
        raise ValueError(f"Unknown peak finding method: {method}")


def _find_global_max(angle_grid: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Find global maximum."""
    max_idx = np.argmax(spectrum)
    return np.array([angle_grid[max_idx]])


def _find_threshold_peaks(angle_grid: np.ndarray, 
                         spectrum: np.ndarray,
                         threshold: Optional[float] = None,
                         min_separation: Optional[float] = None,
                         prominence: Optional[float] = None) -> np.ndarray:
    """Find peaks above threshold."""
    if threshold is None:
        threshold = 0.5 * np.max(spectrum)
        
    # Convert min_separation to indices if provided
    distance = None
    if min_separation is not None:
        angle_step = np.mean(np.diff(angle_grid))
        distance = int(min_separation / angle_step)
        
    peaks, _ = scipy_find_peaks(spectrum, 
                               height=threshold,
                               distance=distance,
                               prominence=prominence)
    
    if len(peaks) == 0:
        return _find_global_max(angle_grid, spectrum)
        
    return angle_grid[peaks]


def _find_adaptive_peaks(angle_grid: np.ndarray,
                        spectrum: np.ndarray,
                        min_separation: Optional[float] = None,
                        prominence: Optional[float] = None) -> np.ndarray:
    """Find peaks using adaptive threshold."""
    # Adaptive threshold based on spectrum statistics
    spectrum_mean = np.mean(spectrum)
    spectrum_std = np.std(spectrum)
    
    # Set threshold to mean + 2*std (captures ~95% of noise)
    threshold = spectrum_mean + 2 * spectrum_std
    
    # If threshold is too high, reduce it
    if threshold > 0.8 * np.max(spectrum):
        threshold = 0.5 * np.max(spectrum)
        
    return _find_threshold_peaks(angle_grid, spectrum, threshold, 
                               min_separation, prominence)


def _find_local_maxima(angle_grid: np.ndarray,
                      spectrum: np.ndarray,
                      prominence: Optional[float] = None,
                      min_separation: Optional[float] = None) -> np.ndarray:
    """Find all significant local maxima."""
    if prominence is None:
        # Default prominence: 10% of spectrum range
        prominence = 0.1 * (np.max(spectrum) - np.min(spectrum))
        
    distance = None
    if min_separation is not None:
        angle_step = np.mean(np.diff(angle_grid))
        distance = int(min_separation / angle_step)
        
    peaks, properties = scipy_find_peaks(spectrum,
                                        prominence=prominence,
                                        distance=distance)
    
    if len(peaks) == 0:
        return _find_global_max(angle_grid, spectrum)
        
    # Sort peaks by prominence (highest first)
    if 'prominences' in properties:
        sort_idx = np.argsort(properties['prominences'])[::-1]
        peaks = peaks[sort_idx]
        
    return angle_grid[peaks]


def peak_width_analysis(angle_grid: np.ndarray,
                       spectrum: np.ndarray,
                       peak_angles: np.ndarray,
                       width_level: float = 0.5) -> np.ndarray:
    """
    Analyze peak widths.
    
    Parameters
    ----------
    angle_grid : np.ndarray
        Angle grid
    spectrum : np.ndarray
        Spectrum values
    peak_angles : np.ndarray
        Peak angles
    width_level : float
        Level for width measurement (0.5 for FWHM)
        
    Returns
    -------
    np.ndarray
        Peak widths in radians
    """
    widths = []
    
    for peak_angle in peak_angles:
        # Find peak index
        peak_idx = np.argmin(np.abs(angle_grid - peak_angle))
        peak_value = spectrum[peak_idx]
        
        # Find width at specified level
        target_level = width_level * peak_value
        
        # Search left side
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > target_level:
            left_idx -= 1
            
        # Search right side  
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > target_level:
            right_idx += 1
            
        # Interpolate for more accurate width
        if left_idx < peak_idx:
            left_angle = _interpolate_crossing(angle_grid[left_idx:left_idx+2],
                                             spectrum[left_idx:left_idx+2],
                                             target_level)
        else:
            left_angle = angle_grid[left_idx]
            
        if right_idx > peak_idx:
            right_angle = _interpolate_crossing(angle_grid[right_idx-1:right_idx+1],
                                              spectrum[right_idx-1:right_idx+1],
                                              target_level)
        else:
            right_angle = angle_grid[right_idx]
            
        width = right_angle - left_angle
        widths.append(width)
        
    return np.array(widths)


def _interpolate_crossing(angles: np.ndarray, values: np.ndarray, target: float) -> float:
    """Interpolate to find where spectrum crosses target level."""
    if len(angles) != 2 or len(values) != 2:
        return angles[0]
        
    if values[1] == values[0]:
        return angles[0]
        
    # Linear interpolation
    t = (target - values[0]) / (values[1] - values[0])
    return angles[0] + t * (angles[1] - angles[0])


def estimate_noise_floor(spectrum: np.ndarray, percentile: float = 25) -> float:
    """
    Estimate noise floor of spectrum.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum values
    percentile : float
        Percentile for noise floor estimation
        
    Returns
    -------
    float
        Estimated noise floor
    """
    return np.percentile(spectrum, percentile)


def signal_to_noise_ratio(spectrum: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    """
    Estimate signal-to-noise ratio for peaks.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum values
    peak_indices : np.ndarray
        Peak indices
        
    Returns
    -------
    np.ndarray
        SNR estimates in dB
    """
    noise_floor = estimate_noise_floor(spectrum)
    peak_values = spectrum[peak_indices]
    
    snr_linear = peak_values / noise_floor
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db