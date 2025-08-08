"""
DOA Estimation Metrics
======================

Comprehensive performance metrics for evaluating DOA estimation methods.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import chi2
import warnings


class DOAMetrics:
    """
    Performance metrics for DOA estimation evaluation.
    
    This class provides various metrics commonly used in DOA literature
    to evaluate the performance of estimation algorithms.
    """
    
    @staticmethod
    def rmse(doas_true: np.ndarray, doas_estimated: np.ndarray, 
             match_sources: bool = True) -> float:
        """
        Root Mean Square Error between true and estimated DOAs.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs in radians
        doas_estimated : np.ndarray
            Estimated DOAs in radians
        match_sources : bool
            Whether to match sources by proximity
            
        Returns
        -------
        float
            RMSE in radians
        """
        if match_sources:
            errors = DOAMetrics._match_and_compute_errors(doas_true, doas_estimated)
        else:
            # Simple element-wise comparison (requires same ordering)
            min_len = min(len(doas_true), len(doas_estimated))
            errors = np.abs(doas_true[:min_len] - doas_estimated[:min_len])
        
        if len(errors) == 0:
            return np.inf
        
        return np.sqrt(np.mean(errors**2))
    
    @staticmethod
    def mae(doas_true: np.ndarray, doas_estimated: np.ndarray,
            match_sources: bool = True) -> float:
        """
        Mean Absolute Error.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_estimated : np.ndarray
            Estimated DOAs
        match_sources : bool
            Whether to match sources
            
        Returns
        -------
        float
            MAE in radians
        """
        if match_sources:
            errors = DOAMetrics._match_and_compute_errors(doas_true, doas_estimated)
        else:
            min_len = min(len(doas_true), len(doas_estimated))
            errors = np.abs(doas_true[:min_len] - doas_estimated[:min_len])
        
        if len(errors) == 0:
            return np.inf
        
        return np.mean(errors)
    
    @staticmethod
    def bias(doas_true: np.ndarray, doas_estimated: np.ndarray,
             match_sources: bool = True) -> float:
        """
        Bias of estimates.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_estimated : np.ndarray
            Estimated DOAs
        match_sources : bool
            Whether to match sources
            
        Returns
        -------
        float
            Bias in radians
        """
        if match_sources:
            # For bias, we need signed errors
            matched_pairs = DOAMetrics._match_sources(doas_true, doas_estimated)
            if len(matched_pairs) == 0:
                return np.inf
            
            errors = []
            for true_doa, est_doa in matched_pairs:
                if est_doa is not None:
                    errors.append(est_doa - true_doa)
        else:
            min_len = min(len(doas_true), len(doas_estimated))
            errors = doas_estimated[:min_len] - doas_true[:min_len]
        
        if len(errors) == 0:
            return np.inf
        
        return np.mean(errors)
    
    @staticmethod
    def variance(doas_true: np.ndarray, doas_estimated_list: List[np.ndarray],
                 match_sources: bool = True) -> np.ndarray:
        """
        Variance of estimates across multiple trials.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_estimated_list : list of np.ndarray
            List of estimates from multiple trials
        match_sources : bool
            Whether to match sources
            
        Returns
        -------
        np.ndarray
            Variance for each source
        """
        K = len(doas_true)
        all_estimates = np.full((len(doas_estimated_list), K), np.nan)
        
        for trial, doas_est in enumerate(doas_estimated_list):
            if match_sources:
                matched_pairs = DOAMetrics._match_sources(doas_true, doas_est)
                for i, (true_doa, est_doa) in enumerate(matched_pairs):
                    if est_doa is not None and i < K:
                        all_estimates[trial, i] = est_doa
            else:
                min_len = min(K, len(doas_est))
                all_estimates[trial, :min_len] = doas_est[:min_len]
        
        # Compute variance for each source
        variances = np.nanvar(all_estimates, axis=0)
        
        return variances
    
    @staticmethod
    def success_rate(doas_true: np.ndarray, doas_estimated_list: List[np.ndarray],
                     threshold_deg: float = 5.0) -> float:
        """
        Success rate based on estimation accuracy.
        
        A trial is considered successful if all sources are estimated
        within the specified threshold.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_estimated_list : list
            List of estimates
        threshold_deg : float
            Success threshold in degrees
            
        Returns
        -------
        float
            Success rate (0 to 1)
        """
        threshold_rad = np.deg2rad(threshold_deg)
        successful_trials = 0
        
        for doas_est in doas_estimated_list:
            if len(doas_est) == len(doas_true):
                errors = DOAMetrics._match_and_compute_errors(doas_true, doas_est)
                if len(errors) == len(doas_true) and np.all(errors < threshold_rad):
                    successful_trials += 1
        
        return successful_trials / len(doas_estimated_list)
    
    @staticmethod
    def resolution_probability(doas_true: np.ndarray, 
                              doas_estimated_list: List[np.ndarray],
                              min_separation_factor: float = 0.5) -> float:
        """
        Probability of resolving closely spaced sources.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs (should be closely spaced)
        doas_estimated_list : list
            List of estimates
        min_separation_factor : float
            Minimum separation factor (fraction of true separation)
            
        Returns
        -------
        float
            Resolution probability
        """
        if len(doas_true) < 2:
            return 1.0  # Single source is always "resolved"
        
        # Compute minimum required separation
        true_separations = np.diff(np.sort(doas_true))
        min_true_separation = np.min(true_separations)
        required_separation = min_separation_factor * min_true_separation
        
        resolved_trials = 0
        
        for doas_est in doas_estimated_list:
            if len(doas_est) >= len(doas_true):
                # Check if estimated sources are sufficiently separated
                est_separations = np.diff(np.sort(doas_est[:len(doas_true)]))
                if np.min(est_separations) >= required_separation:
                    resolved_trials += 1
        
        return resolved_trials / len(doas_estimated_list)
    
    @staticmethod
    def cramer_rao_lower_bound(array, doas: np.ndarray, snr_db: float, 
                              N_snapshots: int) -> np.ndarray:
        """
        Compute Cramer-Rao Lower Bound for DOA estimation.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        doas : np.ndarray
            DOAs in radians
        snr_db : float
            SNR in dB
        N_snapshots : int
            Number of snapshots
            
        Returns
        -------
        np.ndarray
            CRLB for each DOA (variance)
        """
        M = array.M
        K = len(doas)
        snr_linear = 10**(snr_db/10)
        
        # Array manifold and its derivative
        A = array.array_manifold(doas)  # M x K
        
        # Derivative of array manifold w.r.t. DOAs
        dA_dtheta = np.zeros_like(A)
        for k, theta in enumerate(doas):
            # d/dtheta [exp(-j*2*pi*d*m*sin(theta))] = -j*2*pi*d*m*cos(theta) * exp(...)
            m = np.arange(M).reshape(-1, 1)
            dA_dtheta[:, k] = -1j * 2 * np.pi * array.d * m.flatten() * np.cos(theta) * A[:, k]
        
        # Fisher Information Matrix
        # For uncorrelated sources with equal power
        P = np.eye(K)  # Source covariance (assuming unit power)
        sigma2 = 1 / snr_linear  # Noise variance
        
        # FIM for DOA parameters
        R_inv = np.linalg.inv(A @ P @ A.conj().T + sigma2 * np.eye(M))
        
        FIM = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(K):
                FIM[i, j] = 2 * N_snapshots / sigma2 * np.real(
                    dA_dtheta[:, i].conj().T @ 
                    (np.eye(M) - A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T) @ 
                    dA_dtheta[:, j]
                )
        
        # CRLB is diagonal of inverse FIM
        try:
            FIM_inv = np.linalg.inv(FIM.real)
            crlb = np.diag(FIM_inv)
        except np.linalg.LinAlgError:
            # Singular FIM - return infinite bound
            crlb = np.full(K, np.inf)
        
        return crlb
    
    @staticmethod
    def efficiency(variance_estimated: np.ndarray, crlb: np.ndarray) -> np.ndarray:
        """
        Compute estimation efficiency relative to CRLB.
        
        Parameters
        ----------
        variance_estimated : np.ndarray
            Estimated variance
        crlb : np.ndarray
            Cramer-Rao Lower Bound
            
        Returns
        -------
        np.ndarray
            Efficiency (0 to 1, where 1 is optimal)
        """
        return crlb / variance_estimated
    
    @staticmethod
    def _match_sources(doas_true: np.ndarray, doas_estimated: np.ndarray) -> List[Tuple]:
        """
        Match estimated sources to true sources by proximity.
        
        Returns list of (true_doa, estimated_doa) pairs.
        estimated_doa can be None if no match found.
        """
        if len(doas_estimated) == 0:
            return [(true_doa, None) for true_doa in doas_true]
        
        matched_pairs = []
        available_estimates = list(doas_estimated)
        
        for true_doa in doas_true:
            if len(available_estimates) > 0:
                # Find closest estimate
                distances = np.abs(np.array(available_estimates) - true_doa)
                closest_idx = np.argmin(distances)
                closest_estimate = available_estimates.pop(closest_idx)
                matched_pairs.append((true_doa, closest_estimate))
            else:
                matched_pairs.append((true_doa, None))
        
        return matched_pairs
    
    @staticmethod
    def _match_and_compute_errors(doas_true: np.ndarray, 
                                 doas_estimated: np.ndarray) -> np.ndarray:
        """
        Match sources and compute absolute errors.
        """
        matched_pairs = DOAMetrics._match_sources(doas_true, doas_estimated)
        
        errors = []
        for true_doa, est_doa in matched_pairs:
            if est_doa is not None:
                errors.append(np.abs(est_doa - true_doa))
        
        return np.array(errors)
    
    @staticmethod
    def detection_probability(doas_true: np.ndarray,
                             doas_estimated_list: List[np.ndarray],
                             threshold_deg: float = 10.0) -> Tuple[float, np.ndarray]:
        """
        Compute detection probability for each source.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_estimated_list : list
            List of estimates
        threshold_deg : float
            Detection threshold in degrees
            
        Returns
        -------
        overall_prob : float
            Overall detection probability
        individual_probs : np.ndarray
            Detection probability for each source
        """
        threshold_rad = np.deg2rad(threshold_deg)
        K = len(doas_true)
        
        detection_counts = np.zeros(K)
        
        for doas_est in doas_estimated_list:
            matched_pairs = DOAMetrics._match_sources(doas_true, doas_est)
            
            for i, (true_doa, est_doa) in enumerate(matched_pairs):
                if est_doa is not None:
                    error = np.abs(est_doa - true_doa)
                    if error < threshold_rad:
                        detection_counts[i] += 1
        
        individual_probs = detection_counts / len(doas_estimated_list)
        overall_prob = np.mean(individual_probs)
        
        return overall_prob, individual_probs
    
    @staticmethod
    def false_alarm_rate(K_true: int, doas_estimated_list: List[np.ndarray],
                        max_spurious: int = 5) -> float:
        """
        Compute false alarm rate (spurious peaks).
        
        Parameters
        ----------
        K_true : int
            True number of sources
        doas_estimated_list : list
            List of estimates
        max_spurious : int
            Maximum number of spurious detections to consider
            
        Returns
        -------
        float
            False alarm rate
        """
        false_alarms = 0
        total_trials = len(doas_estimated_list)
        
        for doas_est in doas_estimated_list:
            num_estimates = len(doas_est)
            if num_estimates > K_true:
                spurious = min(num_estimates - K_true, max_spurious)
                false_alarms += spurious
        
        return false_alarms / total_trials if total_trials > 0 else 0.0
    
    @staticmethod
    def comprehensive_evaluation(doas_true: np.ndarray,
                               doas_estimated_list: List[np.ndarray],
                               array=None, snr_db: float = None,
                               N_snapshots: int = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_estimated_list : list
            List of estimates from multiple trials
        array : UniformLinearArray, optional
            Array for CRLB computation
        snr_db : float, optional
            SNR for CRLB computation
        N_snapshots : int, optional
            Number of snapshots for CRLB computation
            
        Returns
        -------
        dict
            Comprehensive metrics
        """
        # Filter out failed estimates
        valid_estimates = [est for est in doas_estimated_list 
                          if est is not None and len(est) > 0]
        
        if len(valid_estimates) == 0:
            return {'success_rate': 0.0, 'rmse': np.inf, 'mae': np.inf, 'bias': np.inf}
        
        # Basic metrics
        rmse_values = []
        mae_values = []
        bias_values = []
        
        for doas_est in valid_estimates:
            rmse_val = DOAMetrics.rmse(doas_true, doas_est)
            mae_val = DOAMetrics.mae(doas_true, doas_est)
            bias_val = DOAMetrics.bias(doas_true, doas_est)
            
            if np.isfinite(rmse_val):
                rmse_values.append(rmse_val)
            if np.isfinite(mae_val):
                mae_values.append(mae_val)
            if np.isfinite(bias_val):
                bias_values.append(bias_val)
        
        metrics = {
            'success_rate': DOAMetrics.success_rate(doas_true, valid_estimates),
            'rmse': np.mean(rmse_values) if rmse_values else np.inf,
            'mae': np.mean(mae_values) if mae_values else np.inf,
            'bias': np.mean(bias_values) if bias_values else np.inf,
            'rmse_std': np.std(rmse_values) if len(rmse_values) > 1 else 0,
            'variance': DOAMetrics.variance(doas_true, valid_estimates),
            'resolution_prob': DOAMetrics.resolution_probability(doas_true, valid_estimates),
            'false_alarm_rate': DOAMetrics.false_alarm_rate(len(doas_true), valid_estimates)
        }
        
        # Add CRLB comparison if parameters provided
        if array is not None and snr_db is not None and N_snapshots is not None:
            try:
                crlb = DOAMetrics.cramer_rao_lower_bound(array, doas_true, snr_db, N_snapshots)
                efficiency = DOAMetrics.efficiency(metrics['variance'], crlb)
                metrics['crlb'] = crlb
                metrics['efficiency'] = efficiency
            except:
                pass  # CRLB computation failed
        
        return metrics