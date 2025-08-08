"""
Monte Carlo Simulation
=====================

Tools for running Monte Carlo simulations to evaluate DOA estimation methods.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
import time
from concurrent.futures import ProcessPoolExecutor
from ..array_processing import UniformLinearArray, SignalModel


class MonteCarlo:
    """
    Monte Carlo simulation framework for DOA estimation methods.
    
    This class provides tools to run repeated experiments with different
    noise realizations to evaluate the statistical performance of DOA methods.
    """
    
    def __init__(self, array: UniformLinearArray):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry for simulations
        """
        self.array = array
        self.signal_model = SignalModel(array)
        
    def run_single_trial(self, 
                        estimator: Callable,
                        doas_true: np.ndarray,
                        N_snapshots: int,
                        snr_db: float,
                        seed: Optional[int] = None,
                        **estimator_kwargs) -> Dict[str, Any]:
        """
        Run a single Monte Carlo trial.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function that takes (X, array) and returns DOAs
        doas_true : np.ndarray
            True DOAs in radians
        N_snapshots : int
            Number of snapshots
        snr_db : float
            SNR in dB
        seed : int, optional
            Random seed for this trial
        **estimator_kwargs
            Additional arguments for the estimator
            
        Returns
        -------
        dict
            Trial results
        """
        # Generate data
        X, S, N = self.signal_model.generate_signals(
            doas=doas_true,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            seed=seed
        )
        
        # Run estimator and measure time
        start_time = time.time()
        try:
            doas_est = estimator(X, self.array, **estimator_kwargs)
            success = True
            error_msg = None
        except Exception as e:
            doas_est = None
            success = False
            error_msg = str(e)
        
        compute_time = time.time() - start_time
        
        # Compute errors if successful
        if success and doas_est is not None:
            errors = self._compute_errors(doas_true, doas_est)
        else:
            errors = None
            
        return {
            'doas_true': doas_true,
            'doas_est': doas_est,
            'errors': errors,
            'success': success,
            'error_msg': error_msg,
            'compute_time': compute_time,
            'snr_db': snr_db,
            'N_snapshots': N_snapshots
        }
    
    def run_monte_carlo(self,
                       estimator: Callable,
                       doas_true: np.ndarray,
                       N_trials: int = 100,
                       N_snapshots: int = 100,
                       snr_db: float = 10,
                       parallel: bool = True,
                       n_workers: Optional[int] = None,
                       **estimator_kwargs) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        doas_true : np.ndarray
            True DOAs in radians
        N_trials : int
            Number of Monte Carlo trials
        N_snapshots : int
            Number of snapshots per trial
        snr_db : float
            SNR in dB
        parallel : bool
            Use parallel processing
        n_workers : int, optional
            Number of worker processes
        **estimator_kwargs
            Additional arguments for the estimator
            
        Returns
        -------
        dict
            Monte Carlo results
        """
        print(f"Running {N_trials} Monte Carlo trials...")
        
        if parallel:
            return self._run_parallel(estimator, doas_true, N_trials, 
                                    N_snapshots, snr_db, n_workers, 
                                    **estimator_kwargs)
        else:
            return self._run_sequential(estimator, doas_true, N_trials,
                                      N_snapshots, snr_db, **estimator_kwargs)
    
    def _run_sequential(self, estimator, doas_true, N_trials, N_snapshots, 
                       snr_db, **estimator_kwargs):
        """Run trials sequentially."""
        results = []
        
        for trial in range(N_trials):
            if (trial + 1) % 10 == 0:
                print(f"Trial {trial + 1}/{N_trials}")
                
            seed = trial  # Use trial number as seed for reproducibility
            result = self.run_single_trial(
                estimator, doas_true, N_snapshots, snr_db, seed, 
                **estimator_kwargs
            )
            results.append(result)
            
        return self._process_results(results)
    
    def _run_parallel(self, estimator, doas_true, N_trials, N_snapshots,
                     snr_db, n_workers, **estimator_kwargs):
        """Run trials in parallel."""
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for trial in range(N_trials):
                seed = trial
                future = executor.submit(
                    self.run_single_trial,
                    estimator, doas_true, N_snapshots, snr_db, seed,
                    **estimator_kwargs
                )
                futures.append(future)
            
            results = []
            for i, future in enumerate(futures):
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{N_trials} trials")
                results.append(future.result())
                
        return self._process_results(results)
    
    def _process_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and summarize Monte Carlo results."""
        successful_results = [r for r in results if r['success']]
        N_successful = len(successful_results)
        N_total = len(results)
        
        if N_successful == 0:
            return {
                'success_rate': 0.0,
                'N_trials': N_total,
                'N_successful': 0,
                'errors': None,
                'statistics': None,
                'raw_results': results
            }
        
        # Collect errors from successful trials
        all_errors = [r['errors'] for r in successful_results if r['errors'] is not None]
        
        if len(all_errors) == 0:
            return {
                'success_rate': N_successful / N_total,
                'N_trials': N_total,
                'N_successful': N_successful,
                'errors': None,
                'statistics': None,
                'raw_results': results
            }
        
        # Compute statistics
        rmse_values = [e['rmse'] for e in all_errors]
        bias_values = [e['bias'] for e in all_errors]
        mae_values = [e['mae'] for e in all_errors]
        
        compute_times = [r['compute_time'] for r in successful_results]
        
        statistics = {
            'rmse': {
                'mean': np.mean(rmse_values),
                'std': np.std(rmse_values),
                'median': np.median(rmse_values)
            },
            'bias': {
                'mean': np.mean(bias_values),
                'std': np.std(bias_values), 
                'median': np.median(bias_values)
            },
            'mae': {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values),
                'median': np.median(mae_values)
            },
            'compute_time': {
                'mean': np.mean(compute_times),
                'std': np.std(compute_times),
                'median': np.median(compute_times)
            }
        }
        
        return {
            'success_rate': N_successful / N_total,
            'N_trials': N_total,
            'N_successful': N_successful,
            'statistics': statistics,
            'raw_results': results
        }
    
    def _compute_errors(self, doas_true: np.ndarray, doas_est: np.ndarray) -> Dict[str, float]:
        """
        Compute estimation errors.
        
        Parameters
        ----------
        doas_true : np.ndarray
            True DOAs
        doas_est : np.ndarray
            Estimated DOAs
            
        Returns
        -------
        dict
            Error metrics
        """
        doas_true = np.atleast_1d(doas_true)
        doas_est = np.atleast_1d(doas_est)
        
        if len(doas_est) != len(doas_true):
            # Handle different number of estimates
            # This is a simplified approach - more sophisticated matching could be used
            if len(doas_est) < len(doas_true):
                # Pad with NaNs
                doas_est = np.concatenate([doas_est, np.full(len(doas_true) - len(doas_est), np.nan)])
            else:
                # Truncate
                doas_est = doas_est[:len(doas_true)]
        
        # Sort both arrays for matching
        doas_true_sorted = np.sort(doas_true)
        doas_est_sorted = np.sort(doas_est[~np.isnan(doas_est)])
        
        if len(doas_est_sorted) == 0:
            return {'rmse': np.inf, 'mae': np.inf, 'bias': np.inf}
        
        # Pad if necessary
        if len(doas_est_sorted) < len(doas_true_sorted):
            doas_est_sorted = np.concatenate([
                doas_est_sorted, 
                np.full(len(doas_true_sorted) - len(doas_est_sorted), np.nan)
            ])
        
        # Compute errors for valid estimates
        valid_mask = ~np.isnan(doas_est_sorted)
        if np.sum(valid_mask) == 0:
            return {'rmse': np.inf, 'mae': np.inf, 'bias': np.inf}
        
        errors = doas_est_sorted[valid_mask] - doas_true_sorted[valid_mask]
        
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        bias = np.mean(errors)
        
        return {
            'rmse': rmse,
            'mae': mae, 
            'bias': bias,
            'individual_errors': errors
        }
    
    def snr_sweep(self,
                 estimator: Callable,
                 doas_true: np.ndarray,
                 snr_range: np.ndarray = np.arange(-10, 31, 5),
                 N_trials: int = 100,
                 N_snapshots: int = 100,
                 parallel: bool = True,
                 **estimator_kwargs) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation over a range of SNRs.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        doas_true : np.ndarray
            True DOAs
        snr_range : np.ndarray
            Range of SNR values in dB
        N_trials : int
            Number of trials per SNR
        N_snapshots : int
            Number of snapshots per trial
        parallel : bool
            Use parallel processing
        **estimator_kwargs
            Additional estimator arguments
            
        Returns
        -------
        dict
            SNR sweep results
        """
        print(f"Running SNR sweep from {snr_range[0]} to {snr_range[-1]} dB")
        
        snr_results = {}
        
        for snr_db in snr_range:
            print(f"\nSNR = {snr_db} dB")
            result = self.run_monte_carlo(
                estimator, doas_true, N_trials, N_snapshots, snr_db,
                parallel=parallel, **estimator_kwargs
            )
            snr_results[snr_db] = result
            
        return {
            'snr_range': snr_range,
            'results': snr_results,
            'doas_true': doas_true,
            'N_trials': N_trials,
            'N_snapshots': N_snapshots
        }