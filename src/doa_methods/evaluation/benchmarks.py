"""
DOA Estimation Benchmarks
=========================

Standardized benchmark tests for DOA estimation methods.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import time
import warnings
from ..array_processing import UniformLinearArray, SignalModel
from ..simulation import SimulationScenario, MonteCarlo
from .metrics import DOAMetrics


class DOABenchmark:
    """
    Standardized benchmarks for DOA estimation methods.
    
    This class provides a comprehensive suite of benchmark tests
    commonly used in DOA literature for method evaluation.
    """
    
    def __init__(self, array: UniformLinearArray):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry for benchmarks
        """
        self.array = array
        self.signal_model = SignalModel(array)
        self.monte_carlo = MonteCarlo(array)
        
    def basic_performance_test(self, 
                              estimator: Callable,
                              N_trials: int = 100,
                              **estimator_kwargs) -> Dict[str, Dict]:
        """
        Basic performance test across standard scenarios.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        N_trials : int
            Number of Monte Carlo trials
        **estimator_kwargs
            Additional arguments for estimator
            
        Returns
        -------
        dict
            Benchmark results for each scenario
        """
        print("Running Basic Performance Benchmark...")
        
        # Define test scenarios
        scenarios = {
            'two_close_sources': {
                'doas': np.deg2rad([-5, 5]),
                'N_snapshots': 200,
                'snr_db': 10
            },
            'well_separated': {
                'doas': np.deg2rad([-30, 30]),
                'N_snapshots': 100,
                'snr_db': 10
            },
            'three_sources': {
                'doas': np.deg2rad([-20, 0, 20]),
                'N_snapshots': 200,
                'snr_db': 10
            },
            'low_snr': {
                'doas': np.deg2rad([-15, 15]),
                'N_snapshots': 500,
                'snr_db': 0
            },
            'few_snapshots': {
                'doas': np.deg2rad([-15, 15]),
                'N_snapshots': 20,
                'snr_db': 10
            }
        }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            print(f"  Testing {scenario_name}...")
            
            # Run Monte Carlo simulation
            mc_result = self.monte_carlo.run_monte_carlo(
                estimator=estimator,
                doas_true=params['doas'],
                N_trials=N_trials,
                N_snapshots=params['N_snapshots'],
                snr_db=params['snr_db'],
                parallel=True,
                **estimator_kwargs
            )
            
            results[scenario_name] = {
                'parameters': params,
                'monte_carlo_results': mc_result,
                'rmse_deg': np.rad2deg(mc_result['statistics']['rmse']['mean']) if mc_result['statistics'] else np.inf,
                'success_rate': mc_result['success_rate'],
                'mean_compute_time': mc_result['statistics']['compute_time']['mean'] if mc_result['statistics'] else np.inf
            }
        
        return results
    
    def snr_threshold_test(self,
                          estimator: Callable,
                          doas: np.ndarray = None,
                          snr_range: np.ndarray = None,
                          N_trials: int = 100,
                          N_snapshots: int = 200,
                          **estimator_kwargs) -> Dict[str, np.ndarray]:
        """
        Test performance vs SNR to find threshold.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        doas : np.ndarray, optional
            DOAs to test (default: two sources at ±15°)
        snr_range : np.ndarray, optional
            SNR range in dB
        N_trials : int
            Number of trials per SNR
        N_snapshots : int
            Number of snapshots
        **estimator_kwargs
            Estimator arguments
            
        Returns
        -------
        dict
            SNR threshold results
        """
        print("Running SNR Threshold Test...")
        
        if doas is None:
            doas = np.deg2rad([-15, 15])
        
        if snr_range is None:
            snr_range = np.arange(-10, 21, 2)
        
        # Run SNR sweep
        snr_results = self.monte_carlo.snr_sweep(
            estimator=estimator,
            doas_true=doas,
            snr_range=snr_range,
            N_trials=N_trials,
            N_snapshots=N_snapshots,
            **estimator_kwargs
        )
        
        # Extract key metrics
        snr_values = snr_results['snr_range']
        rmse_values = []
        success_rates = []
        
        for snr in snr_values:
            result = snr_results['results'][snr]
            if result['statistics']:
                rmse_values.append(np.rad2deg(result['statistics']['rmse']['mean']))
            else:
                rmse_values.append(np.inf)
            success_rates.append(result['success_rate'])
        
        # Find SNR threshold (where success rate drops below 90%)
        success_rates = np.array(success_rates)
        valid_snr_idx = np.where(success_rates >= 0.9)[0]
        
        if len(valid_snr_idx) > 0:
            snr_threshold = snr_values[valid_snr_idx[-1]]  # Last SNR with >90% success
        else:
            snr_threshold = np.inf
        
        return {
            'snr_range': snr_values,
            'rmse_deg': np.array(rmse_values),
            'success_rates': success_rates,
            'snr_threshold_90pct': snr_threshold,
            'full_results': snr_results
        }
    
    def resolution_test(self,
                       estimator: Callable,
                       separation_range: np.ndarray = None,
                       N_trials: int = 100,
                       snr_db: float = 15,
                       N_snapshots: int = 500,
                       **estimator_kwargs) -> Dict[str, np.ndarray]:
        """
        Test angular resolution capability.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        separation_range : np.ndarray, optional
            Angular separations to test (degrees)
        N_trials : int
            Number of trials per separation
        snr_db : float
            SNR in dB
        N_snapshots : int
            Number of snapshots
        **estimator_kwargs
            Estimator arguments
            
        Returns
        -------
        dict
            Resolution test results
        """
        print("Running Resolution Test...")
        
        if separation_range is None:
            separation_range = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])
        
        resolution_probs = []
        rmse_values = []
        
        for separation_deg in separation_range:
            print(f"  Testing {separation_deg}° separation...")
            
            # Symmetric sources around 0°
            sep_rad = np.deg2rad(separation_deg)
            doas = np.array([-sep_rad/2, sep_rad/2])
            
            # Run Monte Carlo
            mc_result = self.monte_carlo.run_monte_carlo(
                estimator=estimator,
                doas_true=doas,
                N_trials=N_trials,
                N_snapshots=N_snapshots,
                snr_db=snr_db,
                **estimator_kwargs
            )
            
            # Compute resolution probability
            valid_estimates = [r['doas_est'] for r in mc_result['raw_results'] 
                             if r['success'] and r['doas_est'] is not None]
            
            res_prob = DOAMetrics.resolution_probability(doas, valid_estimates, 
                                                       min_separation_factor=0.6)
            resolution_probs.append(res_prob)
            
            # RMSE
            if mc_result['statistics']:
                rmse_values.append(np.rad2deg(mc_result['statistics']['rmse']['mean']))
            else:
                rmse_values.append(np.inf)
        
        # Find resolution limit (50% probability)
        resolution_probs = np.array(resolution_probs)
        resolution_indices = np.where(resolution_probs >= 0.5)[0]
        
        if len(resolution_indices) > 0:
            resolution_limit = separation_range[resolution_indices[0]]
        else:
            resolution_limit = np.inf
        
        return {
            'separation_range': separation_range,
            'resolution_probabilities': resolution_probs,
            'rmse_deg': np.array(rmse_values),
            'resolution_limit_50pct': resolution_limit
        }
    
    def computational_complexity_test(self,
                                    estimators_dict: Dict[str, Callable],
                                    array_sizes: List[int] = None,
                                    N_snapshots: int = 100,
                                    doas: np.ndarray = None,
                                    N_runs: int = 10) -> Dict[str, Dict]:
        """
        Test computational complexity vs array size.
        
        Parameters
        ----------
        estimators_dict : dict
            Dictionary of {method_name: estimator}
        array_sizes : list, optional
            Array sizes to test
        N_snapshots : int
            Number of snapshots
        doas : np.ndarray, optional
            DOAs for test data
        N_runs : int
            Number of runs for timing
            
        Returns
        -------
        dict
            Complexity test results
        """
        print("Running Computational Complexity Test...")
        
        if array_sizes is None:
            array_sizes = [8, 16, 32, 64]
        
        if doas is None:
            doas = np.deg2rad([-20, 20])
        
        results = {}
        
        for method_name, estimator_func in estimators_dict.items():
            print(f"  Testing {method_name}...")
            
            method_results = {
                'array_sizes': array_sizes,
                'mean_times': [],
                'std_times': [],
                'complexity_order': None
            }
            
            for M in array_sizes:
                # Create temporary array
                temp_array = UniformLinearArray(M=M, d=0.5)
                temp_signal_model = SignalModel(temp_array)
                
                # Generate test data
                X, _, _ = temp_signal_model.generate_signals(
                    doas=doas,
                    N_snapshots=N_snapshots,
                    snr_db=10,
                    seed=42
                )
                
                # Time the estimator
                times = []
                for run in range(N_runs):
                    try:
                        start_time = time.time()
                        
                        # Create estimator for this array
                        if hasattr(estimator_func, '__class__'):
                            # It's a class - instantiate
                            estimator = estimator_func(temp_array)
                            _ = estimator.estimate(X, K=len(doas))
                        else:
                            # It's a function
                            _ = estimator_func(X, temp_array, K=len(doas))
                        
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        
                    except Exception as e:
                        print(f"    Error with {method_name} at M={M}: {str(e)}")
                        times.append(np.inf)
                
                method_results['mean_times'].append(np.mean(times))
                method_results['std_times'].append(np.std(times))
            
            # Estimate complexity order
            times_array = np.array(method_results['mean_times'])
            if np.all(np.isfinite(times_array)):
                # Fit polynomial to log-log plot
                log_M = np.log(array_sizes)
                log_t = np.log(times_array)
                
                # Linear fit: log(t) = a*log(M) + b  =>  complexity ~ M^a
                coeffs = np.polyfit(log_M, log_t, 1)
                method_results['complexity_order'] = coeffs[0]
            
            results[method_name] = method_results
        
        return results
    
    def robustness_test(self,
                       estimator: Callable,
                       test_conditions: Dict[str, Dict] = None,
                       N_trials: int = 50,
                       **estimator_kwargs) -> Dict[str, Dict]:
        """
        Test robustness to various conditions.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        test_conditions : dict, optional
            Test conditions to evaluate
        N_trials : int
            Number of trials per condition
        **estimator_kwargs
            Estimator arguments
            
        Returns
        -------
        dict
            Robustness test results
        """
        print("Running Robustness Test...")
        
        if test_conditions is None:
            test_conditions = {
                'correlated_sources': {
                    'doas': np.deg2rad([-15, 15]),
                    'N_snapshots': 200,
                    'snr_db': 10,
                    'correlation': 0.8
                },
                'unequal_powers': {
                    'doas': np.deg2rad([-20, 20]),
                    'N_snapshots': 200,
                    'snr_db': [15, 5],  # Different SNRs
                },
                'array_mismatch': {
                    'doas': np.deg2rad([-10, 10]),
                    'N_snapshots': 200,
                    'snr_db': 10,
                    'element_errors': 0.1  # 10% element position errors
                }
            }
        
        results = {}
        
        for condition_name, params in test_conditions.items():
            print(f"  Testing {condition_name}...")
            
            if condition_name == 'correlated_sources':
                # Test with correlated sources
                scenario_data = SimulationScenario.correlated_sources(
                    self.array, 
                    correlation=params['correlation'],
                    N_snapshots=params['N_snapshots'],
                    snr_db=params['snr_db']
                )
                
                doas_true = scenario_data['doas_true']
                
            elif condition_name == 'unequal_powers':
                # Test with different source powers
                doas_true = params['doas']
                scenario_data = SimulationScenario.varying_snr_sources(
                    self.array,
                    snr_db_list=params['snr_db'],
                    N_snapshots=params['N_snapshots']
                )
                
            else:
                # Default test
                doas_true = params['doas']
            
            # Run Monte Carlo for this condition
            try:
                if 'scenario_data' in locals():
                    # Use scenario data directly for first trial, then generate similar
                    X = scenario_data['X']
                    
                    # Run estimator once with scenario data
                    result_single = estimator(X, self.array, K=len(doas_true), **estimator_kwargs)
                    
                    # For full Monte Carlo, generate similar data
                    mc_result = self.monte_carlo.run_monte_carlo(
                        estimator=estimator,
                        doas_true=doas_true,
                        N_trials=N_trials,
                        N_snapshots=params['N_snapshots'],
                        snr_db=params.get('snr_db', 10),
                        **estimator_kwargs
                    )
                else:
                    mc_result = self.monte_carlo.run_monte_carlo(
                        estimator=estimator,
                        doas_true=doas_true,
                        N_trials=N_trials,
                        N_snapshots=params['N_snapshots'],
                        snr_db=params['snr_db'],
                        **estimator_kwargs
                    )
                
                results[condition_name] = {
                    'parameters': params,
                    'success_rate': mc_result['success_rate'],
                    'rmse_deg': np.rad2deg(mc_result['statistics']['rmse']['mean']) if mc_result['statistics'] else np.inf
                }
                
            except Exception as e:
                print(f"    Error in {condition_name}: {str(e)}")
                results[condition_name] = {
                    'parameters': params,
                    'success_rate': 0.0,
                    'rmse_deg': np.inf,
                    'error': str(e)
                }
        
        return results
    
    def comprehensive_benchmark(self,
                              estimator: Callable,
                              **estimator_kwargs) -> Dict[str, Dict]:
        """
        Run comprehensive benchmark suite.
        
        Parameters
        ----------
        estimator : callable
            DOA estimation function
        **estimator_kwargs
            Estimator arguments
            
        Returns
        -------
        dict
            Complete benchmark results
        """
        print(f"Running Comprehensive Benchmark for {getattr(estimator, '__name__', 'Unknown Method')}")
        print("="*60)
        
        benchmark_results = {}
        
        # 1. Basic performance
        try:
            benchmark_results['basic_performance'] = self.basic_performance_test(
                estimator, **estimator_kwargs)
        except Exception as e:
            print(f"Basic performance test failed: {str(e)}")
            benchmark_results['basic_performance'] = {'error': str(e)}
        
        # 2. SNR threshold
        try:
            benchmark_results['snr_threshold'] = self.snr_threshold_test(
                estimator, **estimator_kwargs)
        except Exception as e:
            print(f"SNR threshold test failed: {str(e)}")
            benchmark_results['snr_threshold'] = {'error': str(e)}
        
        # 3. Resolution test
        try:
            benchmark_results['resolution'] = self.resolution_test(
                estimator, **estimator_kwargs)
        except Exception as e:
            print(f"Resolution test failed: {str(e)}")
            benchmark_results['resolution'] = {'error': str(e)}
        
        # 4. Robustness test
        try:
            benchmark_results['robustness'] = self.robustness_test(
                estimator, **estimator_kwargs)
        except Exception as e:
            print(f"Robustness test failed: {str(e)}")
            benchmark_results['robustness'] = {'error': str(e)}
        
        print("Benchmark completed!")
        return benchmark_results
    
    def generate_benchmark_report(self, results: Dict, method_name: str = "Unknown") -> str:
        """
        Generate human-readable benchmark report.
        
        Parameters
        ----------
        results : dict
            Benchmark results
        method_name : str
            Name of the method
            
        Returns
        -------
        str
            Formatted report
        """
        report = f"DOA ESTIMATION BENCHMARK REPORT\n"
        report += f"{'='*50}\n"
        report += f"Method: {method_name}\n"
        report += f"Array: {self.array.M} elements, d={self.array.d}λ\n\n"
        
        # Basic performance
        if 'basic_performance' in results and 'error' not in results['basic_performance']:
            report += "BASIC PERFORMANCE:\n"
            report += "-" * 20 + "\n"
            
            for scenario, data in results['basic_performance'].items():
                report += f"{scenario:<20}: "
                report += f"RMSE={data['rmse_deg']:.2f}°, "
                report += f"Success={data['success_rate']:.1%}, "
                report += f"Time={data['mean_compute_time']*1000:.1f}ms\n"
            report += "\n"
        
        # SNR threshold
        if 'snr_threshold' in results and 'error' not in results['snr_threshold']:
            snr_data = results['snr_threshold']
            report += f"SNR THRESHOLD: {snr_data['snr_threshold_90pct']} dB (90% success)\n\n"
        
        # Resolution
        if 'resolution' in results and 'error' not in results['resolution']:
            res_data = results['resolution']
            report += f"ANGULAR RESOLUTION: {res_data['resolution_limit_50pct']}° (50% probability)\n\n"
        
        # Robustness
        if 'robustness' in results and 'error' not in results['robustness']:
            report += "ROBUSTNESS:\n"
            report += "-" * 12 + "\n"
            
            for condition, data in results['robustness'].items():
                if 'error' not in data:
                    report += f"{condition:<20}: "
                    report += f"RMSE={data['rmse_deg']:.2f}°, "
                    report += f"Success={data['success_rate']:.1%}\n"
            report += "\n"
        
        report += "End of Report\n"
        
        return report