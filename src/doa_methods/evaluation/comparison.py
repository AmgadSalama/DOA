"""
Method Comparison Tools
======================

Tools for comparing multiple DOA estimation methods side-by-side.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import time
import pandas as pd
from ..array_processing import UniformLinearArray
from ..simulation import MonteCarlo
from .metrics import DOAMetrics
from .benchmarks import DOABenchmark


class MethodComparison:
    """
    Compare multiple DOA estimation methods across various scenarios.
    
    This class provides comprehensive comparison capabilities for
    evaluating and ranking different DOA estimation algorithms.
    """
    
    def __init__(self, array: UniformLinearArray):
        """
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry for comparison
        """
        self.array = array
        self.monte_carlo = MonteCarlo(array)
        self.benchmark = DOABenchmark(array)
        
    def compare_basic_performance(self,
                                 methods_dict: Dict[str, Callable],
                                 test_scenarios: Dict = None,
                                 N_trials: int = 100,
                                 **kwargs) -> pd.DataFrame:
        """
        Compare methods on basic performance scenarios.
        
        Parameters
        ----------
        methods_dict : dict
            Dictionary of {method_name: estimator_callable}
        test_scenarios : dict, optional
            Custom test scenarios
        N_trials : int
            Number of Monte Carlo trials
        **kwargs
            Additional arguments for estimators
            
        Returns
        -------
        pd.DataFrame
            Comparison results table
        """
        print("Comparing Basic Performance...")
        
        if test_scenarios is None:
            test_scenarios = {
                'Two Close (5°)': {
                    'doas': np.deg2rad([-2.5, 2.5]),
                    'N_snapshots': 200,
                    'snr_db': 10
                },
                'Two Separated (30°)': {
                    'doas': np.deg2rad([-15, 15]),
                    'N_snapshots': 100,
                    'snr_db': 10
                },
                'Three Sources': {
                    'doas': np.deg2rad([-20, 0, 20]),
                    'N_snapshots': 200,
                    'snr_db': 10
                },
                'Low SNR': {
                    'doas': np.deg2rad([-15, 15]),
                    'N_snapshots': 300,
                    'snr_db': 0
                }
            }
        
        # Results storage
        results_data = []
        
        for method_name, estimator in methods_dict.items():
            print(f"  Testing {method_name}...")
            
            for scenario_name, params in test_scenarios.items():
                try:
                    # Run Monte Carlo
                    mc_result = self.monte_carlo.run_monte_carlo(
                        estimator=estimator,
                        doas_true=params['doas'],
                        N_trials=N_trials,
                        N_snapshots=params['N_snapshots'],
                        snr_db=params['snr_db'],
                        **kwargs
                    )
                    
                    # Extract metrics
                    if mc_result['statistics']:
                        rmse_deg = np.rad2deg(mc_result['statistics']['rmse']['mean'])
                        bias_deg = np.rad2deg(mc_result['statistics']['bias']['mean'])
                        compute_time_ms = mc_result['statistics']['compute_time']['mean'] * 1000
                    else:
                        rmse_deg = np.inf
                        bias_deg = np.inf
                        compute_time_ms = np.inf
                    
                    results_data.append({
                        'Method': method_name,
                        'Scenario': scenario_name,
                        'RMSE (°)': rmse_deg,
                        'Bias (°)': bias_deg,
                        'Success Rate': mc_result['success_rate'],
                        'Compute Time (ms)': compute_time_ms,
                        'Sources': len(params['doas']),
                        'SNR (dB)': params['snr_db']
                    })
                    
                except Exception as e:
                    print(f"    Error with {method_name} on {scenario_name}: {str(e)}")
                    results_data.append({
                        'Method': method_name,
                        'Scenario': scenario_name,
                        'RMSE (°)': np.inf,
                        'Bias (°)': np.inf,
                        'Success Rate': 0.0,
                        'Compute Time (ms)': np.inf,
                        'Sources': len(params['doas']),
                        'SNR (dB)': params['snr_db']
                    })
        
        return pd.DataFrame(results_data)
    
    def compare_snr_performance(self,
                               methods_dict: Dict[str, Callable],
                               doas: np.ndarray = None,
                               snr_range: np.ndarray = None,
                               N_trials: int = 50,
                               N_snapshots: int = 200) -> Dict[str, Dict]:
        """
        Compare methods across SNR range.
        
        Parameters
        ----------
        methods_dict : dict
            Dictionary of methods
        doas : np.ndarray, optional
            DOAs to test
        snr_range : np.ndarray, optional
            SNR range in dB
        N_trials : int
            Number of trials per SNR
        N_snapshots : int
            Number of snapshots
            
        Returns
        -------
        dict
            SNR performance comparison results
        """
        print("Comparing SNR Performance...")
        
        if doas is None:
            doas = np.deg2rad([-15, 15])
        
        if snr_range is None:
            snr_range = np.arange(-5, 21, 2)
        
        comparison_results = {}
        
        for method_name, estimator in methods_dict.items():
            print(f"  Testing {method_name}...")
            
            try:
                snr_results = self.monte_carlo.snr_sweep(
                    estimator=estimator,
                    doas_true=doas,
                    snr_range=snr_range,
                    N_trials=N_trials,
                    N_snapshots=N_snapshots
                )
                
                # Extract metrics
                rmse_values = []
                success_rates = []
                
                for snr in snr_range:
                    result = snr_results['results'][snr]
                    if result['statistics']:
                        rmse_values.append(np.rad2deg(result['statistics']['rmse']['mean']))
                    else:
                        rmse_values.append(np.inf)
                    success_rates.append(result['success_rate'])
                
                comparison_results[method_name] = {
                    'snr_range': snr_range,
                    'rmse_deg': np.array(rmse_values),
                    'success_rates': np.array(success_rates),
                    'full_results': snr_results
                }
                
            except Exception as e:
                print(f"    Error with {method_name}: {str(e)}")
                comparison_results[method_name] = {
                    'error': str(e)
                }
        
        return comparison_results
    
    def compare_resolution_capability(self,
                                     methods_dict: Dict[str, Callable],
                                     separation_range: np.ndarray = None,
                                     N_trials: int = 100,
                                     snr_db: float = 15) -> Dict[str, Dict]:
        """
        Compare angular resolution capabilities.
        
        Parameters
        ----------
        methods_dict : dict
            Dictionary of methods
        separation_range : np.ndarray, optional
            Angular separations in degrees
        N_trials : int
            Number of trials per separation
        snr_db : float
            SNR in dB
            
        Returns
        -------
        dict
            Resolution comparison results
        """
        print("Comparing Resolution Capability...")
        
        if separation_range is None:
            separation_range = np.array([1, 2, 3, 4, 5, 7, 10, 15])
        
        comparison_results = {}
        
        for method_name, estimator in methods_dict.items():
            print(f"  Testing {method_name}...")
            
            try:
                resolution_results = self.benchmark.resolution_test(
                    estimator=estimator,
                    separation_range=separation_range,
                    N_trials=N_trials,
                    snr_db=snr_db
                )
                
                comparison_results[method_name] = resolution_results
                
            except Exception as e:
                print(f"    Error with {method_name}: {str(e)}")
                comparison_results[method_name] = {'error': str(e)}
        
        return comparison_results
    
    def compare_computational_complexity(self,
                                       methods_dict: Dict[str, Callable],
                                       array_sizes: List[int] = None,
                                       N_runs: int = 5) -> pd.DataFrame:
        """
        Compare computational complexity.
        
        Parameters
        ----------
        methods_dict : dict
            Dictionary of methods
        array_sizes : list, optional
            Array sizes to test
        N_runs : int
            Number of runs for timing
            
        Returns
        -------
        pd.DataFrame
            Complexity comparison table
        """
        print("Comparing Computational Complexity...")
        
        complexity_results = self.benchmark.computational_complexity_test(
            estimators_dict=methods_dict,
            array_sizes=array_sizes,
            N_runs=N_runs
        )
        
        # Convert to DataFrame
        data_rows = []
        
        for method_name, results in complexity_results.items():
            if 'error' not in results:
                for i, M in enumerate(results['array_sizes']):
                    data_rows.append({
                        'Method': method_name,
                        'Array Size': M,
                        'Mean Time (ms)': results['mean_times'][i] * 1000,
                        'Std Time (ms)': results['std_times'][i] * 1000,
                        'Complexity Order': results.get('complexity_order', 'N/A')
                    })
        
        return pd.DataFrame(data_rows)
    
    def rank_methods(self,
                    methods_dict: Dict[str, Callable],
                    ranking_criteria: Dict[str, float] = None,
                    N_trials: int = 50) -> pd.DataFrame:
        """
        Rank methods based on multiple criteria.
        
        Parameters
        ----------
        methods_dict : dict
            Dictionary of methods
        ranking_criteria : dict, optional
            Weights for ranking criteria
        N_trials : int
            Number of trials for evaluation
            
        Returns
        -------
        pd.DataFrame
            Method rankings
        """
        print("Ranking Methods...")
        
        if ranking_criteria is None:
            ranking_criteria = {
                'accuracy': 0.4,      # Weight for RMSE (inverted)
                'robustness': 0.3,    # Weight for success rate
                'speed': 0.2,         # Weight for computation time (inverted)
                'resolution': 0.1     # Weight for resolution capability
            }
        
        # Test scenarios for ranking
        ranking_scenarios = {
            'standard': {
                'doas': np.deg2rad([-15, 15]),
                'N_snapshots': 200,
                'snr_db': 10
            },
            'challenging': {
                'doas': np.deg2rad([-3, 3]),  # Close sources
                'N_snapshots': 100,
                'snr_db': 5
            }
        }
        
        method_scores = {}
        
        for method_name, estimator in methods_dict.items():
            print(f"  Evaluating {method_name}...")
            
            scores = {'accuracy': 0, 'robustness': 0, 'speed': 0, 'resolution': 0}
            
            try:
                # Test on scenarios
                rmse_scores = []
                success_scores = []
                time_scores = []
                
                for scenario_name, params in ranking_scenarios.items():
                    mc_result = self.monte_carlo.run_monte_carlo(
                        estimator=estimator,
                        doas_true=params['doas'],
                        N_trials=N_trials,
                        N_snapshots=params['N_snapshots'],
                        snr_db=params['snr_db']
                    )
                    
                    if mc_result['statistics']:
                        rmse_scores.append(mc_result['statistics']['rmse']['mean'])
                        time_scores.append(mc_result['statistics']['compute_time']['mean'])
                    else:
                        rmse_scores.append(np.inf)
                        time_scores.append(np.inf)
                    
                    success_scores.append(mc_result['success_rate'])
                
                # Compute normalized scores (0-1, higher is better)
                scores['accuracy'] = 1.0 / (1.0 + np.mean(rmse_scores))  # Lower RMSE is better
                scores['robustness'] = np.mean(success_scores)
                scores['speed'] = 1.0 / (1.0 + np.mean(time_scores))  # Lower time is better
                
                # Resolution test (simplified)
                try:
                    res_test = self.benchmark.resolution_test(
                        estimator, separation_range=np.array([2, 4, 6]), N_trials=20)
                    scores['resolution'] = np.mean(res_test['resolution_probabilities'])
                except:
                    scores['resolution'] = 0.5  # Default moderate score
                
            except Exception as e:
                print(f"    Error evaluating {method_name}: {str(e)}")
                # Give poor scores for failed methods
                scores = {'accuracy': 0.1, 'robustness': 0.1, 'speed': 0.1, 'resolution': 0.1}
            
            # Compute weighted overall score
            overall_score = sum(ranking_criteria[criterion] * scores[criterion] 
                              for criterion in ranking_criteria.keys())
            
            method_scores[method_name] = {
                'overall_score': overall_score,
                **scores
            }
        
        # Create ranking DataFrame
        ranking_data = []
        for method_name, scores in method_scores.items():
            ranking_data.append({
                'Method': method_name,
                'Overall Score': scores['overall_score'],
                'Accuracy Score': scores['accuracy'],
                'Robustness Score': scores['robustness'],
                'Speed Score': scores['speed'],
                'Resolution Score': scores['resolution']
            })
        
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('Overall Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        return df[['Rank', 'Method', 'Overall Score', 'Accuracy Score', 
                  'Robustness Score', 'Speed Score', 'Resolution Score']]
    
    def generate_comparison_report(self,
                                  methods_dict: Dict[str, Callable],
                                  include_plots: bool = False) -> str:
        """
        Generate comprehensive comparison report.
        
        Parameters
        ----------
        methods_dict : dict
            Dictionary of methods to compare
        include_plots : bool
            Whether to generate plots (requires matplotlib)
            
        Returns
        -------
        str
            Formatted comparison report
        """
        print("Generating Comparison Report...")
        
        report = "DOA METHODS COMPARISON REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Basic information
        report += f"Array Configuration: {self.array.M} elements, d={self.array.d}λ\n"
        report += f"Methods Compared: {list(methods_dict.keys())}\n\n"
        
        # Method rankings
        try:
            rankings = self.rank_methods(methods_dict, N_trials=30)
            report += "METHOD RANKINGS:\n"
            report += "-" * 20 + "\n"
            
            for _, row in rankings.iterrows():
                report += f"{row['Rank']}. {row['Method']} "
                report += f"(Score: {row['Overall Score']:.3f})\n"
                report += f"   Accuracy: {row['Accuracy Score']:.3f}, "
                report += f"Robustness: {row['Robustness Score']:.3f}\n"
                report += f"   Speed: {row['Speed Score']:.3f}, "
                report += f"Resolution: {row['Resolution Score']:.3f}\n\n"
        
        except Exception as e:
            report += f"Error generating rankings: {str(e)}\n\n"
        
        # Basic performance comparison
        try:
            basic_perf = self.compare_basic_performance(methods_dict, N_trials=20)
            
            report += "BASIC PERFORMANCE SUMMARY:\n"
            report += "-" * 30 + "\n"
            
            # Group by method and compute average performance
            method_summary = basic_perf.groupby('Method').agg({
                'RMSE (°)': 'mean',
                'Success Rate': 'mean',
                'Compute Time (ms)': 'mean'
            }).round(3)
            
            for method, row in method_summary.iterrows():
                report += f"{method:<20}: "
                report += f"RMSE={row['RMSE (°)']:.2f}°, "
                report += f"Success={row['Success Rate']:.1%}, "
                report += f"Time={row['Compute Time (ms)']:.1f}ms\n"
            
            report += "\n"
            
        except Exception as e:
            report += f"Error in basic performance comparison: {str(e)}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 16 + "\n"
        report += "• For high accuracy: Use subspace methods (MUSIC, ESPRIT)\n"
        report += "• For low SNR: Consider classical methods with regularization\n"
        report += "• For real-time: Use classical beamforming methods\n"
        report += "• For closely spaced sources: Use high-resolution methods\n\n"
        
        report += "End of Comparison Report\n"
        
        return report
    
    def head_to_head_comparison(self,
                               method1: Tuple[str, Callable],
                               method2: Tuple[str, Callable],
                               test_scenarios: Dict = None,
                               N_trials: int = 100) -> Dict:
        """
        Detailed head-to-head comparison of two methods.
        
        Parameters
        ----------
        method1 : tuple
            (name, estimator) for first method
        method2 : tuple
            (name, estimator) for second method
        test_scenarios : dict, optional
            Test scenarios
        N_trials : int
            Number of trials
            
        Returns
        -------
        dict
            Detailed comparison results
        """
        name1, estimator1 = method1
        name2, estimator2 = method2
        
        print(f"Head-to-Head: {name1} vs {name2}")
        
        if test_scenarios is None:
            test_scenarios = {
                'standard': {'doas': np.deg2rad([-15, 15]), 'N_snapshots': 200, 'snr_db': 10},
                'close_sources': {'doas': np.deg2rad([-3, 3]), 'N_snapshots': 300, 'snr_db': 15},
                'low_snr': {'doas': np.deg2rad([-20, 20]), 'N_snapshots': 500, 'snr_db': 0},
                'three_sources': {'doas': np.deg2rad([-20, 0, 20]), 'N_snapshots': 200, 'snr_db': 10}
            }
        
        comparison_results = {
            'method1_name': name1,
            'method2_name': name2,
            'scenarios': {},
            'summary': {}
        }
        
        wins_method1 = 0
        wins_method2 = 0
        
        for scenario_name, params in test_scenarios.items():
            print(f"  Testing {scenario_name}...")
            
            # Test method 1
            try:
                mc1 = self.monte_carlo.run_monte_carlo(
                    estimator1, params['doas'], N_trials, 
                    params['N_snapshots'], params['snr_db'])
                
                rmse1 = mc1['statistics']['rmse']['mean'] if mc1['statistics'] else np.inf
                success1 = mc1['success_rate']
            except:
                rmse1, success1 = np.inf, 0.0
            
            # Test method 2
            try:
                mc2 = self.monte_carlo.run_monte_carlo(
                    estimator2, params['doas'], N_trials,
                    params['N_snapshots'], params['snr_db'])
                
                rmse2 = mc2['statistics']['rmse']['mean'] if mc2['statistics'] else np.inf
                success2 = mc2['success_rate']
            except:
                rmse2, success2 = np.inf, 0.0
            
            # Determine winner
            score1 = success1 * (1.0 / (1.0 + rmse1))  # Combined metric
            score2 = success2 * (1.0 / (1.0 + rmse2))
            
            winner = name1 if score1 > score2 else name2
            if score1 > score2:
                wins_method1 += 1
            else:
                wins_method2 += 1
            
            comparison_results['scenarios'][scenario_name] = {
                name1: {'rmse': np.rad2deg(rmse1), 'success_rate': success1, 'score': score1},
                name2: {'rmse': np.rad2deg(rmse2), 'success_rate': success2, 'score': score2},
                'winner': winner
            }
        
        comparison_results['summary'] = {
            'overall_winner': name1 if wins_method1 > wins_method2 else name2,
            'wins': {name1: wins_method1, name2: wins_method2}
        }
        
        return comparison_results