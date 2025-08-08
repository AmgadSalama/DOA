"""
Simulation Scenarios
===================

Predefined simulation scenarios for testing DOA estimation methods.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from ..array_processing import UniformLinearArray, SignalModel


class SimulationScenario:
    """
    Class for defining and generating simulation scenarios.
    
    This class provides predefined scenarios commonly used in DOA literature
    for testing and comparing different estimation methods.
    """
    
    @staticmethod
    def two_sources_close(array: UniformLinearArray, 
                         N_snapshots: int = 100,
                         snr_db: float = 10,
                         separation_deg: float = 5,
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Two closely spaced sources scenario.
        
        Tests angular resolution capability of DOA methods.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        N_snapshots : int
            Number of snapshots
        snr_db : float
            SNR in dB
        separation_deg : float
            Angular separation in degrees
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results with keys: 'X', 'doas_true', 'params'
        """
        signal_model = SignalModel(array)
        
        # DOAs: symmetric around 0 degrees
        sep_rad = np.deg2rad(separation_deg)
        doas = np.array([-sep_rad/2, sep_rad/2])
        
        X, S, N = signal_model.generate_signals(
            doas=doas,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            seed=seed
        )
        
        return {
            'X': X,
            'S': S, 
            'N': N,
            'doas_true': doas,
            'params': {
                'scenario': 'two_sources_close',
                'N_snapshots': N_snapshots,
                'snr_db': snr_db,
                'separation_deg': separation_deg,
                'separation_rad': sep_rad
            }
        }
    
    @staticmethod
    def multiple_sources_uncorrelated(array: UniformLinearArray,
                                    K: int = 3,
                                    N_snapshots: int = 100,
                                    snr_db: float = 10,
                                    seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Multiple uncorrelated sources scenario.
        
        Tests performance with multiple sources at different angles.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        K : int
            Number of sources
        N_snapshots : int
            Number of snapshots
        snr_db : float
            SNR in dB
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results
        """
        signal_model = SignalModel(array)
        
        # Generate DOAs uniformly distributed in visible region
        if seed is not None:
            np.random.seed(seed)
            
        doas = np.random.uniform(-np.pi/3, np.pi/3, K)
        doas = np.sort(doas)
        
        X, S, N = signal_model.generate_signals(
            doas=doas,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            correlation=0.0,
            seed=seed
        )
        
        return {
            'X': X,
            'S': S,
            'N': N,
            'doas_true': doas,
            'params': {
                'scenario': 'multiple_sources_uncorrelated',
                'K': K,
                'N_snapshots': N_snapshots,
                'snr_db': snr_db
            }
        }
    
    @staticmethod
    def correlated_sources(array: UniformLinearArray,
                          correlation: float = 0.8,
                          N_snapshots: int = 100,
                          snr_db: float = 10,
                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Two correlated sources scenario.
        
        Tests performance with correlated signals.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        correlation : float
            Correlation coefficient (0-1)
        N_snapshots : int
            Number of snapshots
        snr_db : float
            SNR in dB
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results
        """
        signal_model = SignalModel(array)
        
        # Two sources at -20 and +20 degrees
        doas = np.deg2rad([-20, 20])
        
        X, S, N = signal_model.generate_signals(
            doas=doas,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            correlation=correlation,
            seed=seed
        )
        
        return {
            'X': X,
            'S': S,
            'N': N,
            'doas_true': doas,
            'params': {
                'scenario': 'correlated_sources',
                'correlation': correlation,
                'N_snapshots': N_snapshots,
                'snr_db': snr_db
            }
        }
    
    @staticmethod
    def low_snr_scenario(array: UniformLinearArray,
                        snr_db: float = -5,
                        N_snapshots: int = 1000,
                        seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Low SNR scenario with single source.
        
        Tests performance in low SNR conditions.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        snr_db : float
            SNR in dB (negative for challenging scenario)
        N_snapshots : int
            Number of snapshots (typically high for low SNR)
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results
        """
        signal_model = SignalModel(array)
        
        # Single source at 10 degrees
        doas = np.deg2rad([10])
        
        X, S, N = signal_model.generate_signals(
            doas=doas,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            seed=seed
        )
        
        return {
            'X': X,
            'S': S,
            'N': N,
            'doas_true': doas,
            'params': {
                'scenario': 'low_snr',
                'snr_db': snr_db,
                'N_snapshots': N_snapshots
            }
        }
    
    @staticmethod
    def varying_snr_sources(array: UniformLinearArray,
                           snr_db_list: List[float] = [20, 10, 0],
                           N_snapshots: int = 100,
                           seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Multiple sources with different SNRs.
        
        Tests performance with sources of varying strengths.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        snr_db_list : list
            List of SNR values in dB for each source
        N_snapshots : int
            Number of snapshots
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results
        """
        signal_model = SignalModel(array)
        
        K = len(snr_db_list)
        
        # Generate well-separated DOAs
        if K == 3:
            doas = np.deg2rad([-30, 0, 30])
        else:
            doas = np.linspace(-np.pi/3, np.pi/3, K)
        
        X, S, N = signal_model.generate_signals(
            doas=doas,
            N_snapshots=N_snapshots,
            snr_db=snr_db_list,
            seed=seed
        )
        
        return {
            'X': X,
            'S': S,
            'N': N,
            'doas_true': doas,
            'params': {
                'scenario': 'varying_snr_sources',
                'snr_db_list': snr_db_list,
                'N_snapshots': N_snapshots
            }
        }
    
    @staticmethod
    def limited_snapshots(array: UniformLinearArray,
                         N_snapshots: int = 10,
                         snr_db: float = 10,
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Limited snapshots scenario.
        
        Tests performance with small sample sizes.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        N_snapshots : int
            Small number of snapshots
        snr_db : float
            SNR in dB
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Simulation results
        """
        signal_model = SignalModel(array)
        
        # Two sources at moderate separation
        doas = np.deg2rad([-15, 15])
        
        X, S, N = signal_model.generate_signals(
            doas=doas,
            N_snapshots=N_snapshots,
            snr_db=snr_db,
            seed=seed
        )
        
        return {
            'X': X,
            'S': S,
            'N': N,
            'doas_true': doas,
            'params': {
                'scenario': 'limited_snapshots',
                'N_snapshots': N_snapshots,
                'snr_db': snr_db
            }
        }
    
    @staticmethod
    def get_all_scenarios(array: UniformLinearArray, 
                         seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate all predefined scenarios.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array geometry
        seed : int, optional
            Random seed
            
        Returns
        -------
        dict
            Dictionary of all scenarios
        """
        scenarios = {}
        
        scenarios['two_close'] = SimulationScenario.two_sources_close(
            array, seed=seed)
        scenarios['multiple_uncorr'] = SimulationScenario.multiple_sources_uncorrelated(
            array, seed=seed)
        scenarios['correlated'] = SimulationScenario.correlated_sources(
            array, seed=seed)
        scenarios['low_snr'] = SimulationScenario.low_snr_scenario(
            array, seed=seed)
        scenarios['varying_snr'] = SimulationScenario.varying_snr_sources(
            array, seed=seed)
        scenarios['limited_snapshots'] = SimulationScenario.limited_snapshots(
            array, seed=seed)
            
        return scenarios