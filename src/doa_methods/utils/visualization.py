"""
Visualization Utilities
=======================

Plotting and visualization functions for DOA estimation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Dict, Tuple, Union
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


class DOAPlotter:
    """
    Comprehensive plotting utilities for DOA estimation visualization.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Parameters
        ----------
        figsize : tuple
            Default figure size
        """
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
    def plot_array_geometry(self, array, ax=None, show_elements=True, 
                           show_aperture=True, title="Array Geometry"):
        """
        Plot ULA geometry.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array object
        ax : matplotlib axis, optional
            Axis to plot on
        show_elements : bool
            Show individual elements
        show_aperture : bool
            Show array aperture
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        
        # Element positions in meters
        positions = array.element_positions * array.wavelength
        
        # Plot elements
        if show_elements:
            ax.scatter(positions, np.zeros_like(positions), 
                      s=100, c='blue', marker='o', edgecolors='black')
            
            # Label elements
            for i, pos in enumerate(positions):
                ax.annotate(f'{i}', (pos, 0), xytext=(0, 15), 
                           textcoords='offset points', ha='center')
        
        # Show aperture
        if show_aperture:
            aperture_start = positions[0]
            aperture_end = positions[-1]
            aperture_length = aperture_end - aperture_start
            
            ax.annotate('', xy=(aperture_end, -0.1), xytext=(aperture_start, -0.1),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text((aperture_start + aperture_end)/2, -0.2, 
                   f'Aperture: {aperture_length:.2f} m', 
                   ha='center', color='red')
        
        # Wavelength reference
        ax.axhspan(-0.05, 0.05, xmin=0, xmax=array.wavelength/max(positions), 
                  alpha=0.3, color='green', label=f'λ = {array.wavelength:.2f} m')
        
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Cross-range')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.3, 0.3)
        
        # Array parameters text
        info_text = f'M = {array.M}, d = {array.d}λ, fc = {array.fc/1e9:.1f} GHz'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    def plot_beam_patterns(self, angle_grid, patterns_dict, doas_true=None,
                          normalize=True, db_scale=True, ax=None, 
                          title="Beam Patterns"):
        """
        Plot multiple beam patterns for comparison.
        
        Parameters
        ----------
        angle_grid : np.ndarray
            Angle grid in radians
        patterns_dict : dict
            Dictionary of {method_name: pattern_values}
        doas_true : np.ndarray, optional
            True DOAs for reference lines
        normalize : bool
            Normalize patterns
        db_scale : bool
            Use dB scale
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        angle_deg = np.rad2deg(angle_grid)
        
        for i, (method, pattern) in enumerate(patterns_dict.items()):
            # Normalize if requested
            if normalize:
                pattern = pattern / np.max(pattern)
            
            # Convert to dB if requested
            if db_scale:
                pattern_plot = 10 * np.log10(np.maximum(pattern, 1e-10))
                ylabel = 'Power (dB)'
            else:
                pattern_plot = pattern
                ylabel = 'Power (linear)'
            
            ax.plot(angle_deg, pattern_plot, label=method, 
                   color=self.colors[i % len(self.colors)], linewidth=2)
        
        # True DOA reference lines
        if doas_true is not None:
            for doa in doas_true:
                ax.axvline(np.rad2deg(doa), color='red', linestyle='--', 
                          alpha=0.7, linewidth=1)
            ax.axvline(np.rad2deg(doas_true[0]), color='red', linestyle='--', 
                      alpha=0.7, label='True DOA', linewidth=1)
        
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-90, 90)
    
    def plot_music_spectrum(self, angle_grid, spectrum, doas_true=None, 
                           doas_estimated=None, ax=None, 
                           title="MUSIC Spectrum"):
        """
        Plot MUSIC spectrum with enhanced visualization.
        
        Parameters
        ----------
        angle_grid : np.ndarray
            Angle grid in radians
        spectrum : np.ndarray
            MUSIC spectrum values
        doas_true : np.ndarray, optional
            True DOAs
        doas_estimated : np.ndarray, optional
            Estimated DOAs
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        angle_deg = np.rad2deg(angle_grid)
        spectrum_db = 10 * np.log10(spectrum / np.max(spectrum))
        
        # Main spectrum
        ax.plot(angle_deg, spectrum_db, 'b-', linewidth=2, label='MUSIC Spectrum')
        
        # Fill area
        ax.fill_between(angle_deg, spectrum_db, alpha=0.3)
        
        # True DOAs
        if doas_true is not None:
            for i, doa in enumerate(doas_true):
                ax.axvline(np.rad2deg(doa), color='red', linestyle='--', 
                          linewidth=2, alpha=0.8)
                ax.text(np.rad2deg(doa), ax.get_ylim()[1] - 2, f'True {i+1}', 
                       rotation=90, ha='right', va='top', color='red')
        
        # Estimated DOAs
        if doas_estimated is not None:
            for i, doa in enumerate(doas_estimated):
                ax.axvline(np.rad2deg(doa), color='green', linestyle='-', 
                          linewidth=2, alpha=0.8)
                ax.text(np.rad2deg(doa), ax.get_ylim()[0] + 2, f'Est {i+1}', 
                       rotation=90, ha='left', va='bottom', color='green')
        
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Normalized Power (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90, 90)
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], color='blue', label='MUSIC Spectrum')]
        if doas_true is not None:
            legend_elements.append(plt.Line2D([0], [0], color='red', 
                                             linestyle='--', label='True DOAs'))
        if doas_estimated is not None:
            legend_elements.append(plt.Line2D([0], [0], color='green', 
                                             label='Estimated DOAs'))
        ax.legend(handles=legend_elements)
    
    def plot_error_analysis(self, errors_dict, ax=None, title="DOA Estimation Errors"):
        """
        Plot error analysis results.
        
        Parameters
        ----------
        errors_dict : dict
            Dictionary of {method_name: errors_array}
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(errors_dict.keys())
        errors_list = [np.rad2deg(errors_dict[method]) for method in methods]
        
        # Box plot
        bp = ax.boxplot(errors_list, labels=methods, patch_artist=True)
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Absolute Error (degrees)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for method in methods:
            errors = np.rad2deg(errors_dict[method])
            rmse = np.sqrt(np.mean(errors**2))
            stats_text.append(f'{method}: RMSE = {rmse:.2f}°')
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    def plot_snr_performance(self, snr_results, ax=None, 
                            title="Performance vs SNR"):
        """
        Plot performance curves vs SNR.
        
        Parameters
        ----------
        snr_results : dict
            Dictionary with SNR performance data
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (method, data) in enumerate(snr_results.items()):
            snr_values = data['snr']
            rmse_values = data['rmse']
            
            ax.semilogy(snr_values, rmse_values, 'o-', 
                       label=method, color=self.colors[i % len(self.colors)],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('RMSE (degrees)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_resolution_analysis(self, separation_range, resolution_results, 
                               ax=None, title="Angular Resolution Analysis"):
        """
        Plot resolution analysis results.
        
        Parameters
        ----------
        separation_range : np.ndarray
            Angular separations tested
        resolution_results : dict
            Resolution test results
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (method, data) in enumerate(resolution_results.items()):
            separations = data['separations']
            resolved = np.array(data['resolved'])
            
            # Convert boolean to percentage
            resolution_rate = resolved.astype(float)
            
            ax.plot(separations, resolution_rate, 'o-', 
                   label=method, color=self.colors[i % len(self.colors)],
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Angular Separation (degrees)')
        ax.set_ylabel('Resolution Success Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0%', '50%', '100%'])
    
    def plot_array_manifold(self, array, doas, ax=None, 
                           title="Array Manifold Visualization"):
        """
        Visualize array manifold (steering vectors).
        
        Parameters
        ----------
        array : UniformLinearArray
            Array object
        doas : np.ndarray
            DOAs to visualize
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Compute steering vectors
        A = array.array_manifold(doas)
        
        # Plot magnitude and phase
        element_indices = np.arange(array.M)
        
        for i, doa in enumerate(doas):
            a = A[:, i]
            
            # Magnitude (should be 1 for ULA)
            ax.plot(element_indices, np.abs(a), 'o-', 
                   label=f'|a({np.rad2deg(doa):.1f}°)|',
                   color=self.colors[i % len(self.colors)])
            
            # Phase
            ax.plot(element_indices, np.angle(a), 's--', 
                   label=f'∠a({np.rad2deg(doa):.1f}°)',
                   color=self.colors[i % len(self.colors)], alpha=0.7)
        
        ax.set_xlabel('Element Index')
        ax.set_ylabel('Magnitude / Phase (rad)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_covariance_matrix(self, R, ax=None, title="Covariance Matrix"):
        """
        Visualize covariance matrix.
        
        Parameters
        ----------
        R : np.ndarray
            Covariance matrix
        ax : matplotlib axis, optional
            Axis to plot on
        title : str
            Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot magnitude in dB
        R_db = 20 * np.log10(np.abs(R) + 1e-10)
        
        im = ax.imshow(R_db, cmap='viridis', aspect='equal')
        ax.set_xlabel('Element Index')
        ax.set_ylabel('Element Index')
        ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)')
        
        # Add text annotations for small matrices
        if R.shape[0] <= 8:
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    text = ax.text(j, i, f'{R_db[i, j]:.1f}',
                                 ha="center", va="center", color="white")
    
    def create_comprehensive_figure(self, array, X, doas_true, results_dict):
        """
        Create comprehensive multi-panel figure.
        
        Parameters
        ----------
        array : UniformLinearArray
            Array object
        X : np.ndarray
            Data matrix
        doas_true : np.ndarray
            True DOAs
        results_dict : dict
            Results from multiple methods
        
        Returns
        -------
        fig : matplotlib figure
            Complete figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Layout: 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Array geometry
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_array_geometry(array, ax=ax1, title="Array Geometry")
        
        # 2. Sample covariance matrix
        ax2 = fig.add_subplot(gs[0, 1])
        R = X @ X.conj().T / X.shape[1]
        self.plot_covariance_matrix(R, ax=ax2, title="Sample Covariance")
        
        # 3. Array manifold
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_array_manifold(array, doas_true, ax=ax3, 
                                title="Array Manifold (True DOAs)")
        
        # 4-6. Beam patterns / spectra
        angle_grid = array.angle_grid()
        
        # Classical methods beam patterns
        ax4 = fig.add_subplot(gs[1, 0])
        classical_patterns = {}
        for name in ['Conventional BF', 'Capon']:
            if name in results_dict:
                if hasattr(results_dict[name]['estimator'], 'beam_pattern'):
                    pattern = results_dict[name]['estimator'].beam_pattern(X, angle_grid)
                    classical_patterns[name] = pattern
        
        if classical_patterns:
            self.plot_beam_patterns(angle_grid, classical_patterns, 
                                   doas_true, ax=ax4, 
                                   title="Classical Beam Patterns")
        
        # MUSIC spectrum
        ax5 = fig.add_subplot(gs[1, 1])
        if 'MUSIC' in results_dict:
            music_estimator = results_dict['MUSIC']['estimator']
            # Get noise subspace for spectrum calculation
            R = X @ X.conj().T / X.shape[1]
            eigenvals, eigenvecs = np.linalg.eigh(R)
            idx = np.argsort(eigenvals)[::-1]
            U_n = eigenvecs[:, idx[len(doas_true):]]
            
            spectrum = music_estimator.music_spectrum_vectorized(angle_grid, U_n)
            self.plot_music_spectrum(angle_grid, spectrum, 
                                    doas_true, results_dict['MUSIC']['doas'],
                                    ax=ax5)
        
        # DOA estimates comparison
        ax6 = fig.add_subplot(gs[1, 2])
        methods = list(results_dict.keys())
        estimates = [results_dict[m]['doas'] for m in methods]
        
        y_pos = np.arange(len(methods))
        colors_plot = [self.colors[i % len(self.colors)] for i in range(len(methods))]
        
        for i, (method, doas_est) in enumerate(zip(methods, estimates)):
            ax6.scatter(np.rad2deg(doas_est), [i] * len(doas_est), 
                       color=colors_plot[i], s=100, alpha=0.7, label=method)
        
        # True DOAs as reference
        for doa_true in doas_true:
            ax6.axvline(np.rad2deg(doa_true), color='red', linestyle='--', alpha=0.7)
        
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(methods)
        ax6.set_xlabel('DOA Estimate (degrees)')
        ax6.set_title('DOA Estimates Comparison')
        ax6.grid(True, alpha=0.3)
        
        # 7-9. Error analysis, performance metrics
        ax7 = fig.add_subplot(gs[2, 0])
        errors_dict = {}
        for method, result in results_dict.items():
            if result['doas'] is not None and len(result['doas']) == len(doas_true):
                errors = np.abs(np.sort(result['doas']) - np.sort(doas_true))
                errors_dict[method] = errors
        
        if errors_dict:
            self.plot_error_analysis(errors_dict, ax=ax7)
        
        # Summary statistics
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        # Create summary table
        summary_text = "PERFORMANCE SUMMARY\n" + "="*50 + "\n"
        summary_text += f"Array: {array.M} elements, d={array.d}λ\n"
        summary_text += f"True DOAs: {np.rad2deg(doas_true):.2f}°\n"
        summary_text += f"Data: {X.shape[1]} snapshots\n\n"
        
        summary_text += f"{'Method':<15} {'Estimates':<25} {'RMSE':<10} {'Time':<10}\n"
        summary_text += "-"*70 + "\n"
        
        for method, result in results_dict.items():
            if result['doas'] is not None:
                est_str = f"{np.rad2deg(result['doas']):.1f}°"
                if len(result['doas']) == len(doas_true):
                    errors = np.abs(np.sort(result['doas']) - np.sort(doas_true))
                    rmse = np.sqrt(np.mean(errors**2))
                    rmse_str = f"{np.rad2deg(rmse):.2f}°"
                else:
                    rmse_str = "N/A"
                
                time_str = f"{result.get('time', 0)*1000:.1f}ms"
                summary_text += f"{method:<15} {est_str:<25} {rmse_str:<10} {time_str:<10}\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('DOA Estimation Methods - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold')
        
        return fig


def quick_plot_comparison(array, X, doas_true, methods_dict):
    """
    Quick comparison plot for multiple DOA methods.
    
    Parameters
    ----------
    array : UniformLinearArray
        Array object
    X : np.ndarray
        Data matrix
    doas_true : np.ndarray
        True DOAs
    methods_dict : dict
        Dictionary of {method_name: estimator}
        
    Returns
    -------
    fig : matplotlib figure
        Comparison figure
    """
    plotter = DOAPlotter()
    
    # Run all methods
    results = {}
    for name, estimator in methods_dict.items():
        try:
            import time
            start_time = time.time()
            doas_est = estimator.estimate(X, K=len(doas_true))
            compute_time = time.time() - start_time
            
            results[name] = {
                'estimator': estimator,
                'doas': doas_est,
                'time': compute_time
            }
        except Exception as e:
            print(f"Method {name} failed: {str(e)}")
            results[name] = {
                'estimator': estimator,
                'doas': None,
                'time': None
            }
    
    # Create comprehensive figure
    fig = plotter.create_comprehensive_figure(array, X, doas_true, results)
    
    return fig