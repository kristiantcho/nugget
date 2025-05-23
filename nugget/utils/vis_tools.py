import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import clear_output, display
import math
from typing import List, Dict, Union, Tuple, Optional, Any, Callable
import io # Added for GIF generation
import imageio # Added for GIF generation

# Try importing plotly for interactive 3D plotting, but don't fail if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Visualizer:
    """Base class for visualization tools in geometry optimization."""
    
    # Define plot types as class constants
    PLOT_LOSS = "loss"
    PLOT_UW_LOSS = "uw_loss"
    PLOT_SNR_HISTORY = "snr_history"
    PLOT_3D_POINTS = "3d_points"
    PLOT_STRING_XY = "string_xy"
    PLOT_Z_DIST = "z_distribution"
    PLOT_XY_PROJECTION = "xy_projection"
    PLOT_SIGNAL_CONTOUR = "signal_contour"
    PLOT_BACKGROUND_CONTOUR = "background_contour"
    PLOT_PARAM_1D = "parameter_1d"
    PLOT_PARAM_2D = "parameter_2d"
    PLOT_STRING_DIST = "string_distribution"
    PLOT_TRUE_FUNCTION = "true_function"
    PLOT_INTERP_FUNCTION = "interp_function"
    PLOT_ERROR_FUNCTION = "error_function"
    PLOT_SURROGATE_FUNCTION = "surrogate_function"
    
    def __init__(self, device=None, dim=3, domain_size=2.0):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        dim : int
            Dimensionality of the space (2 or 3).
        domain_size : float
            Size of the domain from -domain_size/2 to domain_size/2 in each dimension.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim
        self.domain_size = domain_size
        self.half_domain = domain_size / 2
        self.gif_frames = [] # Added to store frames for the GIF
    
    def visualize_progress(self, 
                          iteration: int, 
                          points_3d: torch.Tensor, 
                          loss_history: List[float], 
                          additional_metrics: Optional[Dict[str, Any]] = None, 
                          string_indices: Optional[List[int]] = None, 
                          points_per_string_list: Optional[List[int]] = None, 
                          string_xy: Optional[torch.Tensor] = None,
                          slice_res: int = 50, 
                          multi_slice: bool = False, 
                          loss_type: str = 'rbf',
                          plot_types: Optional[List[str]] = None,
                          make_gif: bool = False, # New parameter for GIF creation
                          gif_plot_selection: Optional[List[str]] = None, # New: specific plots for GIF
                          gif_filename: str = "optimization_progress.gif", # New: GIF filename
                          gif_fps: int = 2, # New: GIF frames per second
                          geometry_type: Optional[str] = None, # ADDED geometry_type
                          **kwargs) -> None:
        """
        Visualize optimization progress with customizable plot selection and optional GIF generation.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number.
        points_3d : torch.Tensor
            3D points to visualize (shape: n_points x 3).
        loss_history : list
            History of loss values.
        additional_metrics : dict or None
            Additional metrics to visualize (e.g., SNR history).
        string_indices : list or None
            String index for each point.
        points_per_string_list : list or None
            Number of points on each string.
        string_xy : torch.Tensor or None
            XY positions of strings.
        slice_res : int
            Resolution for visualization slices.
        multi_slice : bool
            Whether to use multiple slices for visualization.
        loss_type : str
            Type of loss function used ('rbf', 'snr', or 'surrogate').
        plot_types : list of str or None
            List of plot types to display. If None, displays default plots for the loss type.
            Available plot types:
            - 'loss': Loss history
            - 'snr_history': SNR history over iterations
            - '3d_points': 3D visualization of points
            - 'string_xy': XY positions of strings with points per string
            - 'z_distribution': Distribution of z values
            - 'xy_projection': XY projection of points colored by Z
            - 'signal_contour': Contour plot of signal function
            - 'background_contour': Contour plot of background function
            - 'parameter_1d': 1D parameter vs SNR plot
            - 'parameter_2d': 2D parameter space contour plot
            - 'string_distribution': String distribution bar plot
            - 'true_function': True function contour
            - 'interp_function': Interpolated function contour
            - 'error_function': Error function contour
            - 'surrogate_function': Surrogate function contour
        make_gif : bool
            Whether to generate and save a GIF of the progress.
        gif_plot_selection : list of str or None
            List of plot types to display in each GIF frame. If None, uses a default set.
            Uses the same plot type strings as 'plot_types'.
        gif_filename : str
            Filename for the generated GIF.
        gif_fps : int
            Frames per second for the generated GIF.
        geometry_type : str, optional
            The type of geometry being used.
        kwargs : dict
            Additional keyword arguments for specific loss types.
            For surrogate visualization:
            - surrogate_funcs: List of surrogate functions
            - surrogate_model: The surrogate model instance
            - compute_rbf_interpolant: Function to compute RBF interpolant
        """
        # Clear previous output
        # clear_output(wait=True)

        # GIF Generation Logic
        if make_gif:
            current_gif_plot_types = []
            if gif_plot_selection is not None:
                current_gif_plot_types = gif_plot_selection
            else:
                # Default plot types for GIF: Loss and 3D points if available
                if hasattr(self, 'PLOT_LOSS'):
                    current_gif_plot_types.append(self.PLOT_LOSS)
                if hasattr(self, 'PLOT_3D_POINTS'):
                    current_gif_plot_types.append(self.PLOT_3D_POINTS)
                # Add more defaults if desired, e.g., based on loss_type or available data

            if not current_gif_plot_types:
                print("No plot types selected for GIF frame, skipping GIF update for this iteration.")
            else:
                num_gif_plots = len(current_gif_plot_types)
                num_gif_cols = 3  # Or a different layout, e.g., 2 for smaller GIF frames
                num_gif_rows = (num_gif_plots + num_gif_cols - 1) // num_gif_cols
                
                gif_fig_size = (12, 4 * num_gif_rows) 

                fig_gif, axes_gif_array = plt.subplots(num_gif_rows, num_gif_cols, 
                                                       figsize=gif_fig_size, squeeze=False,
                                                       ) # MODIFIED
                axes_gif_flat_for_loop = axes_gif_array.flatten()

                for i, plot_type_gif in enumerate(current_gif_plot_types):
                    ax_gif = axes_gif_flat_for_loop[i]
                    self._create_plot(
                        plot_type=plot_type_gif,
                        ax=ax_gif,
                        fig=fig_gif,
                        iteration=iteration,
                        points_3d=points_3d,
                        loss_history=loss_history,
                        additional_metrics=additional_metrics,
                        string_indices=string_indices,
                        points_per_string_list=points_per_string_list,
                        string_xy=string_xy,
                        slice_res=slice_res,
                        multi_slice=multi_slice,
                        loss_type=loss_type,
                        geometry_type=geometry_type, # PASSING geometry_type
                        **kwargs
                    )
                
                for i in range(num_gif_plots, num_gif_rows * num_gif_cols):
                    axes_gif_flat_for_loop[i].axis('off')
                
                fig_gif.tight_layout() # ADDED
                
                img_buf = io.BytesIO()
                fig_gif.savefig(img_buf, format='png')
                img_buf.seek(0)
                self.gif_frames.append(imageio.v3.imread(img_buf))
                img_buf.close()
                plt.close(fig_gif)

                if self.gif_frames:
                    try:
                        imageio.mimsave(gif_filename, self.gif_frames, fps=gif_fps)
                        # print(f"GIF '{gif_filename}' updated with {len(self.gif_frames)} frames (Iteration {iteration}).")
                    except Exception as e:
                        print(f"Error saving GIF: {e}")
                        
            return
        
        # Set default plot types based on loss type if not specified
        clear_output(wait=True)
        if plot_types is None:
            if loss_type == 'rbf':
                plot_types = [
                    self.PLOT_UW_LOSS,
                    self.PLOT_LOSS,
                    self.PLOT_3D_POINTS,
                    self.PLOT_STRING_XY if string_xy is not None else self.PLOT_XY_PROJECTION
                ]
            elif loss_type == 'snr':
                plot_types = [
                    self.PLOT_LOSS,
                    self.PLOT_SNR_HISTORY,
                    self.PLOT_3D_POINTS,
                    self.PLOT_SIGNAL_CONTOUR,
                    self.PLOT_BACKGROUND_CONTOUR,
                    self.PLOT_PARAM_1D if 'optimize_params' in kwargs and len(kwargs['optimize_params']) == 1 else self.PLOT_PARAM_2D
                ]
            elif loss_type == 'surrogate':
                plot_types = [
                    self.PLOT_LOSS,
                    self.PLOT_3D_POINTS,
                    self.PLOT_STRING_XY if string_xy is not None else self.PLOT_XY_PROJECTION,
                    self.PLOT_SURROGATE_FUNCTION,
                    self.PLOT_INTERP_FUNCTION,
                    self.PLOT_ERROR_FUNCTION
                ]
        
        # Create figure with proper layout based on number of plots
        num_plots = len(plot_types)
        num_rows = (num_plots + 2) // 3  # Ceiling division to get number of rows needed
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows)) # MODIFIED
        
        # If only one row, ensure axes is still a 2D array
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Generate each requested plot
        for i, plot_type in enumerate(plot_types):
            row_idx = i // 3
            col_idx = i % 3
            ax = axes[row_idx, col_idx]
            
            # Create the specified plot type
            self._create_plot(
                plot_type=plot_type,
                ax=ax,
                fig=fig,
                iteration=iteration,
                points_3d=points_3d,
                loss_history=loss_history,
                additional_metrics=additional_metrics,
                string_indices=string_indices,
                points_per_string_list=points_per_string_list,
                string_xy=string_xy,
                slice_res=slice_res,
                multi_slice=multi_slice,
                loss_type=loss_type,
                geometry_type=geometry_type, # PASSING geometry_type
                **kwargs
            )
        
        # Hide unused axes
        for i in range(num_plots, num_rows * 3):
            row_idx = i // 3
            col_idx = i % 3
            axes[row_idx, col_idx].axis('off')
        
        fig.tight_layout() # ADDED
        plt.show()
    
    def _create_plot(self, 
                   plot_type: str, 
                   ax: plt.Axes, 
                   fig: plt.Figure, 
                   iteration: int, 
                   points_3d: torch.Tensor, 
                   loss_history: List[float], 
                   additional_metrics: Optional[Dict[str, Any]], 
                   string_indices: Optional[List[int]], 
                   points_per_string_list: Optional[List[int]], 
                   string_xy: Optional[torch.Tensor],
                   slice_res: int, 
                   multi_slice: bool, 
                   loss_type: str,
                   **kwargs) -> None:
        """
        Create a specific type of plot on the given axes.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot to create.
        ax : plt.Axes
            Matplotlib axes to draw on.
        fig : plt.Figure
            Matplotlib figure containing the axes.
        iteration : int
            Current iteration number.
        points_3d : torch.Tensor
            3D points to visualize.
        loss_history : list
            History of loss values.
        additional_metrics : dict or None
            Additional metrics to visualize.
        string_indices : list or None
            String index for each point.
        points_per_string_list : list or None
            Number of points on each string.
        string_xy : torch.Tensor or None
            XY positions of strings.
        slice_res : int
            Resolution for visualization slices.
        multi_slice : bool
            Whether to use multiple slices for visualization.
        loss_type : str
            Type of loss function used.
        kwargs : dict
            Additional keyword arguments for specific plot types.
        """
        # Convert points to numpy for plotting
        points_xyz = points_3d.detach().cpu().numpy()
        geometry_type = kwargs.get('geometry_type', None) # Get geometry_type from kwargs
        
        # Create the requested plot type
        if plot_type == self.PLOT_LOSS:
            # Loss history plot
            ax.plot(loss_history)
            ax.set_title(f"Loss (Iteration {iteration})")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            if loss_type == 'rbf':    
                ax.set_yscale('log')
        
        elif plot_type == self.PLOT_UW_LOSS:
            ax.plot(loss_history)
            ax.set_title(f"(unweighted) Loss (Iteration {iteration})")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            if loss_type == 'rbf':
                ax.set_yscale('log')
            
        elif plot_type == self.PLOT_SNR_HISTORY:
            # SNR history plot
            if additional_metrics and 'snr_history' in additional_metrics:
                snr_history = additional_metrics['snr_history']
                ax.plot(snr_history)
                ax.set_title(f"Signal-to-Noise Ratio")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("SNR")
            else:
                ax.text(0.5, 0.5, "SNR history not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_3D_POINTS:
            # 3D visualization of points
            fig.delaxes(ax)  # Remove the current axis
            ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')
            
            if string_indices is not None:
                # Color by string index for string-based methods
                unique_strings = len(points_per_string_list)
                string_colors = plt.cm.rainbow(np.linspace(0, 1, unique_strings))
                colors = np.array([string_colors[idx] for idx in string_indices])
                
                ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                          c=colors, s=30, alpha=0.8)
                
                if string_xy is not None:
                    # Draw vertical lines for strings
                    xy_np = string_xy.detach().cpu().numpy()
                    for i, (x, y) in enumerate(xy_np):
                        ax.plot([x, x], [y, y], [-self.half_domain, self.half_domain], 
                               color=string_colors[i], alpha=0.3, linestyle='--')
            else:
                # Color by z-coordinate for non-string methods
                ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                          c=points_xyz[:, 2], cmap='rainbow', s=30, alpha=0.8)
            
            title_line1 = f"Optimized Points: {len(points_xyz)} total"
            if geometry_type:
                title_line2 = f"Geometry: {geometry_type}"
                ax.set_title(f"{title_line1}\n{title_line2}")
            else:
                ax.set_title(title_line1)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
            ax.set_zlim(-self.half_domain, self.half_domain)
            # Adjust box aspect ratio
            # ax.set_box_aspect(None, zoom=0.85)
            
            # # For 3D plots in the 3rd column, reduce spacing by allowing overlap with adjacent plots
            # if ax.get_subplotspec().get_position(fig).x0 > 0.6:  # If in 3rd column
            #     # Instead of manually setting position, use figure-level layout adjustments
                
            #     # Ensure this 3D plot is drawn on top of any overlapped content
            #     ax.set_zorder(10)  # Higher zorder means it's drawn on top
                
            #     # Make the background of this plot transparent to see overlapped content
            #     ax.patch.set_alpha(0.0)
                
            #     # Adjust the right side padding of the figure to allow the plot to extend
            #     # This effectively allows the 3D plot to use more horizontal space
            #     right_padding = 0.05  # Reduced right padding
                
            #     # Apply tight layout with custom padding
            #     # The small right padding allows 3D plots to extend further left
            #     fig.subplots_adjust(right=1-right_padding, wspace=0.1)
                
            #     # For the specific 3D plot, we can adjust its own margins
            #     for spine in ax.spines.values():
            #         spine.set_linewidth(0.5)  # Thinner borders
                
            #     # Move axis labels closer to the plot
            #     ax.tick_params(pad=2)
            
        elif plot_type == self.PLOT_STRING_XY:
            # String positions in XY plane
            if string_xy is not None:
                xy_np = string_xy.detach().cpu().numpy()
                
                # Create colormap based on number of points per string
                if points_per_string_list is not None:
                    cmap = plt.cm.viridis
                    norm = Normalize(vmin=min(points_per_string_list), 
                                    vmax=max(points_per_string_list))
                    
                    # Plot strings with size proportional to number of points
                    sc = ax.scatter(
                        [xy_np[s, 0] for s in range(len(xy_np)) if points_per_string_list[s] > 0],
                        [xy_np[s, 1] for s in range(len(xy_np)) if points_per_string_list[s] > 0],
                        s=[20 + 10 * (points_per_string_list[s] / max(points_per_string_list)) 
                          for s in range(len(xy_np)) if points_per_string_list[s] > 0],
                        c=[points_per_string_list[s] for s in range(len(xy_np)) 
                          if points_per_string_list[s] > 0],
                        cmap=cmap,
                        alpha=0.8,
                        norm=norm
                    )
                    
                    # Add a colorbar to show the mapping from color to number of points
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.set_label('Number of points on string')
                else:
                    # Basic scatter plot if no point count information
                    ax.scatter(xy_np[:, 0], xy_np[:, 1], s=30, alpha=0.8)
                
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                ax.set_title('String Positions in XY Plane')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            else:
                ax.text(0.5, 0.5, "String XY data not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_Z_DIST:
            # Z value distribution histogram
            z_values = points_xyz[:, 2]
            ax.hist(z_values, bins=20, color='skyblue', edgecolor='black')
            ax.set_xlabel('Z Position')
            ax.set_ylabel('Count')
            ax.set_title('Z-Value Distribution')
            # Set z limits to match domain for consistency
            ax.set_xlim(-self.half_domain, self.half_domain)
            
        elif plot_type == self.PLOT_XY_PROJECTION:
            # XY projection with points colored by Z
            sc = ax.scatter(points_xyz[:, 0], points_xyz[:, 1], 
                         c=points_xyz[:, 2], cmap='rainbow', alpha=0.8, s=40)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('XY Projection (colored by Z)')
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
            fig.colorbar(sc, ax=ax, label='Z Position')
            
        elif plot_type == self.PLOT_SIGNAL_CONTOUR:
            # Signal function contour plot
            signal_funcs = kwargs.get('signal_funcs', [])
            if signal_funcs:
                # Create a 2D grid in the XY plane at Z=0 for visualization
                resolution = slice_res
                x_grid = torch.linspace(-self.half_domain, self.half_domain, resolution, device=self.device)
                y_grid = torch.linspace(-self.half_domain, self.half_domain, resolution, device=self.device)
                X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
                
                grid_z = 0.0  # Z-slice at z=0
                if multi_slice:
                    # Create multiple slices if multi_slice is True
                    z_slices = np.linspace(-self.half_domain, self.half_domain, resolution)
                    grid_points = []
                    for z in z_slices:
                        grid_points.append(torch.stack([X.flatten(), Y.flatten(), 
                                                       torch.ones_like(X.flatten()) * z], dim=1))
                else:
                    grid_points = torch.stack([X.flatten(), Y.flatten(), 
                                            torch.ones_like(X.flatten()) * grid_z], dim=1)
                
                # Choose first signal function for visualization or use all
                vis_all_signals = kwargs.get('vis_all_signals', False)
                signal_values = np.zeros((resolution, resolution))
                
                if not multi_slice:
                    if not vis_all_signals:
                        signal_func = signal_funcs[np.random.randint(0, len(signal_funcs))]
                        signal_values = signal_func(grid_points).reshape(resolution, resolution).detach().cpu().numpy()
                    else:
                        for i in range(len(signal_funcs)):
                            signal_values += signal_funcs[i](grid_points).reshape(resolution, resolution).detach().cpu().numpy()
                        signal_values /= len(signal_funcs)
                else:
                    if not vis_all_signals:
                        signal_func = signal_funcs[np.random.randint(0, len(signal_funcs))]
                        for i in range(len(z_slices)):
                            signal_values += signal_func(grid_points[i]).reshape(resolution, resolution).detach().cpu().numpy()
                        signal_values /= len(z_slices)
                    else:
                        for signal_func in signal_funcs:
                            for i in range(len(z_slices)):
                                signal_values += signal_func(grid_points[i]).reshape(resolution, resolution).detach().cpu().numpy()
                        signal_values /= len(signal_funcs) * len(z_slices)
                
                # Plot signal function
                c1 = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), signal_values, cmap='viridis', levels=20)
                fig.colorbar(c1, ax=ax)
                
                # Show points near the slice
                ax.scatter(points_xyz[:, 0], points_xyz[:, 1], c='red', s=30, alpha=0.8, edgecolor='black')
                
                if not vis_all_signals:
                    ax.set_title("Sample Signal Function")
                else:   
                    ax.set_title("Combined Signal Function")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                # Set consistent domain boundaries
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
            else:
                ax.text(0.5, 0.5, "Signal function data not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_BACKGROUND_CONTOUR:
            # Background function contour plot
            background_funcs = kwargs.get('background_funcs', [])
            # if background_funcs:
            # Create a 2D grid in the XY plane at Z=0 for visualization
            resolution = slice_res
            x_grid = torch.linspace(-self.half_domain, self.half_domain, resolution, device=self.device)
            y_grid = torch.linspace(-self.half_domain, self.half_domain, resolution, device=self.device)
            X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
            
            grid_z = 0.0  # Z-slice at z=0
            if multi_slice:
                # Create multiple slices if multi_slice is True
                z_slices = np.linspace(-self.half_domain, self.half_domain, resolution)
                grid_points = []
                for z in z_slices:
                    grid_points.append(torch.stack([X.flatten(), Y.flatten(), 
                                                    torch.ones_like(X.flatten()) * z], dim=1))
            else:
                grid_points = torch.stack([X.flatten(), Y.flatten(), 
                                            torch.ones_like(X.flatten()) * grid_z], dim=1)
                
            # Compute combined background
            bkg_values = np.zeros((resolution, resolution))
            no_background = kwargs.get('no_background', False)
            
            if not no_background:
                # Regular background with functions
                if background_funcs:
                    for background_func in background_funcs:
                        if not multi_slice:
                            bkg_values += background_func(grid_points).reshape(resolution, resolution).detach().cpu().numpy()*kwargs.get('background_scale', 1.0)
                        else:
                            temp_bkg_values = np.zeros((resolution, resolution))
                            for i in range(len(z_slices)):
                                temp_bkg_values += background_func(grid_points[i]).reshape(resolution, resolution).detach().cpu().numpy()*kwargs.get('background_scale', 1.0)
                            bkg_values += temp_bkg_values/len(z_slices)
            else:
                # For no_background=True case, fill with constant value matching the SNR loss
                bkg_values.fill(kwargs.get('background_scale', 1.0))  # Matching the constant value in SNR loss
            
            # Plot background (either combined functions or constant)
            plot_title = "Combined Background" if not no_background else "No Background"
            c2 = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), bkg_values, 
                            cmap='plasma', levels=20)
            fig.colorbar(c2, ax=ax)
            
            ax.scatter(points_xyz[:, 0], points_xyz[:, 1], c='red', s=30, alpha=0.8, edgecolor='black')
            
            ax.set_title(plot_title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            # Set consistent domain boundaries
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
        # else:
        #     ax.text(0.5, 0.5, "Background function data not available", 
        #             ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_PARAM_1D:
            # 1D parameter vs SNR plot
            optimize_params = kwargs.get('optimize_params', [])
            param_values = kwargs.get('param_values', {})
            all_snr = kwargs.get('all_snr', None)
            
            if len(optimize_params) == 1 and all_snr is not None:
                param_name = optimize_params[0]
                if param_name in param_values:
                    param_vals = param_values[param_name].detach().cpu().numpy()
                    snr_vals = all_snr.cpu().numpy()
                    
                    # Sort by parameter value
                    sort_idx = np.argsort(param_vals)
                    sorted_param_vals = param_vals[sort_idx]
                    sorted_snr_vals = snr_vals[sort_idx]
                    
                    ax.plot(sorted_param_vals, sorted_snr_vals, 'o-')
                    ax.set_title(f"SNR vs {param_name}")
                    ax.set_xlabel(param_name)
                    ax.set_ylabel("SNR")
                else:
                    ax.text(0.5, 0.5, f"Parameter {param_name} not in parameter values", 
                          ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "1D parameter plot not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_PARAM_2D:
            # 2D parameter space contour plot
            optimize_params = kwargs.get('optimize_params', [])
            param_values = kwargs.get('param_values', {})
            all_snr = kwargs.get('all_snr', None)
            
            if len(optimize_params) == 2 and all_snr is not None:
                param1, param2 = optimize_params
                
                if param1 in param_values and param2 in param_values:
                    # Get parameter values
                    param1_vals = param_values[param1].detach().cpu().numpy()
                    param2_vals = param_values[param2].detach().cpu().numpy()
                    snr_vals = all_snr.detach().cpu().numpy()
                    
                    # Create a grid of unique parameter values
                    param1_unique = np.unique(param1_vals)
                    param2_unique = np.unique(param2_vals)
                    
                    # Reshape data for contour plot
                    P1, P2 = np.meshgrid(param1_unique, param2_unique)
                    SNR_grid = snr_vals
                    
                    # Create the contour plot
                    c3 = ax.contourf(P1, P2, SNR_grid, cmap='viridis', levels=20)
                    fig.colorbar(c3, ax=ax)
                    ax.set_title(f"SNR: {param1} vs {param2}")
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                    
                    # Add contour lines
                    ax.contour(P1, P2, SNR_grid, colors='k', alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f"Parameters {param1} and {param2} not in parameter values", 
                          ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "2D parameter plot not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_STRING_DIST:
            # String distribution bar plot
            string_logits = kwargs.get('string_logits', None)
            
            if string_logits is not None:
                # Get probabilities from logits
                import torch.nn.functional as F
                probs = F.softmax(string_logits, dim=0).detach().cpu().numpy()
                
                # Generate colors
                unique_strings = len(probs)
                string_colors = plt.cm.rainbow(np.linspace(0, 1, unique_strings))
                
                # Plot string probabilities
                ax.bar(range(len(probs)), probs, color=string_colors)
                ax.set_title("String Distribution")
                ax.set_xlabel("String Index")
                ax.set_ylabel("Probability")
                # Set y-axis limit from 0 to slightly above max probability for consistency
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, "String distribution data not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_SURROGATE_FUNCTION:
            surrogate_funcs_input = kwargs.get('surrogate_funcs', [])
            vis_all_surrogates = kwargs.get('vis_all_surrogates', False)
            surrogate_model = kwargs.get('surrogate_model', None)

            # Ensure surrogate_funcs_list is a list
            if not isinstance(surrogate_funcs_input, list):
                surrogate_funcs_list = [surrogate_funcs_input] if surrogate_funcs_input else []
            else:
                surrogate_funcs_list = surrogate_funcs_input

            if not surrogate_funcs_list and not surrogate_model:
                ax.text(0.5, 0.5, "Surrogate data not available", ha='center', va='center', transform=ax.transAxes)
                return

            resolution = slice_res
            x_grid_np = np.linspace(-self.half_domain, self.half_domain, resolution)
            y_grid_np = np.linspace(-self.half_domain, self.half_domain, resolution)
            X_np, Y_np = np.meshgrid(x_grid_np, y_grid_np)
            
            final_values_for_contour = np.zeros((resolution, resolution))
            
            # Helper for evaluation
            def _eval_sfunc(sfunc_obj, points_to_eval):
                if callable(sfunc_obj):
                    return sfunc_obj(points_to_eval).reshape(resolution, resolution).detach().cpu().numpy()
                # Check for __call__ if not directly callable (e.g. for some class instances)
                elif hasattr(sfunc_obj, '__call__'):
                    return sfunc_obj.__call__(points_to_eval).reshape(resolution, resolution).detach().cpu().numpy()
                raise TypeError("Surrogate function object is not callable and has no __call__ method.")

            if multi_slice:
                # Use a modest number of slices for performance, e.g., 5. This could be a parameter.
                z_slices_for_multi = np.linspace(-self.half_domain, self.half_domain, 5) 
                accumulated_slice_values = np.zeros((resolution, resolution))
                num_successful_slices = 0

                for z_val in z_slices_for_multi:
                    grid_points_current_slice = torch.tensor(
                        np.column_stack([X_np.flatten(), Y_np.flatten(), np.ones(X_np.size) * z_val]),
                        device=self.device, dtype=torch.float32
                    )
                    
                    current_slice_sum_val = np.zeros((resolution, resolution))
                    num_funcs_evaluated_on_slice = 0
                    
                    if vis_all_surrogates and surrogate_funcs_list:
                        for s_func_item in surrogate_funcs_list:
                            try:
                                current_slice_sum_val += _eval_sfunc(s_func_item, grid_points_current_slice)
                                num_funcs_evaluated_on_slice += 1
                            except Exception as e:
                                print(f"Warning (multi-slice, all_surrogates): Evaluation failed for a surrogate function on slice z={z_val}: {e}")
                    elif surrogate_funcs_list: # Not vis_all_surrogates, but list is available. Pick one.
                        # Using the first function as the sample. Could be random.
                        s_func_to_use = surrogate_funcs_list[0] 
                        try:
                            current_slice_sum_val = _eval_sfunc(s_func_to_use, grid_points_current_slice)
                            num_funcs_evaluated_on_slice = 1
                        except Exception as e:
                            print(f"Warning (multi-slice, sample_surrogate): Evaluation failed for the sampled surrogate function on slice z={z_val}: {e}")
                            # Fallback to surrogate_model if the sampled function fails for this slice
                            if surrogate_model:
                                print(f"Info (multi-slice, sample_surrogate): Attempting fallback to surrogate_model for slice z={z_val}.")
                                try:
                                    model_s_func = surrogate_model(1, test_points=None)
                                    current_slice_sum_val = _eval_sfunc(model_s_func, grid_points_current_slice)
                                    num_funcs_evaluated_on_slice = 1 # Mark as one successful (fallback) evaluation
                                except Exception as e_model:
                                    print(f"Warning (multi-slice, sample_surrogate): Fallback to surrogate_model also failed for slice z={z_val}: {e_model}")
                    elif surrogate_model: # No surrogate_funcs_list, but surrogate_model is available
                        print(f"Info (multi-slice, model_only): Using surrogate_model for slice z={z_val}.")
                        try:
                            model_s_func = surrogate_model(1, test_points=None)
                            current_slice_sum_val = _eval_sfunc(model_s_func, grid_points_current_slice)
                            num_funcs_evaluated_on_slice = 1
                        except Exception as e_model:
                             print(f"Warning (multi-slice, model_only): surrogate_model evaluation failed for slice z={z_val}: {e_model}")
                    
                    if num_funcs_evaluated_on_slice > 0:
                        accumulated_slice_values += (current_slice_sum_val / num_funcs_evaluated_on_slice)
                        num_successful_slices += 1
                
                if num_successful_slices > 0:
                    final_values_for_contour = accumulated_slice_values / num_successful_slices
                else:
                    ax.text(0.5, 0.5, "Multi-slice surrogate evaluation failed for all slices.", ha='center', va='center', transform=ax.transAxes)
                    return # Cannot plot if all slices failed
            
            else: # Single slice (z=0)
                middle_z = 0.0
                grid_points_single_slice = torch.tensor(
                    np.column_stack([X_np.flatten(), Y_np.flatten(), np.ones(X_np.size) * middle_z]),
                    device=self.device, dtype=torch.float32
                )
                
                single_slice_sum_val = np.zeros((resolution, resolution))
                num_funcs_evaluated_on_single_slice = 0

                if vis_all_surrogates and surrogate_funcs_list:
                    for s_func_item in surrogate_funcs_list:
                        try:
                            single_slice_sum_val += _eval_sfunc(s_func_item, grid_points_single_slice)
                            num_funcs_evaluated_on_single_slice += 1
                        except Exception as e:
                            print(f"Warning (single-slice, all_surrogates): Evaluation failed for a surrogate function: {e}")
                elif surrogate_funcs_list: # Not vis_all_surrogates, but list is available. Pick one.
                    s_func_to_use = surrogate_funcs_list[0] # Using the first function as the sample
                    try:
                        single_slice_sum_val = _eval_sfunc(s_func_to_use, grid_points_single_slice)
                        num_funcs_evaluated_on_single_slice = 1
                    except Exception as e:
                        print(f"Warning (single-slice, sample_surrogate): Evaluation failed for the sampled surrogate function: {e}")
                        if surrogate_model: # Fallback for the single chosen function
                            print(f"Info (single-slice, sample_surrogate): Attempting fallback to surrogate_model.")
                            try:
                                model_s_func = surrogate_model(1, test_points=None)
                                single_slice_sum_val = _eval_sfunc(model_s_func, grid_points_single_slice)
                                num_funcs_evaluated_on_single_slice = 1 # Reset count for successful fallback
                            except Exception as e_model:
                                print(f"Warning (single-slice, sample_surrogate): Fallback to surrogate_model also failed: {e_model}")
                elif surrogate_model: # No surrogate_funcs_list, but surrogate_model is available
                    print(f"Info (single-slice, model_only): Using surrogate_model.")
                    try:
                        model_s_func = surrogate_model(1, test_points=None)
                        single_slice_sum_val = _eval_sfunc(model_s_func, grid_points_single_slice)
                        num_funcs_evaluated_on_single_slice = 1
                    except Exception as e_model:
                        print(f"Warning (single-slice, model_only): surrogate_model evaluation failed: {e_model}")

                if num_funcs_evaluated_on_single_slice > 0:
                    final_values_for_contour = single_slice_sum_val / num_funcs_evaluated_on_single_slice
                else:
                    ax.text(0.5, 0.5, "Single-slice surrogate evaluation failed.", ha='center', va='center', transform=ax.transAxes)
                    return # Cannot plot if evaluation failed

            # Plotting (common for both multi and single slice results)
            c1 = ax.contourf(X_np, Y_np, final_values_for_contour, cmap='viridis', levels=20)
            fig.colorbar(c1, ax=ax)
            
            points_np = points_3d.detach().cpu().numpy()
            title_str = "Surrogate Function"
            if multi_slice:
                title_str += " (Multi-Slice Avg)"
                # For multi-slice, show all points projected to XY plane
                ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=30, alpha=0.8, edgecolor='black')
            else:
                title_str += " (Z=0)"
                # For single-slice, show points near the z=0 slice
                xy_points_z0 = points_np[np.abs(points_np[:, 2] - 0.0) < 0.2] # Check points close to z=0
                if len(xy_points_z0) > 0:
                    ax.scatter(xy_points_z0[:, 0], xy_points_z0[:, 1], c='r', s=30, alpha=0.8, edgecolor='black')
                else: # If no points are near z=0, show all points projected
                    ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=30, alpha=0.8, edgecolor='black')

            # Add detail to title based on what was visualized
            if vis_all_surrogates and len(surrogate_funcs_list) > 1:
                title_str += " (Avg All Provided)"
            elif not vis_all_surrogates and surrogate_funcs_list: # A sample from the list was used
                 title_str += " (Sample)"
            elif surrogate_model and not surrogate_funcs_list : # Model was the primary source
                 title_str += " (Model Generated)"
            # Consider if a fallback to model was used when a list was provided
            # This title logic might need further refinement if very specific sourcing is required.

            ax.set_title(title_str)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            # Set consistent domain boundaries
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
            
        elif plot_type in [self.PLOT_TRUE_FUNCTION, self.PLOT_INTERP_FUNCTION, self.PLOT_ERROR_FUNCTION]:
            # Retrieve necessary parameters from kwargs
            surrogate_model = kwargs.get('surrogate_model', None)
            surrogate_funcs = kwargs.get('surrogate_funcs', []) # Ensure it's a list
            if not isinstance(surrogate_funcs, list): # Handle if a single func is passed
                surrogate_funcs = [surrogate_funcs] if surrogate_funcs else []

            compute_rbf_interpolant = kwargs.get('compute_rbf_interpolant', None)
            test_points = kwargs.get('test_points', None)
            epsilon = kwargs.get('epsilon', 30.0)
            num_funcs_viz = kwargs.get('num_funcs_viz', 10) # Number of functions to average if multiple are available

            # Create visualization grid (common for all slices)
            resolution = slice_res
            x_grid_np = np.linspace(-self.half_domain, self.half_domain, resolution)
            y_grid_np = np.linspace(-self.half_domain, self.half_domain, resolution)
            X_np, Y_np = np.meshgrid(x_grid_np, y_grid_np)

            # Accumulators for the final values to be plotted
            # These will store averages (either over funcs for single slice, or over funcs-then-slices for multi-slice)
            accumulated_true_values = np.zeros((resolution, resolution))
            accumulated_interp_values = np.zeros((resolution, resolution))
            
            # Counter for slices (multi-slice) or sets of functions (single-slice) successfully processed
            processed_items_count = 0

            if multi_slice:
                z_slices_for_multi = np.linspace(-self.half_domain, self.half_domain, resolution)  # e.g., 5 Z-slices

                for z_val in z_slices_for_multi:
                    grid_points_current_slice = torch.tensor(
                        np.column_stack([X_np.flatten(), Y_np.flatten(), np.ones(X_np.size) * z_val]),
                        device=self.device, dtype=torch.float32
                    )

                    # Accumulators for the current slice (averaging over num_funcs_viz)
                    current_slice_true_sum = np.zeros((resolution, resolution))
                    current_slice_interp_sum = np.zeros((resolution, resolution))
                    num_funcs_evaluated_on_this_slice = 0

                    # Determine the list of true function callables for this slice
                    list_of_true_func_callables = []
                    if isinstance(surrogate_funcs, list) and len(surrogate_funcs) > 0:
                        indices = np.random.choice(len(surrogate_funcs), min(num_funcs_viz, len(surrogate_funcs)), replace=False)
                        list_of_true_func_callables = [surrogate_funcs[i] for i in indices]
                    elif surrogate_model:
                        list_of_true_func_callables = [surrogate_model(1, test_points=None) for _ in range(num_funcs_viz)]
                    elif kwargs.get('make_test_funcs'):
                        make_test_funcs = kwargs.get('make_test_funcs')
                        if test_points is not None: # make_test_funcs often requires test_points
                            for _ in range(num_funcs_viz):
                                true_func_from_test, _, _, _ = make_test_funcs(1, test_points=test_points)
                                list_of_true_func_callables.append(true_func_from_test)
                    
                    for true_func_callable in list_of_true_func_callables:
                        try:
                            grid_true_single_func = true_func_callable(grid_points_current_slice).reshape(resolution, resolution).detach().cpu().numpy()
                            current_slice_true_sum += grid_true_single_func

                            if plot_type != self.PLOT_TRUE_FUNCTION and compute_rbf_interpolant:
                                f_values_at_data = true_func_callable(points_3d)
                                # Compute RBF interpolant weights and kernel matrix
                                w, K = compute_rbf_interpolant(
                                    points_3d, f_values_at_data, grid_points_current_slice
                                )
                                # Calculate interpolated values by multiplying the kernel matrix with weights
                                grid_interp_single_func = (K @ w).reshape(resolution, resolution).detach().cpu().numpy()
                                current_slice_interp_sum += grid_interp_single_func
                            
                            num_funcs_evaluated_on_this_slice += 1
                        except Exception as e:
                            print(f"Warning (multi-slice func plot): Eval failed for a func on slice z={z_val:.2f}: {e}")

                    if num_funcs_evaluated_on_this_slice > 0:
                        accumulated_true_values += (current_slice_true_sum / num_funcs_evaluated_on_this_slice)
                        if plot_type != self.PLOT_TRUE_FUNCTION and compute_rbf_interpolant:
                            accumulated_interp_values += (current_slice_interp_sum / num_funcs_evaluated_on_this_slice)
                        processed_items_count += 1
                
                if processed_items_count > 0:
                    # Average over the successfully processed slices
                    final_true_values = accumulated_true_values / processed_items_count
                    if plot_type != self.PLOT_TRUE_FUNCTION and compute_rbf_interpolant:
                        final_interp_values = accumulated_interp_values / processed_items_count
                    else:
                        final_interp_values = None # Ensure it's defined
                else:
                    ax.text(0.5, 0.5, "Multi-slice function evaluation failed for all slices.", ha='center', va='center', transform=ax.transAxes)
                    return
            
            else: # Single slice (z=0.0)
                middle_z = 0.0
                grid_points_single_slice = torch.tensor(
                    np.column_stack([X_np.flatten(), Y_np.flatten(), np.ones(X_np.size) * middle_z]),
                    device=self.device, dtype=torch.float32
                )

                # Determine list of functions to evaluate for this single slice/set
                list_of_true_func_callables = []
                if isinstance(surrogate_funcs, list) and len(surrogate_funcs) > 0:
                    indices = np.random.choice(len(surrogate_funcs), min(num_funcs_viz, len(surrogate_funcs)), replace=False)
                    list_of_true_func_callables = [surrogate_funcs[i] for i in indices]
                elif surrogate_model:
                    list_of_true_func_callables = [surrogate_model(1, test_points=None) for _ in range(num_funcs_viz)]
                elif kwargs.get('make_test_funcs'):
                    make_test_funcs = kwargs.get('make_test_funcs')
                    if test_points is not None:
                        for _ in range(num_funcs_viz):
                            true_func_from_test, _, _, _ = make_test_funcs(1, test_points=test_points)
                            list_of_true_func_callables.append(true_func_from_test)

                if not list_of_true_func_callables:
                    ax.text(0.5, 0.5, "Function visualization data not available (no functions to process).", 
                            ha='center', va='center', transform=ax.transAxes)
                    return

                num_funcs_evaluated_on_single_set = 0
                for true_func_callable in list_of_true_func_callables:
                    try:
                        grid_true_single_func = true_func_callable(grid_points_single_slice).reshape(resolution, resolution).detach().cpu().numpy()
                        accumulated_true_values += grid_true_single_func # Summing directly

                        if plot_type != self.PLOT_TRUE_FUNCTION and compute_rbf_interpolant:
                            f_values_at_data = true_func_callable(points_3d)
                            # Compute RBF interpolant weights and kernel matrix
                            w, K = compute_rbf_interpolant(
                                points_3d, f_values_at_data, grid_points_single_slice
                            )
                            # Calculate interpolated values by multiplying the kernel matrix with weights
                            grid_interp_single_func = (K @ w).reshape(resolution, resolution).detach().cpu().numpy()
                            accumulated_interp_values += grid_interp_single_func # Summing directly
                        
                        num_funcs_evaluated_on_single_set += 1
                    except Exception as e:
                        print(f"Warning (single-slice func plot): Evaluation failed for a function: {e}")
                
                if num_funcs_evaluated_on_single_set > 0:
                    final_true_values = accumulated_true_values / num_funcs_evaluated_on_single_set
                    if plot_type != self.PLOT_TRUE_FUNCTION and compute_rbf_interpolant:
                        final_interp_values = accumulated_interp_values / num_funcs_evaluated_on_single_set
                    else:
                        final_interp_values = None # Ensure it's defined
                    processed_items_count = 1 # Indicate data is ready
                else:
                    ax.text(0.5, 0.5, "Single-slice function evaluation failed for all functions.", ha='center', va='center', transform=ax.transAxes)
                    return

            # Plotting logic, common for both multi-slice and single-slice if processed_items_count > 0
            if processed_items_count == 0: # Should have been caught, but as a safeguard
                 ax.text(0.5, 0.5, "No data to plot after processing.", ha='center', va='center', transform=ax.transAxes)
                 return

            points_np = points_3d.detach().cpu().numpy()
            title_prefix = ""
            title_suffix = ""

            if plot_type == self.PLOT_TRUE_FUNCTION:
                title_prefix = "True Function"
                c_map = 'viridis'
                values_to_plot = final_true_values
            elif plot_type == self.PLOT_INTERP_FUNCTION:
                title_prefix = "Interpolated Function"
                c_map = 'viridis'
                if compute_rbf_interpolant and final_interp_values is not None:
                    values_to_plot = final_interp_values
                else:
                    ax.text(0.5, 0.5, "Interpolation data not available.", ha='center', va='center', transform=ax.transAxes)
                    return
            elif plot_type == self.PLOT_ERROR_FUNCTION:
                title_prefix = "Error Function"
                c_map = 'coolwarm'
                if compute_rbf_interpolant and final_interp_values is not None:
                    values_to_plot = final_true_values - final_interp_values
                else:
                    ax.text(0.5, 0.5, "Error data not available (no interpolation).", ha='center', va='center', transform=ax.transAxes)
                    return
            
            title_suffix = " (Multi-Slice Avg)" if multi_slice else f" (Z={0.0})"
            ax.set_title(f"{title_prefix}{title_suffix}")

            c1 = ax.contourf(X_np, Y_np, values_to_plot, cmap=c_map, levels=20)
            fig.colorbar(c1, ax=ax)

            if multi_slice:
                ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=30, alpha=0.8, edgecolor='black')
            else: # Single slice (Z=0.0)
                xy_points_z0 = points_np[np.abs(points_np[:, 2] - 0.0) < 0.2] # Points near Z=0
                if len(xy_points_z0) > 0:
                    ax.scatter(xy_points_z0[:, 0], xy_points_z0[:, 1], c='r', s=30, alpha=0.8, edgecolor='black')
                else: # If no points near Z=0, show all points projected
                    ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=30, alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
        
        else:
            # Unknown plot type
            ax.text(0.5, 0.5, f"Unknown plot type: {plot_type}", 
                  ha='center', va='center', transform=ax.transAxes)
    
    def visualize_function(self, points_3d, test_points=None, num_funcs_viz=100, 
                         slice_res=50, multi_slice=False, make_test_funcs=None,
                         compute_rbf_interpolant=None, epsilon=30.0,
                         plot_types=None, surrogate_model=None, surrogate_funcs=None):
        """
        Visualize function interpolation quality with customizable plot selection.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            3D points to visualize.
        test_points : torch.Tensor or None
            Test points for visualization.
        num_funcs_viz : int
            Number of functions to average for visualization.
        slice_res : int
            Resolution for visualization slices.
        multi_slice : bool
            Whether to use multiple slices for visualization.
        make_test_funcs : callable or None
            Function to generate test functions.
        compute_rbf_interpolant : callable or None
            Function to compute RBF interpolant.
        epsilon : float
            RBF kernel parameter.
        plot_types : list of str or None
            List of plot types to display. If None, displays default function visualization plots.
            Suggested values for function visualization:
            - 'true_function': True function contour
            - 'interp_function': Interpolated function contour
            - 'error_function': Error function contour
            - 'surrogate_function': Surrogate function contour
        surrogate_model : object or None
            The surrogate model to use for generating functions. 
        surrogate_funcs : list or callable or None
            Pre-generated surrogate functions to visualize.
        """
        # Set default plot types if not specified
        if plot_types is None:
            if surrogate_model is not None or surrogate_funcs is not None:
                plot_types = [self.PLOT_SURROGATE_FUNCTION, self.PLOT_INTERP_FUNCTION, self.PLOT_ERROR_FUNCTION]
            else:
                plot_types = [self.PLOT_TRUE_FUNCTION, self.PLOT_INTERP_FUNCTION, self.PLOT_ERROR_FUNCTION]
        
        # Create kwargs dict for the visualization
        kwargs = {
            'make_test_funcs': make_test_funcs,
            'compute_rbf_interpolant': compute_rbf_interpolant,
            'test_points': test_points,
            'epsilon': epsilon,
            'num_funcs_viz': num_funcs_viz,
            'surrogate_model': surrogate_model,
            'surrogate_funcs': surrogate_funcs
        }
        
        # Use the general visualization function
        self.visualize_progress(
            iteration=0,  # Not relevant for function visualization
            points_3d=points_3d,
            loss_history=[],  # Not used for function visualization
            slice_res=slice_res,
            multi_slice=multi_slice,
            loss_type='surrogate' if surrogate_model is not None or surrogate_funcs is not None else 'rbf',
            plot_types=plot_types,
            **kwargs
        )
    
    def create_interactive_3d_plot(self, points_3d, string_indices=None, 
                                 points_per_string_list=None, string_xy=None):
        """
        Create an interactive 3D plot with Plotly.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            3D points to visualize.
        string_indices : list or None
            String index for each point.
        points_per_string_list : list or None
            Number of points on each string.
        string_xy : torch.Tensor or None
            XY positions of strings.
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
            Interactive 3D plot if Plotly is available, otherwise None.
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly is required for interactive 3D plotting.")
            print("Install with: pip install plotly")
            return None
        
        # Convert to numpy for plotting
        if torch.is_tensor(points_3d):
            points_np = points_3d.detach().cpu().numpy()
        else:
            points_np = points_3d
            
        # Create figure
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]],
            subplot_titles=["Interactive 3D Visualization"]
        )
        
        if string_indices is not None and points_per_string_list is not None:
            # Color by string for string-based methods
            n_strings = len(points_per_string_list)
            
            # Generate colors using matplotlib's colormap
            import matplotlib.cm as cm
            colormap = cm.rainbow(np.linspace(0, 1, n_strings))
            colormap_hex = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                          for r, g, b, _ in colormap]
            
            # Add traces for each string
            for s in range(n_strings):
                # Skip empty strings
                if points_per_string_list[s] == 0:
                    continue
                    
                # Get points for this string
                mask = np.array(string_indices) == s
                string_points = points_np[mask]
                
                # Add vertical line for string if string_xy is provided
                if string_xy is not None:
                    x_pos, y_pos = string_xy[s].detach().cpu().numpy()
                    
                    # Add a vertical line for the string
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x_pos, x_pos],
                            y=[y_pos, y_pos],
                            z=[-self.half_domain, self.half_domain],
                            mode='lines',
                            line=dict(
                                color='rgba(0,0,0,0.2)',  # Black with alpha 0.2
                                width=2,
                            ),
                            showlegend=False
                        )
                    )
                    
                    # Create hover text for string positions
                    hovertext = [f"String {s+1}: {points_per_string_list[s]} points"]
                    
                    # Add string position markers
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x_pos],
                            y=[y_pos],
                            z=[-self.half_domain],  # Place at bottom of domain
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=colormap_hex[s],
                                symbol='diamond',
                                opacity=0.8
                            ),
                            name=f'String {s} ({points_per_string_list[s]} pts)',
                            text=hovertext,
                            hoverinfo='text'
                        )
                    )
                
                # Add points with same colors as in the visualization
                fig.add_trace(
                    go.Scatter3d(
                        x=string_points[:, 0],
                        y=string_points[:, 1],
                        z=string_points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=colormap_hex[s],
                            opacity=0.9,
                            symbol='circle'
                        ),
                        name=f'String {s} ({points_per_string_list[s]} pts)'
                    )
                )
        else:
            # Single color for all points if not using strings
            fig.add_trace(
                go.Scatter3d(
                    x=points_np[:, 0],
                    y=points_np[:, 1],
                    z=points_np[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=points_np[:, 2],  # Color by z-coordinate
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name='Optimized Points'
                )
            )
        
        # Update layout for better 3D visualization
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis=dict(range=[-self.half_domain*1.05, self.half_domain*1.05]),
                yaxis=dict(range=[-self.half_domain*1.05, self.half_domain*1.05]),
                zaxis=dict(range=[-self.half_domain*1.05, self.half_domain*1.05])
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            title=dict(
                text=f"Interactive 3D Visualization of {len(points_np)} Points on {len(points_per_string_list) if points_per_string_list else 'N/A'} Strings",
                font=dict(size=16)
            ),
            height=700
        )
        
        return fig