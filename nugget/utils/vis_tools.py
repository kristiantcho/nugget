import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from IPython.display import clear_output, display
import math
import re # Added for regex pattern matching in GIF frame sorting
from typing import List, Dict, Union, Tuple, Optional, Any, Callable
import io # Added for GIF generation
import imageio # Added for GIF generation
from scipy.interpolate import griddata
import os # Added for file management
import tempfile # Added for temporary directory management
import glob # Added for file pattern matching
import shutil # Added for directory operations

# Try importing plotly for interactive 3D plotting, but don't fail if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Visualizer:
    """Base class for visualization tools in geometry optimization."""
    
    @staticmethod
    def _safe_tensor_convert(tensor_input, allow_none=True):
        """
        Safely convert torch tensors by cloning and detaching them.
        Other data types are returned unchanged.
        
        Parameters:
        -----------
        tensor_input : Any
            Input that might be a torch tensor, list of tensors, or other data type.
        allow_none : bool
            Whether to allow None values to pass through.
            
        Returns:
        --------
        Any
            For torch.Tensor: cloned and detached tensor
            For list of tensors: list of cloned and detached tensors
            For other types: unchanged input
        """
        if tensor_input is None and allow_none:
            return None
        if torch.is_tensor(tensor_input):
            return tensor_input.clone().detach()
        elif isinstance(tensor_input, list):
            # Handle lists that might contain tensors
            return [Visualizer._safe_tensor_convert(item, allow_none) for item in tensor_input]
        return tensor_input
    
    # Define plot types as class constants
    PLOT_LOSS = "loss"
    PLOT_UW_LOSS = "uw_loss"
    PLOT_SNR_HISTORY = "snr_history"
    PLOT_LLR_HISTORY = "llr_history"
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
    PLOT_STRING_WEIGHTS_SCATTER = "string_weights_scatter"
    PLOT_LLR_CONTOUR = "llr_contour"
    PLOT_SIGNAL_LLR_CONTOUR = "signal_llr_contour"
    PLOT_BACKGROUND_LLR_CONTOUR = "background_llr_contour"
    PLOT_SIGNAL_LLR_CONTOUR_POINTS = "signal_llr_contour_points"
    PLOT_BACKGROUND_LLR_CONTOUR_POINTS = "background_llr_contour_points"
    PLOT_LLR_HISTOGRAM = "llr_histogram"
    PLOT_SNR_CONTOUR = "snr_contour"
    PLOT_TRUE_SIGNAL_LLR_CONTOUR = "true_signal_llr_contour"
    PLOT_TRUE_BACKGROUND_LLR_CONTOUR = "true_background_llr_contour"
    PLOT_SIGNAL_LIGHT_YIELD_CONTOUR = "signal_light_yield_contour"
    PLOT_SIGNAL_LIGHT_YIELD_CONTOUR_POINTS = "signal_light_yield_contour_points"
    PLOT_FISHER_INFO_CONTOUR = "fisher_info_contour"
    PLOT_ANGULAR_RESOLUTION = "angular_resolution"
    PLOT_ENERGY_RESOLUTION = "energy_resolution"
    PLOT_ANGULAR_RESOLUTION_HISTORY = "angular_resolution_history"
    PLOT_ENERGY_RESOLUTION_HISTORY = "energy_resolution_history"
    PLOT_LOSS_COMPONENTS = "loss_components"
    PLOT_UW_LOSS_COMPONENTS = "uw_loss_components"
    PLOT_LLR_HISTOGRAM_POINTS = "llr_histogram_points"

    
    def __init__(self, device=None, dim=3, domain_size=2.0, gif_temp_dir=None):
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
        self.gif_temp_dir = gif_temp_dir# Temporary directory for storing individual images
        self.gif_image_paths = [] # List to track saved image paths
    
    def _standardize_axis_formatting(self, ax, max_ticks=5, label_precision=2, fontsize=8):
        """
        Standardize axis formatting for consistent GIF frame sizing.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to format
        max_ticks : int
            Maximum number of ticks on each axis
        label_precision : int
            Number of decimal places for tick labels
        fontsize : int
            Font size for tick labels
        """
        if hasattr(ax, 'xaxis') and hasattr(ax, 'yaxis'):
            # Limit number of ticks to prevent overcrowding
            # ax.locator_params(axis='x', nbins=max_ticks)
            # ax.locator_params(axis='y', nbins=max_ticks)
            
            # Format tick labels to consistent precision only if not log scaled
            if ax.get_xscale() != 'log':
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.{label_precision}f}'))
            if ax.get_yscale() != 'log':
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.{label_precision}f}'))
            
            # Set consistent tick label size
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            
            # Ensure tick labels don't extend beyond plot area
            ax.tick_params(axis='x', rotation=0, pad=2)
            ax.tick_params(axis='y', rotation=0, pad=2)
    
    def _draw_rov_safe_space(self, ax, rov_penalty=None, position='bottom_left', scale_factor=1):
        """
        Draw ROV safe space shape on the given axes.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to draw on
        rov_penalty : ROVPenalty object or None
            ROV penalty object to get dimensions from
        position : str
            Where to place the ROV shape ('bottom_left', 'bottom_right', etc.)
        scale_factor : float
            Scale factor for the ROV shape relative to plot domain
        """
        if rov_penalty is None:
            return
            
        # Get ROV dimensions
        rov_rec_width = getattr(rov_penalty, 'rov_rec_width', 0.3)
        rov_height = getattr(rov_penalty, 'rov_height', 0.16) 
        rov_tri_length = getattr(rov_penalty, 'rov_tri_length', 0.08)
        
        # Scale dimensions to fit in corner of plot
        scale = scale_factor #* self.domain_size
        rec_width = rov_rec_width * scale
        rec_height = rov_height * scale
        tri_length = rov_tri_length * scale
        
        # Position in bottom left corner
        if position == 'bottom_left':
            x_offset = -self.half_domain + 0.05 * self.domain_size
            y_offset = -self.half_domain + 0.05 * self.domain_size
        elif position == 'bottom_right':
            x_offset = self.half_domain - (rec_width + tri_length) - 0.05 * self.domain_size
            y_offset = -self.half_domain + 0.05 * self.domain_size
        else:  # default to bottom_left
            x_offset = -self.half_domain + 0.05 * self.domain_size
            y_offset = -self.half_domain + 0.05 * self.domain_size
        
        # Draw rectangular part
        rect_x = [x_offset, x_offset + rec_width, x_offset + rec_width, x_offset, x_offset]
        rect_y = [y_offset - rec_height/2, y_offset - rec_height/2, y_offset + rec_height/2, y_offset + rec_height/2, y_offset - rec_height/2]
        ax.plot(rect_x, rect_y, 'r-', linewidth=2, alpha=0.7, label='ROV Safe Space')
        
        # Draw triangular part
        tri_x = [x_offset + rec_width, x_offset + rec_width + tri_length, x_offset + rec_width, x_offset + rec_width]
        tri_y = [y_offset - rec_height/2, y_offset, y_offset + rec_height/2, y_offset - rec_height/2]
        ax.plot(tri_x, tri_y, 'r-', linewidth=2, alpha=0.7)
        
        # Fill the shape with semi-transparent red
        # Combine rectangle and triangle vertices for filling
        fill_x = [x_offset, x_offset + rec_width, x_offset + rec_width + tri_length, x_offset + rec_width, x_offset]
        fill_y = [y_offset - rec_height/2, y_offset - rec_height/2, y_offset, y_offset + rec_height/2, y_offset + rec_height/2]
        ax.fill(fill_x, fill_y, 'red', alpha=0.2)
        
        # Add "ROV" text inside the rectangular part of the safe space
        text_x = x_offset + rec_width/2  # Center of rectangle
        text_y = y_offset  # Center vertically
        ax.text(text_x, text_y, 'ROV', fontsize=21*rec_width, fontweight='bold', 
                ha='center', va='center', color='darkred', alpha=0.8)

    def _safe_griddata_interpolation(self, points_xy, values, grid_points, resolution, method='linear', fill_value=None):
        """
        Safely perform griddata interpolation with proper error handling.
        
        Parameters:
        -----------
        points_xy : array-like
            2D array of point coordinates (N, 2)
        values : array-like
            Values at each point (N,)
        grid_points : array-like
            Grid points for interpolation (M, 2)
        resolution : int
            Grid resolution for reshaping
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')
        fill_value : float or None
            Value to use for points outside the convex hull
            
        Returns:
        --------
        tuple : (success, grid_values, error_message)
            success : bool - whether interpolation succeeded
            grid_values : ndarray or None - interpolated values reshaped to (resolution, resolution)
            error_message : str or None - error message if failed
        """
        # Safely handle torch tensor inputs by cloning and detaching them
        points_xy = self._safe_tensor_convert(points_xy, allow_none=False)
        values = self._safe_tensor_convert(values, allow_none=False)  
        grid_points = self._safe_tensor_convert(grid_points, allow_none=False)
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(points_xy):
            points_xy = points_xy.detach().cpu().numpy()
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        if torch.is_tensor(grid_points):
            grid_points = grid_points.detach().cpu().numpy()
        
        # Handle finite values
        finite_mask = np.isfinite(values)
        num_finite = np.sum(finite_mask)
        
        if num_finite < 3:
            return False, None, f"Too few finite values ({num_finite}) for triangulation (need ≥3)"
        
        # Extract finite data
        finite_points = points_xy[finite_mask]
        finite_values = values[finite_mask]
        
        # Set fill value if not provided
        if fill_value is None:
            fill_value = np.min(finite_values)
        
        try:
            # Perform interpolation
            interpolated = griddata(
                finite_points, 
                finite_values, 
                grid_points,
                method=method, 
                fill_value=fill_value
            )
            
            # Reshape to grid
            grid_values = interpolated.reshape(resolution, resolution)
            
            return True, grid_values, None
            
        except Exception as e:
            return False, None, str(e)
    
    def visualize_progress(self, 
                          iteration: int = None, 
                          points: torch.Tensor=None, 
                          loss_history: List[float]=None, 
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
                          save_individual_images: bool = True, # New: Save images to disk instead of memory
                          compile_gif_on_iteration: bool = False, # New: Whether to compile GIF on each iteration
                          gif_fixed_figsize: Optional[tuple] = None, # New: Fixed figure size for consistent GIFs
                          gif_fixed_rows: int = 4, # New: Fixed number of rows for consistent layout
                          gif_standardize_ticks: bool = True, # New: Whether to standardize tick formatting
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
            Type of loss function used ('rbf', 'snr', 'surrogate', or 'llr').
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
            - 'string_weights_scatter': String weights scatter plot with variable alpha
            - 'llr_contour': Combined LLR contour plot based on per-string values
            - 'signal_llr_contour': Signal-only LLR contour plot
            - 'background_llr_contour': Background-only LLR contour plot
            - 'signal_llr_contour_points': Signal-only LLR contour plot based on per-point values
            - 'background_llr_contour_points': Background-only LLR contour plot based on per-point values
            - 'llr_histogram': LLR density histogram comparing signal and background distributions
            - 'llr_histogram_points': LLR density histogram comparing signal and background distributions per point
            - 'snr_contour': SNR contour plot based on per-string values
            - 'signal_light_yield_contour': Signal light yield contour plot based on per-string values
            - 'signal_light_yield_contour_points': Signal light yield contour plot based on per-point values
            - 'fisher_info_logdet': Log determinant of Fisher Information matrix contour plot
            - 'angular_resolution': Angular resolution from Fisher Information using Cramér-Rao bound
            - 'energy_resolution': Energy resolution from Fisher Information using Cramér-Rao bound
            - 'loss_components': Individual loss components and total loss from loss dictionary
            - 'uw_loss_components': Individual unweighted loss components and total unweighted loss
        make_gif : bool
            Whether to generate and save a GIF of the progress.
        gif_plot_selection : list of str or None
            List of plot types to display in each GIF frame. If None, uses a default set.
            Uses the same plot type strings as 'plot_types'.
        gif_filename : str
            Filename for the generated GIF.
        gif_fps : int
            Frames per second for the generated GIF.
        save_individual_images : bool
            If True, save individual images to disk instead of storing frames in memory.
            This is more memory efficient and allows for better GIF management.
        compile_gif_on_iteration : bool
            If True, compile/update the GIF on each iteration. If False, only save images
            and require manual compilation via finalize_gif().
        gif_fixed_figsize : tuple or None
            Fixed figure size (width, height) for GIF frames to ensure consistent sizing.
            If None, defaults to (15, 12) for consistent 3x4 layout regardless of plot count.
        gif_fixed_rows : int
            Fixed number of rows for GIF layout to ensure consistent sizing.
            Defaults to 4 rows for a 3x4 grid layout.
        gif_standardize_ticks : bool
            Whether to standardize tick formatting across all plots for consistent sizing.
            Helps prevent layout shifts due to varying tick label lengths.
        geometry_type : str, optional
            The type of geometry being used.
        kwargs : dict
            Additional keyword arguments for specific loss types.
            For surrogate visualization:
            - surrogate_funcs: List of surrogate functions
            - surrogate_model: The surrogate model instance
            - compute_rbf_interpolant: Function to compute RBF interpolant
            For signal/background contour plots:
            - signal_funcs: List of signal functions (old format)
            - background_funcs: List of background functions (old format)
            - signal_surrogate_func: Surrogate function for signal (e.g., light_yield_surrogate method)
            - signal_event_params: Event parameters dict for signal surrogate function
            - background_surrogate_func: Surrogate function for background
            - background_event_params: Event parameters dict for background surrogate function
            - rov_penalty: ROVPenalty object for displaying ROV safe space on string_xy and string_weights_scatter plots
        """
        # Safely handle torch tensor inputs by cloning and detaching them
        points = self._safe_tensor_convert(points)
        string_xy = self._safe_tensor_convert(string_xy)
        
        
        # Handle potential torch tensors in kwargs
        for key in ['test_points', 'string_weights', 'signal_funcs', 'background_funcs']:
            if key in kwargs:
                kwargs[key] = self._safe_tensor_convert(kwargs[key])
        
        
        # Clear previous output
        # clear_output(wait=True)

        # GIF Generation Logic
        if make_gif:
            # Initialize temporary directory for saving images if needed
            if save_individual_images and self.gif_temp_dir is None:
                self.gif_temp_dir = tempfile.mkdtemp(prefix="gif_frames_")
                print(f"Created temporary directory for GIF frames: {self.gif_temp_dir}")
            
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
                
                gif_fig_size = (12, 3.5 * num_gif_rows) 

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
                        points=points,
                        loss_history=loss_history,
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
                
                # Apply consistent formatting to all axes to handle tick label length variations
                if gif_standardize_ticks:
                    for ax in axes_gif_flat_for_loop:
                        self._standardize_axis_formatting(ax)
                
                # Use constrained layout instead of tight_layout for more consistent spacing
                fig_gif.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, 
                                      wspace=0.3, hspace=0.4)
                
                if save_individual_images:
                    # Save individual image to disk
                    add_on = 0
                    if self.gif_temp_dir is not None:
                        # Ensure the temporary directory exists
                        if not os.path.exists(self.gif_temp_dir):
                            os.makedirs(self.gif_temp_dir)
                        # check if there are existing images
                        if os.listdir(self.gif_temp_dir):
                            existing_files = [f for f in os.listdir(self.gif_temp_dir) if f.startswith("frame_") and f.endswith(".png")]
                            if existing_files:
                                # Extract numbers from existing filenames and find the highest
                                numbers = []
                                for f in existing_files:
                                    try:
                                        num = int(f.split("_")[1].split(".")[0])
                                        numbers.append(num)
                                    except (ValueError, IndexError):
                                        continue
                                add_on = max(numbers) + 1 if numbers else 0
                            else:
                                add_on = 0
                    image_filename = f"frame_{iteration+add_on:04d}.png"
                    image_path = os.path.join(self.gif_temp_dir, image_filename)
                    # Use consistent save parameters for identical image sizes
                    fig_gif.savefig(image_path, format='png', dpi=100, bbox_inches=None, 
                                   facecolor='white', edgecolor='none', pad_inches=0.1)
                    self.gif_image_paths.append(image_path)
                    print(f"Saved GIF frame {len(self.gif_image_paths)} to {image_path}")
                    
                    # Compile GIF if requested on each iteration
                    if compile_gif_on_iteration:
                        self._compile_gif_from_images(gif_filename, gif_fps)
                else:
                    # Original method: store frames in memory
                    img_buf = io.BytesIO()
                    # Use consistent save parameters for identical image sizes
                    fig_gif.savefig(img_buf, format='png', dpi=100, bbox_inches=None,
                                   facecolor='white', edgecolor='none', pad_inches=0.1)
                    img_buf.seek(0)
                    self.gif_frames.append(imageio.v3.imread(img_buf))
                    img_buf.close()
                    
                    # Compile GIF from memory frames
                    if self.gif_frames and compile_gif_on_iteration:
                        try:
                            imageio.mimsave(gif_filename, self.gif_frames, fps=gif_fps)
                            print(f"GIF '{gif_filename}' updated with {len(self.gif_frames)} frames (Iteration {iteration}).")
                        except Exception as e:
                            print(f"Error saving GIF: {e}")
                
                plt.close(fig_gif)
                        
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
            elif loss_type == 'llr':
                plot_types = [
                    self.PLOT_LOSS,
                    self.PLOT_LLR_HISTORY if kwargs.get('llr_history') is not None else self.PLOT_LOSS,
                    self.PLOT_3D_POINTS,
                    self.PLOT_STRING_XY if string_xy is not None else self.PLOT_XY_PROJECTION,
                    self.PLOT_LLR_CONTOUR,
                    self.PLOT_SIGNAL_LLR_CONTOUR,
                    self.PLOT_BACKGROUND_LLR_CONTOUR,
                    self.PLOT_LLR_HISTOGRAM
                ]
        
        # Create figure with proper layout based on number of plots
        num_plots = len(plot_types)
        num_rows = (num_plots + 2) // 3  # Ceiling division to get number of rows needed
        if num_plots < 3:
            ncols = num_plots
        else:
            ncols = 3
        fig, axes = plt.subplots(num_rows, ncols, figsize=(5 * ncols, 4.5 * num_rows)) # MODIFIED

        # If only one row, ensure axes is still a 2D array
        if num_rows == 1 and ncols > 1:
            axes = axes.reshape(1, -1)
        
        # Generate each requested plot
        for i, plot_type in enumerate(plot_types):
            if len(plot_types) > 1:
                row_idx = i // ncols
                col_idx = i % ncols
                ax = axes[row_idx, col_idx]
            else:
                ax = axes

            # Create the specified plot type
            self._create_plot(
                plot_type=plot_type,
                ax=ax,
                fig=fig,
                iteration=iteration,
                points=points,
                loss_history=loss_history,
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
        if num_plots > 1:
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
        # Safely handle torch tensor inputs by cloning and detaching them
        points = self._safe_tensor_convert(points_3d)
        string_xy = self._safe_tensor_convert(string_xy)
        if kwargs.get('string_weights') is not None:    
            kwargs['string_weights'] = torch.sigmoid(kwargs['string_weights'].clone())
        # Handle potential torch tensors in common kwargs
        tensor_kwargs = ['string_weights', 'signal_funcs', 'background_funcs', 'test_points', 
                        'llr_per_string', 'signal_llr_per_string', 'background_llr_per_string',
                        'signal_yield_per_string', 'snr_per_string', 'fisher_info_per_string']
        for key in tensor_kwargs:
            if key in kwargs and kwargs.get(key) is not None:
                kwargs[key] = self._safe_tensor_convert(kwargs[key])
        
        # Convert points to numpy for plotting
        points_xyz = points.detach().cpu().numpy()
        geometry_type = kwargs.get('geometry_type', None) # Get geometry_type from kwargs
        
        # Create the requested plot type
        if plot_type == self.PLOT_LOSS:
            # Loss history plot
            ax.plot(loss_history)
            ax.set_title(f"Loss (Iteration {iteration})")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            if loss_type == 'rbf' or np.all(np.array(loss_history) > 0):    
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
            snr_history = kwargs.get('snr_history', None)
            if snr_history is not None:
                ax.plot(snr_history)
                ax.set_title(f"Total Signal-to-Noise Ratio")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("SNR")
            else:
                ax.text(0.5, 0.5, "SNR history not available", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_LLR_HISTORY:
            # LLR history plot
            llr_history = kwargs.get('llr_history', None)
            if llr_history is not None:
                llr_history = np.array(llr_history)/len(points)
                ax.plot(llr_history)
                ax.set_title(f"Mean Log-Likelihood Ratio")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("LLR")
            else:
                ax.text(0.5, 0.5, "LLR history not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_3D_POINTS:
            # 3D visualization of points
            fig.delaxes(ax)  # Remove the current axis
            ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')
            
            # Get string weights for alpha transparency
            string_weights = kwargs.get('string_weights', None)
            
            if string_indices is not None:
                # print("string_indices:", string_indices)
                # Color by string index for string-based methods
                unique_strings = np.unique(string_indices)
                string_colors = plt.cm.rainbow(np.linspace(0, 1, unique_strings.size))
                # Map each point to its string's color
                colors = np.array([string_colors[idx] for idx in unique_strings])
                
                # Calculate alpha values based on string weights
                if string_weights is not None:
                    
                    # Convert string weights to point-wise alpha values
                    alpha_values = np.array([string_weights[idx] for idx in unique_strings])
                    # Apply sigmoid to convert to [0,1] range if not already
                    # alpha_values = 1 / (1 + np.exp(-alpha_values)) if np.any(alpha_values < 0) or np.any(alpha_values > 1) else alpha_values
                    # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                    # Ensure minimum visibility
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                    
                full_colors = []
                full_alphas = []
                # print("Points per string list:", points_per_string_list)
                for string_num, num_points in enumerate(points_per_string_list):
                    full_colors.extend([colors[string_num]] * num_points)
                    if string_weights is not None:
                        full_alphas.extend([alpha_values[string_num]] * num_points)
                    
                
                ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                          c=full_colors, s=min([30,30*200/len(points_per_string_list)]), alpha=full_alphas if string_weights is not None else 0.8)
                
                if string_xy is not None:
                    # Draw vertical lines for strings with alpha based on string weights
                    xy_np = string_xy.detach().cpu().numpy()
                    for i, (x, y) in enumerate(xy_np):
                        line_alpha = string_weights[i] if string_weights is not None else 0.3
                        # Apply sigmoid if needed
                        if string_weights is not None:
                            line_alpha = np.clip(line_alpha, 0.1, 1.0)  # Ensure minimum visibility
                            # line_alpha = 1 / (1 + np.exp(-line_alpha)) if line_alpha < 0 or line_alpha > 1 else line_alpha
                            # line_alpha = max(0.1, line_alpha)  # Minimum visibility
                        ax.plot([x, x], [y, y], [-self.half_domain, self.half_domain], 
                               color=string_colors[i], alpha=line_alpha, linestyle='--')
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
                string_weights = kwargs.get('string_weights', None)
                weight_threshold = kwargs.get('weight_threshold', 0.7)
                
                # Create colormap based on number of points per string
                if points_per_string_list is not None:
                    cmap = plt.cm.viridis
                    norm = Normalize(vmin=min(points_per_string_list), 
                                    vmax=max(points_per_string_list))
                    
                    # Calculate alpha values based on string weights
                    if string_weights is not None:
                        # Apply sigmoid to convert to [0,1] range if not already
                        # print("String weights:", string_weights)
                        alpha_vals = np.array([string_weights[idx] for idx in string_indices])
                        # if np.any(alpha_vals < 0) or np.any(alpha_vals > 1):
                        #     alpha_vals = 1 / (1 + np.exp(-alpha_vals))
                        # alpha_vals = torch.nn.functional.softplus(torch.tensor(alpha_vals)).detach().cpu().numpy()  # Apply softplus for smoothness
                        # Ensure minimum visibility and filter active strings
                        alpha_vals = np.clip(alpha_vals, 0.05, 1.0)
                        # print("Alpha values:", alpha_vals)
                        # active_mask = np.array(points_per_string_list) > 0
                        # alpha_vals = alpha_vals[active_mask] if len(alpha_vals) == len(points_per_string_list) else [0.8] * sum(active_mask)
                        weight_mask = np.array([string_weights[idx] >= weight_threshold for idx in string_indices])
                    else:
                        alpha_vals = 0.8
                        weight_mask = np.array([True]*len(xy_np))
                    
                    
                    
                    # Plot strings with size proportional to number of points and alpha based on weights
                    if np.any(weight_mask):    
                        sc = ax.scatter(
                            np.array([xy_np[s, 0] for s in range(len(xy_np)) if points_per_string_list[s] > 0])[weight_mask],
                            np.array([xy_np[s, 1] for s in range(len(xy_np)) if points_per_string_list[s] > 0])[weight_mask],
                            s=[min([40, 30 * 200 / len(xy_np[weight_mask])]) 
                            for s in range(len(xy_np[weight_mask])) if np.array(points_per_string_list)[weight_mask][s] > 0],
                            c=[np.array(points_per_string_list)[weight_mask][s] for s in range(len(xy_np[weight_mask])) 
                            if np.array(points_per_string_list)[weight_mask][s] > 0],
                            cmap=cmap,
                            alpha=alpha_vals,
                            norm=norm
                        )
                    
                        # Add a colorbar to show the mapping from color to number of points
                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label('Number of points on string')
                else:
                    # Basic scatter plot with alpha based on string weights
                    if string_weights is not None:
                        alpha_vals = np.array([string_weights[idx] for idx in string_indices])
                        # if np.any(alpha_vals < 0) or np.any(alpha_vals > 1):
                        #     alpha_vals = 1 / (1 + np.exp(-alpha_vals))
                        # alpha_vals = torch.nn.functional.softplus(torch.tensor(alpha_vals)).detach().cpu().numpy()
                        alpha_vals = np.clip(alpha_vals, 0.05, 1.0)
                        weight_mask = np.array([string_weights[idx] >= weight_threshold for idx in string_indices])
                    else:
                        alpha_vals = 0.8
                        weight_mask = np.array([True]*len(xy_np))

                    if np.any(weight_mask):    
                        ax.scatter(xy_np[:, 0][weight_mask], xy_np[:, 1][weight_mask], s=min([40,30*200/len(xy_np[weight_mask])]), alpha=alpha_vals[weight_mask])

                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                ax.set_title('String Positions in XY Plane')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                
                # Add ROV safe space visualization if ROV penalty is available

                rov_penalty = kwargs.get('rov_penalty', None)
                if rov_penalty is not None:
                    self._draw_rov_safe_space(ax, rov_penalty)
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
            string_weights = kwargs.get('string_weights', None)
            
            # Calculate alpha values based on string weights
            if string_weights is not None and string_indices is not None:
                alpha_values = np.array([string_weights[idx] for idx in string_indices])
                # Apply sigmoid to convert to [0,1] range if not already
                # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                    # alpha_values = 1 / (1 + np.exp(-alpha_values))
                # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                # Ensure minimum visibility
                alpha_values = np.clip(alpha_values, 0.05, 1.0)
            else:
                alpha_values = 0.8
            
            sc = ax.scatter(points_xyz[:, 0], points_xyz[:, 1], 
                         c=points_xyz[:, 2], cmap='rainbow', alpha=alpha_values, s=40)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('XY Projection (colored by Z)')
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
            fig.colorbar(sc, ax=ax, label='Z Position')
            
        elif plot_type == self.PLOT_SIGNAL_CONTOUR:
            # Signal function contour plot
            signal_funcs = kwargs.get('signal_funcs', [])
            signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
            signal_event_params = kwargs.get('signal_event_params', None)
            
            # Check if we have either the old format or new surrogate format
            if signal_funcs or (signal_surrogate_func is not None and signal_event_params is not None):
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
                
                signal_values = np.zeros((resolution, resolution))
                
                # Handle new surrogate function format
                if signal_surrogate_func is not None and signal_event_params is not None:
                    if not multi_slice:
                        # Evaluate surrogate function at each grid point
                        grid_values = []
                        for point in grid_points:
                            value = signal_surrogate_func(opt_point=point, event_params=signal_event_params)
                            grid_values.append(value)
                        signal_values = torch.stack(grid_values).reshape(resolution, resolution).detach().cpu().numpy()
                    else:
                        # Multi-slice evaluation
                        for i, z in enumerate(z_slices):
                            slice_values = []
                            for point in grid_points[i]:
                                value = signal_surrogate_func(opt_point=point, event_params=signal_event_params)
                                slice_values.append(value)
                            signal_values += torch.stack(slice_values).reshape(resolution, resolution).detach().cpu().numpy()
                        signal_values /= len(z_slices)
                
                # Handle old signal functions format (backward compatibility)
                elif signal_funcs:
                    vis_all_signals = kwargs.get('vis_all_signals', False)
                    
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
                
                # Show points near the slice with alpha based on string weights
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    # print("Test!")
                    alpha_values = np.array([string_weights[idx] for idx in range(len(string_weights))])
                    # Apply sigmoid to convert to [0,1] range if not already
                    # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                        # alpha_values = 1 / (1 + np.exp(-alpha_values))
                        
                    # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                    # Ensure minimum visibility
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                # print("Alpha values:", alpha_values)
                # alpha_values = [alpha_values[i] if alpha_values[i] > 0.7 else 0.1 for i in range(len(alpha_values))]
                
                ax.scatter(string_xy[:, 0], string_xy[:, 1], c='red', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')
                
                # Set appropriate title based on input type
                if signal_surrogate_func is not None:
                    ax.set_title("Signal Surrogate Function")
                else:
                    vis_all_signals = kwargs.get('vis_all_signals', False)
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
                ax.text(0.5, 0.5, "Signal function data not available\n(Pass 'signal_funcs' or 'signal_surrogate_func' + 'signal_event_params')", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_BACKGROUND_CONTOUR:
            # Background function contour plot
            background_funcs = kwargs.get('background_funcs', [])
            background_surrogate_func = kwargs.get('background_surrogate_func', None)
            background_event_params = kwargs.get('background_event_params', None)
            
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
                # Handle new surrogate function format
                if background_surrogate_func is not None and background_event_params is not None:
                    if not multi_slice:
                        # Evaluate surrogate function at each grid point
                        grid_values = []
                        for point in grid_points:
                            value = background_surrogate_func(opt_point=point, event_params=background_event_params)
                            grid_values.append(value)
                        bkg_values = torch.stack(grid_values).reshape(resolution, resolution).detach().cpu().numpy() * kwargs.get('background_scale', 1.0)
                    else:
                        # Multi-slice evaluation
                        for i, z in enumerate(z_slices):
                            slice_values = []
                            for point in grid_points[i]:
                                value = background_surrogate_func(opt_point=point, event_params=background_event_params)
                                slice_values.append(value)
                            bkg_values += torch.stack(slice_values).reshape(resolution, resolution).detach().cpu().numpy() * kwargs.get('background_scale', 1.0)
                        bkg_values /= len(z_slices)
                
                # Handle old background functions format (backward compatibility)
                elif background_funcs:
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
            if background_surrogate_func is not None:
                plot_title = "Background Surrogate Function"
            elif no_background:
                plot_title = "No Background"
            else:
                plot_title = "Combined Background"
                
            c2 = ax.contourf(X.cpu().numpy(), Y.cpu().numpy(), bkg_values, 
                            cmap='plasma', levels=20)
            fig.colorbar(c2, ax=ax)
            
            # Show points with alpha based on string weights
            string_weights = kwargs.get('string_weights', None)
            if string_weights is not None and string_indices is not None:
                alpha_values = np.array([string_weights[idx] for idx in range(len(string_weights))])
                # Apply sigmoid to convert to [0,1] range if not already
                # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                    # alpha_values = 1 / (1 + np.exp(-alpha_values))
                # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                # Ensure minimum visibility
                alpha_values = np.clip(alpha_values, 0.05, 1.0)
            else:
                alpha_values = 0.8
            
            # alpha_values = [alpha_values[i] if alpha_values[i] > 0.7 else 0.1 for i in range(len(alpha_values))]
                
            ax.scatter(string_xy[:, 0], string_xy[:, 1], c='red', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')
            
            ax.set_title(plot_title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            # Set consistent domain boundaries
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
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
                    snr_vals = all_snr.detach().cpu().numpy();
                    
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
            
                probs = torch.nn.functional.softmax(string_logits, dim=0).detach().cpu().numpy()
                
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
            
            points_np = points.detach().cpu().numpy()
            title_str = "Surrogate Function"
            
            # Get string weights for alpha transparency
            string_weights = kwargs.get('string_weights', None)
            string_indices_from_kwargs = kwargs.get('string_indices', None)
            
            if multi_slice:
                title_str += " (Multi-Slice Avg)"
                # For multi-slice, show all points projected to XY plane
                if string_weights is not None and string_indices_from_kwargs is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices_from_kwargs])
                    # Apply sigmoid to convert to [0,1] range if not already
                    # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                        # alpha_values = 1 / (1 + np.exp(-alpha_values))
                    # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                    # Ensure minimum visibility
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')
            else:
                title_str += " (Z=0)"
                # For single-slice, show points near the z=0 slice
                xy_points_z0 = points_np[np.abs(points_np[:, 2] - 0.0) < 0.2] # Check points close to z=0
                if len(xy_points_z0) > 0:
                    if string_weights is not None and string_indices is not None:
                        # Filter alpha values for points near z=0
                        z0_mask = np.abs(points_np[:, 2] - 0.0) < 0.2
                        alpha_values = np.array([string_weights[string_indices[i]] for i in range(len(string_indices)) if z0_mask[i]])
                        # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                            # alpha_values = 1 / (1 + np.exp(-alpha_values))
                        # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()
                        alpha_values = np.clip(alpha_values, 0.05, 1.0)
                    else:
                        alpha_values = 0.8
                    ax.scatter(xy_points_z0[:, 0], xy_points_z0[:, 1], c='r', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')
                else: # If no points are near z=0, show all points projected
                    if string_weights is not None and string_indices is not None:
                        alpha_values = np.array([string_weights[idx] for idx in string_indices])
                        # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                        #     alpha_values = 1 / (1 + np.exp(-alpha_values))
                        # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                        alpha_values = np.clip(alpha_values, 0.05, 1.0)
                    else:
                        alpha_values = 0.8
                    ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')

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
                                f_values_at_data = true_func_callable(points)
                                # Compute RBF interpolant weights and kernel matrix
                                w, K = compute_rbf_interpolant(
                                    points, f_values_at_data, grid_points_current_slice
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
                            f_values_at_data = true_func_callable(points)
                            # Compute RBF interpolant weights and kernel matrix
                            w, K = compute_rbf_interpolant(
                                points, f_values_at_data, grid_points_single_slice
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

            points_np = points.detach().cpu().numpy()
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

            # Calculate alpha values based on string weights
            string_weights = kwargs.get('string_weights', None)
            if string_weights is not None and string_indices is not None:
                alpha_values = np.array([string_weights[idx] for idx in string_indices])
                # Apply sigmoid to convert to [0,1] range if not already
                # if np.any(alpha_values < 0) or np.any(alpha_values > 1):
                #     alpha_values = 1 / (1 + np.exp(-alpha_values))
                # alpha_values = torch.nn.functional.softplus(torch.tensor(alpha_values)).detach().cpu().numpy()  # Apply softplus for smoothness
                # Ensure minimum visibility
                alpha_values = np.clip(alpha_values, 0.05, 1.0)
            else:
                alpha_values = 0.8

            if multi_slice:
                ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')
            else: # Single slice (Z=0.0)
                xy_points_z0 = points_np[np.abs(points_np[:, 2] - 0.0) < 0.2] # Points near Z=0
                if len(xy_points_z0) > 0:
                    # Get alpha values for points near Z=0
                    z0_indices = np.where(np.abs(points_np[:, 2] - 0.0) < 0.2)[0]
                    if isinstance(alpha_values, np.ndarray):
                        z0_alpha_values = alpha_values[z0_indices]
                    else:
                        z0_alpha_values = alpha_values
                    ax.scatter(xy_points_z0[:, 0], xy_points_z0[:, 1], c='r', s=min([40,30*200/len(string_indices)]), alpha=z0_alpha_values, edgecolor='black')
                else: # If no points near Z=0, show all points projected
                    ax.scatter(points_np[:, 0], points_np[:, 1], c='r', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='black')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(-self.half_domain, self.half_domain)
            ax.set_ylim(-self.half_domain, self.half_domain)
        
        elif plot_type == self.PLOT_STRING_WEIGHTS_SCATTER:
            # String weights scatter plot with variable alpha
            if string_xy is not None:
                string_weights = kwargs.get('string_weights', None)
                weight_threshold = kwargs.get('weight_threshold', 0.7)
                
                if string_weights is not None:
                    # Convert tensors to numpy arrays
                    xy_np = string_xy.detach().cpu().numpy()
                    weights_np = string_weights
                    # Create alpha values: 1 if weight > 0.7, else 0.5
                    alphas = [1 if weights_np[i] > 0.7 else 0.5 for i in range(len(weights_np))]
                    edge_colors=['k' if weights_np[i] > 0.7 else 'none' for i in range(len(weights_np))]
                    # Create scatter plot
                    scatter = ax.scatter(
                        xy_np[:, 0], 
                        xy_np[:, 1], 
                        c=weights_np,
                        cmap='Greens',
                        alpha=alphas,
                        edgecolors=edge_colors,
                        s=min([40,30*200/len(weights_np)])
                        )
                    
                    # Add colorbar
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('String Weight')
                    
                    # Set labels and title
                    ax.set_xlabel('X Coordinate')
                    ax.set_ylabel('Y Coordinate')
                    ax.set_title(f'Active strings = {len(weights_np[weights_np > 0.7])}, Total strings = {len(weights_np)}')
                    ax.set_xlim(-self.half_domain, self.half_domain)
                    ax.set_ylim(-self.half_domain, self.half_domain)
                    
                    # Add ROV safe space visualization if ROV penalty is available
                    rov_penalty = kwargs.get('rov_penalty', None)
                    if rov_penalty is not None:
                        self._draw_rov_safe_space(ax, rov_penalty)
                else:
                    ax.text(0.5, 0.5, "String weights data not available", 
                          ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "String XY data not available", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_LLR_CONTOUR:
            # Combined LLR contour plot based on per-string values
            llr_per_string = kwargs.get('llr_per_string', None)
            
            if llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(llr_per_string, 'detach'):
                    llr_values_np = llr_per_string.detach().cpu().numpy()
                else:
                    llr_values_np = np.array(llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding LLR values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate LLR values from string positions to grid
                # Use the minimum value in the data as fill_value to preserve negative values
                fill_val = np.min(llr_values_np) if len(llr_values_np) > 0 else np.nan
                llr_grid = griddata(
                    np.column_stack([string_x, string_y]), 
                    llr_values_np, 
                    grid_points,
                    method='linear', 
                    fill_value=fill_val
                ).reshape(resolution, resolution)
                
                # Create the contour plot
                c1 = ax.contourf(X_np, Y_np, llr_grid, cmap='RdYlBu_r', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Log-Likelihood Ratio')
                
                # Overlay string positions with their LLR values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their LLR values
                scatter = ax.scatter(string_x, string_y, c=llr_values_np, 
                                   cmap='RdYlBu_r', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"Combined LLR per String (n={len(llr_values_np)} strings)")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "LLR per string data not available\n(Requires 'llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SIGNAL_LLR_CONTOUR:
            # Signal-only LLR contour plot based on per-string values
            signal_llr_per_string = kwargs.get('signal_llr_per_string', None)
            
            if signal_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_string, 'detach'):
                    signal_llr_values_np = signal_llr_per_string.detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding signal LLR values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate signal LLR values from string positions to grid
                fill_val = np.min(signal_llr_values_np) if len(signal_llr_values_np) > 0 else np.nan
                signal_llr_grid = griddata(
                    np.column_stack([string_x, string_y]), 
                    signal_llr_values_np, 
                    grid_points,
                    method='linear', 
                    fill_value=fill_val
                ).reshape(resolution, resolution)
                
                # Create the contour plot with signal-appropriate colormap
                c1 = ax.contourf(X_np, Y_np, signal_llr_grid, cmap='Reds', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Signal Log-Likelihood Ratio')
                
                # Overlay string positions with their signal LLR values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their signal LLR values
                scatter = ax.scatter(string_x, string_y, c=signal_llr_values_np, 
                                   cmap='Reds', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"Pred. Signal LLR per String")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "Signal LLR per string data not available\n(Requires 'signal_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_TRUE_SIGNAL_LLR_CONTOUR:
            # Signal-only LLR contour plot based on per-string values
            signal_llr_per_string = kwargs.get('true_signal_llr_per_string', None)
            
            if signal_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_string, 'detach'):
                    signal_llr_values_np = signal_llr_per_string.detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding signal LLR values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate signal LLR values from string positions to grid
                fill_val = np.min(signal_llr_values_np) if len(signal_llr_values_np) > 0 else np.nan
                signal_llr_grid = griddata(
                    np.column_stack([string_x, string_y]), 
                    signal_llr_values_np, 
                    grid_points,
                    method='linear', 
                    fill_value=fill_val
                ).reshape(resolution, resolution)
                
                # Create the contour plot with signal-appropriate colormap
                c1 = ax.contourf(X_np, Y_np, signal_llr_grid, cmap='Reds', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Signal Log-Likelihood Ratio')
                
                # Overlay string positions with their signal LLR values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their signal LLR values
                scatter = ax.scatter(string_x, string_y, c=signal_llr_values_np, 
                                   cmap='Reds', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"True Signal LLR per String")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "True Signal LLR per string data not available\n(Requires 'true_signal_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SIGNAL_LLR_CONTOUR_POINTS:
            # Signal-only LLR contour plot based on per-point values
            signal_llr_per_points = kwargs.get('signal_llr_per_point', None)
            
            if signal_llr_per_points is not None and points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_points, 'detach'):
                    signal_llr_values_np = signal_llr_per_points.detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_points)
                
                points_np = points.detach().cpu().numpy()
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use point XY positions and their corresponding signal LLR values
                points_x = points_np[:, 0]
                points_y = points_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Use the safe interpolation method
                success, signal_llr_grid, error_msg = self._safe_griddata_interpolation(
                    np.column_stack([points_x, points_y]),
                    signal_llr_values_np,
                    grid_points,
                    resolution,
                    method='linear'
                )
                
                if success:
                    # Create the contour plot with signal-appropriate colormap
                    c1 = ax.contourf(X_np, Y_np, signal_llr_grid, cmap='Reds', levels=20)
                    cbar = fig.colorbar(c1, ax=ax)
                    cbar.set_label('Signal Log-Likelihood Ratio (per Point)')
                    
                    # Show point positions colored by their signal LLR values
                    scatter = ax.scatter(points_x, points_y, c=signal_llr_values_np, 
                                       cmap='Reds', s=10, alpha=0.6, edgecolor='black', linewidth=0.2,
                                       label='Point Positions')
                    
                    ax.set_title(f"Signal LLR per Point")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_xlim(-self.half_domain, self.half_domain)
                    ax.set_ylim(-self.half_domain, self.half_domain)
                else:
                    ax.text(0.5, 0.5, f"Signal LLR interpolation failed:\n{error_msg}", 
                          ha='center', va='center', transform=ax.transAxes)
                
            else:
                ax.text(0.5, 0.5, "Signal LLR per point data not available\n(Requires 'signal_llr_per_points' and 'points_3d' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_BACKGROUND_LLR_CONTOUR:
            # Background-only LLR contour plot based on per-string values
            background_llr_per_string = kwargs.get('background_llr_per_string', None)
            
            if background_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(background_llr_per_string, 'detach'):
                    background_llr_values_np = background_llr_per_string.detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding background LLR values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate background LLR values from string positions to grid
                fill_val = np.min(background_llr_values_np) if len(background_llr_values_np) > 0 else np.nan
                background_llr_grid = griddata(
                    np.column_stack([string_x, string_y]), 
                    background_llr_values_np, 
                    grid_points,
                    method='linear', 
                    fill_value=fill_val
                ).reshape(resolution, resolution)
                
                # Create the contour plot with background-appropriate colormap
                c1 = ax.contourf(X_np, Y_np, background_llr_grid, cmap='Blues', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Background Log-Likelihood Ratio')
                
                # Overlay string positions with their background LLR values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their background LLR values
                scatter = ax.scatter(string_x, string_y, c=background_llr_values_np, 
                                   cmap='Blues', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"Pred. Background LLR per String")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "Background LLR per string data not available\n(Requires 'background_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_TRUE_BACKGROUND_LLR_CONTOUR:
            # Background-only LLR contour plot based on per-string values
            background_llr_per_string = kwargs.get('true_background_llr_per_string', None)
            
            if background_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(background_llr_per_string, 'detach'):
                    background_llr_values_np = background_llr_per_string.detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding background LLR values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate background LLR values from string positions to grid
                fill_val = np.min(background_llr_values_np) if len(background_llr_values_np) > 0 else np.nan
                background_llr_grid = griddata(
                    np.column_stack([string_x, string_y]), 
                    background_llr_values_np, 
                    grid_points,
                    method='linear', 
                    fill_value=fill_val
                ).reshape(resolution, resolution)
                
                # Create the contour plot with background-appropriate colormap
                c1 = ax.contourf(X_np, Y_np, background_llr_grid, cmap='Blues', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Background Log-Likelihood Ratio')
                
                # Overlay string positions with their background LLR values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their background LLR values
                scatter = ax.scatter(string_x, string_y, c=background_llr_values_np, 
                                   cmap='Blues', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"True Background LLR per String")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "True Background LLR per string data not available\n(Requires 'true_background_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_BACKGROUND_LLR_CONTOUR_POINTS:
            # Background-only LLR contour plot based on per-point values
            background_llr_per_points = kwargs.get('background_llr_per_point', None)
            
            if background_llr_per_points is not None and points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(background_llr_per_points, 'detach'):
                    background_llr_values_np = background_llr_per_points.detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_points)
                
                points_np = points.detach().cpu().numpy()
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use point XY positions and their corresponding background LLR values
                points_x = points_np[:, 0]
                points_y = points_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Use the safe interpolation method
                success, background_llr_grid, error_msg = self._safe_griddata_interpolation(
                    np.column_stack([points_x, points_y]),
                    background_llr_values_np,
                    grid_points,
                    resolution,
                    method='linear'
                )
                
                if success:
                    # Create the contour plot with background-appropriate colormap
                    c1 = ax.contourf(X_np, Y_np, background_llr_grid, cmap='Blues', levels=20)
                    cbar = fig.colorbar(c1, ax=ax)
                    cbar.set_label('Background Log-Likelihood Ratio (per Point)')
                    
                    # Show point positions colored by their background LLR values
                    scatter = ax.scatter(points_x, points_y, c=background_llr_values_np, 
                                       cmap='Blues', s=10, alpha=0.6, edgecolor='black', linewidth=0.2,
                                       label='Point Positions')
                    
                    ax.set_title(f"Background LLR per Point")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_xlim(-self.half_domain, self.half_domain)
                    ax.set_ylim(-self.half_domain, self.half_domain)
                else:
                    ax.text(0.5, 0.5, f"Background LLR interpolation failed:\n{error_msg}", 
                          ha='center', va='center', transform=ax.transAxes)
                
            else:
                ax.text(0.5, 0.5, "Background LLR per point data not available\n(Requires 'background_llr_per_points' and 'points_3d' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_LLR_HISTOGRAM:
            # LLR density histogram plot with signal and background distributions
            signal_llr_per_string = kwargs.get('signal_llr_per_string', None)/kwargs.get('points_per_string', 1)
            background_llr_per_string = kwargs.get('background_llr_per_string', None)/kwargs.get('points_per_string', 1)
            string_weights = kwargs.get('string_weights', None)
            
            if signal_llr_per_string is not None and background_llr_per_string is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_string, 'detach'):
                    signal_llr_values_np = signal_llr_per_string.detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_string)
                    
                if hasattr(background_llr_per_string, 'detach'):
                    background_llr_values_np = background_llr_per_string.detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_string)
                
                # Apply string weights if available
                if string_weights is not None:
                    if hasattr(string_weights, 'detach'):
                        weights_np = string_weights.detach().cpu().numpy()
                    else:
                        weights_np = np.array(string_weights)
                    
                    # Ensure weights are the same length as LLR values
                    if len(weights_np) == len(signal_llr_values_np) == len(background_llr_values_np):
                        # Apply weights to the LLR values for histogram density
                        signal_weights = weights_np
                        background_weights = weights_np
                    else:
                        print(f"Warning: String weights length ({len(weights_np)}) doesn't match LLR values length ({len(signal_llr_values_np)}). Using uniform weights.")
                        signal_weights = np.ones_like(signal_llr_values_np)
                        background_weights = np.ones_like(background_llr_values_np)
                else:
                    signal_weights = np.ones_like(signal_llr_values_np)
                    background_weights = np.ones_like(background_llr_values_np)
                
                # Determine histogram range to include both distributions
                all_llr_values = np.concatenate([signal_llr_values_np, background_llr_values_np])
                hist_range = (np.min(all_llr_values) - 0.1 * np.abs(np.min(all_llr_values)), 
                             np.max(all_llr_values) + 0.1 * np.abs(np.max(all_llr_values)))
                
                # Create histograms with weights
                bins = 30
                
                # Signal LLR histogram
                ax.hist(signal_llr_values_np, bins=bins, range=hist_range, 
                       weights=signal_weights, alpha=0.7, color='red', 
                       label=f'Signal LLR (n={len(signal_llr_values_np)})', 
                       density=True, edgecolor='darkred', linewidth=0.5)
                
                # Background LLR histogram
                ax.hist(background_llr_values_np, bins=bins, range=hist_range, 
                       weights=background_weights, alpha=0.7, color='blue', 
                       label=f'Background LLR (n={len(background_llr_values_np)})', 
                       density=True, edgecolor='darkblue', linewidth=0.5)
                
                # Calculate weighted means
                signal_mean = np.average(signal_llr_values_np, weights=signal_weights)
                background_mean = np.average(background_llr_values_np, weights=background_weights)
                
                # Plot mean lines
                ax.axvline(signal_mean, color='darkred', linestyle='--', linewidth=2, 
                        #   label=f'Signal Mean: {signal_mean:.3f}'
                          )
                ax.axvline(background_mean, color='darkblue', linestyle='--', linewidth=2, 
                        #   label=f'Background Mean: {background_mean:.3f}'
                          )
                
                # Calculate separation metrics
                separation = abs(signal_mean - background_mean)
                
                # Set labels and title
                ax.set_xlabel('Log-Likelihood Ratio')
                ax.set_ylabel('Density')
                ax.set_title(f'LLR Distribution Comparison')
                ax.legend(fontsize='small')
                ax.grid(True, alpha=0.3)
                
                # Add text box with statistics
                stats_text = f'Signal Strings: {np.sum(signal_weights > 0.1):.0f}/{len(signal_weights)}\n'
                stats_text += f'Background Strings: {np.sum(background_weights > 0.1):.0f}/{len(background_weights)}\n'
                if string_weights is not None:
                    active_strings = np.sum(weights_np > 0.7)
                    stats_text += f'Active Strings: {active_strings}/{len(weights_np)}'
                
                # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                #        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                #        fontsize='small')
                
            elif signal_llr_per_string is not None or background_llr_per_string is not None:
                # Only one type of LLR data available
                available_data = signal_llr_per_string if signal_llr_per_string is not None else background_llr_per_string
                data_type = "Signal" if signal_llr_per_string is not None else "Background"
                color = 'red' if signal_llr_per_string is not None else 'blue'
                
                # Convert to numpy array
                if hasattr(available_data, 'detach'):
                    llr_values_np = available_data.detach().cpu().numpy()
                else:
                    llr_values_np = np.array(available_data)
                
                # Apply string weights if available
                if string_weights is not None:
                    if hasattr(string_weights, 'detach'):
                        weights_np = string_weights.detach().cpu().numpy()
                    else:
                        weights_np = np.array(string_weights)
                    
                    if len(weights_np) == len(llr_values_np):
                        llr_weights = weights_np
                    else:
                        llr_weights = np.ones_like(llr_values_np)
                else:
                    llr_weights = np.ones_like(llr_values_np)
                
                # Create histogram
                bins = 30
                ax.hist(llr_values_np, bins=bins, weights=llr_weights, alpha=0.7, color=color, 
                       label=f'{data_type} LLR (n={len(llr_values_np)})', 
                       density=True, edgecolor='black', linewidth=0.5)
                
                # Calculate and plot weighted mean
                weighted_mean = np.average(llr_values_np, weights=llr_weights)
                ax.axvline(weighted_mean, color='black', linestyle='--', linewidth=2, 
                          label=f'{data_type} Mean: {weighted_mean:.3f}')
                
                ax.set_xlabel('Log-Likelihood Ratio')
                ax.set_ylabel('Density')
                ax.set_title(f'{data_type} LLR Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            else:
                ax.text(0.5, 0.5, "LLR histogram data not available\n(Requires 'signal_llr_per_string' and/or 'background_llr_per_string' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_LLR_HISTOGRAM_POINTS:
            # LLR density histogram plot with signal and background distributions per point
            signal_llr_per_points = kwargs.get('signal_llr_per_point', None)
            background_llr_per_points = kwargs.get('background_llr_per_point', None)
            
            if signal_llr_per_points is not None and background_llr_per_points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_points, 'detach'):
                    signal_llr_values_np = signal_llr_per_points.detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_points)
                    
                if hasattr(background_llr_per_points, 'detach'):
                    background_llr_values_np = background_llr_per_points.detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_points)
                
                # Determine histogram range to include both distributions
                all_llr_values = np.concatenate([signal_llr_values_np, background_llr_values_np])
                hist_range = (np.min(all_llr_values) - 0.1 * np.abs(np.min(all_llr_values)), 
                             np.max(all_llr_values) + 0.1 * np.abs(np.max(all_llr_values)))
                
                # Create histograms
                bins = 30
                
                # Signal LLR histogram
                ax.hist(signal_llr_values_np, bins=bins, range=hist_range, 
                       alpha=0.7, color='red', 
                       label=f'Signal LLR (n={len(signal_llr_values_np)})', 
                       density=True, edgecolor='darkred', linewidth=0.5)
                
                # Background LLR histogram
                ax.hist(background_llr_values_np, bins=bins, range=hist_range, 
                       alpha=0.7, color='blue', 
                       label=f'Background LLR (n={len(background_llr_values_np)})', 
                       density=True, edgecolor='darkblue', linewidth=0.5)
                
                # Calculate means
                signal_mean = np.mean(signal_llr_values_np)
                background_mean = np.mean(background_llr_values_np)
                
                # Plot mean lines
                ax.axvline(signal_mean, color='darkred', linestyle='--', linewidth=2)
                ax.axvline(background_mean, color='darkblue', linestyle='--', linewidth=2)
                
                # Calculate separation metrics
                separation = abs(signal_mean - background_mean)
                
                # Set labels and title
                ax.set_xlabel('Log-Likelihood Ratio per Point')
                ax.set_ylabel('Density')
                ax.set_title(f'LLR Distribution Comparison (Per Point)')
                ax.legend(fontsize='small')
                ax.grid(True, alpha=0.3)
                
                # Add text box with statistics
                stats_text = f'Signal Points: {len(signal_llr_values_np)}\n'
                stats_text += f'Background Points: {len(background_llr_values_np)}\n'
                stats_text += f'Signal Mean: {signal_mean:.3f}\n'
                stats_text += f'Background Mean: {background_mean:.3f}\n'
                stats_text += f'Separation: {separation:.3f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize='small')
                
            elif signal_llr_per_points is not None or background_llr_per_points is not None:
                # Only one type of LLR data available
                available_data = signal_llr_per_points if signal_llr_per_points is not None else background_llr_per_points
                data_type = "Signal" if signal_llr_per_points is not None else "Background"
                color = 'red' if signal_llr_per_points is not None else 'blue'
                
                # Convert to numpy array
                if hasattr(available_data, 'detach'):
                    llr_values_np = available_data.detach().cpu().numpy()
                else:
                    llr_values_np = np.array(available_data)
                
                # Create histogram
                bins = 30
                ax.hist(llr_values_np, bins=bins, alpha=0.7, color=color, 
                       label=f'{data_type} LLR (n={len(llr_values_np)})', 
                       density=True, edgecolor='black', linewidth=0.5)
                
                # Calculate and plot mean
                mean_value = np.mean(llr_values_np)
                ax.axvline(mean_value, color='black', linestyle='--', linewidth=2, 
                          label=f'{data_type} Mean: {mean_value:.3f}')
                
                ax.set_xlabel('Log-Likelihood Ratio per Point')
                ax.set_ylabel('Density')
                ax.set_title(f'{data_type} LLR Distribution (Per Point)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            else:
                ax.text(0.5, 0.5, "LLR per-point histogram data not available\n(Requires 'signal_llr_per_points' and/or 'background_llr_per_points' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SIGNAL_LIGHT_YIELD_CONTOUR:
            # Signal light yield contour plot based on per-string values
            signal_light_yield_per_string = kwargs.get('signal_yield_per_string', None)
            
            if signal_light_yield_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_light_yield_per_string, 'detach'):
                    signal_light_yield_values_np = signal_light_yield_per_string.detach().cpu().numpy()
                else:
                    signal_light_yield_values_np = np.array(signal_light_yield_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding signal light yield values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate signal light yield values from string positions to grid
                if np.any(signal_light_yield_values_np != signal_light_yield_values_np[0]):
                    fill_val = np.min(signal_light_yield_values_np) if len(signal_light_yield_values_np) > 0 else np.nan
                    signal_light_yield_grid = griddata(
                        np.column_stack([string_x, string_y]), 
                        signal_light_yield_values_np, 
                        grid_points,
                        method='linear', 
                        fill_value=fill_val
                    ).reshape(resolution, resolution)
                    
                    # Create the contour plot with signal-appropriate colormap
                    c1 = ax.contourf(X_np, Y_np, signal_light_yield_grid, cmap='Oranges', levels=20)
                else:
                    # If all values are identical, create a uniform grid
                    signal_light_yield_grid = np.full((resolution, resolution), signal_light_yield_values_np[0])
                    c1 = ax.contourf(X_np, Y_np, signal_light_yield_grid, cmap='Oranges', levels=1)
                    # force colorbar to just show that single value
                    c1.set_clim(signal_light_yield_values_np[0]-0.5, signal_light_yield_values_np[0]+0.5)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Signal Light Yield')
                
                # Overlay string positions with their signal light yield values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in range(len(string_weights))])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their signal light yield values
                scatter = ax.scatter(string_x, string_y, c=signal_light_yield_values_np, 
                                   cmap='Oranges', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"Signal Light Yield per String")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "Signal light yield per string data not available\n(Requires 'signal_light_yield_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SIGNAL_LIGHT_YIELD_CONTOUR_POINTS:
            # Signal light yield contour plot based on per-point values
            signal_light_yield_per_points = kwargs.get('signal_yield_per_point', None)
            
            if signal_light_yield_per_points is not None and points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_light_yield_per_points, 'detach'):
                    signal_light_yield_values_np = signal_light_yield_per_points.detach().cpu().numpy()
                else:
                    signal_light_yield_values_np = np.array(signal_light_yield_per_points)
                
                points_np = points.detach().cpu().numpy()
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use point XY positions and their corresponding signal light yield values
                points_x = points_np[:, 0]
                points_y = points_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Use the safe interpolation method
                success, signal_light_yield_grid, error_msg = self._safe_griddata_interpolation(
                    np.column_stack([points_x, points_y]),
                    signal_light_yield_values_np,
                    grid_points,
                    resolution,
                    method='linear'
                )
                
                if success:
                    # Create the contour plot with signal light yield-appropriate colormap
                    c1 = ax.contourf(X_np, Y_np, signal_light_yield_grid, cmap='Oranges', levels=20)
                    cbar = fig.colorbar(c1, ax=ax)
                    cbar.set_label('Signal Light Yield (per Point)')
                    
                    # Show point positions colored by their signal light yield values
                    scatter = ax.scatter(points_x, points_y, c=signal_light_yield_values_np, 
                                       cmap='Oranges', s=10, alpha=0.6, edgecolor='black', linewidth=0.2,
                                       label='Point Positions')
                    
                    ax.set_title(f"Signal Light Yield per Point")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_xlim(-self.half_domain, self.half_domain)
                    ax.set_ylim(-self.half_domain, self.half_domain)
                else:
                    ax.text(0.5, 0.5, f"Signal light yield interpolation failed:\n{error_msg}", 
                          ha='center', va='center', transform=ax.transAxes)
                
            else:
                ax.text(0.5, 0.5, "Signal light yield per point data not available\n(Requires 'signal_yield_per_points' and 'points' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SNR_CONTOUR:
            # SNR contour plot based on per-string values
            snr_per_string = kwargs.get('snr_per_string', None)
            
            if snr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(snr_per_string, 'detach'):
                    snr_values_np = snr_per_string.detach().cpu().numpy()
                else:
                    snr_values_np = np.array(snr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding SNR values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Interpolate SNR values from string positions to grid
                # Use the minimum value in the data as fill_value to preserve negative values
                fill_val = np.min(snr_values_np) if len(snr_values_np) > 0 else np.nan
                snr_grid = griddata(
                    np.column_stack([string_x, string_y]), 
                    snr_values_np, 
                    grid_points,
                    method='linear', 
                    fill_value=fill_val
                ).reshape(resolution, resolution)
                
                # Create the contour plot with a colormap suitable for SNR (higher values = better)
                c1 = ax.contourf(X_np, Y_np, snr_grid, cmap='viridis', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Signal-to-Noise Ratio')
                
                # Overlay string positions with their SNR values as color
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in string_indices])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                
                # Show string positions colored by their SNR values
                scatter = ax.scatter(string_x, string_y, c=snr_values_np, 
                                   cmap='viridis', s=min([60, 40*200/len(string_indices)]), 
                                   alpha=alpha_values, edgecolor='black', linewidth=1,
                                   label='String Positions')
                
                ax.set_title(f"SNR per String (n={len(snr_values_np)} strings)")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "SNR per string data not available\n(Requires 'snr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_FISHER_INFO_CONTOUR:
            # Log determinant of Fisher Information matrix
            # fisher_info_per_string = kwargs.get('fisher_info_per_string', None)
            # string_weights = kwargs.get('string_weights', None)
            
            # if fisher_info_per_string is not None and string_xy is not None:
            #     # Convert to numpy arrays if they're tensors
            #     if hasattr(string_xy, 'detach'):
            #         string_positions_np = string_xy.detach().cpu().numpy()
            #     else:
            #         string_positions_np = np.array(string_xy)
                
            #     # Compute Fisher Information matrix per string and its log determinant
            #     fisher_logdet_values = []
            #     for s_idx in range(len(fisher_info_per_string)):
            #         fisher_matrix = fisher_info_per_string[s_idx]
            #         # if hasattr(fisher_matrix, 'detach'):
            #         #     fisher_matrix = fisher_matrix.detach().cpu().numpy()
                    
            #         # Add regularization for numerical stability
            #         reg_matrix = torch.eye(fisher_matrix.shape[0]) * 1e-6
            #         regularized_fisher = fisher_matrix + reg_matrix
            #         fisher_logdet = torch.logdet(regularized_fisher).detach().cpu().numpy()
            #         fisher_logdet_values.append(fisher_logdet)
                    # Compute log determinant
                #     try:
                #         # Compute eigenvalues to check positive definiteness
                #         eigenvals = np.linalg.eigvals(regularized_fisher)
                #         if np.all(eigenvals > 0):
                #             logdet = np.log(np.linalg.det(regularized_fisher))
                #         else:
                #             # Use pseudodeterminant for non-positive definite matrices
                #             eigenvals_pos = eigenvals[eigenvals > 1e-12]
                #             logdet = np.sum(np.log(eigenvals_pos)) if len(eigenvals_pos) > 0 else -np.inf
                #         fisher_logdet_values.append(logdet)
                #     except:
                #         fisher_logdet_values.append(-np.inf)
                # fisher_logdet_values = np.array(fisher_logdet_values)
            fisher_info_per_string_per_event = kwargs.get('fisher_info_per_string_per_event', None)
            string_weights = kwargs.get('string_weights', None)
            trace_fisher_info_per_string_per_event = np.zeros((fisher_info_per_string_per_event.shape[0], len(string_xy)))
            if fisher_info_per_string_per_event is not None:
                fisher_info_per_string_per_event = np.array(fisher_info_per_string_per_event)
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                
                for event_idx in range(fisher_info_per_string_per_event.shape[0]):
                    for s_idx in range(len(string_positions_np)):
                        trace_fisher_info_per_string_per_event[event_idx, s_idx] = np.trace(np.linalg.inv(fisher_info_per_string_per_event[event_idx, s_idx] + np.eye(fisher_info_per_string_per_event.shape[-1])*1e-6))
                fisher_logdet_values = np.mean(trace_fisher_info_per_string_per_event, axis=0)
                

                # Create a grid for interpolation in XY plane
                resolution = slice_res
                x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Use string XY positions and their corresponding Fisher log-det values
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Create grid points for interpolation
                grid_points = np.column_stack([X_np.flatten(), Y_np.flatten()])
                
                # Use safe griddata interpolation
                points_xy = np.column_stack([string_x, string_y])
                fill_val = np.min(fisher_logdet_values[np.isfinite(fisher_logdet_values)]) if np.any(np.isfinite(fisher_logdet_values)) else np.nan
                
                success, fisher_logdet_grid, error_msg = self._safe_griddata_interpolation(
                    points_xy, fisher_logdet_values, grid_points, resolution, 
                    method='linear', fill_value=fill_val
                )
                
                if success:
                    # Create the contour plot
                    c1 = ax.contourf(X_np, Y_np, fisher_logdet_grid, cmap='plasma', levels=20)
                    cbar = fig.colorbar(c1, ax=ax)
                    cbar.set_label(r'tr(I$_F^{-1}$)')
                    
                    # Overlay string positions
                    if string_weights is not None:
                        alpha_values = np.array([string_weights[idx] for idx in range(len(string_weights))])
                        alpha_values = np.clip(alpha_values, 0.05, 1.0)
                    else:
                        alpha_values = 0.8
                    
                    scatter = ax.scatter(string_x, string_y, c=fisher_logdet_values, 
                                       cmap='plasma', s=min([60, 40*200/len(string_x)]), 
                                       alpha=alpha_values, edgecolor='black', linewidth=1)
                    
                    ax.set_title(f"Fisher Info Inv. Trace per String")
                else:
                    # Fallback based on error type
                    finite_mask = np.isfinite(fisher_logdet_values)
                    num_finite = np.sum(finite_mask)
                    if num_finite > 0:
                        ax.scatter(string_x[finite_mask], string_y[finite_mask], 
                                 c=fisher_logdet_values[finite_mask], cmap='plasma', 
                                 s=min([60, 40*200/len(string_x)]), alpha=0.8, 
                                 edgecolor='black', linewidth=1)
                        ax.set_title(f"Fisher Info Inv. Trace per String")
                        ax.text(0.5, 0.02, f"Interpolation failed: {error_msg}", 
                              ha='center', va='bottom', transform=ax.transAxes, fontsize=8)
                    else:
                        ax.text(0.5, 0.5, "All Fisher Information matrices are singular", 
                              ha='center', va='center', transform=ax.transAxes)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(-self.half_domain, self.half_domain)
                ax.set_ylim(-self.half_domain, self.half_domain)
                
            else:
                ax.text(0.5, 0.5, "Fisher Information data not available\n(Requires 'fisher_info_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_ANGULAR_RESOLUTION:
            # Angular resolution history from Fisher Information matrix using Cramér-Rao bound
            loss_dict = kwargs.get('uw_loss_dict', None)
            
            if loss_dict is not None:
                angular_resolution_history = loss_dict.get('angular_resolution_loss', None)

            if angular_resolution_history is not None:
                angular_resolution_history = np.array(angular_resolution_history) * (180.0/np.pi)  # Convert to degrees
                # Plot the history of weighted total angular resolution
                ax.plot(angular_resolution_history, color='blue', linewidth=2, markersize=4)
                ax.set_title('Angular Resolution History')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Angular Resolution (degrees)')
                ax.grid(True, alpha=0.3)
                
                # # Add current value annotation
                # if len(angular_resolution_history) > 0:
                #     current_val = angular_resolution_history[-1]
                #     ax.annotate(f'Current: {current_val:.2f}°', 
                #               xy=(len(angular_resolution_history)-1, current_val),
                #               xytext=(10, 10), textcoords='offset points',
                #               fontsize=10, ha='left')
            else:
                ax.text(0.5, 0.5, "Angular resolution history not available\n(Pass 'angular_resolution_history' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_ENERGY_RESOLUTION:
            # Energy resolution history from Fisher Information matrix using Cramér-Rao bound
            loss_dict = kwargs.get('uw_loss_dict', None)

            if loss_dict is not None:
                energy_resolution_history = loss_dict.get('energy_resolution_loss', None)

            if energy_resolution_history is not None:
                energy_resolution_history = np.array(energy_resolution_history)
                
                # Plot the history of weighted total energy resolution
                ax.plot(energy_resolution_history, color='red', linewidth=2, markersize=4)
                ax.set_title('Energy Resolution History')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Energy Resolution [GeV]')
                ax.grid(True, alpha=0.3)
                
                # Add current value annotation
                # if len(energy_resolution_history) > 0:
                #     current_val = energy_resolution_history[-1]
                #     ax.annotate(f'Current: {current_val:.4f}', 
                #               xy=(len(energy_resolution_history)-1, current_val),
                #               xytext=(10, 10), textcoords='offset points',
                #               fontsize=10, ha='left')
            else:
                ax.text(0.5, 0.5, "Energy resolution history not available\n(Pass 'energy_resolution_history' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_LOSS_COMPONENTS:
            # Loss components plot from loss dictionary
            loss_dict = kwargs.get('loss_dict', None)
            loss_filter_list = kwargs.get('loss_filter', [])
            loss_weights_dict = kwargs.get('loss_weights_dict', None)
            loss_iterations_dict = kwargs.get('loss_iterations_dict', None)
            if loss_dict is not None and isinstance(loss_dict, dict) and loss_dict:
                # Plot each loss component
                for loss_name, loss_history in loss_dict.items():
                    if loss_name in loss_filter_list:
                        continue
                    if loss_weights_dict is not None and loss_name in loss_weights_dict:
                        weight = loss_weights_dict[loss_name]
                        if weight == 0.0:
                            continue
                    if loss_iterations_dict is not None:
                        iterations = loss_iterations_dict.get(loss_name, None)
                        if iterations is not None and len(iterations) == len(loss_history):
                            # If iterations have gaps, we need to handle missing iterations
                            # Create a full range from 0 to max iteration
                            max_iter = max(iterations)
                            full_range = list(range(max_iter + 1))
                            
                            # Create loss values array with None for missing iterations
                            full_loss_history = []
                            iter_idx = 0
                            for i in full_range:
                                if iter_idx < len(iterations) and iterations[iter_idx] == i:
                                    full_loss_history.append(loss_history[iter_idx])
                                    iter_idx += 1
                                else:
                                    full_loss_history.append(None)
                            
                            # Plot with gaps handled
                            ax.plot(full_range, full_loss_history, label=loss_name, alpha=0.8, linewidth=2)
                            continue
                    if loss_history and len(loss_history) > 0:
                        ax.plot(loss_history, label=loss_name, alpha=0.8, linewidth=2)
                
                # Calculate and plot total loss (sum of all components)
                # Find the maximum length of all loss histories
                max_length = max(len(history) for history in loss_dict.values() if history)
                
                # Calculate total loss at each iteration
                total_loss = []
                for i in range(max_length):
                    iteration_total = 0.0
                    for loss_name, loss_history in loss_dict.items():
                        if loss_name in loss_filter_list:
                            continue
                        if loss_weights_dict is not None and loss_name in loss_weights_dict:
                            weight = loss_weights_dict[loss_name]
                            if weight == 0.0:
                                continue
                        if loss_history and i < len(loss_history):
                            iteration_total += loss_history[i]
                    total_loss.append(iteration_total)
                
                # Plot total loss with a distinct style
                ax.plot(total_loss, label='Total Loss', color='black', 
                       linewidth=3, linestyle='--', alpha=0.9)
                
                ax.set_title(f"Loss Components")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss Value")
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, alpha=0.3)
                
                # Use log scale if all values are positive
                all_values = [val for history in loss_dict.values() for val in history if val is not None and val != 0]
                all_values.extend(total_loss)
                if all_values and all(val > 0 for val in all_values):
                    ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, "Loss dictionary not available or empty\n(Pass 'loss_dict' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_UW_LOSS_COMPONENTS:
            # Unweighted loss components plot from unweighted loss dictionary
            uw_loss_dict = kwargs.get('uw_loss_dict', None)
            loss_weights_dict = kwargs.get('loss_weights_dict', None)
            loss_iterations_dict = kwargs.get('loss_iterations_dict', None)
            if uw_loss_dict is not None and isinstance(uw_loss_dict, dict) and uw_loss_dict:
                # Plot each unweighted loss component
                for loss_name, loss_history in uw_loss_dict.items():
                    if loss_history and len(loss_history) > 0:
                        # transform each component to [1, 0] range for better visibility (min at 0, max at 1)
                        loss_array = np.log10(loss_history)
                        if len(loss_array) > 0 and np.max(loss_array) > np.min(loss_array):
                            # Normalize to [0, 1] first, then scale to [1e-2, 1]
                            normalized_loss = (loss_array - np.min(loss_array)) / (np.max(loss_array) - np.min(loss_array))
                            # normalized_loss = normalized_loss * (1 - 0.01) + 0.01
                        else:
                            # If all values are the same, set them to middle of range
                            normalized_loss = np.full_like(loss_array, 0.5)
                        
                        if loss_iterations_dict is None:    
                            ax.plot(normalized_loss, label=f"{loss_name}", alpha=0.8, linewidth=2)
                        else:
                            iterations = loss_iterations_dict.get(loss_name, None)
                            if iterations is not None:
                                # If iterations have gaps, we need to handle missing iterations
                                # Create a full range from 0 to max iteration
                                max_iter = max(iterations)
                                full_range = list(range(max_iter + 1))
                                
                                # Create loss values array with None for missing iterations
                                full_loss_history = []
                                iter_idx = 0
                                for i in full_range:
                                    if iter_idx < len(iterations) and iterations[iter_idx] == i:
                                        full_loss_history.append(normalized_loss[iter_idx])
                                        iter_idx += 1
                                    else:
                                        full_loss_history.append(None)
                                # Plot with gaps handled
                                ax.plot(full_range, full_loss_history, label=f"{loss_name}", alpha=0.8, linewidth=2)
                            else:
                                ax.plot(normalized_loss, label=f"{loss_name}", alpha=0.8, linewidth=2)

                # Calculate and plot total unweighted loss (sum of all components)
                # Find the maximum length of all loss histories
                max_length = max(len(history) for history in uw_loss_dict.values() if history)
               
                
                # Calculate total unweighted loss at each iteration
                # total_uw_loss = []
                # for i in range(max_length):
                #     iteration_total = 0.0
                #     for loss_history in uw_loss_dict.values():
                #         if loss_history and i < len(loss_history):
                #             iteration_total += loss_history[i]
                #     total_uw_loss.append(iteration_total)
                
                # Plot total unweighted loss with a distinct style
                # ax.plot(total_uw_loss, label='Total UW Loss', color='black', 
                #        linewidth=3, linestyle='--', alpha=0.9)
                
                ax.set_title(f"Unweighted Loss Components")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Normalized Loss Value (log scale)")
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, alpha=0.3)
                
                # Use log scale if all values are positive
                # all_values = [val for history in uw_loss_dict.values() for val in history if val is not None and val != 0]
                # # all_values.extend(total_uw_loss)
                # if all_values and all(val > 0 for val in all_values):
                #     ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, "Unweighted loss dictionary not available or empty\n(Pass 'uw_loss_dict' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
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
        # Safely handle torch tensor inputs by cloning and detaching them
        points_3d = self._safe_tensor_convert(points_3d)
        test_points = self._safe_tensor_convert(test_points)
        
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
    
    def create_interactive_3d_plot(self, points_3d, weight_threshold=None,
                                 points_per_string_list=None, string_xy=None, string_weights=None):
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
        # Safely handle torch tensor inputs by cloning and detaching them
        points_3d = self._safe_tensor_convert(points_3d)
        string_xy = self._safe_tensor_convert(string_xy)
        string_weights = self._safe_tensor_convert(string_weights)
        
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
        
        if points_per_string_list is not None:
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
                if string_weights is not None and weight_threshold is not None:
                    string_probs = torch.sigmoid(string_weights).detach().cpu().numpy()
                    if string_probs[s] < weight_threshold:
                        continue
                
                # Get points for this string
                # mask = np.array(string_indices) == s
                # if len(mask) != len(points_np):
                #     full_mask = np.zeros(len(points_np), dtype=bool)
                #     for k, pps in enumerate(points_per_string_list):
                #         if pps > 0: # set bool values to the mask value at s
                #            full_mask[k*pps:(k+1)*pps] = mask[s]
                              
                #     mask = full_mask  
                mask = (points_np[:,0] == string_xy[s][0]) & (points_np[:,1] == string_xy[s][1]) if string_xy is not None else np.array([True]*len(points_np))     
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
                    # fig.add_trace(
                    #     go.Scatter3d(
                    #         x=[x_pos],
                    #         y=[y_pos],
                    #         z=[-self.half_domain],  # Place at bottom of domain
                    #         mode='markers',
                    #         marker=dict(
                    #             size=8,
                    #             color=colormap_hex[s],
                    #             symbol='diamond',
                    #             opacity=0.8
                    #         ),
                    #         name=f'String {s} ({points_per_string_list[s]} pts)',
                    #         text=hovertext,
                    #         hoverinfo='text'
                    #     )
                    # )
                if string_weights is not None:
                    string_probs = torch.sigmoid(string_weights).detach().cpu().numpy()
                    # Use string weights for alpha transparency
                    alpha_value = 0.9 if string_probs[s] > 0.7 else 0.2
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
                            opacity=alpha_value if string_weights is not None else 0.9,
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
    
    def _compile_gif_from_images(self, gif_filename: str, gif_fps: int = 2) -> bool:
        """
        Compile a GIF from saved image files.
        
        Parameters:
        -----------
        gif_filename : str
            Output filename for the GIF.
        gif_fps : int
            Frames per second for the GIF.
            
        Returns:
        --------
        bool
            True if successful, False otherwise.
        """
        if not self.gif_image_paths:
            print("No images available to compile into GIF.")
            return False
            
        try:
            # Sort image paths numerically by extracting frame numbers
            def extract_frame_number(path):
                """Extract frame number from filename for proper numerical sorting."""
                filename = os.path.basename(path)
                # Look for pattern like "frame_0001.png" or "frame_1.png"
                match = re.search(r'frame_(\d+)', filename)
                if match:
                    return int(match.group(1))
                else:
                    # Fallback to alphabetical sorting if no number found
                    return float('inf')
            
            # Sort paths by frame number
            sorted_paths = sorted(self.gif_image_paths, key=extract_frame_number)
            
            # Load images in order and create GIF
            images = []
            for image_path in sorted_paths:
                if os.path.exists(image_path):
                    images.append(imageio.v3.imread(image_path))
                else:
                    print(f"Warning: Image file not found: {image_path}")
            
            if images:
                imageio.mimsave(gif_filename, images, fps=gif_fps)
                print(f"Successfully compiled GIF '{gif_filename}' with {len(images)} frames.")
                return True
            else:
                print("No valid images found to compile into GIF.")
                return False
                
        except Exception as e:
            print(f"Error compiling GIF: {e}")
            return False
    
    def finalize_gif(self, gif_filename: str = "optimization_progress.gif", 
                     gif_fps: int = 2, cleanup_images: bool = True) -> bool:
        """
        Finalize GIF creation by compiling from saved images and optionally cleaning up.
        
        Parameters:
        -----------
        gif_filename : str
            Output filename for the GIF.
        gif_fps : int
            Frames per second for the GIF.
        cleanup_images : bool
            If True, remove temporary image files after creating GIF.
            
        Returns:
        --------
        bool
            True if successful, False otherwise.
        """
        success = False
        
        # Compile GIF from saved images
        if self.gif_image_paths:
            success = self._compile_gif_from_images(gif_filename, gif_fps)
        elif self.gif_frames:
            # Fallback: compile from memory frames if no saved images
            try:
                imageio.mimsave(gif_filename, self.gif_frames, fps=gif_fps)
                print(f"Successfully compiled GIF '{gif_filename}' from {len(self.gif_frames)} memory frames.")
                success = True
            except Exception as e:
                print(f"Error compiling GIF from memory frames: {e}")
        else:
            print("No frames available to create GIF.")
        
        # Clean up temporary files if requested
        if cleanup_images and success:
            self.cleanup_gif_temp_files()
    
    def cleanup_gif_temp_files(self) -> None:
        """
        Clean up temporary files and directories created for GIF generation.
        """
        # Clear image paths list
        self.gif_image_paths.clear()
        
        # Clear memory frames
        self.gif_frames.clear()
        
        # Remove temporary directory and all its contents
        if self.gif_temp_dir and os.path.exists(self.gif_temp_dir):
            try:
                shutil.rmtree(self.gif_temp_dir)
                print(f"Cleaned up temporary directory: {self.gif_temp_dir}")
                self.gif_temp_dir = None
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
        print("GIF temporary files cleanup completed.")