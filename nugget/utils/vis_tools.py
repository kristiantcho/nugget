import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter
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

# IPython is optional. In some HPC/CLI environments, importing IPython can fail
# due to missing native runtime symbols (e.g. sqlite/libstdc++ ABI issues).
try:
    from IPython.display import clear_output, display
except Exception:
    def clear_output(wait=False):
        return None

    def display(*args, **kwargs):
        return None

# Try importing plotly for interactive 3D plotting, but don't fail if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def sph_to_cart(theta, phi):
    """Converts spherical coordinates (zenith, azimuth) to a 3D Cartesian vector."""
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    return torch.stack([st * cp, st * sp, ct], dim=-1)

def cart_to_sph(vec):
    """Converts a 3D Cartesian vector to spherical coordinates (zenith, azimuth)."""
    vec = vec.squeeze()
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))  # Zenith
    phi = torch.atan2(y, x)  # Azimuth
    return theta, phi


class Visualizer:
    """Base class for visualization tools in geometry optimization."""
    
    @staticmethod
    def _safe_tensor_convert(tensor_input, allow_none=True):
        """
        Safely convert torch tensors by cloning, detaching, and moving to CPU.
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
            For torch.Tensor: cloned, detached tensor on CPU
            For list of tensors: list of cloned, detached tensors on CPU
            For other types: unchanged input
        """
        if tensor_input is None and allow_none:
            return None
        if torch.is_tensor(tensor_input):
            return tensor_input.clone().detach().cpu()
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
    PLOT_POINTSOURCE_FOM = "pointsource_fom"
    PLOT_ANGULAR_RESOLUTION_HISTORY = "angular_resolution_history"
    PLOT_ENERGY_RESOLUTION_HISTORY = "energy_resolution_history"
    PLOT_ANGULAR_RESOLUTION_VS_ZENITH = "angular_resolution_vs_zenith"
    PLOT_ANGULAR_RESOLUTION_VS_ENERGY = "angular_resolution_vs_energy"
    PLOT_ENERGY_RESOLUTION_VS_ENERGY = "energy_resolution_vs_energy"
    PLOT_POINTSOURCE_FOM_VS_ENERGY = "pointsource_fom_vs_energy"
    PLOT_LOSS_COMPONENTS = "loss_components"
    PLOT_UW_LOSS_COMPONENTS = "uw_loss_components"
    PLOT_LLR_HISTOGRAM_POINTS = "llr_histogram_points"
    PLOT_STRING_XY_ROV_PENALTY = "string_xy_rov_penalty"
    PLOT_STRING_XY_LOCAL_STRING_REPULSION = "string_xy_local_string_repulsion_penalty"
    PLOT_ALM_MU = "alm_mu"
    PLOT_ALM_LAMBDA = "alm_lambda"

    
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
        self.device = device if device is not None else torch.device('cpu')
        self.dim = dim
        self.domain_size = domain_size
        self.half_domain = domain_size / 2
        self.gif_frames = [] # Added to store frames for the GIF
        self.gif_temp_dir = gif_temp_dir# Temporary directory for storing individual images
        self.gif_image_paths = [] # List to track saved image paths

    @staticmethod
    def _z_value_for_confidence(confidence_level: float = 0.95) -> float:
        """Return the (two-sided) z-value for a given confidence level.

        Falls back to the 95% normal z-value if SciPy stats is unavailable.
        """
        try:
            confidence_level = float(confidence_level)
        except Exception:
            confidence_level = 0.95

        confidence_level = float(np.clip(confidence_level, 0.0, 1.0))
        if confidence_level <= 0.0:
            return 0.0

        try:
            from scipy.stats import norm

            return float(norm.ppf(0.5 + confidence_level / 2.0))
        except Exception:
            # Common default.
            return 1.959963984540054

    @staticmethod
    def _compute_fom_from_resolution(values, min_resolution=1e-12):
        """Compute FOM and propagated uncertainty from per-event resolutions.

        FOM is defined as sqrt(sum_i(1 / r_i^2)). The uncertainty is propagated as
        (1 / (2 * FOM)) * sqrt(sum_i(1 / r_i^4)).
        """
        # Use float64 explicitly to avoid overflow in 1/r^4 on float32 inputs.
        vals = np.asarray(values, dtype=np.float64)
        min_resolution = float(min_resolution)
        valid_mask = np.isfinite(vals) & (vals > min_resolution)
        vals = vals[valid_mask]
        if vals.size == 0:
            return np.nan, np.nan

        inv_sq = 1.0 / np.square(vals)
        fom = float(np.sqrt(np.sum(inv_sq)))
        if not np.isfinite(fom) or fom <= 0.0:
            return np.nan, np.nan

        inv_four = np.square(inv_sq)
        fom_err = float((0.5 / fom) * np.sqrt(np.sum(inv_four)))
        if not np.isfinite(fom_err):
            fom_err = np.nan
        return fom, fom_err

    @staticmethod
    def _compute_pointsource_fom_from_resolution_and_aeff(res_values, aeff_values, min_resolution=1e-12):
        """Compute pointsource FoM and propagated uncertainty.

        FoM is defined as sqrt(sum_i(A_i / (4*pi*r_i^2))).
        """
        res = np.asarray(res_values, dtype=np.float64)
        aeff = np.asarray(aeff_values, dtype=np.float64)
        min_resolution = float(min_resolution)

        valid_mask = np.isfinite(res) & np.isfinite(aeff) & (res > min_resolution) & (aeff >= 0.0)
        res = res[valid_mask]
        aeff = aeff[valid_mask]
        if res.size == 0:
            return np.nan, np.nan

        terms = aeff / (4.0 * np.pi * np.square(res))
        fom = float(np.sqrt(np.sum(terms)))
        if not np.isfinite(fom) or fom <= 0.0:
            return np.nan, np.nan

        # First-order propagation for F = sqrt(S): sigma_F ≈ (1/(2F)) * sigma_S.
        # Here we estimate sigma_S via sqrt(sum(term_i^2)).
        fom_err = float((0.5 / fom) * np.sqrt(np.sum(np.square(terms))))
        if not np.isfinite(fom_err):
            fom_err = np.nan
        return fom, fom_err

    @staticmethod
    def _pad_frames_to_max_size(frames, background_value=255):
        """Pad frames to the maximum (H, W) using a white background.

        Matplotlib can produce slightly different pixel sizes between frames when
        saving with bbox_inches='tight' (e.g., due to colorbars, legends, or twinx axes).
        Padding keeps the final GIF size consistent while preserving each frame's
        original rendering.
        """
        if not frames:
            return frames

        valid = [f for f in frames if isinstance(f, np.ndarray) and f.ndim >= 2]
        if not valid:
            return frames

        target_h = int(max(f.shape[0] for f in valid))
        target_w = int(max(f.shape[1] for f in valid))

        padded_frames = []
        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.ndim < 2:
                padded_frames.append(frame)
                continue

            h, w = frame.shape[0], frame.shape[1]
            if h == target_h and w == target_w:
                padded_frames.append(frame)
                continue

            y0 = (target_h - h) // 2
            x0 = (target_w - w) // 2

            if frame.ndim == 2:
                canvas = np.full((target_h, target_w), background_value, dtype=frame.dtype)
                canvas[y0:y0 + h, x0:x0 + w] = frame
                padded_frames.append(canvas)
            else:
                channels = frame.shape[2]
                canvas = np.full((target_h, target_w, channels), background_value, dtype=frame.dtype)
                canvas[y0:y0 + h, x0:x0 + w, :] = frame
                padded_frames.append(canvas)

        return padded_frames
    
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
        def _ticks_are_integers(ticks, atol=1e-9):
            ticks = np.asarray(ticks, dtype=float)
            if ticks.size == 0:
                return False
            finite = np.isfinite(ticks)
            if not np.any(finite):
                return False
            ticks = ticks[finite]
            return np.allclose(ticks, np.round(ticks), atol=atol, rtol=0.0)

        if hasattr(ax, 'xaxis') and hasattr(ax, 'yaxis'):
            # Limit number of ticks to prevent overcrowding
            # ax.locator_params(axis='x', nbins=max_ticks)
            # ax.locator_params(axis='y', nbins=max_ticks)
            
            # Format tick labels to consistent precision only if not log scaled
            if ax.get_xscale() != 'log':
                xticks = ax.get_xticks()
                if _ticks_are_integers(xticks):
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, integer=True))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(round(x))}'))
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.{label_precision}f}'))
            if ax.get_yscale() != 'log':
                yticks = ax.get_yticks()
                if _ticks_are_integers(yticks):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, integer=True))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{int(round(y))}'))
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.{label_precision}f}'))
            
            # Set consistent tick label size
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            
            # Ensure tick labels don't extend beyond plot area
            ax.tick_params(axis='x', rotation=0, pad=3)
            ax.tick_params(axis='y', rotation=0, pad=3)
    
    def _draw_rov_safe_space(self, ax, rov_penalty=None, position='bottom_left', scale_factor=1, zoom_range=None):
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
        # rov_rec_width = rov_penalty.rov_rec_width
        rov_rec_width = rov_penalty.rov_rec_width
        rov_height = rov_penalty.rov_height
        rov_tri_length = rov_penalty.rov_tri_length
        if zoom_range is not None:
            ax_lims = zoom_range*2
        else:
            ax_lims = self.domain_size
        
        # Scale dimensions to fit in corner of plot
        scale = scale_factor #* self.domain_size
        rec_width = rov_rec_width * scale
        rec_height = rov_height * scale
        tri_length = rov_tri_length * scale
        
        # Position in bottom left corner
        if position == 'bottom_left':
            x_offset = -ax_lims/2 + 0.05 * ax_lims
            y_offset = -ax_lims/2 + 0.05 * ax_lims
        elif position == 'bottom_right':
            x_offset = ax_lims/2 - (rec_width + tri_length) - 0.05 * ax_lims
            y_offset = -ax_lims/2 + 0.05 * ax_lims
        else:  # default to bottom_left
            x_offset = -ax_lims/2 + 0.05 * ax_lims
            y_offset = -ax_lims/2 + 0.05 * ax_lims

        # Intended shape (to match ROVPenalty): triangle nose first, then rectangle.
        # Triangle from x_offset .. x_offset+tri_length widening to full height,
        # followed by rectangle of constant height.

        # Draw triangular part (nose)
        tri_x = [x_offset, x_offset + tri_length, x_offset + tri_length, x_offset]
        tri_y = [y_offset, y_offset - rec_height/2, y_offset + rec_height/2, y_offset]
        ax.plot(tri_x, tri_y, 'r-', linewidth=2, alpha=0.7, label='ROV Safe Space')

        # Draw rectangular part (corridor)
        rect_x = [x_offset + tri_length, x_offset + tri_length + rec_width, x_offset + tri_length + rec_width, x_offset + tri_length, x_offset + tri_length]
        rect_y = [y_offset - rec_height/2, y_offset - rec_height/2, y_offset + rec_height/2, y_offset + rec_height/2, y_offset - rec_height/2]
        ax.plot(rect_x, rect_y, 'r-', linewidth=2, alpha=0.7)

        # Fill the combined shape
        fill_x = [x_offset, x_offset + tri_length, x_offset + tri_length + rec_width, x_offset + tri_length + rec_width, x_offset + tri_length]
        fill_y = [y_offset, y_offset - rec_height/2, y_offset - rec_height/2, y_offset + rec_height/2, y_offset + rec_height/2]
        ax.fill(fill_x, fill_y, 'red', alpha=0.2)

        # Add "ROV" text inside the rectangular part of the safe space.
        # Compute a fontsize that fits the rectangle *in display pixels*, so it
        # stays consistent under different axis limits/zoom levels.
        text_x = x_offset + tri_length + rec_width / 2  # Center of rectangle
        text_y = y_offset  # Center vertically

        text_artist = ax.text(
            text_x,
            text_y,
            'ROV',
            fontsize=10.0,
            fontweight='bold',
            ha='center',
            va='center',
            color='darkred',
            alpha=0.8,
        )

        # Fit fontsize to the rectangle bounds. This requires a renderer.
        try:
            fig = ax.figure
            if fig is not None and fig.canvas is not None:
                # Ensure a renderer exists.
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

                rect_x0 = x_offset + tri_length
                rect_x1 = rect_x0 + rec_width
                rect_y0 = y_offset - rec_height / 2
                rect_y1 = y_offset + rec_height / 2

                (x0_px, y0_px), (x1_px, y1_px) = ax.transData.transform(
                    np.array([[rect_x0, rect_y0], [rect_x1, rect_y1]], dtype=float)
                )
                rect_w_px = float(abs(x1_px - x0_px))
                rect_h_px = float(abs(y1_px - y0_px))

                bbox = text_artist.get_window_extent(renderer=renderer)
                text_w_px = float(bbox.width)
                text_h_px = float(bbox.height)

                if rect_w_px > 0 and rect_h_px > 0 and text_w_px > 0 and text_h_px > 0:
                    # Scale linearly with fontsize; keep a small margin.
                    margin = 0.90
                    scale = margin * min(rect_w_px / text_w_px, rect_h_px / text_h_px)
                    new_fontsize = float(np.clip(10.0 * scale, 1.0, 72.0))
                    if np.isfinite(new_fontsize) and new_fontsize > 0:
                        text_artist.set_fontsize(new_fontsize)
        except Exception:
            # Best-effort sizing; keep default fontsize on failure.
            pass

    def _draw_rov_safe_space_at_string(
        self,
        ax,
        origin_xy,
        angle_rad,
        rov_penalty=None,
        *,
        alpha=0.12,
        line_alpha=0.55,
        linewidth=1.5,
        zorder=2,
    ):
        """Draw a rotated ROV safe-space corridor anchored at a string.

        The geometry matches `ROVPenalty`.
        Note: the angle convention is the one used in `ROVPenalty`'s rotation
        (local->world is a rotation by `-angle_rad`).
        """
        if rov_penalty is None or origin_xy is None or angle_rad is None:
            return

        try:
            x0 = float(origin_xy[0])
            y0 = float(origin_xy[1])
            a = float(angle_rad)
        except Exception:
            return

        L_rect = float(rov_penalty.rov_rec_width)
        W_rect = float(rov_penalty.rov_height)
        L_tri = float(rov_penalty.rov_tri_length)
        half_height = W_rect / 2.0

        # Local polygon (x forward, y sideways): triangle nose then rectangle.
        poly_local = np.array(
            [
                [0.0, 0.0],
                [L_tri, -half_height],
                [L_tri + L_rect, -half_height],
                [L_tri + L_rect, half_height],
                [L_tri, half_height],
            ],
            dtype=float,
        )

        # Transform local -> world. (Inverse of the rotation used in ROVPenalty.)
        c = float(np.cos(a))
        s = float(np.sin(a))
        rot = np.array([[c, s], [-s, c]], dtype=float)
        poly_world = poly_local @ rot.T
        poly_world[:, 0] += x0
        poly_world[:, 1] += y0

        poly_world_closed = np.vstack([poly_world, poly_world[0]])
        ax.plot(
            poly_world_closed[:, 0],
            poly_world_closed[:, 1],
            color='red',
            linewidth=linewidth,
            alpha=float(np.clip(line_alpha, 0.0, 1.0)),
            zorder=zorder,
        )
        ax.fill(
            poly_world[:, 0],
            poly_world[:, 1],
            color='red',
            alpha=float(np.clip(alpha, 0.0, 1.0)),
            zorder=zorder,
        )

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
        
        # Convert tensors to numpy if needed (already cloned/detached/CPU by _safe_tensor_convert)
        if torch.is_tensor(points_xy):
            points_xy = points_xy.numpy()
        if torch.is_tensor(values):
            values = values.numpy()
        if torch.is_tensor(grid_points):
            grid_points = grid_points.numpy()
        
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
                          points_3d: torch.Tensor=None,
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
            - 'string_xy_local_string_repulsion_penalty': String XY scatter colored by per-string local string repulsion penalty
            - 'signal_light_yield_contour': Signal light yield contour plot based on per-string values
            - 'signal_light_yield_contour_points': Signal light yield contour plot based on per-point values
            - 'fisher_info_logdet': Log determinant of Fisher Information matrix contour plot
            - 'angular_resolution': Angular resolution from Fisher Information using Cramér-Rao bound
            - 'energy_resolution': Energy resolution from Fisher Information using Cramér-Rao bound
            - 'pointsource_fom': Pointsource FoM history from unweighted loss dictionary
            - 'angular_resolution_vs_zenith': Binned angular resolution vs zenith
            - 'angular_resolution_vs_energy': Binned angular resolution vs energy
            - 'energy_resolution_vs_energy': Binned energy resolution vs energy
            - 'loss_components': Individual loss components and total loss from loss dictionary
            - 'uw_loss_components': Individual unweighted loss components and total unweighted loss
            - 'alm_mu': ALM penalty parameters (mu) history for each constraint
            - 'alm_lambda': ALM Lagrange multipliers (lambda) history for each constraint
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
                        - rov_penalty / rov_penalty_func: ROVPenalty object used by `string_xy_rov_penalty`
                        - rov_draw_safe_space_on_violations: bool, optional. If True, the `string_xy_rov_penalty` plot will
                            draw a per-string ROV safe-space corridor for strings with violation >= 1, oriented by
                            `rov_least_blocked_angle_per_string` (both are expected to be present in kwargs from `ROVPenalty`).
            - zoom_range: float, optional. If provided, sets axis limits for 2D contour plots to [-zoom_range, zoom_range] 
              instead of the default domain boundaries [-half_domain, half_domain]
            - plot_with_surrogate: bool, optional. If True and 'light_surrogate_func' and 'surrogate_event_params' 
              are provided, will generate full domain contour plot using the surrogate function for 'signal_light_yield_contour'
            - light_surrogate_func: callable, optional. Surrogate function to evaluate light yield across the full domain.
              Expected to accept 'opt_point' and 'event_params' keyword arguments and return light yield values.
            - surrogate_event_params: dict or list of dicts, optional. Event parameters to pass to the surrogate function.
              Can be a single dict containing 'position', 'zenith', 'azimuth', 'energy', etc., or a list of such dicts.
              If a list is provided, the light yield will be averaged over all events in the list.

                        For resolution-vs-* plots ('angular_resolution_vs_zenith', 'angular_resolution_vs_energy', 'energy_resolution_vs_energy'):
                        - resolution_per_event: array-like, per-event resolution values
                        - resolution_params: list of dicts, each containing 'zenith' and/or 'energy'
                                                - resolution_stat: {'median', 'mean'}, optional. Defaults to 'median'.
                                                        If 'median': the line is the median and `resolution_ci_percentiles` apply to residuals around the median.
                                                        If 'mean': the line is the mean and the band is ±2σ per bin (ignores residual quantiles).
                                                        (Backwards-compat alias: resolution_use_mean=True)
                                                - resolution_use_fom: bool, optional. If True, plots per-bin FOM = sqrt(sum(1/resolution^2)).
                                                    Error bars are propagated as (1/(2*FOM))*sqrt(sum(1/resolution^4)).
                                                - resolution_fom_min_resolution: float, optional. Minimum allowed resolution used in FOM mode
                                                    to avoid divide-by-zero (default: 1e-12).
                                                - show_resolution_ci: bool, optional. If True, draws a two-sided residual-quantile band around the median in each bin
                                                - resolution_ci_percentiles: tuple(float, float), optional. Percentiles for the residual band (default: (16, 84))
                                                - resolution_ci_level: float in (0, 1), optional. Alternative specification as a central containment level. Ignored if
                                                    resolution_ci_percentiles is provided.
                        - zenith_range / zenith_range_deg: tuple(min, max), optional. Restrict zenith range for binning.
                        - energy_range: tuple(min, max), optional. Restrict energy range for binning.
                                                - resolution_logy: bool, optional. If True, uses log scale for y-axis on all resolution/FoM-vs plots
                                                    ('angular_resolution_vs_zenith', 'angular_resolution_vs_energy',
                                                    'energy_resolution_vs_energy', 'pointsource_fom_vs_energy').
                                                    (Backwards-compat aliases: resolution_logy_angular, resolution_logy_vs_zenith,
                                                    resolution_logy_vs_energy)
                        - n_zenith_bins / n_energy_bins: int, optional. Number of bins
        """
        # Backwards-compat: allow callers to pass `points_3d`.
        if points is None and points_3d is not None:
            points = points_3d

        # Safely handle torch tensor inputs by cloning and detaching them
        points = self._safe_tensor_convert(points)
        string_xy = self._safe_tensor_convert(string_xy)
        
        
        # Handle potential torch tensors in kwargs
        for key in ['test_points', 'string_weights', 'signal_funcs', 'background_funcs', 'string_spacing']:
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
                # Render GIF frames using the same layout defaults as the regular (non-GIF)
                # plotting path so the plot appears "as is".
                # Keep up to 3 plots per row; don't allocate empty 3rd column for 1-2 plots.
                num_gif_cols = 3 if num_gif_plots >= 3 else int(num_gif_plots)
                num_gif_cols = max(1, int(num_gif_cols))
                num_gif_rows = (num_gif_plots + num_gif_cols - 1) // num_gif_cols
                num_gif_rows = max(1, int(num_gif_rows))

                if gif_fixed_figsize is not None:
                    gif_fig_size = gif_fixed_figsize
                else:
                    gif_fig_size = (5 * num_gif_cols, 4.5 * num_gif_rows)

                fig_gif, axes_gif_array = plt.subplots(
                    num_gif_rows,
                    num_gif_cols,
                    figsize=gif_fig_size,
                    squeeze=False,
                )
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
                # Keep plots as-is for GIF frames (no extra tick/layout standardization).
                # Match the regular display path by using tight_layout.
                try:
                    fig_gif.tight_layout()
                except Exception:
                    pass
                for ax3d in fig_gif.axes:
                    if getattr(ax3d, 'name', '') == '3d':
                        shift_left = float(getattr(ax3d, '_plot3d_shift_left', 0.0))
                        if shift_left != 0.0:
                            pos = ax3d.get_position()
                            ax3d.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])
                
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
                    # Save the frame exactly as rendered (no bbox tightening).
                    # Frame sizes may vary; we pad them during GIF compilation.
                    fig_gif.savefig(
                        image_path,
                        format='png',
                        dpi=100,
                        bbox_inches=None,
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.1,
                    )
                    self.gif_image_paths.append(image_path)
                    print(f"Saved GIF frame {len(self.gif_image_paths)} to {image_path}")
                    
                    # Compile GIF if requested on each iteration
                    if compile_gif_on_iteration:
                        self._compile_gif_from_images(gif_filename, gif_fps)
                else:
                    # Original method: store frames in memory
                    img_buf = io.BytesIO()
                    fig_gif.savefig(
                        img_buf,
                        format='png',
                        dpi=100,
                        bbox_inches=None,
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.1,
                    )
                    img_buf.seek(0)
                    self.gif_frames.append(imageio.v3.imread(img_buf))
                    img_buf.close()
                    
                    # Compile GIF from memory frames
                    if self.gif_frames and compile_gif_on_iteration:
                        try:
                            frames = self._pad_frames_to_max_size(self.gif_frames)
                            imageio.mimsave(gif_filename, frames, fps=gif_fps)
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
        # Keep up to 3 plots per row; don't allocate empty 3rd column for 1-2 plots.
        ncols = 3 if num_plots >= 3 else int(num_plots)
        ncols = max(1, int(ncols))
        num_rows = (num_plots + 2) // 3
        num_rows = max(1, int(num_rows))
        fig, axes = plt.subplots(num_rows, ncols, figsize=(5 * ncols, 4.5 * num_rows), squeeze=False) # MODIFIED
        axes_flat = axes.flatten()
        
        # Generate each requested plot
        for i, plot_type in enumerate(plot_types):
            ax = axes_flat[i]

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
        total_axes = num_rows * ncols
        for i in range(num_plots, total_axes):
            axes_flat[i].axis('off')
        
        # tight_layout can fail for some edge cases (e.g. pathological font sizes).
        # Visualization should not crash the evaluator, so make this best-effort.
        try:
            fig.tight_layout()
        except Exception as exc:
            print(f"Warning: fig.tight_layout() failed: {exc}")
        for ax3d in fig.axes:
            if getattr(ax3d, 'name', '') == '3d':
                shift_left = float(getattr(ax3d, '_plot3d_shift_left', 0.0))
                if shift_left != 0.0:
                    pos = ax3d.get_position()
                    ax3d.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])
        plt.show()

    def visualize_multi_progress(
        self,
        *,
        geom_vis_kwargs: Dict[str, Dict[str, Any]],
        plot_types: Optional[List[str]] = None,
        iteration: int = 0,
        slice_res: int = 50,
        multi_slice: bool = False,
        loss_type: str = 'rbf',
        make_gif: bool = False,
        ratio_baseline_geometry: Optional[str] = None,
        **shared_kwargs,
    ) -> None:
        """Visualize multiple geometries in a single call.

        Notes
        -----
        - History/performance plots (loss, SNR, LLR) are overlaid with one line per geometry.
        - Geometry plots (e.g. string_xy, string_xy_rov_penalty) are rendered as separate
          subplots per geometry, with the geometry name annotated in each subplot.
        - GIF generation is currently not supported in multi-geometry mode.
        """

        if make_gif:
            make_gif = False

        if not geom_vis_kwargs:
            print("Warning: visualize_multi_progress called with no geometries.")
            return

        # Determine plot_types if not provided.
        if plot_types is None:
            first_payload = next(iter(geom_vis_kwargs.values()))
            plot_types = first_payload.get('plot_types', None)
        if plot_types is None:
            plot_types = [self.PLOT_LOSS]

        # Clear output once for the whole multi-geometry rendering.
        clear_output(wait=True)

        overlay_plot_types = {
            self.PLOT_LOSS,
            self.PLOT_UW_LOSS,
            self.PLOT_SNR_HISTORY,
            self.PLOT_LLR_HISTORY,
            # Curve/metric plots (overlay is meaningful and does not require plotting XY geometry)
            self.PLOT_PARAM_1D,
            self.PLOT_ANGULAR_RESOLUTION,
            self.PLOT_ENERGY_RESOLUTION,
            self.PLOT_POINTSOURCE_FOM,
            self.PLOT_ANGULAR_RESOLUTION_VS_ZENITH,
            self.PLOT_ANGULAR_RESOLUTION_VS_ENERGY,
            self.PLOT_ENERGY_RESOLUTION_VS_ENERGY,
            self.PLOT_POINTSOURCE_FOM_VS_ENERGY,
            self.PLOT_LOSS_COMPONENTS,
            self.PLOT_UW_LOSS_COMPONENTS,
            self.PLOT_ALM_MU,
            self.PLOT_ALM_LAMBDA,
        }
        # Everything not in overlay_plot_types is rendered as one subplot per geometry.

        geom_items = list(geom_vis_kwargs.items())
        n_geoms = len(geom_items)

        # Stable, consistent colors per geometry across overlay plots.
        geom_names = [str(name) for name, _ in geom_items]
        default_colors = plt.rcParams.get('axes.prop_cycle', None)
        if default_colors is not None:
            try:
                default_colors = list(default_colors.by_key().get('color', []))
            except Exception:
                default_colors = []
        if not default_colors:
            default_colors = [f"C{i}" for i in range(max(10, len(geom_names)))]
        geom_color = {name: default_colors[i % len(default_colors)] for i, name in enumerate(geom_names)}

        ratio_plot_types = {
            self.PLOT_ANGULAR_RESOLUTION_VS_ZENITH,
            self.PLOT_ANGULAR_RESOLUTION_VS_ENERGY,
            self.PLOT_ENERGY_RESOLUTION_VS_ENERGY,
            self.PLOT_POINTSOURCE_FOM_VS_ENERGY,
        }

        if ratio_baseline_geometry is not None:
            requested_baseline = str(ratio_baseline_geometry)
            if requested_baseline not in geom_color:
                print(
                    "Warning: ratio_baseline_geometry='{}' not found in geometries: {}".format(
                        requested_baseline,
                        ", ".join(geom_names),
                    )
                )

        def _baseline_name() -> Optional[str]:
            if ratio_baseline_geometry is None:
                return None
            baseline = str(ratio_baseline_geometry)
            return baseline if baseline in geom_color else None

        for plot_type in plot_types:
            if plot_type in overlay_plot_types:
                baseline = _baseline_name() if plot_type in ratio_plot_types else None
                if baseline is not None and n_geoms >= 2:
                    fig, (ax, ax_ratio) = plt.subplots(
                        2,
                        1,
                        figsize=(6.5, 5.6),
                        sharex=True,
                        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
                    )
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
                    ax_ratio = None

                any_series = False
                any_fom_series = False

                # Shared y-log toggle across all resolution/FoM-vs overlay plots.
                if plot_type in ratio_plot_types:
                    overlay_resolution_logy = bool(shared_kwargs.get('resolution_logy', False))
                    if not overlay_resolution_logy:
                        overlay_resolution_logy = bool(shared_kwargs.get('resolution_logy_angular', False))
                    if not overlay_resolution_logy:
                        overlay_resolution_logy = bool(
                            shared_kwargs.get('resolution_logy_vs_zenith', shared_kwargs.get('resolution_logy_vs_energy', False))
                        )
                    if not overlay_resolution_logy:
                        overlay_resolution_logy = any(
                            bool(payload.get('resolution_logy', payload.get('resolution_logy_angular', False)))
                            or bool(payload.get('resolution_logy_vs_zenith', payload.get('resolution_logy_vs_energy', False)))
                            for _, payload in geom_items
                        )
                else:
                    overlay_resolution_logy = False

                # For ratio subplot: store y(x) per geometry.
                ratio_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for geom_name, payload in geom_items:
                    payload = dict(payload)
                    payload.update(shared_kwargs)
                    geom_name_str = str(geom_name)

                    # Pull the relevant series.
                    if plot_type in (self.PLOT_LOSS, self.PLOT_UW_LOSS):
                        series = payload.get('loss_history', None)
                        if series is not None:
                            ax.plot(series, label=geom_name_str, color=geom_color.get(geom_name_str, None))
                            any_series = True
                        continue

                    if plot_type == self.PLOT_SNR_HISTORY:
                        series = payload.get('snr_history', None)
                        if series is not None:
                            ax.plot(series, label=geom_name_str, color=geom_color.get(geom_name_str, None))
                            any_series = True
                        continue

                    if plot_type == self.PLOT_LLR_HISTORY:
                        series = payload.get('llr_history', None)
                        if series is not None:
                            pts = payload.get('points', payload.get('points_3d', None))
                            try:
                                npts = len(pts) if pts is not None else None
                            except Exception:
                                npts = None
                            if npts:
                                series = np.array(series) / float(npts)
                            ax.plot(series, label=geom_name_str, color=geom_color.get(geom_name_str, None))
                            any_series = True
                        continue

                    if plot_type == self.PLOT_PARAM_1D:
                        optimize_params = payload.get('optimize_params', [])
                        param_values = payload.get('param_values', {})
                        all_snr = payload.get('all_snr', None)

                        if len(optimize_params) == 1 and all_snr is not None:
                            param_name = optimize_params[0]
                            if param_name in param_values:
                                try:
                                    param_vals = param_values[param_name].clone().detach().cpu().numpy()
                                except Exception:
                                    param_vals = np.array(param_values[param_name])
                                try:
                                    snr_vals = all_snr.clone().detach().cpu().numpy()
                                except Exception:
                                    snr_vals = np.array(all_snr)
                                sort_idx = np.argsort(param_vals)
                                ax.plot(
                                    param_vals[sort_idx],
                                    snr_vals[sort_idx],
                                    marker='o',
                                    linewidth=2,
                                    label=geom_name_str,
                                    color=geom_color.get(geom_name_str, None),
                                )
                                any_series = True
                        continue

                    if plot_type in (self.PLOT_ANGULAR_RESOLUTION, self.PLOT_ENERGY_RESOLUTION, self.PLOT_POINTSOURCE_FOM):
                        uw_loss_dict = payload.get('uw_loss_dict', None)
                        if isinstance(uw_loss_dict, dict):
                            if plot_type == self.PLOT_ANGULAR_RESOLUTION:
                                series = uw_loss_dict.get('angular_resolution_loss', None)
                            elif plot_type == self.PLOT_ENERGY_RESOLUTION:
                                series = uw_loss_dict.get('energy_resolution_loss', None)
                            else:
                                series = uw_loss_dict.get('pointsource_fom_loss', None)
                                if series is None:
                                    series = uw_loss_dict.get('effective_area_resolution_loss', None)
                                if series is None:
                                    series = uw_loss_dict.get('pointsource_fom', None)
                            if series is not None:
                                series = np.array(series)
                                if plot_type == self.PLOT_ANGULAR_RESOLUTION:
                                    series = series * 180.0
                                ax.plot(series, linewidth=2, label=geom_name_str, color=geom_color.get(geom_name_str, None))
                                any_series = True
                        continue

                    if plot_type == self.PLOT_ANGULAR_RESOLUTION_VS_ZENITH:
                        resolution_per_event = payload.get('resolution_per_event', None)
                        resolution_params = payload.get('resolution_params', None)
                        n_bins = payload.get('n_zenith_bins', 10)
                        resolution_stat = payload.get('resolution_stat', None)
                        if resolution_stat is None and bool(payload.get('resolution_use_mean', False)):
                            resolution_stat = 'mean'
                        resolution_stat = str(resolution_stat).lower() if resolution_stat is not None else 'median'
                        if resolution_stat not in ('median', 'mean', 'fom'):
                            resolution_stat = 'median'
                        resolution_use_fom = bool(payload.get('resolution_use_fom', False)) or resolution_stat == 'fom'
                        if resolution_use_fom:
                            resolution_stat = 'fom'
                        resolution_fom_min_resolution = payload.get('resolution_fom_min_resolution', 1e-12)
                        show_resolution_ci = bool(payload.get('show_resolution_ci', False))
                        resolution_ci_percentiles = payload.get('resolution_ci_percentiles', None)
                        resolution_ci_level = payload.get('resolution_ci_level', None)
                        zenith_range = payload.get('zenith_range', None)
                        zenith_range_deg = payload.get('zenith_range_deg', None)
                        resolution_logy = bool(payload.get('resolution_logy', payload.get('resolution_logy_angular', False)))
                        min_ang_res = payload.get('min_angular_resolution', None)
                        max_ang_res = payload.get('max_angular_resolution', None)
                        if not resolution_logy:
                            resolution_logy = bool(payload.get('resolution_logy_vs_zenith', payload.get('resolution_logy_vs_energy', False)))

                        if resolution_per_event is not None and resolution_params is not None:
                            try:
                                res_values = resolution_per_event.clone().detach().cpu().numpy()
                            except Exception:
                                res_values = np.array(resolution_per_event)

                            zenith_values = []
                            for event_params in resolution_params:
                                if isinstance(event_params, dict) and 'zenith' in event_params:
                                    zenith = event_params['zenith']
                                    try:
                                        zenith_values.append(float(zenith.detach().cpu().item()))
                                    except Exception:
                                        try:
                                            zenith_values.append(float(zenith))
                                        except Exception:
                                            pass
                            zenith_values = np.array(zenith_values)

                            valid_mask = np.isfinite(res_values) & np.isfinite(zenith_values)
                            res_values = np.array(res_values)[valid_mask]
                            zenith_values = zenith_values[valid_mask]

                            if len(res_values) > 0 and len(zenith_values) > 0:
                                zenith_deg = np.rad2deg(zenith_values)
                                # Optional zenith range restriction.
                                zmin_deg, zmax_deg = 0.0, 180.0
                                if zenith_range_deg is not None and len(zenith_range_deg) == 2:
                                    try:
                                        zmin_deg, zmax_deg = float(zenith_range_deg[0]), float(zenith_range_deg[1])
                                    except Exception:
                                        zmin_deg, zmax_deg = 0.0, 180.0
                                elif zenith_range is not None and len(zenith_range) == 2:
                                    try:
                                        zmin_deg = float(np.rad2deg(float(zenith_range[0])))
                                        zmax_deg = float(np.rad2deg(float(zenith_range[1])))
                                    except Exception:
                                        zmin_deg, zmax_deg = 0.0, 180.0
                                if zmax_deg < zmin_deg:
                                    zmin_deg, zmax_deg = zmax_deg, zmin_deg

                                range_mask = (zenith_deg >= zmin_deg) & (zenith_deg <= zmax_deg)
                                zenith_deg = zenith_deg[range_mask]
                                res_values = np.array(res_values)[range_mask]

                                if resolution_logy:
                                    pos_mask = np.array(res_values) > 0
                                    zenith_deg = zenith_deg[pos_mask]
                                    res_values = np.array(res_values)[pos_mask]

                                bin_edges = np.linspace(zmin_deg, zmax_deg, int(n_bins) + 1)
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                                bin_medians = []
                                band_lower = []
                                band_upper = []
                                fom_errors = []
                                bin_counts = []
                                for i in range(int(n_bins)):
                                    mask = (zenith_deg >= bin_edges[i]) & (zenith_deg < bin_edges[i + 1])
                                    if mask.sum() > 0:
                                        vals = np.array(res_values[mask], dtype=float)
                                        if resolution_use_fom:
                                            center_val, fom_err = self._compute_fom_from_resolution(
                                                vals,
                                                min_resolution=resolution_fom_min_resolution,
                                            )
                                            bin_medians.append(center_val)
                                            fom_errors.append(fom_err)
                                            if np.isfinite(center_val) and np.isfinite(fom_err):
                                                band_lower.append(center_val - fom_err)
                                                band_upper.append(center_val + fom_err)
                                            else:
                                                band_lower.append(np.nan)
                                                band_upper.append(np.nan)
                                        elif resolution_stat == 'mean':
                                            center_val = float(np.nanmean(vals))
                                            spread_val = float(np.nanstd(vals))
                                            bin_medians.append(center_val)
                                            fom_errors.append(np.nan)
                                        else:
                                            center_val = float(np.nanmedian(vals))
                                            spread_val = np.nan
                                            bin_medians.append(center_val)
                                            fom_errors.append(np.nan)
                                        if (not resolution_use_fom) and show_resolution_ci:
                                            if resolution_stat == 'mean':
                                                lo = center_val - 2.0 * spread_val
                                                hi = center_val + 2.0 * spread_val
                                                if min_ang_res is not None:
                                                    lo = max(float(min_ang_res), lo)
                                                if max_ang_res is not None:
                                                    hi = min(float(max_ang_res), hi)
                                                band_lower.append(lo)
                                                band_upper.append(float(hi))
                                            else:
                                                q_lo = None
                                                q_hi = None
                                                if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                                    try:
                                                        q_lo = float(resolution_ci_percentiles[0])
                                                        q_hi = float(resolution_ci_percentiles[1])
                                                    except Exception:
                                                        q_lo, q_hi = None, None
                                                if q_lo is None or q_hi is None:
                                                    if resolution_ci_level is not None:
                                                        try:
                                                            lvl = float(resolution_ci_level)
                                                            lvl = float(np.clip(lvl, 0.0, 1.0))
                                                            alpha = 0.5 * (1.0 - lvl)
                                                            q_lo = 100.0 * alpha
                                                            q_hi = 100.0 * (1.0 - alpha)
                                                        except Exception:
                                                            q_lo, q_hi = 16.0, 84.0
                                                    else:
                                                        q_lo, q_hi = 16.0, 84.0
                                                if q_hi < q_lo:
                                                    q_lo, q_hi = q_hi, q_lo
                                                resid = vals - center_val
                                                band_lower.append(center_val + np.nanpercentile(resid, q_lo))
                                                band_upper.append(center_val + np.nanpercentile(resid, q_hi))
                                        else:
                                            if not resolution_use_fom:
                                                band_lower.append(np.nan)
                                                band_upper.append(np.nan)
                                        bin_counts.append(int(mask.sum()))
                                    else:
                                        bin_medians.append(np.nan)
                                        band_lower.append(np.nan)
                                        band_upper.append(np.nan)
                                        fom_errors.append(np.nan)
                                        bin_counts.append(0)

                                bin_medians = np.array(bin_medians)
                                band_lower = np.array(band_lower)
                                band_upper = np.array(band_upper)
                                fom_errors = np.array(fom_errors)
                                bin_counts = np.array(bin_counts)
                                if min_ang_res is not None or max_ang_res is not None:
                                    lo_lim = -np.inf
                                    hi_lim = np.inf
                                    try:
                                        if min_ang_res is not None:
                                            lo_lim = float(min_ang_res)
                                    except Exception:
                                        lo_lim = -np.inf
                                    try:
                                        if max_ang_res is not None:
                                            hi_lim = float(max_ang_res)
                                    except Exception:
                                        hi_lim = np.inf
                                    if np.isfinite(lo_lim) and np.isfinite(hi_lim) and hi_lim < lo_lim:
                                        lo_lim, hi_lim = hi_lim, lo_lim

                                    bin_medians = np.clip(bin_medians, lo_lim, hi_lim)
                                    if show_resolution_ci or resolution_use_fom:
                                        band_lower = np.clip(band_lower, lo_lim, hi_lim)
                                        band_upper = np.clip(band_upper, lo_lim, hi_lim)
                                        band_lower = np.minimum(band_lower, band_upper)
                                valid_bins = np.isfinite(bin_medians)
                                if np.any(valid_bins):
                                    xvals = np.array(bin_centers)[valid_bins]
                                    yvals = np.array(bin_medians)[valid_bins]
                                    ratio_cache[geom_name_str] = (xvals, yvals)
                                    if resolution_use_fom:
                                        any_fom_series = True
                                        valid_err = valid_bins & np.isfinite(fom_errors)
                                        if np.any(valid_err):
                                            ax.errorbar(
                                                bin_centers[valid_err],
                                                bin_medians[valid_err],
                                                yerr=fom_errors[valid_err],
                                                fmt='o-',
                                                linewidth=2,
                                                markersize=6,
                                                capsize=3,
                                                label=geom_name_str,
                                                color=geom_color.get(geom_name_str, None),
                                            )
                                        else:
                                            line = ax.plot(
                                                bin_centers[valid_bins],
                                                bin_medians[valid_bins],
                                                'o-',
                                                linewidth=2,
                                                markersize=6,
                                                label=geom_name_str,
                                                color=geom_color.get(geom_name_str, None),
                                            )[0]
                                    else:
                                        line = ax.plot(
                                            bin_centers[valid_bins],
                                            bin_medians[valid_bins],
                                            'o-',
                                            linewidth=2,
                                            markersize=6,
                                            label=geom_name_str,
                                            color=geom_color.get(geom_name_str, None),
                                        )[0]

                                    if (not resolution_use_fom) and show_resolution_ci:
                                        valid_band = valid_bins & np.isfinite(band_lower) & np.isfinite(band_upper)
                                        if np.any(valid_band):
                                            ax.plot(
                                                bin_centers[valid_band],
                                                band_lower[valid_band],
                                                linestyle='--',
                                                linewidth=1.5,
                                                color=line.get_color(),
                                                alpha=0.8,
                                                zorder=1,
                                            )
                                            ax.plot(
                                                bin_centers[valid_band],
                                                band_upper[valid_band],
                                                linestyle='--',
                                                linewidth=1.5,
                                                color=line.get_color(),
                                                alpha=0.8,
                                                zorder=1,
                                            )

                                    if resolution_logy:
                                        ax.set_yscale('log')

                                any_series = True
                        continue

                    if plot_type == self.PLOT_ANGULAR_RESOLUTION_VS_ENERGY:
                        resolution_per_event = payload.get('resolution_per_event', None)
                        resolution_params = payload.get('resolution_params', None)
                        n_bins = payload.get('n_energy_bins', 10)
                        resolution_stat = payload.get('resolution_stat', None)
                        if resolution_stat is None and bool(payload.get('resolution_use_mean', False)):
                            resolution_stat = 'mean'
                        resolution_stat = str(resolution_stat).lower() if resolution_stat is not None else 'median'
                        if resolution_stat not in ('median', 'mean', 'fom'):
                            resolution_stat = 'median'
                        resolution_use_fom = bool(payload.get('resolution_use_fom', False)) or resolution_stat == 'fom'
                        if resolution_use_fom:
                            resolution_stat = 'fom'
                        resolution_fom_min_resolution = payload.get('resolution_fom_min_resolution', 1e-12)
                        show_resolution_ci = bool(payload.get('show_resolution_ci', False))
                        resolution_ci_percentiles = payload.get('resolution_ci_percentiles', None)
                        resolution_ci_level = payload.get('resolution_ci_level', None)
                        energy_range = payload.get('energy_range', None)
                        resolution_logy = bool(payload.get('resolution_logy', payload.get('resolution_logy_angular', False)))
                        if not resolution_logy:
                            resolution_logy = bool(payload.get('resolution_logy_vs_zenith', payload.get('resolution_logy_vs_energy', False)))
                        min_ang_res = payload.get('min_angular_resolution', None)
                        max_ang_res = payload.get('max_angular_resolution', None)

                        if resolution_per_event is not None and resolution_params is not None:
                            try:
                                res_values = resolution_per_event.clone().detach().cpu().numpy().flatten()
                            except Exception:
                                res_values = np.array(resolution_per_event).flatten()

                            energy_values = []
                            for event_params in resolution_params:
                                if isinstance(event_params, dict) and 'energy' in event_params:
                                    energy = event_params['energy']
                                    try:
                                        energy_values.append(float(energy.detach().cpu().item()))
                                    except Exception:
                                        try:
                                            energy_values.append(float(energy))
                                        except Exception:
                                            pass
                            energy_values = np.array(energy_values)

                            valid_mask = np.isfinite(res_values) & np.isfinite(energy_values) & (energy_values > 0)
                            res_values = res_values[valid_mask]
                            energy_values = energy_values[valid_mask]

                            if energy_range is not None and len(energy_range) == 2:
                                try:
                                    emin, emax = float(energy_range[0]), float(energy_range[1])
                                    if emax < emin:
                                        emin, emax = emax, emin
                                    range_mask = (energy_values >= emin) & (energy_values <= emax)
                                    res_values = res_values[range_mask]
                                    energy_values = energy_values[range_mask]
                                except Exception:
                                    pass

                            if resolution_logy:
                                pos_mask = np.array(res_values) > 0
                                res_values = np.array(res_values)[pos_mask]
                                energy_values = np.array(energy_values)[pos_mask]

                            if len(res_values) > 0 and len(energy_values) > 0:
                                log_energy_min = np.log10(energy_values.min())
                                log_energy_max = np.log10(energy_values.max())
                                bin_edges = np.logspace(log_energy_min, log_energy_max, int(n_bins) + 1)
                                bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

                                bin_medians = []
                                band_lower = []
                                band_upper = []
                                fom_errors = []
                                bin_counts = []
                                for i in range(int(n_bins)):
                                    mask = (energy_values >= bin_edges[i]) & (energy_values < bin_edges[i + 1])
                                    if mask.sum() > 0:
                                        vals = np.array(res_values[mask], dtype=float)
                                        if resolution_use_fom:
                                            center_val, fom_err = self._compute_fom_from_resolution(
                                                vals,
                                                min_resolution=resolution_fom_min_resolution,
                                            )
                                            bin_medians.append(center_val)
                                            fom_errors.append(fom_err)
                                            if np.isfinite(center_val) and np.isfinite(fom_err):
                                                band_lower.append(center_val - fom_err)
                                                band_upper.append(center_val + fom_err)
                                            else:
                                                band_lower.append(np.nan)
                                                band_upper.append(np.nan)
                                        elif resolution_stat == 'mean':
                                            center_val = float(np.nanmean(vals))
                                            spread_val = float(np.nanstd(vals))
                                            bin_medians.append(center_val)
                                            fom_errors.append(np.nan)
                                        else:
                                            center_val = float(np.nanmedian(vals))
                                            spread_val = np.nan
                                            bin_medians.append(center_val)
                                            fom_errors.append(np.nan)
                                        if (not resolution_use_fom) and show_resolution_ci:
                                            if resolution_stat == 'mean':
                                                lo = center_val - 2.0 * spread_val
                                                hi = center_val + 2.0 * spread_val
                                                if min_ang_res is not None:
                                                    lo = float(max(lo, min_ang_res))
                                                if max_ang_res is not None:    
                                                    hi = float(min(hi, max_ang_res))
                                                band_lower.append(lo)
                                                band_upper.append(float(hi))
                                            else:
                                                q_lo = None
                                                q_hi = None
                                                if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                                    try:
                                                        q_lo = float(resolution_ci_percentiles[0])
                                                        q_hi = float(resolution_ci_percentiles[1])
                                                    except Exception:
                                                        q_lo, q_hi = None, None
                                                if q_lo is None or q_hi is None:
                                                    if resolution_ci_level is not None:
                                                        try:
                                                            lvl = float(resolution_ci_level)
                                                            lvl = float(np.clip(lvl, 0.0, 1.0))
                                                            alpha = 0.5 * (1.0 - lvl)
                                                            q_lo = 100.0 * alpha
                                                            q_hi = 100.0 * (1.0 - alpha)
                                                        except Exception:
                                                            q_lo, q_hi = 16.0, 84.0
                                                    else:
                                                        q_lo, q_hi = 16.0, 84.0
                                                if q_hi < q_lo:
                                                    q_lo, q_hi = q_hi, q_lo
                                                resid = vals - center_val
                                                band_lower.append(center_val + np.nanpercentile(resid, q_lo))
                                                band_upper.append(center_val + np.nanpercentile(resid, q_hi))
                                        else:
                                            if not resolution_use_fom:
                                                band_lower.append(np.nan)
                                                band_upper.append(np.nan)
                                        bin_counts.append(int(mask.sum()))
                                    else:
                                        bin_medians.append(np.nan)
                                        band_lower.append(np.nan)
                                        band_upper.append(np.nan)
                                        fom_errors.append(np.nan)
                                        bin_counts.append(0)

                                bin_medians = np.array(bin_medians)
                                band_lower = np.array(band_lower)
                                band_upper = np.array(band_upper)
                                fom_errors = np.array(fom_errors)
                                bin_counts = np.array(bin_counts)
                                if min_ang_res is not None or max_ang_res is not None:
                                    lo_lim = -np.inf
                                    hi_lim = np.inf
                                    try:
                                        if min_ang_res is not None:
                                            lo_lim = float(min_ang_res)
                                    except Exception:
                                        lo_lim = -np.inf
                                    try:
                                        if max_ang_res is not None:
                                            hi_lim = float(max_ang_res)
                                    except Exception:
                                        hi_lim = np.inf
                                    if np.isfinite(lo_lim) and np.isfinite(hi_lim) and hi_lim < lo_lim:
                                        lo_lim, hi_lim = hi_lim, lo_lim

                                    bin_medians = np.clip(bin_medians, lo_lim, hi_lim)
                                    if show_resolution_ci or resolution_use_fom:
                                        band_lower = np.clip(band_lower, lo_lim, hi_lim)
                                        band_upper = np.clip(band_upper, lo_lim, hi_lim)
                                        band_lower = np.minimum(band_lower, band_upper)
                                x_plot = np.log10(bin_centers)
                                valid_bins = np.isfinite(bin_medians)
                                if np.any(valid_bins):
                                    ratio_cache[geom_name_str] = (
                                        np.array(x_plot)[valid_bins],
                                        np.array(bin_medians)[valid_bins],
                                    )
                                    if resolution_use_fom:
                                        any_fom_series = True
                                        valid_err = valid_bins & np.isfinite(fom_errors)
                                        if np.any(valid_err):
                                            ax.errorbar(
                                                x_plot[valid_err],
                                                bin_medians[valid_err],
                                                yerr=fom_errors[valid_err],
                                                fmt='o-',
                                                linewidth=2,
                                                markersize=6,
                                                capsize=3,
                                                label=geom_name_str,
                                                color=geom_color.get(geom_name_str, None),
                                            )
                                        else:
                                            line = ax.plot(
                                                x_plot[valid_bins],
                                                bin_medians[valid_bins],
                                                'o-',
                                                linewidth=2,
                                                markersize=6,
                                                label=geom_name_str,
                                                color=geom_color.get(geom_name_str, None),
                                            )[0]
                                    else:
                                        line = ax.plot(
                                            x_plot[valid_bins],
                                            bin_medians[valid_bins],
                                            'o-',
                                            linewidth=2,
                                            markersize=6,
                                            label=geom_name_str,
                                            color=geom_color.get(geom_name_str, None),
                                        )[0]

                                    if (not resolution_use_fom) and show_resolution_ci:
                                        valid_band = valid_bins & np.isfinite(band_lower) & np.isfinite(band_upper)
                                        if np.any(valid_band):
                                            ax.plot(
                                                x_plot[valid_band],
                                                band_lower[valid_band],
                                                linestyle='--',
                                                linewidth=1.5,
                                                color=line.get_color(),
                                                alpha=0.8,
                                                zorder=1,
                                            )
                                            ax.plot(
                                                x_plot[valid_band],
                                                band_upper[valid_band],
                                                linestyle='--',
                                                linewidth=1.5,
                                                color=line.get_color(),
                                                alpha=0.8,
                                                zorder=1,
                                            )

                                    if resolution_logy:
                                        ax.set_yscale('log')

                                any_series = True
                        continue

                    if plot_type == self.PLOT_ENERGY_RESOLUTION_VS_ENERGY:
                        resolution_per_event = payload.get('resolution_per_event', None)
                        resolution_params = payload.get('resolution_params', None)
                        n_bins = payload.get('n_energy_bins', 10)
                        use_relative_energy = payload.get('use_relative_energy', False)
                        resolution_stat = payload.get('resolution_stat', None)
                        if resolution_stat is None and bool(payload.get('resolution_use_mean', False)):
                            resolution_stat = 'mean'
                        resolution_stat = str(resolution_stat).lower() if resolution_stat is not None else 'median'
                        if resolution_stat not in ('median', 'mean', 'fom'):
                            resolution_stat = 'median'
                        resolution_use_fom = bool(payload.get('resolution_use_fom', False)) or resolution_stat == 'fom'
                        if resolution_use_fom:
                            resolution_stat = 'fom'
                        resolution_fom_min_resolution = payload.get('resolution_fom_min_resolution', 1e-12)
                        show_resolution_ci = bool(payload.get('show_resolution_ci', False))
                        resolution_ci_percentiles = payload.get('resolution_ci_percentiles', None)
                        resolution_ci_level = payload.get('resolution_ci_level', None)
                        energy_range = payload.get('energy_range', None)
                        resolution_logy = bool(payload.get('resolution_logy', payload.get('resolution_logy_angular', False)))
                        if not resolution_logy:
                            resolution_logy = bool(payload.get('resolution_logy_vs_zenith', payload.get('resolution_logy_vs_energy', False)))

                        if resolution_per_event is not None and resolution_params is not None:
                            try:
                                res_values = resolution_per_event.clone().detach().cpu().numpy().flatten()
                            except Exception:
                                res_values = np.array(resolution_per_event).flatten()

                            energy_values = []
                            for event_params in resolution_params:
                                if isinstance(event_params, dict) and 'energy' in event_params:
                                    energy = event_params['energy']
                                    try:
                                        energy_values.append(float(energy.detach().cpu().item()))
                                    except Exception:
                                        try:
                                            energy_values.append(float(energy))
                                        except Exception:
                                            pass

                            energy_values = np.array(energy_values)
                            valid_mask = np.isfinite(res_values) & np.isfinite(energy_values) & (energy_values > 0)
                            res_values = res_values[valid_mask]
                            energy_values = energy_values[valid_mask]

                            if energy_range is not None and len(energy_range) == 2:
                                try:
                                    emin, emax = float(energy_range[0]), float(energy_range[1])
                                    if emax < emin:
                                        emin, emax = emax, emin
                                    range_mask = (energy_values >= emin) & (energy_values <= emax)
                                    res_values = res_values[range_mask]
                                    energy_values = energy_values[range_mask]
                                except Exception:
                                    pass

                            if resolution_logy:
                                pos_mask = np.array(res_values) > 0
                                res_values = np.array(res_values)[pos_mask]
                                energy_values = np.array(energy_values)[pos_mask]

                            if len(res_values) > 0 and len(energy_values) > 0:
                                log_energy_min = np.log10(energy_values.min())
                                log_energy_max = np.log10(energy_values.max())
                                bin_edges = np.logspace(log_energy_min, log_energy_max, int(n_bins) + 1)
                                bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

                                bin_medians = []
                                band_lower = []
                                band_upper = []
                                fom_errors = []
                                bin_counts = []
                                for i in range(int(n_bins)):
                                    mask = (energy_values >= bin_edges[i]) & (energy_values < bin_edges[i + 1])
                                    if mask.sum() > 0:
                                        vals = np.array(res_values[mask], dtype=float)
                                        if resolution_use_fom:
                                            center_val, fom_err = self._compute_fom_from_resolution(
                                                vals,
                                                min_resolution=resolution_fom_min_resolution,
                                            )
                                            bin_medians.append(center_val)
                                            fom_errors.append(fom_err)
                                            if np.isfinite(center_val) and np.isfinite(fom_err):
                                                band_lower.append(center_val - fom_err)
                                                band_upper.append(center_val + fom_err)
                                            else:
                                                band_lower.append(np.nan)
                                                band_upper.append(np.nan)
                                        elif resolution_stat == 'mean':
                                            center_val = float(np.nanmean(vals))
                                            spread_val = float(np.nanstd(vals))
                                            bin_medians.append(center_val)
                                            fom_errors.append(np.nan)
                                        else:
                                            center_val = float(np.nanmedian(vals))
                                            spread_val = np.nan
                                            bin_medians.append(center_val)
                                            fom_errors.append(np.nan)
                                        if (not resolution_use_fom) and show_resolution_ci:
                                            if resolution_stat == 'mean':
                                                band_lower.append(center_val - 2.0 * spread_val)
                                                band_upper.append(center_val + 2.0 * spread_val)
                                            else:
                                                q_lo = None
                                                q_hi = None
                                                if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                                    try:
                                                        q_lo = float(resolution_ci_percentiles[0])
                                                        q_hi = float(resolution_ci_percentiles[1])
                                                    except Exception:
                                                        q_lo, q_hi = None, None
                                                if q_lo is None or q_hi is None:
                                                    if resolution_ci_level is not None:
                                                        try:
                                                            lvl = float(resolution_ci_level)
                                                            lvl = float(np.clip(lvl, 0.0, 1.0))
                                                            alpha = 0.5 * (1.0 - lvl)
                                                            q_lo = 100.0 * alpha
                                                            q_hi = 100.0 * (1.0 - alpha)
                                                        except Exception:
                                                            q_lo, q_hi = 16.0, 84.0
                                                    else:
                                                        q_lo, q_hi = 16.0, 84.0
                                                if q_hi < q_lo:
                                                    q_lo, q_hi = q_hi, q_lo
                                                resid = vals - center_val
                                                band_lower.append(center_val + np.nanpercentile(resid, q_lo))
                                                band_upper.append(center_val + np.nanpercentile(resid, q_hi))
                                        else:
                                            if not resolution_use_fom:
                                                band_lower.append(np.nan)
                                                band_upper.append(np.nan)
                                        bin_counts.append(int(mask.sum()))
                                    else:
                                        bin_medians.append(np.nan)
                                        band_lower.append(np.nan)
                                        band_upper.append(np.nan)
                                        fom_errors.append(np.nan)
                                        bin_counts.append(0)

                                bin_medians = np.array(bin_medians)
                                band_lower = np.array(band_lower)
                                band_upper = np.array(band_upper)
                                fom_errors = np.array(fom_errors)
                                bin_counts = np.array(bin_counts)

                                valid_bins = np.isfinite(bin_medians)
                                if resolution_logy:
                                    valid_bins = valid_bins & (np.array(bin_medians) > 0)
                                if np.any(valid_bins):
                                    ratio_cache[geom_name_str] = (
                                        np.array(bin_centers)[valid_bins],
                                        np.array(bin_medians)[valid_bins],
                                    )
                                    if resolution_use_fom:
                                        any_fom_series = True
                                        valid_err = valid_bins & np.isfinite(fom_errors)
                                        if np.any(valid_err):
                                            ax.errorbar(
                                                bin_centers[valid_err],
                                                bin_medians[valid_err],
                                                yerr=fom_errors[valid_err],
                                                fmt='o-',
                                                linewidth=2,
                                                markersize=6,
                                                capsize=3,
                                                label=geom_name_str,
                                                color=geom_color.get(geom_name_str, None),
                                            )
                                        else:
                                            line = ax.plot(
                                                bin_centers[valid_bins],
                                                bin_medians[valid_bins],
                                                'o-',
                                                linewidth=2,
                                                markersize=6,
                                                label=geom_name_str,
                                                color=geom_color.get(geom_name_str, None),
                                            )[0]
                                    else:
                                        line = ax.plot(
                                            bin_centers[valid_bins],
                                            bin_medians[valid_bins],
                                            'o-',
                                            linewidth=2,
                                            markersize=6,
                                            label=geom_name_str,
                                            color=geom_color.get(geom_name_str, None),
                                        )[0]

                                    if (not resolution_use_fom) and show_resolution_ci:
                                        valid_band = valid_bins & np.isfinite(band_lower) & np.isfinite(band_upper)
                                        if np.any(valid_band):
                                            ax.plot(
                                                bin_centers[valid_band],
                                                band_lower[valid_band],
                                                linestyle='--',
                                                linewidth=1.5,
                                                color=line.get_color(),
                                                alpha=0.8,
                                                zorder=1,
                                            )
                                            ax.plot(
                                                bin_centers[valid_band],
                                                band_upper[valid_band],
                                                linestyle='--',
                                                linewidth=1.5,
                                                color=line.get_color(),
                                                alpha=0.8,
                                                zorder=1,
                                            )

                                any_series = True
                        continue

                    if plot_type == self.PLOT_POINTSOURCE_FOM_VS_ENERGY:
                        resolution_per_event = payload.get('resolution_per_event', None)
                        effective_area_per_event = payload.get('effective_area_per_event', None)
                        event_params = payload.get('resolution_params', None)
                        if event_params is None:
                            event_params = payload.get('effective_area_params', None)
                        if event_params is None:
                            event_params = payload.get('signal_event_params', None)
                        n_bins = payload.get('n_energy_bins', 10)
                        energy_range = payload.get('energy_range', None)
                        fom_min_resolution = payload.get('resolution_fom_min_resolution', 1e-12)
                        resolution_logy = bool(payload.get('resolution_logy', payload.get('resolution_logy_angular', False)))
                        if not resolution_logy:
                            resolution_logy = bool(payload.get('resolution_logy_vs_zenith', payload.get('resolution_logy_vs_energy', False)))

                        if (
                            resolution_per_event is not None
                            and effective_area_per_event is not None
                            and event_params is not None
                        ):
                            try:
                                res_values = resolution_per_event.clone().detach().cpu().numpy().flatten()
                            except Exception:
                                res_values = np.array(resolution_per_event).flatten()

                            try:
                                aeff_values = effective_area_per_event.clone().detach().cpu().numpy().flatten()
                            except Exception:
                                aeff_values = np.array(effective_area_per_event).flatten()

                            energy_values = []
                            for ep in event_params:
                                if isinstance(ep, dict) and 'energy' in ep:
                                    energy = ep['energy']
                                    try:
                                        energy_values.append(float(energy.detach().cpu().item()))
                                    except Exception:
                                        try:
                                            energy_values.append(float(energy))
                                        except Exception:
                                            pass
                            energy_values = np.array(energy_values)

                            n = min(len(res_values), len(aeff_values), len(energy_values))
                            if n > 0:
                                res_values = res_values[:n]
                                aeff_values = aeff_values[:n]
                                energy_values = energy_values[:n]

                            valid_mask = (
                                np.isfinite(res_values)
                                & np.isfinite(aeff_values)
                                & np.isfinite(energy_values)
                                & (energy_values > 0)
                            )
                            res_values = res_values[valid_mask]
                            aeff_values = aeff_values[valid_mask]
                            energy_values = energy_values[valid_mask]

                            if energy_range is not None and len(energy_range) == 2:
                                try:
                                    emin, emax = float(energy_range[0]), float(energy_range[1])
                                    if emax < emin:
                                        emin, emax = emax, emin
                                    range_mask = (energy_values >= emin) & (energy_values <= emax)
                                    res_values = res_values[range_mask]
                                    aeff_values = aeff_values[range_mask]
                                    energy_values = energy_values[range_mask]
                                except Exception:
                                    pass

                            if len(res_values) > 0 and len(aeff_values) > 0 and len(energy_values) > 0:
                                log_energy_min = np.log10(energy_values.min())
                                log_energy_max = np.log10(energy_values.max())
                                bin_edges = np.logspace(log_energy_min, log_energy_max, int(n_bins) + 1)
                                bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

                                bin_fom = []
                                bin_fom_err = []
                                for i in range(int(n_bins)):
                                    mask = (energy_values >= bin_edges[i]) & (energy_values < bin_edges[i + 1])
                                    if mask.sum() > 0:
                                        fval, ferr = self._compute_pointsource_fom_from_resolution_and_aeff(
                                            res_values[mask],
                                            aeff_values[mask],
                                            min_resolution=fom_min_resolution,
                                        )
                                        bin_fom.append(fval)
                                        bin_fom_err.append(ferr)
                                    else:
                                        bin_fom.append(np.nan)
                                        bin_fom_err.append(np.nan)

                                bin_fom = np.array(bin_fom)
                                bin_fom_err = np.array(bin_fom_err)
                                x_plot = np.log10(bin_centers)
                                valid_bins = np.isfinite(bin_fom)
                                if resolution_logy:
                                    valid_bins = valid_bins & (bin_fom > 0)
                                if np.any(valid_bins):
                                    ratio_cache[geom_name_str] = (
                                        np.array(x_plot)[valid_bins],
                                        np.array(bin_fom)[valid_bins],
                                    )
                                    any_fom_series = True

                                    valid_err = valid_bins & np.isfinite(bin_fom_err)
                                    if np.any(valid_err):
                                        ax.errorbar(
                                            x_plot[valid_err],
                                            bin_fom[valid_err],
                                            yerr=bin_fom_err[valid_err],
                                            fmt='o-',
                                            linewidth=2,
                                            markersize=6,
                                            capsize=3,
                                            label=geom_name_str,
                                            color=geom_color.get(geom_name_str, None),
                                        )
                                    else:
                                        ax.plot(
                                            x_plot[valid_bins],
                                            bin_fom[valid_bins],
                                            'o-',
                                            linewidth=2,
                                            markersize=6,
                                            label=geom_name_str,
                                            color=geom_color.get(geom_name_str, None),
                                        )
                                    any_series = True
                        continue

                # Add ratio subplot if requested for resolution-vs-* plots.
                if ax_ratio is not None and baseline is not None:
                    base_xy = ratio_cache.get(str(baseline), None)
                    if base_xy is None:
                        ax_ratio.axis('off')
                    else:
                        base_x, base_y = base_xy
                        base_x = np.array(base_x, dtype=float)
                        base_y = np.array(base_y, dtype=float)
                        valid_base = np.isfinite(base_x) & np.isfinite(base_y)
                        base_x = base_x[valid_base]
                        base_y = base_y[valid_base]
                        if len(base_x) < 2:
                            ax_ratio.axis('off')
                        else:
                            sort_idx = np.argsort(base_x)
                            bx = base_x[sort_idx]
                            by = base_y[sort_idx]

                            for name, (xvals, yvals) in ratio_cache.items():
                                if name == str(baseline):
                                    continue
                                xs = np.array(xvals, dtype=float)
                                ys = np.array(yvals, dtype=float)
                                valid = np.isfinite(xs) & np.isfinite(ys)
                                xs = xs[valid]
                                ys = ys[valid]
                                if len(xs) < 2:
                                    continue
                                in_range = (xs >= float(np.nanmin(bx))) & (xs <= float(np.nanmax(bx)))
                                xs = xs[in_range]
                                ys = ys[in_range]
                                if len(xs) < 2:
                                    continue
                                by_interp = np.interp(xs, bx, by)
                                denom = np.where(np.abs(by_interp) > 0, by_interp, np.nan)
                                ratio = ys / denom

                                ax_ratio.plot(
                                    xs,
                                    ratio,
                                    'o-',
                                    linewidth=1.6,
                                    markersize=4,
                                    color=geom_color.get(name, None),
                                )

                            ax_ratio.axhline(1.0, color='0.3', linewidth=1.0, linestyle='--', alpha=0.8)
                            ax_ratio.set_ylabel('ratio', fontsize=9)
                            ax_ratio.tick_params(axis='both', labelsize=8)

                if not any_series:
                    ax.text(
                        0.5,
                        0.5,
                        f"{plot_type}: no data available",
                        ha='center',
                        va='center',
                        transform=ax.transAxes,
                    )

                # Titles/labels similar to _create_plot.
                if plot_type == self.PLOT_LOSS:
                    ax.set_title(f"Loss (Iteration {iteration})")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss")
                    if loss_type == 'rbf' and any_series:
                        ax.set_yscale('log')
                elif plot_type == self.PLOT_UW_LOSS:
                    ax.set_title(f"(unweighted) Loss (Iteration {iteration})")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss")
                    if loss_type == 'rbf' and any_series:
                        ax.set_yscale('log')
                elif plot_type == self.PLOT_SNR_HISTORY:
                    ax.set_title("Total Signal-to-Noise Ratio")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("SNR")
                elif plot_type == self.PLOT_LLR_HISTORY:
                    ax.set_title("Mean Log-Likelihood Ratio")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("LLR")
                elif plot_type == self.PLOT_PARAM_1D:
                    ax.set_title("SNR vs Parameter")
                    ax.set_xlabel("Parameter")
                    ax.set_ylabel("SNR")
                elif plot_type == self.PLOT_ANGULAR_RESOLUTION:
                    ax.set_title('Angular Resolution History')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Angular Resolution (degrees)')
                    ax.grid(True, alpha=0.3)
                elif plot_type == self.PLOT_ENERGY_RESOLUTION:
                    ax.set_title('Energy Resolution History')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Energy Resolution [GeV]')
                    ax.grid(True, alpha=0.3)
                elif plot_type == self.PLOT_POINTSOURCE_FOM:
                    ax.set_title('Pointsource FoM History')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Pointsource FoM')
                    ax.grid(True, alpha=0.3)
                elif plot_type == self.PLOT_ANGULAR_RESOLUTION_VS_ZENITH:
                    ax.set_title('Angular FOM vs Zenith' if any_fom_series else 'Angular Resolution vs Zenith')
                    ax.set_xlabel('Zenith Angle (degrees)')
                    ax.set_ylabel('FOM (rad$^{-1}$)' if any_fom_series else 'Angular Resolution (radians)')
                    ax.grid(True, alpha=0.3)
                    if overlay_resolution_logy:
                        ax.set_yscale('log')
                    if not any_fom_series:
                        try:
                            ax2 = ax.twinx()
                            ax2.set_ylabel('Angular Resolution (degrees)')
                            ax2.set_yscale(ax.get_yscale())
                            y0, y1 = ax.get_ylim()
                            ax2.set_ylim(np.rad2deg(y0), np.rad2deg(y1))
                            ax2.tick_params(axis='y')
                        except Exception:
                            pass
                elif plot_type == self.PLOT_ANGULAR_RESOLUTION_VS_ENERGY:
                    ax.set_title('Angular FOM vs log$_{10}$(Energy)' if any_fom_series else 'Angular Resolution vs log$_{10}$(Energy)')
                    ax.set_xlabel('log$_{10}$(Energy / GeV)')
                    ax.set_ylabel('FOM (rad$^{-1}$)' if any_fom_series else 'Angular Resolution (radians)')
                    ax.grid(True, alpha=0.3)
                    if overlay_resolution_logy:
                        ax.set_yscale('log')
                    if not any_fom_series:
                        try:
                            ax2 = ax.twinx()
                            ax2.set_ylabel('Angular Resolution (degrees)')
                            ax2.set_yscale(ax.get_yscale())
                            y0, y1 = ax.get_ylim()
                            ax2.set_ylim(np.rad2deg(y0), np.rad2deg(y1))
                            ax2.tick_params(axis='y')
                        except Exception:
                            pass
                elif plot_type == self.PLOT_ENERGY_RESOLUTION_VS_ENERGY:
                    ax.set_title('Energy FOM vs Energy' if any_fom_series else 'Energy Resolution vs Energy')
                    ax.set_xlabel('Energy (GeV)')
                    ax.set_ylabel('FOM (resolution$^{-1}$)' if any_fom_series else 'Energy Resolution (GeV)')
                    ax.grid(True, alpha=0.3, which='both')
                    ax.set_xscale('log')
                    if overlay_resolution_logy:
                        ax.set_yscale('log')
                elif plot_type == self.PLOT_POINTSOURCE_FOM_VS_ENERGY:
                    ax.set_title('Pointsource FoM vs log$_{10}$(Energy)')
                    ax.set_xlabel('log$_{10}$(Energy / GeV)')
                    ax.set_ylabel('Pointsource FoM')
                    ax.grid(True, alpha=0.3)
                    if overlay_resolution_logy:
                        ax.set_yscale('log')
                elif plot_type == self.PLOT_LOSS_COMPONENTS:
                    ax.set_title('Loss Components')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Loss Value')
                    ax.grid(True, alpha=0.3)
                elif plot_type == self.PLOT_UW_LOSS_COMPONENTS:
                    ax.set_title('Unweighted Loss Components')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Loss Value')
                    ax.grid(True, alpha=0.3)
                elif plot_type == self.PLOT_ALM_MU:
                    ax.set_title('ALM Penalty Parameters (μ) History')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('μ (Penalty Parameter)')
                    ax.grid(True, alpha=0.3)
                    if any_series:
                        ax.set_yscale('log')
                elif plot_type == self.PLOT_ALM_LAMBDA:
                    ax.set_title('ALM Lagrange Multipliers (λ) History')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('λ (Lagrange Multiplier)')
                    ax.grid(True, alpha=0.3)

                # In ratio mode we use a shared-x two-row layout; the x-label should live
                # on the bottom axis (ratio subplot) to avoid being hidden/overlapping.
                if ax_ratio is not None:
                    try:
                        if bool(getattr(ax_ratio, 'axison', True)):
                            xlabel = ax.get_xlabel()
                            if isinstance(xlabel, str) and xlabel.strip():
                                ax_ratio.set_xlabel(xlabel)
                                ax.set_xlabel('')
                    except Exception:
                        pass

                if any_series:
                    ax.legend(fontsize=8)

                try:
                    fig.tight_layout()
                except Exception:
                    pass
                for ax3d in fig.axes:
                    if getattr(ax3d, 'name', '') == '3d':
                        shift_left = float(getattr(ax3d, '_plot3d_shift_left', 0.0))
                        if shift_left != 0.0:
                            pos = ax3d.get_position()
                            ax3d.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])
                plt.show()
                continue

            # Everything else: one subplot per geometry.
            # Keep up to 3 geometries per row; don't allocate empty 3rd column for 1-2 geometries.
            ncols = 3 if n_geoms >= 3 else int(n_geoms)
            ncols = max(1, int(ncols))
            nrows = (n_geoms + ncols - 1) // ncols


            # Everything else: one subplot per geometry.
            # Keep up to 3 geometries per row; don't allocate empty 3rd column for 1-2 geometries.
            ncols = 3 if n_geoms >= 3 else int(n_geoms)
            ncols = max(1, int(ncols))
            nrows = (n_geoms + ncols - 1) // ncols
            nrows = max(1, int(nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
            axes_flat = axes.flatten()

            for idx, (geom_name, payload) in enumerate(geom_items):
                ax = axes_flat[idx]
                payload = dict(payload)
                payload.update(shared_kwargs)

                per_iter = payload.get('iteration', iteration)
                per_slice_res = payload.get('slice_res', slice_res)
                per_multi_slice = payload.get('multi_slice', multi_slice)
                per_loss_type = payload.get('loss_type', loss_type)

                points = payload.get('points', payload.get('points_3d', None))
                loss_history = payload.get('loss_history', None)
                string_indices = payload.get('string_indices', None)
                points_per_string_list = payload.get('points_per_string_list', None)
                string_xy = payload.get('string_xy', None)

                # Avoid passing base args twice via **payload.
                base_keys = {
                    'iteration',
                    'points',
                    'points_3d',
                    'loss_history',
                    'string_indices',
                    'points_per_string_list',
                    'string_xy',
                    'slice_res',
                    'multi_slice',
                    'loss_type',
                    'plot_types',
                    'make_gif',
                }
                extra = {k: v for k, v in payload.items() if k not in base_keys}

                self._create_plot(
                    plot_type=plot_type,
                    ax=ax,
                    fig=fig,
                    iteration=per_iter,
                    points=points,
                    loss_history=loss_history,
                    string_indices=string_indices,
                    points_per_string_list=points_per_string_list,
                    string_xy=string_xy,
                    slice_res=per_slice_res,
                    multi_slice=per_multi_slice,
                    loss_type=per_loss_type,
                    **extra,
                )

                # Some plot types (3D) swap axes; grab the most recently created axis.
                ax_label = ax
                if plot_type == self.PLOT_3D_POINTS and len(fig.axes) > 0:
                    ax_label = fig.axes[-1]

                # Always annotate geometry name on per-geometry subplots.
                try:
                    ax_label.text(
                        0.02,
                        0.98,
                        str(geom_name),
                        transform=ax_label.transAxes,
                        ha='left',
                        va='top',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.7, edgecolor='none'),
                    )
                except Exception:
                    pass

            # Hide unused axes (if any)
            for j in range(n_geoms, nrows * ncols):
                try:
                    axes_flat[j].axis('off')
                except Exception:
                    pass

            try:
                fig.tight_layout()
            except Exception:
                pass
            for ax3d in fig.axes:
                if getattr(ax3d, 'name', '') == '3d':
                    shift_left = float(getattr(ax3d, '_plot3d_shift_left', 0.0))
                    if shift_left != 0.0:
                        pos = ax3d.get_position()
                        ax3d.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])
            plt.show()
    
    def _create_plot(self, 
                   plot_type: str, 
                   ax: plt.Axes, 
                   fig: plt.Figure, 
                   iteration: int, 
                   points: torch.Tensor = None,
                   points_3d: torch.Tensor = None,
                   loss_history: List[float] = None, 
                   string_indices: Optional[List[int]] = None, 
                   points_per_string_list: Optional[List[int]] = None, 
                   string_xy: Optional[torch.Tensor] = None,
                   slice_res: int = 50, 
                   multi_slice: bool = False, 
                   loss_type: str = 'rbf',
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
        # Backwards-compat: allow callers to pass `points_3d`.
        if points is None and points_3d is not None:
            points = points_3d

        # Safely handle torch tensor inputs by cloning and detaching them
        points = self._safe_tensor_convert(points)
        string_xy = self._safe_tensor_convert(string_xy)
        string_indices = self._safe_tensor_convert(string_indices)
        points_per_string_list = self._safe_tensor_convert(points_per_string_list)

        # Normalize common index/count containers to CPU-native types for plotting.
        if torch.is_tensor(string_indices):
            string_indices = string_indices.detach().cpu().long().tolist()
        elif isinstance(string_indices, np.ndarray):
            string_indices = string_indices.astype(np.int64).tolist()

        if torch.is_tensor(points_per_string_list):
            points_per_string_list = points_per_string_list.detach().cpu().numpy()
        elif points_per_string_list is not None and not isinstance(points_per_string_list, np.ndarray):
            points_per_string_list = np.asarray(points_per_string_list)

        if points_per_string_list is not None:
            points_per_string_list = np.asarray(points_per_string_list).reshape(-1)
            if points_per_string_list.dtype.kind not in ('i', 'u'):
                points_per_string_list = np.rint(points_per_string_list).astype(np.int64)
            else:
                points_per_string_list = points_per_string_list.astype(np.int64, copy=False)
            points_per_string_list = np.clip(points_per_string_list, 0, None)

        if kwargs.get('string_weights') is not None:    
            kwargs['string_weights'] = torch.sigmoid(kwargs['string_weights'].clone())
        # Handle potential torch tensors in common kwargs
        tensor_kwargs = ['string_weights', 'signal_funcs', 'background_funcs', 'test_points', 
                        'llr_per_string', 'signal_llr_per_string', 'background_llr_per_string',
                        'signal_yield_per_string', 'snr_per_string', 'fisher_info_per_string',
                        'local_string_repulsion_penalty_per_string']
        for key in tensor_kwargs:
            if key in kwargs and kwargs.get(key) is not None:
                kwargs[key] = self._safe_tensor_convert(kwargs[key])
        
        # Convert points to numpy for plotting
        points_xyz = points.clone().detach().cpu().numpy()
        geometry_type = kwargs.get('geometry_type', None) # Get geometry_type from kwargs
        
        # Extract zoom_range parameter for contour plots
        zoom_range = kwargs.get('zoom_range', None)
        
        # Helper function to set axis limits based on zoom_range or default domain
        def set_axis_limits(ax_obj):
            if zoom_range is not None:
                ax_obj.set_xlim(-zoom_range, zoom_range)
                ax_obj.set_ylim(-zoom_range, zoom_range)
            else:
                ax_obj.set_xlim(-self.half_domain, self.half_domain)
                ax_obj.set_ylim(-self.half_domain, self.half_domain)
        
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
            # Store desired shift and apply it after tight_layout so it is not reset.
            ax._plot3d_shift_left = float(kwargs.get('plot_3d_shift_left', 0.03))
            
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

                # Size markers by both global density and per-string population.
                # Dense scenes get smaller points; larger strings get slightly larger markers.
                n_total_points = max(1, int(len(points_xyz)))
                n_strings = max(1, int(len(points_per_string_list)))
                base_size = 1800.0 / (n_total_points ** 0.55)
                base_size *= (80.0 / (n_strings + 10.0)) ** 0.25
                base_size = float(np.clip(base_size, 3.0, 24.0))

                string_counts = np.asarray(points_per_string_list, dtype=np.float64)
                mean_count = max(1.0, float(np.mean(np.clip(string_counts, 1.0, None))))
                per_string_scale = np.sqrt(np.clip(string_counts, 1.0, None) / mean_count)
                per_string_sizes = np.clip(base_size * (0.75 + 0.45 * per_string_scale), 2.0, 32.0)
                    
                full_colors = []
                full_alphas = []
                full_sizes = []
                # print("Points per string list:", points_per_string_list)
                for string_num, num_points in enumerate(points_per_string_list):
                    count_i = int(num_points)
                    full_colors.extend([colors[string_num]] * count_i)
                    full_sizes.extend([float(per_string_sizes[string_num])] * count_i)
                    if string_weights is not None:
                        full_alphas.extend([alpha_values[string_num]] * count_i)
                    
                marker_sizes = full_sizes if len(full_sizes) == n_total_points else base_size
                
                ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                          c=full_colors, s=marker_sizes, alpha=full_alphas if string_weights is not None else 0.8)
                
                if string_xy is not None:
                    # Draw vertical lines for strings with alpha based on string weights
                    xy_np = string_xy.clone().detach().cpu().numpy()
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
                n_total_points = max(1, int(len(points_xyz)))
                marker_size = float(np.clip(1800.0 / (n_total_points ** 0.55), 3.0, 24.0))
                ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], 
                          c=points_xyz[:, 2], cmap='rainbow', s=marker_size, alpha=0.8)
            
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
                xy_np = string_xy.clone().detach().cpu().numpy()
                string_weights = kwargs.get('string_weights', None)
                weight_threshold = kwargs.get('weight_threshold', 0.7)
                max_radius = kwargs.get('max_radius', None)
                draw_radius = kwargs.get('draw_radius', False)
                draw_weighted_cylinder = kwargs.get('draw_weighted_cylinder', False)
                
                # Create colormap based on number of points per string
                if points_per_string_list is not None:
                    cmap = plt.cm.viridis
                    norm = Normalize(vmin=min(points_per_string_list), 
                                    vmax=max(points_per_string_list))
                    points_per_string_arr = np.asarray(points_per_string_list)
                    active_mask = points_per_string_arr > 0
                    xy_active = xy_np[active_mask]
                    points_active = points_per_string_arr[active_mask]
                    
                    # Calculate alpha values based on string weights
                    if string_weights is not None:
                        # Apply sigmoid to convert to [0,1] range if not already
                        # print("String weights:", string_weights)
                        string_weight_vals = np.array([string_weights[idx] for idx in string_indices])
                        # if np.any(alpha_vals < 0) or np.any(alpha_vals > 1):
                        #     alpha_vals = 1 / (1 + np.exp(-alpha_vals))
                        # alpha_vals = torch.nn.functional.softplus(torch.tensor(alpha_vals)).detach().cpu().numpy()  # Apply softplus for smoothness
                        # Ensure minimum visibility and filter active strings
                        alpha_vals = np.clip(string_weight_vals[active_mask], 0.05, 1.0)
                        # Handle NaN values if they exist
                        if np.any(np.isnan(alpha_vals)):
                            alpha_vals = np.nan_to_num(alpha_vals, nan=0.5)
                        # print("Alpha values:", alpha_vals)
                        # active_mask = np.array(points_per_string_list) > 0
                        # alpha_vals = alpha_vals[active_mask] if len(alpha_vals) == len(points_per_string_list) else [0.8] * sum(active_mask)
                        weight_mask = string_weight_vals[active_mask] >= weight_threshold
                        # weight_mask = np.array([True]*len(alpha_vals))
                    else:
                        alpha_vals = np.full(len(xy_active), 0.8, dtype=float)
                        weight_mask = np.ones(len(xy_active), dtype=bool)
                    
                    
                    
                    # Plot strings with size proportional to number of points and alpha based on weights
                    if np.any(weight_mask):    
                        xy_weighted = xy_active[weight_mask]
                        points_weighted = points_active[weight_mask]
                        sc = ax.scatter(
                            xy_weighted[:, 0],
                            xy_weighted[:, 1],
                            s=min([40, 30 * 200 / max(1, len(xy_weighted))]),
                            c=points_weighted,
                            cmap=cmap,
                            alpha=alpha_vals[weight_mask],
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
                        alpha_vals = np.full(len(xy_np), 0.8, dtype=float)
                        weight_mask = np.array([True]*len(xy_np))

                    if np.any(weight_mask):    
                        ax.scatter(xy_np[:, 0][weight_mask], xy_np[:, 1][weight_mask], s=min([40,30*200/len(xy_np[weight_mask])]), alpha=alpha_vals[weight_mask])

                set_axis_limits(ax)
                ax.set_title('String Positions in XY Plane')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                
                # Draw radius circle around origin if requested
                if draw_radius and max_radius is not None:
                    circle = plt.Circle((0, 0), max_radius, color='blue', fill=False, 
                                       linewidth=5, linestyle='--', alpha=0.2)
                    ax.add_patch(circle)
                    # ax.legend()

                # Draw weighted bounding cylinder overlay if requested.
                if draw_weighted_cylinder:
                    cyl_center = kwargs.get('weighted_bounding_cylinder_center', kwargs.get('bounding_cylinder_center', None))
                    cyl_radius = kwargs.get('weighted_bounding_cylinder_radius', kwargs.get('bounding_cylinder_radius', None))
                    # print("Cylinder center:", cyl_center)
                    # print("Cylinder radius:", cyl_radius)

                    if cyl_center is not None and cyl_radius is not None:
                        if torch.is_tensor(cyl_center):
                            cyl_center_np = cyl_center.detach().cpu().numpy().reshape(-1)
                        else:
                            cyl_center_np = np.asarray(cyl_center).reshape(-1)

                        if torch.is_tensor(cyl_radius):
                            cyl_radius_val = float(cyl_radius.detach().cpu().reshape(-1)[0].item())
                        else:
                            cyl_radius_arr = np.asarray(cyl_radius).reshape(-1)
                            cyl_radius_val = float(cyl_radius_arr[0])

                        if cyl_center_np.shape[0] >= 2 and np.isfinite(cyl_radius_val) and cyl_radius_val > 0:
                            weighted_circle = plt.Circle(
                                (float(cyl_center_np[0]), float(cyl_center_np[1])),
                                cyl_radius_val,
                                color='orange',
                                fill=False,
                                linewidth=2,
                                linestyle='-',
                                alpha=0.9,
                            )
                            ax.add_patch(weighted_circle)
                            ax.scatter(
                                [float(cyl_center_np[0])],
                                [float(cyl_center_np[1])],
                                c='orange',
                                s=30,
                                marker='x',
                                linewidths=1.5,
                                alpha=0.9,
                            )
                
            else:
                ax.text(0.5, 0.5, "String XY data not available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        elif plot_type == self.PLOT_STRING_XY_ROV_PENALTY:
            # String positions in XY plane colored by ROV penalty per string
            if string_xy is not None:
                xy_np = string_xy.clone().detach().cpu().numpy()
                rov_penalty_per_string = kwargs.get('rov_penalty_per_string', None)
                rov_least_blocked_angle_per_string = kwargs.get('rov_least_blocked_angle_per_string', None)
                string_weights = kwargs.get('string_weights', None)
                draw_rov_safe_space_on_violations = bool(kwargs.get('rov_draw_safe_space_on_violations', False))
                weight_threshold = kwargs.get('weight_threshold', 0.7)
                if rov_penalty_per_string is not None:
                    # Convert ROV penalty per string to numpy
                    if torch.is_tensor(rov_penalty_per_string):
                        rov_penalty_np = rov_penalty_per_string.clone().detach().cpu().numpy()
                    else:
                        rov_penalty_np = np.array(rov_penalty_per_string)
                    rov_penalty_np*= len(xy_np)

                    # Optionally draw the per-string ROV safe-space corridor for
                    # strings with a (displayed) violation >= 1, oriented by the
                    # least-blocked angle.
                    if draw_rov_safe_space_on_violations:
                        rov_penalty_func = kwargs.get('rov_penalty_func', None) or kwargs.get('rov_penalty', None)
                        if rov_penalty_func is not None and rov_least_blocked_angle_per_string is not None:
                            if torch.is_tensor(rov_least_blocked_angle_per_string):
                                rov_angles_np = rov_least_blocked_angle_per_string.detach().cpu().numpy()
                            else:
                                rov_angles_np = np.array(rov_least_blocked_angle_per_string)

                            if rov_angles_np.shape[0] == xy_np.shape[0]:
                                violation_mask = rov_penalty_np >= weight_threshold
                                viol_idx = np.where(violation_mask)[0]
                                for i in viol_idx:
                                    self._draw_rov_safe_space_at_string(
                                        ax,
                                        origin_xy=xy_np[i],
                                        angle_rad=rov_angles_np[i],
                                        rov_penalty=rov_penalty_func,
                                        zorder=1,
                                    )
                    
                    # Use string weights for alpha transparency (no threshold filtering)
                    if string_weights is not None:
                        alpha_vals = np.array([string_weights[idx] for idx in string_indices])
                        # Clip to ensure minimum visibility and maximum opacity
                        alpha_vals = np.clip(alpha_vals, 0.05, 1.0)
                    else:
                        alpha_vals = 0.8
                    
                    # Create colormap for ROV penalty
                    cmap = plt.cm.RdYlGn_r  # Red for high penalty, green for low penalty
                    
                    # Normalize penalties for colormap
                    vmin = np.min(rov_penalty_np)
                    vmax = np.max(rov_penalty_np)
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    
                    # Plot strings colored by ROV penalty with alpha based on weights
                    sc = ax.scatter(
                        xy_np[:, 0],
                        xy_np[:, 1],
                        s=min([30, 50 * 200 / len(xy_np)]),
                        c=rov_penalty_np,
                        cmap=cmap,
                        norm=norm,
                        alpha=alpha_vals if isinstance(alpha_vals, np.ndarray) else alpha_vals,
                        edgecolors='black',
                        linewidths=0.1
                    )
                    
                    # Add colorbar
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.set_label('ROV Penalty')
                    
                    rov_penalty = kwargs.get('rov_penalty_func', None) or kwargs.get('rov_penalty', None)
                    if rov_penalty is not None:
                        self._draw_rov_safe_space(ax, rov_penalty, zoom_range=zoom_range)
                    
                    set_axis_limits(ax)
                    ax.set_title('ROV Penalty per String')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                else:
                    ax.text(0.5, 0.5, "ROV penalty per string data not available", 
                          ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "String XY data not available", 
                      ha='center', va='center', transform=ax.transAxes)

        elif plot_type == self.PLOT_STRING_XY_LOCAL_STRING_REPULSION:
            # String positions in XY plane colored by local string repulsion per string
            if string_xy is not None:
                xy_np = string_xy.clone().detach().cpu().numpy()
                local_repulsion_per_string = kwargs.get('local_string_repulsion_penalty_per_string', None)
                string_weights = kwargs.get('string_weights', None)

                if local_repulsion_per_string is not None:
                    if torch.is_tensor(local_repulsion_per_string):
                        local_repulsion_np = local_repulsion_per_string.clone().detach().cpu().numpy()
                    else:
                        local_repulsion_np = np.array(local_repulsion_per_string)

                    if local_repulsion_np.shape[0] != xy_np.shape[0]:
                        ax.text(0.5, 0.5, "Repulsion/string count mismatch", 
                              ha='center', va='center', transform=ax.transAxes)
                    else:
                        if string_weights is not None:
                            alpha_vals = np.array([string_weights[idx] for idx in string_indices])
                            alpha_vals = np.clip(alpha_vals, 0.05, 1.0)
                        else:
                            alpha_vals = 0.8

                        cmap = plt.cm.RdYlGn_r  # Red for high penalty, green for low penalty
                        vmin = 0
                        vmax = np.max(local_repulsion_np) if np.max(local_repulsion_np) > 0 else 1.0
                        norm = Normalize(vmin=vmin, vmax=vmax)

                        sc = ax.scatter(
                            xy_np[:, 0],
                            xy_np[:, 1],
                            s=min([30, 50 * 200 / len(xy_np)]),
                            c=local_repulsion_np,
                            cmap=cmap,
                            norm=norm,
                            alpha=alpha_vals if isinstance(alpha_vals, np.ndarray) else alpha_vals,
                            edgecolors='black',
                            linewidths=0.1
                        )

                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label('Local String Repulsion Penalty')

                        set_axis_limits(ax)
                        ax.set_title('Local String Repulsion per String')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                else:
                    ax.text(0.5, 0.5, "Local string repulsion per string data not available", 
                          ha='center', va='center', transform=ax.transAxes)
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
            set_axis_limits(ax)
            fig.colorbar(sc, ax=ax, label='Z Position')
            
        elif plot_type == self.PLOT_SIGNAL_CONTOUR:
            # Signal function contour plot
            signal_funcs = kwargs.get('signal_funcs', [])
            signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
            signal_event_params = kwargs.get('resolution_params', None)
            if signal_event_params is None:    
                signal_event_params = kwargs.get('signal_event_params', None)
            if kwargs.get("plot_geom_contour_only", False):
                x_lim_min = min(points_xyz[:, 0])
                x_lim_max = max(points_xyz[:, 0])
                y_lim_min = min(points_xyz[:, 1])
                y_lim_max = max(points_xyz[:, 1])
                z_lim_min = min(points_xyz[:, 2])
                z_lim_max = max(points_xyz[:, 2])
            else:
                x_lim_min, x_lim_max = -self.half_domain, self.half_domain
                y_lim_min, y_lim_max = -self.half_domain, self.half_domain
                z_lim_min, z_lim_max = -self.half_domain, self.half_domain
            
            # Check if we have either the old format or new surrogate format
            if signal_funcs or (signal_surrogate_func is not None and signal_event_params is not None):
                # Create a 2D grid in the XY plane at Z=0 for visualization
                resolution = slice_res
                x_grid = torch.linspace(x_lim_min, x_lim_max, resolution, device=self.device)
                y_grid = torch.linspace(y_lim_min, y_lim_max, resolution, device=self.device)
                X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
                
                grid_z = 0.0  # Z-slice at z=0
                if multi_slice:
                    # Create multiple slices if multi_slice is True
                    z_slices = np.linspace(z_lim_min, z_lim_max, resolution)
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
                    # Check if signal_event_params is a list of events or a single event
                    if isinstance(signal_event_params, list):
                        event_params_list = signal_event_params
                    else:
                        event_params_list = [signal_event_params]
                    
                    if not multi_slice:
                        # Evaluate surrogate function at all grid points at once for each event
                        event_values = []
                        for event_params in event_params_list:
                            values = signal_surrogate_func(opt_point=grid_points, event_params=event_params).reshape(resolution, resolution).clone().detach().cpu().numpy()
                            event_values.append(values)
                        # Average over all events
                        signal_values = np.mean(event_values, axis=0)
                    else:
                        # Multi-slice evaluation - evaluate entire 3D grid at once for each event
                        all_grid_points = torch.cat(grid_points, dim=0)
                        event_values = []
                        for event_params in event_params_list:
                            all_values = signal_surrogate_func(opt_point=all_grid_points, event_params=event_params)
                            # Reshape to (num_z_slices, resolution, resolution) and average over z dimension
                            event_value = all_values.reshape(len(z_slices), resolution, resolution).mean(dim=0).clone().detach().cpu().numpy()
                            event_values.append(event_value)
                        # Average over all events
                        signal_values = np.mean(event_values, axis=0)
                
                # Handle old signal functions format (backward compatibility)
                elif signal_funcs:
                    vis_all_signals = kwargs.get('vis_all_signals', False)
                    
                    if not multi_slice:
                        if not vis_all_signals:
                            signal_func = signal_funcs[np.random.randint(0, len(signal_funcs))]
                            signal_values = signal_func(grid_points).reshape(resolution, resolution).clone().detach().cpu().numpy()
                        else:
                            for i in range(len(signal_funcs)):
                                signal_values += signal_funcs[i](grid_points).reshape(resolution, resolution).detach().cpu().numpy()
                            signal_values /= len(signal_funcs)
                    else:
                        if not vis_all_signals:
                            signal_func = signal_funcs[np.random.randint(0, len(signal_funcs))]
                            for i in range(len(z_slices)):
                                signal_values += signal_func(grid_points[i]).reshape(resolution, resolution).clone().detach().cpu().numpy()
                            signal_values /= len(z_slices)
                        else:
                            for signal_func in signal_funcs:
                                for i in range(len(z_slices)):
                                    signal_values += signal_func(grid_points[i]).reshape(resolution, resolution).clone().detach().cpu().numpy()
                            signal_values /= len(signal_funcs) * len(z_slices)
                
                # Apply log transformation if requested
                use_log_charge = kwargs.get('use_log_charge', False)
                if use_log_charge:
                    # Replace zeros or very small values with minimum value to avoid log(0)
                    min_val = np.min(signal_values[signal_values > 0]) if np.any(signal_values > 0) else 0.1
                    signal_values = np.where(signal_values <= 0, min_val, signal_values)
                    signal_values = np.log10(signal_values)
                
                # Plot signal function
                c1 = ax.contourf(X.clone().detach().cpu().numpy(), Y.clone().detach().cpu().numpy(), signal_values, cmap='viridis', levels=20)
                colorbar_label = 'Log10(Signal)' if use_log_charge else 'Signal'
                fig.colorbar(c1, ax=ax, label=colorbar_label)
                
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
                
                ax.scatter(string_xy[:, 0], string_xy[:, 1], c='red', s=min([40,30*200/len(string_indices)]), alpha=alpha_values, edgecolor='white')
                
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
                set_axis_limits(ax)
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
                    # Check if background_event_params is a list of events or a single event
                    if isinstance(background_event_params, list):
                        event_params_list = background_event_params
                    else:
                        event_params_list = [background_event_params]
                    
                    if not multi_slice:
                        # Evaluate surrogate function at all grid points at once for each event
                        event_values = []
                        for event_params in event_params_list:
                            values = background_surrogate_func(opt_point=grid_points, event_params=event_params).reshape(resolution, resolution).clone().detach().cpu().numpy() * kwargs.get('background_scale', 1.0)
                            event_values.append(values)
                        # Average over all events
                        bkg_values = np.mean(event_values, axis=0)
                    else:
                        # Multi-slice evaluation - evaluate entire 3D grid at once for each event
                        all_grid_points = torch.cat(grid_points, dim=0)
                        event_values = []
                        for event_params in event_params_list:
                            all_values = background_surrogate_func(opt_point=all_grid_points, event_params=event_params)
                            # Reshape to (num_z_slices, resolution, resolution) and average over z dimension
                            event_value = all_values.reshape(len(z_slices), resolution, resolution).mean(dim=0).clone().detach().cpu().numpy() * kwargs.get('background_scale', 1.0)
                            event_values.append(event_value)
                        # Average over all events
                        bkg_values = np.mean(event_values, axis=0)
                
                # Handle old background functions format (backward compatibility)
                elif background_funcs:
                    for background_func in background_funcs:
                        if not multi_slice:
                            bkg_values += background_func(grid_points).reshape(resolution, resolution).clone().detach().cpu().numpy()*kwargs.get('background_scale', 1.0)
                        else:
                            temp_bkg_values = np.zeros((resolution, resolution))
                            for i in range(len(z_slices)):
                                temp_bkg_values += background_func(grid_points[i]).reshape(resolution, resolution).clone().detach().cpu().numpy()*kwargs.get('background_scale', 1.0)
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
                
            c2 = ax.contourf(X.clone().detach().cpu().numpy(), Y.clone().detach().cpu().numpy(), bkg_values, 
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
            set_axis_limits(ax)
        elif plot_type == self.PLOT_PARAM_1D:
            # 1D parameter vs SNR plot
            optimize_params = kwargs.get('optimize_params', [])
            param_values = kwargs.get('param_values', {})
            all_snr = kwargs.get('all_snr', None)
            
            if len(optimize_params) == 1 and all_snr is not None:
                param_name = optimize_params[0]
                if param_name in param_values:
                    param_vals = param_values[param_name].clone().detach().cpu().numpy()
                    snr_vals = all_snr.clone().detach().cpu().numpy()
                    
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
                    param1_vals = param_values[param1].clone().detach().cpu().numpy()
                    param2_vals = param_values[param2].clone().detach().cpu().numpy()
                    snr_vals = all_snr.clone().detach().cpu().numpy();
                    
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
                    return sfunc_obj(points_to_eval).reshape(resolution, resolution).clone().detach().cpu().numpy()
                # Check for __call__ if not directly callable (e.g. for some class instances)
                elif hasattr(sfunc_obj, '__call__'):
                    return sfunc_obj.__call__(points_to_eval).reshape(resolution, resolution).clone().detach().cpu().numpy()
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
            
            points_np = points.clone().detach().cpu().numpy()
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
            set_axis_limits(ax)
            
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
                            grid_true_single_func = true_func_callable(grid_points_current_slice).reshape(resolution, resolution).clone().detach().cpu().numpy()
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
                        grid_true_single_func = true_func_callable(grid_points_single_slice).reshape(resolution, resolution).clone().detach().cpu().numpy()
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

            points_np = points.clone().detach().cpu().numpy()
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
            set_axis_limits(ax)
        
        elif plot_type == self.PLOT_STRING_WEIGHTS_SCATTER:
            # String weights scatter plot with variable alpha
            if string_xy is not None:
                string_weights = kwargs.get('string_weights', None)
                weight_threshold = kwargs.get('weight_threshold', 0.7)
                
                if string_weights is not None:
                    # Convert tensors to numpy arrays
                    xy_np = string_xy.clone().detach().cpu().numpy()
                    weights_np = string_weights
                    # Create alpha values: 1 if weight > 0.7, else 0.5
                    alphas = [1 if weights_np[i] > 0.7 else 0.6 for i in range(len(weights_np))]
                    # edge_colors=['k' if weights_np[i] > 0.7 else 'none' for i in range(len(weights_np))]
                    # Create scatter plot with explicit normalization
                    
                    norm = Normalize(vmin=0, vmax=1)
                    scatter = ax.scatter(
                        xy_np[:, 0], 
                        xy_np[:, 1], 
                        c=weights_np,
                        cmap='Greens',
                        alpha=alphas,
                        edgecolors=None,
                        s=min([40,30*200/len(weights_np)]),
                        norm=norm
                        )
                    
                    # Add colorbar (that is consistently scaled from 0 to 1 for all iterations)
                  
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('String Weight')
                    
                    # Set labels and title
                    ax.set_xlabel('X Coordinate')
                    ax.set_ylabel('Y Coordinate')
                    ax.set_title(f'Active strings = {len(weights_np[weights_np > 0.7])}, Total strings = {len(weights_np)}')
                    set_axis_limits(ax)
                    
                    # Add ROV safe space visualization if ROV penalty is available
                    # rov_penalty = kwargs.get('rov_penalty', None)
                    # if rov_penalty is not None:
                    #     self._draw_rov_safe_space(ax, rov_penalty, zoom_range=zoom_range)
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
                    llr_values_np = llr_per_string.clone().detach().cpu().numpy()
                else:
                    llr_values_np = np.array(llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
            else:
                ax.text(0.5, 0.5, "LLR per string data not available\n(Requires 'llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SIGNAL_LLR_CONTOUR:
            # Signal-only LLR contour plot based on per-string values
            signal_llr_per_string = kwargs.get('signal_llr_per_string', None)
            
            if signal_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_string, 'detach'):
                    signal_llr_values_np = signal_llr_per_string.clone().detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
            else:
                ax.text(0.5, 0.5, "Signal LLR per string data not available\n(Requires 'signal_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_TRUE_SIGNAL_LLR_CONTOUR:
            # Signal-only LLR contour plot based on per-string values
            signal_llr_per_string = kwargs.get('true_signal_llr_per_string', None)
            
            if signal_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_string, 'detach'):
                    signal_llr_values_np = signal_llr_per_string.clone().detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
            else:
                ax.text(0.5, 0.5, "True Signal LLR per string data not available\n(Requires 'true_signal_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_SIGNAL_LLR_CONTOUR_POINTS:
            # Signal-only LLR contour plot based on per-point values
            signal_llr_per_points = kwargs.get('signal_llr_per_point', None)
            
            if signal_llr_per_points is not None and points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_llr_per_points, 'detach'):
                    signal_llr_values_np = signal_llr_per_points.clone().detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_points)
                
                points_np = points.clone().detach().cpu().numpy()
                
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
                    set_axis_limits(ax)
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
                    background_llr_values_np = background_llr_per_string.clone().detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
            else:
                ax.text(0.5, 0.5, "Background LLR per string data not available\n(Requires 'background_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_TRUE_BACKGROUND_LLR_CONTOUR:
            # Background-only LLR contour plot based on per-string values
            background_llr_per_string = kwargs.get('true_background_llr_per_string', None)
            
            if background_llr_per_string is not None and string_xy is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(background_llr_per_string, 'detach'):
                    background_llr_values_np = background_llr_per_string.clone().detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
            else:
                ax.text(0.5, 0.5, "True Background LLR per string data not available\n(Requires 'true_background_llr_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_BACKGROUND_LLR_CONTOUR_POINTS:
            # Background-only LLR contour plot based on per-point values
            background_llr_per_points = kwargs.get('background_llr_per_point', None)
            
            if background_llr_per_points is not None and points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(background_llr_per_points, 'detach'):
                    background_llr_values_np = background_llr_per_points.clone().detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_points)
                
                points_np = points.clone().detach().cpu().numpy()
                
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
                    set_axis_limits(ax)
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
                    signal_llr_values_np = signal_llr_per_string.clone().detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_string)
                    
                if hasattr(background_llr_per_string, 'detach'):
                    background_llr_values_np = background_llr_per_string.clone().detach().cpu().numpy()
                else:
                    background_llr_values_np = np.array(background_llr_per_string)
                
                # Apply string weights if available
                if string_weights is not None:
                    if hasattr(string_weights, 'detach'):
                        weights_np = string_weights.clone().detach().cpu().numpy()
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
                    llr_values_np = available_data.clone().detach().cpu().numpy()
                else:
                    llr_values_np = np.array(available_data)
                
                # Apply string weights if available
                if string_weights is not None:
                    if hasattr(string_weights, 'detach'):
                        weights_np = string_weights.clone().detach().cpu().numpy()
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
                    signal_llr_values_np = signal_llr_per_points.clone().detach().cpu().numpy()
                else:
                    signal_llr_values_np = np.array(signal_llr_per_points)
                    
                if hasattr(background_llr_per_points, 'detach'):
                    background_llr_values_np = background_llr_per_points.clone().detach().cpu().numpy()
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
                    llr_values_np = available_data.clone().detach().cpu().numpy()
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
            plot_with_surrogate = kwargs.get('plot_with_surrogate', False)
            light_surrogate_func = kwargs.get('signal_surrogate_func', None)
            surrogate_event_params = kwargs.get('signal_event_params', None)
            if surrogate_event_params is None:
                surrogate_event_params = kwargs.get('resolution_params', None)
            
            # Check if we should use surrogate function for full domain contour
            if (plot_with_surrogate or (signal_light_yield_per_string is not None)) and (light_surrogate_func is not None) and (surrogate_event_params is not None):
                # Handle multiple sets of event parameters
                if isinstance(surrogate_event_params, list):
                    event_params_list = surrogate_event_params
                    num_events = len(event_params_list)
                else:
                    event_params_list = [surrogate_event_params]
                    num_events = 1
                
                # Create a dense grid for surrogate function evaluation
                resolution = slice_res
                if kwargs.get('zoom_range', None) is None:
                    x_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                    y_grid = np.linspace(-self.half_domain, self.half_domain, resolution)
                else:
                    zoom_num = kwargs['zoom_range']
                    x_grid = np.linspace(-zoom_num, zoom_num, resolution)
                    y_grid = np.linspace(-zoom_num, zoom_num, resolution)
                X_np, Y_np = np.meshgrid(x_grid, y_grid)
                
                # Initialize grid to accumulate averages over events
                signal_light_yield_grid = np.zeros((resolution, resolution))
                
                # Loop over each event parameter set
                for event_idx, event_params in enumerate(event_params_list):
                    if multi_slice:
                        # For multi-slice, generate points across different z values and average
                        z_slices = np.linspace(-self.half_domain, self.half_domain, slice_res)  # 5 z slices
                        event_grid = np.zeros((resolution, resolution))
                        
                        for z_val in z_slices:
                            # Create 3D points for this z slice
                            Z_np = np.full_like(X_np, z_val)
                            grid_points_3d = np.column_stack([X_np.flatten(), Y_np.flatten(), Z_np.flatten()])
                            grid_points_tensor = torch.tensor(grid_points_3d, dtype=torch.float32, device=self.device)
                            
                            # Evaluate surrogate function at all grid points for this slice
                            slice_values = []
                            for i in range(grid_points_tensor.shape[0]):
                                opt_point = grid_points_tensor[i:i+1]  # Keep batch dimension
                                light_yield_val = light_surrogate_func(
                                    opt_point=opt_point,
                                    event_params=event_params
                                )
                                slice_values.append(light_yield_val.clone().detach().cpu().numpy().item())
                            
                            # Reshape and add to z-slice average for this event
                            slice_grid = np.array(slice_values).reshape(resolution, resolution)
                            event_grid += slice_grid / len(z_slices)
                        
                        # Add this event's contribution to the overall average
                        signal_light_yield_grid += event_grid / num_events
                        
                    else:
                        # For single slice, use z=0 plane
                        Z_np = np.zeros_like(X_np)
                        grid_points_3d = np.column_stack([X_np.flatten(), Y_np.flatten(), Z_np.flatten()])
                        grid_points_tensor = torch.tensor(grid_points_3d, dtype=torch.float32, device=self.device)
                        
                        # Evaluate surrogate function at all grid points for this event
                        grid_values = []
                        for i in range(grid_points_tensor.shape[0]):
                            opt_point = grid_points_tensor[i:i+1]  # Keep batch dimension
                            light_yield_val = light_surrogate_func(
                                opt_point=opt_point,
                                event_params=event_params
                            )
                            grid_values.append(light_yield_val.clone().detach().cpu().numpy().item())
                        
                        # Reshape to grid and add to average
                        event_grid = np.array(grid_values).reshape(resolution, resolution)
                        signal_light_yield_grid += event_grid / num_events
                
                # Create the contour plot with surrogate-based values
                c1 = ax.contourf(X_np, Y_np, signal_light_yield_grid, cmap='Oranges', levels=20)
                cbar = fig.colorbar(c1, ax=ax)
                cbar.set_label('Light Yield')
                
                title_text = f"Signal Yield Contour"
                ax.set_title(title_text)
                
            elif signal_light_yield_per_string is not None and string_xy is not None:
                # Original implementation using per-string values
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_light_yield_per_string, 'detach'):
                    signal_light_yield_values_np = signal_light_yield_per_string.clone().detach().cpu().numpy()
                else:
                    signal_light_yield_values_np = np.array(signal_light_yield_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                cbar.set_label('Light Yield')
                
                ax.set_title(f"Signal Light Yield per String")
                
            else:
                ax.text(0.5, 0.5, "Signal light yield data not available\n(Requires either surrogate function setup or 'signal_light_yield_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Signal Light Yield - No Data")
                
            # Always overlay string positions if available (regardless of method used)
            if string_xy is not None:
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
                else:
                    string_positions_np = np.array(string_xy)
                    
                string_x = string_positions_np[:, 0]
                string_y = string_positions_np[:, 1]
                
                # Get string weights and spacing for visualization
                string_weights = kwargs.get('string_weights', None)
                if string_weights is not None and string_indices is not None:
                    alpha_values = np.array([string_weights[idx] for idx in range(len(string_weights))])
                    alpha_values = np.clip(alpha_values, 0.05, 1.0)
                else:
                    alpha_values = 0.8
                    
                if kwargs.get('string_spacing', None) is not None:
                    size_factor = kwargs['string_spacing']/2
                else:
                    size_factor = 1.0
                # if kwargs.get('zoom_range', None) is not None:
                #     size_factor *= (self.domain_size/kwargs['zoom_range'])
                # else:
                #     size_factor *= self.domain_size
                
                # Color string positions by their per-string light yield if available
                if signal_light_yield_per_string is not None:
                    if hasattr(signal_light_yield_per_string, 'detach'):
                        signal_light_yield_values_np = signal_light_yield_per_string.clone().detach().cpu().numpy()
                    else:
                        signal_light_yield_values_np = np.array(signal_light_yield_per_string)
                        
                    scatter = ax.scatter(string_x, string_y, c=signal_light_yield_values_np, 
                                       cmap='Oranges', s=min([60, 40*200*size_factor/len(string_indices)]), 
                                       alpha=alpha_values, edgecolor='black', linewidth=1,
                                       label='String Positions')
                else:
                    # Just show string positions without color coding
                    point_size = min([60, 40*200*size_factor/len(string_indices)]) if (string_indices is not None and len(string_indices) > 0) else 60
                    scatter = ax.scatter(string_x, string_y, c='red', 
                                       s=point_size, 
                                       alpha=alpha_values, edgecolor='black', linewidth=1,
                                       label='String Positions')
                
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            set_axis_limits(ax)
        
        elif plot_type == self.PLOT_SIGNAL_LIGHT_YIELD_CONTOUR_POINTS:
            # Signal light yield contour plot based on per-point values
            signal_light_yield_per_points = kwargs.get('signal_yield_per_point', None)
            
            if signal_light_yield_per_points is not None and points is not None:
                # Convert to numpy arrays if they're tensors
                if hasattr(signal_light_yield_per_points, 'detach'):
                    signal_light_yield_values_np = signal_light_yield_per_points.clone().detach().cpu().numpy()
                else:
                    signal_light_yield_values_np = np.array(signal_light_yield_per_points)
                
                points_np = points.clone().detach().cpu().numpy()
                
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
                    set_axis_limits(ax)
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
                    snr_values_np = snr_per_string.clone().detach().cpu().numpy()
                else:
                    snr_values_np = np.array(snr_per_string)
                    
                if hasattr(string_xy, 'detach'):
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
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
            #         string_positions_np = string_xy.clone().detach().cpu().numpy()
            #     else:
            #         string_positions_np = np.array(string_xy)
                
            #     # Compute Fisher Information matrix per string and its log determinant
            #     fisher_logdet_values = []
            #     for s_idx in range(len(fisher_info_per_string)):
            #         fisher_matrix = fisher_info_per_string[s_idx]
            #         # if hasattr(fisher_matrix, 'detach'):
            #         #     fisher_matrix = fisher_matrix.clone().detach().cpu().numpy()
                    
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
                    string_positions_np = string_xy.clone().detach().cpu().numpy()
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
                set_axis_limits(ax)
                
            else:
                ax.text(0.5, 0.5, "Fisher Information data not available\n(Requires 'fisher_info_per_string' and 'string_xy' in kwargs)", 
                      ha='center', va='center', transform=ax.transAxes)
        
        elif plot_type == self.PLOT_ANGULAR_RESOLUTION:
            # Angular resolution history from Fisher Information matrix using Cramér-Rao bound
            loss_dict = kwargs.get('uw_loss_dict', None)
            
            if loss_dict is not None:
                angular_resolution_history = loss_dict.get('angular_resolution_loss', None)

            if angular_resolution_history is not None:
                angular_resolution_history = np.array(angular_resolution_history) * 180.0  # Convert to degrees
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

        elif plot_type == self.PLOT_POINTSOURCE_FOM:
            # Pointsource FoM history from unweighted loss dictionary.
            loss_dict = kwargs.get('uw_loss_dict', None)

            pointsource_fom_history = None
            if loss_dict is not None:
                pointsource_fom_history = loss_dict.get('pointsource_fom_loss', None)
                if pointsource_fom_history is None:
                    pointsource_fom_history = loss_dict.get('effective_area_resolution_loss', None)
                if pointsource_fom_history is None:
                    pointsource_fom_history = loss_dict.get('pointsource_fom', None)

            if pointsource_fom_history is not None:
                pointsource_fom_history = np.array(pointsource_fom_history)
                ax.plot(pointsource_fom_history, color='purple', linewidth=2, markersize=4)
                ax.set_title('Pointsource FoM History')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Pointsource FoM')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Pointsource FoM history not available\n(Pass 'uw_loss_dict' with pointsource_fom/effective_area_resolution history)",
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                )
        
        elif plot_type == self.PLOT_ANGULAR_RESOLUTION_VS_ZENITH:
            # Plot binned angular resolution vs zenith angle
            resolution_per_event = kwargs.get('resolution_per_event', None)
            signal_event_params = kwargs.get('resolution_params', None)
            # max_angular_resolution = kwargs.get('max_angular_resolution', np.pi)
            n_bins = kwargs.get('n_zenith_bins', 10)
            resolution_stat = kwargs.get('resolution_stat', None)
            if resolution_stat is None and bool(kwargs.get('resolution_use_mean', False)):
                resolution_stat = 'mean'
            resolution_stat = str(resolution_stat).lower() if resolution_stat is not None else 'median'
            if resolution_stat not in ('median', 'mean', 'fom'):
                resolution_stat = 'median'
            resolution_use_fom = bool(kwargs.get('resolution_use_fom', False)) or resolution_stat == 'fom'
            if resolution_use_fom:
                resolution_stat = 'fom'
            resolution_fom_min_resolution = kwargs.get('resolution_fom_min_resolution', 1e-12)
            show_resolution_ci = bool(kwargs.get('show_resolution_ci', False))
            resolution_ci_percentiles = kwargs.get('resolution_ci_percentiles', None)
            resolution_ci_level = kwargs.get('resolution_ci_level', None)
            zenith_range = kwargs.get('zenith_range', None)
            zenith_range_deg = kwargs.get('zenith_range_deg', None)
            resolution_logy = bool(kwargs.get('resolution_logy', kwargs.get('resolution_logy_angular', False)))
            min_ang_res = kwargs.get('min_angular_resolution', None)
            max_ang_res = kwargs.get('max_angular_resolution', None)
            if not resolution_logy:
                resolution_logy = bool(kwargs.get('resolution_logy_vs_zenith', kwargs.get('resolution_logy_vs_energy', False)))
            
            if resolution_per_event is not None and signal_event_params is not None:
                # Convert to numpy
                if isinstance(resolution_per_event, torch.Tensor):
                    res_values = resolution_per_event.clone().detach().cpu().numpy()
                else:
                    res_values = np.array(resolution_per_event)
                
                # Extract zenith angles from event parameters
                zenith_values = []
                for event_params in signal_event_params:
                    if isinstance(event_params, dict) and 'zenith' in event_params:
                        zenith = event_params['zenith']
                        if isinstance(zenith, torch.Tensor):
                            zenith_values.append(zenith.detach().cpu().item())
                        else:
                            zenith_values.append(float(zenith))
                
                zenith_values = np.array(zenith_values)
                
                # Filter out NaN/Inf values
                valid_mask = np.isfinite(res_values) & np.isfinite(zenith_values)
                res_values = res_values[valid_mask]
                zenith_values = zenith_values[valid_mask]
                
                if len(res_values) > 0 and len(zenith_values) > 0:
                    # Convert to degrees for easier interpretation
                    zenith_deg = np.rad2deg(zenith_values)

                    # Optional zenith range restriction.
                    zmin_deg, zmax_deg = 0.0, 180.0
                    if zenith_range_deg is not None and len(zenith_range_deg) == 2:
                        try:
                            zmin_deg, zmax_deg = float(zenith_range_deg[0]), float(zenith_range_deg[1])
                        except Exception:
                            zmin_deg, zmax_deg = 0.0, 180.0
                    elif zenith_range is not None and len(zenith_range) == 2:
                        try:
                            zmin_deg = float(np.rad2deg(float(zenith_range[0])))
                            zmax_deg = float(np.rad2deg(float(zenith_range[1])))
                        except Exception:
                            zmin_deg, zmax_deg = 0.0, 180.0
                    if zmax_deg < zmin_deg:
                        zmin_deg, zmax_deg = zmax_deg, zmin_deg

                    range_mask = (zenith_deg >= zmin_deg) & (zenith_deg <= zmax_deg)
                    zenith_deg = zenith_deg[range_mask]
                    res_values = res_values[range_mask]

                    if resolution_logy:
                        pos_mask = np.array(res_values) > 0
                        zenith_deg = zenith_deg[pos_mask]
                        res_values = res_values[pos_mask]
                    
                    # Create bins
                    bin_edges = np.linspace(zmin_deg, zmax_deg, n_bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Compute binned statistics
                    bin_medians = []
                    band_lower = []
                    band_upper = []
                    fom_errors = []
                    bin_counts = []
                    
                    for i in range(n_bins):
                        mask = (zenith_deg >= bin_edges[i]) & (zenith_deg < bin_edges[i+1])
                        if mask.sum() > 0:
                            vals = np.array(res_values[mask], dtype=float)
                            if resolution_use_fom:
                                center_val, fom_err = self._compute_fom_from_resolution(
                                    vals,
                                    min_resolution=resolution_fom_min_resolution,
                                )
                                bin_medians.append(center_val)
                                fom_errors.append(fom_err)
                                if np.isfinite(center_val) and np.isfinite(fom_err):
                                    band_lower.append(center_val - fom_err)
                                    band_upper.append(center_val + fom_err)
                                else:
                                    band_lower.append(np.nan)
                                    band_upper.append(np.nan)
                            elif resolution_stat == 'mean':
                                center_val = float(np.nanmean(vals))
                                spread_val = float(np.nanstd(vals))
                                bin_medians.append(center_val)
                                fom_errors.append(np.nan)
                            else:
                                center_val = float(np.nanmedian(vals))
                                spread_val = np.nan
                                bin_medians.append(center_val)
                                fom_errors.append(np.nan)
                            if (not resolution_use_fom) and show_resolution_ci:
                                if resolution_stat == 'mean':
                                    lo = center_val - 2.0 * spread_val
                                    hi = center_val + 2.0 * spread_val
                                    if min_ang_res is not None:    
                                        lo = float(max(lo, min_ang_res))
                                    if max_ang_res is not None:
                                        hi = float(min(hi, max_ang_res))
                                    band_lower.append(lo)
                                    band_upper.append(float(hi))
                                else:
                                    q_lo = None
                                    q_hi = None
                                    if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                        try:
                                            q_lo = float(resolution_ci_percentiles[0])
                                            q_hi = float(resolution_ci_percentiles[1])
                                        except Exception:
                                            q_lo, q_hi = None, None
                                    if q_lo is None or q_hi is None:
                                        if resolution_ci_level is not None:
                                            try:
                                                lvl = float(resolution_ci_level)
                                                lvl = float(np.clip(lvl, 0.0, 1.0))
                                                alpha = 0.5 * (1.0 - lvl)
                                                q_lo = 100.0 * alpha
                                                q_hi = 100.0 * (1.0 - alpha)
                                            except Exception:
                                                q_lo, q_hi = 16.0, 84.0
                                        else:
                                            q_lo, q_hi = 16.0, 84.0
                                    if q_hi < q_lo:
                                        q_lo, q_hi = q_hi, q_lo
                                    resid = vals - center_val
                                    band_lower.append(center_val + np.nanpercentile(resid, q_lo))
                                    band_upper.append(center_val + np.nanpercentile(resid, q_hi))
                            else:
                                if not resolution_use_fom:
                                    band_lower.append(np.nan)
                                    band_upper.append(np.nan)
                            bin_counts.append(mask.sum())
                        else:
                            bin_medians.append(np.nan)
                            band_lower.append(np.nan)
                            band_upper.append(np.nan)
                            fom_errors.append(np.nan)
                            bin_counts.append(0)
                    
                    bin_medians = np.array(bin_medians)
                    band_lower = np.array(band_lower)
                    band_upper = np.array(band_upper)
                    fom_errors = np.array(fom_errors)
                    bin_counts = np.array(bin_counts)
                    if min_ang_res is not None or max_ang_res is not None:
                        lo_lim = -np.inf
                        hi_lim = np.inf
                        try:
                            if min_ang_res is not None:
                                lo_lim = float(min_ang_res)
                        except Exception:
                            lo_lim = -np.inf
                        try:
                            if max_ang_res is not None:
                                hi_lim = float(max_ang_res)
                        except Exception:
                            hi_lim = np.inf
                        if np.isfinite(lo_lim) and np.isfinite(hi_lim) and hi_lim < lo_lim:
                            lo_lim, hi_lim = hi_lim, lo_lim

                        bin_medians = np.clip(bin_medians, lo_lim, hi_lim)
                        if show_resolution_ci or resolution_use_fom:
                            band_lower = np.clip(band_lower, lo_lim, hi_lim)
                            band_upper = np.clip(band_upper, lo_lim, hi_lim)
                            band_lower = np.minimum(band_lower, band_upper)

                    ci_lower = band_lower
                    ci_upper = band_upper
                    
                    # Plot with error bars
                    valid_bins = np.isfinite(bin_medians)
                    # ax.errorbar(bin_centers[valid_bins], bin_means[valid_bins], 
                    #            yerr=bin_stds[valid_bins], fmt='o-', capsize=5, 
                    #            linewidth=2, markersize=8, label='Mean ± Std')
                    if resolution_use_fom:
                        valid_err = valid_bins & np.isfinite(fom_errors)
                        if np.any(valid_err):
                            ax.errorbar(
                                bin_centers[valid_err],
                                bin_medians[valid_err],
                                yerr=fom_errors[valid_err],
                                fmt='o-',
                                linewidth=2,
                                markersize=8,
                                capsize=4,
                                label='FOM',
                            )
                        else:
                            ax.plot(
                                bin_centers[valid_bins],
                                bin_medians[valid_bins],
                                'o-',
                                linewidth=2,
                                markersize=8,
                                label='FOM',
                            )
                    elif show_resolution_ci and ci_lower is not None and ci_upper is not None:
                        valid_ci = valid_bins & np.isfinite(ci_lower) & np.isfinite(ci_upper)
                        if np.any(valid_ci):
                            q_lo, q_hi = 16.0, 84.0
                            ci_label = None
                            if resolution_stat == 'mean':
                                ci_label = 'Mean ± 2σ'
                            else:
                                if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                    try:
                                        q_lo = float(resolution_ci_percentiles[0])
                                        q_hi = float(resolution_ci_percentiles[1])
                                    except Exception:
                                        q_lo, q_hi = 16.0, 84.0
                                elif resolution_ci_level is not None:
                                    try:
                                        lvl = float(resolution_ci_level)
                                        lvl = float(np.clip(lvl, 0.0, 1.0))
                                        alpha = 0.5 * (1.0 - lvl)
                                        q_lo = 100.0 * alpha
                                        q_hi = 100.0 * (1.0 - alpha)
                                    except Exception:
                                        q_lo, q_hi = 16.0, 84.0
                                if q_hi < q_lo:
                                    q_lo, q_hi = q_hi, q_lo
                                ci_label = f"Residual band (p{q_lo:g}-p{q_hi:g})"
                            ax.plot(
                                bin_centers[valid_ci],
                                ci_lower[valid_ci],
                                linestyle='--',
                                linewidth=1.5,
                                color='gray',
                                alpha=0.9,
                                label=str(ci_label),
                                zorder=1,
                            )
                            ax.plot(
                                bin_centers[valid_ci],
                                ci_upper[valid_ci],
                                linestyle='--',
                                linewidth=1.5,
                                color='gray',
                                alpha=0.9,
                                label='_nolegend_',
                                zorder=1,
                            )
                    if not resolution_use_fom:
                        ax.plot(
                            bin_centers[valid_bins],
                            bin_medians[valid_bins],
                            'o-',
                            linewidth=2,
                            markersize=8,
                            label=('Mean' if resolution_stat == 'mean' else 'Median'),
                        )
                    if resolution_logy:
                        ax.set_yscale('log')

                    if resolution_stat == 'mean':
                        try:
                            y0, y1 = ax.get_ylim()
                            if y0 < 1e-5:
                                ax.set_ylim(bottom=1e-5)
                        except Exception:
                            pass
                    # Add scatter plot of raw data points (semi-transparent)
                    # ax.scatter(zenith_deg, res_values, alpha=0.3, s=20, 
                    #           c='gray', label='Individual events')
                    
                    ax.set_xlabel('Zenith Angle (degrees)', fontsize=10)
                    ax.set_ylabel('FOM (rad$^{-1}$)' if resolution_use_fom else 'Angular Resolution (radians)', fontsize=10)
                    ax.set_title('Angular FOM vs Zenith' if resolution_use_fom else 'Angular Resolution vs Zenith', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Add secondary y-axis for resolution in degrees (not used in FOM mode).
                    if not resolution_use_fom:
                        ax2 = ax.twinx()
                        ax2.set_ylabel('Angular Resolution (degrees)', fontsize=10)
                        ax2.set_yscale(ax.get_yscale())
                        ax2.set_ylim(np.rad2deg(ax.get_ylim()[0]), np.rad2deg(ax.get_ylim()[1]))
                        ax2.tick_params(axis='y')
                    
                    # Add text showing bin counts
                    # textstr = f'Total events: {len(res_values)}\n'
                    # textstr += f'Bins: {n_bins}\n'
                    # textstr += f'Mean resolution: {np.mean(res_values):.3f} rad ({np.rad2deg(np.mean(res_values)):.2f}°)'
                    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                    #        verticalalignment='top', bbox=dict(boxstyle='round', 
                    #        facecolor='wheat', alpha=0.5), fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Data not available\\nProvide resolution_per_event and signal_event_params', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        elif plot_type == self.PLOT_ANGULAR_RESOLUTION_VS_ENERGY:
            # Plot binned angular resolution vs log10(energy)
            resolution_per_event = kwargs.get('resolution_per_event', None)
            signal_event_params = kwargs.get('resolution_params', None)
            n_bins = kwargs.get('n_energy_bins', 10)
            resolution_stat = kwargs.get('resolution_stat', None)
            if resolution_stat is None and bool(kwargs.get('resolution_use_mean', False)):
                resolution_stat = 'mean'
            resolution_stat = str(resolution_stat).lower() if resolution_stat is not None else 'median'
            if resolution_stat not in ('median', 'mean', 'fom'):
                resolution_stat = 'median'
            resolution_use_fom = bool(kwargs.get('resolution_use_fom', False)) or resolution_stat == 'fom'
            if resolution_use_fom:
                resolution_stat = 'fom'
            resolution_fom_min_resolution = kwargs.get('resolution_fom_min_resolution', 1e-12)
            show_resolution_ci = bool(kwargs.get('show_resolution_ci', False))
            resolution_ci_percentiles = kwargs.get('resolution_ci_percentiles', None)
            resolution_ci_level = kwargs.get('resolution_ci_level', None)
            energy_range = kwargs.get('energy_range', None)
            resolution_logy = bool(kwargs.get('resolution_logy', kwargs.get('resolution_logy_angular', False)))
            min_ang_res = kwargs.get('min_angular_resolution', None)
            max_ang_res = kwargs.get('max_angular_resolution', None)
            if not resolution_logy:
                resolution_logy = bool(kwargs.get('resolution_logy_vs_zenith', kwargs.get('resolution_logy_vs_energy', False)))

            if resolution_per_event is not None and signal_event_params is not None:
                # Convert to numpy
                if isinstance(resolution_per_event, torch.Tensor):
                    res_values = resolution_per_event.clone().detach().cpu().numpy().flatten()
                else:
                    res_values = np.array(resolution_per_event).flatten()

                # Extract energy values from event parameters
                energy_values = []
                for event_params in signal_event_params:
                    if isinstance(event_params, dict) and 'energy' in event_params:
                        energy = event_params['energy']
                        if isinstance(energy, torch.Tensor):
                            energy_values.append(energy.detach().cpu().item())
                        else:
                            energy_values.append(float(energy))

                energy_values = np.array(energy_values)

                # Filter out NaN/Inf and non-positive energy values
                valid_mask = np.isfinite(res_values) & np.isfinite(energy_values) & (energy_values > 0)
                res_values = res_values[valid_mask]
                energy_values = energy_values[valid_mask]

                if energy_range is not None and len(energy_range) == 2:
                    try:
                        emin, emax = float(energy_range[0]), float(energy_range[1])
                        if emax < emin:
                            emin, emax = emax, emin
                        range_mask = (energy_values >= emin) & (energy_values <= emax)
                        res_values = res_values[range_mask]
                        energy_values = energy_values[range_mask]
                    except Exception:
                        pass

                if resolution_logy:
                    pos_mask = np.array(res_values) > 0
                    res_values = np.array(res_values)[pos_mask]
                    energy_values = np.array(energy_values)[pos_mask]

                if len(res_values) > 0 and len(energy_values) > 0:
                    # Create logarithmic bins for energy
                    log_energy_min = np.log10(energy_values.min())
                    log_energy_max = np.log10(energy_values.max())
                    bin_edges = np.logspace(log_energy_min, log_energy_max, n_bins + 1)
                    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean

                    # Compute binned statistics
                    bin_medians = []
                    band_lower = []
                    band_upper = []
                    fom_errors = []
                    bin_counts = []

                    for i in range(n_bins):
                        mask = (energy_values >= bin_edges[i]) & (energy_values < bin_edges[i+1])
                        if mask.sum() > 0:
                            vals = np.array(res_values[mask], dtype=float)
                            if resolution_use_fom:
                                center_val, fom_err = self._compute_fom_from_resolution(
                                    vals,
                                    min_resolution=resolution_fom_min_resolution,
                                )
                                bin_medians.append(center_val)
                                fom_errors.append(fom_err)
                                if np.isfinite(center_val) and np.isfinite(fom_err):
                                    band_lower.append(center_val - fom_err)
                                    band_upper.append(center_val + fom_err)
                                else:
                                    band_lower.append(np.nan)
                                    band_upper.append(np.nan)
                            elif resolution_stat == 'mean':
                                center_val = float(np.nanmean(vals))
                                spread_val = float(np.nanstd(vals))
                                bin_medians.append(center_val)
                                fom_errors.append(np.nan)
                            else:
                                center_val = float(np.nanmedian(vals))
                                spread_val = np.nan
                                bin_medians.append(center_val)
                                fom_errors.append(np.nan)
                            if (not resolution_use_fom) and show_resolution_ci:
                                if resolution_stat == 'mean':
                                    lo = center_val - 2.0 * spread_val
                                    hi = center_val + 2.0 * spread_val
                                    if min_ang_res is not None:    
                                        lo = float(max(lo, min_ang_res))
                                    if max_ang_res is not None:
                                        hi = float(min(hi, max_ang_res))
                                    band_lower.append(lo)
                                    band_upper.append(float(hi))
                                else:
                                    q_lo = None
                                    q_hi = None
                                    if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                        try:
                                            q_lo = float(resolution_ci_percentiles[0])
                                            q_hi = float(resolution_ci_percentiles[1])
                                        except Exception:
                                            q_lo, q_hi = None, None
                                    if q_lo is None or q_hi is None:
                                        if resolution_ci_level is not None:
                                            try:
                                                lvl = float(resolution_ci_level)
                                                lvl = float(np.clip(lvl, 0.0, 1.0))
                                                alpha = 0.5 * (1.0 - lvl)
                                                q_lo = 100.0 * alpha
                                                q_hi = 100.0 * (1.0 - alpha)
                                            except Exception:
                                                q_lo, q_hi = 16.0, 84.0
                                        else:
                                            q_lo, q_hi = 16.0, 84.0
                                    if q_hi < q_lo:
                                        q_lo, q_hi = q_hi, q_lo
                                    resid = vals - center_val
                                    band_lower.append(center_val + np.nanpercentile(resid, q_lo))
                                    band_upper.append(center_val + np.nanpercentile(resid, q_hi))
                            else:
                                if not resolution_use_fom:
                                    band_lower.append(np.nan)
                                    band_upper.append(np.nan)
                            bin_counts.append(mask.sum())
                        else:
                            bin_medians.append(np.nan)
                            band_lower.append(np.nan)
                            band_upper.append(np.nan)
                            fom_errors.append(np.nan)
                            bin_counts.append(0)

                    bin_medians = np.array(bin_medians)
                    band_lower = np.array(band_lower)
                    band_upper = np.array(band_upper)
                    fom_errors = np.array(fom_errors)
                    bin_counts = np.array(bin_counts)
                    if min_ang_res is not None or max_ang_res is not None:
                        lo_lim = -np.inf
                        hi_lim = np.inf
                        try:
                            if min_ang_res is not None:
                                lo_lim = float(min_ang_res)
                        except Exception:
                            lo_lim = -np.inf
                        try:
                            if max_ang_res is not None:
                                hi_lim = float(max_ang_res)
                        except Exception:
                            hi_lim = np.inf
                        if np.isfinite(lo_lim) and np.isfinite(hi_lim) and hi_lim < lo_lim:
                            lo_lim, hi_lim = hi_lim, lo_lim

                        bin_medians = np.clip(bin_medians, lo_lim, hi_lim)
                        if show_resolution_ci or resolution_use_fom:
                            band_lower = np.clip(band_lower, lo_lim, hi_lim)
                            band_upper = np.clip(band_upper, lo_lim, hi_lim)
                            band_lower = np.minimum(band_lower, band_upper)
                    ci_lower = band_lower
                    ci_upper = band_upper
                    # print(fom_errors)
                    # Plot mean line vs log10(energy)
                    valid_bins = np.isfinite(bin_medians)
                    x_plot = np.log10(bin_centers)
                    if resolution_use_fom:
                        valid_err = valid_bins & np.isfinite(fom_errors)
                        if np.any(valid_err):
                            ax.errorbar(
                                x_plot[valid_err],
                                bin_medians[valid_err],
                                yerr=fom_errors[valid_err],
                                fmt='o-',
                                linewidth=2,
                                markersize=8,
                                capsize=4,
                                label='FOM',
                            )
                            
                        else:
                            ax.plot(
                                x_plot[valid_bins],
                                bin_medians[valid_bins],
                                'o-',
                                linewidth=2,
                                markersize=8,
                                label='FOM',
                            )
                    elif show_resolution_ci and ci_lower is not None and ci_upper is not None:
                        valid_ci = valid_bins & np.isfinite(ci_lower) & np.isfinite(ci_upper)
                        if np.any(valid_ci):
                            q_lo, q_hi = 16.0, 84.0
                            ci_label = None
                            if resolution_stat == 'mean':
                                ci_label = 'Mean ± 2σ'
                            else:
                                if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                    try:
                                        q_lo = float(resolution_ci_percentiles[0])
                                        q_hi = float(resolution_ci_percentiles[1])
                                    except Exception:
                                        q_lo, q_hi = 16.0, 84.0
                                elif resolution_ci_level is not None:
                                    try:
                                        lvl = float(resolution_ci_level)
                                        lvl = float(np.clip(lvl, 0.0, 1.0))
                                        alpha = 0.5 * (1.0 - lvl)
                                        q_lo = 100.0 * alpha
                                        q_hi = 100.0 * (1.0 - alpha)
                                    except Exception:
                                        q_lo, q_hi = 16.0, 84.0
                                if q_hi < q_lo:
                                    q_lo, q_hi = q_hi, q_lo
                                ci_label = f"Residual band (p{q_lo:g}-p{q_hi:g})"
                            ax.plot(
                                x_plot[valid_ci],
                                ci_lower[valid_ci],
                                linestyle='--',
                                linewidth=1.5,
                                color='gray',
                                alpha=0.9,
                                label=str(ci_label),
                                zorder=1,
                            )
                            ax.plot(
                                x_plot[valid_ci],
                                ci_upper[valid_ci],
                                linestyle='--',
                                linewidth=1.5,
                                color='gray',
                                alpha=0.9,
                                label='_nolegend_',
                                zorder=1,
                            )
                    if not resolution_use_fom:
                        ax.plot(
                            x_plot[valid_bins],
                            bin_medians[valid_bins],
                            'o-',
                            linewidth=2,
                            markersize=8,
                            label=('Mean' if resolution_stat == 'mean' else 'Median'),
                        )
                    if resolution_logy:
                        ax.set_yscale('log')

                    if resolution_stat == 'mean':
                        try:
                            y0, y1 = ax.get_ylim()
                            if y0 < 1e-5:
                                ax.set_ylim(bottom=1e-5)
                        except Exception:
                            pass

                    ax.set_xlabel('log$_{10}$(Energy / GeV)', fontsize=10)
                    ax.set_ylabel('FOM (rad$^{-1}$)' if resolution_use_fom else 'Angular Resolution (radians)', fontsize=10)
                    ax.set_title('Angular FOM vs log$_{10}$(Energy)' if resolution_use_fom else 'Angular Resolution vs log$_{10}$(Energy)', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    # Add secondary y-axis in degrees (not used in FOM mode).
                    if not resolution_use_fom:
                        ax2 = ax.twinx()
                        ax2.set_ylabel('Angular Resolution (degrees)', fontsize=10)
                        ax2.set_yscale(ax.get_yscale())
                        ax2.set_ylim(np.rad2deg(ax.get_ylim()[0]), np.rad2deg(ax.get_ylim()[1]))
                        ax2.tick_params(axis='y')
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Data not available\nProvide resolution_per_event and resolution_params',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)

        elif plot_type == self.PLOT_POINTSOURCE_FOM_VS_ENERGY:
            # Plot binned pointsource FoM vs log10(energy)
            resolution_per_event = kwargs.get('resolution_per_event', None)
            effective_area_per_event = kwargs.get('effective_area_per_event', None)
            signal_event_params = kwargs.get('resolution_params', None)
            if signal_event_params is None:
                signal_event_params = kwargs.get('effective_area_params', None)
            if signal_event_params is None:
                signal_event_params = kwargs.get('signal_event_params', None)
            n_bins = kwargs.get('n_energy_bins', 10)
            energy_range = kwargs.get('energy_range', None)
            fom_min_resolution = kwargs.get('resolution_fom_min_resolution', 1e-12)
            resolution_logy = bool(kwargs.get('resolution_logy', kwargs.get('resolution_logy_angular', False)))
            if not resolution_logy:
                resolution_logy = bool(kwargs.get('resolution_logy_vs_zenith', kwargs.get('resolution_logy_vs_energy', False)))

            if (
                resolution_per_event is not None
                and effective_area_per_event is not None
                and signal_event_params is not None
            ):
                if isinstance(resolution_per_event, torch.Tensor):
                    res_values = resolution_per_event.clone().detach().cpu().numpy().flatten()
                else:
                    res_values = np.array(resolution_per_event).flatten()

                if isinstance(effective_area_per_event, torch.Tensor):
                    aeff_values = effective_area_per_event.clone().detach().cpu().numpy().flatten()
                else:
                    aeff_values = np.array(effective_area_per_event).flatten()

                energy_values = []
                for event_params in signal_event_params:
                    if isinstance(event_params, dict) and 'energy' in event_params:
                        energy = event_params['energy']
                        if isinstance(energy, torch.Tensor):
                            energy_values.append(energy.detach().cpu().item())
                        else:
                            energy_values.append(float(energy))

                energy_values = np.array(energy_values)

                n = min(len(res_values), len(aeff_values), len(energy_values))
                if n > 0:
                    res_values = res_values[:n]
                    aeff_values = aeff_values[:n]
                    energy_values = energy_values[:n]

                valid_mask = (
                    np.isfinite(res_values)
                    & np.isfinite(aeff_values)
                    & np.isfinite(energy_values)
                    & (energy_values > 0)
                )
                res_values = res_values[valid_mask]
                aeff_values = aeff_values[valid_mask]
                energy_values = energy_values[valid_mask]

                if energy_range is not None and len(energy_range) == 2:
                    try:
                        emin, emax = float(energy_range[0]), float(energy_range[1])
                        if emax < emin:
                            emin, emax = emax, emin
                        range_mask = (energy_values >= emin) & (energy_values <= emax)
                        res_values = res_values[range_mask]
                        aeff_values = aeff_values[range_mask]
                        energy_values = energy_values[range_mask]
                    except Exception:
                        pass

                if len(res_values) > 0 and len(aeff_values) > 0 and len(energy_values) > 0:
                    log_energy_min = np.log10(energy_values.min())
                    log_energy_max = np.log10(energy_values.max())
                    bin_edges = np.logspace(log_energy_min, log_energy_max, n_bins + 1)
                    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

                    bin_fom = []
                    bin_fom_err = []
                    for i in range(n_bins):
                        mask = (energy_values >= bin_edges[i]) & (energy_values < bin_edges[i + 1])
                        if mask.sum() > 0:
                            fval, ferr = self._compute_pointsource_fom_from_resolution_and_aeff(
                                res_values[mask],
                                aeff_values[mask],
                                min_resolution=fom_min_resolution,
                            )
                            bin_fom.append(fval)
                            bin_fom_err.append(ferr)
                        else:
                            bin_fom.append(np.nan)
                            bin_fom_err.append(np.nan)

                    bin_fom = np.array(bin_fom)
                    bin_fom_err = np.array(bin_fom_err)
                    valid_bins = np.isfinite(bin_fom)
                    if resolution_logy:
                        valid_bins = valid_bins & (bin_fom > 0)
                    x_plot = np.log10(bin_centers)

                    valid_err = valid_bins & np.isfinite(bin_fom_err)
                    if np.any(valid_err):
                        ax.errorbar(
                            x_plot[valid_err],
                            bin_fom[valid_err],
                            yerr=bin_fom_err[valid_err],
                            fmt='o-',
                            linewidth=2,
                            markersize=8,
                            capsize=4,
                            label='Pointsource FoM',
                        )
                    else:
                        ax.plot(
                            x_plot[valid_bins],
                            bin_fom[valid_bins],
                            'o-',
                            linewidth=2,
                            markersize=8,
                            label='Pointsource FoM',
                        )

                    ax.set_xlabel('log$_{10}$(Energy / GeV)', fontsize=10)
                    ax.set_ylabel('Pointsource FoM', fontsize=10)
                    ax.set_title('Pointsource FoM vs log$_{10}$(Energy)', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    if resolution_logy:
                        ax.set_yscale('log')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=14)
            else:
                ax.text(
                    0.5,
                    0.5,
                    'Data not available\nProvide resolution_per_event, effective_area_per_event, and event params',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=12,
                )

        elif plot_type == self.PLOT_ENERGY_RESOLUTION_VS_ENERGY:
            # Plot binned energy resolution vs energy
            resolution_per_event = kwargs.get('resolution_per_event', None)
            signal_event_params = kwargs.get('resolution_params', None)
            n_bins = kwargs.get('n_energy_bins', 10)
            use_relative_energy = kwargs.get('use_relative_energy', False)
            resolution_stat = kwargs.get('resolution_stat', None)
            if resolution_stat is None and bool(kwargs.get('resolution_use_mean', False)):
                resolution_stat = 'mean'
            resolution_stat = str(resolution_stat).lower() if resolution_stat is not None else 'median'
            if resolution_stat not in ('median', 'mean', 'fom'):
                resolution_stat = 'median'
            resolution_use_fom = bool(kwargs.get('resolution_use_fom', False)) or resolution_stat == 'fom'
            if resolution_use_fom:
                resolution_stat = 'fom'
            resolution_fom_min_resolution = kwargs.get('resolution_fom_min_resolution', 1e-12)
            show_resolution_ci = bool(kwargs.get('show_resolution_ci', False))
            resolution_ci_percentiles = kwargs.get('resolution_ci_percentiles', None)
            resolution_ci_level = kwargs.get('resolution_ci_level', None)
            energy_range = kwargs.get('energy_range', None)
            resolution_logy = bool(kwargs.get('resolution_logy', kwargs.get('resolution_logy_angular', False)))
            if not resolution_logy:
                resolution_logy = bool(kwargs.get('resolution_logy_vs_zenith', kwargs.get('resolution_logy_vs_energy', False)))
            
            if resolution_per_event is not None and signal_event_params is not None:
                # Convert to numpy
                if isinstance(resolution_per_event, torch.Tensor):
                    res_values = resolution_per_event.clone().detach().cpu().numpy().flatten()
                else:
                    res_values = np.array(resolution_per_event).flatten()
                
                # Extract energy values from event parameters
                energy_values = []
                for event_params in signal_event_params:
                    if isinstance(event_params, dict) and 'energy' in event_params:
                        energy = event_params['energy']
                        if isinstance(energy, torch.Tensor):
                            energy_values.append(energy.detach().cpu().item())
                        else:
                            energy_values.append(float(energy))
                
                energy_values = np.array(energy_values)
                
                # Filter out NaN/Inf values
                valid_mask = np.isfinite(res_values) & np.isfinite(energy_values)
                res_values = res_values[valid_mask]
                energy_values = energy_values[valid_mask]

                if energy_range is not None and len(energy_range) == 2:
                    try:
                        emin, emax = float(energy_range[0]), float(energy_range[1])
                        if emax < emin:
                            emin, emax = emax, emin
                        range_mask = (energy_values >= emin) & (energy_values <= emax)
                        res_values = res_values[range_mask]
                        energy_values = energy_values[range_mask]
                    except Exception:
                        pass

                if resolution_logy:
                    pos_mask = np.array(res_values) > 0
                    res_values = np.array(res_values)[pos_mask]
                    energy_values = np.array(energy_values)[pos_mask]
                
                if len(res_values) > 0 and len(energy_values) > 0:
                    # Create logarithmic bins for energy
                    log_energy_min = np.log10(energy_values.min())
                    log_energy_max = np.log10(energy_values.max())
                    bin_edges = np.logspace(log_energy_min, log_energy_max, n_bins + 1)
                    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log scale
                    
                    # Compute binned statistics
                    bin_medians = []
                    band_lower = []
                    band_upper = []
                    fom_errors = []
                    bin_counts = []
                    
                    for i in range(n_bins):
                        mask = (energy_values >= bin_edges[i]) & (energy_values < bin_edges[i+1])
                        if mask.sum() > 0:
                            vals = np.array(res_values[mask], dtype=float)
                            if resolution_use_fom:
                                center_val, fom_err = self._compute_fom_from_resolution(
                                    vals,
                                    min_resolution=resolution_fom_min_resolution,
                                )
                                bin_medians.append(center_val)
                                fom_errors.append(fom_err)
                                if np.isfinite(center_val) and np.isfinite(fom_err):
                                    band_lower.append(center_val - fom_err)
                                    band_upper.append(center_val + fom_err)
                                else:
                                    band_lower.append(np.nan)
                                    band_upper.append(np.nan)
                            elif resolution_stat == 'mean':
                                center_val = float(np.nanmean(vals))
                                spread_val = float(np.nanstd(vals))
                                bin_medians.append(center_val)
                                fom_errors.append(np.nan)
                            else:
                                center_val = float(np.nanmedian(vals))
                                spread_val = np.nan
                                bin_medians.append(center_val)
                                fom_errors.append(np.nan)
                            if (not resolution_use_fom) and show_resolution_ci:
                                if resolution_stat == 'mean':
                                    band_lower.append(center_val - 2.0 * spread_val)
                                    band_upper.append(center_val + 2.0 * spread_val)
                                else:
                                    q_lo = None
                                    q_hi = None
                                    if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                        try:
                                            q_lo = float(resolution_ci_percentiles[0])
                                            q_hi = float(resolution_ci_percentiles[1])
                                        except Exception:
                                            q_lo, q_hi = None, None
                                    if q_lo is None or q_hi is None:
                                        if resolution_ci_level is not None:
                                            try:
                                                lvl = float(resolution_ci_level)
                                                lvl = float(np.clip(lvl, 0.0, 1.0))
                                                alpha = 0.5 * (1.0 - lvl)
                                                q_lo = 100.0 * alpha
                                                q_hi = 100.0 * (1.0 - alpha)
                                            except Exception:
                                                q_lo, q_hi = 16.0, 84.0
                                        else:
                                            q_lo, q_hi = 16.0, 84.0
                                    if q_hi < q_lo:
                                        q_lo, q_hi = q_hi, q_lo
                                    resid = vals - center_val
                                    band_lower.append(center_val + np.nanpercentile(resid, q_lo))
                                    band_upper.append(center_val + np.nanpercentile(resid, q_hi))
                            else:
                                if not resolution_use_fom:
                                    band_lower.append(np.nan)
                                    band_upper.append(np.nan)
                            bin_counts.append(mask.sum())
                        else:
                            bin_medians.append(np.nan)
                            band_lower.append(np.nan)
                            band_upper.append(np.nan)
                            fom_errors.append(np.nan)
                            bin_counts.append(0)
                    
                    bin_medians = np.array(bin_medians)
                    band_lower = np.array(band_lower)
                    band_upper = np.array(band_upper)
                    fom_errors = np.array(fom_errors)
                    bin_counts = np.array(bin_counts)

                    ci_lower = band_lower
                    ci_upper = band_upper
                    
                    # Plot with error bars
                    valid_bins = np.isfinite(bin_medians)
                    if resolution_logy:
                        valid_bins = valid_bins & (np.array(bin_medians) > 0)
                    if resolution_use_fom:
                        valid_err = valid_bins & np.isfinite(fom_errors)
                        if np.any(valid_err):
                            ax.errorbar(
                                bin_centers[valid_err],
                                bin_medians[valid_err],
                                yerr=fom_errors[valid_err],
                                fmt='o-',
                                linewidth=2,
                                markersize=8,
                                capsize=4,
                                label='FOM',
                            )
                        else:
                            ax.plot(
                                bin_centers[valid_bins],
                                bin_medians[valid_bins],
                                'o-',
                                linewidth=2,
                                markersize=8,
                                label='FOM',
                            )
                    elif show_resolution_ci and ci_lower is not None and ci_upper is not None:
                        valid_ci = valid_bins & np.isfinite(ci_lower) & np.isfinite(ci_upper)
                        if np.any(valid_ci):
                            q_lo, q_hi = 16.0, 84.0
                            ci_label = None
                            if resolution_stat == 'mean':
                                ci_label = 'Mean ± 2σ'
                            else:
                                if resolution_ci_percentiles is not None and len(resolution_ci_percentiles) == 2:
                                    try:
                                        q_lo = float(resolution_ci_percentiles[0])
                                        q_hi = float(resolution_ci_percentiles[1])
                                    except Exception:
                                        q_lo, q_hi = 16.0, 84.0
                                elif resolution_ci_level is not None:
                                    try:
                                        lvl = float(resolution_ci_level)
                                        lvl = float(np.clip(lvl, 0.0, 1.0))
                                        alpha = 0.5 * (1.0 - lvl)
                                        q_lo = 100.0 * alpha
                                        q_hi = 100.0 * (1.0 - alpha)
                                    except Exception:
                                        q_lo, q_hi = 16.0, 84.0
                                if q_hi < q_lo:
                                    q_lo, q_hi = q_hi, q_lo
                                ci_label = f"Residual band (p{q_lo:g}-p{q_hi:g})"
                            ax.fill_between(
                                bin_centers[valid_ci],
                                ci_lower[valid_ci],
                                ci_upper[valid_ci],
                                alpha=0.2,
                                label=str(ci_label),
                                zorder=1,
                            )
                    if not resolution_use_fom:
                        ax.plot(
                            bin_centers[valid_bins],
                            bin_medians[valid_bins],
                            'o-',
                            linewidth=2,
                            markersize=8,
                            label=('Mean' if resolution_stat == 'mean' else 'Median'),
                        )
                    
                    ax.set_xlabel('Energy (GeV)', fontsize=10)
                    if resolution_use_fom:
                        ax.set_ylabel('FOM (dimensionless)' if use_relative_energy else 'FOM (GeV$^{-1}$)', fontsize=10)
                        ax.set_title('Energy FOM vs Energy', fontsize=12)
                    elif use_relative_energy:
                        ax.set_ylabel('Relative Energy Resolution (ΔE/E)', fontsize=10)
                        ax.set_title(f'Relative Energy Resolution vs Energy', fontsize=12)
                    else:
                        ax.set_ylabel('Energy Resolution (GeV)', fontsize=10)
                        ax.set_title(f'Energy Resolution vs Energy', fontsize=12)
                    
                    ax.set_xscale('log')
                    if resolution_logy:
                        ax.set_yscale('log')
                    ax.grid(True, alpha=0.3, which='both')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Data not available\nProvide resolution_per_event and signal_event_params', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
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
                    # Set y-axis limits
                    min_val = min(all_values) if all_values else 1e-4
                    max_val = max(total_loss) if total_loss else 1.0
                    
                    # Set lower limit to 1e-4 if any loss reaches that value
                    if min_val <= 1e-4:
                        ax.set_ylim(bottom=1e-4)
                    
                    # Adjust upper limit based on total loss with some margin
                    ax.set_ylim(top=max_val * 1.5)
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
                        loss_array[np.isnan(loss_array)] = 0.0  # Handle NaN values by setting them to 0 (log10(1))
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
        
        elif plot_type == self.PLOT_ALM_MU:
            # Plot ALM penalty parameter (mu) history for each constraint
            alm_mus_history = kwargs.get('alm_mus_history', {})
            loss_iterations_dict = kwargs.get('loss_iterations_dict', {})
            
            if not alm_mus_history:
                ax.text(0.5, 0.5, 'No ALM mu history available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('ALM Penalty Parameters (μ)')
            else:
                # Get the iteration numbers (use the first available loss component's iterations)
                if loss_iterations_dict:
                    iterations = loss_iterations_dict[list(loss_iterations_dict.keys())[0]]
                else:
                    # Fall back to assuming sequential iterations
                    max_len = max(len(v) for v in alm_mus_history.values()) if alm_mus_history else 0
                    iterations = list(range(max_len))
                
                # Plot mu for each constraint
                for constraint_name, mu_history in alm_mus_history.items():
                    if len(mu_history) > 0:
                        # Align iterations with history length
                        plot_iterations = iterations[:len(mu_history)]
                        ax.plot(plot_iterations, mu_history, label=f'{constraint_name}', linewidth=2)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('μ (Penalty Parameter)')
                ax.set_title('ALM Penalty Parameters (μ) History')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')  # Often mu values vary over orders of magnitude
        
        elif plot_type == self.PLOT_ALM_LAMBDA:
            # Plot ALM Lagrange multiplier (lambda) history for each constraint
            alm_lambdas_history = kwargs.get('alm_lambdas_history', {})
            loss_iterations_dict = kwargs.get('loss_iterations_dict', {})
            
            if not alm_lambdas_history:
                ax.text(0.5, 0.5, 'No ALM lambda history available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('ALM Lagrange Multipliers (λ)')
            else:
                # Get the iteration numbers (use the first available loss component's iterations)
                if loss_iterations_dict:
                    iterations = loss_iterations_dict[list(loss_iterations_dict.keys())[0]]
                else:
                    # Fall back to assuming sequential iterations
                    max_len = max(len(v) for v in alm_lambdas_history.values()) if alm_lambdas_history else 0
                    iterations = list(range(max_len))
                
                # Plot lambda for each constraint
                for constraint_name, lambda_history in alm_lambdas_history.items():
                    if len(lambda_history) > 0:
                        # Align iterations with history length
                        plot_iterations = iterations[:len(lambda_history)]
                        ax.plot(plot_iterations, lambda_history, label=f'{constraint_name}', linewidth=2)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('λ (Lagrange Multiplier)')
                ax.set_title('ALM Lagrange Multipliers (λ) History')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
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
            points_np = points_3d.clone().detach().cpu().numpy()
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
                    x_pos, y_pos = string_xy[s].clone().detach().cpu().numpy()
                    
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
                images = self._pad_frames_to_max_size(images)
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
                frames = self._pad_frames_to_max_size(self.gif_frames)
                imageio.mimsave(gif_filename, frames, fps=gif_fps)
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
                # self.gif_temp_dir = None
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
        print("GIF temporary files cleanup completed.")

def plot_nll_landscape(llrnet, signal_sampler, signal_surrogate_func,
                       param_names=None, param_ranges=None, n_points=50,
                       event_labels=['position', 'energy', 'zenith', 'azimuth'],
                       true_event=None, detector_point=None, figsize=(10, 8),
                       contour_levels=[0, 1, 4, 9], cmap='viridis',
                       use_mollweide=False, skip_zero_response=False, use_patd=False,
                       num_detector_points=1, min_detector_points=1,
                       min_detector_response=0.0, max_detector_resample_attempts=1000,
                       plot_opposite_direction_true_params=False,
                       use_rich_features=False,
                       progress_print_every_n_points=None):
    """
    Plot negative log-likelihood landscape for a trained signal-only LLRnet.
    
    This function visualizes how the predicted negative log-likelihood changes when
    varying one or two event parameters while keeping the detector response fixed
    to that of a true event. This helps assess whether the network has learned
    the correct relationship between event parameters and detector response.
    
    The predicted log-likelihood is normalized such that the true event parameters
    have NLL = 0, allowing interpretation of contours as confidence levels
    (NLL = 1, 4, 9 correspond to 1σ, 2σ, 3σ for chi-squared with 1-2 DOF).
    
    When multiple detector points are provided, the log-likelihoods are summed
    across all points to produce a combined likelihood landscape.
    
    Parameters:
    -----------
    llrnet : LLRnet
        Trained LLRnet instance (must be trained with signal-only approach)
    signal_sampler : ToySampler
        Sampler instance for generating signal event parameters and detector points
    signal_surrogate_func : callable
        Function to calculate light yield for signal events (or PATD generator if use_patd=True)
    param_names : list of str, optional
        Names of 1 or 2 parameters to vary. Must be keys in event_labels.
        If None, defaults to ['energy', 'zenith'] for 2D or ['energy'] for 1D.
        Examples: ['energy'], ['zenith', 'azimuth'], ['energy', 'zenith']
    param_ranges : dict, optional
        Dictionary mapping parameter names to (min, max) tuples.
        If None, uses sampler's default ranges.
        Example: {'energy': (1.0, 10.0), 'zenith': (0, np.pi)}
    n_points : int
        Number of points to sample along each parameter axis
    event_labels : list
        List of event parameter keys to include as features
    true_event : dict, optional
        True event parameters. If None, samples a new event from signal_sampler.
    detector_point : torch.Tensor or list of torch.Tensor, optional
        Detector point coordinates. Can be a single point or a list/tensor of multiple points.
        If None, detector points are sampled from signal_sampler until at least
        min_detector_points satisfy response >= min_detector_response.
        If multiple points, log-likelihoods are summed.
    num_detector_points : int
        Number of detector points to sample per resampling attempt when detector_point is None.
    min_detector_points : int
        Minimum number of detector points required to satisfy the detector response threshold.
    min_detector_response : float
        Minimum detector response threshold used when selecting sampled detector points.
    max_detector_resample_attempts : int
        Maximum number of resampling attempts before raising an error.
    plot_opposite_direction_true_params : bool
        If True, and if zenith and/or azimuth are plotted, also mark the opposite
        direction corresponding to the same physical line through the detector.
        Opposite direction is computed as:
        - zenith' = pi - zenith
        - azimuth' = (azimuth + pi) mod 2*pi
    figsize : tuple
        Figure size (width, height)
    contour_levels : list
        NLL contour levels to plot (default: [0, 1, 4, 9])
    cmap : str
        Colormap for the plot
    use_mollweide : bool
        If True and param_names are ['zenith', 'azimuth'] or ['azimuth', 'zenith'],
        use Mollweide projection for plotting (default: False)
    skip_zero_response : bool
        If True, skip detector points with zero response when calculating total LLR
        (effectively adds zero to the sum for those points). This can be useful when
        some detector points have zero expected response for certain parameter values.
        (default: False)
    use_patd : bool
        If True, uses PATD (Photon Arrival Time Distribution) mode with evaluate_patd_likelihood
        method. The likelihoods from all photon hits across all detector points are summed.
        (default: False)
    use_rich_features : bool
        If True, passes use_rich_features=True to evaluate_patd_likelihood, which uses
        prepare_features_patd instead of prepare_data_from_raw_patd. Must match the flag
        used during training. Only relevant when use_patd=True. (default: False)
    progress_print_every_n_points : int or None
        If set to a positive integer, print progress every N processed landscape points.

   """

    
    if not llrnet.is_trained:
        raise RuntimeError("LLRnet must be trained before plotting NLL landscape")
    
    # Default parameter names
    if param_names is None:
        param_names = ['energy', 'zenith']
    
    if len(param_names) > 2:
        raise ValueError("Can only vary up to 2 parameters")
    
    # Validate parameter names
    # for param_name in param_names:
    #     if param_name not in event_labels and param_name not in ['position', 'x', 'y', 'z']:
    #         raise ValueError(f"Parameter '{param_name}' not in event_labels or position coordinates: {event_labels}")
    
    # Sample true event and detector point if not provided
    if true_event is None:
        true_event = signal_sampler.sample_events(1)[0]

    progress_print_every_n_points = (
        int(progress_print_every_n_points)
        if progress_print_every_n_points is not None
        else None
    )
    if progress_print_every_n_points is not None and progress_print_every_n_points <= 0:
        progress_print_every_n_points = None

    def _maybe_print_landscape_progress(processed_points, total_points, landscape_name):
        if progress_print_every_n_points is None:
            return
        if processed_points % progress_print_every_n_points != 0 and processed_points != total_points:
            return
        print(
            f"{landscape_name}: processed {processed_points}/{total_points} landscape points"
        )

    def _extract_response_scalar(response_obj):
        """Convert surrogate response to a float for thresholding/filtering."""
        resp = response_obj
        if use_patd and isinstance(resp, dict):
            resp = resp.get('num_photons', 0.0)
        if isinstance(resp, torch.Tensor):
            if resp.numel() == 1:
                return float(resp.item())
            return float(resp.reshape(-1)[0].item())
        try:
            return float(resp)
        except Exception:
            return float('nan')

    def _event_direction_angles(event_data):
        """Return (zenith, azimuth) as floats when available."""
        zen = None
        azi = None

        if event_data.get('zenith') is not None:
            zen = _extract_response_scalar(event_data.get('zenith'))
        if event_data.get('azimuth') is not None:
            azi = _extract_response_scalar(event_data.get('azimuth'))

        if (zen is None or not np.isfinite(zen) or azi is None or not np.isfinite(azi)) and event_data.get('direction') is not None:
            try:
                theta_tmp, phi_tmp = cart_to_sph(event_data['direction'])
                if zen is None or not np.isfinite(zen):
                    zen = _extract_response_scalar(theta_tmp)
                if azi is None or not np.isfinite(azi):
                    azi = _extract_response_scalar(phi_tmp)
            except Exception:
                pass

        if zen is None or not np.isfinite(zen) or azi is None or not np.isfinite(azi):
            return None, None

        azi = float(np.mod(azi, 2 * np.pi))
        return float(zen), azi

    def _opposite_direction_angles(zen, azi):
        """Return opposite direction angles for the same line through detector."""
        if zen is None or azi is None:
            return None, None
        return float(np.pi - zen), float(np.mod(azi + np.pi, 2 * np.pi))

    # Normalize detector point inputs into a list of tensors.
    def _to_detector_points_list(detector_point_input):
        if isinstance(detector_point_input, list):
            return [
                p.to(llrnet.device) if isinstance(p, torch.Tensor)
                else torch.tensor(p, device=llrnet.device)
                for p in detector_point_input
            ]
        if isinstance(detector_point_input, torch.Tensor):
            if detector_point_input.ndim == 1:
                return [detector_point_input.to(llrnet.device)]
            return [p.to(llrnet.device) for p in detector_point_input]
        return [torch.tensor(detector_point_input, device=llrnet.device)]

    true_detector_responses = None
    if detector_point is None:
        batch_size = max(1, int(num_detector_points))
        min_required = max(1, int(min_detector_points))
        threshold = float(min_detector_response)
        max_attempts = max(1, int(max_detector_resample_attempts))

        selected_points = []
        selected_responses = []
        attempts = 0
        while len(selected_points) < min_required:
            attempts += 1
            sampled_raw = signal_sampler.sample_detector_points(batch_size)
            sampled_points = _to_detector_points_list(sampled_raw)

            batch_effective_points = []
            batch_effective_responses = []

            for p in sampled_points:
                resp_scalar = _extract_response_scalar(
                    signal_surrogate_func(opt_point=p, event_params=true_event)
                )
                if np.isfinite(resp_scalar) and resp_scalar >= threshold:
                    batch_effective_points.append(p)
                    batch_effective_responses.append(resp_scalar)

            if batch_effective_points:
                remaining_needed = min_required - len(selected_points)
                if len(batch_effective_points) <= remaining_needed:
                    selected_points.extend(batch_effective_points)
                    selected_responses.extend(batch_effective_responses)
                else:
                    keep_indices = np.random.choice(
                        len(batch_effective_points),
                        size=remaining_needed,
                        replace=False,
                    )
                    for keep_idx in keep_indices:
                        selected_points.append(batch_effective_points[keep_idx])
                        selected_responses.append(batch_effective_responses[keep_idx])

        # if len(selected_points) < min_required:
        #     raise RuntimeError(
        #         f"Unable to find enough detector points meeting response >= {threshold}. "
        #         f"Found {len(selected_points)} points after {attempts} attempts; "
        #         f"required at least {min_required}."
        #     )

        detector_points = selected_points
        true_detector_responses = selected_responses
    else:
        detector_points = _to_detector_points_list(detector_point)
    
    # num_detector_points = len(detector_points)
    
    if true_event.get('azimuth') is not None:
        if true_event['azimuth'] < 0:
            true_event['azimuth'] += 2 * np.pi

    true_zenith, true_azimuth = _event_direction_angles(true_event)
    opp_zenith, opp_azimuth = _opposite_direction_angles(true_zenith, true_azimuth)

    # Get default parameter ranges from sampler if not provided
    if param_ranges is None:
        param_ranges = {}
        for param_name in param_names:
            if param_name == 'energy':
                param_ranges['energy'] = (0.8, 1.0)  # Default energy range
            elif param_name == 'zenith':
                param_ranges['zenith'] = (-np.pi, np.pi)
            elif param_name == 'azimuth':
                param_ranges['azimuth'] = (0, 2*np.pi)
            elif param_name == 'position':
                param_ranges['position'] = (-llrnet.domain_size/2, llrnet.domain_size/2)
            elif param_name in ['x', 'y', 'z']:
                param_ranges[param_name] = (-llrnet.domain_size/2, llrnet.domain_size/2)
            else:
                # For other parameters, try to infer from true event
                if param_name in true_event:
                    val = true_event[param_name]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    param_ranges[param_name] = (val * 0.5, val * 2.0)
                else:
                    param_ranges[param_name] = (0.0, 1.0)
    
    # Calculate true detector response for all detector points (fixed for all parameter variations).
    # In PATD mode we also store the full surrogate result so that the same photon times
    # are reused for every hypothesis in the grid — matching the non-PATD behaviour where
    # the true light yield is held fixed while only the hypothesis parameters change.
    true_patd_results = None  # list of raw surrogate dicts, only populated when use_patd=True
    if true_detector_responses is None:
        true_detector_responses = []
        if use_patd:
            true_patd_results = []
        for det_point in detector_points:
            response = signal_surrogate_func(
                opt_point=det_point,
                event_params=true_event
            )
            true_detector_responses.append(_extract_response_scalar(response))
            if use_patd:
                true_patd_results.append(response)  # store full dict for later reuse
    elif use_patd:
        # responses were pre-computed during resampling (scalar only); call surrogate again
        # to get the full PATD dicts.  This is one surrogate call per detector point, done
        # once before the grid loop, which is the same cost as the non-PATD resampling path.
        true_patd_results = []
        for det_point in detector_points:
            true_patd_results.append(
                signal_surrogate_func(opt_point=det_point, event_params=true_event)
            )

    # Count effective detector points (non-zero response)
    num_effective_detector_points = sum(1 for resp in true_detector_responses if resp != 0.0)

    # Get true event features for all detector points and sum their log-likelihoods
    true_llr_sum = 0.0
    with torch.no_grad():
        if use_patd:
            # In PATD mode, sum likelihoods across all photon hits from all detector points.
            # Pass the pre-computed patd_result so the surrogate is not called again.
            for det_point, true_patd in zip(detector_points, true_patd_results):
                llr_result = llrnet.evaluate_patd_likelihood(
                    point=det_point,
                    event_data=true_event,
                    signal_surrogate_func=signal_surrogate_func,
                    event_labels=event_labels,
                    use_rich_features=use_rich_features,
                    patd_result=true_patd,
                )
                true_llr_sum += llr_result['joint_log_likelihood']
        else:
            # Standard mode using light yield features
            for det_point in detector_points:
                true_features = llrnet.prepare_data_from_raw(
                    point=det_point,
                    event_data=true_event,
                    surrogate_func=signal_surrogate_func,
                    event_labels=event_labels,
                    noise_scale=llrnet.signal_noise_scale,
                )
                true_llr_sum += llrnet.predict_log_likelihood_ratio(true_features.unsqueeze(0)).item()
    
    # Create parameter grids
    if len(param_names) == 1:
        # 1D case
        param_name = param_names[0]
        param_min, param_max = param_ranges[param_name]
        # Use log spacing for energy
        if param_name == 'energy':
            param_values = np.logspace(np.log10(param_min), np.log10(param_max), n_points)
        else:
            param_values = np.linspace(param_min, param_max, n_points)
        
        # Calculate NLL for each parameter value (summed across detector points)
        nll_values = []
        processed_landscape_points = 0
        total_landscape_points = len(param_values)
        
        for param_val in param_values:
            # Create modified event with varied parameter
            modified_event = {k: v.clone() if torch.is_tensor(v) else v for k, v in true_event.items()}
            
            # Update direction if varying zenith/azimuth and network trained with direction
            if 'direction' in event_labels and (param_name == 'zenith' or param_name == 'azimuth'):
                theta = true_event['zenith'].item() if 'zenith' in true_event else cart_to_sph(modified_event['direction'])[0].item()
                phi = true_event['azimuth'].item() if 'azimuth' in true_event else cart_to_sph(modified_event['direction'])[1].item()
                
                if param_name == 'zenith':
                    theta = torch.tensor(param_val, dtype=torch.float32)
                elif param_name == 'azimuth':
                    phi = torch.tensor(param_val, dtype=torch.float32)
                modified_event['direction'] = sph_to_cart(theta, phi)
            
            # Set zenith/azimuth in modified_event if they are being varied
            if param_name == 'zenith':
                modified_event['zenith'] = torch.tensor([param_val], dtype=torch.float32)
            elif param_name == 'azimuth':
                modified_event['azimuth'] = torch.tensor([param_val], dtype=torch.float32)
            
            # Update the specific parameter
            if param_name == 'position':
                # Special handling for position (3D vector)
                if isinstance(modified_event[param_name], torch.Tensor):
                    modified_event[param_name] = modified_event[param_name].clone()
                    modified_event[param_name][0] = param_val  # Vary first coordinate
                else:
                    modified_event[param_name][0] = param_val
            elif param_name in ['x', 'y', 'z']:
                # Handle individual position coordinates
                coord_idx = {'x': 0, 'y': 1, 'z': 2}[param_name]
                if 'position' in modified_event:
                    if isinstance(modified_event['position'], torch.Tensor):
                        modified_event['position'] = modified_event['position'].clone()
                        modified_event['position'][0][coord_idx] = param_val
                    else:
                        modified_event['position'][0][coord_idx] = param_val
            elif param_name in event_labels:
                # Only set if parameter is in event_labels
                modified_event[param_name] = torch.tensor([param_val], dtype=torch.float32)
            
            # Sum log-likelihoods across all detector points
            llr_sum = 0.0
            filtered_true_event = {k: v for k, v in true_event.items() if k in event_labels}

            patd_iter = true_patd_results if use_patd else [None] * len(detector_points)
            for det_point, true_response, true_patd in zip(detector_points, true_detector_responses, patd_iter):
                # Skip if response is zero and skip_zero_response is True
                if skip_zero_response and true_response == 0.0:
                    continue

                with torch.no_grad():
                    if use_patd:
                        # Pass the pre-computed true patd_result so the fixed observation
                        # (photon times from the true event) is reused for every hypothesis.
                        llr_result = llrnet.evaluate_patd_likelihood(
                            point=det_point,
                            event_data=modified_event,
                            signal_surrogate_func=signal_surrogate_func,
                            event_labels=event_labels,
                            use_rich_features=use_rich_features,
                            patd_result=true_patd,
                        )
                        llr_sum += llr_result['joint_log_likelihood']
                    else:
                        # Standard mode: create features with modified parameters but TRUE detector response
                        features = llrnet.prepare_data_from_raw(
                            point=det_point,
                            event_data=modified_event,  # Full event for surrogate (if called)
                            surrogate_func=signal_surrogate_func,
                            signal_event_data=true_event,  # Filtered for feature extraction
                            event_labels=event_labels,
                            noise_scale=0.0,  # No noise for evaluation
                        )

                        # Predict LLR and add to sum
                        llr = llrnet.predict_log_likelihood_ratio(features.unsqueeze(0)).item()
                        llr_sum += llr

            # Store raw NLL (will normalize later)
            nll = -llr_sum
            nll_values.append(nll)
            processed_landscape_points += 1
            _maybe_print_landscape_progress(
                processed_landscape_points,
                total_landscape_points,
                f"NLL landscape ({param_name})",
            )
        
        nll_values = np.array(nll_values)
        
        # Normalize to minimum NLL value
        min_nll = np.min(nll_values)
        nll_values = nll_values - min_nll
        
        # Find minimum location
        min_idx = np.argmin(nll_values)
        min_param_val = param_values[min_idx]
        
        # Create 1D plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(param_values, nll_values, 'b-', linewidth=2)
        
        # Mark minimum NLL value
        ax.plot(min_param_val, 0.0, 'g*', markersize=15, 
               markeredgecolor='black', markeredgewidth=1.5, label='Minimum NLL', zorder=5)
        
        # Mark true parameter value
        if param_name in ['x', 'y', 'z']:
            coord_idx = {'x': 0, 'y': 1, 'z': 2}[param_name]
            true_param_val = true_event['position'][0][coord_idx]
            if isinstance(true_param_val, torch.Tensor):
                true_param_val = true_param_val.item()
        else:
            true_param_val = true_event[param_name]
            if isinstance(true_param_val, torch.Tensor):
                true_param_val = true_param_val.item() if true_param_val.numel() == 1 else true_param_val[0].item()
        true_nll_val = nll_values[np.argmin(np.abs(param_values - true_param_val))]
        label_text = f'True value'
        ax.axvline(true_param_val, color='r', linestyle='--', linewidth=2, label=label_text)
        ax.plot(
            true_param_val,
            true_nll_val,
            'r*',
            markersize=12,
            markeredgecolor='white',
            markeredgewidth=1.0,
            zorder=6,
        )

        if plot_opposite_direction_true_params and param_name in ('zenith', 'azimuth'):
            opp_param_val = None
            if param_name == 'zenith':
                opp_param_val = opp_zenith
            elif param_name == 'azimuth':
                opp_param_val = opp_azimuth

            if opp_param_val is not None and np.isfinite(opp_param_val):
                if not np.isfinite(true_param_val) or abs(float(opp_param_val) - float(true_param_val)) > 1e-12:
                    ax.axvline(
                        opp_param_val,
                        color='magenta',
                        linestyle='--',
                        linewidth=2,
                        label='Opposite-direction true value',
                    )
                    opp_nll_val = nll_values[np.argmin(np.abs(param_values - opp_param_val))]
                    ax.plot(
                        opp_param_val,
                        opp_nll_val,
                        'm*',
                        markersize=11,
                        markeredgecolor='white',
                        markeredgewidth=1.0,
                        zorder=6,
                    )
        
        # Add horizontal lines for contour levels
        for level in contour_levels:
            ax.axhline(level, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(param_min + 0.02*(param_max-param_min), level, f'NLL={level}', 
                   fontsize=9, va='bottom', color='gray')
        
        ax.set_xlabel(param_name.capitalize(), fontsize=12)
        ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
        title_suffix = f' ({num_effective_detector_points} effective detector points)'
        ax.set_title(f'NLL Landscape: {param_name}{title_suffix}', fontsize=14)
        if param_name == 'energy':
            ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        result = {
            'fig': fig,
            'ax': ax,
            'true_event': true_event,
            'true_nll': true_nll_val,
            'detector_points': detector_points,
            'eff_num_detector_points': num_effective_detector_points,
            'param_grid': param_values,
            'nll_grid': nll_values
        }
        
    else:
        # 2D case
        param1_name, param2_name = param_names
        param1_min, param1_max = param_ranges[param1_name]
        param2_min, param2_max = param_ranges[param2_name]
        
        # Check if we should use Mollweide projection
        is_mollweide = (use_mollweide and 
                       set(param_names) == {'zenith', 'azimuth'})
        
        # Use log spacing for energy parameters
        if param1_name == 'energy':
            param1_values = np.logspace(np.log10(param1_min), np.log10(param1_max), n_points)
        else:
            param1_values = np.linspace(param1_min, param1_max, n_points)
        
        if param2_name == 'energy':
            param2_values = np.logspace(np.log10(param2_min), np.log10(param2_max), n_points)
        else:
            param2_values = np.linspace(param2_min, param2_max, n_points)
        
        param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
        nll_grid = np.zeros_like(param1_grid)
        processed_landscape_points = 0
        total_landscape_points = n_points * n_points
        
        # Calculate NLL for each parameter combination (summed across detector points)
        for i in range(n_points):
            for j in range(n_points):
                # Create modified event
                modified_event = {k: v.clone() if torch.is_tensor(v) else v for k, v in true_event.items()}
                
                # Update direction if varying zenith/azimuth and network trained with direction
                if 'direction' in event_labels:
                    if param1_name == 'zenith' or param2_name == 'zenith' or param1_name == 'azimuth' or param2_name == 'azimuth':
                        theta = true_event['zenith'] if 'zenith' in true_event else cart_to_sph(true_event['direction'])[0]
                        phi = true_event['azimuth'] if 'azimuth' in true_event else cart_to_sph(true_event['direction'])[1]
                        
                        if param1_name == 'zenith':
                            theta = torch.tensor(param1_grid[i, j], dtype=torch.float32)
                        elif param1_name == 'azimuth':
                            phi = torch.tensor(param1_grid[i, j], dtype=torch.float32)
                        if param2_name == 'zenith':
                            theta = torch.tensor(param2_grid[i, j], dtype=torch.float32)
                        elif param2_name == 'azimuth':
                            phi = torch.tensor(param2_grid[i, j], dtype=torch.float32)
                        
                        modified_event['direction'] = sph_to_cart(theta, phi)
                
                # Set zenith/azimuth in modified_event if they are being varied
                if param1_name == 'zenith':
                    modified_event['zenith'] = torch.tensor([param1_grid[i, j]], dtype=torch.float32)
                elif param1_name == 'azimuth':
                    modified_event['azimuth'] = torch.tensor([param1_grid[i, j]], dtype=torch.float32)
                if param2_name == 'zenith':
                    modified_event['zenith'] = torch.tensor([param2_grid[i, j]], dtype=torch.float32)
                elif param2_name == 'azimuth':
                    modified_event['azimuth'] = torch.tensor([param2_grid[i, j]], dtype=torch.float32)
                
                # Update both parameters (only if in event_labels or position-related)
                for param_name, param_val in [(param1_name, param1_grid[i, j]), 
                                               (param2_name, param2_grid[i, j])]:
                    if param_name == 'position':
                        if isinstance(modified_event[param_name], torch.Tensor):
                            modified_event[param_name] = modified_event[param_name].clone()
                            modified_event[param_name][0] = param_val
                        else:
                            modified_event[param_name][0] = param_val
                    elif param_name in ['x', 'y', 'z']:
                        # Handle individual position coordinates
                        coord_idx = {'x': 0, 'y': 1, 'z': 2}[param_name]
                        if 'position' in modified_event:
                            if isinstance(modified_event['position'], torch.Tensor):
                                modified_event['position'] = modified_event['position'].clone()
                                modified_event['position'][0][coord_idx] = param_val
                            else:
                                modified_event['position'][0][coord_idx] = param_val
                    elif param_name in event_labels:
                        # Only set if parameter is in event_labels
                        modified_event[param_name] = torch.tensor([param_val], dtype=torch.float32)
                
                # Sum log-likelihoods across all detector points
                llr_sum = 0.0
                filtered_true_event = {k: v for k, v in true_event.items() if k in event_labels}

                patd_iter = true_patd_results if use_patd else [None] * len(detector_points)
                for det_point, true_response, true_patd in zip(detector_points, true_detector_responses, patd_iter):
                    # Skip if response is zero and skip_zero_response is True
                    if skip_zero_response and true_response == 0.0:
                        continue

                    with torch.no_grad():
                        if use_patd:
                            # Pass the pre-computed true patd_result so the fixed observation
                            # (photon times from the true event) is reused for every hypothesis.
                            llr_result = llrnet.evaluate_patd_likelihood(
                                point=det_point,
                                event_data=modified_event,
                                signal_surrogate_func=signal_surrogate_func,
                                event_labels=event_labels,
                                use_rich_features=use_rich_features,
                                patd_result=true_patd,
                            )
                            llr_sum += llr_result['joint_log_likelihood']
                        else:
                            # Standard mode: create features with TRUE detector response
                            features = llrnet.prepare_data_from_raw(
                                point=det_point,
                                event_data=modified_event,  # Full event for surrogate (if called)
                                surrogate_func=signal_surrogate_func,
                                signal_event_data=filtered_true_event,  # Filtered for feature extraction
                                event_labels=event_labels,
                                noise_scale=0.0,
                            )

                            # Predict LLR and add to sum
                            llr = llrnet.predict_log_likelihood_ratio(features.unsqueeze(0)).item()
                            llr_sum += llr
                
                # Store raw NLL (will normalize later)
                nll_grid[i, j] = -llr_sum
                processed_landscape_points += 1
                _maybe_print_landscape_progress(
                    processed_landscape_points,
                    total_landscape_points,
                    f"NLL landscape ({param1_name} vs {param2_name})",
                )
        
        # Normalize to minimum NLL value
        min_nll = np.min(nll_grid)
        nll_grid = nll_grid - min_nll
        
        # Find minimum location
        min_idx = np.unravel_index(np.argmin(nll_grid), nll_grid.shape)
        min_param1_val = param1_grid[min_idx]
        min_param2_val = param2_grid[min_idx]
        
        # Create 2D contour plot
        if is_mollweide:
            # For Mollweide projection, we need azimuth in [-pi, pi] and zenith as latitude
            # Convert zenith (0 to pi) to latitude (-pi/2 to pi/2)
            # and azimuth (0 to 2pi) to longitude (-pi to pi)
            
            # Determine which parameter is which
            if param1_name == 'azimuth':
                lon_grid = param1_grid - np.pi  # Convert [0, 2pi] to [-pi, pi]
                lat_grid = np.pi/2 - param2_grid  # Convert zenith [0, pi] to latitude [pi/2, -pi/2]
                lon_values = param1_values - np.pi
                lat_values = np.pi/2 - param2_values
            else:  # param1_name == 'zenith'
                lon_grid = param2_grid - np.pi  # Convert [0, 2pi] to [-pi, pi]
                lat_grid = np.pi/2 - param1_grid  # Convert zenith [0, pi] to latitude [pi/2, -pi/2]
                lon_values = param2_values - np.pi
                lat_values = np.pi/2 - param1_values
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='mollweide')
            
            # Plot filled contours
            contourf = ax.contourf(lon_grid, lat_grid, nll_grid, 
                                   levels=20, cmap=cmap, alpha=0.7)
            
            # Plot contour lines at specific levels
            contour = ax.contour(lon_grid, lat_grid, nll_grid, 
                                levels=contour_levels, colors='white', 
                                linewidths=2, alpha=0.8)
            try:
                ax.clabel(contour, inline=True, fontsize=10, fmt='%.0f')
            except (IndexError, ValueError):
                # Skip labeling if contours are invalid or too sparse in Mollweide projection
                pass
            
            # Mark minimum NLL location (convert to lat/lon)
            if param1_name == 'azimuth':
                min_lon = min_param1_val - np.pi
                min_lat = np.pi/2 - min_param2_val
            else:
                min_lon = min_param2_val - np.pi
                min_lat = np.pi/2 - min_param1_val
            
            ax.plot(min_lon, min_lat, 'g*', markersize=20, 
                   markeredgecolor='black', markeredgewidth=2, label='Minimum NLL', zorder=5)
            
            # Mark true parameter values (convert to lat/lon)
            if param1_name in ['x', 'y', 'z']:
                coord_idx = {'x': 0, 'y': 1, 'z': 2}[param1_name]
                true_param1_val = true_event['position'][0][coord_idx]
                if isinstance(true_param1_val, torch.Tensor):
                    true_param1_val = true_param1_val.item()
            else:
                true_param1_val = true_event[param1_name]
                if isinstance(true_param1_val, torch.Tensor):
                    true_param1_val = true_param1_val.item() if true_param1_val.numel() == 1 else true_param1_val[0].item()
            
            if param2_name in ['x', 'y', 'z']:
                coord_idx = {'x': 0, 'y': 1, 'z': 2}[param2_name]
                true_param2_val = true_event['position'][0][coord_idx]
                if isinstance(true_param2_val, torch.Tensor):
                    true_param2_val = true_param2_val.item()
            else:
                true_param2_val = true_event[param2_name]
                if isinstance(true_param2_val, torch.Tensor):
                    true_param2_val = true_param2_val.item() if true_param2_val.numel() == 1 else true_param2_val[0].item()
            
            if param1_name == 'azimuth':
                true_lon = true_param1_val - np.pi
                true_lat = np.pi/2 - true_param2_val
            else:
                true_lon = true_param2_val - np.pi
                true_lat = np.pi/2 - true_param1_val
            
            # Get NLL at true parameter values
            true_idx1 = np.argmin(np.abs(param1_values - true_param1_val))
            true_idx2 = np.argmin(np.abs(param2_values - true_param2_val))
            true_nll_val = nll_grid[true_idx2, true_idx1]
            
            ax.plot(true_lon, true_lat, 'r*', markersize=20, 
                   markeredgecolor='white', markeredgewidth=2, label='True values', zorder=5)

            if plot_opposite_direction_true_params:
                opp_param1_val = true_param1_val
                opp_param2_val = true_param2_val
                if param1_name == 'zenith' and opp_zenith is not None:
                    opp_param1_val = opp_zenith
                elif param1_name == 'azimuth' and opp_azimuth is not None:
                    opp_param1_val = opp_azimuth
                if param2_name == 'zenith' and opp_zenith is not None:
                    opp_param2_val = opp_zenith
                elif param2_name == 'azimuth' and opp_azimuth is not None:
                    opp_param2_val = opp_azimuth

                if np.isfinite(opp_param1_val) and np.isfinite(opp_param2_val):
                    if param1_name == 'azimuth':
                        opp_lon = opp_param1_val - np.pi
                        opp_lat = np.pi/2 - opp_param2_val
                    else:
                        opp_lon = opp_param2_val - np.pi
                        opp_lat = np.pi/2 - opp_param1_val

                    if abs(float(opp_lon) - float(true_lon)) > 1e-12 or abs(float(opp_lat) - float(true_lat)) > 1e-12:
                        ax.plot(
                            opp_lon,
                            opp_lat,
                            'm*',
                            markersize=16,
                            markeredgecolor='white',
                            markeredgewidth=1.5,
                            label='Opposite true direction',
                            zorder=5,
                        )
            
            # Set custom tick labels in degrees
            # Azimuth: convert from [-π, π] to [0°, 360°]
            ax.set_xlabel('Azimuth (degrees)', fontsize=12)
            xticks_rad = ax.get_xticks()
            xticks_deg = [(x + np.pi) * 180 / np.pi for x in xticks_rad]
            ax.set_xticklabels([f'{int(deg)}°' for deg in xticks_deg])
            
            # Zenith: convert from [-π/2, π/2] to [0°, 180°]
            ax.set_ylabel('Zenith (degrees)', fontsize=12)
            yticks_rad = ax.get_yticks()
            yticks_deg = [(np.pi/2 - y) * 180 / np.pi for y in yticks_rad]
            ax.set_yticklabels([f'{int(deg)}°' for deg in yticks_deg])
            
            title_suffix = f' ({num_effective_detector_points} effective detector points)'
            ax.set_title(f'NLL Landscape{title_suffix}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', pad=0.07, fraction=0.046)
            cbar.set_label('Negative Log-Likelihood', fontsize=11)
            
        else:
            # Standard Cartesian plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot filled contours
            contourf = ax.contourf(param1_grid, param2_grid, nll_grid, 
                                   levels=20, cmap=cmap, alpha=0.7)
            
            # Plot contour lines at specific levels
            contour = ax.contour(param1_grid, param2_grid, nll_grid, 
                                levels=contour_levels, colors='white', 
                                linewidths=2, alpha=0.8)
            ax.clabel(contour, inline=True, fontsize=10, fmt='%.0f')
            
            # Mark minimum NLL location
            ax.plot(min_param1_val, min_param2_val, 'g*', markersize=20, 
                   markeredgecolor='black', markeredgewidth=2, label='Minimum NLL', zorder=5)
            
            # Mark true parameter values
            if param1_name in ['x', 'y', 'z']:
                coord_idx = {'x': 0, 'y': 1, 'z': 2}[param1_name]
                true_param1_val = true_event['position'][0][coord_idx]
                if isinstance(true_param1_val, torch.Tensor):
                    true_param1_val = true_param1_val.item()
          
            else:
                true_param1_val = true_event[param1_name]
                if isinstance(true_param1_val, torch.Tensor):
                    true_param1_val = true_param1_val.item() if true_param1_val.numel() == 1 else true_param1_val[0].item()
            
            if param2_name in ['x', 'y', 'z']:
                coord_idx = {'x': 0, 'y': 1, 'z': 2}[param2_name]
                true_param2_val = true_event['position'][0][coord_idx]
                if isinstance(true_param2_val, torch.Tensor):
                    true_param2_val = true_param2_val.item()
            else:
                true_param2_val = true_event[param2_name]
                if isinstance(true_param2_val, torch.Tensor):
                    true_param2_val = true_param2_val.item() if true_param2_val.numel() == 1 else true_param2_val[0].item()
            
            # Get NLL at true parameter values
            true_idx1 = np.argmin(np.abs(param1_values - true_param1_val))
            true_idx2 = np.argmin(np.abs(param2_values - true_param2_val))
            true_nll_val = nll_grid[true_idx2, true_idx1]  # Note: meshgrid uses (y, x) indexing
                
            ax.plot(true_param1_val, true_param2_val, 'r*', markersize=20, 
                   markeredgecolor='white', markeredgewidth=2, label='True values', zorder=5)

            if plot_opposite_direction_true_params:
                opp_param1_val = true_param1_val
                opp_param2_val = true_param2_val
                if param1_name == 'zenith' and opp_zenith is not None:
                    opp_param1_val = opp_zenith
                elif param1_name == 'azimuth' and opp_azimuth is not None:
                    opp_param1_val = opp_azimuth
                if param2_name == 'zenith' and opp_zenith is not None:
                    opp_param2_val = opp_zenith
                elif param2_name == 'azimuth' and opp_azimuth is not None:
                    opp_param2_val = opp_azimuth

                if np.isfinite(opp_param1_val) and np.isfinite(opp_param2_val):
                    if abs(float(opp_param1_val) - float(true_param1_val)) > 1e-12 or abs(float(opp_param2_val) - float(true_param2_val)) > 1e-12:
                        ax.plot(
                            opp_param1_val,
                            opp_param2_val,
                            'm*',
                            markersize=16,
                            markeredgecolor='white',
                            markeredgewidth=1.5,
                            label='Opposite true direction',
                            zorder=5,
                        )
            
            ax.set_xlabel(param1_name.capitalize(), fontsize=12)
            ax.set_ylabel(param2_name.capitalize(), fontsize=12)
            title_suffix = f' ({num_effective_detector_points} effective detector points)'
            ax.set_title(f'NLL Landscape: {param1_name} vs {param2_name}{title_suffix}', fontsize=14)
            if param1_name == 'energy':
                ax.set_xscale('log')
            if param2_name == 'energy':
                ax.set_yscale('log')
            #fix legend to bottom left corner, next to cbar
            fig.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], loc='lower left')
            
            # Add colorbar
            cbar = plt.colorbar(contourf, ax=ax)
            cbar.set_label('Negative Log-Likelihood', fontsize=11)
        
        result = {
            'fig': fig,
            'ax': ax,
            'true_event': true_event,
            'true_nll': true_nll_val,
            'min_nll_params': {param1_name: min_param1_val, param2_name: min_param2_val},
            'detector_points': detector_points,
            'eff_num_detector_points': num_effective_detector_points,
            'param_grid': (param1_grid, param2_grid),
            'nll_grid': nll_grid
        }
    
    plt.tight_layout()
    return result

def plot_nll_landscape_compare(
    llrnet, 
    signal_sampler, 
    signal_surrogate_func,
    param_names=None, 
    param_ranges=None, 
    n_points=50,
    event_labels=['position', 'energy', 'zenith', 'azimuth'],
    true_event=None, 
    detector_point=None, 
    figsize=(10, 8),
    contour_levels=[1, 4, 9], 
    cmap='viridis',
    logscale_e = True
):
    """
    Compares the NLL landscape from LLRnet with a true Poisson NLL landscape.

    This function visualizes how the predicted negative log-likelihood from the network
    compares to an analytical NLL derived from a Poisson distribution. It plots contour
    lines for both landscapes on the same axes.
    
    When multiple detector points are provided, the log-likelihoods are summed
    across all points to produce a combined likelihood landscape for both the
    network prediction and the true Poisson likelihood.

    Parameters:
    -----------
    llrnet : LLRnet
        Trained LLRnet instance (must be trained with signal-only approach).
    signal_sampler : ToySampler
        Sampler for generating signal event parameters and detector points.
    signal_surrogate_func : callable
        Function to calculate light yield for signal events (e.g., LightSabre).
    param_names : list of str, optional
        Names of 2 parameters to vary. Defaults to ['energy', 'zenith'].
    param_ranges : dict, optional
        Dictionary mapping parameter names to (min, max) tuples.
    n_points : int
        Number of points to sample along each parameter axis.
    event_labels : list
        List of event parameter keys to include as features.
    true_event : dict, optional
        True event parameters. If None, a new event is sampled.
    detector_point : torch.Tensor or list of torch.Tensor, optional
        Detector point coordinates. Can be a single point or a list/tensor of multiple points.
        If None, a new point is sampled. If multiple points, log-likelihoods are summed.
    figsize : tuple
        Figure size for the plot.
    contour_levels : list
        NLL contour levels to plot (default: [1, 4, 9] for ~1-3 sigma).
    cmap : str
        Colormap for the plot background.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not llrnet.is_trained:
        raise RuntimeError("LLRnet must be trained before plotting NLL landscape")

    if param_names is None:
        param_names = ['energy', 'zenith']
    if len(param_names) != 2:
        raise ValueError("This comparison function is designed for 2 parameters.")

    # 1. Setup: Get true event, detector point, and observed light yield
    if true_event is None:
        true_event = signal_sampler.sample_events(1)[0]
    if detector_point is None:
        detector_point = signal_sampler.sample_detector_points(1).squeeze()
    
    # Handle multiple detector points
    if isinstance(detector_point, list):
        detector_points = [p.to(llrnet.device) if isinstance(p, torch.Tensor) else torch.tensor(p, device=llrnet.device) for p in detector_point]
    elif isinstance(detector_point, torch.Tensor):
        if detector_point.ndim == 1:
            detector_points = [detector_point.to(llrnet.device)]
        else:
            detector_points = [p.to(llrnet.device) for p in detector_point]
    else:
        detector_points = [torch.tensor(detector_point, device=llrnet.device)]
    
    num_detector_points = len(detector_points)

    # Get observed light yields for all detector points
    ly_observed_list = []
    with torch.no_grad():
        for det_point in detector_points:
            ly_obs = signal_surrogate_func(opt_point=det_point, event_params=true_event)
            # Ensure observed light yield is non-negative for Poisson likelihood
            ly_obs = torch.clamp(ly_obs, min=0.0)
            ly_observed_list.append(ly_obs)

    # 2. Create parameter grids
    param1_name, param2_name = param_names
    if param_ranges is None:
        param_ranges = {}
        for param_name in param_names:
            true_val = true_event[param_name].item()
            param_ranges[param_name] = (true_val * 0.5, true_val * 1.5)

    param1_min, param1_max = param_ranges[param1_name]
    param2_min, param2_max = param_ranges[param2_name]
    
    # Use log spacing for energy parameters
    if param1_name == 'energy':
        param1_values = np.logspace(np.log10(param1_min), np.log10(param1_max), n_points)
    else:
        param1_values = np.linspace(param1_min, param1_max, n_points)
    
    if param2_name == 'energy':
        param2_values = np.logspace(np.log10(param2_min), np.log10(param2_max), n_points)
    else:
        param2_values = np.linspace(param2_min, param2_max, n_points)
    
    param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
    
    nll_grid_net = np.zeros_like(param1_grid)
    nll_grid_true = np.zeros_like(param1_grid)

    # 3. Calculate NLL grids for both models (summed across detector points)
    for i in range(n_points):
        for j in range(n_points):
            hypothesis_event = {k: v.clone() if torch.is_tensor(v) else v for k, v in true_event.items()}

            # Update direction if varying zenith/azimuth and network trained with direction
            if 'direction' in event_labels:
                if param1_name == 'zenith' or param2_name == 'zenith' or param1_name == 'azimuth' or param2_name == 'azimuth':
                    # Get current angles from direction
                    theta = true_event['zenith'] if 'zenith' in true_event else cart_to_sph(true_event['direction'])[0]
                    phi = true_event['azimuth'] if 'azimuth' in true_event else cart_to_sph(true_event['direction'])[1]
                    
                    # Update the angle being varied
                    if param1_name == 'zenith':
                        theta = torch.tensor(param1_grid[i, j], dtype=torch.float32)
                    elif param1_name == 'azimuth':
                        phi = torch.tensor(param1_grid[i, j], dtype=torch.float32)
                    if param2_name == 'zenith':
                        theta = torch.tensor(param2_grid[i, j], dtype=torch.float32)
                    elif param2_name == 'azimuth':
                        phi = torch.tensor(param2_grid[i, j], dtype=torch.float32)
                    
                    # Update direction vector
                    hypothesis_event['direction'] = sph_to_cart(theta, phi)
            
            # Set zenith/azimuth explicitly if they are being varied
            if param1_name == 'zenith':
                hypothesis_event['zenith'] = torch.tensor([param1_grid[i, j]], device=llrnet.device, dtype=torch.float32)
            elif param1_name == 'azimuth':
                hypothesis_event['azimuth'] = torch.tensor([param1_grid[i, j]], device=llrnet.device, dtype=torch.float32)
            if param2_name == 'zenith':
                hypothesis_event['zenith'] = torch.tensor([param2_grid[i, j]], device=llrnet.device, dtype=torch.float32)
            elif param2_name == 'azimuth':
                hypothesis_event['azimuth'] = torch.tensor([param2_grid[i, j]], device=llrnet.device, dtype=torch.float32)
            
            # Set parameters if they're in event_labels
            if param1_name in event_labels:
                hypothesis_event[param1_name] = torch.tensor([param1_grid[i, j]], device=llrnet.device, dtype=torch.float32)
            if param2_name in event_labels:
                hypothesis_event[param2_name] = torch.tensor([param2_grid[i, j]], device=llrnet.device, dtype=torch.float32)
            
            # Sum across all detector points for both network and true likelihood
            llr_sum = 0.0
            nll_true_sum = 0.0
            
            for det_point, ly_obs in zip(detector_points, ly_observed_list):
                # a) NLL from LLRnet
                features = llrnet.prepare_data_from_raw(
                    point=det_point,
                    event_data=hypothesis_event,
                    surrogate_func=signal_surrogate_func,
                    signal_event_data=true_event,
                    event_labels=event_labels,
                )
                
                with torch.no_grad():
                    llr = llrnet.predict_log_likelihood_ratio(features.unsqueeze(0)).item()
                    llr_sum += llr
                
                # b) "True" NLL from Poisson Likelihood
                with torch.no_grad():
                    ly_expected = signal_surrogate_func(opt_point=det_point, event_params=hypothesis_event)
                    ly_expected = torch.clamp(ly_expected, min=1e-10)
                    
                    # Poisson log-likelihood: log(P(obs|expected)) = obs*log(expected) - expected - log(obs!)
                    # For NLL we need -log(P(obs|expected))
                    
                    poisson_nll = (ly_expected - ly_obs * torch.log(ly_expected)).sum().item() + torch.lgamma(ly_obs + 1).sum().item()
                    nll_true_sum += poisson_nll
            
            nll_grid_net[i, j] = -llr_sum
            nll_grid_true[i, j] = nll_true_sum

    # 4. Normalize grids and plot
    nll_grid_net -= np.min(nll_grid_net)
    nll_grid_true -= np.min(nll_grid_true)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot contour lines for both NLLs (no filled contours)
    contour_true = ax.contour(param1_grid, param2_grid, nll_grid_true, levels=contour_levels, colors='black', linewidths=2, linestyles='--')
    contour_net = ax.contour(param1_grid, param2_grid, nll_grid_net, levels=contour_levels, colors='red', linewidths=2)
    
    # Add contour labels
    ax.clabel(contour_true, inline=True, fontsize=10, fmt='%.0f')
    ax.clabel(contour_net, inline=True, fontsize=10, fmt='%.0f')
    
    # Create proxy artists for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='LLRnet Predicted NLL'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Poisson NLL'),
        Line2D([0], [0], color='green', marker='*', markersize=15, label='True Parameters'),
        Line2D([0], [0], color='red', marker='o', markersize=10, label='LLRnet Min NLL', linestyle='None'),
        Line2D([0], [0], color='black', marker='s', markersize=10, label='Poisson Min NLL', linestyle='None')
    ]

    # Mark true parameter values and minimum NLL locations
    true_param1_val = true_event[param1_name].item()
    true_param2_val = true_event[param2_name].item()
    ax.plot(true_param1_val, true_param2_val, 'g*', markersize=15, markeredgecolor='white', label='True Parameters', zorder=10)
    ax.plot(param1_grid[np.unravel_index(np.argmin(nll_grid_net), nll_grid_net.shape)],
            param2_grid[np.unravel_index(np.argmin(nll_grid_net), nll_grid_net.shape)],
            'ro', markersize=10, label='LLRnet Min NLL', zorder=10)
    ax.plot(param1_grid[np.unravel_index(np.argmin(nll_grid_true), nll_grid_true.shape)],
            param2_grid[np.unravel_index(np.argmin(nll_grid_true), nll_grid_true.shape)],
            'ks', markersize=10, label='Poisson Min NLL', zorder=10)

    ax.set_xlabel(param1_name.capitalize(), fontsize=12)
    ax.set_ylabel(param2_name.capitalize(), fontsize=12)
    title_suffix = f' ({num_detector_points} detector points)' if num_detector_points > 1 else ''
    ax.set_title(f'NLL Comparison: {param1_name} vs {param2_name}{title_suffix}', fontsize=14)
    if param1_name == 'energy':
        ax.set_xscale('log')
    if param2_name == 'energy':
        ax.set_yscale('log')
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return {
        'fig': fig, 'ax': ax, 'true_event': true_event, 'detector_points': detector_points,
        'num_detector_points': num_detector_points,
        'param_grid': (param1_grid, param2_grid), 'nll_grid_net': nll_grid_net, 'nll_grid_true': nll_grid_true
    }


def plot_nll_landscape_with_sampling(
    llrnet, 
    signal_sampler, 
    signal_surrogate_func,
    param_name='energy',
    param_range=None,
    n_param_points=50,
    num_detector_points=10,
    num_iterations=100,
    event_labels=['position', 'energy', 'zenith', 'azimuth'],
    true_event=None,
    figsize=(10, 8),
    contour_percentiles=[2.5, 97.5],  # 95% confidence interval
    cmap='viridis',
    skip_zero_response=False,
    min_detector_points = 1
):
    """
    Plot NLL landscape with uncertainty from random detector point sampling.
    
    This function evaluates the NLL landscape for a single parameter while randomly
    sampling detector points multiple times. For each parameter value, it:
    1. Randomly samples N detector points (num_detector_points times)
    2. Calculates the summed NLL across those points
    3. Repeats this process multiple times (num_iterations)
    4. Plots the median NLL and confidence interval (default 95%)
    
    This provides insight into how the NLL landscape varies with different random
    selections of detector points, showing both the typical behavior (median) and
    the uncertainty (confidence interval).
    
    Parameters:
    -----------
    llrnet : LLRnet
        Trained LLRnet instance (must be trained with signal-only approach)
    signal_sampler : ToySampler
        Sampler instance for generating signal event parameters and detector points
    signal_surrogate_func : callable
        Function to calculate light yield for signal events
    param_name : str
        Name of the parameter to vary. Must be a key in event_labels.
        Examples: 'energy', 'zenith', 'azimuth'
    param_range : tuple, optional
        (min, max) range for the parameter. If None, uses default ranges.
    n_param_points : int
        Number of points to sample along the parameter axis
    num_detector_points : int
        Number of detector points to randomly sample for each iteration
    num_iterations : int
        Number of random sampling iterations to perform at each parameter value
    event_labels : list
        List of event parameter keys to include as features
    true_event : dict, optional
        True event parameters. If None, samples a new event from signal_sampler.
    figsize : tuple
        Figure size (width, height)
    contour_percentiles : list
        Percentiles for confidence interval (default [2.5, 97.5] for 95% CI)
    cmap : str
        Colormap for the plot
    skip_zero_response : bool
        If True, skip detector points with zero response when calculating total LLR
        (effectively adds zero to the sum for those points). The title will show the
        average number of effective detector points used. (default: False)
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'fig': matplotlib figure
        - 'ax': matplotlib axis
        - 'true_event': true event parameters used
        - 'param_values': parameter values tested
        - 'nll_median': median NLL at each parameter value
        - 'nll_lower': lower confidence bound
        - 'nll_upper': upper confidence bound
        - 'nll_all': all NLL values (n_param_points, num_iterations)
        
    Example:
    --------
    >>> # Plot energy landscape with uncertainty
    >>> result = plot_nll_landscape_with_sampling(
    ...     llrnet=trained_model,
    ...     signal_sampler=sampler,
    ...     signal_surrogate_func=signal_func,
    ...     param_name='energy',
    ...     param_range=(1.0, 100.0),
    ...     n_param_points=50,
    ...     num_detector_points=20,
    ...     num_iterations=100
    ... )
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not llrnet.is_trained:
        raise RuntimeError("LLRnet must be trained before plotting NLL landscape")
    
    # Sample true event if not provided
    if true_event is None:
        true_event = signal_sampler.sample_events(1)[0]
    
    # Get parameter range
    if param_range is None:
        if param_name == 'energy':
            param_range = (0.8, 1.0)
        elif param_name == 'zenith':
            param_range = (-np.pi, np.pi)
        elif param_name == 'azimuth':
            param_range = (0, 2*np.pi)
        elif param_name in ['x', 'y', 'z']:
            param_range = (-llrnet.domain_size/2, llrnet.domain_size/2)
        else:
            # Try to infer from true event
            if param_name in true_event:
                val = true_event[param_name]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                param_range = (val * 0.5, val * 2.0)
            else:
                param_range = (0.0, 1.0)
    
    param_min, param_max = param_range
    
    # Use log spacing for energy
    if param_name == 'energy':
        param_values = np.logspace(np.log10(param_min), np.log10(param_max), n_param_points)
    else:
        param_values = np.linspace(param_min, param_max, n_param_points)
    
    # Store NLL values for all iterations
    nll_all = np.zeros((n_param_points, num_iterations))
    
    # Track effective detector points (non-zero response) for computing average
    effective_detector_counts = []
    
    # For each parameter value
    for param_idx, param_val in enumerate(param_values):
        # Create modified event with varied parameter
        modified_event = {k: v.clone() if torch.is_tensor(v) else v for k, v in true_event.items()}
        
        # Update direction if varying zenith/azimuth
        if 'direction' in event_labels and (param_name == 'zenith' or param_name == 'azimuth'):
            theta, phi = cart_to_sph(modified_event['direction'])
            if param_name == 'zenith':
                theta = torch.tensor(param_val, dtype=torch.float32)
            elif param_name == 'azimuth':
                phi = torch.tensor(param_val, dtype=torch.float32)
            modified_event['direction'] = sph_to_cart(theta, phi)
        
        # Set zenith/azimuth explicitly
        if param_name == 'zenith':
            modified_event['zenith'] = torch.tensor([param_val], dtype=torch.float32)
        elif param_name == 'azimuth':
            modified_event['azimuth'] = torch.tensor([param_val], dtype=torch.float32)
        
        # Update the specific parameter
        if param_name == 'position':
            if isinstance(modified_event[param_name], torch.Tensor):
                modified_event[param_name] = modified_event[param_name].clone()
                modified_event[param_name][0] = param_val
            else:
                modified_event[param_name][0] = param_val
        elif param_name in ['x', 'y', 'z']:
            coord_idx = {'x': 0, 'y': 1, 'z': 2}[param_name]
            if 'position' in modified_event:
                if isinstance(modified_event['position'], torch.Tensor):
                    modified_event['position'] = modified_event['position'].clone()
                    modified_event['position'][0][coord_idx] = param_val
                else:
                    modified_event['position'][0][coord_idx] = param_val
        elif param_name in event_labels:
            modified_event[param_name] = torch.tensor([param_val], dtype=torch.float32)
        
        # For each iteration, sample random detector points
        iter_idx = 0
        max_resample_attempts = 1000  # Prevent infinite loops
        while iter_idx < num_iterations:
            # Sample random detector points
            detector_points = signal_sampler.sample_detector_points(num_detector_points)
            
            # Get true detector responses for these points
            true_detector_responses = []
            for det_point in detector_points:
                true_detector_responses.append(signal_surrogate_func(
                    opt_point=det_point,
                    event_params=true_event
                ))
            
            # Count effective detector points (non-zero response)
            num_effective = sum(1 for resp in true_detector_responses if resp != 0.0)
            
            # If skip_zero_response is True, resample if no effective points
            if skip_zero_response and num_effective < min_detector_points:
                max_resample_attempts -= 1
                if max_resample_attempts > 0:
                    continue  # Resample detector points
            
            # Valid set of detector points found
            effective_detector_counts.append(num_effective)
            
            # Sum log-likelihoods across all detector points
            llr_sum = 0.0
            # filtered_true_event = {k: v for k, v in true_event.items() if k in event_labels}
            
            for det_point, true_response in zip(detector_points, true_detector_responses):
                # Skip if response is zero and skip_zero_response is True
                if skip_zero_response and true_response == 0.0:
                    continue
                    
                # Create features
                features = llrnet.prepare_data_from_raw(
                    point=det_point,
                    event_data=modified_event,
                    surrogate_func=signal_surrogate_func,
                    signal_event_data=true_event,
                    event_labels=event_labels,
                    noise_scale=0.0,
                )
                
                # Predict LLR and add to sum
                with torch.no_grad():
                    llr = llrnet.predict_log_likelihood_ratio(features.unsqueeze(0)).item()
                    llr_sum += llr
            
            # Store NLL for this iteration
            nll_all[param_idx, iter_idx] = -llr_sum
            iter_idx += 1  # Move to next iteration
    
    # Normalize NLL values for each iteration separately
    for iter_idx in range(num_iterations):
        min_nll = np.min(nll_all[:, iter_idx])
        nll_all[:, iter_idx] = nll_all[:, iter_idx] - min_nll
    
    # Calculate median and percentiles
    nll_median = np.median(nll_all, axis=1)
    nll_lower = np.percentile(nll_all, contour_percentiles[0], axis=1)
    nll_upper = np.percentile(nll_all, contour_percentiles[1], axis=1)
    
    min_nll = np.min(nll_median)
    nll_median = nll_median - min_nll
    nll_lower = nll_lower - min_nll
    nll_upper = nll_upper - min_nll
    
    # Find minimum of median
    min_idx = np.argmin(nll_median)
    min_param_val = param_values[min_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot median line
    ax.plot(param_values, nll_median, 'b-', linewidth=2, label='Median NLL')
    
    # Plot confidence interval as shaded region
    ci_label = f'{contour_percentiles[1] - contour_percentiles[0]}% CI'
    ax.fill_between(param_values, nll_lower, nll_upper, alpha=0.3, label=ci_label)
    
    # Mark minimum NLL value
    ax.plot(min_param_val, nll_median[min_idx], 'g*', markersize=15,
            markeredgecolor='black', markeredgewidth=1.5, label='Minimum Median NLL', zorder=5)
    
    # Mark true parameter value
    if param_name in ['x', 'y', 'z']:
        coord_idx = {'x': 0, 'y': 1, 'z': 2}[param_name]
        true_param_val = true_event['position'][0][coord_idx]
        if isinstance(true_param_val, torch.Tensor):
            true_param_val = true_param_val.item()
    else:
        true_param_val = true_event[param_name]
        if isinstance(true_param_val, torch.Tensor):
            true_param_val = true_param_val.item() if true_param_val.numel() == 1 else true_param_val[0].item()
    
    ax.axvline(true_param_val, color='r', linestyle='--', linewidth=2,
               label=f'True value')
    
    ax.set_xlabel(param_name.capitalize(), fontsize=12)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    
    # Calculate average effective detector points and create title
    if skip_zero_response and len(effective_detector_counts) > 0:
        avg_effective = np.mean(effective_detector_counts)
        title = f'NLL Landscape for: {param_name} with avg {avg_effective:.1f}/{num_detector_points} effective detector points'
    else:
        title = f'NLL Landscape for: {param_name} with {num_detector_points} detector points'
    ax.set_title(title, fontsize=14)
    
    if param_name == 'energy':
        ax.set_xscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'fig': fig,
        'ax': ax,
        'true_event': true_event,
        'param_values': param_values,
        'nll_median': nll_median,
        'nll_lower': nll_lower,
        'nll_upper': nll_upper,
        'nll_all': nll_all,
        'num_detector_points': num_detector_points,
        'num_iterations': num_iterations
    }
