#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoOptimizer: A class for optimizing geometric point configurations using various strategies.

This class integrates all the utility modules from NuGeo/utils:
- geometries: Different point arrangement strategies (FreePoints, DynamicString, ContinuousString)
- losses: Loss functions for optimization (RBFInterpolationLoss, SNRloss)
- surrogates: Test function generators (SkewedGaussian)
- schedulers: Learning rate scheduling methods
- vis_tools: Visualization tools
"""

import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

# Import all utilities from utils
from ..utils.geometries import Geometry, FreePoints, DynamicString, ContinuousString
from ..utils.losses import LossFunction, RBFInterpolationLoss, SNRloss
from ..utils.surrogates import Surrogate, SkewedGaussian
from ..utils.schedulers import create_scheduler
from ..utils.vis_tools import Visualizer

class GeoOptimizer:
    """
    A class for optimizing geometric point configurations using various strategies.
    
    This class integrates different geometry types, loss functions, surrogate models,
    and visualization tools to find optimal geometry points.
    """
    
    def __init__(self, 
                 dim: int = 3, 
                 domain_size: float = 2.0, 
                 epsilon: float = 30.0, 
                 device: Optional[torch.device] = None,
                 # Common optimization parameters
                 num_test_points: int = 2000, 
                 batch_size: int = 20, 
                 num_iterations: int = 300,
                 learning_rate: float = 0.1, 
                 repulsion_weight: float = 0.0001, 
                 boundary_weight: float = 40.0,
                 sampling_weight: float = 0.05, 
                 visualize_every: int = 10, 
                 slice_res: int = 50,
                 decay_sampling: bool = False, 
                 lr_scheduler_type: Optional[str] = None, 
                 lr_scheduler_params: Optional[Dict] = None,
                 # String-based optimization parameters
                 string_repulsion_weight: float = 0.0005, 
                 min_spacing: float = 0.005,
                 order_weight: float = 1.0, 
                 optimize_xy: bool = True, 
                 xy_learning_rate: float = 0.05,
                 xy_lr_scheduler_type: Optional[str] = None, 
                 xy_lr_scheduler_params: Optional[Dict] = None,
                 # SNR optimization parameters
                 signal_scale: float = 1.0,
                 background_scale: float = 10.0,
                 snr_weight: float = 1.0,
                 no_background: bool = False):
        """
        Initialize the GeoOptimizer.
        
        Parameters:
        -----------
        dim : int
            Dimensionality of the space (2 or 3)
        domain_size : float
            Size of the domain from -domain_size/2 to domain_size/2 in each dimension
        epsilon : float
            Parameter for the RBF kernel (larger values give more local influence)
        device : torch.device or None
            Device to use for computations (None for automatic selection)
            
        # Common optimization parameters
        num_test_points : int
            Number of test points to evaluate at
        batch_size : int
            Number of test functions to use in each batch
        num_iterations : int
            Number of optimization iterations
        learning_rate : float
            Learning rate for optimizer
        repulsion_weight : float
            Weight for repulsion penalty
        boundary_weight : float
            Weight for boundary penalty
        sampling_weight : float
            Weight for sampling bias (focus on high-value regions)
        visualize_every : int
            Interval for visualization during optimization
        slice_res : int
            Resolution for visualization slices
        decay_sampling : bool
            Whether to decay the sampling weight over iterations
        lr_scheduler_type : str or None
            Type of learning rate scheduler to use ('cosine', 'step', 'exp', 'linear', None)
        lr_scheduler_params : dict or None
            Parameters for the learning rate scheduler
            
        # String-based optimization parameters
        string_repulsion_weight : float
            Weight for repulsion between strings (for string-based methods)
        min_spacing : float
            Minimum spacing between points on a string (for string-based methods)
        order_weight : float
            Weight for order penalty (for string-based methods)
        optimize_xy : bool
            Whether to optimize the XY positions of strings (for string-based methods)
        xy_learning_rate : float
            Learning rate for XY position optimization (for string-based methods)
        xy_lr_scheduler_type : str or None
            Type of learning rate scheduler for XY position optimization
        xy_lr_scheduler_params : dict or None
            Parameters for the XY position learning rate scheduler
            
        # SNR optimization parameters
        signal_scale : float
            Scale factor for signal functions
        background_scale : float
            Scale factor for background functions
        snr_weight : float
            Weight for the SNR loss term
        no_background : bool
            If True, uses a constant background value instead of generated functions
        """
        self.dim = dim
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.half_domain = domain_size / 2
        
        # Set device (CPU or GPU)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Common optimization parameters
        self.num_test_points = num_test_points
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.repulsion_weight = repulsion_weight
        self.boundary_weight = boundary_weight
        self.sampling_weight = sampling_weight
        self.visualize_every = visualize_every
        self.slice_res = slice_res
        self.decay_sampling = decay_sampling
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_params = lr_scheduler_params
        
        # String-based optimization parameters
        self.string_repulsion_weight = string_repulsion_weight
        self.min_spacing = min_spacing
        self.order_weight = order_weight
        self.optimize_xy = optimize_xy
        self.xy_learning_rate = xy_learning_rate
        self.xy_lr_scheduler_type = xy_lr_scheduler_type
        self.xy_lr_scheduler_params = xy_lr_scheduler_params
        
        # SNR optimization parameters
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.snr_weight = snr_weight
        self.no_background = no_background
        
        # Initialize components
        self.surrogate_model = SkewedGaussian(device=self.device, dim=self.dim, domain_size=self.domain_size)
        self.visualizer = Visualizer(device=self.device, dim=self.dim, domain_size=self.domain_size)
        
        # Initialize other variables to be set during optimization
        self.geometry = None
        self.loss_function = None
        self.test_points = None
        self.background_funcs = []
        self.signal_funcs = []
        self.loss_history = []
        self.snr_history = []
        self.points = None
        self.optimizers = {}
        self.schedulers = {}
        
    def setup_geometry(self, geometry_type: str, **kwargs) -> None:
        """
        Set up the geometry for optimization.
        
        Parameters:
        -----------
        geometry_type : str
            Type of geometry to use ('free_points', 'dynamic_string', 'continuous_string')
        **kwargs : dict
            Additional parameters for the specific geometry type
        """
        if geometry_type.lower() == 'free_points':
            self.geometry = FreePoints(device=self.device, dim=self.dim, domain_size=self.domain_size)
        elif geometry_type.lower() == 'dynamic_string':
            self.geometry = DynamicString(
                device=self.device, 
                dim=self.dim, 
                domain_size=self.domain_size,
                random_initial_dist=kwargs.get('random_initial_dist', False),
                total_points=kwargs.get('total_points', 140),
                n_strings=kwargs.get('n_strings', 20),
                initial_random_spread=kwargs.get('initial_random_spread', 0.1),
                optimize_positions_only=kwargs.get('optimize_positions_only', False),
                min_points_per_string=kwargs.get('min_points_per_string', 2),
                optimize_xy=kwargs.get('optimize_xy', self.optimize_xy),
                even_distribution=kwargs.get('even_distribution', True)
            )
        elif geometry_type.lower() == 'continuous_string':
            self.geometry = ContinuousString(
                device=self.device, 
                dim=self.dim, 
                domain_size=self.domain_size,
                optimize_xy=kwargs.get('optimize_xy', self.optimize_xy),
                total_points=kwargs.get('total_points', 140),
                n_strings=kwargs.get('n_strings', 20),
                optimize_positions_only=kwargs.get('optimize_positions_only', False)
            )
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")

    def setup_loss(self, loss_type: str, **kwargs) -> None:
        """
        Set up the loss function for optimization.
        
        Parameters:
        -----------
        loss_type : str
            Type of loss function to use ('rbf', 'snr')
        **kwargs : dict
            Additional parameters for the specific loss function
        """
        if loss_type.lower() == 'rbf':
            self.loss_function = RBFInterpolationLoss(
                device=self.device,
                repulsion_weight=kwargs.get('repulsion_weight', self.repulsion_weight),
                boundary_weight=kwargs.get('boundary_weight', self.boundary_weight),
                epsilon=kwargs.get('epsilon', self.epsilon),
                string_repulsion_weight=kwargs.get('string_repulsion_weight', self.string_repulsion_weight),
                path_repulsion_weight=kwargs.get('path_repulsion_weight', 0),
                z_repulsion_weight=kwargs.get('z_repulsion_weight', self.repulsion_weight),
                min_dist=kwargs.get('min_dist', 1e-3),
                sampling_weight=kwargs.get('sampling_weight', self.sampling_weight),
                domain_size=self.domain_size
            )
        elif loss_type.lower() == 'snr':
            self.loss_function = SNRloss(
                device=self.device,
                repulsion_weight=kwargs.get('repulsion_weight', self.repulsion_weight),
                boundary_weight=kwargs.get('boundary_weight', self.boundary_weight),
                string_repulsion_weight=kwargs.get('string_repulsion_weight', self.string_repulsion_weight),
                path_repulsion_weight=kwargs.get('path_repulsion_weight', self.string_repulsion_weight),
                z_repulsion_weight=kwargs.get('z_repulsion_weight', self.repulsion_weight),
                min_dist=kwargs.get('min_dist', 1e-3),
                domain_size=self.domain_size,
                snr_weight=kwargs.get('snr_weight', self.snr_weight),
                signal_scale=kwargs.get('signal_scale', self.signal_scale),
                background_scale=kwargs.get('background_scale', self.background_scale),
                no_background=kwargs.get('no_background', self.no_background)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    

    def optimize(self, 
               geometry_type: str, 
               loss_type: str, 
               num_iterations: Optional[int] = None,
               visualize_every: Optional[int] = None,
               initial_geometry_dict: Optional[Dict[str, Any]] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Run the optimization process for the specified geometry and loss function.
        
        Parameters:
        -----------
        geometry_type : str
            Type of geometry to use ('free_points', 'dynamic_string', 'continuous_string')
        loss_type : str
            Type of loss function to use ('rbf', 'snr')
        num_iterations : int or None
            Number of optimization iterations. If None, uses the value from initialization.
        visualize_every : int or None
            Interval for visualization during optimization. If None, uses the value from initialization.
        initial_geometry_dict : dict or None
            Optional dictionary containing a pre-trained geometry to use as a starting point.
            This should be the result dictionary from a previous optimization, containing keys
            specific to the geometry type being used:
            - For 'free_points': should contain 'points'
            - For 'dynamic_string': should contain 'string_xy', 'z_values', 'string_indices', etc.
            - For 'continuous_string': should contain 'path_positions', 'string_xy', etc.
        **kwargs : dict
            Additional parameters for the specific geometry and loss types, and for the optimization process
            
        Returns:
        --------
        dict
            Dictionary containing optimization results
        """
        # Set up parameters
        num_iterations = num_iterations if num_iterations is not None else self.num_iterations
        visualize_every = visualize_every if visualize_every is not None else self.visualize_every
        
        # Extract optimization-specific parameters
        alternating_training = kwargs.pop('alternating_training', False)
        alternating_steps = kwargs.pop('alternating_steps', 20)
        
        # Print key parameters for debugging
        # print(f"DEBUG: even_distribution = {kwargs.get('even_distribution', True)}")
        
        # Setup geometry
        self.setup_geometry(geometry_type, **kwargs)
        
        # Set up loss function
        self.setup_loss(loss_type, **kwargs)
        
        # Generate test points
        num_test_points = kwargs.get('num_test_points', self.num_test_points)
        self.test_points = torch.rand(num_test_points, self.dim, device=self.device) * self.domain_size - self.half_domain
        
        
        # Initialize geometry
        if initial_geometry_dict is not None:
            # Pass the initial geometry to the geometry's initialize_points method
            geom_dict = self.geometry.initialize_points(initial_geometry=initial_geometry_dict)
        else:
            geom_dict = self.geometry.initialize_points()
        
        # Initialize points
        self.points = geom_dict['points']
        
        # Initialize loss histories
        self.loss_history = []
        self.snr_history = []
        self.uw_loss_history = []
        
        # Get optimizable parameters from the geometry based on type
        optimizers = {}
        schedulers = {}
        
        if geometry_type.lower() == 'free_points':
            # For free points, simply optimize the point positions
            points = geom_dict['points']
            points.requires_grad_(True)
            geom_dict['points'] = points
            optimizer = torch.optim.Adam([points], lr=self.learning_rate)
            optimizers['points'] = optimizer
            
            scheduler = create_scheduler(
                optimizer, num_iterations, 
                self.lr_scheduler_type, self.lr_scheduler_params
            )
            if scheduler:
                schedulers['points'] = scheduler
                
        elif geometry_type.lower() == 'dynamic_string':
            # For dynamic strings, optimize z-values and potentially string positions
            z_values = geom_dict.get('z_values')
            string_xy = geom_dict.get('string_xy')
            string_logits = geom_dict.get('string_logits')
            
            # Create optimizers if parameters are available
            if z_values is not None and not kwargs.get('optimize_positions_only', False):
                print(f"Optimizing z_values with shape {z_values.shape}")
                z_values.requires_grad_(True)
                geom_dict['z_values'] = z_values
                z_optimizer = torch.optim.Adam([z_values], lr=self.learning_rate)
                optimizers['z_values'] = z_optimizer
                
                scheduler = create_scheduler(
                    z_optimizer, num_iterations, 
                    self.lr_scheduler_type, self.lr_scheduler_params
                )
                if scheduler:
                    schedulers['z_values'] = scheduler
                
            # Always ensure string_xy exists and requires_grad,
            # but only add to optimizers if optimize_xy is True
            if string_xy is not None and self.optimize_xy:
                print(f"Optimizing string_xy with shape {string_xy.shape}")
                string_xy.requires_grad_(True)
                geom_dict['string_xy'] = string_xy
                xy_optimizer = torch.optim.Adam([string_xy], lr=self.xy_learning_rate)
                optimizers['string_xy'] = xy_optimizer
                
                scheduler = create_scheduler(
                    xy_optimizer, num_iterations, 
                    self.xy_lr_scheduler_type, self.xy_lr_scheduler_params
                )
                if scheduler:
                    schedulers['string_xy'] = scheduler
                    
            if string_logits is not None and not kwargs.get("even_distribution", True):
                # print(f"Optimizing string_logits with shape {string_logits.shape}")
                # print(f"DEBUG: string_logits value = {string_logits}")
                # print(f"DEBUG: even_distribution = {kwargs.get('even_distribution', True)}")
                string_logits.requires_grad_(True)
                geom_dict['string_logits'] = string_logits
                redis_optimizer = torch.optim.Adam([string_logits], lr=kwargs.get('redis_learning_rate', 0.1))
                optimizers['string_logits'] = redis_optimizer
                
                scheduler = create_scheduler(
                    redis_optimizer, num_iterations, 
                    kwargs.get('redis_lr_scheduler_type'), kwargs.get('redis_lr_scheduler_params')
                )
                if scheduler:
                    schedulers['string_logits'] = scheduler
                
        elif geometry_type.lower() == 'continuous_string':
            # For continuous strings, optimize path positions and potentially string positions
            path_positions = geom_dict.get('path_positions')
            string_xy = geom_dict.get('string_xy')
            geom_dict['num_strings'] = len(string_xy)
            
            # Create optimizers if parameters are available
            if path_positions is not None:
                path_positions.requires_grad_(True)
                geom_dict['path_positions'] = path_positions
                path_optimizer = torch.optim.Adam([path_positions], lr=self.learning_rate)
                optimizers['path_positions'] = path_optimizer
                
                scheduler = create_scheduler(
                    path_optimizer, num_iterations, 
                    self.lr_scheduler_type, self.lr_scheduler_params
                )
                if scheduler:
                    schedulers['path_positions'] = scheduler
                
            if string_xy is not None and self.optimize_xy:
                string_xy.requires_grad_(True)
                geom_dict['string_xy'] = string_xy
                xy_optimizer = torch.optim.Adam([string_xy], lr=self.xy_learning_rate)
                optimizers['string_xy'] = xy_optimizer
                
                scheduler = create_scheduler(
                    xy_optimizer, num_iterations, 
                    self.xy_lr_scheduler_type, self.xy_lr_scheduler_params
                )
                if scheduler:
                    schedulers['string_xy'] = scheduler
        
        # Store optimizers and schedulers
        self.optimizers = optimizers
        self.schedulers = schedulers
        
        # Check what kind of surrogate function mode we're in
        if loss_type.lower() == 'snr':
            # SNR loss needs both signal and background functions
            num_background_funcs = kwargs.get('num_background_funcs', 1)
            optimize_params = kwargs.get('optimize_params', None)
            parameter_ranges = kwargs.get('parameter_ranges', None)
            position_ranges = kwargs.get('position_ranges', None)
            background_params = kwargs.get('background_params', None)
            show_all_signals = kwargs.get('show_all_signals', True)
            
            # Generate background functions once at the beginning
            print(f"Creating background distribution from {num_background_funcs} functions...")
            if not self.no_background:
                self.background_funcs = self.surrogate_model.generate_surrogate_functions(
                    num_background_funcs, 'background', background_params, position_ranges
                )
            else:
                # Use an empty list as placeholder
                self.background_funcs = []
            
            # Signal functions will be generated for each batch
            batch_size = kwargs.get('batch_size', self.batch_size)
            param_grid_size = kwargs.get('param_grid_size', 20)
            keep_param_const = kwargs.get('keep_param_const', None)
            rand_params_in_grid = kwargs.get('rand_params_in_grid', False)
            rand_other_grid = kwargs.get('rand_other_grid', False)
            
            # Initialize visualization signal functions
            # vis_signal_funcs = self.generate_surrogate_functions(
            #     10, 'signal', parameter_ranges, position_ranges, optimize_params,
            #     param_grid_size, 0, keep_param_const, rand_params_in_grid
            # )
            
        elif loss_type.lower() == 'rbf':  # RBF interpolation loss
            # Generate test functions for optimization
            num_test_funcs = kwargs.get('num_test_funcs', None)
            if num_test_funcs is not None:
                print(f"Generating {num_test_funcs} test functions...")
                self.surrogate_funcs = self.surrogate_model.generate_surrogate_functions(
                    num_test_funcs, 'background'
                )
            else:
                self.surrogate_funcs = None
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Prepare for alternating training if requested
        training_phases = []
        if alternating_training:
            if 'z_values' in optimizers and 'string_xy' in optimizers:
                # For dynamic strings with alternating z and xy optimization
                training_phases = [
                    {'z_values': False, 'string_xy': True, 'string_logits': False},
                    
                ]
                if 'string_logits' in optimizers:
                    # Add redistribution phase
                    training_phases.append({'z_values': False, 'string_xy': False, 'string_logits': True})
                if not kwargs.get('optimize_positions_only', True):
                    training_phases.append({'z_values': True, 'string_xy': False, 'string_logits': False})
            elif 'path_positions' in optimizers and 'string_xy' in optimizers:
                # For continuous strings with alternating path and xy optimization
                training_phases = [
                    {'path_positions': False, 'string_xy': True},
                    {'path_positions': True, 'string_xy': False}
                ]
        
        # Start optimization loop
        print(f"Starting optimization with {num_iterations} iterations...")
        
        # for param_name in optimizers.keys():
        #     geom_dict[param_name].requires_grad_(True)
        
        current_phase_idx = 0
        for iteration in range(num_iterations):
            # Apply alternating training if requested
            if alternating_training and training_phases:
                if iteration % alternating_steps == 0:
                    # Switch to the next training phase
                    current_phase_idx = (current_phase_idx + 1) % len(training_phases)
                    phase = training_phases[current_phase_idx]
                    
                    # Print current phase for debugging
                    # active_params = [k for k, v in phase.items() if v]
                    # print(f"Iteration {iteration}: Phase {current_phase_idx}, active parameters: {active_params}")
                    
                    # Zero gradients based on current phase
                for param_name, enabled in phase.items():
                    if param_name in optimizers:
                        # if enabled:
                        optimizers[param_name].zero_grad()
                        if param_name == 'string_logits':
                            geom_dict['redis_phase'] = enabled        
            else:
                # Zero gradients for all optimizers
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                            
            # Compute loss based on loss type
            if loss_type.lower() == 'rbf':
                # RBF interpolation loss
                batch_size = kwargs.get('batch_size', self.batch_size)
                if self.surrogate_funcs is not None:
                    batch_indices = np.random.choice(len(self.surrogate_funcs), batch_size)
                    batch_surrogate_funcs = [self.surrogate_funcs[i] for i in batch_indices]
                else:
                    batch_surrogate_funcs = self.surrogate_model.generate_surrogate_functions(
                    batch_size, 'background'
                )
    
                # Call the loss function with the current geometry
                loss, uw_loss, _, _, _ = self.loss_function(
                    points_3d=self.points,
                    surrogate_funcs=batch_surrogate_funcs,
                    test_points=self.test_points,
                    **geom_dict
                )
                
                # Store loss history
                self.loss_history.append(loss.item())
                
            elif loss_type.lower() == 'snr':
                # SNR loss - generate a new batch of signal functions
                batch_size = kwargs.get('batch_size', self.batch_size)
                optimize_params = kwargs.get('optimize_params', None)
                
                # Generate signal functions for this batch
                signal_funcs = []
                for batch_num in range(batch_size):
                    signal_func = self.surrogate_model.generate_surrogate_functions(
                        1, 'signal', parameter_ranges, position_ranges, optimize_params,
                        param_grid_size, batch_num, keep_param_const, rand_params_in_grid, rand_other_grid
                    )
                    signal_funcs.extend(signal_func)
                    
                # Call the loss function with the current geometry
                loss, avg_snr, all_snr = self.loss_function(
                    points_3d=self.points,
                    signal_funcs=signal_funcs,
                    background_funcs=self.background_funcs,
                    optimize_params=optimize_params,
                    grid_size=param_grid_size,
                    **geom_dict
                )
                
                # Store loss and SNR history
                self.loss_history.append(loss.item())
                self.snr_history.append(avg_snr)
                if not loss_type.lower() == 'snr':
                    self.uw_loss_history.append(uw_loss)
                
                # Store signal functions for visualization
                if iteration == 0 or (visualize_every > 0 and iteration % visualize_every == 0):
                    if not show_all_signals:
                        indices = np.random.permutation(len(signal_funcs))[:min(len(signal_funcs), 200)]
                        vis_signal_funcs = [signal_funcs[i] for i in indices]  # Keep a few for visualization
                    else:
                        vis_signal_funcs = signal_funcs
            
            
            # print(f"\nIteration {iteration} - Parameter gradients:")
            # for param_name, optimizer in optimizers.items():
            #     param = None
            #     if param_name in geom_dict:
            #         param = geom_dict[param_name]
                
                # print(f"DEBUG: Checking param '{param_name}', in geom_dict: {param_name in geom_dict}")
                
                # if param is not None and param.grad is not None:
                #     grad_norm = param.grad.norm().item()
                #     print(f"  - {param_name}: grad_norm = {grad_norm:.6f}, requires_grad = {param.requires_grad}")
                    
                #     # For path_positions or z_values, check if they're being updated when they shouldn't be
                #     if param_name in ['path_positions', 'z_values'] and kwargs.get('optimize_positions_only', False):
                #         print(f"    WARNING: {param_name} still has gradients despite optimize_positions_only=True")
                    
                #     # For string_logits, check if they're being updated when even_distribution=True
                #     if param_name == 'string_logits' and kwargs.get('even_distribution', True):
                #         print(f"    WARNING: {param_name} still has gradients despite even_distribution=True")
                #         print(f"    This is likely a configuration issue. Make sure you're passing 'even_distribution=False' when calling optimize()")
                # else:
                #     if param is None:
                #         print(f"  - {param_name}: No parameter found in geom_dict")
                #     elif param.grad is None:
                #         print(f"  - {param_name}: Parameter exists but has no gradients, requires_grad = {param.requires_grad}")
                #     else:
                #         print(f"  - {param_name}: No gradients or parameter not found")
            
            # Backpropagate and update
            loss.backward()
            
            # Apply optimizer steps based on alternating training
            if alternating_training and training_phases:
                phase = training_phases[current_phase_idx]
                for param_name, enabled in phase.items():
                    if enabled and param_name in optimizers:
                        optimizers[param_name].step()
                        if param_name in schedulers:
                            schedulers[param_name].step()
            else:
                # Apply all optimizers
                for optimizer in optimizers.values():
                    optimizer.step()
                # Step schedulers
                for scheduler in schedulers.values():
                    scheduler.step()
            
            # Update the geometry
            geom_dict = self.geometry.update_points(**geom_dict)
            
            if geom_dict.get('z_values') is not None:
                if z_values is not None  and 'z_values' in optimizers and "string_logits" in optimizers:
                    # If z_values tensor object was replaced by update_points:
                    new_z_tensor = geom_dict['z_values']
                    original_z_tensor_in_optimizer = optimizers['z_values'].param_groups[0]['params'][0]
                    if new_z_tensor is not original_z_tensor_in_optimizer:
                        # The optimizer is tracking original_z_tensor_in_optimizer.
                        # The new_z_tensor was used to compute self.points for the next iteration.
                        # We need to ensure the optimizer tracks the new_z_tensor.
                        if original_z_tensor_in_optimizer.requires_grad:
                            new_z_tensor.requires_grad_(True) # Ensure grad status is maintained
                        optimizers['z_values'].param_groups[0]['params'] = [new_z_tensor]
              

            self.points = geom_dict['points']
            if loss_type.lower() == 'snr':
                self.signal_funcs = signal_funcs
                self.background_funcs = self.background_funcs
                self.optimize_params = optimize_params
                self.param_values = self.surrogate_model.param_values
            self.geom_dict = geom_dict
            if (visualize_every > 0 and (iteration % visualize_every == 0 or iteration == num_iterations - 1)) or kwargs.get('make_gif', True):
                vis_kwargs = {}
                if loss_type.lower() == 'snr':
                    vis_kwargs.update({
                        'signal_funcs': vis_signal_funcs,
                        'background_funcs': self.background_funcs, 
                        'optimize_params': kwargs.get('optimize_params', None),
                        'param_values': self.surrogate_model.param_values,
                        'vis_all_signals': kwargs.get('vis_all_signals', True),
                        'all_snr': all_snr,
                        'no_background': self.no_background,
                        'background_scale': self.background_scale,
                    })
                elif loss_type.lower() == 'rbf':
                    vis_kwargs.update({
                        'surrogate_funcs': batch_surrogate_funcs,
                        'surrogate_model': self.surrogate_model,
                        'compute_rbf_interpolant': self.loss_function.compute_rbf_interpolant if hasattr(self.loss_function, 'compute_rbf_interpolant') else None,
                        'uw_loss': self.uw_loss_history,
                        'vis_all_surrogates': kwargs.get('vis_all_surrogates', True),
                        
                    })
                
                if geom_dict.get('num_strings') is not None:
                    vis_kwargs.update({
                        'num_strings': geom_dict['num_strings'],
                        })
                vis_kwargs.update({"geometry_type": geometry_type})
                
                additional_metrics = {'snr_history': self.snr_history} if loss_type.lower() == 'snr' else None
            
            # Visualize progress
            if visualize_every > 0 and (iteration % visualize_every == 0 or iteration == num_iterations - 1):
       
                self.visualizer.visualize_progress(
                    iteration=iteration,
                    points_3d=self.points,
                    loss_history=self.loss_history,
                    additional_metrics=additional_metrics,
                    string_indices=geom_dict.get('string_indices'),
                    points_per_string_list=geom_dict.get('points_per_string_list'),
                    string_xy=geom_dict.get('string_xy'),
                    slice_res=self.slice_res,
                    multi_slice=kwargs.get('multi_slice', False),
                    loss_type=loss_type.lower(),
                    string_logits=geom_dict.get('string_logits'),
                    plot_types=kwargs.get("plot_types", ['loss', '3d_points']),
                    **vis_kwargs
                )
                
            if kwargs.get('make_gif', True):
                self.visualizer.visualize_progress(
                    iteration=iteration,
                    points_3d=self.points,
                    loss_history=self.loss_history,
                    additional_metrics=additional_metrics,
                    string_indices=geom_dict.get('string_indices'),
                    points_per_string_list=geom_dict.get('points_per_string_list'),
                    string_xy=geom_dict.get('string_xy'),
                    slice_res=self.slice_res,
                    multi_slice=kwargs.get('multi_slice', False),
                    loss_type=loss_type.lower(),
                    string_logits=geom_dict.get('string_logits'),
                    plot_types=kwargs.get("plot_types", ['loss', '3d_points']),
                    make_gif=True,
                    gif_filename=kwargs.get('gif_filename', 'optimization.gif'),
                    gif_fps=kwargs.get('gif_fps', 2),
                    gif_plot_selection=kwargs.get('gif_plot_selection', ['3d_points']),
                    **vis_kwargs
                )
        
        # Return the results
        results = {
            'points': self.points.detach(),
            'loss_history': self.loss_history,
            'points_per_string_list': geom_dict.get('points_per_string_list'),
            'string_indices': geom_dict.get('string_indices'),
            'string_xy': geom_dict.get('string_xy'),
            'path_positions': geom_dict.get('path_positions')
        }
        
        if loss_type.lower() == 'snr':
            results['snr_history'] = self.snr_history
            results['signal_funcs'] = vis_signal_funcs
            results['background_funcs'] = self.background_funcs
            
        return results

    def create_interactive_plot(self, points=None, string_indices=None, points_per_string_list=None, string_xy=None):
        """
        Create an interactive 3D plot of the optimized geometry.
        
        Parameters:
        -----------
        points : torch.Tensor or None
            Points to visualize. If None, uses the current points.
        string_indices : list or None
            String index for each point. If None, uses the current indices.
        points_per_string_list : list or None
            Number of points on each string. If None, uses the current counts.
        string_xy : torch.Tensor or None
            XY positions of strings. If None, uses the current positions.
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
            Interactive 3D plot if Plotly is available, None otherwise
        """
        # Use current values if not provided
        points = self.points if points is None else points
        
        return self.visualizer.create_interactive_3d_plot(
            points_3d=points,
            string_indices=string_indices,
            points_per_string_list=points_per_string_list,
            string_xy=string_xy
        )
    
    def visualize_functions(self, points=None, num_funcs=10, slice_res=None, multi_slice=False):
        """
        Visualize test functions and their interpolation quality on the optimized geometry.
        
        Parameters:
        -----------
        points : torch.Tensor or None
            Points to visualize. If None, uses the current points.
        num_funcs : int
            Number of functions to average for visualization.
        slice_res : int or None
            Resolution for visualization slices. If None, uses the current resolution.
        multi_slice : bool
            Whether to use multiple slices for visualization.
        """
        # Use current values if not provided
        points = self.points if points is None else points
        slice_res = self.slice_res if slice_res is None else slice_res
        
        # Use the visualizer
        self.visualizer.visualize_function(
            points_3d=points,
            test_points=self.test_points,
            num_funcs_viz=num_funcs,
            slice_res=slice_res,
            multi_slice=multi_slice,
            surrogate_model=self.surrogate_model,
            surrogate_funcs=self.surrogate_funcs[:num_funcs] if hasattr(self, 'surrogate_funcs') else None,
            compute_rbf_interpolant=self.loss_function.compute_rbf_interpolant if hasattr(self.loss_function, 'compute_rbf_interpolant') else None
        )
    
    def save_geometry(self, file_path: str, results: Dict[str, Any] = None) -> None:
        """
        Save optimization results or current geometry to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the geometry file
        results : dict or None
            Optimization results to save. If None, uses the current geometry.
        """
        if results is None:
            # Use current geometry state
            if not hasattr(self, 'points') or self.points is None:
                raise ValueError("No geometry has been optimized yet. Run optimize() first or provide results.")
            
            # Create a dictionary with the current geometry state
            results = {
                'points': self.points.detach().cpu()
            }
            
            # Add additional geometry information if available
            if hasattr(self, 'geom_dict'):
                for key, value in self.geom_dict.items():
                    if isinstance(value, torch.Tensor):
                        results[key] = value.detach().cpu()
                    elif isinstance(value, (list, dict)):
                        results[key] = value
        
        # Convert tensors to CPU for saving
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.detach().cpu()
        
        # Save to file
        torch.save(results, file_path)
        print(f"Geometry saved to {file_path}")
    
    def load_geometry(self, file_path: str) -> Dict[str, Any]:
        """
        Load a saved geometry from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved geometry file
            
        Returns:
        --------
        dict
            Loaded geometry dictionary that can be used as initial_geometry_dict
        """
        results = torch.load(file_path)
        print(f"Loaded geometry from {file_path}")
        return results

    def clone(self):
        """
        Create a deep copy of the GeoOptimizer instance, properly handling PyTorch tensors.
        
        Returns:
        --------
        GeoOptimizer
            A new GeoOptimizer instance with identical parameters but independent tensors.
        """
        
        
        # Create a new instance with the same initialization parameters
        new_optimizer = GeoOptimizer(
            dim=self.dim,
            domain_size=self.domain_size,
            epsilon=self.epsilon,
            device=self.device,
            # Common optimization parameters
            num_test_points=self.num_test_points,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            learning_rate=self.learning_rate,
            repulsion_weight=self.repulsion_weight,
            boundary_weight=self.boundary_weight,
            sampling_weight=self.sampling_weight,
            visualize_every=self.visualize_every,
            slice_res=self.slice_res,
            decay_sampling=self.decay_sampling,
            lr_scheduler_type=self.lr_scheduler_type if hasattr(self, 'lr_scheduler_type') else None,
            lr_scheduler_params=self.lr_scheduler_params if hasattr(self, 'lr_scheduler_params') else None,
            # String-based optimization parameters
            string_repulsion_weight=self.string_repulsion_weight,
            min_spacing=self.min_spacing,
            order_weight=self.order_weight,
            optimize_xy=self.optimize_xy,
            xy_learning_rate=self.xy_learning_rate,
            xy_lr_scheduler_type=self.xy_lr_scheduler_type if hasattr(self, 'xy_lr_scheduler_type') else None,
            xy_lr_scheduler_params=self.xy_lr_scheduler_params if hasattr(self, 'xy_lr_scheduler_params') else None,
            # SNR optimization parameters
            signal_scale=self.signal_scale,
            background_scale=self.background_scale,
            snr_weight=self.snr_weight,
            no_background=self.no_background
        )
        
        # Deep copy attributes that are not tensors
        for attr_name, attr_value in self.__dict__.items():
            if attr_name not in new_optimizer.__dict__:
                if isinstance(attr_value, torch.Tensor):
                    # Create a new tensor with the same data but detached from the computation graph
                    new_optimizer.__dict__[attr_name] = attr_value.clone().detach()
                elif isinstance(attr_value, dict):
                    # Handle dictionaries that might contain tensors
                    new_optimizer.__dict__[attr_name] = self._deep_copy_with_tensors(attr_value)
                elif isinstance(attr_value, list):
                    # Handle lists that might contain tensors
                    new_optimizer.__dict__[attr_name] = [
                        item.clone().detach() if isinstance(item, torch.Tensor) else 
                        self._deep_copy_with_tensors(item) if isinstance(item, (dict, list, tuple)) else 
                        copy.deepcopy(item)
                        for item in attr_value
                    ]
                else:
                    # Standard deep copy for other types
                    new_optimizer.__dict__[attr_name] = copy.deepcopy(attr_value)
        
        return new_optimizer
    
    def _deep_copy_with_tensors(self, obj):
        """
        Helper method to deep copy objects containing PyTorch tensors.
        
        Parameters:
        -----------
        obj : any
            The object to deep copy
            
        Returns:
        --------
        any
            A deep copy of the input object with tensors properly cloned
        """
        import copy
        
        if isinstance(obj, torch.Tensor):
            return obj.clone().detach()
        elif isinstance(obj, dict):
            return {k: self._deep_copy_with_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy_with_tensors(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._deep_copy_with_tensors(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__ attribute
            new_obj = copy.copy(obj)  # Shallow copy the object itself
            for k, v in obj.__dict__.items():
                new_obj.__dict__[k] = self._deep_copy_with_tensors(v)  # Deep copy its attributes
            return new_obj
        else:
            # For other types, use standard deepcopy
            return copy.deepcopy(obj)