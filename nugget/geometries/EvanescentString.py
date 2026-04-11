from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np
import torch.nn.functional as F


class EvanescentString(Geometry):
    """Evanescent string geometry optimizer."""
    
    def __init__(
        self,
        device=None,
        dim=3,
        domain_size=2,
        random_weights=False,
        hybrid_mix_init=0.5,
        custom_z_spacing=None,
        n_strings=1000,
        points_per_string=5,
        starting_weight=1.0,
        custom_string_spacing=None,
        hex_type='hexagonal',
        active_weights_mode: bool = False,
    ):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.n_strings = n_strings
        self.points_per_string = points_per_string
        self.starting_weight = starting_weight
        self.custom_string_spacing = custom_string_spacing
        self.random_weights = random_weights
        self.active_weights_mode = active_weights_mode
        if hex_type == 'hexagonal':
            self.hex_func = self.create_uniform_hexagonal_grid
        elif hex_type == 'circular':
            self.hex_func = self.create_circular_hexagonal_grid
        elif hex_type == 'sunflower':
            self.hex_func = self.create_sunflower_grid
        else:
            self.hex_func = self.create_uniform_hexagonal_grid
        # Create hexagonal grid for strings
        original_dim = self.dim
        self.hybrid_mix = hybrid_mix_init
        self.dim = 2
        self.hex_grid = self.hex_func(n_points=self.n_strings, optimal_spacing=self.custom_string_spacing, hybrid_mix=self.hybrid_mix)
        self.dim = original_dim
        self.custom_z_spacing = custom_z_spacing # distance between points along z-axis
        
        
        
        # Half domain size for z-value mapping
        self.half_domain = domain_size / 2.0
    
    def initialize_points(self, initial_geometry=None, **kwargs):
        """
        Initialize points in an evanescent string configuration.
        
        Parameters:
        -----------
        initial_geometry : dict or None
            Optional dictionary containing pre-trained geometry parameters to use as a starting point.
            Should contain keys like 'string_xy', 'z_values', etc.
        
        Returns:
        --------
        dict
            Dictionary with initialized torch tensors
        """
        
        active_weights_mode = kwargs.get('active_weights_mode', self.active_weights_mode)
        threshold = kwargs.get('weight_threshold', 0.7)

        def _as_tensor(value, *, dtype=torch.float32):
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=dtype)
            return torch.tensor(value, device=self.device, dtype=dtype)

        def _default_string_xy() -> torch.Tensor:
            return self.hex_grid.clone()

        def _default_z_values(n_strings: int) -> torch.Tensor:
            if self.custom_z_spacing is not None:
                z_line = self.custom_z_spacing * (
                    torch.arange(self.points_per_string, device=self.device, dtype=torch.float32)
                    - (self.points_per_string - 1) / 2.0
                )
            else:
                z_line = torch.linspace(
                    -self.half_domain,
                    self.half_domain,
                    self.points_per_string,
                    device=self.device,
                )
            return z_line.repeat(n_strings)

        def _default_raw_weights(n_strings: int) -> torch.Tensor:
            if not self.random_weights:
                return torch.ones(n_strings, device=self.device, dtype=torch.float32) * self.starting_weight
            return torch.rand(n_strings, device=self.device, dtype=torch.float32) * 8 - 4

        if initial_geometry is not None:
            print("Using pre-trained evanescent string geometry as starting point")

            if 'string_xy' in initial_geometry and initial_geometry['string_xy'] is not None:
                string_xy = _as_tensor(initial_geometry['string_xy'])
                self.n_strings = int(string_xy.shape[0])
            else:
                if 'n_strings' in initial_geometry and initial_geometry['n_strings'] is not None:
                    self.n_strings = int(initial_geometry['n_strings'])
                string_xy = _default_string_xy()

            expected_points = int(self.n_strings) * int(self.points_per_string)
            z_values = None
            if 'z_values' in initial_geometry and initial_geometry['z_values'] is not None:
                z_values_candidate = _as_tensor(initial_geometry['z_values']).reshape(-1)
                if z_values_candidate.numel() == self.points_per_string:
                    z_values = z_values_candidate.repeat(self.n_strings)
                elif z_values_candidate.numel() == expected_points:
                    z_values = z_values_candidate

            if z_values is None:
                z_values = _default_z_values(self.n_strings)

            raw_weights = None
            if 'old_string_weights' in initial_geometry and initial_geometry['old_string_weights'] is not None:
                raw_weights = _as_tensor(initial_geometry['old_string_weights']).reshape(-1)
            elif 'old_weights' in initial_geometry and initial_geometry['old_weights'] is not None:
                raw_weights = _as_tensor(initial_geometry['old_weights']).reshape(-1)
            elif 'string_weights' in initial_geometry and initial_geometry['string_weights'] is not None:
                raw_weights = _as_tensor(initial_geometry['string_weights']).reshape(-1)

            if raw_weights is None or raw_weights.numel() != self.n_strings:
                raw_weights = _default_raw_weights(self.n_strings)

            if active_weights_mode:
                old_string_weights = raw_weights
                string_weights = 2000 * (torch.sigmoid(old_string_weights) > threshold).to(dtype=torch.float32) - 1000
            else:
                string_weights = raw_weights
                old_string_weights = raw_weights
        else:
            string_xy = _default_string_xy()
            z_values = _default_z_values(self.n_strings)
            raw_weights = _default_raw_weights(self.n_strings)
            if active_weights_mode:
                old_string_weights = raw_weights
                string_weights = 2000 * (torch.sigmoid(old_string_weights) > threshold).to(dtype=torch.float32) - 1000
            else:
                string_weights = raw_weights
                old_string_weights = raw_weights

        string_indices = torch.arange(self.n_strings, device=self.device, dtype=torch.long)
        
        points_3d = torch.zeros(self.n_strings * self.points_per_string, 3, device=self.device)
        # Fill points_3d with string_xy and z_values
        for s_idx in range(self.n_strings):
            start_idx = s_idx * self.points_per_string
            end_idx = start_idx + self.points_per_string
            points_3d[start_idx:end_idx, 0] = string_xy[s_idx, 0]  # x value
            points_3d[start_idx:end_idx, 1] = string_xy[s_idx, 1]  # y value
            points_3d[start_idx:end_idx, 2] = z_values[start_idx:end_idx]  # z value
            
        return {
            # 'hybrid_mix': self.hybrid_mix,
            'points_3d': points_3d,
            'active_points': points_3d,  # Initially all points are active
            'string_xy': string_xy,
            'z_values': z_values,
            'string_weights': string_weights,
            'old_string_weights': old_string_weights,
            'string_indices': string_indices,
            'active_string_indices': string_indices,  # Initially all strings are active
            'points_per_string_list': [self.points_per_string] * self.n_strings,  # Each string has points_per_string points
            'active_weights_mode': active_weights_mode,
            'weight_threshold': threshold,
            }

    def update_points(self, string_xy, z_values, string_weights, string_indices, old_string_weights=None, **kwargs):
        """
        Update the points based on current optimization state.
        
        Parameters:
        -----------
        string_xy : torch.Tensor
            XY coordinates for each string (n_strings, 2)
        z_values : torch.Tensor
            Z values for all points (n_strings * points_per_string,)
        string_weights : torch.Tensor
            Raw weights for each string (n_strings,)
        string_indices : torch.Tensor
            Indices for each string (n_strings,)
            
        Returns:
        --------
        dict
            Dictionary with updated tensors
        """
        active_weights_mode = kwargs.get('active_weights_mode', self.active_weights_mode)

        # Backwards compatibility: older saved geom dicts won't have old_string_weights.
        if old_string_weights is None:
            old_string_weights = kwargs.get('old_weights', None)
        if old_string_weights is None:
            old_string_weights = string_weights

        # Choose which weights to threshold on (raw/old weights).
        threshold = kwargs.get('weight_threshold', 0.7)
        weights_for_thresholding = old_string_weights
        

        if active_weights_mode:
            string_weights_to_return = 200*(torch.sigmoid(weights_for_thresholding) > threshold).to(dtype=torch.float32) - 100
            old_string_weights_to_return = old_string_weights
        else:
            string_weights_to_return = old_string_weights
            old_string_weights_to_return = string_weights_to_return
        # hybrid_mix = kwargs.get('hybrid_mix', self.hybrid_mix)
        # Count how many strings are active
      
        
        # if n_active_strings == 0:
        #     # If no strings are active, return empty tensors
        #     empty_points = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
           
            
        #     return {
        #         'points_3d': empty_points,
        #         'string_xy': string_xy,  # Keep original string_xy
        #         'z_values': z_values,
        #         'string_weights': string_weights,  # Keep original weights
        #         'string_indices': string_indices,
        #         'active_string_indices': []
        #     }
        
        # Create new points tensor for only the active strings
        total_points = len(string_indices) * self.points_per_string
        new_points_3d = torch.zeros(total_points, 3, device=self.device)
       
        # new_z_values = torch.zeros(total_active_points, device=self.device)
        # new_string_indices = []
        
        # Fill the new points tensor with data from active strings only
        for new_idx, original_string_idx in enumerate(string_indices):
            # Calculate indices for the original z_values
            original_start_idx = new_idx * self.points_per_string
            original_end_idx = original_start_idx + self.points_per_string
            
            # Set XY coordinates from the string position
            new_points_3d[original_start_idx:original_end_idx, 0] = string_xy[new_idx, 0]  # x
            new_points_3d[original_start_idx:original_end_idx, 1] = string_xy[new_idx, 1]  # y
            
            # Set Z coordinates from the original z_values
            new_points_3d[original_start_idx:original_end_idx, 2] = z_values[original_start_idx:original_end_idx]
            # new_z_values[new_start_idx:new_end_idx] = z_values[original_start_idx:original_end_idx]
            
            # Update string indices to point to the new string index
            # new_string_indices.extend([new_idx] * self.points_per_string)
        
       
            
        
        return {
            'points_3d': new_points_3d,
            'string_xy': string_xy,  # Keep original string_xy (never changes)
            'z_values': z_values,  # Only z_values for active strings
            'string_weights': string_weights_to_return,
            'old_string_weights': old_string_weights_to_return,
            'string_indices': string_indices,  # Updated indices for active strings only
            'points_per_string_list': [self.points_per_string] * len(string_indices),  # Each active string has points_per_string points  
            'weight_threshold': threshold,
            'active_weights_mode': active_weights_mode,
            # 'hybrid_mix': hybrid_mix
        }
        
