from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np
import torch.nn.functional as F


class SpaceString(Geometry):
    """Hexagonal string geometry optimizer."""
    
    def __init__(self, device=None, dim=3, domain_size=2,
                n_strings=1000, points_per_string=5, starting_spacing=0.1, hex_type='hexagonal'):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.n_strings = n_strings
        self.points_per_string = points_per_string
        self.starting_spacing = torch.tensor(starting_spacing, device=self.device)
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
        self.dim = 2
        self.hex_grid = self.hex_func(n_points=self.n_strings, optimal_spacing=self.starting_spacing)
        self.dim = original_dim
        
        
        # Half domain size for z-value mapping
        self.half_domain = domain_size / 2.0
    
    def initialize_points(self, initial_geometry=None, **kwargs):
        """
        Initialize points in a hexagonal string configuration.

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
        
        if initial_geometry is not None:
            print(f"Using pre-trained evanescent string geometry as starting point")
            result = {}
            
            # Process string_xy if available
            
            if 'string_xy' in initial_geometry:
                string_xy = initial_geometry['string_xy']
                if not isinstance(string_xy, torch.Tensor):
                    string_xy = torch.tensor(string_xy, device=self.device, dtype=torch.float32)
                elif string_xy.device != self.device:
                    string_xy = string_xy.to(self.device)
                result['string_xy'] = string_xy
            else:
                string_xy = self.hex_grid.clone()
                result['string_xy'] = string_xy
            z_values = None
            not_matched = False
            if 'points_per_string_list' in initial_geometry:
                points_per_string_list = initial_geometry['points_per_string_list']
                if points_per_string_list != [self.points_per_string]*len(string_xy):
                    print(f"Warning: points_per_string_list {points_per_string_list} does not match expected {[self.points_per_string]*len(string_xy)}.")
                    not_matched = True
            if 'z_values' in initial_geometry:
                z_values = initial_geometry['z_values']
                # if z_values != len(result['string_xy'])*self.points_per_string:
                #     print(f"Warning: z_values length {len(z_values)} does not match expected {len(result['string_xy']) * self.points_per_string}.")
                #     not_matched = True
                if not isinstance(z_values, torch.Tensor):
                    z_values = torch.tensor(z_values, device=self.device, dtype=torch.float32)
                elif z_values.device != self.device:
                    z_values = z_values.to(self.device)
                result['z_values'] = z_values
            if z_values is None or not_matched:
                z_values = torch.linspace(-self.half_domain, self.half_domain, self.points_per_string, device=self.device)
                for s_idx in range(self.n_strings):
                    if s_idx == 0:
                        result['z_values'] = z_values
                    else:
                        result['z_values'] = torch.cat((result['z_values'], z_values), dim=0)
            if 'string_spacing' in initial_geometry:
                string_spacing = initial_geometry['string_spacing']
                if not isinstance(string_spacing, torch.Tensor):
                    string_spacing = torch.tensor(string_spacing, device=self.device, dtype=torch.float32)
                elif string_spacing.device != self.device:
                    string_spacing = string_spacing.to(self.device)
                result['string_spacing'] = string_spacing
            else:
                dists = torch.norm(string_xy[:, None, :] - string_xy[None, :, :], axis=-1)  # Pairwise distances
                string_spacing = torch.min(dists[dists > 0])
                result['string_spacing'] = string_spacing
           
        else:
            # Create string_xy (hex grid or random based on self.optimize_xy)
            string_xy = self.hex_grid.clone()
            string_spacing = self.starting_spacing
            # Initialize z_values uniformly along each string
            z_values = torch.linspace(-self.half_domain, self.half_domain, self.points_per_string, device=self.device)
            z_values = z_values.repeat(self.n_strings)
            
            
        
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
            'points_3d': points_3d,
            'string_spacing': string_spacing,
            'string_xy': string_xy,
            'z_values': z_values,
            'string_indices': string_indices,
            'points_per_string_list': [self.points_per_string] * self.n_strings,  # Each string has points_per_string points
            }

    def update_points(self, string_xy, z_values, string_indices, string_spacing, points_per_string_list, **kwargs):
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
        # Apply sigmoid to string weights to get probabilities between 0 and 1
    
        
        total_points = len(string_indices) * self.points_per_string
        new_points_3d = torch.zeros(total_points, 3, device=self.device)
        original_dim = self.dim
        self.dim = 2  # Temporarily set dim to 2 for hex grid generation
        string_xy = self.hex_func(n_points=len(string_indices), optimal_spacing=string_spacing)
        self.dim = original_dim


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
      
    
        return {
            'points_3d': new_points_3d,
            'string_xy': string_xy,  # Keep original string_xy (never changes)
            'z_values': z_values,  # Only z_values for active strings
            'string_indices': string_indices,  # Updated indices for active strings only
            'points_per_string_list': points_per_string_list,  # Each active string has points_per_string points
            'string_spacing': string_spacing,  # Keep original spacing (never changes)
        }
        
