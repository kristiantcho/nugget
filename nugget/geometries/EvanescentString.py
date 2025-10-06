from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np
import torch.nn.functional as F


class EvanescentString(Geometry):
    """Evanescent string geometry optimizer."""
    
    def __init__(self, device=None, dim=3, domain_size=2,
                n_strings=1000, points_per_string=5, starting_weight=1.0):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.n_strings = n_strings
        self.points_per_string = points_per_string
        self.starting_weight = starting_weight
        
        # Create hexagonal grid for strings
        original_dim = self.dim
        self.dim = 2
        self.hex_grid = self.create_uniform_hexagonal_grid(n_points=self.n_strings)
        self.dim = original_dim
        
        
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
            if 'string_weights' in initial_geometry:
                string_weights = initial_geometry['string_weights']
                if not isinstance(string_weights, torch.Tensor):
                    string_weights = torch.tensor(string_weights, device=self.device, dtype=torch.float32)
                elif string_weights.device != self.device:
                    string_weights = string_weights.to(self.device)
                result['string_weights'] = string_weights
            else:
                # Default to uniform weights if not provided
                result['string_weights'] = torch.ones(self.n_strings, device=self.device, dtype=torch.float32)*self.starting_weight
        else:
            # Create string_xy (hex grid or random based on self.optimize_xy)
            string_xy = self.hex_grid.clone()
            
            # Initialize z_values uniformly along each string
            z_values = torch.linspace(-self.half_domain, self.half_domain, self.points_per_string, device=self.device)
            z_values = z_values.repeat(self.n_strings)
            
            string_weights = torch.ones(self.n_strings, device=self.device, dtype=torch.float32)*self.starting_weight
        
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
            'active_points': points_3d,  # Initially all points are active
            'string_xy': string_xy,
            'z_values': z_values,
            'string_weights': string_weights,
            'string_indices': string_indices,
            'active_string_indices': string_indices,  # Initially all strings are active
            'points_per_string_list': [self.points_per_string] * self.n_strings,  # Each string has points_per_string points
            }
    
    def update_points(self, string_xy, z_values, string_weights, string_indices, **kwargs):
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
        # string_probs = torch.sigmoid(string_weights)
        string_probs = string_weights  # Keep original probabilities for later use
        # Determine which strings to include based on their sigmoid weights
        # You can adjust this threshold as needed 
        threshold = kwargs.get('weight_threshold', 0.7)
        active_strings_mask = string_probs > threshold
        active_string_indices = torch.where(active_strings_mask)[0]
        
        # Count how many strings are active
        n_active_strings = string_indices[active_strings_mask]
        
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
        total_active_points = len(active_string_indices) * self.points_per_string
        active_points_3d = torch.zeros(total_active_points, 3, device=self.device)
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
        
        for new_idx, original_string_idx in enumerate(active_string_indices):
            # Calculate indices for the original z_values
            original_start_idx = original_string_idx * self.points_per_string
            original_end_idx = original_start_idx + self.points_per_string
            
            # Calculate indices for the new points tensor
            new_start_idx = new_idx * self.points_per_string
            new_end_idx = new_start_idx + self.points_per_string
            
            # Set XY coordinates from the string position
            active_points_3d[new_start_idx:new_end_idx, 0] = string_xy[original_string_idx, 0]
            active_points_3d[new_start_idx:new_end_idx, 1] = string_xy[original_string_idx, 1]
            
            # Set Z coordinates from the original z_values
            active_points_3d[new_start_idx:new_end_idx, 2] = z_values[original_start_idx:original_end_idx]
            
        
        return {
            'points_3d': new_points_3d,
            'string_xy': string_xy,  # Keep original string_xy (never changes)
            'z_values': z_values,  # Only z_values for active strings
            'string_weights': string_weights,  # Keep original weights
            'string_indices': string_indices,  # Updated indices for active strings only
            'active_string_indices': active_string_indices,  # Which strings are active
            'active_points': active_points_3d, # Points for active strings only
            'points_per_string_list': [self.points_per_string] * len(string_indices),  # Each active string has points_per_string points  
            'weight_threshold': threshold
        }
        
