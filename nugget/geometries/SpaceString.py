from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np
import torch.nn.functional as F


class SpaceString(Geometry):
    """Hexagonal string geometry optimizer."""
    
    def __init__(self, device=None, dim=3, domain_size=2, hybrid_mix_init=0.5, make_hybrid_iter=True, hybrid_iter_step = 0.01,
                n_strings=1000, points_per_string=5, starting_spacing=0.1, hex_type='hexagonal', starting_z_spacing=None, optimize_z=False):
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
        elif hex_type == 'hybrid':
            self.hex_func = self.create_hybrid_hex_sunflower_grid
        else:
            self.hex_func = self.create_uniform_hexagonal_grid
        # Create hexagonal grid for strings
        original_dim = self.dim
        self.dim = 2
        
        self.dim = original_dim
        self.starting_z_spacing = starting_z_spacing
        self.optimize_z_spacing = optimize_z
        self.hybrid_mix_init = hybrid_mix_init
        self.make_hybrid_iter = make_hybrid_iter
        self.hybrid_iter_step = hybrid_iter_step
        # Half domain size for z-value mapping
        self.half_domain = domain_size / 2.0
        self.hex_grid = self.hex_func(n_points=self.n_strings, optimal_spacing=self.starting_spacing, hybrid_mix=hybrid_mix_init, 
                                      iterative_hungarian=self.make_hybrid_iter, iter_step=self.hybrid_iter_step)
    
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
                    points_per_string_list = [self.points_per_string] * len(string_xy)
            else:
                points_per_string_list = [self.points_per_string] * len(string_xy)
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
                if self.starting_z_spacing is None:
                    z_values = torch.linspace(-self.half_domain, self.half_domain, self.points_per_string, device=self.device)
                else:
                    z_values = torch.arange(0, self.points_per_string, device=self.device, dtype=torch.float32) * self.starting_z_spacing - (self.points_per_string - 1) * self.starting_z_spacing / 2.0
                z_values = z_values.repeat(self.n_strings)
                result['z_values'] = torch.tensor(z_values, device=self.device)
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
       
            if 'z_spacing' in initial_geometry:
                z_spacing = initial_geometry['z_spacing']
                if not isinstance(z_spacing, torch.Tensor):
                    z_spacing = torch.tensor(z_spacing, device=self.device, dtype=torch.float32)
                elif z_spacing.device != self.device:
                    z_spacing = z_spacing.to(self.device)
                result['z_spacing'] = z_spacing
            else:
                z_count = 0
                z_dists = []
                for n_points in points_per_string_list:
                    if n_points >= 1:
                        z_vals = result['z_values'][z_count:z_count+n_points]
                        z_count += n_points
                        # pairwise differences between each z_val in z_values
                        z_dists.append(torch.abs(z_vals[:, None] - z_vals[None, :]))
                if len(z_dists) > 0:
                    z_dists = torch.cat(z_dists, dim=0)
                    z_spacing = torch.min(z_dists[z_dists > 0])
                    z_spacing = torch.tensor(z_spacing, device=self.device)
                else:
                    z_spacing = torch.tensor(0, device=self.device)   

        else:
            # Create string_xy (hex grid or random based on self.optimize_xy)
            string_xy = self.hex_grid.clone()
            string_spacing = self.starting_spacing
            # Initialize z_values uniformly along each string
            if self.starting_z_spacing is None:
                z_values = torch.linspace(-self.half_domain, self.half_domain, self.points_per_string, device=self.device)
                z_spacing = torch.tensor(torch.abs(z_values[1]-z_values[0]), device=self.device)
            else:
                z_spacing = self.starting_z_spacing
                z_values = torch.arange(0, self.points_per_string, device=self.device, dtype=torch.float32) * self.starting_z_spacing - (self.points_per_string - 1) * self.starting_z_spacing / 2.0
            z_values = z_values.repeat(self.n_strings)
            points_per_string_list = [self.points_per_string] * self.n_strings
            
            
        
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
            'points_per_string_list': points_per_string_list,  # Each string has points_per_string points
            'z_spacing': z_spacing,
            'hybrid_mix': self.hybrid_mix_init,
            }

    def update_points(self, string_xy, z_values, string_indices, string_spacing, z_spacing, points_per_string_list, **kwargs):
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
    
        hybrid_mix = kwargs.get('hybrid_mix', self.hybrid_mix_init)
        total_points = torch.sum(torch.tensor(points_per_string_list, device=self.device)).item()
        new_points_3d = torch.zeros(total_points, 3, device=self.device)
        original_dim = self.dim
        self.dim = 2  # Temporarily set dim to 2 for hex grid generation
        string_xy = self.hex_func(n_points=len(string_indices), optimal_spacing=string_spacing, hybrid_mix=hybrid_mix, 
                                 iterative_hungarian=self.make_hybrid_iter, iter_step=self.hybrid_iter_step)
        self.dim = original_dim
        


        # new_z_values = torch.zeros(total_active_points, device=self.device)
        # new_string_indices = []
        
        # Fill the new points tensor with data from active strings only
        count = 0
        for new_idx, n_points in enumerate(points_per_string_list):
            # Calculate indices for the original z_values
            original_start_idx = count
            original_end_idx = count + n_points

            # Set XY coordinates from the string position
            new_points_3d[original_start_idx:original_end_idx, 0] = string_xy[new_idx, 0]  # x
            new_points_3d[original_start_idx:original_end_idx, 1] = string_xy[new_idx, 1]  # y
            
            # Set Z coordinates from the original z_values
            if not self.optimize_z_spacing:
                new_points_3d[original_start_idx:original_end_idx, 2] = z_values[original_start_idx:original_end_idx]
            else:
                # Recompute z_values based on optimized z_spacing
                mid_point = (n_points - 1) / 2.0
                new_z = (torch.arange(n_points, device=self.device, dtype=torch.float32) - mid_point) * z_spacing
                new_points_3d[original_start_idx:original_end_idx, 2] = new_z
            count += n_points
            # new_z_values[new_start_idx:new_end_idx] = z_values[original_start_idx:original_end_idx]

        if not self.optimize_z_spacing:
            z_count = 0
            z_dists = []
            for n_points in points_per_string_list:
                if n_points >= 1:
                    z_vals = z_values[z_count:z_count+n_points]
                    z_count += n_points
                    # pairwise differences between each z_val in z_values
                    z_dists.append(torch.abs(z_vals[:, None] - z_vals[None, :]))
            if len(z_dists) > 0:
                z_dists = torch.cat(z_dists, dim=0)
                z_spacing = torch.min(z_dists[z_dists > 0])
            else:
                z_spacing = torch.tensor(0, device=self.device) 
    
        return {
            'points_3d': new_points_3d,
            'string_xy': string_xy,  # Keep original string_xy (never changes)
            'z_values': z_values,  # Only z_values for active strings
            'string_indices': string_indices,  # Updated indices for active strings only
            'points_per_string_list': points_per_string_list,  # Each active string has points_per_string points
            'string_spacing': string_spacing,  # Keep original spacing (never changes)
            'z_spacing': z_spacing,  # Updated z_spacing if optimizing
        }
        
