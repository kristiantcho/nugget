from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np
import torch.nn.functional as F

class ContinuousString(Geometry):
    """Continuous string geometry optimizer."""
    
    def __init__(self, device=None, dim=3, domain_size=2, optimize_xy=False, 
                total_points=150, n_strings=30, optimize_positions_only=False):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.total_points = total_points
        self.n_strings = n_strings
        self.optimize_xy = optimize_xy
        self.optimize_positions_only = optimize_positions_only
        
        # Create hexagonal grid for strings
        original_dim = self.dim
        self.dim = 2
        self.hex_grid = self.create_uniform_hexagonal_grid(n_points=self.n_strings)
        self.dim = original_dim
            
    
    def initialize_points(self, initial_geometry=None, **kwargs):
        """
        Initialize points in a continuous string configuration.
        
        Parameters:
        -----------
        initial_geometry : dict or None
            Optional dictionary containing pre-trained geometry parameters to use as a starting point.
            Should contain keys like 'path_positions', 'string_xy', etc.
        
        Returns:
        --------
        dict
            Dictionary with initialized torch tensors
        """
        if initial_geometry is not None:
            print(f"Using pre-trained continuous string geometry as starting point")
            # Extract components from the initial geometry
            
            # Process path_positions if available
            if 'path_positions' in initial_geometry:
                path_positions = initial_geometry['path_positions']
                if not isinstance(path_positions, torch.Tensor):
                    path_positions = torch.tensor(path_positions, device=self.device, dtype=torch.float32)
                elif path_positions.device != self.device:
                    path_positions = path_positions.to(self.device)
            else:
                # Fall back to default initialization
                path_positions = torch.linspace(0, 1, self.total_points, device=self.device)
            
            # Process string_xy if available
            if 'string_xy' in initial_geometry:
                string_xy = initial_geometry['string_xy']
                if not isinstance(string_xy, torch.Tensor):
                    string_xy = torch.tensor(string_xy, device=self.device, dtype=torch.float32)
                elif string_xy.device != self.device:
                    string_xy = string_xy.to(self.device)
            else:
                # Fall back to default initialization
                if self.optimize_xy:
                    string_xy = torch.rand(self.n_strings, 2, device=self.device) * self.domain_size - self.half_domain
                else:
                    string_xy = self.hex_grid.clone()
            
            # Return updated points using the initial geometry
            return self.update_points(path_positions=path_positions, string_xy=string_xy)
            
        # Create parameters for 1D continuous path from 0 to 1
        path_positions = torch.linspace(0, 1, self.total_points, device=self.device)
        
        # Initialize string positions (XY coordinates)
        if self.optimize_xy:
            string_xy = torch.rand(self.n_strings, 2, device=self.device) * self.domain_size - self.half_domain
        else:
            string_xy = self.hex_grid.clone()
        
        # Map path positions to 3D points initially
        return self.update_points(path_positions=path_positions, string_xy=string_xy)
    
    def update_points(self, path_positions, string_xy, **kwargs):
        """
        Update the points based on current optimization state.
        
        Parameters:
        -----------
        path_positions : torch.Tensor
            Position of each point along the continuous path from 0 to 1 (total_points,)
        string_xy : torch.Tensor
            XY coordinates for each string (n_strings, 2)
            
        Returns:
        --------
        dict
            Dictionary with updated tensors and metadata
        """
        # Map 1D path positions to actual 3D points and string assignments
        points_3d, string_indices, points_per_string_list = self.map_path_to_3d(path_positions, string_xy)
        
        return {
            "points": points_3d,
            "path_positions": path_positions, 
            "string_xy": string_xy, 
            "string_indices": string_indices, 
            "points_per_string_list": points_per_string_list
        }
    
    def map_path_to_3d(self, path_positions, string_xy):
        """
        Map 1D path positions to 3D coordinates.
        
        Parameters:
        -----------
        path_positions : torch.Tensor
            Position of each point along the continuous path from 0 to 1 (total_points,)
        string_xy : torch.Tensor
            XY coordinates for each string (n_strings, 2)
            
        Returns:
        --------
        tuple
            (points_3d, string_indices, points_per_string_list)
        """
        # First, determine which string each point belongs to
        # We divide the path into n_strings segments
        segment_width = 1.0 / self.n_strings
        string_indices_float = path_positions / segment_width
        string_indices = torch.floor(string_indices_float).clamp(0, self.n_strings-1).long()
        
        # Calculate how far along each string the point is (0 = bottom, 1 = top)
        relative_position = (string_indices_float - string_indices)
        
        # Map to z coordinate (-half_domain to half_domain)
        z_values = (2 * relative_position - 1) * self.half_domain
        
        # Create 3D points
        points_3d = torch.zeros(self.total_points, 3, device=self.device)
        
        # Set xy coordinates based on string
        points_3d[:, 0] = string_xy[string_indices, 0]  # x coordinate
        points_3d[:, 1] = string_xy[string_indices, 1]  # y coordinate
        points_3d[:, 2] = z_values  # z coordinate
        
        # Count points per string
        string_indices_np = string_indices.detach().cpu().numpy()
        points_per_string_list = []
        for s in range(self.n_strings):
            count = np.sum(string_indices_np == s)
            points_per_string_list.append(int(count))
        
        return points_3d, string_indices.tolist(), points_per_string_list