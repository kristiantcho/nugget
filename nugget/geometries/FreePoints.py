from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np
import torch.nn.functional as F

class FreePoints(Geometry):
    """Free points geometry"""
    
    def __init__(self, device=None, dim=3, domain_size=2):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
    
    def initialize_points(self, n_points=50, initial_geometry=None, **kwargs):
        """
        Initialize free points in the geometry.
        
        Parameters:
        -----------
        n_points : int
            Number of points to initialize
        initial_geometry : dict or None
            Optional dictionary containing pre-trained geometry parameters to use as a starting point.
            Should contain a 'points_3d' key with the point coordinates.
            
        Returns:
        --------
        dict
            Dictionary with initialized tensors and metadata
        
        """
        if initial_geometry is not None:
            print(f"Using pre-trained free points geometry as starting point")
            
            # Handle string weight filtering if coming from string-based geometry
            filtered_points = None
            weight_threshold = initial_geometry.get('weight_threshold', 0.7)
            
            if 'string_weights' in initial_geometry and 'points_3d' in initial_geometry and 'string_indices' in initial_geometry:
                # Filter points based on string weights
                string_weights = initial_geometry['string_weights']
                points = initial_geometry['points_3d']
                string_indices = initial_geometry['string_indices']
                
                # Convert to tensors if needed
                if not isinstance(string_weights, torch.Tensor):
                    string_weights = torch.tensor(string_weights, device=self.device, dtype=torch.float32)
                elif string_weights.device != self.device:
                    string_weights = string_weights.to(self.device)
                
                if not isinstance(points, torch.Tensor):
                    points = torch.tensor(points, device=self.device, dtype=torch.float32)
                elif points.device != self.device:
                    points = points.to(self.device)
                
                if not isinstance(string_indices, torch.Tensor):
                    string_indices = torch.tensor(string_indices, device=self.device, dtype=torch.long)
                elif string_indices.device != self.device:
                    string_indices = string_indices.to(self.device)
                
                # Apply weight filtering
                string_probs = torch.sigmoid(string_weights)  # Keep original probabilities
                active_strings_mask = string_probs > weight_threshold
                active_string_indices = torch.where(active_strings_mask)[0]
                
                # Filter points to only include those from active strings
                active_points_mask = torch.isin(string_indices, active_string_indices)
                filtered_points = points[active_points_mask]
                print(f"Filtered points from string-based geometry: {len(filtered_points)} out of {len(points)} points active")
                
            # Extract points from the initial geometry
            if filtered_points is not None:
                points = filtered_points
            elif 'points_3d' in initial_geometry:
                points = initial_geometry['points_3d']
                if not isinstance(points, torch.Tensor):
                    points = torch.tensor(points, device=self.device, dtype=torch.float32)
                elif points.device != self.device:
                    points = points.to(self.device)
            else:
                # Fall back to random initialization
                points = torch.rand((n_points, self.dim), device=self.device) * self.domain_size - self.half_domain
        else:
            # Generate random points
            points = torch.rand((n_points, self.dim), device=self.device) * self.domain_size - self.half_domain
            
        return {
            'points_3d': points,
        }
    
    def update_points(self, points, **kwargs):
        """
        Update the points based on current state in optimization.
        
        Returns:
        --------
        torch.Tensor
            Updated points (n_points, dim)
        """
        # Placeholder for actual update logic
        return {
            'points_3d': points,
        }