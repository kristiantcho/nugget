import torch
import numpy as np

class Geometry:
    """Base class for geometry optimization strategies."""
    
    def __init__(self, device=None, dim=3, domain_size=2):
        """
        Initialize the geometry
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim
        self.domain_size = domain_size
        self.half_domain = domain_size / 2
    
    def initialize_points(self, **kwargs):
        """
        Initialize points according to the geometry strategy.
        
        Returns:
        --------
        dict
            Dictionary with initialized tensors and metadata
        """
        raise NotImplementedError("Subclasses must implement initialize_points")
    
    def update_points(self, **kwargs):
        """
        Update points based on current optimization state.
        
        Returns:
        --------
        dict
            Dictionary with updated tensors and metadata
        """
        raise NotImplementedError("Subclasses must implement update_points")
    
    def create_uniform_hexagonal_grid(self, n_points=50):
        """
        Create a uniform hexagonal lattice of points starting from center and building outward.
        Starts with one point in the center, then adds points in concentric hexagons clockwise.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
            
        Returns:
        --------
        torch.Tensor
            Grid points (n_points, 2)
        """
        if self.dim != 2:
            raise ValueError("Hexagonal grid is only available for 2D")
        
        # Calculate optimal spacing to fit the desired number of points within domain
        # For a hexagonal grid, the number of points in rings 0 to r is: 1 + 3*r*(r+1)
        # We need to find the minimum spacing such that all points fit within the domain
        
        def calculate_max_rings(spacing):
            """Calculate how many complete rings can fit within the domain"""
            max_radius = self.half_domain / spacing
            return int(max_radius)
        
        def points_in_rings(num_rings):
            """Calculate total points in rings 0 to num_rings-1"""
            if num_rings <= 0:
                return 0
            return 1 + 3 * (num_rings - 1) * num_rings
        
        # Binary search to find optimal spacing
        min_spacing = 0.0
        max_spacing = self.domain_size - self.domain_size/10
        
        # Find the largest spacing that allows at least n_points
        optimal_spacing = max_spacing
        for _ in range(100):  # Binary search iterations
            mid_spacing = (min_spacing + max_spacing) / 2
            max_rings = calculate_max_rings(mid_spacing)
            total_possible_points = points_in_rings(max_rings + 1)  # +1 because we can have partial rings
            
            if total_possible_points >= n_points:
                optimal_spacing = mid_spacing
                min_spacing = mid_spacing
            else:
                max_spacing = mid_spacing
                
            if max_spacing - min_spacing < 1e-6:
                break
        
        # Generate points starting from center
        points = []
        spacing = optimal_spacing
        
        # Add center point
        points.append([0.0, 0.0])
        
        if n_points == 1:
            hex_points = torch.tensor(points[:n_points], dtype=torch.float32, device=self.device)
            return hex_points
        
        # Generate concentric hexagonal rings
        ring = 1
        while len(points) < n_points:
            # Check if this ring would fit within domain
            # For hexagonal lattice, the distance from center to corner is ring * spacing
            max_distance = ring * spacing - self.half_domain/10
            if max_distance > self.half_domain:
                break
                
            # Generate hexagonal ring points
            ring_points = []
            
            # For each side of the hexagon
            for side in range(6):
                # Number of points on this side (including corners, but avoid duplicates)
                points_on_side = ring
                
                for i in range(points_on_side):
                    # Calculate position along this side of the hexagon
                    # Each side goes from one corner to the next
                    
                    # Hexagon vertices (corners) in clockwise order starting from rightmost
                    corners = [
                        (ring * spacing, 0),  # Right
                        (ring * spacing * 0.5, ring * spacing * np.sqrt(3) / 2),  # Top-right
                        (-ring * spacing * 0.5, ring * spacing * np.sqrt(3) / 2),  # Top-left
                        (-ring * spacing, 0),  # Left
                        (-ring * spacing * 0.5, -ring * spacing * np.sqrt(3) / 2),  # Bottom-left
                        (ring * spacing * 0.5, -ring * spacing * np.sqrt(3) / 2),  # Bottom-right
                    ]
                    
                    # Get start and end corners for this side
                    start_corner = corners[side]
                    end_corner = corners[(side + 1) % 6]
                    
                    # Interpolate along the side (skip the end point to avoid duplicates)
                    if i < points_on_side:
                        t = i / points_on_side
                        x = start_corner[0] + t * (end_corner[0] - start_corner[0])
                        y = start_corner[1] + t * (end_corner[1] - start_corner[1])
                        
                        # Check if point is within domain bounds
                        if abs(x) <= self.half_domain and abs(y) <= self.half_domain:
                            ring_points.append([x, y])
            
            # Add ring points
            points.extend(ring_points)
            ring += 1
        
        # If we have more points than needed, take the first n_points
        if len(points) > n_points:
            final_points = points[:n_points]
        else:
            final_points = points
            # If we don't have enough points, fill with the last point
            while len(final_points) < n_points:
                final_points.append(final_points[-1] if final_points else [0.0, 0.0])
        
        # Convert to torch tensor
        hex_points = torch.tensor(final_points[:n_points], dtype=torch.float32, device=self.device)
        
        return hex_points