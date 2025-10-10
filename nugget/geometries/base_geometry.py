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
        self.device = device if device is not None else torch.device('cpu')
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
    
    def create_uniform_hexagonal_grid(self, n_points=50, optimal_spacing=None):
        """
        Create a uniform hexagonal lattice of points starting from center and building outward.
        Starts with one point in the center, then adds points in concentric hexagons clockwise.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
        optimal_spacing : torch.Tensor or None
            Spacing parameter as a torch tensor (can be optimized)
            
        Returns:
        --------
        torch.Tensor
            Grid points (n_points, 2)
        """

        
        # Calculate optimal spacing to fit the desired number of points within domain
        # For a hexagonal grid, the number of points in rings 0 to r is: 1 + 3*r*(r+1)
        # We need to find the minimum spacing such that all points fit within the domain
        
        def calculate_max_rings(spacing):
            """Calculate how many complete rings can fit within the domain"""
            max_radius = self.half_domain / spacing
            return torch.floor(max_radius).int()
        
        def points_in_rings(num_rings):
            """Calculate total points in rings 0 to num_rings-1"""
            num_rings = torch.clamp(num_rings, min=0)
            return torch.where(num_rings <= 0, 
                             torch.tensor(0, device=self.device, dtype=torch.int32),
                             1 + 3 * (num_rings - 1) * num_rings)
        
        # Binary search to find optimal spacing using torch operations
        if optimal_spacing is None:
            min_spacing = torch.tensor(1e-6, device=self.device, dtype=torch.float32)
            max_spacing = torch.tensor(self.domain_size - self.domain_size/10, 
                                     device=self.device, dtype=torch.float32)
            
            # Find the largest spacing that allows at least n_points
            optimal_spacing = max_spacing.clone()
            for _ in range(100):  # Binary search iterations
                mid_spacing = (min_spacing + max_spacing) / 2
                max_rings = calculate_max_rings(mid_spacing)
                total_possible_points = points_in_rings(max_rings + 1)  # +1 because we can have partial rings
                
                condition = total_possible_points >= n_points
                optimal_spacing = torch.where(condition, mid_spacing, optimal_spacing)
                min_spacing = torch.where(condition, mid_spacing, min_spacing)
                max_spacing = torch.where(~condition, mid_spacing, max_spacing)
                    
                if torch.abs(max_spacing - min_spacing) < 1e-6:
                    break
        else:
            # Ensure optimal_spacing is a torch tensor
            if not isinstance(optimal_spacing, torch.Tensor):
                optimal_spacing = torch.tensor(optimal_spacing, device=self.device, dtype=torch.float32)
            optimal_spacing = optimal_spacing.to(device=self.device, dtype=torch.float32)
        
        
        # Generate points starting from center using torch operations
        spacing = optimal_spacing
        
        # Start with center point
        all_points = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
        center_point = torch.zeros(1, 2, device=self.device, dtype=torch.float32)
        all_points = torch.cat([all_points, center_point], dim=0)
        
        if n_points == 1:
            return all_points[:n_points]
        
        # Pre-calculate maximum number of rings we might need
        max_possible_rings = int(torch.ceil(self.half_domain / spacing).item()) + 1
        
        # Generate all rings at once using vectorized operations
        for ring in range(1, max_possible_rings + 1):
            # Check if this ring would fit within domain
            max_distance = ring * spacing - self.half_domain/10
            if max_distance > self.half_domain:
                break
                
            ring_tensor = torch.tensor(ring, device=self.device, dtype=torch.float32)
            
            # Define hexagon corners using torch operations
            sqrt3_half = torch.sqrt(torch.tensor(3.0, device=self.device)) / 2
            corners = torch.stack([
                torch.stack([ring_tensor * spacing, torch.zeros_like(spacing)]),  # Right
                torch.stack([ring_tensor * spacing * 0.5, ring_tensor * spacing * sqrt3_half]),  # Top-right
                torch.stack([-ring_tensor * spacing * 0.5, ring_tensor * spacing * sqrt3_half]),  # Top-left
                torch.stack([-ring_tensor * spacing, torch.zeros_like(spacing)]),  # Left
                torch.stack([-ring_tensor * spacing * 0.5, -ring_tensor * spacing * sqrt3_half]),  # Bottom-left
                torch.stack([ring_tensor * spacing * 0.5, -ring_tensor * spacing * sqrt3_half]),  # Bottom-right
            ])  # Shape: (6, 2)
            
            ring_points = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
            
            # Generate points for each side of the hexagon
            for side in range(6):
                start_corner = corners[side]  # Shape: (2,)
                end_corner = corners[(side + 1) % 6]  # Shape: (2,)
                
                # Create interpolation parameters for this side
                t_values = torch.linspace(0, 1, ring + 1, device=self.device)[:-1]  # Exclude end to avoid duplicates
                
                # Vectorized interpolation
                side_points = start_corner.unsqueeze(0) + t_values.unsqueeze(1) * (end_corner - start_corner).unsqueeze(0)
                
                # Check if points are within domain bounds using torch operations
                x_coords = side_points[:, 0]
                y_coords = side_points[:, 1]
                valid_mask = (torch.abs(x_coords) <= self.half_domain) & (torch.abs(y_coords) <= self.half_domain)
                
                # Add valid points
                valid_points = side_points[valid_mask]
                if valid_points.shape[0] > 0:
                    ring_points = torch.cat([ring_points, valid_points], dim=0)
            
            # Add ring points to all points
            if ring_points.shape[0] > 0:
                all_points = torch.cat([all_points, ring_points], dim=0)
            
            # Early stopping if we have enough points
            if all_points.shape[0] >= n_points:
                break
        
        # Handle cases where we have more or fewer points than needed
        if all_points.shape[0] >= n_points:
            hex_points = all_points[:n_points]
        else:
            # Pad with the last point or center if no points exist
            needed_points = n_points - all_points.shape[0]
            if all_points.shape[0] > 0:
                last_point = all_points[-1:].repeat(needed_points, 1)
            else:
                last_point = torch.zeros(needed_points, 2, device=self.device, dtype=torch.float32)
            hex_points = torch.cat([all_points, last_point], dim=0)
        
        return hex_points
    
    def create_circular_hexagonal_grid(self, n_points=50, optimal_spacing=None):
        """
        Create a hexagonal grid arranged in concentric circles.
        Points are arranged in a hexagonal pattern but confined within circular boundaries,
        creating a more circular overall distribution than the standard hexagonal grid.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
        optimal_spacing : torch.Tensor or None
            Spacing parameter as a torch tensor (can be optimized)
            
        Returns:
        --------
        torch.Tensor
            Grid points (n_points, 2)
        """
     
        
        # Calculate optimal spacing to fit the desired number of points within circular domain
        if optimal_spacing is None:
            # Estimate spacing based on circular area and desired points
            max_radius = self.half_domain * 0.95  # Leave small margin
            area_per_point = (np.pi * max_radius**2) / n_points
            # For hexagonal packing, each point occupies sqrt(3)/2 * spacing^2 area
            hexagonal_area_factor = np.sqrt(3) / 2
            optimal_spacing = torch.sqrt(torch.tensor(area_per_point / hexagonal_area_factor, 
                                                    device=self.device, dtype=torch.float32))
        else:
            # Ensure optimal_spacing is a torch tensor
            if not isinstance(optimal_spacing, torch.Tensor):
                optimal_spacing = torch.tensor(optimal_spacing, device=self.device, dtype=torch.float32)
            optimal_spacing = optimal_spacing.to(device=self.device, dtype=torch.float32)
        
        spacing = optimal_spacing
        max_radius = self.half_domain * 0.95
        
        # Start with center point
        all_points = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
        center_point = torch.zeros(1, 2, device=self.device, dtype=torch.float32)
        all_points = torch.cat([all_points, center_point], dim=0)
        
        if n_points == 1:
            return all_points[:n_points]
        
        # Calculate maximum number of rings we might need
        max_possible_rings = int(torch.ceil(max_radius / spacing).item()) + 1
        
        sqrt3_half = torch.sqrt(torch.tensor(3.0, device=self.device)) / 2
        
        # Generate concentric circular rings with hexagonal packing
        for ring in range(1, max_possible_rings + 1):
            ring_radius = ring * spacing
            
            # Skip if ring is outside circular domain
            if ring_radius > max_radius:
                break
            
            ring_tensor = torch.tensor(ring, device=self.device, dtype=torch.float32)
            
            # Calculate number of points in this ring based on circumference
            # For hexagonal packing, points are spaced by 'spacing' along the circumference
            circumference = 2 * np.pi * ring_radius
            n_points_in_ring = max(6, int(circumference / spacing))  # Minimum 6 for hexagonal structure
            
            # Generate points evenly around the circle
            angles = torch.linspace(0, 2 * np.pi, n_points_in_ring + 1, device=self.device)[:-1]  # Exclude last to avoid duplicate
            
            # Add slight hexagonal bias to angles for better packing
            # Adjust angles to align better with hexagonal structure
            angle_offset = (ring % 2) * (np.pi / n_points_in_ring)  # Alternate rings for better packing
            angles = angles + angle_offset
            
            # Convert to Cartesian coordinates
            x = ring_radius * torch.cos(angles)
            y = ring_radius * torch.sin(angles)
            ring_points = torch.stack([x, y], dim=1)
            
            # Filter points to ensure they're within the circular domain
            distances = torch.sqrt(ring_points[:, 0]**2 + ring_points[:, 1]**2)
            valid_mask = distances <= max_radius
            valid_points = ring_points[valid_mask]
            
            # Also ensure points are within the square domain bounds
            x_coords = valid_points[:, 0]
            y_coords = valid_points[:, 1]
            bounds_mask = (torch.abs(x_coords) <= self.half_domain) & (torch.abs(y_coords) <= self.half_domain)
            valid_points = valid_points[bounds_mask]
            
            # Add valid points
            if valid_points.shape[0] > 0:
                all_points = torch.cat([all_points, valid_points], dim=0)
            
            # Early stopping if we have enough points
            if all_points.shape[0] >= n_points:
                break
        
        # Handle cases where we have more or fewer points than needed
        if all_points.shape[0] >= n_points:
            circular_hex_points = all_points[:n_points]
        else:
            # Pad with points on the outer ring if needed
            needed_points = n_points - all_points.shape[0]
            if all_points.shape[0] > 0:
                # Generate additional points on the boundary
                boundary_angles = torch.linspace(0, 2 * np.pi, needed_points + 1, device=self.device)[:-1]
                boundary_x = max_radius * torch.cos(boundary_angles)
                boundary_y = max_radius * torch.sin(boundary_angles)
                boundary_points = torch.stack([boundary_x, boundary_y], dim=1)
                circular_hex_points = torch.cat([all_points, boundary_points], dim=0)
            else:
                # Fallback to center points
                padding_points = torch.zeros(needed_points, 2, device=self.device, dtype=torch.float32)
                circular_hex_points = torch.cat([all_points, padding_points], dim=0)
        
        return circular_hex_points
    
    def create_sunflower_grid(self, n_points=50, optimal_spacing=None):
        """
        Create a sunflower-like spiral grid using the golden angle.
        Points are distributed in a spiral pattern with increasing radius,
        similar to the arrangement of seeds in a sunflower head.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
        scaling_factor : torch.Tensor or None
            Scaling factor for the spiral (can be optimized)
            
        Returns:
        --------
        torch.Tensor
            Grid points (n_points, 2)
        """
        scaling_factor = optimal_spacing
            
        if scaling_factor is None:
            # Calculate scaling factor to fit points within domain
            max_radius = self.half_domain * 0.95  # Leave small margin
            scaling_factor = torch.tensor(max_radius / torch.sqrt(torch.tensor(n_points, dtype=torch.float32)), 
                                        device=self.device, dtype=torch.float32)
        else:
            if not isinstance(scaling_factor, torch.Tensor):
                scaling_factor = torch.tensor(scaling_factor, device=self.device, dtype=torch.float32)
            scaling_factor = scaling_factor.to(device=self.device, dtype=torch.float32)
        
        # Golden angle in radians (approximately 137.5 degrees)
        golden_angle = torch.tensor(np.pi * (3 - np.sqrt(5)), device=self.device, dtype=torch.float32)
        
        # Generate point indices
        indices = torch.arange(n_points, device=self.device, dtype=torch.float32)
        
        # Calculate angles using golden angle spiral
        angles = indices * golden_angle
        
        # Calculate radii using square root scaling for uniform density
        radii = scaling_factor * torch.sqrt(indices)
        
        # Convert to Cartesian coordinates
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        
        return torch.stack([x, y], dim=1)
    
    