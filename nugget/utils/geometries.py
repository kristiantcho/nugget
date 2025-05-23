import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

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
        Create a uniform hexagonal lattice of points (for 2D only).
        
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
            
        # Calculate spacing based on number of points
        area = self.domain_size * self.domain_size
        area_per_point = area / n_points
        
        # Spacing for a regular hexagonal grid
        horizontal_spacing = np.sqrt(2 * area_per_point / np.sqrt(3))
        vertical_spacing = horizontal_spacing * np.sqrt(3) / 2
        
        # Generate more points than needed to ensure complete coverage
        x_count = int(np.ceil(self.domain_size / horizontal_spacing)) + 3
        y_count = int(np.ceil(self.domain_size / vertical_spacing)) + 3
        
        # Generate initial grid with hexagonal pattern
        all_points = []
        
        # Calculate center offset to perfectly center the entire grid
        center_x_offset = (x_count % 2) * horizontal_spacing / 2
        center_y_offset = (y_count % 2) * vertical_spacing / 2
        
        for j in range(-y_count, y_count + 1):
            # Hexagonal pattern: offset every other row
            offset = horizontal_spacing / 2 if j % 2 else 0
            
            for i in range(-x_count, x_count + 1):
                x = i * horizontal_spacing + offset - center_x_offset
                y = j * vertical_spacing - center_y_offset
                
                # Check if within domain bounds
                if -self.half_domain <= x <= self.half_domain and -self.half_domain <= y <= self.half_domain:
                    all_points.append([x, y])
        
        # If we have too many points, select them to ensure uniform distribution
        if len(all_points) > n_points:
            # Convert to numpy array for processing
            points_array = np.array(all_points)
            
            # First compute pairwise distances between all candidate points
            distances = squareform(pdist(points_array))
            
            # Start with the point closest to the center
            center_distances = np.sum(points_array**2, axis=1)
            selected_indices = [np.argmin(center_distances)]  # Start with center point
            
            # Iteratively add the point that is farthest from all previously selected points
            while len(selected_indices) < n_points:
                # Calculate minimum distance from each remaining point to any selected point
                mask = np.ones(len(all_points), dtype=bool)
                mask[selected_indices] = False
                
                # For each candidate point, find its minimum distance to any selected point
                min_distances = np.min(distances[mask][:, selected_indices], axis=1)
                
                # Choose the point with the maximum minimum distance (farthest point)
                next_idx = np.where(mask)[0][np.argmax(min_distances)]
                selected_indices.append(next_idx)
            
            # Keep only the selected points
            final_points = [all_points[i] for i in selected_indices]
        else:
            final_points = all_points
        
        # If we don't have enough points (unlikely), duplicate the last point
        while len(final_points) < n_points:
            final_points.append(final_points[-1])
        
        # Convert to torch tensor
        hex_points = torch.tensor(final_points[:n_points], dtype=torch.float32, device=self.device)
        
        return hex_points
    
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
            Should contain a 'points' key with the point coordinates.
            
        Returns:
        --------
        dict
            Dictionary with initialized tensors and metadata
        
        """
        if initial_geometry is not None:
            print(f"Using pre-trained free points geometry as starting point")
            # Extract points from the initial geometry
            if 'points' in initial_geometry:
                points = initial_geometry['points']
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
            'points': points,
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
            'points': points,
        }

class DynamicString(Geometry):
    """Dynamic string geometry optimizer."""
    
    def __init__(self, device=None, dim=3, domain_size=2, random_initial_dist = False, 
                total_points = 150, n_strings = 30, initial_random_spread = 0.1, optimize_positions_only = False,
                min_points_per_string = 2, optimize_xy = False, even_distribution = False):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.random_initial_dist = random_initial_dist
        self.total_points = total_points
        self.n_strings = n_strings
        self.initial_random_spread = initial_random_spread
        self.min_points_per_string = min_points_per_string
        self.optimize_xy = optimize_xy
        self.even_distribution = even_distribution # False if you want to redistribute points with string logits (not working optimally yet)
        self.optimize_positions_only = optimize_positions_only
        original_dim = self.dim
        self.dim = 2
        self.hex_grid = self.create_uniform_hexagonal_grid(n_points=self.n_strings)
        self.dim = original_dim
            
    
    def initialize_points(self, initial_geometry=None, **kwargs):
        """
        Initialize points in a dynamic string configuration.
        
        Parameters:
        -----------
        initial_geometry : dict or None
            Optional dictionary containing pre-trained geometry parameters to use as a starting point.
            Should contain keys like 'string_xy', 'z_values', 'string_indices', 'points_per_string_list', etc.
        
        Returns:
        --------
        dict
            Dictionary with initialized torch tensors
        
        """
        if initial_geometry is not None:
            print(f"Using pre-trained dynamic string geometry as starting point")
            # Extract and validate components from the initial geometry
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
                # Fall back to default initialization
                string_xy = self.hex_grid.clone()
                if self.optimize_xy:
                    string_xy = torch.rand(self.n_strings, 2, device=self.device) * self.domain_size - self.half_domain
                result['string_xy'] = string_xy
            
            # Process z_values if available
            if 'z_values' in initial_geometry:
                z_values = initial_geometry['z_values']
                if not isinstance(z_values, torch.Tensor):
                    z_values = torch.tensor(z_values, device=self.device, dtype=torch.float32)
                elif z_values.device != self.device:
                    z_values = z_values.to(self.device)
                result['z_values'] = z_values
            elif 'points' in initial_geometry:
                # Extract z-values from points if available
                points = result['points']
                result['z_values'] = points[:, 2]  # z-coordinates
            else:
                # Generate default z-values based on the number of points per string
                if 'string_xy' in result:
                    # Determine total number of points based on total_points parameter
                    n_points = self.total_points
                    
                    # Create initial distribution of points across strings
                    default_points_per_string = [n_points // self.n_strings] * self.n_strings
                    remainder = n_points % self.n_strings
                    for i in range(remainder):
                        default_points_per_string[i] += 1
                    
                    # Generate z-values for each string
                    z_values_list = []
                    for s_idx in range(self.n_strings):
                        n_pts = default_points_per_string[s_idx]
                        if n_pts > 0:
                            string_z_segment = torch.linspace(-self.half_domain, self.half_domain, n_pts, device=self.device)
                            z_values_list.append(string_z_segment)
                    
                    # Combine all z-values
                    if z_values_list:
                        result['z_values'] = torch.cat(z_values_list)
            
            # Process string_indices if available
            if 'string_indices' in initial_geometry:
                result['string_indices'] = initial_geometry['string_indices']
            else:
                # If we have z_values and string_xy, calculate string_indices
                if 'z_values' in result and 'string_xy' in result:
                    # Get number of points from z_values
                    n_points = len(result['z_values'])
                    # If points are already available, use their XY coordinates to assign to nearest string
                    if 'points' in initial_geometry:
                        points = result['points']
                        # Calculate distances between each point and each string
                        distances = torch.zeros(n_points, self.n_strings, device=self.device)
                        for s_idx in range(self.n_strings):
                            string_xy_pos = result['string_xy'][s_idx]
                            # Calculate Euclidean distance in XY plane
                            distances[:, s_idx] = torch.sqrt((points[:, 0] - string_xy_pos[0])**2 + 
                                                            (points[:, 1] - string_xy_pos[1])**2)
                        
                        # Assign each point to the closest string
                        string_indices = torch.argmin(distances, dim=1).tolist()
                        result['string_indices'] = string_indices
                    else:
                        # Without points, distribute evenly
                        default_points_per_string = [n_points // self.n_strings] * self.n_strings
                        remainder = n_points % self.n_strings
                        for i in range(remainder):
                            default_points_per_string[i] += 1
                        
                        # Create string_indices list
                        default_string_indices = []
                        for s_idx in range(self.n_strings):
                            default_string_indices.extend([s_idx] * default_points_per_string[s_idx])
                        
                        result['string_indices'] = default_string_indices
            
            # Process points_per_string_list if available
            if 'points_per_string_list' in initial_geometry:
                result['points_per_string_list'] = initial_geometry['points_per_string_list']
            else:
                # Calculate points per string from string_indices if available
                if 'string_indices' in result:
                    string_indices = result['string_indices']
                    default_points_per_string = [0] * self.n_strings
                    for idx in string_indices:
                        if 0 <= idx < self.n_strings:  # Validate index
                            default_points_per_string[idx] += 1
                    
                    result['points_per_string_list'] = default_points_per_string
            
            # Process string_logits if available
            if 'string_logits' in initial_geometry and not self.even_distribution:
                string_logits = initial_geometry['string_logits']
                if not isinstance(string_logits, torch.Tensor):
                    string_logits = torch.tensor(string_logits, device=self.device, dtype=torch.float32)
                elif string_logits.device != self.device:
                    string_logits = string_logits.to(self.device)
                result['string_logits'] = string_logits
            elif not self.even_distribution:
                string_logits = torch.ones(self.n_strings, device=self.device, dtype=torch.float32) / self.n_strings
                result['string_logits'] = string_logits
            
            # Get the points
            if 'points' in initial_geometry:
                points = initial_geometry['points']
                if not isinstance(points, torch.Tensor):
                    points = torch.tensor(points, device=self.device, dtype=torch.float32)
                elif points.device != self.device:
                    points = points.to(self.device)
                result['points'] = points
            else:
                # Construct points from available data if possible
                if 'string_xy' in result and 'z_values' in result and 'string_indices' in result:
                    # Create points tensor
                    n_points = len(result['z_values'])
                    points_3d = torch.zeros(n_points, 3, device=self.device)
                    
                    # Set xy and z coordinates
                    for i, (s_idx, z_val) in enumerate(zip(result['string_indices'], result['z_values'])):
                        if 0 <= s_idx < self.n_strings:  # Validate index
                            points_3d[i, 0] = result['string_xy'][s_idx, 0]  # x value
                            points_3d[i, 1] = result['string_xy'][s_idx, 1]  # y value
                            points_3d[i, 2] = z_val  # z value
                    
                    result['points'] = points_3d
            
            # Final check to ensure all necessary components are available
            # If we still don't have points, but have other components, construct them now
            if 'points' not in result and 'string_xy' in result and 'z_values' in result and 'string_indices' in result:
                n_points = len(result['z_values'])
                points_3d = torch.zeros(n_points, 3, device=self.device)
                
                for i, (s_idx, z_val) in enumerate(zip(result['string_indices'], result['z_values'])):
                    if 0 <= s_idx < self.n_strings:  # Validate index
                        points_3d[i, 0] = result['string_xy'][s_idx, 0]  # x value
                        points_3d[i, 1] = result['string_xy'][s_idx, 1]  # y value
                        points_3d[i, 2] = z_val  # z value
                
                result['points'] = points_3d
            
            # Return the initialized geometry dict
            return self.update_points(**result)
            
        # Regular initialization if no initial geometry is provided
        # Initialize string_xy (hex grid or random based on self.optimize_xy)
        string_xy = self.hex_grid.clone()
        if self.optimize_xy:
            string_xy = torch.rand(self.n_strings, 2, device=self.device) * self.domain_size - self.half_domain

        string_logits = None
        points_3d = torch.zeros(self.total_points, 3, device=self.device)
        z_values_final = None
        string_indices_final = []
        points_per_string_list_final = None

        # Determine initialization strategy
        if not self.even_distribution:
            # This block implements point assignment based on weighted_xy and closest string,
            # preserving string_logits initialization based on self.random_initial_dist.
            # This replaces the previous soft assignment and commented-out hard assignment logic.
            if self.random_initial_dist:
                string_logits = self.initial_random_spread * torch.randn(self.n_strings, device=self.device)
            else:
                string_logits = torch.ones(self.n_strings, device=self.device, dtype=torch.float32) / self.n_strings
            
            string_probs = F.softmax(string_logits, dim=0)

            n_points = self.total_points
            # Using self.n_strings and self.device directly from the class instance

            # Calculate weighted_xy for each point based on global string_probs
            # string_xy_exp_for_weighted_avg: (n_points, self.n_strings, 2)
            string_xy_exp_for_weighted_avg = string_xy.unsqueeze(0).expand(n_points, self.n_strings, 2)
            # probs_exp_for_weighted_avg: (1, self.n_strings, 1) -> broadcasts
            probs_exp_for_weighted_avg = string_probs.unsqueeze(0).unsqueeze(-1)
            # weighted_xy: (n_points, 2) - each point gets its own weighted XY
            weighted_xy = (string_xy_exp_for_weighted_avg * probs_exp_for_weighted_avg).sum(dim=1)

            # Assign points to the closest string based on their weighted_xy
            # weighted_xy_expanded: (n_points, 1, 2)
            # string_xy_expanded_for_dist: (1, self.n_strings, 2)
            weighted_xy_expanded = weighted_xy.unsqueeze(1)
            string_xy_expanded_for_dist = string_xy.unsqueeze(0)

            # Calculate squared Euclidean distances: (n_points, self.n_strings)
            distances_sq = torch.sum((weighted_xy_expanded - string_xy_expanded_for_dist)**2, dim=2)
            # assigned_string_indices_for_points: (n_points,) - index of closest string for each point
            # Replace argmin with Gumbel-Softmax for differentiability
            assignment_logits = -distances_sq  # Higher score for smaller distance
            TAU = 0.01  # Temperature for Gumbel-Softmax; can be tuned or annealed
            # assigned_string_one_hot is (n_points, n_strings), one-hot in fwd, differentiable in bwd
            assigned_string_one_hot = F.gumbel_softmax(assignment_logits, tau=TAU, hard=True)
            # For subsequent logic needing integer indices (forward pass for these parts)
            assigned_string_indices_for_points = torch.argmax(assigned_string_one_hot, dim=1)

            # Calculate points per string
            points_per_string_counts = torch.zeros(self.n_strings, dtype=torch.long, device=self.device)
            for s_idx_loop in range(self.n_strings):
                points_per_string_counts[s_idx_loop] = (assigned_string_indices_for_points == s_idx_loop).sum()
            
            points_per_string_list_final = points_per_string_counts.tolist() # Store as list of ints

            # Reconstruct points_3d, z_values, and string_indices
            # points_3d is already torch.zeros(self.total_points, 3, device=self.device)
            current_point_fill_idx = 0
            _z_values_list = []
            _string_indices_list = [] # This will be assigned to string_indices_final

            for s_idx_loop in range(self.n_strings):
                num_points_on_this_string = points_per_string_counts[s_idx_loop].item()
                if num_points_on_this_string > 0:
                    start_idx = current_point_fill_idx
                    end_idx = current_point_fill_idx + num_points_on_this_string
                    
                    # Set XY for these points to the string's XY
                    points_3d[start_idx:end_idx, 0] = string_xy[s_idx_loop, 0]
                    points_3d[start_idx:end_idx, 1] = string_xy[s_idx_loop, 1]
                    
                    # Remap Z values uniformly along this string segment
                    string_z_segment = torch.linspace(-self.half_domain, self.half_domain, num_points_on_this_string, device=self.device)
                    points_3d[start_idx:end_idx, 2] = string_z_segment
                    
                    _z_values_list.append(string_z_segment)
                    _string_indices_list.extend([s_idx_loop] * num_points_on_this_string)
                    current_point_fill_idx += num_points_on_this_string
            
            if _z_values_list: # Ensure list is not empty before cat
                z_values_final = torch.cat(_z_values_list)
            else: # Handle case where no points are assigned (e.g. total_points = 0 or all strings get 0 points)
                z_values_final = torch.tensor([], device=self.device, dtype=torch.float32)

            string_indices_final = _string_indices_list
            # string_logits is already set from the start of this block
            # points_3d is modified in-place
            # points_per_string_list_final is set
        else: # self.even_distribution is True
            # Strategy 3: Even distribution -> HARD ASSIGNMENT, equal distribution
            # print(f"DEBUG GEOMETRY (initialize_points): Using EVEN distribution")
            string_logits = None 
            
            points_per_string_counts = [self.total_points // self.n_strings] * self.n_strings
            remainder = self.total_points % self.n_strings
            for i in range(remainder):
                points_per_string_counts[i] += 1
            
            _z_values_list = []
            current_idx = 0
            for s in range(self.n_strings):
                n_pts = points_per_string_counts[s]
                if n_pts > 0:
                    string_z_segment = torch.linspace(-self.half_domain, self.half_domain, n_pts, device=self.device)
                    points_3d[current_idx:current_idx+n_pts, 0] = string_xy[s, 0]
                    points_3d[current_idx:current_idx+n_pts, 1] = string_xy[s, 1]
                    points_3d[current_idx:current_idx+n_pts, 2] = string_z_segment
                    _z_values_list.append(string_z_segment)
                    string_indices_final.extend([s] * n_pts)
                    current_idx += n_pts
            z_values_final = torch.cat(_z_values_list) if _z_values_list else torch.tensor([], device=self.device)
            points_per_string_list_final = torch.tensor(points_per_string_counts, dtype=torch.float32, device=self.device) # Tensor

        # Ensure points_3d is correctly filled if total_points was not perfectly met by allocation
        # This is more relevant for hard allocation if current_idx < self.total_points
        # For soft allocation, points_3d is always (self.total_points, 3)

        return {
            "points": points_3d, 
            "z_values": z_values_final, 
            "string_xy": string_xy,
            "string_indices": string_indices_final, 
            "points_per_string_list": points_per_string_list_final, # This is now a tensor
            "string_logits": string_logits
        }
    
     
    def compute_string_distribution(self, string_logits):
        # Apply softmax to get probability distribution over strings
        string_probs = F.softmax(string_logits, dim=0)
        
        # Calculate ideal number of points per string based on probabilities
        # Each string gets at least min_points_per_string, then remaining points distributed by probability
        remaining_points = self.total_points - self.n_strings * self.min_points_per_string
        
        # Base distribution ensures minimum points per string
        ideal_points = [self.min_points_per_string] * self.n_strings
        ideal_points = torch.tensor(ideal_points, device=self.device)
        
        if remaining_points > 0:
            # Keep everything in PyTorch tensors to maintain gradient flow
            weighted_probs = string_probs * remaining_points
            integer_alloc = torch.floor(weighted_probs)
            allocated = integer_alloc.sum()
            if allocated < remaining_points:
                decimal_part = weighted_probs - integer_alloc
                _, sorted_indices = torch.sort(decimal_part, descending=True)
                extra_needed = int(remaining_points - allocated)
                for i in range(extra_needed):
                    integer_alloc[sorted_indices[i]] += 1
            # Do NOT cast to .long() or int here; keep as float for gradient flow
            for i in range(self.n_strings):
                ideal_points[i] = ideal_points[i] + integer_alloc[i]
        
        return ideal_points 
        
    def update_points(self, z_values, string_xy, points_per_string_list, string_indices, string_logits=None, **kwargs):
        """Update the points based on current optimization state.
        Parameters:
        -----------
        z_values : torch.Tensor
            Current z values for each point (n_points,) - Not used in the 'redis_phase' logic below.
        string_xy : torch.Tensor
            Current XY coordinates for each string (n_strings, 2)
        points_per_string_list : list
            Number of points per string (n_strings,) - Not used in the 'redis_phase' logic below.
        string_indices : list
            List of string indices for each point - Not used in the 'redis_phase' logic below.
        string_logits : torch.Tensor or None
            Logits for string importance (n_strings,)
        
        returns:
        --------
        dict
            Dictionary with updated tensors 
            """
        
        # print(f"DEBUG GEOMETRY: update_points called with even_distribution={self.even_distribution}, string_logits is None: {string_logits is None}")
        # if string_logits is not None:
            # print(f"DEBUG GEOMETRY: string_logits.requires_grad = {string_logits.requires_grad}")
            
        if not self.even_distribution and string_logits is not None and kwargs.get("redis_phase", False):
            # New logic:
            # 1. Calculate weighted_xy for all n_points.
            # 2. Assign each point to the string whose string_xy is closest to the point's weighted_xy.
            # 3. Reconstruct points_3d: XY from assigned string, Z remapped uniformly along that string.

            string_probs = F.softmax(string_logits, dim=0)  # (n_strings,)
            n_points = self.total_points
            n_strings = self.n_strings
            device = self.device

            # Calculate weighted_xy for each point based on global string_probs
            # string_xy_exp: (n_points, n_strings, 2)
            string_xy_exp_for_weighted_avg = string_xy.unsqueeze(0).expand(n_points, n_strings, 2)
            # probs_exp: (1, n_strings, 1) -> broadcasts to (n_points, n_strings, 1)
            probs_exp_for_weighted_avg = string_probs.unsqueeze(0).unsqueeze(-1)
            # weighted_xy: (n_points, 2) - each point gets its own weighted XY
            weighted_xy = (string_xy_exp_for_weighted_avg * probs_exp_for_weighted_avg).sum(dim=1)

            # Assign points to the closest string based on their weighted_xy
            # weighted_xy_expanded: (n_points, 1, 2)
            # string_xy_expanded_for_dist: (1, n_strings, 2)
            weighted_xy_expanded = weighted_xy.unsqueeze(1)
            string_xy_expanded_for_dist = string_xy.unsqueeze(0)

            # Calculate squared Euclidean distances: (n_points, n_strings)
            distances_sq = torch.sum((weighted_xy_expanded - string_xy_expanded_for_dist)**2, dim=2)
            # assigned_string_indices_for_points: (n_points,) - index of closest string for each original point
            # Replace argmin with Gumbel-Softmax for differentiability
            assignment_logits = -distances_sq  # Higher score for smaller distance
            TAU = 1.0  # Temperature for Gumbel-Softmax; can be tuned or annealed
            # assigned_string_one_hot is (n_points, n_strings), one-hot in fwd, differentiable in bwd
            assigned_string_one_hot = F.gumbel_softmax(assignment_logits, tau=TAU, hard=True)
            # For subsequent logic needing integer indices (forward pass for these parts)
            assigned_string_indices_for_points = torch.argmax(assigned_string_one_hot, dim=1)

            # Prepare for reconstructing points_3d, ordered by string
            new_points_3d = torch.zeros(n_points, 3, device=device)
            
            points_per_string_counts = torch.zeros(n_strings, dtype=torch.long, device=device)
            for s_idx in range(n_strings):
                points_per_string_counts[s_idx] = (assigned_string_indices_for_points == s_idx).sum()
            
            points_per_string_counts_list_output = points_per_string_counts.tolist()
            
            current_point_fill_idx = 0
            new_z_values_segments_list = []
            final_string_indices_output_list = []

            for s_idx in range(n_strings):
                num_points_on_this_string = points_per_string_counts[s_idx].item()
                if num_points_on_this_string > 0:
                    start_idx = current_point_fill_idx
                    end_idx = current_point_fill_idx + num_points_on_this_string
                    
                    # Set XY for these points to the string's XY
                    new_points_3d[start_idx:end_idx, 0] = string_xy[s_idx, 0]
                    new_points_3d[start_idx:end_idx, 1] = string_xy[s_idx, 1]
                    
                    # Remap Z values uniformly along this string segment
                    string_z_segment = torch.linspace(-self.half_domain, self.half_domain, num_points_on_this_string, device=device)
                    new_points_3d[start_idx:end_idx, 2] = string_z_segment
                    
                    new_z_values_segments_list.append(string_z_segment)
                    final_string_indices_output_list.extend([s_idx] * num_points_on_this_string)
                    current_point_fill_idx += num_points_on_this_string
            
            final_concat_z_values = torch.cat(new_z_values_segments_list) if new_z_values_segments_list else torch.tensor([], device=device, dtype=torch.float32)

            # Ensure all points were processed
            if current_point_fill_idx != n_points:
                # This might happen if some points couldn't be assigned or counts were off.
                # For robustness, one might add error handling or a fallback.
                # Assuming for now that all points get assigned.
                pass

            return {
                "points": new_points_3d,
                "z_values": final_concat_z_values,
                "string_xy": string_xy, # Passed through
                "string_indices": final_string_indices_output_list,
                "points_per_string_list": points_per_string_counts_list_output,
                "string_logits": string_logits # Passed through
            }
        
        # Original hard allocation logic (if the above condition is not met)
        # Create new points_3d with updated distribution
        new_points_3d = torch.zeros(self.total_points, 3, device=self.device)
        current_idx = 0
        for s in range(self.n_strings):
            n_pts = int(points_per_string_list[s])
            if n_pts > 0:  # Skip empty strings
                if self.optimize_xy:
                    new_points_3d[current_idx:current_idx+n_pts, 0] = string_xy[s, 0]  # x value
                    new_points_3d[current_idx:current_idx+n_pts, 1] = string_xy[s, 1]  # y value
                else:
                    new_points_3d[current_idx:current_idx+n_pts, 0] = self.hex_grid[s, 0]  # x value
                    new_points_3d[current_idx:current_idx+n_pts, 1] = self.hex_grid[s, 1]  # y value
                
                new_points_3d[current_idx:current_idx+n_pts, 2] = z_values[current_idx:current_idx+n_pts]  # z value
                current_idx += n_pts
        
        # Update the variables
        points_3d = new_points_3d
        
        return {
            "points": points_3d, 
            "z_values": z_values, 
            "string_xy": string_xy,
            "string_indices": string_indices, 
            "points_per_string_list": points_per_string_list,
            "string_logits": string_logits if not self.even_distribution else None}
            
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
            
        # Create parameters for a 1D continuous path from 0 to 1
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


