import torch
import numpy as np
from scipy.optimize import linear_sum_assignment




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
    
    def create_uniform_hexagonal_grid(self, n_points=50, optimal_spacing=None, **kwaargs):
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
        
        # Calculate number of rings needed to reach n_points
        # For hexagonal grid: points in rings 0 to r = 1 + 3*r*(r+1)
        # Solve for r: 3r^2 + 3r + 1 - n_points = 0
        # Using quadratic formula: r = (-3 + sqrt(9 + 12*(n_points-1))) / 6
        n_points_tensor = torch.tensor(n_points - 1, device=self.device, dtype=torch.float32)
        required_rings = torch.ceil((-3 + torch.sqrt(9 + 12 * n_points_tensor)) / 6).int().item()
        
        # Use required rings to generate exactly n_points (allow points beyond domain)
        max_possible_rings = required_rings + 1  # +1 for safety margin
        
        # Generate rings; for the final (partial) ring, distribute points evenly along the hex perimeter
        for ring in range(1, max_possible_rings + 1):
                
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
            
            # Determine how many points we still need
            points_remaining = n_points - all_points.shape[0]
            if points_remaining <= 0:
                break

            full_ring_points = 6 * ring

            if points_remaining >= full_ring_points:
                # Generate a full ring: ring points per side (total 6*ring), excluding duplicate corners
                ring_points = torch.zeros(0, 2, device=self.device, dtype=torch.float32)
                for side in range(6):
                    start_corner = corners[side]  # (2,)
                    end_corner = corners[(side + 1) % 6]
                    # Create interpolation parameters for this side
                    t_values = torch.linspace(0, 1, ring + 1, device=self.device)[:-1]  # Exclude end to avoid duplicates
                    side_points = start_corner.unsqueeze(0) + t_values.unsqueeze(1) * (end_corner - start_corner).unsqueeze(0)
                    if side_points.shape[0] > 0:
                        ring_points = torch.cat([ring_points, side_points], dim=0)
                if ring_points.shape[0] > 0:
                    all_points = torch.cat([all_points, ring_points], dim=0)
            else:
                # Partial last ring: evenly spread the required number of points along the hexagon perimeter
                m = points_remaining  # number of points to place on this ring
                # Edge length (distance between consecutive corners) = ring * spacing
                edge_vec = corners[1] - corners[0]
                edge_len = torch.norm(edge_vec)
                total_perimeter = edge_len * 6.0

                # Evenly spaced arc-length positions along [0, total_perimeter)
                # Use arange to avoid reliance on endpoint kwarg
                positions = (torch.arange(m, device=self.device, dtype=torch.float32) / m) * total_perimeter
                # Side index for each position (0..5)
                side_idx = torch.floor(positions / edge_len).to(torch.int64)
                side_idx = torch.clamp(side_idx, 0, 5)
                # Local parameter t in [0,1) along each side
                t_on_side = (positions - side_idx.to(torch.float32) * edge_len) / edge_len

                # Gather start and end corners for each side
                start_corners = corners[side_idx]               # (m, 2)
                next_idx = (side_idx + 1) % 6
                end_corners = corners[next_idx]                 # (m, 2)
                ring_points = start_corners + t_on_side.unsqueeze(1) * (end_corners - start_corners)

                all_points = torch.cat([all_points, ring_points], dim=0)
                # We've placed exactly the needed points; stop
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
    
    def create_circular_hexagonal_grid(self, n_points=50, optimal_spacing=None, **kwaargs):
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
    
    def create_sunflower_grid(self, n_points=50, optimal_spacing=None, **kwaargs):
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

    def create_hybrid_hex_sunflower_grid(self, n_points=50, optimal_spacing=None, **kwaargs):
        """Create a hybrid grid blending a hexagonal lattice and a sunflower (golden-angle) spiral.

        Steps:
        1. Generate hexagonal lattice (center, then rings).
        2. Generate sunflower point set via create_sunflower_grid.
        3. Optionally rescale sunflower so its max radius equals the hex lattice max radius.
        4. Match points (Hungarian) and interpolate linearly in Cartesian space.

        mix semantics:
        - mix = 0.0 : pure hex lattice.
        - mix = 1.0 : pure sunflower (possibly radius-adjusted if match_hex_radius True).
        - (0,1)   : blended positions.

        Kwargs:
        - hybrid_mix (float)          : blend factor in [0,1].
        - return_matches (bool)       : return (points, match_index_tensor).
        - skip_hungarian (bool)       : use index-wise correspondence instead of matching.
        - iterative_hungarian (bool)  : perform incremental morph with re-matching each step.
        - iter_step (float)           : step size for iterative morph (default 0.01, clamped).
        - match_hex_radius (bool)     : rescale sunflower to hex max radius (default True).
        """
        mix = kwaargs.get('hybrid_mix', 0.5)
        return_matches = kwaargs.get('return_matches', False)
        skip_hungarian = kwaargs.get('skip_hungarian', False)
        iterative = kwaargs.get('iterative_hungarian', False)
        iter_step = float(kwaargs.get('iter_step', 0.001))
        match_hex_radius = bool(kwaargs.get('match_hex_radius', True))
        max_iters = kwaargs.get('max_iters', 1000)

        # Normalize mix
        if not isinstance(mix, (torch.Tensor)):
            mix_t = torch.tensor(mix, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            mix_t = mix.clamp(0.0, 1.0)

        # Hex lattice
        hex_points = self.create_uniform_hexagonal_grid(n_points=n_points, optimal_spacing=optimal_spacing)
        # if hex_points.shape[0] < n_points:
        #     need = n_points - hex_points.shape[0]
        #     pad = hex_points[-1:].repeat(need, 1) if hex_points.shape[0] > 0 else torch.zeros(need, 2, device=self.device)
        #     hex_points = torch.cat([hex_points, pad], dim=0)

        # Spacing (kept for potential downstream use)
        # if optimal_spacing is None and hex_points.shape[0] > 1:
            # dists = torch.cdist(hex_points, hex_points)
            # inf = torch.tensor(float('inf'), device=self.device, dtype=dists.dtype)
            # idx = torch.arange(dists.shape[0], device=self.device)
            # dists[idx, idx] = inf
            # nn = torch.min(dists, dim=1).values
            # _median_spacing = torch.median(nn)
        # else: use provided optimal_spacing (not needed explicitly here)

        # Sunflower: compute spacing to match hex max radius if requested
        # print(mix_t)
        if match_hex_radius:
            # Determine hex outer radius
            hex_radii = torch.sqrt((hex_points ** 2).sum(dim=1))
            hex_max = hex_radii.max()
            if n_points <= 1:
                sunflower_spacing = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                sunflower_spacing = hex_max / torch.sqrt(torch.tensor(float(n_points - 1), device=self.device))
            sunflower_points = self.create_sunflower_grid(n_points=n_points, optimal_spacing=sunflower_spacing)
        else:
            sunflower_points = self.create_sunflower_grid(n_points=n_points, optimal_spacing=optimal_spacing)
        # print('sunflower spacing:', sunflower_spacing.item() if match_hex_radius else 'N/A')
        # Fast exits for pure endpoints (handle skip_hungarian flag coherently)
        if mix_t.item() <= 1e-5:
            if return_matches:
                idxs = torch.arange(n_points, device=self.device)
                matches_tensor = torch.stack([idxs, idxs], dim=1)
                return hex_points[:n_points], matches_tensor
            return hex_points[:n_points]
        if mix_t.item() >= 1.0 - 1e-5 and not iterative:
            if skip_hungarian or not return_matches:
                return sunflower_points[:n_points] if not return_matches else (sunflower_points[:n_points], None)
            # Provide Hungarian matches at endpoint
            dist_matrix = torch.cdist(hex_points, sunflower_points, p=2)
            row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
            matches_tensor = torch.stack([
                torch.tensor(row_idx, device=self.device),
                torch.tensor(col_idx, device=self.device)
            ], dim=1)
            return sunflower_points[:n_points], matches_tensor

        # Index-wise interpolation option
        if skip_hungarian and not iterative:
            blended = hex_points + mix_t * (sunflower_points - hex_points)
            return blended[:n_points] if not return_matches else (blended[:n_points], None)

        # Iterative Hungarian morph
        if iterative:
            step_abs = max(1e-4, min(0.25, float(iter_step)))
            target_mix = mix_t
            current_mix = 0.0
            current = hex_points.clone()
            last_row_idx = None
            last_col_idx = None
            max
            # max_iters = int(np.ceil(target_mix / step_abs)) + 2
            dist_matrix = torch.cdist(current, sunflower_points, p=2)
            r0, c0 = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
            old_dist_sum = torch.sum(dist_matrix[r0, c0])
            for _ in range(max_iters):
                
                # dist_matrix = torch.cdist(current, sunflower_points, p=2)
                # row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
                # last_row_idx, last_col_idx = row_idx, col_idx
                # matched_targets = torch.empty_like(current)
                # matched_targets[row_idx] = sunflower_points[col_idx]
                # remaining = target_mix - current_mix
                # this_step = min(step_abs, remaining)
                # current = current + this_step * (matched_targets - current)
                # current_mix += this_step
                dist_matrix = torch.cdist(current, sunflower_points, p=2)
                row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
                new_total_dist_matrix = torch.sum(dist_matrix[row_idx, col_idx])
                current_mix = 1 - (new_total_dist_matrix.item() / old_dist_sum.item()) 
                if current_mix >= target_mix - 1e-3:
                    break
                
                last_row_idx, last_col_idx = row_idx, col_idx

                # Build targets aligned to hex indices (row indices are current/hex indices)
                targets = torch.empty_like(current)
                targets[row_idx] = sunflower_points[col_idx]

                # Advance absolute mix and blend w.r.t. original hex base
                # next_mix = min(target_mix, current_mix + step_abs)
                # alpha = float(next_mix)
                # current = (1.0 - alpha) * current + alpha * targets
                current = current + step_abs * (targets - current)
                # current_mix = next_mix
            if return_matches and last_row_idx is not None:
                matches_tensor = torch.stack([
                    torch.tensor(last_row_idx, device=self.device),
                    torch.tensor(last_col_idx, device=self.device)
                ], dim=1)
                return current[:n_points], matches_tensor
            return current[:n_points]

        # One-shot Hungarian blend
        dist_matrix = torch.cdist(hex_points, sunflower_points, p=2)
        row_idx, col_idx = linear_sum_assignment(dist_matrix.detach().cpu().numpy())
        blended = torch.zeros_like(hex_points)
        for r, c in zip(row_idx, col_idx):
            h_pt = hex_points[r]
            s_pt = sunflower_points[c]
            blended[r] = h_pt + mix_t * (s_pt - h_pt)
        # Unmatched safety (unlikely with equal counts)
        mask = torch.zeros(n_points, dtype=torch.bool, device=self.device)
        mask[row_idx] = True
        if (~mask).any():
            blended[~mask] = hex_points[~mask]
        if return_matches:
            matches_tensor = torch.stack([
                torch.tensor(row_idx, device=self.device),
                torch.tensor(col_idx, device=self.device)
            ], dim=1)
            return blended[:n_points], matches_tensor
        return blended[:n_points]
    
def compare_geometries(geometry1, geometry2, points_key='points', 
                       unmatched_penalty='domain_diagonal', penalty_scale=1.0,
                       use_string_weights=False, string_weights_key='string_weights',
                       string_xy_key='string_xy', apply_sigmoid=True):
    """
    Compare two geometries by finding optimal point correspondence using Hungarian matching
    and computing the average distance between matched points.
    
    Handles geometries with different numbers of points by using a rectangular cost matrix
    and applying penalties for unmatched points.
    
    Can optionally handle string-based geometries where multiple points belong to the same
    string (XY location). String weights are assigned to each point, and pairwise distances
    are divided by the product of corresponding point weights.
    
    Parameters:
    -----------
    geometry1 : dict
        First geometry dictionary containing 3D points
    geometry2 : dict
        Second geometry dictionary containing 3D points
    points_key : str
        Key to access points in the geometry dictionaries (default: 'points')
    unmatched_penalty : str or float
        Method to penalize unmatched points. Options:
        - 'domain_diagonal': Use the diagonal of the domain as penalty (default)
        - 'max_distance': Use the maximum pairwise distance in the cost matrix
        - 'mean_distance': Use the mean pairwise distance in the cost matrix
        - float: Use a specific penalty value
    penalty_scale : float
        Multiplier for the penalty value (default: 1.0)
    use_string_weights : bool
        If True, use string-based weighting for evanescent string geometries (default: False)
    string_weights_key : str
        Key to access string weights in geometry dictionaries (default: 'string_weights')
    string_xy_key : str
        Key to access string XY coordinates (default: 'string_xy')
    apply_sigmoid : bool
        If True, apply sigmoid to string weights before using them (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'average_distance': Mean distance between all points (including penalty for unmatched)
        - 'matched_average_distance': Mean distance between only matched pairs
        - 'total_distance': Sum of all matched distances plus penalties
        - 'matches': Array of (index1, index2) pairs showing correspondence
        - 'distances': Array of distances for each matched pair
        - 'n_matched': Number of matched pairs
        - 'n_unmatched': Number of unmatched points
        - 'penalty_contribution': Total penalty added for unmatched points
    """
    # Extract points from geometries
    points1 = geometry1[points_key]
    points2 = geometry2[points_key]
    
    # Assign weights to each point based on string membership
    if use_string_weights:
        string_weights1 = geometry1.get(string_weights_key, None)
        string_weights2 = geometry2.get(string_weights_key, None)
        string_xy1 = geometry1.get(string_xy_key, None)
        string_xy2 = geometry2.get(string_xy_key, None)
        
        # Apply sigmoid to string weights if requested
        if string_weights1 is not None and apply_sigmoid:
            string_weights1 = torch.sigmoid(string_weights1)
        if string_weights2 is not None and apply_sigmoid:
            string_weights2 = torch.sigmoid(string_weights2)
        
        # Assign string weights to each point
        point_weights1 = _assign_string_weights_to_points(
            points1, string_xy1, string_weights1, geometry1
        )
        point_weights2 = _assign_string_weights_to_points(
            points2, string_xy2, string_weights2, geometry2
        )
    else:
        point_weights1 = point_weights2 = None
    
    # Get number of points
    n_points1 = points1.shape[0]
    n_points2 = points2.shape[0]
    n_matched = min(n_points1, n_points2)
    n_unmatched = abs(n_points1 - n_points2)
    
    # Compute pairwise distance matrix
    # shape: (n_points1, n_points2)
    distance_matrix = torch.cdist(points1, points2, p=2)
    
    # Divide distances by product of point weights if using string weights
    if use_string_weights and point_weights1 is not None and point_weights2 is not None:
        # Create weight product matrix: (n_points1, n_points2)
        weight_products = point_weights1.unsqueeze(1) * point_weights2.unsqueeze(0)
        # Avoid division by zero by adding small epsilon
        weight_products = torch.clamp(weight_products, min=1e-8)
        # Divide distances by weight products
        distance_matrix = distance_matrix / weight_products
    
    # Determine penalty value for unmatched points
    if isinstance(unmatched_penalty, (int, float)):
        penalty_value = float(unmatched_penalty) * penalty_scale
    elif unmatched_penalty == 'domain_diagonal':
        # Estimate domain size from the data
        all_points = torch.cat([points1, points2], dim=0)
        domain_range = (all_points.max(dim=0).values - all_points.min(dim=0).values)
        domain_diagonal = torch.sqrt((domain_range ** 2).sum()).item()
        penalty_value = domain_diagonal * penalty_scale
    elif unmatched_penalty == 'max_distance':
        penalty_value = distance_matrix.max().item() * penalty_scale
    elif unmatched_penalty == 'mean_distance':
        penalty_value = distance_matrix.mean().item() * penalty_scale
    else:
        raise ValueError(f"Unknown unmatched_penalty method: {unmatched_penalty}")
    
    # Create square cost matrix by padding with penalty values
    max_n = max(n_points1, n_points2)
    cost_matrix = np.full((max_n, max_n), penalty_value, dtype=np.float32)
    
    # Fill in the actual distances
    distance_matrix_np = distance_matrix.cpu().detach().numpy()
    cost_matrix[:n_points1, :n_points2] = distance_matrix_np
    
    # Apply Hungarian matching to find optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out dummy matches (where penalty was used)
    valid_matches_mask = (row_indices < n_points1) & (col_indices < n_points2)
    valid_row_indices = row_indices[valid_matches_mask]
    valid_col_indices = col_indices[valid_matches_mask]
    
    # Get distances for valid matched pairs
    if len(valid_row_indices) > 0:
        matched_distances = distance_matrix[valid_row_indices, valid_col_indices]
        matched_average_distance = torch.mean(matched_distances).item()
        total_matched_distance = torch.sum(matched_distances).item()
    else:
        matched_distances = torch.tensor([], device=points1.device)
        matched_average_distance = 0.0
        total_matched_distance = 0.0
    
    # Calculate penalty contribution
    penalty_contribution = n_unmatched * penalty_value
    total_distance = total_matched_distance + penalty_contribution
    
    # Calculate overall average (including penalties)
    average_distance = total_distance / max_n
    
    # Create matches array (only valid matches)
    if len(valid_row_indices) > 0:
        matches = np.stack([valid_row_indices, valid_col_indices], axis=1)
    else:
        matches = np.array([]).reshape(0, 2)
    
    return {
        'average_distance': average_distance,
        'matched_average_distance': matched_average_distance,
        'total_distance': total_distance,
        'matches': matches,
        'distances': matched_distances.cpu().detach().numpy(),
        'n_matched': len(valid_row_indices),
        'n_unmatched': n_unmatched,
        'penalty_contribution': penalty_contribution,
        'penalty_value': penalty_value
    }


def _assign_string_weights_to_points(points, string_xy, string_weights, geometry_dict):
    """
    Assign string weights to individual points based on which string they belong to.
    
    For evanescent string geometries, multiple points share the same XY location (string).
    This function identifies which string each point belongs to and assigns the 
    corresponding string weight.
    
    Parameters:
    -----------
    points : torch.Tensor
        Points tensor of shape (n_points, 3) with XYZ coordinates
    string_xy : torch.Tensor or None
        XY coordinates for each string of shape (n_strings, 2)
    string_weights : torch.Tensor or None
        Weights for each string of shape (n_strings,)
    geometry_dict : dict
        Full geometry dictionary that may contain additional info like 'points_per_string_list'
        
    Returns:
    --------
    torch.Tensor
        Point weights of shape (n_points,) where each point has the weight of its string
    """
    device = points.device
    n_points = points.shape[0]
    
    # If no string structure, return uniform weights
    if string_xy is None or string_weights is None:
        return torch.ones(n_points, device=device, dtype=torch.float32)
    
    n_strings = string_xy.shape[0]
    point_weights = torch.zeros(n_points, device=device, dtype=torch.float32)
    
    # Get points_per_string from geometry if available
    points_per_string_list = geometry_dict.get('points_per_string_list', None)
    
    if points_per_string_list is not None:
        # Use the provided structure - points are organized sequentially by string
        for string_idx in range(n_strings):
            # Assume uniform points per string for simplicity
            points_per_string = points_per_string_list[0] if isinstance(points_per_string_list, list) else points_per_string_list
            
            start_idx = string_idx * points_per_string
            end_idx = start_idx + points_per_string
            
            if end_idx <= n_points:
                # Assign this string's weight to all its points
                point_weights[start_idx:end_idx] = string_weights[string_idx]
            else:
                # Handle edge case where we run out of points
                point_weights[start_idx:n_points] = string_weights[string_idx]
                break
    else:
        # Fallback: match points to strings by XY proximity
        for point_idx in range(n_points):
            point_xy = points[point_idx, :2]
            # Find closest string
            xy_distances = torch.norm(string_xy - point_xy.unsqueeze(0), dim=1)
            closest_string_idx = torch.argmin(xy_distances)
            point_weights[point_idx] = string_weights[closest_string_idx]
    
    return point_weights
    