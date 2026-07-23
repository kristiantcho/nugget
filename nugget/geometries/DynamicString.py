from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np

class DynamicString(Geometry):
    """Dynamic string geometry optimizer."""
    
    def __init__(self, device=None, dim=3, domain_size=2,
                total_points = 150, n_strings = 30,
                random_xy = False,
                custom_z_spacing = None, points_per_string = None,
                custom_string_spacing = None, hex_type='hexagonal', hybrid_mix_init=0.5):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.n_strings = n_strings
        self.points_per_string = points_per_string
        self.total_points = int(self.n_strings * self.points_per_string) if self.points_per_string is not None else total_points
        self.random_xy = random_xy
        self.custom_z_spacing = custom_z_spacing
        self.custom_string_spacing = custom_string_spacing
        self.hybrid_mix = hybrid_mix_init
        if hex_type == 'hexagonal':
            self.hex_func = self.create_uniform_hexagonal_grid
        elif hex_type == 'circular':
            self.hex_func = self.create_circular_hexagonal_grid
        elif hex_type == 'sunflower':
            self.hex_func = self.create_sunflower_grid
        else:
            self.hex_func = self.create_uniform_hexagonal_grid
        original_dim = self.dim
        self.dim = 2
        self.hex_grid = self.hex_func(
            n_points=self.n_strings,
            optimal_spacing=self.custom_string_spacing,
            hybrid_mix=self.hybrid_mix,
        )
        self.dim = original_dim

    def _make_z_segment(self, n_points):
        if n_points <= 0:
            return torch.tensor([], device=self.device, dtype=torch.float32)
        if self.custom_z_spacing is not None:
            return self.custom_z_spacing * (
                torch.arange(n_points, device=self.device, dtype=torch.float32)
                - (n_points - 1) / 2.0
            )
        return torch.linspace(-self.half_domain, self.half_domain, n_points, device=self.device)

    def _sync_total_points_from_points_per_string(self):
        if self.points_per_string is not None:
            self.total_points = int(self.n_strings * self.points_per_string)
            
    
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
            
            # Handle string weight filtering first
            active_strings_mask = None
            weight_threshold = initial_geometry.get('weight_threshold', 0.7)
            
            if 'string_weights' in initial_geometry:
                string_weights = initial_geometry['string_weights']
                if not isinstance(string_weights, torch.Tensor):
                    string_weights = torch.tensor(string_weights, device=self.device, dtype=torch.float32)
                elif string_weights.device != self.device:
                    string_weights = string_weights.to(self.device)
                
                # Apply weight filtering - use same logic as EvanescentString
                string_probs = torch.sigmoid(string_weights) 
                active_strings_mask = string_probs > weight_threshold
                active_string_indices = torch.where(active_strings_mask)[0]
                print(f"Filtering strings: {len(active_string_indices)} out of {len(string_weights)} strings active")
            
            # Process string_xy if available
            if 'string_xy' in initial_geometry:
                string_xy = initial_geometry['string_xy']
                if not isinstance(string_xy, torch.Tensor):
                    string_xy = torch.tensor(string_xy, device=self.device, dtype=torch.float32)
                elif string_xy.device != self.device:
                    string_xy = string_xy.to(self.device)
                
                # Apply string filtering if weight mask is available
                if active_strings_mask is not None:
                    string_xy = string_xy[active_strings_mask]
                    # Update n_strings to reflect filtered strings
                    self.n_strings = len(string_xy)
                    self._sync_total_points_from_points_per_string()
                    print(f"Filtered string_xy to {self.n_strings} strings")
                result['string_xy'] = string_xy
            else:
                # Fall back to default initialization
                string_xy = self.hex_grid.clone()
                if self.random_xy:
                    string_xy = torch.rand(self.n_strings, 2, device=self.device) * self.domain_size - self.half_domain
                # if active_strings_mask is not None:
                #     string_xy = string_xy[active_strings_mask]
                #     self.n_strings = len(string_xy)
                #     self._sync_total_points_from_points_per_string()
                result['string_xy'] = string_xy
            
            # Process z_values if available
            if 'z_values' in initial_geometry:
                z_values = initial_geometry['z_values']
                if not isinstance(z_values, torch.Tensor):
                    z_values = torch.tensor(z_values, device=self.device, dtype=torch.float32)
                elif z_values.device != self.device:
                    z_values = z_values.to(self.device)
                
                # If filtering strings, we need to filter points that belong to inactive strings
                if active_strings_mask is not None:
                    # For EvanescentString geometry, we need to handle the case where
                    # z_values contains all points but string_indices are per-string
                    if 'points_per_string_list' in initial_geometry:
                        points_per_string_list = initial_geometry['points_per_string_list']
                        if isinstance(points_per_string_list, list):
                            points_per_string_list = torch.tensor(points_per_string_list, device=self.device)
                        
                        active_string_indices = torch.where(active_strings_mask)[0]
                        
                        # Create a mask for all points based on which strings are active
                        active_points_mask = torch.zeros(len(z_values), dtype=torch.bool, device=self.device)
                        current_point_idx = 0
                        
                        for string_idx in range(len(points_per_string_list)):
                            points_in_string = int(points_per_string_list[string_idx])
                            if string_idx in active_string_indices:
                                # Mark these points as active
                                active_points_mask[current_point_idx:current_point_idx + points_in_string] = True
                            current_point_idx += points_in_string
                        
                        z_values = z_values[active_points_mask]
                        print(f"Filtered z_values from {len(initial_geometry['z_values'])} to {len(z_values)} points")
                    elif 'string_indices' in initial_geometry and len(initial_geometry['string_indices']) == len(z_values):
                        # This is the case where string_indices is per-point (like DynamicString output)
                        string_indices = initial_geometry['string_indices']
                        if not isinstance(string_indices, torch.Tensor):
                            string_indices = torch.tensor(string_indices, device=self.device, dtype=torch.long)
                        elif string_indices.device != self.device:
                            string_indices = string_indices.to(self.device)
                        
                        active_string_indices = torch.where(active_strings_mask)[0]
                        # Filter points to only include those from active strings
                        active_points_mask = torch.isin(string_indices, active_string_indices)
                        z_values = z_values[active_points_mask]
                        print(f"Filtered z_values from {len(initial_geometry['z_values'])} to {len(z_values)} points")
                
                result['z_values'] = z_values
            elif 'points_3d' in initial_geometry:
                # Extract z-values from points if available
                points = initial_geometry['points_3d']
                if not isinstance(points, torch.Tensor):
                    points = torch.tensor(points, device=self.device, dtype=torch.float32)
                elif points.device != self.device:
                    points = points.to(self.device)
                
                # If filtering strings, we need to filter points that belong to inactive strings
                if active_strings_mask is not None:
                    # For EvanescentString geometry, we need to handle the case where
                    # points contain all points but string_indices are per-string
                    if 'points_per_string_list' in initial_geometry:
                        points_per_string_list = initial_geometry['points_per_string_list']
                        if isinstance(points_per_string_list, list):
                            points_per_string_list = torch.tensor(points_per_string_list, device=self.device)
                        
                        active_string_indices = torch.where(active_strings_mask)[0]
                        
                        # Create a mask for all points based on which strings are active
                        active_points_mask = torch.zeros(len(points), dtype=torch.bool, device=self.device)
                        current_point_idx = 0
                        
                        for string_idx in range(len(points_per_string_list)):
                            points_in_string = int(points_per_string_list[string_idx])
                            if string_idx in active_string_indices:
                                # Mark these points as active
                                active_points_mask[current_point_idx:current_point_idx + points_in_string] = True
                            current_point_idx += points_in_string
                        
                        points = points[active_points_mask]
                        print(f"Filtered points from {len(initial_geometry['points_3d'])} to {len(points)} points")
                    elif 'string_indices' in initial_geometry and len(initial_geometry['string_indices']) == len(points):
                        # This is the case where string_indices is per-point (like DynamicString output)
                        string_indices = initial_geometry['string_indices']
                        if not isinstance(string_indices, torch.Tensor):
                            string_indices = torch.tensor(string_indices, device=self.device, dtype=torch.long)
                        elif string_indices.device != self.device:
                            string_indices = string_indices.to(self.device)
                        
                        active_string_indices = torch.where(active_strings_mask)[0]
                        # Filter points to only include those from active strings
                        active_points_mask = torch.isin(string_indices, active_string_indices)
                        points = points[active_points_mask]
                        print(f"Filtered points from {len(initial_geometry['points_3d'])} to {len(points)} points")
                
                result['z_values'] = points[:, 2]  # z-coordinates
            else:
                # Generate default z-values based on the number of points per string
                if 'string_xy' in result:
                    # Determine total number of points based on the total_points parameter
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
                            string_z_segment = self._make_z_segment(n_pts)
                            z_values_list.append(string_z_segment)
                    
                    # Combine all z-values
                    if z_values_list:
                        result['z_values'] = torch.cat(z_values_list)
            
            # Process string_indices if available
            if 'string_indices' in initial_geometry:
                string_indices = initial_geometry['string_indices']
                
                # If filtering strings, we need to remap string indices and filter points
                if active_strings_mask is not None:
                    if not isinstance(string_indices, torch.Tensor):
                        string_indices = torch.tensor(string_indices, device=self.device, dtype=torch.long)
                    elif string_indices.device != self.device:
                        string_indices = string_indices.to(self.device)
                    
                    active_string_indices = torch.where(active_strings_mask)[0]
                    # Create a mapping from old indices to new indices
                    old_to_new_mapping = torch.full((active_strings_mask.size(0),), -1, device=self.device, dtype=torch.long)
                    old_to_new_mapping[active_string_indices] = torch.arange(len(active_string_indices), device=self.device)
                    
                    # Handle different cases of string_indices
                    if len(string_indices) == len(active_strings_mask):
                        # EvanescentString case: string_indices is per-string, need to expand to per-point
                        if 'points_per_string_list' in initial_geometry:
                            points_per_string_list = initial_geometry['points_per_string_list']
                            if isinstance(points_per_string_list, list):
                                points_per_string_list = torch.tensor(points_per_string_list, device=self.device)
                            
                            # Create per-point string indices for active strings only
                            new_string_indices = []
                            new_string_idx = 0
                            for old_string_idx in range(len(active_strings_mask)):
                                points_in_string = int(points_per_string_list[old_string_idx])
                                if old_string_idx in active_string_indices:
                                    # Add points for this active string
                                    new_string_indices.extend([new_string_idx] * points_in_string)
                                    new_string_idx += 1
                            
                            string_indices = torch.tensor(new_string_indices, device=self.device, dtype=torch.long)
                            self.total_points = len(string_indices)
                            print(f"Created per-point string_indices for active strings, total_points now: {self.total_points}")
                        else:
                            # Fallback: just filter the string indices
                            string_indices = string_indices[active_string_indices]
                            string_indices = old_to_new_mapping[string_indices]
                            print(f"Filtered string-level indices")
                    else:
                        # DynamicString case: string_indices is per-point
                        # Filter points to only include those from active strings
                        active_points_mask = torch.isin(string_indices, active_string_indices)
                        string_indices = string_indices[active_points_mask]
                        
                        # Remap string indices to new numbering
                        string_indices = old_to_new_mapping[string_indices]
                        
                        # Update total_points to reflect filtering
                        self.total_points = len(string_indices)
                        print(f"Filtered per-point string_indices, total_points now: {self.total_points}")
                
                result['string_indices'] = string_indices.tolist() if isinstance(string_indices, torch.Tensor) else string_indices
            else:
                # If we have z_values and string_xy, calculate string_indices
                if 'z_values' in result and 'string_xy' in result:
                    # Get number of points from z_values
                    n_points = len(result['z_values'])
                    # If points are already available, use their XY coordinates to assign to nearest string
                    if 'points_3d' in initial_geometry:
                        points = result['points_3d']
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
                points_per_string_list = initial_geometry['points_per_string_list']
                
                # If filtering strings, we need to filter the points_per_string_list
                if active_strings_mask is not None:
                    if isinstance(points_per_string_list, torch.Tensor):
                        points_per_string_list = points_per_string_list[active_strings_mask]
                    else:
                        # Convert to tensor, filter, then back to list
                        points_per_string_tensor = torch.tensor(points_per_string_list, device=self.device)
                        points_per_string_list = points_per_string_tensor[active_strings_mask].tolist()
                
                result['points_per_string_list'] = points_per_string_list
            else:
                # Calculate points per string from string_indices if available
                if 'string_indices' in result:
                    string_indices = result['string_indices']
                    default_points_per_string = [0] * self.n_strings
                    for idx in string_indices:
                        if 0 <= idx < self.n_strings:  # Validate index
                            default_points_per_string[idx] += 1
                    
                    result['points_per_string_list'] = default_points_per_string
            
            # Get the points
            if 'points_3d' in initial_geometry:
                points = initial_geometry['points_3d']
                if not isinstance(points, torch.Tensor):
                    points = torch.tensor(points, device=self.device, dtype=torch.float32)
                elif points.device != self.device:
                    points = points.to(self.device)
                
                # Points filtering should already be handled above when filtering z_values/string_indices
                # If we have string filtering, points should already be filtered
                result['points_3d'] = points
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
                    
                    result['points_3d'] = points_3d
            
            # Final check to ensure all necessary components are available
            # If we still don't have points, but have other components, construct them now
            if 'points_3d' not in result and 'string_xy' in result and 'z_values' in result and 'string_indices' in result:
                n_points = len(result['z_values'])
                points_3d = torch.zeros(n_points, 3, device=self.device)
                
                for i, (s_idx, z_val) in enumerate(zip(result['string_indices'], result['z_values'])):
                    if 0 <= s_idx < self.n_strings:  # Validate index
                        points_3d[i, 0] = result['string_xy'][s_idx, 0]  # x value
                        points_3d[i, 1] = result['string_xy'][s_idx, 1]  # y value
                        points_3d[i, 2] = z_val  # z value
                
                result['points_3d'] = points_3d
            
            # Return the initialized geometry dict
            return self.update_points(**result)
            
        # Regular initialization if no initial geometry is provided
        # Initialize string_xy (hex grid or random based on self.optimize_xy)
        string_xy = self.hex_grid.clone()
        if self.random_xy:
            string_xy = torch.rand(self.n_strings, 2, device=self.device) * self.domain_size - self.half_domain

        points_3d = torch.zeros(self.total_points, 3, device=self.device)
        z_values_final = None
        string_indices_final = []
        points_per_string_list_final = None

        points_per_string_counts = [self.total_points // self.n_strings] * self.n_strings
        remainder = self.total_points % self.n_strings
        for i in range(remainder):
            points_per_string_counts[i] += 1

        _z_values_list = []
        current_idx = 0
        for s in range(self.n_strings):
            n_pts = points_per_string_counts[s]
            if n_pts > 0:
                string_z_segment = self._make_z_segment(n_pts)
                points_3d[current_idx:current_idx+n_pts, 0] = string_xy[s, 0]
                points_3d[current_idx:current_idx+n_pts, 1] = string_xy[s, 1]
                points_3d[current_idx:current_idx+n_pts, 2] = string_z_segment
                _z_values_list.append(string_z_segment)
                string_indices_final.extend([s] * n_pts)
                current_idx += n_pts
        z_values_final = torch.cat(_z_values_list) if _z_values_list else torch.tensor([], device=self.device)
        points_per_string_list_final = torch.tensor(points_per_string_counts, dtype=torch.float32, device=self.device)

        # Ensure points_3d is correctly filled if total_points was not perfectly met by allocation
        # This is more relevant for hard allocation if current_idx < self.total_points
        # For soft allocation, points_3d is always (self.total_points, 3)

        return {
            "points_3d": points_3d, 
            "z_values": z_values_final, 
            "string_xy": string_xy,
            "string_indices": string_indices_final, 
            "points_per_string_list": points_per_string_list_final,
        }
    
     
    def _point_to_string_index(self, points_per_string_list):
        """Per-point -> string index map (n_points,), cached across steps.

        The counts in points_per_string_list are structural metadata that do not
        change during optimization (only string_xy / z_values *values* change),
        so the expansion index is computed once and reused. Crucially this avoids
        a per-string ``int(points_per_string_list[s])`` in the hot update_points
        loop: when points_per_string_list is a CUDA tensor (as produced by
        initialize_points), each such int() forces a device->host sync, i.e.
        n_strings synchronizations *per optimization step* -- the main reason
        DynamicString ran slower than EvanescentString (whose counts are a plain
        Python list) on GPU.
        """
        # Fast path: reuse the cached index without ANY device->host transfer.
        # The counts are structural (fixed across optimization steps), so we key
        # the cache on the object's identity + version, avoiding a .tolist()
        # (itself a sync on CUDA) on every step. Only a genuine change in the
        # counts object triggers a recompute.
        if isinstance(points_per_string_list, torch.Tensor):
            key = (id(points_per_string_list), points_per_string_list._version)
        else:
            key = tuple(int(c) for c in points_per_string_list)  # small, CPU-only

        cache = getattr(self, '_p2s_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]

        # Cache miss: read the counts to the CPU once (single .tolist(), not one
        # int() per element) and build the per-point string index.
        if isinstance(points_per_string_list, torch.Tensor):
            counts = [int(c) for c in points_per_string_list.detach().cpu().tolist()]
        else:
            counts = [int(c) for c in points_per_string_list]

        idx = torch.repeat_interleave(
            torch.arange(len(counts), device=self.device),
            torch.tensor(counts, device=self.device, dtype=torch.long),
        )  # (n_points,) — string index for each point, in point order
        self._p2s_cache = (key, idx)
        return idx

    def update_points(self, z_values, string_xy, points_per_string_list, string_indices, **kwargs):
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

        returns:
        --------
        dict
            Dictionary with updated tensors
            """

        # Build points_3d from the (optimizable) string_xy and z_values in a
        # single vectorized, gradient-preserving, sync-free op. Each point's XY
        # is gathered from its string via a precomputed per-point string index;
        # z comes straight from z_values. This replaces the old per-string Python
        # loop (which called int(points_per_string_list[s]) every string, forcing
        # a device sync per string per step on GPU) -- see _point_to_string_index.
        if z_values.shape[0] > 0:
            p2s = self._point_to_string_index(points_per_string_list)  # (n_points,)
            xy = string_xy[p2s]                                        # (n_points, 2), differentiable gather
            points_3d = torch.cat([xy, z_values.reshape(-1, 1)], dim=1)  # (n_points, 3)
        else:
            # Fallback if no points are assigned
            points_3d = torch.zeros(0, 3, device=self.device)

        return {
            "points_3d": points_3d, 
            "z_values": z_values, 
            "string_xy": string_xy,
            "string_indices": string_indices, 
            "points_per_string_list": points_per_string_list,
        }