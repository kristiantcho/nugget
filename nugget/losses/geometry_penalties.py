import torch
from torch.nn import functional as F
from nugget.losses.base_loss import LossFunction

class BoundaryPenalty(LossFunction):
    """Loss function for boundary penalties."""
    def __init__(self, device=None):
        """
        Initialize the boundary penalties loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        domain_size : float
            Size of the domain.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute boundary penalty to keep points in domain.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The boundary penalty value (weighted).
        """
        points_3d = geom_dict.get('points_3d', None)
        domain_size = kwargs.get('boundary_range', 2.0)
        return {'boundary_penalty': torch.mean(torch.clamp(torch.abs(points_3d) - domain_size/2, min=0.0) ** 2)}
    
class StringBoundaryPenaltySquare(LossFunction):
    """Loss function for square boundary penalties."""
    def __init__(self, device=None):
        """
        Initialize the square boundary penalties loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        domain_size : float
            Size of the domain.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute boundary penalty to keep points in domain.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The boundary penalty value (weighted).
        """
        string_xy = geom_dict.get('string_xy', None)
        domain_size = kwargs.get('boundary_range', 2.0)
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else 1.0
        clamped_string_xy = torch.clamp(torch.abs(string_xy) - domain_size/2, min=0.0)** 2
        clamped_string_xy = torch.sqrt(torch.sum(clamped_string_xy, dim=1))
        return {'string_boundary_penalty': torch.mean(clamped_string_xy * string_probs)}
    

class StringBoundaryPenaltyCircle(LossFunction):
    """Loss function for circle boundary penalties."""
    def __init__(self, device=None):
        """
        Initialize the circle boundary penalties loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        domain_size : float
            Size of the domain.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute boundary penalty to keep points in domain.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The boundary penalty value (weighted).
        """
        string_xy = geom_dict.get('string_xy', None)
        domain_size = kwargs.get('boundary_range', 2.0)
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else 1.0
        boundary_sharpness = kwargs.get('boundary_sharpness', 10.0)
        clamped_string_xy = torch.sigmoid(boundary_sharpness * (torch.sqrt(string_xy[:,0] ** 2 + string_xy[:,1] ** 2) - domain_size/2))
        # clamped_string_xy = torch.sqrt(torch.sum(clamped_string_xy))
        return {'string_boundary_penalty': torch.mean(clamped_string_xy * string_probs)}

class RepulsionPenalty(LossFunction):
    """Loss function for repulsion penalties to keep points apart."""
    def __init__(self, device=None):
        """
        Initialize the repulsion penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute repulsion penalty to keep points apart.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'points_3d' key.
        **kwargs
            Additional keyword arguments including 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The 3D repulsion penalty value (weighted).
        """
        points_3d = geom_dict.get('points_3d', None)
        min_dist = kwargs.get('min_dist', 1e-3)
        
        repulsion = 0.0
        total_points = len(points_3d)
        for k in range(total_points):
            for j in range(k + 1, total_points):
                # Use distance in path space
                dist_sq = torch.sum((points_3d[k] - points_3d[j]) ** 2)
                repulsion += 1.0 / (dist_sq + min_dist)
        
        return {'repulsion_penalty': repulsion}


class LocalRepulsionPenalty(LossFunction):
    """Loss function for local repulsion penalties to keep points apart within a local radius."""
    def __init__(self, device=None):
        """
        Initialize the local repulsion penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute repulsion penalty to keep points apart, but only for pairs within a given radius.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'points_3d' key.
        **kwargs
            Additional keyword arguments including 'max_radius' and 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The local repulsion penalty value (weighted).
        """
        points_3d = geom_dict.get('points_3d', None)
        max_radius = kwargs.get('max_radius', 0.1)
        min_dist = kwargs.get('min_dist', 1e-3)
        ignore_border = kwargs.get('ignore_border', False)
        
        if points_3d is None:
            return {'local_repulsion_penalty': torch.tensor(0.0)}
            
        n = len(points_3d)
        if n == 0:
            return {'local_repulsion_penalty': torch.tensor(0.0)}

        sharpness = kwargs.get('local_sharpness', 10.0)  # Controls steepness of sigmoid transition
        
      
        # Compute pairwise squared distances
        diff = points_3d.unsqueeze(1) - points_3d.unsqueeze(0)  # (n, n, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        dist = torch.sqrt(dist_sq)  
        # print(f"Min distance between points: {torch.min(dist[dist>0])}")
        
        # Soft mask using sigmoid - smoother transition at radius boundary
        self_mask = torch.eye(n, dtype=torch.bool, device=points_3d.device)
        radius_weight = torch.sigmoid((max_radius - dist) * sharpness)  # Sharp transition around max_radius
        # radius_mask  = (dist < max_radius) & (~self_mask)
        # radius_weight = radius_mask.int()
        radius_weight = radius_weight * (~self_mask).float()  # Zero out self-pairs
        # print(f"total radius weight: {torch.sum(radius_weight)}")
        repulsion_matrix = radius_weight / (dist_sq + min_dist)
        # print(f"Max repulsion value: {torch.max(repulsion_matrix)}")
        repulsion = torch.sum(repulsion_matrix) / n
        
        return {'local_repulsion_penalty': repulsion}


class StringRepulsionPenalty(LossFunction):
    """Loss function for string repulsion penalties."""
    def __init__(self, device=None):
        """
        Initialize the string repulsion penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute repulsion penalty between strings.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_xy' and optional 'string_weights' keys.
        **kwargs
            Additional keyword arguments including 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The local string repulsion penalty value (weighted).
        """
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        min_dist = kwargs.get('min_dist', 1e-3)
        ignore_border = kwargs.get('ignore_border', False)
        
        if string_xy is None:
            return torch.tensor(0.0)
        n = string_xy.shape[0]
        # Compute pairwise squared distances
        diff = string_xy.unsqueeze(1) - string_xy.unsqueeze(0)  # (n, n, 2)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        # Mask: ignore self-pairs
        mask = (dist_sq > 0)
        repulsion = 0.0
        if string_weights is not None:
            string_probs = torch.sigmoid(string_weights)
            # Outer product for all pairs
            weight_matrix = string_probs.unsqueeze(1) * string_probs.unsqueeze(0)  # (n, n)
            repulsion_matrix = torch.zeros_like(dist_sq)
            repulsion_matrix[mask] = weight_matrix[mask] / (dist_sq[mask] + min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        else:
            repulsion_matrix = torch.zeros_like(dist_sq)
            repulsion_matrix[mask] = 1.0 / (dist_sq[mask] + min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        return {'string_repulsion_penalty': repulsion}

class LocalStringRepulsionPenalty(LossFunction):
    """Loss function for local string repulsion penalties."""
    def __init__(self, device=None):
        """
        Initialize the local string repulsion penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute repulsion penalty between strings, but only for pairs within a given radius.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_xy' and optional 'string_weights' keys.
        **kwargs
            Additional keyword arguments including 'max_radius' and 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The local string repulsion penalty value (weighted).
        """
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        max_radius = kwargs.get('max_radius', 0.1)
        min_dist = kwargs.get('min_dist', 1e-3)
        sharpness = kwargs.get('local_sharpness', 10.0)  # Controls steepness of sigmoid transition
        ignore_border = kwargs.get('ignore_border', False)
        domain_size = kwargs.get('boundary_range', 2.0)
        
        if string_xy is None:
            return {'local_string_repulsion_penalty': torch.tensor(0.0)}
        n = string_xy.shape[0]
        if n == 0:
            return {'local_string_repulsion_penalty': torch.tensor(0.0)}
        # Compute pairwise squared distances
        diff = string_xy.unsqueeze(1) - string_xy.unsqueeze(0)  # (n, n, 2)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        dist = torch.sqrt(dist_sq + 1e-10)  # Add small epsilon for numerical stability
        
        # Soft mask using sigmoid - smoother transition at radius boundary
        self_mask = torch.eye(n, dtype=torch.bool, device=string_xy.device)
        radius_weight = torch.sigmoid((max_radius - dist) * sharpness)  # Sharp transition around max_radius
        # radius_weight = torch.ones_like(dist)
        # radius_mask = dist < max_radius
        # radius_weight = radius_weight * radius_mask.float()
        radius_weight = radius_weight * (~self_mask).float()  # Zero out self-pairs
        
        if ignore_border:
            clamped_string_xy = torch.clamp(torch.abs(string_xy) - domain_size/2, min=0.0)** 2
            clamped_string_xy = torch.sqrt(torch.sum(clamped_string_xy, dim=1))
            border_mask = (clamped_string_xy < 1e-3).unsqueeze(1) | (clamped_string_xy < 1e-3).unsqueeze(0)
            radius_weight = radius_weight * (~border_mask).float()
        repulsion = 0.0
        repulsion_per_string = torch.zeros(n, device=string_xy.device)
        if string_weights is not None:
            string_probs = torch.sigmoid(string_weights)
            # Outer product for all pairs
            weight_matrix = string_probs.unsqueeze(1) * string_probs.unsqueeze(0)  # (n, n)
            # repulsion_matrix = weight_matrix * radius_weight / (dist_sq + min_dist)
            repulsion_matrix = weight_matrix * radius_weight #* torch.exp(-dist_sq / (max_radius**2 + 1e-10))
            # repulsion = torch.sum(repulsion_matrix) / n
            # num_neighbors_per_string = radius_weight.sum(dim=1)  # Count neighbors for each string
            repulsion_per_string = repulsion_matrix.sum(dim=1)  # Total repulsion for each string
            
            # Avoid division by zero and normalize
            # normalized_repulsion = torch.where(
            #     num_neighbors_per_string > 0,
            #     repulsion_per_string / (num_neighbors_per_string + 1e-10),
            #     torch.zeros_like(repulsion_per_string)
            # )
            # repulsion_per_string = normalized_repulsion
            repulsion = repulsion_per_string.mean() #* string_probs.sum()
        else:
            # Mirror weighted behavior with implicit unit weights.
            # This keeps the same smooth, differentiable dependence on string_xy.
            repulsion_matrix = radius_weight
            # num_neighbors_per_string = radius_weight.sum(dim=1)
            repulsion_per_string = repulsion_matrix.sum(dim=1)
            # normalized_repulsion = torch.where(
            #     num_neighbors_per_string > 0,
            #     repulsion_per_string / (num_neighbors_per_string + 1e-10),
            #     torch.zeros_like(repulsion_per_string)
            # )
            # repulsion_per_string = normalized_repulsion
            repulsion = repulsion_per_string.mean()
        return {
            'local_string_repulsion_penalty': repulsion,
            'local_string_repulsion_penalty_per_string': repulsion_per_string,
        }
    

class PathRepulsionPenalty(LossFunction):
    """Loss function for path repulsion penalties in continuous string path space."""
    def __init__(self, device=None):
        """
        Initialize the path repulsion penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute repulsion penalty to keep points apart in continuous string path space (normalized).
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'path_positions' key.
        **kwargs
            Additional keyword arguments including 'number_of_strings', 'min_dist', and 'domain_size'.
            
        Returns:
        --------
        torch.Tensor
            The path repulsion penalty value (weighted).
        """
        path_positions = geom_dict.get('path_positions', None)
        number_of_strings = kwargs.get('number_of_strings', 1)
        min_dist = kwargs.get('min_dist', 1e-3)
        domain_size = kwargs.get('domain_size', 2)
        
        path_penalty = 0.0
        total_points = len(path_positions)
        path_min_dist = min_dist / (number_of_strings*domain_size)
        for k in range(total_points):
            for j in range(k + 1, total_points):
                # Use distance in path space
                dist_sq = torch.sum((path_positions[k] - path_positions[j]) ** 2)
                path_penalty += 1.0 / (dist_sq + path_min_dist)
                
        return {'path_repulsion_penalty': path_penalty}


class ZDistRepulsionPenalty(LossFunction):
    """Loss function for z distance penalty to keep points apart along the same string."""
    def __init__(self, device=None):
        """
        Initialize the z distance penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute z distance penalty to keep points apart along the same string.
        
        Uses z_values and points_per_string_list if available for efficient computation,
        otherwise falls back to points_3d-based computation.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'z_values', 'points_per_string_list', or 'points_3d' keys.
        **kwargs
            Additional keyword arguments including 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The z distance penalty value (weighted).
        """
        z_values = geom_dict.get('z_values', None)
        points_per_string_list = geom_dict.get('points_per_string_list', None)
        points_3d = geom_dict.get('points_3d', None)
        min_dist = kwargs.get('min_dist', 1e-3)
        
        z_dist_penalty = torch.tensor(0.0, device=self.device)
        
        # Use z_values and points_per_string_list if available (more efficient)
        if z_values is not None and points_per_string_list is not None:
            current_idx = 0
            for string_idx, num_points in enumerate(points_per_string_list):
                if num_points > 1:  # Only compute repulsion if string has multiple points
                    # Get z values for this string
                    string_z_values = z_values[current_idx:current_idx + num_points]
                    
                    # Compute pairwise repulsion within this string
                    for i in range(num_points):
                        for j in range(i + 1, num_points):
                            z_dist_sq = (string_z_values[i] - string_z_values[j]) ** 2
                            z_dist_penalty += 1.0 / (z_dist_sq + min_dist)
                
                current_idx += num_points
        
        # Fallback to points_3d-based computation if z_values/points_per_string_list not available
        elif points_3d is not None:
            total_points = len(points_3d)
            for k in range(total_points):
                for j in range(k + 1, total_points):
                    # Check if points are on the same string (same x,y coordinates)
                    if torch.allclose(points_3d[k][:2], points_3d[j][:2], atol=1e-6):
                        # Compute z distance only
                        z_dist_sq = (points_3d[k][2] - points_3d[j][2]) ** 2
                        z_dist_penalty += 1.0 / (z_dist_sq + min_dist)
        
        return {'z_dist_repulsion_penalty': z_dist_penalty}


class LocalZDistRepulsionPenalty(LossFunction):
    """Loss function for local z distance repulsion penalty to keep points apart along the same string within a local radius."""
    def __init__(self, device=None):
        """
        Initialize the local z distance repulsion penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute repulsion penalty between points on the same string within a given z-distance radius.
        
        Uses z_values and points_per_string_list if available for efficient computation,
        otherwise falls back to points_3d-based computation.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'z_values', 'points_per_string_list', or 'points_3d' keys.
        **kwargs
            Additional keyword arguments including 'max_radius' and 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The local z distance repulsion penalty value (weighted).
        """
        z_values = geom_dict.get('z_values', None)
        points_per_string_list = geom_dict.get('points_per_string_list', None)
        points_3d = geom_dict.get('points_3d', None)
        max_radius = kwargs.get('max_radius', 0.1)
        min_dist = kwargs.get('min_dist', 1e-3)
        sharpness = kwargs.get('local_sharpness', 100.0)  # Controls steepness of sigmoid transition
        repulsion = torch.tensor(0.0, device=self.device)
        total_valid_pairs = 0
        
        # Use z_values and points_per_string_list if available (more efficient)
        if (z_values is not None or points_3d is not None) and points_per_string_list is not None:
            current_idx = 0
            if z_values is None:
                z_values = points_3d[:, 2]
            for string_idx, num_points in enumerate(points_per_string_list):
                if num_points > 1:  # Only compute repulsion if string has multiple points
                    # Get z values for this string
                    string_z_values = z_values[current_idx:current_idx + num_points]
                    z_dist_sq = (string_z_values.unsqueeze(1) - string_z_values.unsqueeze(0)) ** 2
                    z_dist = torch.sqrt(z_dist_sq + 1e-10)  # Add small epsilon for numerical stability
                    self_mask = torch.eye(num_points, dtype=torch.bool, device=z_values.device)
                    radius_weight = torch.sigmoid((max_radius - z_dist) * sharpness)  # Sharp transition around max_radius
                    radius_weight = radius_weight * (~self_mask).float()  # Zero out self-pairs

                    repulsion += torch.sum(radius_weight * (1.0 / (z_dist_sq + min_dist)))
                    total_valid_pairs += torch.sum(radius_weight > 0).item()

                current_idx += num_points
        
        # Normalize by number of valid pairs or total points
        if total_valid_pairs > 0:
            repulsion = repulsion / total_valid_pairs

        
        return {'local_z_dist_repulsion_penalty': repulsion}

class StringWeightsPenalty(LossFunction):
    """Loss function for string weights penalty to balance the amount of active strings."""
    def __init__(self, device=None):
        """
        Initialize the string weights penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute penalty for string weights to balance the amount of active strings.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_weights' key.
        **kwargs
            Additional keyword arguments (currently unused).
            
        Returns:
        --------
        torch.Tensor
            The string weights penalty value (weighted).
        """
        string_weights = geom_dict.get('string_weights', None)
        
        string_probs = torch.sigmoid(string_weights)
        # string_probs = string_weights
        return {'string_weights_penalty': torch.sum(torch.sqrt(string_probs)) / len(string_probs)}

class StringWeightsBoundaryPenalty(LossFunction):
    """Loss function for string weights boundary penalty to keep them within [0,1]."""
    def __init__(self, device=None):
        """
        Initialize the string weights boundary penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute boundary penalty for string weights to keep them within a [0,1].
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_weights' key.
        **kwargs
            Additional keyword arguments (currently unused).
            
        Returns:
        --------
        torch.Tensor
            The string weights boundary penalty value (weighted).
        """
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None
        
        return {'string_weight_boundary_penalty':torch.mean((string_probs - torch.clamp(string_probs, min=0.0, max=0.8)) ** 2)}

class StringNumberPenalty(LossFunction):
    """Loss function for string number penalty to keep the number of active strings balanced."""
    def __init__(self, device=None):
        """
        Initialize the string number penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute penalty for the number of strings to keep the number of active strings balanced.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_weights' key.
        **kwargs
            Additional keyword arguments including 'eva_min_num_strings'.
            
        Returns:
        --------
        torch.Tensor
            The string number penalty value (weighted).
        """
        string_weights = geom_dict.get('string_weights', None)
        eva_min_num_strings = kwargs.get('eva_min_num_strings', 70)
        string_number_beta = kwargs.get('string_number_beta', 1.0)
        
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None
        # string_probs = string_weights
        if string_probs is not None:
            return {'string_number_penalty': F.softplus(torch.sum(string_probs) - eva_min_num_strings, beta=string_number_beta)/string_probs.shape[0]}
        else:
            return {'string_number_penalty': torch.tensor(0.0)}

class WeightBinarizationPenalty(LossFunction):
    """Loss function for weight binarization penalty to encourage binarization."""
    def __init__(self, device=None):
        """
        Initialize the weight binarization penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute penalty for string weights to encourage binarization.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_weights' key.
        **kwargs
            Additional keyword arguments (currently unused).
            
        Returns:
        --------
        torch.Tensor
            The binarization penalty value (weighted).
        """
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None
        # string_probs_cut = torch.clamp(string_probs, min=0.0, max=1.0)
        
        if string_probs is not None:
            return {'weight_binarization_penalty': torch.sum(-string_probs * torch.log(string_probs + 1e-6) - (1 - string_probs) * torch.log(1 - string_probs + 1e-6))/ len(string_probs)}
        else:
            return {'weight_binarization_penalty': torch.tensor(0.0)}


class ROVPenalty(LossFunction):
    """Loss function for ROV penalty to maintain ROV capability for each string."""
    def __init__(self, device=None, rov_rec_width=0.3, rov_height=0.16, rov_tri_length=0.08):
        """
        Initialize the string number penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)
        self.rov_rec_width = rov_rec_width
        self.rov_height = rov_height
        self.rov_tri_length = rov_tri_length
        
    def _compute_blockage_per_angle_alt(self, all_relative, num_angles, half_height, tri_length, rec_width, other_probs=None):
        """Exact hard-mask triangle+rectangle blockage via interval accumulation.

        This mode matches the non-alt hard geometry rules while avoiding the large
        (N, N-1, num_angles) tensor. Each blocker contributes a small set of angular
        intervals that are accumulated with a per-string difference array.
        """
        device = all_relative.device
        dtype = all_relative.dtype
        N = all_relative.shape[0]

        two_pi = 2.0 * torch.pi
        angle_step = two_pi / float(num_angles)
        blockage_per_angle = torch.zeros(N, num_angles, device=device, dtype=dtype)

        L_tri = float(tri_length)
        L_tot = float(tri_length + rec_width)
        beta = float(torch.atan(torch.tensor(half_height / max(L_tri, 1e-12))))
        eps = 1e-12

        # If blocker distance is larger than this, neither triangle nor rectangle can include it.
        max_dist = (L_tot ** 2 + half_height ** 2) ** 0.5

        def _add_theta_intervals(diff, start, end, w):
            """Add inclusive angular intervals [start, end] on periodic [0, 2pi)."""
            if start.numel() == 0:
                return

            start = torch.remainder(start, two_pi)
            end = torch.remainder(end, two_pi)
            wrap = start > end

            start_idx = torch.ceil((start - eps) / angle_step).to(torch.long)
            end_idx = torch.floor((end + eps) / angle_step).to(torch.long)
            start_idx = torch.clamp(start_idx, 0, num_angles - 1)
            end_idx = torch.clamp(end_idx, 0, num_angles - 1)

            non_wrap = ~wrap
            if torch.any(non_wrap):
                s_nw = start_idx[non_wrap]
                e_nw = end_idx[non_wrap]
                w_nw = w[non_wrap]
                keep = s_nw <= e_nw
                if torch.any(keep):
                    s_nw = s_nw[keep]
                    e_nw = e_nw[keep]
                    w_nw = w_nw[keep]
                    diff.index_add_(0, s_nw, w_nw)
                    diff.index_add_(0, e_nw + 1, -w_nw)

            if torch.any(wrap):
                s_w = start_idx[wrap]
                e_w = end_idx[wrap]
                w_w = w[wrap]

                keep_head = e_w >= 0
                if torch.any(keep_head):
                    e_h = e_w[keep_head]
                    w_h = w_w[keep_head]
                    diff[0] += torch.sum(w_h)
                    diff.index_add_(0, e_h + 1, -w_h)

                keep_tail = s_w <= (num_angles - 1)
                if torch.any(keep_tail):
                    s_t = s_w[keep_tail]
                    w_t = w_w[keep_tail]
                    diff.index_add_(0, s_t, w_t)
                    diff[num_angles] -= torch.sum(w_t)

        for i in range(N):
            rel_i = all_relative[i]  # (N-1, 2)
            dx = rel_i[:, 0]
            dy = rel_i[:, 1]

            dist = torch.sqrt(dx * dx + dy * dy + 1e-12)
            valid = (dist > 1e-8) & (dist <= max_dist)
            if not torch.any(valid):
                continue

            phi = torch.atan2(dy, dx)
            phi = torch.remainder(phi, two_pi)

            if other_probs is not None:
                pair_weight = other_probs[i]
            else:
                pair_weight = torch.ones_like(dist)

            phi = phi[valid]
            dist = dist[valid]
            pair_weight = pair_weight[valid]

            diff = torch.zeros(num_angles + 1, device=device, dtype=dtype)

            # Build |delta| intervals for triangle and rectangle pieces, then
            # combine with inclusion-exclusion (tri + rect - overlap) so each
            # blocker contributes exactly once where shapes overlap.

            # Triangle piece:
            # 0 <= x <= L_tri and |y| <= (half_height/L_tri) * x.
            tri_low = torch.where(
                dist > L_tri,
                torch.acos(torch.clamp(L_tri / dist, max=1.0)),
                torch.zeros_like(dist),
            )
            tri_high = torch.full_like(dist, beta)
            tri_valid = tri_low <= (tri_high + eps)

            # Rectangle piece:
            # L_tri <= x <= L_tot and |y| <= half_height.
            rect_high = torch.minimum(
                torch.asin(torch.clamp(half_height / dist, max=1.0)),
                torch.acos(torch.clamp(L_tri / dist, max=1.0)),
            )
            rect_low = torch.where(
                dist > L_tot,
                torch.acos(torch.clamp(L_tot / dist, max=1.0)),
                torch.zeros_like(dist),
            )
            rect_valid = (dist >= L_tri) & (rect_low <= (rect_high + eps))

            # Intersection on |delta| side for inclusion-exclusion.
            inter_low = torch.maximum(tri_low, rect_low)
            inter_high = torch.minimum(tri_high, rect_high)
            inter_valid = tri_valid & rect_valid & (inter_low <= (inter_high + eps))

            for lo, hi, is_valid, sign in (
                (tri_low, tri_high, tri_valid, 1.0),
                (rect_low, rect_high, rect_valid, 1.0),
                (inter_low, inter_high, inter_valid, -1.0),
            ):
                if not torch.any(is_valid):
                    continue

                lo_v = lo[is_valid]
                hi_v = hi[is_valid]
                phi_v = phi[is_valid]
                w_v = pair_weight[is_valid] * sign

                # Non-alt path uses y_rot = dx*sin(theta) + dy*cos(theta),
                # so blocked intervals are centered at theta = -phi.
                # delta in [lo, hi] => theta in [lo-phi, hi-phi]
                _add_theta_intervals(diff, lo_v - phi_v, hi_v - phi_v, w_v)
                # delta in [-hi, -lo] => theta in [-hi-phi, -lo-phi]
                _add_theta_intervals(diff, -hi_v - phi_v, -lo_v - phi_v, w_v)

            blockage_per_angle[i] = torch.cumsum(diff, dim=0)[:num_angles]
            blockage_per_angle[i].clamp_(min=0.0)

        return blockage_per_angle

   

    def __call__(self, geom_dict, **kwargs):
        """
        points: (N, 2) tensor of 2D points
        Returns: scalar penalty loss
        """
        points = geom_dict.get('string_xy', None)
        num_angles = kwargs.get('num_angles', 6)
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None

        # Backward-compatible options (defaults preserve old behavior):
        # - soft_inside=False: use hard boolean masks for corridor membership (non-differentiable w.r.t. positions)
        # - angle_softmin_tau=0.0: use hard min over angles (non-differentiable at argmin switches)
        # - detach_other_probs=True: do not backprop into blocking strings' weights
        soft_inside = kwargs.get('rov_soft_inside', False)
        inside_sharpness = float(kwargs.get('rov_inside_sharpness', 5.0))
        angle_softmin_tau = float(kwargs.get('rov_angle_softmin_tau', 0.0))
        detach_other_probs = kwargs.get('detach_other_probs', True)
        alt_mode = bool(kwargs.get('rov_alt_mode', False))

        N = points.shape[0]
        
        # Vectorized computation
        # Compute all relative positions at once
        mask = ~torch.eye(N, dtype=torch.bool, device=points.device)
        all_relative = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, 2)
        all_relative = all_relative[mask].reshape(N, N-1, 2)  # (N, N-1, 2)
        
        # Prepare angles (used by both branches for reporting least-blocked heading).
        angles = torch.linspace(
            0,
            2 * torch.pi * (num_angles - 1) / num_angles,
            num_angles,
            device=points.device,
        )
        
        # Geometry checks
        # Intended shape: a triangular "nose" starting at the string that widens
        # to the corridor width, followed by a rectangular corridor.
        # In rotated coordinates (x_rot forward, |y| sideways):
        # - Triangle:   0 <= x <= L_tri and |y| <= (W/2) * (x/L_tri)
        # - Rectangle:  L_tri <= x <= L_tri + L_rect and |y| <= W/2
        L_rect = float(self.rov_rec_width)
        W_rect = float(self.rov_height)
        L_tri = float(self.rov_tri_length)
        half_height = W_rect / 2.0
        slope = half_height / max(L_tri, 1e-12)

        if alt_mode and (not soft_inside):
            if string_probs is not None:
                other_probs = string_probs.unsqueeze(0).expand(N, N)[mask].reshape(N, N-1)
                if detach_other_probs:
                    other_probs = other_probs.detach()
            else:
                other_probs = None

            # Exact alternative: interval accumulation over heading bins.
            blockage_per_angle = self._compute_blockage_per_angle_alt(
                all_relative=all_relative,
                num_angles=num_angles,
                half_height=half_height,
                tri_length=L_tri,
                rec_width=L_rect,
                other_probs=other_probs,
            )
            angle_scores_per_angle = blockage_per_angle

            if angle_softmin_tau > 0.0:
                penalty_per_string = -angle_softmin_tau * torch.logsumexp(
                    -blockage_per_angle / angle_softmin_tau, dim=1
                )
            else:
                penalty_per_string = blockage_per_angle.min(dim=1)[0]

            if string_probs is not None:
                loss = (penalty_per_string * string_probs).sum()
            else:
                loss = penalty_per_string.sum()

        else:
            c = torch.cos(angles)
            s = torch.sin(angles)

            # Apply rotation for all angles at once: (N, N-1, num_angles, 2)
            rel_expanded = all_relative.unsqueeze(2)  # (N, N-1, 1, 2)
            x_rot = rel_expanded[..., 0] * c - rel_expanded[..., 1] * s
            y_rot_abs = (rel_expanded[..., 0] * s + rel_expanded[..., 1] * c).abs()

            if not soft_inside:
                inside_tri = (x_rot >= 0) & (x_rot <= L_tri) & (y_rot_abs <= slope * x_rot)
                inside_rect = (x_rot >= L_tri) & (x_rot <= L_tri + L_rect) & (y_rot_abs <= half_height)
                inside = (inside_rect | inside_tri).float()  # (N, N-1, num_angles)
            else:
                k = inside_sharpness

                def _soft_between(x, lo, hi):
                    # ~1 if lo<=x<=hi, ~0 otherwise
                    return torch.sigmoid(k * (x - lo)) * torch.sigmoid(k * (hi - x))

                # Triangle: 0<=x<=L_tri and |y| <= slope*x
                tri_x = _soft_between(x_rot, 0.0, L_tri)
                tri_y_bound = slope * x_rot
                tri_y = torch.sigmoid(k * (tri_y_bound - y_rot_abs))
                inside_tri = tri_x * tri_y

                # Rectangle: L_tri<=x<=L_tri+L_rect and |y|<=W_rect/2
                rect_x = _soft_between(x_rot, L_tri, L_tri + L_rect)
                rect_y = torch.sigmoid(k * (half_height - y_rot_abs))
                inside_rect = rect_x * rect_y

                # Soft OR (union)
                inside = 1.0 - (1.0 - inside_rect) * (1.0 - inside_tri)  # (N, N-1, num_angles)

            if string_probs is not None:
                # Get weights of other strings for each position
                other_probs = string_probs.unsqueeze(0).expand(N, N)[mask].reshape(N, N-1)
                if detach_other_probs:
                    other_probs = other_probs.detach()

                # For each angle, sum weights of strings inside safe space
                # (N, num_angles) = sum over other strings
                blockage_per_angle = (inside.float() * other_probs.unsqueeze(-1)).sum(dim=1)
                angle_scores_per_angle = blockage_per_angle

                # For each string: minimum blockage across all angles (optionally smooth)
                if angle_softmin_tau > 0.0:
                    penalty_per_string = -angle_softmin_tau * torch.logsumexp(
                        -blockage_per_angle / angle_softmin_tau, dim=1
                    )
                else:
                    penalty_per_string = blockage_per_angle.min(dim=1)[0]  # (N,)

                # Weight by string probability and sum
                loss = (penalty_per_string * string_probs).sum()
            else:
                # Count number of strings inside for each angle
                # (N, num_angles) = count of other strings inside
                count_per_angle = inside.sum(dim=1)  # (N, num_angles)
                angle_scores_per_angle = count_per_angle

                # For each string: minimum count across all angles (optionally smooth)
                if angle_softmin_tau > 0.0:
                    penalty_per_string = -angle_softmin_tau * torch.logsumexp(
                        -count_per_angle / angle_softmin_tau, dim=1
                    )
                else:
                    penalty_per_string = count_per_angle.min(dim=1)[0]  # (N,)

                loss = penalty_per_string.sum()

        # Reporting: least-blocked angle per string (hard argmin), regardless of
        # whether a softmin was used for the penalty aggregation.
        # Use a small tolerance when selecting the least-blocked angle so
        # numerically equivalent boundary bins resolve to the first bin
        # consistently across alt and non-alt implementations.
        min_scores = angle_scores_per_angle.min(dim=1, keepdim=True)[0]
        tie_tol = 1e-6
        near_min_mask = angle_scores_per_angle <= (min_scores + tie_tol)
        least_blocked_angle_idx_per_string = near_min_mask.to(torch.int64).argmax(dim=1)  # (N,)
        least_blocked_angle_per_string = angles[least_blocked_angle_idx_per_string]  # (N,)
        # least_blocked_angle_deg_per_string = least_blocked_angle_per_string * (180.0 / torch.pi)
        penalty_per_string = torch.clamp(penalty_per_string, min=0.0)  # Ensure non-negative for reporting
        return {
            'rov_penalty': loss / N,
            'rov_penalty_per_string': penalty_per_string / N,
            'rov_least_blocked_angle_per_string': least_blocked_angle_per_string,
        }

            
        