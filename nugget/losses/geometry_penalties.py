import math

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

try:
    from geomloss import SamplesLoss
except ImportError:
    SamplesLoss = None

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
        
        use_binarization_weight = kwargs.get('string_number_use_binarization_weight', False)

        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None
        if string_probs is not None:
            if use_binarization_weight:
                # Per-string binary entropy normalised to [0, 1]; invert so weight
                # is 1 for fully binarized strings (p near 0 or 1) and 0 for
                # maximally ambiguous strings (p near 0.5).
                h = -string_probs * torch.log(string_probs + 1e-6) - (1 - string_probs) * torch.log(1 - string_probs + 1e-6)
                binarization_weight = 1.0 - h / math.log(2)  # in [0, 1], 1 at extremes
                effective_probs = string_probs * binarization_weight
            else:
                effective_probs = string_probs
            return {'string_number_penalty': F.softplus(torch.sum(effective_probs) - eva_min_num_strings, beta=string_number_beta) / string_probs.shape[0]}
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
            return {'weight_binarization_penalty': torch.sum(-string_probs * torch.log(string_probs + 1e-6) - (1 - string_probs) * torch.log(1 - string_probs + 1e-6)) / len(string_probs)}
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
        """Vectorized hard-mask triangle+rectangle blockage via flat diff-array scatter.

        Eliminates the Python for-loop over N strings by flattening all N*(N-1) pairs,
        computing geometry quantities in bulk, and scattering into a flat diff array of
        length N*(num_angles+1) before a single row-wise cumsum.
        """
        device = all_relative.device
        dtype = all_relative.dtype
        N = all_relative.shape[0]
        P = N * (N - 1)
        A1 = num_angles + 1  # diff array has one guard slot per row

        two_pi = 2.0 * torch.pi
        angle_step = two_pi / float(num_angles)

        L_tri = float(tri_length)
        L_tot = float(tri_length + rec_width)
        beta = float(torch.atan(torch.tensor(half_height / max(L_tri, 1e-12))))
        eps = 1e-12

        max_dist = (L_tot ** 2 + half_height ** 2) ** 0.5

        # --- Step 1: flatten all pairs ---
        rel_flat = all_relative.reshape(P, 2)                                   # (P, 2)
        row_idx = torch.arange(N, device=device).repeat_interleave(N - 1)      # (P,)

        if other_probs is not None:
            pair_weight = other_probs.reshape(P)                                # (P,)
        else:
            pair_weight = torch.ones(P, device=device, dtype=dtype)             # (P,)

        # --- Step 2: geometric quantities, filter by valid distance ---
        dx = rel_flat[:, 0]
        dy = rel_flat[:, 1]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-12)                           # (P,)
        valid = (dist > 1e-8) & (dist <= max_dist)                              # (P,)

        dist_v = dist[valid]                                                    # (V,)
        phi_v  = torch.remainder(torch.atan2(dy[valid], dx[valid]), two_pi)    # (V,)
        pw_v   = pair_weight[valid]                                             # (V,)
        row_v  = row_idx[valid]                                                 # (V,)

        # --- Step 3: angular interval bounds (vectorized over V) ---
        tri_low = torch.where(
            dist_v > L_tri,
            torch.acos(torch.clamp(L_tri / dist_v, max=1.0)),
            torch.zeros_like(dist_v),
        )                                                                       # (V,)
        tri_high = torch.full_like(dist_v, beta)                               # (V,)
        tri_ok   = tri_low <= (tri_high + eps)                                 # (V,)

        rect_high = torch.minimum(
            torch.asin(torch.clamp(half_height / dist_v, max=1.0)),
            torch.acos(torch.clamp(L_tri / dist_v, max=1.0)),
        )                                                                       # (V,)
        rect_low = torch.where(
            dist_v > L_tot,
            torch.acos(torch.clamp(L_tot / dist_v, max=1.0)),
            torch.zeros_like(dist_v),
        )                                                                       # (V,)
        rect_ok = (dist_v >= L_tri) & (rect_low <= (rect_high + eps))         # (V,)

        inter_low  = torch.maximum(tri_low,  rect_low)                        # (V,)
        inter_high = torch.minimum(tri_high, rect_high)                       # (V,)
        inter_ok   = tri_ok & rect_ok & (inter_low <= (inter_high + eps))     # (V,)

        # --- Step 4: flat diff array ---
        diff_flat = torch.zeros(N * A1, device=device, dtype=dtype)            # (N*A1,)

        def _add_intervals_flat(start_raw, end_raw, w, row_ids):
            """Scatter signed angular intervals into diff_flat (no Python loops over pairs)."""
            start = torch.remainder(start_raw, two_pi)
            end   = torch.remainder(end_raw,   two_pi)

            start_idx = torch.clamp(
                torch.ceil((start - eps) / angle_step).to(torch.long), 0, num_angles - 1)
            end_idx = torch.clamp(
                torch.floor((end + eps) / angle_step).to(torch.long), 0, num_angles - 1)

            wrap     = start > end
            non_wrap = ~wrap

            # Non-wrapping: valid where start_idx <= end_idx
            nw = non_wrap & (start_idx <= end_idx)
            if nw.any():
                flat_s = row_ids[nw] * A1 + start_idx[nw]
                flat_e = row_ids[nw] * A1 + end_idx[nw] + 1
                diff_flat.index_add_(0, flat_s,  w[nw])
                diff_flat.index_add_(0, flat_e, -w[nw])

            # Wrapping: head piece [0, end_idx] + tail piece [start_idx, num_angles-1]
            if wrap.any():
                # Head: +w at col 0, -w at end_idx+1
                flat_head_s = row_ids[wrap] * A1
                flat_head_e = row_ids[wrap] * A1 + end_idx[wrap] + 1
                diff_flat.index_add_(0, flat_head_s,  w[wrap])
                diff_flat.index_add_(0, flat_head_e, -w[wrap])
                # Tail: +w at start_idx, -w at guard slot (col num_angles)
                flat_tail_s = row_ids[wrap] * A1 + start_idx[wrap]
                flat_tail_e = row_ids[wrap] * A1 + num_angles
                diff_flat.index_add_(0, flat_tail_s,  w[wrap])
                diff_flat.index_add_(0, flat_tail_e, -w[wrap])

        # --- Step 5: scatter all 6 interval sets (3 shapes × 2 symmetric halves) ---
        for lo_all, hi_all, ok_all, sign in (
            (tri_low,   tri_high,   tri_ok,   1.0),
            (rect_low,  rect_high,  rect_ok,  1.0),
            (inter_low, inter_high, inter_ok, -1.0),
        ):
            if not ok_all.any():
                continue
            lo_k  = lo_all[ok_all]
            hi_k  = hi_all[ok_all]
            phi_k = phi_v[ok_all]
            w_k   = pw_v[ok_all] * sign
            row_k = row_v[ok_all]

            # Positive half: delta in [lo, hi] => theta in [lo-phi, hi-phi]
            _add_intervals_flat(lo_k - phi_k, hi_k - phi_k,   w_k, row_k)
            # Negative half: delta in [-hi, -lo] => theta in [-hi-phi, -lo-phi]
            _add_intervals_flat(-hi_k - phi_k, -lo_k - phi_k, w_k, row_k)

        # --- Step 6: cumsum, slice, clamp ---
        diff_2d = diff_flat.reshape(N, A1)
        blockage_per_angle = torch.cumsum(diff_2d, dim=1)[:, :num_angles]     # (N, num_angles)
        blockage_per_angle = torch.clamp(blockage_per_angle, min=0.0)

        return blockage_per_angle

    def _compute_blockage_per_angle_default(
        self, all_relative, angles, half_height, tri_length, rec_width,
        soft_inside=False, inside_sharpness=5.0, other_probs=None,
        use_chunked=False, chunk_size=16,
    ):
        """Vectorized triangle+rectangle blockage via per-angle rotation.

        Pure-tensor core (no kwargs/dict access) so it can be wrapped with
        torch.compile independently of __call__'s dynamic argument parsing.
        Mirrors _compute_blockage_per_angle_alt's role for the non-alt path.
        """
        L_tri = tri_length
        L_rect = rec_width
        slope = half_height / max(L_tri, 1e-12)

        c = torch.cos(angles)
        s = torch.sin(angles)

        rel_expanded = all_relative.unsqueeze(2)  # (N, N-1, 1, 2)

        num_angles = angles.shape[0]

        if use_chunked:
            k = inside_sharpness

            def _soft_between(x, lo, hi):
                return torch.sigmoid(k * (x - lo)) * torch.sigmoid(k * (hi - x))

            blockage_per_angle = torch.zeros(
                all_relative.shape[0], num_angles, device=all_relative.device, dtype=all_relative.dtype)
            for a_start in range(0, num_angles, chunk_size):
                a_end = min(a_start + chunk_size, num_angles)
                c_ch = c[a_start:a_end]
                s_ch = s[a_start:a_end]

                x_rot     = rel_expanded[..., 0] * c_ch - rel_expanded[..., 1] * s_ch   # (N, N-1, chunk)
                y_rot_abs = (rel_expanded[..., 0] * s_ch + rel_expanded[..., 1] * c_ch).abs()

                tri_x      = _soft_between(x_rot, 0.0, L_tri)
                tri_y      = torch.sigmoid(k * (slope * x_rot - y_rot_abs))
                inside_tri = tri_x * tri_y

                rect_x      = _soft_between(x_rot, L_tri, L_tri + L_rect)
                rect_y      = torch.sigmoid(k * (half_height - y_rot_abs))
                inside_rect = rect_x * rect_y

                inside_ch = 1.0 - (1.0 - inside_rect) * (1.0 - inside_tri)   # (N, N-1, chunk)

                if other_probs is not None:
                    blockage_per_angle[:, a_start:a_end] = (inside_ch * other_probs.unsqueeze(-1)).sum(dim=1)
                else:
                    blockage_per_angle[:, a_start:a_end] = inside_ch.sum(dim=1)

            return blockage_per_angle

        # Apply rotation for all angles at once: (N, N-1, num_angles)
        x_rot     = rel_expanded[..., 0] * c - rel_expanded[..., 1] * s
        y_rot_abs = (rel_expanded[..., 0] * s + rel_expanded[..., 1] * c).abs()

        if not soft_inside:
            inside_tri  = (x_rot >= 0) & (x_rot <= L_tri) & (y_rot_abs <= slope * x_rot)
            inside_rect = (x_rot >= L_tri) & (x_rot <= L_tri + L_rect) & (y_rot_abs <= half_height)
            inside = (inside_rect | inside_tri).float()
        else:
            k = inside_sharpness

            def _soft_between(x, lo, hi):
                return torch.sigmoid(k * (x - lo)) * torch.sigmoid(k * (hi - x))

            tri_x      = _soft_between(x_rot, 0.0, L_tri)
            tri_y      = torch.sigmoid(k * (slope * x_rot - y_rot_abs))
            inside_tri = tri_x * tri_y

            rect_x      = _soft_between(x_rot, L_tri, L_tri + L_rect)
            rect_y      = torch.sigmoid(k * (half_height - y_rot_abs))
            inside_rect = rect_x * rect_y

            inside = 1.0 - (1.0 - inside_rect) * (1.0 - inside_tri)

        if other_probs is not None:
            blockage_per_angle = (inside.float() * other_probs.unsqueeze(-1)).sum(dim=1)
        else:
            blockage_per_angle = inside.float().sum(dim=1)

        return blockage_per_angle

    def _compute_away_theta(self, all_relative, num_neighbours, soft=True, nn_tau=1.0):
        """Per-string heading pointing away from its `num_neighbours` nearest strings.

        For each string i, take the (soft or hard) nearest-neighbour set among the
        other strings, sum the unit vectors pointing *to* those neighbours, and
        negate: the result points away from the local cluster.

        Parameters
        ----------
        all_relative : (N, N-1, 2) tensor
            all_relative[i, j] = (position of other string j) - (position of string i),
            i.e. the vector from string i toward other string j.
        num_neighbours : int
            Number of nearest neighbours (k). Clamped to available count.
        soft : bool
            If True, neighbour membership is a softmax over -distance/nn_tau (fully
            differentiable). If False, a hard top-k membership mask (0/1) is used.
        nn_tau : float or None
            Softmax temperature for the soft neighbour selection, as a *multiple of the
            geometry's own distance scale* (the median k-th nearest-neighbour distance),
            so it is robust to the absolute units of `points`. Smaller -> sharper top-k.
            If None, defaults to 0.5 * that scale.

        Returns
        -------
        theta_away : (N,) tensor of headings in radians (gradients flow w.r.t. positions).
        away_valid : (N,) bool tensor, False where the away vector is ~0 (undefined).
        """
        n_other = all_relative.shape[1]
        k = int(max(1, min(int(num_neighbours), n_other)))

        dist = torch.sqrt((all_relative ** 2).sum(dim=-1) + 1e-12)  # (N, N-1)
        # Unit vectors from string i toward each other string j.
        unit = all_relative / dist.unsqueeze(-1)  # (N, N-1, 2)

        if soft:
            # Scale the softmax temperature by the geometry's own distance scale (median
            # k-th nearest distance) so the soft top-k spreads over ~k neighbours
            # regardless of the absolute coordinate units.
            kth_dist = torch.topk(dist, k=k, dim=1, largest=False).values[:, -1]  # (N,)
            dist_scale = torch.median(kth_dist).clamp_min(1e-12)
            tau_mult = 0.5 if nn_tau is None else float(nn_tau)
            tau = (tau_mult * dist_scale).clamp_min(1e-12)
            # Soft top-k emphasis: softmax over -distance puts weight on the nearest
            # strings. Scaled by k so the effective membership mass ~ k neighbours.
            nn_weights = torch.softmax(-dist / tau, dim=1) * k  # (N, N-1)
        else:
            # Hard top-k membership mask (1 for the k nearest, else 0).
            nn_weights = torch.zeros_like(dist)
            topk_idx = torch.topk(dist, k=k, dim=1, largest=False).indices  # (N, k)
            nn_weights.scatter_(1, topk_idx, 1.0)

        # Sum unit vectors to neighbours, then negate to point away from them.
        toward_vec = (unit * nn_weights.unsqueeze(-1)).sum(dim=1)  # (N, 2)
        away_vec = -toward_vec  # (N, 2)

        away_norm = torch.linalg.norm(away_vec, dim=1)  # (N,)
        away_valid = away_norm > 1e-8
        # Offset degenerate (~0) rows so atan2 stays finite in fwd/bwd; away_valid
        # zeroes their contribution downstream.
        safe_away_vec = torch.where(
            away_valid.unsqueeze(1),
            away_vec,
            away_vec + torch.tensor([1.0, 0.0], device=all_relative.device, dtype=away_vec.dtype),
        )
        # NOTE: `angles` (and blockage_per_angle's heading grid) use ROVPenalty's own
        # corridor-rotation convention, under which a heading `theta` maps to the WORLD
        # direction (cos(theta), -sin(theta)) -- i.e. angles increase clockwise, not the
        # standard (counter-clockwise) atan2 convention. `away_vec` is an ordinary
        # world-space Cartesian vector, so we must negate its y-component before
        # atan2 to express it in that same clockwise convention; otherwise theta_away
        # would be the mirror image (about the x-axis) of the true away direction,
        # visibly pointing the wrong way once compared against `angles`.
        theta_away = torch.atan2(-safe_away_vec[:, 1], safe_away_vec[:, 0])  # (N,)
        return theta_away, away_valid

    def _select_away_best_angle(
        self,
        blockage_per_angle,
        angles,
        theta_away,
        away_valid,
        *,
        away_weight=0.0,
        soft=True,
        select_tau=0.1,
    ):
        """Choose the "best" ROV heading by trading off blockage against outwardness.

        Selection uses a combined per-angle score

            score[a] = blockage[a] + away_weight * misalign[a]

        where misalign[a] = (1 - cos(angle_a - theta_away)) / 2 in [0, 1] measures how
        much heading `a` points *toward* the nearest neighbours (0 = straight away,
        1 = straight toward). Blockage is the primary term: because misalign is bounded
        by away_weight, a uniquely-clear ("no alternative") corridor keeps the lowest
        score and is still chosen, so it is not punished. The away term only tips the
        choice among headings with comparable blockage.

        The returned penalty is the *blockage* at the chosen heading (NOT the combined
        score), so preferring a more-outward-but-equally-clear angle costs nothing extra.

        Soft mode selects via softmin weights (differentiable) over the combined score;
        hard mode uses a hard argmin. For strings with an undefined outward direction
        (`away_valid` False), the away term is disabled (falls back to pure blockage).

        Returns
        -------
        penalty_per_string : (N,) selected blockage.
        best_angle : (N,) chosen heading in radians.
        """
        misalign = (1.0 - torch.cos(angles.unsqueeze(0) - theta_away.unsqueeze(1))) / 2.0  # (N, A)
        if away_valid is not None:
            # Zero the away influence where the outward direction is undefined.
            misalign = misalign * away_valid.unsqueeze(1).to(misalign.dtype)

        score = blockage_per_angle + away_weight * misalign  # (N, A)

        if soft:
            # Softmin selection weights over the combined score (differentiable).
            w = torch.softmax(-score / max(float(select_tau), 1e-12), dim=1)  # (N, A)
            # Penalty = expected blockage under the selection distribution.
            penalty_per_string = (w * blockage_per_angle).sum(dim=1)  # (N,)
            # Chosen heading = circular mean of angles under the selection weights.
            cos_b = (w * torch.cos(angles).unsqueeze(0)).sum(dim=1)  # (N,)
            sin_b = (w * torch.sin(angles).unsqueeze(0)).sum(dim=1)  # (N,)
            best_angle = torch.atan2(sin_b, cos_b)  # (N,)
        else:
            best_idx = score.argmin(dim=1)  # (N,)
            penalty_per_string = blockage_per_angle.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # (N,)
            best_angle = angles[best_idx]  # (N,)

        return penalty_per_string, best_angle

    def __call__(self, geom_dict, **kwargs):
        """
        points: (N, 2) tensor of 2D points
        Returns: dict with 'rov_penalty' (scalar), 'rov_penalty_per_string' (N,), and
                 'rov_least_blocked_angle_per_string' (N,).

        "Point away from the nearest neighbours" options (opt-in; defaults off):
        - rov_away_weight: float, default 0.0. If > 0, the chosen ROV heading trades off
            blockage (primary) against pointing away from the string's nearest
            neighbours (secondary): score[a] = blockage[a] + rov_away_weight*misalign[a],
            with misalign in [0, 1] (0 = points straight away). The reported
            least-blocked angle becomes this chosen heading, and the penalty is the
            blockage at it. Blockage dominates, so a single narrow clear corridor is
            still chosen and not punished; the away term only tips ties among comparably
            clear headings.
        - rov_away_soft: bool, default = rov_soft_inside. Soft variant (soft top-k
            neighbours via softmax over -distance + softmin heading selection, fully
            differentiable) vs hard variant (hard top-k + argmin selection).
        - rov_away_num_neighbours: int, default 5. Number of nearest neighbours defining
            the outward direction.
        - rov_away_nn_tau: float or None, default None. Softmax temperature for the soft
            neighbour selection, as a multiple of the geometry's own distance scale
            (median k-th nearest distance), so it is unit-robust. None -> 0.5.
        - rov_away_select_tau: float, default 0.1. Softmin temperature (blockage units)
            for the soft heading selection.
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

        # "Point away from the nearest neighbours" options (opt-in; default off).
        # When rov_away_weight > 0, the "best" ROV heading is chosen by trading off
        # blockage (primary) against pointing away from the string's nearest neighbours
        # (secondary): score[a] = blockage[a] + rov_away_weight * misalign[a]. The
        # reported least-blocked angle becomes this chosen heading, and the penalty is
        # the blockage at that heading. Because blockage dominates, a string with a
        # single narrow clear corridor (no alternative) still selects it and is not
        # punished; the away term only tips the choice among comparably-clear headings.
        # Two variants:
        #   - soft (rov_away_soft=True): soft top-k neighbours (softmax over -distance)
        #     + softmin heading selection, matching the other soft tricks; differentiable.
        #   - hard (rov_away_soft=False): hard top-k neighbours + hard argmin selection.
        rov_away_weight = float(kwargs.get('rov_away_weight', 0.0))
        rov_away_enabled = rov_away_weight > 0.0
        rov_away_soft = bool(kwargs.get('rov_away_soft', soft_inside))
        rov_away_num_neighbours = int(kwargs.get('rov_away_num_neighbours', 5))
        # None -> scale-aware default (0.5 * median k-th nearest distance).
        _nn_tau = kwargs.get('rov_away_nn_tau', None)
        rov_away_nn_tau = None if _nn_tau is None else float(_nn_tau)
        # Softmin temperature (blockage units) for the soft heading selection.
        rov_away_select_tau = float(kwargs.get('rov_away_select_tau', 0.1))

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

        # Outward "away from nearest neighbours" heading per string (opt-in).
        theta_away = None
        away_valid = None
        if rov_away_enabled:
            theta_away, away_valid = self._compute_away_theta(
                all_relative,
                num_neighbours=rov_away_num_neighbours,
                soft=rov_away_soft,
                nn_tau=rov_away_nn_tau,
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

        else:
            if string_probs is not None:
                other_probs = string_probs.unsqueeze(0).expand(N, N)[mask].reshape(N, N-1)
                if detach_other_probs:
                    other_probs = other_probs.detach()
            else:
                other_probs = None

            # soft_inside on GPU: chunk over angles to avoid materialising the full
            # (N, N-1, num_angles) float tensor. Hard-inside uses bool which is 8x
            # smaller so chunking is only worthwhile for soft.
            use_chunked = soft_inside and points.is_cuda
            chunk_size = kwargs.get('rov_angle_chunk_size', 16)

            blockage_per_angle = self._compute_blockage_per_angle_default(
                all_relative=all_relative,
                angles=angles,
                half_height=half_height,
                tri_length=L_tri,
                rec_width=L_rect,
                soft_inside=soft_inside,
                inside_sharpness=inside_sharpness,
                other_probs=other_probs,
                use_chunked=use_chunked,
                chunk_size=chunk_size,
            )

        angle_scores_per_angle = blockage_per_angle

        # --- Per-string penalty and reported least-blocked heading ---
        if rov_away_enabled and theta_away is not None:
            # Choose the heading by trading off blockage (primary) against pointing
            # away from the nearest neighbours (secondary). The penalty is the blockage
            # at the chosen heading, and the reported angle IS that heading.
            penalty_per_string, least_blocked_angle_per_string = self._select_away_best_angle(
                blockage_per_angle,
                angles,
                theta_away,
                away_valid,
                away_weight=rov_away_weight,
                soft=rov_away_soft,
                select_tau=rov_away_select_tau,
            )
        else:
            # Default behavior: least-blocked path only.
            if angle_softmin_tau > 0.0:
                penalty_per_string = -angle_softmin_tau * torch.logsumexp(
                    -blockage_per_angle / angle_softmin_tau, dim=1
                )
                penalty_per_string = penalty_per_string.clamp(min=0.0)
            else:
                penalty_per_string = blockage_per_angle.min(dim=1)[0]  # (N,)

            # Reported least-blocked angle (hard argmin over near-min bins). A small
            # tolerance resolves numerically-equivalent boundary bins to the first bin
            # consistently across the alt and non-alt implementations.
            min_scores = angle_scores_per_angle.min(dim=1, keepdim=True)[0]
            tie_tol = 1e-6
            near_min_mask = angle_scores_per_angle <= (min_scores + tie_tol)
            least_blocked_angle_idx_per_string = near_min_mask.to(torch.int64).argmax(dim=1)  # (N,)
            least_blocked_angle_per_string = angles[least_blocked_angle_idx_per_string]  # (N,)

        # Aggregate into the scalar loss, weighting by string probability if available.
        if string_probs is not None:
            loss = (penalty_per_string * string_probs).sum()
        else:
            loss = penalty_per_string.sum()

        # least_blocked_angle_deg_per_string = least_blocked_angle_per_string * (180.0 / torch.pi)
        penalty_per_string = torch.clamp(penalty_per_string, min=0.0)  # Ensure non-negative for reporting
        return {
            'rov_penalty': loss / N,
            'rov_penalty_per_string': penalty_per_string / N,
            'rov_least_blocked_angle_per_string': least_blocked_angle_per_string,
        }

class DiversityPenalty(LossFunction):
    """Loss function for diversity penalty to encourage diversity among optimal geometries."""
    def __init__(self, device=None):
        """
        Initialize the diversity penalty loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        """
        super().__init__(device)
        self._sinkhorn_loss_cache = {}

    def _get_sinkhorn_loss(self, blur):
        if SamplesLoss is None:
            raise ImportError(
                "geomloss is required for diversity_use_sinkhorn=True. Install geomloss to enable this path."
            )

        blur = float(blur)
        sinkhorn_loss = self._sinkhorn_loss_cache.get(blur)
        if sinkhorn_loss is None:
            sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
            self._sinkhorn_loss_cache[blur] = sinkhorn_loss
        return sinkhorn_loss

    def _geometry_distance(self, geometry_a, geometry_b, use_hungarian=False):
        """Compute a distance between two geometries.

        Default behavior assumes both geometries use the same string ordering and
        compares the weight tensors directly. Optional Hungarian matching can be
        enabled to align nearby strings first and then compare their strengths.
        """
        string_xy_a = geometry_a.get('string_xy', None)
        string_xy_b = geometry_b.get('string_xy', None)
        string_weights_a = geometry_a.get('string_weights', None)
        string_weights_b = geometry_b.get('string_weights', None)

        if string_weights_a is None or string_weights_b is None:
            return None

        weights_a = torch.sigmoid(string_weights_a)
        weights_b = torch.sigmoid(string_weights_b)

        if weights_a.shape == weights_b.shape and not use_hungarian:
            return torch.mean((weights_a - weights_b) ** 2)

        if not use_hungarian:
            min_len = min(weights_a.shape[0], weights_b.shape[0])
            if min_len == 0:
                return torch.tensor(0.0, device=weights_a.device)
            matched_distance = torch.mean((weights_a[:min_len] - weights_b[:min_len]) ** 2)
            if weights_a.shape[0] > min_len:
                matched_distance = matched_distance + torch.mean(weights_a[min_len:] ** 2)
            if weights_b.shape[0] > min_len:
                matched_distance = matched_distance + torch.mean(weights_b[min_len:] ** 2)
            return matched_distance

        if string_xy_a is None or string_xy_b is None or string_xy_a.numel() == 0 or string_xy_b.numel() == 0:
            min_len = min(weights_a.shape[0], weights_b.shape[0])
            if min_len == 0:
                return torch.tensor(0.0, device=weights_a.device)
            return torch.mean((weights_a[:min_len] - weights_b[:min_len]) ** 2)

        xy_cost = torch.cdist(string_xy_a, string_xy_b, p=2)
        weight_cost = torch.cdist(weights_a.unsqueeze(1), weights_b.unsqueeze(1), p=2)
        # Normalize spatial cost so it doesn't dominate weight cost.
        # Align the means of the two cost matrices (with safety clamps).
        mean_xy = torch.mean(xy_cost)
        mean_w = torch.mean(weight_cost)
        scale = mean_w / mean_xy
        xy_cost = xy_cost * scale
        cost_matrix = xy_cost + weight_cost
        row_indices, col_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        row_indices = torch.as_tensor(row_indices, device=weights_a.device, dtype=torch.long)
        col_indices = torch.as_tensor(col_indices, device=weights_b.device, dtype=torch.long)

        if row_indices.numel() == 0:
            return torch.tensor(0.0, device=weights_a.device)

        # matched_distance = torch.mean((weights_a[row_indices] - weights_b[col_indices]) ** 2)
        # distance should use cost matrix values to reflect both spatial and weight differences, not just weight differences
        matched_distance = torch.mean(cost_matrix[row_indices, col_indices] ** 2)
        
        if weights_a.shape[0] != weights_b.shape[0]:
            mask_a = torch.ones(weights_a.shape[0], dtype=torch.bool, device=weights_a.device)
            mask_b = torch.ones(weights_b.shape[0], dtype=torch.bool, device=weights_b.device)
            mask_a[row_indices] = False
            mask_b[col_indices] = False

            if torch.any(mask_a):
                matched_distance = matched_distance + torch.mean(weights_a[mask_a] ** 2)
            if torch.any(mask_b):
                matched_distance = matched_distance + torch.mean(weights_b[mask_b] ** 2)

        return matched_distance

    def _sinkhorn_distance(self, geometry_a, geometry_b, epsilon=0.01, niter=100):
        """Differentiable regularized OT distance via GeomLoss Sinkhorn loss.

        String weights (after sigmoid) define the transport masses and string
        positions define the sample support. Both remain in the compute graph,
        so gradients flow to positions and weights.

        Falls back to unmatched weight MSE when positions are unavailable in
        either geometry.
        """
        string_xy_a = geometry_a.get('string_xy', None)
        string_xy_b = geometry_b.get('string_xy', None)
        string_weights_a = geometry_a.get('string_weights', None)
        string_weights_b = geometry_b.get('string_weights', None)

        if string_weights_a is None or string_weights_b is None:
            return None

        if string_xy_a is None or string_xy_b is None or string_xy_a.numel() == 0 or string_xy_b.numel() == 0:
            # No positions: fall back to weight-only MSE (no matching needed)
            min_len = min(string_weights_a.shape[0], string_weights_b.shape[0])
            if min_len == 0:
                return torch.tensor(0.0, device=string_weights_a.device)
            wa = torch.sigmoid(string_weights_a)
            wb = torch.sigmoid(string_weights_b)
            return torch.mean((wa[:min_len] - wb[:min_len]) ** 2)

        weights_a = torch.sigmoid(string_weights_a)
        weights_b = torch.sigmoid(string_weights_b)

        # Compute a shared, robust scale from both geometries so distances
        # are compared on the same normalized domain. Use median pairwise
        # distance across the combined support (robust to outliers).
        def _robust_scale(xy1, xy2):
            xy = torch.cat([xy1.reshape(-1, xy1.shape[-1]), xy2.reshape(-1, xy2.shape[-1])], dim=0)
            if xy.shape[0] < 2:
                return torch.tensor(1.0, device=xy.device, dtype=xy.dtype)
            d = torch.cdist(xy, xy, p=2)
            d = d.reshape(-1)
            d = d[d > 0]
            if d.numel() == 0:
                return torch.tensor(1.0, device=xy.device, dtype=xy.dtype)
            return torch.median(d).clamp(min=1e-6)

        scale = _robust_scale(string_xy_a, string_xy_b)

        # Center both point clouds and normalize by the shared scale so the
        # Sinkhorn distance measures shape similarity rather than absolute
        # size/position. We keep absolute translation by centering both with
        # the combined mean to preserve relative alignment.
        combined_mean = torch.cat([string_xy_a, string_xy_b], dim=0).mean(dim=0)
        norm_xy_a = (string_xy_a - combined_mean) / scale
        norm_xy_b = (string_xy_b - combined_mean) / scale

        blur = math.sqrt(max(float(epsilon), 1e-12))
        sinkhorn_loss = self._get_sinkhorn_loss(blur=blur)
        return sinkhorn_loss(weights_a, norm_xy_a, weights_b, norm_xy_b)

    def _mmd_distance(self, geometry_a, geometry_b, kernel_sigma=None):
        """Differentiable MMD² between two string distributions.

        Treats sigmoid(string_weights) as distribution masses and string
        positions as support points.  Uses a Gaussian kernel on positions —
        three kernel-matrix evaluations, no iterations, no matching.

        kernel_sigma : squared bandwidth of the Gaussian kernel k(x,y)=exp(-D/σ).
                       Defaults to the median heuristic (median of all pairwise
                       squared distances across both geometries).
        """
        string_xy_a = geometry_a.get('string_xy', None)
        string_xy_b = geometry_b.get('string_xy', None)
        string_weights_a = geometry_a.get('string_weights', None)
        string_weights_b = geometry_b.get('string_weights', None)

        if string_weights_a is None or string_weights_b is None:
            return None

        wa = torch.sigmoid(string_weights_a)
        wa = wa / (wa.sum() + 1e-10)   # (n,) normalized
        wb = torch.sigmoid(string_weights_b)
        wb = wb / (wb.sum() + 1e-10)   # (m,) normalized

        if string_xy_a is None or string_xy_b is None or string_xy_a.numel() == 0 or string_xy_b.numel() == 0:
            min_len = min(wa.shape[0], wb.shape[0])
            if min_len == 0:
                return torch.tensor(0.0, device=wa.device)
            return torch.mean((wa[:min_len] - wb[:min_len]) ** 2)

        D_aa = torch.cdist(string_xy_a, string_xy_a, p=2) ** 2   # (n, n)
        D_ab = torch.cdist(string_xy_a, string_xy_b, p=2) ** 2   # (n, m)
        D_bb = torch.cdist(string_xy_b, string_xy_b, p=2) ** 2   # (m, m)

        if kernel_sigma is None:
            all_sq_dists = torch.cat([D_aa.reshape(-1), D_ab.reshape(-1), D_bb.reshape(-1)])
            kernel_sigma = torch.median(all_sq_dists).clamp(min=1e-10)

        K_aa = torch.exp(-D_aa / kernel_sigma)
        K_ab = torch.exp(-D_ab / kernel_sigma)
        K_bb = torch.exp(-D_bb / kernel_sigma)

        # MMD² = wᵀK_aa w  −  2 wᵀK_ab v  +  vᵀK_bb v
        return wa @ K_aa @ wa - 2.0 * (wa @ K_ab @ wb) + wb @ K_bb @ wb

    def _to_device_geometry(self, geometry):
        """Move supported geometry tensors onto the loss device."""
        if geometry is None:
            return {}

        moved_geometry = dict(geometry)
        for key in ('string_xy', 'string_weights'):
            value = moved_geometry.get(key, None)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            elif value.device != self.device:
                value = value.to(self.device)
            moved_geometry[key] = value

        return moved_geometry

    def __call__(self, geom_dict, **kwargs):
        """
        Compute diversity penalty for the current geometry against prior geometries.
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'string_xy' and optionally 'string_weights' keys.
        **kwargs
            Additional keyword arguments including 'max_radius', 'min_dist', 'domain_size', and 'ignore_border'.
            
        Returns:
        --------
        torch.Tensor
            The diversity penalty value (weighted).
        """
        other_geoms = kwargs.get('other_geoms', None)
        diversity_min = kwargs.get('diversity_min', 20)
        use_hungarian = kwargs.get('diversity_use_hungarian', False)
        use_sinkhorn = kwargs.get('diversity_use_sinkhorn', False)
        sinkhorn_epsilon = kwargs.get('sinkhorn_epsilon', 0.01)
        sinkhorn_niter = kwargs.get('sinkhorn_niter', 100)
        use_mmd = kwargs.get('diversity_use_mmd', False)
        mmd_kernel_sigma = kwargs.get('mmd_kernel_sigma', None)

        if other_geoms is None:
            zero = torch.tensor(0.0, device=self.device)
            return {'diversity_penalty': zero, 'diversity_score': zero}

        if isinstance(other_geoms, dict):
            other_geoms = [other_geoms]

        other_geoms = [self._to_device_geometry(geometry) for geometry in other_geoms if geometry is not geom_dict]
        if len(other_geoms) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {'diversity_penalty': zero, 'diversity_score': zero}

        current_geometry = self._to_device_geometry(geom_dict)

        pairwise_distances = []
        for other_geometry in other_geoms:
            if use_sinkhorn:
                pair_distance = self._sinkhorn_distance(
                    current_geometry, other_geometry,
                    epsilon=sinkhorn_epsilon, niter=sinkhorn_niter,
                )
            elif use_mmd:
                pair_distance = self._mmd_distance(
                    current_geometry, other_geometry,
                    kernel_sigma=mmd_kernel_sigma,
                )
            else:
                pair_distance = self._geometry_distance(current_geometry, other_geometry, use_hungarian=use_hungarian)
            if pair_distance is not None:
                pairwise_distances.append(pair_distance)

        if len(pairwise_distances) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {'diversity_penalty': zero, 'diversity_score': zero}

        pairwise_distances = torch.stack(pairwise_distances)
        diversity_score = torch.min(pairwise_distances)
        diversity_threshold = torch.as_tensor(diversity_min, device=diversity_score.device, dtype=diversity_score.dtype)
        diversity_penalty = diversity_threshold - diversity_score
        # print(f"Diversity score: {diversity_score.item():.4f}, penalty: {diversity_penalty.item():.4f}")
        return {
            'diversity_penalty': diversity_penalty,
            'diversity_score': diversity_score,
            'diversity_min_distance_to_others': pairwise_distances,
        }


