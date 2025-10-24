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
    
class StringBoundaryPenalty(LossFunction):
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
        string_xy = geom_dict.get('string_xy', None)
        domain_size = kwargs.get('boundary_range', 2.0)
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else 1.0
        clamped_string_xy = torch.clamp(torch.abs(string_xy) - domain_size/2, min=0.0)** 2
        clamped_string_xy = torch.sqrt(torch.sum(clamped_string_xy, dim=1))
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
        
        if points_3d is None:
            return {'local_repulsion_penalty': torch.tensor(0.0)}
            
        n = len(points_3d)
        if n == 0:
            return {'local_repulsion_penalty': torch.tensor(0.0)}

        sharpness = kwargs.get('local_sharpness', 100.0)  # Controls steepness of sigmoid transition
        
      
        # Compute pairwise squared distances
        diff = points_3d.unsqueeze(1) - points_3d.unsqueeze(0)  # (n, n, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        dist = torch.sqrt(dist_sq + 1e-10)  # Add small epsilon for numerical stability
        
        # Soft mask using sigmoid - smoother transition at radius boundary
        self_mask = torch.eye(n, dtype=torch.bool, device=points_3d.device)
        radius_weight = torch.sigmoid((max_radius - dist) * sharpness)  # Sharp transition around max_radius
        radius_weight = radius_weight * (~self_mask).float()  # Zero out self-pairs

        repulsion_matrix = radius_weight / (dist_sq + min_dist)
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
        sharpness = kwargs.get('local_sharpness', 100.0)  # Controls steepness of sigmoid transition
        
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
        radius_weight = radius_weight * (~self_mask).float()  # Zero out self-pairs
        
        repulsion = 0.0
        if string_weights is not None:
            string_probs = torch.sigmoid(string_weights)
            # Outer product for all pairs
            weight_matrix = string_probs.unsqueeze(1) * string_probs.unsqueeze(0)  # (n, n)
            repulsion_matrix = weight_matrix * radius_weight / (dist_sq + min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        else:
            repulsion_matrix = radius_weight / (dist_sq + min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        return {'local_string_repulsion_penalty': repulsion}
    

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
            return {'string_number_penalty': F.softplus(torch.sum(string_probs) - eva_min_num_strings, beta=string_number_beta)/len(string_probs)}
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
            return {'weight_binarization_penalty': torch.sum(-string_probs * torch.log(string_probs + 1e-10) - (1 - string_probs) * torch.log(1 - string_probs + 1e-10))/ len(string_probs)}
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
        
   

    def __call__(self, geom_dict, **kwargs):
        """
        points: (N, 2) tensor of 2D points
        Returns: scalar penalty loss
        """
        points = geom_dict.get('string_xy', None)
        num_angles = kwargs.get('num_angles', 6)
        string_weights = geom_dict.get('string_weights', None)
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None

        N = points.shape[0]
        
        # Vectorized computation
        # Compute all relative positions at once
        mask = ~torch.eye(N, dtype=torch.bool, device=points.device)
        all_relative = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, 2)
        all_relative = all_relative[mask].reshape(N, N-1, 2)  # (N, N-1, 2)
        
        # Prepare angles
        angles = torch.linspace(0, 2 * torch.pi * (num_angles - 1) / num_angles, 
                            num_angles, device=points.device)
        c = torch.cos(angles)
        s = torch.sin(angles)
        
        # Apply rotation for all angles at once: (N, N-1, num_angles, 2)
        rel_expanded = all_relative.unsqueeze(2)  # (N, N-1, 1, 2)
        x_rot = rel_expanded[..., 0] * c - rel_expanded[..., 1] * s
        y_rot_abs = (rel_expanded[..., 0] * s + rel_expanded[..., 1] * c).abs()
        
        # Geometry checks
        L_rect = self.rov_rec_width
        W_rect = self.rov_height
        L_tri = self.rov_tri_length
        W_tri = self.rov_tri_length
        slope = W_tri / L_tri
        
        inside_rect = (x_rot >= 0) & (x_rot <= L_rect) & (y_rot_abs <= W_rect / 2)
        inside_tri = (x_rot >= L_rect) & (x_rot <= L_rect + L_tri) & \
                    (y_rot_abs <= slope * (L_rect + L_tri - x_rot))
        inside = inside_rect | inside_tri  # (N, N-1, num_angles)
        
        if string_probs is not None:
            # Get weights of other strings for each position
            other_probs = string_probs.unsqueeze(0).expand(N, N)[mask].reshape(N, N-1)
            
            # For each angle, sum weights of strings inside safe space
            # (N, num_angles) = sum over other strings
            blockage_per_angle = (inside.float() * other_probs.unsqueeze(-1)).sum(dim=1)
            
            # For each string: minimum blockage across all angles
            penalty_per_string = blockage_per_angle.min(dim=1)[0]  # (N,)
            
            # Weight by string probability and sum
            loss = (penalty_per_string * string_probs).sum()
        else:
            # Count number of strings inside for each angle
            # (N, num_angles) = count of other strings inside
            count_per_angle = inside.float().sum(dim=1)  # (N, num_angles)
            
            # For each string: minimum count across all angles
            penalty_per_string = count_per_angle.min(dim=1)[0]  # (N,)
            
            loss = penalty_per_string.sum()
        
        return {'rov_penalty': loss / N, 'rov_penalty_per_string': penalty_per_string/N}

            
        