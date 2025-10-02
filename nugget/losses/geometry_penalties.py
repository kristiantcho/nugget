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
        clamped_string_xy = torch.sum(clamped_string_xy, dim=1)
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
            return torch.tensor(0.0)
            
        n = len(points_3d)
        if n == 0:
            return torch.tensor(0.0)
            
        # Stack points for efficient computation
        points_tensor = points_3d # (n, 3)
        
        # Compute pairwise squared distances
        diff = points_tensor.unsqueeze(1) - points_tensor.unsqueeze(0)  # (n, n, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        
        # Mask: ignore self-pairs and pairs outside radius
        self_mask = torch.eye(n, dtype=torch.bool, device=points_tensor.device)
        radius_mask = dist_sq < max_radius ** 2
        
        mask = (~self_mask) & radius_mask
        
        # Compute repulsion for valid pairs
        repulsion_matrix = torch.zeros_like(dist_sq)
        repulsion_matrix[mask] = 1.0 / (dist_sq[mask] + min_dist)
        repulsion = torch.sum(repulsion_matrix) / n if n > 0 else torch.tensor(0.0)
        
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
        
        if string_xy is None:
            return torch.tensor(0.0)
        n = string_xy.shape[0]
        # Compute pairwise squared distances
        diff = string_xy.unsqueeze(1) - string_xy.unsqueeze(0)  # (n, n, 2)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        # Mask: ignore self-pairs and pairs outside radius
        mask = (dist_sq > 0) & (dist_sq < max_radius ** 2)
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
        
        repulsion = torch.tensor(0.0, device=self.device)
        total_valid_pairs = 0
        
        # Use z_values and points_per_string_list if available (more efficient)
        if z_values is not None and points_per_string_list is not None:
            current_idx = 0
            for string_idx, num_points in enumerate(points_per_string_list):
                if num_points > 1:  # Only compute repulsion if string has multiple points
                    # Get z values for this string
                    string_z_values = z_values[current_idx:current_idx + num_points]
                    
                    # Compute pairwise repulsion within this string, within radius
                    for i in range(num_points):
                        for j in range(i + 1, num_points):
                            z_dist_sq = (string_z_values[i] - string_z_values[j]) ** 2
                            z_dist = torch.sqrt(z_dist_sq + 1e-10)  # Add small epsilon for numerical stability
                            
                            if z_dist < max_radius:
                                repulsion += 1.0 / (z_dist_sq + min_dist)
                                total_valid_pairs += 1
                
                current_idx += num_points
        
        # Fallback to points_3d-based computation if z_values/points_per_string_list not available
        elif points_3d is not None:
            n = len(points_3d)
            if n > 0:
                # Stack points for efficient computation
                points_tensor = points_3d  # (n, 3)
                
                # Compute pairwise squared distances
                diff = points_tensor.unsqueeze(1) - points_tensor.unsqueeze(0)  # (n, n, 3)
                dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
                
                # Check for same string (same x,y coordinates)
                xy_coords = points_tensor[:, :2]  # (n, 2)
                xy_diff = xy_coords.unsqueeze(1) - xy_coords.unsqueeze(0)  # (n, n, 2)
                xy_dist_sq = torch.sum(xy_diff ** 2, dim=-1)  # (n, n)
                
                # Get z-coordinates for distance computation
                z_coords = points_tensor[:, 2]  # (n,)
                z_diff = z_coords.unsqueeze(1) - z_coords.unsqueeze(0)  # (n, n)
                z_dist_sq = z_diff ** 2
                
                # Mask: same string (same x,y), ignore self-pairs, and within z-radius
                same_string_mask = xy_dist_sq < 1e-6  # Same x,y coordinates (same string)
                self_mask = torch.eye(n, dtype=torch.bool, device=points_tensor.device)
                radius_mask = torch.sqrt(z_dist_sq + 1e-10) < max_radius
                
                mask = same_string_mask & (~self_mask) & radius_mask
                
                # Compute repulsion for valid pairs
                repulsion_matrix = torch.zeros_like(dist_sq)
                repulsion_matrix[mask] = 1.0 / (z_dist_sq[mask] + min_dist)
                repulsion = torch.sum(repulsion_matrix)
                total_valid_pairs = torch.sum(mask.float()).item()
        
        # Normalize by number of valid pairs or total points
        if total_valid_pairs > 0:
            repulsion = repulsion / total_valid_pairs
        elif points_3d is not None:
            repulsion = repulsion / len(points_3d) if len(points_3d) > 0 else repulsion
        
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
        
        string_probs = torch.sigmoid(string_weights) if string_weights is not None else None
        # string_probs = string_weigh
        return {'string_number_penalty': F.softplus(torch.sum(string_probs) - eva_min_num_strings)}

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
        string_probs_cut = torch.clamp(string_probs, min=0.0, max=1.0)
        return {'weight_binarization_penalty': torch.sum(-string_probs_cut * torch.log(string_probs_cut + 1e-10) - (1 - string_probs_cut) * torch.log(1 - string_probs_cut + 1e-10))}
    
    
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
        
   

    def inside_safe_space(self, points, theta):
        """
        points: (N, 2) tensor of 2D points relative to candidate ROV position
        theta: scalar angle in radians (rotation of safe space)
        returns: (N,) tensor ~ soft indicator if points are inside region
        """

        # Rotation matrix
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], device=points.device)
        rot_points = points @ R.T  # rotate into canonical frame

        # Canonical safe space dimensions (from diagram)
        L_rect = self.rov_rec_width
        W_rect = self.rov_height
        L_tri = self.rov_tri_length
        W_tri = self.rov_tri_length

        x, y = rot_points[:, 0], rot_points[:, 1].abs()

        # Inside rectangular part
        inside_rect = (x >= 0) & (x <= L_rect) & (y <= W_rect / 2)

        # Inside triangular part (slope check)
        slope = W_tri / L_tri  # 80/80 = 1.0
        inside_tri = (x >= L_rect) & (x <= L_rect + L_tri) & (y <= slope * (L_rect + L_tri - x))

        inside = inside_rect | inside_tri
        return inside.float()

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
        loss = 0.0

        for i in range(N):
            others = torch.cat([points[:i], points[i+1:]], dim=0) - points[i]  # relative coords

            ok = []
            for k in range(num_angles):
                theta = torch.tensor(2 * torch.pi * k / num_angles)
                inside = self.inside_safe_space(others, theta)
                ok.append(inside.any().float())  # 1 if blocked, 0 if free

            # If all orientations blocked -> penalty = 1
            penalty = torch.stack(ok).min()  # min over orientations
            if string_probs is not None:
                penalty *= string_probs[i]
            loss += penalty

        return {'rov_penalty': loss/N}

            
        