import torch
from torch.nn import functional as F
from nugget.losses.base_loss import LossFunction

class BoundaryPenalties(LossFunction):
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
        domain_size = kwargs.get('domain_size', 2.0)
        return torch.mean(torch.clamp(torch.abs(points_3d) - domain_size/2, min=0.0) ** 2
        )

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
        
        return repulsion


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
            The string repulsion penalty value (weighted).
        """
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        min_dist = kwargs.get('min_dist', 1e-3)
        
        # Return zero penalty if string_xy is None
        if string_xy is None:
            return torch.tensor(0.0)
            
        string_repulsion = 0.0
        total_strings = len(string_xy)
        if string_weights is not None:
            # string_probs = torch.sigmoid(string_weights)
            string_probs = string_weights
        for i in range(total_strings):
            for j in range(i + 1, total_strings):
                if string_weights is None:
                    dist_sq = torch.sum((string_xy[i] - string_xy[j]) ** 2)
                    string_repulsion += 1.0/ (dist_sq + min_dist)
                else:
                    dist_sq = torch.sum(((string_xy[i] - string_xy[j]) ** 2))
                    string_repulsion += string_probs[i] * string_probs[j] / (dist_sq + min_dist)
                
        
        return string_repulsion/total_strings

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
            string_probs = string_weights
            # Outer product for all pairs
            weight_matrix = string_probs.unsqueeze(1) * string_probs.unsqueeze(0)  # (n, n)
            repulsion_matrix = torch.zeros_like(dist_sq)
            repulsion_matrix[mask] = weight_matrix[mask] / (dist_sq[mask] + min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        else:
            repulsion_matrix = torch.zeros_like(dist_sq)
            repulsion_matrix[mask] = 1.0 / (dist_sq[mask] + min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        return repulsion
    

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
                
        return path_penalty


class ZDistPenalty(LossFunction):
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
        
        Parameters:
        -----------
        geom_dict : dict
            Geometry dictionary containing 'points_3d' key.
        **kwargs
            Additional keyword arguments including 'min_dist'.
            
        Returns:
        --------
        torch.Tensor
            The z distance penalty value (weighted).
        """
        points_3d = geom_dict.get('points_3d', None)
        min_dist = kwargs.get('min_dist', 1e-3)
        
        z_dist_penalty = 0.0
        total_points = len(points_3d)
        for k in range(total_points):
            for j in range(k + 1, total_points):
                if points_3d[k][2] == points_3d[j][2]:
                    # Use distance in path space
                    dist_sq = torch.sum((points_3d[k] - points_3d[j]) ** 2)
                    z_dist_penalty += 1.0 / (dist_sq + min_dist)
        
        return z_dist_penalty

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
        
        # string_probs = torch.sigmoid(string_weights)
        string_probs = string_weights
        return torch.sum(torch.sqrt(string_probs)) / len(string_probs)

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
        
        return torch.mean((string_weights - torch.clamp(string_weights, min=0.0, max=0.8)) ** 2)

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
        
        # string_probs = torch.sigmoid(string_weights)
        string_probs = string_weights
        return F.softplus(torch.sum(string_probs) - eva_min_num_strings)

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
        
        string_weights_cut = torch.clamp(string_weights, min=0.0, max=1.0)
        return torch.sum(-string_weights_cut * torch.log(string_weights_cut + 1e-10) - (1 - string_weights_cut) * torch.log(1 - string_weights_cut + 1e-10))