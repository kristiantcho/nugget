import torch
import numpy as np
import torch.nn.functional as F

class LossFunction:
    """Base class for loss functions in geometry optimization."""
    
    def __init__(self, device=None, repulsion_weight=0.001, boundary_weight=100.0,
                 string_repulsion_weight=0.001, path_repulsion_weight=0.001, z_repulsion_weight=0.001, min_dist=1e-3, domain_size=2):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        repulsion_weight : float
            Weight for repulsion penalty between points.
        boundary_weight : float
            Weight for boundary penalty.
        string_repulsion_weight : float
            Weight for repulsion between strings.
        path_repulsion_weight : float
            Weight for repulsion in path space.
        z_repulsion_weight : float
            Weight for z distance (along string) penalty.
            
        min_dist : float
            Minimum distance threshold.
        domain : float
            Size of the domain.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.repulsion_weight = repulsion_weight
        self.boundary_weight = boundary_weight
        self.string_repulsion_weight = string_repulsion_weight
        self.path_repulsion_weight = path_repulsion_weight
        self.z_repulsion_weight = z_repulsion_weight
        # self.order_weight = order_weight
        self.min_dist = min_dist
        self.domain_size = domain_size
    
    def __call__(self, points, **kwargs):
        """
        Compute the loss.
        
        Parameters:
        -----------
        points : torch.Tensor
            The points to evaluate the loss at.
        **kwargs : dict
            Additional arguments specific to the loss function.
            
        Returns:
        --------
        float
            The computed loss value.
        """
        raise NotImplementedError("Subclasses must implement __call__")
    
    
    def compute_boundary_penalty(self, points_3d):
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
        return self.boundary_weight*torch.mean(
            torch.clamp(torch.abs(points_3d) - self.domain_size/2, min=0.0) ** 2
        )
        
    def compute_repulsion_penalty(self, points_3d):
        """
        Compute repulsion penalty to keep points apart.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The 3D repulsion penalty value (weighted).
        """
        repulsion = 0.0
        total_points = len(points_3d)
        for k in range(total_points):
            for j in range(k + 1, total_points):
                # Use distance in path space
                dist_sq = torch.sum((points_3d[k] - points_3d[j]) ** 2)
                repulsion += 1.0 / (dist_sq + self.min_dist)
        
        return self.repulsion_weight*repulsion
    
    
    def compute_string_repulsion_penalty(self, string_xy, points_per_string_list):
        """
        Compute repulsion penalty between strings.
        
        Parameters:
        -----------
        string_xy : list of torch.Tensor or None
            The 2D points of the strings to compute the penalty for.
        points_per_string_list : list of int
            The number of points in each string.
            
        Returns:
        --------
        torch.Tensor
            The string repulsion penalty value (weighted).
        """
        # Return zero penalty if string_xy is None
        if string_xy is None:
            return torch.tensor(0.0, device=self.device)
            
        string_repulsion = 0.0
        total_strings = len(string_xy)
        
        for i in range(total_strings):
            for j in range(i + 1, total_strings):
                # Use distance in path space
                if points_per_string_list[i] == 0 or points_per_string_list[j] == 0:
                    continue
                dist_sq = torch.sum((string_xy[i] - string_xy[j]) ** 2)
                string_repulsion += 1.0 / (dist_sq + self.min_dist)
        
        return self.string_repulsion_weight*string_repulsion
    
    
    def path_repulsion_penalty(self, path_positions, number_of_strings):
        """
        Compute repulsion penalty to keep points apart in continuous string path space (normalized).
        
        Parameters:
        -----------
        path_positions : torch.Tensor
            The positions of points along the continuous string path to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The path repulsion penalty value (weighted).
        """
        path_penalty = 0.0
        total_points = len(path_positions)
        path_min_dist = self.min_dist / (number_of_strings*self.domain_size)
        for k in range(total_points):
            for j in range(k + 1, total_points):
                # Use distance in path space
                dist_sq = torch.sum((path_positions[k] - path_positions[j]) ** 2)
                path_penalty += 1.0 / (dist_sq + path_min_dist)
                
        return self.path_repulsion_weight*path_penalty
    
    
    def z_dist_penalty(self, points_3d):
        """
        Compute z distance penalty to keep points apart along the same string.
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to compute the penalty for.
        Returns:
        --------
        torch.Tensor
            The z distance penalty value (weighted).
        """
        z_dist_penalty = 0.0
        total_points = len(points_3d)
        for k in range(total_points):
            for j in range(k + 1, total_points):
                if points_3d[k][2] == points_3d[j][2]:
                    # Use distance in path space
                    dist_sq = torch.sum((points_3d[k] - points_3d[j]) ** 2)
                    z_dist_penalty += 1.0 / (dist_sq + self.min_dist)
        
        return self.z_repulsion_weight*z_dist_penalty
    
    
    



class RBFInterpolationLoss(LossFunction):
    """Loss function for RBF interpolation."""
    
    def __init__(self, device=None, repulsion_weight=0.0004, boundary_weight=100.0, epsilon=30.0,
                 string_repulsion_weight=0.001, path_repulsion_weight=0.001, z_repulsion_weight=0.001, min_dist=1e-3, sampling_weight=0.001, domain_size=2):
        """
        Initialize the RBF interpolation loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        repulsion_weight : float
            Weight for repulsion penalty between points.
        boundary_weight : float
            Weight for boundary penalty.
        string_repulsion_weight : float
            Weight for repulsion between strings.
        path_repulsion_weight : float
            Weight for repulsion in path space.
        z_repulsion_weight : float
            Weight for z distance (along string) penalty.
            
        min_dist : float
            Minimum distance threshold.
        domain : float
            Size of the domain.
        """
        super().__init__(device, repulsion_weight, boundary_weight,
                         string_repulsion_weight, path_repulsion_weight,
                         z_repulsion_weight, min_dist, domain_size)
        self.sampling_weight = sampling_weight
        self.epsilon = epsilon
    
    
    def compute_rbf_interpolant(self, x_points, f_values, test_points):
        """
        Compute RBF interpolant weights and kernel matrix.
        
        Parameters:
        -----------
        x_points : torch.Tensor
            Interpolation points (N, dim)
        f_values : torch.Tensor
            Function values at interpolation points (N,)
        test_points : torch.Tensor
            Points where the interpolant will be evaluated (M, dim)
            
        Returns:
        --------
        tuple (torch.Tensor, torch.Tensor)
            (weights, kernel matrix)
        """
        # Compute pairwise distances
        r = torch.cdist(x_points, x_points)  # Shape (N, N)
        
        # Compute RBF kernel matrix
        K = self.rbf(r)  # Shape (N, N)
        
        # Using linear solve instead of lstsq to maintain gradient flow
        # Add small regularization to ensure stability
        reg = torch.eye(K.size(0), device=K.device) * 1e-8
        w = torch.linalg.solve(K + reg, f_values)  # This maintains gradients better than lstsq
        
        # Compute distances to test points
        r_test = torch.cdist(test_points, x_points)  # Shape (M, N)
        K_test = self.rbf(r_test)  # Shape (M, N)
        
        return w, K_test
    
    def rbf(self, r):
        """
        Radial Basis Function (Gaussian).
        
        Parameters:
        -----------
        r : torch.Tensor
            Distances
            
        Returns:
        --------
        torch.Tensor
            RBF values
        """
        return torch.exp(-self.epsilon * r**2)
        
    
    def __call__(self, points_3d, surrogate_funcs, test_points=None, points_per_string_list=None, string_xy=None, path_positions=None, loss_func = F.mse_loss, num_strings = None, **kwargs):
        """
        Compute the RBF interpolation loss.
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        surrogate_funcs : list of callable
            List of surrogate functions for evaluation.
        test_points : torch.Tensor or None
            Test points to test against RBF interpolated function. If None, random points are generated within the geometry domain.
        points_per_string_list : list of int
            The number of points in each string.
        string_xy : list of torch.Tensor
            The 2D points of the strings.
        path_positions : torch.Tensor
            The positions of points along the continuous string path.
        loss_func : callable
            Loss function (pyTorch) to use for computing the loss.
        num_strings : int
            Number of strings in the geometry.
        **kwargs : dict
            Additional arguments specific to the loss function.
        Returns:
        --------
        float, float, torch.Tensor, torch.Tensor, torch.Tensor
            The computed loss value, the unweighted loss value, the surrogate function values at test points, rbf interpolated values at test points, and the surrogate function values at the geometry points.
        """
        
        if test_points is None:
            test_points = torch.rand(1000, 3).to(self.device) * self.domain_size - self.domain_size / 2
        
        f_tests = []
        s_tests = []
        f_value_sets = []
        for i, surrogate_func in enumerate(list(surrogate_funcs)):
            
            f_values = surrogate_func(points_3d)
            f_test = surrogate_func(test_points)
            w, K = self.compute_rbf_interpolant(points_3d, f_values, test_points)
            s_test = K @ w
            
            f_tests.append(f_test)
            s_tests.append(s_test)
            f_value_sets.append(f_values)
        f_tests = torch.stack(f_tests)
        s_tests = torch.stack(s_tests)
        f_value_sets = torch.stack(f_value_sets)
        
        loss = loss_func(f_tests, s_tests)
        uw_loss = loss.item()
        
        # Compute penalties
        if points_per_string_list is not None and self.string_repulsion_weight > 0:
            string_repulsion_penalty = self.compute_string_repulsion_penalty(string_xy, points_per_string_list)
            loss += string_repulsion_penalty
        if path_positions is not None and self.path_repulsion_weight > 0:
            if num_strings is None:
                num_strings = len(points_per_string_list)
            path_repulsion_penalty = self.path_repulsion_penalty(path_positions, num_strings)
            loss += path_repulsion_penalty
        if self.repulsion_weight > 0:
            repulsion_penalty = self.compute_repulsion_penalty(points_3d)
            loss += repulsion_penalty
        if self.z_repulsion_weight > 0:
            z_repulsion_penalty = self.z_dist_penalty(points_3d)
            loss += z_repulsion_penalty
        if self.boundary_weight > 0:
            boundary_penalty = self.compute_boundary_penalty(points_3d)
            loss += boundary_penalty
        if self.sampling_weight > 0:
            loss += self.sampling_weight/torch.mean(f_value_sets)
        
        return loss, uw_loss, f_tests, s_tests, f_value_sets
    
class SNRloss(LossFunction):
    """Loss function for SNR (Signal-to-Noise Ratio) optimization."""
    
    def __init__(self, device=None, repulsion_weight=0.001, boundary_weight=1.0,
                 string_repulsion_weight=0.0005, path_repulsion_weight=0.001,
                 z_repulsion_weight=0.001, min_dist=1e-3, domain_size=2, snr_weight=1.0, signal_scale=1.0, background_scale=1.0, no_background=False):
        """
        Initialize the SNR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        repulsion_weight : float
            Weight for repulsion penalty between points on the same string.
        boundary_weight : float
            Weight for boundary penalty.
        string_repulsion_weight : float
            Weight for repulsion between strings.
        path_repulsion_weight : float
            Weight for repulsion in path space.
        z_repulsion_weight : float
            Weight for z distance (along string) penalty.
        min_dist : float
            Minimum distance threshold.
        domain_size : float
            Size of the domain.
        min_spacing : float
            Minimum spacing between points on a string.
        order_weight : float
            Weight for order penalty to maintain point ordering.
        snr_weight : float
            Weight for the SNR loss term.
        """
        super().__init__(device, repulsion_weight, boundary_weight,
                         string_repulsion_weight, path_repulsion_weight,
                         z_repulsion_weight, min_dist, domain_size)
        self.snr_weight = snr_weight
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.no_background = no_background
        
    
    def __call__(self, points_3d, signal_funcs, background_funcs, string_xy=None, 
                 points_per_string_list=None, path_positions=None, num_strings=None, **kwargs):
        """
        Compute the SNR loss.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Points in 3D space (total_points, 3).
        signal_func : callable
            Function that takes points_3d and returns signal values.
        background_func : callable
            Function that takes points_3d and returns background values.
        z_values : torch.Tensor or None
            Z-coordinates of points.
        string_xy : torch.Tensor or None
            XY positions of strings if optimizing them.
        string_indices : list or None
            Indices indicating which string each point belongs to.
        points_per_string_list : list or None
            Number of points on each string.
        path_positions : torch.Tensor or None
            Positions along the continuous path for path-based optimization.
        num_strings : int or None
            Number of strings in the geometry.
        **kwargs : dict
            Additional arguments specific to the loss function.
            
        Returns:
        --------
        torch.Tensor, float, torch.Tensor
            The computed loss value, the average SNR over all signal functions, and the SNR values for each signal function.
        """
        signal_total = torch.zeros(len(signal_funcs), device=self.device)

        # signal_values = torch.zeros(len(signal_funcs), points_3d.shape[0], device=self.device)
        for i, func in enumerate(signal_funcs):
            signal_total[i] = torch.sum(func(points_3d)) * self.signal_scale
        
        # signal_total = torch.sum(signal_values, dim=1)  # Sum across all points for each function
        
        # Compute background light yield (sum of background function values)
        background_total = torch.zeros(len(background_funcs), device=self.device)
        for i, func in enumerate(background_funcs):
            background_total[i] = torch.sum(func(points_3d)) * self.background_scale
        
        # background_total = torch.sum(background_values, dim=1)  # Sum across all points for each function
        
        # Compute SNR for each signal function against the average background
        avg_background = torch.mean(background_total)
        
        # Fix for no_background=True: Create a consistent background value that doesn't change between batches
        if self.no_background:
            # Use a constant value of 1 instead of a newly created tensor to ensure consistency across batches
            avg_background = torch.tensor(self.background_scale, device=self.device)
            
        # Compute SNR with a small epsilon to avoid division by zero
        epsilon = 1e-10 if not self.no_background else 0
        snr = signal_total / torch.sqrt(avg_background + epsilon)
        
        # Average SNR across all signal functions
        avg_snr = torch.mean(snr)
        
        # SNR loss (negative since we want to maximize SNR)
        snr_loss = -self.snr_weight * avg_snr
        
        # Initialize total loss with SNR loss
        total_loss = snr_loss
        
        optimize_params = kwargs.get('optimize_params', None)
        grid_size = kwargs.get('grid_size', None)
        
        if len(optimize_params) == 2:
            # Reshape for 2D grid visualization
            all_snr = torch.zeros(grid_size, grid_size, device=self.device)
            for i in range(grid_size):
                for j in range(grid_size):
                    all_snr[i, j] = snr[i * grid_size + j]
        elif len(optimize_params) == 1:
            all_snr = snr.detach()
        
        # Compute penalties
        if points_per_string_list is not None and self.string_repulsion_weight > 0 and string_xy is not None:
            string_repulsion_penalty = self.compute_string_repulsion_penalty(string_xy, points_per_string_list)
            total_loss += string_repulsion_penalty
        
        if path_positions is not None and self.path_repulsion_weight > 0:
            if num_strings is None:
                num_strings = len(points_per_string_list)
            path_repulsion_penalty = self.path_repulsion_penalty(path_positions, num_strings)
            total_loss += path_repulsion_penalty
        
        if self.repulsion_weight > 0:
            repulsion_penalty = self.compute_repulsion_penalty(points_3d)
            total_loss += repulsion_penalty
        
        if self.z_repulsion_weight > 0:
            z_repulsion_penalty = self.z_dist_penalty(points_3d)
            total_loss += z_repulsion_penalty
        
        if self.boundary_weight > 0:
            boundary_penalty = self.compute_boundary_penalty(points_3d)
            total_loss += boundary_penalty
        
        return total_loss, avg_snr.item(), all_snr






