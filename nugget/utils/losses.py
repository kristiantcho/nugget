import torch
import numpy as np
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score

class LossFunction:
    """Base class for loss functions in geometry optimization."""
    
    def __init__(self, device=None, repulsion_weight=0.001, boundary_weight=100.0, eva_weight=0.001, eva_boundary_weight=100, eva_string_num_weight=0.001,
                 eva_binary_weight = 10, eva_min_num_strings = 20, string_repulsion_weight=0.001, max_radius=0.1, path_repulsion_weight=0.001, z_repulsion_weight=0.001, min_dist=1e-3, domain_size=2):
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
        self.eva_weight = eva_weight
        # self.order_weight = order_weight
        self.min_dist = min_dist
        self.domain_size = domain_size
        self.eva_boundary_weight = eva_boundary_weight
        self.eva_string_num_weight = eva_string_num_weight
        self.eva_binary_weight = eva_binary_weight
        self.eva_min_num_strings = eva_min_num_strings
        self.max_radius = max_radius
    
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
    
    
    def compute_string_repulsion_penalty(self, string_xy, string_weights=None):
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
        if string_weights is not None:
            # string_probs = torch.sigmoid(string_weights)
            string_probs = string_weights
        for i in range(total_strings):
            for j in range(i + 1, total_strings):
                if string_weights is None:
                    dist_sq = torch.sum((string_xy[i] - string_xy[j]) ** 2)
                    string_repulsion += 1.0/ (dist_sq + self.min_dist)
                else:
                    dist_sq = torch.sum(((string_xy[i] - string_xy[j]) ** 2))
                    string_repulsion += string_probs[i] * string_probs[j] / (dist_sq + self.min_dist)
                
        
        return self.string_repulsion_weight*string_repulsion/total_strings
    
    def compute_local_string_repulsion_penalty(self, string_xy, string_weights=None):
        """
        Compute repulsion penalty between strings, but only for pairs within a given radius.
        Parameters:
        -----------
        string_xy : torch.Tensor (n_strings, 2)
            The 2D positions of the strings.
        string_weights : torch.Tensor (n_strings,) or None
            Optional weights for each string.
        radius : float
            Only pairs within this distance are considered.
        Returns:
        --------
        torch.Tensor
            The local string repulsion penalty value (weighted).
        """
        if string_xy is None:
            return torch.tensor(0.0, device=self.device)
        n = string_xy.shape[0]
        # Compute pairwise squared distances
        diff = string_xy.unsqueeze(1) - string_xy.unsqueeze(0)  # (n, n, 2)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (n, n)
        # Mask: ignore self-pairs and pairs outside radius
        mask = (dist_sq > 0) & (dist_sq < self.max_radius ** 2)
        repulsion = 0.0
        if string_weights is not None:
            string_probs = string_weights
            # Outer product for all pairs
            weight_matrix = string_probs.unsqueeze(1) * string_probs.unsqueeze(0)  # (n, n)
            repulsion_matrix = torch.zeros_like(dist_sq)
            repulsion_matrix[mask] = weight_matrix[mask] / (dist_sq[mask] + self.min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        else:
            repulsion_matrix = torch.zeros_like(dist_sq)
            repulsion_matrix[mask] = 1.0 / (dist_sq[mask] + self.min_dist)
            repulsion = torch.sum(repulsion_matrix) / n
        return self.string_repulsion_weight * repulsion
        
    
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
    
    def string_weights_penalty(self, string_weights):
        """
        Compute penalty for string weights to balance the amount of active strings.
        
        Parameters:
        -----------
        string_weights : torch.Tensor
            The weights of the strings to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The string weights penalty value (weighted).
        """
        # string_probs = torch.sigmoid(string_weights)
        string_probs = string_weights
        return self.eva_weight * torch.sum(torch.sqrt(string_probs)) / len(string_probs)
    
    def string_weights_boundary_penalty(self, string_weights):
        """
        Compute boundary penalty for string weights to keep them within a [0,1].
        
        Parameters:
        -----------
        string_weights : torch.Tensor
            The weights of the strings to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The string weights boundary penalty value (weighted).
        """
        return self.eva_boundary_weight * torch.mean((string_weights - torch.clamp(
            string_weights, min=0.0, max=0.8
        )) ** 2)
    
    
    def string_number_penalty(self, string_weights):
        """
        Compute penalty for the number of strings to keep the number of active strings balanced.
        
        Parameters:
        -----------
        string_weights : torch.Tensor
            The weights of the strings to compute the penalty for.
            
        Returns:
        --------
        torch.Tensor
            The string number penalty value (weighted).
        """
        # string_probs = torch.sigmoid(string_weights)
        string_probs = string_weights
        return self.eva_string_num_weight * F.softplus(torch.sum(string_probs) - self.eva_min_num_strings)
    
    def weight_binarization_penalty(self, string_weights):
        """
        Compute penalty for string weights to encourage binarization.
        
        Parameters:
        -----------
        string_weights : torch.Tensor
            The weights of the strings to compute the penalty for.
        threshold : float
            Threshold for binarization.
            
        Returns:
        --------
        torch.Tensor
            The binarization penalty value (weighted).
        """
        string_weights_cut = torch.clamp(string_weights, min=0.0, max=1.0)
        return self.eva_binary_weight * torch.sum(-string_weights_cut * torch.log(string_weights_cut + 1e-10) - (1 - string_weights_cut) * torch.log(1 - string_weights_cut + 1e-10))


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

class WeightedSNRLoss(LossFunction):
    """Loss function for weighted SNR optimization per string."""
    
    def __init__(self, device=None, repulsion_weight=0.001, boundary_weight=1.0, eva_string_num_weight=0.001,
                 eva_binary_weight = 10, eva_min_num_strings=20, string_repulsion_weight=0.0005, max_local_rad = 0.1, path_repulsion_weight=0.001,
                 z_repulsion_weight=0.001, eva_weight=0.001, eva_boundary_weight=10, min_dist=1e-3, domain_size=2, 
                 snr_weight=1.0, signal_scale=1.0, background_scale=10.0, no_background=False, conflict_free=False, use_llr=False, llr_model=None, llr_noise=0.1,
                 print_loss=False, num_bkg_samples=100, num_signal_samples=100):
        """
        Initialize the weighted SNR loss function.
        
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
        eva_weight : float
            Weight for evanescent string weights penalty.
        min_dist : float
            Minimum distance threshold.
        domain_size : float
            Size of the domain.
        snr_weight : float
            Weight for the SNR loss term.
        signal_scale : float
            Scaling factor for signal values.
        background_scale : float
            Scaling factor for background values.
        no_background : bool
            Whether to use background functions or not.
        """
        super().__init__(device, repulsion_weight, boundary_weight, eva_weight, eva_boundary_weight, 
                         eva_string_num_weight, eva_binary_weight, eva_min_num_strings,
                         string_repulsion_weight, max_local_rad, path_repulsion_weight,
                         z_repulsion_weight, min_dist, domain_size)
        self.snr_weight = snr_weight
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.no_background = no_background
        self.conflict_free = conflict_free
        self.use_llr = use_llr
        self.llr_model = llr_model
        self.llr_noise = llr_noise
        self.print_loss = print_loss
        self.num_bkg_samples = num_bkg_samples
        self.num_signal_samples = num_signal_samples
    
    def compute_snr_per_string(self, points_3d, signal_funcs, background_funcs,
                               n_strings, string_xy, points_per_string_list=None):
        """
        Compute SNR for each string separately.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Points in 3D space (total_points, 3).
        signal_funcs : list of callable
            List of signal functions.
        background_funcs : list of callable
            List of background functions.
        string_indices : list
            Indices indicating which string each point belongs to.
        points_per_string_list : list
            Number of points on each string.
        n_strings : int
            Total number of strings.
            
        Returns:
        --------
        torch.Tensor
            SNR values for each string (n_strings, n_signal_funcs).
        """
        snr_per_string = torch.zeros(n_strings, device=self.device)
        
        # Compute SNR for each string
        current_idx = 0
        # print(f"Computing SNR for {n_strings} strings with {n_signal_funcs} signal functions")
        for s_idx in range(int(n_strings)):
            # n_pts = points_per_string_list[s_idx] if points_per_string_list is not None else int(n_strings/len(points_3d))
            # print(f"String {s_idx}: {n_pts} points")
            # if n_pts > 0:
            # Get points for this string
            # string_points = points_3d[current_idx:current_idx + n_pts]
            # print(string_points)
            
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]
            # print(f"String {s_idx} points: {string_points}")
            if not self.use_llr:
                if not self.no_background:
                    background_sum = torch.zeros(len(string_points), device=self.device)
                    for i, func in enumerate(background_funcs):
                        background_sum += torch.tensor(func(string_points)) * self.background_scale
                    background_sum /= len(background_funcs)
                else:
                    background_sum = torch.tensor(1.0, device=self.device) * self.background_scale  # Use a constant value for background if no_background is True
                # Compute signal for this string
                
                for func_idx, signal_func in enumerate(signal_funcs):
                    signal_sum = torch.tensor(signal_func(string_points)) * self.signal_scale
                    # print(signal_sum, flush=True)
                    
                    # Compute SNR with epsilon to avoid division by zero
                    epsilon = 1e-10 if not self.no_background else 0
                    snr_per_string[s_idx] += torch.sum(signal_sum / torch.sqrt(background_sum + epsilon))
                snr_per_string[s_idx] /= len(signal_funcs)
            else: # Use LLR model to compute "SNR"
                total_signal_light_yield = torch.zeros(string_points.shape[0], device=self.device)
                total_background_light_yield = torch.zeros(string_points.shape[0], device=self.device)
                for signal_func in signal_funcs:
                    total_signal_light_yield += signal_func(string_points) * self.signal_scale
                total_signal_light_yield /= len(signal_funcs)
                for background_func in background_funcs:
                    total_background_light_yield += background_func(string_points) * self.background_scale
                total_background_light_yield /= len(background_funcs)
                for i in range(total_signal_light_yield.shape[0]):
                    signal_samples = np.random.normal(
                        loc=total_signal_light_yield[i].item(),
                        scale=self.llr_noise,
                        size=self.num_signal_samples
                    )
                    background_samples = np.random.normal(
                        loc=total_background_light_yield[i].item(),
                        scale=self.llr_noise,
                        size=self.num_bkg_samples
                    )
                    signal_samples = np.clip(signal_samples, 0, None)
                    background_samples = np.clip(background_samples, 0, None)
                    total_samples = torch.tensor(np.concatenate([signal_samples, background_samples]), device=self.device)
                    # print(total_samples)
                    test_string = torch.cat([string_points[i].unsqueeze(0).repeat(total_samples.shape[0], 1), total_samples.unsqueeze(1)], dim=1)
                    test_string = test_string.to(torch.float32)
                    expected_llr = torch.mean(self.llr_model(
                        test_string
                    ))
                    snr_per_string[s_idx] += expected_llr
                # current_idx += n_pts
        # time.sleep(2)
        if not self.use_llr:
            print(f"SNR per string: {snr_per_string}")
        else:
            print(f"Expected LLR per string: {snr_per_string}")
        
        return snr_per_string
    
    def __call__(self, points_3d, signal_funcs, background_funcs, string_xy=None,
                 points_per_string_list=None, string_weights=None,
                 path_positions=None, num_strings=None, precomputed_snr_per_string=None, **kwargs):
        """
        Compute the weighted SNR loss.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Points in 3D space (total_points, 3).
        signal_funcs : list of callable
            List of signal functions.
        background_funcs : list of callable
            List of background functions.
        string_xy : torch.Tensor or None
            XY positions of strings if optimizing them.
        points_per_string_list : list or None
            Number of points on each string.
        string_indices : list or None
            Indices indicating which string each point belongs to.
        string_weights : torch.Tensor or None
            Weights for each string for weighted averaging.
        path_positions : torch.Tensor or None
            Positions along the continuous path for path-based optimization.
        num_strings : int or None
            Number of strings in the geometry.
        precomputed_snr_per_string : torch.Tensor or None
            Precomputed SNR values per string (n_strings, n_signal_funcs).
            If provided, computation is skipped.
        **kwargs : dict
            Additional arguments specific to the loss function.
            
        Returns:
        --------
        tuple
            (total_loss, weighted_avg_snr_no_penalties, snr_per_string)
            - total_loss: Loss with all penalties applied
            - weighted_avg_snr_no_penalties: Weighted average SNR without penalties
            - snr_per_string: SNR values for each string (n_strings, n_signal_funcs)
        """
        # Determine number of strings
        if num_strings is None:
            if points_per_string_list is not None:
                num_strings = len(points_per_string_list)
            elif string_weights is not None:
                num_strings = len(string_weights)
            else:
                raise ValueError("Cannot determine number of strings")
        
        # Use precomputed SNR if provided, otherwise compute it
        if precomputed_snr_per_string is not None:
            snr_per_string = precomputed_snr_per_string
        else:
            snr_per_string = self.compute_snr_per_string(
                points_3d, signal_funcs, background_funcs, num_strings, string_xy, points_per_string_list
            )
        # print(f"SNR per string: {snr_per_string}")
        # Apply string weights if provided
        if string_weights is not None:
            # Apply sigmoid to get probabilities
            # string_probs = torch.sigmoid(string_weights)
            string_probs = string_weights
            clamped_string_probs = torch.sigmoid(string_probs)  # Apply clamp to string weights
            # Normalize weights to sum to 1
            # snr_per_string: (n_strings, n_signal_funcs)
            # normalized_weights: (n_strings,)
            no_norm_weighted_snr_per_string = torch.sum(snr_per_string * clamped_string_probs, dim=0)/torch.mean(snr_per_string, dim=0, keepdim=True)  # (n_signal_funcs,)
            # print(snr_per_string*clamped_string_probs)
        else:
            # If no weights provided, use simple average
            
            # weighted_snr_per_func = torch.mean(snr_per_string/torch.mean(snr_per_string, dim=0, keepdim=True))  # (n_signal_funcs,)
            no_norm_weighted_snr_per_string = torch.mean(snr_per_string, dim=0)  # (n_signal_funcs,)
        # Average across all signal functions
        # weighted_avg_snr = torch.mean(weighted_snr_per_func)
        no_norm_weighted_avg_snr = torch.mean(no_norm_weighted_snr_per_string)
        
        # SNR loss (negative since we want to maximize SNR)
        
        snr_loss =  self.snr_weight / (no_norm_weighted_avg_snr + 1e-10)
        # print(f"SNR loss: {snr_loss.item()}")
        # Initialize total loss with SNR loss
        total_loss = snr_loss
        if self.conflict_free:
            total_loss = [snr_loss]
            if self.print_loss:
                print(f"weighted SNR: {total_loss[0].item()}")
        elif self.print_loss:
            print(f'weighted snr: {total_loss.item()}')
        
        # Compute penalties
        if self.string_repulsion_weight > 0 and string_xy is not None and string_weights is not None:
            string_repulsion_penalty = self.compute_local_string_repulsion_penalty(string_xy, clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_repulsion_penalty)
            else:    
                total_loss += string_repulsion_penalty
            if self.print_loss:
                print(f"String repulsion penalty: {string_repulsion_penalty}")
        
        if self.eva_boundary_weight > 0 and string_weights is not None:
            string_weights_boundary_penalty = self.string_weights_boundary_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_weights_boundary_penalty)
            else:
                total_loss += string_weights_boundary_penalty
            if self.print_loss:
                print(f"String weights boundary penalty: {string_weights_boundary_penalty}")
            
        if self.eva_string_num_weight > 0 and string_weights is not None:
            string_number_penalty = self.string_number_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_number_penalty)
            else:
                total_loss += string_number_penalty
            if self.print_loss:
                print(f"String number penalty: {string_number_penalty}")
            
        if self.eva_binary_weight > 0 and string_weights is not None:
            binarization_penalty = self.weight_binarization_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(binarization_penalty)
            else:
                total_loss += binarization_penalty
            if self.print_loss:
                print(f"Binarization penalty: {binarization_penalty}")

        # if path_positions is not None and self.path_repulsion_weight > 0:
        #     if num_strings is None:
        #         num_strings = len(points_per_string_list)
        #     path_repulsion_penalty = self.path_repulsion_penalty(path_positions, num_strings)
        #     total_loss += path_repulsion_penalty
        
        # if self.repulsion_weight > 0:
        #     repulsion_penalty = self.compute_repulsion_penalty(points_3d)
        #     total_loss += repulsion_penalty
        
        # if self.z_repulsion_weight > 0:
        #     z_repulsion_penalty = self.z_dist_penalty(points_3d)
        #     total_loss += z_repulsion_penalty
        
        # if self.boundary_weight > 0:
        #     boundary_penalty = self.compute_boundary_penalty(points_3d)
        #     total_loss += boundary_penalty
        
        # Add string weights penalty if using evanescent strings
        if string_weights is not None and self.eva_weight > 0:
            string_weights_penalty = self.string_weights_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_weights_penalty)
            else:
                total_loss += string_weights_penalty
            if self.print_loss:
                print(f"String weights penalty: {string_weights_penalty}")
        if self.print_loss:    
            if self.conflict_free:
                print(f"Total loss: {torch.stack(total_loss).sum().item()}")
            else:
                print(f"Total loss: {total_loss.item()}")

        
        return total_loss, no_norm_weighted_avg_snr.item(), snr_per_string



class WeightedLLRLoss(LossFunction):
    """Loss function for weighted LLR optimization using surrogate functions and LLRnet or KLNet.
    
    This loss function supports two modes of operation:
    1. Traditional mode: Uses LLRnet with surrogate functions to generate events and compute LLR
    2. KLNet mode: Uses a pre-trained KLNet model to directly predict expected LLR values
    
    KLNet mode is more efficient as it bypasses event generation and directly predicts 
    expected LLR values at detector positions given event parameters.
    

    """
    
    def __init__(self, device=None, repulsion_weight=0.001, boundary_weight=1.0, eva_string_num_weight=0.001,
                 eva_binary_weight=10, eva_min_num_strings=20, string_repulsion_weight=0.0005, max_local_rad=0.1, 
                 path_repulsion_weight=0.001, z_repulsion_weight=0.001, eva_weight=0.001, eva_boundary_weight=10, 
                 min_dist=1e-3, domain_size=2, llr_weight=1.0, signal_scale=1.0, background_scale=1.0, 
                 add_noise=True, sig_noise_scale=0.1, bkg_noise_scale=0.1, num_samples=100,
                 conflict_free=False, print_loss=False, llr_net=None, signal_surrogate_func=None, 
                 background_surrogate_func=None, signal_event_params=None, background_event_params=None,
                 batch_size_per_point=32, random_seed=None, use_klnet=False, keep_opt_point=True,
                 klnet_model=None, signal_ratio = 0.5, use_bce_loss=False, boost_signal=True, signal_boost_weight=1.0,
                 boost_signal_yield=False, signal_yield_boost_weight=1.0, compute_fisher_info=False, 
                 fisher_info_weight=1.0):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        repulsion_weight : float
            Weight for repulsion penalty between points on the same string.
        boundary_weight : float
            Weight for boundary penalty.
        eva_string_num_weight : float
            Weight for controlling number of active strings.
        eva_binary_weight : float
            Weight for encouraging binary string weights.
        eva_min_num_strings : int
            Minimum number of active strings.
        string_repulsion_weight : float
            Weight for repulsion between strings.
        max_local_rad : float
            Maximum radius for local string repulsion.
        path_repulsion_weight : float
            Weight for repulsion in path space.
        z_repulsion_weight : float
            Weight for z distance penalty.
        eva_weight : float
            Weight for evanescent string weights penalty.
        eva_boundary_weight : float
            Weight for string weights boundary penalty.
        min_dist : float
            Minimum distance threshold.
        domain_size : float
            Size of the domain.
        llr_weight : float
            Weight for the LLR loss term.
        signal_scale : float
            Scaling factor for signal values.
        background_scale : float
            Scaling factor for background values.
        add_noise : bool
            Whether to add noise to light yield values.
        noise_scale : float
            Standard deviation of noise to add to light yields.
        num_signal_samples : int
            Number of signal samples to generate per evaluation point.
        num_background_samples : int
            Number of background samples to generate per evaluation point.
        conflict_free : bool
            Whether to return losses as separate terms for multi-objective optimization.
        print_loss : bool
            Whether to print loss components during computation.
        llr_net : LLRnet or None
            Trained LLR network for computing log-likelihood ratios.
        signal_surrogate_func : callable or None
            Function that computes signal light yield from event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
        signal_event_params : dict or None
            Dictionary containing signal event parameters.
        background_event_params : dict or None
            Dictionary containing background event parameters.
        batch_size_per_string : int
            Number of samples to generate per string for LLR computation.
        random_seed : int or None
            Random seed for reproducibility.
        use_klnet : bool
            Whether to use KLNet for expected LLR computation instead of traditional method.
        klnet_model : KLnet or None
            Trained KLNet model for computing expected LLR values directly.
        compute_fisher_info : bool
            Whether to compute Fisher Information with respect to signal light yield.
        fisher_info_weight : float
            Weight for Fisher Information term in the loss function.
        """
        super().__init__(device, repulsion_weight, boundary_weight, eva_weight, eva_boundary_weight, 
                         eva_string_num_weight, eva_binary_weight, eva_min_num_strings,
                         string_repulsion_weight, max_local_rad, path_repulsion_weight,
                         z_repulsion_weight, min_dist, domain_size)
        
        self.llr_weight = llr_weight
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.add_noise = add_noise
        self.sig_noise_scale = sig_noise_scale
        self.bkg_noise_scale = bkg_noise_scale
        self.total_samples_per_point = num_samples
        self.conflict_free = conflict_free
        self.print_loss = print_loss
        self.batch_size_per_point = batch_size_per_point
        self.random_seed = random_seed
        self.keep_opt_point = keep_opt_point
        
        # LLR computation components
        self.llr_net = llr_net
        self.signal_surrogate_func = signal_surrogate_func
        self.background_surrogate_func = background_surrogate_func
        self.signal_event_params = signal_event_params
        self.background_event_params = background_event_params
       
        self.use_klnet = use_klnet
        self.klnet_model = klnet_model
        self.signal_ratio = signal_ratio
        self.use_bce_loss = use_bce_loss
        self.boost_signal = boost_signal
        self.signal_boost_weight = signal_boost_weight
        self.boost_signal_yield = boost_signal_yield
        self.signal_yield_boost_weight = signal_yield_boost_weight
        self.compute_fisher_info = compute_fisher_info
        self.fisher_info_weight = fisher_info_weight
        self.fisher_info_params = ['energy', 'azimuth', 'zenith']  # Default parameters for Fisher Info

        # Validate inputs
        if not self.use_klnet:
            if self.llr_net is None:
                raise ValueError("llr_net must be provided and trained when not using KLNet")
            if not self.llr_net.is_trained:
                raise ValueError("llr_net must be trained before use when not using KLNet")
            if self.signal_surrogate_func is None or self.background_surrogate_func is None:
                raise ValueError("Both signal_surrogate_func and background_surrogate_func must be provided when not using KLNet")
            if self.signal_event_params is None or self.background_event_params is None:
                raise ValueError("Both signal_event_params and background_event_params must be provided when not using KLNet")
        else:
            if self.klnet_model is None:
                raise ValueError("klnet_model must be provided when use_klnet=True")
            if not self.klnet_model.is_trained:
                raise ValueError("klnet_model must be trained before use")
            if self.signal_event_params is None or self.background_event_params is None:
                raise ValueError("Both signal_event_params and background_event_params must be provided for KLNet")
    
    def compute_llr_per_string(self, string_xy, n_strings, points_3d, compute_snr=False, compute_signal_yield = False,
                              signal_surrogate_func=None, background_surrogate_func=None, 
                              num_samples_per_point=100, compute_true_llr=False, signal_x_bias=0, background_x_bias=0, zenith_bias=1.5):
        """
        Compute expected LLR for each string using surrogate functions and noise modeling.
        Optionally compute SNR per string as well.
        
        Parameters:
        -----------
        string_xy : torch.Tensor
            XY positions of strings, shape (n_strings, 2).
        n_strings : int
            Number of strings.
        points_3d : torch.Tensor
            Points in 3D space for the geometry.
        compute_snr : bool
            Whether to also compute SNR per string using surrogate functions.
        signal_surrogate_func : callable or None
            Light yield surrogate function for signal events. Required if compute_snr=True.
        background_surrogate_func : callable or None
            Light yield surrogate function for background events. Required if compute_snr=True.
        num_samples_per_point : int
            Number of event samples to use per point when computing SNR.
            
        Returns:
        --------
        tuple
            (llr_per_string, signal_llr_per_string, background_llr_per_string, 
             snr_per_string, true_llr_per_string, true_signal_llr_per_string, 
             true_background_llr_per_string, signal_yield_per_string, fisher_info_per_string)
            - llr_per_string: Expected LLR values for each string, shape (n_strings,)
            - signal_llr_per_string: Signal LLR values for each string
            - background_llr_per_string: Background LLR values for each string  
            - snr_per_string: SNR values for each string (if compute_snr=True, else None)
            - true_llr_per_string: True LLR values (if compute_true_llr=True, else None)
            - true_signal_llr_per_string: True signal LLR values (if compute_true_llr=True, else None)
            - true_background_llr_per_string: True background LLR values (if compute_true_llr=True, else None)
            - signal_yield_per_string: Signal light yield values (if compute_signal_yield=True, else None)
            - fisher_info_per_string: Fisher Information values (if compute_fisher_info=True, else None)
        """
        # Validate SNR computation inputs
        if compute_snr:
            if signal_surrogate_func is None or background_surrogate_func is None:
                raise ValueError("Both signal_surrogate_func and background_surrogate_func must be provided when compute_snr=True")
        llr_per_string = torch.zeros(n_strings, device=self.device)
        snr_per_string = torch.zeros(n_strings, device=self.device) if compute_snr else None
        signal_yield_per_string = torch.zeros(n_strings, device=self.device) if compute_signal_yield else None
        true_llr_per_string = torch.zeros(n_strings, device=self.device) if compute_true_llr else None
        # Track signal and background LLR separately for loss computation
        signal_llr_per_string = torch.zeros(n_strings, device=self.device)
        background_llr_per_string = torch.zeros(n_strings, device=self.device)
        true_signal_llr_per_string = torch.zeros(n_strings, device=self.device) if compute_true_llr else None
        true_background_llr_per_string = torch.zeros(n_strings, device=self.device) if compute_true_llr else None
        
        # For BCE loss, we need to store predicted probabilities for signal and background separately
        signal_probs_per_string = torch.zeros(n_strings, device=self.device) if self.use_bce_loss else None
        background_probs_per_string = torch.zeros(n_strings, device=self.device) if self.use_bce_loss else None
        
        # For Fisher Information computation
        fisher_info_per_string = None
        if self.compute_fisher_info:
            n_params = len(self.fisher_info_params)
            fisher_info_per_string = torch.zeros(n_strings, n_params, n_params, device=self.device)
        # Set random seed for reproducibility if provided
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        if not self.use_klnet: 
            print(f"Computing LLR for {n_strings} strings with {self.total_samples_per_point} samples per point")
        else:
            print(f"Computing expected LLR for {n_strings} strings with {self.total_samples_per_point} samples per string using KLNet")
        
        # Cache surrogate functions for Fisher Information computation
        if self.compute_fisher_info:
            self._cached_signal_surrogate_func = signal_surrogate_func
            self._cached_background_surrogate_func = background_surrogate_func
        
        for s_idx in range(n_strings):
            # Create optimization points for this string
            # Sample points along the string (assuming strings extend in z-direction)
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]
            
            # Generate batch data using LLRnet's method
            # if not self.use_klnet:
            total_llr = 0.0
            total_signal_llr = 0.0
            total_background_llr = 0.0
            total_signal_prob = 0.0
            total_background_prob = 0.0
           
            total_true_llr = 0.0
            total_true_llr_signal = 0.0
            total_true_llr_background = 0.0
        
            for point in string_points:    
                # CORRECTED APPROACH: Generate mixed batches like in training
                # This computes the expected LLR by evaluating how the network responds
                # to a realistic mix of signal and background events at detector positions
                
                mixed_data_loader = self.llr_net.generate_batch_data_from_events(
                    optimization_points=[point],
                    signal_surrogate_func=self.signal_surrogate_func,
                    background_surrogate_func=self.background_surrogate_func,
                    signal_event_params=self.signal_event_params,
                    background_event_params=self.background_event_params,
                    batch_size=self.batch_size_per_point,
                    signal_ratio=self.signal_ratio,  # Use the configured signal ratio
                    add_noise=self.add_noise,
                    sig_noise_scale=self.sig_noise_scale,
                    bkg_noise_scale=self.bkg_noise_scale,
                    random_seed=self.random_seed,
                    keep_opt_point=self.keep_opt_point,
                    output_denoised=compute_true_llr,
                    samples_per_epoch=self.total_samples_per_point,
                    return_event_params=self.compute_fisher_info  # Get event params for Fisher Info
                )
                
                # Compute expected LLR for this string
               
                num_batches = 0
                batch_fisher_matrices = []
                with torch.no_grad():
                    for info in mixed_data_loader:
                        if self.compute_fisher_info:
                            if not compute_true_llr:
                                features, labels, event_params_batch = info
                            else:
                                features, labels, denoised_yield, event_params_batch = info
                        else:
                            if not compute_true_llr:
                                features, labels = info
                            else:
                                features, labels, denoised_yield = info
                        # Compute LLR for this mixed batch
                        batch_llr = self.llr_net.predict_log_likelihood_ratio(features)
                        
                        # Separate LLR by true labels for signal/background tracking
                        signal_mask = labels == 1
                        background_mask = labels == 0
                        
                        if signal_mask.any():
                            total_signal_llr += torch.mean(batch_llr[signal_mask]).item()
                        if background_mask.any():
                            total_background_llr += torch.mean(batch_llr[background_mask]).item()
                        
                        if self.use_bce_loss:
                            batch_pred = self.llr_net.predict_proba(features)
                            
                            if signal_mask.any():
                                total_signal_prob += torch.mean(batch_pred[signal_mask]).item()
                            if background_mask.any():
                                total_background_prob += torch.mean(batch_pred[background_mask]).item()
                        
                        # Expected LLR for this batch (weighted by signal_ratio)
                        # This correctly reflects how signal vs background events 
                        # are distinguished at this detector position
                        total_llr += torch.mean(batch_llr).item()
                        
                        # Compute Fisher Information matrix if requested
                        if self.compute_fisher_info and event_params_batch is not None:
                            # Filter event parameters for signal events only
                            signal_indices = torch.where(signal_mask)[0]
                            signal_event_params = [event_params_batch[i.item()] for i in signal_indices]
                            with torch.enable_grad():
                                batch_fisher_matrix = self._compute_fisher_information(signal_event_params, point)
                            batch_fisher_matrices.append(batch_fisher_matrix)
                        if compute_true_llr:
                            features = np.array(features).squeeze()
                            denoised_yield = denoised_yield.cpu().numpy().squeeze()
                            signal_mean_denoised_yield = np.mean(denoised_yield[signal_mask.cpu().numpy()])
                            background_mean_denoised_yield = np.mean(denoised_yield[background_mask.cpu().numpy()])
                            temp_total_true_llr = np.zeros_like(batch_llr)   
                            temp_total_true_llr += np.log(features[:, -2]) # energy llr
                            temp_total_true_llr += np.log((2+zenith_bias) / (1 + zenith_bias * (1 - np.abs(np.cos(features[:, -4]))))) # zenith llr
                            temp_total_true_llr += np.log(self.bkg_noise_scale/self.sig_noise_scale) # light yield noise llr
                            temp_total_true_llr += 0.5*(np.log(features[:,-1]) - np.log(background_mean_denoised_yield))**2 / self.bkg_noise_scale**2
                            temp_total_true_llr -= 0.5*(np.log(features[:,-1]) - np.log(signal_mean_denoised_yield))**2 / self.sig_noise_scale**2
                            if signal_x_bias != 0 or background_x_bias != 0: # spatial bias llr
                                temp_total_true_llr += (signal_x_bias - background_x_bias)*(features[:,0] + features[:,3]) + (background_x_bias**2 - signal_x_bias**2)/2
                            total_true_llr += np.mean(temp_total_true_llr)
                            total_true_llr_background += np.mean(temp_total_true_llr[background_mask])
                            total_true_llr_signal += np.mean(temp_total_true_llr[signal_mask])
                            

                        num_batches += 1
                        
                        

                # Average expected LLR across all batches for this string
                if num_batches > 0:
                    llr_per_string[s_idx] += total_llr / num_batches
                    signal_llr_per_string[s_idx] += total_signal_llr / num_batches
                    background_llr_per_string[s_idx] += total_background_llr / num_batches
                    if self.use_bce_loss:
                        signal_probs_per_string[s_idx] += total_signal_prob / num_batches
                        background_probs_per_string[s_idx] += total_background_prob / num_batches
                    if compute_true_llr:
                        true_llr_per_string[s_idx] += total_true_llr / num_batches
                        true_signal_llr_per_string[s_idx] += total_true_llr_signal / num_batches
                        true_background_llr_per_string[s_idx] += total_true_llr_background / num_batches
                    if self.compute_fisher_info and batch_fisher_matrices:
                        # Average Fisher Information matrices across batches
                        avg_fisher_matrix = torch.stack(batch_fisher_matrices).mean(dim=0)
                        fisher_info_per_string[s_idx] = avg_fisher_matrix
               
                
            # llr_per_string[s_idx] /=len(string_points)
            # signal_llr_per_string[s_idx] /=len(string_points)
            # background_llr_per_string[s_idx] /=len(string_points)
            # if self.use_bce_loss:
            #     signal_probs_per_string[s_idx] /=len(string_points)
            #     background_probs_per_string[s_idx] /=len(string_points)
            # if compute_true_llr:
            #     true_llr_per_string[s_idx] /=len(string_points)
            #     true_signal_llr_per_string[s_idx] /=len(string_points)
            #     true_background_llr_per_string[s_idx] /=len(string_points)

                # except Exception as e:
                #     print(f"Error computing LLR for string {s_idx}: {e}")
                #     llr_per_string[s_idx] = 0.0
                #     signal_llr_per_string[s_idx] = 0.0
                #     background_llr_per_string[s_idx] = 0.0
                #     if self.use_bce_loss:
                #         signal_probs_per_string[s_idx] = 0.0
                #         background_probs_per_string[s_idx] = 0.0
            #     #     if compute_true_llr:
            #     #         true_llr_per_string[s_idx] = 0.0
            #     #         true_signal_llr_per_string[s_idx] = 0.0
            #     #         true_background_llr_per_string[s_idx] = 0.0

            # else:
            #     # Use expected LLR computation or KLNet
            #     # try:
            #     # Create mixed event parameters for combined signal/background events
            #     batch_event_params = self._create_mixed_event_batch(
            #         self.signal_event_params, self.background_event_params, 
            #         self.batch_size_per_string, signal_ratio=self.signal_ratio
            #     )
                
            #     # Predict expected LLR for this string position using KLNet
            #     expected_llr = self.klnet_model.predict_expected_llr(
            #         string_points, batch_event_params
            #     )
            #     # print(expected_llr)
            #     # print(expected_llr.shape)
            #     # Sum expected LLR across all points on the string
            #     # (each point already represents average over multiple events)
            #     if expected_llr.dim() > 1:
            #         expected_llr = expected_llr.mean(dim=1)  # average over events
            #     else:    
            #         new_expected_llr = torch.zeros(len(string_points), device=self.device)
            #         batch_len = self.batch_size_per_string
            #         for string_point_idx in range(string_points.shape[0]):
            #             new_expected_llr[string_point_idx] += expected_llr[batch_len*string_point_idx:batch_len*(string_point_idx+1)].mean()
            #         expected_llr = new_expected_llr
            #     print(f"Expected LLR: {expected_llr}")
            #     llr_per_string[s_idx] = expected_llr.sum().item()  # Average LLR for this string
            #     # except Exception as e:
            #     #     print(f"Error computing KLNet LLR for string {s_idx}: {e}")
            #     #     llr_per_string[s_idx] = 0.0
            
            # Compute SNR if requested
            if compute_snr or compute_signal_yield:
                
                    # Sample events for SNR computation using the same event parameters as LLR
                signal_indices = torch.randint(0, len(self.signal_event_params['position']), 
                                                (int(self.total_samples_per_point*0.5),), device=self.device)
                background_indices = torch.randint(0, len(self.background_event_params['position']), 
                                                    (self.total_samples_per_point-int(self.total_samples_per_point*0.5),), device=self.device)

                snr_for_string = torch.zeros(len(string_points), device=self.device)
                signal_yield_for_string = torch.zeros(len(string_points), device=self.device)
                
                for point_idx, point in enumerate(string_points):
                    # Compute signal light yield for this point
                    signal_yields = []
                    for sample_idx in range(len(signal_indices)):
                        event_idx = signal_indices[sample_idx].item()
                        signal_yield = signal_surrogate_func(
                            self.signal_event_params, point, event_idx
                        )
                        if self.add_noise:
                            signal_yield = signal_yield + signal_yield*torch.randn(size=signal_yield.shape, device=self.device) * self.sig_noise_scale
                        
                        signal_yields.append(signal_yield * self.signal_scale)
                    
                    # Compute background light yield for this point
                    background_yields = []
                    for sample_idx in range(len(background_indices)):
                        event_idx = background_indices[sample_idx].item()
                        background_yield = background_surrogate_func(
                            self.background_event_params, point, event_idx
                        )
                        if self.add_noise:
                            background_yield = background_yield + background_yield*torch.randn(size=background_yield.shape, device=self.device) * self.bkg_noise_scale
                        background_yields.append(background_yield * self.background_scale)
                    
                    # Convert to tensors and compute average
                    signal_yields = torch.stack(signal_yields) if signal_yields else torch.zeros(1, device=self.device)
                    background_yields = torch.stack(background_yields) if background_yields else torch.ones(1, device=self.device)
                    
                    avg_signal = torch.mean(signal_yields)
                    avg_background = torch.mean(background_yields)
                    
                    # Compute SNR: signal / sqrt(background) with epsilon to avoid division by zero
                    epsilon = 1e-10
                    snr_for_string[point_idx] = avg_signal / torch.sqrt(avg_background + epsilon)
                    signal_yield_for_string[point_idx] = avg_signal
                # summed SNR across all points on this string
                if compute_snr:
                    snr_per_string[s_idx] = torch.sum(snr_for_string)
                if compute_signal_yield:
                    signal_yield_per_string[s_idx] = torch.sum(signal_yield_for_string)
                        
             
           
        if self.print_loss:
            print(f"Expected LLR per string: {llr_per_string}")
            print(f"Signal LLR per string: {signal_llr_per_string}")
            print(f"Background LLR per string: {background_llr_per_string}")
            if compute_snr:
                print(f"SNR per string: {snr_per_string}")
            if self.use_bce_loss:
                print(f"Signal probs per string: {signal_probs_per_string}")
                print(f"Background probs per string: {background_probs_per_string}")

       
        return llr_per_string, signal_llr_per_string, background_llr_per_string, snr_per_string, true_llr_per_string, true_signal_llr_per_string, true_background_llr_per_string, signal_yield_per_string, fisher_info_per_string
       
    
    # def _create_mixed_event_batch(self, signal_event_params, background_event_params, 
    #                               batch_size, signal_ratio=0.5):
    #     """
    #     Create a mixed batch of signal and background event parameters.
        
    #     Parameters:
    #     -----------
    #     signal_event_params : dict
    #         Signal event parameters
    #     background_event_params : dict
    #         Background event parameters
    #     batch_size : int
    #         Size of the batch to create
    #     signal_ratio : float
    #         Probability of selecting a signal event (vs background)
            
    #     Returns:
    #     --------
    #     dict : Mixed event parameters batch
    #     """
    #     # Set random seed for reproducibility if provided
    #     if self.random_seed is not None:
    #         torch.manual_seed(self.random_seed)
    #         np.random.seed(self.random_seed)
        
    #     # Determine number of signal and background events
    #     n_signal_events = int(batch_size * signal_ratio)
    #     n_background_events = batch_size - n_signal_events
        
    #     # Get available event indices
    #     signal_indices = list(range(len(signal_event_params['position'])))
    #     background_indices = list(range(len(background_event_params['position'])))
        
    #     # Randomly sample events
    #     if n_signal_events > 0:
    #         selected_signal_indices = np.random.choice(
    #             signal_indices, size=min(n_signal_events, len(signal_indices)), replace=True
    #         )
    #     else:
    #         selected_signal_indices = []
            
    #     if n_background_events > 0:
    #         selected_background_indices = np.random.choice(
    #             background_indices, size=min(n_background_events, len(background_indices)), replace=True
    #         )
    #     else:
    #         selected_background_indices = []
        
    #     # Create mixed batch
    #     mixed_params = {}
        
    #     # Combine parameters from selected events
    #     for key in signal_event_params.keys():
    #         signal_values = []
    #         background_values = []
            
    #         # Get signal values
    #         if len(selected_signal_indices) > 0:
    #             signal_values = [signal_event_params[key][i] for i in selected_signal_indices]
            
    #         # Get background values  
    #         if len(selected_background_indices) > 0:
    #             background_values = [background_event_params[key][i] for i in selected_background_indices]
            
    #         # Combine and shuffle
    #         all_values = signal_values + background_values
    #         if all_values:
    #             # Shuffle the combined list
    #             np.random.shuffle(all_values)
    #             mixed_params[key] = all_values
    #         else:
    #             mixed_params[key] = []
        
    #     return mixed_params

    def set_fisher_info_params(self, params=['energy', 'azimuth', 'zenith']):
        """
        Set which parameters to compute Fisher Information for.
        
        Parameters:
        -----------
        params : list
            List of parameter names to compute Fisher Information for.
            Available options: ['energy', 'azimuth', 'zenith', 'position']
        """
        self.fisher_info_params = params
        
    def set_signal_surrogate_func(self, signal_surrogate_func):
        """
        Set the signal surrogate function for Fisher Information computation.
        
        Parameters:
        -----------
        signal_surrogate_func : callable
            Signal surrogate function that takes (event_params, detector_point, event_idx)
            and returns light yield mean value.
        """
        self.signal_surrogate_func = signal_surrogate_func

    def _compute_fisher_information(self, event_params_batch, detector_point):
        """
        Compute Fisher Information matrix with respect to specified parameters using Poisson light yield.
        
        Assumes light yield follows a Poisson distribution where the surrogate function output
        represents the mean (λ). For Poisson distributions, Fisher Information matrix with respect
        to parameters θ is: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features for the batch
        labels : torch.Tensor  
            True labels (0=background, 1=signal)
        event_params_batch : list
            Event parameters corresponding to each feature in the batch
        llr_values : torch.Tensor
            Log-likelihood ratio values for the batch
        detector_point : torch.Tensor
            Current detector point being evaluated
            
        Returns:
        --------
        torch.Tensor
            Fisher Information matrix for this batch (n_params, n_params)
        """
        n_params = len(self.fisher_info_params)
        fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
        
        # if event_params_batch is None or len(event_params_batch) == 0:
        #     return fisher_matrix
            
      
        # # Extract signal events only (Fisher Info computed for signal parameters)
        # signal_mask = labels == 1
        # if not signal_mask.any():
        #     return fisher_matrix
            
        # # Get signal events
        # signal_indices = torch.where(signal_mask)[0]
        # if len(signal_indices) == 0:
        #     return fisher_matrix
            
        num_valid_events = 0
        
        # Compute Fisher Information for each signal event
        for batch_idx in range(len(event_params_batch)):
            
                
            event_params = event_params_batch[batch_idx]
            if event_params is None:
                continue
            
            # Extract parameter values for this event
            if not isinstance(event_params, dict):
                continue

            # Handle both dictionary and direct parameter formats
            if isinstance(event_params, dict):
                # Direct dictionary access
                energy = event_params.get('energy', torch.tensor(1.0, device=self.device))
                azimuth = event_params.get('azimuth', torch.tensor(0.0, device=self.device))
                zenith = event_params.get('zenith', torch.tensor(0.0, device=self.device))
                position = event_params.get('position', torch.zeros(3, device=self.device))
           
                
                # Ensure parameters require gradients
                grad_event_params = {}
                for param_name in self.fisher_info_params:
                  
                    if param_name == 'energy':
                        grad_event_params[param_name] = energy.clone().detach().requires_grad_(True)
                    elif param_name == 'azimuth':
                        grad_event_params[param_name] = azimuth.clone().detach().requires_grad_(True)
                    elif param_name == 'zenith':
                        grad_event_params[param_name] = zenith.clone().detach().requires_grad_(True)
                    else:
                        print(f'Using default value for {param_name}')
                        grad_event_params[param_name] = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Add position parameter
                grad_event_params['position'] = position.clone().detach()

                # Compute light yield mean (λ) using the signal surrogate function with gradients
                if hasattr(self, '_cached_signal_surrogate_func') and self._cached_signal_surrogate_func is not None:
                    light_yield_mean = self._cached_signal_surrogate_func(grad_event_params, detector_point, 0, using_gradient =True)
                elif hasattr(self, 'signal_surrogate_func') and self.signal_surrogate_func is not None:
                    light_yield_mean = self.signal_surrogate_func(grad_event_params, detector_point, 0, using_gradient=True)
                else:
                    # Skip Fisher Information computation if no surrogate function available
                    if self.print_loss and batch_idx == 0:  # Only print once per batch
                        print("Warning: No signal surrogate function available for Fisher Information")
                    continue
                    
                # Ensure light yield is positive for Poisson distribution
                # light_yield_mean = torch.clamp(light_yield_mean, min=1e-6)
                
                
                
                # Compute gradients with respect to each parameter
                param_gradients = []
                
                for param_name in self.fisher_info_params:
                    if param_name in grad_event_params:
                        param_tensor = grad_event_params[param_name]
                        
                        # Compute gradient ∂λ/∂θ
                     
                        grad_outputs = torch.ones_like(light_yield_mean)
                        # print(f'Computing gradient for {param_name} = {param_tensor}')
                        param_grad = torch.autograd.grad(
                            outputs=light_yield_mean,
                            inputs=param_tensor,
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=True,
                            only_inputs=True,
                            allow_unused=True
                        )[0]
                        # print(f'Param: {param_name}, Value: {param_tensor}, Grad: {param_grad}')
                        if param_grad is not None:
                            param_gradients.append(param_grad)
                        else:
                            param_gradients.append(torch.zeros_like(param_tensor))
                        # except Exception as grad_e:
                        #     if self.print_loss:
                        #         print(f"Warning: Gradient computation failed for {param_name}: {grad_e}")
                        #     param_gradients.append(torch.zeros_like(param_tensor))
                    else:
                        # Parameter not available, use zero gradient
                        print(f'Warning: Parameter {param_name} not found, using zero gradient.')
                        param_gradients.append(torch.tensor(0.0, device=self.device))
                
                # Compute Fisher Information matrix: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
                if len(param_gradients) == n_params:
                    for i_param in range(n_params):
                        for j_param in range(n_params):
                            try:
                                grad_i = param_gradients[i_param]
                                grad_j = param_gradients[j_param]
                                
                                # Fisher Information element: (∂λ/∂θ_i)(∂λ/∂θ_j)/λ
                                fisher_element = (grad_i * grad_j) / light_yield_mean
                                fisher_matrix[i_param, j_param] += fisher_element.item()
                            except Exception as e:
                                if self.print_loss:
                                    print(f"Warning: Fisher matrix element computation failed: {e}")
                                continue
                    
                    num_valid_events += 1
        
        # Average Fisher Information matrix across all valid signal events
        if num_valid_events > 0:
            fisher_matrix /= num_valid_events
            
        return fisher_matrix
                
       

    def __call__(self, string_xy=None, string_weights=None, num_strings=None, precomputed_llr_per_string=None,  precomputed_signal_yield_per_string=None,
                 points_3d=None, compute_snr=False, signal_surrogate_func=None, background_surrogate_func=None, precomputed_fisher_info_per_string=None,
                 num_samples_per_point=100, precomputed_signal_probs_per_string=None, 
                 precomputed_background_probs_per_string=None, precomputed_signal_llr=None, precomputed_background_llr=None, 
                 signal_x_bias = 0, background_x_bias = 0, compute_true_llr=False, points_per_string=1, **kwargs):
        """
        Compute the weighted LLR loss.
        
        Parameters:
        -----------
        points_3d : torch.Tensor or None
            Points in 3D space (not used directly, but kept for compatibility).
        signal_funcs : list of callable or None
            Not used directly (kept for compatibility).
        background_funcs : list of callable or None
            Not used directly (kept for compatibility).
        string_xy : torch.Tensor
            XY positions of strings, shape (n_strings, 2).
        points_per_string_list : list or None
            Number of points on each string (for compatibility).
        string_weights : torch.Tensor or None
            Weights for each string for weighted averaging.
        path_positions : torch.Tensor or None
            Positions along the continuous path (for compatibility).
        num_strings : int or None
            Number of strings in the geometry.
        precomputed_llr_per_string : torch.Tensor or None
            Precomputed LLR values per string. If provided, computation is skipped.
        points_3d : torch.Tensor or None
            Points in 3D space for the geometry.
        compute_snr : bool
            Whether to also compute SNR per string using surrogate functions.
        signal_surrogate_func : callable or None
            Light yield surrogate function for signal events. Required if compute_snr=True.
        background_surrogate_func : callable or None
            Light yield surrogate function for background events. Required if compute_snr=True.
        num_samples_per_point : int
            Number of event samples to use per point when computing SNR.
        points_per_string : list or int
            Number of points at each string.
        **kwargs : dict
            Additional arguments.
            
        Returns:
        --------
        tuple
            If compute_snr=False: (total_loss, weighted_avg_llr_no_penalties, llr_per_string)
            If compute_snr=True: (total_loss, weighted_avg_llr_no_penalties, llr_per_string, snr_per_string)
            - total_loss: Loss with all penalties applied
            - weighted_avg_llr_no_penalties: Weighted average LLR without penalties
            - llr_per_string: LLR values for each string
            - snr_per_string: SNR values for each string (only if compute_snr=True)
        """
        # Determine number of strings
        if num_strings is None:
            if string_weights is not None:
                num_strings = len(string_weights)
            elif string_xy is not None:
                num_strings = len(string_xy)
            else:
                raise ValueError("Cannot determine number of strings")
        
        # Validate string_xy
        if string_xy is None:
            raise ValueError("string_xy must be provided for LLR computation")
        
        # Use precomputed LLR if provided, otherwise compute it
        if precomputed_llr_per_string is not None and precomputed_background_llr is not None and precomputed_signal_llr is not None:
            llr_per_string = precomputed_llr_per_string
            signal_llr_per_string = precomputed_signal_llr
            background_llr_per_string = precomputed_background_llr
            
            
            # For now, assume we don't have precomputed signal/background LLR
            # signal_llr_per_string = torch.zeros_like(llr_per_string)
            # background_llr_per_string = torch.zeros_like(llr_per_string)
            if precomputed_signal_probs_per_string is not None and precomputed_background_probs_per_string is not None and self.use_bce_loss:
                signal_probs_per_string = precomputed_signal_probs_per_string
                background_probs_per_string = precomputed_background_probs_per_string
            if precomputed_signal_yield_per_string is not None:
                signal_yield_per_string = precomputed_signal_yield_per_string
            if precomputed_fisher_info_per_string is not None:
                fisher_info_per_string = precomputed_fisher_info_per_string
        else:
            llr_result = self.compute_llr_per_string(
                string_xy, num_strings, points_3d, compute_snr=compute_snr, 
                signal_surrogate_func=signal_surrogate_func,
                background_surrogate_func=background_surrogate_func,
                num_samples_per_point=num_samples_per_point, signal_x_bias=signal_x_bias,
                background_x_bias=background_x_bias, compute_true_llr=compute_true_llr
            )
            llr_per_string, signal_llr_per_string, background_llr_per_string, snr_per_string, true_llr_per_string, true_signal_llr_per_string, true_background_llr_per_string, signal_yield_per_string, fisher_info_per_string = llr_result

        if type(points_per_string) is int:
            points_per_string = [points_per_string] * num_strings
            points_per_string = torch.tensor(points_per_string, device=self.device)
        elif type(points_per_string) is list:
            points_per_string = torch.tensor(points_per_string, device=self.device)

        # Apply string weights if provided
        if string_weights is not None:
            # Apply sigmoid to get probabilities
            string_probs = string_weights
            clamped_string_probs = torch.sigmoid(string_probs)
            
            # Compute weighted average LLR for signal and background separately
            weighted_signal_llr = torch.mean(signal_llr_per_string * clamped_string_probs / points_per_string)
            weighted_background_llr = torch.mean(background_llr_per_string * clamped_string_probs / points_per_string)
            weighted_llr = torch.sum(llr_per_string * clamped_string_probs)  # Keep for compatibility

            # For BCE loss, compute weighted BCE using string weights as sample weights
            if self.use_bce_loss:
                # Create targets: 1 for signal, 0 for background
                signal_targets = torch.ones(num_strings, device=self.device)
                background_targets = torch.zeros(num_strings, device=self.device)
                
                # Convert probabilities back to logits for binary_cross_entropy_with_logits
                # logit(p) = log(p / (1 - p))
                eps = 1e-7  # Small epsilon to avoid log(0)
                signal_logits = torch.log((signal_probs_per_string + eps) / (1 - signal_probs_per_string + eps))
                background_logits = torch.log((background_probs_per_string + eps) / (1 - background_probs_per_string + eps))
                
                # Compute BCE with logits for signal and background separately, then combine
                signal_bce = F.binary_cross_entropy_with_logits(signal_logits, signal_targets, weight=clamped_string_probs, reduction='mean')
                background_bce = F.binary_cross_entropy_with_logits(background_logits, background_targets, weight=clamped_string_probs, reduction='mean')

                # Weighted BCE combining signal and background
                weighted_bce = (signal_bce + background_bce) / 2
        else:
            # If no weights provided, use simple average
            weighted_signal_llr = torch.mean(signal_llr_per_string)
            weighted_background_llr = torch.mean(background_llr_per_string)
            weighted_llr = torch.mean(llr_per_string)  # Keep for compatibility
            if self.use_bce_loss:
                # Create targets: 1 for signal, 0 for background
                signal_targets = torch.ones(num_strings, device=self.device)
                background_targets = torch.zeros(num_strings, device=self.device)
                
                # Convert probabilities back to logits for binary_cross_entropy_with_logits
                # logit(p) = log(p / (1 - p))
                eps = 1e-7  # Small epsilon to avoid log(0)
                signal_logits = torch.log((signal_probs_per_string + eps) / (1 - signal_probs_per_string + eps))
                background_logits = torch.log((background_probs_per_string + eps) / (1 - background_probs_per_string + eps))
                
                # Compute unweighted BCE with logits
                signal_bce = F.binary_cross_entropy_with_logits(signal_logits, signal_targets, reduction='mean')
                background_bce = F.binary_cross_entropy_with_logits(background_logits, background_targets, reduction='mean')
                
                # Average BCE
                weighted_bce = (signal_bce + background_bce) / 2
            clamped_string_probs = None
        
        # NEW LLR LOSS: Reciprocal of absolute difference between mean signal and background LLR
        if not self.use_bce_loss:
            # Compute absolute difference between signal and background LLR
            llr_difference = torch.abs(weighted_signal_llr - weighted_background_llr)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            # Loss is reciprocal of the difference (we want to maximize the difference)
            
            llr_loss = self.llr_weight / (llr_difference + epsilon)
            if self.boost_signal:
                # Boost signal LLR if configured
                llr_loss += self.signal_boost_weight/ weighted_signal_llr
            if self.boost_signal_yield and signal_yield_per_string is not None:
                llr_loss += self.signal_yield_boost_weight / torch.mean(signal_yield_per_string*clamped_string_probs/points_per_string)
        else:
            llr_loss = self.llr_weight * weighted_bce
        
        # Add Fisher Information term if computed
        if self.compute_fisher_info and fisher_info_per_string is not None:
            try:
                # Sum Fisher Information matrices across strings with optional weighting
                if string_weights is not None:
                    # Apply string weights to Fisher Information matrices
                    clamped_probs_for_fisher = torch.sigmoid(string_weights)
                    # Weight each matrix: sum over strings of (weight * fisher_matrix)
                    total_fisher_matrix = torch.zeros_like(fisher_info_per_string[0])
                    for s_idx in range(len(fisher_info_per_string)):
                        total_fisher_matrix += clamped_probs_for_fisher[s_idx] * fisher_info_per_string[s_idx]
                else:
                    # Simple sum across all strings
                    total_fisher_matrix = torch.sum(fisher_info_per_string, dim=0)
                
                # Compute log determinant of the total Fisher Information matrix
                # Add small regularization to ensure positive definiteness
                reg_matrix = torch.eye(total_fisher_matrix.size(0), device=self.device) * 1e-6
                regularized_fisher = total_fisher_matrix + reg_matrix
                
                # Compute log determinant
                try:
                    fisher_logdet = torch.det(regularized_fisher)
                    if torch.isfinite(fisher_logdet):
                        # Fisher Information bonus (negative log det to encourage higher Fisher Info)
                        fisher_bonus = self.fisher_info_weight / fisher_logdet
                        llr_loss += fisher_bonus
                        
                        if self.print_loss:
                            print(f"Fisher Information matrix trace: {torch.trace(total_fisher_matrix).item():.6f}")
                            print(f"Fisher Information log det: {fisher_logdet.item():.6f}")
                            print(f"Fisher Information bonus: {fisher_bonus.item():.6f}")
                    else:
                        if self.print_loss:
                            print("Warning: Fisher Information log determinant is not finite")
                except Exception as det_e:
                    if self.print_loss:
                        print(f"Warning: Fisher Information log determinant computation failed: {det_e}")
                        
            except Exception as fisher_e:
                if self.print_loss:
                    print(f"Warning: Fisher Information matrix processing failed: {fisher_e}")
        
        # Initialize total loss with LLR loss
        total_loss = llr_loss
        if self.conflict_free:
            total_loss = [llr_loss]
            if self.print_loss:
                if not self.use_bce_loss:
                    print(f"Signal LLR mean: {weighted_signal_llr.item():.6f}")
                    print(f"Background LLR mean: {weighted_background_llr.item():.6f}")
                    print(f"LLR difference: {torch.abs(weighted_signal_llr - weighted_background_llr).item():.6f}")
                print(f"Weighted LLR loss: {llr_loss.item()}")
        elif self.print_loss:
            if not self.use_bce_loss:
                print(f"Signal LLR mean: {weighted_signal_llr.item():.6f}")
                print(f"Background LLR mean: {weighted_background_llr.item():.6f}")
                print(f"LLR difference: {torch.abs(weighted_signal_llr - weighted_background_llr).item():.6f}")
            print(f"Weighted LLR loss: {llr_loss.item()}")
        
        # Compute penalties (same as WeightedSNRLoss)
        if self.string_repulsion_weight > 0 and string_xy is not None and string_weights is not None:
            string_repulsion_penalty = self.compute_local_string_repulsion_penalty(string_xy, clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_repulsion_penalty)
            else:    
                total_loss += string_repulsion_penalty
            if self.print_loss:
                print(f"String repulsion penalty: {string_repulsion_penalty.item()}")
        
        if self.eva_boundary_weight > 0 and string_weights is not None:
            string_weights_boundary_penalty = self.string_weights_boundary_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_weights_boundary_penalty)
            else:
                total_loss += string_weights_boundary_penalty
            if self.print_loss:
                print(f"String weights boundary penalty: {string_weights_boundary_penalty.item()}")
            
        if self.eva_string_num_weight > 0 and string_weights is not None:
            string_number_penalty = self.string_number_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_number_penalty)
            else:
                total_loss += string_number_penalty
            if self.print_loss:
                print(f"String number penalty: {string_number_penalty.item()}")
            
        if self.eva_binary_weight > 0 and string_weights is not None:
            binarization_penalty = self.weight_binarization_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(binarization_penalty)
            else:
                total_loss += binarization_penalty
            if self.print_loss:
                print(f"Binarization penalty: {binarization_penalty.item()}")
        
        # Add string weights penalty if using evanescent strings
        if string_weights is not None and self.eva_weight > 0:
            string_weights_penalty = self.string_weights_penalty(clamped_string_probs)
            if self.conflict_free:
                total_loss.append(string_weights_penalty)
            else:
                total_loss += string_weights_penalty
            if self.print_loss:
                print(f"String weights penalty: {string_weights_penalty.item()}")
        
        if self.print_loss:    
            if self.conflict_free:
                print(f"Total loss: {torch.stack(total_loss).sum().item()}")
            else:
                print(f"Total loss: {total_loss.item()}")
        
        # Prepare weighted Fisher Information output (if computed)
        if self.compute_fisher_info and 'fisher_info_per_string' in locals() and fisher_info_per_string is not None:
            weighted_fisher_info_value = total_fisher_matrix if 'total_fisher_matrix' in locals() else None
        else:
            weighted_fisher_info_value = None

        if compute_snr and points_3d is not None and not self.use_bce_loss:
            return total_loss, weighted_llr.item(), llr_per_string, snr_per_string, weighted_fisher_info_value
        elif compute_snr and points_3d is not None and self.use_bce_loss:
            return total_loss, weighted_llr.item(), weighted_bce.item(), llr_per_string, snr_per_string, signal_probs_per_string, background_probs_per_string, weighted_fisher_info_value
        elif points_3d is not None and not self.use_bce_loss:
            return total_loss, weighted_llr.item(), llr_per_string, weighted_fisher_info_value
        elif points_3d is not None and self.use_bce_loss:
            return total_loss, weighted_llr.item(), weighted_bce.item(), llr_per_string, snr_per_string, signal_probs_per_string, background_probs_per_string, weighted_fisher_info_value
        else:
            return total_loss, weighted_llr.item(), llr_per_string, weighted_fisher_info_value




