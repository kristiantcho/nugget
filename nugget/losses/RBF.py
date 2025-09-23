from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F

class RBFInterpolationLoss(LossFunction):
    """Loss function for RBF interpolation."""
    
    def __init__(self, device=None, epsilon=30.0):
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
        super().__init__(device)
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
        
    
    def __call__(self, geom_dict, **kwargs):
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
        
        points_3d = geom_dict.get('points_3d', None)
        surrogate_func = kwargs.get('signal_surrogate_func')
        test_points = kwargs.get('test_points', None)
        event_params = kwargs.get('signal_event_params', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)
        loss_func = kwargs.get('loss_func', F.mse_loss)
        
        
        if test_points is None:
            test_points = torch.rand(1000, 3).to(self.device) * self.domain_size - self.domain_size / 2
        
        f_tests = []
        s_tests = []
        f_value_sets = []
        for params in event_params:
            
            f_values = surrogate_func(opt_point=points_3d, event_params=params)  # Shape (num_points,)
            f_test = surrogate_func(opt_point=test_points, event_params=params)
            w, K = self.compute_rbf_interpolant(points_3d, f_values, test_points)
            s_test = K @ w
            
            f_tests.append(f_test)
            s_tests.append(s_test)
            f_value_sets.append(f_values)
        f_tests = torch.stack(f_tests)
        s_tests = torch.stack(s_tests)
        f_value_sets = torch.stack(f_value_sets)
        
        loss = loss_func(f_tests, s_tests)
    
        
        # return loss,  f_tests, s_tests, f_value_sets
        return {'rbf_loss': loss, 'f_tests': f_tests, 's_tests': s_tests, 'f_value_sets': f_value_sets}