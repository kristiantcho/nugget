from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F

class SNRloss(LossFunction):
    """Loss function for SNR (Signal-to-Noise Ratio) optimization."""
    
    def __init__(self, device=None, signal_scale=1.0, background_scale=1.0, no_background=False):
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
        super().__init__(device)
      
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.no_background = no_background
        
    
    def __call__(self, geom_dict, **kwargs):
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
        points_3d = geom_dict.get('points_3d', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        background_surrogate_func = kwargs.get('background_surrogate_func', None)
        background_event_params = kwargs.get('background_event_params', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        background_sampler = kwargs.get('background_sampler', None)
        num_events = kwargs.get('num_events', 100)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        if background_event_params is None and background_sampler is not None:
            background_event_params = background_sampler.sample_events(num_events)
        signal_total = torch.zeros(len(signal_event_params), device=self.device)

        # signal_values = torch.zeros(len(signal_funcs), points_3d.shape[0], device=self.device)
        for i, params in enumerate(signal_event_params):
            signal_total[i] = torch.sum(signal_surrogate_func(opt_point=points_3d, event_params=params)) * self.signal_scale
        
        # signal_total = torch.sum(signal_values, dim=1)  # Sum across all points for each function
        
        # Compute background light yield (sum of background function values)
        background_total = torch.zeros(len(background_event_params), device=self.device)
        for i, params in enumerate(background_event_params):
            background_total[i] = torch.sum(background_surrogate_func(opt_point=points_3d, event_params=params)) * self.background_scale
        
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
        snr_loss = 1/avg_snr
        
        # Initialize total loss with SNR loss
        # total_loss = snr_loss
        
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
        
        # return snr_loss, avg_snr.item(), all_snr
        return {'snr_loss': snr_loss, 'avg_snr': avg_snr.item(), 'all_snr': all_snr}
    
    
class WeightedSNRLoss(LossFunction):
    """Loss function for weighted SNR optimization per string."""
    
    def __init__(self, device=None, signal_scale=1.0, background_scale=10.0, no_background=False, print_loss=False, num_bkg_samples=100, num_signal_samples=100):
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
        super().__init__(device)
    
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.no_background = no_background
        self.print_loss = print_loss
        self.num_bkg_samples = num_bkg_samples
        self.num_signal_samples = num_signal_samples
    
    def compute_snr_per_string(self, points_3d, signal_surrogate_func, background_surrogate_func, signal_event_params, background_event_params, string_xy):
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
        n_strings = len(string_xy)
        snr_per_string = torch.zeros(n_strings, device=self.device)
        
        # Compute SNR for each string
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
            signal_sum = torch.tensor(0.0, device=self.device)
            background_sum = torch.tensor(0.0, device=self.device)
            for point in string_points:
                if not self.no_background:
                    
                    for params in background_event_params:
                        background_sum += background_surrogate_func(point, *params) * self.background_scale
                    
                else:
                    background_sum = torch.tensor(1.0, device=self.device) * self.background_scale  # Use a constant value for background if no_background is True
                
                for params in signal_event_params:
                    signal_sum += signal_surrogate_func(point, *params) * self.signal_scale
        
            background_sum /= len(background_event_params)
            signal_sum /= len(signal_event_params)
            epsilon = 1e-10 if not self.no_background else 0
            snr_per_string[s_idx] += torch.sum(signal_sum / torch.sqrt(background_sum + epsilon))
        
        return snr_per_string
    
    def __call__(self, geom_dict, **kwargs):
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
        precomputed_snr_per_string : torch.Tensor or None
            Precomputed SNR values per string (n_strings, n_signal_events).
            If provided, computation is skipped.
        **kwargs : dict
            Additional arguments specific to the loss function.
            
        Returns:
        --------
        dict
            (total_loss, avg_snr, snr_per_string)
            - total_loss: Loss with all penalties applied
            - avg_snr: Weighted average SNR
            - snr_per_string: SNR values for each string (n_strings, n_signal_events)
        """
        
        points_3d = geom_dict.get('points_3d', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        background_surrogate_func = kwargs.get('background_surrogate_func', None)
        background_event_params = kwargs.get('background_event_params', None)
        string_xy = geom_dict.get('string_xy', None)
        points_per_string_list = geom_dict.get('points_per_string_list', None)
        string_weights = geom_dict.get('string_weights', None)
        num_strings = geom_dict.get('num_strings', None)
        precomputed_snr_per_string = kwargs.get('precomputed_snr_per_string', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        background_sampler = kwargs.get('background_sampler', None)
        num_events = kwargs.get('num_events', 100)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        if background_event_params is None and background_sampler is not None:
            background_event_params = background_sampler.sample_events(num_events)
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
                points_3d, signal_surrogate_func, background_surrogate_func, signal_event_params, background_event_params, string_xy)
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
            no_norm_weighted_snr_per_string = torch.sum(snr_per_string * clamped_string_probs, dim=0)#/torch.mean(snr_per_string, dim=0, keepdim=True)  # (n_signal_funcs,)
            # print(snr_per_string*clamped_string_probs)
        else:
            # If no weights provided, use simple average
            
            # weighted_snr_per_func = torch.mean(snr_per_string/torch.mean(snr_per_string, dim=0, keepdim=True))  # (n_signal_funcs,)
            no_norm_weighted_snr_per_string = torch.mean(snr_per_string, dim=0)  # (n_signal_funcs,)
        # Average across all signal functions
        # weighted_avg_snr = torch.mean(weighted_snr_per_func)
        no_norm_weighted_avg_snr = torch.mean(no_norm_weighted_snr_per_string)
        
        # SNR loss (negative since we want to maximize SNR)
        
        snr_loss =  1 / (no_norm_weighted_avg_snr + 1e-10)
        
        # return snr_loss, no_norm_weighted_avg_snr.item(), snr_per_string
        return {'snr_loss': snr_loss, 'avg_snr': no_norm_weighted_avg_snr.item(), 'snr_per_string': snr_per_string}