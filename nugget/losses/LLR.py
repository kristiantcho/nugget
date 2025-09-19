from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F

class WeightedLLRLoss(LossFunction):
    """Loss function for log-likelihood ratio (LLR) maximization."""
    
    def __init__(self, device=None, LLRnet = None, noise_scale=0.0, print_loss=False, event_labels=['position', 'energy', 'zenith', 'azimuth']):
        """
        Initialize the LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        LLRnet : torch.nn.Module
            Pre-trained neural network model to compute LLR.
        noise_scale : float
            Scale of noise to add to surrogate response.
        """
        super().__init__(device)
        self.llr_net = LLRnet.to(self.device) if LLRnet is not None else None
        self.print_loss = print_loss
        self.noise_scale = noise_scale
        self.event_labels = event_labels
    
    def compute_LLR_per_string(self, string_xy, points_3d, event_params, surrogate_func):
        """
        Compute the LLR for each string.
        
        Parameters:
        -----------
        string_xy : torch.Tensor
            Tensor of shape (n_strings, 2) with the (x, y) positions of each string.
        points_3d : torch.Tensor
            Tensor of shape (n_points, 3) with the (x, y, z) positions of all points.
        event_params : list of dict
            List of parameter dictionaries for events.
        surrogate_func : callable
            Surrogate function to compute response.
            
        Returns:
        --------
        torch.Tensor
            Tensor of shape (n_strings,) with the computed LLR for each string.
        """
        llr_per_string = torch.zeros(len(string_xy), device=self.device)
        for s_idx in range(len(string_xy)):
            # Create optimization points for this string
            # Sample points along the string (assuming strings extend in z-direction)
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]    
            for point in string_points:
                # Compute surrogate response for each event
                avg_llr = torch.tensor(0.0, device=self.device)
                for params in event_params:
                    features = self.llr_net.prepare_data_from_raw(point, params, surrogate_func, self.event_labels, self.noise_scale)
                    avg_llr += self.llr_net.predict_log_likelihood_ratio(features)  # Shape (n_events,)
                avg_llr /= len(event_params)
                llr_per_string[s_idx] += avg_llr
        return llr_per_string
    
    def __call__(self, geom_dict, **kwargs):
        
        precomputed_llr_per_string = kwargs.get('precomputed_llr_per_string', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)

        if precomputed_llr_per_string is not None:
            llr_per_string = precomputed_llr_per_string
        else:
            llr_per_string = self.compute_LLR_per_string(string_xy, points_3d, event_params, surrogate_func)

        if string_weights is None:
            total_llr = torch.sum(llr_per_string)  # Sum over strings
        else:
            string_probs = torch.sigmoid(string_weights)
            total_llr = torch.sum(llr_per_string * string_probs)  # Weighted sum
        
        llr_loss = 1/(total_llr + 1e-6)  # Add small value for numerical stability
        
        return {'llr_loss': llr_loss, 'llr_per_string': llr_per_string, 'total_llr': total_llr}