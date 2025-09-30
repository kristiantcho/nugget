from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F

class WeightedLLRLoss(LossFunction):
    """Loss function for log-likelihood ratio (LLR) maximization."""
    
    def __init__(self, device=None, llr_net = None, print_loss=False, event_labels=['position', 'energy', 'zenith', 'azimuth'], no_grad=True):
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
        self.llr_net = llr_net
        self.print_loss = print_loss
        self.event_labels = event_labels
        self.no_grad = no_grad
    
    def compute_LLR_per_string(self, string_xy, points_3d, event_params, surrogate_func, noise_scale=0):
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
        llr_values = []
        for s_idx in range(len(string_xy)):
            # Create optimization points for this string
            # Sample points along the string (assuming strings extend in z-direction)
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]    
            avg_llr = torch.tensor(0.0, device=self.device)
            features = []           
            for point in string_points:
                # Compute surrogate response for each event
                for params in event_params:
                    features.append(self.llr_net.prepare_data_from_raw(point, params, surrogate_func, self.event_labels, noise_scale, self.llr_net.add_relative_pos))
                #     avg_llr += self.llr_net.predict_log_likelihood_ratio(features)
                # avg_llr /= len(event_params)
            if features:  # Only compute if we have features
                if self.no_grad:
                    with torch.no_grad():    
                        avg_llr = torch.sum(self.llr_net.predict_log_likelihood_ratio(torch.stack(features))) / len(event_params)
                else:    
                    avg_llr = torch.sum(self.llr_net.predict_log_likelihood_ratio(torch.stack(features))) / len(event_params)           
            llr_values.append(avg_llr)
        
        # Stack to preserve gradients
        llr_per_string = torch.stack(llr_values) if llr_values else torch.tensor([], device=self.device)
        return llr_per_string
    
    def __call__(self, geom_dict, **kwargs):
        
        precomputed_llr_per_string = kwargs.get('precomputed_signal_llr_per_string', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        noise_scale = kwargs.get('signal_noise_scale', 0.0)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)

        if precomputed_llr_per_string is not None:
            llr_per_string = precomputed_llr_per_string
        else:
            llr_per_string = self.compute_LLR_per_string(string_xy, points_3d, event_params, surrogate_func, noise_scale)

        if string_weights is None:
            total_llr = torch.sum(llr_per_string)  # Sum over strings
        else:
            string_probs = torch.sigmoid(string_weights)
            total_llr = torch.sum(llr_per_string * string_probs) / len(string_probs)  # Weighted sum
        
        llr_loss = 1/(total_llr + 1e-6)  # Add small value for numerical stability
        
        return {'signal_llr_loss': llr_loss, 'signal_llr_per_string': llr_per_string, 'signal_total_llr': total_llr}
    

class WeightedMeanDifLLRLoss(WeightedLLRLoss):
    """Loss function for mean difference in log-likelihood ratio (LLR) maximization."""
    
    def __init__(self, device=None, llr_net = None, print_loss=False, event_labels=['position', 'energy', 'zenith', 'azimuth'], no_grad=True):
        """
        Initialize the mean difference LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        LLRnet : torch.nn.Module
            Pre-trained neural network model to compute LLR.
        noise_scale : float
            Scale of noise to add to surrogate response.
        """
        super().__init__(device, llr_net, print_loss, event_labels, no_grad)
    
    def __call__(self, geom_dict, **kwargs):
        
        precomputed_signal_llr_per_string = kwargs.get('precomputed_signal_llr_per_string', None)
        precomputed_background_llr_per_string = kwargs.get('precomputed_background_llr_per_string', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        background_event_params = kwargs.get('background_event_params', None)
        background_surrogate_func = kwargs.get('background_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        background_sampler = kwargs.get('background_sampler', None)
        num_events = kwargs.get('num_events', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', 0.0)
        background_noise_scale = kwargs.get('background_noise_scale', 0.0)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        if background_event_params is None and background_sampler is not None:
            background_event_params = background_sampler.sample_events(num_events)

        if precomputed_signal_llr_per_string is not None:
            signal_llr_per_string = precomputed_signal_llr_per_string
        else:
            signal_llr_per_string = self.compute_LLR_per_string(string_xy, points_3d, signal_event_params, signal_surrogate_func, signal_noise_scale)
        if precomputed_background_llr_per_string is not None:
            background_llr_per_string = precomputed_background_llr_per_string
        else:
            background_llr_per_string = self.compute_LLR_per_string(string_xy, points_3d, background_event_params, background_surrogate_func, background_noise_scale)

        if string_weights is None:
            signal_total_llr = torch.mean(signal_llr_per_string)  # Mean over strings
            background_total_llr = torch.mean(background_llr_per_string)  # Mean over strings
            llr_diff = torch.abs(signal_total_llr - background_total_llr)
        else:
            string_probs = torch.sigmoid(string_weights)
            signal_total_llr = torch.sum(signal_llr_per_string * string_probs) / len(string_probs)  # Weighted mean
            background_total_llr = torch.sum(background_llr_per_string * string_probs) / len(string_probs)  # Weighted mean
            llr_diff = torch.abs(signal_total_llr - background_total_llr)
        
        llr_loss = 1/(llr_diff + 1e-6)  # Add small value for numerical stability
        
        return {'mean_dif_llr_loss': llr_loss, 'signal_llr_per_string': signal_llr_per_string, 'background_llr_per_string': background_llr_per_string, 'signal_total_llr': signal_total_llr, 'background_total_llr': background_total_llr}

class LLRLoss(LossFunction):
    """Loss function for log-likelihood ratio (LLR) maximization."""
    
    def __init__(self, device=None, llr_net = None, print_loss=False, event_labels=['position', 'energy', 'zenith', 'azimuth']):
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
        self.llr_net = llr_net
        self.print_loss = print_loss
        self.event_labels = event_labels
    
    def compute_LLR_per_point(self, points_3d, event_params, surrogate_func, noise_scale=0):
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
          
        llr_values = []         
        for ip, point in enumerate(points_3d):
            # Compute surrogate response for each event
            features = []
            for params in event_params:
                features.append(self.llr_net.prepare_data_from_raw(point, params, surrogate_func, self.event_labels, noise_scale, self.llr_net.add_relative_pos))
            #     avg_llr += self.llr_net.predict_log_likelihood_ratio(features)
            # avg_llr /= len(event_params)
            avg_llr = torch.sum(self.llr_net.predict_log_likelihood_ratio(torch.stack(features))) / len(event_params)
            llr_values.append(avg_llr)
        
        # Stack to preserve gradients
        llr_per_point = torch.stack(llr_values) if llr_values else torch.tensor([], device=self.device)
        return llr_per_point
    
    def __call__(self, geom_dict, **kwargs):
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        noise_scale = kwargs.get('signal_noise_scale', 0.0)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)

        
        llr_per_point = self.compute_LLR_per_point(points_3d, event_params, surrogate_func, noise_scale)
        total_llr = torch.sum(llr_per_point)  # Sum over points
        
        llr_loss = 1/(total_llr + 1e-6)  # Add small value for numerical stability
        
        return {'signal_llr_loss': llr_loss, 'signal_total_llr': total_llr, 'signal_llr_per_point': llr_per_point}
    

class MeanDifLLRLoss(LLRLoss):
    """Loss function for mean difference in log-likelihood ratio (LLR) maximization."""
    
    def __init__(self, device=None, llr_net = None, print_loss=False, event_labels=['position', 'energy', 'zenith', 'azimuth']):
        """
        Initialize the mean difference LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations. If None, uses cuda if available, else cpu.
        LLRnet : torch.nn.Module
            Pre-trained neural network model to compute LLR.
        noise_scale : float
            Scale of noise to add to surrogate response.
        """
        super().__init__(device, llr_net, print_loss, event_labels)
    
    def __call__(self, geom_dict, **kwargs):
        
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        background_event_params = kwargs.get('background_event_params', None)
        background_surrogate_func = kwargs.get('background_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        background_sampler = kwargs.get('background_sampler', None)
        num_events = kwargs.get('num_events', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', 0.0)
        background_noise_scale = kwargs.get('background_noise_scale', 0.0)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        if background_event_params is None and background_sampler is not None:
            background_event_params = background_sampler.sample_events(num_events)

        signal_llr_per_point = self.compute_LLR_per_point(points_3d, signal_event_params, signal_surrogate_func, signal_noise_scale)
        background_llr_per_point = self.compute_LLR_per_point(points_3d, background_event_params, background_surrogate_func, background_noise_scale)
        signal_total_llr = torch.sum(signal_llr_per_point)  # Mean over points
        background_total_llr = torch.sum(background_llr_per_point)  # Mean over points
        llr_diff = torch.abs(signal_total_llr - background_total_llr)/len(points_3d)
        llr_loss = 1/(llr_diff + 1e-6)  # Add small value for numerical stability
        
        return {'mean_dif_llr_loss': llr_loss, 'signal_total_llr': signal_total_llr, 'background_total_llr': background_total_llr, 'signal_llr_per_point': signal_llr_per_point, 'background_llr_per_point': background_llr_per_point}