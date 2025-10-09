from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F


def compute_fisher_info_single(fisher_info_params, point, event_params, surrogate_func, llr_net=None, signal_noise_scale=None, add_relative_pos=False):
    """
    Compute the Fisher information matrix for a single point and event parameters.
    
    Parameters:
    -----------
    point : torch.Tensor
        The 3D point to evaluate the Fisher information at.
    event_params : dict
        Dictionary containing event parameters.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
        
    Returns:
    --------
    torch.Tensor
        The Fisher information matrix (n_params, n_params).
    """
    n_params = len(fisher_info_params)
    fisher_matrix = torch.zeros(n_params, n_params)
    
    # Ensure parameters require gradients
    grad_event_params = {}
    for param_name in fisher_info_params:
        grad_event_params[param_name] = event_params.get(param_name).clone().detach().requires_grad_(True)

    for param_name in event_params.keys():
        if param_name not in fisher_info_params:
            grad_event_params[param_name] = event_params.get(param_name).clone().detach().requires_grad_(False)
    

    # Compute light yield mean (λ) using the signal surrogate function with gradients
    if llr_net is None:
        light_yield_mean = surrogate_func(opt_point=point, event_params=grad_event_params)  # Shape (1,)
        if signal_noise_scale is not None:
            light_yield_mean = torch.normal(0, signal_noise_scale)*light_yield_mean + light_yield_mean
    
    else:
        features = llr_net.prepare_data_from_raw(point, grad_event_params, surrogate_func, noise_scale=signal_noise_scale, add_relative_pos=add_relative_pos)
        light_yield_mean = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-9)
    param_gradients = []
    
    for param_name in fisher_info_params:
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
            
            param_gradients.append(param_grad)

    # Compute Fisher Information matrix: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
    if len(param_gradients) == n_params:
        for i_param in range(n_params):
            for j_param in range(n_params):
                
                grad_i = param_gradients[i_param]
                grad_j = param_gradients[j_param]
                
                # Fisher Information element: (∂λ/∂θ_i)(∂λ/∂θ_j)/λ
                if llr_net is None:    
                    fisher_element = (grad_i * grad_j) / light_yield_mean
                else: 
                    fisher_element = grad_i * grad_j
                fisher_matrix[i_param, j_param] += fisher_element.item()
            
    return fisher_matrix


def compute_fisher_info_batch_events(fisher_info_params, point, event_params_list, surrogate_func, device=None):
    """
    Compute the Fisher information matrix for a single point with multiple events (batched).
    
    This is much more efficient than calling compute_fisher_info_single multiple times
    as it processes all events simultaneously and computes gradients in batch.
    
    Parameters:
    -----------
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
    point : torch.Tensor
        The 3D point to evaluate the Fisher information at.
    event_params_list : list of dict
        List of dictionaries containing event parameters.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    device : torch.device or None
        Device to use for computations.
        
    Returns:
    --------
    torch.Tensor
        The Fisher information matrix (n_params, n_params).
    """
    if device is None:
        device = point.device
        
    n_params = len(fisher_info_params)
    n_events = len(event_params_list)
    fisher_matrix = torch.zeros(n_params, n_params, device=device)
    
    if n_events == 0:
        return fisher_matrix
    
    # Stack event parameters into batched tensors
    batched_grad_params = {}
    batched_fixed_params = {}
    
    for param_name in fisher_info_params:
        param_values = []
        for event_params in event_params_list:
            param_values.append(event_params.get(param_name))
        batched_grad_params[param_name] = torch.stack(param_values).to(device).requires_grad_(True)
    
    # Handle non-gradient parameters
    for event_params in event_params_list:
        for param_name in event_params.keys():
            if param_name not in fisher_info_params:
                if param_name not in batched_fixed_params:
                    param_values = []
                    for ep in event_params_list:
                        param_values.append(ep.get(param_name))
                    batched_fixed_params[param_name] = torch.stack(param_values).to(device)
    
    # Combine all parameters
    batched_event_params = {**batched_grad_params, **batched_fixed_params}
    
    # Expand point to match batch size
    batched_points = point.unsqueeze(0).expand(n_events, -1)  # Shape: (n_events, 3)
    
    # Compute light yields for all events at once
    light_yields = []
    for i in range(n_events):
        event_params_dict = {k: v[i] for k, v in batched_event_params.items()}
        light_yield = surrogate_func(opt_point=batched_points[i], event_params=event_params_dict)
        light_yields.append(light_yield)
    
    light_yields = torch.stack(light_yields)  # Shape: (n_events,)
    
    # Compute gradients for all parameters
    param_gradients = []
    for param_name in fisher_info_params:
        param_tensor = batched_grad_params[param_name]
        
        # Compute gradient ∂λ/∂θ for all events
        grad_outputs = torch.ones_like(light_yields)
        param_grad = torch.autograd.grad(
            outputs=light_yields,  # Sum to get scalar output
            inputs=param_tensor,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]  # Shape: (n_events,)
        
        param_gradients.append(param_grad)
    
    # Compute Fisher Information matrix: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
    if len(param_gradients) == n_params:
        for i_param in range(n_params):
            for j_param in range(n_params):
                grad_i = param_gradients[i_param]  # Shape: (n_events,)
                grad_j = param_gradients[j_param]  # Shape: (n_events,)
                
                # Fisher Information elements for all events: (∂λ/∂θ_i)(∂λ/∂θ_j)/λ
                fisher_elements = (grad_i * grad_j) / light_yields  # Shape: (n_events,)
                
                # Average across events
                fisher_matrix[i_param, j_param] = fisher_elements.mean()
    
    return fisher_matrix


def compute_fisher_info_batch_points(fisher_info_params, points, event_params_list, surrogate_func, device=None):
    """
    Compute the Fisher information matrix for multiple points with multiple events (fully batched).
    
    This is the most efficient approach as it processes all points and events simultaneously.
    
    Parameters:
    -----------
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
    points : torch.Tensor
        The 3D points to evaluate the Fisher information at. Shape: (n_points, 3)
    event_params_list : list of dict
        List of dictionaries containing event parameters.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    device : torch.device or None
        Device to use for computations.
        
    Returns:
    --------
    torch.Tensor
        The Fisher information matrix summed over all points (n_params, n_params).
    """
    if device is None:
        device = points.device
        
    n_params = len(fisher_info_params)
    n_events = len(event_params_list)
    n_points = points.shape[0]
    
    total_fisher_matrix = torch.zeros(n_params, n_params, device=device)
    
    if n_events == 0 or n_points == 0:
        return total_fisher_matrix
    
    # For now, we'll batch over events but process points sequentially
    # Future optimization could batch both dimensions if surrogate_func supports it
    for point in points:
        fisher_matrix = compute_fisher_info_batch_events(
            fisher_info_params, point, event_params_list, surrogate_func, device
        )
        total_fisher_matrix += fisher_matrix
    
    return total_fisher_matrix

class FisherInfoLoss(LossFunction):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith']):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
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
        fisher_info_params : list of str
            List of event parameters to compute Fisher information for.
        """
        super().__init__(device)
        
        # self.total_samples_per_point = num_samples
        self.print_loss = print_loss
        # self.batch_size_per_point = batch_size_per_point
        self.random_seed = random_seed
        # self.signal_surrogate_func = signal_surrogate_func
        # self.background_surrogate_func = background_surrogate_func
        # self.signal_event_params = signal_event_params
        # self.background_event_params = background_event_params
        self.fisher_info_params = fisher_info_params # Default parameters for Fisher Info  
    
    def __call__experimental(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(FisherInfo).
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        event_params : list of dict
            List of dictionaries containing event parameters.
        surrogate_func : callable
            Function that computes light yield from event parameters.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 1)
        
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)
        
        # Use batch computation for all points and events
        total_fisher_info = compute_fisher_info_batch_points(
            self.fisher_info_params, 
            points_3d, 
            event_params, 
            surrogate_func, 
            self.device
        )
            
        fisher_loss = 1/(torch.det(total_fisher_info) + 1e-6)  # Add small value to diagonal for numerical stability
        
        if self.print_loss:
            print(f"Fisher Info Loss: {fisher_loss.item()}")
        
        # return fisher_loss, total_fisher_info
        return {'fisher_loss': fisher_loss, 'total_fisher_info': total_fisher_info}
        
    def __call__(self, geom_dict, **kwargs):
        """
        Legacy method - kept for backward compatibility and testing.
        Compute the total Fisher information loss using the original single-event approach.
        """
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)
        n_params = len(self.fisher_info_params)
        total_fisher_info = torch.zeros(n_params, n_params, device=self.device)
        fisher_info_per_point = torch.zeros((len(points_3d), n_params, n_params), device=self.device)
        for i, point in enumerate(points_3d):
            fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
            for params in event_params:
                for _ in range(llr_iterations):
                    fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, params, surrogate_func, llr_net, signal_noise_scale, add_relative_pos=add_relative_pos)/len(event_params)
            total_fisher_info += fisher_matrix/llr_iterations
            fisher_info_per_point[i] += fisher_matrix

        fisher_loss = torch.trace(torch.inverse(total_fisher_info + 1e-6 * torch.eye(total_fisher_info.shape[0], device=total_fisher_info.device)))  # Add small value to diagonal for numerical stability

        if self.print_loss:
            print(f"Fisher Info Loss: {fisher_loss.item()}")
        
        return {'fisher_loss': fisher_loss, 'total_fisher_info': total_fisher_info, 'fisher_info_per_point': fisher_info_per_point}
        
class WeightedFisherInfoLoss(LossFunction):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith']):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
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
        fisher_info_params : list of str
            List of event parameters to compute Fisher information for.
        """
        super().__init__(device)
        
        # self.total_samples_per_point = num_samples
        self.print_loss = print_loss
        # self.batch_size_per_point = batch_size_per_point
        self.random_seed = random_seed
        # self.signal_surrogate_func = signal_surrogate_func
        # self.background_surrogate_func = background_surrogate_func
        # self.signal_event_params = signal_event_params
        # self.background_event_params = background_event_params
        self.fisher_info_params = fisher_info_params # Default parameters for Fisher Info

    def compute_fisher_info_per_string_per_event(self, string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net=None, signal_noise_scale=None, llr_iterations=1, add_relative_pos=False):
        n_strings = len(string_xy)
        n_params = len(self.fisher_info_params)
        fisher_per_string_per_event = torch.zeros(len(signal_event_params), n_strings, n_params, n_params, device=self.device)
        
        for i, signal_params in enumerate(signal_event_params):
            for s_idx in range(n_strings):
                mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
                string_points = points_3d[mask]
                fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
                for point in string_points: 
                    for _ in range(llr_iterations):    
                        fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, event_params=signal_params, surrogate_func=signal_surrogate_func, llr_net=llr_net, signal_noise_scale=signal_noise_scale, add_relative_pos=add_relative_pos)
                    fisher_matrix += fisher_matrix/llr_iterations
                fisher_per_string_per_event[i, s_idx] += fisher_matrix

        return fisher_per_string_per_event

    def compute_fisher_info_per_string_experimental(self, string_xy, points_3d, signal_event_params, signal_surrogate_func):
        """
        Compute the Fisher information for each string using batch computation.
        
        Parameters:
        -----------
        string_xy : list of torch.Tensor or None
            The 2D points of the strings to compute the penalty for.
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        signal_event_params : list of dict
            List of dictionaries containing signal event parameters.
        signal_surrogate_func : callable
            Function that computes signal light yield from event parameters.
        background_event_params : list of dict or None
            List of dictionaries containing background event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
        """
        n_strings = len(string_xy)
        n_params = len(self.fisher_info_params)
        fisher_info_per_string = torch.zeros(n_strings, n_params, n_params, device=self.device)
        
        for s_idx in range(n_strings):
            # Create optimization points for this string
            # Sample points along the string (assuming strings extend in z-direction)
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]
            
            if len(string_points) > 0:
                # Use batch computation for all points on this string with all events
                fisher_matrix = compute_fisher_info_batch_points(
                    self.fisher_info_params, 
                    string_points, 
                    signal_event_params, 
                    signal_surrogate_func, 
                    self.device
                )
                fisher_info_per_string[s_idx] = fisher_matrix
            
        return fisher_info_per_string
        
    def compute_fisher_info_per_string(self, string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net=None, signal_noise_scale=None, llr_iterations=1, add_relative_pos=False):
        """
        Legacy method - kept for backward compatibility and testing.
        Compute the Fisher information for each string using the original single-event approach.
        """
        n_strings = len(string_xy)
        n_params = len(self.fisher_info_params)
        fisher_info_per_string = torch.zeros(n_strings, n_params, n_params, device=self.device)
        
        for s_idx in range(n_strings):
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]
            fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
            for point in string_points:  
                for signal_params in signal_event_params:
                    for _ in range(llr_iterations):    
                        fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, event_params=signal_params, surrogate_func=signal_surrogate_func, llr_net=llr_net, signal_noise_scale=signal_noise_scale, add_relative_pos=add_relative_pos)/len(signal_event_params)
            fisher_info_per_string[s_idx] += fisher_matrix/llr_iterations
            
        return fisher_info_per_string
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(WeightedFisherInfo).
        
        Parameters:
        -----------
        string_xy : list of torch.Tensor or None
            The 2D points of the strings to compute the penalty for.
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        signal_event_params : list of dict
            List of dictionaries containing signal event parameters.
        signal_surrogate_func : callable
            Function that computes signal light yield from event parameters.
        background_event_params : list of dict or None
            List of dictionaries containing background event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        # precomputed_fisher_info_per_string = kwargs.get('precomputed_fisher_info_per_string', None)
        precomputed_fisher_info_per_string_per_event = kwargs.get('precomputed_fisher_info_per_string_per_event', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        # background_event_params = kwargs.get('background_event_params', None)
        # background_surrogate_func = kwargs.get('background_surrogate_func', None)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        # if precomputed_fisher_info_per_string is None:
        #     fisher_info_per_string = self.compute_fisher_info_per_string(string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos)
        if precomputed_fisher_info_per_string_per_event is None:
            fisher_info_per_string_per_event = self.compute_fisher_info_per_string_per_event(string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos) 
        # else:
        #     fisher_info_per_string = precomputed_fisher_info_per_string
        else:
            fisher_info_per_string_per_event = precomputed_fisher_info_per_string_per_event
        # if string_weights is None:
        #     total_fisher_info = torch.sum(fisher_info_per_string, dim=0)  # Sum over strings, keep matrix form
        # else:
        #     string_probs = torch.sigmoid(string_weights)
        #     total_fisher_info = torch.sum(string_probs.unsqueeze(1).unsqueeze(2) * fisher_info_per_string, dim=0)  # Weighted sum
        # fisher_loss = torch.det(torch.inverse(total_fisher_info + 1e-6 * torch.eye(total_fisher_info.shape[0], device=total_fisher_info.device)))  # Add small value to diagonal for numerical stability
        if string_weights is None:
            total_fisher_info_per_event = torch.sum(fisher_info_per_string_per_event, dim=1)  # Sum over strings, keep matrix form
        else:
            string_probs = torch.sigmoid(string_weights)
            total_fisher_info_per_event = torch.sum(string_probs.unsqueeze(1).unsqueeze(2) * fisher_info_per_string_per_event, dim=1)
        mean_fisher_inv_trace = 0
        for i in range(total_fisher_info_per_event.shape[0]):
            mean_fisher_inv_trace += torch.trace(torch.inverse(total_fisher_info_per_event[i] + 1e-6 * torch.eye(total_fisher_info_per_event.shape[1], device=total_fisher_info_per_event.device)))/total_fisher_info_per_event.shape[0]  # Add small value to diagonal for numerical stability
        fisher_loss = mean_fisher_inv_trace
        
        # return {'fisher_loss': fisher_loss, 'fisher_info_per_string': fisher_info_per_string, 'total_fisher_info': total_fisher_info}
        return {'fisher_loss': fisher_loss, 'fisher_info_per_string_per_event': fisher_info_per_string_per_event, 'total_fisher_info_per_event': total_fisher_info_per_event, 'fisher_signal_params': signal_event_params}
class WeightedResolutionLoss(WeightedFisherInfoLoss):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith'], resolution_type='angular'):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
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
        fisher_info_params : list of str
            List of event parameters to compute Fisher information for.
        resolution_type : str
            Type of resolution to compute ('angular' or 'energy').
        """
        super().__init__(device, print_loss, random_seed, fisher_info_params)
        
        self.resolution_type = resolution_type # 'angular' or 'energy'

    def __call__(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(WeightedFisherInfo).
        
        Parameters:
        -----------
        string_xy : list of torch.Tensor or None
            The 2D points of the strings to compute the penalty for.
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        signal_event_params : list of dict
            List of dictionaries containing signal event parameters.
        signal_surrogate_func : callable
            Function that computes signal light yield from event parameters.
        background_event_params : list of dict or None
            List of dictionaries containing background event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        precomputed_fisher_info_per_string_per_event = kwargs.get('precomputed_fisher_info_per_string_per_event', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        # background_event_params = kwargs.get('background_event_params', None)
        # background_surrogate_func = kwargs.get('background_surrogate_func', None)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        if precomputed_fisher_info_per_string_per_event is None:
            fisher_info_per_string_per_event = self.compute_fisher_info_per_string_per_event(string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos)
        else:
            fisher_info_per_string_per_event = precomputed_fisher_info_per_string_per_event
        if string_weights is None:
            total_fisher_info = torch.sum(fisher_info_per_string_per_event, dim=1)  # Sum over strings, keep matrix form
        else:
            string_probs = torch.sigmoid(string_weights)
            total_fisher_info = torch.sum(string_probs.unsqueeze(1).unsqueeze(2) * fisher_info_per_string_per_event, dim=1)
        resolution_per_event = []
        if self.resolution_type == 'angular':
            for i, params in enumerate(signal_event_params):
                zenith = params['zenith']
                # azimuth = params['azimuth']
                azimuth_idx = self.fisher_info_params.index('azimuth')
                zenith_idx = self.fisher_info_params.index('zenith')
                cov_matrix = torch.inverse(total_fisher_info[i])
        
                # Angular resolution: sqrt(var_azimuth + var_zenith)
                var_azimuth = cov_matrix[azimuth_idx, azimuth_idx]
                var_zenith = cov_matrix[zenith_idx, zenith_idx]
                covar_zenith_azimuth = cov_matrix[zenith_idx, azimuth_idx]
                angular_resolution_rad = torch.sqrt(var_zenith + torch.sin(zenith)*var_azimuth + 2*torch.sin(zenith)*torch.cos(zenith)*covar_zenith_azimuth)
                resolution_per_event.append(angular_resolution_rad)
            resolution_per_event = torch.stack(resolution_per_event)
            total_resolution = torch.mean(resolution_per_event)
            return {'angular_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': signal_event_params}
        elif self.resolution_type == 'energy':
            for i, params in enumerate(signal_event_params):
                energy_idx = self.fisher_info_params.index('energy')
                cov_matrix = torch.inverse(total_fisher_info)
        
                # Energy resolution: sqrt(var_energy)/energy
                var_energy = cov_matrix[energy_idx, energy_idx]
                energy_resolution = torch.sqrt(var_energy)
            resolution_per_event.append(energy_resolution)
            resolution_per_event = torch.stack(resolution_per_event)
            total_resolution = torch.mean(resolution_per_event)

            return {'energy_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': signal_event_params}

class ResolutionLoss(FisherInfoLoss):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith'], resolution_type='angular'):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
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
        fisher_info_params : list of str
            List of event parameters to compute Fisher information for.
        resolution_type : str
            Type of resolution to compute ('angular' or 'energy').
        """
        super().__init__(device, print_loss, random_seed, fisher_info_params)
        
        self.resolution_type = resolution_type # 'angular' or 'energy'
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(FisherInfo).
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        event_params : list of dict
            List of dictionaries containing event parameters.
        surrogate_func : callable
            Function that computes light yield from event parameters.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)
        n_params = len(self.fisher_info_params)
        resolution_per_event = []
        for params in event_params:
            total_fisher_info = torch.zeros(n_params, n_params, device=self.device)
            for i, point in enumerate(points_3d):
                fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
                for _ in range(llr_iterations):
                    fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, params, surrogate_func, llr_net, signal_noise_scale, add_relative_pos=add_relative_pos)/len(event_params)
                total_fisher_info += fisher_matrix/llr_iterations
                
            if self.resolution_type == 'angular':
                zenith = params['zenith']
                # azimuth = params['azimuth']
                azimuth_idx = self.fisher_info_params.index('azimuth')
                zenith_idx = self.fisher_info_params.index('zenith')
                cov_matrix = torch.inverse(total_fisher_info)
        
                # Angular resolution: sqrt(var_azimuth + var_zenith)
                var_azimuth = cov_matrix[azimuth_idx, azimuth_idx]
                var_zenith = cov_matrix[zenith_idx, zenith_idx]
                covar_zenith_azimuth = cov_matrix[zenith_idx, azimuth_idx]
                angular_resolution_rad = torch.sqrt(var_azimuth + torch.sin(zenith)*var_azimuth + 2*torch.sin(zenith)*torch.cos(zenith)*covar_zenith_azimuth)

                resolution_per_event.append(angular_resolution_rad)

            elif self.resolution_type == 'energy':
                energy_idx = self.fisher_info_params.index('energy')
                cov_matrix = torch.inverse(total_fisher_info)
        
                # Energy resolution: sqrt(var_energy)/energy
                var_energy = cov_matrix[energy_idx, energy_idx]
                mean_energy = torch.mean(torch.tensor([params['energy'] for params in event_params], device=self.device))
                energy_resolution = torch.sqrt(var_energy)/mean_energy
                resolution_per_event.append(energy_resolution)
        resolution_per_event = torch.stack(resolution_per_event)
        total_resolution = torch.mean(resolution_per_event)
        if self.resolution_type == 'energy':
            return {'energy_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': event_params}
        elif self.resolution_type == 'angular':
            return {'angular_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': event_params}
