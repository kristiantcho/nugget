from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F


def compute_fisher_info_single(fisher_info_params, point, event_params, surrogate_func):
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
    light_yield_mean = surrogate_func(opt_point=point, event_params=grad_event_params)  # Shape (1,)
        
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
                fisher_element = (grad_i * grad_j) / light_yield_mean
                fisher_matrix[i_param, j_param] += fisher_element.item()
            
    return fisher_matrix

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
        event_params = kwargs.get('event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        n_params = len(self.fisher_info_params)
        total_fisher_info = torch.zeros(n_params, n_params, device=self.device)
        
        for point in points_3d:
            fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
            for params in event_params:
                fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, params, surrogate_func)/len(event_params)
            total_fisher_info += fisher_matrix
            
        fisher_loss = 1/(torch.det(total_fisher_info) + 1e-6)  # Add small value to diagonal for numerical stability
        
        if self.print_loss:
            print(f"Fisher Info Loss: {fisher_loss.item()}")
        
        # return fisher_loss, total_fisher_info
        return {'fisher_loss': fisher_loss, 'total_fisher_info': total_fisher_info}
        
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

    def compute_fisher_info_per_string(self, string_xy, points_3d, signal_event_params, signal_surrogate_func):
        """
        Compute the Fisher information for each string.
        
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
            fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
            for point in string_points:  
                for signal_params in signal_event_params:
                    fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, signal_params=signal_params, surrogate_func=signal_surrogate_func)/len(signal_event_params)
            fisher_info_per_string[s_idx] += fisher_matrix
            
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
        precomputed_fisher_info_per_string = kwargs.get('precomputed_fisher_info_per_string', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        # background_event_params = kwargs.get('background_event_params', None)
        # background_surrogate_func = kwargs.get('background_surrogate_func', None)
        if precomputed_fisher_info_per_string is None:
            fisher_info_per_string = self.compute_fisher_info_per_string(string_xy, points_3d, signal_event_params, signal_surrogate_func)
        else:
            fisher_info_per_string = precomputed_fisher_info_per_string
        if string_weights is None:
            total_fisher_info = torch.sum(fisher_info_per_string, dim=0)  # Sum over strings, keep matrix form
        else:
            string_probs = torch.sigmoid(string_weights)
            total_fisher_info = torch.sum(fisher_info_per_string * string_probs.unsqueeze(1), dim=0)  # Weighted sum

        fisher_loss = 1/(torch.det(total_fisher_info) + 1e-6)  # Add small value to diagonal for numerical stability
        
        
        # return fisher_loss, fisher_info_per_string, total_fisher_info
        return {'fisher_loss': fisher_loss, 'fisher_info_per_string': fisher_info_per_string, 'total_fisher_info': total_fisher_info}