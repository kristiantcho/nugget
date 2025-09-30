from nugget.losses.base_loss import LossFunction
import torch

class LightYieldLoss(LossFunction):
    def __init__(self, device=None, print_loss=False, ):
        """
        Initialize the light yield loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
        """
        super().__init__(device)
        self.print_loss = print_loss


    def __call__(self, geom_dict, **kwargs):
        loss = torch.tensor(0.0, device=self.device)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        event_params = kwargs.get('signal_event_params', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        noise_scale = kwargs.get('signal_noise_scale', 0.0)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)

        for params in event_params:
            # Compute light yield for each event
            light_yield = surrogate_func(opt_point=points_3d, event_params=params)  # Shape (num_points,)
            # We want to maximize light yield
            if noise_scale > 0.0:
                light_yield = light_yield + light_yield*torch.randn(size=light_yield.shape, device=self.device) * noise_scale
            loss += 1/(torch.sum(light_yield)/len(event_params) + 1e-6)  # Add small value to avoid division by zero  
        if self.print_loss:
            print(f"Light yield loss: {loss.item()}")
        return {'signal_yield_loss': loss, 'signal_yield_per_point': light_yield/len(event_params), 'total_signal_yield': torch.sum(light_yield)/len(event_params)}

class WeightedLightYieldLoss(LossFunction):
    
    def __init__(self, device=None, print_loss=False):
        """
        Initialize the light yield loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
        """
        super().__init__(device)
        self.print_loss = print_loss
    

    def light_yield_per_string(self, surrogate_func, event_params, string_xy, points_3d, noise_scale=0.0):
        
        signal_yield_per_string = torch.zeros(len(string_xy), device=self.device)
        n_strings = len(string_xy)
        for s_idx in range(n_strings):
            # Create optimization points for this string
            # Sample points along the string (assuming strings extend in z-direction)
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]    
            
            signal_yield_for_string = torch.zeros(len(string_points), device=self.device)
        
            for point_idx, point in enumerate(string_points):
                # Compute signal light yield for this point
                signal_yields = []
                for params in event_params:
                    signal_yield = surrogate_func(opt_point=point, event_params=params)  # Shape (num_points,)
                    if noise_scale > 0.0:
                        signal_yield = signal_yield + signal_yield*torch.randn(size=signal_yield.shape, device=self.device) * noise_scale
                    
                    signal_yields.append(signal_yield)
                
                
                # Convert to tensors and compute average
                signal_yields = torch.stack(signal_yields)
                avg_signal = torch.mean(signal_yields)
                signal_yield_for_string[point_idx] = avg_signal
            # summed SNR across all points on this string
      
                signal_yield_per_string[s_idx] = torch.sum(signal_yield_for_string)
        return signal_yield_per_string
    
    def __call__(self, geom_dict, **kwargs):
        
        precomputed_light_yield = kwargs.get('precomputed_signal_yield_per_string', None)
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
        
        if precomputed_light_yield is None:
            signal_yield_per_string = self.light_yield_per_string(surrogate_func, event_params, string_xy, points_3d, noise_scale)
        else:
            signal_yield_per_string = precomputed_light_yield
        if string_weights is None:
            total_light_yield = torch.sum(signal_yield_per_string)  # Sum over strings
        else:
            string_probs = torch.sigmoid(string_weights)
            total_light_yield = torch.sum(signal_yield_per_string * string_probs)  # Weighted sum
        light_yield_loss = 1/(total_light_yield + 1e-6)  # Add small value to avoid division by zero
        if self.print_loss:
            print(f"Weighted light yield loss: {light_yield_loss.item()}")
        
        # return light_yield_loss, signal_yield_per_string, total_light_yield
        return {'signal_yield_loss': light_yield_loss, 'signal_yield_per_string': signal_yield_per_string, 'total_signal_yield': total_light_yield}

        

