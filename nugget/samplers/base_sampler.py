import torch

class Sampler:
    
    def __init__(self, device=None, dim=3, domain_size=2):
        """Base class for event and detector point sampler."""
        self.device = device if device is not None else torch.device("cpu")
        self.dim = dim
        self.domain_size = domain_size
        self.half_domain = domain_size / 2
        
    
    def sample_events(self, num_events, **kwargs):
        """
        Call the surrogate model with keyword arguments.
        
        Parameters:
        -----------
        num_events : int
            Number of events to sample
        kwargs : dict
            Keyword arguments for the sampler
            
        Returns:
        --------
        dict
            Dictionary with sampled event parameters
        """
        event_params = {}
        event_params['position'] = None
        event_params['energy'] = None
        event_params['zenith'] = None
        event_params['azimuth'] = None
        event_params['lepton'] = None
        
        raise NotImplementedError("Surrogate model not implemented.")
    
    def sample_detector_points(self, num_points, **kwargs):
        """
        Sample detector points in the geometry.
        
        Parameters:
        -----------
        num_points : int
            Number of detector points to sample
        kwargs : dict
            Additional arguments for sampling
            
        Returns:
        --------
        torch.Tensor
            Sampled detector points (num_points, dim)
        """
        raise NotImplementedError("Point sampling not implemented.")