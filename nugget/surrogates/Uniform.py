import torch
from nugget.surrogates.base_surrogate import Surrogate

class Uniform(Surrogate):
    
    def __init__(self, device=None, dim=3, domain_size=2, **kwargs):
        """
        Initialize the Uniform surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (2D or 3D)
        domain_size : int
            length of the domain 
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.kwargs = kwargs
        
    def __call__(self, opt_point, event_params=None):
        if opt_point.ndim == 1:
            opt_point = opt_point.unsqueeze(0)
        factor = self.kwargs.get('factor', 1.0)
        return factor * torch.ones(opt_point.shape[0], device=self.device, dtype=torch.float32)