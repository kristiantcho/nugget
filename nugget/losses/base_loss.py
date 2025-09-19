import torch
import numpy as np
import torch.nn.functional as F

class LossFunction:
    """Base class for loss functions in geometry optimization."""
    
    def __init__(self, device=None):
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
  
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute the loss.
        
        Parameters:
        -----------
        geom_dict : dict
            Dictionary containing geometric information (e.g., points, strings).
        **kwargs : dict
            Additional arguments specific to the loss function.
            
        Returns:
        --------
        dict
            The computed loss value (ending in `_loss`) and other computed values.
        """
        raise NotImplementedError("Subclasses must implement __call__")







