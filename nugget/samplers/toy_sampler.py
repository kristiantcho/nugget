from nugget.samplers.base_sampler import Sampler
import torch
import numpy as np


class ToySampler(Sampler):
    
    def __init__(self, device=None, dim=3, domain_size=2, **kwargs):
        """Sampler for toy model."""
        super().__init__(device, dim, domain_size)
        self.kwargs = kwargs
        self.event_type = kwargs.get('event_type','signal')  # 'signal' or 'background'
    
    def sample_power_law(self, E_min=0.8, E_max=1, gamma=2.7, n_samples=1):
        """
        Sample from a power law distribution.
        
        Parameters:
        -----------
        E_min : float
            Minimum value
        E_max : float
            Maximum value
        gamma : float
            Power law index
            
        Returns:
        --------
        torch.Tensor
            Sampled value
        """
        
        r = torch.rand(n_samples, device=self.device)
        exponent = 1 - gamma
        return ((E_max**exponent - E_min**exponent) * r + E_min**exponent) ** (1 / exponent)
    

    def sample_background_zenith(self, a=1.5, n_samples=1):
        """
        Sample zenith angles for atmospheric background neutrinos.
        Biases toward the horizon using a simple analytical form.
        Returns angles in radians.
        
        The larger `a` is, the stronger the bias toward horizontal.
        """
        def pdf(cos_theta):
            return 1 + a * (1 - np.abs(cos_theta))

        cos_theta_vals = []
        n_accepted = 0
        n_trials = 0
        while n_accepted < n_samples:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(0, 1 + a)  # max of PDF = 1 + a
            if y <= pdf(x):
                cos_theta_vals.append(x)
                n_accepted += 1
            n_trials += 1
            if n_trials > 20 * n_samples:
                raise RuntimeError("Rejection sampling failed to converge")

        cos_theta = np.array(cos_theta_vals)
        theta = np.arccos(cos_theta)
        return torch.tensor(theta, device=self.device)
    
    def sample_events(self, num_events):
        """
        Sample event parameters for toy model.
        
        Parameters:
        -----------
        num_events : int
            Number of events to sample
        event_type : str
            Type of event ('signal' or 'background')
            
        Returns:
        --------
        list of dict
            List of event parameters
        """
        
        E_min = self.kwargs.get('E_min', 0.8)
        E_max = self.kwargs.get('E_max', 1)
        if self.event_type == 'signal':
            gamma = self.kwargs.get('gamma',2.7)
        else:
            gamma = self.kwargs.get('gamma',3.7)
        a = self.kwargs.get('a',1.5)
        x_bias = self.kwargs.get('x_bias',0)
        y_bias = self.kwargs.get('y_bias',0)
        z_bias = self.kwargs.get('z_bias',0)
        
        event_params_list = []
        for _ in range(num_events):
            event_params = {}
            event_params['energy'] = self.sample_power_law(n_samples=1, E_min=E_min, E_max=E_max, gamma=gamma)
            if self.event_type == 'signal':
                event_params['zenith'] = torch.rand(1, device=self.device) * np.pi
            else:
                event_params['zenith'] = self.sample_background_zenith(a, 1)
            # event_params['lepton'] = None
            event_params['azimuth'] = torch.rand(1, device=self.device) * 2 * np.pi
            event_params['position'] = torch.rand(1, 3, device=self.device) * self.domain_size*1.5 - self.domain_size*1.5/2
            
            event_params['position'] += torch.tensor([x_bias, y_bias, z_bias])*self.domain_size
            event_params_list.append(event_params)
        
        return event_params_list        
        
    def sample_detector_points(self, num_points):
        """Sample points within the detector volume.

        Args:
            num_points (int): Number of points to sample.

        Returns:
            torch.Tensor: Sampled points within the detector volume.
        """
        return torch.rand((num_points, self.dim), device=self.device, dtype=torch.float32) * self.domain_size - self.domain_size/2
        
        