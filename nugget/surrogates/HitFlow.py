import torch
import numpy as np
import math
from nugget.surrogates.base_surrogate import Surrogate
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform, Transform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
import time

class LogTransform(Transform):
    """Log transform for photon arrival times."""
    
    def forward(self, x, context=None):
        # x > 0
        y = torch.log10(x) / 5.0
        # y = log10(x)/5 => dy/dx = 1 / (5 * ln(10) * x)
        # log|det J| = sum_i [ -log(x_i) - log(5 * ln(10)) ]
        log_scale = math.log(5.0 * math.log(10.0))
        log_det = -torch.log(x).sum(dim=-1) - x.shape[-1] * log_scale
        return y, log_det

    def inverse(self, y, context=None):
        x = 10 ** (y * 5.0)
        # dx/dy = 5 * ln(10) * x
        # log|det J| = sum_i [ log(x_i) + log(5 * ln(10)) ]
        log_scale = math.log(5.0 * math.log(10.0))
        log_det = torch.log(x).sum(dim=-1) + y.shape[-1] * log_scale
        return x, log_det


class HitFlow(Surrogate):
    """
    HitFlow: A normalizing flow-based surrogate for photon arrival time distributions.
    
    This model learns to generate photon arrival time distributions (PATD) conditioned
    on event parameters and detector positions using normalizing flows.
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, **kwargs):
        """
        Initialize the HitFlow surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (must be 3D for this model)
        domain_size : int
            Length of the domain
        num_layers : int
            Number of flow layers (default: 10)
        hidden_features : int
            Number of hidden features in autoregressive transforms (default: 64)
        num_bins : int
            Number of bins for rational quadratic splines (default: 10)
        tail_bound : float
            Tail bound for spline transforms (default: 4)
        context_features : int
            Number of context features (default: 11 or 12 with varying cylinder)
        vary_cylinder : bool
            If True, randomly varies cylinder size during training (default: False)
        min_domain_size : float
            Minimum domain size when varying cylinder (default: 1000)
        max_domain_size : float
            Maximum domain size when varying cylinder (default: 5000)
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        self.num_layers = kwargs.get('num_layers', 10)
        self.hidden_features = kwargs.get('hidden_features', 64)
        self.num_bins = kwargs.get('num_bins', 10)
        self.tail_bound = kwargs.get('tail_bound', 4)
        self.vary_cylinder = kwargs.get('vary_cylinder', False)
        self.min_domain_size = kwargs.get('min_domain_size', 1000)
        self.max_domain_size = kwargs.get('max_domain_size', 5000)
        
        # Context features don't change with varying cylinder
        self.context_features = kwargs.get('context_features', 11)
        
        # Build the flow model
        self._build_flow()
        
        # Training parameters
        self.optimizer = None
        self.is_trained = False
        
    def _build_flow(self):
        """Build the normalizing flow architecture."""
        base_dist = StandardNormal(shape=[1])
        
        transforms = [LogTransform()]
        
        for _ in range(self.num_layers):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=self.hidden_features,
                    context_features=self.context_features,
                    num_bins=self.num_bins,
                    tails="linear",
                    tail_bound=self.tail_bound,
                    activation=torch.nn.SiLU()
                )
            )
        
        transform = CompositeTransform(transforms)
        self.flow = Flow(transform, base_dist).to(self.device)
        
    def create_event_features(self, event_params, coordinate, num_hits, t_geom_min=None):
        """
        Create feature vector from event parameters and detector position.
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'direction', 'energy'
        coordinate : torch.Tensor
            Detector position (x, y, z)
        num_hits : int or torch.Tensor
            Number of photon hits
        t_geom_min : torch.Tensor or None
            Minimum geometric time (optional)
            
        Returns:
        --------
        torch.Tensor
            Feature vector of shape (num_hits, context_features)
        """
        # Always use the model's fixed domain_size for normalization.
        # If domain_size is (width, height), normalize x,y by (width/2) and z by (height/2).
        domain_size = self.domain_size
        if isinstance(domain_size, torch.Tensor):
            domain_size = domain_size.item()

        if isinstance(domain_size, (tuple, list)) and len(domain_size) == 2:
            width, height = domain_size
            if isinstance(width, torch.Tensor):
                width = width.item()
            if isinstance(height, torch.Tensor):
                height = height.item()
            x_scale = width / 2.0
            y_scale = width / 2.0
            z_scale = height / 2.0
        else:
            # Original behavior: divide all coordinates by scalar domain_size
            x_scale = domain_size/2
            y_scale = domain_size/2
            z_scale = domain_size/2
        
        # Extract and scale parameters
        energy = torch.log10(event_params['energy']).squeeze()/8.0
        
        # Scale coordinates
        x = coordinate[0] / x_scale
        y = coordinate[1] / y_scale
        z = coordinate[2] / z_scale
        
        # Vertex position
        v_x = event_params['position'][0][0] / x_scale
        v_y = event_params['position'][0][1] / y_scale
        v_z = event_params['position'][0][2] / z_scale
        
        # Direction§
        d_x = event_params['direction'][0]
        d_y = event_params['direction'][1]
        d_z = event_params['direction'][2]
        
        # Log of number of hits
        if isinstance(num_hits, torch.Tensor):
            log_hits = torch.log10(num_hits.float())
        else:
            log_hits = torch.log10(torch.tensor(num_hits, dtype=torch.float32, device=self.device))/6.0
        
        # Build feature list
        feature_list = [energy, x, y, z, v_x, v_y, v_z, d_x, d_y, d_z, log_hits]
        
        if t_geom_min is not None:
            feature_list.append(torch.log10(t_geom_min))
        
        # Stack features and repeat for each hit
        features = torch.stack([f.to(self.device) for f in feature_list]).unsqueeze(0)
        
        # Convert num_hits to int
        num_hits_int = int(num_hits.item()) if isinstance(num_hits, torch.Tensor) else int(num_hits)
        features = features.repeat(num_hits_int, 1)
        
        return features
    
    def train_model(self, event_sampler, light_yield_surrogate_func, num_iterations=1000, sampling_timeout=None,
                    epoch_size=800, batch_size=None, lr=1e-4, min_hits=10, max_hits_per_event=None, save_interval=100, verbose=True, save_path=None,):
        """
        Train the HitFlow model on randomly sampled events.
        
        Parameters:
        -----------
        event_sampler : Sampler
            Sampler object with sample_events() method
        light_yield_surrogate_func : callable
            Function that takes opt_point and event_params and returns PATD dict
            with 'hit_times' and 'num_photons' keys
        num_iterations : int
            Number of training iterations
        epoch_size : int
            Maximum number of samples per epoch
        batch_size : int or None
            Number of hits per batch for training. If None, all epoch data is processed in one step
        lr : float
            Learning rate
        min_hits : int
            Minimum number of hits required per event
        max_hits_per_event : int or None
            Maximum number of hits to use per event (for memory management)
        save_interval : int
            Interval at which to print/save training progress
        verbose : bool
            Whether to print training progress
        sampling_timeout : int or None
            Maximum attempts to find an event with min_hits
        save_path : str or None
            Path to save intermediate models (if None, no saving)    
        Returns:
        --------
        list
            Training losses
        """
        self.flow.train()
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        
        losses = []
        print("Starting HitFlow training...", flush=True)
        
        # Print initial model summary
        if verbose:
            total_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
            print(f"Model has {total_params} trainable parameters.", flush=True)
            print(self.flow, flush=True)
            
        for iteration in range(num_iterations):
            epoch_start_time = time.time()
            
            # Step 1: Collect all hits and features for the epoch
            all_hit_times = []
            all_features = []
            total_hits = 0
            
            if verbose and iteration == 0:
                sampling_start = time.time()
            
            while total_hits < epoch_size:
                # Randomly vary cylinder size if enabled
                if self.vary_cylinder:
                    current_domain_size = np.random.uniform(self.min_domain_size, self.max_domain_size)
                    if hasattr(event_sampler, 'cylinder'):
                        event_sampler.cylinder.height = torch.tensor(current_domain_size, dtype=torch.float32, device=self.device)
                        event_sampler.cylinder.radius = torch.tensor(current_domain_size / 2.0, dtype=torch.float32, device=self.device)
                
                # Sample event and detector position
                ly = 0
                signal_event = event_sampler.sample_events(1)
                counter = 0
                
                while ly < min_hits:
                    coordinate = event_sampler.sample_detector_points(1)
                    patd_dict = light_yield_surrogate_func(
                        opt_point=coordinate, 
                        event_params=signal_event[0]
                    )
                    ly = patd_dict['num_photons']
                    counter += 1
                    if sampling_timeout is not None and counter >= sampling_timeout:
                        break
                        
                if ly < min_hits and sampling_timeout is not None and counter >= sampling_timeout:
                    continue
                
                # Get hit times and shift to start near zero
                hit_times = patd_dict['hit_times']
                min_time = torch.min(hit_times)
                if max_hits_per_event is not None and len(hit_times) > max_hits_per_event:
                    # randomly sample max_hits_per_event from hit_times
                    indices = torch.randperm(len(hit_times))[:max_hits_per_event]
                    hit_times = hit_times[indices]
                    # hit_times = hit_times[:max_hits_per_event]
                    
                
                hit_times_shifted = (hit_times - min_time + 1e-5).unsqueeze(1)
                
                # Create context features
                features_batch = self.create_event_features(
                    signal_event[0], 
                    coordinate.squeeze(0), 
                    len(hit_times_shifted)
                )
                
                # Store hits and features
                all_hit_times.append(hit_times_shifted)
                all_features.append(features_batch)
                total_hits += len(hit_times_shifted)
            
            if verbose and iteration == 0:
                print(f'Collected {total_hits} hits from {len(all_hit_times)} events in {time.time() - sampling_start:.2f} seconds.', flush=True)
            
            # Step 2: Concatenate all data
            all_hit_times_tensor = torch.cat(all_hit_times, dim=0)
            all_features_tensor = torch.cat(all_features, dim=0)
            
            # Step 3: Train on batches
            total_loss = 0.0
            num_batches = 0
            
            if verbose and iteration == 0:
                training_start = time.time()
            
            # If batch_size is None, process all data in one optimization step
            if batch_size is None:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute log probabilities for all data
                log_probs = self.flow.log_prob(all_hit_times_tensor, context=all_features_tensor)
                
                # Compute loss (negative mean log probability)
                loss = -log_probs.mean()
                
                # Backward pass
                loss.backward()
                
                # Optimization step
                self.optimizer.step()
                
                # Track loss for reporting
                total_loss = loss.item()
                num_batches = 1
                loss = total_loss
            else:
                # Process in batches with optimization step per batch
                for i in range(0, len(all_hit_times_tensor), batch_size):
                    batch_hit_times = all_hit_times_tensor[i:i+batch_size]
                    batch_features = all_features_tensor[i:i+batch_size]
                    
                    # Zero gradients for this batch
                    self.optimizer.zero_grad()
                    
                    # Compute log probabilities for this batch
                    log_probs = self.flow.log_prob(batch_hit_times, context=batch_features)
                    
                    # Compute loss (negative mean log probability)
                    batch_loss = -log_probs.mean()
                    
                    # Backward pass
                    batch_loss.backward()
                    
                    # Optimization step
                    self.optimizer.step()
                    
                    # Track loss for reporting
                    total_loss += batch_loss.item()
                    num_batches += 1
                
                # Average loss across all batches for reporting
                loss = total_loss / num_batches
            
            if verbose and iteration == 0:
                print(f'Completed training on {num_batches} batches in {time.time() - training_start:.2f} seconds.', flush=True)
            
            losses.append(loss)
            
            if verbose and ((iteration + 1) % save_interval == 0 or iteration == num_iterations - 1):
                epoch_time = time.time() - epoch_start_time
                print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s", flush=True)
                
            if save_path is not None and ((iteration + 1) % save_interval == 0 or iteration == num_iterations - 1):
                self.save_model(f"{save_path}")
        
        self.is_trained = True
        if verbose:
            print(f"HitFlow training completed.", flush=True)
        return losses
    
    def light_yield_surrogate(self, **kwargs):
        """
        Generate photon arrival time distribution using the trained flow.
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'direction', 'energy'
        opt_point : torch.Tensor
            Detector position (single point)
        light_yield : int or torch.Tensor
            Number of photons to generate
        min_hit_time : float or torch.Tensor
            Minimum hit time to shift the distribution
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'hit_times': Tensor of photon hit times
            - 'num_photons': Number of photons generated
        """
        # if not self.is_trained:
        #     raise RuntimeError("HitFlow model must be trained before generating samples. "
        #                      "Call train_model() first.")
        
        # Extract parameters
        event_params = kwargs.get('event_params', None)
        opt_point = kwargs.get('opt_point', None)
        light_yield = kwargs.get('light_yield', None)
        min_hit_time = kwargs.get('min_hit_time', 0.0)
        
        if event_params is None or opt_point is None or light_yield is None:
            raise ValueError("event_params, opt_point, and light_yield are required")
        
        # Convert light_yield to int
        num_photons = int(light_yield.item()) if isinstance(light_yield, torch.Tensor) else int(light_yield)
        
        # Convert min_hit_time to tensor
        if not isinstance(min_hit_time, torch.Tensor):
            min_hit_time = torch.tensor(min_hit_time, dtype=torch.float32, device=self.device)
        
        # Ensure opt_point is on the correct device
        if isinstance(opt_point, torch.Tensor):
            opt_point = opt_point.to(self.device).squeeze()
        else:
            opt_point = torch.tensor(opt_point, dtype=torch.float32, device=self.device).squeeze()
        
        # Create context features (use first row as they're all identical)
        features = self.create_event_features(
            event_params, 
            opt_point, 
            num_photons
        )
        
        # Generate samples from the flow
        with torch.no_grad():
            samples = self.flow.sample(num_photons, context=features[0].unsqueeze(0))
        
        # Shift samples by minimum hit time
        hit_times = samples.squeeze() + min_hit_time
        
        # Return PATD dict
        return {
            'hit_times': hit_times,
            'num_photons': num_photons
        }
    
    def evaluate_pdf(self, hit_times, event_params, opt_point, min_hit_time=None):
        """
        Evaluate the log probability density of provided hit times under the trained flow.
        
        Parameters:
        -----------
        hit_times : torch.Tensor
            Photon hit times to evaluate (can be 1D tensor)
        event_params : dict
            Contains 'position', 'direction', 'energy'
        opt_point : torch.Tensor
            Detector position (single point)
        min_hit_time : float or torch.Tensor
            Minimum hit time used to shift the distribution (should match training)
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'log_prob': Sum of log probabilities for all hit times
            - 'log_probs': Individual log probabilities for each hit time
            - 'num_photons': Number of photons evaluated
        """
        # if not self.is_trained:
        #     raise RuntimeError("HitFlow model must be trained before evaluating. "
        #                      "Call train_model() first.")
        
        # Convert min_hit_time to tensor
        if min_hit_time is None:
            min_hit_time = torch.tensor(min(hit_times), dtype=torch.float32, device=self.device)
        elif not isinstance(min_hit_time, torch.Tensor):
            min_hit_time = torch.tensor(min_hit_time, dtype=torch.float32, device=self.device)
        
        # Ensure hit_times is on the correct device and has proper shape
        if isinstance(hit_times, torch.Tensor):
            hit_times = hit_times.to(self.device)
        else:
            hit_times = torch.tensor(hit_times, dtype=torch.float32, device=self.device)
        
        # Ensure 1D and shift by min_hit_time
        if hit_times.dim() > 1:
            hit_times = hit_times.squeeze()
        hit_times_shifted = (hit_times - min_hit_time +1e-3).unsqueeze(1)
        
        # Ensure opt_point is on the correct device
        if isinstance(opt_point, torch.Tensor):
            opt_point = opt_point.to(self.device).squeeze()
        else:
            opt_point = torch.tensor(opt_point, dtype=torch.float32, device=self.device).squeeze()
        
        num_hits = len(hit_times_shifted)
        
        # Create context features
        features = self.create_event_features(
            event_params, 
            opt_point, 
            num_hits
        )
        
        # Set flow to eval mode and evaluate log probabilities
        self.flow.eval()
        log_probs = self.flow.log_prob(hit_times_shifted, context=features)
        
        # Sum log probabilities
        total_log_prob = torch.sum(log_probs)
        
        return {
            'log_prob': total_log_prob,
            'log_probs': log_probs,
            'num_photons': num_hits
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        torch.save({
            'flow_state_dict': self.flow.state_dict(),
            'num_layers': self.num_layers,
            'hidden_features': self.hidden_features,
            'num_bins': self.num_bins,
            'tail_bound': self.tail_bound,
            'context_features': self.context_features,
            'is_trained': self.is_trained,
            'device': str(self.device),
            'dim': self.dim,
            'domain_size': self.domain_size,
            'vary_cylinder': self.vary_cylinder,
            'min_domain_size': self.min_domain_size,
            'max_domain_size': self.max_domain_size
        }, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update model parameters
        self.num_layers = checkpoint['num_layers']
        self.hidden_features = checkpoint['hidden_features']
        self.num_bins = checkpoint['num_bins']
        self.tail_bound = checkpoint['tail_bound']
        self.context_features = checkpoint['context_features']
        self.is_trained = checkpoint['is_trained']
        
        # Load cylinder variation parameters if available
        # self.vary_cylinder = checkpoint.get('vary_cylinder', False)
        # self.min_domain_size = checkpoint.get('min_domain_size', 1000)
        # self.max_domain_size = checkpoint.get('max_domain_size', 5000)
        
        # Rebuild and load flow
        self._build_flow()
        self.flow.load_state_dict(checkpoint['flow_state_dict'])
        self.flow.eval()
