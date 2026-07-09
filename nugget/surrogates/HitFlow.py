import torch
import numpy as np
import math
from nugget.surrogates.base_surrogate import Surrogate
# from nflows.distributions.normal import StandardNormal
# from nflows.flows.base import Flow
# from nflows.transforms.base import CompositeTransform, Transform
# from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
import time

class LogTransform(Transform):
    """Log transform for photon arrival times."""
    def __init__(self, scale_fac=1.0):
        super().__init__()
        self.scale_fac = scale_fac
    def forward(self, x, context=None):
        # x > 0
        y = torch.log10(x) / self.scale_fac
        # y = log10(x)/scale_fac => dy/dx = 1 / (scale_fac * ln(10) * x)
        # log|det J| = sum_i [ -log(x_i) - log(scale_fac * ln(10)) ]
        log_scale = math.log(self.scale_fac * math.log(10.0))
        log_det = -torch.log(x).sum(dim=-1) - x.shape[-1] * log_scale
        return y, log_det

    def inverse(self, y, context=None):
        x = 10 ** (y * self.scale_fac)
        # dx/dy = scale_fac * ln(10) * x
        # log|det J| = sum_i [ log(x_i) + log(scale_fac * ln(10)) ]
        log_scale = math.log(self.scale_fac * math.log(10.0))
        log_det = torch.log(x).sum(dim=-1) + y.shape[-1] * log_scale
        return x, log_det


class MinGeomLogTransform(Transform):
    """Shift by the geometric-time minimum, scale, and apply log10."""
    def __init__(self, scale_fac=1.0, geom_time_scale=1e5, eps=1e-12):
        super().__init__()
        self.scale_fac = scale_fac
        self.geom_time_scale = geom_time_scale
        self.eps = eps

    def _t_geom_min(self, x, context):
        if context is None or context.shape[-1] < 13:
            return x.new_zeros(x.shape[:-1] + (1,))
        return context[:, -2].unsqueeze(1)

    def forward(self, x, context=None):
        t_geom_min = self._t_geom_min(x, context)
        shifted_scaled = (x - t_geom_min) / self.geom_time_scale
        shifted_scaled = torch.clamp(shifted_scaled, min=self.eps)
        y = torch.log10(shifted_scaled) / self.scale_fac

        log_scale = math.log(self.scale_fac * math.log(10.0))
        log_det = (
            -math.log(self.geom_time_scale) * x.shape[-1]
            - torch.log(shifted_scaled).sum(dim=-1)
            - x.shape[-1] * log_scale
        )
        return y, log_det

    def inverse(self, y, context=None):
        t_geom_min = self._t_geom_min(y, context)
        shifted_scaled = 10 ** (y * self.scale_fac)
        x = shifted_scaled * self.geom_time_scale + t_geom_min

        log_scale = math.log(self.scale_fac * math.log(10.0))
        log_det = (
            math.log(self.geom_time_scale) * y.shape[-1]
            + torch.log(shifted_scaled).sum(dim=-1)
            + y.shape[-1] * log_scale
        )
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
        
        self.num_layers = kwargs.get('num_layers', 3)
        self.hidden_features = kwargs.get('hidden_features', 800)
        self.num_bins = kwargs.get('num_bins', 4)
        self.tail_bound = kwargs.get('tail_bound', 5)
        self.vary_cylinder = kwargs.get('vary_cylinder', False)
        self.min_domain_size = kwargs.get('min_domain_size', 1000)
        self.max_domain_size = kwargs.get('max_domain_size', 5000)
        self.scale_fac = kwargs.get('scale_fac', 4.0)
        self.geom_time_scale = kwargs.get('geom_time_scale', 1e5)
        self.beam_distance_scale = kwargs.get('beam_distance_scale', 1250.0)
        self.use_min_hit_time = kwargs.get('use_min_hit_time', True)
        self.use_geom_time_shift = kwargs.get('use_geom_time_shift', True)
        self.shuffle_training_batches = kwargs.get('shuffle_training_batches', True)
        self.reduce_lr_on_plateau = kwargs.get('reduce_lr_on_plateau', False)
        self.lr_scheduler_factor = kwargs.get('lr_scheduler_factor', 0.5)
        self.lr_scheduler_patience = kwargs.get('lr_scheduler_patience', 30)
        self.lr_scheduler_threshold = kwargs.get('lr_scheduler_threshold', 1e-4)
        self.lr_scheduler_min_lr = kwargs.get('lr_scheduler_min_lr', 1e-6)
        
        self.context_features = kwargs.get('context_features', 14)
        
        # Physics parameters for geometric time calculation
        self.refractive_index = kwargs.get('refractive_index', 1.33)  # Water/ice refractive index
        self.v_mu = kwargs.get('v_mu', 0.299792458)  # Muon velocity in m/ns (speed of light)
        
        # Build the flow model
        self._build_flow()
        
        # Training parameters
        self.optimizer = None
        self.is_trained = False
        
    def _build_flow(self):
        """Build the normalizing flow architecture."""
        base_dist = StandardNormal(shape=[1])
        
        if self.use_geom_time_shift:
            transforms = [
                MinGeomLogTransform(
                    scale_fac=self.scale_fac,
                    geom_time_scale=self.geom_time_scale,
                )
            ]
        else:
            transforms = [LogTransform(scale_fac=self.scale_fac)]
        
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

    def calculate_t_geom_min(self, event_params, coordinate):
        """
        Calculate the minimum geometric time using LightSabre formula.
        
        t_geom_min = foot_length / (c / n) + t_foot / v_mu
        
        Where:
        - foot_length: perpendicular distance from detector to the muon track
        - t_foot: distance along track to closest approach (foot point)
        - c: speed of light (0.299792458 m/ns)
        - n: refractive index of medium
        - v_mu: muon velocity
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'direction' (or 'zenith'/'azimuth'), and 'energy'
        coordinate : torch.Tensor
            Detector position (x, y, z)
            
        Returns:
        --------
        torch.Tensor or None
            Calculated t_geom_min, or None if direction cannot be determined
        """
        c = 0.299792458  # speed of light in m/ns
        
        # Convert coordinate to tensor
        if isinstance(coordinate, torch.Tensor):
            det_pos = coordinate.to(self.device).squeeze()
        else:
            det_pos = torch.tensor(coordinate, dtype=torch.float32, device=self.device).squeeze()
        
        # Convert vertex position to tensor
        if isinstance(event_params['position'], torch.Tensor):
            vertex_pos = event_params['position'].to(self.device).squeeze()
        else:
            vertex_pos = torch.tensor(event_params['position'], dtype=torch.float32, device=self.device).squeeze()
        
        # Get direction vector
        direction = event_params.get('direction', None)
        if direction is None:
            zenith = event_params.get('zenith', None)
            azimuth = event_params.get('azimuth', None)
            if zenith is None or azimuth is None:
                return None  # Cannot compute without direction
            theta = zenith.to(self.device).squeeze() if isinstance(zenith, torch.Tensor) else torch.tensor(zenith, dtype=torch.float32, device=self.device).squeeze()
            phi = azimuth.to(self.device).squeeze() if isinstance(azimuth, torch.Tensor) else torch.tensor(azimuth, dtype=torch.float32, device=self.device).squeeze()
            track_dir = torch.stack([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ])
        else:
            if not isinstance(direction, torch.Tensor):
                track_dir = torch.tensor(direction, dtype=torch.float32, device=self.device).squeeze()
            else:
                track_dir = direction.to(self.device).squeeze()
        
        # Normalize direction
        track_dir_norm = torch.norm(track_dir).clamp_min(1e-12)
        track_dir_normalized = track_dir / track_dir_norm
        
        # Compute vector from vertex to detector
        to_detector = det_pos - vertex_pos
        
        # Compute t_foot: distance along track to foot point (closest approach)
        t_foot = torch.dot(to_detector, track_dir_normalized)
        
        # Compute foot_length: perpendicular distance from detector to track
        cross_product = torch.cross(to_detector, track_dir_normalized, dim=0)
        foot_length = torch.norm(cross_product)
        
        # Compute t_geom_min using LightSabre formula
        t_geom_min = foot_length / (c / self.refractive_index) + t_foot / self.v_mu
        
        return t_geom_min

    def calculate_distance_to_beam(self, event_params, coordinate):
        """Compute the perpendicular distance from a detector point to the event beam."""
        if isinstance(coordinate, torch.Tensor):
            coordinate = coordinate.to(self.device).squeeze()
        else:
            coordinate = torch.tensor(coordinate, dtype=torch.float32, device=self.device).squeeze()

        if isinstance(event_params['position'], torch.Tensor):
            vertex = event_params['position'].to(self.device).squeeze()
        else:
            vertex = torch.tensor(event_params['position'], dtype=torch.float32, device=self.device).squeeze()

        direction = event_params.get('direction', None)
        if direction is None:
            zenith = event_params.get('zenith', None)
            azimuth = event_params.get('azimuth', None)
            if zenith is None or azimuth is None:
                raise ValueError("event_params must contain either 'direction' or both 'zenith' and 'azimuth'")
            theta = zenith.to(self.device).squeeze() if isinstance(zenith, torch.Tensor) else torch.tensor(zenith, dtype=torch.float32, device=self.device).squeeze()
            phi = azimuth.to(self.device).squeeze() if isinstance(azimuth, torch.Tensor) else torch.tensor(azimuth, dtype=torch.float32, device=self.device).squeeze()
            direction = torch.stack([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ])
        elif not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32, device=self.device).squeeze()
        else:
            direction = direction.to(self.device).squeeze()

        direction_norm = torch.norm(direction).clamp_min(1e-12)
        return torch.norm(torch.cross(coordinate - vertex, direction, dim=0)) / direction_norm

    def _prepare_scalar_feature(self, value, default=0.0):
        if value is None:
            return torch.tensor(default, dtype=torch.float32, device=self.device)
        if isinstance(value, torch.Tensor):
            return value.to(self.device).squeeze().to(torch.float32)
        return torch.tensor(value, dtype=torch.float32, device=self.device)
        
    def create_event_features(self, event_params, coordinate, num_hits, t_geom_min=None, d_beam=None):
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
        d_beam : torch.Tensor or None
            Optional beam-distance feature. If omitted, it is computed from the
            event vertex and direction when the extended context expects it.
            
        Returns:
        --------
        torch.Tensor
            Feature vector of shape (num_hits, context_features)
        """
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
            x_scale = domain_size / 2.0
            y_scale = domain_size / 2.0
            z_scale = domain_size / 2.0

        if isinstance(coordinate, torch.Tensor):
            coordinate = coordinate.to(self.device).squeeze()
        else:
            coordinate = torch.tensor(coordinate, dtype=torch.float32, device=self.device).squeeze()

        energy = self._prepare_scalar_feature(event_params['energy'])
        energy = torch.log10(torch.clamp(energy, min=1e-12)) / 8.0

        x = coordinate[0] / x_scale
        y = coordinate[1] / y_scale
        z = coordinate[2] / z_scale

        if isinstance(event_params['position'], torch.Tensor):
            vertex = event_params['position'].to(self.device).squeeze()
        else:
            vertex = torch.tensor(event_params['position'], dtype=torch.float32, device=self.device).squeeze()
        v_x = vertex[0] / x_scale
        v_y = vertex[1] / y_scale
        v_z = vertex[2] / z_scale

        direction = event_params.get('direction', None)
        if direction is None:
            zenith = event_params.get('zenith', None)
            azimuth = event_params.get('azimuth', None)
            if zenith is None or azimuth is None:
                raise ValueError("event_params must contain either 'direction' or both 'zenith' and 'azimuth'")
            theta = zenith.to(self.device).squeeze() if isinstance(zenith, torch.Tensor) else torch.tensor(zenith, dtype=torch.float32, device=self.device).squeeze()
            phi = azimuth.to(self.device).squeeze() if isinstance(azimuth, torch.Tensor) else torch.tensor(azimuth, dtype=torch.float32, device=self.device).squeeze()
            direction = torch.stack([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ])
        elif not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32, device=self.device).squeeze()
        else:
            direction = direction.to(self.device).squeeze()

        d_x = direction[0]
        d_y = direction[1]
        d_z = direction[2]

        vertex_to_detector = coordinate - vertex
        vert_dist = torch.norm(vertex_to_detector)
        direction_norm = torch.norm(direction).clamp_min(1e-12)
        cos_angle = torch.dot(direction, vertex_to_detector) / (direction_norm * vert_dist.clamp_min(1e-12))

        if isinstance(num_hits, torch.Tensor):
            num_hits_int = int(num_hits.item())
        else:
            num_hits_int = int(num_hits)

        if self.context_features <= 11:
            if isinstance(num_hits, torch.Tensor):
                log_hits = torch.log10(num_hits.float().clamp_min(1.0)) / 4.0
            else:
                log_hits = torch.log10(torch.tensor(max(num_hits_int, 1), dtype=torch.float32, device=self.device)) / 4.0
            feature_list = [energy, x, y, z, v_x, v_y, v_z, d_x, d_y, d_z, log_hits]
        else:
            feature_list = [energy, x, y, z, v_x, v_y, v_z, d_x, d_y, d_z, vert_dist, cos_angle]

            if self.context_features >= 13:
                if t_geom_min is None:
                    t_geom_min = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    t_geom_min = self._prepare_scalar_feature(t_geom_min)
                feature_list.append(t_geom_min / self.geom_time_scale)

            if self.context_features >= 14:
                if d_beam is None:
                    d_beam = self.calculate_distance_to_beam(event_params, coordinate)
                else:
                    d_beam = self._prepare_scalar_feature(d_beam)
                feature_list.append(d_beam / self.beam_distance_scale)

        features = torch.stack([f.to(self.device) for f in feature_list]).unsqueeze(0)
        features = features.repeat(num_hits_int, 1)
        return features
    
    def train_model(self, event_sampler, light_yield_surrogate_func, num_iterations=1000, sampling_timeout=None,
                    epoch_size=800, batch_size=None, lr=1e-4, min_hits=10, max_hits_per_event=None, save_interval=100,
                    verbose=True, save_path=None, shuffle_training_batches=None, reduce_lr_on_plateau=None,
                    lr_scheduler_factor=None, lr_scheduler_patience=None, lr_scheduler_threshold=None,
                    lr_scheduler_min_lr=None, min_hits_levels=None, max_hits_per_subsample=None):
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
            Minimum number of hits required per event (used if min_hits_levels is None)
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
        shuffle_training_batches : bool or None
            Whether to shuffle concatenated hits/features before mini-batching.
            If None, uses the model configuration.
        reduce_lr_on_plateau : bool or None
            Whether to enable ReduceLROnPlateau on the optimizer.
            If None, uses the model configuration.
        lr_scheduler_factor : float or None
            Multiplicative LR reduction factor for ReduceLROnPlateau.
        lr_scheduler_patience : int or None
            Number of epochs with no improvement before reducing LR.
        lr_scheduler_threshold : float or None
            Minimum loss improvement treated as progress by the scheduler.
        lr_scheduler_min_lr : float or None
            Lower bound for the learning rate when using ReduceLROnPlateau.
        min_hits_levels : list or None
            List of minimum hit thresholds to cycle through during training.
            If provided, training cycles through these levels and ignores min_hits.
            Example: [1, 10, 100, 1000]
        max_hits_per_subsample : int or None
            Maximum number of hits to randomly subsample from each event.
            If provided, limits the number of hits used per event coordinate pair.
            Example: 10 (max 10 hits per event)
        Returns:
        --------
        list
            Training losses
        """
        self.flow.train()
        if shuffle_training_batches is None:
            shuffle_training_batches = self.shuffle_training_batches
        if reduce_lr_on_plateau is None:
            reduce_lr_on_plateau = self.reduce_lr_on_plateau
        if lr_scheduler_factor is None:
            lr_scheduler_factor = self.lr_scheduler_factor
        if lr_scheduler_patience is None:
            lr_scheduler_patience = self.lr_scheduler_patience
        if lr_scheduler_threshold is None:
            lr_scheduler_threshold = self.lr_scheduler_threshold
        if lr_scheduler_min_lr is None:
            lr_scheduler_min_lr = self.lr_scheduler_min_lr

        # Setup min_hits cycling if levels are provided
        use_min_hits_cycling = min_hits_levels is not None and len(min_hits_levels) > 0
        if use_min_hits_cycling:
            num_levels = len(min_hits_levels)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = None
        if reduce_lr_on_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                threshold=lr_scheduler_threshold,
                min_lr=lr_scheduler_min_lr,
            )
        
        losses = []
        print("Starting HitFlow training...", flush=True)
        
        # Print initial model summary
        if verbose:
            total_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
            print(f"Model has {total_params} trainable parameters.", flush=True)
            print(self.flow, flush=True)
            
        for iteration in range(num_iterations):
            epoch_start_time = time.time()
            
            # Determine current min_hits for this iteration (if cycling through levels)
            if use_min_hits_cycling:
                level_idx = iteration // (num_iterations // num_levels)
                level_idx = min(level_idx, num_levels - 1)  # Clamp to valid range
                current_min_hits = min_hits_levels[level_idx]
            else:
                current_min_hits = min_hits
            
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
                
                while ly < current_min_hits:
                    coordinate = event_sampler.sample_detector_points(1)
                    patd_dict = light_yield_surrogate_func(
                        opt_point=coordinate, 
                        event_params=signal_event[0]
                    )
                    ly = patd_dict['num_photons']
                    counter += 1
                    if sampling_timeout is not None and counter >= sampling_timeout:
                        break
                        
                if ly < current_min_hits and sampling_timeout is not None and counter >= sampling_timeout:
                    continue
                
                # Get hit times and shift to start near zero
                hit_times = patd_dict['hit_times']
                if self.use_min_hit_time:
                    min_time = torch.min(hit_times)
                else:
                    min_time = 0.0
                
                # Subsample hits per event if max_hits_per_subsample is provided
                if max_hits_per_subsample is not None and len(hit_times) > max_hits_per_subsample:
                    indices = torch.randperm(len(hit_times), device=hit_times.device)[:max_hits_per_subsample]
                    hit_times = hit_times[indices]
                    # If patd_probs is in patd_dict, subsample that too
                    if 'patd_probs' in patd_dict and patd_dict['patd_probs'] is not None:
                        patd_dict['patd_probs'] = patd_dict['patd_probs'][indices]
                
                # Also apply max_hits_per_event if provided (additional limiting)
                if max_hits_per_event is not None and len(hit_times) > max_hits_per_event:
                    indices = torch.randperm(len(hit_times), device=hit_times.device)[:max_hits_per_event]
                    hit_times = hit_times[indices]
                    if 'patd_probs' in patd_dict and patd_dict['patd_probs'] is not None:
                        patd_dict['patd_probs'] = patd_dict['patd_probs'][indices]
                    
                
                hit_times_shifted = (hit_times - min_time + 1e-5).unsqueeze(1)
                
                # Calculate t_geom_min if not provided in patd_dict
                if 't_geom_min' not in patd_dict or patd_dict['t_geom_min'] is None:
                    # Try to compute using LightSabre formula
                    t_geom_min = self.calculate_t_geom_min(signal_event[0], coordinate.squeeze(0))
                    if t_geom_min is None:
                        # Fallback to minimum hit time if direction not available
                        t_geom_min = torch.min(hit_times)
                else:
                    t_geom_min = patd_dict['t_geom_min']
                
                # Create context features
                features_batch = self.create_event_features(
                    signal_event[0], 
                    coordinate.squeeze(0), 
                    len(hit_times_shifted),
                    t_geom_min=t_geom_min,
                    d_beam=patd_dict.get('d_geom', None),
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

            if shuffle_training_batches and len(all_hit_times_tensor) > 1:
                permutation = torch.randperm(len(all_hit_times_tensor), device=all_hit_times_tensor.device)
                all_hit_times_tensor = all_hit_times_tensor[permutation]
                all_features_tensor = all_features_tensor[permutation]
            
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

            if scheduler is not None:
                scheduler.step(loss)
                
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
        t_geom_min = kwargs.get('t_geom_min', None)
        d_beam = kwargs.get('d_beam', None)
        
        if event_params is None or opt_point is None or light_yield is None:
            raise ValueError("event_params, opt_point, and light_yield are required")
        
        # Convert light_yield to int
        num_photons = int(light_yield.item()) if isinstance(light_yield, torch.Tensor) else int(light_yield)
        
        # Convert min_hit_time to tensor
        if (not isinstance(min_hit_time, torch.Tensor)) and self.use_min_hit_time:
            min_hit_time = torch.tensor(min_hit_time, dtype=torch.float32, device=self.device)
        elif not self.use_min_hit_time:
            min_hit_time = 0.0
        # Ensure opt_point is on the correct device
        if isinstance(opt_point, torch.Tensor):
            opt_point = opt_point.to(self.device).squeeze()
        else:
            opt_point = torch.tensor(opt_point, dtype=torch.float32, device=self.device).squeeze()
        
        # Create context features (use first row as they're all identical)
        features = self.create_event_features(
            event_params, 
            opt_point, 
            num_photons,
            t_geom_min=t_geom_min,
            d_beam=d_beam,
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
    
    def evaluate_pdf(self, hit_times, event_params, opt_point, min_hit_time=None, t_geom_min=None, d_beam=None):
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
        t_geom_min : float or torch.Tensor or None
            Optional geometric-time minimum feature used during training.
        d_beam : float or torch.Tensor or None
            Optional beam-distance feature used during training.
            
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
        if not self.use_min_hit_time:
            min_hit_time = 0.0
        
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
            num_hits,
            t_geom_min=t_geom_min,
            d_beam=d_beam,
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
            'max_domain_size': self.max_domain_size,
            'scale_fac': self.scale_fac,
            'geom_time_scale': self.geom_time_scale,
            'beam_distance_scale': self.beam_distance_scale,
            'use_min_hit_time': self.use_min_hit_time,
            'use_geom_time_shift': self.use_geom_time_shift,
            'refractive_index': self.refractive_index,
            'v_mu': self.v_mu,
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
        self.scale_fac = checkpoint.get('scale_fac', 1.0)
        self.geom_time_scale = checkpoint.get('geom_time_scale', 1e5)
        self.beam_distance_scale = checkpoint.get('beam_distance_scale', 1250.0)
        self.use_min_hit_time = checkpoint.get('use_min_hit_time', True)
        self.use_geom_time_shift = checkpoint.get('use_geom_time_shift', True)
        self.refractive_index = checkpoint.get('refractive_index', 1.33)
        self.v_mu = checkpoint.get('v_mu', 0.299792458)
        
        # Load cylinder variation parameters if available
        # self.vary_cylinder = checkpoint.get('vary_cylinder', False)
        # self.min_domain_size = checkpoint.get('min_domain_size', 1000)
        # self.max_domain_size = checkpoint.get('max_domain_size', 5000)
        
        # Rebuild and load flow
        self._build_flow()
        self.flow.load_state_dict(checkpoint['flow_state_dict'])
        self.flow.eval()
