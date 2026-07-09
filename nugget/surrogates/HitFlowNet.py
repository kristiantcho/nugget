from nugget.surrogates.base_surrogate import Surrogate
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import warnings
# import h5py
# from nflows.distributions.normal import StandardNormal
# from nflows.transforms.nonlinearities import PiecewiseRationalQuadraticCDF
# from nflows.flows.base import Flow
# from nflows.transforms.base import CompositeTransform, Transform

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def sph_to_cart(theta, phi):
    """Converts spherical coordinates (zenith, azimuth) to a 3D Cartesian vector."""
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    return torch.stack([st * cp, st * sp, ct], dim=-1)


def cart_to_sph(vec):
    """Converts a 3D Cartesian vector to spherical coordinates (zenith, azimuth)."""
    vec = vec.squeeze()
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))  # Zenith
    phi = torch.atan2(y, x)  # Azimuth
    return theta, phi


class LogTransform(Transform):
    """Log10 transform for normalizing flows."""
    
    def forward(self, x, context=None):
        """Transform data to log space."""
        y = torch.log10(x)
        log_det = -torch.log(x).sum(dim=-1)
        return y, log_det

    def inverse(self, y, context=None):
        """Transform from log space back to data space."""
        x = 10**y
        log_det = y.sum(dim=-1) * np.log(10)
        return x, log_det


class FourierFeatures(torch.nn.Module):
    """
    Multiscale Fourier feature mapping for neural networks.
    
    This module applies a Fourier feature transformation to input coordinates
    using multiple frequency scales. The transformation maps input coordinates
    to a higher-dimensional space using sine and cosine functions.
    """
    
    def __init__(self, input_dim, num_frequencies=64, frequency_scale=1.0, learnable=False):
        """
        Initialize Fourier feature mapping.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input coordinates
        num_frequencies : int
            Number of frequency components (output will be 2 * num_frequencies)
        frequency_scale : float
            Scale factor for frequency sampling
        learnable : bool
            If True, frequencies are learnable parameters. If False, they are fixed.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.frequency_scale = frequency_scale
        self.output_dim = 2 * num_frequencies
        
        # Generate frequency matrix
        frequencies = torch.randn(num_frequencies, input_dim) * frequency_scale
        
        if learnable:
            self.frequencies = torch.nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)
    
    def forward(self, x):
        """
        Apply Fourier feature mapping.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input coordinates of shape (..., input_dim)
            
        Returns:
        --------
        torch.Tensor
            Fourier features of shape (..., 2 * num_frequencies)
        """
        projected = torch.matmul(x, self.frequencies.T) * 2 * torch.pi
        sin_features = torch.sin(projected)
        cos_features = torch.cos(projected)
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        return fourier_features


class ResidualBlock(torch.nn.Module):
    """A residual block with optional dimension matching."""
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main path
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Skip connection - add projection if dimensions don't match
        if input_dim != output_dim:
            self.skip_projection = torch.nn.Linear(input_dim, output_dim)
        else:
            self.skip_projection = None
    
    def forward(self, x):
        # Main path
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Skip connection
        if self.skip_projection is not None:
            skip = self.skip_projection(x)
        else:
            skip = x
        
        # Add residual connection
        return out + skip


class HitFlowNet(Surrogate):
    """
    HitFlowNet: Neural network for predicting normalizing flow parameters for photon arrival time distributions.
    
    This model learns to predict the parameters of normalizing flows that can generate photon arrival
    time distributions (PATD) for neutrino events. The training process involves:
    
    1. Sample events and detector positions with sufficient photon yield
    2. For each sample, train an individual normalizing flow on the PATD until convergence
    3. Extract the trained flow parameters as training targets
    4. Train HitFlowNet to predict these flow parameters from event features
    
    The normalizing flow architecture uses:
    - LogTransform to map times to log space
    - Multiple layers of PiecewiseRationalQuadraticCDF transforms
    
    Architecture:
    - Input: Event features (energy, position, direction) + detector position
    - Output: Flow parameters (all learnable parameters of the flow transforms)
    
    Example Usage:
    --------------
    
    # Create model
    model = HitFlowNet(
        dim=3,
        num_flow_layers=6,
        num_bins=4,
        tail_bound=6.0,
        hidden_dims=[256, 128, 64]
    )
    
    # Train with event sampler and PATD surrogate
    model.train_hitflownet(
        sampler=signal_sampler,
        patd_surrogate=light_yield_patd_surrogate,
        num_training_samples=1000,
        min_photons=1000,
        flow_training_iterations=1000
    )
    
    # Use trained model to generate PATD
    event_params = {...}
    opt_point = torch.tensor([10., 5., 3.])
    samples = model.sample_patd(opt_point=opt_point, event_params=event_params, num_samples=5000)
    """
    
    def __init__(self, device=None, dim=3, domain_size=2500, 
                 hidden_dims=[256, 128, 64], dropout_rate=0.1, learning_rate=1e-3,
                 use_fourier_features=True, num_frequencies=64, frequency_scale=1.0,
                 learnable_frequencies=False, num_parallel_branches=1, frequency_scales=None,
                 num_frequencies_per_branch=None, shared_mlp=False, use_residual_connections=False,
                 add_relative_pos=True, add_distance_from_beam=False,
                 norm_pos=True, log_scale_energy=True,
                 num_flow_layers=6, num_bins=4, tail_bound=6.0, tails='linear',
                 reduce_lr_on_plateau=True, lr_scheduler_patience=10,
                 lr_scheduler_factor=0.5, lr_scheduler_min_lr=1e-6):
        """
        Initialize the HitFlowNet model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (must be 3D)
        domain_size : float
            Size of the detector domain in meters
        hidden_dims : list
            List of hidden layer dimensions for the MLP
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for HitFlowNet training
        use_fourier_features : bool
            Whether to use Fourier feature mapping
        num_frequencies : int
            Number of frequency components for Fourier features
        frequency_scale : float
            Scale factor for frequency sampling
        learnable_frequencies : bool
            If True, Fourier frequencies are learnable
        num_parallel_branches : int
            Number of parallel Fourier+MLP branches to use
        frequency_scales : list or None
            List of frequency scales for each branch
        num_frequencies_per_branch : list or None
            List of number of frequencies for each branch
        shared_mlp : bool
            If True, uses a single shared MLP for all Fourier branches
        use_residual_connections : bool
            If True, uses residual connections in MLP
        add_relative_pos : bool
            If True, adds relative position as features
        add_distance_from_beam : bool
            If True, adds perpendicular distance from beam
        norm_pos : bool
            If True, normalizes position coordinates
        log_scale_energy : bool
            If True, uses log10(energy) as input
        num_flow_layers : int
            Number of PiecewiseRationalQuadraticCDF layers in the flow
        num_bins : int
            Number of bins for spline transforms
        tail_bound : float
            Tail bound for spline transforms
        tails : str
            Tail type ('linear' or 'circular')
        reduce_lr_on_plateau : bool
            If True, reduces learning rate on plateau
        lr_scheduler_patience : int
            Patience for learning rate scheduler
        lr_scheduler_factor : float
            Factor to reduce learning rate
        lr_scheduler_min_lr : float
            Minimum learning rate
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_fourier_features = use_fourier_features
        self.num_frequencies = num_frequencies
        self.frequency_scale = frequency_scale
        self.learnable_frequencies = learnable_frequencies
        self.num_parallel_branches = num_parallel_branches
        self.shared_mlp = shared_mlp
        self.use_residual_connections = use_residual_connections
        self.add_relative_pos = add_relative_pos
        self.add_distance_from_beam = add_distance_from_beam
        self.norm_pos = norm_pos
        self.log_scale_energy = log_scale_energy
        
        # Handle multiple branch configurations
        if num_parallel_branches > 1:
            if frequency_scales is None:
                # Default: geometric progression of scales
                self.frequency_scales = [frequency_scale * (2 ** i) for i in range(num_parallel_branches)]
            else:
                self.frequency_scales = frequency_scales
            
            if num_frequencies_per_branch is None:
                self.num_frequencies_per_branch = [num_frequencies] * num_parallel_branches
            else:
                self.num_frequencies_per_branch = num_frequencies_per_branch
        else:
            self.frequency_scales = [frequency_scale]
            self.num_frequencies_per_branch = [num_frequencies]
        
        # Flow architecture parameters
        self.num_flow_layers = num_flow_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.tails = tails
        
        # Learning rate scheduler
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        
        # Calculate number of flow parameters per layer
        # PiecewiseRationalQuadraticCDF has parameters: widths, heights, derivatives
        # For shape=[1], num_bins: (num_bins-1) widths, (num_bins-1) heights, (num_bins+1) derivatives
        self.params_per_layer = (num_bins - 1) + (num_bins - 1) + (num_bins + 1)
        self.total_flow_params = num_flow_layers * self.params_per_layer
        
        # Network components (will be built later)
        self.fourier_features_list = None
        self.mlp_branches = None
        self.shared_branch_mlp = None
        self.final_mlp = None
        self.optimizer = None
        self.lr_scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def _build_network(self, input_dim):
        """
        Build the parallel MLP network architecture with multiple Fourier feature mappings.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features (before Fourier features)
        """
        print(f"Building HitFlowNet architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Number of parallel branches: {self.num_parallel_branches}")
        print(f"  Shared MLP: {self.shared_mlp}")
        print(f"  Flow parameters to predict: {self.total_flow_params}")
        
        # Create Fourier features for each branch
        if self.use_fourier_features:
            self.fourier_features_list = torch.nn.ModuleList()
            fourier_output_dims = []
            
            for i, (scale, n_freq) in enumerate(zip(self.frequency_scales, self.num_frequencies_per_branch)):
                fourier = FourierFeatures(
                    input_dim=input_dim,
                    num_frequencies=n_freq,
                    frequency_scale=scale,
                    learnable=self.learnable_frequencies
                ).to(self.device)
                self.fourier_features_list.append(fourier)
                fourier_output_dims.append(fourier.output_dim)
                print(f"  Branch {i}: Fourier features {input_dim} -> {fourier.output_dim} (scale={scale}, freq={n_freq})")
        else:
            fourier_output_dims = [input_dim] * self.num_parallel_branches
        
        # Create MLP branches
        if self.shared_mlp:
            # Single shared MLP for all branches
            max_fourier_dim = max(fourier_output_dims)
            mlp_input_dim = max_fourier_dim
            
            layers = []
            prev_dim = mlp_input_dim
            
            for hidden_dim in self.hidden_dims:
                if self.use_residual_connections:
                    layers.append(ResidualBlock(prev_dim, hidden_dim, self.dropout_rate))
                else:
                    layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                    layers.append(torch.nn.SiLU())
                    layers.append(torch.nn.Dropout(self.dropout_rate))
                prev_dim = hidden_dim
            
            self.shared_branch_mlp = torch.nn.Sequential(*layers).to(self.device)
            branch_output_dims = [prev_dim] * self.num_parallel_branches
            
            print(f"  Shared branch MLP: {mlp_input_dim} -> {' -> '.join(map(str, self.hidden_dims))}")
        else:
            # Separate MLP for each branch
            self.mlp_branches = torch.nn.ModuleList()
            branch_output_dims = []
            
            for i, fourier_dim in enumerate(fourier_output_dims):
                layers = []
                prev_dim = fourier_dim
                
                for hidden_dim in self.hidden_dims:
                    if self.use_residual_connections:
                        layers.append(ResidualBlock(prev_dim, hidden_dim, self.dropout_rate))
                    else:
                        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                        layers.append(torch.nn.SiLU())
                        layers.append(torch.nn.Dropout(self.dropout_rate))
                    prev_dim = hidden_dim
                
                branch_mlp = torch.nn.Sequential(*layers).to(self.device)
                self.mlp_branches.append(branch_mlp)
                branch_output_dims.append(prev_dim)
                
                print(f"  Branch {i} MLP: {fourier_dim} -> {' -> '.join(map(str, self.hidden_dims))}")
        
        # Create final MLP that combines all branch outputs and predicts flow parameters
        total_branch_output_dim = sum(branch_output_dims)
        final_layers = []
        
        final_hidden_dim = min(128, total_branch_output_dim // 2)
        
        final_layers.append(torch.nn.Linear(total_branch_output_dim, final_hidden_dim))
        final_layers.append(torch.nn.SiLU())
        final_layers.append(torch.nn.Dropout(self.dropout_rate))
        
        # Output layer to predict flow parameters
        final_layers.append(torch.nn.Linear(final_hidden_dim, self.total_flow_params))
        
        self.final_mlp = torch.nn.Sequential(*final_layers).to(self.device)
        
        print(f"  Final MLP: {total_branch_output_dim} -> {final_hidden_dim} -> {self.total_flow_params}")
        
        # Initialize optimizer
        all_params = []
        
        if self.shared_mlp:
            all_params.extend(self.shared_branch_mlp.parameters())
        else:
            for branch in self.mlp_branches:
                all_params.extend(branch.parameters())
        
        all_params.extend(self.final_mlp.parameters())
        
        if self.fourier_features_list is not None and self.learnable_frequencies:
            for fourier in self.fourier_features_list:
                all_params.extend(fourier.parameters())
        
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
        # Initialize learning rate scheduler
        if self.reduce_lr_on_plateau:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=self.lr_scheduler_min_lr,
                verbose=True
            )
        
        print(f"  Total trainable parameters: {sum(p.numel() for p in all_params if p.requires_grad):,}")
    
    def _forward_pass(self, features):
        """
        Internal method for forward pass through the parallel network.
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features
            
        Returns:
        --------
        torch.Tensor
            Predicted flow parameters
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        else:
            features = features.to(self.device)
        
        # Process through each parallel branch
        branch_outputs = []
        
        if self.shared_mlp:
            max_fourier_dim = max([f.output_dim for f in self.fourier_features_list]) if self.fourier_features_list else features.shape[-1]
            
            for i in range(self.num_parallel_branches):
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](features)
                    if branch_input.shape[-1] < max_fourier_dim:
                        padding = torch.zeros(*branch_input.shape[:-1], max_fourier_dim - branch_input.shape[-1], device=self.device)
                        branch_input = torch.cat([branch_input, padding], dim=-1)
                else:
                    branch_input = features
                
                branch_output = self.shared_branch_mlp(branch_input)
                branch_outputs.append(branch_output)
        else:
            for i in range(self.num_parallel_branches):
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](features)
                else:
                    branch_input = features
                
                branch_output = self.mlp_branches[i](branch_input)
                branch_outputs.append(branch_output)
        
        # Concatenate all branch outputs
        concatenated_features = torch.cat(branch_outputs, dim=-1)
        
        # Final MLP to predict flow parameters
        flow_params = self.final_mlp(concatenated_features)
        
        return flow_params
    
    def compute_distance_from_beam(self, point, track_pos, track_dir):
        """
        Compute perpendicular distance from point to beam/track.
        
        Parameters:
        -----------
        point : torch.Tensor
            Detector position (3,)
        track_pos : torch.Tensor
            Track position (3,)
        track_dir : torch.Tensor
            Track direction (3,)
            
        Returns:
        --------
        torch.Tensor
            Perpendicular distance
        """
        # Normalize direction
        track_dir_norm = track_dir / torch.norm(track_dir)
        
        # Vector from track position to point
        vec = point - track_pos
        
        # Project onto track direction
        proj_length = torch.dot(vec, track_dir_norm)
        
        # Perpendicular component
        perp_vec = vec - proj_length * track_dir_norm
        
        # Distance
        distance = torch.norm(perp_vec)
        
        return distance
    
    def prepare_features(self, point, event_data, event_labels=['position', 'energy', 'zenith', 'azimuth']):
        """
        Prepare input features for the network from event data and detector position.
        
        Parameters:
        -----------
        point : torch.Tensor
            Detector position (3,)
        event_data : dict
            Dictionary containing event parameters
        event_labels : list
            List of keys to extract from event_data
            
        Returns:
        --------
        torch.Tensor
            Feature vector
        """
        features = []
        
        # Convert point to tensor
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point = point.to(self.device)
        point = point.squeeze()
        
        # Extract and process features based on event_labels
        extracted_data = {}
        for label in event_labels:
            value = event_data.get(label)
            
            if value is None:
                # Provide default values for missing data
                raise ValueError(f"Event data missing required label: {label}")
            
            # Convert to tensor if needed
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
            else:
                value = value.to(self.device)
            
            extracted_data[label] = value.squeeze()
        
        # Add energy if present (log scale if specified)
        if 'energy' in extracted_data:
            if self.log_scale_energy:
                features.append(torch.log10(extracted_data['energy']))
            else:
                features.append(extracted_data['energy'])
        
        # Add detector position (normalized if specified)
        if self.norm_pos:
            features.extend([
                point[0] / self.domain_size,
                point[1] / self.domain_size,
                point[2] / self.domain_size
            ])
        else:
            features.extend([point[0], point[1], point[2]])
        
        # Add track position if present
        if 'position' in extracted_data:
            track_pos = extracted_data['position']
            if self.norm_pos:
                features.extend([
                    track_pos[0] / self.domain_size,
                    track_pos[1] / self.domain_size,
                    track_pos[2] / self.domain_size
                ])
            else:
                features.extend([track_pos[0], track_pos[1], track_pos[2]])
        
        # Add direction vector (use provided direction or calculate from angles)
        if 'direction' in extracted_data:
            # Use provided direction directly
            track_dir = extracted_data['direction']
            features.extend([track_dir[0], track_dir[1], track_dir[2]])
        elif 'zenith' in extracted_data and 'azimuth' in extracted_data:
            # Convert angles to Cartesian direction
            track_dir = sph_to_cart(extracted_data['zenith'], extracted_data['azimuth'])
            features.extend([track_dir[0], track_dir[1], track_dir[2]])
        
        # Add relative position if specified and position is available
        if self.add_relative_pos and 'position' in extracted_data:
            track_pos = extracted_data['position']
            rel_pos = point - track_pos
            if self.norm_pos:
                rel_pos = rel_pos / self.domain_size
            features.extend([rel_pos[0], rel_pos[1], rel_pos[2]])
        
        # Add distance from beam if specified and all required data is available
        if self.add_distance_from_beam and 'position' in extracted_data:
            # Get direction from 'direction' label or calculate from zenith/azimuth
            if 'direction' in extracted_data:
                track_dir = extracted_data['direction']
            elif 'zenith' in extracted_data and 'azimuth' in extracted_data:
                track_dir = sph_to_cart(extracted_data['zenith'], extracted_data['azimuth'])
            else:
                track_dir = None
            
            if track_dir is not None:
                track_pos = extracted_data['position']
                distance = self.compute_distance_from_beam(point, track_pos, track_dir)
                if self.norm_pos:
                    distance = distance / self.domain_size
                features.append(distance)
        
        # Add any other scalar features directly (e.g., 'time' or custom features)
        for label in event_labels:
            if label not in ['position', 'energy', 'zenith', 'azimuth']:
                value = extracted_data.get(label)
                if value is not None:
                    # Check if it's a scalar
                    if value.dim() == 0:
                        features.append(value)
                    # If it's a vector, add all components
                    elif value.dim() == 1:
                        features.extend(list(value))
        
        # Stack into tensor
        feature_tensor = torch.stack([f if isinstance(f, torch.Tensor) else torch.tensor(f, device=self.device) for f in features])
        
        return feature_tensor
    
    def create_flow(self):
        """
        Create a normalizing flow with the specified architecture.
        
        Returns:
        --------
        Flow
            A normalizing flow model
        """
        base_dist = StandardNormal(shape=[1])
        transforms = [LogTransform()]
        
        for _ in range(self.num_flow_layers):
            transform = PiecewiseRationalQuadraticCDF(
                shape=[1],
                num_bins=self.num_bins,
                tails=self.tails,
                tail_bound=self.tail_bound
            )
            transforms.append(transform)
        
        composite_transform = CompositeTransform(transforms)
        flow = Flow(composite_transform, base_dist)
        
        return flow
    
    def extract_flow_parameters(self, flow):
        """
        Extract learnable parameters from a trained flow.
        
        Parameters:
        -----------
        flow : Flow
            Trained normalizing flow
            
        Returns:
        --------
        torch.Tensor
            Flattened parameter vector
        """
        params = []
        
        # Extract parameters from each PiecewiseRationalQuadraticCDF layer
        # Skip the LogTransform (index 0)
        for i in range(1, len(flow._transform._transforms)):
            transform = flow._transform._transforms[i]
            
            # Get all parameters from this transform
            for param in transform.parameters():
                params.append(param.detach().flatten())
        
        # Concatenate all parameters
        param_vector = torch.cat(params)
        
        return param_vector
    
    def set_flow_parameters(self, flow, param_vector):
        """
        Set flow parameters from a predicted parameter vector.
        
        Parameters:
        -----------
        flow : Flow
            Flow model to set parameters for
        param_vector : torch.Tensor
            Predicted parameter vector
        """
        idx = 0
        
        # Set parameters for each PiecewiseRationalQuadraticCDF layer
        # Skip the LogTransform (index 0)
        for i in range(1, len(flow._transform._transforms)):
            transform = flow._transform._transforms[i]
            
            # Set parameters
            for param in transform.parameters():
                num_params = param.numel()
                param.data = param_vector[idx:idx+num_params].view(param.shape)
                idx += num_params
    
    def train_single_flow(self, hit_times, max_iterations=1000, lr=1e-3, 
                         convergence_threshold=1e-4, patience=50, verbose=False):
        """
        Train a single normalizing flow on photon arrival times.
        
        Parameters:
        -----------
        hit_times : torch.Tensor
            Photon arrival times (N, 1)
        max_iterations : int
            Maximum number of training iterations
        lr : float
            Learning rate for flow training
        convergence_threshold : float
            Threshold for loss improvement to determine convergence
        patience : int
            Number of iterations without improvement before stopping
        verbose : bool
            If True, print training progress
            
        Returns:
        --------
        tuple
            (trained_flow, final_loss, training_history)
        """
        flow = self.create_flow().to(self.device)
        flow.train()  # Set to training mode
        
        # Ensure all parameters require gradients
        for param in flow.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        
        training_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute negative log likelihood
            log_probs = flow.log_prob(hit_times)
            loss = -log_probs.mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            training_history.append(current_loss)
            
            # Check for convergence
            if current_loss < best_loss - convergence_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Flow converged at iteration {iteration+1} with loss {current_loss:.4f}")
                break
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Flow iteration {iteration+1}/{max_iterations}, Loss: {current_loss:.4f}")
        
        return flow, best_loss, training_history
    
    def create_and_save_flow_dataset(self, sampler, patd_surrogate, num_samples, save_path,
                                     min_photons=1000, flow_training_iterations=1000, event_labels=['position', 'energy', 'zenith', 'azimuth'],
                                     flow_lr=1e-3, flow_convergence_threshold=1e-4,
                                     flow_patience=50, save_every=10, resume=True, verbose=True, verbose_flow_training=False):
        """
        Create and save a dataset of flow parameters and features to disk as HDF5.
        
        This function samples events, trains flows, and saves the resulting
        (features, flow_params) pairs progressively. Data is saved every N samples
        to prevent data loss if the process crashes.
        
        Parameters:
        -----------
        sampler : Sampler
            Sampler for generating event parameters
        patd_surrogate : callable
            Function to generate PATD
        num_samples : int
            Number of samples to generate
        save_path : str
            Path to save the dataset (will create .h5 or .hdf5 file)
        min_photons : int
            Minimum number of photons required for a sample
        flow_training_iterations : int
            Maximum iterations for training each individual flow
        flow_lr : float
            Learning rate for flow training
        flow_convergence_threshold : float
            Convergence threshold for flow training
        flow_patience : int
            Patience for early stopping in flow training
        save_every : int
            Save to disk every N samples
        resume : bool
            If True and file exists, resume from where it left off
        verbose : bool
            If True, print progress
        verbose_flow_training : bool
            If True, print progress during flow training
        """
        # Ensure save_path has correct extension
        if not (save_path.endswith('.h5') or save_path.endswith('.hdf5')):
            save_path += '.h5'
        
        # Check if we should resume from existing file
        start_idx = 0
        current_batch_features = []
        current_batch_flow_params = []
        
        if resume and os.path.exists(save_path):
            if verbose:
                print(f"Resuming from existing dataset: {save_path}")
            
            with h5py.File(save_path, 'r') as f:
                start_idx = f.attrs['num_samples']
            
            if start_idx >= num_samples:
                if verbose:
                    print(f"Dataset already complete with {start_idx} samples.")
                # Load and return final data
                with h5py.File(save_path, 'r') as f:
                    all_features = torch.tensor(f['features'][:])
                    all_flow_params = torch.tensor(f['flow_params'][:])
                return all_features, all_flow_params
            
            if verbose:
                print(f"  Loaded {start_idx} existing samples")
                print(f"  Generating {num_samples - start_idx} more samples...")
        else:
            if verbose:
                print(f"Creating flow parameter dataset with {num_samples} samples...")
                print(f"Saving every {save_every} samples to {save_path}")
        
        for i in range(start_idx, num_samples):
            # Keep sampling until we get enough photons
            while True:
                coordinate = sampler.sample_detector_points(1)
                signal_event = sampler.sample_events(1)
                
                patd_dict = patd_surrogate(opt_point=coordinate, event_params=signal_event[0])
                ly = patd_dict['num_photons']
                
                if ly >= min_photons:
                    break
            
            # Train flow on this PATD
            hit_times = patd_dict['hit_times'].view(-1, 1).to(self.device)
            
            flow, final_loss, history = self.train_single_flow(
                hit_times,
                max_iterations=flow_training_iterations,
                lr=flow_lr,
                convergence_threshold=flow_convergence_threshold,
                patience=flow_patience,
                verbose=verbose_flow_training
            )
            
            # Extract features and flow parameters
            features = self.prepare_features(coordinate.squeeze(), signal_event[0], event_labels=event_labels)
            flow_params = self.extract_flow_parameters(flow)
            
            current_batch_features.append(features.cpu())
            current_batch_flow_params.append(flow_params.cpu())
            
            # Save periodically and clear memory
            if (i + 1) % save_every == 0 or (i + 1) == num_samples:
                batch_features = torch.stack(current_batch_features)
                batch_flow_params = torch.stack(current_batch_flow_params)
                
                # Append to HDF5 file
                if os.path.exists(save_path):
                    # Append to existing file
                    with h5py.File(save_path, 'a') as f:
                        # Get current size
                        current_size = f['features'].shape[0]
                        new_size = current_size + len(current_batch_features)
                        
                        # Resize datasets
                        f['features'].resize((new_size, batch_features.shape[1]))
                        f['flow_params'].resize((new_size, batch_flow_params.shape[1]))
                        
                        # Write new data
                        f['features'][current_size:new_size] = batch_features.numpy()
                        f['flow_params'][current_size:new_size] = batch_flow_params.numpy()
                        
                        # Update attributes
                        f.attrs['num_samples'] = new_size
                else:
                    # Create new file
                    with h5py.File(save_path, 'w') as f:
                        f.create_dataset('features', data=batch_features.numpy(), 
                                       maxshape=(None, batch_features.shape[1]), chunks=True)
                        f.create_dataset('flow_params', data=batch_flow_params.numpy(),
                                       maxshape=(None, batch_flow_params.shape[1]), chunks=True)
                        f.attrs['num_samples'] = len(current_batch_features)
                        f.attrs['min_photons'] = min_photons
                        f.attrs['num_flow_layers'] = self.num_flow_layers
                        f.attrs['num_bins'] = self.num_bins
                        f.attrs['tail_bound'] = self.tail_bound
                        f.attrs['tails'] = self.tails
                
                if verbose:
                    print(f"  Generated {i+1}/{num_samples} samples (last flow loss: {final_loss:.4f}) - Saved to disk", flush=True)
                
                # Clear batch lists to free memory
                current_batch_features = []
                current_batch_flow_params = []
                
            # elif verbose and (i + 1) % 50 == 0:
            #     print(f"  Generated {i+1}/{num_samples} samples (last flow loss: {final_loss:.4f})")
        
        # Load final dataset to return
        with h5py.File(save_path, 'r') as f:
            all_features = torch.tensor(f['features'][:])
            all_flow_params = torch.tensor(f['flow_params'][:])
        
        if verbose:
            print(f"Dataset complete and saved to {save_path}", flush=True)
            print(f"  Features shape: {all_features.shape}", flush=True)
            print(f"  Flow params shape: {all_flow_params.shape}", flush=True)
    
    class PrecomputedFlowDataset(Dataset):
        """
        Dataset that loads precomputed flow parameters from disk.
        
        This is much faster than HitFlowDataset since it doesn't train flows on-the-fly.
        Use create_and_save_flow_dataset() to generate the data files first.
        """
        
        def __init__(self, dataset_path):
            """
            Initialize dataset from saved file.
            
            Parameters:
            -----------
            dataset_path : str
                Path to saved dataset (.h5, .hdf5, or .pt file)
            """
            # Support both HDF5 and legacy .pt format
            if dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
                with h5py.File(dataset_path, 'r') as f:
                    self.features = torch.tensor(f['features'][:])
                    self.flow_params = torch.tensor(f['flow_params'][:])
                    self.num_samples = f.attrs['num_samples']
            else:
                # Legacy .pt format
                data = torch.load(dataset_path)
                self.features = data['features']
                self.flow_params = data['flow_params']
                self.num_samples = data['num_samples']
            
            print(f"Loaded precomputed dataset from {dataset_path}")
            print(f"  Samples: {self.num_samples}")
            print(f"  Features shape: {self.features.shape}")
            print(f"  Flow params shape: {self.flow_params.shape}")
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.features[idx], self.flow_params[idx]
    
    class HitFlowDataset(Dataset):
        """
        Dataset for training HitFlowNet to predict flow parameters.
        
        Generates training samples by:
        1. Sampling detector points and event parameters
        2. Training a normalizing flow on the PATD
        3. Extracting flow parameters as training targets
        4. Optionally resampling if photon count is below threshold
        """
        
        def __init__(self, sampler, patd_surrogate, num_samples, hitflownet_model,
                     min_photons=1000, flow_training_iterations=1000, flow_lr=1e-3,
                     flow_convergence_threshold=1e-4, flow_patience=50,
                     event_labels=['position', 'energy', 'zenith', 'azimuth']):
            """
            Initialize dataset.
            
            Parameters:
            -----------
            sampler : Sampler
                Sampler for generating event parameters
            patd_surrogate : callable
                Function to generate PATD
            num_samples : int
                Number of samples in epoch
            hitflownet_model : HitFlowNet
                HitFlowNet model instance (for prepare_features method)
            min_photons : int
                Minimum number of photons required for a sample
            flow_training_iterations : int
                Maximum iterations for training each individual flow
            flow_lr : float
                Learning rate for flow training
            flow_convergence_threshold : float
                Convergence threshold for flow training
            flow_patience : int
                Patience for early stopping in flow training
            event_labels : list
                List of event parameter keys
            """
            self.sampler = sampler
            self.patd_surrogate = patd_surrogate
            self.num_samples = num_samples
            self.hitflownet_model = hitflownet_model
            self.min_photons = min_photons
            # self.max_resample_attempts = max_resample_attempts
            self.flow_training_iterations = flow_training_iterations
            self.flow_lr = flow_lr
            self.flow_convergence_threshold = flow_convergence_threshold
            self.flow_patience = flow_patience
            self.event_labels = event_labels
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Sample with resampling if needed
            while True:
                # Sample detector point and event parameters
                coordinate = self.sampler.sample_detector_points(1)
                signal_event = self.sampler.sample_events(1)
                
                # Get PATD
                patd_dict = self.patd_surrogate(opt_point=coordinate, event_params=signal_event[0])
                ly = patd_dict['num_photons']
                
                # Check if photon count meets minimum threshold
                if ly >= self.min_photons:
                    # Train flow on this PATD
                    hit_times = patd_dict['hit_times'].view(-1, 1).to(self.hitflownet_model.device)
                    
                    flow, final_loss, history = self.hitflownet_model.train_single_flow(
                        hit_times,
                        max_iterations=self.flow_training_iterations,
                        lr=self.flow_lr,
                        convergence_threshold=self.flow_convergence_threshold,
                        patience=self.flow_patience,
                        verbose=False
                    )
                    
                    # Extract features and flow parameters
                    features = self.hitflownet_model.prepare_features(coordinate.squeeze(), signal_event[0])
                    flow_params = self.hitflownet_model.extract_flow_parameters(flow)
                    
                    return features, flow_params
            
            # # If we exhausted all attempts, train on the last sample anyway
            # hit_times = patd_dict['hit_times'].view(-1, 1).to(self.hitflownet_model.device)
            
            # flow, final_loss, history = self.hitflownet_model.train_single_flow(
            #     hit_times,
            #     max_iterations=self.flow_training_iterations,
            #     lr=self.flow_lr,
            #     convergence_threshold=self.flow_convergence_threshold,
            #     patience=self.flow_patience,
            #     verbose=False
            # )
            
            # features = self.hitflownet_model.prepare_features(coordinate.squeeze(), signal_event[0])
            # flow_params = self.hitflownet_model.extract_flow_parameters(flow)
            
            # return features, flow_params
    
    def create_hitflow_dataloader(self, sampler, patd_surrogate, num_samples_per_epoch=1000,
                                  batch_size=32, shuffle=True, num_workers=0,
                                  min_photons=1000, flow_training_iterations=1000, flow_lr=1e-3,
                                  flow_convergence_threshold=1e-4, flow_patience=50,
                                  event_labels=['position', 'energy', 'zenith', 'azimuth']):
        """
        Create a DataLoader for training HitFlowNet.
        
        Parameters:
        -----------
        sampler : Sampler
            Sampler for generating event parameters
        patd_surrogate : callable
            Function to generate PATD
        num_samples_per_epoch : int
            Number of samples per epoch
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle data
        num_workers : int
            Number of worker processes
        min_photons : int
            Minimum number of photons required for a sample
        flow_training_iterations : int
            Maximum iterations for training each individual flow
        flow_lr : float
            Learning rate for flow training
        flow_convergence_threshold : float
            Convergence threshold for flow training
        flow_patience : int
            Patience for early stopping in flow training
        event_labels : list
            List of event parameter keys
            
        Returns:
        --------
        DataLoader
        """
        dataset = self.HitFlowDataset(
            sampler=sampler,
            patd_surrogate=patd_surrogate,
            num_samples=num_samples_per_epoch,
            hitflownet_model=self,
            min_photons=min_photons,
            flow_training_iterations=flow_training_iterations,
            flow_lr=flow_lr,
            flow_convergence_threshold=flow_convergence_threshold,
            flow_patience=flow_patience,
            event_labels=event_labels
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def create_precomputed_dataloader(self, dataset_path, batch_size=32, shuffle=True, num_workers=0):
        """
        Create a DataLoader from precomputed flow parameter dataset.
        
        Parameters:
        -----------
        dataset_path : str
            Path to saved dataset (.h5, .hdf5, or .pt file)
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle data
        num_workers : int
            Number of worker processes
            
        Returns:
        --------
        DataLoader
        """
        dataset = self.PrecomputedFlowDataset(dataset_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                              verbose=True, early_stopping_patience=15, input_dim=None):
        """
        Train HitFlowNet using dataloaders.
        
        Parameters:
        -----------
        train_dataloader : DataLoader
            Training data loader
        val_dataloader : DataLoader or None
            Validation data loader
        epochs : int
            Number of epochs to train
        verbose : bool
            If True, print training progress
        early_stopping_patience : int
            Patience for early stopping
        input_dim : int or None
            Input feature dimension (auto-detected from first batch if None)
            
        Returns:
        --------
        dict
            Training history
        """
        # Build network if not already built
        if self.final_mlp is None:
            # Get input dimension from first batch
            if input_dim is None:
                for batch_features, _ in train_dataloader:
                    input_dim = batch_features.shape[1]
                    break
            self._build_network(input_dim=input_dim)
        
        if verbose:
            print(f"Starting HitFlowNet training for {epochs} epochs...")
            print(f"Training samples per epoch: {len(train_dataloader.dataset)}")
            if val_dataloader is not None:
                print(f"Validation samples per epoch: {len(val_dataloader.dataset)}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            if self.shared_mlp:
                self.shared_branch_mlp.train()
            if self.mlp_branches:
                for branch in self.mlp_branches:
                    branch.train()
            self.final_mlp.train()
            if self.fourier_features_list is not None:
                for fourier in self.fourier_features_list:
                    fourier.train()
            
            train_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_targets in train_dataloader:
                self.optimizer.zero_grad()
                
                # Move to device
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass through parallel branches
                predictions = self._forward_pass(batch_features)
                
                # MSE loss between predicted and actual flow parameters
                loss = torch.nn.functional.mse_loss(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            self.train_losses.append(train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                if self.shared_mlp:
                    self.shared_branch_mlp.eval()
                if self.mlp_branches:
                    for branch in self.mlp_branches:
                        branch.eval()
                self.final_mlp.eval()
                if self.fourier_features_list is not None:
                    for fourier in self.fourier_features_list:
                        fourier.eval()
                
                val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_dataloader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        predictions = self._forward_pass(batch_features)
                        loss = torch.nn.functional.mse_loss(predictions, batch_targets)
                        val_loss += loss.item()
                        num_val_batches += 1
                
                val_loss /= num_val_batches
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {
                        'shared_branch_mlp': self.shared_branch_mlp.state_dict() if self.shared_mlp else None,
                        'mlp_branches': [branch.state_dict() for branch in self.mlp_branches] if self.mlp_branches else None,
                        'final_mlp': self.final_mlp.state_dict(),
                        'fourier_features_list': [f.state_dict() for f in self.fourier_features_list] if self.fourier_features_list else None,
                    }
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                # No validation, just print training loss
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            if self.best_model_state['shared_branch_mlp']:
                self.shared_branch_mlp.load_state_dict(self.best_model_state['shared_branch_mlp'])
            if self.best_model_state['mlp_branches']:
                for i, branch_state in enumerate(self.best_model_state['mlp_branches']):
                    self.mlp_branches[i].load_state_dict(branch_state)
            self.final_mlp.load_state_dict(self.best_model_state['final_mlp'])
            if self.best_model_state['fourier_features_list']:
                for i, fourier_state in enumerate(self.best_model_state['fourier_features_list']):
                    self.fourier_features_list[i].load_state_dict(fourier_state)
        
        if verbose:
            if val_dataloader is not None:
                print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
            else:
                print(f"\nTraining completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss if val_dataloader is not None else None
        }
    
    def predict_flow_parameters(self, point, event_data):
        """
        Predict flow parameters for given event and detector position.
        
        Parameters:
        -----------
        point : torch.Tensor
            Detector position
        event_data : dict
            Event parameters
            
        Returns:
        --------
        torch.Tensor
            Predicted flow parameters
        """
        # Set to eval mode
        if self.shared_mlp:
            self.shared_branch_mlp.eval()
        if self.mlp_branches:
            for branch in self.mlp_branches:
                branch.eval()
        self.final_mlp.eval()
        if self.fourier_features_list is not None:
            for fourier in self.fourier_features_list:
                fourier.eval()
        
        with torch.no_grad():
            features = self.prepare_features(point, event_data)
            features = features.unsqueeze(0)  # Add batch dimension
            predictions = self._forward_pass(features)
        
        return predictions.squeeze(0)
    
    def sample_patd(self, opt_point, event_params, num_samples=1000):
        """
        Generate photon arrival time distribution samples using predicted flow parameters.
        
        Parameters:
        -----------
        opt_point : torch.Tensor
            Detector position
        event_params : dict
            Event parameters
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        torch.Tensor
            Sampled photon arrival times
        """
        # Predict flow parameters
        param_vector = self.predict_flow_parameters(opt_point, event_params)
        
        # Create flow and set predicted parameters
        flow = self.create_flow().to(self.device)
        self.set_flow_parameters(flow, param_vector)
        
        # Generate samples
        flow.eval()
        with torch.no_grad():
            samples = flow.sample(num_samples)
        
        return samples.squeeze()
    
    def __call__(self, opt_point, event_params, num_samples=1000):
        """
        Generate PATD samples (alias for sample_patd).
        
        Parameters:
        -----------
        opt_point : torch.Tensor
            Detector position
        event_params : dict
            Event parameters
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        dict
            Dictionary containing 'hit_times' and 'num_photons'
        """
        samples = self.sample_patd(opt_point, event_params, num_samples)
        
        return {
            'hit_times': samples,
            'num_photons': len(samples)
        }
    
    def light_yield_surrogate(self, **kwargs):
        """
        Surrogate interface for generating PATD.
        
        Parameters:
        -----------
        opt_point : torch.Tensor
            Detector position
        event_params : dict
            Event parameters
        num_samples : int (optional)
            Number of samples to generate (default: 1000)
            
        Returns:
        --------
        dict
            Dictionary containing 'hit_times' and 'num_photons'
        """
        opt_point = kwargs.get('opt_point')
        event_params = kwargs.get('event_params')
        num_samples = kwargs.get('num_samples', 1000)
        
        return self.__call__(opt_point, event_params, num_samples)
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        save_dict = {
            'model_config': {
                'dim': self.dim,
                'domain_size': self.domain_size,
                'hidden_dims': self.hidden_dims,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'use_fourier_features': self.use_fourier_features,
                'num_frequencies': self.num_frequencies,
                'frequency_scale': self.frequency_scale,
                'learnable_frequencies': self.learnable_frequencies,
                'num_parallel_branches': self.num_parallel_branches,
                'frequency_scales': self.frequency_scales,
                'num_frequencies_per_branch': self.num_frequencies_per_branch,
                'shared_mlp': self.shared_mlp,
                'use_residual_connections': self.use_residual_connections,
                'add_relative_pos': self.add_relative_pos,
                'add_distance_from_beam': self.add_distance_from_beam,
                'norm_pos': self.norm_pos,
                'log_scale_energy': self.log_scale_energy,
                'num_flow_layers': self.num_flow_layers,
                'num_bins': self.num_bins,
                'tail_bound': self.tail_bound,
                'tails': self.tails,
            },
            'shared_branch_mlp_state_dict': self.shared_branch_mlp.state_dict() if self.shared_mlp else None,
            'mlp_branches_state_dicts': [branch.state_dict() for branch in self.mlp_branches] if self.mlp_branches else None,
            'final_mlp_state_dict': self.final_mlp.state_dict() if self.final_mlp else None,
            'fourier_features_state_dicts': [f.state_dict() for f in self.fourier_features_list] if self.fourier_features_list else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore configuration
        config = checkpoint['model_config']
        self.dim = config['dim']
        self.domain_size = config['domain_size']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.use_fourier_features = config['use_fourier_features']
        self.num_frequencies = config['num_frequencies']
        self.frequency_scale = config['frequency_scale']
        self.learnable_frequencies = config['learnable_frequencies']
        self.num_parallel_branches = config.get('num_parallel_branches', 1)
        self.frequency_scales = config.get('frequency_scales', [config['frequency_scale']])
        self.num_frequencies_per_branch = config.get('num_frequencies_per_branch', [config['num_frequencies']])
        self.shared_mlp = config.get('shared_mlp', False)
        self.use_residual_connections = config['use_residual_connections']
        self.add_relative_pos = config['add_relative_pos']
        self.add_distance_from_beam = config['add_distance_from_beam']
        self.norm_pos = config['norm_pos']
        self.log_scale_energy = config['log_scale_energy']
        self.num_flow_layers = config['num_flow_layers']
        self.num_bins = config['num_bins']
        self.tail_bound = config['tail_bound']
        self.tails = config['tails']
        
        # Recalculate derived parameters
        self.params_per_layer = (self.num_bins - 1) + (self.num_bins - 1) + (self.num_bins + 1)
        self.total_flow_params = self.num_flow_layers * self.params_per_layer
        
        # Load network states
        if checkpoint['final_mlp_state_dict'] is not None:
            # Need to rebuild network first
            # Get input dimension from Fourier features or first layer
            if checkpoint['fourier_features_state_dicts'] and checkpoint['fourier_features_state_dicts'][0]:
                input_dim = checkpoint['fourier_features_state_dicts'][0]['frequencies'].shape[1]
            else:
                # Try to infer from MLP branch or shared branch MLP
                if checkpoint['mlp_branches_state_dicts']:
                    first_layer_key = list(checkpoint['mlp_branches_state_dicts'][0].keys())[0]
                    if 'weight' in first_layer_key:
                        input_dim = checkpoint['mlp_branches_state_dicts'][0][first_layer_key].shape[1]
                elif checkpoint['shared_branch_mlp_state_dict']:
                    first_layer_key = list(checkpoint['shared_branch_mlp_state_dict'].keys())[0]
                    if 'weight' in first_layer_key:
                        input_dim = checkpoint['shared_branch_mlp_state_dict'][first_layer_key].shape[1]
            
            self._build_network(input_dim)
            
            # Load states
            if checkpoint['shared_branch_mlp_state_dict']:
                self.shared_branch_mlp.load_state_dict(checkpoint['shared_branch_mlp_state_dict'])
            
            if checkpoint['mlp_branches_state_dicts']:
                for i, branch_state in enumerate(checkpoint['mlp_branches_state_dicts']):
                    self.mlp_branches[i].load_state_dict(branch_state)
            
            self.final_mlp.load_state_dict(checkpoint['final_mlp_state_dict'])
            
            if checkpoint['fourier_features_state_dicts']:
                for i, fourier_state in enumerate(checkpoint['fourier_features_state_dicts']):
                    self.fourier_features_list[i].load_state_dict(fourier_state)
            
            if checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Model loaded from {filepath}")


class HitFlowNetSurrogate(Surrogate):
    """
    HitFlowNet-based surrogate for PATD generation.
    
    This class wraps a trained HitFlowNet model and provides a light_yield_surrogate
    interface compatible with other surrogate models like LightSabrePATD.
    
    Example Usage:
    --------------
    # Train a HitFlowNet model
    hitflownet = HitFlowNet(dim=3, num_flow_layers=6)
    hitflownet.train_hitflownet(sampler, patd_surrogate, num_training_samples=1000)
    
    # Create surrogate wrapper
    surrogate = HitFlowNetSurrogate(hitflownet_model=hitflownet)
    
    # Use like LightSabrePATD
    event_params = {...}
    opt_point = torch.tensor([10., 5., 3.])
    
    patd_dict = surrogate.light_yield_surrogate(
        opt_point=opt_point,
        event_params=event_params,
        num_samples=5000
    )
    """
    
    def __init__(self, hitflownet_model, device=None, dim=3, domain_size=2500):
        """
        Initialize HitFlowNet surrogate wrapper.
        
        Parameters:
        -----------
        hitflownet_model : HitFlowNet
            Trained HitFlowNet model
        device : torch.device
            Device to run on
        dim : int
            Dimension (must be 3)
        domain_size : float
            Domain size in meters
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.hitflownet = hitflownet_model
    
    def light_yield_surrogate(self, **kwargs):
        """
        Generate PATD using HitFlowNet.
        
        Parameters:
        -----------
        opt_point : torch.Tensor
            Detector position
        event_params : dict
            Event parameters
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        dict
            Dictionary with 'hit_times' and 'num_photons'
        """
        return self.hitflownet.light_yield_surrogate(**kwargs)
    
    def __call__(self, **kwargs):
        """
        Generate PATD using HitFlowNet (alias for light_yield_surrogate).
        
        Parameters:
        -----------
        opt_point : torch.Tensor
            Detector position
        event_params : dict
            Event parameters
        num_samples : int
            Number of samples to generate
            
        Returns:
        --------
        dict
            Dictionary with 'hit_times' and 'num_photons'
        """
        return self.light_yield_surrogate(**kwargs)
