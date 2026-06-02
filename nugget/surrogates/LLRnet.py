import numpy as np
import torch
import time
from nugget.surrogates.base_surrogate import Surrogate
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
# import matplotlib.pyplot as plt
import h5py
import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

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
        # Each row corresponds to one frequency component for all input dimensions
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
        # x shape: (..., input_dim)
        # frequencies shape: (num_frequencies, input_dim)
        
        # Compute 2π * frequencies * x for all frequency components
        # This does a dot product between each input vector and each frequency vector
        # Shape: (..., num_frequencies)
        projected = torch.matmul(x, self.frequencies.T) * 2 * torch.pi
        
        # Apply sine and cosine to get Fourier features
        # Shape: (..., num_frequencies)
        sin_features = torch.sin(projected)
        cos_features = torch.cos(projected)
        
        # Concatenate sine and cosine features
        # Shape: (..., 2 * num_frequencies)
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


class LLRnet(Surrogate):
    """
    Log-Likelihood Ratio network for training an MLP classifier to estimate LLR.
    
    This network is trained as a binary classifier but uses the sigmoid trick to compute
    Log-Likelihood Ratios. The network outputs probabilities through a sigmoid activation,
    and the LLR is computed as log(p/(1-p)) where p is the output probability.
    
    The network supports parallel Fourier mapping layers with corresponding MLPs that
    process different frequency scales simultaneously. This allows the network to capture
    patterns at multiple scales and combine them for improved performance.
    
    Architecture:
    - Multiple parallel branches, each with:
      * Optional Fourier feature mapping at different frequency scales
      * Either separate MLPs per branch OR a single shared MLP (when shared_mlp=True)
    - Final MLP that concatenates all branch outputs and applies sigmoid
    
    When shared_mlp=True, each Fourier branch output is separately fed to the same
    shared MLP and the outputs are concatenated before the final layer. If branches
    have different Fourier output dimensions, smaller inputs are zero-padded to match
    the maximum dimension.
    
    Example Usage:
    --------------
    
    # Single branch (traditional architecture)
    model = LLRnet(dim=3, num_parallel_branches=1, frequency_scale=1.0)
    
    # Multiple branches with different frequency scales and shared MLP
    model = LLRnet(
        dim=3, 
        num_parallel_branches=3,
        frequency_scales=[0.5, 2.0, 8.0],
        num_frequencies_per_branch=[32, 64, 32],
        shared_mlp=True  # Use single MLP for all branches
    )
    
    # Train with event data
    history = model.train_with_event_data(
        signal_event_params=signal_params,
        background_event_params=background_params,
        signal_surrogate_func=signal_func,
        background_surrogate_func=background_func,
        epochs=100
    )
    
    Can work with provided background/signal functions or raw neutrino event data.
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, hidden_dims=[128, 64, 32], 
                 dropout_rate=0.1, learning_rate=1e-3, use_fourier_features=True,
                 num_frequencies=64, frequency_scale=1.0, learnable_frequencies=False,
                 num_parallel_branches=1, frequency_scales=None, num_frequencies_per_branch=None, log_scale_ly=False, norm_pos=False,
                 shared_mlp=False, use_residual_connections=False, signal_noise_scale=0.0, background_noise_scale=0.0, add_relative_pos=True,
                 add_distance_from_beam=False, log_scale_energy=False, reduce_lr_on_plateau=False, lr_scheduler_patience=10, input_delta_time=False,
                 lr_scheduler_factor=0.5, lr_scheduler_min_lr=1e-6, use_patd=False, min_photons=1, num_photons_per_sample=None, rel_time=False, input_charge=False):
        """
        Initialize the LLRnet surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (2D or 3D)
        domain_size : int
            Length of the domain 
        hidden_dims : list
            List of hidden layer dimensions for the MLP
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for training
        use_fourier_features : bool
            Whether to use Fourier feature mapping at the input
        num_frequencies : int
            Number of frequency components for Fourier features (used if single branch)
        frequency_scale : float
            Scale factor for frequency sampling in Fourier features (used if single branch)
        learnable_frequencies : bool
            If True, Fourier frequencies are learnable parameters
        num_parallel_branches : int
            Number of parallel Fourier+MLP branches to use
        frequency_scales : list or None
            List of frequency scales for each branch. If None, uses geometric progression
        num_frequencies_per_branch : list or None
            List of number of frequencies for each branch. If None, uses num_frequencies for all
        shared_mlp : bool
            If True, uses a single shared MLP for all Fourier branches instead of separate MLPs
            This reduces parameters while still allowing multiple frequency scales
        use_residual_connections : bool
            If True, uses residual connections in the MLP layers for better gradient flow
        signal_noise_scale : float
            Scale for adding noise to signal detector response
        background_noise_scale : float
            Scale for adding noise to background detector response
        add_relative_pos : bool
            If True, adds relative position (detector_point - event_position) as features
        add_distance_from_beam : bool
            If True, adds perpendicular distance from detector point to beam/track as a feature.
            Requires 'position', 'zenith', and 'azimuth' in event data.
        log_scale_ly : bool
            If True, applies log10 scaling to light yield values
        norm_pos : bool
            If True, normalizes position coordinates by domain_size/2
        reduce_lr_on_plateau : bool
            If True, reduces learning rate when validation loss plateaus
        lr_scheduler_patience : int
            Number of epochs with no improvement after which learning rate will be reduced
        lr_scheduler_factor : float
            Factor by which the learning rate will be reduced (new_lr = lr * factor)
        lr_scheduler_min_lr : float
            Minimum learning rate allowed
        use_patd : bool
            If True, uses Photon Arrival Time Distributions instead of light yield
        min_photons : int
            Minimum number of photons required for an event to be valid in PATD mode (default: 1)
        num_photons_per_sample : int or None
            Number of photons to sample from each valid event in PATD mode.
            If None, defaults to min_photons. Can be set higher than min_photons to
            sample more photons from each event, or lower (but >= 1) to sample fewer.
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
        self.signal_noise_scale = signal_noise_scale
        self.background_noise_scale = background_noise_scale
        self.add_relative_pos = add_relative_pos
        self.add_distance_from_beam = add_distance_from_beam
        self.log_scale_ly = log_scale_ly
        self.rel_time = rel_time
        self.input_charge = input_charge  # If using PATD, we will add number of photons as an input feature
        self.norm_pos = norm_pos
        self.log_scale_energy = log_scale_energy
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.use_patd = use_patd
        self.min_photons = min_photons
        self.input_delta_time = input_delta_time
        self.num_photons_per_sample = num_photons_per_sample if num_photons_per_sample is not None else min_photons
        # Handle multiple branch configurations
        if num_parallel_branches > 1:
            # Set up frequency scales for each branch
            if frequency_scales is None:
                # Use geometric progression of frequency scales
                self.frequency_scales = [frequency_scale * (2.0 ** i) for i in range(num_parallel_branches)]
            else:
                if len(frequency_scales) != num_parallel_branches:
                    raise ValueError(f"frequency_scales must have length {num_parallel_branches}")
                self.frequency_scales = frequency_scales
            
            # Set up number of frequencies for each branch
            if num_frequencies_per_branch is None:
                self.num_frequencies_per_branch = [num_frequencies] * num_parallel_branches
            else:
                if len(num_frequencies_per_branch) != num_parallel_branches:
                    raise ValueError(f"num_frequencies_per_branch must have length {num_parallel_branches}")
                self.num_frequencies_per_branch = num_frequencies_per_branch
        else:
            # Single branch case
            self.frequency_scales = [frequency_scale]
            self.num_frequencies_per_branch = [num_frequencies]
        
        # Initialize network architecture
        self.fourier_features_list = None
        self.mlp_branches = None
        self.shared_branch_mlp = None  # Single shared MLP for all branches
        self.final_mlp = None
        self.optimizer = None
        self.lr_scheduler = None  # Learning rate scheduler
        self.loss_fn = torch.nn.BCELoss()  # Changed from BCEWithLogitsLoss
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_trained = False

    def _pos_norm_divisor(self):
        """Return a divisor for normalizing (x,y,z) positions.

        - If domain_size is scalar: divisor is (domain_size/2) (original behavior).
        - If domain_size is (width, height): divisor is (width/2, width/2, height/2).

        Returns a float or a (3,) torch.Tensor on self.device.
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
            return torch.tensor(
                [width / 2.0, width / 2.0, height / 2.0],
                device=self.device,
                dtype=torch.float32,
            )

        return domain_size / 2.0
        
    def _build_network(self, input_dim):
        """Build the parallel MLP network architecture with multiple Fourier feature mappings."""
        
        print(f"Building parallel network architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Number of parallel branches: {self.num_parallel_branches}")
        print(f"  Shared MLP: {self.shared_mlp}")
        
        if self.use_fourier_features:
            # Create multiple Fourier feature layers
            self.fourier_features_list = torch.nn.ModuleList()
            fourier_output_dims = []
            
            for i in range(self.num_parallel_branches):
                fourier_layer = FourierFeatures(
                    input_dim=input_dim,
                    num_frequencies=self.num_frequencies_per_branch[i],
                    frequency_scale=self.frequency_scales[i],
                    learnable=self.learnable_frequencies
                ).to(self.device)
                
                self.fourier_features_list.append(fourier_layer)
                fourier_output_dims.append(fourier_layer.output_dim)
                
                print(f"  Branch {i}: {self.num_frequencies_per_branch[i]} frequencies, "
                      f"scale {self.frequency_scales[i]:.3f}, output dim {fourier_layer.output_dim}")
        else:
            # No Fourier features, use raw input for all branches
            self.fourier_features_list = None
            fourier_output_dims = [input_dim] * self.num_parallel_branches
        
        # Create MLP branches - either shared or separate
        if self.shared_mlp:
            # For shared MLP, create a single MLP that can handle different input dimensions
            # We'll use the maximum Fourier output dimension or pad smaller inputs
            max_fourier_dim = max(fourier_output_dims)
            
            # Create shared MLP layers with optional residual connections
            shared_layers = []
            current_dim = max_fourier_dim
            
            # Input layer
            shared_layers.append(torch.nn.Linear(current_dim, self.hidden_dims[0]))
            shared_layers.append(torch.nn.SiLU())
            shared_layers.append(torch.nn.Dropout(self.dropout_rate))
            current_dim = self.hidden_dims[0]
            
            # Hidden layers with optional residual connections
            if self.use_residual_connections and len(self.hidden_dims) > 1:
                # Use residual blocks for hidden layers
                for j in range(len(self.hidden_dims) - 1):
                    shared_layers.append(ResidualBlock(current_dim, self.hidden_dims[j + 1], self.dropout_rate))
                    current_dim = self.hidden_dims[j + 1]
            else:
                # Regular linear layers
                for j in range(len(self.hidden_dims) - 1):
                    shared_layers.append(torch.nn.Linear(self.hidden_dims[j], self.hidden_dims[j + 1]))
                    shared_layers.append(torch.nn.SiLU())
                    shared_layers.append(torch.nn.Dropout(self.dropout_rate))
            
            # Output layer for shared MLP (no activation yet)
            branch_output_dim = self.hidden_dims[-1]
            shared_layers.append(torch.nn.Linear(current_dim, branch_output_dim))
            
            self.shared_branch_mlp = torch.nn.Sequential(*shared_layers).to(self.device)
            self.mlp_branches = None  # Not used in shared mode
            
            # Store input dimensions for padding logic
            self.fourier_output_dims = fourier_output_dims
            self.max_fourier_dim = max_fourier_dim
            
            # All branches will have same output dimension
            branch_output_dims = [branch_output_dim] * self.num_parallel_branches
            
            residual_info = "with residual connections" if self.use_residual_connections else "without residual connections"
            print(f"  Shared MLP {residual_info}: {max_fourier_dim} -> {self.hidden_dims} -> {branch_output_dim}")
            print(f"  Fourier output dims: {fourier_output_dims}")
            
        else:
            # Create separate MLP branches (original behavior)
            self.mlp_branches = torch.nn.ModuleList()
            self.shared_branch_mlp = None  # Not used in separate mode
            self.shared_core_mlp = None
            branch_output_dims = []
            
            for i in range(self.num_parallel_branches):
                mlp_input_dim = fourier_output_dims[i]
                
                # Create MLP for this branch with optional residual connections
                branch_layers = []
                current_dim = mlp_input_dim
                
                # Input layer
                branch_layers.append(torch.nn.Linear(current_dim, self.hidden_dims[0]))
                branch_layers.append(torch.nn.SiLU())
                branch_layers.append(torch.nn.Dropout(self.dropout_rate))
                current_dim = self.hidden_dims[0]
                
                # Hidden layers with optional residual connections
                if self.use_residual_connections and len(self.hidden_dims) > 1:
                    # Use residual blocks for hidden layers
                    for j in range(len(self.hidden_dims) - 1):
                        branch_layers.append(ResidualBlock(current_dim, self.hidden_dims[j + 1], self.dropout_rate))
                        current_dim = self.hidden_dims[j + 1]
                else:
                    # Regular linear layers
                    for j in range(len(self.hidden_dims) - 1):
                        branch_layers.append(torch.nn.Linear(self.hidden_dims[j], self.hidden_dims[j + 1]))
                        branch_layers.append(torch.nn.SiLU())
                        branch_layers.append(torch.nn.Dropout(self.dropout_rate))
                
                # Output layer for this branch (no activation yet)
                branch_output_dim = self.hidden_dims[-1]
                branch_layers.append(torch.nn.Linear(current_dim, branch_output_dim))
                
                branch_mlp = torch.nn.Sequential(*branch_layers).to(self.device)
                self.mlp_branches.append(branch_mlp)
                branch_output_dims.append(branch_output_dim)
                
                residual_info = "with residual connections" if self.use_residual_connections else "without residual connections"
                print(f"  Branch {i} MLP {residual_info}: {mlp_input_dim} -> {self.hidden_dims} -> {branch_output_dim}")
        
        # Create final MLP that combines all branch outputs
        total_branch_output_dim = sum(branch_output_dims)
        final_layers = []
        
        # Optionally add more layers to process the concatenated features
        final_hidden_dim = min(64, total_branch_output_dim // 2)  # Adaptive sizing
        
        final_layers.append(torch.nn.Linear(total_branch_output_dim, final_hidden_dim))
        final_layers.append(torch.nn.SiLU())
        final_layers.append(torch.nn.Dropout(self.dropout_rate))
        
        # # Final output layer with sigmoid
        final_layers.append(torch.nn.Linear(final_hidden_dim, 1))
        final_layers.append(torch.nn.Sigmoid())
        
        self.final_mlp = torch.nn.Sequential(*final_layers).to(self.device)
        
        print(f"  Final MLP: {total_branch_output_dim} -> {final_hidden_dim} -> 1")
        # print(f"  Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
        # Create optimizer for all trainable parameters
        all_params = []
        
        # Add MLP branch parameters
        if self.shared_mlp:
            # Add shared MLP parameters
            all_params.extend(self.shared_branch_mlp.parameters())
        else:
            for branch in self.mlp_branches:
                all_params.extend(branch.parameters())
        
        # Add final MLP parameters
        all_params.extend(self.final_mlp.parameters())
        
        # Add Fourier feature parameters if learnable
        if self.fourier_features_list is not None and self.learnable_frequencies:
            for fourier_layer in self.fourier_features_list:
                all_params.extend(fourier_layer.parameters())
        
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
        # Initialize learning rate scheduler if requested
        if self.reduce_lr_on_plateau:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=self.lr_scheduler_min_lr,
            )
        
        print(f"  Total trainable parameters: {sum(p.numel() for p in all_params if p.requires_grad):,}")
        
    
    def compute_distance_from_beam(self, point, track_pos, track_dir):
        """
        Compute longitudinal and perpendicular distances from the track.
        
        Returns:
        --------
        dist_long : torch.Tensor
            Distance along the track direction (projection).
        dist_perp : torch.Tensor
            Perpendicular distance from the track.
        """
        # Vector from track vertex to point
        # point: (..., 3), track_pos: (..., 3)
        if point.dim() == 1:
            point = point.unsqueeze(0)
        if track_pos.dim() == 1:
            track_pos = track_pos.unsqueeze(0)
        if track_dir.dim() == 1:
            track_dir = track_dir.unsqueeze(0)
            
        rel_pos = point - track_pos
        
        # Project onto track direction
        # track_dir should be normalized
        # dist_long = rel_pos . track_dir
        dist_long = torch.sum(rel_pos * track_dir, dim=-1, keepdim=True)
        
        # Perpendicular distance vector
        perp_vec = rel_pos - dist_long * track_dir
        dist_perp = torch.norm(perp_vec, dim=-1, keepdim=True)
        
        return dist_long, dist_perp
    
    # def prepare_data_from_raw(self, point, event_data, surrogate_func, event_labels=['position', 'energy', 'zenith', 'azimuth'], noise_scale=0.0, add_relative_pos=True, signal_event_data=None, output_true_light_yield=False, device=None):
    #     """
    #     Prepare feature vector for the network.
        
    #     Logic:
    #     - The 'hypothesis' parameters come from `event_data`.
    #     - The 'observed' data (light yield) comes from `signal_event_data` (if provided) 
    #       OR is generated from `event_data` (if signal_event_data is None).
    #     """
    #     if device is None:
    #         device = self.device

    #     # 1. Extract Hypothesis Parameters (Theta)
    #     # ----------------------------------------
    #     # Convert dictionary to tensor if necessary, handling batches
    #     # This part depends on how your event_data is structured (dict of tensors or single tensor)
    #     # Assuming event_data is a dict of tensors/floats here for clarity
        
    #     # Helper to ensure tensor
    #     def to_tensor(val):
    #         if torch.is_tensor(val): return val.to(device).float()
    #         return torch.tensor(val, dtype=torch.float32, device=device)

    #     # Extract track parameters for geometry calculation
    #     track_pos = to_tensor(event_data['position'])
    #     if track_pos.dim() == 1: track_pos = track_pos.unsqueeze(0)
        
    #     # Calculate track direction from zenith/azimuth
    #     if 'direction' in event_data:
    #         track_dir = to_tensor(event_data['direction'])
    #     elif 'zenith' in event_data and 'azimuth' in event_data:
    #         zenith = to_tensor(event_data['zenith'])
    #         azimuth = to_tensor(event_data['azimuth'])
            
    #         sz, cz = torch.sin(zenith), torch.cos(zenith)
    #         sa, ca = torch.sin(azimuth), torch.cos(azimuth)
    #         track_dir = torch.stack([sz*ca, sz*sa, cz], dim=-1) # (Batch, 3)
    #         if track_dir.dim() == 3: # Handle stack adding a dimension if inputs were (N, 1)
    #             track_dir = track_dir.squeeze(1)
        
    #     else:
    #         # Default or error handling
    #         track_dir = torch.zeros_like(track_pos)
    #         track_dir[:, 2] = 1.0 # Default z-axis
        
    #     # 2. Calculate Geometric Features
    #     # -------------------------------
    #     # Ensure point is correct shape
    #     if isinstance(point, np.ndarray):
    #         point = torch.tensor(point, device=device, dtype=torch.float32)
    #     else:
    #         point = point.to(device).float()
            
    #     if point.dim() == 1: point = point.unsqueeze(0)
        
      
        
    #     # Calculate invariant distances (Crucial for physics performance)
    #     if self.add_distance_from_beam:   
    #         dist_long, dist_perp = self.compute_distance_from_beam(point, track_pos, track_dir)
        
    #     # 3. Get Observed Data (x)
    #     # ------------------------
    #     if signal_event_data is not None:
    #         # Case: "Marginal" sample or pre-computed observation
    #         # signal_event_data can be either:
    #         # 1. A dict of event parameters (generate LY from these parameters)
    #         # 2. A tensor (pre-computed light yield to reuse)
   
    #         observed_ly = surrogate_func(opt_point=point, event_params=signal_event_data)
    #     else:
    #         # Case: "Joint" sample - observed data matches the hypothesis parameters
    
    #         observed_ly = surrogate_func(opt_point=point, event_params=event_data)
    #         # Add noise if requested (simulating detector resolution)
    #     if noise_scale > 0:
    #         noise = torch.randn_like(observed_ly) * noise_scale
    #         observed_ly = observed_ly + noise
        
    #     # Log scale light yield (it spans orders of magnitude)
    #     if self.log_scale_ly:
    #         # Add small epsilon to avoid log(0)
    #         observed_ly = torch.log10(torch.abs(observed_ly) + 1e-10)

    #     # 4. Construct Feature Vector
    #     # ---------------------------
    #     # We want to feed the network: [Observed_LY, Geometric_Features, Energy]
    #     # We usually DON'T feed raw absolute positions (x,y,z) as they aren't translation invariant.
        
    #     feature_list = []
        
    #     if add_relative_pos:
    #         # Add relative cartesian coordinates
    #         if self.norm_pos:
    #             rel_pos = point/(self.domain_size/2)
    #         else:
    #             rel_pos = point
    #         feature_list.append(rel_pos.flatten())
            
    #     if self.add_distance_from_beam:
    #         # Add powerful geometric invariants
    #         feature_list.append(dist_long.flatten())
    #         feature_list.append(dist_perp.flatten())
            
    #     # Add Energy (Hypothesis)
    #     if 'energy' in event_labels:
    #         energy = to_tensor(event_data['energy']).view(-1, 1)
    #         # Log scale energy is usually better for NN
    #         feature_list.append(torch.log10(energy.flatten()))
            
    #     # Add Direction (Hypothesis) - usually better as vector than angles
    #     if 'zenith' in event_labels and 'azimuth' in event_labels:
    #         feature_list.append(track_dir.flatten())
        
    #     if 'position' in event_labels:
    #         feature_list.append(track_pos.flatten())

    #     # Concatenate
    #     features = torch.cat(feature_list, dim=0)
        
    #     if output_true_light_yield:
    #         return features, observed_ly
            
    #     return features
    
    def prepare_features_patd(self, point, event_data, patd_result):
        """
        Build per-photon feature vectors from a pre-computed PATD surrogate result.

        This is an alternative to prepare_data_from_raw_patd that mirrors the feature
        engineering used in the NSF notebook (create_event_features / sample_detector_features).
        It exposes richer geometric features that are strong discriminators for the
        hit-time LLR classifier:

          [det_x, det_y, det_z,           normalised detector position        (3)
           v_x,   v_y,   v_z,             normalised vertex position           (3)
           d_x,   d_y,   d_z,             unit direction vector                (3)
           log10(E)/8,                    log-scaled energy                    (1)
           vert_dist,                     L2(detector - vertex) normalised     (1)
           cos_angle,                     cos(direction ∠ vertex→detector)     (1)
           t_hit (log-sign scaled)]       per-photon arrival time              (1)
                                                                           total = 13

        The first 12 entries are the same for every photon in the event and are
        replicated across rows.  Only the last column varies per photon.

        Normalisation uses self.domain_size via _pos_norm_divisor(), so it is
        consistent with the rest of LLRnet regardless of detector geometry.

        Parameters
        ----------
        point : torch.Tensor or np.ndarray
            Detector position, shape (3,) or (1, 3).
        event_data : dict
            Event parameters.  Must contain 'position', 'energy', 'direction'.
        patd_result : dict
            Output of the PATD surrogate.  Must contain 'hit_times' and 'num_photons'.

        Returns
        -------
        features : torch.Tensor, shape (num_photons, 13)
            Per-photon feature matrix, or None if num_photons == 0.
        num_photons : int
        """
        num_photons = patd_result['num_photons']
        if isinstance(num_photons, torch.Tensor):
            num_photons = int(num_photons.item())
        if num_photons == 0:
            return None, 0

        if isinstance(point, np.ndarray):
            point = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point = point.float().to(self.device)
        point = point.squeeze()  # (3,)

        norm = self._pos_norm_divisor()  # scalar or (3,) tensor

        # --- detector and vertex positions, normalised ---
        det = point / norm  # (3,)

        vert = event_data['position']
        if isinstance(vert, np.ndarray):
            vert = torch.tensor(vert, device=self.device, dtype=torch.float32)
        else:
            vert = vert.float().to(self.device)
        vert = vert.squeeze() / norm  # (3,)

        # --- direction (already a unit vector) ---
        direction = event_data['direction']
        if isinstance(direction, np.ndarray):
            direction = torch.tensor(direction, device=self.device, dtype=torch.float32)
        else:
            direction = direction.float().to(self.device)
        direction = direction.squeeze()  # (3,)

        # --- log-scaled energy ---
        energy = event_data['energy']
        if isinstance(energy, np.ndarray):
            energy = torch.tensor(energy, device=self.device, dtype=torch.float32)
        else:
            energy = energy.float().to(self.device)
        log_energy = torch.log10(energy.squeeze() + 1e-10) / 8.0  # scalar

        # --- derived geometric scalars ---
        rel = det - vert  # detector relative to vertex, in normalised coords
        vert_dist = torch.norm(rel)
        cos_angle = torch.dot(direction, rel) / (torch.norm(direction) * vert_dist + 1e-8)

        # --- t_geom_min: minimum geometric photon arrival time ---
        # t_geom_min = patd_result['t_geom_min']
        # if isinstance(t_geom_min, np.ndarray):
        #     t_geom_min = torch.tensor(t_geom_min, device=self.device, dtype=torch.float32)
        # else:
        #     t_geom_min = t_geom_min.float().to(self.device)
        # t_geom_min_scalar = t_geom_min.squeeze().mean() / 1e5  # scalar

        # --- assemble the 13 event-level context features ---
        event_features = torch.stack([
            det[0], det[1], det[2],
            vert[0], vert[1], vert[2],
            direction[0], direction[1], direction[2],
            log_energy,
            vert_dist,
            cos_angle,
            # t_geom_min_scalar,
        ])  # (13,)

        # --- per-photon hit times: log-sign scaled ---
        hit_times = patd_result['hit_times'].float().to(self.device)  # (N,)
        t_scaled = torch.where(
            hit_times < 0,
            -torch.log10(-hit_times + 1e-4) / 4.0,
            torch.log10(hit_times + 1e-4) / 4.0,
        )  # (N,)

        # --- replicate event features and append per-photon time ---
        event_features_batch = event_features.unsqueeze(0).expand(num_photons, -1)  # (N, 13)
        features = torch.cat([event_features_batch, t_scaled.unsqueeze(1)], dim=1)  # (N, 14)

        return features.clone().detach(), num_photons

    def prepare_features_charge(self, point, event_data, light_yield):
        """
        Build a feature vector for the charge (non-PATD) LLR network from a
        pre-computed light yield value.

        This is the charge analogue of prepare_features_patd.  It uses the same
        rich geometry as the notebook's create_event_features — adding vert_dist
        and cos_angle instead of raw relative position — and takes a pre-computed
        light yield so that the observation can be held fixed while only the
        hypothesis parameters change (needed for correct NLL landscape evaluation).

        Feature layout (10 features total):
          [det_x, det_y, det_z,       normalised detector position    (3)
           v_x,   v_y,   v_z,         normalised vertex position       (3)
           d_x,   d_y,   d_z,         unit direction vector            (3)
           log10(E)/8,                log-scaled energy                (1)
           vert_dist,                 L2(detector - vertex) normalised (1)
           cos_angle,                 cos(direction ∠ vertex→detector) (1)
           log_ly]                    log-scaled light yield           (1)
                                                               total = 13

        Normalisation uses self.domain_size via _pos_norm_divisor().

        Parameters
        ----------
        point : torch.Tensor or np.ndarray
            Detector position, shape (3,) or (1, 3).
        event_data : dict
            Event parameters. Must contain 'position', 'energy', 'direction'.
        light_yield : torch.Tensor or float
            Pre-computed light yield scalar (the observation).

        Returns
        -------
        features : torch.Tensor, shape (13,)
        """
        if isinstance(point, np.ndarray):
            point = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point = point.float().to(self.device)
        point = point.squeeze()  # (3,)

        norm = self._pos_norm_divisor()  # scalar or (3,) tensor

        # --- detector and vertex positions, normalised ---
        det = point / norm  # (3,)

        vert = event_data['position']
        if isinstance(vert, np.ndarray):
            vert = torch.tensor(vert, device=self.device, dtype=torch.float32)
        else:
            vert = vert.float().to(self.device)
        vert = vert.squeeze() / norm  # (3,)

        # --- direction (already a unit vector) ---
        direction = event_data['direction']
        if isinstance(direction, np.ndarray):
            direction = torch.tensor(direction, device=self.device, dtype=torch.float32)
        else:
            direction = direction.float().to(self.device)
        direction = direction.squeeze()  # (3,)

        # --- log-scaled energy ---
        energy = event_data['energy']
        if isinstance(energy, np.ndarray):
            energy = torch.tensor(energy, device=self.device, dtype=torch.float32)
        else:
            energy = energy.float().to(self.device)
        log_energy = torch.log10(energy.squeeze() + 1e-10) / 8.0  # scalar

        # --- derived geometric scalars ---
        rel = det - vert
        vert_dist = torch.norm(rel)
        cos_angle = torch.dot(direction, rel) / (torch.norm(direction) * vert_dist + 1e-8)

        # --- log-scaled light yield ---
        if not isinstance(light_yield, torch.Tensor):
            light_yield = torch.tensor(light_yield, device=self.device, dtype=torch.float32)
        else:
            light_yield = light_yield.float().to(self.device)
        log_ly = torch.log10(torch.abs(light_yield.squeeze()) + 1e-10) / 4.0  # scalar

        features = torch.stack([
            det[0], det[1], det[2],
            vert[0], vert[1], vert[2],
            direction[0], direction[1], direction[2],
            log_energy,
            vert_dist,
            cos_angle,
            log_ly,
        ])  # (13,)

        return features.clone().detach()

    def prepare_data_from_raw_patd(self, point, event_data, surrogate_func, event_labels=['position', 'energy', 'zenith', 'azimuth'], signal_event_data=None, num_samples=1, input_photons=None):
        """
        Prepare training data from raw neutrino event data in PATD mode.
        
        This method is specifically for use_patd=True mode. It calls the surrogate function
        to get photon arrival times and prepares features for each photon hit.
        
        Parameters:
        -----------
        point : torch.Tensor or np.ndarray
            Detector point coordinates
        event_data : dict
            Raw event data dictionary with keys like 'position', 'energy', 'zenith', 'azimuth'
        surrogate_func : callable
            Function to calculate PATD (returns dict with 'hit_times' and 'num_photons')
        event_labels : list
            List of event parameter keys to include as features
        signal_event_data : dict, optional
            If provided, uses these parameters for hypothesis while event_data provides observation
        num_samples : int
            Number of times to call surrogate function (for generating multiple PATD realizations).
            If > 1, generates multiple independent PATD samples and concatenates all photon features.
            
        Returns:
        --------
        tuple or None
            If num_samples=1:
                - (features_batch, num_photons) where features_batch has shape (num_photons, feature_dim)
                - Returns (None, 0) if no photons detected
            If num_samples>1:
                - (features_batch, total_num_photons) where features_batch has shape (total_photons, feature_dim)
                - total_photons is sum of photons across all samples
                - Returns (None, 0) if no photons detected in any sample
        """
        if not self.use_patd:
            raise ValueError("prepare_data_from_raw_patd should only be called when use_patd=True")
        
        if signal_event_data is None:
            signal_event_data = event_data
        
        # Convert point to tensor
        if isinstance(point, np.ndarray):
            point_tensor = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point_tensor = point.float().to(self.device)
        
        if point_tensor.dim() == 1:
            point_tensor = point_tensor.unsqueeze(0)  # (1, 3)
        
        # Build base event features (will be replicated for each photon)
        feature_list = []
        
        # Add point coordinates
        if self.norm_pos:
            norm_points = point_tensor / self._pos_norm_divisor()
        else:
            norm_points = point_tensor
        feature_list.extend(norm_points.flatten())  # (3,)
        
        # Add relative position
        if self.add_relative_pos and 'position' in signal_event_data:
            event_pos = signal_event_data['position']
            if isinstance(event_pos, np.ndarray):
                event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
            else:
                event_pos_tensor = event_pos.float().to(self.device)
            if event_pos_tensor.dim() == 1:
                event_pos_tensor = event_pos_tensor.unsqueeze(0)
            relative_pos = point_tensor - event_pos_tensor
            feature_list.extend(relative_pos.flatten())  # (3,)
        
        # Add distance from beam
        if self.add_distance_from_beam and "direction" in signal_event_data:
            track_dir = signal_event_data["direction"]
            if isinstance(track_dir, np.ndarray):
                track_dir = torch.tensor(track_dir, device=self.device, dtype=torch.float32)
            else:
                track_dir = track_dir.float().to(self.device)
            event_pos = signal_event_data['position']
            if isinstance(event_pos, np.ndarray):
                event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
            else:
                event_pos_tensor = event_pos.float().to(self.device)
            if event_pos_tensor.dim() == 1:
                event_pos_tensor = event_pos_tensor.unsqueeze(0)
            if track_dir.dim() == 1:
                track_dir = track_dir.unsqueeze(0)
            dist_long, dist_perp = self.compute_distance_from_beam(point_tensor, event_pos_tensor, track_dir)
            feature_list.extend(dist_perp.flatten())  # (1,)
        
        # Add event parameters
        for key in event_labels:
            if key in signal_event_data:
                feature = signal_event_data[key]
                if isinstance(feature, np.ndarray):
                    feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                elif not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                if self.log_scale_energy and key == 'energy':
                    feature = torch.log10(feature + 1e-10)
                if self.norm_pos and key == 'position':
                    feature = feature / self._pos_norm_divisor()
                feature_list.extend(feature.flatten())  # (feature_dim,)
        
        # Build base features (same for all photons in an event)
          # (base_feature_dim,)
        
        # Generate PATD samples (multiple calls to surrogate if num_samples > 1)
        all_features_batches = []
        total_photons = 0
        if not self.input_charge:
            base_features = torch.stack(feature_list).to(self.device)  # (base_feature_dim,)
        
        for sample_idx in range(num_samples):
            if self.input_charge:
                feature_list_copy = feature_list
            # Call surrogate function to get PATD for this sample
            if isinstance(input_photons, list) and len(input_photons) == num_samples:
                final_photons = input_photons[sample_idx]
            elif isinstance(input_photons, int):
                final_photons = input_photons
            else:
                final_photons = None
            
            if final_photons is not None:
                if final_photons == 0:
                    continue  # Skip this sample if zero photons requested
            
            with torch.no_grad():
                patd_result = surrogate_func(opt_point=point_tensor.squeeze(0), event_params=event_data, input_photons=final_photons)
            
            hit_times = patd_result['hit_times']
            num_photons = patd_result['num_photons']
            
            
            # Skip if no photons in this sample
            if num_photons == 0:
                continue
            
            # Process hit times
            if self.rel_time or self.input_delta_time:
                vertex_times = patd_result.get('vertex_times', None)
                emmission_points = patd_result.get('emission_points', None)
                if vertex_times is not None and emmission_points is not None:
                    # Calculate expected arrival time based on direct path from emission point to detector
                    # This gives us a more physical "relative time" that accounts for geometry and speed of light
                    speed_of_light = 299792458.0/(1.3e9)  # m/ns
                    emmission_points_tensor = torch.as_tensor(emmission_points, dtype=torch.float32, device=self.device)
                    vertex_times_tensor = torch.as_tensor(vertex_times, dtype=torch.float32, device=self.device)
                    
                    # Calculate distance from emission points to detector point
                    distances = torch.norm(emmission_points_tensor - point_tensor, dim=-1)  # (num_photons,)
                    
                    expected_arrival_times = vertex_times_tensor + distances / speed_of_light  # (num_photons,)
                    if self.rel_time:
                        hit_times = hit_times - expected_arrival_times  # Relative times from expected arrival
                    elif self.input_delta_time:
                        delta_times = hit_times - expected_arrival_times
            
            if self.log_scale_ly:
                # account 
                # hit_times_processed = torch.log10(torch.abs(hit_times) + 1e-4).view(-1, 1)/4
                hit_times_processed = torch.where(hit_times<0, -torch.log10(-hit_times)/4, torch.log10(hit_times)/4).view(-1, 1)
                if self.input_delta_time:
                    delta_times_processed = torch.where(delta_times<0, -torch.log10(-delta_times)/4, torch.log10(delta_times)/4).view(-1, 1)
            else:
                hit_times_processed = hit_times.view(-1, 1)/1e4
                if self.input_delta_time:
                    delta_times_processed = delta_times.view(-1, 1)/1e4
                
            hit_times_processed = hit_times_processed.sort(dim=0).values  # Sort hit times for better learning
            
            
            if self.input_charge:
                feature_list_copy.extend(torch.tensor([num_photons]))  # Add number of photons as a feature
            # print(feature_list)
                base_features = torch.stack(feature_list_copy).to(self.device)  # (base_feature_dim,)
            # Replicate base features for each photon
            num_photons_int = int(num_photons.item()) if isinstance(num_photons, torch.Tensor) else int(num_photons)
            base_features_batch = base_features.unsqueeze(0).repeat(num_photons_int, 1)
            
            # Concatenate base features with hit times
            if self.input_delta_time:
                features_batch = torch.cat([base_features_batch, hit_times_processed, delta_times_processed], dim=1)
            else:    
                features_batch = torch.cat([base_features_batch, hit_times_processed], dim=1)
            
            all_features_batches.append(features_batch)
            total_photons += num_photons_int
        
        # Handle case where no photons were detected in any sample
        if total_photons == 0:
            return None, 0
        
        # Concatenate all samples if multiple
        if num_samples == 1:
            return all_features_batches[0], total_photons
        else:
            # Concatenate features from all samples
            combined_features = torch.cat(all_features_batches, dim=0)  # (total_photons, feature_dim)
            return combined_features, total_photons
    
    def prepare_data_from_raw(self, point, event_data, surrogate_func, event_labels=['position', 'energy', 'zenith', 'azimuth'], noise_scale=0.0, signal_event_data=None, output_true_light_yield=False, num_samples=1):
        """
        Prepare training data from raw neutrino event data (non-PATD mode).
        
        For PATD mode, use prepare_data_from_raw_patd instead.
        
        Parameters:
        -----------
        point : torch.Tensor or np.ndarray
            Detector point coordinates
        event_data : dict
            Raw event data dictionary with keys like 'position', 'energy', 'zenith', 'azimuth'
        surrogate_func : callable
            Function to calculate detector response/light yield
        event_labels : list
            List of event parameter keys to include as features
        noise_scale : float
            Scale for adding noise to generate multiple realizations
        signal_event_data : dict, optional
            If provided, uses these parameters for hypothesis while event_data provides observation
        output_true_light_yield : bool
            If True, returns tuple of (features, light_yields)
        num_samples : int
            Number of noise realizations to generate (for batched feature generation).
            If > 1, returns batched features with different noise per sample.
            
        Returns:
        --------
        torch.Tensor or tuple
            If num_samples=1:
                - features tensor of shape (feature_dim,)
            If num_samples>1:
                - features tensor of shape (num_samples * n_points, feature_dim)
                - if output_true_light_yield=True: (features, light_yields) where light_yields has shape (num_samples * n_points,)
        """
        if self.use_patd:
            raise ValueError("prepare_data_from_raw should not be called when use_patd=True. Use prepare_data_from_raw_patd instead.")
        
        if signal_event_data is None:
            signal_event_data = event_data
        
        # Handle batched sampling with multiple realizations
        # Generate event features once, then only regenerate detector responses
        if num_samples > 1:
            # Convert point to tensor once
            if isinstance(point, np.ndarray):
                point_tensor = torch.tensor(point, device=self.device, dtype=torch.float32)
            else:
                point_tensor = point.float().to(self.device)
            
            if point_tensor.dim() == 1:
                point_tensor = point_tensor.unsqueeze(0)  # (1, 3)
            
            n_points = point_tensor.shape[0]
            
            # Generate event features for all points (vectorized)
            # Point coordinates - shape: (n_points, 3)
            if self.norm_pos:
                norm_points = point_tensor / self._pos_norm_divisor()
            else:
                norm_points = point_tensor
            
            # Relative position - shape: (n_points, 3)
            relative_pos = None
            if self.add_relative_pos and 'position' in signal_event_data:
                event_pos = signal_event_data['position']
                if isinstance(event_pos, np.ndarray):
                    event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
                else:
                    event_pos_tensor = event_pos.float().to(self.device)
                if event_pos_tensor.dim() == 1:
                    event_pos_tensor = event_pos_tensor.unsqueeze(0)
                relative_pos = point_tensor - event_pos_tensor  # (n_points, 3)
            
            # Distance from beam - shape: (n_points, 1)
            dist_perp = None
            if self.add_distance_from_beam and "direction" in signal_event_data:
                track_dir = signal_event_data["direction"]
                if isinstance(track_dir, np.ndarray):
                    track_dir = torch.tensor(track_dir, device=self.device, dtype=torch.float32)
                else:
                    track_dir = track_dir.float().to(self.device)
                event_pos = signal_event_data['position']
                if isinstance(event_pos, np.ndarray):
                    event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
                else:
                    event_pos_tensor = event_pos.float().to(self.device)
                if event_pos_tensor.dim() == 1:
                    event_pos_tensor = event_pos_tensor.unsqueeze(0)
                if track_dir.dim() == 1:
                    track_dir = track_dir.unsqueeze(0)
                dist_long, dist_perp = self.compute_distance_from_beam(point_tensor, event_pos_tensor, track_dir)
            
            # Event parameters (same for all points) - will be replicated
            event_param_features = []
            for key in event_labels:
                if key in signal_event_data:
                    feature = signal_event_data[key]
                    if isinstance(feature, np.ndarray):
                        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                    elif not isinstance(feature, torch.Tensor):
                        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                    if self.log_scale_energy and key == 'energy':
                        feature = torch.log10(feature + 1e-10)
                    if self.norm_pos and key == 'position':
                        feature = feature / self._pos_norm_divisor()
                    event_param_features.append(feature.flatten())
            
            # Concatenate event params (scalar/vector features)
            if event_param_features:
                event_params_cat = torch.cat(event_param_features, dim=0)  # (param_dim,)
            else:
                event_params_cat = torch.tensor([], device=self.device)
            
            # Build event features for each point (n_points, event_feature_dim)
            point_event_features_list = []
            point_event_features_list.append(norm_points)  # (n_points, 3)
            if relative_pos is not None:
                point_event_features_list.append(relative_pos)  # (n_points, 3)
            if dist_perp is not None:
                point_event_features_list.append(dist_perp)  # (n_points, 1)
            
            # Replicate event params for each point
            if len(event_params_cat) > 0:
                event_params_replicated = event_params_cat.unsqueeze(0).expand(n_points, -1)  # (n_points, param_dim)
                point_event_features_list.append(event_params_replicated)
            
            point_event_features = torch.cat(point_event_features_list, dim=1)  # (n_points, event_feature_dim)
            n_points = point_event_features.shape[0]
            
            # Now generate detector responses for all points, num_samples times
            # Call surrogate once for all points (vectorized), then loop only over samples
            all_detector_responses = []
            all_light_yields = []
            
            for _ in range(num_samples):
                # Generate detector response for all points at once (vectorized)
                with torch.no_grad():
                    responses = surrogate_func(opt_point=point_tensor, event_params=event_data)
                
                if isinstance(responses, np.ndarray):
                    responses = torch.tensor(responses, device=self.device, dtype=torch.float32)
                elif not isinstance(responses, torch.Tensor):
                    responses = torch.tensor(responses, device=self.device, dtype=torch.float32)
                responses = responses.float().to(self.device)

                # Surrogate functions sometimes return a scalar for a single point.
                # Normalize to shape (n_points,) so batching logic below is consistent.
                responses = responses.reshape(-1)
                if responses.numel() != n_points:
                    if responses.numel() == 1:
                        responses = responses.expand(n_points)
                    else:
                        raise ValueError(
                            f"surrogate_func returned {responses.numel()} responses, expected {n_points}. "
                            f"point_tensor shape={tuple(point_tensor.shape)}"
                        )
                
                if output_true_light_yield:
                    all_light_yields.append(responses.clone())
                
                # Add noise if requested
                if noise_scale is not None and noise_scale > 0:
                    noise = torch.randn_like(responses) * noise_scale
                    responses = responses + noise
                
                # Apply log scaling if needed
                if self.log_scale_ly:
                    responses = torch.log10(torch.abs(responses) + 1e-10)/4
                
                all_detector_responses.append(responses)  # (n_points,)
            
            # Stack: (num_samples, n_points)
            detector_responses_batched = torch.stack(all_detector_responses)

            # If n_points==1 and surrogate returned scalars, stack could produce (num_samples,).
            # Force (num_samples, n_points) to make unsqueeze(2) valid.
            if detector_responses_batched.dim() == 1:
                detector_responses_batched = detector_responses_batched.unsqueeze(1)
            
            # Replicate point_event_features for each sample: (num_samples, n_points, event_feature_dim)
            point_event_features_batched = point_event_features.unsqueeze(0).expand(num_samples, -1, -1)
            
            # Add detector response dimension: (num_samples, n_points, 1)
            detector_responses_expanded = detector_responses_batched.unsqueeze(2)
            
            # Concatenate: (num_samples, n_points, total_feature_dim)
            features_batched = torch.cat([point_event_features_batched, detector_responses_expanded], dim=2)
            
            # Reshape to (num_samples * n_points, total_feature_dim)
            features_batched = features_batched.reshape(num_samples * n_points, -1)
            
            if output_true_light_yield:
                light_yields_batched = torch.stack(all_light_yields).reshape(num_samples * n_points)
                return features_batched, light_yields_batched
            return features_batched
        
        if isinstance(event_data, dict):
            # Extract features from dictionary
            
            # Add detector point coordinates
            if isinstance(point, np.ndarray):
                point_tensor = torch.tensor(point, device=self.device, dtype=torch.float32)
            else:
                point_tensor = point.float().to(self.device)
            
            if self.use_patd:
                # For PATD mode, surrogate_func returns dict with 'hit_times' and 'num_photons'
                with torch.no_grad():    
                    patd_result = surrogate_func(opt_point=point_tensor, event_params=event_data)
                hit_times = patd_result['hit_times']
                num_photons = patd_result['num_photons']
                if num_photons == 0:
                    # Return empty features if no photons
                    return None, 0
            
            
            # Determine if we have batched points
            if point_tensor.dim() == 1:
                # Single point (3,) - keep original behavior
                is_batched = False
                n_points = 1
                point_tensor = point_tensor.unsqueeze(0)  # (1, 3)
            elif point_tensor.dim() == 2:
                # Batched points (N, 3)
                is_batched = True
                n_points = point_tensor.shape[0]
            else:
                raise ValueError(f"point must be 1D or 2D, got shape {point_tensor.shape}")
            
            # Initialize feature list for batch
            feature_list = []
            
            # Flatten point coordinates to (N, 3) or (N*3,) depending on batch
            if self.norm_pos:
                norm_points = point_tensor / self._pos_norm_divisor()
            else:
                norm_points = point_tensor
            
            # For batched: keep as (N, 3), for single: flatten to (3,)
            if is_batched:
                feature_list.append(norm_points)  # (N, 3)
            else:
                feature_list.append(norm_points.flatten())  # (3,)
            if self.add_relative_pos and 'position' in signal_event_data:
                event_pos = signal_event_data['position']
                if isinstance(event_pos, np.ndarray):
                    event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
                else:
                    event_pos_tensor = event_pos.float().to(self.device)
                # Ensure event_pos is broadcasted correctly for batch
                if event_pos_tensor.dim() == 1:
                    event_pos_tensor = event_pos_tensor.unsqueeze(0)  # (1, 3)
                relative_pos = point_tensor - event_pos_tensor  # (N, 3)
                if is_batched:
                    feature_list.append(relative_pos)  # (N, 3)
                else:
                    feature_list.append(relative_pos.flatten())  # (3,)
            
            if self.add_distance_from_beam:
                if "direction" in signal_event_data:
                    track_dir = signal_event_data["direction"]
                    if isinstance(track_dir, np.ndarray):
                        track_dir = torch.tensor(track_dir, device=self.device, dtype=torch.float32)
                    else:
                        track_dir = track_dir.float().to(self.device)
                    event_pos = signal_event_data['position']
                    if isinstance(event_pos, np.ndarray):
                        event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
                    else:
                        event_pos_tensor = event_pos.float().to(self.device)
                    # Ensure proper broadcasting
                    if event_pos_tensor.dim() == 1:
                        event_pos_tensor = event_pos_tensor.unsqueeze(0)
                    if track_dir.dim() == 1:
                        track_dir = track_dir.unsqueeze(0)
                    dist_long, dist_perp = self.compute_distance_from_beam(point_tensor, event_pos_tensor, track_dir)
                    if is_batched:
                        feature_list.append(dist_perp)  # (N, 1)
                    else:
                        feature_list.append(dist_perp.flatten())  # (1,)
            # Add event parameters
            for key in event_labels:
                if key in signal_event_data:
                    feature = signal_event_data[key]
                    if isinstance(feature, np.ndarray):
                        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                    elif not isinstance(feature, torch.Tensor):
                        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                    if self.log_scale_energy and key == 'energy':
                        feature = torch.log10(feature + 1e-10)
                    if self.norm_pos and key == 'position':
                        feature = feature / self._pos_norm_divisor()
                    # Handle batching: replicate event parameter for each point
                    if is_batched:
                        # Ensure feature is at least 1D and flatten
                        if feature.dim() == 0:
                            feature = feature.unsqueeze(0)  # (1,)
                        else:
                            feature = feature.flatten()  # Flatten any multi-dimensional params to 1D
                        
                        # Now feature is 1D - expand to match batch
                        feature = feature.unsqueeze(0).expand(n_points, -1)  # (N, feature_dim)
                        feature_list.append(feature)
                    else:
                        feature_list.append(feature.flatten())  # (feature_dim,)
                else:
                    raise KeyError(f"Key '{key}' not found in event_data")
                    
            # Calculate detector response (standard light yield mode)
            if is_batched:
                # Compute detector response for each point in batch
                detector_responses = []
                for i in range(n_points):
                    pt = point_tensor[i]
                    with torch.no_grad():
                        response = surrogate_func(opt_point=pt, event_params=event_data)
                    if isinstance(response, np.ndarray):
                        response = torch.tensor(response, device=self.device, dtype=torch.float32)
                    elif not isinstance(response, torch.Tensor):
                        response = torch.tensor(response, device=self.device, dtype=torch.float32)
                    detector_responses.append(response.float().to(self.device))
                detector_response = torch.stack(detector_responses)  # (N,)
            else:
                with torch.no_grad():
                    detector_response = surrogate_func(opt_point=point_tensor.squeeze(0), event_params=event_data)
                if isinstance(detector_response, np.ndarray):
                    detector_response = torch.tensor(detector_response, device=self.device, dtype=torch.float32)
                elif not isinstance(detector_response, torch.Tensor):
                    detector_response = torch.tensor(detector_response, device=self.device, dtype=torch.float32)
                detector_response = detector_response.float().to(self.device)
            
            if output_true_light_yield:
                true_light_yield = detector_response.clone()
            
            # Apply log scaling if needed
            if self.log_scale_ly:
                detector_response = torch.log10(torch.abs(detector_response) + 1e-10)
            
            # Add to feature list
            if is_batched:
                feature_list.append(detector_response.unsqueeze(1))  # (N, 1)
            else:
                feature_list.append(detector_response.flatten())  # (1,)

            # Combine all features
            if is_batched:
                # Concatenate along feature dimension: (N, total_features)
                features = torch.cat(feature_list, dim=1)
            else:
                # Concatenate to 1D tensor
                features = torch.cat(feature_list, dim=0)
            
            if output_true_light_yield:
                return features, true_light_yield
            else:
                return features
        
    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                             verbose=True, early_stopping_patience=10, input_dim=None,
                             grad_clip=None, save_every_n_epochs=None,
                             checkpoint_path=None):
        """
        Train the LLR network using PyTorch DataLoader with balanced signal/background events.
        
        This method is designed to work with EventDataset that dynamically generates
        balanced signal and background events. The dataset ensures that for every signal
        event there is a corresponding background event with the same detector point and
        shared parameters, differing only in detector response.
        
        Parameters:
        -----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader providing training batches of (features, labels)
        val_dataloader : torch.utils.data.DataLoader, optional
            DataLoader for validation data. If None, no validation is performed.
        epochs : int
            Number of training epochs  
        verbose : bool
            Whether to print training progress
        early_stopping_patience : int
            Number of epochs to wait for improvement before early stopping
        save_every_n_epochs : int or None
            If set, save a checkpoint every N epochs during training.
        checkpoint_path : str or None
            File path to overwrite on each periodic save. Required when
            save_every_n_epochs is set.
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        if save_every_n_epochs is not None and save_every_n_epochs <= 0:
            raise ValueError("save_every_n_epochs must be a positive integer or None")
        if save_every_n_epochs is not None and checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided when save_every_n_epochs is set")

        # Build network if not already built
        # We need to get a sample to determine the feature dimension
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            if input_dim is None:
                sample_batch = next(iter(train_dataloader))
                # print(f"Sampled batch in {time.time() - start_time:.4f} seconds")
                sample_features, _ = sample_batch
                # Features are now (batch_size, feature_dim) since each sample is an individual event
                feature_dim = sample_features.shape[1]
                print(f"Number of datapoints per batch: {sample_features.shape[0]}")
                self._build_network(feature_dim)
            else:
                self._build_network(input_dim)
        
        # Training loop
        best_val_loss = float('inf') if val_dataloader is not None else None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            if self.shared_mlp:
                self.shared_branch_mlp.train()
            else:
                for branch in self.mlp_branches:
                    branch.train()
            self.final_mlp.train()
            if self.fourier_features_list is not None:
                for fourier_layer in self.fourier_features_list:
                    fourier_layer.train()
                    
            train_loss = 0.0
            n_batches = 0
            
            # Iterate through batches of individual events
            time_start = time.time()
            for batch_features, batch_labels in train_dataloader:
                
                
                # Debug: Check device and dtype before transfer
                if n_batches == 0 and epoch == 0:
                    print(f"Batch loaded in {time.time() - time_start:.4f} seconds")
                    # print(f"  Effective batch size: {batch_features.shape[0]} samples (photons for PATD)")
                    # print(f"  Feature dim: {batch_features.shape[1]}")
                    # print(f"  Features requires_grad: {batch_features.requires_grad}")
                    # print(f"  Features min: {batch_features[:,-1].min().item():.4f}, max: {batch_features[:,-1].max().item():.4f}")
                    # print(f"  Features mean: {batch_features[:,-1].mean().item():.4f}, std: {batch_features[:,-1].std().item():.4f}")
                    time_start = time.time()
                
                    # 
                # Each sample is now an individual event
                # batch_features shape: (batch_size, feature_dim)
                # batch_labels shape: (batch_size,)
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # if n_batches == 0 and epoch == 0:
                #     print(f"  Features device after .to(): {batch_features.device}")
                #     print(f"  Labels device after .to(): {batch_labels.device}")
                
                self.optimizer.zero_grad()
                outputs = self._forward_pass(batch_features)
            
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                if grad_clip is not None:
                    all_params = (
                        list(self.shared_branch_mlp.parameters()) if self.shared_mlp
                        else [p for branch in self.mlp_branches for p in branch.parameters()]
                    ) + list(self.final_mlp.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                self.optimizer.step()
                
                train_loss += loss.item()
                if n_batches == 0 and epoch == 0:
                    print(f"Batch trained in {time.time() - time_start:.4f} seconds", flush=True)
                n_batches += 1
                    
                    
                time_start = time.time()
            train_loss /= n_batches
            
            # Validation phase
            val_loss = None
            if val_dataloader is not None:
                if self.shared_mlp:
                    self.shared_branch_mlp.eval()
                else:
                    for branch in self.mlp_branches:
                        branch.eval()
                self.final_mlp.eval()
                if self.fourier_features_list is not None:
                    for fourier_layer in self.fourier_features_list:
                        fourier_layer.eval()
                        
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for batch_features, batch_labels in val_dataloader:
                        # Each sample is now an individual event
                        # batch_features shape: (batch_size, feature_dim)
                        # batch_labels shape: (batch_size,)
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        val_outputs = self._forward_pass(batch_features)
                        val_batch_loss = self.loss_fn(val_outputs, batch_labels)
                        val_loss += val_batch_loss.item()
                        n_val_batches += 1
                        
                val_loss /= n_val_batches
                
            # Store history
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            # Update learning rate scheduler if enabled
            if self.reduce_lr_on_plateau and self.lr_scheduler is not None:
                if val_loss is not None:
                    # Use validation loss for scheduler
                    self.lr_scheduler.step(val_loss)
                else:
                    # Use training loss if no validation data
                    self.lr_scheduler.step(train_loss)
            
            # Early stopping check (only if validation data provided)
            if val_dataloader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state_dict = {
                    'mlp_branches': [branch.state_dict().copy() for branch in self.mlp_branches] if not self.shared_mlp else None,
                    'shared_branch_mlp': self.shared_branch_mlp.state_dict().copy() if self.shared_mlp else None,
                    'final_mlp': self.final_mlp.state_dict().copy(),
                    'fourier_features_list': [fourier_layer.state_dict().copy() for fourier_layer in self.fourier_features_list] if self.fourier_features_list is not None else None
                }
            elif val_dataloader is not None:
                patience_counter += 1
            
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
            
            # Early stopping (only if validation data provided)
            if val_dataloader is not None and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                if self.shared_mlp:
                    self.shared_branch_mlp.load_state_dict(self.best_state_dict['shared_branch_mlp'])
                else:
                    for i, branch in enumerate(self.mlp_branches):
                        branch.load_state_dict(self.best_state_dict['mlp_branches'][i])
                self.final_mlp.load_state_dict(self.best_state_dict['final_mlp'])
                if self.fourier_features_list is not None and self.best_state_dict['fourier_features_list'] is not None:
                    for i, fourier_layer in enumerate(self.fourier_features_list):
                        fourier_layer.load_state_dict(self.best_state_dict['fourier_features_list'][i])
                break

            if save_every_n_epochs is not None and ((epoch + 1) % save_every_n_epochs == 0 or (epoch+1) == epochs):
                checkpoint_dirname = os.path.dirname(checkpoint_path)
                if checkpoint_dirname:
                    os.makedirs(checkpoint_dirname, exist_ok=True)
                self._save_model_state(checkpoint_path)
        
        self.is_trained = True
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if val_dataloader is not None else []
        }
        
    def _forward_pass(self, points):
        """
        Internal method for forward pass through the parallel network.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features
            
        Returns:
        --------
        torch.Tensor
            Network output (probabilities)
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=self.device, dtype=torch.float32)
        else:
            points = points.float().to(self.device)  # Ensure float32 and correct device
        
        # Process through each parallel branch
        branch_outputs = []
        
        if self.shared_mlp:
            # Use shared MLP for all branches - process each branch separately
            for i in range(self.num_parallel_branches):
                branch_input = points
                
                # Apply Fourier feature mapping if enabled
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](branch_input)
                
                # Pad input to max dimension if necessary
                current_dim = branch_input.size(-1)
                if current_dim < self.max_fourier_dim:
                    padding_size = self.max_fourier_dim - current_dim
                    padding = torch.zeros(branch_input.size(0), padding_size, device=self.device, dtype=branch_input.dtype)
                    branch_input = torch.cat([branch_input, padding], dim=-1)
                
                # Forward pass through the shared MLP for this branch
                branch_output = self.shared_branch_mlp(branch_input)
                branch_outputs.append(branch_output)
        else:
            # Use separate MLPs for each branch (original behavior)
            for i in range(self.num_parallel_branches):
                branch_input = points
                
                # Apply Fourier feature mapping if enabled
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](branch_input)
                
                # Forward pass through branch MLP
                branch_output = self.mlp_branches[i](branch_input)
                branch_outputs.append(branch_output)
        
        # Concatenate all branch outputs
        concatenated_features = torch.cat(branch_outputs, dim=-1)
        
        # Final MLP with sigmoid activation
        final_output = self.final_mlp(concatenated_features)
        
        # Ensure output maintains batch dimension - squeeze only last dim if it's 1
        if final_output.dim() > 1 and final_output.shape[-1] == 1:
            return final_output.squeeze(-1)
        else:
            return final_output
    
    def __call__(self, points, return_probabilities=True):
        """
        Evaluate the trained LLR network on input points.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
        return_probabilities : bool
            If True, return probabilities (default behavior since network outputs probabilities)
            If False, return LLR values
            
        Returns:
        --------
        torch.Tensor
            Probabilities or LLR values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling. Use .train() method first.")
        
        if self.shared_mlp:
            self.shared_branch_mlp.eval()
        else:
            for branch in self.mlp_branches:
                branch.eval()
        self.final_mlp.eval()
        if self.fourier_features_list is not None:
            for fourier_layer in self.fourier_features_list:
                fourier_layer.eval()
            
      
        probabilities = self._forward_pass(points)
        
        if return_probabilities:
            return probabilities
        else:
            # Convert probabilities to LLR: log(p/(1-p))
            epsilon = 1e-7  # Small value to prevent log(0)
            prob_clamped = torch.clamp(probabilities, epsilon, 1 - epsilon)
            return torch.log(prob_clamped / (1 - prob_clamped))

    def evaluate_patd_likelihood(self, point, event_data, signal_surrogate_func,
                                 event_labels=['position', 'energy', 'zenith', 'azimuth'],
                                 use_rich_features=False, patd_result=None):
        """
        Evaluate joint log-likelihood for all photon hits at a detector position.

        Computes the sum of per-photon log-likelihood ratios log(p/(1-p)), which equals
        the joint log-LLR under the assumption of independent hits.

        Parameters
        ----------
        point : torch.Tensor or np.ndarray
            Detector point coordinates.
        event_data : dict
            Event parameters (hypothesis).
        signal_surrogate_func : callable
            Function to calculate PATD.  Called once as
            ``signal_surrogate_func(opt_point=point, event_params=event_data)``
            unless patd_result is provided.
        event_labels : list
            Event parameter keys passed to prepare_data_from_raw_patd.
            Ignored when use_rich_features=True.
        use_rich_features : bool
            If True, uses prepare_features_patd (14-feature geometry-rich builder).
            Must match the flag used during training.
        patd_result : dict or None
            Pre-computed surrogate result dict (with 'hit_times', 'num_photons',
            't_geom_min', ...).  When provided the surrogate is NOT called and
            these fixed photon times are used as the observation — event_data is
            used only for the hypothesis features.  This is the correct mode for
            NLL landscape evaluation where the observation must be held fixed while
            only the hypothesis parameters change.

        Returns
        -------
        dict with keys:
            'joint_log_likelihood' : scalar Tensor, sum of per-photon log-LLRs
            'num_photons'          : int
            'individual_llrs'      : Tensor of shape (num_photons,)
        """
        if not self.use_patd:
            raise ValueError("evaluate_patd_likelihood can only be used when use_patd=True")

        if isinstance(point, np.ndarray):
            point_t = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point_t = point.float().to(self.device)

        if use_rich_features:
            if patd_result is None:
                with torch.no_grad():
                    patd_result = signal_surrogate_func(opt_point=point_t, event_params=event_data)
            features_batch, num_photons = self.prepare_features_patd(
                point=point_t,
                event_data=event_data,
                patd_result=patd_result,
            )
        else:
            if patd_result is not None:
                features_batch, num_photons = self.prepare_data_from_raw_patd(
                    point=point_t,
                    event_data=event_data,
                    surrogate_func=lambda **kwargs: patd_result,
                    event_labels=event_labels,
                )
            else:
                features_batch, num_photons = self.prepare_data_from_raw_patd(
                    point=point_t,
                    event_data=event_data,
                    surrogate_func=signal_surrogate_func,
                    event_labels=event_labels,
                )

        if num_photons == 0 or features_batch is None:
            return {
                'joint_log_likelihood': torch.tensor(0.0, device=self.device),
                'num_photons': 0,
                'individual_llrs': torch.tensor([], device=self.device),
            }

        individual_llrs = self.predict_log_likelihood_ratio(features_batch)
        joint_log_likelihood = torch.sum(individual_llrs)

        return {
            'joint_log_likelihood': joint_log_likelihood,
            'num_photons': num_photons,
            'individual_llrs': individual_llrs,
        }
    
    def predict_log_likelihood_ratio(self, features, epsilon=1e-7):
        """
        Compute the Log-Likelihood Ratio using the sigmoid trick.
        
        This method computes log(p/(1-p)) where p is the output probability from the network.

    
        Parameters:
        -----------
        features : torch.Tensor
            Input features to evaluate
        epsilon : float
            Small value to prevent log(0)
        Returns:
        --------
        torch.Tensor
            Log-likelihood ratios
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling. Use .train() method first.")
        
        if self.shared_mlp:
            self.shared_branch_mlp.eval()
        else:
            for branch in self.mlp_branches:
                branch.eval()
        self.final_mlp.eval()
        if self.fourier_features_list is not None:
            for fourier_layer in self.fourier_features_list:
                fourier_layer.eval()
            
        
        # Get probabilities from the network (already has sigmoid)
        probabilities = self._forward_pass(features)
        
        # Compute LLR: log(p/(1-p))
        # epsilon = 1e-7  # Small value to prevent log(0)
        prob_clamped = torch.clamp(probabilities, epsilon, 1 - epsilon)
        llr = torch.log(prob_clamped / (1 - prob_clamped))
        # llr = torch.log(probabilities + 1e-10) - torch.log(1 - (probabilities + 1e-10))
        return llr
    
    def predict_likelihood_ratio(self, features):
        """
        Compute the Likelihood Ratio (not log).
        
        This method computes p/(1-p) where p is the sigmoid of the network output.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
            
        Returns:
        --------
        torch.Tensor
            Likelihood ratios
        """
        log_ratios = self.predict_log_likelihood_ratio(features)
        return torch.exp(log_ratios)
    
    def predict_proba(self, features):
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features
            
        Returns:
        --------
        torch.Tensor
            Probabilities of being signal (class 1)
        """
        return self.__call__(features, return_probabilities=True)
    
    def predict(self, features, threshold=0.5):
        """
        Get binary predictions.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features
        threshold : float
            Decision threshold for classification
            
        Returns:
        --------
        torch.Tensor
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(features)
        return (probabilities > threshold).float()
    
    def evaluate(self, features, labels, metrics=['accuracy', 'precision', 'recall', 'f1']):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        features : torch.Tensor
            Test features
        labels : torch.Tensor
            True labels
        metrics : list
            List of metrics to compute
            
        Returns:
        --------
        dict
            Dictionary of computed metrics
        """
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        results = {}
        
        # Convert to numpy for sklearn metrics
        y_true = labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities.cpu().numpy()
        
        if 'accuracy' in metrics:
            from sklearn.metrics import accuracy_score
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in metrics:
            from sklearn.metrics import precision_score
            results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        if 'recall' in metrics:
            from sklearn.metrics import recall_score
            results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        if 'f1' in metrics:
            from sklearn.metrics import f1_score
            results['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if 'auc' in metrics:
            from sklearn.metrics import roc_auc_score
            results['auc'] = roc_auc_score(y_true, y_prob)
        
        return results
    
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LLRnet Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _save_model_state(self, filepath):
        save_dict = {
            'mlp_branches_state_dict': [branch.state_dict() for branch in self.mlp_branches] if not self.shared_mlp else None,
            'shared_branch_mlp_state_dict': self.shared_branch_mlp.state_dict() if self.shared_mlp else None,
            'final_mlp_state_dict': self.final_mlp.state_dict(),
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'device': self.device,
            'dim': self.dim,
            'domain_size': self.domain_size,
            'use_fourier_features': self.use_fourier_features,
            'num_frequencies': self.num_frequencies,
            'frequency_scale': self.frequency_scale,
            'learnable_frequencies': self.learnable_frequencies,
            'num_parallel_branches': self.num_parallel_branches,
            'frequency_scales': self.frequency_scales,
            'num_frequencies_per_branch': self.num_frequencies_per_branch,
            'shared_mlp': self.shared_mlp,
            'reduce_lr_on_plateau': self.reduce_lr_on_plateau,
            'lr_scheduler_patience': self.lr_scheduler_patience,
            'lr_scheduler_factor': self.lr_scheduler_factor,
            'lr_scheduler_min_lr': self.lr_scheduler_min_lr,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }
        
        if self.fourier_features_list is not None:
            save_dict['fourier_features_list_state_dict'] = [fourier.state_dict() for fourier in self.fourier_features_list]
        else:
            save_dict['fourier_features_list_state_dict'] = None

        torch.save(save_dict, filepath)

    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")
        self._save_model_state(filepath)
    
    def load_model(self, filepath):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update model parameters
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # Load Fourier feature parameters if available (for backward compatibility)
        self.use_fourier_features = checkpoint.get('use_fourier_features', False)
        self.num_frequencies = checkpoint.get('num_frequencies', 64)
        self.frequency_scale = checkpoint.get('frequency_scale', 1.0)
        self.learnable_frequencies = checkpoint.get('learnable_frequencies', False)
        
        # Load parallel branch parameters (new format)
        self.num_parallel_branches = checkpoint.get('num_parallel_branches', 1)
        self.frequency_scales = checkpoint.get('frequency_scales', [self.frequency_scale])
        self.num_frequencies_per_branch = checkpoint.get('num_frequencies_per_branch', [self.num_frequencies])
        self.shared_mlp = checkpoint.get('shared_mlp', False)
        
        # Load learning rate scheduler parameters
        self.reduce_lr_on_plateau = checkpoint.get('reduce_lr_on_plateau', False)
        self.lr_scheduler_patience = checkpoint.get('lr_scheduler_patience', 10)
        self.lr_scheduler_factor = checkpoint.get('lr_scheduler_factor', 0.5)
        self.lr_scheduler_min_lr = checkpoint.get('lr_scheduler_min_lr', 1e-6)
        
        # Determine if this is old format (single MLP) or new format (parallel branches)
        is_old_format = 'model_state_dict' in checkpoint
        
        if is_old_format:
            # Convert old format to new format for backward compatibility
            print("Loading old format model - converting to single branch architecture")
            self.num_parallel_branches = 1
            self.frequency_scales = [self.frequency_scale]
            self.num_frequencies_per_branch = [self.num_frequencies]
            
            # Infer input_dim from old format
            if self.use_fourier_features and 'fourier_features_state_dict' in checkpoint:
                fourier_state = checkpoint['fourier_features_state_dict']
                if fourier_state is not None:
                    frequencies_shape = fourier_state['frequencies'].shape
                    input_dim = frequencies_shape[1]
                else:
                    first_layer_weight = checkpoint['model_state_dict']['0.weight']
                    fourier_output_dim = first_layer_weight.shape[1]
                    input_dim = fourier_output_dim // (2 * self.num_frequencies)
            else:
                first_layer_weight = checkpoint['model_state_dict']['0.weight']
                input_dim = first_layer_weight.shape[1]
            
            # Build network
            self._build_network(input_dim)
            
            # Load old states into first branch
            self.mlp_branches[0].load_state_dict(checkpoint['model_state_dict'])
            
            if self.fourier_features_list is not None and 'fourier_features_state_dict' in checkpoint:
                fourier_state = checkpoint['fourier_features_state_dict']
                if fourier_state is not None:
                    self.fourier_features_list[0].load_state_dict(fourier_state)
                    
        else:
            # New format with parallel branches
            # Infer input_dim from first branch's Fourier features
            if self.use_fourier_features and 'fourier_features_list_state_dict' in checkpoint:
                fourier_states = checkpoint['fourier_features_list_state_dict']
                if fourier_states is not None and len(fourier_states) > 0:
                    frequencies_shape = fourier_states[0]['frequencies'].shape
                    input_dim = frequencies_shape[1]
                else:
                    # Fallback: infer from first MLP branch
                    first_branch_state = checkpoint['mlp_branches_state_dict'][0]
                    first_layer_key = next(iter(first_branch_state.keys()))
                    if '0.weight' in first_layer_key:
                        input_dim = first_branch_state['0.weight'].shape[1]
                    else:
                        # Try different naming convention
                        for key in first_branch_state.keys():
                            if 'weight' in key and len(first_branch_state[key].shape) == 2:
                                input_dim = first_branch_state[key].shape[1]
                                break
            else:
                # No Fourier features, get input dim from first MLP branch
                first_branch_state = checkpoint['mlp_branches_state_dict'][0]
                first_layer_key = next(iter(first_branch_state.keys()))
                if '0.weight' in first_layer_key:
                    input_dim = first_branch_state['0.weight'].shape[1]
                else:
                    for key in first_branch_state.keys():
                        if 'weight' in key and len(first_branch_state[key].shape) == 2:
                            input_dim = first_branch_state[key].shape[1]
                            break
            
            # Build network
            self._build_network(input_dim)
            
            # Load parallel branch states
            if self.shared_mlp and 'shared_branch_mlp_state_dict' in checkpoint:
                # Load shared MLP state
                self.shared_branch_mlp.load_state_dict(checkpoint['shared_branch_mlp_state_dict'])
            elif not self.shared_mlp and 'mlp_branches_state_dict' in checkpoint:
                # Load separate branch states
                for i, branch_state in enumerate(checkpoint['mlp_branches_state_dict']):
                    self.mlp_branches[i].load_state_dict(branch_state)
            
            # Load final MLP state
            self.final_mlp.load_state_dict(checkpoint['final_mlp_state_dict'])
            
            # Load Fourier features if available
            if self.fourier_features_list is not None and 'fourier_features_list_state_dict' in checkpoint:
                fourier_states = checkpoint['fourier_features_list_state_dict']
                if fourier_states is not None:
                    for i, fourier_state in enumerate(fourier_states):
                        self.fourier_features_list[i].load_state_dict(fourier_state)
            
            # Load learning rate scheduler state if available
            if self.reduce_lr_on_plateau and self.lr_scheduler is not None:
                if 'lr_scheduler_state_dict' in checkpoint and checkpoint['lr_scheduler_state_dict'] is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.is_trained = True

    class EventDataset(Dataset):
        """
        Balanced PyTorch Dataset for signal/background events for LLR training.
        
        This dataset dynamically generates balanced signal and background events
        using ToySamplers and surrogate functions. It generates events on-demand
        while maintaining proper pairing for balanced training using a deterministic
        approach based on the sample index.
        
        This inner class has access to the parent LLRnet's prepare_data_from_raw method.
        """

        def __init__(self, llrnet_instance, signal_sampler, background_sampler, signal_surrogate_func, background_surrogate_func,
                     num_samples_per_epoch=1000, output_true_light_yield=False,
                     event_labels=['position', 'energy', 'zenith', 'azimuth']):
            """
            Initialize the EventDataset for balanced signal/background training.
            
            This dataset generates balanced pairs of signal and background events on-demand.
            Each pair uses the same detector point and shared event parameters,
            differing only in the detector response calculated by different surrogate functions.
            
            Parameters:
            -----------
            llrnet_instance : LLRnet
                Reference to the parent LLRnet instance to access prepare_data_from_raw
            signal_sampler : ToySampler
                Sampler instance for generating signal event parameters
            background_sampler : ToySampler
                Sampler instance for generating background event parameters
            signal_surrogate_func : callable
                Function to calculate light yield for signal events
            background_surrogate_func : callable  
                Function to calculate light yield for background events
            num_samples_per_epoch : int
                Total number of signal/background pairs to generate per epoch
            event_labels : list
                List of event parameter keys to include as features
            """
            
            self.llrnet = llrnet_instance
            self.signal_sampler = signal_sampler
            self.background_sampler = background_sampler
            self.signal_surrogate_func = signal_surrogate_func
            self.background_surrogate_func = background_surrogate_func
            self.num_samples_per_epoch = num_samples_per_epoch
            self.event_labels = event_labels
            self.output_true_light_yield = output_true_light_yield
            
            # Cache for the current epoch's generated pairs
            self.epoch_cache = {}
            # Track which events from each pair have been accessed
            self.pair_access_tracker = {}
            self.current_epoch_id = 0
            
        def _generate_pair_data(self, pair_idx):
            """
            Generate signal/background pair data for a specific pair index.
            
            This method generates both signal and background events for a given pair
            using the same detector point and event parameters.
            
            Parameters:
            -----------
            pair_idx : int
                The index of the pair to generate
                
            Returns:
            --------
            tuple : (signal_data, background_data)
                signal_data: (features, label) for signal event
                background_data: (features, label) for background event
            """
            # Use deterministic seeding based on pair index and epoch
            # This ensures reproducibility within an epoch while varying across epochs
          
        
    
            # Sample a random detector point for this event pair
            detector_point = self.signal_sampler.sample_detector_points(1).squeeze()
            
            # Generate signal event parameters (shared between signal and background)
            signal_event_data = self.signal_sampler.sample_events(1)[0]
            
            # Generate background event parameters
            background_event_data = self.background_sampler.sample_events(1)[0]
            
            # Create signal features using signal surrogate function
            if not self.output_true_light_yield:
                signal_features = self.llrnet.prepare_data_from_raw(
                    detector_point, signal_event_data, self.signal_surrogate_func, 
                    self.event_labels, self.llrnet.signal_noise_scale, self.llrnet.add_relative_pos
                )
                # Create background features using background surrogate function 
                # but with the signal event data (for balanced training)
                background_features = self.llrnet.prepare_data_from_raw(
                    detector_point, background_event_data, self.background_surrogate_func, 
                    self.event_labels, self.llrnet.background_noise_scale, self.llrnet.add_relative_pos, 
                    signal_event_data
                )
            else:
                signal_features, true_signal_light_yield = self.llrnet.prepare_data_from_raw(
                    detector_point, signal_event_data, self.signal_surrogate_func, 
                    self.event_labels, self.llrnet.signal_noise_scale, self.llrnet.add_relative_pos,
                    output_true_light_yield=True
                )
                # Create background features using background surrogate function 
                # but with the signal event data (for balanced training)
                background_features, true_background_light_yield = self.llrnet.prepare_data_from_raw(
                    detector_point, background_event_data, self.background_surrogate_func, 
                    self.event_labels, self.llrnet.background_noise_scale, self.llrnet.add_relative_pos, 
                    signal_event_data, output_true_light_yield=True
                )
            
            signal_label = torch.tensor(1.0, device=self.llrnet.device)
            background_label = torch.tensor(0.0, device=self.llrnet.device)
            if not self.output_true_light_yield:
                return (signal_features, signal_label), (background_features, background_label)
            else:
                return (signal_features, signal_label, true_signal_light_yield), (background_features, background_label, true_background_light_yield)

        def __len__(self):
            """Return the number of individual events per epoch (2 * num_samples_per_epoch)."""
            return self.num_samples_per_epoch * 2
        
        def __getitem__(self, idx):
            """
            Get individual signal or background event for balanced training.
            
            This method generates pairs on-demand and caches them for the current epoch.
            It uses deterministic seeding to ensure consistency within epochs while
            allowing variation across epochs.
            
            Parameters:
            -----------
            idx : int
                Sample index (even indices = signal, odd indices = background)
            
            Returns:
            --------
            tuple : (features, label)
                features: torch.Tensor of shape (feature_dim,) for individual event
                label: torch.Tensor scalar (1.0 for signal, 0.0 for background)
            """
            # Detect new epoch when idx resets to 0
            if idx == 0:
                self.current_epoch_id += 1
                self.epoch_cache.clear()
                self.pair_access_tracker.clear()
            
            # Determine which pair this event belongs to and whether it's signal or background
            pair_idx = idx // 2
            is_signal = (idx % 2 == 0)
            
            # Check if we already have this pair cached
            if pair_idx not in self.epoch_cache:
                # Generate the pair data
                signal_data, background_data = self._generate_pair_data(pair_idx)
                self.epoch_cache[pair_idx] = (signal_data, background_data)
                # Initialize access tracker for this pair
                self.pair_access_tracker[pair_idx] = {'signal_accessed': False, 'background_accessed': False}
            
            # Get the appropriate event from the cached pair
            if is_signal:
                result = self.epoch_cache[pair_idx][0]  # signal data
                self.pair_access_tracker[pair_idx]['signal_accessed'] = True
            else:
                result = self.epoch_cache[pair_idx][1]  # background data
                self.pair_access_tracker[pair_idx]['background_accessed'] = True
            
            # Check if both signal and background have been accessed for this pair
            if (self.pair_access_tracker[pair_idx]['signal_accessed'] and 
                self.pair_access_tracker[pair_idx]['background_accessed']):
                # Both events accessed, clear from cache to free memory
                del self.epoch_cache[pair_idx]
                del self.pair_access_tracker[pair_idx]
            
            return result
        
    
    class SignalOnlyDataset(Dataset):
        """
        Balanced PyTorch Dataset for signal-only LLR training with matched/mismatched light yields.
        
        This dataset uses only signal events but trains the network to distinguish between:
        - Class 1 (matched): Signal event parameters with their corresponding light yield
        - Class 0 (mismatched): Signal event parameters with light yield from a different event
        
        This approach trains the network to learn whether the light yield is consistent with
        the given event parameters, which can be useful for detecting anomalies or verifying
        event reconstruction quality.
        
        Similar to EventDataset, this generates pairs on-demand and caches them for the epoch.
        Each pair consists of:
        - Matched event: same signal event used for both parameters and light yield (label=1)
        - Mismatched event: parameters from one event, light yield from different event (label=0)
        """

        def __init__(self, llrnet_instance, signal_sampler, signal_surrogate_func,
                     num_samples_per_epoch=1000, output_true_light_yield=False,
                     event_labels=['position', 'energy', 'zenith', 'azimuth'],
                     **kwargs):
            """
            Initialize the SignalOnlyDataset for matched/mismatched training.
            
            This dataset generates balanced pairs where one has matched parameters and light yield,
            and the other has mismatched parameters and light yield.
            
            Parameters:
            -----------
            llrnet_instance : LLRnet
                Reference to the parent LLRnet instance to access prepare_data_from_raw
            signal_sampler : ToySampler
                Sampler instance for generating signal event parameters
            signal_surrogate_func : callable
                Function to calculate light yield for signal events
            num_samples_per_epoch : int
                Total number of matched/mismatched pairs to generate per epoch
            output_true_light_yield : bool
                Whether to output the true light yield (for debugging/analysis)
            event_labels : list
                List of event parameter keys to include as features
            **kwargs:
                presampled_events : list, optional
                    Pre-sampled event parameters to reuse
                presampled_detector_points : torch.Tensor, optional
                    Pre-sampled detector points to reuse
                min_light_yield : float, optional
                    Minimum light yield threshold for resampling
                max_resample_attempts : int, optional
                    Maximum number of resampling attempts
                samples_per_event : int, optional
                    Number of times to sample from the same detector point and event.
                    Useful when surrogate_func includes randomness (e.g., Poisson sampling).
                    Default is 1 (no resampling). If > 1, each unique event/detector pair
                    will be sampled multiple times with different random realizations.
                use_rich_features : bool, optional (default False)
                    If True, uses prepare_features_charge instead of prepare_data_from_raw.
                    Produces a 13-feature vector per event with richer geometry:
                    det(3) + vertex(3) + dir(3) + log_E(1) + vert_dist(1) + cos_angle(1)
                    + log_ly(1).  event_labels is ignored in this mode.
            """

            self.llrnet = llrnet_instance
            self.signal_sampler = signal_sampler
            self.signal_surrogate_func = signal_surrogate_func
            self.num_samples_per_epoch = num_samples_per_epoch
            self.event_labels = event_labels
            self.output_true_light_yield = output_true_light_yield
            self.presampled_events = kwargs.get('presampled_events', None)
            self.presampled_detector_points = kwargs.get('presampled_detector_points', None)
            self.samples_per_event = kwargs.get('samples_per_event', 1)
            self.vary_cylinder = kwargs.get('vary_cylinder', False)
            self.cylinder_sampler = kwargs.get('cylinder_sampler', None)
            self.use_rich_features = kwargs.get('use_rich_features', False)
            self.domain_size = llrnet_instance.domain_size
            # Resampling configuration: discard near-zero light yield pairs (uninformative)
            # min_light_yield: if provided, pairs where BOTH matched & mismatched light yields have
            #                  mean absolute value below this threshold are resampled.
            # max_resample_attempts: cap on resampling attempts to avoid infinite loops.
            self.min_light_yield = kwargs.get('min_light_yield', None)
            self.max_resample_attempts = kwargs.get('max_resample_attempts', 10)
            # Cache for the current epoch's generated pairs
            self.epoch_cache = {}
            # Track which events from each pair have been accessed
            self.pair_access_tracker = {}
            self.current_epoch_id = 0
            
        def _generate_pair_data(self, pair_idx):
            """
            Generate matched/mismatched pair data for a specific pair index.
            
            If samples_per_event > 1, generates multiple samples from the same
            detector point and event parameters with different random realizations
            from the surrogate function.
            
            Returns:
            --------
            For samples_per_event=1:
                tuple: ((matched_features, matched_label), (mismatched_features, mismatched_label))
            
            For samples_per_event>1:
                list of tuples: [((matched_features_1, matched_label), (mismatched_features_1, mismatched_label)),
                                 ((matched_features_2, matched_label), (mismatched_features_2, mismatched_label)),
                                 ...]
            """
            attempt = 0
            
            if self.vary_cylinder and self.cylinder_sampler is not None:
                # Sample a new cylinder configuration
                # new height and radius
                new_heights = torch.linspace(self.domain_size * 0.01, self.domain_size, steps=100)
                new_radii = torch.linspace(self.domain_size * 0.01/2, self.domain_size / 2, steps=100)
                height = new_heights[torch.randint(0, len(new_heights), (1,)).item()]
                radius = new_radii[torch.randint(0, len(new_radii), (1,)).item()]
                self.signal_sampler = self.cylinder_sampler(cylinder_height=height.item(), cylinder_radius=radius.item(), domain_size=self.domain_size, E_min=1e2, E_max=1e8, energy_dist='log_uniform', find_exact_intersection=True)
            if self.presampled_detector_points is not None:
                det_indx = np.random.randint(0, len(self.presampled_detector_points))
                detector_point = self.presampled_detector_points[det_indx]
            else:
                detector_point = self.signal_sampler.sample_detector_points(1).squeeze()
            while True:
                # Sample two different signal events
                if self.presampled_events is not None:
                    event_indices = torch.randperm(len(self.presampled_events))
                    signal_event_data_1 = self.presampled_events[event_indices[0]]
                    signal_event_data_2 = self.presampled_events[event_indices[1]]
                else:
                    signal_event_data_1 = self.signal_sampler.sample_events(1)[0]
                    signal_event_data_2 = self.signal_sampler.sample_events(1)[0]

                # Deep copies (avoid in-place tensor mutation side effects)
                def deep_copy_event(ev):
                    new = {}
                    for k, v in ev.items():
                        if torch.is_tensor(v):
                            new[k] = v.clone()
                        else:
                            new[k] = v
                    return new

                event_for_params_matched = deep_copy_event(signal_event_data_1)
                event_for_light_yield_matched = deep_copy_event(signal_event_data_1)

                # MISMATCHED: parameters from event1, light yield from event2 (DO NOT overwrite)
                event_for_params_mismatched = deep_copy_event(signal_event_data_1)   # keep Θ from event1
                event_for_light_yield_mismatched = deep_copy_event(signal_event_data_2)  # use LY from event2

                # If resampling is enabled, evaluate light yields and decide if informative
                if self.min_light_yield is not None:
                    with torch.no_grad():
                        matched_ly = self.signal_surrogate_func(opt_point=detector_point, event_params=event_for_light_yield_matched)
                        mismatched_ly = self.signal_surrogate_func(opt_point=detector_point, event_params=event_for_light_yield_mismatched)
                        # Use mean absolute value as magnitude metric
                        matched_mag = torch.mean(torch.abs(matched_ly)).item() if torch.is_tensor(matched_ly) else float(np.mean(np.abs(matched_ly)))
                        mismatched_mag = torch.mean(torch.abs(mismatched_ly)).item() if torch.is_tensor(mismatched_ly) else float(np.mean(np.abs(mismatched_ly)))
                    # Resample only if BOTH are below threshold (uninformative contrast)
                    if matched_mag < self.min_light_yield and mismatched_mag < self.min_light_yield and attempt < self.max_resample_attempts:
                        attempt += 1
                        # if attempt >= self.max_resample_attempts:
                        #     print(f"Max resample attempts reached")
                        continue  # try again
                break  # either informative or resampling disabled / attempts exhausted

            matched_label = torch.tensor(1.0, device=self.llrnet.device)
            mismatched_label = torch.tensor(0.0, device=self.llrnet.device)

            def _build_pair(event_for_params_m, event_for_ly_m,
                            event_for_params_mm, event_for_ly_mm):
                """Return (matched_features, matched_true_ly, mismatched_features, mismatched_true_ly).
                matched_true_ly / mismatched_true_ly are None when not requested."""
                if self.use_rich_features:
                    with torch.no_grad():
                        ly_m = self.signal_surrogate_func(
                            opt_point=detector_point, event_params=event_for_ly_m
                        )
                        ly_mm = self.signal_surrogate_func(
                            opt_point=detector_point, event_params=event_for_ly_mm
                        )
                    f_m = self.llrnet.prepare_features_charge(
                        detector_point, event_for_params_m, ly_m
                    )
                    f_mm = self.llrnet.prepare_features_charge(
                        detector_point, event_for_params_mm, ly_mm
                    )
                    true_ly_m = ly_m if self.output_true_light_yield else None
                    true_ly_mm = ly_mm if self.output_true_light_yield else None
                else:
                    if self.output_true_light_yield:
                        f_m, true_ly_m = self.llrnet.prepare_data_from_raw(
                            detector_point, event_for_ly_m, self.signal_surrogate_func,
                            self.event_labels, self.llrnet.signal_noise_scale,
                            event_for_params_m, output_true_light_yield=True,
                        )
                        f_mm, true_ly_mm = self.llrnet.prepare_data_from_raw(
                            detector_point, event_for_ly_mm, self.signal_surrogate_func,
                            self.event_labels, self.llrnet.signal_noise_scale,
                            event_for_params_mm, output_true_light_yield=True,
                        )
                    else:
                        f_m = self.llrnet.prepare_data_from_raw(
                            detector_point, event_for_ly_m, self.signal_surrogate_func,
                            self.event_labels, self.llrnet.signal_noise_scale,
                            event_for_params_m,
                        )
                        f_mm = self.llrnet.prepare_data_from_raw(
                            detector_point, event_for_ly_mm, self.signal_surrogate_func,
                            self.event_labels, self.llrnet.signal_noise_scale,
                            event_for_params_mm,
                        )
                        true_ly_m = true_ly_mm = None
                return f_m, true_ly_m, f_mm, true_ly_mm

            if self.samples_per_event > 1:
                all_samples = []
                for _ in range(self.samples_per_event):
                    f_m, tly_m, f_mm, tly_mm = _build_pair(
                        event_for_params_matched, event_for_light_yield_matched,
                        event_for_params_mismatched, event_for_light_yield_mismatched,
                    )
                    if self.output_true_light_yield:
                        all_samples.append((
                            (f_m, matched_label, tly_m),
                            (f_mm, mismatched_label, tly_mm),
                        ))
                    else:
                        all_samples.append((
                            (f_m, matched_label),
                            (f_mm, mismatched_label),
                        ))
                return all_samples
            else:
                f_m, tly_m, f_mm, tly_mm = _build_pair(
                    event_for_params_matched, event_for_light_yield_matched,
                    event_for_params_mismatched, event_for_light_yield_mismatched,
                )
                if self.output_true_light_yield:
                    return (f_m, matched_label, tly_m), (f_mm, mismatched_label, tly_mm)
                else:
                    return (f_m, matched_label), (f_mm, mismatched_label)
        
        def __len__(self):
            """Return the number of individual events per epoch (2 * num_samples_per_epoch * samples_per_event)."""
            return self.num_samples_per_epoch * 2 * self.samples_per_event
        
        def __getitem__(self, idx):
            """
            Get individual matched or mismatched event for balanced training.
            
            This method generates pairs on-demand and caches them for the current epoch.
            When samples_per_event > 1, multiple samples are generated from the same
            detector point and event parameters.
            
            Parameters:
            -----------
            idx : int
                Sample index (even indices = matched, odd indices = mismatched)
            
            Returns:
            --------
            tuple : (features, label)
                features: torch.Tensor of shape (feature_dim,) for individual event
                label: torch.Tensor scalar (1.0 for matched, 0.0 for mismatched)
            """
            # Detect new epoch when idx resets to 0
            if idx == 0:
                self.epoch_cache.clear()
                self.pair_access_tracker.clear()
                self.current_epoch_id += 1
            
            # Determine which base pair and which sample within that pair
            samples_per_pair = 2 * self.samples_per_event
            pair_idx = idx // samples_per_pair
            idx_within_pair = idx % samples_per_pair
            
            # Determine which sample iteration and whether it's matched or mismatched
            sample_iteration = idx_within_pair // 2
            is_matched = (idx_within_pair % 2 == 0)
            
            # Check if we already have this pair cached
            if pair_idx not in self.epoch_cache:
                # Generate new pair(s) and cache it/them
                pair_data = self._generate_pair_data(pair_idx)
                self.epoch_cache[pair_idx] = pair_data
                # Initialize access tracker based on samples_per_event
                if self.samples_per_event > 1:
                    self.pair_access_tracker[pair_idx] = {
                        'accessed_count': 0,
                        'total_accesses': samples_per_pair
                    }
                else:
                    self.pair_access_tracker[pair_idx] = {'matched_accessed': False, 'mismatched_accessed': False}
            
            # Get the appropriate event from the cached pair
            if self.samples_per_event > 1:
                # Multiple samples per event - pair_data is a list of tuples
                sample_pair = self.epoch_cache[pair_idx][sample_iteration]
                if is_matched:
                    result = sample_pair[0]  # First element is matched
                else:
                    result = sample_pair[1]  # Second element is mismatched
                
                # Track access count
                self.pair_access_tracker[pair_idx]['accessed_count'] += 1
                
                # Check if all samples have been accessed
                if self.pair_access_tracker[pair_idx]['accessed_count'] >= self.pair_access_tracker[pair_idx]['total_accesses']:
                    # All samples from this pair have been used, can delete from cache
                    del self.epoch_cache[pair_idx]
                    del self.pair_access_tracker[pair_idx]
            else:
                # Original behavior for samples_per_event=1
                if is_matched:
                    result = self.epoch_cache[pair_idx][0]  # First element is matched
                    self.pair_access_tracker[pair_idx]['matched_accessed'] = True
                else:
                    result = self.epoch_cache[pair_idx][1]  # Second element is mismatched
                    self.pair_access_tracker[pair_idx]['mismatched_accessed'] = True
                
                # Check if both matched and mismatched have been accessed for this pair
                if (self.pair_access_tracker[pair_idx]['matched_accessed'] and 
                    self.pair_access_tracker[pair_idx]['mismatched_accessed']):
                    # Both events from this pair have been used, can delete from cache
                    del self.epoch_cache[pair_idx]
                    del self.pair_access_tracker[pair_idx]
            
            return result
    
    
    class PATDDataset(IterableDataset):
        """
        Iterable Dataset for training with Photon Arrival Time Distributions (PATD).
        
        Similar to SignalOnlyDataset but for PATD mode. Can operate in two modes:
        
        1. Event mode (shuffle_photons=False, default): Returns ALL photon features from one event at once.
           Each iteration yields (features_batch, labels) where features_batch has shape (num_photons, feature_dim).
           
        2. Photon mode (shuffle_photons=True): Returns individual photons one at a time, shuffling them across batches.
           Each iteration yields (features, label) for a single photon, with photons from different events mixed.
        
        For matched samples (label=1), both the hypothesis and observation come from the same event.
        For mismatched samples (label=0), hypothesis from one event, observation from different event.
        
        As an IterableDataset, this works seamlessly with DataLoader multiprocessing - each worker
        generates its own independent stream of data.
        """
        
        def __init__(self, signal_sampler, signal_surrogate_func, llrnet_instance,
                     num_samples_per_epoch=1000, event_labels=['position', 'energy', 'zenith', 'azimuth'],
                     min_photons=1, num_photons_per_sample=None, shuffle_photons=False, **kwargs):
            """
            Initialize PATD iterable dataset.
            
            Parameters:
            -----------
            signal_sampler : Sampler
                Sampler for generating signal events
            signal_surrogate_func : callable
                Function to calculate PATD (returns dict with 'hit_times', 'num_photons')
            llrnet_instance : LLRnet
                LLRnet instance for accessing prepare_data_from_raw
            num_samples_per_epoch : int
                Number of matched/mismatched pairs to generate per epoch.
                With multiple workers, each worker generates this many pairs.
            event_labels : list
                List of event parameter keys
            min_photons : int
                Minimum number of photons required for an event to be valid
            num_photons_per_sample : int or None
                Maximum number of photons to use per event (for capping).
                If None, uses all photons from the event.
            shuffle_photons : bool
                If False (default): Returns all photons from one event together (event mode)
                If True: Returns individual photons one at a time, shuffled across batches (photon mode)
            **kwargs:
                vary_cylinder : bool, optional
                    If True, randomly vary cylinder dimensions for each pair
                cylinder_sampler : callable, optional
                    Function to create new sampler with varied cylinder dimensions
                use_rich_features : bool, optional (default False)
                    If True, uses prepare_features_patd instead of prepare_data_from_raw_patd.
                    This produces a 14-feature vector per photon with richer geometry:
                    det(3) + vertex(3) + dir(3) + log_E(1) + vert_dist(1) + cos_angle(1)
                    + t_geom_min(1) + t_hit_scaled(1).
                    Requires the surrogate to return 't_geom_min' in its result dict.
                    event_labels is ignored in this mode.
            """
            super().__init__()
            self.signal_sampler = signal_sampler
            self.signal_surrogate_func = signal_surrogate_func
            self.llrnet = llrnet_instance
            self.num_samples_per_epoch = num_samples_per_epoch
            self.event_labels = event_labels
            self.min_photons = min_photons
            self.num_photons_per_sample = num_photons_per_sample
            self.shuffle_photons = shuffle_photons
            self.vary_cylinder = kwargs.get('vary_cylinder', False)
            self.cylinder_sampler = kwargs.get('cylinder_sampler', None)
            self.use_rich_features = kwargs.get('use_rich_features', False)
            self.domain_size = llrnet_instance.domain_size
        
        def _generate_event_with_photons(self, detector_pos=None):
            """Generate an event that produces at least min_photons hits."""
            while True:
                # Sample event and detector position
                event_params = self.signal_sampler.sample_events(1)[0]
                if detector_pos is None:
                    detector_pos = self.signal_sampler.sample_detector_points(1)
                
                # Get PATD - pass num_photons_per_sample to limit photons if specified
                patd_result = self.signal_surrogate_func(
                    opt_point=detector_pos, 
                    event_params=event_params,
                    max_photons=self.num_photons_per_sample
                )
                num_photons = patd_result['num_photons']
                
                if num_photons >= self.min_photons:
                    return event_params, detector_pos, patd_result
                
        def _generate_pair_data(self, pair_idx):
            """
            Generate matched/mismatched pair data for PATD events.
            
            Similar to SignalOnlyDataset but returns ALL photon features from each event.
            
            Returns:
            --------
            tuple: ((matched_features_batch, matched_labels), (mismatched_features_batch, mismatched_labels))
                   where features_batch has shape (num_photons, feature_dim)
                   and labels has shape (num_photons,) with all same value
            """
            
            if self.vary_cylinder and self.cylinder_sampler is not None:
                # Sample a new cylinder configuration
                # new height and radius
                new_heights = torch.linspace(self.domain_size * 0.01, self.domain_size, steps=100)
                new_radii = torch.linspace(self.domain_size * 0.01/2, self.domain_size / 2, steps=100)
                height = new_heights[torch.randint(0, len(new_heights), (1,)).item()]
                radius = new_radii[torch.randint(0, len(new_radii), (1,)).item()]
                self.signal_sampler = self.cylinder_sampler(cylinder_height=height.item(), cylinder_radius=radius.item(), domain_size=self.domain_size, E_min=1e2, E_max=1e8, energy_dist='log_uniform', find_exact_intersection=True)
            
            # Sample detector point (shared for this pair)
            detector_point = self.signal_sampler.sample_detector_points(1).squeeze()
            
            # Generate MATCHED sample (hypothesis and observation from same event)
            event_params_matched, _, patd_result_matched = self._generate_event_with_photons(detector_point)
            
            # Create feature vectors for all photon hits
            if self.use_rich_features:
                matched_features_batch, _ = self.llrnet.prepare_features_patd(
                    point=detector_point,
                    event_data=event_params_matched,
                    patd_result=patd_result_matched,
                )
            else:
                matched_features_batch, _ = self.llrnet.prepare_data_from_raw_patd(
                    point=detector_point,
                    event_data=event_params_matched,
                    surrogate_func=lambda **kwargs: patd_result_matched,
                    event_labels=self.event_labels,
                )

            # Create labels for all matched photons (all are class 1)
            num_matched_photons = matched_features_batch.shape[0]
            matched_labels = torch.ones(num_matched_photons, dtype=torch.float32, device=self.llrnet.device)

            # Generate MISMATCHED sample: observation from a different event, hypothesis from matched event.
            # event_params_obs, _, patd_result_obs = self._generate_event_with_photons(detector_point)

            # if self.use_rich_features:
            #     # Hypothesis features come from event_params_matched; observation times from patd_result_obs.
            #     # prepare_features_patd only takes one event_data dict, so we build a merged view:
            #     # keep all params from matched (hypothesis) but swap in the obs hit_times via patd_result_obs.
            #     mismatched_features_batch, _ = self.llrnet.prepare_features_patd(
            #         point=detector_point,
            #         event_data=event_params_matched,   # hypothesis: position, energy, direction from matched
            #         patd_result=patd_result_obs,        # observation: photon times from a different event
            #     )
            # else:
            #     # event_data drives the surrogate call (provides photon times via patd_result_obs),
            #     # signal_event_data provides the hypothesis features (position, energy, direction).
            #     mismatched_features_batch, _ = self.llrnet.prepare_data_from_raw_patd(
            #         point=detector_point,
            #         event_data=event_params_obs,            # observation: photon times from a different event
            #         surrogate_func=lambda **kwargs: patd_result_obs,
            #         signal_event_data=event_params_matched, # hypothesis: parameters from matched event
            #         event_labels=self.event_labels,
            #     )
            event_params_hypothesis_mismatch = self.signal_sampler.sample_events(1)[0]

            if self.use_rich_features:
                mismatched_features_batch, _ = self.llrnet.prepare_features_patd(
                    point=detector_point,
                    event_data=event_params_hypothesis_mismatch,  # hypothesis: parameters from different event
                    patd_result=patd_result_matched,              # observation: same hit times as matched
                )
            else:
                mismatched_features_batch, _ = self.llrnet.prepare_data_from_raw_patd(
                    point=detector_point,
                    event_data=event_params_matched,               # observation: same hit times as matched
                    surrogate_func=lambda **kwargs: patd_result_matched,
                    signal_event_data=event_params_hypothesis_mismatch,  # hypothesis: from different event
                    event_labels=self.event_labels,
                )
            # Create labels for all mismatched photons (all are class 0)
            num_mismatched_photons = mismatched_features_batch.shape[0]
            mismatched_labels = torch.zeros(num_mismatched_photons, dtype=torch.float32, device=self.llrnet.device)
            
            # Clone and detach to ensure clean tensors without computational graph
            return ((matched_features_batch.clone().detach(), matched_labels), 
                    (mismatched_features_batch.clone().detach(), mismatched_labels))
        
        def __iter__(self):
            """
            Generate photon features and labels as an iterator.
            
            Yields:
            -------
            If shuffle_photons=False (event mode):
                (features_batch, labels) : tuple
                    features_batch : torch.Tensor, shape (num_photons, feature_dim)
                    labels : torch.Tensor, shape (num_photons,)
            
            If shuffle_photons=True (photon mode):
                (features, label) : tuple
                    features : torch.Tensor, shape (feature_dim,)
                    label : torch.Tensor, scalar
            """
            # Get worker info for multiprocessing
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                # Single-process data loading
                num_workers = 1
                worker_id = 0
            else:
                # Multi-process data loading
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
            
            # Each worker generates its share of pairs
            pairs_per_worker = self.num_samples_per_epoch // num_workers
            if worker_id < self.num_samples_per_epoch % num_workers:
                pairs_per_worker += 1
            
            with torch.no_grad():
                if self.shuffle_photons:
                    # Photon mode: build separate pools for matched and mismatched photons
                    matched_pool = []
                    mismatched_pool = []
                    
                    for pair_idx in range(pairs_per_worker):
                        pair_data = self._generate_pair_data(pair_idx)
                        
                        # Extract matched photons
                        matched_features_batch, matched_labels = pair_data[0]
                        for i in range(matched_features_batch.shape[0]):
                            matched_pool.append((
                                matched_features_batch[i].clone().detach(),
                                matched_labels[i]
                            ))
                        
                        # Extract mismatched photons
                        mismatched_features_batch, mismatched_labels = pair_data[1]
                        for i in range(mismatched_features_batch.shape[0]):
                            mismatched_pool.append((
                                mismatched_features_batch[i].clone().detach(),
                                mismatched_labels[i]
                            ))
                    random.shuffle(matched_pool)
                    random.shuffle(mismatched_pool)
                    # Truncate both pools to same length to ensure perfect balance
                    min_length = min(len(matched_pool), len(mismatched_pool))
                    matched_pool = matched_pool[:min_length]
                    mismatched_pool = mismatched_pool[:min_length]
                    
                    # Shuffle each pool independently
                    
                    
                    # Alternate between matched and mismatched to ensure each batch is balanced
                    for matched_photon, mismatched_photon in zip(matched_pool, mismatched_pool):
                        yield matched_photon
                        yield mismatched_photon
                        
                else:
                    # Event mode: yield all photons from one event together
                    for pair_idx in range(pairs_per_worker):
                        pair_data = self._generate_pair_data(pair_idx)
                        
                        # Yield matched event (all photons together)
                        yield pair_data[0]
                        
                        # Yield mismatched event (all photons together)
                        yield pair_data[1]
    
    
    class BatchSignalOnlyDataset(Dataset):
        """
        Efficient dataset that generates batches of Joint and Marginal samples on the fly.
        
        Strategy:
        1. Sample N event parameters (Theta).
        2. Sample N detector points.
        3. Calculate N observations (x) where x_i ~ p(x | Theta_i).
        4. Create Joint Batch (Class 1): Pairs (x_i, Theta_i).
        5. Create Marginal Batch (Class 0): Pairs (x_i, Theta_j) where i != j (shuffled).
        """
        def __init__(self, llrnet, signal_sampler, signal_surrogate_func, 
                     num_batches=100, batch_size=128, 
                     event_labels=['position', 'energy', 'zenith', 'azimuth']):
            self.llrnet = llrnet
            self.signal_sampler = signal_sampler
            self.signal_surrogate_func = signal_surrogate_func
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.event_labels = event_labels
            # Use CPU for data generation to support multi-worker loading
            self.device = 'cpu'

        def __len__(self):
            return self.num_batches

        def __getitem__(self, idx):
            # 1. Sample Batch of Parameters (Theta) and Points
            # ------------------------------------------------
            # Assuming sampler returns dict of tensors
            event_params_list = self.signal_sampler.sample_events(self.batch_size)
            
            # Collate list of dicts into dict of tensors
            event_params = {}
            if len(event_params_list) > 0:
                first_event = event_params_list[0]
                keys = first_event.keys()
                for key in keys:
                    # Stack tensors
                    val = first_event[key]
                    if torch.is_tensor(val):
                        # Handle scalar tensors vs 1D tensors
                        stacked = torch.stack([e[key] for e in event_params_list])
                        if stacked.dim() > 1 and stacked.shape[1] == 1:
                             stacked = stacked.squeeze(1)
                        event_params[key] = stacked.to(self.device)
                    elif isinstance(val, np.ndarray):
                        stacked = np.stack([e[key] for e in event_params_list])
                        event_params[key] = torch.tensor(stacked, device=self.device).float()
                        if event_params[key].dim() > 1 and event_params[key].shape[1] == 1:
                             event_params[key] = event_params[key].squeeze(1)
                    else:
                        event_params[key] = torch.tensor([e[key] for e in event_params_list], device=self.device).float()
            
            points = self.signal_sampler.sample_detector_points(self.batch_size)
            if isinstance(points, np.ndarray):
                points = torch.tensor(points, device=self.device).float()
            else:
                points = points.to(self.device).float()
            
            # 2. Generate Observations (x) - The "True" Data
            # ----------------------------------------------
            # We generate x from Theta. This is the "Joint" relationship.
            # We use the LLRnet helper to get the features for the Joint case (Class 1)
            # prepare_data_from_raw handles the surrogate call + noise + feature assembly
            joint_features = self.llrnet.prepare_data_from_raw(
                point=points,
                event_data=event_params,
                surrogate_func=self.signal_surrogate_func,
                event_labels=self.event_labels,
                noise_scale=self.llrnet.signal_noise_scale,
                add_relative_pos=self.llrnet.add_relative_pos,
                signal_event_data=None, # This triggers generation of x from event_params
                device=self.device
            )
            
            # 3. Create Marginal Samples (Class 0) via Shuffling
            # --------------------------------------------------
            # We want pairs (x_i, Theta_j).
            # We already have x_i embedded in joint_features.
            # However, prepare_data_from_raw combines x and Theta into one tensor.
            # We need to extract x, shuffle Theta, and recombine.
            
            # To do this efficiently, we first get the raw observation values (Light Yield)
            # We can ask prepare_data_from_raw to return the raw LY used.
            _, observed_ly = self.llrnet.prepare_data_from_raw(
                point=points,
                event_data=event_params,
                surrogate_func=self.signal_surrogate_func,
                event_labels=self.event_labels,
                noise_scale=self.llrnet.signal_noise_scale,
                add_relative_pos=self.llrnet.add_relative_pos,
                signal_event_data=None,
                output_true_light_yield=True,
                device=self.device
            )
            
            # Shuffle the parameters (Theta)
            # We roll the tensors by 1 to ensure no i == j matches
            shuffled_params = {}
            for key, val in event_params.items():
                if torch.is_tensor(val):
                    shuffled_params[key] = torch.roll(val, shifts=1, dims=0)
                else:
                    # Handle list/numpy if necessary, though tensors preferred

                    pass
            
            # We also need to shuffle the points if they are considered part of the "event setup"
            # usually points are fixed per observation.
            # In the marginal case: We have observation x_i (at point_i).
            # We pair it with hypothesis Theta_j.
            # The geometric features must be calculated between point_i and Theta_j.
            
            marginal_features = self.llrnet.prepare_data_from_raw(
                point=points, # Keep points aligned with observation x_i
                event_data=shuffled_params, # Use shuffled hypothesis
                surrogate_func=self.signal_surrogate_func,
                event_labels=self.event_labels,
                noise_scale=0.0, # No noise here, we are just building features
                add_relative_pos=self.llrnet.add_relative_pos,
                signal_event_data=observed_ly, # FORCE the observation to be x_i
                device=self.device
            )
            
            # 4. Combine and Label
            # --------------------
            X = torch.cat([joint_features, marginal_features], dim=0)
            y = torch.cat([torch.ones(self.batch_size, device=self.device), torch.zeros(self.batch_size, device=self.device)], dim=0)
            
            return X, y


    def create_patd_dataloader(self, signal_sampler, signal_surrogate_func,
                              num_samples_per_epoch=1000, batch_size=32,
                              num_workers=0,
                              event_labels=['position', 'energy', 'zenith', 'azimuth'],
                              shuffle_photons=False, use_rich_features=False, other_kwargs={}):
        """
        Create a DataLoader for PATD training using IterableDataset.
        
        Can operate in two modes:
        1. Event mode (shuffle_photons=False): Each sample returns all photons from one event
        2. Photon mode (shuffle_photons=True): Each sample returns individual photons, shuffled across events
        
        Note: As an IterableDataset, shuffling is handled internally for photon mode.
        Data generation is distributed across workers automatically.
        
        Parameters:
        -----------
        signal_sampler : Sampler
            Sampler for generating signal events
        signal_surrogate_func : callable
            Function to calculate PATD
        num_samples_per_epoch : int
            Number of matched/mismatched pairs to generate per epoch.
            Each worker will generate approximately num_samples_per_epoch / num_workers pairs.
        batch_size : int
            If shuffle_photons=False: Number of events per batch
            If shuffle_photons=True: Number of individual photons per batch
        num_workers : int
            Number of worker processes for data loading. Each worker generates
            its own independent stream of data.
        event_labels : list
            List of event parameter keys
        shuffle_photons : bool
            If False (default): Event mode - returns all photons from one event together
            If True: Photon mode - returns individual photons shuffled across batches
            
        Returns:
        --------
        DataLoader
            PyTorch DataLoader for PATD training
            
            Event mode (shuffle_photons=False):
            - features: (total_photons_in_batch, feature_dim)
            - labels: (total_photons_in_batch,)
            where total_photons_in_batch is sum of photons from all events in batch
            
            Photon mode (shuffle_photons=True):
            - features: (batch_size, feature_dim)
            - labels: (batch_size,)
            where batch_size is number of individual photons
            
        Example:
        --------
        # Event mode (default): batch contains photons from multiple events concatenated
        train_loader = llrnet.create_patd_dataloader(
            signal_sampler=sampler,
            signal_surrogate_func=surrogate,
            num_samples_per_epoch=1000,
            batch_size=32,  # 32 events per batch
            num_workers=4,
            shuffle_photons=False
        )
        
        # Photon mode: batch contains individual photons from different events mixed
        train_loader = llrnet.create_patd_dataloader(
            signal_sampler=sampler,
            signal_surrogate_func=surrogate,
            num_samples_per_epoch=1000,
            batch_size=128,  # 128 individual photons per batch
            num_workers=4,
            shuffle_photons=True
        )
        """
        if not self.use_patd:
            raise ValueError("create_patd_dataloader can only be used when use_patd=True")
        
        if shuffle_photons:
            # Photon mode: simple collate for individual photons
            def patd_collate_fn(batch):
                """
                Collate function for individual photons.
                
                Each item is (features, label) where features is (feature_dim,) and label is scalar.
                """
                features_list = [item[0] for item in batch]
                labels_list = [item[1] for item in batch]
                
                batch_features = torch.stack(features_list, dim=0)
                batch_labels = torch.stack(labels_list, dim=0)
                
                return batch_features, batch_labels
        else:
            # Event mode: concatenate all photons from multiple events
            def patd_collate_fn(batch):
                """
                Collate function that concatenates all photons from multiple events.
                
                Each item in batch is (features_batch, labels) where:
                - features_batch: (num_photons_in_event, feature_dim)
                - labels: (num_photons_in_event,)
                
                Output:
                - batch_features: (total_photons, feature_dim)
                - batch_labels: (total_photons,)
                """
                all_features = []
                all_labels = []
                
                for features, labels in batch:
                    all_features.append(features)
                    all_labels.append(labels)
                
                # Concatenate all photons from all events in the batch
                batch_features = torch.cat(all_features, dim=0)
                batch_labels = torch.cat(all_labels, dim=0)
                
                return batch_features, batch_labels
        
        dataset = self.PATDDataset(
            signal_sampler=signal_sampler,
            signal_surrogate_func=signal_surrogate_func,
            llrnet_instance=self,
            num_samples_per_epoch=num_samples_per_epoch,
            event_labels=event_labels,
            min_photons=self.min_photons,
            num_photons_per_sample=self.num_photons_per_sample,
            shuffle_photons=shuffle_photons,
            use_rich_features=use_rich_features,
            **other_kwargs
        )
        
        # IterableDataset: no shuffle parameter needed (handled internally)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=patd_collate_fn
        )
        
        return dataloader
    
    
    def create_batch_signal_only_dataloader(self, signal_sampler, signal_surrogate_func,
                                           num_batches=1000, batch_size=128, num_workers=0,
                                           event_labels=['position', 'energy', 'zenith', 'azimuth']):
        
        dataset = self.BatchSignalOnlyDataset(
            llrnet=self,
            signal_sampler=signal_sampler,
            signal_surrogate_func=signal_surrogate_func,
            num_batches=num_batches,
            batch_size=batch_size,
            event_labels=event_labels
        )
        
        # Since the dataset returns a full batch at once, batch_size for DataLoader must be None
        # or 1 (if we squeeze later). Usually None disables automatic batching.
        return DataLoader(dataset, batch_size=None, num_workers=num_workers)
    
    def create_event_dataloader(self, signal_sampler, background_sampler, signal_surrogate_func, background_surrogate_func,
                               num_samples_per_epoch=1000, batch_size=32, 
                               shuffle=True, num_workers=0, output_true_light_yield=False,
                               event_labels=['position', 'energy', 'zenith', 'azimuth']):
        """
        Create a DataLoader for balanced signal/background training using the EventDataset class.
        
        This method creates an EventDataset that generates balanced signal and background
        events with shared detector points and event parameters, ensuring perfectly balanced
        training with matched features except for detector response. The dataset internally
        generates pairs but returns individual events to the DataLoader.
        
        Parameters:
        -----------
        signal_sampler : ToySampler
            Sampler instance for generating signal event parameters
        background_sampler : ToySampler
            Sampler instance for generating background event parameters
        signal_surrogate_func : callable
            Function to calculate light yield for signal events  
        background_surrogate_func : callable
            Function to calculate light yield for background events
        num_samples_per_epoch : int
            Total number of signal/background pairs to generate per epoch
            (will result in 2 * num_samples_per_epoch individual events)
        batch_size : int
            Number of individual events per batch 
        shuffle : bool
            Whether to shuffle the individual events
        num_workers : int
            Number of worker processes for data loading
        event_labels : list
            List of event parameter keys to include as features
            
        Returns:
        --------
        torch.utils.data.DataLoader
            Configured DataLoader for balanced training
            
        Example:
        --------
        >>> # Create model and samplers
        >>> model = LLRnet(dim=3, domain_size=2, device=device)
        >>> signal_sampler = ToySampler(device=device, dim=3, domain_size=2)
        >>> background_sampler = ToySampler(device=device, dim=3, domain_size=2)
        >>> 
        >>> # Create DataLoader for balanced training
        >>> train_loader = model.create_event_dataloader(
        ...     signal_sampler=signal_sampler,
        ...     background_sampler=background_sampler,
        ...     signal_surrogate_func=signal_func,
        ...     background_surrogate_func=background_func,
        ...     num_samples_per_epoch=2500,  # 2500 pairs = 5000 total events per epoch
        ...     batch_size=64                # 64 individual events per batch
        ... )
        >>>
        >>> # Train model
        >>> history = model.train_with_dataloader(train_loader, epochs=100)
        """
        dataset = self.EventDataset(
            llrnet_instance=self,
            signal_sampler=signal_sampler,
            background_sampler=background_sampler,
            signal_surrogate_func=signal_surrogate_func,
            background_surrogate_func=background_surrogate_func,
            num_samples_per_epoch=num_samples_per_epoch,
            event_labels=event_labels, output_true_light_yield=output_true_light_yield
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Speed up GPU transfers
        )
        
        return dataloader
    
    def create_signal_only_dataloader(self, signal_sampler, signal_surrogate_func,
                                     num_samples_per_epoch=1000, batch_size=32,
                                     shuffle=True, num_workers=0, output_true_light_yield=False,
                                     event_labels=['position', 'energy', 'zenith', 'azimuth'],
                                     use_rich_features=False, **other_kwargs):
        """
        Create a DataLoader for signal-only training with matched/mismatched light yields.
        
        This method creates a SignalOnlyDataset that trains the network to distinguish between:
        - Matched (label=1): Event parameters with their corresponding light yield
        - Mismatched (label=0): Event parameters with light yield from a different event
        
        This is useful for training the network to verify if light yield is consistent with
        given event parameters, which can help detect anomalies or validate reconstructions.
        
        Parameters:
        -----------
        signal_sampler : ToySampler
            Sampler instance for generating signal event parameters
        signal_surrogate_func : callable
            Function to calculate light yield for signal events
        num_samples_per_epoch : int
            Total number of matched/mismatched pairs to generate per epoch
            (will result in 2 * num_samples_per_epoch individual events)
        batch_size : int
            Number of individual events per batch
        shuffle : bool
            Whether to shuffle the individual events
        num_workers : int
            Number of worker processes for data loading
        output_true_light_yield : bool
            Whether to output true light yield (for debugging/analysis)
        event_labels : list
            List of event parameter keys to include as features
        **other_kwargs : dict
            Additional keyword arguments passed to SignalOnlyDataset, including:
            - samples_per_event : int
                Number of times to sample from the same detector point and event.
                Useful when surrogate_func includes randomness (e.g., Poisson sampling).
                Default is 1 (no resampling). If > 1, each unique event/detector pair
                will be sampled multiple times with different random realizations.
                Total dataset size will be 2 * num_samples_per_epoch * samples_per_event.
            - presampled_events : list, optional
                Pre-sampled event parameters to reuse
            - presampled_detector_points : torch.Tensor, optional
                Pre-sampled detector points to reuse
            - min_light_yield : float, optional
                Minimum light yield threshold for resampling
            - max_resample_attempts : int, optional
                Maximum number of resampling attempts
            
        Returns:
        --------
        torch.utils.data.DataLoader
            Configured DataLoader for signal-only training
            
        """
        dataset = self.SignalOnlyDataset(
            llrnet_instance=self,
            signal_sampler=signal_sampler,
            signal_surrogate_func=signal_surrogate_func,
            num_samples_per_epoch=num_samples_per_epoch,
            event_labels=event_labels,
            output_true_light_yield=output_true_light_yield,
            use_rich_features=use_rich_features,
            **other_kwargs
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Speed up GPU transfers
        )
        
        return dataloader


