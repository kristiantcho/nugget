from nugget.surrogates.base_surrogate import Surrogate
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
import numpy as np
import os

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


class ChargeNet(Surrogate):
    """
    Charge prediction network for training an MLP to estimate light yield/DOM response.
    
    This network is trained as a regressor to predict the detector response (light yield)
    at a given detector location for a given neutrino event. Unlike LLRnet which performs
    binary classification, ChargeNet directly predicts continuous light yield values.
    
    The network supports parallel Fourier mapping layers with corresponding MLPs that
    process different frequency scales simultaneously, similar to LLRnet architecture.
    
    Architecture:
    - Multiple parallel branches, each with:
      * Optional Fourier feature mapping at different frequency scales
      * Either separate MLPs per branch OR a single shared MLP (when shared_mlp=True)
    - Final MLP that concatenates all branch outputs and produces light yield prediction
    
    Example Usage:
    --------------
    
    # Single branch architecture
    model = ChargeNet(dim=3, num_parallel_branches=1, frequency_scale=1.0)
    
    # Multiple branches with different frequency scales
    model = ChargeNet(
        dim=3, 
        num_parallel_branches=3,
        frequency_scales=[0.5, 2.0, 8.0],
        num_frequencies_per_branch=[32, 64, 32],
        shared_mlp=True
    )
    
    # Train with event data
    history = model.train_with_dataloader(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=100
    )
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, hidden_dims=[128, 64, 32], 
                 dropout_rate=0.1, learning_rate=1e-3, use_fourier_features=True,
                 num_frequencies=64, frequency_scale=1.0, learnable_frequencies=False,
                 num_parallel_branches=1, frequency_scales=None, num_frequencies_per_branch=None, 
                 log_scale_ly=False, norm_pos=False, shared_mlp=False, use_residual_connections=False,
                 add_relative_pos=True, add_distance_from_beam=False, log_scale_energy=False, 
                 reduce_lr_on_plateau=False, lr_scheduler_patience=10, lr_scheduler_factor=0.5, 
                 lr_scheduler_min_lr=1e-6):
        """
        Initialize the ChargeNet surrogate model.
        
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
            Number of frequency components for Fourier features
        frequency_scale : float
            Scale factor for frequency sampling in Fourier features
        learnable_frequencies : bool
            If True, Fourier frequencies are learnable parameters
        num_parallel_branches : int
            Number of parallel Fourier+MLP branches to use
        frequency_scales : list or None
            List of frequency scales for each branch
        num_frequencies_per_branch : list or None
            List of number of frequencies for each branch
        shared_mlp : bool
            If True, uses a single shared MLP for all Fourier branches
        use_residual_connections : bool
            If True, uses residual connections in the MLP layers
        add_relative_pos : bool
            If True, adds relative position as features
        add_distance_from_beam : bool
            If True, adds perpendicular distance from detector point to beam/track
        log_scale_ly : bool
            If True, trains to predict log10 of light yield (recommended for better convergence)
        norm_pos : bool
            If True, normalizes position coordinates by domain_size/2
        log_scale_energy : bool
            If True, uses log10(energy) as input feature
        reduce_lr_on_plateau : bool
            If True, reduces learning rate when validation loss plateaus
        lr_scheduler_patience : int
            Number of epochs with no improvement after which learning rate will be reduced
        lr_scheduler_factor : float
            Factor by which the learning rate will be reduced
        lr_scheduler_min_lr : float
            Minimum learning rate allowed
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
        self.log_scale_ly = log_scale_ly
        self.norm_pos = norm_pos
        self.log_scale_energy = log_scale_energy
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        
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
        
        # Initialize network architecture
        self.fourier_features_list = None
        self.mlp_branches = None
        self.shared_branch_mlp = None
        self.final_mlp = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = torch.nn.MSELoss()  # MSE loss for regression
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_trained = False
        
    def _build_network(self, input_dim):
        """Build the parallel MLP network architecture with multiple Fourier feature mappings."""
        
        print(f"Building ChargeNet architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Number of parallel branches: {self.num_parallel_branches}")
        print(f"  Shared MLP: {self.shared_mlp}")
        print(f"  Log scale output: {self.log_scale_ly}")
        
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
        
        # Create final MLP that combines all branch outputs
        total_branch_output_dim = sum(branch_output_dims)
        final_layers = []
        
        final_hidden_dim = min(64, total_branch_output_dim // 2)
        
        final_layers.append(torch.nn.Linear(total_branch_output_dim, final_hidden_dim))
        final_layers.append(torch.nn.SiLU())
        final_layers.append(torch.nn.Dropout(self.dropout_rate))
        
        # Final output layer - LINEAR (no sigmoid) for regression
        final_layers.append(torch.nn.Linear(final_hidden_dim, 1))
        
        self.final_mlp = torch.nn.Sequential(*final_layers).to(self.device)
        
        print(f"  Final MLP: {total_branch_output_dim} -> {final_hidden_dim} -> 1 (linear)")
        
        # Create optimizer
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
        
        # Initialize learning rate scheduler if requested
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
        if point.dim() == 1:
            point = point.unsqueeze(0)
        if track_pos.dim() == 1:
            track_pos = track_pos.unsqueeze(0)
        if track_dir.dim() == 1:
            track_dir = track_dir.unsqueeze(0)
            
        rel_pos = point - track_pos
        dist_long = torch.sum(rel_pos * track_dir, dim=-1, keepdim=True)
        perp_vec = rel_pos - dist_long * track_dir
        dist_perp = torch.norm(perp_vec, dim=-1, keepdim=True)
        
        return dist_long, dist_perp
    
    def prepare_data_from_raw(self, point, event_data, surrogate_func, 
                             event_labels=['position', 'energy', 'zenith', 'azimuth'], 
                             noise_scale=0.0):
        """
        Prepare training data from raw neutrino event data.
        
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
            Scale for adding noise to light yield values
            
        Returns:
        --------
        tuple : (features, light_yield)
            features : torch.Tensor of shape (feature_dim,)
            light_yield : torch.Tensor of shape (1,)
        """
        # Helper to ensure tensor
        def to_tensor(val):
            if torch.is_tensor(val):
                return val.to(self.device).float()
            return torch.tensor(val, dtype=torch.float32, device=self.device)
        
        # Extract track parameters
        track_pos = to_tensor(event_data['position'])
        if track_pos.dim() == 1:
            track_pos = track_pos.unsqueeze(0)
        
        # Calculate track direction
        if 'direction' in event_data:
            track_dir = to_tensor(event_data['direction'])
        elif 'zenith' in event_data and 'azimuth' in event_data:
            zenith = to_tensor(event_data['zenith'])
            azimuth = to_tensor(event_data['azimuth'])
            
            sz, cz = torch.sin(zenith), torch.cos(zenith)
            sa, ca = torch.sin(azimuth), torch.cos(azimuth)
            track_dir = torch.stack([sz*ca, sz*sa, cz], dim=-1)
            if track_dir.dim() == 3:
                track_dir = track_dir.squeeze(1)
        else:
            track_dir = torch.zeros_like(track_pos)
            track_dir[:, 2] = 1.0
        
        # Ensure point is correct shape
        if isinstance(point, np.ndarray):
            point = torch.tensor(point, device=self.device, dtype=torch.float32)
        else:
            point = point.to(self.device).float()
            
        if point.dim() == 1:
            point = point.unsqueeze(0)
        
        # Calculate geometric features
        if self.add_distance_from_beam:
            dist_long, dist_perp = self.compute_distance_from_beam(point, track_pos, track_dir)
        
        # Get light yield
        light_yield = surrogate_func(opt_point=point, event_params=event_data)
        
        # Add noise if requested
        if noise_scale > 0:
            noise = torch.randn_like(light_yield) * noise_scale
            light_yield = light_yield + noise
        
        # Store original light yield for target
        
        
        # Log scale light yield if requested (for better convergence)
        if self.log_scale_ly:
            light_yield = torch.log10(torch.abs(light_yield) + 1e-10)
       
        # Construct feature vector
        feature_list = []
        
        # # Add light yield as input feature
        # feature_list.append(light_yield.flatten())
        
        if self.add_relative_pos:
            if self.norm_pos:
                rel_pos = (point - track_pos) / (self.domain_size / 2)
            else:
                rel_pos = point - track_pos
        else:
            if self.norm_pos:
                rel_pos = point / (self.domain_size / 2)
            else:
                rel_pos = point
            feature_list.extend(rel_pos.flatten())
            
        if self.add_distance_from_beam:
            feature_list.extend(dist_long.flatten())
            feature_list.extend(dist_perp.flatten())
            
        if 'energy' in event_labels:
            energy = to_tensor(event_data['energy']).view(-1, 1)
            if self.log_scale_energy:
                feature_list.extend(torch.log10(energy.flatten()))
            else:
                feature_list.extend(energy.flatten())
            
        if ('zenith' in event_labels and 'azimuth' in event_labels) or ('direction' in event_labels):
            feature_list.extend(track_dir.flatten())
        
        if 'position' in event_labels:
            feature_list.extend(track_pos.flatten()) 
        
        features = torch.stack(feature_list, dim=0)
        
        # Ensure light_yield is a scalar tensor for proper batching
        return features, light_yield.squeeze()
    
    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                             verbose=True, early_stopping_patience=10, input_dim=None):
        """
        Train the ChargeNet network using PyTorch DataLoader.
        
        Parameters:
        -----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader providing training batches of (features, light_yields)
        val_dataloader : torch.utils.data.DataLoader, optional
            DataLoader for validation data
        epochs : int
            Number of training epochs  
        verbose : bool
            Whether to print training progress
        early_stopping_patience : int
            Number of epochs to wait for improvement before early stopping
        input_dim : int, optional
            Input dimension (will be inferred from first batch if not provided)
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        # Build network if not already built
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            if input_dim is None:
                # Get a sample to determine feature dimension
                sample_features, _ = next(iter(train_dataloader))
                if isinstance(sample_features, torch.Tensor):
                    input_dim = sample_features.shape[-1]
                else:
                    raise ValueError("Could not determine input dimension from dataloader")
            
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
                for fourier in self.fourier_features_list:
                    fourier.train()
            
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch_features, batch_targets in train_dataloader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                predictions = self._forward_pass(batch_features)
                
                # Compute MSE loss
                loss = self.loss_fn(predictions, batch_targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                if self.shared_mlp:
                    self.shared_branch_mlp.eval()
                else:
                    for branch in self.mlp_branches:
                        branch.eval()
                self.final_mlp.eval()
                if self.fourier_features_list is not None:
                    for fourier in self.fourier_features_list:
                        fourier.eval()
                
                epoch_val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_dataloader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        predictions = self._forward_pass(batch_features)
                        loss = self.loss_fn(predictions, batch_targets)
                        
                        epoch_val_loss += loss.item()
                        num_val_batches += 1
                
                avg_val_loss = epoch_val_loss / num_val_batches
                self.val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
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
            Network output (predicted light yield)
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=self.device, dtype=torch.float32)
        else:
            points = points.to(self.device)
        
        # Process through each parallel branch
        branch_outputs = []
        
        if self.shared_mlp:
            max_fourier_dim = max([f.output_dim for f in self.fourier_features_list]) if self.fourier_features_list else points.shape[-1]
            
            for i in range(self.num_parallel_branches):
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](points)
                    if branch_input.shape[-1] < max_fourier_dim:
                        padding = torch.zeros(*branch_input.shape[:-1], max_fourier_dim - branch_input.shape[-1], device=self.device)
                        branch_input = torch.cat([branch_input, padding], dim=-1)
                else:
                    branch_input = points
                
                branch_output = self.shared_branch_mlp(branch_input)
                branch_outputs.append(branch_output)
        else:
            for i in range(self.num_parallel_branches):
                if self.fourier_features_list is not None:
                    branch_input = self.fourier_features_list[i](points)
                else:
                    branch_input = points
                
                branch_output = self.mlp_branches[i](branch_input)
                branch_outputs.append(branch_output)
        
        # Concatenate all branch outputs
        concatenated_features = torch.cat(branch_outputs, dim=-1)
        
        # Final MLP (linear output, no sigmoid)
        final_output = self.final_mlp(concatenated_features)
        
        return final_output.squeeze(-1)
    
    def __call__(self, points):
        """
        Evaluate the trained ChargeNet on input points.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
            
        Returns:
        --------
        torch.Tensor
            Predicted light yield values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        if self.shared_mlp:
            self.shared_branch_mlp.eval()
        else:
            for branch in self.mlp_branches:
                branch.eval()
        self.final_mlp.eval()
        if self.fourier_features_list is not None:
            for fourier in self.fourier_features_list:
                fourier.eval()
        
        with torch.no_grad():
            predictions = self._forward_pass(points)
        
        return predictions
    
    def predict(self, features):
        """
        Predict light yield values for given features.
        
        Parameters:
        -----------
        features : torch.Tensor or np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray
            Predicted light yield values
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        predictions = self(features)
        
        # Convert back from log scale if necessary
        if self.log_scale_ly:
            predictions = 10 ** predictions
        else:
            predictions = torch.clamp(predictions, min=0.0)  # Ensure non-negative
        
        # Round to integers for discrete photon counts
        # predictions = torch.round(predictions)
       
        
        return predictions.cpu().numpy()
    
    def evaluate(self, features, targets):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features
        targets : torch.Tensor
            True light yield values
            
        Returns:
        --------
        dict : Dictionary containing evaluation metrics
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        self.eval()
        with torch.no_grad():
            predictions = self._forward_pass(features)
        
        # Compute MSE
        mse = torch.nn.functional.mse_loss(predictions, targets).item()
        
        # Compute MAE
        mae = torch.nn.functional.l1_loss(predictions, targets).item()
        
        # Compute R² score
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def save_model(self, filepath):
        """Save model state to file."""
        state = {
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
            'log_scale_ly': self.log_scale_ly,
            'norm_pos': self.norm_pos,
            'log_scale_energy': self.log_scale_energy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'is_trained': self.is_trained,
        }
        
        if self.fourier_features_list is not None:
            state['fourier_features_state'] = [f.state_dict() for f in self.fourier_features_list]
        
        if self.shared_mlp:
            state['shared_branch_mlp_state'] = self.shared_branch_mlp.state_dict()
        else:
            state['mlp_branches_state'] = [branch.state_dict() for branch in self.mlp_branches]
        
        state['final_mlp_state'] = self.final_mlp.state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()
        
        if self.lr_scheduler is not None:
            state['lr_scheduler_state'] = self.lr_scheduler.state_dict()
        
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model state from file."""
        state = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        self.hidden_dims = state['hidden_dims']
        self.dropout_rate = state['dropout_rate']
        self.learning_rate = state['learning_rate']
        self.use_fourier_features = state['use_fourier_features']
        self.num_frequencies = state['num_frequencies']
        self.frequency_scale = state['frequency_scale']
        self.learnable_frequencies = state['learnable_frequencies']
        self.num_parallel_branches = state['num_parallel_branches']
        self.frequency_scales = state['frequency_scales']
        self.num_frequencies_per_branch = state['num_frequencies_per_branch']
        self.shared_mlp = state['shared_mlp']
        self.use_residual_connections = state['use_residual_connections']
        self.add_relative_pos = state['add_relative_pos']
        self.add_distance_from_beam = state['add_distance_from_beam']
        self.log_scale_ly = state['log_scale_ly']
        self.norm_pos = state['norm_pos']
        self.log_scale_energy = state['log_scale_energy']
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']
        self.is_trained = state['is_trained']
        
        # Determine input dimension from first Fourier layer or final MLP
        if 'fourier_features_state' in state:
            first_fourier_state = state['fourier_features_state'][0]
            input_dim = first_fourier_state['frequencies'].shape[1]
        else:
            # Infer from MLP input
            if self.shared_mlp:
                mlp_state = state['shared_branch_mlp_state']
                # Check if using residual connections (ResidualBlock has 'linear.weight')
                if '0.linear.weight' in mlp_state:
                    first_layer_weight = mlp_state['0.linear.weight']
                else:
                    first_layer_weight = mlp_state['0.weight']
            else:
                mlp_state = state['mlp_branches_state'][0]
                # Check if using residual connections
                if '0.linear.weight' in mlp_state:
                    first_layer_weight = mlp_state['0.linear.weight']
                else:
                    first_layer_weight = mlp_state['0.weight']
            input_dim = first_layer_weight.shape[1]
        
        # Rebuild network
        self._build_network(input_dim)
        
        # Load state dicts
        if 'fourier_features_state' in state:
            for fourier, fourier_state in zip(self.fourier_features_list, state['fourier_features_state']):
                fourier.load_state_dict(fourier_state)
        
        if self.shared_mlp:
            self.shared_branch_mlp.load_state_dict(state['shared_branch_mlp_state'])
        else:
            for branch, branch_state in zip(self.mlp_branches, state['mlp_branches_state']):
                branch.load_state_dict(branch_state)
        
        self.final_mlp.load_state_dict(state['final_mlp_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        
        if 'lr_scheduler_state' in state and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state['lr_scheduler_state'])
        
        print(f"Model loaded from {filepath}")
    
    class ChargeDataset(Dataset):
        """
        Dataset for training ChargeNet to predict light yield.
        
        Generates training samples by:
        1. Sampling detector points
        2. Sampling event parameters
        3. Computing light yield using surrogate function
        4. Optionally resampling if light yield is below threshold
        """
        
        def __init__(self, sampler, surrogate_func, num_samples, chargenet_model,
                     event_labels=['position', 'energy', 'zenith', 'azimuth'], noise_scale=0.0,
                     min_light_yield=None, max_resample_attempts=10):
            """
            Initialize dataset.
            
            Parameters:
            -----------
            sampler : Sampler
                Sampler for generating event parameters
            surrogate_func : callable
                Function to compute light yield
            num_samples : int
                Number of samples in epoch
            chargenet_model : ChargeNet
                ChargeNet model instance (for prepare_data_from_raw method)
            event_labels : list
                List of event parameter keys
            noise_scale : float
                Scale for adding noise
            min_light_yield : float or None
                Minimum light yield required. If provided, events with light yield below
                this threshold will be resampled (up to max_resample_attempts)
            max_resample_attempts : int
                Maximum number of resampling attempts if light yield is below threshold
            """
            self.sampler = sampler
            self.surrogate_func = surrogate_func
            self.num_samples = num_samples
            self.chargenet_model = chargenet_model
            self.event_labels = event_labels
            self.noise_scale = noise_scale
            self.min_light_yield = min_light_yield
            self.max_resample_attempts = max_resample_attempts
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Sample with resampling if needed
            for attempt in range(self.max_resample_attempts):
                # Sample detector point and event parameters
                point = self.sampler.sample_detector_points(1).squeeze()
                event_params = self.sampler.sample_events(1)[0]
                
                # Prepare features and target
                features, target = self.chargenet_model.prepare_data_from_raw(
                    point, event_params, self.surrogate_func,
                    event_labels=self.event_labels,
                    noise_scale=self.noise_scale
                )
                
                # Check if light yield meets minimum threshold
                if self.min_light_yield is None:
                    # No threshold, accept sample
                    return features, target
                
                # Get the actual light yield value (handle both tensor and scalar)
                if isinstance(target, torch.Tensor):
                    ly_value = target.item() if target.numel() == 1 else target.max().item()
                else:
                    ly_value = float(target)
                
                # If we're using log scale, convert back to check threshold
                if self.chargenet_model.log_scale_ly:
                    ly_value = 10 ** ly_value
                
                if ly_value >= self.min_light_yield:
                    return features, target
            
            # If we exhausted all attempts, return the last sample anyway
            return features, target
    
    def create_charge_dataloader(self, sampler, surrogate_func, num_samples_per_epoch=1000,
                                 batch_size=32, shuffle=True, num_workers=0,
                                 event_labels=['position', 'energy', 'zenith', 'azimuth'],
                                 noise_scale=0.0, min_light_yield=None, max_resample_attempts=10):
        """
        Create a DataLoader for training ChargeNet.
        
        Parameters:
        -----------
        sampler : Sampler
            Sampler for generating event parameters
        surrogate_func : callable
            Function to compute light yield
        num_samples_per_epoch : int
            Number of samples per epoch
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle data
        num_workers : int
            Number of worker processes
        event_labels : list
            List of event parameter keys
        noise_scale : float
            Scale for adding noise
        min_light_yield : float or None
            Minimum light yield required. If provided, events with light yield below
            this threshold will be resampled (up to max_resample_attempts). This is
            useful for ensuring training samples have sufficient signal.
        max_resample_attempts : int
            Maximum number of resampling attempts if light yield is below threshold
            
        Returns:
        --------
        DataLoader
        """
        dataset = self.ChargeDataset(
            sampler=sampler,
            surrogate_func=surrogate_func,
            num_samples=num_samples_per_epoch,
            chargenet_model=self,
            event_labels=event_labels,
            noise_scale=noise_scale,
            min_light_yield=min_light_yield,
            max_resample_attempts=max_resample_attempts
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return dataloader


class ChargeNetSurrogate(Surrogate):
    """
    ChargeNet-based surrogate for light yield prediction.
    
    This class wraps a trained ChargeNet model and provides a light_yield_surrogate
    interface compatible with other surrogate models like LightSabre. It can be used
    as a drop-in replacement for physics-based surrogates in optimization and sampling workflows.
    
    Example Usage:
    --------------
    # Train a ChargeNet model
    chargenet = ChargeNet(dim=3, log_scale_ly=True)
    chargenet.train_with_dataloader(train_loader, val_loader, epochs=100)
    
    # Create surrogate wrapper
    surrogate = ChargeNetSurrogate(chargenet_model=chargenet)
    
    # Use like LightSabre
    event_params = {
        'position': torch.tensor([0., 0., 0.]),
        'zenith': torch.tensor(0.5),
        'azimuth': torch.tensor(1.0),
        'energy': torch.tensor(1000.)
    }
    opt_point = torch.tensor([10., 5., 3.])
    
    light_yield = surrogate.light_yield_surrogate(
        opt_point=opt_point,
        event_params=event_params
    )
    """
    
    def __init__(self, chargenet_model, device=None, dim=3, domain_size=2):
        """
        Initialize ChargeNetSurrogate with a trained ChargeNet model.
        
        Parameters:
        -----------
        chargenet_model : ChargeNet
            A trained ChargeNet model instance
        device : torch.device, optional
            Device to run predictions on (uses model's device if not specified)
        dim : int
            Dimension of the input space
        domain_size : int
            Length of the domain
        """
        if device is None:
            device = chargenet_model.device
        
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        self.chargenet = chargenet_model
        
        if not self.chargenet.is_trained:
            raise ValueError("ChargeNet model must be trained before using as surrogate")
    
    def light_yield_surrogate(self, **kwargs):
        """
        Surrogate function that computes light yield using trained ChargeNet model.
        
        This method provides a consistent interface with physics-based surrogate models
        and can be used in optimization workflows.
        
        Parameters:
        -----------
        event_params : dict
            Contains 'position', 'zenith', 'azimuth', 'energy'
        opt_point : torch.Tensor
            Optimization point where light yield is evaluated (single point or array)
        use_poisson : bool, optional
            If True, samples from Poisson distribution with predicted mean (default: False)
            
        Returns:
        --------
        torch.Tensor
            Predicted light yield value(s) at the optimization point(s)
        """
        # Extract parameters
        opt_point = kwargs.get('opt_point', None)
        event_params = kwargs.get('event_params', None)
        use_poisson = kwargs.get('use_poisson', False)
        
        if event_params is None:
            raise ValueError("event_params must be provided")
        
        if opt_point is None:
            raise ValueError("opt_point must be provided")
        
        # Validate event parameters
        required_params = ['position', 'zenith', 'azimuth', 'energy']
        for param in required_params:
            if param not in event_params:
                raise ValueError(f"event_params must contain '{param}'")
        
        # Prepare features using ChargeNet's data preparation method
        # Note: We need a dummy surrogate function since prepare_data_from_raw expects one
        # but for prediction we don't actually compute light yield - we predict it
        def dummy_surrogate(opt_point, event_params):
            # Return zeros as placeholder - these will be replaced by the model's prediction
            if isinstance(opt_point, torch.Tensor):
                return torch.zeros(opt_point.shape[0] if opt_point.dim() > 1 else 1, device=self.device)
            return torch.zeros(1, device=self.device)
        
        # Handle single point or batch of points
        if isinstance(opt_point, torch.Tensor):
            opt_point_tensor = opt_point.to(self.device)
        else:
            opt_point_tensor = torch.tensor(opt_point, device=self.device, dtype=torch.float32)
        
        # Check if we have a batch or single point
        is_batch = opt_point_tensor.dim() > 1
        
        if is_batch:
            # Process batch of points
            batch_size = opt_point_tensor.shape[0]
            predictions = []
            
            for i in range(batch_size):
                point = opt_point_tensor[i]
                features, _ = self.chargenet.prepare_data_from_raw(
                    point=point,
                    event_data=event_params,
                    surrogate_func=dummy_surrogate,
                    noise_scale=0.0
                )
                
                # Make prediction
                self.chargenet.eval()
                with torch.no_grad():
                    pred = self.chargenet._forward_pass(features.unsqueeze(0))
                
                # Convert from log scale if necessary
                if self.chargenet.log_scale_ly:
                    pred = 10 ** pred
                
                predictions.append(pred)
            
            light_yield = torch.stack(predictions).squeeze()
        else:
            # Single point
            features, _ = self.chargenet.prepare_data_from_raw(
                point=opt_point_tensor,
                event_data=event_params,
                surrogate_func=dummy_surrogate,
                noise_scale=0.0
            )
            
            # Make prediction
            self.chargenet.eval()
            with torch.no_grad():
                light_yield = self.chargenet._forward_pass(features.unsqueeze(0))
            
            # Convert from log scale if necessary
            if self.chargenet.log_scale_ly:
                light_yield = 10 ** light_yield
            
            light_yield = light_yield.squeeze()
        
        # Ensure non-negative
        light_yield = torch.clamp(light_yield, min=0)
        
        # Optional Poisson sampling
        if use_poisson:
            light_yield = torch.poisson(light_yield)
        
        return light_yield
    
    def __call__(self, **kwargs):
        """
        Call the surrogate function (alias for light_yield_surrogate).
        
        Parameters:
        -----------
        **kwargs : dict
            Same parameters as light_yield_surrogate
            
        Returns:
        --------
        torch.Tensor
            Predicted light yield
        """
        return self.light_yield_surrogate(**kwargs)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, dim=3, domain_size=2):
        """
        Create a ChargeNetSurrogate from a saved checkpoint.
        
        Parameters:
        -----------
        checkpoint_path : str
            Path to saved ChargeNet model checkpoint
        device : torch.device, optional
            Device to load model on
        dim : int
            Dimension of the input space
        domain_size : int
            Length of the domain
            
        Returns:
        --------
        ChargeNetSurrogate
            Surrogate instance with loaded model
        """
        # Load the checkpoint to get model hyperparameters
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        state = torch.load(checkpoint_path, map_location=device)
        
        # Create ChargeNet with saved hyperparameters
        chargenet = ChargeNet(
            device=device,
            dim=dim,
            domain_size=domain_size,
            hidden_dims=state['hidden_dims'],
            dropout_rate=state['dropout_rate'],
            learning_rate=state['learning_rate'],
            use_fourier_features=state['use_fourier_features'],
            num_frequencies=state['num_frequencies'],
            frequency_scale=state['frequency_scale'],
            learnable_frequencies=state['learnable_frequencies'],
            num_parallel_branches=state['num_parallel_branches'],
            frequency_scales=state['frequency_scales'],
            num_frequencies_per_branch=state['num_frequencies_per_branch'],
            shared_mlp=state['shared_mlp'],
            use_residual_connections=state['use_residual_connections'],
            add_relative_pos=state['add_relative_pos'],
            add_distance_from_beam=state['add_distance_from_beam'],
            log_scale_ly=state['log_scale_ly'],
            norm_pos=state['norm_pos'],
            log_scale_energy=state['log_scale_energy']
        )
        
        # Load the model weights
        chargenet.load_model(checkpoint_path)
        
        return cls(chargenet_model=chargenet, device=device, dim=dim, domain_size=domain_size)
