from nugget.surrogates.base_surrogate import Surrogate
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
        self.activation = torch.nn.ReLU()
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
                 num_parallel_branches=1, frequency_scales=None, num_frequencies_per_branch=None,
                 shared_mlp=False, use_residual_connections=False, signal_noise_scale=0.0, background_noise_scale=0.0, add_relative_pos=True):
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
        self.loss_fn = torch.nn.BCELoss()  # Changed from BCEWithLogitsLoss
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_trained = False
        
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
            shared_layers.append(torch.nn.ReLU())
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
                    shared_layers.append(torch.nn.ReLU())
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
                branch_layers.append(torch.nn.ReLU())
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
                        branch_layers.append(torch.nn.ReLU())
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
        final_layers.append(torch.nn.ReLU())
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
        
        print(f"  Total trainable parameters: {sum(p.numel() for p in all_params if p.requires_grad):,}")
        
    
    
    def prepare_data_from_raw(self, point, event_data, surrogate_func, event_labels = ['position', 'energy', 'zenith', 'azimuth'], noise_scale=0.0, add_relative_pos=True):
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
            Scale for adding noise (not used currently)
            
        Returns:
        --------
        torch.Tensor
            features: torch.Tensor of shape (feature_dim,) for single event
        """
        if isinstance(event_data, dict):
            # Extract features from dictionary
            feature_list = []
            
            # Add detector point coordinates
            if isinstance(point, np.ndarray):
                point_tensor = torch.tensor(point, device=self.device, dtype=torch.float32)
            else:
                point_tensor = point.float().to(self.device)
            
            # Flatten point coordinates to 1D
            feature_list.append(point_tensor.flatten())
            if add_relative_pos and 'position' in event_data:
                event_pos = event_data['position']
                if isinstance(event_pos, np.ndarray):
                    event_pos_tensor = torch.tensor(event_pos, device=self.device, dtype=torch.float32)
                else:
                    event_pos_tensor = event_pos.float().to(self.device)
                relative_pos = point_tensor - event_pos_tensor
                feature_list.append(relative_pos.flatten())
            
            # Add event parameters
            for key in event_labels:
                if key in event_data:
                    feature = event_data[key]
                    if isinstance(feature, np.ndarray):
                        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                    elif not isinstance(feature, torch.Tensor):
                        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
                    
                    # Flatten feature to 1D
                    feature_list.append(feature.flatten())
                else:
                    raise KeyError(f"Key '{key}' not found in event_data")
                    
            # Calculate detector response
            detector_response = surrogate_func(opt_point=point_tensor, event_params=event_data)
            if noise_scale > 0.0:
                noise = np.random.normal(0, noise_scale, size=detector_response.shape)
                detector_response += torch.tensor(noise, device=self.device, dtype=torch.float32) * detector_response
            if isinstance(detector_response, np.ndarray):
                detector_response = torch.tensor(detector_response, device=self.device, dtype=torch.float32)
            elif not isinstance(detector_response, torch.Tensor):
                detector_response = torch.tensor(detector_response, device=self.device, dtype=torch.float32)
            
            # Ensure detector response is on correct device and flattened
            detector_response = detector_response.float().to(self.device).flatten()
            feature_list.append(detector_response)

            # Combine all features into a single 1D tensor
            features = torch.cat(feature_list, dim=0)
        
        return features
    
    # def train(self, features, labels, epochs=100, batch_size=256, validation_split=0.2, 
    #           verbose=True, early_stopping_patience=10):
    #     """
    #     Train the LLR network.
        
    #     Parameters:
    #     -----------
    #     features : torch.Tensor
    #         Input features of shape (N, feature_dim)
    #     labels : torch.Tensor
    #         Binary labels of shape (N,)
    #     epochs : int
    #         Number of training epochs
    #     batch_size : int
    #         Batch size for training
    #     validation_split : float
    #         Fraction of data to use for validation
    #     verbose : bool
    #         Whether to print training progress
    #     early_stopping_patience : int
    #         Number of epochs to wait for improvement before early stopping
            
    #     Returns:
    #     --------
    #     dict : Training history with 'train_loss' and 'val_loss' keys
    #     """
    #     # Build network if not already built
    #     if self.mlp_branches is None and self.shared_branch_mlp is None:
    #         self._build_network(features.shape[1])
        
    #     # Split data into train and validation
    #     n_samples = len(features)
    #     n_val = int(n_samples * validation_split)
    #     n_train = n_samples - n_val
        
    #     # Shuffle data
    #     perm = torch.randperm(n_samples)
    #     features = features[perm]
    #     labels = labels[perm]
        
    #     train_features = features[:n_train]
    #     train_labels = labels[:n_train]
    #     val_features = features[n_train:]
    #     val_labels = labels[n_train:]
        
    #     # Training loop
    #     best_val_loss = float('inf')
    #     patience_counter = 0
        
    #     for epoch in range(epochs):
    #         # Training phase
    #         if self.shared_mlp:
    #             self.shared_branch_mlp.train()
    #         else:
    #             for branch in self.mlp_branches:
    #                 branch.train()
    #         self.final_mlp.train()
    #         if self.fourier_features_list is not None:
    #             for fourier_layer in self.fourier_features_list:
    #                 fourier_layer.train()
                
    #         train_loss = 0.0
    #         n_batches = 0
            
    #         for i in range(0, n_train, batch_size):
    #             batch_features = train_features[i:i+batch_size]
    #             batch_labels = train_labels[i:i+batch_size]
                
    #             self.optimizer.zero_grad()
    #             outputs = self._forward_pass(batch_features)
    #             loss = self.loss_fn(outputs, batch_labels)
    #             loss.backward()
    #             self.optimizer.step()
                
    #             train_loss += loss.item()
    #             n_batches += 1
            
    #         train_loss /= n_batches
            
    #         # Validation phase
    #         if self.shared_mlp:
    #             self.shared_branch_mlp.eval()
    #         else:
    #             for branch in self.mlp_branches:
    #                 branch.eval()
    #         self.final_mlp.eval()
    #         if self.fourier_features_list is not None:
    #             for fourier_layer in self.fourier_features_list:
    #                 fourier_layer.eval()
                
    #         with torch.no_grad():
    #             val_outputs = self._forward_pass(val_features)
    #             val_loss = self.loss_fn(val_outputs, val_labels).item()
            
    #         # Store history
    #         self.train_losses.append(train_loss)
    #         self.val_losses.append(val_loss)
            
    #         # Early stopping check
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             patience_counter = 0
    #             # Save best model state
    #             self.best_state_dict = {
    #                 'mlp_branches': [branch.state_dict().copy() for branch in self.mlp_branches] if not self.shared_mlp else None,
    #                 'shared_branch_mlp': self.shared_branch_mlp.state_dict().copy() if self.shared_mlp else None,
    #                 'final_mlp': self.final_mlp.state_dict().copy(),
    #                 'fourier_features_list': [fourier_layer.state_dict().copy() for fourier_layer in self.fourier_features_list] if self.fourier_features_list is not None else None
    #             }
    #         else:
    #             patience_counter += 1
            
    #         if verbose and (epoch + 1) % 10 == 0:
    #             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
    #         # Early stopping
    #         if patience_counter >= early_stopping_patience:
    #             if verbose:
    #                 print(f"Early stopping at epoch {epoch+1}")
    #             # Load best model
    #             if self.shared_mlp:
    #                 self.shared_branch_mlp.load_state_dict(self.best_state_dict['shared_branch_mlp'])
    #             else:
    #                 for i, branch in enumerate(self.mlp_branches):
    #                     branch.load_state_dict(self.best_state_dict['mlp_branches'][i])
    #             self.final_mlp.load_state_dict(self.best_state_dict['final_mlp'])
    #             if self.fourier_features_list is not None and self.best_state_dict['fourier_features_list'] is not None:
    #                 for i, fourier_layer in enumerate(self.fourier_features_list):
    #                     fourier_layer.load_state_dict(self.best_state_dict['fourier_features_list'][i])
    #             break
        
    #     self.is_trained = True
        
    #     return {
    #         'train_loss': self.train_losses,
    #         'val_loss': self.val_losses
    #     }
        
    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                             verbose=True, early_stopping_patience=10):
        """
        Train the LLR network using PyTorch DataLoader.
        
        This method is designed to work with EventDataset that dynamically generates
        events from different detector points. Each batch from the DataLoader contains
        events from a single detector point.
        
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
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        # Build network if not already built
        # We need to get a sample to determine the feature dimension
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            sample_batch = next(iter(train_dataloader))
            sample_features, _ = sample_batch
            # Features are now (batch_size, feature_dim) since each sample is a single event
            feature_dim = sample_features.shape[1]
            self._build_network(feature_dim)
        
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
            
            # Iterate through batches of single events
            for batch_features, batch_labels in train_dataloader:
                # Move to device (no need to flatten since each sample is already a single event)
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self._forward_pass(batch_features)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
                
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
                        # Move to device (no need to flatten since each sample is already a single event)
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
            
            if verbose and (epoch + 1) % 10 == 0:
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
            
        with torch.no_grad():
            probabilities = self._forward_pass(points)
            
            if return_probabilities:
                return probabilities
            else:
                # Convert probabilities to LLR: log(p/(1-p))
                epsilon = 1e-7  # Small value to prevent log(0)
                prob_clamped = torch.clamp(probabilities, epsilon, 1 - epsilon)
                return torch.log(prob_clamped / (1 - prob_clamped))

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
            
        with torch.no_grad():
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
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")
        
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
            'shared_mlp': self.shared_mlp
        }
        
        if self.fourier_features_list is not None:
            save_dict['fourier_features_list_state_dict'] = [fourier.state_dict() for fourier in self.fourier_features_list]
        else:
            save_dict['fourier_features_list_state_dict'] = None
            
        torch.save(save_dict, filepath)
    
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
        
        self.is_trained = True

    class EventDataset(Dataset):
        """
        Simplified PyTorch Dataset for sampling background and signal events for LLR training.
        
        This dataset dynamically generates single events using a ToySampler, calculates light yield
        using surrogate functions, and provides proper labels for training. Each sample 
        represents a single event with a randomly sampled detector point.
        
        This inner class has access to the parent LLRnet's prepare_data_from_raw method.
        """

        def __init__(self, llrnet_instance, signal_sampler, background_sampler, signal_surrogate_func, background_surrogate_func,
                     num_samples_per_epoch=1000, signal_fraction=0.5,
                     event_labels=['position', 'energy', 'zenith', 'azimuth']):
            """
            Initialize the EventDataset.
            
            Parameters:
            -----------
            llrnet_instance : LLRnet
                Reference to the parent LLRnet instance to access prepare_data_from_raw
            sampler : ToySampler
                Sampler instance for generating event parameters
            signal_surrogate_func : callable
                Function to calculate light yield for signal events
            background_surrogate_func : callable  
                Function to calculate light yield for background events
            num_samples_per_epoch : int
                Total number of samples (events) to generate per epoch
            signal_fraction : float
                Fraction of events that should be signal (rest are background)
            event_labels : list
                List of event parameter keys to include as features
            """
            
            self.llrnet = llrnet_instance
            self.signal_sampler = signal_sampler
            self.background_sampler = background_sampler
            self.signal_surrogate_func = signal_surrogate_func
            self.background_surrogate_func = background_surrogate_func
            self.num_samples_per_epoch = num_samples_per_epoch
            self.signal_fraction = signal_fraction
            self.event_labels = event_labels
              # Fixed noise scale for signal
            
            # Calculate number of signal and background samples
            self.num_signal_samples = int(num_samples_per_epoch * signal_fraction)
            self.num_background_samples = num_samples_per_epoch - self.num_signal_samples
            
        def __len__(self):
            """Return the number of samples per epoch."""
            return self.num_samples_per_epoch
        
        def __getitem__(self, idx):
            """
            Generate a single event with a randomly sampled detector point.
            
            Parameters:
            -----------
            idx : int
                Sample index (used to determine if this should be signal or background)
            
            Returns:
            --------
            tuple : (features, label)
                features: torch.Tensor of shape (feature_dim,) for single event
                label: torch.Tensor scalar (0 for background, 1 for signal)
            """
            # Sample a random detector point for this event
            
            
            # Determine if this should be a signal or background event
            # Use deterministic assignment based on index to ensure exact signal_fraction
            is_signal = idx < self.num_signal_samples
            
            if is_signal:
                detector_point = self.signal_sampler.sample_detector_points(1).squeeze()
                # Generate a signal event
                event_data = self.signal_sampler.sample_events(1, 'signal')[0]
                event_features = self.llrnet.prepare_data_from_raw(
                    detector_point, event_data, self.signal_surrogate_func, self.event_labels, self.llrnet.signal_noise_scale, self.llrnet.add_relative_pos
                )
                label = torch.tensor(1.0, device=self.llrnet.device)
            else:
                detector_point = self.background_sampler.sample_detector_points(1).squeeze()
                # Generate a background event
                event_data = self.background_sampler.sample_events(1, 'background')[0]
                event_features = self.llrnet.prepare_data_from_raw(
                    detector_point, event_data, self.background_surrogate_func, self.event_labels, self.llrnet.background_noise_scale, self.llrnet.add_relative_pos
                )
                label = torch.tensor(0.0, device=self.llrnet.device)
            
            return event_features, label

    def create_event_dataloader(self, signal_sampler, background_sampler, signal_surrogate_func, background_surrogate_func,
                               num_samples_per_epoch=1000, signal_fraction=0.5, batch_size=32, 
                               shuffle=True, num_workers=0,
                               event_labels=['position', 'energy', 'zenith', 'azimuth']):
        """
        Create a DataLoader for event-based training using the simplified inner EventDataset class.
        
        This method creates an EventDataset that generates single events with randomly
        sampled detector points, making the training process much simpler.
        
        Parameters:
        -----------
        sampler : ToySampler
            Sampler instance for generating event parameters
        signal_surrogate_func : callable
            Function to calculate light yield for signal events  
        background_surrogate_func : callable
            Function to calculate light yield for background events
        num_samples_per_epoch : int
            Total number of samples (events) to generate per epoch
        signal_fraction : float
            Fraction of events that should be signal (rest are background)
        batch_size : int
            Number of events per batch (now can be larger since each sample is a single event)
        shuffle : bool
            Whether to shuffle the events
        num_workers : int
            Number of worker processes for data loading
        event_labels : list
            List of event parameter keys to include as features
            
        Returns:
        --------
        torch.utils.data.DataLoader
            Configured DataLoader for training
            
        Example:
        --------
        >>> # Create model and sampler
        >>> model = LLRnet(dim=3, domain_size=2, device=device)
        >>> sampler = ToySampler(device=device, dim=3, domain_size=2)
        >>> 
        >>> # Create DataLoader using simplified method
        >>> train_loader = model.create_event_dataloader(
        ...     sampler=sampler,
        ...     signal_surrogate_func=signal_func,
        ...     background_surrogate_func=background_func,
        ...     num_samples_per_epoch=5000,  # 5000 events per epoch
        ...     signal_fraction=0.6,         # 60% signal, 40% background
        ...     batch_size=64                # 64 events per batch
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
            signal_fraction=signal_fraction,
            event_labels=event_labels
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Speed up GPU transfers
        )
        
        return dataloader
