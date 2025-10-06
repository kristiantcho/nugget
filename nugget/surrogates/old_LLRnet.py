from nugget.surrogates.base_surrogate import Surrogate
import torch
import numpy as np


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
                 shared_mlp=False, use_residual_connections=False):
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
        
    def prepare_data_from_functions(self, points, signal_funcs, background_funcs, 
                                  n_samples_per_func=1000, signal_ratio=0.5):
        """
        Prepare training data from provided signal and background functions.
        
        Parameters:
        -----------
        points : torch.Tensor
            Points to evaluate functions at (N, dim)
        signal_funcs : list
            List of signal functions
        background_funcs : list
            List of background functions
        n_samples_per_func : int
            Number of samples to generate per function
        signal_ratio : float
            Ratio of signal to total samples
            
        Returns:
        --------
        tuple : (features, labels)
            features: torch.Tensor of shape (total_samples, feature_dim)
            labels: torch.Tensor of shape (total_samples,)
        """
        all_features = []
        all_labels = []
        
        n_signal_samples = int(n_samples_per_func * signal_ratio)
        n_background_samples = n_samples_per_func - n_signal_samples
        
        # Generate signal samples
        for signal_func in signal_funcs:
            # Sample random points
            sample_indices = torch.randint(0, len(points), (n_signal_samples,))
            sample_points = points[sample_indices]
            
            # Evaluate signal function
            signal_values = signal_func(sample_points)
            
            # Create features (points + signal values + additional features)
            features = torch.cat([
                sample_points,
                signal_values.unsqueeze(-1) if signal_values.dim() == 1 else signal_values
            ], dim=1)
            
            all_features.append(features)
            all_labels.append(torch.ones(n_signal_samples, device=self.device))
        
        # Generate background samples
        for background_func in background_funcs:
            # Sample random points
            sample_indices = torch.randint(0, len(points), (n_background_samples,))
            sample_points = points[sample_indices]
            
            # Evaluate background function
            background_values = background_func(sample_points)
            
            # Create features (points + background values + additional features)
            features = torch.cat([
                sample_points,
                background_values.unsqueeze(-1) if background_values.dim() == 1 else background_values
            ], dim=1)
            
            all_features.append(features)
            all_labels.append(torch.zeros(n_background_samples, device=self.device))
        
        # Combine all data
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Shuffle the data
        perm = torch.randperm(len(features))
        features = features[perm]
        labels = labels[perm]
        
        return features, labels
    
    def prepare_data_from_raw(self, event_data, labels, additional_features=None):
        """
        Prepare training data from raw neutrino event data.
        
        Parameters:
        -----------
        event_data : torch.Tensor or dict
            Raw event data. Can be:
            - Tensor of shape (N, feature_dim) for direct feature input
            - Dict with keys like 'positions', 'light_yield', 'energy', etc.
        labels : torch.Tensor
            Binary labels (0 for background, 1 for signal)
        additional_features : torch.Tensor or None
            Additional engineered features to include
            
        Returns:
        --------
        tuple : (features, labels)
            features: torch.Tensor of shape (N, feature_dim)
            labels: torch.Tensor of shape (N,)
        """
        if isinstance(event_data, dict):
            # Extract features from dictionary
            feature_list = []
            
            # Add position features if available
            if 'positions' in event_data:
                positions = event_data['positions']
                if not isinstance(positions, torch.Tensor):
                    positions = torch.tensor(positions, device=self.device, dtype=torch.float32)
                feature_list.append(positions)
            
            # Add light yield features if available
            if 'light_yield' in event_data:
                light_yield = event_data['light_yield']
                if not isinstance(light_yield, torch.Tensor):
                    light_yield = torch.tensor(light_yield, device=self.device, dtype=torch.float32)
                if light_yield.dim() == 1:
                    light_yield = light_yield.unsqueeze(-1)
                feature_list.append(light_yield)
            
            # # Add energy features if available
            # if 'energy' in event_data:
            #     energy = event_data['energy']
            #     if not isinstance(energy, torch.Tensor):
            #         energy = torch.tensor(energy, device=self.device, dtype=torch.float32)
            #     if energy.dim() == 1:
            #         energy = energy.unsqueeze(-1)
            #     feature_list.append(energy)
            
            # # Add timing features if available
            # if 'timing' in event_data:
            #     timing = event_data['timing']
            #     if not isinstance(timing, torch.Tensor):
            #         timing = torch.tensor(timing, device=self.device, dtype=torch.float32)
            #     if timing.dim() == 1:
            #         timing = timing.unsqueeze(-1)
            #     feature_list.append(timing)
            
            # Combine all features
            features = torch.cat(feature_list, dim=1)
            
        else:
            # Direct tensor input
            features = event_data
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, device=self.device, dtype=torch.float32)
        
        # Add additional engineered features if provided
        if additional_features is not None:
            if not isinstance(additional_features, torch.Tensor):
                additional_features = torch.tensor(additional_features, device=self.device, dtype=torch.float32)
            features = torch.cat([features, additional_features], dim=1)
        
        # Ensure labels are tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device, dtype=torch.float32)
        
        return features, labels
    
    def train(self, features, labels, epochs=100, batch_size=256, validation_split=0.2, 
              verbose=True, early_stopping_patience=10):
        """
        Train the LLR network.
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features of shape (N, feature_dim)
        labels : torch.Tensor
            Binary labels of shape (N,)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : bool
            Whether to print training progress
        early_stopping_patience : int
            Number of epochs to wait for improvement before early stopping
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        # Build network if not already built
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            self._build_network(features.shape[1])
        
        # Split data into train and validation
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle data
        perm = torch.randperm(n_samples)
        features = features[perm]
        labels = labels[perm]
        
        train_features = features[:n_train]
        train_labels = labels[:n_train]
        val_features = features[n_train:]
        val_labels = labels[n_train:]
        
        # Training loop
        best_val_loss = float('inf')
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
            
            for i in range(0, n_train, batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self._forward_pass(batch_features)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation phase
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
                val_outputs = self._forward_pass(val_features)
                val_loss = self.loss_fn(val_outputs, val_labels).item()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state_dict = {
                    'mlp_branches': [branch.state_dict().copy() for branch in self.mlp_branches] if not self.shared_mlp else None,
                    'shared_branch_mlp': self.shared_branch_mlp.state_dict().copy() if self.shared_mlp else None,
                    'final_mlp': self.final_mlp.state_dict().copy(),
                    'fourier_features_list': [fourier_layer.state_dict().copy() for fourier_layer in self.fourier_features_list] if self.fourier_features_list is not None else None
                }
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
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
            'val_loss': self.val_losses
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
        
        return final_output.squeeze()
    
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
    
    def predict_log_likelihood_ratio(self, points):
        """
        Compute the Log-Likelihood Ratio using the sigmoid trick.
        
        This method computes log(p/(1-p)) where p is the output probability from the network.
        
        Parameters:
        -----------
        points : torch.Tensor
            Input points/features to evaluate
            
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
            probabilities = self._forward_pass(points)
            
            # Compute LLR: log(p/(1-p))
            # epsilon = 1e-7  # Small value to prevent log(0)
            # prob_clamped = torch.clamp(probabilities, epsilon, 1 - epsilon)
            # llr = torch.log(prob_clamped / (1 - prob_clamped))
            llr = torch.log(probabilities + 1e-10) - torch.log(1 - (probabilities + 1e-10))
            return llr
    
    def predict_likelihood_ratio(self, points):
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
        log_ratios = self.predict_log_likelihood_ratio(points)
        return torch.exp(log_ratios)
    
    def predict_proba(self, points):
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
        return self.__call__(points, return_probabilities=True)
    
    def predict(self, points, threshold=0.5):
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
        probabilities = self.predict_proba(points)
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
    
    def generate_batch_data_from_events(self, event_indices=None, optimization_points=None, signal_surrogate_func=None, 
                                       background_surrogate_func=None, signal_event_params=None,
                                       background_event_params=None, batch_size=256, 
                                       signal_ratio=0.5, add_noise=True, sig_noise_scale=0.1,
                                       bkg_noise_scale = 0.2, random_seed=None, 
                                       samples_per_epoch=10, keep_opt_point=False, sr_mode=False, output_denoised=False, 
                                       return_event_params=False):
        """
        Generate training data on-the-fly from neutrino event parameters for batch training.
        
        Features include neutrino event data remapped relative to optimization points:
        - Relative position coordinates (event center - optimization point)
        - Relative incoming angles (zenith, azimuth)
        - Light yield values computed from surrogate functions (with optional noise)
        - Other event parameters
        
        The optimization point coordinates themselves are NOT included in features.
        
        Parameters:
        -----------
        event_indices : list or torch.Tensor or None
            Indices of events to use. If None, uses all available events.
        optimization_points : torch.Tensor or None
            Points at which to evaluate/optimize, shape (N, dim). If None, 
            random points will be sampled from the domain for each batch.
        signal_surrogate_func : callable or None
            Function that takes event parameters and optimization point, returns light yield
            Signature: light_yield = func(event_params, opt_point)
            where event_params is a dict with keys like 'position', 'zenith', 'azimuth', 'energy', etc.
        background_surrogate_func : callable or None
            Similar function for background events
        signal_event_params : dict or None
            Dictionary containing signal event parameters:
            {
                'positions': torch.Tensor, shape (n_signal_events, dim) - event centers
                'zenith_angles': torch.Tensor, shape (n_signal_events,) - incoming zenith angles
                'azimuth_angles': torch.Tensor, shape (n_signal_events,) - incoming azimuth angles  
                'energies': torch.Tensor, shape (n_signal_events,) - event energies (optional)
                'other_features': torch.Tensor, shape (n_signal_events, n_features) - additional features (optional)
            }
        background_event_params : dict or None
            Similar dictionary for background events
        batch_size : int
            Size of each training batch
        signal_ratio : float
            Fraction of signal events in each batch (0 to 1)
        add_noise : bool
            Whether to add noise to light yield values
        sig_noise_scale : float
            Standard deviation of Gaussian noise to add to signal light yields
        bkg_noise_scale : float
            Standard deviation of Gaussian noise to add to background light yields
        random_seed : int or None
            Random seed for reproducibility
        samples_per_event : int
            Number of optimization points to sample per event in each batch
        return_event_params : bool
            Whether to return the original event parameters for selected events
            
        Returns:
        --------
        torch.utils.data.DataLoader
            DataLoader yielding batches of:
            - If return_event_params=False and output_denoised=False: (features, labels)
            - If return_event_params=False and output_denoised=True: (features, labels, denoised_yield)
            - If return_event_params=True and output_denoised=False: (features, labels, event_params_batch)
            - If return_event_params=True and output_denoised=True: (features, labels, denoised_yield, event_params_batch)
            
            Features shape: (batch_size, feature_dim)
            Labels shape: (batch_size,) - binary (0=background, 1=signal)
            event_params_batch: list of dicts with original event parameters corresponding to each sample
                               - Each dict contains the original event parameters for the selected event
                               - The order corresponds directly to the labels (label[i] corresponds to event_params_batch[i])
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Validate inputs - need either surrogate functions or direct light yields
        if signal_surrogate_func is None and background_surrogate_func is None:
            if signal_event_params is None and background_event_params is None:
                raise ValueError("Must provide either surrogate functions or event parameters with light yields")
            # Check if light yields are provided in event params
            if signal_event_params and 'light_yields' not in signal_event_params:
                raise ValueError("signal_event_params must contain 'light_yields' if no signal_surrogate_func provided")
            if background_event_params and 'light_yields' not in background_event_params:
                raise ValueError("background_event_params must contain 'light_yields' if no background_surrogate_func provided")
        
        if signal_event_params is None and background_event_params is None:
            raise ValueError("At least one of signal_event_params or background_event_params must be provided")
        
        # Convert optimization points to tensor if provided and ensure float32
        # if optimization_points is not None:
        #     if not isinstance(optimization_points, torch.Tensor):
        #         optimization_points = torch.tensor(optimization_points, device=self.device, dtype=torch.float32)
        #     else:
        #         optimization_points = optimization_points.float().to(self.device)
        
        # Handle event indices - determine which events to use
        total_signal_events = len(signal_event_params['position']) if signal_event_params else 0
        total_background_events = len(background_event_params['position']) if background_event_params else 0
        
        if event_indices is None:
            # Use all events
            signal_indices = list(range(total_signal_events)) if total_signal_events > 0 else []
            background_indices = list(range(total_background_events)) if total_background_events > 0 else []
        else:
            # Filter provided indices by signal/background
            if isinstance(event_indices, torch.Tensor):
                event_indices = event_indices.tolist()
            
            # For simplicity, assume event_indices refers to combined pool
            # In practice, you might want to pass separate signal_indices and background_indices
            # total_events = total_signal_events + total_background_events
            # signal_indices = [i for i in event_indices if i < total_signal_events]
            # background_indices = [i - total_signal_events for i in event_indices 
            #                     if i >= total_signal_events and i < total_events]
        
        class EventDataset(torch.utils.data.Dataset):
            def __init__(self, sig_indices, bg_indices, opt_points, sig_func, bg_func, sig_params, bg_params, sig_ratio, add_noise, sig_noise_scale, bkg_noise_scale, device, dim, samples_per_epoch, domain_size, keep_opt_point=False, sr_mode=False, output_denoised=False, return_event_params=False):
                self.sig_indices = sig_indices
                self.bg_indices = bg_indices
                self.opt_points = opt_points
                self.sig_surrogate_func = sig_func
                self.bg_surrogate_func = bg_func
                self.sig_params = sig_params
                self.bg_params = bg_params
                self.sig_ratio = sig_ratio
                self.add_noise = add_noise
                self.sig_noise_scale = sig_noise_scale
                self.bkg_noise_scale = bkg_noise_scale
                self.device = device
                self.dim = dim
                self.samples_per_epoch = samples_per_epoch
                self.domain_size = domain_size
                self.keep_opt_point = keep_opt_point
                self.sr_mode = sr_mode
                self.return_event_params = return_event_params
                # Calculate total number of available event-optimization point combinations
                # self.total_combinations = (len(self.sig_indices) + len(self.bg_indices)) * self.samples_per_event
                self.total_combinations = samples_per_epoch
                self.output_denoised = output_denoised
                # Determine feature dimension
                self.feature_dim = self._calculate_feature_dim()
                
                
                
            def _calculate_feature_dim(self):
                """Calculate the total feature dimension."""
                # Base features: relative position (dim) + relative angles (2) + light_yield (1)
                if not self.sr_mode:
                    feature_dim = self.dim + 2 + 1
                    if self.keep_opt_point:
                        feature_dim += self.dim
                    
                    # Add energy if available
                    if (self.sig_params and 'energy' in self.sig_params) or \
                    (self.bg_params and 'energy' in self.bg_params):
                        feature_dim += 1
                else:
                    feature_dim = 6
                
                # Add other features if available
                if self.sig_params and 'other_features' in self.sig_params or \
                   self.bg_params and 'other_features' in self.bg_params:
                    feature_dim += self.sig_params['other_features'].shape[1]
                    
                
                return feature_dim
                
            def __len__(self):
                return self.total_combinations
                
            def __getitem__(self, idx):
                """Generate a training sample by sampling an event and optimization point."""
                # Calculate which event and which sample within that event
                # total_events = len(self.sig_indices) + len(self.bg_indices)
                # event_group_idx = idx // self.samples_per_event
                # sample_within_event = idx % self.samples_per_event
                # event_idx = idx // self.samples_per_event
                # opt_point_sample = idx % self.samples_per_event
                
                # Decide if this should be a signal or background sample based on ratio
                is_signal = torch.rand(1).item() < self.sig_ratio
                
                # Sample an event
                if is_signal and len(self.sig_indices) > 0:
                    event_idx = torch.randint(0, len(self.sig_indices), (1,)).item()
                    actual_event_idx = self.sig_indices[event_idx]
                    background = False
                elif not is_signal and len(self.bg_indices) > 0:
                    event_idx = torch.randint(0, len(self.bg_indices), (1,)).item()
                    actual_event_idx = self.bg_indices[event_idx]
                    background = True
                
                # Sample an optimization point (this is where samples_per_event matters)
                if self.opt_points is not None:
                    # Sample from provided optimization points
                    opt_idx = torch.randint(0, len(self.opt_points), (1,)).item()
                    opt_point = torch.tensor(self.opt_points[opt_idx], device=self.device, dtype=torch.float32)
                else:
                    # Generate random optimization point from domain
                    # Use sample_within_event as part of random seed for reproducibility within event
                    # torch.manual_seed(torch.initial_seed() + idx)
                    opt_point = torch.rand(self.dim, device=self.device, dtype=torch.float32) * self.domain_size - self.domain_size/2
                
                # Generate sample data
                if not self.output_denoised:
                    features, label = self._generate_sample(opt_point, actual_event_idx, background)
                    if not self.return_event_params:
                        return features, label
                    else:
                        # Extract event parameters for the selected event
                        if background:
                            selected_event_params = self._extract_event_params(self.bg_params, actual_event_idx)
                        else:
                            selected_event_params = self._extract_event_params(self.sig_params, actual_event_idx)
                        return features, label, selected_event_params
                else:
                    features, label, denoised = self._generate_sample(opt_point, actual_event_idx, background)
                    if not self.return_event_params:
                        return features, label, denoised
                    else:
                        # Extract event parameters for the selected event
                        if background:
                            selected_event_params = self._extract_event_params(self.bg_params, actual_event_idx)
                        else:
                            selected_event_params = self._extract_event_params(self.sig_params, actual_event_idx)
                        return features, label, denoised, selected_event_params
                    
            
                
            def _extract_event_params(self, event_params, event_idx):
                """Extract event parameters for a specific event index."""
                if event_params is None:
                    return {}
                
                extracted_params = {}
                for key, values in event_params.items():
                    if isinstance(values, torch.Tensor):
                        extracted_params[key] = values[event_idx].clone()
                    elif isinstance(values, (list, tuple)):
                        extracted_params[key] = values[event_idx]
                    else:
                        # For scalar values, just copy
                        extracted_params[key] = values
                        
                return extracted_params
                
            def _generate_sample(self, opt_point, event_idx, background=False):
                """Generate a surrogate sample relative to the optimization point."""
                # Randomly select a background event
                if background:
                    sample_event_params = self.bg_params
                    surrogate_func = self.bg_surrogate_func
                    noise_scale = self.bkg_noise_scale
                else:                    
                    sample_event_params = self.sig_params
                    surrogate_func = self.sig_surrogate_func
                    noise_scale = self.sig_noise_scale
                # n_events = len(sample_event_params['positions'])
                # event_idx = torch.randint(0, n_events, (1,)).item()
                denoised = None
                # Compute light yield using surrogate function or use provided value
                if surrogate_func is not None:
                    light_yield = surrogate_func(sample_event_params, opt_point, event_idx)
                    if not isinstance(light_yield, torch.Tensor):
                        light_yield = torch.tensor(light_yield, device=self.device, dtype=torch.float32)
                    else:
                        light_yield = light_yield.float()  # Ensure float32
                else:
                    # Use pre-computed light yield
                    light_yield = sample_event_params['light_yield'][event_idx].clone().float()
                
                # Add noise to light yield if requested
                if self.add_noise:
                    noise = torch.randn(1, device=self.device, dtype=torch.float32) * noise_scale
                    denoised = light_yield.clone()
                    light_yield = light_yield + light_yield*noise
                
                # Create relative features
                features = self._create_relative_features(
                    opt_point, light_yield,
                    sample_event_params, event_idx
                )
                if background:
                    if features.dim() > 1:
                        labels = torch.zeros(features.shape[0], device=self.device, dtype=torch.float32)  # Background label
                    else:
                        labels = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                else:
                    if features.dim() > 1:
                        labels = torch.ones(features.shape[0], device=self.device, dtype=torch.float32)  # Signal label
                    else:
                        labels = torch.tensor(1.0, device=self.device, dtype=torch.float32)

                if not self.output_denoised:
                    return features, labels
                else:
                    # Return features, labels, and denoised light yield
                    return features, labels, denoised

            def _create_relative_features(self, opt_point, light_yield, event_params, event_idx):
                """Create feature vector with relative positioning and angles.
                Order or features:
                1. Relative position (event center - optimization point)
                1.5. Optionally include optimization point itself
                2. Relative zenith and azimuth angles
                3. Energy (if available)
                4. Light yield (with optional noise)
                5. Other features (if available)"""
                feature_list = []
                
                # Ensure all inputs are float32
                event_pos = event_params['position'][event_idx].float()
                zenith = event_params['zenith'][event_idx].float()
                azimuth = event_params['azimuth'][event_idx].float()
                opt_point = opt_point.float()
                light_yield = light_yield.float()
                energy = event_params.get('energy', None)
                if energy is not None:
                    energy = event_params['energy'][event_idx].float()
                
                # Relative position: event_center - optimization_point
                # if not self.sr_mode:
                relative_pos = event_pos - opt_point
                feature_list.append(relative_pos)
                if self.keep_opt_point:
                    # Optionally include the optimization point itself
                    feature_list.append(opt_point)
                
                # Relative angles (zenith and azimuth)
                # Note: These are already in the event coordinate system
                # Could be modified to be relative to optimization point if needed
                feature_list.append(torch.tensor([zenith, azimuth], device=self.device, dtype=torch.float32))
                if energy is not None:
                    energy_feature = energy.unsqueeze(0) if energy.dim() == 0 else energy
                    feature_list.append(energy_feature.float())
                # else:
                #     sr_model = SymbolicReg(num_rotations=1)
                    
                #     # dist = torch.norm(event_pos - opt_point)   
                #     direction = torch.tensor([
                #         torch.sin(zenith) * torch.cos(azimuth),
                #         torch.sin(zenith) * torch.sin(azimuth),
                #         torch.cos(zenith)
                #     ], device=self.device)
                #     input_buffer = sr_model._create_model_input(event_pos, direction, energy, opt_point)
                #     for feature in input_buffer:
                #         feature_list.append(feature.float())
                # Light yield (possibly with noise)
                light_yield_feature = light_yield.unsqueeze(0) if light_yield.dim() == 0 else light_yield
                feature_list.append(light_yield_feature.float())
                
                # Add energy if available
                # if energy is not None:
                #     feature_list.append(energy_feature.float())

                # Add other features if available
                if 'other_features' in event_params:
                    other_feats = event_params['other_features'][event_idx].float()
                    feature_list.append(other_feats)
                
                # Concatenate all features - all should now be float32
                features = torch.cat(feature_list, dim=0)
                
                return features
        
        # Create dataset
        dataset = EventDataset(
            signal_indices, background_indices, optimization_points, 
            signal_surrogate_func, background_surrogate_func,
            signal_event_params, background_event_params,
            signal_ratio, add_noise, sig_noise_scale, bkg_noise_scale, 
            self.device, self.dim, samples_per_epoch, self.domain_size, keep_opt_point, 
            sr_mode, output_denoised, return_event_params
        )
        
        # Create DataLoader with appropriate collate function based on return structure
        def collate_batch(batch):
            """Custom collate function to handle different return structures."""
            if not return_event_params and not output_denoised:
                # Standard: (features, labels)
                return (
                    torch.stack([item[0] for item in batch]),
                    torch.stack([item[1] for item in batch])
                )
            elif not return_event_params and output_denoised:
                # With denoised: (features, labels, denoised)
                return (
                    torch.stack([item[0] for item in batch]),
                    torch.stack([item[1] for item in batch]),
                    torch.stack([item[2] for item in batch])
                )
            elif return_event_params and not output_denoised:
                # With event params: (features, labels, event_params)
                features = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
                
                # Collect event params - each item[2] is a single event parameter dict
                event_params_batch = [item[2] for item in batch]
                
                return features, labels, event_params_batch
            else:
                # With both denoised and event params: (features, labels, denoised, event_params)
                features = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
                denoised = torch.stack([item[2] for item in batch])
                
                # Collect event params - each item[3] is a single event parameter dict
                event_params_batch = [item[3] for item in batch]
                
                return features, labels, denoised, event_params_batch
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_batch
        )
        
        return dataloader
        
        # Example usage with return_event_params=True:
        # dataloader = generate_batch_data_from_events(..., return_event_params=True)
        # for batch_data in dataloader:
        #     if output_denoised:
        #         features, labels, denoised, event_params_batch = batch_data
        #     else:
        #         features, labels, event_params_batch = batch_data
        #     # event_params_batch is a list of dicts with length equal to batch_size
        #     # Each dict contains the original event parameters for the corresponding sample
        #     # The order corresponds directly to labels: event_params_batch[i] goes with labels[i]
    
    def train_with_event_data(self, optimization_points=None, signal_surrogate_func=None,
                            background_surrogate_func=None, signal_event_params=None,
                            background_event_params=None, epochs=100, batch_size=256,
                            signal_ratio=0.5, add_noise=True, signal_noise_scale=0.1,
                            background_noise_scale=0.2,
                            validation_split=0.2, verbose=True, early_stopping_patience=10,
                            random_seed=None, early_stopping=True, samples_per_epoch=10, keep_opt_point=False, sr_mode=False):
        """
        Train the LLR network using on-the-fly generated data from neutrino events.
        
        This version splits the neutrino events for train/validation rather than 
        optimization points. Each batch samples random optimization points and 
        random events to create training samples.
        
        Parameters:
        -----------
        optimization_points : torch.Tensor or None
            Points at which to evaluate/optimize, shape (N, dim). If None,
            random points will be sampled from the domain for each batch.
        signal_surrogate_func : callable or None
            Function that takes event parameters and optimization point, returns light yield for signal events
        background_surrogate_func : callable or None
            Function that takes event parameters and optimization point, returns light yield for background events
        signal_event_params : dict or None
            Dictionary containing signal event parameters (see generate_batch_data_from_events)
        background_event_params : dict or None
            Dictionary containing background event parameters
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        signal_ratio : float
            Fraction of signal events in each batch
        add_noise : bool
            Whether to add noise to light yield values
        signal_noise_scale : float
            Standard deviation of noise to add to signal light yields
        background_noise_scale : float
            Standard deviation of noise to add to background light yields
        validation_split : float
            Fraction of neutrino events to use for validation
        verbose : bool
            Whether to print training progress
        early_stopping_patience : int
            Epochs to wait for improvement before early stopping
        random_seed : int or None
            Random seed for reproducibility
        early_stopping : bool
            Whether to use early stopping based on validation loss
        samples_per_event : int
            Number of optimization points to sample per event in each batch
            
        Returns:
        --------
        dict : Training history with 'train_loss' and 'val_loss' keys
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Convert optimization points to tensor if provided and ensure float32
        if optimization_points is not None:
            if not isinstance(optimization_points, torch.Tensor):
                optimization_points = torch.tensor(optimization_points, device=self.device, dtype=torch.float32)
            else:
                optimization_points = optimization_points.float().to(self.device)
        
        # Split neutrino events into train and validation
        # Determine total number of signal and background events
        n_signal_events = len(signal_event_params['position']) if signal_event_params else 0
        n_background_events = len(background_event_params['position']) if background_event_params else 0
        
        # Split signal events
        if n_signal_events > 0:
            n_val_signal = int(n_signal_events * validation_split)
            signal_perm = torch.randperm(n_signal_events)
            train_signal_indices = signal_perm[:-n_val_signal].tolist() if n_val_signal > 0 else signal_perm.tolist()
            val_signal_indices = signal_perm[-n_val_signal:].tolist() if n_val_signal > 0 else []
        else:
            train_signal_indices = []
            val_signal_indices = []
        
        # Split background events
        if n_background_events > 0:
            n_val_background = int(n_background_events * validation_split)
            background_perm = torch.randperm(n_background_events)
            train_background_indices = background_perm[:-n_val_background].tolist() if n_val_background > 0 else background_perm.tolist()
            val_background_indices = background_perm[-n_val_background:].tolist() if n_val_background > 0 else []
        else:
            train_background_indices = []
            val_background_indices = []
        
        # Create filtered event parameters for train and validation
        def filter_event_params(event_params, indices):
            if event_params is None or len(indices) == 0:
                return None
            
            filtered_params = {}
            for key, values in event_params.items():
                if isinstance(values, torch.Tensor):
                    filtered_params[key] = values[indices]
                elif isinstance(values, (list, np.ndarray)):
                    filtered_params[key] = torch.tensor([values[i] for i in indices], 
                                                      device=self.device, dtype=torch.float32)
            return filtered_params
        
        train_signal_params = filter_event_params(signal_event_params, train_signal_indices)
        val_signal_params = filter_event_params(signal_event_params, val_signal_indices)
        train_background_params = filter_event_params(background_event_params, train_background_indices)
        val_background_params = filter_event_params(background_event_params, val_background_indices)
        
        # Create training event indices (just range since we've already filtered the params)
        train_sig_indices = list(range(len(train_signal_indices))) if train_signal_params else []
        train_bg_indices = list(range(len(train_background_indices))) if train_background_params else []
        val_sig_indices = list(range(len(val_signal_indices))) if val_signal_params else []
        val_bg_indices = list(range(len(val_background_indices))) if val_background_params else []
        
        # Create data loaders
        train_loader = self.generate_batch_data_from_events(
            event_indices=None,  # Use all filtered events
            optimization_points=optimization_points,
            signal_surrogate_func=signal_surrogate_func, 
            background_surrogate_func=background_surrogate_func,
            signal_event_params=train_signal_params, 
            background_event_params=train_background_params,
            batch_size=batch_size, signal_ratio=signal_ratio, add_noise=add_noise, 
            sig_noise_scale=signal_noise_scale, bkg_noise_scale=background_noise_scale, 
            random_seed=random_seed, samples_per_epoch=samples_per_epoch, keep_opt_point=keep_opt_point, sr_mode=sr_mode
        )
        
        val_loader = self.generate_batch_data_from_events(
            event_indices=None,  # Use all filtered events
            optimization_points=optimization_points,
            signal_surrogate_func=signal_surrogate_func, 
            background_surrogate_func=background_surrogate_func,
            signal_event_params=val_signal_params, 
            background_event_params=val_background_params,
            batch_size=min(batch_size, len(val_sig_indices) + len(val_bg_indices)) if (val_sig_indices or val_bg_indices) else batch_size, 
            signal_ratio=signal_ratio, add_noise=add_noise, 
            sig_noise_scale=signal_noise_scale, bkg_noise_scale=background_noise_scale, 
            random_seed=random_seed, samples_per_epoch=samples_per_epoch, keep_opt_point=keep_opt_point, sr_mode=sr_mode
        )
        
        # Get a sample batch to determine feature dimension
        sample_batch = next(iter(train_loader))
        raw_feature_dim = sample_batch[0].shape[1]
        
        # Build network if not already built
        if self.mlp_branches is None and self.shared_branch_mlp is None:
            self._build_network(raw_feature_dim)
        
        # Training loop
        best_val_loss = float('inf')
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
            
            for batch_features, batch_labels in train_loader:
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
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    val_outputs = self._forward_pass(batch_features)
                    loss = self.loss_fn(val_outputs, batch_labels)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            val_loss /= n_val_batches if n_val_batches > 0 else 1
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_state_dict = {
                        'mlp_branches': [branch.state_dict().copy() for branch in self.mlp_branches] if not self.shared_mlp else None,
                        'shared_branch_mlp': self.shared_branch_mlp.state_dict().copy() if self.shared_mlp else None,
                        'final_mlp': self.final_mlp.state_dict().copy(),
                        'fourier_features_list': [fourier_layer.state_dict().copy() for fourier_layer in self.fourier_features_list] if self.fourier_features_list is not None else None
                    }
                else:
                    patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping and patience_counter >= early_stopping_patience:
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
            'val_loss': self.val_losses
        }
