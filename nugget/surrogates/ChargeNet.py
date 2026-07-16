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
                 lr_scheduler_min_lr=1e-6, use_rich_features=False, rich_rel_pos_mode=False,
                 add_vertex_distance=True, add_pmt_direction=False, log_charge_scale=4,
                 ly_eps=1e-10):
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
        use_rich_features : bool
            If True, features are built with prepare_features_charge (the rich
            geometric/event feature vector) instead of prepare_data_from_raw.
        rich_rel_pos_mode : bool
            If True, the rich features use only the relative position
            (detector - vertex) instead of both absolute positions.
        add_vertex_distance : bool
            If True, the rich features include the detector-to-vertex distance.
        add_pmt_direction : bool
            If True, the rich features append the hit-PMT direction (a unit
            vector, relative to the optical module) as 3 extra features.
        log_charge_scale : float
            Scale factor applied to log10 of light yield in the rich features.
        ly_eps : float
            Small epsilon value to prevent taking log of zero in scaling.
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
        self.use_rich_features = use_rich_features
        self.rich_rel_pos_mode = rich_rel_pos_mode
        self.add_vertex_distance = add_vertex_distance
        # If True, prepare_features_charge appends the hit-PMT direction (a unit
        # vector, relative to the optical module) as 3 extra features. The
        # direction is read from event_data['pmt_direction'].
        self.add_pmt_direction = add_pmt_direction
        self.log_charge_scale = log_charge_scale
        self.ly_eps = ly_eps
        # All unique PMT directions seen when add_pmt_direction is used, as a
        # (n_unique, 3) tensor. Populated by the light-yield parquet dataset from
        # the geometry CSV, and persisted via save_model / load_model. None until
        # a dataset with add_pmt_direction populates it.
        self.pmt_directions = None

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

    def set_pmt_directions_from_csv(self, geometry_csv_path):
        """Populate self.pmt_directions from a geometry CSV.

        Reads the ``pmt_dir_x/y/z`` columns of the geometry CSV produced by
        ``extract_geom.py`` and stores the unique PMT pointing directions on the
        model as a (n_unique, 3) tensor. This lets a model know the geometry's
        PMT directions without building a dataloader (e.g. at inference time).
        The stored value is persisted via save_model / load_model.

        Parameters
        ----------
        geometry_csv_path : str
            Path to the geometry CSV (with pmt_dir_x/y/z columns).

        Returns
        -------
        torch.Tensor
            The unique PMT directions, shape (n_unique, 3), on self.device.
        """
        import pandas as pd
        df = pd.read_csv(geometry_csv_path)
        missing = {'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'} - set(df.columns)
        if missing:
            raise ValueError(
                f"geometry CSV '{geometry_csv_path}' is missing column(s): {sorted(missing)}"
            )
        dirs = df[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy()
        return self.set_pmt_directions(dirs)

    def set_pmt_directions(self, directions):
        """Set self.pmt_directions to the unique directions in `directions`.

        Parameters
        ----------
        directions : array-like or torch.Tensor
            PMT directions, shape (n, 3). Duplicates are collapsed (rounded to
            6 decimals) so only unique pointing vectors are stored.

        Returns
        -------
        torch.Tensor
            The unique PMT directions, shape (n_unique, 3), on self.device.
        """
        if isinstance(directions, torch.Tensor):
            arr = directions.detach().cpu().numpy()
        else:
            arr = np.asarray(directions)
        arr = arr.reshape(-1, 3)
        unique_dirs = np.unique(np.round(arr, 6), axis=0)
        self.pmt_directions = torch.tensor(
            unique_dirs, device=self.device, dtype=torch.float32
        )
        return self.pmt_directions

    def prepare_features_charge(self, point, event_data, light_yield=None, device=None):
        """
        Build a rich feature vector for the charge (light-yield) regressor.

        This is the regression analogue of LLRnet.prepare_features_charge. Unlike
        the classifier version, the light yield is the TARGET, not an input, so no
        log_ly column is appended here. The ``light_yield`` argument is accepted
        only for call-signature compatibility and is ignored.

        Feature layout (default flags):
          [det_x, det_y, det_z,       normalised detector position     (3)
           v_x,   v_y,   v_z,         normalised vertex position       (3)
           d_x,   d_y,   d_z,         unit direction vector            (3)
           log10(E)/8,                log-scaled energy                (1)
           vert_dist,                 L2(detector - vertex) normalised (1, optional)
           cos_angle,                 cos(direction ∠ vertex→detector) (1)
           dist_perp,                 perp. distance to beam normalised(1, optional)
           pmt_dir_x, pmt_dir_y, pmt_dir_z]  hit-PMT direction         (3, optional)

        The pmt_dir block is included only when self.add_pmt_direction is True,
        in which case event_data must provide 'pmt_direction' (a 3-vector).

        Normalisation uses self.domain_size via _pos_norm_divisor().

        Parameters
        ----------
        point : torch.Tensor or np.ndarray
            Detector position, shape (3,) or (1, 3).
        event_data : dict
            Event parameters. Must contain 'position', 'energy', 'direction'.
        light_yield : ignored
            Accepted for call-signature compatibility; not used.

        Returns
        -------
        features : torch.Tensor, shape (feature_dim,)
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
        log_energy = torch.log10(energy.squeeze() + self.ly_eps) / 8.0  # scalar

        # --- derived geometric scalars ---
        rel = det - vert
        vert_dist = torch.norm(rel)
        cos_angle = torch.dot(direction, rel) / (torch.norm(direction) * vert_dist + 1e-8)

        if self.rich_rel_pos_mode:
            # Use only relative position (detector - vertex) instead of both absolute positions
            feature_values = [
                rel[0], rel[1], rel[2],
                direction[0], direction[1], direction[2],
                log_energy,
            ]
        else:
            feature_values = [
                det[0], det[1], det[2],
                vert[0], vert[1], vert[2],
                direction[0], direction[1], direction[2],
                log_energy,
            ]
        if self.add_vertex_distance:
            feature_values.append(vert_dist)
        feature_values.append(cos_angle)
        if self.add_distance_from_beam:
            track_pos = vert * norm  # back to original scale for distance calculation
            track_dir = direction    # already a unit vector
            _, dist_perp = self.compute_distance_from_beam(point, track_pos, track_dir)
            feature_values.append(dist_perp.reshape(()) / (self.domain_size / 2))
        # --- hit-PMT direction (unit vector relative to the optical module) ---
        if self.add_pmt_direction:
            pmt_dir = event_data['pmt_direction']
            if isinstance(pmt_dir, np.ndarray):
                pmt_dir = torch.tensor(pmt_dir, device=self.device, dtype=torch.float32)
            else:
                pmt_dir = pmt_dir.float().to(self.device)
            pmt_dir = pmt_dir.squeeze()  # (3,)
            feature_values.extend([pmt_dir[0], pmt_dir[1], pmt_dir[2]])

        features = torch.stack(feature_values)

        return features.clone().detach()

    def prepare_features_charge_batched(self, points, event_data, light_yields=None,
                                        pmt_directions=None):
        """Batched rich charge features for many detector points, one event.

        Vectorised equivalent of prepare_features_charge over a set of detector
        points that share the same event params (position, energy, direction).
        As in the single-point version, the light yield is the target, so no
        log_ly column is appended and ``light_yields`` is ignored (kept only for
        call-signature compatibility).

        Parameters
        ----------
        points : torch.Tensor or np.ndarray, shape (n_det, 3)
            Detector (OM) positions.
        event_data : dict
            Event params; must contain 'position', 'energy', 'direction'
            (shared across all detector points).
        light_yields : ignored
            Accepted for call-signature compatibility; not used.
        pmt_directions : torch.Tensor or np.ndarray, shape (n_det, 3) or None
            Per-detector PMT directions; required when self.add_pmt_direction.

        Returns
        -------
        torch.Tensor, shape (n_det, feature_dim)
            Same column layout as prepare_features_charge.
        """
        dev = self.device

        def _t(x):
            if isinstance(x, torch.Tensor):
                return x.float().to(dev)
            return torch.tensor(x, device=dev, dtype=torch.float32)

        pts = _t(points).reshape(-1, 3)              # (n, 3)
        n = pts.shape[0]
        norm = self._pos_norm_divisor()              # scalar or (3,)

        det = pts / norm                             # (n, 3)
        vert = (_t(event_data['position']).squeeze() / norm).reshape(3)   # (3,)
        direction = _t(event_data['direction']).squeeze().reshape(3)      # (3,)
        log_energy = torch.log10(_t(event_data['energy']).squeeze() + self.ly_eps) / 8.0  # scalar

        rel = det - vert.unsqueeze(0)                                 # (n, 3)
        vert_dist = torch.norm(rel, dim=1)                            # (n,)
        cos_angle = (rel @ direction) / (torch.norm(direction) * vert_dist + 1e-8)  # (n,)

        dir_rep = direction.unsqueeze(0).expand(n, -1)               # (n, 3)
        log_e_rep = log_energy.expand(n).unsqueeze(1)                # (n, 1)

        cols = []
        if self.rich_rel_pos_mode:
            cols.append(rel)                    # (n, 3)
            cols.append(dir_rep)                # (n, 3)
            cols.append(log_e_rep)              # (n, 1)
        else:
            cols.append(det)                    # (n, 3)
            cols.append(vert.unsqueeze(0).expand(n, -1))  # (n, 3)
            cols.append(dir_rep)                # (n, 3)
            cols.append(log_e_rep)              # (n, 1)
        if self.add_vertex_distance:
            cols.append(vert_dist.unsqueeze(1))     # (n, 1)
        cols.append(cos_angle.unsqueeze(1))         # (n, 1)
        if self.add_distance_from_beam:
            track_pos = vert * norm                 # (3,) original scale
            _, dist_perp = self.compute_distance_from_beam(
                det * norm, track_pos.unsqueeze(0), direction.unsqueeze(0)
            )  # dist_perp: (n, 1)
            half = self.domain_size / 2
            cols.append(dist_perp.reshape(n, 1) / half)
        if self.add_pmt_direction:
            if pmt_directions is None:
                raise ValueError("pmt_directions is required when add_pmt_direction=True")
            cols.append(_t(pmt_directions).reshape(n, 3))  # (n, 3)

        features = torch.cat(cols, dim=1)           # (n, feature_dim)
        return features.clone().detach()

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
        
        
        # Log scale light yield if requested (for better convergence). We use
        # log10(count + 1) so a zero light yield maps to 0.0 (representable) and
        # the inverse is 10**pred - 1 (see predict / the surrogate). This matches
        # the parquet dataset's target transform.
        if self.log_scale_ly:
            light_yield = torch.log10(torch.abs(light_yield) + 1.0)
       
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
                             verbose=True, early_stopping_patience=10, input_dim=None,
                             save_every_n_epochs=None, checkpoint_path=None):
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
        save_every_n_epochs : int or None
            If set, save a checkpoint every N epochs during training (and at the
            final epoch). Requires checkpoint_path.
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
                    # Save a final checkpoint before breaking out of training.
                    if save_every_n_epochs is not None:
                        checkpoint_dirname = os.path.dirname(checkpoint_path)
                        if checkpoint_dirname:
                            os.makedirs(checkpoint_dirname, exist_ok=True)
                        self.save_model(checkpoint_path)
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

            # Periodic checkpointing (also fires on the final epoch).
            if save_every_n_epochs is not None and ((epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs):
                checkpoint_dirname = os.path.dirname(checkpoint_path)
                if checkpoint_dirname:
                    os.makedirs(checkpoint_dirname, exist_ok=True)
                self.save_model(checkpoint_path)

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
        
        # Convert back from log scale if necessary. Forward transform is
        # log10(count + 1), so the inverse is 10**pred - 1.
        if self.log_scale_ly:
            predictions = 10 ** predictions - 1.0
            predictions = torch.clamp(predictions, min=0.0)
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
            'use_rich_features': self.use_rich_features,
            'rich_rel_pos_mode': self.rich_rel_pos_mode,
            'add_vertex_distance': self.add_vertex_distance,
            'add_pmt_direction': self.add_pmt_direction,
            'log_charge_scale': self.log_charge_scale,
            'ly_eps': self.ly_eps,
            'pmt_directions': self.pmt_directions.detach().cpu() if self.pmt_directions is not None else None,
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
        self.use_rich_features = state.get('use_rich_features', False)
        self.rich_rel_pos_mode = state.get('rich_rel_pos_mode', False)
        self.add_vertex_distance = state.get('add_vertex_distance', True)
        self.add_pmt_direction = state.get('add_pmt_direction', False)
        self.log_charge_scale = state.get('log_charge_scale', 4)
        self.ly_eps = state.get('ly_eps', 1e-10)
        pmt_directions = state.get('pmt_directions', None)
        self.pmt_directions = pmt_directions.to(self.device) if pmt_directions is not None else None
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
                # (inverse of log10(count + 1)).
                if self.chargenet_model.log_scale_ly:
                    ly_value = 10 ** ly_value - 1.0
                
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

    class LightYieldParquetDataset(Dataset):
        """
        Regression dataset for the charge (light-yield) ChargeNet, sourced from a
        parquet file produced by ``extract_accepted_photons.py`` in light-yield
        mode plus a geometry CSV from ``extract_geom.py``.

        Each parquet row is one hit PMT in one event and carries:

            string, om, pmt, count, muon_x, muon_y, muon_z,
            muon_energy, neutrino_energy, zenith, azimuth

        where muon_x/y/z is the muon interaction vertex, muon_energy the muon
        (CC daughter) energy, and zenith/azimuth the primary-neutrino direction.

        The geometry CSV maps ``(string, om, pmt)`` to the optical-module
        position (``om_x/y/z``) and the hit-PMT direction (``pmt_dir_x/y/z``).

        For every row we build a (features, target) pair for ChargeNet:

            * features    = chargenet.prepare_features_charge(point, event_data),
                            i.e. the rich geometric/event feature vector WITHOUT
                            the light-yield column (the yield is the target).
            * target      = the row's light yield (count), optionally log-scaled.
            * point       = OM position (detector position),   from the CSV
            * event_data:
                - position    = muon interaction vertex (muon_x/y/z)
                - energy      = neutrino_energy
                - direction   = unit vector from neutrino (zenith, azimuth)
                - pmt_direction = hit-PMT direction (used only when the model has
                                  add_pmt_direction=True)

        A "params" row is drawn at random (with replacement) on every
        __getitem__ call, so the epoch length (num_samples_per_epoch) is
        decoupled from the number of rows. With uniform_energy_zenith=True the
        draw is stratified so the network sees (neutrino energy, cos zenith)
        approximately uniformly (pick a non-empty log10-energy x cos-zenith bin
        uniformly, then a row within it).

        Optional zero light-yield augmentation (zero_ly_prob > 0): with low
        probability any item is replaced by a *zero-LY* sample -- the event's
        params observed at a (string, om, pmt) that was NOT hit in that event
        (sampled from the geometry keys), with light yield ``zero_ly_value``.
        This teaches the network that an unhit PMT has (near) zero yield.
        Requires run_id/event_id columns in the parquet to group hits per event
        (otherwise each row is treated as its own event).
        """

        def __init__(self, chargenet_instance, parquet_path, geometry_csv_path,
                     num_samples_per_epoch=None, seed=None,
                     zero_ly_prob=0.0, zero_ly_value=0.0,
                     uniform_energy_zenith=False, n_energy_bins=20,
                     n_coszen_bins=20, filter_vertex_in_domain=True,
                     event_filter=None):
            """
            Parameters
            ----------
            chargenet_instance : ChargeNet
                Parent model; provides prepare_features_charge, device, flags.
            parquet_path : str
                Path to the light-yield parquet file.
            geometry_csv_path : str
                Path to the geometry CSV (string, om, pmt -> om pos + pmt dir).
            num_samples_per_epoch : int or None
                Number of samples per epoch. Defaults to the number of usable
                parquet rows.
            seed : int or None
                Seed for the sampling RNG (reproducible draws).
            zero_ly_prob : float
                Probability (per item) of instead emitting a zero light-yield
                sample: the event's params observed at a PMT that was NOT hit in
                that event (sampled from the geometry), with light yield
                ``zero_ly_value``. Default 0.0 (disabled).
            zero_ly_value : float
                Light-yield value used for zero-LY samples (default 0.0).
            uniform_energy_zenith : bool
                If True, the event-params row is drawn by importance sampling so
                the network sees (neutrino energy, cos zenith) approximately
                uniformly: each __getitem__ first picks a non-empty
                (log10 energy, cos zenith) bin uniformly at random, then a row
                uniformly from the rows in that bin. Default False (uniform over
                rows). See _build_energy_coszen_bins for the binning.
            n_energy_bins : int
                Number of bins in log10(neutrino_energy) for uniform sampling.
            n_coszen_bins : int
                Number of bins in cos(zenith) for uniform sampling.
            filter_vertex_in_domain : bool
                If True (default), drop any row whose muon interaction vertex
                (muon_x/y/z) falls outside the model's domain. The domain is a
                box centred at the origin with half-extents derived from
                chargenet_instance.domain_size: a scalar gives a cube of side
                domain_size (|x|,|y|,|z| <= domain_size/2); a (width, height)
                pair gives |x|,|y| <= width/2 and |z| <= height/2.
            event_filter : set or None
                If given, keep only rows whose event id is in this set (event
                id is (run_id, event_id) when those columns exist, else the row
                index). Used to restrict the dataset to a train/test subset.
            """
            import pandas as pd

            self.chargenet = chargenet_instance
            self.device = chargenet_instance.device
            self.zero_ly_prob = float(zero_ly_prob)
            self.zero_ly_value = float(zero_ly_value)

            # ---- load geometry CSV -> (string, om, pmt) lookup ----
            geo = pd.read_csv(geometry_csv_path)
            self._om_pos = {}
            self._pmt_dir = {}
            for r in geo.itertuples(index=False):
                key = (int(r.string), int(r.om), int(r.pmt))
                self._om_pos[key] = np.array([r.om_x, r.om_y, r.om_z], dtype=np.float32)
                self._pmt_dir[key] = np.array(
                    [r.pmt_dir_x, r.pmt_dir_y, r.pmt_dir_z], dtype=np.float32
                )
            # All geometry keys, in a fixed order, for sampling unhit PMTs.
            self._geo_keys = list(self._om_pos.keys())

            # Per-axis half-extents of the domain box (centred at origin), from
            # the model's domain_size. Scalar -> cube; (width, height) -> box.
            self.filter_vertex_in_domain = bool(filter_vertex_in_domain)
            ds = chargenet_instance.domain_size
            if isinstance(ds, torch.Tensor):
                ds = ds.tolist() if ds.dim() > 0 else ds.item()
            if isinstance(ds, (tuple, list)) and len(ds) == 2:
                width, height = float(ds[0]), float(ds[1])
                half_extent = np.array([width / 2.0, width / 2.0, height / 2.0],
                                       dtype=np.float64)
            else:
                half = float(ds) / 2.0
                half_extent = np.array([half, half, half], dtype=np.float64)
            self._domain_half_extent = half_extent

            # ---- load parquet and keep only rows with a matching geometry and,
            #      optionally, whose muon vertex lies inside the domain ----
            df = pd.read_parquet(parquet_path)
            has_event_id = {'run_id', 'event_id'}.issubset(df.columns)
            # Optional event-level subset (e.g. train/test split): keep only rows
            # whose event id is in this set. Keys match the event identity used
            # below: (run_id, event_id) when present, else the row index.
            self.event_filter = set(event_filter) if event_filter is not None else None
            keep = []
            n_out_of_domain = 0
            n_filtered_events = 0
            for row_idx, r in enumerate(df.itertuples(index=False)):
                key = (int(r.string), int(r.om), int(r.pmt))
                if key not in self._om_pos:
                    continue
                if self.filter_vertex_in_domain:
                    if (abs(float(r.muon_x)) > half_extent[0] or
                            abs(float(r.muon_y)) > half_extent[1] or
                            abs(float(r.muon_z)) > half_extent[2]):
                        n_out_of_domain += 1
                        continue
                if self.event_filter is not None:
                    ev = (int(r.run_id), int(r.event_id)) if has_event_id else row_idx
                    if ev not in self.event_filter:
                        n_filtered_events += 1
                        continue
                keep.append(r)
            if len(keep) == 0:
                raise ValueError(
                    "No usable parquet rows: none matched the geometry CSV "
                    "(and/or all muon vertices were outside the domain). Check "
                    "that the files correspond to the same detector and that "
                    "domain_size is large enough."
                )
            if self.filter_vertex_in_domain and n_out_of_domain > 0:
                print(f"LightYieldParquetDataset: dropped {n_out_of_domain} row(s) "
                      f"with muon vertex outside domain half-extents "
                      f"{half_extent.tolist()}.")

            # Precompute per-row arrays (as numpy; converted to tensors per item).
            n = len(keep)
            self._point = np.empty((n, 3), dtype=np.float32)   # OM position
            self._pmt_direction = np.empty((n, 3), dtype=np.float32)
            self._muon_pos = np.empty((n, 3), dtype=np.float32)
            self._energy = np.empty((n,), dtype=np.float32)
            self._zenith = np.empty((n,), dtype=np.float32)
            self._azimuth = np.empty((n,), dtype=np.float32)
            self._count = np.empty((n,), dtype=np.float32)     # light yield
            # Per-event set of hit (string, om, pmt) keys, so a zero-LY sample can
            # pick a PMT that was NOT hit in the same event. Keyed by event id.
            self._event_hit_keys = {}
            self._row_event = [None] * n
            for i, r in enumerate(keep):
                key = (int(r.string), int(r.om), int(r.pmt))
                self._point[i] = self._om_pos[key]
                self._pmt_direction[i] = self._pmt_dir[key]
                self._muon_pos[i] = (r.muon_x, r.muon_y, r.muon_z)
                self._energy[i] = r.neutrino_energy
                self._zenith[i] = r.zenith
                self._azimuth[i] = r.azimuth
                self._count[i] = r.count
                # Event identity: (run_id, event_id) if present, else the row
                # itself (each row is then treated as its own event).
                ev = (int(r.run_id), int(r.event_id)) if has_event_id else i
                self._row_event[i] = ev
                self._event_hit_keys.setdefault(ev, set()).add(key)

            # Distinct events and per-event row indices (kept for parity with the
            # LLRnet dataset; used by the sampling helpers).
            self._events = list(self._event_hit_keys.keys())
            self._n_events = len(self._events)
            # event -> np.array of row indices belonging to it.
            self._event_rows = {}
            for i, ev in enumerate(self._row_event):
                self._event_rows.setdefault(ev, []).append(i)
            self._event_rows = {k: np.asarray(v) for k, v in self._event_rows.items()}

            self._n_rows = n
            self.num_samples_per_epoch = (
                num_samples_per_epoch if num_samples_per_epoch is not None else n
            )
            # Dedicated RNG so sampling is reproducible and independent of global
            # torch/numpy state (also safe across DataLoader workers).
            self._rng = np.random.default_rng(seed)

            # Importance sampling to flatten (energy, cos zenith): group rows into
            # (log10 energy, cos zenith) bins so a params row can be drawn by
            # first picking a non-empty bin uniformly, then a row within it.
            self.uniform_energy_zenith = bool(uniform_energy_zenith)
            if self.uniform_energy_zenith:
                self._build_energy_coszen_bins(int(n_energy_bins), int(n_coszen_bins))

            # When the model uses the PMT direction as a feature, record all the
            # unique PMT directions from the geometry on the model itself, so they
            # are available at inference time and persisted via save/load_model.
            if getattr(chargenet_instance, 'add_pmt_direction', False):
                all_dirs = np.stack(list(self._pmt_dir.values()), axis=0)  # (n_pmts, 3)
                chargenet_instance.set_pmt_directions(all_dirs)

        def _build_energy_coszen_bins(self, n_energy_bins, n_coszen_bins):
            """Group row indices into (log10 energy, cos zenith) bins.

            Builds ``self._bin_rows``: a list of int arrays, one per NON-EMPTY
            2-D bin, each holding the indices of the rows that fall in that bin.
            Uniform sampling then picks one of these lists uniformly, then a row
            uniformly from within it -- flattening the empirical (energy, cos
            zenith) distribution the network sees over the occupied grid.
            """
            log_e = np.log10(np.clip(self._energy, 1e-12, None))
            coszen = np.cos(self._zenith)

            # Bin edges spanning the observed range (guard against zero width).
            def _edges(vals, nb):
                lo, hi = float(np.min(vals)), float(np.max(vals))
                if hi <= lo:
                    hi = lo + 1e-6
                return np.linspace(lo, hi, nb + 1)

            e_edges = _edges(log_e, n_energy_bins)
            c_edges = _edges(coszen, n_coszen_bins)

            # Bin index per row (clipped to the last bin at the upper edge).
            ei = np.clip(np.digitize(log_e, e_edges) - 1, 0, n_energy_bins - 1)
            ci = np.clip(np.digitize(coszen, c_edges) - 1, 0, n_coszen_bins - 1)
            flat = ei * n_coszen_bins + ci  # unique id per 2-D bin

            order = np.argsort(flat, kind='stable')
            flat_sorted = flat[order]
            # Split the sorted row indices at bin boundaries into per-bin groups.
            boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
            self._bin_rows = np.split(order, boundaries)
            self._n_bins = len(self._bin_rows)

        def _sample_params_row(self):
            """Draw an event-params row index according to the sampling scheme."""
            if getattr(self, 'uniform_energy_zenith', False) and self._n_bins > 0:
                # Uniform over non-empty bins, then uniform within the bin.
                b = int(self._rng.integers(0, self._n_bins))
                group = self._bin_rows[b]
                return int(group[int(self._rng.integers(0, len(group)))])
            return int(self._rng.integers(0, self._n_rows))

        def _event_data(self, i, pmt_direction=None):
            """Build the event_data dict (hypothesis params) for row i.

            If ``pmt_direction`` (a (3,) array) is given it overrides the row's
            own PMT direction -- used for zero-LY samples observed at a PMT that
            was not hit in the event.
            """
            zenith = torch.tensor(self._zenith[i], device=self.device, dtype=torch.float32)
            azimuth = torch.tensor(self._azimuth[i], device=self.device, dtype=torch.float32)
            direction = sph_to_cart(zenith, azimuth)  # (3,), unit vector
            pmt_dir = self._pmt_direction[i] if pmt_direction is None else pmt_direction
            return {
                'position': torch.tensor(self._muon_pos[i], device=self.device, dtype=torch.float32),
                'energy': torch.tensor(self._energy[i], device=self.device, dtype=torch.float32),
                'direction': direction,
                'pmt_direction': torch.tensor(pmt_dir, device=self.device, dtype=torch.float32),
            }

        def _features(self, i, point=None, pmt_direction=None):
            """Feature vector for row i's params observed at the given point.

            ``point`` and ``pmt_direction`` optionally override the detector
            position / PMT direction (used for zero-LY samples at an unhit PMT).
            The light yield is the TARGET, not a feature, so it is not passed in.
            """
            pt = self._point[i] if point is None else point
            point_t = torch.tensor(pt, device=self.device, dtype=torch.float32)
            return self.chargenet.prepare_features_charge(
                point_t, self._event_data(i, pmt_direction=pmt_direction)
            )

        def _transform_target(self, light_yield):
            """Apply the model's target transform to a raw light-yield value.

            If chargenet.log_scale_ly is True the target is log10(count + 1.0)
            (so a zero count is representable as 0.0); otherwise the raw count is
            used. Returns a scalar float32 tensor.
            """
            ly = float(light_yield)
            if self.chargenet.log_scale_ly:
                # log10(count + 1.0): keeps zeros representable (-> 0.0).
                target = np.log10(ly + 1.0)/self.chargenet.log_charge_scale
            else:
                # Raw count.
                target = ly
            return torch.tensor(target, device=self.device, dtype=torch.float32)

        def _sample_unhit_key(self, row):
            """Sample a geometry (string, om, pmt) key NOT hit in row's event.

            Returns None if the event hit every PMT in the geometry (no unhit
            PMT available).
            """
            hit = self._event_hit_keys.get(self._row_event[row], ())
            n_geo = len(self._geo_keys)
            if len(hit) >= n_geo:
                return None
            # Rejection sampling: unhit PMTs vastly outnumber hit ones in practice.
            for _ in range(100):
                k = self._geo_keys[int(self._rng.integers(0, n_geo))]
                if k not in hit:
                    return k
            # Fallback: scan for any unhit key (guaranteed to exist here).
            for k in self._geo_keys:
                if k not in hit:
                    return k
            return None

        def __len__(self):
            return self.num_samples_per_epoch

        def __getitem__(self, idx):
            # The "params" row is drawn at random (with replacement) each call, so
            # the epoch length (num_samples_per_epoch) is independent of the file
            # size and every item is an i.i.d. draw rather than a fixed
            # permutation of rows. With uniform_energy_zenith the draw is
            # stratified over (log10 energy, cos zenith) bins (see
            # _sample_params_row).
            row = self._sample_params_row()

            # With low probability, emit a zero-LY sample instead: the event's
            # params observed at a PMT that was NOT hit in the event, with a
            # (transformed) zero light yield -> the network should learn this is
            # (near) zero yield.
            if self.zero_ly_prob > 0.0 and self._rng.random() < self.zero_ly_prob:
                unhit = self._sample_unhit_key(row)
                if unhit is not None:
                    features = self._features(
                        row,
                        point=self._om_pos[unhit],
                        pmt_direction=self._pmt_dir[unhit],
                    )
                    target = self._transform_target(self.zero_ly_value)
                    return features, target

            features = self._features(row)
            target = self._transform_target(self._count[row])
            return features, target

    def create_light_yield_parquet_dataloader(self, parquet_path, geometry_csv_path,
                                              num_samples_per_epoch=None, batch_size=32,
                                              shuffle=True, num_workers=0, seed=None,
                                              zero_ly_prob=0.05, zero_ly_value=0.0,
                                              uniform_energy_zenith=False,
                                              n_energy_bins=20, n_coszen_bins=20,
                                              filter_vertex_in_domain=True,
                                              test_save_path=None, test_frac=0.1,
                                              split_seed=None,
                                              pin_memory=None, pin_memory_device=None):
        """
        Create a DataLoader for the charge ChargeNet regressor from a light-yield
        parquet file and a geometry CSV.

        See LightYieldParquetDataset for the data model. Set the model's
        ``add_pmt_direction=True`` to include the hit-PMT direction in the
        feature vector. Each item is (features, target) where target is the
        row's light yield, optionally log-scaled (log10(count + 1.0)) when the
        model has log_scale_ly=True.

        Parameters
        ----------
        parquet_path : str
            Light-yield parquet file (from extract_accepted_photons.py --ly_mode).
        geometry_csv_path : str
            Geometry CSV (from extract_geom.py).
        num_samples_per_epoch : int or None
            Samples per epoch (defaults to number of rows).
        batch_size, shuffle, num_workers, pin_memory, pin_memory_device
            Standard DataLoader options.
        seed : int or None
            Seed for reproducible sampling.
        zero_ly_prob : float
            Probability of replacing an item with a zero light-yield sample: the
            event's params observed at a PMT NOT hit in that event, with light
            yield zero_ly_value. Default 0.0 (disabled).
        zero_ly_value : float
            Light-yield value used for zero-LY samples (default 0.0).
        uniform_energy_zenith : bool
            If True, importance-sample the event-params row so the network sees
            (neutrino energy, cos zenith) approximately uniformly (stratified
            over log10-energy x cos-zenith bins). Default False.
        n_energy_bins, n_coszen_bins : int
            Bin counts for the uniform (energy, cos zenith) sampling.
        filter_vertex_in_domain : bool
            If True (default), drop rows whose muon interaction vertex lies
            outside the model's domain (box from self.domain_size, centred at
            the origin).
        test_save_path : str or None
            If given, hold out ``test_frac`` of the EVENTS as a test set: the
            held-out events' rows are written to this parquet path, and the
            returned DataLoader is built from the remaining (train) events only.
            The split is by event (all rows of an event go to the same side).
            If None (default), no split is done and the whole file is used.
        test_frac : float
            Fraction of events to hold out for testing (default 0.1).
        split_seed : int or None
            Seed for the train/test event split (falls back to ``seed`` if None),
            so the split is reproducible.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        # Optional event-level train/test split. Choose the held-out events from
        # the parquet's event ids, write their rows to test_save_path, and build
        # the training dataset from only the remaining (train) events.
        train_event_filter = None
        if test_save_path is not None:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            has_event_id = {'run_id', 'event_id'}.issubset(df.columns)
            if has_event_id:
                ev_series = list(zip(df['run_id'].astype(int), df['event_id'].astype(int)))
                unique_events = sorted(set(ev_series))
            else:
                # No event ids: treat each row as its own event.
                ev_series = list(range(len(df)))
                unique_events = list(ev_series)

            rng = np.random.default_rng(split_seed if split_seed is not None else seed)
            n_events = len(unique_events)
            n_test = int(round(test_frac * n_events))
            perm = rng.permutation(n_events)
            test_idx = set(perm[:n_test].tolist())
            test_events = {unique_events[i] for i in test_idx}
            train_event_filter = {unique_events[i] for i in range(n_events) if i not in test_idx}

            # Mask of rows whose event is held out for testing, and write them.
            # Build the per-row event key as a pandas Series (avoids turning a
            # list of (run_id, event_id) tuples into a 2-D numpy array, which
            # would make membership tests iterate over unhashable row arrays).
            ev_col = pd.Series(ev_series, index=df.index, dtype=object)
            test_mask = ev_col.isin(test_events).to_numpy()
            df.loc[test_mask].to_parquet(test_save_path, index=False)
            print(f"create_light_yield_parquet_dataloader: held out "
                  f"{len(test_events)}/{n_events} events "
                  f"({int(test_mask.sum())} rows) for testing -> {test_save_path}. "
                  f"Training on the remaining {len(train_event_filter)} events.")

        dataset = self.LightYieldParquetDataset(
            chargenet_instance=self,
            parquet_path=parquet_path,
            geometry_csv_path=geometry_csv_path,
            num_samples_per_epoch=num_samples_per_epoch,
            seed=seed,
            zero_ly_prob=zero_ly_prob,
            zero_ly_value=zero_ly_value,
            uniform_energy_zenith=uniform_energy_zenith,
            n_energy_bins=n_energy_bins,
            n_coszen_bins=n_coszen_bins,
            filter_vertex_in_domain=filter_vertex_in_domain,
            event_filter=train_event_filter,
        )

        dl_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        # ChargeNet has no _resolve_pin_memory: pass pin_memory through directly.
        if pin_memory is not None:
            dl_kwargs['pin_memory'] = pin_memory
            if pin_memory and pin_memory_device:
                dl_kwargs['pin_memory_device'] = pin_memory_device

        return DataLoader(**dl_kwargs)


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
        
        # When the model was trained on rich features (prepare_features_charge),
        # build a rich event dict that carries a unit 'direction' vector (derived
        # from zenith/azimuth if not already present) and, for an
        # add_pmt_direction model, requires a 'pmt_direction' in event_params.
        use_rich_features = getattr(self.chargenet, 'use_rich_features', False)
        rich_event_params = None
        if use_rich_features:
            rich_event_params = dict(event_params)
            if 'direction' not in rich_event_params:
                zenith = rich_event_params['zenith']
                azimuth = rich_event_params['azimuth']
                if not isinstance(zenith, torch.Tensor):
                    zenith = torch.tensor(zenith, device=self.device, dtype=torch.float32)
                else:
                    zenith = zenith.float().to(self.device)
                if not isinstance(azimuth, torch.Tensor):
                    azimuth = torch.tensor(azimuth, device=self.device, dtype=torch.float32)
                else:
                    azimuth = azimuth.float().to(self.device)
                rich_event_params['direction'] = sph_to_cart(zenith, azimuth)
            if getattr(self.chargenet, 'add_pmt_direction', False) and \
                    'pmt_direction' not in rich_event_params:
                raise ValueError(
                    "This ChargeNet was trained with add_pmt_direction=True, so "
                    "event_params must supply 'pmt_direction' (the hit-PMT "
                    "direction, a 3-vector) to build the rich features."
                )

        # Check if we have a batch or single point
        is_batch = opt_point_tensor.dim() > 1

        if is_batch:
            # Process batch of points
            batch_size = opt_point_tensor.shape[0]
            predictions = []

            for i in range(batch_size):
                point = opt_point_tensor[i]
                if use_rich_features:
                    features = self.chargenet.prepare_features_charge(
                        point, rich_event_params
                    )
                else:
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

                # Convert from log scale if necessary (inverse of log10(count+1)).
                if self.chargenet.log_scale_ly:
                    pred = 10 ** (pred* self.chargenet.log_charge_scale) - 1.0

                predictions.append(pred)

            light_yield = torch.stack(predictions).squeeze()
        else:
            # Single point
            if use_rich_features:
                features = self.chargenet.prepare_features_charge(
                    opt_point_tensor, rich_event_params
                )
            else:
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

            # Convert from log scale if necessary (inverse of log10(count+1)).
            if self.chargenet.log_scale_ly:
                light_yield = 10 ** (light_yield * self.chargenet.log_charge_scale) - 1.0

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
            log_scale_energy=state['log_scale_energy'],
            use_rich_features=state.get('use_rich_features', False),
            rich_rel_pos_mode=state.get('rich_rel_pos_mode', False),
            add_vertex_distance=state.get('add_vertex_distance', True),
            add_pmt_direction=state.get('add_pmt_direction', False),
            log_charge_scale=state.get('log_charge_scale', 4)
        )
        
        # Load the model weights
        chargenet.load_model(checkpoint_path)
        
        return cls(chargenet_model=chargenet, device=device, dim=dim, domain_size=domain_size)
