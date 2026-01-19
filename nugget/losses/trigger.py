import torch
import numpy as np
from itertools import combinations
from nugget.losses.base_loss import LossFunction
import time


class TriggerLoss(LossFunction):
    """
    Loss function for detector trigger efficiency based on multi-level conditions.
    
    This loss evaluates detector efficiency by testing whether events will be detected
    given three cascading conditions:
    1. t1_i = point_weight_i x sigmoid(light_yield_i - threshold)
    2. t2_ij = sigmoid(threshold - |point_i - point_j|) for all pairs i != j
    3. t3 = sigmoid(sum_ij(t2_ij x t1_i x t1_j) - threshold) for unique pairs
    """
    
    def __init__(self, 
                 device=None, 
                 light_yield_threshold=10.0,
                 pairwise_distance_threshold=100.0,
                 min_pairs_threshold=10.0,
                 t1_temperature=1.0,
                 t2_temperature=1.0,
                 t3_temperature=1.0,
                 t_temperature=1.0,
                 weight_sigmoid_sharpness=1.0,
                 print_loss=False):
        """
        Initialize the trigger loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        light_yield_threshold : float
            Minimum light yield threshold for point detection (t1).
        pairwise_distance_threshold : float
            Maximum pairwise distance threshold (t2).
        min_pairs_threshold : float
            Minimum threshold for sum of pair contributions (t3).
        t1_temperature : float
            Temperature for t1 sigmoid (higher = sharper transition).
        t2_temperature : float
            Temperature for t2 sigmoid (higher = sharper transition).
        t3_temperature : float
            Temperature for t3 sigmoid (higher = sharper transition).
        weight_sigmoid_sharpness : float
            Sharpness parameter for string weight sigmoid (higher = sharper transition).
        print_loss : bool
            Whether to print loss components during computation.
        """
        super().__init__(device)
        
        self.light_yield_threshold = light_yield_threshold
        self.pairwise_distance_threshold = pairwise_distance_threshold
        self.min_pairs_threshold = min_pairs_threshold
        self.t1_temperature = t1_temperature
        self.t2_temperature = t2_temperature
        self.t3_temperature = t3_temperature
        self.t_temperature = t_temperature
        self.weight_sigmoid_sharpness = weight_sigmoid_sharpness
        self.print_loss = print_loss
    
    def map_string_weights_to_points(self, points_3d, string_xy, string_weights):
        """
        Map string weights to point weights based on xy positions.
        
        Points with the same xy coordinates (belonging to the same string) 
        will receive the same weight. Weights are passed through sigmoid.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            All detector points, shape (n_points, 3)
        string_xy : torch.Tensor
            String xy positions, shape (n_strings, 2)
        string_weights : torch.Tensor
            Weights for each string, shape (n_strings,)
            
        Returns:
        --------
        torch.Tensor
            Weights for each point (after sigmoid), shape (n_points,)
        """
        n_points = len(points_3d)
        n_strings = len(string_xy)
        
        # Extract xy coordinates from points
        points_xy = points_3d[:, :2]  # Shape: (n_points, 2)
        
        # Initialize point weights
        point_weights = torch.zeros(n_points, device=self.device)
        
        # For each point, find the closest string and assign its weight
        for i, point_xy in enumerate(points_xy):
            # Calculate distance to all strings
            distances = torch.norm(string_xy - point_xy.unsqueeze(0), dim=1)
            
            # Find the closest string
            closest_string_idx = torch.argmin(distances)
            
            # Assign the string's weight to this point
            point_weights[i] = string_weights[closest_string_idx]
        
        # Apply sigmoid to weights (not controlled by temperature)
        point_weights_sigmoid = torch.sigmoid(self.weight_sigmoid_sharpness * point_weights)
        
        return point_weights_sigmoid
    
    def compute_pairwise_distances_along_track(self, points_3d, track_pos, track_dir):
        """
        Compute pairwise distances between points along the track direction.
        
        This projects all points onto the track line and computes the absolute
        difference between their projection positions.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Points, shape (n_points, 3)
        track_pos : torch.Tensor
            Point on track line, shape (3,)
        track_dir : torch.Tensor
            Track direction, shape (3,)
            
        Returns:
        --------
        torch.Tensor
            Pairwise distances along track, shape (n_points, n_points)
        """
        # Normalize direction
        track_dir = track_dir / torch.norm(track_dir)
        
        # Project all points onto the track: (p - x0) · d
        diff = points_3d - track_pos.unsqueeze(0)
        projections = torch.sum(diff * track_dir.unsqueeze(0), dim=1)  # Shape: (n_points,)
        
        # Compute pairwise differences: |proj_i - proj_j|
        # Using broadcasting: (n, 1) - (1, n) = (n, n)
        pairwise_distances = torch.abs(projections.unsqueeze(1) - projections.unsqueeze(0))
        
        return pairwise_distances
    
    def compute_trigger_probability_batch_events(self,
                                                 points_3d,
                                                 event_params_list,
                                                 surrogate_func,
                                                 string_weights=None,
                                                 precomputed_light_yield=None):
        """
        Compute trigger probability for all events in batch using vectorized operations.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Detector points, shape (n_points, 3)
        event_params_list : list of dict
            List of event parameters
        surrogate_func : callable
            Light yield surrogate function
        string_weights : torch.Tensor or None
            Weights for each point (after sigmoid), shape (n_points,)
        precomputed_light_yield : torch.Tensor or None
            Precomputed light yields, shape (n_events, n_points)
            
        Returns:
        --------
        torch.Tensor
            Trigger probabilities for all events, shape (n_events,)
        """
        n_points = len(points_3d)
        n_events = len(event_params_list)
        
        # Step 1: Compute t1 values (light yield condition) for all events
        if precomputed_light_yield is not None:
            light_yields = precomputed_light_yield  # Shape: (n_events, n_points)
        else:
            # Batch compute light yields for all events
            light_yields = torch.zeros(n_events, n_points, device=points_3d.device)
            for i, event_params in enumerate(event_params_list):
                light_yields[i] = surrogate_func(opt_point=points_3d, event_params=event_params)
        
        # Default weights (if not provided, all points have weight 1)
        if string_weights is None:
            string_weights = torch.ones(n_points, device=points_3d.device)
        
        # t1_i = point_weight_i x sigmoid(light_yield_i - threshold)
        # Shape: (n_events, n_points)
        t1_sigmoid = torch.sigmoid(self.t1_temperature * (light_yields - self.light_yield_threshold))
        t1_values = string_weights.unsqueeze(0) * t1_sigmoid  # Broadcast weights
        
        # Extract track parameters for all events
        track_positions = []
        track_directions = []
        
        for event_params in event_params_list:
            track_pos = event_params.get('position')
            zenith = event_params.get('zenith')
            azimuth = event_params.get('azimuth')
            
            # Ensure track_pos is 1D with shape (3,)
            if isinstance(track_pos, torch.Tensor):
                track_pos = track_pos.flatten()
            else:
                track_pos = torch.tensor(track_pos, device=points_3d.device).flatten()
            
            # Convert to direction vector
            if isinstance(zenith, torch.Tensor):
                theta = zenith.squeeze()
                phi = azimuth.squeeze()
            else:
                theta = torch.tensor(zenith, device=points_3d.device).squeeze()
                phi = torch.tensor(azimuth, device=points_3d.device).squeeze()
            
            track_dir = torch.stack([
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta)
            ])
            
            track_positions.append(track_pos)
            track_directions.append(track_dir)
        
        # Stack into batched tensors
        track_positions = torch.stack(track_positions)  # Shape: (n_events, 3)
        track_directions = torch.stack(track_directions)  # Shape: (n_events, 3)
        
        # Normalize directions
        track_directions = track_directions / torch.norm(track_directions, dim=1, keepdim=True)
        
        # Step 2: Compute t2 matrix (pairwise distance condition along track) for all events
        # Batch compute projections: (points - track_pos) · track_dir
        # Shape manipulations: points (n_points, 3) -> (1, n_points, 3)
        #                     track_pos (n_events, 3) -> (n_events, 1, 3)
        #                     result: (n_events, n_points, 3)
        diff = points_3d.unsqueeze(0) - track_positions.unsqueeze(1)  # (n_events, n_points, 3)
        
        # Compute dot product with track direction
        # (n_events, n_points, 3) * (n_events, 1, 3) -> sum over dim=2 -> (n_events, n_points)
        projections = torch.sum(diff * track_directions.unsqueeze(1), dim=2)
        
        # Compute pairwise differences: |proj_i - proj_j| for each event
        # (n_events, n_points, 1) - (n_events, 1, n_points) = (n_events, n_points, n_points)
        pairwise_distances = torch.abs(projections.unsqueeze(2) - projections.unsqueeze(1))
        
        # Apply sigmoid: sigmoid(threshold - distance)
        t2_matrix = torch.sigmoid(self.t2_temperature * (self.pairwise_distance_threshold - pairwise_distances))
        
        # Set diagonal to 0 (i != j condition) for all events
        eye_mask = 1 - torch.eye(n_points, device=points_3d.device).unsqueeze(0)  # (1, n_points, n_points)
        t2_matrix = t2_matrix * eye_mask  # Broadcast to (n_events, n_points, n_points)
        
        # Step 3: Compute t3 (overall trigger probability) for all events
        # Compute t2_ij x t1_i x t1_j for all pairs and all events
        # t1_outer: (n_events, n_points, 1) * (n_events, 1, n_points) = (n_events, n_points, n_points)
        t1_outer = t1_values.unsqueeze(2) * t1_values.unsqueeze(1)
        pair_contributions = t2_matrix * t1_outer  # (n_events, n_points, n_points)
        
        # Sum over upper triangle only (to avoid counting pairs twice)
        upper_triangular_mask = torch.triu(torch.ones(n_points, n_points, device=points_3d.device), diagonal=1)
        upper_triangular_mask = upper_triangular_mask.unsqueeze(0)  # (1, n_points, n_points)
        pair_contributions_unique = pair_contributions * upper_triangular_mask
        
        # Sum all unique pair contributions for each event
        pair_sums = torch.sum(pair_contributions_unique, dim=2)  # Shape: (n_events, n_points) sum over each points neighbours
        
        # Apply final sigmoid for all events
        t3_values = torch.sigmoid(self.t3_temperature * (pair_sums - self.min_pairs_threshold))
        
        # Take softmax weighted sum to see if there exists 1 point with enough neighbours that fulfill the condition
        t_values = torch.sum(t3_values*torch.softmax(t3_values*self.t_temperature, dim=1), dim=1)
        return t_values  # Shape: (n_events,)
    
    def compute_trigger_probability_single_event(self, 
                                                   points_3d, 
                                                   event_params, 
                                                   surrogate_func,
                                                   string_weights=None,
                                                   precomputed_light_yield=None):
        """
        Compute trigger probability for a single event using three-step process.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Detector points, shape (n_points, 3)
        event_params : dict
            Event parameters
        surrogate_func : callable
            Light yield surrogate function
        string_weights : torch.Tensor or None
            Weights for each point (after sigmoid), shape (n_points,)
        precomputed_light_yield : torch.Tensor or None
            Precomputed light yields, shape (n_points,)
            
        Returns:
        --------
        dict
            Contains 't3' (trigger probability for this event), 't1_values', 't2_matrix', 'pair_sum'
        """
        n_points = len(points_3d)
        # time_start = time.time()
        # print("starting trigger computation...", flush=True)
        
        # Step 1: Compute t1 values (light yield condition)
        if precomputed_light_yield is not None:
            light_yields = precomputed_light_yield
        else:
            light_yields = surrogate_func(opt_point=points_3d, event_params=event_params)
        
        # print(f"Light yield computation time: {time.time() - time_start:.4f} s", flush=True)
        # time_start = time.time()
        
        # Default weights (if not provided, all points have weight 1)
        if string_weights is None:
            string_weights = torch.ones(n_points, device=points_3d.device)
        
        # t1_i = point_weight_i x sigmoid(light_yield_i - threshold)
        t1_sigmoid = torch.sigmoid(self.t1_temperature * (light_yields - self.light_yield_threshold))
        t1_values = string_weights * t1_sigmoid
        
        # print(f"T1 computation time: {time.time() - time_start:.4f} s", flush=True)
        # time_start = time.time()
        
        # Extract track parameters
        track_pos = event_params.get('position')
        zenith = event_params.get('zenith')
        azimuth = event_params.get('azimuth')
        
        # Ensure track_pos is 1D with shape (3,)
        if isinstance(track_pos, torch.Tensor):
            track_pos = track_pos.flatten()
        else:
            track_pos = torch.tensor(track_pos, device=points_3d.device).flatten()
        
        # Convert to direction vector
        if isinstance(zenith, torch.Tensor):
            theta = zenith.squeeze()
            phi = azimuth.squeeze()
        else:
            theta = torch.tensor(zenith, device=points_3d.device).squeeze()
            phi = torch.tensor(azimuth, device=points_3d.device).squeeze()
        
        track_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ])
        
        # Step 2: Compute t2 matrix (pairwise distance condition along track)
        # t2_ij = sigmoid(threshold - |point_i - point_j|_track) where i != j
        pairwise_distances = self.compute_pairwise_distances_along_track(points_3d, track_pos, track_dir)
        
        # Apply sigmoid: sigmoid(threshold - distance)
        t2_matrix = torch.sigmoid(self.t2_temperature * (self.pairwise_distance_threshold - pairwise_distances))
        
        # Set diagonal to 0 (i != j condition)
        t2_matrix = t2_matrix * (1 - torch.eye(n_points, device=points_3d.device))
        
        # print(f"T2 computation time: {time.time() - time_start:.4f} s", flush=True)
        # time_start = time.time()
        
        # Step 3: Compute t3 (overall trigger probability)
        # t3 = sigmoid(sum_ij(t2_ij x t1_i x t1_j) - threshold)
        # where i != j and pairs are not repeated
        
        # Compute t2_ij x t1_i x t1_j for all pairs
        # Using broadcasting: t1_i (n,1) * t1_j (1,n) * t2_ij (n,n)
        t1_outer = t1_values.unsqueeze(1) * t1_values.unsqueeze(0)  # (n, n)
        pair_contributions = t2_matrix * t1_outer  # (n, n)
        
        # Sum over upper triangle only (to avoid counting pairs twice)
        # Create upper triangular mask (excluding diagonal)
        upper_triangular_mask = torch.triu(torch.ones(n_points, n_points, device=points_3d.device), diagonal=1)
        pair_contributions_unique = pair_contributions * upper_triangular_mask
        
        # Sum all unique pair contributions
        pair_sum = torch.sum(pair_contributions_unique, dim=1)  # Shape: (n_points,) sum over each points neighbours
        
        # Apply final sigmoid
        t3 = torch.sigmoid(self.t3_temperature * (pair_sum - self.min_pairs_threshold))
        
        # print(f"T3 computation time: {time.time() - time_start:.4f} s", flush=True)
        t_values = torch.sum(t3*torch.softmax(t3*self.t_temperature))
        return {
            't3_values': t3,
            't1_values': t1_values,
            't2_values': t2_matrix,
            't_values': t_values
        }
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute trigger loss for detector geometry (batched version).
        
        Parameters:
        -----------
        geom_dict : dict
            Dictionary containing 'points_3d' and optionally 'string_xy' and 'string_weights'.
            If 'string_xy' and 'string_weights' are provided, weights will be mapped to points
            based on xy positions (points at same xy get same weight).
        **kwargs : dict
            Must contain:
            - signal_surrogate_func : callable
            - signal_event_params : list of dict (event parameters)
            Optional:
            - precomputed_light_yield_per_point_per_event : torch.Tensor, shape (n_events, n_points)
            
        Returns:
        --------
        dict
            Contains 'trigger_loss' and 'detector_efficiency'
        """
        points_3d = geom_dict.get('points_3d', None)
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        event_params_list = kwargs.get('signal_event_params', None)
        precomputed_light_yield = kwargs.get('precomputed_light_yield_per_point_per_event', None)
        
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        
        # Generate events if not provided
        if event_params_list is None and signal_sampler is not None:
            event_params_list = [signal_sampler.sample() for _ in range(num_events)]
        
        if points_3d is None or surrogate_func is None or event_params_list is None:
            raise ValueError("points_3d, signal_surrogate_func, and signal_event_params must be provided")
        
        # Map string weights to point weights if string_xy is provided
        point_weights = None
        if string_weights is not None:
            if string_xy is None:
                # Get unique xy coordinates in order of first appearance
                seen = set()
                unique_indices = []
                for i, xy in enumerate(points_3d[:, :2]):
                    xy_tuple = tuple(xy.tolist())
                    if xy_tuple not in seen:
                        seen.add(xy_tuple)
                        unique_indices.append(i)
                
                string_xy = points_3d[unique_indices, :2]
            
            point_weights = self.map_string_weights_to_points(points_3d, string_xy, string_weights)
        
        # Batch compute trigger probabilities for all events
        if precomputed_light_yield is None:
            precomputed_light_yield = torch.zeros((len(event_params_list), len(points_3d)), device=self.device)
            for i, event_params in enumerate(event_params_list):
                precomputed_light_yield[i] = surrogate_func(opt_point=points_3d, event_params=event_params)
        
        t_values = self.compute_trigger_probability_batch_events(
            points_3d=points_3d,
            event_params_list=event_params_list,
            surrogate_func=surrogate_func,
            string_weights=point_weights,
            precomputed_light_yield=precomputed_light_yield
        )
        
        # Calculate detector efficiency: mean of t3 values
        detector_efficiency = torch.mean(t_values)
        
        # Calculate loss: 1 - detector efficiency
        trigger_loss = 1.0 - detector_efficiency
        
        if self.print_loss:
            print(f"Trigger Loss: {trigger_loss.item():.6f}")
            print(f"Detector Efficiency: {detector_efficiency.item():.6f}")
            print(f"Mean T3: {detector_efficiency.item():.6f}")
        
        return {
            'trigger_loss': trigger_loss,
            'detector_efficiency': detector_efficiency,
            't_per_event': t_values
        }
