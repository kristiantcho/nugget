import torch
import numpy as np
from itertools import combinations
from nugget.losses.base_loss import LossFunction


class TriggerLoss(LossFunction):
    """
    Loss function for detector trigger efficiency based on multi-level conditions.
    
    This loss evaluates detector efficiency by testing whether events will be detected
    given three cascading conditions:
    1. Individual points must detect sufficient light yield (above threshold)
    2. Minimum number of points must pass condition 1
    3. Maximum distance along track for selected points must be below threshold
    """
    
    def __init__(self, 
                 device=None, 
                 light_yield_threshold=10.0,
                 min_triggered_points=5,
                 max_track_distance=500.0,
                 light_yield_sigmoid_sharpness=1.0,
                 distance_sigmoid_sharpness=0.01,
                 softmax_temperature=10.0,
                 weight_sigmoid_sharpness=1.0,
                 print_loss=False):
        """
        Initialize the trigger loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        light_yield_threshold : float
            Minimum light yield threshold for point detection.
        min_triggered_points : int
            Minimum number of points that must be triggered (N).
        max_track_distance : float
            Maximum distance along track for triggered point spread.
        light_yield_sigmoid_sharpness : float
            Sharpness parameter for light yield sigmoid (higher = sharper transition).
        distance_sigmoid_sharpness : float
            Sharpness parameter for distance sigmoid (higher = sharper transition).
        softmax_temperature : float
            Temperature parameter for softmax (higher = closer to max function).
        weight_sigmoid_sharpness : float
            Sharpness parameter for string weight sigmoid (higher = sharper transition).
        print_loss : bool
            Whether to print loss components during computation.
        """
        super().__init__(device)
        
        self.light_yield_threshold = light_yield_threshold
        self.min_triggered_points = min_triggered_points
        self.max_track_distance = max_track_distance
        self.light_yield_sigmoid_sharpness = light_yield_sigmoid_sharpness
        self.distance_sigmoid_sharpness = distance_sigmoid_sharpness
        self.softmax_temperature = softmax_temperature
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
        
        # Apply sigmoid to weights
        point_weights_sigmoid = torch.sigmoid(self.weight_sigmoid_sharpness * point_weights)
        
        return point_weights_sigmoid
    
    def compute_light_yield_per_point(self, points_3d, event_params, surrogate_func):
        """
        Compute light yield at each detector point for given event parameters.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Detector point positions, shape (n_points, 3)
        event_params : dict
            Event parameters containing position, zenith, azimuth, energy
        surrogate_func : callable
            Light yield surrogate function
            
        Returns:
        --------
        torch.Tensor
            Light yield at each point, shape (n_points,)
        """
        light_yields = []
        for point in points_3d:
            ly = surrogate_func(opt_point=point, event_params=event_params)
            light_yields.append(ly)
        
        return torch.stack(light_yields)
    
    def compute_track_projection(self, points, track_pos, track_dir):
        """
        Compute projection of points onto the track line.
        
        Parameters:
        -----------
        points : torch.Tensor
            Points, shape (n_points, 3)
        track_pos : torch.Tensor
            Point on track line, shape (3,)
        track_dir : torch.Tensor
            Track direction (normalized), shape (3,)
            
        Returns:
        --------
        torch.Tensor
            Projection distances along track, shape (n_points,)
        """
        # Normalize direction
        track_dir = track_dir / torch.norm(track_dir)
        
        # Calculate projection: (p - x0) · d
        diff = points - track_pos.unsqueeze(0)
        projections = torch.sum(diff * track_dir.unsqueeze(0), dim=1)
        
        return projections
    
    def max_distance_along_track(self, points, track_pos, track_dir):
        """
        Calculate maximum distance between neighboring points along the track line.
        
        This computes the projections of all points onto the track, sorts them,
        and returns the maximum distance between consecutive points.
        
        Parameters:
        -----------
        points : torch.Tensor
            Set of points, shape (n_points, 3)
        track_pos : torch.Tensor
            Point on track line, shape (3,)
        track_dir : torch.Tensor
            Track direction, shape (3,)
            
        Returns:
        --------
        torch.Tensor
            Maximum distance along track between consecutive neighboring points
        """
        projections = self.compute_track_projection(points, track_pos, track_dir)
        
        # Sort projections to get ordered positions along track
        sorted_projections = torch.sort(projections)[0]
        
        # Calculate distances between consecutive points
        if len(sorted_projections) < 2:
            return torch.tensor(0.0, device=self.device)
        
        consecutive_distances = sorted_projections[1:] - sorted_projections[:-1]
        
        # Return maximum distance between neighbors
        max_dist = torch.max(consecutive_distances)
        return max_dist
    
    def compute_trigger_probability_single_event(self, 
                                                   points_3d, 
                                                   event_params, 
                                                   surrogate_func,
                                                   string_weights=None,
                                                   precomputed_light_yield=None):
        """
        Compute trigger probability for a single event.
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            Detector points, shape (n_points, 3)
        event_params : dict
            Event parameters
        surrogate_func : callable
            Light yield surrogate function
        string_weights : torch.Tensor or None
            Weights for each point, shape (n_points,)
        precomputed_light_yield : torch.Tensor or None
            Precomputed light yields, shape (n_points,)
            
        Returns:
        --------
        dict
            Contains 'weighted_sum' (for loss), 'max_prob' (for efficiency), 
            't1_values', 't2_values', 't3_values'
        """
        n_points = len(points_3d)
        
        # Step 1: Compute light yield condition (t1 values)
        if precomputed_light_yield is not None:
            light_yields = precomputed_light_yield
        else:
            light_yields = self.compute_light_yield_per_point(points_3d, event_params, surrogate_func)
        
        # Sigmoid of (light_yield - threshold)
        t1 = torch.sigmoid(self.light_yield_sigmoid_sharpness * (light_yields - self.light_yield_threshold))
        
        # Default weights
        if string_weights is None:
            string_weights = torch.ones(n_points, device=self.device)
        
        # Step 2: Compute all combinations of N points
        N = self.min_triggered_points
        
        if N > n_points:
            # Not enough points to trigger
            return {
                'weighted_sum': torch.tensor(0.0, device=self.device),
                'max_prob': torch.tensor(0.0, device=self.device),
                't1_values': t1,
                't2_values': torch.tensor([], device=self.device),
                't3_values': torch.tensor([], device=self.device)
            }
        
        # Generate all combinations of N points
        point_indices = list(range(n_points))
        all_combinations = list(combinations(point_indices, N))
        n_combinations = len(all_combinations)
        
        # Calculate t2 values (weighted average of t1 for each combination)
        t2_values = torch.zeros(n_combinations, device=self.device)
        for i, combo in enumerate(all_combinations):
            combo_indices = torch.tensor(combo, device=self.device)
            combo_t1 = t1[combo_indices]
            combo_weights = string_weights[combo_indices]
            
            # Weighted average
            t2_values[i] = torch.sum(combo_t1 * combo_weights) / N
        
        # Step 3: Compute track distance condition (t3 values)
        # Extract track parameters
        track_pos = event_params.get('position')
        zenith = event_params.get('zenith')
        azimuth = event_params.get('azimuth')
        
        # Convert to direction vector
        if isinstance(zenith, torch.Tensor):
            theta = zenith.squeeze()
            phi = azimuth.squeeze()
        else:
            theta = torch.tensor(zenith, device=self.device).squeeze()
            phi = torch.tensor(azimuth, device=self.device).squeeze()
        
        track_dir = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ])
        
        # Calculate t3 values
        t3_values = torch.zeros(n_combinations, device=self.device)
        for i, combo in enumerate(all_combinations):
            combo_indices = torch.tensor(combo, device=self.device)
            combo_points = points_3d[combo_indices]
            
            # Maximum distance along track
            max_dist = self.max_distance_along_track(combo_points, track_pos, track_dir)
            
            # Sigmoid of (threshold - max_distance)
            distance_sigmoid = torch.sigmoid(
                self.distance_sigmoid_sharpness * (self.max_track_distance - max_dist)
            )
            
            # Multiply with t2
            t3_values[i] = t2_values[i] * distance_sigmoid
        
        # Calculate maximum (for efficiency)
        max_prob = torch.max(t3_values) if len(t3_values) > 0 else torch.tensor(0.0, device=self.device)
        
        # Calculate softmax weighted sum (for loss)
        if len(t3_values) > 0:
            softmax_weights = torch.softmax(self.softmax_temperature * t3_values, dim=0)
            weighted_sum = torch.sum(softmax_weights * t3_values)
        else:
            weighted_sum = torch.tensor(0.0, device=self.device)
        
        return {
            'weighted_sum': weighted_sum,
            'max_prob': max_prob,
            't1_values': t1,
            't2_values': t2_values,
            't3_values': t3_values
        }
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute trigger loss for detector geometry.
        
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
            if string_xy is not None:
                # If string_weights is provided without string_xy, assume it's already per-point
                # and apply sigmoid
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
        
        n_events = len(event_params_list)
        
        # Process each event
        total_weighted_sum = torch.tensor(0.0, device=self.device)
        total_max_prob = torch.tensor(0.0, device=self.device)
        
        for i, event_params in enumerate(event_params_list):
            # Get precomputed light yield for this event if available
            event_light_yield = None
            if precomputed_light_yield is not None:
                event_light_yield = precomputed_light_yield[i]
            
            result = self.compute_trigger_probability_single_event(
                points_3d=points_3d,
                event_params=event_params,
                surrogate_func=surrogate_func,
                string_weights=point_weights,  # Use mapped point weights
                precomputed_light_yield=event_light_yield
            )
            
            total_weighted_sum += result['weighted_sum']
            total_max_prob += result['max_prob']
        
        # Calculate detector efficiency (using max)
        detector_efficiency = total_max_prob / n_events
        
        # Calculate loss (using softmax weighted sum)
        trigger_loss = (n_events - total_weighted_sum) / n_events
  
        
        # if self.print_loss:
        #     print(f"Trigger Loss: {trigger_loss.item():.6f}")
        #     print(f"Detector Efficiency: {detector_efficiency.item():.6f}")
        #     print(f"Total Weighted Sum: {total_weighted_sum.item():.6f}")
        #     print(f"Total Max Prob: {total_max_prob.item():.6f}")
        
        return {
            'trigger_loss': trigger_loss,
            'detector_efficiency': detector_efficiency,
            'total_weighted_sum': total_weighted_sum,
            'total_max_prob': total_max_prob
        }
