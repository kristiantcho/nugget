import torch
from nugget.losses.base_loss import LossFunction
from nugget.losses.fisher_info import WeightedResolutionLoss

class TriggerLoss(LossFunction):
    """
    Trigger loss based on a sliding bar along the event track.

    For each event:
    1. Compute per-point activity weights t1.
    2. Project points onto the track direction.
    3. Slide a 1D bar across the projection range.
    4. For each bar position, score whether enough t1-weighted points are inside.
    5. Aggregate bar scores with a softmax-weighted sum.
    """
    
    def __init__(self, 
                 device=None, 
                 light_yield_threshold=6.0,
                 distance_bar_length=550.0,
                 distance_bar_step=None,
                 min_points_threshold=30.0,
                 t1_temperature=1.0,
                 t3_temperature=1.0,
                 t_temperature=1.0,
                 use_hard_cuts=False,
                 print_loss=False):
        """
        Initialize the trigger loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        light_yield_threshold : float
            Minimum light yield threshold for point detection (t1).
        distance_bar_length : float
            Length of the sliding bar along the track projection.
        distance_bar_step : float or None
            Step size for sliding the bar. If None, bar positions are at each point's projection
            along the track, continuing until the bar's right end passes the last point.
        min_points_threshold : float
            Minimum t1-weighted points inside a bar to produce strong trigger response.
        t1_temperature : float
            Temperature for t1 sigmoid (higher = sharper transition).
        t3_temperature : float
            Temperature for per-bar thresholding sigmoid (higher = sharper transition).
        t_temperature : float
            Temperature for aggregating bar scores (higher = more focus on max).
        use_hard_cuts : bool
            If True, use hard thresholds to produce binary trigger outputs (0/1) instead of
            smooth differentiable sigmoid/softmax aggregations.
        weight_sigmoid_sharpness : float
            Sharpness parameter for string weight sigmoid (higher = sharper transition).
        print_loss : bool
            Whether to print loss components during computation.
        """
        super().__init__(device)
        
        self.light_yield_threshold = light_yield_threshold
        self.distance_bar_length = distance_bar_length
        self.distance_bar_step = distance_bar_step
        self.min_points_threshold = min_points_threshold
        self.t1_temperature = t1_temperature
        self.t3_temperature = t3_temperature
        self.t_temperature = t_temperature
        self.use_hard_cuts = use_hard_cuts
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
        point_weights_sigmoid = torch.sigmoid(point_weights)
        
        return point_weights_sigmoid
    
    def compute_track_projection(self, points_3d, track_pos, track_dir):
        """
        Project all points onto the track direction.

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
            Scalar projections along the track, shape (n_points,)
        """
        track_dir = track_dir / torch.norm(track_dir)
        diff = points_3d - track_pos.unsqueeze(0)
        return torch.sum(diff * track_dir.unsqueeze(0), dim=1)
    
    def compute_trigger_probability_single_event(self, 
                                                   points_3d, 
                                                   event_params, 
                                                   surrogate_func,
                                                   string_weights=None,
                                                   precomputed_light_yield=None):
        """
        Compute trigger probability for a single event using a sliding bar process.
        
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
            Contains per-bar scores, point activities, and final trigger score.
        """
        n_points = len(points_3d)
        
        # Step 1: Compute t1 values (light yield condition)
        if precomputed_light_yield is not None:
            light_yields = precomputed_light_yield
        else:
            light_yields = surrogate_func(opt_point=points_3d, event_params=event_params)
        
        # Default weights (if not provided, all points have weight 1)
        if string_weights is None:
            string_weights = torch.ones(n_points, device=points_3d.device)
        
        # t1_i: soft (sigmoid) or hard (binary threshold)
        if self.use_hard_cuts:
            t1_values = (light_yields >= self.light_yield_threshold).to(light_yields.dtype)
        else:
            t1_sigmoid = torch.sigmoid(self.t1_temperature * (light_yields - self.light_yield_threshold))
            t1_values = string_weights * t1_sigmoid
        
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

        # Step 2: Project points onto track
        projections = self.compute_track_projection(points_3d, track_pos, track_dir)
        s_min = torch.min(projections)
        s_max = torch.max(projections)

        # Step 3: Slide distance bar from start to end of geometry along track
        bar_length = torch.as_tensor(self.distance_bar_length, device=points_3d.device, dtype=projections.dtype)

        if self.distance_bar_step is None:
            # Point-by-point stepping: bar starts at each point's projection
            bar_starts = torch.sort(projections)[0]
            # Ensure we cover until the bar's right end goes past the last point
            if bar_starts[-1] + bar_length < s_max:
                bar_starts = torch.cat([bar_starts, s_max.unsqueeze(0)])
        else:
            # Uniform stepping with fixed step size
            bar_step = torch.as_tensor(self.distance_bar_step, device=points_3d.device, dtype=projections.dtype)
            effective_span = torch.clamp(s_max - s_min - bar_length, min=0.0)

            if effective_span <= 0:
                bar_starts = s_min.unsqueeze(0)
            else:
                bar_starts = torch.arange(s_min, s_min + effective_span + bar_step, bar_step, device=points_3d.device)
                if bar_starts[-1] < s_min + effective_span:
                    bar_starts = torch.cat([bar_starts, (s_min + effective_span).unsqueeze(0)])

        bar_ends = bar_starts + bar_length

        # For each bar position, accumulate t1 from points that fall inside the bar.
        point_s = projections.unsqueeze(0)
        in_bar_mask = (point_s >= bar_starts.unsqueeze(1)) & (point_s <= bar_ends.unsqueeze(1))
        bar_activity = torch.sum(in_bar_mask.float() * t1_values.unsqueeze(0), dim=1)

        # Threshold each bar and aggregate.
        if self.use_hard_cuts:
            t3 = (bar_activity >= self.min_points_threshold).to(bar_activity.dtype)
            t_values = torch.max(t3)
        else:
            t3 = torch.sigmoid(self.t3_temperature * (bar_activity - self.min_points_threshold))
            # Soft-max over bars: an event triggers if ANY bar has enough activity.
            # Weight by bar_activity (which meaningfully ranks bars) rather than by
            # t3 itself; t3 saturates at 1, which makes softmax(t3) near-uniform and
            # dilutes a single strongly-triggering bar by 1/n_bars (capping t_values
            # well below 1 regardless of geometry).
            t_values = torch.sum(t3 * torch.softmax(bar_activity * self.t_temperature, dim=0))

        return {
            't3_values': t3,
            't1_values': t1_values,
            'bar_starts': bar_starts,
            'bar_ends': bar_ends,
            'bar_activity': bar_activity,
            't_values': t_values
            }

    def compute_trigger_probability_batched_events(
        self,
        points_3d,
        event_params_list,
        surrogate_func,
        string_weights=None,
        precomputed_light_yield=None,
    ):
        """Compute trigger probability for many events.

        Parameters
        ----------
        points_3d : torch.Tensor
            Detector points, shape (n_points, 3)
        event_params_list : list of dict
            Per-event parameters. Each dict must contain:
            - 'position': (3,)
            - 'zenith': scalar
            - 'azimuth': scalar
        surrogate_func : callable
            Light yield surrogate function. If precomputed yields are not provided, this will
            be called once per event (since it only supports one event at a time).
        string_weights : torch.Tensor or None
            Point weights after sigmoid, shape (n_points,)
        precomputed_light_yield : torch.Tensor or None
            Precomputed light yields, shape (n_events, n_points)

        Returns
        -------
        dict
            Batched outputs. Bar-related tensors are dense:
            - 't_values': (n_events,)
            - 't1_values': (n_events, n_points)
            - 't3_values': (n_events, n_bars)
            - 'bar_activity': (n_events, n_bars)
            - 'bar_starts': (n_events, n_bars) or (n_bars,)
            - 'bar_ends': (n_events, n_bars) or (n_bars,)
        """
        n_points = len(points_3d)
        n_events = len(event_params_list)

        # Step 1: light yields per event per point
        if precomputed_light_yield is not None:
            light_yields = precomputed_light_yield
        else:
            if surrogate_func is None:
                raise ValueError("surrogate_func must be provided when precomputed_light_yield is None")
            light_yields = []
            for event_params in event_params_list:
                light_yields.append(surrogate_func(opt_point=points_3d, event_params=event_params))
            light_yields = torch.stack(light_yields, dim=0)

        if light_yields.ndim != 2:
            raise ValueError(
                "Batched trigger expects light_yields with shape (n_events, n_points). "
                f"Got shape {tuple(light_yields.shape)}"
            )
        if light_yields.shape[0] != n_events:
            raise ValueError(
                f"Mismatch: got {light_yields.shape[0]} events in light_yields but {n_events} event_params"
            )

        # Default point weights
        if string_weights is None:
            string_weights = torch.ones(n_points, device=points_3d.device, dtype=light_yields.dtype)
        else:
            string_weights = string_weights.to(device=points_3d.device, dtype=light_yields.dtype)

        # t1(e, i): soft (sigmoid) or hard (binary threshold)
        if self.use_hard_cuts:
            t1_values = (light_yields >= self.light_yield_threshold).to(light_yields.dtype) * string_weights.unsqueeze(0)
        else:
            t1_sigmoid = torch.sigmoid(self.t1_temperature * (light_yields - self.light_yield_threshold))
            t1_values = t1_sigmoid * string_weights.unsqueeze(0)

        # Track parameters (batched)
        track_pos = torch.stack(
            [
                ep['position'].flatten() if isinstance(ep['position'], torch.Tensor)
                else torch.tensor(ep['position'], device=points_3d.device).flatten()
                for ep in event_params_list
            ],
            dim=0,
        ).to(dtype=points_3d.dtype, device=points_3d.device)

        theta = torch.stack(
            [
                ep['zenith'].squeeze() if isinstance(ep['zenith'], torch.Tensor)
                else torch.tensor(ep['zenith'], device=points_3d.device).squeeze()
                for ep in event_params_list
            ],
            dim=0,
        ).to(dtype=points_3d.dtype, device=points_3d.device)

        phi = torch.stack(
            [
                ep['azimuth'].squeeze() if isinstance(ep['azimuth'], torch.Tensor)
                else torch.tensor(ep['azimuth'], device=points_3d.device).squeeze()
                for ep in event_params_list
            ],
            dim=0,
        ).to(dtype=points_3d.dtype, device=points_3d.device)

        track_dir = torch.stack(
            [
                torch.sin(theta) * torch.cos(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=1,
        )  # (n_events, 3)
        track_dir = track_dir / torch.norm(track_dir, dim=1, keepdim=True)

        # Step 2: projections per event per point: (n_events, n_points)
        diff = points_3d.unsqueeze(0) - track_pos.unsqueeze(1)
        projections = torch.sum(diff * track_dir.unsqueeze(1), dim=2)
        s_min = torch.min(projections, dim=1).values
        s_max = torch.max(projections, dim=1).values

        bar_length = torch.as_tensor(
            self.distance_bar_length, device=points_3d.device, dtype=projections.dtype
        )

        if self.distance_bar_step is None:
            # Point-by-point stepping: for each event use all projected points as bar starts.
            # This yields exactly n_points bars per event, so we can batch this path fully.
            bar_starts = torch.sort(projections, dim=1).values  # (n_events, n_points)
            bar_ends = bar_starts + bar_length

            point_s = projections.unsqueeze(1)  # (n_events, 1, n_points)
            in_bar_mask = (point_s >= bar_starts.unsqueeze(2)) & (point_s <= bar_ends.unsqueeze(2))
            bar_activity = torch.sum(
                in_bar_mask.to(dtype=t1_values.dtype) * t1_values.unsqueeze(1),
                dim=2,
            )  # (n_events, n_points)

            if self.use_hard_cuts:
                t3 = (bar_activity >= self.min_points_threshold).to(bar_activity.dtype)
                t_values = torch.max(t3, dim=1).values
            else:
                t3 = torch.sigmoid(self.t3_temperature * (bar_activity - self.min_points_threshold))
                # Soft-max over bars weighted by bar_activity (not t3): see the
                # single-event path for why weighting by t3 caps t_values ~1/n_bars.
                t_values = torch.sum(t3 * torch.softmax(bar_activity * self.t_temperature, dim=1), dim=1)

            return {
                't3_values': t3,
                't1_values': t1_values,
                'bar_starts': bar_starts,
                'bar_ends': bar_ends,
                'bar_activity': bar_activity,
                't_values': t_values,
            }

        # Fixed stepping: same number of bars per event is achievable via padding to max bars
        bar_step = torch.as_tensor(self.distance_bar_step, device=points_3d.device, dtype=projections.dtype)
        effective_span = torch.clamp(s_max - s_min - bar_length, min=0.0)  # (n_events,)
        n_steps = torch.floor(effective_span / bar_step).to(torch.long) + 1
        max_steps = int(torch.max(n_steps).item())

        # Build bar_starts per event: start + step*k, clamped to last allowed start
        k = torch.arange(max_steps, device=points_3d.device, dtype=projections.dtype).unsqueeze(0)  # (1, K)
        bar_starts = s_min.unsqueeze(1) + k * bar_step  # (n_events, K)
        last_start = s_min + effective_span
        bar_starts = torch.minimum(bar_starts, last_start.unsqueeze(1))

        # Mask out bars beyond n_steps per event
        valid_bar = k < n_steps.unsqueeze(1).to(dtype=projections.dtype)
        valid_bar_bool = valid_bar.bool()

        bar_ends = bar_starts + bar_length

        # For each event and bar, accumulate t1 from points that fall inside the bar.
        point_s = projections.unsqueeze(1)  # (n_events, 1, n_points)
        in_bar_mask = (point_s >= bar_starts.unsqueeze(2)) & (point_s <= bar_ends.unsqueeze(2))
        bar_activity = torch.sum(in_bar_mask.to(dtype=t1_values.dtype) * t1_values.unsqueeze(1), dim=2)  # (n_events, K)

        # Zero out invalid bars (so they don't contribute)
        bar_activity = torch.where(valid_bar_bool, bar_activity, torch.zeros_like(bar_activity))

        if self.use_hard_cuts:
            t3 = (bar_activity >= self.min_points_threshold).to(bar_activity.dtype)
            t3 = torch.where(valid_bar_bool, t3, torch.zeros_like(t3))
            t_values = torch.max(t3, dim=1).values
        else:
            t3 = torch.sigmoid(self.t3_temperature * (bar_activity - self.min_points_threshold))
            t3 = torch.where(valid_bar_bool, t3, torch.zeros_like(t3))

            # Soft-max over bars weighted by bar_activity (not t3): see the
            # single-event path for why weighting by t3 caps t_values ~1/n_bars.
            # Invalid (padding) bars get -inf logits so they take zero softmax weight.
            softmax_logits = torch.where(
                valid_bar_bool,
                bar_activity * self.t_temperature,
                torch.full_like(bar_activity, torch.finfo(bar_activity.dtype).min),
            )
            t_values = torch.sum(t3 * torch.softmax(softmax_logits, dim=1), dim=1)

        return {
            't3_values': t3,
            't1_values': t1_values,
            'bar_starts': bar_starts,
            'bar_ends': bar_ends,
            'bar_activity': bar_activity,
            't_values': t_values,
        }

  
    
    def _compute_chunk_light_yield(self, points_3d, event_params_chunk, surrogate_func, batched_surrogate_func, detach_light_yields):
        """Compute (or fetch) light yields for one chunk of events."""
        if batched_surrogate_func is not None:
            light_yield = batched_surrogate_func(om_positions=points_3d, event_params_list=event_params_chunk)
        else:
            light_yield = torch.stack(
                [surrogate_func(opt_point=points_3d, event_params=ep) for ep in event_params_chunk],
                dim=0,
            )
        if detach_light_yields:
            light_yield = light_yield.detach()
        return light_yield

    def _compute_t_values(self, points_3d, event_params_list, surrogate_func, point_weights, precomputed_light_yield, use_batched_trigger):
        """Compute per-event trigger t-values for a single (already chunked) batch of events."""
        if use_batched_trigger is None:
            use_batched_trigger = precomputed_light_yield is not None

        if use_batched_trigger:
            event_trigger = self.compute_trigger_probability_batched_events(
                points_3d=points_3d,
                event_params_list=event_params_list,
                surrogate_func=surrogate_func,
                string_weights=point_weights,
                precomputed_light_yield=precomputed_light_yield,
            )
            return event_trigger['t_values']

        t_values = []
        for i, event_params in enumerate(event_params_list):
            event_light_yield = None
            if precomputed_light_yield is not None:
                event_light_yield = precomputed_light_yield[i]

            event_trigger = self.compute_trigger_probability_single_event(
                points_3d=points_3d,
                event_params=event_params,
                surrogate_func=surrogate_func,
                string_weights=point_weights,
                precomputed_light_yield=event_light_yield,
            )
            t_values.append(event_trigger['t_values'])

        return torch.stack(t_values)

    def __call__(self, geom_dict, **kwargs):
        """
        Compute trigger loss for detector geometry, chunking over events internally to cap memory.

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
            - batched_surrogate_func : callable, used to compute light yields per chunk when
              precomputed yields are not provided
            - binned_trigger_batch_size : int, max number of events processed per chunk
              (None processes all events in a single chunk)
            - detach_light_yields : bool, detach computed light yields from the autograd graph
            - perfect_efficiency : bool, short-circuits to a trigger value of 1 for every event

        Returns:
        --------
        dict
            Contains 'trigger_loss', 'detector_efficiency', and 't_per_event'
        """
        points_3d = geom_dict.get('points_3d', None)
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)

        surrogate_func = kwargs.get('signal_surrogate_func', None)
        event_params_list = kwargs.get('signal_event_params', None)
        precomputed_light_yield = kwargs.get('precomputed_light_yield_per_point_per_event', None)
        batched_surrogate_func = kwargs.get('batched_surrogate_func', None)
        chunk_size = kwargs.get('binned_trigger_batch_size', None)
        detach_light_yields = kwargs.get('detach_light_yields', False)

        # Optional override: choose computation mode
        # - None: auto (batched if precomputed yields provided, else single loop)
        # - True: force batched
        # - False: force single-event loop
        use_batched_trigger = kwargs.get('use_batched_trigger', None)

        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)

        # Optional per-call overrides for sliding-bar trigger parameters.
        self.distance_bar_length = kwargs.get('distance_bar_length', self.distance_bar_length)
        self.distance_bar_step = kwargs.get('distance_bar_step', self.distance_bar_step)
        self.min_points_threshold = kwargs.get('min_points_threshold', self.min_points_threshold)

        # Generate events if not provided
        if event_params_list is None and signal_sampler is not None:
            event_params_list = [signal_sampler.sample() for _ in range(num_events)]

        if points_3d is None or surrogate_func is None or event_params_list is None:
            raise ValueError("points_3d, signal_surrogate_func, and signal_event_params must be provided")

        n_events = len(event_params_list)

        # Perfect trigger short-circuit: every event triggers with probability 1.
        if kwargs.get('perfect_efficiency', False):
            t_values = torch.ones(n_events, device=points_3d.device, dtype=points_3d.dtype)
            detector_efficiency = torch.mean(t_values)
            trigger_loss = 1.0 - detector_efficiency
            return {
                'trigger_loss': trigger_loss,
                'detector_efficiency': detector_efficiency,
                't_per_event': t_values
            }

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

        if chunk_size is None or chunk_size >= n_events:
            # Single pass — precompute light yields with batched surrogate if available
            ly = precomputed_light_yield
            if ly is None and batched_surrogate_func is not None:
                ly = self._compute_chunk_light_yield(
                    points_3d, event_params_list, surrogate_func, batched_surrogate_func, detach_light_yields
                )
            elif ly is not None and detach_light_yields:
                ly = ly.detach()

            t_values = self._compute_t_values(
                points_3d, event_params_list, surrogate_func, point_weights, ly, use_batched_trigger
            )
        else:
            # Chunked pass — process chunk_size events at a time to cap memory
            t_values = torch.zeros(n_events, device=points_3d.device, dtype=points_3d.dtype)
            for chunk_start in range(0, n_events, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_events)
                chunk_events = event_params_list[chunk_start:chunk_end]

                if precomputed_light_yield is not None:
                    chunk_ly = precomputed_light_yield[chunk_start:chunk_end]
                    if detach_light_yields:
                        chunk_ly = chunk_ly.detach()
                elif batched_surrogate_func is not None:
                    chunk_ly = self._compute_chunk_light_yield(
                        points_3d, chunk_events, surrogate_func, batched_surrogate_func, detach_light_yields
                    )
                else:
                    chunk_ly = None

                chunk_t_values = self._compute_t_values(
                    points_3d, chunk_events, surrogate_func, point_weights, chunk_ly, use_batched_trigger
                )
                t_values[chunk_start:chunk_end] = chunk_t_values

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


class ResolutionSelectionLoss(LossFunction):
    """
    Resolution selection loss based on Fisher information.
    This loss encourages the detector geometry to achieve a desired resolution for specific event parameters.
    Parameters:
    -----------
    device : torch.device or None
        Device to use for computations.
    resolution_type : str
        Type of resolution to consider. Options are 'angular' or 'energy'.
    fisher_info_params : list of str
        List of event parameters to consider for Fisher information..
    """
    


    def __init__(self, 
                 device=None, 
                 resolution_type='angular',
                 fisher_info_params=['energy', 'azimuth', 'zenith'],
                 ):
        
        super().__init__(device)
        self.resolution_type = resolution_type
        self.fisher_info_params = fisher_info_params
      


    def soft_between(self, selection_thresholds, lower, upper, temperature=1.0):

        """
        Compute a soft mask for whether [lower, upper] intersects the selection threshold range.

        Parameters:
        -----------
        selection_thresholds : list
            List containing lower and upper thresholds range, shape (2,)
        lower : float or torch.Tensor
            Lower bound(s), shape (n_events,) or scalar
        upper : float or torch.Tensor
            Upper bound(s), shape (n_events,) or scalar
        temperature : float
            Temperature for the soft selection (higher = sharper transition)
        """
        # Intersection requires lower <= thresholds[1] and upper >= thresholds[0]
        lower_mask = torch.sigmoid(temperature * (selection_thresholds[1] - lower))
        upper_mask = torch.sigmoid(temperature * (upper - selection_thresholds[0]))
        return lower_mask * upper_mask

    def __call__(self, geom_dict, **kwargs):
        """
        Compute resolution selection loss for detector geometry.
        
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
            - precomputed_fisher_info_per_string_per_event : torch.Tensor, shape (n_events, n_strings, 3, 3)
        """
        selection_soft_temperature = kwargs.get('selection_soft_temperature', 1.0)
        precalculated_resolution_loss = kwargs.get('precalculated_resolution_loss', None)
        
        selection_thresholds=kwargs.get('selection_thresholds', [-1, 0.1])
        self.selection_soft_temperature = kwargs.get('selection_soft_temperature', selection_soft_temperature)
        if precalculated_resolution_loss is None:    
            weighted_resolution_loss=WeightedResolutionLoss(
                device=self.device,
                resolution_type=self.resolution_type,
                fisher_info_params=self.fisher_info_params
            ) 
            loss_stuff = weighted_resolution_loss(geom_dict, **kwargs)
            resolution_per_event = loss_stuff['resolution_per_event'].squeeze()
            signal_event_params = loss_stuff['resolution_params']
        else:
            resolution_per_event = precalculated_resolution_loss['resolution_per_event'].squeeze()
            signal_event_params = precalculated_resolution_loss['resolution_params']
        true_params = []
        if self.resolution_type =='angular':
            for event in signal_event_params:
                if 'zenith' not in event or 'azimuth' not in event:
                    raise ValueError("For angular resolution, each event must have 'zenith' and 'azimuth' parameters.")
                true_params.append(event['zenith'])
        elif self.resolution_type =='energy':
            for event in signal_event_params:
                if 'energy' not in event:
                    raise ValueError("For energy resolution, each event must have 'energy' parameter.")
                true_params.append(event['energy'])
        else:
            raise ValueError(f"Unsupported resolution type: {self.resolution_type}. Supported types are 'angular' and 'energy'.")
        true_params = torch.stack(true_params).to(device=self.device).squeeze()
        if kwargs.get('hard_selection', False):
            # check if the resolution contour around the true parameter intersects the threshold range
            if self.resolution_type =='angular': # take cosine of zenith angle for angular resolution
                cos_true_params = torch.cos(true_params)
                lower_bound = cos_true_params - torch.sin(true_params) * resolution_per_event
                upper_bound = cos_true_params + torch.sin(true_params) * resolution_per_event
                selection_mask = (lower_bound <= selection_thresholds[1]) & (upper_bound >= selection_thresholds[0])
            elif self.resolution_type =='energy':
                lower_bound = true_params - resolution_per_event
                upper_bound = true_params + resolution_per_event
                selection_mask = (lower_bound <= selection_thresholds[1]) & (upper_bound >= selection_thresholds[0])
        else: # use soft selection based on resolution
            if self.resolution_type =='angular':
                cos_true_params = torch.cos(true_params)
                lower_bound = cos_true_params - torch.sin(true_params) * resolution_per_event
                upper_bound = cos_true_params + torch.sin(true_params) * resolution_per_event
                selection_mask = self.soft_between(selection_thresholds, lower_bound, upper_bound, temperature=self.selection_soft_temperature)
            elif self.resolution_type =='energy':
                lower_bound = true_params - resolution_per_event
                upper_bound = true_params + resolution_per_event
                selection_mask = self.soft_between(selection_thresholds, lower_bound, upper_bound, temperature=self.selection_soft_temperature)



        selection_efficiency = torch.mean(selection_mask.float())
        selection_loss = 1.0 - selection_efficiency

        return {
            'selection_loss': selection_loss,
            'selection_efficiency': selection_efficiency,
            'resolution_per_event': resolution_per_event,
            'resolution_params': signal_event_params,
            'selection_per_event': selection_mask
        }          
        

    


