import torch

from nugget.losses.base_loss import LossFunction
from nugget.losses.effective_area import EffectiveAreaLoss, get_weighted_min_enclosing_circle
from nugget.losses.fisher_info import ResolutionLoss, WeightedResolutionLoss
from nugget.samplers.cyl_sampler import CylinderSampler


class FoMLoss(LossFunction):
    """Combine per-event effective area with angular resolution.

    Final loss follows the same structure as angular-resolution loss:
        L = 1 / sqrt(sum_i(term_i))
    with
        term_i = A_eff_i / (4*pi*res_i^2)

    where A_eff_i is per-event effective area and res_i is per-event angular resolution.
    """

    def __init__(
        self,
        device=None,
        use_weighted_resolution=True,
        fisher_info_params=None,
        resolution_loss=None,
        effective_area_loss=None,
        resolution_loss_kwargs=None,
        effective_area_loss_kwargs=None,
    ):
        super().__init__(device)

        if fisher_info_params is None:
            fisher_info_params = ["energy", "azimuth", "zenith"]

        self.use_weighted_resolution = use_weighted_resolution

        if resolution_loss is None:
            resolution_loss_kwargs = {} if resolution_loss_kwargs is None else dict(resolution_loss_kwargs)
            if self.use_weighted_resolution:
                self.resolution_loss = WeightedResolutionLoss(
                    device=self.device,
                    fisher_info_params=fisher_info_params,
                    resolution_type="angular",
                    **resolution_loss_kwargs,
                )
            else:
                self.resolution_loss = ResolutionLoss(
                    device=self.device,
                    fisher_info_params=fisher_info_params,
                    resolution_type="angular",
                    **resolution_loss_kwargs,
                )
        else:
            self.resolution_loss = resolution_loss

        if effective_area_loss is None:
            effective_area_loss_kwargs = {} if effective_area_loss_kwargs is None else dict(effective_area_loss_kwargs)
            self.effective_area_loss = EffectiveAreaLoss(
                device=self.device,
                **effective_area_loss_kwargs,
            )
        else:
            self.effective_area_loss = effective_area_loss

    def _get_events(self, kwargs):
        event_params = kwargs.get("signal_event_params", None)
        signal_sampler = kwargs.get("signal_sampler", None)
        num_events = kwargs.get("num_events", 100)

        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)

        if event_params is None:
            raise ValueError(
                "signal_event_params must be provided (or signal_sampler must be provided) "
                "for EffectiveAreaResolutionLoss"
            )

        return event_params

    def _get_geometry_bounding_cylinder(self, geom_dict, temperature, include_height=True, **circle_kwargs):
        """Fit a cylinder to the current geometry: XY center/radius come from the
        smooth weighted minimum enclosing circle of the string positions
        (weighted continuously in [0, 1] by string_weights); height (if
        requested) is the unweighted z-extent of the detector's points_3d.

        Extra keyword arguments are forwarded to get_weighted_min_enclosing_circle
        (e.g. downweight_untriggerable and its trigger_* parameters).
        """
        string_xy = geom_dict.get("string_xy", None)
        if string_xy is None:
            raise ValueError("geom_dict must provide 'string_xy' to adjust the cylinder to geometry")
        string_weights = geom_dict.get("string_weights", None)
        string_probs = None
        if string_weights is not None:
            string_probs = torch.sigmoid(string_weights)

        center_xy, radius = get_weighted_min_enclosing_circle(
            string_xy, string_weights=string_probs, temperature=temperature, **circle_kwargs
        )

        if not include_height:
            center_z = torch.zeros((), device=self.device, dtype=center_xy.dtype)
            height = torch.zeros((), device=self.device, dtype=center_xy.dtype)
        else:
            points_3d = geom_dict.get("points_3d")
            z_positions = points_3d[:, 2]
            z_max = torch.max(z_positions)
            z_min = torch.min(z_positions)
            center_z = 0.5 * (z_min + z_max)
            height = z_max - z_min

        center = torch.stack([center_xy[0], center_xy[1], center_z])
        return center, radius, height

    # Constructor args of CylinderSampler that are captured explicitly (not in
    # self.kwargs), so a from-scratch cylinder can be safely merged into a
    # clone's **kwargs without colliding with these.
    _CYLINDER_SAMPLER_RESERVED_KEYS = ("device", "dim", "domain_size", "cylinder_center", "cylinder_height", "cylinder_radius")

    def _get_geometry_adjusted_sampler(self, geom_dict, kwargs):
        """Clone the configured signal_sampler onto the cylinder derived from the current geometry."""
        signal_sampler = kwargs.get("signal_sampler", None)
        if signal_sampler is None:
            raise ValueError("signal_sampler must be provided when adjust_cylinder_to_geometry=True")
        if not isinstance(signal_sampler, CylinderSampler):
            raise TypeError(
                "adjust_cylinder_to_geometry=True requires signal_sampler to be a CylinderSampler, "
                f"got {type(signal_sampler)}"
            )

        temperature = kwargs.get("bounding_cylinder_temperature", 1)
        include_height = kwargs.get("fom_adjust_cylinder_height", True)
        # Forward the triggerability-gating options so the FoM sampling cylinder
        # matches the radius EffectiveAreaLoss derives (keeps the two consistent).
        circle_kwargs = {
            "downweight_untriggerable": kwargs.get("downweight_untriggerable", False),
            "trigger_neighbor_distance": kwargs.get("trigger_neighbor_distance", 550.0),
            "trigger_min_neighbors": kwargs.get("trigger_min_neighbors", 30.0),
            "trigger_distance_sharpness": kwargs.get("trigger_distance_sharpness", 0.05),
            "trigger_count_sharpness": kwargs.get("trigger_count_sharpness", 1.0),
        }
        center, radius, height = self._get_geometry_bounding_cylinder(
            geom_dict, temperature, include_height=include_height, **circle_kwargs
        )

        sampler_kwargs = {
            k: v for k, v in signal_sampler.kwargs.items()
            if k not in self._CYLINDER_SAMPLER_RESERVED_KEYS
        }

        return CylinderSampler(
            device=signal_sampler.device,
            dim=signal_sampler.dim,
            domain_size=signal_sampler.domain_size,
            cylinder_center=center.detach(),
            cylinder_height=height.detach(),
            cylinder_radius=radius.detach(),
            **sampler_kwargs,
        )

    def __call__(self, geom_dict, **kwargs):
        # use_irregular_cylinder = kwargs.get("use_irregular_cylinder", False)
        # use_batched_effective_area = kwargs.get("use_batched_effective_area", False)

        adjust_cylinder_to_geometry = kwargs.get("fom_adjust_cylinder_to_geometry", False)

        # Let resolution choose/load/subsample events first (e.g. weighted Fisher path).
        resolution_kwargs = dict(kwargs)
        normalize_fom_by_energy = kwargs.get("normalize_fom_by_energy", True)
        # resolution_kwargs.pop("use_irregular_cylinder", None)
        # resolution_kwargs.pop("use_batched_effective_area", None)

        if adjust_cylinder_to_geometry:
            # Recompute the sampling cylinder from the current geometry's bounding
            # cylinder (same as EffectiveAreaLoss would derive it) and resample fresh
            # events from it, rather than reusing whatever events/cylinder the caller
            # passed in.
            geometry_adjusted_sampler = self._get_geometry_adjusted_sampler(geom_dict, kwargs)
            resolution_kwargs["signal_sampler"] = geometry_adjusted_sampler
            resolution_kwargs["signal_event_params"] = None

        # Angular resolution (weighted or non-weighted)
        resolution_out = self.resolution_loss(geom_dict, **resolution_kwargs)
        resolution_per_event = resolution_out.get("resolution_per_event", None)
        if resolution_per_event is None:
            raise ValueError("Resolution loss did not return 'resolution_per_event'")
        resolution_per_event = torch.as_tensor(resolution_per_event, device=self.device)

        # Use the exact event list that resolution used after any internal subsampling.
        shared_events = resolution_out.get("resolution_params", None)
        if shared_events is None:
            shared_events = self._get_events(resolution_kwargs)

        effective_area_kwargs = dict(resolution_kwargs)
        effective_area_kwargs["signal_event_params"] = shared_events
        effective_area_kwargs["per_event_effective_area_loss"] = True
        # effective_area_kwargs["use_irregular_cylinder"] = use_irregular_cylinder
        # effective_area_kwargs["use_batched_effective_area"] = use_batched_effective_area

        # If precomputed per-event light yields are for a different event count,
        # drop them so EffectiveAreaLoss recomputes consistently for shared_events.
        # precomp = effective_area_kwargs.get("precomputed_light_yield_per_point_per_event", None)
        # if precomp is not None and hasattr(precomp, "shape"):
        #     if int(precomp.shape[0]) != len(shared_events):
                # effective_area_kwargs.pop("precomputed_light_yield_per_point_per_event", None)
                # effective_area_kwargs.pop("pc_ly_per_point_per_event_per_e_per_ct", None)

        # Effective area per event
        effective_area_out = self.effective_area_loss(geom_dict, **effective_area_kwargs)
        effective_area_per_event = effective_area_out.get("effective_area_per_event", None)
        if effective_area_per_event is None:
            raise ValueError("EffectiveAreaLoss did not return 'effective_area_per_event'")
        effective_area_per_event = torch.as_tensor(effective_area_per_event, device=self.device)

        resolution_per_event = resolution_per_event.reshape(-1)
        effective_area_per_event = effective_area_per_event.reshape(-1)

        if resolution_per_event.shape[0] != effective_area_per_event.shape[0]:
            raise ValueError(
                "Mismatch between per-event tensors: "
                f"resolution_per_event has {resolution_per_event.shape[0]} entries, "
                f"effective_area_per_event has {effective_area_per_event.shape[0]} entries"
            )

        finite_mask = (
            torch.isfinite(resolution_per_event)
            & torch.isfinite(effective_area_per_event)
            & (resolution_per_event > 1e-12)
            & (effective_area_per_event >= 0.0)
        )
        if normalize_fom_by_energy:
            energies = [event['energy'] for event in shared_events]
            energies = torch.as_tensor(energies, device=self.device)
            norm_effective_area_per_event = effective_area_per_event/energies
        else:
            norm_effective_area_per_event = effective_area_per_event
        if finite_mask.any():
            # safe_res = torch.clamp_min(resolution_per_event[finite_mask], 1e-12)
            # safe_aeff = torch.clamp_min(effective_area_per_event[finite_mask], 0.0)
            safe_res = resolution_per_event[finite_mask]
            safe_aeff = norm_effective_area_per_event[finite_mask]
            sum_term = torch.sum(safe_aeff / (4.0 * torch.pi * (safe_res ** 2)))
            combined_loss = 1.0 / torch.sqrt(torch.clamp_min(sum_term, 1e-20))
        else:
            combined_loss = torch.tensor(1.0, device=self.device, requires_grad=True)

        return {
            "pointsource_fom_loss": combined_loss,
            "resolution_per_event": resolution_per_event,
            "effective_area_per_event": effective_area_per_event,
            "weighted_bounding_cylinder_center": effective_area_out.get("weighted_bounding_cylinder_center", effective_area_out.get("bounding_cylinder_center", None)),
            "weighted_bounding_cylinder_radius": effective_area_out.get("weighted_bounding_cylinder_radius", effective_area_out.get("bounding_cylinder_radius", None)),
            "weighted_bounding_cylinder_height": effective_area_out.get("weighted_bounding_cylinder_height", effective_area_out.get("bounding_cylinder_height", None)),
            "bounding_cylinder_center": effective_area_out.get("bounding_cylinder_center", effective_area_out.get("weighted_bounding_cylinder_center", None)),
            "resolution_loss": resolution_out.get("angular_resolution_loss", None),
            "effective_area_loss": effective_area_out.get("effective_area_loss", None),
            "resolution_details": resolution_out,
            "effective_area_details": effective_area_out,
            "signal_event_params": shared_events,
            "detector_efficiencies": effective_area_out.get("detector_efficiencies", None),
            "use_weighted_resolution": self.use_weighted_resolution,
        }
