import math
import torch
from nugget.losses.base_loss import LossFunction
from nugget.losses.trigger import _resolve_batched_surrogate


class PhotonCountTrigger(LossFunction):
    """Soft trigger on the total number of detected photons per event.

    N_tot = sum over points of (string activation * expected photons), then a soft cut at n_photons:
      'sigmoid': sigmoid(ln(N_tot / n_photons) / softness), softness in units of ln N
      'poisson': P(N >= n_photons | mean N_tot), the probability of actually seeing that many photons
    Same call contract and return keys as TriggerLoss, so it can stand in for it.

    With NuSmoothie, photons='expected' uses pi * E[q | hit] and photons='sample' one hit x flow draw per
    PMT for this call; both are differentiable in the module positions whenever points_3d requires grad.
    """

    def __init__(self, device=None, n_photons=10.0, softness=0.2, cut='sigmoid', photons='expected',
                 print_loss=False):
        super().__init__(device=device)
        if cut not in ('sigmoid', 'poisson'):
            raise ValueError(f"cut must be 'sigmoid' or 'poisson', got {cut!r}")
        if photons not in ('expected', 'sample'):
            raise ValueError(f"photons must be 'expected' or 'sample', got {photons!r}")
        self.n_photons = float(n_photons)
        self.softness = float(softness)
        self.cut = cut
        self.photons = photons
        self.print_loss = print_loss

    @staticmethod
    def point_weights(points_3d, string_xy, string_weights):
        """(n_points,) sigmoid(string_weights) of each point's nearest string."""
        nearest = torch.cdist(points_3d[:, :2], string_xy).argmin(dim=1)
        return torch.sigmoid(string_weights)[nearest]

    def efficiency(self, n_tot):
        """Soft detection probability for total expected photon counts n_tot."""
        if self.cut == 'poisson':
            # Regularised lower incomplete gamma: P(N >= k | lambda) = P(k, lambda) for integer k.
            k = torch.full_like(n_tot, math.ceil(self.n_photons))
            return torch.special.gammainc(k, n_tot.clamp_min(0.0))
        return torch.sigmoid(torch.log(n_tot.clamp_min(1e-30) / self.n_photons) / self.softness)

    def _light_yield(self, points_3d, events, surrogate_func, batched_func, seed=None):
        if batched_func is not None:
            owner = getattr(batched_func, '__self__', None)
            if hasattr(owner, 'hit_model') and hasattr(owner, 'ly_model'):  # NuSmoothie
                grad = torch.is_grad_enabled() and points_3d.requires_grad
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=owner.device).manual_seed(int(seed))
                # fast: the same cheap, pruned estimator whether or not a graph is kept, so the
                # accumulator's graph-free pass and its gradient pass compute the same counts.
                return batched_func(om_positions=points_3d, event_params_list=events, grad=grad,
                                    fast=True, sample=self.photons == 'sample', generator=generator)
            return batched_func(om_positions=points_3d, event_params_list=events)
        return torch.stack([surrogate_func(opt_point=points_3d, event_params=ep).reshape(-1) for ep in events])

    def __call__(self, geom_dict, **kwargs):
        points_3d = geom_dict['points_3d']
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        events = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        precomputed = kwargs.get('precomputed_light_yield_per_point_per_event', None)
        batched_func = kwargs.get('batched_surrogate_func', None)
        if batched_func is None and kwargs.get('use_batched_surrogate', True):
            batched_func = _resolve_batched_surrogate(surrogate_func)
        chunk = kwargs.get('binned_trigger_batch_size', None) or len(events)
        detach = kwargs.get('detach_light_yields', False)
        # Set per batch by AnalysisAccumulator and replayed in its gradient pass: same draws in both passes.
        seed = kwargs.get('accumulator_seed', None)
        if precomputed is None and surrogate_func is None and batched_func is None:
            raise ValueError("PhotonCountTrigger needs signal_surrogate_func or precomputed_light_yield_per_point_per_event")

        weights = None
        if string_weights is not None and string_xy is not None:
            weights = self.point_weights(points_3d, string_xy, string_weights)

        n_tot = []
        for start in range(0, len(events), chunk):
            if precomputed is not None:
                ly = precomputed[start:start + chunk]
            else:
                ly = self._light_yield(points_3d, events[start:start + chunk], surrogate_func, batched_func,
                                       seed=None if seed is None else seed + start)
            if detach:
                ly = ly.detach()
            if weights is not None:
                # Surrogates with one output per PMT return K values per point, point-major.
                w = weights.repeat_interleave(ly.shape[1] // weights.shape[0]) if ly.shape[1] != weights.shape[0] else weights
                ly = ly * w.to(ly.dtype)
            n_tot.append(ly.sum(dim=1))
        n_tot = torch.cat(n_tot)

        t_values = self.efficiency(n_tot)
        detector_efficiency = t_values.mean()
        if self.print_loss:
            print(f"PhotonCountTrigger: efficiency {detector_efficiency.item():.4f}")
        return {
            'trigger_loss': 1.0 - detector_efficiency,
            'detector_efficiency': detector_efficiency,
            't_per_event': t_values,
            'total_photons_per_event': n_tot,
        }
