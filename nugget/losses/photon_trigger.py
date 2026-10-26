import math
import torch
from nugget.losses.base_loss import LossFunction
from nugget.losses.trigger import _resolve_batched_surrogate


class PhotonCountTrigger(LossFunction):
    """Soft trigger on the total number of detected photons, or of hit PMTs, per event.

    count='photons': N = sum over points of (string activation * expected photons); threshold n_photons.
                     With NuSmoothie, pi * E[q | hit] is differentiable in the module positions through pi.
    count='hits':    N = sum over PMTs of (string activation * hit probability), the expected number of
                     hit PMTs; needs a surrogate with hit_prob_batched (NuSmoothie) and is differentiable in
                     the module positions through the hit probability. Threshold n_hits.
    Soft cut at the threshold k:
      'sigmoid': sigmoid(ln(N / k) / softness), softness in units of ln N
      'poisson': P(N' >= k) for N' Poisson with mean N
    Same call contract and return keys as TriggerLoss, so it can stand in for it.
    """

    def __init__(self, device=None, n_photons=10.0, softness=0.2, cut='sigmoid', count='photons',
                 n_hits=8.0, ly_prune_below=1e-5, print_loss=False):
        super().__init__(device=device)
        if cut not in ('sigmoid', 'poisson'):
            raise ValueError(f"cut must be 'sigmoid' or 'poisson', got {cut!r}")
        if count not in ('photons', 'hits'):
            raise ValueError(f"count must be 'photons' or 'hits', got {count!r}")
        self.n_photons = float(n_photons)
        self.n_hits = float(n_hits)
        # NuSmoothie photons: PMTs with hit probability at or below this skip the flow and count E[q | hit] = 1.
        self.ly_prune_below = float(ly_prune_below)
        self.softness = float(softness)
        self.cut = cut
        self.count = count
        self.print_loss = print_loss

    @property
    def threshold(self):
        return self.n_hits if self.count == 'hits' else self.n_photons

    @staticmethod
    def point_weights(points_3d, string_xy, string_weights):
        """(n_points,) sigmoid(string_weights) of each point's nearest string."""
        nearest = torch.cdist(points_3d[:, :2], string_xy).argmin(dim=1)
        return torch.sigmoid(string_weights)[nearest]

    def efficiency(self, n_tot):
        """Soft detection probability for expected counts n_tot."""
        if self.cut == 'poisson':
            # Regularised lower incomplete gamma: P(N >= k | lambda) = P(k, lambda) for integer k.
            k = torch.full_like(n_tot, math.ceil(self.threshold))
            return torch.special.gammainc(k, n_tot.clamp_min(0.0))
        return torch.sigmoid(torch.log(n_tot.clamp_min(1e-30) / self.threshold) / self.softness)

    def _light_yield(self, points_3d, events, surrogate_func, batched_func):
        if batched_func is not None:
            owner = getattr(batched_func, '__self__', None)
            if hasattr(owner, 'hit_prob_batched'):  # NuSmoothie: differentiable through the hit probability
                grad = torch.is_grad_enabled() and points_3d.requires_grad
                return batched_func(om_positions=points_3d, event_params_list=events, hit_grad=grad,
                                    ly_prune_below=self.ly_prune_below)
            return batched_func(om_positions=points_3d, event_params_list=events)
        return torch.stack([surrogate_func(opt_point=points_3d, event_params=ep).reshape(-1) for ep in events])

    def _hit_probs(self, points_3d, events, surrogate_func):
        owner = getattr(surrogate_func, '__self__', surrogate_func)
        if not hasattr(owner, 'hit_prob_batched'):
            raise ValueError("count='hits' needs a surrogate with hit_prob_batched (NuSmoothie)")
        grad = torch.is_grad_enabled() and points_3d.requires_grad
        return owner.hit_prob_batched(points_3d, events, grad=grad)

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
        if self.count == 'photons' and precomputed is None and surrogate_func is None and batched_func is None:
            raise ValueError("PhotonCountTrigger needs signal_surrogate_func or precomputed_light_yield_per_point_per_event")

        weights = None
        if string_weights is not None and string_xy is not None:
            weights = self.point_weights(points_3d, string_xy, string_weights)

        n_tot = []
        for start in range(0, len(events), chunk):
            if self.count == 'hits':
                vals = self._hit_probs(points_3d, events[start:start + chunk], surrogate_func)
            elif precomputed is not None:
                vals = precomputed[start:start + chunk]
            else:
                vals = self._light_yield(points_3d, events[start:start + chunk], surrogate_func, batched_func)
            if detach:
                vals = vals.detach()
            if weights is not None:
                # Surrogates with one output per PMT return K values per point, point-major.
                w = weights.repeat_interleave(vals.shape[1] // weights.shape[0]) if vals.shape[1] != weights.shape[0] else weights
                vals = vals * w.to(vals.dtype)
            n_tot.append(vals.sum(dim=1))
        n_tot = torch.cat(n_tot)

        t_values = self.efficiency(n_tot)
        detector_efficiency = t_values.mean()
        if self.print_loss:
            print(f"PhotonCountTrigger: efficiency {detector_efficiency.item():.4f}")
        return {
            'trigger_loss': 1.0 - detector_efficiency,
            'detector_efficiency': detector_efficiency,
            't_per_event': t_values,
            'total_photons_per_event' if self.count == 'photons' else 'total_hits_per_event': n_tot,
        }
