import torch
import math
import random
from torch.utils.checkpoint import checkpoint as _checkpoint
from nugget.losses.base_loss import LossFunction
from nugget.samplers.sim_weights import (event_arrays, generation_params, get_simulation_weights,
                                         ranged_generation_density, M2_TO_CM2)
from nugget.samplers.flux_weights import get_flux_weights, model_parameter_values
from nugget.losses.effective_area import CrossSection, TransmissionProb
from nugget.losses.fisher_info import WeightedResolutionLoss
from nugget.losses.fisher_info_om_net import covariance_from_fisher, resolution_from_covariance
from nugget.losses.trigger import TriggerLoss
from nugget.losses.photon_trigger import PhotonCountTrigger

# Trigger names accepted by AnalysisLoss(trigger_loss=...) and the 'analysis_trigger' loss param.
TRIGGERS = {'sliding_bar': TriggerLoss, 'photon_count': PhotonCountTrigger}

PRECOMPUTED_PER_EVENT_KEYS = ('precomputed_fisher_info_per_string_per_event', 'precomputed_light_yield_per_point_per_event')


def soft_bin_indicators(x, edges, sigma):
    """(N, n_bins) smooth bin membership of x with width sigma; the hard indicator as sigma -> 0."""
    # Sum of two tanh steps rather than the poster's 0.5 * (tanh * tanh + 1): that form tends to 0.5 in
    # every bin once sigma >> bin width, whereas this one telescopes so each event's total over bins is <= 1.
    lo = (x.unsqueeze(1) - edges[:-1]) / sigma.unsqueeze(1)
    hi = (edges[1:] - x.unsqueeze(1)) / sigma.unsqueeze(1)
    return 0.5 * (torch.tanh(lo) + torch.tanh(hi))


def soft_histogram(weights, xs, edges, sigmas):
    """(P, n_1 * ... * n_D) smeared histograms of P weight rows (P, N) over D axes, flattened row-major.

    Contracts the weights with one axis' (N, n_d) indicators at a time instead of forming the joint
    (N, n_1 * ... * n_D) indicator matrix.
    """
    letters = 'abcdefgh'[:len(xs)]
    inds = [soft_bin_indicators(x, e, s) for x, e, s in zip(xs, edges, sigmas)]
    spec = 'pi,' + ','.join(f'i{c}' for c in letters) + '->p' + letters
    return torch.einsum(spec, weights, *inds).reshape(weights.shape[0], -1)


def chunked_soft_histogram(weights, xs, edges, sigmas, chunk_size=None):
    """soft_histogram summed over chunks of events; each chunk is checkpointed, so its indicators are not kept for backward."""
    n = weights.shape[1]
    if chunk_size is None or chunk_size >= n:
        return soft_histogram(weights, xs, edges, sigmas)
    total = 0.0
    for s in range(0, n, chunk_size):
        sl = slice(s, s + chunk_size)
        part = [weights[:, sl], *[x[sl] for x in xs], *[w[sl] for w in sigmas]]
        D = len(xs)
        total = total + _checkpoint(lambda w, *a: soft_histogram(w, a[:D], edges, a[D:]), *part,
                                    use_reentrant=False)
    return total


def fit_injection_radius(points, d, point_weights=None, scale=1.5, softness=20.0):
    """(n,) injection-disk radius around the geometry for each travel direction d (n, 3); the disk is centred at the origin.

    The radius is scale x a weighted soft maximum of the modules' distances from the line through the
    origin along d. It is differentiable in the module positions and weights, never exceeds the hard
    maximum, and a module whose weight goes to 0 drops out.
    """
    w = (torch.ones(points.shape[0], dtype=points.dtype, device=points.device) if point_weights is None
         else point_weights.to(points.dtype))
    along = points @ d.to(points.dtype).T                                    # (M, n)
    dist = torch.sqrt(((points * points).sum(1, keepdim=True) - along ** 2).clamp_min(1e-12))
    # Weighted log-mean-exp: between the weighted mean and the max distance, -> max as softness -> 0.
    log_w = torch.log(w.clamp_min(1e-30)).unsqueeze(1)
    soft_max = softness * (torch.logsumexp(dist / softness + log_w, dim=0) - torch.log(w.sum()))
    return scale * soft_max


def poisson_fisher_matrix(mu, grad_mu):
    """F_jk = sum_b dmu_b/dj * dmu_b/dk / mu_b, with grad_mu of shape (n_params, n_bins)."""
    g = grad_mu / torch.sqrt(mu.clamp_min(1e-12))
    return g @ g.T


class AnalysisLoss(LossFunction):
    def __init__(self, trigger_loss=None, resolution_loss=None, xsec=None, transmission=None, device=None,
                 energy_resolution_loss=None, angular_resolution_loss=None, fisher_loss=None):
        super().__init__(device=device)
        # A loss with fisher_per_event (e.g. OMFisherNetLoss) over energy, zenith and azimuth: every
        # resolution then comes from one per-event Fisher matrix, and the resolution losses are unused.
        self.fisher_loss = fisher_loss
        if fisher_loss is not None:
            missing = {'energy', 'zenith', 'azimuth'} - set(fisher_loss.fisher_params)
            if missing:
                raise ValueError(f"fisher_loss needs fisher_params with energy, zenith and azimuth; missing {sorted(missing)}")
        # Any trigger: a name in TRIGGERS, a TriggerLoss-like object, or a callable(geom_dict, **kwargs)
        # returning per-event efficiencies (N,) or a dict with 't_per_event'. None -> 'sliding_bar'.
        self.trigger_loss = self._make_trigger(trigger_loss)
        # Keyed by resolution_type; missing ones default to WeightedResolutionLoss when first needed.
        # resolution_loss (the older single-loss argument) fills the slot its resolution_type names.
        self.resolution_losses = {}
        for loss in (resolution_loss, energy_resolution_loss, angular_resolution_loss):
            if loss is not None:
                self.resolution_losses[loss.resolution_type] = loss
        # CrossSection() takes a while to build, so it is made once and kept.
        self.xsec = xsec
        self.transmission = transmission

    def _make_trigger(self, trigger):
        if isinstance(trigger, str):
            if trigger not in TRIGGERS:
                raise ValueError(f"Unknown trigger {trigger!r}; use one of {sorted(TRIGGERS)} or pass a trigger object")
            return TRIGGERS[trigger](device=self.device)
        return trigger

    def _detector_efficiency(self, trigger, geom_dict, eff_kwargs):
        """(N,) per-event detection efficiency from any trigger."""
        out = trigger(geom_dict, **eff_kwargs)
        if isinstance(out, dict):
            out = out['t_per_event']
        return out.reshape(-1)

    def _fill_from_fisher(self, geom_dict, eff_kwargs, cache):
        """All resolutions from one per-event covariance: 'energy' (sigma_E, GeV), 'zenith' (sigma_theta, rad),
        'angular' (rad) and the covariance itself."""
        F, ev = self.fisher_loss.fisher_per_event(geom_dict, **eff_kwargs)
        names = list(self.fisher_loss.fisher_params)
        cov = covariance_from_fisher(F, names, ev['energy'])
        ie, iz = names.index('energy'), names.index('zenith')
        cache['covariance'] = cov
        cache['energy'] = torch.sqrt(cov[:, ie, ie].clamp_min(0.0))
        cache['zenith'] = torch.sqrt(cov[:, iz, iz].clamp_min(0.0))
        cache['angular'] = resolution_from_covariance(cov, names, ev['energy'], ev['zenith'], 'angular')

    def _resolution_per_event(self, kind, geom_dict, eff_kwargs, cache):
        """(N,) per-event 'energy' (sigma_E, GeV) or 'angular' (rad) resolution, computed at most once per call.

        With a fisher_loss, also 'zenith' (sigma_theta, rad), and all come from a single Fisher evaluation.
        """
        if kind not in cache:
            if self.fisher_loss is not None:
                self._fill_from_fisher(geom_dict, eff_kwargs, cache)
                return cache[kind]
            if kind not in self.resolution_losses:
                self.resolution_losses[kind] = WeightedResolutionLoss(
                    device=self.device, resolution_type=kind, fisher_info_params=['energy', 'zenith', 'azimuth'])
            out = self.resolution_losses[kind](geom_dict, **eff_kwargs)
            cache[kind] = out[f'{kind}_resolution_per_event'].reshape(-1)
        return cache[kind]

    def __call__(self, geom_dict, **kwargs):
        hists = self.histograms(geom_dict, **kwargs)
        hists.pop('analysis_replay')
        head = self.fisher_head(hists['analysis_mu'], hists['analysis_grad_mu'], hists['analysis_param_names'], **kwargs)
        return {**hists, **head}

    def histograms(self, geom_dict, **kwargs):
        """Additive part of the loss: expected counts mu (n_bins,) and dmu/dphi (n_params, n_bins) for one batch of events."""
        binning_var_name       = kwargs.get('analysis_binning_var_name', 'energy') # :str or List[str], 'energy' / 'zenith' / ['energy', 'zenith'] (2D)
        num_events             = kwargs.get('num_events', 1) # :int
        signal_event_params    = kwargs.get('signal_event_params', None) # :List[Dict]
        num_bins               = kwargs.get('analysis_num_bins', 50) # :int, or List[int] per binning variable
        signal_sampler         = kwargs.get('signal_sampler', None) # :CylinderSampler
        sim_weight_kwargs      = kwargs.get('analysis_sim_weight_kwargs', {}) # :Dict, overrides for get_simulation_weights (e.g. geometry-dependent cyl_radius/cyl_height)
        sim_weights            = kwargs.get('analysis_sim_weights', None) # :Tensor (N,), for signal_event_params
        flux_weights           = kwargs.get('analysis_flux_weights', None) # :Tensor (N,)
        grad_weights           = kwargs.get('analysis_grad_flux_weights', None) # :Dict[str, Tensor (N,)]
        live_time              = kwargs.get('live_time', 1.0) # :float, seconds
        perfect_efficiency     = kwargs.get('analysis_perfect_efficiency', False) # :bool, every event detected; skips the trigger
        trigger_override       = kwargs.get('analysis_trigger', None) # :str|object|callable, replaces the constructor's trigger for this call
        energy_res_selection   = kwargs.get('analysis_energy_res_selection', False) # :bool, soft cut on sigma_E / E
        energy_res_threshold   = kwargs.get('analysis_energy_res_threshold', 1.0) # :float, sigma_E / E where the selection weight is 0.5
        energy_res_softness    = kwargs.get('analysis_energy_res_softness', 0.1) # :float, width of the cut in ln(sigma_E / E)
        param_overrides        = kwargs.get('analysis_param_overrides', None) # :Dict
        dataset_name           = kwargs.get('analysis_dataset_name', None) # :str
        batched_sim_weights    = kwargs.get('analysis_batched_sim_weights', True) # :bool, sim weights are precomputed for the whole batch and must be sliced to the subsample
        bin_chunk_size         = kwargs.get('analysis_bin_chunk_size', None) # :int, events per checkpointed chunk of the soft histogram (None = one pass)
        resample_vertices      = kwargs.get('analysis_resample_vertices', False) # :bool, RangeSampler only: redraw each vertex on a disk fitted around the geometry for its direction
        injection_scale        = kwargs.get('analysis_injection_scale', 1.5) # :float, disk radius / soft max distance of the modules from the track line
        injection_softness     = kwargs.get('analysis_injection_softness', 20.0) # :float, m, softness of that maximum
        vertex_uniforms        = kwargs.get('analysis_vertex_uniforms', None) # :Tensor (N, 3), the vertex draws; set by the replay

        friend_config       = kwargs.get('friend_config', None) # :str|dict
        pyff_config         = kwargs.get('pyff_config', None) # :str|dict
        weight_pid          = kwargs.get('weight_pid', 14) # :int
        eff_kwargs = kwargs.copy()

        # prepare events and get weights
        if signal_event_params is None:
            signal_event_params = signal_sampler.sample_events(num_events)
            # Anything precomputed belongs to some other set of events.
            sim_weights = flux_weights = grad_weights = None
            for key in PRECOMPUTED_PER_EVENT_KEYS:
                eff_kwargs[key] = None
        else:
            n_total = len(signal_event_params)
            idx = sorted(random.sample(range(n_total), num_events)) if n_total > num_events else list(range(n_total))
            signal_event_params = [signal_event_params[i] for i in idx]
            if sim_weights is not None:
                # Precomputed with N_gen = n_total; the subsample has to stand in for all of them.
                sim_weights = sim_weights[idx] * n_total / len(idx)
            if flux_weights is not None and grad_weights is not None:
                flux_weights = flux_weights[idx]
                grad_weights = {k: g[idx] for k, g in grad_weights.items()}
            for key in PRECOMPUTED_PER_EVENT_KEYS:
                if eff_kwargs.get(key) is not None:
                    eff_kwargs[key] = eff_kwargs[key][idx]
        eff_kwargs['signal_event_params'] = signal_event_params
        replay_events = signal_event_params

        # Keep energy and direction, redraw the vertex on a disk fitted around the geometry for each direction.
        # Only the disk area carries the geometry gradient (through the sim weights); the vertices are graph-free.
        injection_area = None
        if resample_vertices:
            if not hasattr(signal_sampler, 'place_vertices'):
                raise ValueError("analysis_resample_vertices needs a RangeSampler as signal_sampler")
            points = geom_dict['points_3d']
            point_w = None
            if geom_dict.get('string_weights') is not None and geom_dict.get('string_xy') is not None:
                point_w = PhotonCountTrigger.point_weights(points, geom_dict['string_xy'], geom_dict['string_weights'])
            d = torch.stack([torch.as_tensor(e['direction']).reshape(3) for e in signal_event_params]).to(
                device=points.device, dtype=points.dtype)
            radius = fit_injection_radius(points, d, point_w, injection_scale, injection_softness)
            injection_area = (math.pi * radius ** 2).to(self.device)            # m^2
            n_ev, dev = len(signal_event_params), signal_sampler.device
            if vertex_uniforms is None:
                vertex_uniforms = torch.rand(n_ev, 3, dtype=torch.float64, device=dev)
            energies = torch.stack([torch.as_tensor(e['energy']).reshape(-1)[0] for e in signal_event_params])
            origin = torch.zeros(n_ev, 3, dtype=torch.float64, device=dev)
            vertex, x_total = signal_sampler.place_vertices(
                energies.to(device=dev, dtype=torch.float64), d.to(dev), origin, radius.to(dev), vertex_uniforms)
            signal_event_params = [{**e, 'position': vertex[i:i + 1], 'column_depth': x_total[i:i + 1]}
                                   for i, e in enumerate(signal_event_params)]
            eff_kwargs['signal_event_params'] = signal_event_params
            for key in PRECOMPUTED_PER_EVENT_KEYS:   # computed for the old vertices
                eff_kwargs[key] = None
            sim_weights = None                       # recomputed below for the new vertices and disks
        # Only precomputed sim weights are constants that may be replayed; computed ones can depend on the geometry.
        precomputed_sim_weights = sim_weights
        ev = {k: v.to(self.device) for k, v in event_arrays(signal_event_params, travel_direction=True).items()}

        gen = {}
        if signal_sampler is not None:
            gen.update(generation_params(signal_sampler))
            if hasattr(signal_sampler, 'cylinder'):
                gen.update(cyl_radius=signal_sampler.cylinder.radius, cyl_height=signal_sampler.cylinder.height)
        gen.update(sim_weight_kwargs)
        if gen.get('mode') == 'ranged' and 'column_depth' not in gen:
            gen['column_depth'] = ev.get('column_depth')
        if injection_area is not None:
            # Per-event disk instead of the sampler's fixed pi R_inj^2; N_gen is this batch.
            gen.update(column_depth=ev['column_depth'], include_injection_area=False)
        if sim_weights is None:
            if self.xsec is None:
                self.xsec = CrossSection()
            if self.transmission is None:
                self.transmission = TransmissionProb()
            sim_weights = get_simulation_weights(ev['energy'], ev['cos_zenith'], xsec=self.xsec,
                                                 transmission=self.transmission, batched=batched_sim_weights, **gen)
            if injection_area is not None:
                sim_weights = sim_weights * injection_area
        if flux_weights is None or grad_weights is None:
            flux_weights, grad_weights = get_flux_weights(ev['energy'], ev['zenith'], ev['azimuth'],
                                                          friend_config, pyff_config, pid=weight_pid,
                                                          dataset=dataset_name, params=param_overrides, device=self.device)

        if perfect_efficiency:
            detector_effs = torch.ones_like(ev['energy'])
        else:
            if trigger_override is not None:
                trigger = self._make_trigger(trigger_override)
            else:
                if self.trigger_loss is None:
                    self.trigger_loss = TriggerLoss(device=self.device)
                trigger = self.trigger_loss
            detector_effs = self._detector_efficiency(trigger, geom_dict, eff_kwargs)

        # One histogram axis per binning variable: log10(E) smeared by the energy resolution and/or
        # cos(zenith) smeared by the angular resolution, each in the units of its axis.
        binning_vars = [binning_var_name] if isinstance(binning_var_name, str) else list(binning_var_name)
        bins_per_var = [num_bins] * len(binning_vars) if isinstance(num_bins, int) else list(num_bins)
        res_cache = {}
        bin_edges, true_vals, widths = {}, [], []
        for var, n in zip(binning_vars, bins_per_var):
            if var == 'energy':
                E_min, E_max = gen.get('E_min', 1e2), gen.get('E_max', 1e8)
                bin_edges[var] = torch.linspace(math.log10(E_min), math.log10(E_max), n + 1, device=self.device)
                true_vals.append(torch.log10(ev['energy']))
                sigma_E = self._resolution_per_event('energy', geom_dict, eff_kwargs, res_cache)
                widths.append(sigma_E / (ev['energy'] * math.log(10)))
            elif var == 'zenith':
                # cos_range is in the sampler's travel convention, the events are in the arrival one.
                c_lo, c_hi = gen.get('cos_range', (-1.0, 1.0))
                bin_edges[var] = torch.linspace(-c_hi, -c_lo, n + 1, device=self.device)
                true_vals.append(ev['cos_zenith'])
                if self.fisher_loss is not None:
                    # sigma(cos theta) = |sin theta| sigma_theta, from the zenith entry of the covariance.
                    sigma_th = self._resolution_per_event('zenith', geom_dict, eff_kwargs, res_cache)
                    widths.append(sigma_th * torch.sin(ev['zenith']).abs())
                else:
                    # Without the covariance, the full angular resolution stands in for sigma_theta.
                    sigma_psi = self._resolution_per_event('angular', geom_dict, eff_kwargs, res_cache)
                    widths.append(sigma_psi * torch.sin(ev['zenith']))
            else:
                raise ValueError(f"Unknown binning variable {var!r}; use 'energy' and/or 'zenith'")
        # A singular per-event Fisher gives NaN/inf; such events are smeared out of every bin.
        widths = [torch.nan_to_num(w, nan=1e3, posinf=1e3).clamp_min(1e-6) for w in widths]

        selection = torch.ones_like(detector_effs)
        if energy_res_selection:
            rel_res = self._resolution_per_event('energy', geom_dict, eff_kwargs, res_cache) / ev['energy']
            # A singular per-event Fisher (NaN/inf) counts as unresolved and is cut.
            rel_res = torch.nan_to_num(rel_res, nan=1e6, posinf=1e6).clamp_min(1e-12)
            selection = torch.sigmoid((math.log(energy_res_threshold) - torch.log(rel_res)) / energy_res_softness)

        # w_i and dw_i/dphi_j; the flux gradients are constants, geometry enters via sim weights, efficiency, selection and resolution
        acceptance = live_time * sim_weights * detector_effs * selection
        total_weights = acceptance * flux_weights
        param_names = list(grad_weights)
        grad_total_weights = torch.stack([grad_weights[k] for k in param_names]) * acceptance  # (n_params, N)

        # Joint bin membership is the product over axes, flattened row-major (first variable slowest).
        # mu and dmu/dphi are histograms of different weights, so they share one contraction.
        all_weights = torch.cat([total_weights.unsqueeze(0), grad_total_weights])        # (1 + n_params, N)
        hists = chunked_soft_histogram(all_weights, true_vals, [bin_edges[v] for v in binning_vars],
                                       widths, chunk_size=bin_chunk_size)
        mu, grad_mu = hists[0], hists[1:]                                  # (n_bins,), (n_params, n_bins)

        # Everything needed to recompute exactly this batch again (used by AnalysisAccumulator).
        replay = {
            'signal_event_params': replay_events,
            'num_events': len(replay_events),
            'analysis_flux_weights': flux_weights.detach(),
            'analysis_grad_flux_weights': {k: g.detach() for k, g in grad_weights.items()},
            **{key: eff_kwargs.get(key) for key in PRECOMPUTED_PER_EVENT_KEYS},
        }
        if precomputed_sim_weights is not None:
            replay['analysis_sim_weights'] = precomputed_sim_weights
        if injection_area is not None:
            replay['analysis_vertex_uniforms'] = vertex_uniforms   # the same vertices again

        out = {
            'analysis_mu': mu,
            'analysis_grad_mu': grad_mu,
            'analysis_param_names': param_names,
            'analysis_bin_edges': bin_edges,  # {var: edges}
            'analysis_hist_shape': tuple(len(bin_edges[var]) - 1 for var in binning_vars),  # mu is flattened from this
            'detector_efficiency_per_event': detector_effs,
            'selection_efficiency_per_event': selection,
            'analysis_replay': replay,
        }
        if 'energy' in binning_vars:
            out['energy_resolution_per_event'] = widths[binning_vars.index('energy')]
            out['energy_resolution_units'] = 'log10'  # sigma of log10(E); read by the visualizer's energy panels
        if 'angular' in res_cache:
            out['angular_resolution_per_event'] = res_cache['angular']  # radians, as the visualizer's angular panels expect
        if injection_area is not None:
            # A_eff = T * P_int * A_inj * efficiency * selection, in m^2: the weight times the generation density.
            area = sim_weights * ranged_generation_density(ev['energy'], **gen) / M2_TO_CM2
            out['effective_area_per_event'] = area * detector_effs #* selection
            out['injection_area_per_event'] = injection_area
            out['effective_area_params'] = [{'energy': e['energy'], 'zenith': e['zenith'], 'azimuth': e['azimuth'],
                                             'position': e['position']} for e in signal_event_params]
        return out

    def fisher_head(self, mu, grad_mu, param_names, **kwargs):
        """Nonlinear part of the loss: flux-parameter Fisher matrix from the (pooled) histograms, and the optimality criterion."""
        signal_flux_var_names = kwargs.get('analysis_signal_flux_var_names', ['astro_norm']) # :List[str]
        optimality            = kwargs.get('analysis_optimality', 'a') # :str, 'a' or 'd'

        fisher = poisson_fisher_matrix(mu, grad_mu)
        # The signal block of the full inverse is the covariance marginalised over the nuisance parameters.
        cov = torch.linalg.inv(fisher)
        signal_idx = [param_names.index(name) for name in signal_flux_var_names]
        signal_cov = cov[signal_idx][:, signal_idx]
        if optimality == 'a':
            loss = torch.sqrt(torch.diagonal(signal_cov)).sum()
        elif optimality == 'd':
            loss = torch.logdet(signal_cov)
        else:
            raise NotImplementedError(f"No {optimality} optimality")

        out = {
            'fisher_analysis_loss': loss,
            'flux_param_names': signal_flux_var_names,
            'flux_param_variances': torch.diagonal(signal_cov),
            'flux_param_covariance': cov,
            'analysis_fisher_matrix': fisher,
            'analysis_param_names': param_names,
        }
        if kwargs.get('pyff_config') is not None:
            # The assumed true values the Fisher matrix is evaluated at (centre of a likelihood scan).
            values = model_parameter_values(kwargs['pyff_config'], kwargs.get('analysis_param_overrides'))
            out['analysis_param_values'] = {name: float(values[name]) for name in param_names if name in values}
        return out


def fisher_constraints(fisher, param_names, names=None):
    """{name: (sigma_conditional, sigma_profiled)} from a Fisher matrix over param_names.

    conditional: the other parameters fixed at their true values, 1 / sqrt(F_ii).
    profiled: the others fitted as well, sqrt((F^-1)_ii) = 1 / sqrt(F_ii - F_in F_nn^-1 F_ni), the
    Schur complement of the other parameters' block. Always >= the conditional one.
    """
    F = torch.as_tensor(fisher, dtype=torch.float64)
    cov = torch.linalg.pinv(F)
    out = {}
    for name in (names or param_names):
        i = param_names.index(name)
        out[name] = (float(1.0 / torch.sqrt(F[i, i])), float(torch.sqrt(cov[i, i])))
    return out
        

        


        

        
        
            