import torch
import math
import random
from nugget.losses.base_loss import LossFunction
from nugget.samplers.sim_weights import event_arrays, generation_params, get_simulation_weights
from nugget.samplers.flux_weights import get_flux_weights
from nugget.losses.effective_area import CrossSection, TransmissionProb
from nugget.losses.fisher_info import WeightedResolutionLoss
from nugget.losses.trigger import TriggerLoss

PRECOMPUTED_PER_EVENT_KEYS = ('precomputed_fisher_info_per_string_per_event', 'precomputed_light_yield_per_point_per_event')


def soft_bin_indicators(x, edges, sigma):
    """(N, n_bins) smooth bin membership of x with width sigma; the hard indicator as sigma -> 0."""
    # Sum of two tanh steps rather than the poster's 0.5 * (tanh * tanh + 1): that form tends to 0.5 in
    # every bin once sigma >> bin width, whereas this one telescopes so each event's total over bins is <= 1.
    lo = (x.unsqueeze(1) - edges[:-1]) / sigma.unsqueeze(1)
    hi = (edges[1:] - x.unsqueeze(1)) / sigma.unsqueeze(1)
    return 0.5 * (torch.tanh(lo) + torch.tanh(hi))


def poisson_fisher_matrix(mu, grad_mu):
    """F_jk = sum_b dmu_b/dj * dmu_b/dk / mu_b, with grad_mu of shape (n_params, n_bins)."""
    g = grad_mu / torch.sqrt(mu.clamp_min(1e-12))
    return g @ g.T


class AnalysisLoss(LossFunction):
    def __init__(self, trigger_loss=None, resolution_loss=None, xsec=None, transmission=None, device=None):
        super().__init__(device=device)
        self.trigger_loss = trigger_loss
        self.resolution_loss = resolution_loss
        # CrossSection() takes a while to build, so it is made once and kept.
        self.xsec = xsec
        self.transmission = transmission

    def __call__(self, geom_dict, **kwargs):
        hists = self.histograms(geom_dict, **kwargs)
        hists.pop('analysis_replay')
        head = self.fisher_head(hists['analysis_mu'], hists['analysis_grad_mu'], hists['analysis_param_names'], **kwargs)
        return {**hists, **head}

    def histograms(self, geom_dict, **kwargs):
        """Additive part of the loss: expected counts mu (n_bins,) and dmu/dphi (n_params, n_bins) for one batch of events."""
        binning_var_name       = kwargs.get('analysis_binning_var_name', 'energy') # :str, 'energy' or 'zenith'
        num_events             = kwargs.get('num_events', 1) # :int
        signal_event_params    = kwargs.get('signal_event_params', None) # :List[Dict]
        num_bins               = kwargs.get('analysis_num_bins', 50) # :int
        signal_sampler         = kwargs.get('signal_sampler', None) # :CylinderSampler
        sim_weight_kwargs      = kwargs.get('analysis_sim_weight_kwargs', {}) # :Dict, overrides for get_simulation_weights (e.g. geometry-dependent cyl_radius/cyl_height)
        sim_weights            = kwargs.get('analysis_sim_weights', None) # :Tensor (N,), for signal_event_params
        flux_weights           = kwargs.get('analysis_flux_weights', None) # :Tensor (N,)
        grad_weights           = kwargs.get('analysis_grad_flux_weights', None) # :Dict[str, Tensor (N,)]
        live_time              = kwargs.get('live_time', 1.0) # :float, seconds
        perfect_efficiency     = kwargs.get('analysis_perfect_efficiency', False) # :bool, every event detected; skips the trigger
        energy_res_selection   = kwargs.get('analysis_energy_res_selection', False) # :bool, soft cut on sigma_E / E
        energy_res_threshold   = kwargs.get('analysis_energy_res_threshold', 1.0) # :float, sigma_E / E where the selection weight is 0.5
        energy_res_softness    = kwargs.get('analysis_energy_res_softness', 0.1) # :float, width of the cut in ln(sigma_E / E)
        param_overrides        = kwargs.get('analysis_param_overrides', None) # :Dict
        dataset_name           = kwargs.get('analysis_dataset_name', None) # :str

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
        # Only precomputed sim weights are constants that may be replayed; computed ones can depend on the geometry.
        precomputed_sim_weights = sim_weights
        ev = {k: v.to(self.device) for k, v in event_arrays(signal_event_params, travel_direction=True).items()}

        gen = {}
        if signal_sampler is not None:
            gen.update(generation_params(signal_sampler),
                       cyl_radius=signal_sampler.cylinder.radius, cyl_height=signal_sampler.cylinder.height)
        gen.update(sim_weight_kwargs)
        if sim_weights is None:
            if self.xsec is None:
                self.xsec = CrossSection()
            if self.transmission is None:
                self.transmission = TransmissionProb()
            sim_weights = get_simulation_weights(ev['energy'], ev['cos_zenith'], xsec=self.xsec,
                                                 transmission=self.transmission, **gen)
        if flux_weights is None or grad_weights is None:
            flux_weights, grad_weights = get_flux_weights(ev['energy'], ev['zenith'], ev['azimuth'],
                                                          friend_config, pyff_config, pid=weight_pid,
                                                          dataset=dataset_name, params=param_overrides, device=self.device)

        # prepare the intermediate loss functions
        if self.resolution_loss is None:
            self.resolution_loss = WeightedResolutionLoss(device=self.device, 
                                     resolution_type='energy' if binning_var_name=='energy' else 'angular',
                                     fisher_info_params=['energy', 'zenith', 'azimuth'])
        if self.trigger_loss is None and not perfect_efficiency:
            self.trigger_loss = TriggerLoss(device=self.device)

        res_loss_stuff = self.resolution_loss(geom_dict, **eff_kwargs)
        if perfect_efficiency:
            detector_effs = torch.ones_like(ev['energy'])
        else:
            detector_effs = self.trigger_loss(geom_dict, **eff_kwargs)['t_per_event'].reshape(-1)

        # depending on the binning variable we bin in log10(E) or cos(zenith), with the resolution in the same units
        if binning_var_name == 'energy':
            E_min, E_max = gen.get('E_min', 1e2), gen.get('E_max', 1e8)
            bin_edges = torch.linspace(math.log10(E_min), math.log10(E_max), num_bins + 1, device=self.device)
            true_vals = torch.log10(ev['energy'])
            resolutions = res_loss_stuff['energy_resolution_per_event'].reshape(-1) / (ev['energy'] * math.log(10))
        else:
            # cos_range is in the sampler's travel convention, the events are in the arrival one.
            c_lo, c_hi = gen.get('cos_range', (-1.0, 1.0))
            bin_edges = torch.linspace(-c_hi, -c_lo, num_bins + 1, device=self.device)
            true_vals = ev['cos_zenith']
            resolutions = res_loss_stuff['angular_resolution_per_event'].reshape(-1) * torch.sin(ev['zenith'])
        # A singular per-event Fisher gives NaN/inf; such events are smeared out of every bin.
        resolutions = torch.nan_to_num(resolutions, nan=1e3, posinf=1e3).clamp_min(1e-6)

        selection = torch.ones_like(detector_effs)
        if energy_res_selection:
            if 'energy_resolution_per_event' not in res_loss_stuff:
                raise ValueError("analysis_energy_res_selection needs an energy resolution loss (analysis_binning_var_name='energy')")
            rel_res = res_loss_stuff['energy_resolution_per_event'].reshape(-1) / ev['energy']
            # A singular per-event Fisher (NaN/inf) counts as unresolved and is cut.
            rel_res = torch.nan_to_num(rel_res, nan=1e6, posinf=1e6).clamp_min(1e-12)
            selection = torch.sigmoid((math.log(energy_res_threshold) - torch.log(rel_res)) / energy_res_softness)

        # w_i and dw_i/dphi_j; the flux gradients are constants, geometry enters via sim weights, efficiency, selection and resolution
        acceptance = live_time * sim_weights * detector_effs * selection
        total_weights = acceptance * flux_weights
        param_names = list(grad_weights)
        grad_total_weights = torch.stack([grad_weights[k] for k in param_names]) * acceptance  # (n_params, N)

        indicators = soft_bin_indicators(true_vals, bin_edges, resolutions)  # (N, n_bins)
        mu = total_weights @ indicators                                    # (n_bins,)
        grad_mu = grad_total_weights @ indicators                          # (n_params, n_bins)

        # Everything needed to recompute exactly this batch again (used by AnalysisAccumulator).
        replay = {
            'signal_event_params': signal_event_params,
            'num_events': len(signal_event_params),
            'analysis_flux_weights': flux_weights.detach(),
            'analysis_grad_flux_weights': {k: g.detach() for k, g in grad_weights.items()},
            **{key: eff_kwargs.get(key) for key in PRECOMPUTED_PER_EVENT_KEYS},
        }
        if precomputed_sim_weights is not None:
            replay['analysis_sim_weights'] = precomputed_sim_weights

        return {
            'analysis_mu': mu,
            'analysis_grad_mu': grad_mu,
            'analysis_param_names': param_names,
            'analysis_bin_edges': bin_edges,
            'energy_resolution_per_event' if binning_var_name == 'energy' else 'angular_resolution_per_event': resolutions,
            'detector_efficiency_per_event': detector_effs,
            'selection_efficiency_per_event': selection,
            'analysis_replay': replay,
        }

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

        return {
            'fisher_analysis_loss': loss,
            'flux_param_names': signal_flux_var_names,
            'flux_param_variances': torch.diagonal(signal_cov),
            'flux_param_covariance': cov,
            'analysis_fisher_matrix': fisher,
        }
        

        


        

        
        
            