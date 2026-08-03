from nugget.losses.base_loss import LossFunction
import torch
import numpy as np
import math
import random
import pandas as pd
import os
from nugget.losses.trigger import TriggerLoss, ResolutionSelectionLoss
from nugget.losses.fisher_info import WeightedResolutionLoss
from torch.special import ndtr
from typing import Union, List
from numpy.typing import ArrayLike as Array
from friEnd.api import extend_dataframe, load_config
from friEnd.pyff_friend import PyFF_Friend

# ---------------------------------------------------------------------------
# Flux weighting via friEnd + pyForwardFolding
# ---------------------------------------------------------------------------

SEP = '::'  # separates base key from component index, e.g. 'position::0'

# Fluxes in friEnd/pyForwardFolding are per cm^2 (MCEq, KRA, and the
# PowerLawFlux baseline_norm convention all are), while nugget geometry is in
# metres. This converts the generation area m^2 -> cm^2.
_M2_TO_CM2 = 1e4


def to_pandas_jax(data_list, sep=SEP):
    """List[dict[str, Tensor]] -> pd.DataFrame with flattened columns."""
  

    keys = data_list[0].keys()
    stacked = {}

    for k in keys:
        vals = []
        for d in data_list:
            v = d[k]
            arr = np.asarray(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            # drop a leading dim of size 1, e.g. energy (1,) -> (), position (1,3) -> (3,)
            if arr.ndim >= 1 and arr.shape[0] == 1:
                arr = arr[0]
            vals.append(arr)
        stacked[k] = np.stack(vals)  # (N,) or (N, D)

    df_dict = {}
    shapes = {}
    for k, arr in stacked.items():
        shapes[k] = arr.shape[1:]
        if arr.ndim == 1:
            df_dict[k] = arr
        else:
            flat = arr.reshape(arr.shape[0], -1)
            for i in range(flat.shape[1]):
                df_dict[f'{k}{sep}{i}'] = flat[:, i]

    df = pd.DataFrame(df_dict)
    df.attrs['shapes'] = shapes
    return df


def from_pandas_jax(df, sep=SEP):
    """pd.DataFrame -> List[dict[str, torch.Tensor]], reversing to_pandas_jax."""
  

    groups = {}
    for col in df.columns:
        if sep in col:
            base, idx = col.rsplit(sep, 1)
            groups.setdefault(base, {})[int(idx)] = df[col].to_numpy()
        else:
            groups[col] = df[col].to_numpy()

    full = {}
    for k, v in groups.items():
        if isinstance(v, dict):
            ncols = max(v.keys()) + 1
            full[k] = np.stack([v[i] for i in range(ncols)], axis=1)  # (N, D)
        else:
            full[k] = v[:, None]  # (N, 1)

    shapes = df.attrs.get('shapes', {})
    n_rows = len(df)
    data_list = []
    for row in range(n_rows):
        d = {}
        for k, arr in full.items():
            val = arr[row]  # flat (D,)
            target_shape = shapes.get(k, val.shape)  # fall back to flat shape if unknown (new key)
            d[k] = torch.as_tensor(val).reshape(target_shape)
        data_list.append(d)
    return data_list


def _projected_area_np(cos_theta, cyl_radius, cyl_height):
    """A_proj(theta) = pi R^2 |cos| + 2 R H sin. Mirrors cyl_sampler.projected_area."""
  

    cos_theta = np.asarray(cos_theta, dtype=float)
    cap = np.pi * cyl_radius ** 2
    sides = 2.0 * cyl_radius * cyl_height
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, None))
    return cap * np.abs(cos_theta) + sides * sin_theta


def _as_float(x):
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def describe_sampler_generation(signal_sampler):
    """Read the generation parameters straight off a CylinderSampler.

    Returns a dict with the cylinder geometry, the energy range, the zenith
    window, and — crucially — which *direction* sampling mode was used, since
    that determines the form of the geometric part of the fluxless weight.

    ``uniform_zenith_sampling=False`` (the CylinderSampler default) rejection
    samples cos(theta) proportional to A_proj(theta) and places the vertex
    uniformly over that same A_proj(theta). The two cancel, so the whole
    geometric factor collapses to the single constant G = int A_proj dOmega.

    ``uniform_zenith_sampling=True`` samples theta uniformly on
    [theta_min, theta_max] (note: uniform in *theta*, not in cos(theta) — see
    cyl_sampler.sample_uniform_ray). The vertex is still uniform over
    A_proj(theta), so A_proj no longer cancels and must be applied per event
    together with 1/p(Omega) = 2 pi * dtheta * sin(theta).
    """
    skw = getattr(signal_sampler, 'kwargs', {}) or {}
    cyl = getattr(signal_sampler, 'cylinder', None)
    if cyl is None:
        raise ValueError(
            "signal_sampler has no 'cylinder' attribute; cannot infer the "
            "generation geometry needed for fluxless weights."
        )

    cos_range = skw.get('cos_range', (-1.0, 1.0))
    if isinstance(cos_range, str):
        mode = cos_range.strip().lower()
        # Mirrors _parse_zenith_cos_range / the hardcoded windows in
        # sample_uniform_ray.
        if 'horizontal' in mode:
            cos_window = 'horizontal'
        elif 'vertical' in mode:
            cos_window = 'vertical'
        else:
            raise ValueError(f"Unsupported cos_range string: {cos_range!r}")
    else:
        if torch.is_tensor(cos_range):
            cos_range = cos_range.detach().cpu().tolist()
        cos_window = (float(cos_range[0]), float(cos_range[1]))

    return {
        'cyl_radius': _as_float(cyl.radius),
        'cyl_height': _as_float(cyl.height),
        'E_min': float(skw.get('E_min', 0.8)),
        'E_max': float(skw.get('E_max', 1.0)),
        'energy_dist': skw.get('energy_dist', 'power_law'),
        'gamma': float(skw.get('gamma', 2.7 if getattr(signal_sampler, 'event_type', 'signal') == 'signal' else 3.7)),
        'cos_window': cos_window,
        'uniform_zenith_sampling': bool(skw.get('uniform_zenith_sampling', False)),
    }


def _inv_p_energy(energy, gen):
    """1 / p_gen(E), in GeV."""


    E = np.asarray(energy, dtype=float)
    E_min, E_max = gen['E_min'], gen['E_max']

    if gen['energy_dist'] == 'log_uniform':
        # p(E) = 1 / (E * ln(E_max/E_min))
        return E * np.log(E_max / E_min)

    # sample_power_law: p(E) ~ E^-gamma normalised on [E_min, E_max]
    gamma = gen['gamma']
    if abs(gamma - 1.0) < 1e-12:
        norm = np.log(E_max / E_min)
    else:
        expo = 1.0 - gamma
        norm = (E_max ** expo - E_min ** expo) / expo
    return norm * E ** gamma


def _geometric_weight(cos_theta, gen):
    """Geometric part of the fluxless weight, in m^2 sr.

    Returns (values, description). ``values`` is either a scalar (projected-area
    importance sampling) or a per-event array (uniform-zenith sampling).
    """

    R, H = gen['cyl_radius'], gen['cyl_height']
    window = gen['cos_window']

    if not gen['uniform_zenith_sampling']:
        # G = int dOmega A_proj(Omega) = 2 pi int A_proj dcos.
        # Full sphere closed form: 2 pi^2 R (R + H).
        if window == 'horizontal':
            c_lo, c_hi = -math.cos(math.radians(70.0)), math.cos(math.radians(70.0))
        elif window == 'vertical':
            # |cos| > 0.9, i.e. two symmetric intervals.
            c_grid = np.concatenate([
                np.linspace(-1.0, -0.9, 4001),
                np.linspace(0.9, 1.0, 4001),
            ])
            A = _projected_area_np(c_grid, R, H)
            G = 2.0 * np.pi * (
                np.trapezoid(A[:4001], c_grid[:4001])
                + np.trapezoid(A[4001:], c_grid[4001:])
            )
            return float(G), 'const_G(vertical)'
        else:
            c_lo, c_hi = window

        c_grid = np.linspace(c_lo, c_hi, 20001)
        G = 2.0 * np.pi * np.trapezoid(_projected_area_np(c_grid, R, H), c_grid)
        return float(G), 'const_G'

    # Uniform in theta: A_proj no longer cancels.
    cos_theta = np.asarray(cos_theta, dtype=float)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, None))
    A_proj = _projected_area_np(cos_theta, R, H)

    if window == 'horizontal':
        d_theta = math.radians(110.0 - 70.0)
        inv_p_omega = 2.0 * np.pi * d_theta * sin_theta
    elif window == 'vertical':
        # Mixture of two theta intervals, each chosen with probability 0.5:
        #   [0, acos(0.9)]  and  [acos(-0.9), pi]
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        t_min = math.acos(0.9)
        t_mid = math.acos(-0.9)
        d_lo, d_hi = t_min - 0.0, math.pi - t_mid
        p_theta = np.where(theta <= 0.5 * (t_min + t_mid), 0.5 / d_lo, 0.5 / d_hi)
        inv_p_omega = 2.0 * np.pi * sin_theta / p_theta
    else:
        c_lo, c_hi = window
        # theta is uniform on [acos(c_hi), acos(c_lo)]
        d_theta = math.acos(max(min(c_lo, 1.0), -1.0)) - math.acos(max(min(c_hi, 1.0), -1.0))
        inv_p_omega = 2.0 * np.pi * d_theta * sin_theta

    return inv_p_omega * A_proj, 'per_event_Aproj'


def compute_fluxless_weights(df, gen):
    """OneWeight-style generation weight, in GeV cm^2 sr.

        w_i = (1 / N_gen) * (1 / p(E_i)) * [geometric factor]

    so that ``w_i * dPhi/(dE dOmega)`` is a rate in Hz. The geometric factor is
    the constant G for projected-area sampling, or the per-event
    ``2 pi dtheta sin(theta) A_proj(theta)`` for uniform-zenith sampling.

    Note this covers *generation* only. The interaction and transmission
    probabilities are applied separately as a per-event acceptance (see
    ``effective_area_acceptance``), so that they are not double counted.
    """


    N_gen = len(df)

    energy = df['energy'].to_numpy(dtype=float)
    cos_theta = np.cos(df['zenith'].to_numpy(dtype=float))

    inv_pE = _inv_p_energy(energy, gen)
    geom, geom_kind = _geometric_weight(cos_theta, gen)

    weights = inv_pE * np.asarray(geom) / N_gen
    #weights *= _M2_TO_CM2  # m^2 -> cm^2
    return weights, geom_kind


def add_weights_to_signal_events(
    signal_event_params,
    friend_config,
    pyff_config,
    signal_sampler=None,
    gen=None,
    pid=14,
    temp_path=None,
    clear=True,
    verbose=False,
):
    """Attach ``weights`` and ``grad_weights_*`` to a list of signal events.

    Runs the friEnd step pipeline (coordinates, MCEq atmospheric splines,
    galactic maps, ...) followed by pyForwardFolding, which evaluates the
    parametric flux model and its per-event Jacobian w.r.t. the fit parameters.

    Parameters
    ----------
    signal_event_params : list of dict
        Events as produced by ``CylinderSampler.sample_events``.
    friend_config : str or dict
        friEnd step config (path or already-loaded dict).
    pyff_config : str or dict
        pyForwardFolding config. Its ``datasets[].path`` must point at
        ``temp_path``, since that is the file pyFF reads and writes.
    signal_sampler : CylinderSampler, optional
        Used to infer the generation parameters (cylinder, energy range,
        sampling mode) when ``gen`` is not given.
    gen : dict, optional
        Explicit generation parameters, as returned by
        ``describe_sampler_generation``. Overrides ``signal_sampler``.
    pid : int
        PDG code written to every event, used by the friEnd MCEq step to pick
        the flux species. 14 = nu_mu.
    temp_path : str, optional
        Scratch parquet path handed to pyFF. Must match the pyFF config.

    Returns
    -------
    list of dict
        The events, with ``weights`` and ``grad_weights_<param>`` added.
    """

    if gen is None:
        if signal_sampler is None:
            raise ValueError("Provide either 'gen' or 'signal_sampler'.")
        gen = describe_sampler_generation(signal_sampler)

    df = to_pandas_jax(signal_event_params)

    df['fluxless_weight'], geom_kind = compute_fluxless_weights(
        df, gen
    )
    df['pid'] = pid

    if verbose:
        print(
            f"[add_weights] N={len(df)} sampling={geom_kind} "
            f"R={gen['cyl_radius']:.1f}m H={gen['cyl_height']:.1f}m "
            f"E=[{gen['E_min']:.3g},{gen['E_max']:.3g}]GeV dist={gen['energy_dist']}"
        )

    if isinstance(friend_config, str):
        friend_config = load_config(friend_config)
    df_out = extend_dataframe(df, friend_config)

    if temp_path is None:
        temp_path = os.path.join('.', 'temp.parquet')
    os.makedirs(os.path.dirname(os.path.abspath(temp_path)), exist_ok=True)

    # pyFF reads the dataset from disk and writes 'weights'/'grad_weights_*'
    # back to the same path, so the round trip through parquet is required.
    df_out.to_parquet(temp_path)
    PyFF_Friend(pyffconfig=pyff_config, clear=clear).add_weights()
    df_weighted = pd.read_parquet(temp_path)
    df_weighted.attrs['shapes'] = df_out.attrs.get('shapes', {})

    return from_pandas_jax(df_weighted)


def bKDE(
    binning_var:Union[float,Array],
    bins: Array,
    bandwidth: Union[float,Array],
    reflect_infinities: bool = False,
) -> Array:
    """Differentiable histogram, defined via a binned kernel density estimate (bKDE).

    Parameters
    ----------
    data : Array
        1D array of data to histogram.
    bins : Array
        1D array of bin edges.
    bandwidth : float
        The bandwidth of the kernel. Bigger == lower gradient variance, but more bias.
    density : bool
        Normalise the histogram to unit area.
    reflect_infinities : bool
        If True, define bins at +/- infinity, and reflect their mass into the edge bins.

    Returns
    -------
    Array
        1D array of bKDE counts.
    """

    # Follow the data: binning_var carries the device/dtype of the event tensors,
    # so coerce the bin edges to match instead of assuming the caller built them
    # on the right device (torch.linspace defaults to CPU).
    binning_var = torch.as_tensor(binning_var)
    bins = torch.as_tensor(bins, device=binning_var.device, dtype=binning_var.dtype)
    if torch.is_tensor(bandwidth):
        bandwidth = bandwidth.to(device=binning_var.device, dtype=binning_var.dtype)

    if reflect_infinities:
        inf_edge = torch.tensor([torch.inf], device=bins.device, dtype=bins.dtype)
        bins = torch.cat([-inf_edge, bins.reshape(-1), inf_edge])

    # get cumulative counts (area under kde) for each set of bin edges
    z = ((bins.reshape(-1, 1) - binning_var) / bandwidth)

    cdf = ndtr(z)
    event_cdf = cdf
    #cdf /= weights.sum()
    # sum kde contributions in each bin
    counts = (event_cdf[1:, :] - event_cdf[:-1, :])

    if reflect_infinities:
        # Fold the two overflow bins back into the first/last real bins so no
        # kernel mass is lost outside the binning range.
        counts = counts[1:-1].clone()
        counts[0] = counts[0] + event_cdf[1] - event_cdf[0]
        counts[-1] = counts[-1] + event_cdf[-1] - event_cdf[-2]
    return counts

def bKDEnD(
        binning_vars: list,
        bins: list,
        uncerts: list,
        ):
    count_list = []
    for binning_var, uncert, bin1d in zip(binning_vars,uncerts,bins):
        # bKDE aligns bins/bandwidth to binning_var's device, so a caller that
        # built the bin edges on CPU (torch.linspace default) still works when
        # the event tensors live on a GPU.
        counts = bKDE(binning_var,bin1d,uncert)
        count_list.append(counts)

    out = count_list[0]
    for t in count_list[1:]:
        out = out.unsqueeze(-2) * t
    
    return out

def rearrange_matrix(matrix, indices):
    n = matrix.shape[0]
    
    # Create a new order of indices: specified ones first, then the rest
    new_order = indices + [i for i in range(n) if i not in indices]
    
    # Rearrange the rows and columns
    matrix_rearranged = matrix[new_order, :][:, new_order]
    return matrix_rearranged

def calc_fisher_matrix(mu,grad_hist,ssq,signal_idx):
    eps = 1e-8

    #TODO softmasking using ssq

    values = torch.stack(grad_hist).squeeze()      
    values = values / torch.sqrt(mu + eps)

    fim = torch.einsum('i...,j...->ij', values, values)
    fim = rearrange_matrix(fim,signal_idx)

    k = len(signal_idx)
    A = fim[:k,:k]
    B = fim[:k,k:]
    C = fim[k:,k:]
    marginalized_fim = A - B @ torch.linalg.inv(C) @ B.T

    return marginalized_fim

def calc_cov(fim):
    return torch.linalg.inv(fim)

def calc_weighted_hists(counts,weights):
    return (counts * weights).sum(dim=-1)

# Optimalities

def A_optimality(fim, **kwargs):
    cov = calc_cov(fim)
    diag = torch.diag(cov)
    return torch.sum(torch.sqrt(diag))



class AnalysisLoss(LossFunction):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith'], effective_area_loss=None, trigger_loss=None):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        print_loss : bool
            Whether to print loss components during computation.
        signal_surrogate_func : callable or None
            Function that computes signal light yield from event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
        signal_event_params : dict or None
            Dictionary containing signal event parameters.
        background_event_params : dict or None
            Dictionary containing background event parameters.
        batch_size_per_string : int
            Number of samples to generate per string for LLR computation.
        random_seed : int or None
            Random seed for reproducibility.
        fisher_info_params : list of str
            List of event parameters to compute Fisher information for.
        """
        super().__init__(device)
        
        self.print_loss = print_loss
        self.random_seed = random_seed
        self.fisher_info_params = fisher_info_params
        self.effective_area_loss = effective_area_loss
        self.trigger_loss = trigger_loss


    def _ensure_weights(
        self,
        signal_event_params,
        friend_config=None,
        pyff_config=None,
        signal_sampler=None,
        gen=None,
        pid=14,
        temp_path=None,
    ):
        """Add ``weights``/``grad_weights_*`` to events that lack them.

        Returns the list unchanged if the events already carry weights, so the
        expensive friEnd + pyForwardFolding pass runs at most once per event set
        rather than on every optimizer step.
        """
        if not signal_event_params:
            return signal_event_params

        test_event = signal_event_params[0]
        has_weights = 'weights' in test_event
        has_grads = any(k.startswith('grad_weights_') for k in test_event)
        if has_weights and has_grads:
            return signal_event_params

        if friend_config is None or pyff_config is None:
            raise ValueError(
                "Sampled events carry no flux weights and no 'friend_config'/"
                "'pyff_config' were provided. Either pass events that already "
                "have 'weights' and 'grad_weights_*', or supply both configs so "
                "they can be computed."
            )

        weighted = add_weights_to_signal_events(
            signal_event_params,
            friend_config=friend_config,
            pyff_config=pyff_config,
            signal_sampler=signal_sampler,
            gen=gen,
            pid=pid,
            temp_path=temp_path,
            verbose=self.print_loss,
        )

        # add_weights_to_signal_events round-trips through pandas, which turns
        # every field into a plain CPU tensor. Restore the pre-existing fields
        # from the originals (keeping device and any grad tracking), and move the
        # newly added weight columns onto self.device so they can be combined
        # with the acceptance / bKDE tensors later.
        for original, new in zip(signal_event_params, weighted):
            for key, value in list(new.items()):
                if key in original and torch.is_tensor(original[key]):
                    new[key] = original[key]
                elif torch.is_tensor(value):
                    new[key] = value.to(self.device)

        return weighted

    def __call__(self, geom_dict, **kwargs):
        """
        - True values
        - Uncertanties
        - Acceptance

        """
        # points_3d           = geom_dict.get('points_3d', None)

        # uncertainties       = kwargs.get('uncertainties') # :List[Tensor]
        # acceptance          = kwargs.get('detection_eff_func', None) # :Tensor

        binning_var_names   = kwargs.get('analysis_binning_var_names', ['energy', 'zenith']) # :List[str]
        num_events           = kwargs.get('num_events', 1) # :int
        signal_event_params = kwargs.get('signal_event_params', None) #:List[Dict]
        precomputed_fisher = kwargs.get('precomputed_fisher_info_per_string_per_event', None) # :Tensor
        precomputed_ly = kwargs.get('precomputed_light_yield_per_point_per_event', None) # :Tensor
        num_bins       = kwargs.get('analysis_num_bins',50) # :List[Tensor]
        signal_flux_var_names  = kwargs.get('analysis_signal_flux_var_names', ['astro_norm']) # :List[int]
        signal_sampler         = kwargs.get('signal_sampler') # :Callable
        # weights             = kwargs.get('flux_weights') # :Tensor
        # grad_weights        = kwargs.get('grad_flux_weights') # :List[Tensor]
        trigger_loss        = self.trigger_loss
        optimality          = kwargs.get('analysis_optimality','a') # :str
        live_time           = kwargs.get('live_time', 1.0) # :float
        # Effective-area acceptance: replaces trigger_loss as the acceptance term.
        # Pass an EffectiveAreaLoss instance; it already owns the cross-section
        # and transmission tables and runs the trigger internally.
        effective_area_loss =  self.effective_area_loss
        eff_area_cyl_radius = kwargs.get('eff_area_cyl_radius', None) # :float
        eff_area_cyl_height = kwargs.get('eff_area_cyl_height', None) # :float
        # Flux weighting (friEnd + pyForwardFolding), used when the sampled events
        # do not already carry 'weights'/'grad_weights_*'.
        friend_config       = kwargs.get('friend_config', None) # :str|dict
        pyff_config         = kwargs.get('pyff_config', None) # :str|dict
        weight_temp_path    = kwargs.get('weight_temp_path', None) # :str
        weight_pid          = kwargs.get('weight_pid', 14) # :int
    
        if signal_event_params is None:
            signal_event_params = signal_sampler.sample_events(num_events)
            weight_factor = live_time
            # Freshly sampled events have no flux weights yet. Weight them here,
            # before any subsampling, so N_gen is the full generated sample.
            signal_event_params = self._ensure_weights(
                signal_event_params,
                friend_config=friend_config,
                pyff_config=pyff_config,
                signal_sampler=signal_sampler,
                pid=weight_pid,
                temp_path=weight_temp_path,
            )
            kwargs['signal_event_params'] = signal_event_params
        else:
            # Provided events may or may not already carry weights (e.g. loaded
            # from a parquet that was run through friEnd/pyFF offline).
            signal_event_params = self._ensure_weights(
                signal_event_params,
                friend_config=friend_config,
                pyff_config=pyff_config,
                signal_sampler=signal_sampler,
                pid=weight_pid,
                temp_path=weight_temp_path,
            )
            # randomly sample a subset of the provided signal events and corresponding precomputed values if more than num_events are given
            if len(signal_event_params) > num_events:
                weight_factor = live_time*len(signal_event_params) / num_events
                selected_indices = random.sample(range(len(signal_event_params)), num_events)
                signal_event_params = [signal_event_params[i] for i in selected_indices]
                if precomputed_fisher is not None:
                    precomputed_fisher = precomputed_fisher[selected_indices]
                if precomputed_ly is not None:
                    precomputed_ly = precomputed_ly[selected_indices]
            else:
                weight_factor = live_time
            kwargs['signal_event_params'] = signal_event_params
            kwargs['precomputed_fisher_info_per_string_per_event'] = precomputed_fisher
            kwargs['precomputed_light_yield_per_point_per_event'] = precomputed_ly

        uncertainties = []
        # make a list of energy bin edges in logspace and zenith bin edges in linear space 
        for input_name in binning_var_names:
            if input_name == 'energy':
                weighted_resolution_loss=WeightedResolutionLoss(
                        device=self.device,
                        resolution_type='energy',
                        fisher_info_params=['energy']
                )
            elif input_name == 'zenith':
                weighted_resolution_loss=WeightedResolutionLoss(
                        device=self.device,
                        resolution_type='angular',
                        fisher_info_params=['direction']
                )
            loss_stuff = weighted_resolution_loss(geom_dict, **kwargs)
            uncerainty = loss_stuff['resolution_per_event']
            if input_name == 'energy':
                for i in range(len(uncerainty)):
                    uncerainty[i] =  uncerainty[i] / signal_event_params[i]['energy']
            elif input_name == 'zenith':
                kwargs['precalculated_resolution_loss'] = loss_stuff
                for i in range(len(uncerainty)):
                    uncerainty[i] =  uncerainty[i] * torch.abs(torch.sin(signal_event_params[i]['zenith']))
            uncertainties.append(uncerainty.squeeze())
            if input_name == 'zenith':
                selection_loss = ResolutionSelectionLoss(
                    device=self.device,
                    resolution_type='angular',
                    fisher_info_params=self.fisher_info_params
                )
                selection_acceptance = selection_loss(geom_dict, **kwargs)['selection_per_event']
        
        # Acceptance = analysis selection x [trigger x transmission x interaction].
        # The bracket comes from EffectiveAreaLoss, which runs the trigger itself
        # and multiplies in the neutrino-physics factors. The projected area is
        # deliberately excluded (include_projected_area=False) because A_proj is
        # already inside the fluxless weight -- via the constant G for
        # projected-area sampling, or explicitly per event for uniform-zenith
        # sampling. Including it again would double count a strongly
        # zenith-dependent factor and tilt the cos(zenith) distribution.
        acceptance = selection_acceptance.squeeze()

        if effective_area_loss is not None:
            # Use the *generation* cylinder so that w * P_int refers to
            # interactions in the same target volume the weights were built for.
            # A geometry-derived bounding cylinder would drift against the frozen
            # G and produce a spurious "shrink the detector" gradient.
            if eff_area_cyl_radius is None or eff_area_cyl_height is None:
                if signal_sampler is not None and hasattr(signal_sampler, 'cylinder'):
                    eff_area_cyl_radius = _as_float(signal_sampler.cylinder.radius)
                    eff_area_cyl_height = _as_float(signal_sampler.cylinder.height)

            eff_kwargs = kwargs
            eff_kwargs.update({
                'signal_event_params': signal_event_params,
                'per_event_effective_area_loss': True,
                'include_projected_area': False,
                'use_batched_effective_area': True,})
        
            if eff_area_cyl_radius is not None and eff_area_cyl_height is not None:
                eff_kwargs['use_sampler_cyl_for_volume'] = True
            eff_out = effective_area_loss(geom_dict, **eff_kwargs)
            # Already includes the trigger (t_per_event), so trigger_loss must
            # not be applied on top of it.
            acceptance = acceptance * eff_out['effective_area_per_event'].squeeze()
        elif trigger_loss is not None:
            acceptance = acceptance * trigger_loss(geom_dict, **kwargs)['t_per_event'].squeeze()
        ########
        weights = []
        grad_weights = []
        test_event = signal_event_params[0]
        energy_bins = torch.linspace(2,8, num_bins, device=self.device)
        zenith_bins = torch.linspace(-1,1, num_bins, device=self.device)
        signal_idx = []
        var_count = 0
        input_vars = []
        for input_name in binning_var_names:
            input_vars.append([])
        for key in test_event:
            if key.startswith('grad_weights_'):
                grad_weights.append([])
                for flux_var_name in signal_flux_var_names:
                    if key.endswith(flux_var_name):
                        signal_idx.append(var_count)
                var_count += 1
        for signal_event in signal_event_params:
            weights.append(signal_event['weights']*weight_factor)
            count = 0
            for key in signal_event:
                if key.startswith('grad_weights_'):
                    grad_weights[count].append(signal_event[key]*weight_factor)
                    count += 1
                if key == 'energy' and 'energy' in binning_var_names:
                    signal_event[key] = torch.log10(signal_event[key])
                if key == 'zenith' and 'zenith' in binning_var_names:
                    signal_event[key] = torch.cos(signal_event[key])
        weights = torch.stack(weights).squeeze()
        for i in range(len(grad_weights)):
            grad_weights[i] = torch.stack(grad_weights[i]).squeeze()
        bins = []
        for input_name in binning_var_names:
            if input_name == 'energy':
                bins.append(energy_bins)
            elif input_name == 'zenith':
                bins.append(zenith_bins)
        # print(bins)

        for i, input_name in enumerate(binning_var_names):
            for signal_event in signal_event_params:
                input_vars[i].append(signal_event[input_name])
            input_vars[i] = torch.stack(input_vars[i]).squeeze()
        # print(f"input_vars: {input_vars}")
        per_event_counts = bKDEnD(input_vars,bins,uncertainties)

        mu = calc_weighted_hists(per_event_counts,weights*acceptance)
        #ssq = calc_weighted_hist(per_event_counts,weights**2)
        grad_hist = [calc_weighted_hists(per_event_counts,grad_weight*acceptance) for grad_weight in grad_weights]

        fim = calc_fisher_matrix(mu,grad_hist,ssq=None,signal_idx=signal_idx)

        if optimality == "a":
            opti = A_optimality
        else:
            raise NotImplementedError(f"No {optimality} optimality")
        
        fisher_loss = opti(fim)

        # Marginalized covariance of the signal flux parameters. Its diagonal is
        # the per-parameter variance; ordering matches signal_flux_var_names,
        # since calc_fisher_matrix moves signal_idx to the front.
        cov = calc_cov(fim)
        param_variances = torch.diag(cov)

        if self.print_loss:
            print(f"Fisher Analysis Info Loss: {fisher_loss.item()}")

        return {
            'fisher_analysis_loss': fisher_loss,
            'flux_param_names': signal_flux_var_names,
            'flux_param_covariance': cov,
            'flux_param_variances': param_variances,
        }