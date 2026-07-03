from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F
import numpy as np
import time
import math
import random
import pickle
import gc
import os
from torch.func import jacrev, jvp, vmap, linearize
from nugget.losses.trigger import TriggerLoss, ResolutionSelectionLoss
from nugget.losses.fisher_info import WeightedResolutionLoss
from torch.special import ndtr
from typing import Union, List
from numpy.typing import ArrayLike as Array
import time

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

    bins = torch.Tensor([-torch.inf, *bins, torch.inf]) if reflect_infinities else bins

    # get cumulative counts (area under kde) for each set of bin edges
    z = ((bins.reshape(-1, 1) - binning_var) / bandwidth)

    cdf = ndtr(z)
    event_cdf = cdf
    #cdf /= weights.sum()
    # sum kde contributions in each bin
    counts = (event_cdf[1:, :] - event_cdf[:-1, :])

    if reflect_infinities:
        counts = (
            counts[1:-1]
            + torch.Tensor([counts[0]] + [0] * (len(counts) - 3))
            + torch.Tensor([0] * (len(counts) - 3) + [counts[-1]])
        )
    print(f"counts: {counts}")
    return counts

def bKDEnD(
        binning_vars: list, 
        bins: list,
        uncerts: list,
        ):
    count_list = []
    for binning_var, uncert, bin1d in zip(binning_vars,uncerts,bins):
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
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith']):
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
        num_bins                = kwargs.get('analysis_num_bins',50) # :List[Tensor]
        signal_flux_var_names          = kwargs.get('analysis_signal_flux_var_names', ['astro_norm']) # :List[int]
        signal_sampler         = kwargs.get('signal_sampler') # :Callable
        # weights             = kwargs.get('flux_weights') # :Tensor
        # grad_weights        = kwargs.get('grad_flux_weights') # :List[Tensor]
        trigger_loss        = kwargs.get('trigger_loss', None) # :Tensor
        optimality          = kwargs.get('analysis_optimality','a') # :str
        live_time           = kwargs.get('live_time', 1.0) # :float
        
        

    
        if signal_event_params is None:
            signal_event_params = signal_sampler.sample_events(num_events)
        else:
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
                        fisher_info_params=self.fisher_info_params
                )
            elif input_name == 'zenith':
                weighted_resolution_loss=WeightedResolutionLoss(
                        device=self.device,
                        resolution_type='angular',
                        fisher_info_params=self.fisher_info_params
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
        
        if trigger_loss is not None:
            acceptance = selection_acceptance.squeeze() * trigger_loss(geom_dict, **kwargs)['t_per_event'].squeeze()
        else:
            acceptance = selection_acceptance
        ########
        weights = []
        grad_weights = []
        test_event = signal_event_params[0]
        energy_bins = torch.linspace(2,8, num_bins)
        zenith_bins = torch.linspace(-1,1, num_bins)
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
        
        if self.print_loss:
            print(f"Fisher Analysis Info Loss: {fisher_loss.item()}")
        
        return {'fisher_analysis_loss': fisher_loss,}