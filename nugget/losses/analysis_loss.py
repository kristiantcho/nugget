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

from torch.special import ndtr

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

    keys = list(grad_hist.keys())
    values = torch.stack(grad_hist)          
    values = values / torch.sqrt(mu + eps)

    fim = torch.einsum('ib,jb->ij', values, values)
    fim = rearrange_matrix(fim)

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
    def __init__(self, device=None, print_loss=False, random_seed=None):
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




    def __call__(self, geom_dict, **kwargs):
        """
        - True values
        - Uncertanties
        - Acceptance

        """
        points_3d           = geom_dict.get('points_3d', None)

        uncertainties       = kwargs.get('uncertainties') # :List[Tensor]
        acceptance          = kwargs.get('acceptance', 1.) # :Tensor

        binning_var_names   = kwargs.get('binning_var_names') # :List[str]

        signal_event_params = kwargs.get('signal_event_params')

        bins                = kwargs.get('bins') # :List[Tensor]
        signal_idx          = kwargs.get('signal_idx') # :List[int]

        weights             = kwargs.get('weights') # :Tensor
        grad_weights        = kwargs.get('grad_weights') # :List[Tensor]

        optimality          = kwargs.get('optimality') # :str


        ########

        input_vars = [[a[input_name] for a in signal_event_params] for input_name in binning_var_names]

        per_event_counts = bKDEnd(input_vars,bins,uncertainties)

        mu = calc_weighted_hist(per_event_counts,weights*acceptance)
        #ssq = calc_weighted_hist(per_event_counts,weights**2)
        grad_hist = [calc_weighted_hist(per_event_counts,grad_weight*acceptance) for grad_weight in grad_weights]

        fim = calc_fisher_information_matrix(mu,grad_hist,ssq=None,signal_idx=signal_idx)

        if optimality == "a":
            opti = A_optimality
        else:
            raise NotImplementedError(f"No {optimality} optimality")
        
        fisher_loss = opti(fim)
        
        if self.print_loss:
            print(f"Fisher Analysis Info Loss: {fisher_loss.item()}")
        
        return {'fisher_analysis_loss': fisher_loss,}