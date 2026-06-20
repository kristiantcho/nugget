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

from nugget.losses.fisher_info_helpers import (
    _pos_norm_divisor_from_domain_size,
    _llr_mask_from_true_ly,
    _fisher_chunk_cleanup,
    _llr_out_single_point_all_iters,
    _fisher_one_point_jacrev,
    _unflatten_theta,
    _sample_detector_responses_batched,
    _build_features_from_cached_responses,
    _sample_rich_observations,
    _build_rich_features_from_cached_obs,
    _fisher_points_all_iters_jvp,
    _compute_fisher_llr_over_points,
    directional_resolution,
    compute_fisher_info_single_averaged,
    compute_fisher_info_single,
    compute_fisher_info_strings,
)

class FisherInfoLoss(LossFunction):
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
        
        # self.total_samples_per_point = num_samples
        self.print_loss = print_loss
        # self.batch_size_per_point = batch_size_per_point
        self.random_seed = random_seed
        # self.signal_surrogate_func = signal_surrogate_func
        # self.background_surrogate_func = background_surrogate_func
        # self.signal_event_params = signal_event_params
        # self.background_event_params = background_event_params
        self.fisher_info_params = fisher_info_params # Default parameters for Fisher Info  
    
    # def __call__experimental(self, geom_dict, **kwargs):
    #     """
    #     Compute the total Fisher information loss = 1/det(FisherInfo).
        
    #     Parameters:
    #     -----------
    #     points_3d : torch.Tensor
    #         The 3D points to evaluate the loss at.
    #     event_params : list of dict
    #         List of dictionaries containing event parameters.
    #     surrogate_func : callable
    #         Function that computes light yield from event parameters.
            
    #     Returns:
    #     --------
    #     torch.Tensor
    #         The total Fisher information loss value.
    #     """
    #     points_3d = geom_dict.get('points_3d', None)
    #     event_params = kwargs.get('signal_event_params', None)
    #     surrogate_func = kwargs.get('signal_surrogate_func', None)
    #     signal_sampler = kwargs.get('signal_sampler', None)
    #     num_events = kwargs.get('num_events', 1)
        
    #     if event_params is None and signal_sampler is not None:
    #         event_params = signal_sampler.sample_events(num_events)
        
    #     # Use batch computation for all points and events
    #     total_fisher_info = compute_fisher_info_batch_points(
    #         self.fisher_info_params, 
    #         points_3d, 
    #         event_params, 
    #         surrogate_func, 
    #         self.device
    #     )
            
    #     fisher_loss = 1/(torch.det(total_fisher_info) + 1e-6)  # Add small value to diagonal for numerical stability
        
    #     if self.print_loss:
    #         print(f"Fisher Info Loss: {fisher_loss.item()}")
        
    #     # return fisher_loss, total_fisher_info
    #     return {'fisher_loss': fisher_loss, 'total_fisher_info': total_fisher_info}
        
    def __call__(self, geom_dict, **kwargs):
        """
        Legacy method - kept for backward compatibility and testing.
        Compute the total Fisher information loss using the original single-event approach.
        """
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        max_energy_resolution = kwargs.get('max_energy_resolution', 1)
        max_angular_resolution = kwargs.get('max_angular_resolution', torch.pi)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)
        n_params = len(self.fisher_info_params)
        total_fisher_info = torch.zeros(n_params, n_params, device=self.device)
        fisher_info_per_point = torch.zeros((len(points_3d), n_params, n_params), device=self.device)
        for i, point in enumerate(points_3d):
            fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
            for params in event_params:
                for _ in range(llr_iterations):
                    fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, params, surrogate_func, llr_net, signal_noise_scale, add_relative_pos=add_relative_pos)/len(event_params)
            total_fisher_info += fisher_matrix/llr_iterations
            fisher_info_per_point[i] += fisher_matrix

        fisher_loss = torch.trace(torch.inverse(total_fisher_info + 1e-5 * torch.eye(total_fisher_info.shape[0], device=total_fisher_info.device)))  # Increased regularization for numerical stability
        if 'energy' in self.fisher_info_params and ('azimuth' in self.fisher_info_params or 'zenith' in self.fisher_info_params):
            fisher_loss = fisher_loss/(max_angular_resolution + max_energy_resolution)
        elif 'energy' in self.fisher_info_params:
            fisher_loss = fisher_loss/max_energy_resolution
        elif 'azimuth' in self.fisher_info_params or 'zenith' in self.fisher_info_params:
            fisher_loss = fisher_loss/max_angular_resolution
        
        if self.print_loss:
            print(f"Fisher Info Loss: {fisher_loss.item()}")
        
        return {'fisher_loss': fisher_loss, 'total_fisher_info': total_fisher_info, 'fisher_info_per_point': fisher_info_per_point}
        
class WeightedFisherInfoLoss(LossFunction):
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
        
        # self.total_samples_per_point = num_samples
        self.print_loss = print_loss
        # self.batch_size_per_point = batch_size_per_point
        self.random_seed = random_seed
        # self.signal_surrogate_func = signal_surrogate_func
        # self.background_surrogate_func = background_surrogate_func
        # self.signal_event_params = signal_event_params
        # self.background_event_params = background_event_params
        self.fisher_info_params = fisher_info_params # Default parameters for Fisher Info

    def compute_fisher_info_per_string_per_event(
        self, string_xy, points_3d, signal_event_params, signal_surrogate_func,
        llr_net=None, signal_noise_scale=None, llr_iterations=1, add_relative_pos=False,
        skip_zero_response=True, verbose=False, event_batch_size=1, grad_chunk_size=10,
        jacrev_chunk_size=10000, point_chunk_size=None, llr_autodiff_mode='jacrev',
        detach_fisher_tensors=True, use_patd=False, eval_patd_log_probs=None,
        use_rich_features=False, use_patd_quadrature=False, use_charge_quadrature=False,
        charge_center_on_llr_peak=False, charge_peak_scan_points=64,
        t_offset_ns=100.0, t_max_ns=10000.0, zero_response_threshold=0.5,
        adaptive_grid_retry=True, adaptive_t_max_floor_ns=10.0, uninformative_fisher_value=1e-6,
        precomputed_fisher_per_string_per_event=None, recompute_bad_points=True,
    ):
        n_strings = len(string_xy)
        # use_rich_features is now stored on the model — read from it if available.
        if llr_net is not None and not use_rich_features:
            use_rich_features = bool(getattr(llr_net, 'use_rich_features', False))
        param_dims = []
        param_names_expanded = []
        for param_name in self.fisher_info_params:
            param_value = signal_event_params[0].get(param_name)
            if param_value.dim() == 0 or (param_value.dim() == 1 and param_value.shape[0] == 1):
                # Scalar parameter
                param_dims.append(1)
                param_names_expanded.append(param_name)
            else:
                # Multi-dimensional parameter (e.g., 3D vector)
                dim_size = param_value.numel()
                param_dims.append(dim_size)
                for i in range(dim_size):
                    param_names_expanded.append(f"{param_name}_{i}")
        total_dims = sum(param_dims)
        n_events = len(signal_event_params)

        # If a precomputed Fisher tensor is supplied, start from it and only
        # recompute the "bad" strings (zero / NaN / Inf) per event.  Otherwise
        # start from zeros and compute everything.
        have_precomp = precomputed_fisher_per_string_per_event is not None
        if have_precomp:
            fisher_per_string_per_event = precomputed_fisher_per_string_per_event.to(self.device).clone()
            if fisher_per_string_per_event.shape != (n_events, n_strings, total_dims, total_dims):
                raise ValueError(
                    "precomputed_fisher_per_string_per_event has shape "
                    f"{tuple(fisher_per_string_per_event.shape)}, expected "
                    f"{(n_events, n_strings, total_dims, total_dims)}."
                )
            if not recompute_bad_points:
                # Caller just wants the precomputed tensor back (no recomputation).
                return fisher_per_string_per_event
        else:
            fisher_per_string_per_event = torch.zeros(
                n_events, n_strings, total_dims, total_dims, device=self.device)

        def _bad_strings(F_event):
            # F_event: (n_strings, D, D). Bad = non-finite anywhere, or exactly zero.
            nonfinite = ~torch.isfinite(F_event).all(dim=(1, 2))           # (n_strings,)
            near_zero = F_event.abs().amax(dim=(1, 2)) <= 0.0              # (n_strings,)
            return nonfinite | near_zero

        # Pre-map each string -> the indices of points_3d that belong to it, so a
        # bad-string subset can be turned into a point subset without rescanning.
        def _string_xy_val(s):
            sx, sy = string_xy[s][0], string_xy[s][1]
            sx = sx.to(self.device) if isinstance(sx, torch.Tensor) else torch.tensor(sx, device=self.device)
            sy = sy.to(self.device) if isinstance(sy, torch.Tensor) else torch.tensor(sy, device=self.device)
            return sx, sy

        def _compute(points, sxy):
            return compute_fisher_info_single_averaged(
                        fisher_info_params=self.fisher_info_params,
                        point=points,
                        event_params=signal_params,
                        surrogate_func=signal_surrogate_func,
                        llr_net=llr_net,
                        signal_noise_scale=signal_noise_scale,
                        add_relative_pos=add_relative_pos,
                        skip_zero_response=skip_zero_response,
                        llr_iterations=llr_iterations,
                        string_xy=sxy,
                        device=self.device,
                        grad_chunk_size=grad_chunk_size,
                        jacrev_chunk_size=jacrev_chunk_size,
                        point_chunk_size=point_chunk_size,
                        llr_autodiff_mode=llr_autodiff_mode,
                        detach_fisher_tensors=detach_fisher_tensors,
                        use_patd=use_patd,
                        eval_patd_log_probs=eval_patd_log_probs,
                        use_rich_features=use_rich_features,
                        use_patd_quadrature=use_patd_quadrature,
                        use_charge_quadrature=use_charge_quadrature,
                        charge_center_on_llr_peak=charge_center_on_llr_peak,
                        charge_peak_scan_points=charge_peak_scan_points,
                        t_offset_ns=t_offset_ns,
                        t_max_ns=t_max_ns,
                        zero_response_threshold=zero_response_threshold,
                        adaptive_grid_retry=adaptive_grid_retry,
                        adaptive_t_max_floor_ns=adaptive_t_max_floor_ns,
                        uninformative_fisher_value=uninformative_fisher_value,
                        )

        for i, signal_params in enumerate(signal_event_params):
            if have_precomp:
                # Recompute only the strings flagged bad in the precomputed slice.
                bad_mask = _bad_strings(fisher_per_string_per_event[i])     # (n_strings,)
                bad_strings = bad_mask.nonzero(as_tuple=True)[0]
                if bad_strings.numel() == 0:
                    if verbose and (i % 100 == 0 or i == n_events - 1):
                        print(f"Event {i+1}/{n_events}: 0 bad strings, kept precomputed", flush=True)
                    continue
                # Build the point subset belonging to the bad strings, preserving
                # the bad-string order so the returned (n_bad, D, D) maps back 1:1.
                bad_sxy = [string_xy[int(s)] for s in bad_strings.tolist()]
                point_masks = []
                for s in bad_strings.tolist():
                    sx, sy = _string_xy_val(s)
                    point_masks.append((points_3d[:, 0] == sx) & (points_3d[:, 1] == sy))
                pts_mask = torch.stack(point_masks, dim=0).any(dim=0)       # (n_points,)
                pts_subset = points_3d[pts_mask]
                if pts_subset.shape[0] == 0:
                    # No points map to these strings — assign uninformative defaults.
                    eye = torch.eye(total_dims, device=self.device)
                    fisher_per_string_per_event[i, bad_strings] = uninformative_fisher_value * eye
                else:
                    recomputed = _compute(pts_subset, bad_sxy).to(self.device)  # (n_bad, D, D)
                    fisher_per_string_per_event[i, bad_strings] = recomputed
                if verbose and (i % 100 == 0 or i == n_events - 1):
                    print(f"Event {i+1}/{n_events}: recomputed {bad_strings.numel()} bad strings", flush=True)
            else:
                fisher_matrices = _compute(points_3d, string_xy)
                fisher_per_string_per_event[i] += fisher_matrices.to(self.device)
                if verbose and (i % 100 == 0 or i == n_events - 1):
                    print(f"Computed Fisher info for event {i+1}/{n_events}", flush=True)
        # else:
        #     for i in range(0, len(signal_event_params), event_batch_size):
        #         batch_params = signal_event_params[i:i+event_batch_size]
        #         # Returns (n_strings, n_events, total_dims, total_dims), need to permute to (n_events, n_strings, ...)
        #         fisher_matrices = compute_fisher_info_single_batched(
        #                     fisher_info_params=self.fisher_info_params, 
        #                     point=points_3d, 
        #                     event_params_list=batch_params, 
        #                     surrogate_func=signal_surrogate_func, 
        #                     llr_net=llr_net, 
        #                     signal_noise_scale=signal_noise_scale, 
        #                     add_relative_pos=add_relative_pos, 
        #                     skip_zero_response=skip_zero_response, 
        #                     llr_iterations=llr_iterations, 
        #                     string_xy=string_xy
        #                     )
        #         fisher_per_string_per_event[i:i+event_batch_size] += fisher_matrices.permute(1, 0, 2, 3)   
            

        return fisher_per_string_per_event

    # def compute_fisher_info_per_string_experimental(self, string_xy, points_3d, signal_event_params, signal_surrogate_func):
    #     """
    #     Compute the Fisher information for each string using batch computation.
        
    #     Parameters:
    #     -----------
    #     string_xy : list of torch.Tensor or None
    #         The 2D points of the strings to compute the penalty for.
    #     points_3d : torch.Tensor
    #         The 3D points to evaluate the loss at.
    #     signal_event_params : list of dict
    #         List of dictionaries containing signal event parameters.
    #     signal_surrogate_func : callable
    #         Function that computes signal light yield from event parameters.
    #     background_event_params : list of dict or None
    #         List of dictionaries containing background event parameters.
    #     background_surrogate_func : callable or None
    #         Function that computes background light yield from event parameters.
    #     """
    #     n_strings = len(string_xy)
    #     n_params = len(self.fisher_info_params)
    #     fisher_info_per_string = torch.zeros(n_strings, n_params, n_params, device=self.device)
        
    #     for s_idx in range(n_strings):
    #         # Create optimization points for this string
    #         # Sample points along the string (assuming strings extend in z-direction)
    #         mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
    #         string_points = points_3d[mask]
            
    #         if len(string_points) > 0:
    #             # Use batch computation for all points on this string with all events
    #             fisher_matrix = compute_fisher_info_batch_points(
    #                 self.fisher_info_params, 
    #                 string_points, 
    #                 signal_event_params, 
    #                 signal_surrogate_func, 
    #                 self.device
    #             )
    #             fisher_info_per_string[s_idx] = fisher_matrix
            
    #     return fisher_info_per_string
        
    def compute_fisher_info_per_string(self, string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net=None, signal_noise_scale=None, llr_iterations=1, add_relative_pos=False):
        """
        Legacy method - kept for backward compatibility and testing.
        Compute the Fisher information for each string using the original single-event approach.
        """
        n_strings = len(string_xy)
        n_params = len(self.fisher_info_params)
        fisher_info_per_string = torch.zeros(n_strings, n_params, n_params, device=self.device)
        
        for s_idx in range(n_strings):
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]
            fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
            for point in string_points:  
                for signal_params in signal_event_params:
                    for _ in range(llr_iterations):    
                        fisher_matrix += compute_fisher_info_single(self.fisher_info_params, point, event_params=signal_params, surrogate_func=signal_surrogate_func, llr_net=llr_net, signal_noise_scale=signal_noise_scale, add_relative_pos=add_relative_pos)/len(signal_event_params)
            fisher_info_per_string[s_idx] += fisher_matrix/llr_iterations
            
        return fisher_info_per_string
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(WeightedFisherInfo).
        
        Parameters:
        -----------
        string_xy : list of torch.Tensor or None
            The 2D points of the strings to compute the penalty for.
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        signal_event_params : list of dict
            List of dictionaries containing signal event parameters.
        signal_surrogate_func : callable
            Function that computes signal light yield from event parameters.
        background_event_params : list of dict or None
            List of dictionaries containing background event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        # precomputed_fisher_info_per_string = kwargs.get('precomputed_fisher_info_per_string', None)
        precomputed_fisher_info_per_string_per_event = kwargs.get('precomputed_fisher_info_per_string_per_event', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        max_energy_resolution = kwargs.get('max_energy_resolution', 1)
        max_angular_resolution = kwargs.get('max_angular_resolution', torch.pi)
        # background_event_params = kwargs.get('background_event_params', None)
        # background_surrogate_func = kwargs.get('background_surrogate_func', None)
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        # if precomputed_fisher_info_per_string is None:
        #     fisher_info_per_string = self.compute_fisher_info_per_string(string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos)
        if precomputed_fisher_info_per_string_per_event is None:
            fisher_info_per_string_per_event = self.compute_fisher_info_per_string_per_event(string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos) 
        # else:
        #     fisher_info_per_string = precomputed_fisher_info_per_string
        else:
            fisher_info_per_string_per_event = precomputed_fisher_info_per_string_per_event
        # if string_weights is None:
        #     total_fisher_info = torch.sum(fisher_info_per_string, dim=0)  # Sum over strings, keep matrix form
        # else:
        #     string_probs = torch.sigmoid(string_weights)
        #     total_fisher_info = torch.sum(string_probs.unsqueeze(1).unsqueeze(2) * fisher_info_per_string, dim=0)  # Weighted sum
        # fisher_loss = torch.det(torch.inverse(total_fisher_info + 1e-6 * torch.eye(total_fisher_info.shape[0], device=total_fisher_info.device)))  # Add small value to diagonal for numerical stability
        if string_weights is None:
            total_fisher_info_per_event = torch.sum(fisher_info_per_string_per_event, dim=1)  # Sum over strings, keep matrix form
        else:
            string_probs = torch.sigmoid(string_weights)
            total_fisher_info_per_event = torch.sum(string_probs.unsqueeze(1).unsqueeze(2) * fisher_info_per_string_per_event, dim=1)
        mean_fisher_inv_trace = 0
        for i in range(total_fisher_info_per_event.shape[0]):
            mean_fisher_inv_trace += torch.trace(torch.inverse(total_fisher_info_per_event[i] + 1e-5 * torch.eye(total_fisher_info_per_event.shape[1], device=total_fisher_info_per_event.device)))/total_fisher_info_per_event.shape[0]  # Increased regularization for numerical stability
        fisher_loss = mean_fisher_inv_trace
        if 'energy' in self.fisher_info_params and ('azimuth' in self.fisher_info_params or 'zenith' in self.fisher_info_params):
            fisher_loss = fisher_loss/(max_angular_resolution + max_energy_resolution)
        elif 'energy' in self.fisher_info_params:
            fisher_loss = fisher_loss/max_energy_resolution
        elif 'azimuth' in self.fisher_info_params or 'zenith' in self.fisher_info_params:
            fisher_loss = fisher_loss/max_angular_resolution
        # return {'fisher_loss': fisher_loss, 'fisher_info_per_string': fisher_info_per_string, 'total_fisher_info': total_fisher_info}
        return {'fisher_loss': fisher_loss, 'fisher_info_per_string_per_event': fisher_info_per_string_per_event, 'total_fisher_info_per_event': total_fisher_info_per_event, 'fisher_signal_params': signal_event_params}

class WeightedResolutionLoss(WeightedFisherInfoLoss):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith'], resolution_type='angular'):
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
        resolution_type : str
            Type of resolution to compute ('angular' or 'energy').
        """
        super().__init__(device, print_loss, random_seed, fisher_info_params)
        
        self.resolution_type = resolution_type # 'angular' or 'energy'
    
    def _load_and_batch_events_fisher_info(self, event_paths, fisher_info_paths, 
                                           num_events, signal_event_params, precomputed_fisher_info_per_string_per_event):
        """
        Helper function to load and batch events and Fisher info from files or subset precomputed data.
        
        Parameters:
        -----------
        event_paths : list of str or None
            List of file paths containing precomputed event parameters.
        fisher_info_paths : list of str or None
            List of file paths containing precomputed Fisher info per string per event.
        batch_size_per_iteration : int or None
            Number of events to use per iteration when using batched loading.
        num_events : int
            Default number of events if batch_size_per_iteration not specified.
        signal_event_params : list of dict or None
            Current signal event parameters.
        precomputed_fisher_info_per_string_per_event : torch.Tensor or None
            Current precomputed Fisher info.
            
        Returns:
        --------
        tuple
            (signal_event_params, precomputed_fisher_info_per_string_per_event)
        """
        import random
        import pickle
        
        # If precomputed data is already provided, ignore file paths and just handle subsetting
        if signal_event_params is not None and precomputed_fisher_info_per_string_per_event is not None:
            if num_events is not None:
                # Get total number of events
                n_total_events = len(signal_event_params)
                
                # Randomly select indices if subsetting is needed
                if n_total_events > num_events:
                    selected_indices = random.sample(range(n_total_events), num_events)
                    selected_indices = sorted(selected_indices)  # Sort for consistent ordering
                    
                    # Subset events
                    signal_event_params = [signal_event_params[i] for i in selected_indices]
                    
                    # Subset Fisher info
                    precomputed_fisher_info_per_string_per_event = precomputed_fisher_info_per_string_per_event[selected_indices]
            
            return signal_event_params, precomputed_fisher_info_per_string_per_event
        
        # Handle batched loading from files (only if precomputed data not fully provided)
        if event_paths is not None and fisher_info_paths is not None:
            # Ensure paths are lists
            if not isinstance(event_paths, list):
                event_paths = [event_paths]
            if not isinstance(fisher_info_paths, list):
                fisher_info_paths = [fisher_info_paths]
            
            assert len(event_paths) == len(fisher_info_paths), "event_paths and fisher_info_paths must have the same length"
            
            # Use batch size if provided, otherwise use num_events
            target_batch_size = num_events
            
            # Randomly shuffle file indices
            file_indices = list(range(len(event_paths)))
            random.shuffle(file_indices)
            
            loaded_events = []
            loaded_fisher_info = []
            
            for idx in file_indices:
                if len(loaded_events) >= target_batch_size:
                    break
                
                # Load events
                event_path = event_paths[idx]
                if event_path.endswith('.pkl'):
                    with open(event_path, 'rb') as f:
                        events = pickle.load(f)
                else:
                    events = torch.load(event_path)
                
                # Load Fisher info
                fisher_path = fisher_info_paths[idx]
                fisher_info = torch.load(fisher_path)
                
                # Ensure events are on the correct device
                if isinstance(events, list):
                    for event in events:
                        for key in event:
                            if isinstance(event[key], torch.Tensor):
                                event[key] = event[key].to(self.device)
                
                # Ensure Fisher info is on the correct device
                fisher_info = fisher_info.to(self.device)
                
                # Add to loaded data
                if isinstance(events, list):
                    loaded_events.extend(events)
                else:
                    loaded_events.append(events)
                
                # Handle Fisher info structure
                if fisher_info.dim() == 4:  # (n_events, n_strings, dims, dims)
                    for i in range(fisher_info.shape[0]):
                        loaded_fisher_info.append(fisher_info[i])
                elif fisher_info.dim() == 3:  # Single event (n_strings, dims, dims)
                    loaded_fisher_info.append(fisher_info)
                else:
                    loaded_fisher_info.append(fisher_info)
            
            # Subset to target batch size
            signal_event_params = loaded_events[:target_batch_size]
            
            # Stack Fisher info
            if len(loaded_fisher_info) > 0 and isinstance(loaded_fisher_info[0], torch.Tensor):
                precomputed_fisher_info_per_string_per_event = torch.stack(loaded_fisher_info[:target_batch_size])
            else:
                precomputed_fisher_info_per_string_per_event = None
        
        # Handle case where only precomputed Fisher info is provided (but not events)
        elif precomputed_fisher_info_per_string_per_event is not None and num_events is not None:
            # Get total number of events
            n_total_events = len(signal_event_params) if signal_event_params is not None else precomputed_fisher_info_per_string_per_event.shape[0]
            
            # Randomly select indices
            if n_total_events > num_events:
                selected_indices = random.sample(range(n_total_events), num_events)
                selected_indices = sorted(selected_indices)  # Sort for consistent ordering
                
                # Subset events if they exist
                if signal_event_params is not None:
                    signal_event_params = [signal_event_params[i] for i in selected_indices]
                
                # Subset Fisher info
                precomputed_fisher_info_per_string_per_event = precomputed_fisher_info_per_string_per_event[selected_indices]
        
        return signal_event_params, precomputed_fisher_info_per_string_per_event

    def __call__(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(WeightedFisherInfo).
        
        Parameters:
        -----------
        string_xy : list of torch.Tensor or None
            The 2D points of the strings to compute the penalty for.
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        signal_event_params : list of dict
            List of dictionaries containing signal event parameters.
        signal_surrogate_func : callable
            Function that computes signal light yield from event parameters.
        event_paths : list of str or None
            List of file paths containing precomputed event parameters.
        fisher_info_paths : list of str or None
            List of file paths containing precomputed Fisher info per string per event.
        batch_size_per_iteration : int or None
            Number of events to use per iteration when using batched loading.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        precomputed_fisher_info_per_string_per_event = kwargs.get('precomputed_fisher_info_per_string_per_event', None)
        string_weights = geom_dict.get('string_weights', None)
        string_xy = geom_dict.get('string_xy', None)
        points_3d = geom_dict.get('points_3d', None)
        signal_event_params = kwargs.get('signal_event_params', None)
        signal_surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        max_angular_resolution = kwargs.get('max_angular_resolution', torch.pi) # radians
        max_energy_resolution = kwargs.get('max_energy_resolution', 1) # fraction
        use_relative_energy = kwargs.get('use_relative_energy', False)
        skip_zero_response = kwargs.get('skip_zero_response', True)
        event_batch_size = kwargs.get('fisher_info_event_batch_size', 1)
        grad_chunk_size = kwargs.get('fisher_info_grad_chunk_size', 10)
        jacrev_chunk_size = kwargs.get('fisher_info_jacrev_chunk_size', 10000)
        point_chunk_size = kwargs.get('fisher_info_point_chunk_size', None)
        llr_autodiff_mode = kwargs.get('fisher_info_llr_autodiff_mode', 'jacrev')
        detach_fisher_tensors = kwargs.get('fisher_info_detach_tensors', True)
        use_patd = kwargs.get('fisher_info_use_patd', False)
        eval_patd_log_probs = kwargs.get('eval_patd_log_probs', None)
        use_rich_features = kwargs.get('use_rich_features', False)
        use_patd_quadrature = kwargs.get('use_patd_quadrature', False)
        use_charge_quadrature = kwargs.get('use_charge_quadrature', False)
        charge_center_on_llr_peak = kwargs.get('charge_center_on_llr_peak', False)
        charge_peak_scan_points = kwargs.get('charge_peak_scan_points', 64)
        t_offset_ns = kwargs.get('t_offset_ns', 100.0)
        t_max_ns = kwargs.get('t_max_ns', 10000.0)
        zero_response_threshold = kwargs.get('zero_response_threshold', 0.5)
        adaptive_grid_retry = kwargs.get('adaptive_grid_retry', True)
        adaptive_t_max_floor_ns = kwargs.get('adaptive_t_max_floor_ns', 10.0)
        uninformative_fisher_value = kwargs.get('uninformative_fisher_value', 1e-6)
        # When a precomputed Fisher tensor is supplied, optionally recompute only
        # its bad (zero/NaN/Inf) strings per event instead of using it verbatim.
        recompute_bad_points = kwargs.get('recompute_bad_points', False)

        # New parameters for batched loading from files
        event_paths = kwargs.get('event_paths', None)
        fisher_info_paths = kwargs.get('fisher_info_paths', None)
        # batch_size_per_iteration = kwargs.get('batch_size_per_iteration', None)
        
        # Load and batch events/Fisher info from files or subset precomputed data
        signal_event_params, precomputed_fisher_info_per_string_per_event = self._load_and_batch_events_fisher_info(
            event_paths, fisher_info_paths, num_events,
            signal_event_params, precomputed_fisher_info_per_string_per_event
        )
        
        # Standard event sampling if no precomputed data
        if signal_event_params is None and signal_sampler is not None:
            signal_event_params = signal_sampler.sample_events(num_events)
        
        # Compute Fisher info. Three cases:
        #   1. No precomputed tensor                  -> compute everything.
        #   2. Precomputed + recompute_bad_points      -> recompute only bad strings.
        #   3. Precomputed + not recompute_bad_points  -> use verbatim.
        if precomputed_fisher_info_per_string_per_event is None or recompute_bad_points:
            fisher_info_per_string_per_event = self.compute_fisher_info_per_string_per_event(
                string_xy,
                points_3d,
                signal_event_params,
                signal_surrogate_func,
                llr_net,
                signal_noise_scale,
                llr_iterations,
                add_relative_pos,
                skip_zero_response=skip_zero_response,
                event_batch_size=event_batch_size,
                grad_chunk_size=grad_chunk_size,
                jacrev_chunk_size=jacrev_chunk_size,
                point_chunk_size=point_chunk_size,
                llr_autodiff_mode=llr_autodiff_mode,
                detach_fisher_tensors=detach_fisher_tensors,
                use_patd=use_patd,
                eval_patd_log_probs=eval_patd_log_probs,
                use_rich_features=use_rich_features,
                use_patd_quadrature=use_patd_quadrature,
                use_charge_quadrature=use_charge_quadrature,
                charge_center_on_llr_peak=charge_center_on_llr_peak,
                charge_peak_scan_points=charge_peak_scan_points,
                t_offset_ns=t_offset_ns,
                t_max_ns=t_max_ns,
                zero_response_threshold=zero_response_threshold,
                adaptive_grid_retry=adaptive_grid_retry,
                adaptive_t_max_floor_ns=adaptive_t_max_floor_ns,
                uninformative_fisher_value=uninformative_fisher_value,
                precomputed_fisher_per_string_per_event=precomputed_fisher_info_per_string_per_event,
                recompute_bad_points=recompute_bad_points,
            )
        else:
            fisher_info_per_string_per_event = precomputed_fisher_info_per_string_per_event.to(self.device)
        #sum over strings with weights but keep per event
        if string_weights is None:
            total_fisher_info = torch.sum(fisher_info_per_string_per_event, dim=1)  # Sum over strings, keep matrix form
        else:
            string_probs = torch.sigmoid(string_weights)
            total_fisher_info = torch.sum(string_probs.unsqueeze(1).unsqueeze(2) * fisher_info_per_string_per_event, dim=1)
        resolution_per_event = []
        
        if self.resolution_type == 'angular':
            # Check if using direction or zenith/azimuth
            if 'direction' in self.fisher_info_params:
                # Use directional resolution with tangent space projection
                direction_idx = self.fisher_info_params.index('direction')
                # Calculate start index for direction in the expanded Fisher matrix
                start_idx = 0
                for j, param_name in enumerate(self.fisher_info_params):
                    if j < direction_idx:
                        param_val = signal_event_params[0].get(param_name)
                        if param_val.dim() == 0 or (param_val.dim() == 1 and param_val.shape[0] == 1):
                            start_idx += 1
                        else:
                            start_idx += param_val.numel()
                
                # Batch process all events at once
                # Extract all Fisher submatrices: (N, 3, 3)
                F3_batch = total_fisher_info[:, start_idx:start_idx+3, start_idx:start_idx+3]
                
                # Extract all directions: (N, 3)
                directions_batch = torch.stack([params['direction'].to(self.device) for params in signal_event_params])
                
                # Compute all resolutions at once: (N,)
                resolution_per_event = directional_resolution(F3_batch, directions_batch)
            else:
                # Use traditional zenith/azimuth resolution - need covariance matrix
                # Vectorized batch inverse
                n_events = len(signal_event_params)
                regularized_fisher = total_fisher_info + 1e-5 * torch.eye(
                    total_fisher_info.shape[1], device=self.device
                ).unsqueeze(0).expand(n_events, -1, -1)  # Increased regularization for stability
                
                try:
                    cov_matrix = torch.inverse(regularized_fisher)
                except:
                    # Fall back to loop if batch inverse fails
                    cov_matrix = []
                    for i in range(n_events):
                        try:
                            cov_matrix.append(torch.inverse(regularized_fisher[i]))
                        except:
                            cov_matrix.append(torch.pinverse(regularized_fisher[i]))
                    cov_matrix = torch.stack(cov_matrix)
                
                for i, params in enumerate(signal_event_params):
                    zenith = params['zenith']
                    azimuth_idx = self.fisher_info_params.index('azimuth')
                    zenith_idx = self.fisher_info_params.index('zenith')
                    # Angular resolution: sqrt(var_azimuth + var_zenith)
                    var_azimuth = cov_matrix[i][azimuth_idx, azimuth_idx]
                    var_zenith = cov_matrix[i][zenith_idx, zenith_idx]
                    covar_zenith_azimuth = cov_matrix[i][zenith_idx, azimuth_idx]
                    angular_resolution_rad = torch.sqrt(var_zenith + torch.sin(zenith)*var_azimuth + 2*torch.sin(zenith)*torch.cos(zenith)*covar_zenith_azimuth)
                    resolution_per_event.append(angular_resolution_rad)
                resolution_per_event = torch.stack(resolution_per_event)
            finite_mask = torch.isfinite(resolution_per_event) & (resolution_per_event > 1e-12)
            if finite_mask.any():
                # Replace bad entries with a large sentinel WITHIN the graph (not via
                # boolean indexing) so their 1/r^2 contribution is ~0 AND no NaN/inf
                # gradient flows back through string_weights for those events.
                safe_res = torch.where(
                    finite_mask,
                    torch.clamp_min(resolution_per_event, 1e-12),
                    torch.full_like(resolution_per_event, 1e6),
                )
                total_resolution = 1 / torch.sqrt(torch.sum(1 / (safe_res ** 2)))
            else:
                # Keep optimization stable when all events are invalid/singular.
                total_resolution = torch.tensor(1.0, device=self.device, requires_grad=True)
            return {'angular_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': signal_event_params}
        elif self.resolution_type == 'energy':
            # Compute covariance matrix for energy resolution
            # Vectorized batch inverse
            n_events = len(signal_event_params)
            regularized_fisher = total_fisher_info + 1e-6 * torch.eye(
                total_fisher_info.shape[1], device=self.device
            ).unsqueeze(0).expand(n_events, -1, -1)
            
            try:
                cov_matrix = torch.inverse(regularized_fisher)
            except:
                # Fall back to loop if batch inverse fails
                cov_matrix = []
                for i in range(n_events):
                    try:
                        cov_matrix.append(torch.inverse(regularized_fisher[i]))
                    except:
                        cov_matrix.append(torch.pinverse(regularized_fisher[i]))
                cov_matrix = torch.stack(cov_matrix)
            
            for i, params in enumerate(signal_event_params):
                energy_idx = self.fisher_info_params.index('energy')
                var_energy = cov_matrix[i][energy_idx, energy_idx]
                energy_resolution = torch.sqrt(torch.clamp_min(var_energy, 1e-10))  # Clamp to ensure non-negative
                if use_relative_energy:
                    energy_resolution = energy_resolution/params['energy'].to(self.device)
                resolution_per_event.append(energy_resolution)
            resolution_per_event = torch.stack(resolution_per_event)
            finite_mask = torch.isfinite(resolution_per_event) & (resolution_per_event > 1e-12)
            if finite_mask.any():
                safe_res = torch.where(
                    finite_mask,
                    torch.clamp_min(resolution_per_event, 1e-12),
                    torch.full_like(resolution_per_event, 1e6),
                )
                total_resolution = 1 / torch.sqrt(torch.sum(1 / (safe_res ** 2)))
            else:
                total_resolution = torch.tensor(1.0, device=self.device, requires_grad=True)
            return {'energy_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': signal_event_params}

class ResolutionLoss(FisherInfoLoss):
    def __init__(self, device=None, print_loss=False, random_seed=None, fisher_info_params=['energy', 'azimuth', 'zenith'], resolution_type='angular'):
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
        resolution_type : str
            Type of resolution to compute ('angular' or 'energy').
        """
        super().__init__(device, print_loss, random_seed, fisher_info_params)
        
        self.resolution_type = resolution_type # 'angular' or 'energy'
    
    def compute_fisher_info_per_event(self, points_3d, event_params, surrogate_func, llr_net=None, signal_noise_scale=None, llr_iterations=1, add_relative_pos=False, skip_zero_response=True, llr_event_labels=None):
        param_dims = []
        param_names_expanded = []
        for param_name in self.fisher_info_params:
            param_value = event_params[0].get(param_name)
            if param_value.dim() == 0 or (param_value.dim() == 1 and param_value.shape[0] == 1):
                # Scalar parameter
                param_dims.append(1)
                param_names_expanded.append(param_name)
            else:
                # Multi-dimensional parameter (e.g., 3D vector)
                dim_size = param_value.numel()
                param_dims.append(dim_size)
                for i in range(dim_size):
                    param_names_expanded.append(f"{param_name}_{i}")
        
        total_dims = sum(param_dims)
        total_fisher_info = []
        for params in event_params:
            fisher_matrix = torch.zeros(total_dims, total_dims, device=self.device)
            for _ in range(llr_iterations):
                fisher_matrix += compute_fisher_info_single(self.fisher_info_params, points_3d, params, surrogate_func, llr_net, signal_noise_scale, add_relative_pos=add_relative_pos, skip_zero_response=skip_zero_response, event_param_names=llr_event_labels)
            total_fisher_info.append(fisher_matrix/llr_iterations)
        total_fisher_info = torch.stack(total_fisher_info)
        return total_fisher_info
    
    def __call__(self, geom_dict, **kwargs):
        """
        Compute the total Fisher information loss = 1/det(FisherInfo).
        
        Parameters:
        -----------
        points_3d : torch.Tensor
            The 3D points to evaluate the loss at.
        event_params : list of dict
            List of dictionaries containing event parameters.
        surrogate_func : callable
            Function that computes light yield from event parameters.
            
        Returns:
        --------
        torch.Tensor
            The total Fisher information loss value.
        """
        points_3d = geom_dict.get('points_3d', None)
        event_params = kwargs.get('signal_event_params', None)
        surrogate_func = kwargs.get('signal_surrogate_func', None)
        signal_sampler = kwargs.get('signal_sampler', None)
        num_events = kwargs.get('num_events', 100)
        llr_net = kwargs.get('fisher_info_llr_net', None)
        llr_iterations = kwargs.get('fisher_info_llr_iterations', 100)
        llr_event_labels = kwargs.get('llr_event_labels', None)
        signal_noise_scale = kwargs.get('signal_noise_scale', None)
        add_relative_pos = kwargs.get('add_relative_pos', False)
        max_energy_resolution = kwargs.get('max_energy_resolution', 1.0)
        max_angular_resolution = kwargs.get('max_angular_resolution', torch.pi)
        precalculated_fisher_info = kwargs.get('precomputed_fisher_info_per_event', None)
        use_relative_energy = kwargs.get('use_relative_energy', False)
        skip_zero_response = kwargs.get('skip_zero_response', True)
        if event_params is None and signal_sampler is not None:
            event_params = signal_sampler.sample_events(num_events)
        resolution_per_event = []
        if precalculated_fisher_info is not None:
            total_fisher_info = precalculated_fisher_info
        else:
            total_fisher_info = self.compute_fisher_info_per_event(points_3d, event_params, surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos, skip_zero_response=skip_zero_response, llr_event_labels=llr_event_labels)
        resolution_per_event = []
        
        if self.resolution_type == 'angular':
            # Check if using direction or zenith/azimuth
            if 'direction' in self.fisher_info_params:
                # Use directional resolution with tangent space projection
                direction_idx = self.fisher_info_params.index('direction')
                # Calculate start index for direction in the expanded Fisher matrix
                start_idx = 0
                for j, param_name in enumerate(self.fisher_info_params):
                    if j < direction_idx:
                        param_val = event_params[0].get(param_name)
                        if param_val.dim() == 0 or (param_val.dim() == 1 and param_val.shape[0] == 1):
                            start_idx += 1
                        else:
                            start_idx += param_val.numel()
                
                # Batch process all events at once
                # Extract all Fisher submatrices: (N, 3, 3)
                F3_batch = total_fisher_info[:, start_idx:start_idx+3, start_idx:start_idx+3]
                
                # Extract all directions: (N, 3)
                directions_batch = torch.stack([params['direction'].to(self.device) for params in event_params])
                
                # Compute all resolutions at once: (N,)
                resolution_per_event = directional_resolution(F3_batch, directions_batch)
            else:
                # Use traditional zenith/azimuth resolution - need covariance matrix
                # Vectorized batch inverse
                n_events = len(event_params)
                regularized_fisher = total_fisher_info + 1e-5 * torch.eye(
                    total_fisher_info.shape[1], device=self.device
                ).unsqueeze(0).expand(n_events, -1, -1)  # Increased regularization for stability
                
                try:
                    cov_matrix = torch.inverse(regularized_fisher)
                except:
                    # Fall back to loop if batch inverse fails
                    cov_matrix = []
                    for i in range(n_events):
                        try:
                            cov_matrix.append(torch.inverse(regularized_fisher[i]))
                        except:
                            cov_matrix.append(torch.pinverse(regularized_fisher[i]))
                    cov_matrix = torch.stack(cov_matrix)
                
                for i, params in enumerate(event_params):
                    zenith = params['zenith']
                    azimuth_idx = self.fisher_info_params.index('azimuth')
                    zenith_idx = self.fisher_info_params.index('zenith')
                    # Angular resolution: sqrt(var_azimuth + var_zenith)
                    var_azimuth = cov_matrix[i][azimuth_idx, azimuth_idx]
                    var_zenith = cov_matrix[i][zenith_idx, zenith_idx]
                    covar_zenith_azimuth = cov_matrix[i][zenith_idx, azimuth_idx]
                    angular_variance = var_zenith + torch.sin(zenith)*var_azimuth + 2*torch.sin(zenith)*torch.cos(zenith)*covar_zenith_azimuth
                    angular_resolution_rad = torch.sqrt(torch.clamp_min(angular_variance, 1e-10))  # Clamp to avoid sqrt of negative/zero
                    resolution_per_event.append(angular_resolution_rad)
            resolution_per_event = torch.stack(resolution_per_event)
            finite_mask = torch.isfinite(resolution_per_event) & (resolution_per_event > 1e-12)
            if finite_mask.any():
                safe_res = torch.where(
                    finite_mask,
                    torch.clamp_min(resolution_per_event, 1e-12),
                    torch.full_like(resolution_per_event, 1e6),
                )
                total_resolution = 1 / torch.sqrt(torch.sum(1 / safe_res**2))
            else:
                total_resolution = torch.tensor(1.0, device=self.device, requires_grad=True)
            return {'angular_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': event_params}
        elif self.resolution_type == 'energy':
            # Compute covariance matrix for energy resolution
            # Vectorized batch inverse
            n_events = len(event_params)
            regularized_fisher = total_fisher_info + 1e-5 * torch.eye(
                total_fisher_info.shape[1], device=self.device
            ).unsqueeze(0).expand(n_events, -1, -1)  # Increased regularization for stability
            
            try:
                cov_matrix = torch.inverse(regularized_fisher)
            except:
                # Fall back to loop if batch inverse fails
                cov_matrix = []
                for i in range(n_events):
                    try:
                        cov_matrix.append(torch.inverse(regularized_fisher[i]))
                    except:
                        cov_matrix.append(torch.pinverse(regularized_fisher[i]))
                cov_matrix = torch.stack(cov_matrix)
            
            for i, params in enumerate(event_params):
                energy_idx = self.fisher_info_params.index('energy')
                var_energy = cov_matrix[i][energy_idx, energy_idx]
                var_energy = torch.nn.functional.softplus(var_energy, beta=1000)
                var_energy = torch.clamp_min(var_energy, 1e-10)  # Ensure positive variance
                # finite_inds = torch.isfinite(var_energy)
                # print(var_energy)
                energy_resolution = torch.sqrt(var_energy)
                if use_relative_energy:
                    energy_resolution = energy_resolution/params['energy']
                resolution_per_event.append(energy_resolution)
            resolution_per_event = torch.stack(resolution_per_event)
            finite_mask = torch.isfinite(resolution_per_event) & (resolution_per_event > 1e-12)
            if finite_mask.any():
                safe_res = torch.where(
                    finite_mask,
                    torch.clamp_min(resolution_per_event, 1e-12),
                    torch.full_like(resolution_per_event, 1e6),
                )
                total_resolution = 1 / torch.sqrt(torch.sum(1 / safe_res**2))
            else:
                total_resolution = torch.tensor(1.0, device=self.device, requires_grad=True)

            return {'energy_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': event_params}
