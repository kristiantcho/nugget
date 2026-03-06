from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F
import numpy as np
import time
import random
import pickle
import gc
from torch.func import jacrev, jvp, vmap, linearize


def _llr_mask_from_true_ly(true_ly, *, threshold=0.5, sharpness=12.0):
    return torch.sigmoid((true_ly - threshold) * sharpness) + 1e-6


def _llr_out_single_point_all_iters(
    pt_3,
    *,
    theta_vals,
    fisher_info_params,
    fixed_params,
    llr_net,
    surrogate_func,
    llr_iterations,
    signal_noise_scale,
    skip_zero_response,
    event_param_names,
):
    params = {fisher_info_params[i]: theta_vals[i] for i in range(len(fisher_info_params))}
    params.update(fixed_params)
    features, ly = llr_net.prepare_data_from_raw(
        pt_3.unsqueeze(0),
        params,
        surrogate_func,
        noise_scale=signal_noise_scale,
        output_true_light_yield=True,
        event_labels=fisher_info_params if event_param_names is None else event_param_names,
        num_samples=llr_iterations,
    )
    llr_out = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5).reshape(-1)  # (L,)
    if skip_zero_response:
        llr_out = llr_out * _llr_mask_from_true_ly(ly.reshape(-1))
    return llr_out


def _fisher_one_point_jacrev(
    pt_3,
    *,
    theta_tuple,
    fisher_info_params,
    fixed_params,
    llr_net,
    surrogate_func,
    llr_iterations,
    signal_noise_scale,
    skip_zero_response,
    event_param_names,
    jacrev_chunk_size,
):
    def _theta_only_fn(*theta_vals):
        return _llr_out_single_point_all_iters(
            pt_3,
            theta_vals=theta_vals,
            fisher_info_params=fisher_info_params,
            fixed_params=fixed_params,
            llr_net=llr_net,
            surrogate_func=surrogate_func,
            llr_iterations=llr_iterations,
            signal_noise_scale=signal_noise_scale,
            skip_zero_response=skip_zero_response,
            event_param_names=event_param_names,
        )

    J_tuple = jacrev(
        _theta_only_fn,
        argnums=tuple(range(len(fisher_info_params))),
        chunk_size=jacrev_chunk_size,
    )(*theta_tuple)

    J = torch.cat([j.reshape(llr_iterations, -1) for j in J_tuple], dim=1).detach()
    del J_tuple
    F = torch.einsum('li,lj->ij', J, J) / llr_iterations
    del J
    return F


def _unflatten_theta(theta_flat, *, fisher_info_params, theta_shapes, theta_numels, fixed_params):
    params = {}
    idx = 0
    for name, shape, numel in zip(fisher_info_params, theta_shapes, theta_numels):
        params[name] = theta_flat[idx:idx + numel].reshape(shape)
        idx += numel
    params.update(fixed_params)
    return params


def _sample_detector_responses_batched(
    pts_3,
    *,
    surrogate_func,
    params_for_sampling,
    llr_iterations,
    signal_noise_scale,
    llr_net,
    device,
):
    """Mimic LLRnet.prepare_data_from_raw(num_samples>1) response generation.

    Returns:
      responses_processed: (L, B) after noise + optional log scaling
      light_yields_true:   (L, B) pre-noise (used for masking)
    """
    pts_3 = pts_3.float().to(device)
    B = pts_3.shape[0]
    responses_list = []
    ly_list = []
    with torch.no_grad():
        for _ in range(llr_iterations):
            resp = surrogate_func(opt_point=pts_3, event_params=params_for_sampling)
            if isinstance(resp, np.ndarray):
                resp = torch.tensor(resp, device=device, dtype=torch.float32)
            elif not isinstance(resp, torch.Tensor):
                resp = torch.tensor(resp, device=device, dtype=torch.float32)
            resp = resp.float().to(device).reshape(-1)
            if resp.numel() != B:
                if resp.numel() == 1:
                    resp = resp.expand(B)
                else:
                    raise ValueError(
                        f"surrogate_func returned {resp.numel()} responses, expected {B}. "
                        f"pts_3 shape={tuple(pts_3.shape)}"
                    )

            ly_list.append(resp.clone())

            if signal_noise_scale is not None and signal_noise_scale > 0:
                resp = resp + torch.randn_like(resp) * signal_noise_scale
            if bool(getattr(llr_net, 'log_scale_ly', False)):
                resp = torch.log10(torch.abs(resp) + 1e-10)
            responses_list.append(resp)

    responses = torch.stack(responses_list, dim=0)
    light_yields_true = torch.stack(ly_list, dim=0)
    return responses, light_yields_true


def _build_features_from_cached_responses(
    pts_3,
    *,
    theta_flat,
    cached_responses_processed,
    llr_net,
    fisher_info_params,
    event_param_names,
    theta_shapes,
    theta_numels,
    fixed_params,
    device,
):
    """Rebuild features deterministically, matching prepare_data_from_raw(num_samples>1)."""
    params = _unflatten_theta(
        theta_flat,
        fisher_info_params=fisher_info_params,
        theta_shapes=theta_shapes,
        theta_numels=theta_numels,
        fixed_params=fixed_params,
    )

    pts_3 = pts_3.float().to(device)
    if pts_3.dim() == 1:
        pts_3 = pts_3.unsqueeze(0)
    B = pts_3.shape[0]
    L = cached_responses_processed.shape[0]

    if bool(getattr(llr_net, 'norm_pos', False)):
        domain_size = getattr(llr_net, 'domain_size', None)
        if domain_size is None:
            raise AttributeError("llr_net.norm_pos=True but llr_net.domain_size is missing")
        norm_points = pts_3 / (domain_size / 2)
    else:
        norm_points = pts_3

    relative_pos = None
    if bool(getattr(llr_net, 'add_relative_pos', False)) and ('position' in params):
        event_pos = params['position']
        if isinstance(event_pos, np.ndarray):
            event_pos = torch.tensor(event_pos, device=device, dtype=torch.float32)
        elif not isinstance(event_pos, torch.Tensor):
            event_pos = torch.tensor(event_pos, device=device, dtype=torch.float32)
        event_pos = event_pos.float().to(device)
        if event_pos.dim() == 1:
            event_pos = event_pos.unsqueeze(0)
        relative_pos = pts_3 - event_pos

    dist_perp = None
    if bool(getattr(llr_net, 'add_distance_from_beam', False)) and ("direction" in params) and ("position" in params):
        track_dir = params["direction"]
        if isinstance(track_dir, np.ndarray):
            track_dir = torch.tensor(track_dir, device=device, dtype=torch.float32)
        elif not isinstance(track_dir, torch.Tensor):
            track_dir = torch.tensor(track_dir, device=device, dtype=torch.float32)
        track_dir = track_dir.float().to(device)
        if track_dir.dim() == 1:
            track_dir = track_dir.unsqueeze(0)

        event_pos = params["position"]
        if isinstance(event_pos, np.ndarray):
            event_pos = torch.tensor(event_pos, device=device, dtype=torch.float32)
        elif not isinstance(event_pos, torch.Tensor):
            event_pos = torch.tensor(event_pos, device=device, dtype=torch.float32)
        event_pos = event_pos.float().to(device)
        if event_pos.dim() == 1:
            event_pos = event_pos.unsqueeze(0)

        _, dist_perp = llr_net.compute_distance_from_beam(pts_3, event_pos, track_dir)

    event_labels = fisher_info_params if event_param_names is None else event_param_names
    event_param_features = []
    for key in event_labels:
        if key not in params:
            continue
        feature = params[key]
        if isinstance(feature, np.ndarray):
            feature = torch.tensor(feature, device=device, dtype=torch.float32)
        elif not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, device=device, dtype=torch.float32)
        feature = feature.float().to(device)
        if bool(getattr(llr_net, 'log_scale_energy', False)) and key == 'energy':
            feature = torch.log10(feature + 1e-10)
        if bool(getattr(llr_net, 'norm_pos', False)) and key == 'position':
            domain_size = getattr(llr_net, 'domain_size', None)
            if domain_size is None:
                raise AttributeError("llr_net.norm_pos=True but llr_net.domain_size is missing")
            feature = feature / (domain_size / 2)
        event_param_features.append(feature.flatten())

    if event_param_features:
        event_params_cat = torch.cat(event_param_features, dim=0)
    else:
        event_params_cat = torch.tensor([], device=device, dtype=pts_3.dtype)

    point_event_features_list = [norm_points]
    if relative_pos is not None:
        point_event_features_list.append(relative_pos)
    if dist_perp is not None:
        point_event_features_list.append(dist_perp)
    if event_params_cat.numel() > 0:
        event_params_replicated = event_params_cat.unsqueeze(0).expand(B, -1)
        point_event_features_list.append(event_params_replicated)

    point_event_features = torch.cat(point_event_features_list, dim=1)
    point_event_features_batched = point_event_features.unsqueeze(0).expand(L, -1, -1)
    detector_responses_expanded = cached_responses_processed.unsqueeze(2)
    features_batched = torch.cat([point_event_features_batched, detector_responses_expanded], dim=2)
    return features_batched.reshape(L * B, -1)


def _fisher_points_all_iters_jvp(
    pts_3,
    *,
    llr_net,
    surrogate_func,
    fisher_info_params,
    event_param_names,
    fixed_params,
    theta0_flat,
    theta_shapes,
    theta_numels,
    total_dims,
    llr_iterations,
    signal_noise_scale,
    skip_zero_response,
    basis_chunk_size,
    device,
):
    can_cache_responses = not bool(getattr(llr_net, 'use_patd', False))

    if can_cache_responses:
        params0 = _unflatten_theta(
            theta0_flat,
            fisher_info_params=fisher_info_params,
            theta_shapes=theta_shapes,
            theta_numels=theta_numels,
            fixed_params=fixed_params,
        )
        cached_responses, cached_ly_true = _sample_detector_responses_batched(
            pts_3,
            surrogate_func=surrogate_func,
            params_for_sampling=params0,
            llr_iterations=llr_iterations,
            signal_noise_scale=signal_noise_scale,
            llr_net=llr_net,
            device=device,
        )
        cached_responses = cached_responses.detach()
        cached_ly_true = cached_ly_true.detach()

        def _theta_only_fn(theta_flat):
            B = pts_3.shape[0]
            features = _build_features_from_cached_responses(
                pts_3,
                theta_flat=theta_flat,
                cached_responses_processed=cached_responses,
                llr_net=llr_net,
                fisher_info_params=fisher_info_params,
                event_param_names=event_param_names,
                theta_shapes=theta_shapes,
                theta_numels=theta_numels,
                fixed_params=fixed_params,
                device=device,
            )
            llr_out = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5).reshape(llr_iterations, B)
            if skip_zero_response:
                llr_out = llr_out * _llr_mask_from_true_ly(cached_ly_true)
            return llr_out.transpose(0, 1).contiguous()  # (B, L)
    else:
        def _theta_only_fn(theta_flat):
            params = _unflatten_theta(
                theta_flat,
                fisher_info_params=fisher_info_params,
                theta_shapes=theta_shapes,
                theta_numels=theta_numels,
                fixed_params=fixed_params,
            )
            features, ly = llr_net.prepare_data_from_raw(
                pts_3,
                params,
                surrogate_func,
                noise_scale=signal_noise_scale,
                output_true_light_yield=True,
                event_labels=fisher_info_params if event_param_names is None else event_param_names,
                num_samples=llr_iterations,
            )
            B = pts_3.shape[0]
            llr_out = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5).reshape(llr_iterations, B)
            if skip_zero_response:
                llr_out = llr_out * _llr_mask_from_true_ly(ly.reshape(llr_iterations, B))
            return llr_out.transpose(0, 1).contiguous()

    y0, jvp_fn = linearize(_theta_only_fn, theta0_flat)
    del y0

    cols_parts = []
    for d_start in range(0, total_dims, basis_chunk_size):
        d_end = min(d_start + basis_chunk_size, total_dims)
        k = d_end - d_start

        basis_chunk = torch.zeros(k, total_dims, device=device, dtype=theta0_flat.dtype)
        rows = torch.arange(k, device=device)
        cols_idx = torch.arange(d_start, d_end, device=device)
        basis_chunk[rows, cols_idx] = 1

        cols_chunk = vmap(jvp_fn)(basis_chunk)  # (k, B, L)
        cols_parts.append(cols_chunk)
        del basis_chunk, cols_chunk

    cols = torch.cat(cols_parts, dim=0)
    del cols_parts
    J = cols.permute(1, 2, 0).contiguous()  # (B, L, D)
    del cols
    F = (torch.einsum('bld,ble->bde', J, J) / llr_iterations).detach()
    del J

    if can_cache_responses:
        del cached_responses, cached_ly_true
    del jvp_fn
    return F


def _compute_fisher_llr_over_points(
    *,
    point,
    n_points,
    total_dims,
    fisher_info_params,
    event_params,
    fixed_params,
    llr_net,
    surrogate_func,
    llr_iterations,
    signal_noise_scale,
    skip_zero_response,
    event_param_names,
    llr_autodiff_mode,
    jacrev_chunk_size,
    grad_chunk_size,
    point_chunk_size,
    string_xy,
    sum_over_points,
    device,
):
    theta_tuple = tuple(event_params[p].detach().to(device) for p in fisher_info_params)
    pt_chunk = point_chunk_size if point_chunk_size is not None else n_points

    fisher_sum = None
    fisher_per_point = None
    if string_xy is None and sum_over_points:
        fisher_sum = torch.zeros(total_dims, total_dims, device=device)
    else:
        fisher_per_point = torch.zeros(n_points, total_dims, total_dims, device=device)

    llr_autodiff_mode = (llr_autodiff_mode or 'jacrev').lower()
    if llr_autodiff_mode == 'jacrev':
        def _one(pt_3):
            return _fisher_one_point_jacrev(
                pt_3,
                theta_tuple=theta_tuple,
                fisher_info_params=fisher_info_params,
                fixed_params=fixed_params,
                llr_net=llr_net,
                surrogate_func=surrogate_func,
                llr_iterations=llr_iterations,
                signal_noise_scale=signal_noise_scale,
                skip_zero_response=skip_zero_response,
                event_param_names=event_param_names,
                jacrev_chunk_size=jacrev_chunk_size,
            )

        for p_start in range(0, n_points, pt_chunk):
            p_end = min(p_start + pt_chunk, n_points)
            pts = point[p_start:p_end]
            try:
                fisher_chunk = vmap(_one, randomness='different')(pts)
            except Exception:
                fisher_chunk = torch.stack([_one(pts[i]) for i in range(pts.shape[0])], dim=0)
            fisher_chunk = fisher_chunk.detach()
            if fisher_sum is not None:
                fisher_sum.add_(fisher_chunk.sum(dim=0))
            else:
                fisher_per_point[p_start:p_end] = fisher_chunk
            del fisher_chunk, pts
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return fisher_sum, fisher_per_point

    if llr_autodiff_mode != 'jvp':
        raise ValueError(f"Unsupported llr_autodiff_mode={llr_autodiff_mode!r}; expected 'jacrev' or 'jvp'.")

    theta_shapes = [event_params[p].detach().to(device).shape for p in fisher_info_params]
    theta_numels = [int(event_params[p].detach().to(device).numel()) for p in fisher_info_params]
    theta0_flat = torch.cat([event_params[p].detach().to(device).reshape(-1) for p in fisher_info_params], dim=0)
    basis_chunk_size = grad_chunk_size if grad_chunk_size is not None else total_dims

    for p_start in range(0, n_points, pt_chunk):
        p_end = min(p_start + pt_chunk, n_points)
        pts = point[p_start:p_end]
        fisher_chunk = _fisher_points_all_iters_jvp(
            pts,
            llr_net=llr_net,
            surrogate_func=surrogate_func,
            fisher_info_params=fisher_info_params,
            event_param_names=event_param_names,
            fixed_params=fixed_params,
            theta0_flat=theta0_flat,
            theta_shapes=theta_shapes,
            theta_numels=theta_numels,
            total_dims=total_dims,
            llr_iterations=llr_iterations,
            signal_noise_scale=signal_noise_scale,
            skip_zero_response=skip_zero_response,
            basis_chunk_size=basis_chunk_size,
            device=device,
        ).detach()
        if fisher_sum is not None:
            fisher_sum.add_(fisher_chunk.sum(dim=0))
        else:
            fisher_per_point[p_start:p_end] = fisher_chunk

        del fisher_chunk, pts
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return fisher_sum, fisher_per_point

def compute_fisher_info_single_batched(fisher_info_params, point, event_params_list, surrogate_func, llr_net=None, signal_noise_scale=None, add_relative_pos=False, skip_zero_response=True, event_param_names=None, llr_iterations=1, string_xy=None, sum_over_points=True):
    """
    Compute the Fisher information matrix with batched processing of points, events, and LLR iterations.
    
    This function processes all combinations of (points × events × llr_iterations) simultaneously
    with a single LLR network forward pass and batched gradient computation for maximum efficiency.
    
    Parameters:
    -----------
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
    point : torch.Tensor
        The 3D point(s) to evaluate the Fisher information at. Can be:
        - Single point: shape (3,)
        - Multiple points: shape (n_points, 3)
    event_params_list : list of dict
        List of dictionaries containing event parameters.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    llr_net : optional
        Neural network for computing log-likelihood ratios.
    signal_noise_scale : float or None
        Scale for adding noise to signal.
    add_relative_pos : bool
        Whether to add relative position features.
    skip_zero_response : bool
        Whether to skip points with zero response using soft masking.
    event_param_names : list or None
        Names of event parameters to use for LLR network.
    llr_iterations : int
        Number of LLR iterations to compute and average over for each event.
    string_xy : list of torch.Tensor or None
        Optional list of 2D string coordinates [(x1, y1), (x2, y2), ...].
        If provided, points are grouped by strings and Fisher matrices are summed per string.
    sum_over_points : bool
        If True and string_xy is None, sum Fisher matrices over all points.
        If False, return separate Fisher matrices for each point.
        
    Returns:
    --------
    torch.Tensor
        Fisher information matrices with shape depending on parameters:
        - If string_xy provided: (n_strings, n_events, total_dims, total_dims)
        - If string_xy is None and sum_over_points=True: (n_events, total_dims, total_dims)
        - If string_xy is None and sum_over_points=False: (n_points, n_events, total_dims, total_dims)
        where total_dims is the sum of dimensions of all parameters.
        Each matrix is averaged over llr_iterations realizations.
    """
    if len(event_params_list) == 0:
        return torch.tensor([])
    
    if llr_iterations <= 0:
        llr_iterations = 1
    
    # Determine device
    if isinstance(point, torch.Tensor):
        device = point.device
    else:
        device = torch.device('cpu')
    
    # Handle point dimensions and string grouping
    if point.dim() == 1:
        point = point.unsqueeze(0)  # Convert to (1, 3)
    n_points = point.shape[0]
    
    # If string_xy is provided, create mapping from points to strings
    point_to_string = None
    n_strings = None
    if string_xy is not None:
        n_strings = len(string_xy)
        point_to_string = torch.zeros(n_points, dtype=torch.long, device=device)
        for s_idx in range(n_strings):
            mask = (point[:, 0] == string_xy[s_idx][0]) & (point[:, 1] == string_xy[s_idx][1])
            point_to_string[mask] = s_idx
    
    n_events = len(event_params_list)
    
    # Determine dimensionality of each parameter
    param_dims = []
    param_names_expanded = []
    for param_name in fisher_info_params:
        param_value = event_params_list[0].get(param_name)
        if param_value.dim() == 0 or (param_value.dim() == 1 and param_value.shape[0] == 1):
            param_dims.append(1)
            param_names_expanded.append(param_name)
        else:
            dim_size = param_value.numel()
            param_dims.append(dim_size)
            for i in range(dim_size):
                param_names_expanded.append(f"{param_name}_{i}")
    
    total_dims = sum(param_dims)
    
    # Stack event parameters into batched tensors, replicated for points and llr_iterations
    # Shape will be (n_points * n_events * llr_iterations, ...)
    batched_grad_params = {}
    batched_fixed_params = {}
    
    for param_name in fisher_info_params:
        param_values = []
        for _ in range(n_points):
            for event_params in event_params_list:
                param_value = event_params.get(param_name)
                # Replicate this event's parameter llr_iterations times
                for _ in range(llr_iterations):
                    param_values.append(param_value)
        # Stack and enable gradients
        stacked = torch.stack(param_values).to(device)
        batched_grad_params[param_name] = stacked.clone().detach().requires_grad_(True)
    
    # Handle non-gradient parameters
    all_param_names = set()
    for event_params in event_params_list:
        all_param_names.update(event_params.keys())
    
    for param_name in all_param_names:
        if param_name not in fisher_info_params:
            param_values = []
            for _ in range(n_points):
                for event_params in event_params_list:
                    param_value = event_params.get(param_name)
                    for _ in range(llr_iterations):
                        param_values.append(param_value)
            batched_fixed_params[param_name] = torch.stack(param_values).to(device)
    
    # Compute light yields for all points × events × iterations
    n_total = n_points * n_events * llr_iterations  # Total number of samples
    start_time = time.time()
    if llr_net is None:
        # Generate multiple samples with different noise for each point×event×iteration
        light_yield_means = []
        for point_idx in range(n_points):
            for event_idx in range(n_events):
                for iter_idx in range(llr_iterations):
                    global_idx = point_idx * n_events * llr_iterations + event_idx * llr_iterations + iter_idx
                    event_params_dict = {k: v[global_idx] for k, v in {**batched_grad_params, **batched_fixed_params}.items()}
                    ly = surrogate_func(opt_point=point[point_idx], event_params=event_params_dict)
                    if signal_noise_scale is not None:
                        ly = torch.normal(0, signal_noise_scale) * ly + ly
                    light_yield_means.append(ly)
        light_yield_means = torch.stack(light_yield_means)  # Shape: (n_total,)
    else:
        # Use fully batched feature generation: process all points at once per event
        # This is the most efficient approach as prepare_data_from_raw can handle batched points
        features_list = []
        ly_list = []
        
        # Loop over events and process all points simultaneously for each event
        for event_idx in range(n_events):
            # Get event parameters for this event (first occurrence in batched params)
            event_start_idx = event_idx * llr_iterations
            event_params_dict = {k: v[event_start_idx] for k, v in {**batched_grad_params, **batched_fixed_params}.items()}
            
            # Generate features for ALL points for this event at once with llr_iterations samples each
            # point shape: (n_points, 3)
            # This will return features for all points with llr_iterations noise realizations each
            features_batched, ly_batched = llr_net.prepare_data_from_raw(
                point, event_params_dict, surrogate_func,
                noise_scale=signal_noise_scale,
                output_true_light_yield=True,
                event_labels=fisher_info_params if event_param_names is None else event_param_names,
                num_samples=llr_iterations
            )
            # features_batched shape: (n_points * llr_iterations, feature_dim) 
            # ly_batched shape: (n_points * llr_iterations,)
            
            features_list.append(features_batched)
            ly_list.append(ly_batched)
        
        # Stack all features and light yields
        all_features = torch.cat(features_list, dim=0)  # Shape: (n_total, feature_dim)
        all_ly = torch.cat(ly_list, dim=0)  # Shape: (n_total,)
        
        # Compute LLR for all points×events×iterations in a single forward pass
        light_yield_means = llr_net.predict_log_likelihood_ratio(all_features, epsilon=1e-5)
        # light_yield_means shape: (n_total,)
        
        # Apply masking to entire batch
        if skip_zero_response:
            mask = torch.sigmoid((all_ly - 0.5) * 12.0) + 1e-6
            light_yield_means = light_yield_means * mask
    
    # Compute gradients for each parameter across all points×events×iterations in a single batched operation
    # print(f"Computed {n_total} LLR evaluations in {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    param_gradients = []
    for param_name in fisher_info_params:
        if param_name in batched_grad_params:
            param_tensor = batched_grad_params[param_name]
            param_size = param_tensor[0].numel()
            
            # Compute gradient for all samples at once using grad_outputs
            # This computes ∂(light_yield_means[i])/∂(param_tensor[i]) for all i simultaneously
            param_grad = torch.autograd.grad(
                outputs=light_yield_means,
                inputs=param_tensor,
                grad_outputs=torch.ones_like(light_yield_means),
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            
            if param_grad is not None:
                # Reshape to (n_total, param_size)
                param_grad_batch = param_grad.view(n_total, param_size)
            else:
                # If gradient is None, create zero gradient
                param_grad_batch = torch.zeros(n_total, param_size, device=device)
            
            param_gradients.append(param_grad_batch)
    
    # Compute Fisher Information matrices for all samples
    # print(f"Computed gradients for {len(param_gradients)} parameters in {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    if len(param_gradients) == len(fisher_info_params):
        # Stack all parameter gradients: (n_total, total_dims)
        all_gradients = torch.cat(param_gradients, dim=1)
        
        # Vectorized computation using batched outer product
        grad_outer = torch.bmm(
            all_gradients.unsqueeze(-1),  # (n_total, total_dims, 1)
            all_gradients.unsqueeze(-2)   # (n_total, 1, total_dims)
        )  # Result: (n_total, total_dims, total_dims)
        
        if llr_net is None:
            # Divide by light yields
            fisher_matrices_all = grad_outer / light_yield_means.view(n_total, 1, 1)
        else:
            fisher_matrices_all = grad_outer
        
        # Reshape to (n_points, n_events, llr_iterations, total_dims, total_dims)
        fisher_matrices_all = fisher_matrices_all.view(n_points, n_events, llr_iterations, total_dims, total_dims)
        
        # Average over llr_iterations: (n_points, n_events, total_dims, total_dims)
        fisher_matrices = fisher_matrices_all.mean(dim=2)
        
        # Handle string grouping or point summation
        if string_xy is not None:
            # Sum Fisher matrices by string: (n_strings, n_events, total_dims, total_dims)
            fisher_by_string = torch.zeros(n_strings, n_events, total_dims, total_dims, device=device)
            for s_idx in range(n_strings):
                string_mask = point_to_string == s_idx
                # Sum over points belonging to this string
                fisher_by_string[s_idx] = fisher_matrices[string_mask].sum(dim=0)
            # print(f"Computed Fisher matrices by string in {time.time() - start_time:.2f} seconds")
            return fisher_by_string
        elif sum_over_points:
            # Sum over all points: (n_events, total_dims, total_dims)
            return fisher_matrices.sum(dim=0)
        else:
            # Return per-point results: (n_points, n_events, total_dims, total_dims)
            return fisher_matrices
    else:
        # No valid gradients
        if string_xy is not None:
            return torch.zeros(n_strings, n_events, total_dims, total_dims, device=device)
        elif sum_over_points:
            return torch.zeros(n_events, total_dims, total_dims, device=device)
        else:
            return torch.zeros(n_points, n_events, total_dims, total_dims, device=device)


def directional_resolution(F3, n):
    """Calculate angular resolution from Fisher information matrix (PyTorch version).
    
    Projects 3D Fisher matrix onto tangent space perpendicular to track direction.
    
    Args:
        F3: Fisher information matrix of shape (3, 3) or (N, 3, 3) for batched computation
        n: Unit track direction vector of shape (3,) or (N, 3) for batched computation
        
    Returns:
        68% containment angular resolution in radians (scalar or (N,) tensor)
    """
    # Check if batched input
    is_batched = F3.dim() == 3
    
    if not is_batched:
        # Single input case - normalize and compute
        n = n / torch.norm(n)

        # --- Build tangent basis B (3x2) ---
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype, device=n.device)
        if abs(torch.dot(n, ref)) > 0.9:
            ref = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
        
        b1 = torch.cross(n, ref)
        b1 = b1 / torch.norm(b1)
        b2 = torch.cross(n, b1)
        b2 = b2 / torch.norm(b2)
        B = torch.stack([b1, b2], dim=1)  # 3x2

        # --- Project Fisher ---
        F2 = B.T @ F3 @ B   # 2x2 Fisher in tangent coords
        F2 = F2 + 1e-10 * torch.eye(2, device=F2.device, dtype=F2.dtype)  # Increased regularization for stability
        
        # --- Invert to get covariance ---
        try:
            Cov2 = torch.inverse(F2)
        except RuntimeError:
            Cov2 = torch.pinverse(F2)

        # --- Angular resolution (approx small-angle) ---
        eigvals = torch.linalg.eigvalsh(Cov2)
        eigvals = torch.nn.functional.softplus(eigvals, beta=5)
        eigvals = torch.clamp_min(eigvals, 1e-10)  # Ensure positive eigenvalues
        sigma_eff = torch.sqrt(torch.mean(eigvals) + 1e-10)  # Add epsilon for numerical stability
        r68 = 1.515 * sigma_eff
        
        return r68
    
    else:
        # Batched computation
        batch_size = F3.shape[0]
        device = F3.device
        dtype = F3.dtype
        
        # Normalize direction vectors (N, 3)
        n = n / torch.norm(n, dim=1, keepdim=True)
        
        # --- Build tangent basis B for all directions (N, 3, 2) ---
        # Reference vector
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand(batch_size, 3)
        
        # Check which directions are nearly parallel to ref (use different ref for those)
        dots = torch.abs(torch.sum(n * ref, dim=1))  # (N,)
        parallel_mask = dots > 0.9
        ref[parallel_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        
        # First tangent vector
        b1 = torch.cross(n, ref, dim=1)  # (N, 3)
        b1 = b1 / torch.norm(b1, dim=1, keepdim=True)
        
        # Second tangent vector
        b2 = torch.cross(n, b1, dim=1)  # (N, 3)
        b2 = b2 / torch.norm(b2, dim=1, keepdim=True)
        
        # Stack to form basis: (N, 3, 2)
        B = torch.stack([b1, b2], dim=2)
        
        # --- Project Fisher matrices: (N, 2, 2) ---
        # F2 = B.T @ F3 @ B for each batch element
        F2 = torch.bmm(torch.bmm(B.transpose(1, 2), F3), B)  # (N, 2, 2)
        
        # Add regularization
        F2 = F2 + 1e-5 * torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 2, 2)  # Increased regularization for stability
        
        # --- Invert to get covariance (N, 2, 2) ---
        try:
            Cov2 = torch.inverse(F2)
        except RuntimeError:
            # Fall back to loop with pinverse if needed
            Cov2 = []
            for i in range(batch_size):
                try:
                    Cov2.append(torch.inverse(F2[i]))
                except:
                    Cov2.append(torch.pinverse(F2[i]))
            Cov2 = torch.stack(Cov2)
        
        # --- Angular resolution for all events ---
        eigvals = torch.linalg.eigvalsh(Cov2)  # (N, 2)
        eigvals = torch.nn.functional.softplus(eigvals, beta=1000)
        eigvals = torch.clamp_min(eigvals, 1e-10)  # Ensure positive eigenvalues
        sigma_eff = torch.sqrt(torch.mean(eigvals, dim=1) + 1e-10)  # (N,) Add epsilon for numerical stability
        r68 = 1.515 * sigma_eff  # (N,)
        
        return r68


def compute_fisher_info_single_averaged(fisher_info_params, point, event_params, surrogate_func, llr_iterations=1, llr_net=None, signal_noise_scale=None, add_relative_pos=False, skip_zero_response=True, event_param_names=None, string_xy=None, sum_over_points=True, device=None, grad_chunk_size=None, jacrev_chunk_size=None, point_chunk_size=None, llr_autodiff_mode='jacrev'):
    """
    Compute the Fisher information matrix averaged over multiple LLR iterations.
    
    This function computes the LLR multiple times (with noise if specified) and averages
    the resulting Fisher matrices. More efficient than calling compute_fisher_info_single
    multiple times as it batches the gradient computation.
    
    Parameters:
    -----------
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
    point : torch.Tensor
        The 3D point(s) to evaluate the Fisher information at. Can be:
        - Single point: shape (3,)
        - Multiple points: shape (n_points, 3)
    event_params : dict
        Dictionary containing event parameters.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    llr_iterations : int
        Number of LLR iterations to compute and average over.
    llr_net : optional
        Neural network for computing log-likelihood ratios.
    signal_noise_scale : float or None
        Scale for adding noise to signal.
    add_relative_pos : bool
        Whether to add relative position features.
    skip_zero_response : bool
        Whether to skip points with zero response using soft masking.
    event_param_names : list or None
        Names of event parameters to use for LLR network.
    string_xy : list of torch.Tensor or None
        Optional list of 2D string coordinates [(x1, y1), (x2, y2), ...].
        If provided, points are grouped by strings and Fisher matrices are summed per string.
    sum_over_points : bool
        If True and string_xy is None, sum Fisher matrices over all points.
        If False, return separate Fisher matrices for each point.
    device : torch.device, str, or None
        Device to run on. Defaults to the device of `point`.
    grad_chunk_size : int or None
        Chunk size for outer-product accumulation over llr_iterations.
        Reduce to bound GPU memory. None (default) processes all iterations at once.
        Note: when llr_autodiff_mode='jvp', this value is instead used as the
        chunk size over parameter-basis directions (number of JVP directions
        processed at once). This bounds peak memory when D is large.
    jacrev_chunk_size : int or None
        Passed to jacrev's chunk_size. Controls how many Jacobian rows (backward
        passes) are computed at a time within the single forward graph.
        chunk_size=1 = one backward at a time (minimum memory, slower);
        None (default) = all rows simultaneously.
    llr_autodiff_mode : {'jacrev', 'jvp'}
        Controls the autodiff strategy in the LLR-network path.
        - 'jacrev' (default): reverse-mode Jacobian via jacrev (fast when output dim is small).
        - 'jvp': forward-mode via JVPs. Typically better when parameter dim (D) is small and
          output dim (L = llr_iterations) is larger.
    point_chunk_size : int or None
        Process points in batches of this size. Each batch gets its own jacrev
        call, so the graph size and basis-vector allocation scale with
        (point_chunk_size * llr_iterations) rather than (n_points * llr_iterations).
        This is the primary knob for OOM errors with many points. None (default)
        processes all points in a single call.
        
    Returns:
    --------
    torch.Tensor
        Fisher information matrices with shape depending on parameters:
        - If string_xy provided: (n_strings, total_dims, total_dims)
        - If string_xy is None and sum_over_points=True: (total_dims, total_dims)
        - If string_xy is None and sum_over_points=False: (n_points, total_dims, total_dims)
        where total_dims is the sum of dimensions of all parameters.
        Each matrix is averaged over llr_iterations realizations.
    """
    if llr_iterations <= 0:
        llr_iterations = 1

    # ------------------------------------------------------------------ setup
    # Derive device from point; the explicit `device` argument overrides only if provided.
    device = point.device if device is None else torch.device(device)
    if point.dim() == 1:
        point = point.unsqueeze(0)           # (1, 3)
    point = point.to(device)
    n_points = point.shape[0]

    # String → point mapping
    point_to_string = None
    n_strings = None
    if string_xy is not None:
        n_strings = len(string_xy)
        point_to_string = torch.zeros(n_points, dtype=torch.long, device=device)
        for s_idx in range(n_strings):
            sx = string_xy[s_idx][0].to(device) if isinstance(string_xy[s_idx][0], torch.Tensor) else torch.tensor(string_xy[s_idx][0], device=device)
            sy = string_xy[s_idx][1].to(device) if isinstance(string_xy[s_idx][1], torch.Tensor) else torch.tensor(string_xy[s_idx][1], device=device)
            mask = (point[:, 0] == sx) & (point[:, 1] == sy)
            point_to_string[mask] = s_idx

    # Parameter dimensionality
    param_dims = []
    for param_name in fisher_info_params:
        pv = event_params.get(param_name)
        param_dims.append(1 if (pv.dim() == 0 or (pv.dim() == 1 and pv.shape[0] == 1)) else pv.numel())
    total_dims = sum(param_dims)

    # Fixed (non-gradient) params: detached and on the correct device
    fixed_params = {k: v.detach().to(device) for k, v in event_params.items() if k not in fisher_info_params}

    # -----------------------------------------------------------------------
    if llr_net is None:
        # ---- surrogate-only path -----------------------------------------------
        # The surrogate directly returns the Poisson mean λ (no noise averaging
        # needed — llr_iterations is ignored per specification).
        # Create n_points INDEPENDENT parameter copies so that ly[i] has its own
        # gradient path: J[i] = ∂λ(point_i)/∂theta.
        batched_grad_params = {}
        for param_name in fisher_info_params:
            pv = event_params.get(param_name).to(device)
            # one copy per point
            copies = pv.unsqueeze(0).expand(n_points, *pv.shape).clone().detach().to(device).requires_grad_(True)
            batched_grad_params[param_name] = copies  # (n_points, *param_shape)

        # Fully-batched surrogate call; fall back to sequential if not supported.
        try:
            params_b = {p: batched_grad_params[p] for p in fisher_info_params}
            params_b.update(fixed_params)
            ly_all = surrogate_func(opt_point=point,
                                    event_params={k: (v[0] if k in fisher_info_params else v)
                                                  for k, v in params_b.items()})
            if not isinstance(ly_all, torch.Tensor) or ly_all.shape[0] != n_points:
                raise ValueError
        except Exception:
            ly_list = []
            for pt_idx in range(n_points):
                p_i = {p: batched_grad_params[p][pt_idx] for p in fisher_info_params}
                p_i.update(fixed_params)
                ly_list.append(surrogate_func(opt_point=point[pt_idx], event_params=p_i))
            ly_all = torch.stack(ly_list)   # (n_points,)

        # Per-point gradients: J[i] = ∂ly[i]/∂param_copy_i
        grad_parts = []
        for param_name in fisher_info_params:
            param_tensor = batched_grad_params[param_name]
            param_size   = param_tensor[0].numel()
            g = torch.autograd.grad(
                outputs=ly_all,
                inputs=param_tensor,
                grad_outputs=torch.ones_like(ly_all),
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True,
            )[0]
            grad_parts.append(g.view(n_points, param_size) if g is not None
                               else torch.zeros(n_points, param_size, device=device))

        J       = torch.cat(grad_parts, dim=1)   # (n_points, total_dims)
        ly_vals = ly_all.detach()

        # Fisher formula: F_i = (∂λ_i/∂θ)(∂λ_i/∂θ)ᵀ / λ_i
        grad_outer = torch.bmm(J.unsqueeze(-1), J.unsqueeze(-2))  # (n_points, D, D)
        grad_outer = grad_outer / ly_vals.clamp(min=1e-10).view(n_points, 1, 1)
        fisher_per_point = grad_outer   # (n_points, total_dims, total_dims)

    else:
        # ---- LLR-network path -----------------------------------------------
        # Compute per-point Jacobians (output length = llr_iterations) and vmap
        # across points. This avoids building a huge Jacobian of size (B*L, D)
        # from a single (B*L)-vector output.

        # Align computation device with llr_net to avoid cross-device errors.
        llr_device = device
        if hasattr(llr_net, 'device'):
            llr_device = llr_net.device
            if not isinstance(llr_device, torch.device):
                llr_device = torch.device(llr_device)
        if llr_device != device:
            device = llr_device
            point = point.to(device)
            fixed_params = {k: v.detach().to(device) for k, v in event_params.items() if k not in fisher_info_params}

        fisher_sum, fisher_per_point = _compute_fisher_llr_over_points(
            point=point,
            n_points=n_points,
            total_dims=total_dims,
            fisher_info_params=fisher_info_params,
            event_params=event_params,
            fixed_params=fixed_params,
            llr_net=llr_net,
            surrogate_func=surrogate_func,
            llr_iterations=llr_iterations,
            signal_noise_scale=signal_noise_scale,
            skip_zero_response=skip_zero_response,
            event_param_names=event_param_names,
            llr_autodiff_mode=llr_autodiff_mode,
            jacrev_chunk_size=jacrev_chunk_size,
            grad_chunk_size=grad_chunk_size,
            point_chunk_size=point_chunk_size,
            string_xy=string_xy,
            sum_over_points=sum_over_points,
            device=device,
        )
    # ---------------------------------------------- aggregate over points / strings
    if llr_net is not None and string_xy is None and sum_over_points:
        return fisher_sum
    if string_xy is not None:
        fisher_by_string = torch.zeros(n_strings, total_dims, total_dims, device=device)
        for s_idx in range(n_strings):
            string_mask = (point_to_string == s_idx)
            if string_mask.any():
                fisher_by_string[s_idx] = fisher_per_point[string_mask].sum(dim=0)
        return fisher_by_string                          # (n_strings, total_dims, total_dims)
    elif sum_over_points:
        return fisher_per_point.sum(dim=0)               # (total_dims, total_dims)
    else:
        return fisher_per_point                          # (n_points, total_dims, total_dims)


def compute_fisher_info_single(fisher_info_params, point, event_params, surrogate_func, llr_net=None, signal_noise_scale=None, add_relative_pos=False, skip_zero_response=True, event_param_names=None):
    """
    Compute the Fisher information matrix for a single point and event parameters.
    
    Handles both scalar parameters (e.g., energy) and multi-dimensional parameters (e.g., position, direction).
    
    Parameters:
    -----------
    point : torch.Tensor
        The 3D point to evaluate the Fisher information at.
    event_params : dict
        Dictionary containing event parameters.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
        
    Returns:
    --------
    torch.Tensor
        The Fisher information matrix (total_dims, total_dims) where total_dims is the sum of 
        dimensions of all parameters in fisher_info_params.
    """
    # Determine dimensionality of each parameter and total dimensions
    param_dims = []
    param_names_expanded = []
    for param_name in fisher_info_params:
        param_value = event_params.get(param_name)
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
        
    # Determine device from input point
    if isinstance(point, torch.Tensor):
        device = point.device
    else:
        device = torch.device('cpu')
    
    fisher_matrix = torch.zeros(total_dims, total_dims, device=device)
    
    # Ensure parameters require gradients
    grad_event_params = {}
    for param_name in fisher_info_params:
        grad_event_params[param_name] = event_params.get(param_name).clone().detach().requires_grad_(True)

    for param_name in event_params.keys():
        if param_name not in fisher_info_params:
            grad_event_params[param_name] = event_params.get(param_name).clone().detach().requires_grad_(False)
    

    # Compute light yield mean (λ) using the signal surrogate function with gradients
    if llr_net is None:
        light_yield_mean = surrogate_func(opt_point=point, event_params=grad_event_params)  # Shape (1,)
        if signal_noise_scale is not None:
            light_yield_mean = torch.normal(0, signal_noise_scale)*light_yield_mean + light_yield_mean
    
    else:
        features, ly = llr_net.prepare_data_from_raw(point, grad_event_params, surrogate_func, noise_scale=signal_noise_scale, output_true_light_yield=True, event_labels=fisher_info_params if event_param_names is None else event_param_names)
        light_yield_mean = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5)
        
        # Apply soft masking for zero response points using sigmoid
        if skip_zero_response:
            # Sigmoid with scaling factor to smoothly mask out low/zero responses
            # Factor of 10 gives steep transition around ly=0.5
            # print(ly.sum())
            mask = torch.sigmoid((ly-0.5) * 12.0) + 1e-6  # Small offset to avoid exact zero
           
 
            # Apply mask element-wise if batched, or as scalar if single
            if light_yield_mean.dim() > 0 and light_yield_mean.shape[0] > 1:
                # Batched case: multiply each likelihood by its mask
                light_yield_mean = light_yield_mean * mask
        
        # If batched, sum the log likelihoods (equivalent to taking product of likelihoods)
        if light_yield_mean.dim() > 0 and light_yield_mean.shape[0] > 1:
            light_yield_mean = light_yield_mean.sum()
    
    # Compute gradients for each parameter (handling multi-dimensional parameters)
    param_gradients = []
    for param_name in fisher_info_params:
        if param_name in grad_event_params:
            param_tensor = grad_event_params[param_name]
            
            # Compute gradient ∂λ/∂θ
            # For summed likelihood, gradient is scalar output
            param_grad = torch.autograd.grad(
                outputs=light_yield_mean,
                inputs=param_tensor,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            
            # Flatten gradient if multi-dimensional
            if param_grad is not None:
                if param_grad.dim() > 0:
                    param_grad = param_grad.flatten()
                else:
                    param_grad = param_grad.unsqueeze(0)
            else:
                # If gradient is None, create zero gradient with appropriate size
                param_size = param_tensor.numel()
                param_grad = torch.zeros(param_size)
            
            param_gradients.append(param_grad)

    # Compute Fisher Information matrix: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
    # Now we need to handle multi-dimensional parameters
    if len(param_gradients) == len(fisher_info_params):
        # Flatten all gradients into a single vector
        all_gradients = torch.cat(param_gradients)  # Shape: (total_dims,)
        
        # Compute Fisher matrix using vectorized outer product
        if llr_net is None:
            fisher_matrix = torch.outer(all_gradients, all_gradients) / light_yield_mean
        else:
            fisher_matrix = torch.outer(all_gradients, all_gradients)
            
    return fisher_matrix


# def compute_fisher_info_batch_events(fisher_info_params, point, event_params_list, surrogate_func, device=None):
#     """
#     Compute the Fisher information matrix for a single point with multiple events (batched).
    
#     This is much more efficient than calling compute_fisher_info_single multiple times
#     as it processes all events simultaneously and computes gradients in batch.
    
#     Parameters:
#     -----------
#     fisher_info_params : list of str
#         List of event parameters to compute Fisher information for.
#     point : torch.Tensor
#         The 3D point to evaluate the Fisher information at.
#     event_params_list : list of dict
#         List of dictionaries containing event parameters.
#     surrogate_func : callable
#         Function that computes light yield from event parameters.
#     device : torch.device or None
#         Device to use for computations.
        
#     Returns:
#     --------
#     torch.Tensor
#         The Fisher information matrix (n_params, n_params).
#     """
#     if device is None:
#         device = point.device
        
#     n_params = len(fisher_info_params)
#     n_events = len(event_params_list)
#     fisher_matrix = torch.zeros(n_params, n_params, device=device)
    
#     if n_events == 0:
#         return fisher_matrix
    
#     # Stack event parameters into batched tensors
#     batched_grad_params = {}
#     batched_fixed_params = {}
    
#     for param_name in fisher_info_params:
#         param_values = []
#         for event_params in event_params_list:
#             param_values.append(event_params.get(param_name))
#         batched_grad_params[param_name] = torch.stack(param_values).to(device).requires_grad_(True)
    
#     # Handle non-gradient parameters
#     for event_params in event_params_list:
#         for param_name in event_params.keys():
#             if param_name not in fisher_info_params:
#                 if param_name not in batched_fixed_params:
#                     param_values = []
#                     for ep in event_params_list:
#                         param_values.append(ep.get(param_name))
#                     batched_fixed_params[param_name] = torch.stack(param_values).to(device)
    
#     # Combine all parameters
#     batched_event_params = {**batched_grad_params, **batched_fixed_params}
    
#     # Expand point to match batch size
#     batched_points = point.unsqueeze(0).expand(n_events, -1)  # Shape: (n_events, 3)
    
#     # Compute light yields for all events at once
#     light_yields = []
#     for i in range(n_events):
#         event_params_dict = {k: v[i] for k, v in batched_event_params.items()}
#         light_yield = surrogate_func(opt_point=batched_points[i], event_params=event_params_dict)
#         light_yields.append(light_yield)
    
#     light_yields = torch.stack(light_yields)  # Shape: (n_events,)
    
#     # Compute gradients for all parameters
#     param_gradients = []
#     for param_name in fisher_info_params:
#         param_tensor = batched_grad_params[param_name]
        
#         # Compute gradient ∂λ/∂θ for all events
#         grad_outputs = torch.ones_like(light_yields)
#         param_grad = torch.autograd.grad(
#             outputs=light_yields,  # Sum to get scalar output
#             inputs=param_tensor,
#             grad_outputs=grad_outputs,
#             create_graph=False,
#             retain_graph=True,
#             only_inputs=True,
#             allow_unused=True
#         )[0]  # Shape: (n_events,)
        
#         param_gradients.append(param_grad)
    
#     # Compute Fisher Information matrix: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
#     if len(param_gradients) == n_params:
#         for i_param in range(n_params):
#             for j_param in range(n_params):
#                 grad_i = param_gradients[i_param]  # Shape: (n_events, n_points)
#                 grad_j = param_gradients[j_param]  # Shape: (n_events, n_points)

#                 # Fisher Information elements for all events: (∂λ/∂θ_i)(∂λ/∂θ_j)/λ
#                 fisher_elements = (grad_i * grad_j) / light_yields.unsqueeze(1)  # Shape: (n_events, n_points)

#                 # Average across events
#                 fisher_matrix[i_param, j_param] = fisher_elements.mean()
    
#     return fisher_matrix


# def compute_fisher_info_batch_points(fisher_info_params, points, event_params_list, surrogate_func, device=None):
#     """
#     Compute the Fisher information matrix for multiple points with multiple events (fully batched).
    
#     This is the most efficient approach as it processes all points and events simultaneously.
    
#     Parameters:
#     -----------
#     fisher_info_params : list of str
#         List of event parameters to compute Fisher information for.
#     points : torch.Tensor
#         The 3D points to evaluate the Fisher information at. Shape: (n_points, 3)
#     event_params_list : list of dict
#         List of dictionaries containing event parameters.
#     surrogate_func : callable
#         Function that computes light yield from event parameters.
#     device : torch.device or None
#         Device to use for computations.
        
#     Returns:
#     --------
#     torch.Tensor
#         The Fisher information matrix summed over all points (n_params, n_params).
#     """
#     if device is None:
#         device = points.device
        
#     n_params = len(fisher_info_params)
#     n_events = len(event_params_list)
#     n_points = points.shape[0]
    
#     total_fisher_matrix = torch.zeros(n_params, n_params, device=device)
    
#     if n_events == 0 or n_points == 0:
#         return total_fisher_matrix
    
#     # For now, we'll batch over events but process points sequentially
#     # Future optimization could batch both dimensions if surrogate_func supports it
#     for point in points:
#         fisher_matrix = compute_fisher_info_batch_events(
#             fisher_info_params, point, event_params_list, surrogate_func, device
#         )
#         total_fisher_matrix += fisher_matrix
    
#     return total_fisher_matrix


def compute_fisher_info_strings(fisher_info_params, string_xy, points_3d, event_params, surrogate_func, llr_net=None, signal_noise_scale=None, add_relative_pos=False, skip_zero_response=True, event_param_names=None, device=None):
    """
    Compute the Fisher information matrix for each string in the geometry.
    
    Processes all detector points at once for efficiency, then groups results by string.
    This is more efficient than calling compute_fisher_info_single for each point separately.
    
    Parameters:
    -----------
    fisher_info_params : list of str
        List of event parameters to compute Fisher information for.
    string_xy : list of torch.Tensor
        List of 2D coordinates for each string [(x1, y1), (x2, y2), ...].
    points_3d : torch.Tensor
        All 3D points in the geometry, shape (n_points, 3).
    event_params : dict
        Dictionary containing event parameters for a single event.
    surrogate_func : callable
        Function that computes light yield from event parameters.
    llr_net : optional
        Neural network for computing log-likelihood ratios.
    signal_noise_scale : float or None
        Scale for adding noise to signal.
    add_relative_pos : bool
        Whether to add relative position features.
    skip_zero_response : bool
        Whether to skip points with zero response using soft masking.
    event_param_names : list or None
        Names of event parameters to use for LLR network.
    device : torch.device or None
        Device to use for computations.
        
    Returns:
    --------
    torch.Tensor
        Fisher information matrix per string, shape (n_strings, total_dims, total_dims)
        where total_dims is the sum of dimensions of all parameters in fisher_info_params.
    """
    if device is None:
        device = points_3d.device
    
    n_strings = len(string_xy)
    n_points = points_3d.shape[0]
    
    # Determine dimensionality of each parameter
    param_dims = []
    param_names_expanded = []
    for param_name in fisher_info_params:
        param_value = event_params.get(param_name)
        if param_value.dim() == 0 or (param_value.dim() == 1 and param_value.shape[0] == 1):
            param_dims.append(1)
            param_names_expanded.append(param_name)
        else:
            dim_size = param_value.numel()
            param_dims.append(dim_size)
            for i in range(dim_size):
                param_names_expanded.append(f"{param_name}_{i}")
    
    total_dims = sum(param_dims)
    
    # Initialize output tensor
    fisher_per_string = torch.zeros(n_strings, total_dims, total_dims, device=device)
    
    # Prepare gradient-enabled parameters
    grad_event_params = {}
    for param_name in fisher_info_params:
        grad_event_params[param_name] = event_params.get(param_name).clone().detach().requires_grad_(True)
    
    for param_name in event_params.keys():
        if param_name not in fisher_info_params:
            grad_event_params[param_name] = event_params.get(param_name).clone().detach().requires_grad_(False)
    
    # Compute light yield for all points at once
    if llr_net is None:
        # Try batch computation, fall back to loop if surrogate doesn't support it
        try:
            light_yield_means = surrogate_func(opt_point=points_3d, event_params=grad_event_params)
            if signal_noise_scale is not None:
                noise = torch.normal(0, signal_noise_scale, size=light_yield_means.shape, device=device)
                light_yield_means = noise * light_yield_means + light_yield_means
        except:
            # Fall back to sequential processing if batch not supported
            light_yield_means = []
            for point in points_3d:
                ly = surrogate_func(opt_point=point, event_params=grad_event_params)
                if signal_noise_scale is not None:
                    ly = torch.normal(0, signal_noise_scale) * ly + ly
                light_yield_means.append(ly)
            light_yield_means = torch.stack(light_yield_means)
    else:
        # Use LLR network - process all points at once (batch processing)
        # Prepare features for all points simultaneously
        # features_list = []
        # ly_list = []
        
        # Batch prepare data for all points
        # for point in points_3d:
        features, ly = llr_net.prepare_data_from_raw(
            points_3d, grad_event_params, surrogate_func, 
            noise_scale=signal_noise_scale, 
            output_true_light_yield=True, 
            event_labels=fisher_info_params if event_param_names is None else event_param_names
        )
            # # Handle PATD case where features might be batched per point
            # if features.dim() > 1:
            #     features_list.append(features)
            #     ly_list.append(ly)
            # else:
            #     features_list.append(features.unsqueeze(0))
            #     ly_list.append(ly.unsqueeze(0) if ly.dim() == 0 else ly)
        
        # Stack all features and process through LLR network in one batch
        # features_batch = torch.cat(features_list, dim=0)
        # ly_batch = torch.cat(ly_list, dim=0)
        
        # Single forward pass through LLR network for all points
        light_yield_means = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5)
        
        # Apply soft masking for zero response points
        if skip_zero_response:
            mask = torch.sigmoid((ly - 0.5) * 12.0) + 1e-6
            light_yield_means = light_yield_means * mask
    
    # Sum light yields for each string using vectorized operations
    # Create string indices for each point
    string_indices = torch.zeros(n_points, dtype=torch.long, device=device)
    for s_idx in range(n_strings):
        mask = (points_3d[:, 0] == string_xy[s_idx][0]) & (points_3d[:, 1] == string_xy[s_idx][1])
        string_indices[mask] = s_idx
    
    # Use scatter_add for efficient summation by string
    string_light_yields = torch.zeros(n_strings, device=device)
    string_light_yields.scatter_add_(0, string_indices, light_yield_means)
    string_light_yields = [string_light_yields[i] for i in range(n_strings)]
    
    # Compute gradients for each string using backward passes
    param_gradients = []
    for param_name in fisher_info_params:
        if param_name in grad_event_params:
            param_tensor = grad_event_params[param_name]
            param_size = param_tensor.numel()
            
            # Collect gradients for all strings
            string_grads = []
            for s_idx in range(n_strings):
                # Zero out any existing gradients
                if param_tensor.grad is not None:
                    param_tensor.grad.zero_()
                
                # Backward pass for this string
                string_light_yields[s_idx].backward(retain_graph=True)
                
                # Extract and store gradient
                if param_tensor.grad is not None:
                    grad = param_tensor.grad.clone()
                    # Flatten to 1D
                    if grad.dim() > 0:
                        grad = grad.flatten()
                    else:
                        grad = grad.unsqueeze(0)
                else:
                    grad = torch.zeros(param_size, device=device)
                
                string_grads.append(grad)
            
            # Stack gradients: (n_strings, param_size)
            param_grad_batch = torch.stack(string_grads)
            param_gradients.append(param_grad_batch)  # Shape: (n_strings, param_dim)
    
    # Compute Fisher Information matrix for each string using vectorized operations
    if len(param_gradients) == len(fisher_info_params):
        # Stack all parameter gradients: (n_strings, total_dims)
        all_gradients = torch.cat(param_gradients, dim=1)
        
        # Vectorized computation of Fisher matrix for all strings at once
        # Use batched outer product: bmm(grad.unsqueeze(-1), grad.unsqueeze(-2))
        grad_outer = torch.bmm(
            all_gradients.unsqueeze(-1),  # (n_strings, total_dims, 1)
            all_gradients.unsqueeze(-2)   # (n_strings, 1, total_dims)
        )  # Result: (n_strings, total_dims, total_dims)
        
        if llr_net is None:
            # Divide by string light yields
            string_ly_tensor = torch.stack([string_light_yields[i] for i in range(n_strings)])
            fisher_per_string = grad_outer / string_ly_tensor.view(n_strings, 1, 1)
        else:
            fisher_per_string = grad_outer
    
    return fisher_per_string


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

    def compute_fisher_info_per_string_per_event(self, string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net=None, signal_noise_scale=None, llr_iterations=1, add_relative_pos=False, skip_zero_response=True, verbose=False, event_batch_size=1, grad_chunk_size=10, jacrev_chunk_size=10000, point_chunk_size=None, llr_autodiff_mode='jacrev'):
        n_strings = len(string_xy)
        # n_params = len(self.fisher_info_params)
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
        fisher_per_string_per_event = torch.zeros(len(signal_event_params), n_strings, total_dims, total_dims, device=self.device)
        if event_batch_size == 1:    
            for i, signal_params in enumerate(signal_event_params):
                # for s_idx in range(n_strings):
                #     mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
                #     string_points = points_3d[mask]
                #     fisher_matrix = torch.zeros(total_dims, total_dims, device=self.device)
                #     # for point in string_points: 
                #     # for _ in range(llr_iterations):    
                #         # fisher_matrix += compute_fisher_info_single(self.fisher_info_params, string_points, event_params=signal_params, surrogate_func=signal_surrogate_func, llr_net=llr_net, signal_noise_scale=signal_noise_scale, add_relative_pos=add_relative_pos, skip_zero_response=skip_zero_response)
                #     # fisher_matrix = fisher_matrix/llr_iterations
                #     fisher_matrix = compute_fisher_info_single_averaged(self.fisher_info_params, string_points, llr_iterations=llr_iterations, event_params=signal_params, surrogate_func=signal_surrogate_func, llr_net=llr_net, signal_noise_scale=signal_noise_scale, add_relative_pos=add_relative_pos, skip_zero_response=skip_zero_response)
                #     fisher_per_string_per_event[i, s_idx] += fisher_matrix
                # Returns (n_strings, n_events, total_dims, total_dims), need to permute to (n_events, n_strings, ...)
                fisher_matrices = compute_fisher_info_single_averaged(
                            fisher_info_params=self.fisher_info_params, 
                            point=points_3d, 
                            event_params=signal_params, 
                            surrogate_func=signal_surrogate_func, 
                            llr_net=llr_net, 
                            signal_noise_scale=signal_noise_scale, 
                            add_relative_pos=add_relative_pos, 
                            skip_zero_response=skip_zero_response, 
                            llr_iterations=llr_iterations, 
                            string_xy=string_xy,
                            device=self.device,
                            grad_chunk_size=grad_chunk_size,
                            jacrev_chunk_size=jacrev_chunk_size,
                            point_chunk_size=point_chunk_size,
                            llr_autodiff_mode=llr_autodiff_mode
                            )
                fisher_per_string_per_event[i] += fisher_matrices.to(self.device)
                if verbose and (i % 100 == 0 or i == len(signal_event_params)-1):
                    print(f"Computed Fisher info for event {i+1}/{len(signal_event_params)}", flush=True)        
        else:
            for i in range(0, len(signal_event_params), event_batch_size):
                batch_params = signal_event_params[i:i+event_batch_size]
                # Returns (n_strings, n_events, total_dims, total_dims), need to permute to (n_events, n_strings, ...)
                fisher_matrices = compute_fisher_info_single_batched(
                            fisher_info_params=self.fisher_info_params, 
                            point=points_3d, 
                            event_params_list=batch_params, 
                            surrogate_func=signal_surrogate_func, 
                            llr_net=llr_net, 
                            signal_noise_scale=signal_noise_scale, 
                            add_relative_pos=add_relative_pos, 
                            skip_zero_response=skip_zero_response, 
                            llr_iterations=llr_iterations, 
                            string_xy=string_xy
                            )
                fisher_per_string_per_event[i:i+event_batch_size] += fisher_matrices.permute(1, 0, 2, 3)   
            

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
        
        # Compute or use precomputed Fisher info
        if precomputed_fisher_info_per_string_per_event is None:
            fisher_info_per_string_per_event = self.compute_fisher_info_per_string_per_event(string_xy, points_3d, signal_event_params, signal_surrogate_func, llr_net, signal_noise_scale, llr_iterations, add_relative_pos, skip_zero_response=skip_zero_response, event_batch_size=event_batch_size, grad_chunk_size=grad_chunk_size, jacrev_chunk_size=jacrev_chunk_size, point_chunk_size=point_chunk_size, llr_autodiff_mode=llr_autodiff_mode)
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
            # resolution_per_event = torch.stack(resolution_per_event)
            total_resolution = torch.nanmean(resolution_per_event)/max_angular_resolution
            # finite_mask = torch.isfinite(resolution_per_event)
            # if finite_mask.any():
            #     total_resolution = torch.mean(resolution_per_event[finite_mask])
            #     total_resolution = total_resolution/max_angular_resolution
            # else:
            #     # If all resolutions are NaN/inf, return a large penalty value
            #     total_resolution = torch.tensor(1.0, device=self.device, requires_grad=True)
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
            total_resolution = torch.mean(resolution_per_event[~torch.isnan(resolution_per_event)])
            total_resolution = total_resolution/max_energy_resolution

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
            # resolution_per_event = torch.stack(resolution_per_event)
            total_resolution = torch.mean(resolution_per_event[torch.isfinite(resolution_per_event)])
            total_resolution = total_resolution/max_angular_resolution
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
            total_resolution = torch.mean(resolution_per_event[torch.isfinite(resolution_per_event)])
            total_resolution = total_resolution/max_energy_resolution

            return {'energy_resolution_loss': total_resolution, 'resolution_per_event': resolution_per_event, 'resolution_params': event_params}
