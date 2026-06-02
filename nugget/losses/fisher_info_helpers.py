import torch
import torch.nn.functional as F
import numpy as np
import gc
import os
import math
from torch.func import jacrev, jvp, vmap, linearize


def _pos_norm_divisor_from_domain_size(domain_size, *, device, dtype=torch.float32):
    """Match LLRnet's norm_pos scaling.

    - Scalar domain_size: divide all coordinates by (domain_size/2).
    - Tuple/list (width, height): divide x,y by (width/2) and z by (height/2).

    Returns either a Python float or a (3,) torch.Tensor on `device`.
    """
    if isinstance(domain_size, torch.Tensor):
        # If it's a scalar tensor.
        domain_size = domain_size.item()

    if isinstance(domain_size, (tuple, list)) and len(domain_size) == 2:
        width, height = domain_size
        if isinstance(width, torch.Tensor):
            width = width.item()
        if isinstance(height, torch.Tensor):
            height = height.item()
        width = float(width)
        height = float(height)
        return torch.tensor(
            [width / 2.0, width / 2.0, height / 2.0],
            device=device,
            dtype=dtype,
        )

    return float(domain_size) / 2.0


def _llr_mask_from_true_ly(true_ly, *, threshold=0.5, sharpness=12.0):
    return torch.sigmoid((true_ly - threshold) * sharpness) + 1e-6


def _fisher_chunk_cleanup(device):
    """Release Python-side objects between chunks without forcing CUDA cache flushes.

    Calling torch.cuda.empty_cache() in tight autodiff loops can surface asynchronous
    CUDA faults at cache-flush points and usually hurts throughput. Keep cleanup to
    Python GC and optional explicit synchronization for debugging.
    """
    gc.collect()
    if device.type == 'cuda' and os.environ.get('NUGGET_FISHER_CUDA_SYNC', '0') == '1':
        torch.cuda.synchronize(device)


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
    use_rich_features=False,
):
    params = {fisher_info_params[i]: theta_vals[i] for i in range(len(fisher_info_params))}
    params.update(fixed_params)

    # NOTE: this function is called inside jacrev, so everything here is
    # differentiated w.r.t. theta_vals.  The surrogate call must be hoisted
    # outside via pre-sampled observations; only hypothesis feature assembly
    # and the network forward pass should be inside the trace.
    # For rich features, the caller (_fisher_one_point_jacrev) pre-samples
    # observations and passes them in via fixed_params['_cached_rich_obs'] and
    # fixed_params['_cached_rich_ly'].  The standard path is unchanged.
    if use_rich_features:
        cached_obs = fixed_params.get('_cached_rich_obs')
        cached_ly = fixed_params.get('_cached_rich_ly')  # (L,) or (L, 1)

        theta_shapes = [v.shape for v in theta_vals]
        theta_numels = [v.numel() for v in theta_vals]
        theta_flat = torch.cat([v.reshape(-1) for v in theta_vals])
        # fixed_params without the cache sentinels
        base_fixed = {k: v for k, v in fixed_params.items()
                      if not k.startswith('_cached_rich')}

        features = _build_rich_features_from_cached_obs(
            pt_3.unsqueeze(0),
            theta_flat=theta_flat,
            cached_obs=cached_obs,
            llr_net=llr_net,
            fisher_info_params=fisher_info_params,
            theta_shapes=theta_shapes,
            theta_numels=theta_numels,
            fixed_params=base_fixed,
            device=pt_3.device,
        )
        ly = cached_ly.reshape(-1)
    else:
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
    detach_fisher_tensors=True,
    use_rich_features=False,
):
    if use_rich_features:
        # Pre-sample observations outside the jacrev trace so the surrogate is
        # called exactly once (not once per parameter dimension).
        params0 = {fisher_info_params[i]: theta_tuple[i].detach() for i in range(len(fisher_info_params))}
        params0.update(fixed_params)
        cached_obs, cached_ly = _sample_rich_observations(
            pt_3.unsqueeze(0),
            surrogate_func=surrogate_func,
            params_for_sampling=params0,
            llr_iterations=llr_iterations,
            llr_net=llr_net,
            device=pt_3.device,
        )
        # Inject into fixed_params so _llr_out_single_point_all_iters can read them.
        augmented_fixed = dict(fixed_params)
        augmented_fixed['_cached_rich_obs'] = cached_obs
        augmented_fixed['_cached_rich_ly'] = cached_ly.reshape(-1)  # (L,)
    else:
        augmented_fixed = fixed_params

    def _theta_only_fn(*theta_vals):
        return _llr_out_single_point_all_iters(
            pt_3,
            theta_vals=theta_vals,
            fisher_info_params=fisher_info_params,
            fixed_params=augmented_fixed,
            llr_net=llr_net,
            surrogate_func=surrogate_func,
            llr_iterations=llr_iterations,
            signal_noise_scale=signal_noise_scale,
            skip_zero_response=skip_zero_response,
            event_param_names=event_param_names,
            use_rich_features=use_rich_features,
        )

    J_tuple = jacrev(
        _theta_only_fn,
        argnums=tuple(range(len(fisher_info_params))),
        chunk_size=jacrev_chunk_size,
    )(*theta_tuple)

    J = torch.cat([j.reshape(llr_iterations, -1) for j in J_tuple], dim=1)
    del J_tuple
    F = torch.einsum('li,lj->ij', J, J) / llr_iterations
    del J
    return F.detach() if detach_fisher_tensors else F


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
        divisor = _pos_norm_divisor_from_domain_size(domain_size, device=device, dtype=pts_3.dtype)
        norm_points = pts_3 / divisor
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
            divisor = _pos_norm_divisor_from_domain_size(domain_size, device=device, dtype=feature.dtype)
            feature = feature / divisor
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


def _sample_rich_observations(
    pts_3,
    *,
    surrogate_func,
    params_for_sampling,
    llr_iterations,
    llr_net,
    device,
):
    """Pre-sample surrogate observations for the rich-feature path.

    Like _sample_detector_responses_batched but stores the full surrogate output
    (scalar LY or PATD dict) so that _build_rich_features_from_cached_obs can
    append the fixed observation to the differentiable hypothesis features.

    Returns
    -------
    cached_obs : list of lists — cached_obs[l][b] is the raw surrogate output
        for iteration l, point b. For charge: a scalar tensor. For PATD: a dict.
    cached_ly_true : (L, B) float tensor of light yields for masking.
    """
    pts_3 = pts_3.float().to(device)
    B = pts_3.shape[0]
    is_patd = bool(getattr(llr_net, 'use_patd', False))

    cached_obs = []
    ly_rows = []

    with torch.no_grad():
        for _ in range(llr_iterations):
            # Call surrogate once for all B points together (matches _sample_detector_responses_batched).
            # Fall back to per-point loop only if the surrogate doesn't support batched points.
            try:
                batch_raw = surrogate_func(opt_point=pts_3, event_params=params_for_sampling)
                if is_patd:
                    # Batched PATD returns a list of dicts, one per point
                    if not isinstance(batch_raw, (list, tuple)) or len(batch_raw) != B:
                        raise TypeError
                    obs_row = list(batch_raw)
                    ly_row = [
                        float(r.get('num_photons', 0).item()
                              if isinstance(r.get('num_photons', 0), torch.Tensor)
                              else r.get('num_photons', 0))
                        for r in obs_row
                    ]
                else:
                    # Batched charge returns a (B,) tensor
                    if isinstance(batch_raw, dict):
                        batch_raw = batch_raw.get('light_yield', next(iter(batch_raw.values())))
                    if not isinstance(batch_raw, torch.Tensor):
                        raise TypeError
                    batch_raw = batch_raw.detach().float().reshape(-1)
                    if batch_raw.numel() != B:
                        raise ValueError
                    obs_row = [batch_raw[b] for b in range(B)]
                    ly_row = [float(v.item()) for v in obs_row]
            except Exception:
                # Fall back: call surrogate per point
                obs_row = []
                ly_row = []
                for b in range(B):
                    raw = surrogate_func(opt_point=pts_3[b], event_params=params_for_sampling)
                    if is_patd:
                        n = raw.get('num_photons', 0)
                        ly_val = float(n.item()) if isinstance(n, torch.Tensor) else float(n)
                    else:
                        if isinstance(raw, dict):
                            raw = raw.get('light_yield', next(iter(raw.values())))
                        if isinstance(raw, torch.Tensor):
                            raw = raw.detach().float()
                        else:
                            raw = torch.tensor(float(raw), dtype=torch.float32, device=device)
                        ly_val = float(raw.item())
                    obs_row.append(raw)
                    ly_row.append(ly_val)
            cached_obs.append(obs_row)
            ly_rows.append(ly_row)

    cached_ly_true = torch.tensor(ly_rows, dtype=torch.float32, device=device)  # (L, B)
    return cached_obs, cached_ly_true


def _build_rich_features_from_cached_obs(
    pts_3,
    *,
    theta_flat,
    cached_obs,
    llr_net,
    fisher_info_params,
    theta_shapes,
    theta_numels,
    fixed_params,
    device,
):
    """Build rich features differentiably from cached (fixed) observations.

    The observation (LY scalar or PATD hit times / t_geom_min) is pre-sampled
    and held constant. Only the hypothesis geometry (vert, dir, energy, derived
    geometric scalars) is rebuilt from theta_flat, so gradients flow only through
    the hypothesis parameters — exactly as _build_features_from_cached_responses
    does for the standard path.

    Returns
    -------
    features : (L*B, feat_dim) tensor
    """
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
    L = len(cached_obs)
    is_patd = bool(getattr(llr_net, 'use_patd', False))

    # --- differentiable hypothesis features — computed ONCE over B, then broadcast ---
    norm = llr_net._pos_norm_divisor()

    vert = params.get('position', params.get('vertex', None))
    if vert is None:
        raise KeyError("'position' not found in params for rich feature builder")
    if isinstance(vert, np.ndarray):
        vert = torch.tensor(vert, device=device, dtype=torch.float32)
    vert = vert.float().to(device).reshape(1, 3) / norm  # (1, 3)

    direction = params.get('direction')
    if direction is None:
        raise KeyError("'direction' not found in params for rich feature builder")
    if isinstance(direction, np.ndarray):
        direction = torch.tensor(direction, device=device, dtype=torch.float32)
    direction = direction.float().to(device).reshape(1, 3)  # (1, 3)
    dir_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)  # (1, 1)

    energy = params.get('energy')
    if energy is None:
        raise KeyError("'energy' not found in params for rich feature builder")
    if isinstance(energy, np.ndarray):
        energy = torch.tensor(energy, device=device, dtype=torch.float32)
    log_energy = torch.log10(energy.float().to(device).squeeze() + 1e-10) / 8.0  # scalar

    # Batched geometry over B points — one operation, not B scalar calls
    det = pts_3 / norm                                      # (B, 3)
    rel = det - vert                                        # (B, 3)
    vert_dist = torch.norm(rel, dim=-1, keepdim=True)      # (B, 1)
    cos_angle = (direction * rel).sum(dim=-1, keepdim=True) / (dir_norm * vert_dist.clamp(min=1e-8))  # (B, 1)

    # Hypothesis context: (B, ctx_dim) — does NOT depend on l
    log_energy_expanded = log_energy.expand(B, 1)          # (B, 1)
    ctx = torch.cat([
        det,                    # (B, 3)
        vert.expand(B, -1),     # (B, 3)
        direction.expand(B, -1),# (B, 3)
        log_energy_expanded,    # (B, 1)
        vert_dist,              # (B, 1)
        cos_angle,              # (B, 1)
    ], dim=-1)  # (B, 12) for charge or (B, 12) for PATD

    if not is_patd:
        # Charge path: one scalar observation per (l, b) → (L, B, 1)
        # Build observation tensor from cache: shape (L, B, 1)
        obs_rows = []
        for l in range(L):
            obs_b = []
            for b in range(B):
                ly_raw = cached_obs[l][b]
                if isinstance(ly_raw, torch.Tensor):
                    val = ly_raw.float().to(device).squeeze()
                else:
                    val = torch.tensor(float(ly_raw), dtype=torch.float32, device=device)
                obs_b.append(torch.log10(torch.abs(val) + 1e-10) / 4.0)
            obs_rows.append(torch.stack(obs_b))          # (B,)
        log_ly = torch.stack(obs_rows).unsqueeze(-1)     # (L, B, 1) — detached constants

        # Broadcast ctx across L, append observation: (L, B, 13)
        ctx_expanded = ctx.unsqueeze(0).expand(L, -1, -1)   # (L, B, 12)
        features_batched = torch.cat([ctx_expanded, log_ly], dim=-1)  # (L, B, 13)
        return features_batched.reshape(L * B, -1)
    else:
        # PATD path: variable number of hits per (l, b) — must loop over (l, b)
        # but geometry is read from pre-computed ctx rows, not recomputed each time.
        all_features = []
        for l in range(L):
            for b in range(B):
                raw = cached_obs[l][b]
                hit_times = raw['hit_times'].float().to(device)
                t_scaled = torch.where(
                    hit_times < 0,
                    -torch.log10(-hit_times + 1e-4) / 4.0,
                    torch.log10(hit_times + 1e-4) / 4.0,
                )  # (N_hits,)
                # ctx[b] is already computed — just expand and append hit times
                ctx_rep = ctx[b].unsqueeze(0).expand(t_scaled.shape[0], -1)  # (N_hits, 12)
                feat = torch.cat([ctx_rep, t_scaled.unsqueeze(1)], dim=1)    # (N_hits, 13)
                all_features.append(feat)
        return torch.cat(all_features, dim=0)  # (total_hits, 13)


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
    detach_fisher_tensors=True,
    use_rich_features=False,
):
    # Rich-feature path: can never cache responses (surrogate dict structure differs).
    can_cache_responses = (
        not bool(getattr(llr_net, 'use_patd', False))
        and not use_rich_features
    )

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
        if detach_fisher_tensors:
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
    elif use_rich_features:
        # Pre-sample observations outside linearize so the surrogate is called
        # exactly once — not once per JVP direction.
        params0 = _unflatten_theta(
            theta0_flat,
            fisher_info_params=fisher_info_params,
            theta_shapes=theta_shapes,
            theta_numels=theta_numels,
            fixed_params=fixed_params,
        )
        cached_obs, cached_ly_true = _sample_rich_observations(
            pts_3,
            surrogate_func=surrogate_func,
            params_for_sampling=params0,
            llr_iterations=llr_iterations,
            llr_net=llr_net,
            device=device,
        )
        cached_ly_true = cached_ly_true.detach()  # (L, B)

        # Pre-compute everything that does NOT depend on theta outside the trace.
        # Only vert, direction, log_energy (and derived vert_dist, cos_angle) vary with theta.
        B = pts_3.shape[0]
        L = llr_iterations
        norm_const = llr_net._pos_norm_divisor()
        det_const = (pts_3.float().to(device) / norm_const).detach()  # (B, 3) — constant

        # Observation column: (L, B, 1) — constant, built once here not inside the trace
        log_ly_const = torch.stack([
            torch.stack([
                torch.log10(torch.abs(
                    cached_obs[l][b].float().to(device).squeeze()
                    if isinstance(cached_obs[l][b], torch.Tensor)
                    else torch.tensor(float(cached_obs[l][b]), device=device)
                ) + 1e-10) / 4.0
                for b in range(B)
            ])
            for l in range(L)
        ]).unsqueeze(-1).detach()  # (L, B, 1)

        def _theta_only_fn(theta_flat):
            params = _unflatten_theta(
                theta_flat,
                fisher_info_params=fisher_info_params,
                theta_shapes=theta_shapes,
                theta_numels=theta_numels,
                fixed_params=fixed_params,
            )
            # Hypothesis features — theta-dependent parts only
            vert = params['position'].float().to(device).reshape(1, 3) / norm_const  # (1, 3)
            direction = params['direction'].float().to(device).reshape(1, 3)          # (1, 3)
            dir_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
            energy = params['energy'].float().to(device).squeeze()
            log_energy = (torch.log10(energy + 1e-10) / 8.0).reshape(1, 1).expand(B, 1)

            rel = det_const - vert                                                    # (B, 3)
            vert_dist = torch.norm(rel, dim=-1, keepdim=True)                        # (B, 1)
            cos_angle = (direction * rel).sum(dim=-1, keepdim=True) / (
                dir_norm * vert_dist.clamp(min=1e-8))                                 # (B, 1)

            ctx = torch.cat([
                det_const,               # (B, 3) — constant but needs to be in graph for cat
                vert.expand(B, -1),      # (B, 3)
                direction.expand(B, -1), # (B, 3)
                log_energy,              # (B, 1)
                vert_dist,               # (B, 1)
                cos_angle,               # (B, 1)
            ], dim=-1)  # (B, 12)

            # Broadcast ctx over L and append constant log_ly
            ctx_exp = ctx.unsqueeze(0).expand(L, -1, -1)              # (L, B, 12)
            features = torch.cat([ctx_exp, log_ly_const], dim=-1)     # (L, B, 13)
            features = features.reshape(L * B, -1)                     # (L*B, 13)

            llr_out = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5).reshape(L, B)
            if skip_zero_response:
                llr_out = llr_out * _llr_mask_from_true_ly(cached_ly_true)
            return llr_out.transpose(0, 1).contiguous()  # (B, L)
    else:
        # PATD path (non-rich): call surrogate inside the trace (no caching possible).
        def _theta_only_fn(theta_flat):
            params = _unflatten_theta(
                theta_flat,
                fisher_info_params=fisher_info_params,
                theta_shapes=theta_shapes,
                theta_numels=theta_numels,
                fixed_params=fixed_params,
            )
            B = pts_3.shape[0]
            features, ly = llr_net.prepare_data_from_raw(
                pts_3,
                params,
                surrogate_func,
                noise_scale=signal_noise_scale,
                output_true_light_yield=True,
                event_labels=fisher_info_params if event_param_names is None else event_param_names,
                num_samples=llr_iterations,
            )
            ly = ly.reshape(llr_iterations, B)
            llr_out = llr_net.predict_log_likelihood_ratio(features, epsilon=1e-5).reshape(llr_iterations, B)
            if skip_zero_response:
                llr_out = llr_out * _llr_mask_from_true_ly(ly)
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

        cols_chunk = vmap(jvp_fn, randomness='same')(basis_chunk)  # (k, B, L)
        cols_parts.append(cols_chunk)
        del basis_chunk, cols_chunk

    cols = torch.cat(cols_parts, dim=0)
    del cols_parts
    J = cols.permute(1, 2, 0).contiguous()  # (B, L, D)
    del cols
    F = torch.einsum('bld,ble->bde', J, J) / llr_iterations
    del J

    if can_cache_responses:
        del cached_responses, cached_ly_true
    del jvp_fn
    return F.detach() if detach_fisher_tensors else F


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
    detach_fisher_tensors=True,
    use_rich_features=False,
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
        if use_rich_features:
            # Pre-sample observations for all points BEFORE the per-point loop so
            # that _sample_rich_observations (which has Python control flow) never
            # runs inside vmap.  Each point gets its own list of L cached observations.
            params0 = {fisher_info_params[i]: theta_tuple[i].detach() for i in range(len(fisher_info_params))}
            params0.update(fixed_params)
            all_cached_obs, all_cached_ly = _sample_rich_observations(
                point,  # all n_points at once
                surrogate_func=surrogate_func,
                params_for_sampling=params0,
                llr_iterations=llr_iterations,
                llr_net=llr_net,
                device=device,
            )
            # all_cached_obs[l][b] for b in 0..n_points, all_cached_ly: (L, n_points)

            def _one_rich(pt_3, pt_idx):
                # Extract per-point cache — pure Python indexing, outside jacrev trace.
                pt_cached_obs = [[all_cached_obs[l][pt_idx]] for l in range(llr_iterations)]
                pt_cached_ly = all_cached_ly[:, pt_idx].reshape(-1)  # (L,)
                augmented_fixed = dict(fixed_params)
                augmented_fixed['_cached_rich_obs'] = pt_cached_obs
                augmented_fixed['_cached_rich_ly'] = pt_cached_ly

                def _theta_only_fn(*theta_vals):
                    return _llr_out_single_point_all_iters(
                        pt_3,
                        theta_vals=theta_vals,
                        fisher_info_params=fisher_info_params,
                        fixed_params=augmented_fixed,
                        llr_net=llr_net,
                        surrogate_func=surrogate_func,
                        llr_iterations=llr_iterations,
                        signal_noise_scale=signal_noise_scale,
                        skip_zero_response=skip_zero_response,
                        event_param_names=event_param_names,
                        use_rich_features=use_rich_features,
                    )

                J_tuple = jacrev(
                    _theta_only_fn,
                    argnums=tuple(range(len(fisher_info_params))),
                    chunk_size=jacrev_chunk_size,
                )(*theta_tuple)
                J = torch.cat([j.reshape(llr_iterations, -1) for j in J_tuple], dim=1)
                del J_tuple
                F = torch.einsum('li,lj->ij', J, J) / llr_iterations
                del J
                return F.detach() if detach_fisher_tensors else F

            for p_start in range(0, n_points, pt_chunk):
                p_end = min(p_start + pt_chunk, n_points)
                pts = point[p_start:p_end]
                # Sequential loop — vmap not possible due to Python-indexed cache
                fisher_chunk = torch.stack(
                    [_one_rich(pts[i], p_start + i) for i in range(pts.shape[0])], dim=0
                )
                if detach_fisher_tensors:
                    fisher_chunk = fisher_chunk.detach()
                if fisher_sum is not None:
                    fisher_sum.add_(fisher_chunk.sum(dim=0))
                else:
                    fisher_per_point[p_start:p_end] = fisher_chunk
                del fisher_chunk, pts
                _fisher_chunk_cleanup(device)
        else:
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
                    detach_fisher_tensors=detach_fisher_tensors,
                    use_rich_features=False,
                )

            for p_start in range(0, n_points, pt_chunk):
                p_end = min(p_start + pt_chunk, n_points)
                pts = point[p_start:p_end]
                try:
                    fisher_chunk = vmap(_one, randomness='different')(pts)
                except Exception:
                    fisher_chunk = torch.stack([_one(pts[i]) for i in range(pts.shape[0])], dim=0)
                if detach_fisher_tensors:
                    fisher_chunk = fisher_chunk.detach()
                if fisher_sum is not None:
                    fisher_sum.add_(fisher_chunk.sum(dim=0))
                else:
                    fisher_per_point[p_start:p_end] = fisher_chunk
                del fisher_chunk, pts
                _fisher_chunk_cleanup(device)

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
            detach_fisher_tensors=detach_fisher_tensors,
            use_rich_features=use_rich_features,
        )
        if detach_fisher_tensors:
            fisher_chunk = fisher_chunk.detach()
        if fisher_sum is not None:
            fisher_sum.add_(fisher_chunk.sum(dim=0))
        else:
            fisher_per_point[p_start:p_end] = fisher_chunk

        del fisher_chunk, pts
        _fisher_chunk_cleanup(device)

    return fisher_sum, fisher_per_point

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
        eigvals = torch.nn.functional.softplus(eigvals, beta=5) - (math.log(2.0) / 5)
        # eigvals = torch.clamp_min(eigvals, 1e-10)  # Ensure positive eigenvalues
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
        F2 = F2 + 1e-8 * torch.eye(2, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 2, 2)  # Increased regularization for stability
        
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
        eigvals = torch.nn.functional.softplus(eigvals, beta=5) - (math.log(2.0) / 5)
        # eigvals = torch.clamp_min(eigvals, 1e-10)  # Ensure positive eigenvalues
        sigma_eff = torch.sqrt(torch.mean(eigvals, dim=1) + 1e-10)  # (N,) Add epsilon for numerical stability
        r68 = 1.515 * sigma_eff  # (N,)
        
        return r68


def compute_fisher_info_single_averaged(fisher_info_params, point, event_params, surrogate_func, llr_iterations=1, llr_net=None, signal_noise_scale=None, add_relative_pos=False, skip_zero_response=True, event_param_names=None, string_xy=None, sum_over_points=True, device=None, grad_chunk_size=None, jacrev_chunk_size=None, point_chunk_size=None, llr_autodiff_mode='jacrev', detach_fisher_tensors=True, use_patd=False, eval_patd_log_probs=None, use_rich_features=False):
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
    detach_fisher_tensors : bool
        If True (default), detach Fisher tensors during chunk aggregation to save
        memory and preserve historical behavior. Set False to keep graph
        connectivity (e.g., gradients wrt geometry points/string_xy).
    use_patd : bool
        If True, use the photon arrival time distribution path instead of the
        Poisson-mean or LLR-network paths. llr_net is ignored when use_patd=True.
        Requires eval_patd_log_probs to be provided as a callable parameter.

        Fisher info per detector: F_i = mean_charge_i * (1/N) sum_hits (d log p/dtheta)^2
        where p(t|theta) = CPandel.pdf(t_residual, d=foot_length(theta)).
        mean_charge_i is the average of num_photons over llr_iterations surrogate calls.
        N hits are collected by sampling the surrogate until llr_iterations total
        residual times are accumulated per detector.
    eval_patd_log_probs : callable or None
        Function to evaluate log-probability of photon arrival times.
        Only used when use_patd=True. If use_patd=True and eval_patd_log_probs
        is None, raises an error.

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
    if use_patd:
        # ---- PATD path -----------------------------------------------
        # Fisher info via photon arrival time distribution:
        #   F_i = mean_charge_i * (1/N_hits) Σ_hits (∂ log p(t|θ) / ∂θ)^T (∂ log p(t|θ) / ∂θ)
        # where p(t|θ) = CPandel.pdf(t_residual, d=foot_length(θ)) evaluated
        # at pre-sampled (fixed) residual times.
        #
        # Phase 1 — no-grad sampling:
        #   charge: call surrogate llr_iterations times, average num_photons per detector
        #   hits:   keep calling until each detector has >= llr_iterations residual times
        # Phase 2 — jacrev or jvp over eval_patd_log_probs (CPandel PDF with grad-enabled foot_length)

        if eval_patd_log_probs is None:
            raise ValueError(
                "use_patd=True requires eval_patd_log_probs to be provided as a callable parameter."
            )
        if not callable(eval_patd_log_probs):
            raise TypeError(
                "eval_patd_log_probs must be callable."
            )

        # If the user accidentally passed the surrogate sampling function
        # (e.g. LightSabrePATD.light_yield_surrogate) as eval_patd_log_probs,
        # try to recover the correct log-prob evaluator from the surrogate
        # instance: `surrogate_instance.eval_patd_log_probs`.
        if eval_patd_log_probs == surrogate_func:
            owner = getattr(surrogate_func, '__self__', None)
            if owner is not None and hasattr(owner, 'eval_patd_log_probs'):
                eval_patd_log_probs = getattr(owner, 'eval_patd_log_probs')
            elif hasattr(surrogate_func, 'eval_patd_log_probs'):
                eval_patd_log_probs = getattr(surrogate_func, 'eval_patd_log_probs')
            else:
                raise ValueError(
                    "The provided eval_patd_log_probs appears to be the surrogate sampling function "
                    "(which returns PATD dicts). For gradient computation you must provide a callable "
                    "that evaluates log-probabilities at fixed residual times (for LightSabrePATD this is "
                    "surrogate_instance.eval_patd_log_probs)."
                )

        patd_autodiff_mode = (llr_autodiff_mode or 'jacrev').lower()
        if patd_autodiff_mode not in {'jacrev', 'jvp'}:
            raise ValueError(
                f"Unsupported llr_autodiff_mode={llr_autodiff_mode!r} for PATD path; "
                "expected 'jacrev' or 'jvp'."
            )
        
        # Allow either 'jacrev' or 'jvp' for PATD. The user must ensure the
        # provided `eval_patd_log_probs` is fully traceable for JVP (no .item()
        # or other data-dependent numpy/CPU ops). If it is not, jacrev will be
        # the safe option.

        theta_shapes_patd = [event_params[p].detach().to(device).shape for p in fisher_info_params]
        theta_numels_patd = [event_params[p].detach().to(device).numel() for p in fisher_info_params]
        theta0_flat_patd = torch.cat([
            event_params[p].detach().to(device).reshape(-1) for p in fisher_info_params
        ])
        all_params_detached = {k: v.detach().to(device) for k, v in event_params.items()}

        t_residuals_per_pt = [[] for _ in range(n_points)]
        charge_sums = torch.zeros(n_points, dtype=torch.float32, device=device)

        with torch.no_grad():
            # Charge: llr_iterations calls
            # First, check if expected_photons is already in surrogate results
            charge_sums = torch.zeros(n_points, dtype=torch.float32, device=device)
            use_expected_photons = False
            
            # Do a single initial call to check for expected_photons
            if n_points == 1:
                res = surrogate_func(opt_point=point[0], event_params=all_params_detached)
                if isinstance(res, dict) and 'expected_photons' in res:
                    charge_sums[0] = float(res['expected_photons'])
                    use_expected_photons = True
            else:
                results = surrogate_func(opt_point=point, event_params=all_params_detached)
                if isinstance(results, list):
                    all_have_expected = True
                    for _i, res in enumerate(results):
                        if isinstance(res, dict) and 'expected_photons' in res:
                            charge_sums[_i] = float(res['expected_photons'])
                        else:
                            all_have_expected = False
                    use_expected_photons = all_have_expected
                elif isinstance(results, torch.Tensor):
                    charge_sums = results.detach().float().reshape(-1)[:n_points]
                    use_expected_photons = False
            
            # If expected_photons is not available, accumulate over iterations
            if not use_expected_photons:
                charge_sums = torch.zeros(n_points, dtype=torch.float32, device=device)
                for iter_idx in range(llr_iterations):
                    if n_points == 1:
                        res = surrogate_func(opt_point=point[0], event_params=all_params_detached)
                        if isinstance(res, dict):
                            charge_sums[0] += float(res.get('num_photons', res.get('expected_photons', 0)))
                        else:
                            charge_sums[0] += float(res.item() if isinstance(res, torch.Tensor) else float(res))
                    else:
                        results = surrogate_func(opt_point=point, event_params=all_params_detached)
                        if isinstance(results, list):
                            for _i, res in enumerate(results):
                                if isinstance(res, dict):
                                    charge_sums[_i] += float(res.get('num_photons', res.get('expected_photons', 0)))
                                else:
                                    charge_sums[_i] += float(res)
                        elif isinstance(results, torch.Tensor):
                            charge_sums += results.detach().float().reshape(-1)[:n_points]
                mean_charges = charge_sums / max(llr_iterations, 1)
            else:
                mean_charges = charge_sums  # Use pre-computed expected_photons directly

            # Hits: sample until each detector has >= llr_iterations residual times
            _max_calls = llr_iterations * 200
            _call_count = 0
            while _call_count < _max_calls:
                if all(
                    (skip_zero_response and mean_charges[_i] < 1) or
                    (sum(len(_rt) for _rt in t_residuals_per_pt[_i]) >= llr_iterations)
                    for _i in range(n_points)
                ):
                    break
                _call_count += 1
                if n_points == 1:
                    res = surrogate_func(opt_point=point[0], event_params=all_params_detached)
                    if isinstance(res, dict):
                        _rt = res.get('residual_times', None)
                        if _rt is not None and _rt.numel() > 0:
                            t_residuals_per_pt[0].append(_rt.detach().cpu())
                else:
                    results = surrogate_func(opt_point=point, event_params=all_params_detached)
                    if isinstance(results, list):
                        for _i, res in enumerate(results):
                            _rt = res.get('residual_times', None)
                            if _rt is not None and _rt.numel() > 0:
                                t_residuals_per_pt[_i].append(_rt.detach().cpu())

        # Compile and truncate to llr_iterations hits per detector
        t_residuals_compiled = []
        for _i in range(n_points):
            _parts = t_residuals_per_pt[_i]
            if _parts:
                _all_rt = torch.cat(_parts, dim=0)[:llr_iterations]
            else:
                _all_rt = torch.tensor([], dtype=torch.float32)
            t_residuals_compiled.append(_all_rt.to(device))

        # Grad phase: per-detector Fisher via jacrev (only mode supported for PATD)
        # Vectorize across point chunks using vmap to avoid sequential loop
        fisher_per_point = torch.zeros(n_points, total_dims, total_dims, device=device)
        
        # Determine point chunk size for vectorization
        pt_chunk_patd = point_chunk_size if point_chunk_size is not None else n_points
        
        # Helper to process one point given its residual times
        def _process_point_chunk(pts_chunk, t_residuals_chunk, n_hits_chunk):
            """
            Compute Jacobians and Fishers for a chunk of points via jacrev.
            pts_chunk: (B, 3) or (3,) if B=1
            t_residuals_chunk: list of (N_i,) tensors, one per point
            n_hits_chunk: (B,) tensor with hit counts per point
            Returns: (B, D, D) fisher matrix
            """
            B = pts_chunk.shape[0] if pts_chunk.dim() > 1 else 1
            pts_chunk = pts_chunk.reshape(B, 3) if pts_chunk.dim() == 1 else pts_chunk
            
            fishers_chunk = torch.zeros(B, total_dims, total_dims, device=device)
            
            for _b in range(B):
                _t_res = t_residuals_chunk[_b]
                if _t_res.numel() == 0:
                    continue

                _n_hits = int(n_hits_chunk[_b].item())
                _t_fixed = _t_res
                _pt_fixed = pts_chunk[_b]

                def _patd_log_probs_fn(_theta_flat, _t=_t_fixed, _pt=_pt_fixed):
                    _params = _unflatten_theta(
                        _theta_flat, fisher_info_params=fisher_info_params,
                        theta_shapes=theta_shapes_patd, theta_numels=theta_numels_patd,
                        fixed_params=fixed_params,
                    )
                    return eval_patd_log_probs(
                        t_residuals_fixed=_t, opt_point=_pt, event_params=_params,
                    )

                if patd_autodiff_mode == 'jacrev':
                    # Reverse-mode Jacobian: J is (N_hits, D)
                    _J = jacrev(_patd_log_probs_fn, chunk_size=jacrev_chunk_size)(theta0_flat_patd)
                    _J = _J.reshape(_n_hits, total_dims)
                else:
                    # Forward-mode via JVPs: linearize then apply basis vectors
                    _basis_chunk = grad_chunk_size if grad_chunk_size is not None else total_dims
                    _ly0, _jvp_fn = linearize(_patd_log_probs_fn, theta0_flat_patd)
                    _cols_parts = []
                    for _d_start in range(0, total_dims, _basis_chunk):
                        _d_end = min(_d_start + _basis_chunk, total_dims)
                        _k = _d_end - _d_start
                        _bvecs = torch.zeros(_k, total_dims, device=device, dtype=theta0_flat_patd.dtype)
                        _bvecs[torch.arange(_k, device=device), torch.arange(_d_start, _d_end, device=device)] = 1
                        _cols_parts.append(vmap(_jvp_fn)(_bvecs))  # (k, N_hits)
                    _J = torch.cat(_cols_parts, dim=0).permute(1, 0).contiguous()  # (N_hits, D)
                    del _cols_parts, _jvp_fn, _ly0

                # Compute Fisher: outer product divided by hit count
                _F = torch.einsum('li,lj->ij', _J, _J) / _n_hits
                fishers_chunk[_b] = _F.detach() if detach_fisher_tensors else _F
            
            return fishers_chunk
        
        # Process point chunks
        for p_start in range(0, n_points, pt_chunk_patd):
            p_end = min(p_start + pt_chunk_patd, n_points)
            pts_chunk = point[p_start:p_end]
            t_res_chunk = t_residuals_compiled[p_start:p_end]
            n_hits_chunk = torch.tensor([int(t.numel()) for t in t_res_chunk], device=device, dtype=torch.float32)
            
            # Process chunk
            fishers_chunk = _process_point_chunk(pts_chunk, t_res_chunk, n_hits_chunk)
            fisher_per_point[p_start:p_end] = fishers_chunk
            _fisher_chunk_cleanup(device)
        
        # Scale each detector's Fisher by its mean photon count
        fisher_per_point = fisher_per_point * mean_charges.view(n_points, 1, 1)

    elif llr_net is None:
        # ---- surrogate-only path -----------------------------------------------
        # The surrogate directly returns the Poisson mean λ.
        # We mirror the LLR-mode principle: build a Jacobian J of the vector output
        # (one λ per point) w.r.t. the event parameters, then form per-point Fishers.
        # Note: for a Poisson mean parameterization, Fisher per point is
        #   F_i = (∂λ_i/∂θ)(∂λ_i/∂θ)^T / λ_i

        llr_autodiff_mode_local = (llr_autodiff_mode or 'jacrev').lower()
        if llr_autodiff_mode_local not in {'jacrev', 'jvp'}:
            raise ValueError(
                f"Unsupported llr_autodiff_mode={llr_autodiff_mode!r} for surrogate-only path; "
                "expected 'jacrev' or 'jvp'."
            )

        theta_shapes = [event_params[p].detach().to(device).shape for p in fisher_info_params]
        theta_numels = [int(event_params[p].detach().to(device).numel()) for p in fisher_info_params]
        theta0_flat = torch.cat([event_params[p].detach().to(device).reshape(-1) for p in fisher_info_params], dim=0)
        total_dims_local = int(theta0_flat.numel())

        pt_chunk = point_chunk_size if point_chunk_size is not None else n_points
        fisher_per_point = torch.zeros(n_points, total_dims_local, total_dims_local, device=device)

        def _surrogate_batched(pts_3, params_dict):
            """Evaluate surrogate over a point batch; fall back to a point loop."""
            try:
                ly_b = surrogate_func(opt_point=pts_3, event_params=params_dict)
                if not isinstance(ly_b, torch.Tensor):
                    raise TypeError
                ly_b = ly_b.reshape(-1)
                if ly_b.numel() != pts_3.shape[0]:
                    raise ValueError
                return ly_b
            except Exception:
                ly_list = []
                for i in range(pts_3.shape[0]):
                    ly_list.append(surrogate_func(opt_point=pts_3[i], event_params=params_dict))
                return torch.stack(ly_list).reshape(-1)

        for p_start in range(0, n_points, pt_chunk):
            p_end = min(p_start + pt_chunk, n_points)
            pts = point[p_start:p_end]
            B = pts.shape[0]

            def _theta_only_fn(theta_flat):
                params = _unflatten_theta(
                    theta_flat,
                    fisher_info_params=fisher_info_params,
                    theta_shapes=theta_shapes,
                    theta_numels=theta_numels,
                    fixed_params=fixed_params,
                )
                return _surrogate_batched(pts, params)

            if llr_autodiff_mode_local == 'jacrev':
                # Reverse-mode Jacobian: J is (B, D)
                J = jacrev(_theta_only_fn, chunk_size=jacrev_chunk_size)(theta0_flat)
                if not isinstance(J, torch.Tensor):
                    J = torch.as_tensor(J, device=device)
                J = J.reshape(B, total_dims_local)
                if detach_fisher_tensors:
                    ly_vals = _theta_only_fn(theta0_flat).detach().reshape(-1)
                else:
                    ly_vals = _theta_only_fn(theta0_flat).reshape(-1)

            else:
                # Forward-mode Jacobian via JVPs.
                # The surrogate may contain Poisson draws which linearize would
                # capture and vmap would then error on. Hoist a single surrogate
                # evaluation outside linearize to get the mean λ, then
                # differentiate a surrogate that reuses those fixed mean values.
                # For the Fisher information of a Poisson mean, what matters is
                # ∂λ/∂θ evaluated at θ0, not the stochastic draw — so holding
                # the Poisson draw fixed is the correct thing to do.
                with torch.no_grad():
                    ly_vals_fixed = _surrogate_batched(pts, _unflatten_theta(
                        theta0_flat,
                        fisher_info_params=fisher_info_params,
                        theta_shapes=theta_shapes,
                        theta_numels=theta_numels,
                        fixed_params=fixed_params,
                    )).detach()  # (B,)

                def _theta_only_fn_det(theta_flat):
                    # Differentiable wrapper that avoids re-calling the stochastic
                    # surrogate: instead re-evaluates only the mean λ(θ).
                    # We rely on the surrogate's mean being differentiable w.r.t. θ
                    # even when the Poisson draw is removed.
                    params_det = _unflatten_theta(
                        theta_flat,
                        fisher_info_params=fisher_info_params,
                        theta_shapes=theta_shapes,
                        theta_numels=theta_numels,
                        fixed_params=fixed_params,
                    )
                    return _surrogate_batched(pts, params_det)

                basis_chunk_size = grad_chunk_size if grad_chunk_size is not None else total_dims_local
                ly_vals = ly_vals_fixed.reshape(-1)
                _, jvp_fn = linearize(_theta_only_fn_det, theta0_flat)

                cols_parts = []
                for d_start in range(0, total_dims_local, basis_chunk_size):
                    d_end = min(d_start + basis_chunk_size, total_dims_local)
                    k = d_end - d_start
                    basis_chunk = torch.zeros(k, total_dims_local, device=device, dtype=theta0_flat.dtype)
                    rows = torch.arange(k, device=device)
                    cols_idx = torch.arange(d_start, d_end, device=device)
                    basis_chunk[rows, cols_idx] = 1
                    cols_chunk = vmap(jvp_fn)(basis_chunk)  # (k, B)
                    cols_parts.append(cols_chunk)

                cols = torch.cat(cols_parts, dim=0)  # (D, B)
                J = cols.permute(1, 0).contiguous()  # (B, D)
                del cols_parts, cols, jvp_fn

            # Per-point Fisher (Poisson mean): outer / lambda
            outer = torch.bmm(J.unsqueeze(-1), J.unsqueeze(-2))  # (B, D, D)
            outer = outer / ly_vals.clamp(min=1e-10).view(B, 1, 1)
            fisher_per_point[p_start:p_end] = outer.detach() if detach_fisher_tensors else outer
            del J, outer, ly_vals, pts

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
            detach_fisher_tensors=detach_fisher_tensors,
            use_rich_features=use_rich_features,
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

