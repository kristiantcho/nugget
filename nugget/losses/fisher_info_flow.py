"""Fisher-information resolution loss for the hit / light-yield / arrival-time surrogates.

A light-weight counterpart to ``WeightedResolutionLoss`` in ``fisher_info.py``: same
``__call__(geom_dict, **kwargs)`` contract and the same return dict, but the
likelihood is the learned hurdle model rather than LLRnet/LightSabre, and the knob
count is kept small.

The Fisher matrix
----------------
Per PMT j the observation is (hit?, light yield, arrival times) with

    p(y, q, {t} | th) = (1 - pi)                      if y = 0
                      = pi . p_q(q) . prod_k p_t(t_k) if y = 1

The score is  s = (y - pi) grad l + y [grad log p_q + sum_k grad log p_t],
l = logit(pi). Every cross term in E[s s^T] vanishes because E[grad log p] = 0 for a
normalised density -- including the q-t one, where E_q[q E_t[grad log p_t] ...] = 0.
So the three pieces simply add:

    F_j = pi (1 - pi) grad l grad l^T                       (hit)
        + pi      E_q[ grad log p_q grad log p_q^T ]        (light yield)
        + pi qbar E_t[ grad log p_t grad log p_t^T ]        (arrival time)

with qbar = E[q | hit]. The Bernoulli term is closed form; it is written with the
logit rather than as grad pi grad pi^T / (pi (1 - pi)) because pi ~ 1e-4 makes the
divided form numerically hopeless. qbar is evaluated at the true parameters and is
NOT differentiated -- the information carried by q itself is already the second term.

The expectations are done by quadrature on a fixed grid in the flow's own z, with
weights p_z(z) dz. For the light yield z is a theta-independent transform of q, so
the change of variables leaves the Fisher untouched; for the arrival time z depends
on theta through t_geom, but gridding z at th0 and mapping to fixed t_hit leaves the
Jacobians cancelling inside the integral. qbar falls out of the same grid.

Gradients
---------
``log_prob_*(differentiable=True)`` is required: the default path detaches inside
``_v_and_div`` and returns a constant, so a Fisher built on it is identically zero.

Rows of a batch are independent, so ONE backward pass of ``sum_i log p_i`` yields
d log p_i / d c_i for every row at once; the chain to theta then goes through the
Jacobian of ``build_context``, which is cheap and pure-torch (``jacfwd``). This
costs one backward per batch regardless of how many PMTs or quadrature nodes it
holds. Note that ``jacfwd`` cannot be wrapped around the flow itself -- nesting it
over the ``jvp``-based divergence gives silently wrong numbers.
"""

import gc
import math

import numpy as np
import torch

from nugget.losses.base_loss import LossFunction


def _as_t(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _free(device):
    gc.collect()
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def _log_prob_z(model, z, ctx, n_steps, div_eps=None):
    """log p(z | c), differentiable in ctx.

    div_eps=None uses the model's autograd divergence: exact, and the fastest
    option measured -- but it needs create_graph=True, and torch.compile refuses
    ("does not currently support double backward"). Setting div_eps switches to a
    central difference for dv/dz, which is first-order-only and therefore
    compilable, at the cost of two extra network evaluations per ODE step.
    """
    if div_eps is None:
        return model.log_prob_z(z, ctx, n_steps=n_steps, differentiable=True)
    c = model._apply_context_norm(model._prep(ctx))
    zz = model._prep(z).reshape(-1, 1)
    B = zz.shape[0]
    div = torch.zeros_like(zz)
    dt = 1.0 / n_steps
    for i in reversed(range(n_steps)):
        tm = torch.full((B,), (i + 0.5) * dt, device=zz.device, dtype=zz.dtype)
        v = model._velocity(zz, tm, c)
        vp = model._velocity(zz + div_eps, tm, c)
        vm = model._velocity(zz - div_eps, tm, c)
        zz = zz - dt * v
        div = div + dt * (vp - vm) / (2.0 * div_eps)
    return (-0.5 * zz ** 2 - 0.5 * math.log(2.0 * math.pi) - div).reshape(-1)


def _log_prob_tres(at, t_res, ctx, n_steps, div_eps=None):
    """log p(t_res | c) = log p_z(z) + log|dz/dt_res|."""
    t = at._prep(t_res).reshape(-1)
    return _log_prob_z(at, at.to_z(t), ctx, n_steps, div_eps) + at.log_det_dz_dq(t)


class FlowFisherResolutionLoss(LossFunction):
    """Angular / energy resolution from the learned hurdle likelihood.

    Parameters
    ----------
    hit_model : HitClassifier
    ly_model : FlowMatchLY
    atime_model : FlowMatchATime or None
    mode : {'all', 'atime'}
        'atime' uses only the arrival-time Fisher term, but still evaluates pi and
        qbar because they weight it.
    fisher_info_params : sequence of str
        Scanned parameters, from {'energy', 'zenith', 'azimuth'}. Energy is
        differentiated w.r.t. log10(E), which is what the resolution is usually
        quoted in and keeps the matrix well conditioned.
    n_quad, z_range : int, (float, float)
        Quadrature grid in the flow's z. 48 nodes over [-5, 5] is ample.
    n_steps : int
        ODE steps per log-prob. The gradient path holds all of them in memory.
    pmt_directions, geometry_csv_path, n_pmt_per_om
        The per-OM PMT template; defaults to the geometry the models were trained on.
    """

    def __init__(self, hit_model=None, ly_model=None, atime_model=None,
                 device=None, fisher_info_params=('energy', 'zenith', 'azimuth'),
                 resolution_type='angular', mode='all',
                 n_quad=48, z_range=(-5.0, 5.0), n_steps=8,
                 pmt_directions=None, geometry_csv_path=None, n_pmt_per_om=None,
                 div_mode='autograd', div_eps=1e-6, use_torch_compile=False,
                 torch_compile_kwargs=None, print_loss=False):
        super().__init__(device=device)
        if mode not in ('all', 'atime'):
            raise ValueError("mode must be 'all' or 'atime'")
        if resolution_type not in ('angular', 'energy'):
            raise ValueError("resolution_type must be 'angular' or 'energy'")
        bad = [p for p in fisher_info_params
               if p not in ('energy', 'zenith', 'azimuth')]
        if bad:
            raise ValueError(f"unsupported fisher_info_params: {bad}")

        self.hit_model = hit_model
        self.ly_model = ly_model
        self.atime_model = atime_model
        self.mode = mode
        self.resolution_type = resolution_type
        self.fisher_info_params = list(fisher_info_params)
        self.n_quad = int(n_quad)
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.n_steps = int(n_steps)
        self.print_loss = bool(print_loss)
        if div_mode not in ('autograd', 'fd'):
            raise ValueError("div_mode must be 'autograd' or 'fd'")
        if use_torch_compile and div_mode != 'fd':
            div_mode = 'fd'
            print("FlowFisherResolutionLoss: torch.compile needs div_mode='fd' "
                  "(the autograd divergence uses double backward, which compile "
                  "rejects) -- switching to 'fd'.")
        self.div_mode = div_mode
        self.div_eps = float(div_eps) if div_mode == 'fd' else None
        self.use_torch_compile = bool(use_torch_compile)

        if hit_model is None or ly_model is None:
            raise ValueError("a hit model and a light-yield model are both required "
                             "(pi and qbar weight every term)")
        if mode == 'atime' and atime_model is None:
            raise ValueError("mode='atime' needs an arrival-time model")

        self.dtype = hit_model.param_dtype
        self._pmt_dirs = self._resolve_pmt_template(
            pmt_directions, geometry_csv_path, n_pmt_per_om)
        if self.use_torch_compile:
            kw = dict(dynamic=False)
            kw.update(torch_compile_kwargs or {})
            for m in (hit_model, ly_model, atime_model):
                if m is not None and getattr(m, 'net', None) is not None:
                    m.net = torch.compile(m.net, **kw)

        zs = torch.linspace(self.z_range[0], self.z_range[1], self.n_quad,
                            device=self.device, dtype=self.dtype)
        self._zq = zs
        self._dz = float((self.z_range[1] - self.z_range[0]) / max(self.n_quad - 1, 1))

    # ------------------------------------------------------------------ setup

    def _resolve_pmt_template(self, pmt_directions, geometry_csv_path, n_pmt_per_om):
        from nugget.surrogates.NuSmoothie import NuSmoothie, DEFAULT_GEOMETRY
        if pmt_directions is not None:
            d = torch.as_tensor(pmt_directions, dtype=self.dtype).reshape(-1, 3)
        else:
            d = torch.as_tensor(
                NuSmoothie.pmt_directions_from_geometry(
                    geometry_csv_path or DEFAULT_GEOMETRY), dtype=self.dtype)
        if n_pmt_per_om is not None:
            d = d[:int(n_pmt_per_om)]
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return d.to(self.device)

    # ------------------------------------------------------- theta <-> context

    def _theta0(self, ev):
        """Pack one event's parameters into the scanned theta vector."""
        vals = []
        for p in self.fisher_info_params:
            v = float(_as_t(ev[p], self.device, self.dtype).reshape(-1)[0])
            vals.append(math.log10(max(v, 1e-30)) if p == 'energy' else v)
        return torch.tensor(vals, device=self.device, dtype=self.dtype)

    def _ctx_builder(self, model, pts, dirs, vertex, ev):
        """theta -> (M, context_dim), out of place so jacfwd can differentiate it."""
        M = pts.shape[0]
        fixed = {p: float(_as_t(ev[p], self.device, self.dtype).reshape(-1)[0])
                 for p in ('energy', 'zenith', 'azimuth')}
        idx = {p: i for i, p in enumerate(self.fisher_info_params)}

        def build(theta):
            e = (10.0 ** theta[idx['energy']] if 'energy' in idx
                 else torch.tensor(fixed['energy'], device=self.device, dtype=self.dtype))
            z = (theta[idx['zenith']] if 'zenith' in idx
                 else torch.tensor(fixed['zenith'], device=self.device, dtype=self.dtype))
            a = (theta[idx['azimuth']] if 'azimuth' in idx
                 else torch.tensor(fixed['azimuth'], device=self.device, dtype=self.dtype))
            return model.build_context(
                pts, vertex.reshape(1, 3).expand(M, 3), e.reshape(1).expand(M),
                z.reshape(1).expand(M), a.reshape(1).expand(M), pmt_directions=dirs)

        return build

    # ------------------------------------------------------------- score terms

    def _scores(self, model, build, theta, z_nodes, chunk):
        """d log p_z(z_k | c_j) / d theta for every (PMT j, node k).

        One backward gives d log p / d c for all rows -- they are independent -- and
        the chain to theta goes through jacfwd of build_context.

        Returns (logw, g) with logw the unnormalised log weights (M, K) and
        g the scores (M, K, P).
        """
        M = int(build(theta).shape[0])
        K = z_nodes.shape[0]
        P = theta.shape[0]
        Jc = torch.func.jacfwd(build)(theta)                       # (M, C, P)
        ctx0 = build(theta).detach()

        logw = torch.empty(M, K, device=self.device, dtype=self.dtype)
        g = torch.empty(M, K, P, device=self.device, dtype=self.dtype)
        rep = max(int(chunk) // max(K, 1), 1)
        for s in range(0, M, rep):
            e = min(s + rep, M)
            n = e - s
            c = ctx0[s:e].repeat_interleave(K, 0).detach().requires_grad_(True)
            zz = z_nodes.repeat(n)
            lp = _log_prob_z(model, zz, c, self.n_steps, self.div_eps)
            G = torch.autograd.grad(lp.sum(), c)[0].reshape(n, K, -1)
            logw[s:e] = lp.detach().reshape(n, K)
            g[s:e] = torch.einsum('mkc,mcp->mkp', G, Jc[s:e])
            del c, lp, G
        return logw, g

    @staticmethod
    def _weights(logw):
        """p_z(z_k) dz normalised over the grid -- robust to an imperfect flow norm."""
        w = torch.softmax(logw, dim=1)
        return w

    def _outer(self, w, g):
        """sum_k w_k g_k g_k^T  ->  (M, P, P)."""
        return torch.einsum('mk,mkp,mkq->mpq', w, g, g)

    # ---------------------------------------------------------- per-event core

    def _fisher_per_pmt(self, pts, dirs, ev, chunk):
        """(M, P, P) Fisher contribution of each PMT for one event."""
        theta = self._theta0(ev)
        vertex = _as_t(ev['position'], self.device, self.dtype).reshape(3)
        P = theta.shape[0]
        M = pts.shape[0]

        # ---- hit term: closed form, pi (1-pi) grad l grad l^T ----
        build_h = self._ctx_builder(self.hit_model, pts, dirs, vertex, ev)
        Jh = torch.func.jacfwd(build_h)(theta)                     # (M, C, P)
        ch = build_h(theta).detach().requires_grad_(True)
        ell = self.hit_model.predict_hit_logit(ch, calibrated=True).reshape(-1)
        gl_c = torch.autograd.grad(ell.sum(), ch)[0]               # (M, C)
        gl = torch.einsum('mc,mcp->mp', gl_c, Jh)                  # (M, P)
        pi = torch.sigmoid(ell.detach()).clamp(1e-12, 1 - 1e-12)   # (M,)
        F = (pi * (1.0 - pi)).reshape(M, 1, 1) * gl.unsqueeze(2) * gl.unsqueeze(1)

        # ---- light yield: weights, qbar, and (unless mode='atime') its Fisher ----
        build_l = self._ctx_builder(self.ly_model, pts, dirs, vertex, ev)
        logw_q, g_q = self._scores(self.ly_model, build_l, theta, self._zq, chunk)
        w_q = self._weights(logw_q)                                # (M, K)
        qvals = self.ly_model.from_z(self._zq).reshape(1, -1)      # (1, K)
        qbar = (w_q * qvals).sum(1).clamp_min(1.0)                 # (M,)
        if self.mode == 'all':
            F = F + pi.reshape(M, 1, 1) * self._outer(w_q, g_q)
        del logw_q, g_q

        # ---- arrival time ----
        if self.atime_model is not None:
            at = self.atime_model
            build_t = self._ctx_builder(at, pts, dirs, vertex, ev)
            # grid z at the true theta, map to fixed t_hit, then let t_res move with
            # theta through t_geom -- that dependence is where the timing information
            # on direction actually lives.
            with torch.no_grad():
                tg0 = at.geometric_time(
                    pts, vertex.reshape(1, 3).expand(M, 3),
                    zeniths=_as_t(ev['zenith'], self.device, self.dtype).reshape(1).expand(M),
                    azimuths=_as_t(ev['azimuth'], self.device, self.dtype).reshape(1).expand(M),
                ).reshape(M, 1)
                t_hit = at.from_z(self._zq).reshape(1, -1) + tg0   # (M, K)
            logw_t, g_t = self._scores_time(at, build_t, theta, t_hit, pts, vertex,
                                            ev, chunk)
            w_t = self._weights(logw_t)
            F = F + (pi * qbar).reshape(M, 1, 1) * self._outer(w_t, g_t)
            del logw_t, g_t
        return F

    def _scores_time(self, at, build, theta, t_hit, pts, vertex, ev, chunk):
        """Arrival-time scores, including the theta-dependence of t_geom."""
        M, K = t_hit.shape
        P = theta.shape[0]
        idx = {p: i for i, p in enumerate(self.fisher_info_params)}
        fixed = {p: float(_as_t(ev[p], self.device, self.dtype).reshape(-1)[0])
                 for p in ('zenith', 'azimuth')}

        def t_res_of(theta_, sl):
            z = (theta_[idx['zenith']] if 'zenith' in idx
                 else torch.tensor(fixed['zenith'], device=self.device, dtype=self.dtype))
            a = (theta_[idx['azimuth']] if 'azimuth' in idx
                 else torch.tensor(fixed['azimuth'], device=self.device, dtype=self.dtype))
            n = sl.stop - sl.start
            tg = at.geometric_time(pts[sl], vertex.reshape(1, 3).expand(n, 3),
                                   zeniths=z.reshape(1).expand(n),
                                   azimuths=a.reshape(1).expand(n)).reshape(n, 1)
            return (t_hit[sl] - tg).reshape(-1)

        Jc = torch.func.jacfwd(build)(theta)                       # (M, C, P)
        ctx0 = build(theta).detach()
        logw = torch.empty(M, K, device=self.device, dtype=self.dtype)
        g = torch.empty(M, K, P, device=self.device, dtype=self.dtype)
        rep = max(int(chunk) // max(K, 1), 1)
        for s in range(0, M, rep):
            e = min(s + rep, M)
            n = e - s
            sl = slice(s, e)
            Jt = torch.func.jacfwd(lambda th: t_res_of(th, sl))(theta)  # (n*K, P)
            tr = t_res_of(theta, sl).detach().requires_grad_(True)
            c = ctx0[sl].repeat_interleave(K, 0).detach().requires_grad_(True)
            lp = _log_prob_tres(at, tr, c, self.n_steps, self.div_eps)
            gc_, gt_ = torch.autograd.grad(lp.sum(), (c, tr))
            logw[sl] = lp.detach().reshape(n, K)
            g[sl] = (torch.einsum('mkc,mcp->mkp', gc_.reshape(n, K, -1), Jc[sl])
                     + (gt_.reshape(n, K, 1) * Jt.reshape(n, K, P)))
            del c, tr, lp, gc_, gt_, Jt
        return logw, g

    # ------------------------------------------------------------------- driver

    def compute_fisher_per_string_per_event(self, string_xy, points_3d,
                                            signal_event_params, chunk=8192,
                                            empty_cache_after_event=False,
                                            verbose=False):
        """(n_events, n_strings, P, P).

        Every OM in ``points_3d`` is expanded into the PMT template; a point belongs
        to the string whose (x, y) it matches exactly, as in ``fisher_info.py``.
        """
        pts3 = _as_t(points_3d, self.device, self.dtype).reshape(-1, 3)
        n_pts = pts3.shape[0]
        K_pmt = self._pmt_dirs.shape[0]
        pts = pts3.repeat_interleave(K_pmt, 0)                     # (n_pts*K, 3)
        dirs = self._pmt_dirs.repeat(n_pts, 1)
        P = len(self.fisher_info_params)
        n_ev = len(signal_event_params)

        if string_xy is None:
            sel = [torch.arange(n_pts, device=self.device)]
        else:
            sxy = torch.stack([torch.stack([_as_t(s[0], self.device, self.dtype),
                                            _as_t(s[1], self.device, self.dtype)])
                               for s in string_xy]).detach()
            hit = ((pts3[:, 0].unsqueeze(0) == sxy[:, 0].unsqueeze(1)) &
                   (pts3[:, 1].unsqueeze(0) == sxy[:, 1].unsqueeze(1)))
            sel = [h.nonzero(as_tuple=True)[0] for h in hit]
        n_str = len(sel)

        out = torch.zeros(n_ev, n_str, P, P, device=self.device, dtype=self.dtype)
        for i, ev in enumerate(signal_event_params):
            F_pmt = self._fisher_per_pmt(pts, dirs, ev, chunk)     # (n_pts*K, P, P)
            F_om = F_pmt.reshape(n_pts, K_pmt, P, P).sum(1)        # (n_pts, P, P)
            for s, ids in enumerate(sel):
                if ids.numel():
                    out[i, s] = F_om[ids].sum(0)
            del F_pmt, F_om
            if empty_cache_after_event:
                _free(self.device)
            if verbose:
                print(f'  Fisher: event {i + 1}/{n_ev}', flush=True)
        return out

    # --------------------------------------------------------------- resolution

    def _resolution(self, F, signal_event_params, use_relative_energy):
        names = self.fisher_info_params
        n = F.shape[0]
        eye = torch.eye(F.shape[-1], device=self.device, dtype=F.dtype)
        try:
            cov = torch.linalg.inv(F + 1e-20 * eye)
        except Exception:
            cov = torch.linalg.pinv(F + 1e-20 * eye)
        if self.resolution_type == 'angular':
            iz, ia = names.index('zenith'), names.index('azimuth')
            zen = torch.stack([_as_t(p['zenith'], self.device, self.dtype).reshape(())
                               for p in signal_event_params])
            var = (cov[:, iz, iz]
                   + torch.sin(zen) ** 2 * cov[:, ia, ia]
                   + 2.0 * torch.sin(zen) * cov[:, iz, ia])
            return torch.sqrt(var.clamp_min(0.0))
        ie = names.index('energy')
        # theta uses log10 E, so sqrt(var) is already a relative (dex) resolution
        res = torch.sqrt(cov[:, ie, ie].clamp_min(0.0))
        return res if use_relative_energy else res * math.log(10.0)

    def __call__(self, geom_dict, **kwargs):
        """Same contract as ``WeightedResolutionLoss.__call__``.

        geom_dict : 'points_3d', optionally 'string_xy' and 'string_weights'.
        kwargs    : 'signal_event_params' or ('signal_sampler', 'num_events'),
                    'fisher_res_metric' ('fom' | 'median' | 'mean'),
                    'use_relative_energy', 'empty_cache_after_event',
                    'precomputed_fisher_info_per_string_per_event',
                    'fisher_info_chunk_size', 'verbose'.
        """
        points_3d = geom_dict.get('points_3d', None)
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)

        params = kwargs.get('signal_event_params', None)
        sampler = kwargs.get('signal_sampler', None)
        num_events = int(kwargs.get('num_events', 100))
        metric = kwargs.get('fisher_res_metric', 'fom')
        use_rel_e = bool(kwargs.get('use_relative_energy', False))
        empty_cache = bool(kwargs.get('empty_cache_after_event', False))
        chunk = int(kwargs.get('fisher_info_chunk_size', 8192))
        verbose = bool(kwargs.get('verbose', False))
        precomp = kwargs.get('precomputed_fisher_info_per_string_per_event', None)

        if params is None:
            if sampler is None:
                raise ValueError("provide signal_event_params or a signal_sampler")
            params = sampler.sample_events(num_events)

        if precomp is None:
            F_str = self.compute_fisher_per_string_per_event(
                string_xy, points_3d, params, chunk=chunk,
                empty_cache_after_event=empty_cache, verbose=verbose)
        else:
            F_str = precomp.to(device=self.device, dtype=self.dtype)

        if string_weights is None:
            F = F_str.sum(dim=1)
        else:
            w = torch.sigmoid(_as_t(string_weights, self.device, self.dtype))
            F = (w.reshape(1, -1, 1, 1) * F_str).sum(dim=1)
        if empty_cache:
            _free(self.device)

        res = self._resolution(F, params, use_rel_e)
        ok = torch.isfinite(res) & (res > 1e-15)
        if ok.any():
            clean = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            safe = torch.where(ok, clean.clamp_min(1e-15),
                               torch.full_like(res, 1e6))
            if metric == 'fom':
                total = 1.0 / torch.sqrt(torch.mean(1.0 / safe ** 2))
            elif metric == 'median':
                total = torch.median(safe)
            else:
                total = torch.mean(safe)
        else:
            total = torch.tensor(1.0, device=self.device, dtype=self.dtype,
                                 requires_grad=True)
        if self.print_loss:
            print(f'FlowFisherResolutionLoss[{self.mode}]: {total.item():.6g} '
                  f'({int(ok.sum())}/{len(res)} events usable)')

        key = ('angular_resolution' if self.resolution_type == 'angular'
               else 'energy_resolution')
        return {f'{key}_loss': total, f'{key}_per_event': res,
                'resolution_params': params,
                'fisher_info_per_string_per_event': F_str.detach()}
