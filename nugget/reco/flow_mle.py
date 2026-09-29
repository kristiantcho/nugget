"""Maximum-likelihood track reconstruction on the learned hit / light-yield / time models."""
import math

import numpy as np
import torch
import torch.nn.functional as Fn

# Fit parameters. dir_a, dir_b are offsets in the tangent plane of the seed (true)
# direction, in radians to first order, so they have no pole singularity.
PARAMS = ('x', 'y', 'z', 'dir_a', 'dir_b', 'log10_energy', 't0')

F64 = torch.float64

# Default spread of a random fit start: m, degrees, decades, ns.
START_SIGMA = {'vertex': 20.0, 'direction_deg': 5.0, 'log10_energy': 0.3, 't0': 20.0}


def _as_t(x, device, dtype=F64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _dir_for(model, travel):
    """Direction in the model's training convention (see NuSmoothie._dir_for)."""
    return -travel if getattr(model, 'track_dir_is_arrival', False) else travel


def _log_prob_z(model, z, ctx, n_steps, div_eps=None, need_grad=True):
    """log p(z | c). div_eps=None: exact autograd divergence; otherwise central
    differences, one derivative order lower, so theta-gradients and Hessians are cheaper."""
    if div_eps is None:
        return model.log_prob_z(z, ctx, n_steps=n_steps, differentiable=need_grad)
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


@torch.no_grad()
def _push(model, z0, ctx, n_steps):
    """Base noise z0 -> data-space z, with the same midpoint scheme as sample_z."""
    c = model._apply_context_norm(model._prep(ctx))
    z = model._prep(z0).reshape(-1, 1)
    B = z.shape[0]
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = torch.full((B,), i * dt, device=z.device, dtype=z.dtype)
        tm = torch.full((B,), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
        k1 = model._velocity(z, t0, c)
        z = z + dt * model._velocity(z + 0.5 * dt * k1, tm, c)
    return z.reshape(-1)


class FlowEventLikelihood:
    """Sample one detector response per event and fit it by maximum likelihood.

    Any ``points_3d`` of optical modules; each is expanded into the PMT template
    (16 PMTs from the training geometry by default), as in FlowFisherResolutionLoss.

    Parameters
    ----------
    n_steps, sample_n_steps : int
        ODE steps for the likelihood and for sampling the response.
    div_mode : {'fd', 'autograd'}
        Divergence inside log p. 'fd' is one derivative order cheaper, which matters
        because the fit needs Hessians. div_eps=None picks 1e-3 (float32 nets) / 1e-5.
    n_dequant_nodes : int
        Gauss-Legendre nodes for log P(q) = log int_0^1 p(q+u) du (deterministic).
    noise_frac, noise_window : float
        Timing noise floor p = (1-eps) p_T + eps / window; guards the fit against
        photons far outside the flow's support at a wrong hypothesis.
    prune_pi : float
        Unhit PMTs with P(hit) below this at the truth are left out of the no-hit
        term; the expected hits dropped are reported per response.
    chunk : int
        Rows per backward; bounds memory for the value, gradient and Hessian.
    """

    def __init__(self, hit_model, ly_model, atime_model=None, device=None,
                 pmt_directions=None, geometry_csv_path=None, n_steps=16,
                 sample_n_steps=64, div_mode='fd', div_eps=None, n_dequant_nodes=4,
                 noise_frac=1e-4, noise_window=1e4, prune_pi=1e-6, chunk=8192):
        if div_mode not in ('fd', 'autograd'):
            raise ValueError("div_mode must be 'fd' or 'autograd'")
        self.hit_model, self.ly_model, self.atime_model = hit_model, ly_model, atime_model
        self.device = torch.device(device if device is not None else hit_model.device)
        self.n_steps, self.sample_n_steps = int(n_steps), int(sample_n_steps)
        self.div_mode, self.div_eps = div_mode, div_eps
        self.noise_frac, self.noise_window = float(noise_frac), float(noise_window)
        self.prune_pi, self.chunk = float(prune_pi), max(int(chunk), 1)

        if pmt_directions is None:
            from nugget.surrogates.NuSmoothie import NuSmoothie, DEFAULT_GEOMETRY
            pmt_directions = NuSmoothie.pmt_directions_from_geometry(
                geometry_csv_path or DEFAULT_GEOMETRY)
        d = torch.as_tensor(np.asarray(pmt_directions), dtype=F64).reshape(-1, 3)
        self.pmt_dirs = (d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(self.device)

        x, w = np.polynomial.legendre.leggauss(int(n_dequant_nodes))
        self._u = torch.as_tensor((x + 1.0) / 2.0, dtype=F64, device=self.device)
        self._logw_u = torch.as_tensor(np.log(w / 2.0), dtype=F64, device=self.device)

    @classmethod
    def from_nusmoothie(cls, ns, **kw):
        kw.setdefault('device', ns.device)
        return cls(ns.hit_model, ns.ly_model, ns.atime_model, **kw)

    # ------------------------------------------------------------------ helpers

    def _eps(self, model):
        if self.div_mode != 'fd':
            return None
        if self.div_eps is not None:
            return float(self.div_eps)
        return 1e-3 if model.param_dtype == torch.float32 else 1e-5

    def expand(self, points_3d):
        """OMs (n, 3) -> PMT rows: positions (n*K, 3) and orientations (n*K, 3)."""
        pts3 = _as_t(points_3d, self.device).reshape(-1, 3)
        K = self.pmt_dirs.shape[0]
        return pts3.repeat_interleave(K, 0), self.pmt_dirs.repeat(pts3.shape[0], 1)

    def frame(self, event, t_offset=0.0):
        """Seed theta (the true event) and the tangent basis at its travel direction.

        t_offset is subtracted from the true event time, matching a response whose
        clock was re-zeroed (sample(time_origin='first_hit')).
        """
        dev = self.device
        vert = _as_t(event['position'], dev).reshape(3)
        if event.get('direction') is not None:
            d = _as_t(event['direction'], dev).reshape(3)
        else:
            th = _as_t(event['zenith'], dev).reshape(())
            ph = _as_t(event['azimuth'], dev).reshape(())
            d = torch.stack([torch.sin(th) * torch.cos(ph),
                             torch.sin(th) * torch.sin(ph), torch.cos(th)])
        d = d / d.norm()
        helper = torch.tensor([1.0, 0.0, 0.0] if abs(float(d[0])) < 0.9
                              else [0.0, 1.0, 0.0], dtype=F64, device=dev)
        e1 = helper - (helper @ d) * d
        e1 = e1 / e1.norm()
        e2 = torch.linalg.cross(d, e1)
        E = float(_as_t(event['energy'], dev).reshape(-1)[0])
        t0 = float(_as_t(event.get('time', 0.0), dev).reshape(-1)[0]) - float(t_offset)
        theta0 = torch.tensor([float(vert[0]), float(vert[1]), float(vert[2]),
                               0.0, 0.0, math.log10(E), t0], dtype=F64, device=dev)
        return {'theta0': theta0, 'd0': d, 'e1': e1, 'e2': e2}

    @staticmethod
    def unpack(theta, fr):
        """Full theta (7,) -> vertex (3,), travel direction (3,), energy (), t0 ()."""
        d = fr['d0'] + theta[3] * fr['e1'] + theta[4] * fr['e2']
        return theta[0:3], d / d.norm(), 10.0 ** theta[5], theta[6]

    @staticmethod
    def _full(theta_free, fr, free_idx):
        return fr['theta0'].index_put((free_idx,), theta_free)

    def _ctx(self, model, pts, dirs, vertex, d, E):
        n = pts.shape[0]
        return model.build_context(
            pts, vertex.reshape(1, 3).expand(n, 3), E.reshape(1).expand(n),
            pmt_directions=dirs,
            directions=_dir_for(model, d).reshape(1, 3).expand(n, 3))

    # ----------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample(self, points_3d, event, generator=None, max_photons_per_pmt=None,
               time_origin='vertex'):
        """One detector response: hit flags, counts, photon times (+ their base noise).

        time_origin : 'vertex' keeps times relative to the muon at its vertex;
            'first_hit' re-zeros them to the earliest photon, as real data would be.
            The fit seeds t0 consistently either way, and with t0 free the result is
            identical. The shift is stored as resp['t_offset'].
        max_photons_per_pmt keeps a random subset per PMT with weight q / n_kept.
        That is an approximation of the full likelihood and inflates the fitted spread.
        """
        if time_origin not in ('vertex', 'first_hit'):
            raise ValueError("time_origin must be 'vertex' or 'first_hit'")
        pts, dirs = self.expand(points_3d)
        pts, dirs = pts.detach(), dirs.detach()
        fr = self.frame(event)
        vertex, d, E, t0 = self.unpack(fr['theta0'], fr)
        M, ck, dev = pts.shape[0], self.chunk, self.device

        pi = torch.empty(M, dtype=F64, device=dev)
        for s in range(0, M, ck):
            c = self._ctx(self.hit_model, pts[s:s + ck], dirs[s:s + ck], vertex, d, E)
            pi[s:s + ck] = self.hit_model.predict_hit_prob(c, calibrated=True).reshape(-1).to(F64)
        hit = torch.rand(M, dtype=F64, device=dev, generator=generator) < pi
        hid = torch.nonzero(hit, as_tuple=True)[0]

        counts = torch.zeros(M, dtype=F64, device=dev)
        for s in range(0, hid.numel(), ck):
            ids = hid[s:s + ck]
            c = self._ctx(self.ly_model, pts[ids], dirs[ids], vertex, d, E)
            counts[ids] = self.ly_model.sample_light_yield(
                c, n_steps=self.sample_n_steps, discrete=True,
                generator=generator).reshape(-1).to(F64)

        q = counts[hid]
        n_ph = q if max_photons_per_pmt is None else q.clamp(max=float(max_photons_per_pmt))
        n_ph = n_ph.long()
        owner = hid.repeat_interleave(n_ph)
        w = (q / n_ph.to(F64).clamp_min(1.0)).repeat_interleave(n_ph)
        N = owner.numel()
        z0 = torch.randn(N, dtype=F64, device=dev, generator=generator)
        t_hit = torch.empty(N, dtype=F64, device=dev)
        at = self.atime_model
        if at is not None:
            for s in range(0, N, ck):
                o = owner[s:s + ck]
                n = o.numel()
                c = self._ctx(at, pts[o], dirs[o], vertex, d, E)
                t_res = at.from_z(_push(at, z0[s:s + ck], c, self.sample_n_steps)).to(F64)
                tg = at.geometric_time(pts[o], vertex.reshape(1, 3).expand(n, 3),
                                       directions=_dir_for(at, d).reshape(1, 3).expand(n, 3))
                t_hit[s:s + ck] = t_res.reshape(-1) + tg.reshape(-1).to(F64) + t0

        # A constant shift of every hit time, absorbed exactly by t0 when it is free.
        t_offset = 0.0
        if time_origin == 'first_hit' and at is not None and N > 0:
            t_offset = float(t_hit.min())
            t_hit = t_hit - t_offset

        active = hit | (pi > self.prune_pi)
        return {'points_3d': points_3d, 'event': event, 't_offset': t_offset,
                'hit': hit, 'counts': counts, 'active': active, 'pi_true': pi,
                'pruned_expected_hits': float(pi[~active].sum()),
                'ph_owner': owner, 'ph_t': t_hit, 'ph_w': w, 'ph_z0': z0,
                'n_hits': int(hid.numel()), 'n_photons': float(q.sum())}

    # --------------------------------------------------------------- likelihood

    def _units(self, resp):
        """Work units of about `chunk` rows: ('hit'|'ly'|'t', indices)."""
        ck = self.chunk
        act = torch.nonzero(resp['active'], as_tuple=True)[0]
        hid = torch.nonzero(resp['hit'], as_tuple=True)[0]
        pm = max(ck // self._u.numel(), 1)
        units = [('hit', act[s:s + ck]) for s in range(0, act.numel(), ck)]
        units += [('ly', hid[s:s + pm]) for s in range(0, hid.numel(), pm)]
        if self.atime_model is not None:
            ar = torch.arange(resp['ph_t'].numel(), device=self.device)
            units += [('t', ar[s:s + ck]) for s in range(0, ar.numel(), ck)]
        return units

    def _unit_loglik(self, kind, ids, theta, fr, resp, pts, dirs, need_grad):
        vertex, d, E, t0 = self.unpack(theta, fr)
        if kind == 'hit':
            c = self._ctx(self.hit_model, pts[ids], dirs[ids], vertex, d, E)
            ell = self.hit_model.predict_hit_logit(c, calibrated=True).reshape(-1).to(F64)
            return torch.where(resp['hit'][ids], Fn.logsigmoid(ell),
                               Fn.logsigmoid(-ell)).sum()
        if kind == 'ly':
            m, K = self.ly_model, self._u.numel()
            c = self._ctx(m, pts[ids], dirs[ids], vertex, d, E).repeat_interleave(K, 0)
            qd = m._prep((resp['counts'][ids].reshape(-1, 1)
                          + self._u.reshape(1, -1)).reshape(-1))
            lp = (_log_prob_z(m, m.to_z(qd), c, self.n_steps, self._eps(m), need_grad)
                  + m.log_det_dz_dq(qd)).to(F64)
            return torch.logsumexp(lp.reshape(-1, K) + self._logw_u, dim=1).sum()
        at = self.atime_model
        own = resp['ph_owner'][ids]
        p, dr, n = pts[own], dirs[own], own.numel()
        c = self._ctx(at, p, dr, vertex, d, E)
        tg = at.geometric_time(p, vertex.reshape(1, 3).expand(n, 3),
                               directions=_dir_for(at, d).reshape(1, 3).expand(n, 3))
        tr = at._prep(resp['ph_t'][ids] - t0 - tg.reshape(-1))
        lp = (_log_prob_z(at, at.to_z(tr), c, self.n_steps, self._eps(at), need_grad)
              + at.log_det_dz_dq(tr)).to(F64)
        if self.noise_frac > 0:
            floor = math.log(self.noise_frac) - math.log(self.noise_window)
            lp = torch.logaddexp(lp + math.log1p(-self.noise_frac),
                                 torch.full_like(lp, floor))
        return (resp['ph_w'][ids] * lp).sum()

    def loglik(self, theta, resp, fr=None):
        """log L at a full theta (7,), no graph. Useful for scans."""
        fr = fr or self.frame(resp['event'], resp.get('t_offset', 0.0))
        pts, dirs = self.expand(resp['points_3d'])
        with torch.no_grad():
            return sum(self._unit_loglik(k, i, theta, fr, resp, pts.detach(), dirs, False)
                       for k, i in self._units(resp))

    def _value(self, th, fr, free_idx, resp, pts, dirs, units):
        with torch.no_grad():
            full = self._full(th, fr, free_idx)
            return sum(self._unit_loglik(k, i, full, fr, resp, pts, dirs, False)
                       for k, i in units)

    def _vgh(self, th, fr, free_idx, resp, pts, dirs, units):
        """Value, gradient and Hessian w.r.t. the free parameters, one unit at a time."""
        P = th.numel()
        th = th.detach().clone().requires_grad_(True)
        val = th.new_zeros(())
        g = th.new_zeros(P)
        H = th.new_zeros(P, P)
        for kind, ids in units:
            lc = self._unit_loglik(kind, ids, self._full(th, fr, free_idx), fr, resp,
                                   pts, dirs, True)
            val = val + lc.detach()
            if not lc.requires_grad:
                continue
            gc, = torch.autograd.grad(lc, th, create_graph=True)
            for p in range(P):
                if gc[p].requires_grad:
                    r, = torch.autograd.grad(gc[p], th, retain_graph=True,
                                             allow_unused=True)
                    if r is not None:
                        H[p] += r.detach()
            g = g + gc.detach()
            del lc, gc
        return val, g, 0.5 * (H + H.T)

    # ---------------------------------------------------------------------- fit

    def _start_event(self, fr_true, free, start, start_sigma, gen):
        """Event to start the fit from, or None for the truth.

        'random' moves only the FREE parameters off the truth (fixed ones stay known):
        vertex coordinates by N(0, vertex^2) m, the direction by a Rayleigh-distributed
        angle in a uniform random azimuth about the truth, log10 E and t0 by Gaussians.
        A dict is used as the start event as given (travel direction convention).
        """
        if isinstance(start, dict):
            return start
        if start == 'truth':
            return None
        if start != 'random':
            raise ValueError("start must be 'truth', 'random' or an event dict")
        sg = dict(START_SIGMA)
        sg.update(start_sigma or {})

        def rn(n=1):
            return torch.randn(n, generator=gen, dtype=F64).to(self.device)

        vert, d, E, t0 = self.unpack(fr_true['theta0'], fr_true)
        v = vert.clone()
        for c, k in enumerate(('x', 'y', 'z')):
            if k in free:
                v[c] = v[c] + sg['vertex'] * rn()[0]
        s_dir = math.radians(sg['direction_deg'])
        if 'dir_a' in free and 'dir_b' in free:
            alpha = s_dir * float(rn(2).norm())
            phi = 2.0 * math.pi * float(torch.rand(1, generator=gen, dtype=F64))
            d = (math.cos(alpha) * d + math.sin(alpha)
                 * (math.cos(phi) * fr_true['e1'] + math.sin(phi) * fr_true['e2']))
        elif 'dir_a' in free or 'dir_b' in free:
            e = fr_true['e1'] if 'dir_a' in free else fr_true['e2']
            d = d + s_dir * rn()[0] * e
        log_e = math.log10(float(E))
        if 'log10_energy' in free:
            log_e += sg['log10_energy'] * float(rn()[0])
        t = float(t0) + (sg['t0'] * float(rn()[0]) if 't0' in free else 0.0)
        return {'position': v, 'direction': d / d.norm(), 'energy': 10.0 ** log_e,
                'time': t}

    def fit(self, resp, free=PARAMS, start='truth', start_sigma=None, start_seed=None,
            max_iter=20, tol=1e-6, lam0=1e-3, verbose=False):
        """Levenberg-Marquardt / damped Newton with the exact Hessian.

        start : 'truth', 'random' (perturb the free parameters, scales START_SIGMA,
            overridden by start_sigma) or an event dict. The fit's tangent frame is
            centred on the start; errors are always measured against the truth.
        start_seed : int or torch.Generator for reproducible random starts.
        """
        bad = [p for p in free if p not in PARAMS]
        if bad:
            raise ValueError(f"unknown fit parameters {bad}; choose from {PARAMS}")
        free = tuple(free)
        free_idx = torch.tensor([PARAMS.index(p) for p in free], device=self.device)
        pts, dirs = self.expand(resp['points_3d'])
        pts = pts.detach()
        units = self._units(resp)

        gen = start_seed
        if start_seed is not None and not isinstance(start_seed, torch.Generator):
            gen = torch.Generator().manual_seed(int(start_seed))
        # truth on the response's clock; a random start derives from it and inherits it
        fr_true = self.frame(resp['event'], resp.get('t_offset', 0.0))
        ev0 = self._start_event(fr_true, free, start, start_sigma, gen)
        fr = fr_true if ev0 is None else self.frame(ev0)
        val_true = float(self._value(fr_true['theta0'][free_idx], fr_true, free_idx,
                                     resp, pts, dirs, units))

        th = fr['theta0'][free_idx].clone()
        val, g, H = self._vgh(th, fr, free_idx, resp, pts, dirs, units)
        val_start = float(val)
        lam, converged, it = float(lam0), False, 0
        for it in range(1, max_iter + 1):
            A = -H
            Dg = torch.diag(A.diagonal().abs().clamp_min(1e-12))
            step, accepted = None, False
            while lam < 1e10:
                try:
                    step = torch.linalg.solve(A + lam * Dg, g)
                except RuntimeError:
                    lam *= 10.0
                    continue
                v_new = self._value(th + step, fr, free_idx, resp, pts, dirs, units)
                if torch.isfinite(v_new) and v_new >= val:
                    accepted = True
                    break
                lam *= 10.0
            if not accepted:
                break
            gain = float(g @ step)
            th = th + step
            lam = max(lam / 10.0, 1e-9)
            val, g, H = self._vgh(th, fr, free_idx, resp, pts, dirs, units)
            if verbose:
                print(f'    it {it}: logL {float(val):.6f}  (truth {val_true:.6f})'
                      f'  step.g {gain:.3g}  lam {lam:.1e}')
            if abs(gain) < tol:
                converged = True
                break

        full = self._full(th, fr, free_idx).detach()
        vertex, d, E, t0 = self.unpack(full, fr)
        try:
            cov = torch.linalg.inv(-H)
        except RuntimeError:
            cov = torch.linalg.pinv(-H)
        sig_ang = float('nan')
        if 'dir_a' in free and 'dir_b' in free:
            ia = [free.index('dir_a'), free.index('dir_b')]
            sig_ang = float(torch.sqrt(cov[ia][:, ia].diagonal().sum().clamp_min(0.0)))

        def ang(u):
            c = (u @ fr_true['d0']).clamp(-1.0, 1.0)
            return float(torch.atan2(torch.linalg.cross(u, fr_true['d0']).norm(), c)), c

        angle, cos_a = ang(d)
        return {'theta': full, 'free': free, 'vertex': vertex, 'direction': d,
                'energy': float(E), 't0': float(t0), 'angle': angle,
                'chord2': float(2.0 * (1.0 - cos_a)),
                'dlog10E': float(full[5] - fr_true['theta0'][5]),
                'dt0': float(full[6] - fr_true['theta0'][6]),
                'start_angle': ang(fr['d0'])[0],
                'loglik': float(val), 'loglik_true': val_true,
                'loglik_start': val_start,
                'fit_ge_truth': float(val) >= val_true - 1e-6 * max(1.0, abs(val_true)),
                'n_iter': it, 'converged': converged, 'hessian': H, 'cov': cov,
                'sigma_angle_obs': sig_ang}

    # --------------------------------------------------------------- resolution

    def resolution(self, points_3d, events, generator=None, free=PARAMS, min_hits=1,
                   max_photons_per_pmt=None, time_origin='vertex', verbose=False,
                   **fit_kw):
        """Sample one response per event, fit each, and summarise the errors.

        angular_rms_deg = sqrt(mean 2(1 - cos dpsi)), the MLE counterpart of the
        Fisher sqrt(tr cov); sigma_obs_rms_deg is the same from the observed
        information at each fit. Events with fewer than min_hits hit PMTs are skipped.
        fit_kw go to fit (e.g. start='random', start_sigma, start_seed); an int
        start_seed seeds one generator for all events, so each gets its own start.
        """
        ss = fit_kw.get('start_seed')
        if ss is not None and not isinstance(ss, torch.Generator):
            fit_kw['start_seed'] = torch.Generator().manual_seed(int(ss))
        rows, skipped = [], 0
        for i, ev in enumerate(events):
            resp = self.sample(points_3d, ev, generator, max_photons_per_pmt,
                               time_origin=time_origin)
            if resp['n_hits'] < min_hits:
                skipped += 1
                continue
            r = self.fit(resp, free=free, **fit_kw)
            r.update(n_hits=resp['n_hits'], n_photons=resp['n_photons'],
                     pruned_expected_hits=resp['pruned_expected_hits'], index=i)
            rows.append(r)
            if verbose:
                print(f'  event {i + 1}/{len(events)}: {resp["n_hits"]} hits, '
                      f'{resp["n_photons"]:.0f} photons, start '
                      f'{math.degrees(r["start_angle"]):.2f} deg -> dpsi = '
                      f'{math.degrees(r["angle"]):.3f} deg, sigma_obs = '
                      f'{math.degrees(r["sigma_angle_obs"]):.3f} deg, '
                      f'{r["n_iter"]} it{"" if r["converged"] else " (not converged)"}'
                      f'{"" if r["fit_ge_truth"] else ", BELOW truth logL"}',
                      flush=True)
        if not rows:
            return {'per_event': [], 'n_skipped': skipped}
        ch = np.array([r['chord2'] for r in rows])
        ang = np.array([r['angle'] for r in rows])
        so = np.array([r['sigma_angle_obs'] for r in rows])
        de = np.array([r['dlog10E'] for r in rows])
        return {'per_event': rows, 'n_skipped': skipped,
                'n_converged': int(sum(r['converged'] for r in rows)),
                'n_fit_ge_truth': int(sum(r['fit_ge_truth'] for r in rows)),
                'angular_rms_deg': math.degrees(math.sqrt(ch.mean())),
                'angular_median_deg': math.degrees(float(np.median(ang))),
                'sigma_obs_rms_deg': math.degrees(math.sqrt(np.nanmean(so ** 2))),
                'dlog10E_rms': float(np.sqrt(np.mean(de ** 2))),
                'dlog10E_bias': float(de.mean())}
