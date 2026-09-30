"""Monte Carlo variant of FlowFisherResolutionLoss: sampled light yield and times."""
import torch

from nugget.losses.fisher_info_flow import (FlowFisherResolutionLoss, _log_prob_z,
                                            _model_dir)


def _push(model, z0, ctx, n_steps, chunk):
    """Base noise z0 -> data-space z, sample_z's midpoint scheme; differentiable in ctx."""
    out = []
    for s in range(0, z0.shape[0], chunk):
        c = model._apply_context_norm(model._prep(ctx[s:s + chunk]))
        z = model._prep(z0[s:s + chunk]).reshape(-1, 1)
        B = z.shape[0]
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0 = torch.full((B,), i * dt, device=z.device, dtype=z.dtype)
            tm = torch.full((B,), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
            k1 = model._velocity(z, t0, c)
            z = z + dt * model._velocity(z + 0.5 * dt * k1, tm, c)
        out.append(z.reshape(-1))
    return torch.cat(out)


class FlowFisherMCResolutionLoss(FlowFisherResolutionLoss):
    """FlowFisherResolutionLoss with the light-yield and timing Fisher estimated by MC.

    Per selected PMT, K draws from each flow: F_LY = mean_k s_k s_k^T, with s_k the
    score at the draw held fixed in theta, and F_T likewise per photon, weighted by
    q_bar = mean(q~_k) - 1/2 (the dequantisation offset averages 1/2). Both are the
    continuous (dequantised) likelihoods. The exact hit term, sample_hits, event
    batching and geometry_grads (including 'chunked') are inherited unchanged.

    Draws follow the geometry pathwise, x_k = T(z_k; c(theta_0, x)), so position
    gradients see the sampling distribution move; the base noise z_k is kept in the
    per-row state, so pass 2 of 'chunked' rebuilds exactly the F that pass 1 used.

    Parameters, beyond the parent's (n_quad and z_range are unused)
    ----------
    n_samples_ly, n_samples_t : int
        Draws per PMT for the light yield and for the arrival time.
    sample_n_steps : int or None
        ODE steps for drawing; None uses n_steps.
    hit_sample_seed : int or None
        Seeds the hit selection and the MC draws per call (a reproducible loss).
    """

    def __init__(self, *args, n_samples_ly=4, n_samples_t=4, sample_n_steps=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples_ly = max(int(n_samples_ly), 1)
        self.n_samples_t = max(int(n_samples_t), 1)
        self.sample_n_steps = None if sample_n_steps is None else int(sample_n_steps)

    def _nodes_per_pmt(self):
        return max(self.n_samples_ly, self.n_samples_t)

    def _noise(self, key, M, K, idx, gen, state, new_state):
        """(len(idx), K) base noise for the selected rows; reused from state if present."""
        z = None if state is None else state.get(key)
        if z is None:
            z = torch.randn(M, K, device=self.device, dtype=self.dtype, generator=gen)
        new_state[key] = z
        return z[idx]

    def _scores_rows(self, model, build, d0, z1, chunk):
        """d log p_z(z1_mk | c_m(theta)) / d theta at per-row draws z1 (m, K) -> (m, K, P).

        Only the context is differentiated w.r.t. theta; the draw z1 is data. With
        tracking it still carries its pathwise dependence on the geometry.
        """
        m, K = z1.shape
        P = d0.shape[0]
        Jc = torch.func.jacfwd(build)(d0)                          # (m, C, P)
        ctx0 = self._keep(build(d0))
        g = torch.empty(m, K, P, device=self.device, dtype=self.dtype)
        rep = max(int(chunk) // K, 1)
        for s in range(0, m, rep):
            e = min(s + rep, m)
            n = e - s
            c = self._diff_wrt(ctx0[s:e].repeat_interleave(K, 0))
            lp = _log_prob_z(model, z1[s:e].reshape(-1), c, self.n_steps, self.div_eps)
            G = torch.autograd.grad(lp.sum(), c,
                                    create_graph=self._track)[0].reshape(n, K, -1)
            g[s:e] = torch.einsum('mkc,mcp->mkp', G, Jc[s:e])
            del c, lp, G
        return g

    def _flow_terms(self, F, idx, wf, pts, dirs, rows_ev, eb, chunk, d0, gen, state,
                    new_state):
        M, m = F.shape[0], idx.numel()
        sub_pts, sub_dirs, rev = pts[idx], dirs[idx], rows_ev[idx]
        Kq, Kt = self.n_samples_ly, self.n_samples_t
        ns = self.sample_n_steps or self.n_steps
        ck = max(int(chunk), 1)

        # ---- light yield: draws pathwise in the geometry, fixed in theta ---------
        ly = self.ly_model
        build_l = self._ctx_builder(ly, sub_pts, sub_dirs, rev, eb)
        zq = self._noise('z_ly', M, Kq, idx, gen, state, new_state)   # (m, Kq)
        with torch.set_grad_enabled(self._track):
            c_l = build_l(d0).repeat_interleave(Kq, 0)
            z1 = _push(ly, zq.reshape(-1), c_l, ns, ck).reshape(m, Kq)
        # expected photon count given a hit: E[q] = E[q~] - 1/2. Clamp only at 0: a
        # floor of 1 per draw would bias dim PMTs upward, most of all with one draw
        qbar = (ly.from_z(z1).mean(1) - 0.5).clamp_min(0.0).to(self.dtype)
        if self.include_ly:
            g_q = self._scores_rows(ly, build_l, d0, z1, chunk)
            contrib = (wf.reshape(m, 1, 1)
                       * torch.einsum('mkp,mkq->mpq', g_q, g_q) / Kq)
            F = F + torch.zeros_like(F).index_add(0, idx, contrib.to(F.dtype))
            del g_q, contrib

        # ---- arrival time ----------------------------------------------------------
        if self.include_atime:
            at = self.atime_model
            build_t = self._ctx_builder(at, sub_pts, sub_dirs, rev, eb)
            zt = self._noise('z_t', M, Kt, idx, gen, state, new_state)
            with torch.set_grad_enabled(self._track):
                c_t = build_t(d0).repeat_interleave(Kt, 0)
                t_res = at.from_z(_push(at, zt.reshape(-1), c_t, ns, ck)).reshape(m, Kt)
                # t_hit = t_res + t_geom at the true theta; _scores_time subtracts
                # t_geom(theta), so both carry the same position dependence
                tg0 = at.geometric_time(
                    sub_pts, eb['vertex'][rev],
                    directions=_model_dir(at, eb['zenith'][rev], eb['azimuth'][rev]),
                ).reshape(m, 1)
                t_hit = t_res.to(tg0.dtype) + tg0                     # (m, Kt)
            _, g_t = self._scores_time(at, build_t, d0, t_hit, sub_pts, rev, eb, chunk)
            contrib = ((wf * qbar).reshape(m, 1, 1)
                       * torch.einsum('mkp,mkq->mpq', g_t, g_t) / Kt)
            F = F + torch.zeros_like(F).index_add(0, idx, contrib.to(F.dtype))
            del g_t, contrib
        return F
