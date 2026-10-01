"""Resolution loss from an MLP trained to predict each OM's Fisher matrix."""

import math
import os
import time

import numpy as np
import torch

from nugget.losses.base_loss import LossFunction


def _as_t(x, device, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _unit_travel(zenith, azimuth):
    st = torch.sin(zenith)
    return torch.stack([st * torch.cos(azimuth), st * torch.sin(azimuth),
                        torch.cos(zenith)], dim=-1)


def _om_to_string(pts3, string_xy, device):
    """(n_pts,) string index per OM by exact (x, y) match; n_strings marks 'none'."""
    n = pts3.shape[0]
    if string_xy is None:
        return torch.zeros(n, dtype=torch.long, device=device), 1
    if isinstance(string_xy, torch.Tensor):
        sxy = _as_t(string_xy.detach(), device).reshape(-1, 2)
    else:                                         # list of (x, y) pairs, as in fisher_info
        sxy = torch.stack([torch.stack([_as_t(s[0], device).reshape(()),
                                        _as_t(s[1], device).reshape(())])
                           for s in string_xy]).detach()
    p = pts3.detach().to(sxy.dtype)
    hit = ((p[:, 0].unsqueeze(0) == sxy[:, 0].unsqueeze(1)) &
           (p[:, 1].unsqueeze(0) == sxy[:, 1].unsqueeze(1)))
    n_str = sxy.shape[0]
    om_str = torch.where(hit.any(0), hit.float().argmax(0),
                         torch.full((n,), n_str, device=device))
    return om_str.long(), n_str


def resolution_from_fisher(F, fisher_params, energy, zenith, resolution_type='angular',
                           use_relative_energy=False):
    """Per-event resolution from (n, P, P), inverted in E-scaled units as the Fisher loss does."""
    n, P = F.shape[0], F.shape[-1]
    d = torch.ones(n, P, dtype=F.dtype, device=F.device)
    if 'energy' in fisher_params:
        d[:, fisher_params.index('energy')] = energy.clamp_min(1e-30)
    dd = d.unsqueeze(2) * d.unsqueeze(1)
    eye = torch.eye(P, dtype=F.dtype, device=F.device)
    try:
        cov = dd * torch.linalg.inv(dd * F + 1e-20 * eye)
    except RuntimeError:
        cov = dd * torch.linalg.pinv(dd * F + 1e-20 * eye)
    if resolution_type == 'angular':
        iz, ia = fisher_params.index('zenith'), fisher_params.index('azimuth')
        s = torch.sin(zenith)
        var = cov[:, iz, iz] + s ** 2 * cov[:, ia, ia] + 2.0 * s * cov[:, iz, ia]
        return torch.sqrt(var.clamp_min(0.0))
    res = torch.sqrt(cov[:, fisher_params.index('energy'),
                         fisher_params.index('energy')].clamp_min(0.0))
    return res / energy.clamp_min(1e-30) if use_relative_energy else res


# --------------------------------------------------------------------------- #
#  Training data                                                               #
# --------------------------------------------------------------------------- #

class OMFisherTargets:
    """On-the-fly exact per-OM Fisher targets: fresh events, OMs placed around each.

    Calling it with n_events returns a batch dict (on the Fisher loss's device):
    om_pos, vertex, energy, zenith, azimuth (travel), event_id, F (N, P, P).

    Most OMs sit around the track: d_perp log-uniform in d_perp_range, d_long uniform
    in d_long_range (from the vertex, along the travel direction), uniform azimuth.
    A fraction uniform_frac is uniform in a cube of half-size uniform_half_size about
    the vertex, so far regions are covered too.
    """

    def __init__(self, fisher_loss, sampler, oms_per_event=64, d_perp_range=(1.0, 400.0),
                 d_long_range=(-200.0, 1500.0), uniform_frac=0.1, uniform_half_size=600.0,
                 chunk=8192, seed=None):
        if fisher_loss.sample_hits:
            print("OMFisherTargets: fisher_loss.sample_hits=True samples the hit "
                  "selection; use sample_hits=False for exact targets.")
        self.fisher_loss, self.sampler = fisher_loss, sampler
        self.oms_per_event, self.chunk = int(oms_per_event), int(chunk)
        self.d_perp_range, self.d_long_range = tuple(d_perp_range), tuple(d_long_range)
        self.uniform_frac, self.uniform_half_size = float(uniform_frac), float(uniform_half_size)
        self.rng = np.random.default_rng(seed)

    @property
    def fisher_params(self):
        return list(self.fisher_loss.fisher_info_params)

    def _place(self, ev):
        """OM positions (oms_per_event, 3) around one event's track."""
        n_u = int(round(self.uniform_frac * self.oms_per_event))
        n_t = self.oms_per_event - n_u
        v = np.asarray(torch.as_tensor(ev['position']).detach().cpu(), float).reshape(3)
        z = float(torch.as_tensor(ev['zenith']).reshape(-1)[0])
        a = float(torch.as_tensor(ev['azimuth']).reshape(-1)[0])
        u = np.array([math.sin(z) * math.cos(a), math.sin(z) * math.sin(a), math.cos(z)])
        h = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = h - (h @ u) * u
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        lo, hi = math.log(self.d_perp_range[0]), math.log(self.d_perp_range[1])
        dp = np.exp(self.rng.uniform(lo, hi, n_t))
        dl = self.rng.uniform(self.d_long_range[0], self.d_long_range[1], n_t)
        ph = self.rng.uniform(0.0, 2.0 * math.pi, n_t)
        p_track = (v + dl[:, None] * u
                   + dp[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2))
        p_unif = v + self.rng.uniform(-self.uniform_half_size, self.uniform_half_size,
                                      (n_u, 3))
        return np.concatenate([p_track, p_unif]), v

    def __call__(self, n_events):
        fl = self.fisher_loss
        dev, dt = fl.device, fl.dtype
        evs = self.sampler.sample_events(int(n_events))
        placed = [self._place(ev) for ev in evs]
        idx = torch.arange(len(evs), device=dev).repeat_interleave(self.oms_per_event)
        pos = torch.as_tensor(np.concatenate([p for p, _ in placed]), device=dev, dtype=dt)
        F = fl.fisher_per_om(pos, evs, idx, chunk=self.chunk)     # needs autograd inside
        col = lambda k: torch.tensor([float(torch.as_tensor(e[k]).reshape(-1)[0])
                                      for e in evs], dtype=torch.float64, device=dev)
        vert = torch.as_tensor(np.stack([v for _, v in placed]), dtype=torch.float64,
                               device=dev)
        return {'om_pos': pos.detach().double(), 'vertex': vert[idx],
                'energy': col('energy')[idx], 'zenith': col('zenith')[idx],
                'azimuth': col('azimuth')[idx], 'event_id': idx,
                'F': F.detach().double(), 'fisher_params': self.fisher_params}


# --------------------------------------------------------------------------- #
#  Network                                                                     #
# --------------------------------------------------------------------------- #

class _Block(torch.nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, h):
        x = torch.nn.functional.silu(self.fc1(torch.nn.functional.silu(self.norm(h))))
        return h + self.fc2(self.drop(x))


class _ResMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, dropout):
        super().__init__()
        self.inp = torch.nn.Linear(in_dim, width)
        self.blocks = torch.nn.ModuleList([_Block(width, dropout) for _ in range(depth)])
        self.out_norm = torch.nn.LayerNorm(width)
        self.out = torch.nn.Linear(width, out_dim)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
#  Loss                                                                        #
# --------------------------------------------------------------------------- #

class OMFisherNetLoss(LossFunction):
    """Angular / energy resolution from an MLP's per-OM Fisher (16-PMT template summed).

    Same call contract as FlowFisherResolutionLoss: per-string Fisher summed over OMs,
    optional sigmoid(string_weights), resolution per event, one metric. Positions are
    differentiable through the net itself.

    The net predicts the Cholesky factor of F' = D F D + eps_floor I, with D the event
    energy for 'energy' (so that entry is the Fisher of ln E) and param_scales[p]
    otherwise (default 1): log of its diagonal and the off-diagonals over their
    column's diagonal. F = L L^T is PSD by construction.

    Parameters
    ----------
    domain_size, rich_rel_pos_mode, include_vertex_position, add_vertex_distance,
    add_distance_from_beam, add_dist_long, ly_eps
        Input features, with the same meaning as in the hit / flow models (see
        build_context); always included are the travel direction, log10 E and
        cos(track, vertex -> OM). There are no PMT features: the target is a whole OM.
    add_log_distances : bool
        Also give log10(1 m + d_perp) and log10(1 m + |OM - vertex|). The Fisher falls
        by orders of magnitude over 1-400 m, which linear distances standardised over
        a km-wide spread barely resolve.
    fisher_params : sequence of str
        Must match the targets it is trained on (OMFisherTargets).
    target_mode : 'exact' or 'mc'
        'exact': targets are exact Fishers; MSE on their log-Cholesky factors.
        'mc': targets are single-draw MC Fishers (unbiased, noisy); the loss is the
        squared error of every entry in linear F space, each over
        sqrt(s_i s_j) with s_i = sg(F'_pred,ii) + tau_i. The scale depends on the input
        only, so the minimiser is still the conditional mean E[F'_target | c] = F';
        a log-space loss would converge to E[log F'_target], biased low for noisy
        targets. Per entry rather than per trace, so energy is not drowned out by
        the (much larger) angular entries.
    weight_tau : 'median', float or 0
        'exact': training weight tr F' / (tr F' + tau), so dark OMs, which barely
        enter any sum, count less (0 weights every OM equally). 'mc': 'median' sets
        tau_i to the median F'_ii of the warm-up draw; a number sets every tau_i.
    geometry_grads : 'chunked', 'direct' or False
        'chunked' builds F without a graph and accumulates dL/d(points_3d) in
        bounded memory; 'direct' keeps the whole graph.
    net_dtype : torch dtype or None
        None uses the default (float64 in nugget); float32 halves memory and time.
    """

    def __init__(self, device=None, fisher_params=('energy', 'zenith', 'azimuth'),
                 resolution_type='angular', param_scales=None,
                 domain_size=20000, rich_rel_pos_mode=True, include_vertex_position=False,
                 add_vertex_distance=False, add_distance_from_beam=True,
                 add_dist_long=True, ly_eps=1e-6, add_log_distances=False,
                 width=256, depth=6, dropout=0.0, learning_rate=1e-3,
                 lr_schedule='onecycle', warmup_frac=0.1, weight_decay=0.0,
                 eps_floor=1e-6, weight_tau='median', target_mode='exact',
                 geometry_grads='chunked', net_dtype=None, print_loss=False):
        super().__init__(device=device)
        if resolution_type not in ('angular', 'energy'):
            raise ValueError("resolution_type must be 'angular' or 'energy'")
        if target_mode not in ('exact', 'mc'):
            raise ValueError("target_mode must be 'exact' or 'mc'")
        self.target_mode = target_mode
        if geometry_grads not in ('chunked', 'direct', False):
            raise ValueError("geometry_grads must be 'chunked', 'direct' or False")
        self.fisher_params = list(fisher_params)
        self.resolution_type = resolution_type
        self.param_scales = dict(param_scales or {})
        self.domain_size = domain_size
        self.rich_rel_pos_mode = bool(rich_rel_pos_mode)
        self.include_vertex_position = bool(include_vertex_position)
        self.add_vertex_distance = bool(add_vertex_distance)
        self.add_distance_from_beam = bool(add_distance_from_beam)
        self.add_dist_long = bool(add_dist_long)
        self.ly_eps = float(ly_eps)
        self.add_log_distances = bool(add_log_distances)
        self.width, self.depth, self.dropout = width, depth, dropout
        self.learning_rate, self.lr_schedule = learning_rate, lr_schedule
        self.warmup_frac, self.weight_decay = warmup_frac, weight_decay
        self.eps_floor, self.weight_tau = float(eps_floor), weight_tau
        self.geometry_grads, self.net_dtype = geometry_grads, net_dtype
        self.print_loss = bool(print_loss)

        self.context_mean = self.context_std = None
        self.target_mean = self.target_std = None
        self.tau = self.tau_diag = None
        self.net = self.optimizer = self.lr_scheduler = None
        self.train_losses, self.val_losses = [], []
        self.best_state_dict = None

    _CONTEXT_FLAGS = ('domain_size', 'rich_rel_pos_mode', 'include_vertex_position',
                      'add_vertex_distance', 'add_distance_from_beam', 'add_dist_long',
                      'ly_eps', 'add_log_distances')

    @property
    def context_dim(self):
        d = 3 + 3 + 1                                   # rel, direction, log10 E
        if not self.rich_rel_pos_mode:
            d += 3                                      # absolute det + vert
        elif self.include_vertex_position:
            d += 3
        d += int(self.add_vertex_distance)
        d += 1                                          # cos(track, vertex -> OM)
        d += int(self.add_distance_from_beam) + int(self.add_dist_long)
        return d + 2 * int(self.add_log_distances)

    @property
    def P(self):
        return len(self.fisher_params)

    @property
    def target_dim(self):
        return self.P + self.P * (self.P - 1) // 2

    @property
    def param_dtype(self):
        return next(self.net.parameters()).dtype if self.net is not None \
            else (self.net_dtype or torch.get_default_dtype())

    # ---------------- features and targets ----------------

    def build_context(self, om_pos, vertex, energy, direction):
        """(N,3),(N,3),(N,),(N,3) travel direction -> (N, context_dim).

        Same layout as the hit / flow models' build_context, minus the PMT features.
        The direction is always the travel direction here (event_params convention).
        """
        ds = self.domain_size
        norm = float(ds[0] if isinstance(ds, (tuple, list)) else ds) / 2.0
        om = om_pos.reshape(-1, 3)
        vert_raw = vertex.reshape(-1, 3).to(om.dtype)
        u = direction.reshape(-1, 3).to(om.dtype)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        det, vert = om / norm, vert_raw / norm
        log_e = torch.log10(energy.reshape(-1).to(om.dtype) + self.ly_eps) / 8.0
        rel = det - vert
        vert_dist = torch.linalg.norm(rel, dim=1)
        cos_angle = (u * rel).sum(1) / (vert_dist + 1e-8)

        if self.rich_rel_pos_mode:
            cols = [rel, u, log_e.unsqueeze(1)]
            if self.include_vertex_position:
                cols.append(vert)
        else:
            cols = [det, vert, u, log_e.unsqueeze(1)]
        if self.add_vertex_distance:
            cols.append(vert_dist.unsqueeze(1))
        cols.append(cos_angle.unsqueeze(1))
        if self.add_distance_from_beam or self.add_dist_long:
            rel_m = om - vert_raw
            d_long = (rel_m * u).sum(1)
            d_perp = torch.linalg.norm(rel_m - d_long.unsqueeze(1) * u, dim=1)
            if self.add_distance_from_beam:
                cols.append((d_perp / norm).unsqueeze(1))
            if self.add_dist_long:
                cols.append((d_long / norm).unsqueeze(1))
        if self.add_log_distances:
            rel_m = om - vert_raw
            d_perp = torch.linalg.norm(rel_m - (rel_m * u).sum(1, keepdim=True) * u, dim=1)
            cols += [(torch.log10(d_perp + 1.0) / 3.0).unsqueeze(1),
                     (torch.log10(rel_m.norm(dim=1) + 1.0) / 3.0).unsqueeze(1)]
        return torch.cat(cols, dim=1)

    def _scales(self, energy):
        d = torch.ones(energy.shape[0], self.P, dtype=energy.dtype, device=energy.device)
        for i, p in enumerate(self.fisher_params):
            s = self.param_scales.get(p, 'relative' if p == 'energy' else 1.0)
            d[:, i] = energy if s == 'relative' else float(s)
        return d

    def _scaled(self, F, energy):
        """F' = D F D, symmetrised: the dimensionless matrix the net works in."""
        d = self._scales(energy)
        Fs = d.unsqueeze(2) * F * d.unsqueeze(1)
        return 0.5 * (Fs + Fs.transpose(1, 2))

    def _triu(self, device):
        """Upper-triangle indices of F' and their weights in ||.||_F^2 (1 diag, 2 off)."""
        iu, ju = torch.triu_indices(self.P, self.P, device=device)
        return iu, ju, torch.where(iu == ju, 1.0, 2.0).to(device)

    def fisher_to_target(self, F, energy):
        """(N,P,P) physical Fisher -> (N, target_dim) log-Cholesky targets and tr F'."""
        Fs = self._scaled(F, energy)
        eye = torch.eye(self.P, dtype=F.dtype, device=F.device)
        jitter = self.eps_floor
        for _ in range(8):                       # a few rows may need more than eps
            L, info = torch.linalg.cholesky_ex(Fs + jitter * eye)
            if not bool((info != 0).any()):
                break
            jitter *= 10.0
        diag = L.diagonal(dim1=1, dim2=2)
        ii, jj = torch.tril_indices(self.P, self.P, -1, device=F.device)
        return (torch.cat([torch.log(diag), L[:, ii, jj] / diag[:, jj]], dim=1),
                Fs.diagonal(dim1=1, dim2=2).sum(1))

    def _scaled_from_output(self, y):
        """(N, target_dim) log-Cholesky -> F' = L L^T (N,P,P), PSD by construction."""
        P = self.P
        diag = torch.exp(y[:, :P])
        L = torch.diag_embed(diag)
        ii, jj = torch.tril_indices(P, P, -1, device=y.device)
        if ii.numel():
            L = L.index_put((torch.arange(y.shape[0], device=y.device).unsqueeze(1),
                             ii.unsqueeze(0), jj.unsqueeze(0)), y[:, P:] * diag[:, jj])
        return L @ L.transpose(1, 2)

    def target_to_fisher(self, y, energy):
        """(N, target_dim) -> (N,P,P) physical Fisher."""
        d = self._scales(energy.to(y.dtype))
        return self._scaled_from_output(y) / (d.unsqueeze(2) * d.unsqueeze(1))

    def _tensors(self, data):
        """-> context, log-Cholesky target, tr F', and F' upper triangle (N, P(P+1)/2)."""
        dev = self.device
        E = data['energy'].double().to(dev)
        F = data['F'].double().to(dev)
        u = _unit_travel(data['zenith'].double(), data['azimuth'].double()).to(dev)
        ctx = self.build_context(data['om_pos'].double().to(dev),
                                 data['vertex'].double().to(dev), E, u)
        y, tr = self.fisher_to_target(F, E)
        iu, ju, _ = self._triu(dev)
        return ctx, y, tr, self._scaled(F, E)[:, iu, ju]

    # ---------------- network ----------------

    def build_network(self):
        self.net = _ResMLP(self.context_dim, self.target_dim, self.width, self.depth,
                           self.dropout).to(device=self.device,
                                            dtype=self.net_dtype or torch.get_default_dtype())
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        n = sum(p.numel() for p in self.net.parameters())
        print(f"OMFisherNetLoss: params {self.fisher_params}, target_dim={self.target_dim}, "
              f"width={self.width}, depth={self.depth}, weights={n:,}")

    def predict_fisher(self, om_pos, vertex, energy, direction):
        """(N,P,P) Fisher in physical units; differentiable in all inputs."""
        ctx = self.build_context(om_pos, vertex, energy, direction)
        ctx = (ctx - self.context_mean.to(ctx.dtype)) / self.context_std.to(ctx.dtype)
        y = self.net(ctx.to(self.param_dtype)).to(ctx.dtype)
        y = y * self.target_std.to(ctx.dtype) + self.target_mean.to(ctx.dtype)
        return self.target_to_fisher(y, energy.reshape(-1).to(ctx.dtype))

    def _freeze(self):
        """Inference use: eval mode, and no gradients into the net's own weights."""
        self.net.eval()
        self.net.requires_grad_(False)

    # ---------------- training ----------------

    def fit_normalisers(self, data):
        if list(data['fisher_params']) != self.fisher_params:
            raise ValueError(f"data has {data['fisher_params']}, loss has {self.fisher_params}")
        # with 'mc' targets the log-Cholesky statistics are of noisy draws: they only
        # set the output scaling, the loss itself stays in linear F space
        ctx, y, tr, f_up = self._tensors(data)
        self.context_mean, self.context_std = ctx.mean(0), ctx.std(0).clamp_min(1e-6)
        self.target_mean, self.target_std = y.mean(0), y.std(0).clamp_min(1e-6)
        self.tau = (float(tr.median()) if self.weight_tau == 'median'
                    else float(self.weight_tau or 0.0))
        iu, ju, _ = self._triu(f_up.device)
        diag = f_up[:, iu == ju]                                   # F'_ii, (N, P)
        self.tau_diag = (diag.median(0).values.clamp_min(1e-12) if self.weight_tau == 'median'
                         else torch.full((self.P,), max(float(self.weight_tau or 0.0), 1e-12),
                                         dtype=diag.dtype, device=diag.device))
        print(f"  normalisers from {ctx.shape[0]:,} OMs; tau = {self.tau:.3g}, "
              f"tau_diag = {[f'{t:.3g}' for t in self.tau_diag.tolist()]}")

    @property
    def _buffer_dim(self):
        return self.target_dim if self.target_mode == 'exact' else self.P * (self.P + 1) // 2

    def _standardise(self, data):
        """Batch dict -> (standardised context, target, weight) in the net's dtype.

        'exact': the standardised log-Cholesky target and its tr-based weight.
        'mc': the raw F' upper triangle; its weighting lives in _net_loss.
        """
        ctx, y, tr, f_up = self._tensors(data)
        dt = self.param_dtype
        ctx = ((ctx - self.context_mean) / self.context_std).to(dt)
        if self.target_mode == 'mc':
            return ctx, f_up.to(dt), torch.ones_like(tr).to(dt)
        w = tr / (tr + self.tau) if self.tau > 0 else torch.ones_like(tr)
        return ctx, ((y - self.target_mean) / self.target_std).to(dt), w.to(dt)

    def _net_loss(self, ctx, y, w):
        out = self.net(ctx)
        if self.target_mode == 'exact':
            return (w * ((out - y) ** 2).mean(1)).sum() / w.sum().clamp_min(1e-12)
        # 'mc': squared error per entry in linear F' space, each over a scale that
        # depends on the input only (the detached prediction): minimiser E[F' | c]
        yy = out * self.target_std.to(out.dtype) + self.target_mean.to(out.dtype)
        Fp = self._scaled_from_output(yy)
        iu, ju, fw = self._triu(out.device)
        s = Fp.diagonal(dim1=1, dim2=2).detach() + self.tau_diag.to(out.dtype)  # (B, P)
        rel = (Fp[:, iu, ju] - y) / torch.sqrt(s[:, iu] * s[:, ju])
        return (fw.to(out.dtype) * rel ** 2).sum(1).mean()

    def fit_online(self, targets, n_steps=20_000, events_per_step=16, updates_per_step=4,
                   batch_size=4096, buffer_size=200_000, warmup_events=256,
                   n_val_events=512, val_every=100, early_stopping_patience=20,
                   grad_clip=1.0, save_every=None, checkpoint_path=None,
                   val_targets=None, verbose=True):
        """Train on (event, OM) pairs drawn on the fly by ``targets`` (OMFisherTargets).

        Every step draws events_per_step fresh events into a replay buffer and takes
        updates_per_step minibatch steps from it: target Fishers cost far more than an
        MLP step, so each draw is reused a few times while the buffer keeps the
        minibatches from being dominated by the latest events. warmup_events set the
        normalisers and seed the buffer; n_val_events are drawn once and fixed, from
        val_targets if given (e.g. exact targets while training on 'mc' ones, so the
        validation loss is the true error rather than error plus target noise).
        Validation runs every val_every steps; patience and save_every count those.
        """
        for t in (targets, val_targets):
            if t is not None and list(t.fisher_params) != self.fisher_params:
                raise ValueError(f"targets give {t.fisher_params}, loss has "
                                 f"{self.fisher_params}")
        if self.net is None:
            self.build_network()
        self.net.requires_grad_(True)
        warm = targets(warmup_events)
        if self.context_mean is None:
            self.fit_normalisers(warm)
        val = self._standardise((val_targets or targets)(n_val_events))

        dev, dt = self.device, self.param_dtype
        buf = [torch.empty(buffer_size, d, device=dev, dtype=dt)
               for d in (self.context_dim, self._buffer_dim)]
        buf.append(torch.empty(buffer_size, device=dev, dtype=dt))
        fill, head = 0, 0

        def push(batch):
            nonlocal fill, head
            parts = self._standardise(batch)
            n = min(parts[0].shape[0], buffer_size)
            pos = (head + torch.arange(n, device=dev)) % buffer_size
            for b, p in zip(buf, parts):
                b[pos] = p[-n:]
            head, fill = (head + n) % buffer_size, min(fill + n, buffer_size)

        push(warm)
        if self.lr_schedule == 'onecycle':
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.learning_rate,
                total_steps=n_steps * updates_per_step, pct_start=self.warmup_frac)

        best_val, patience, t0 = float('inf'), 0, time.time()
        tl_sum, tl_n, n_val = 0.0, 0, 0
        for step in range(1, n_steps + 1):
            push(targets(events_per_step))
            self.net.train()
            for _ in range(updates_per_step):
                i = torch.randint(fill, (min(batch_size, fill),), device=dev)
                self.optimizer.zero_grad()
                loss = self._net_loss(buf[0][i], buf[1][i], buf[2][i])
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                self.optimizer.step()
                if self.lr_scheduler is not None and \
                        self.lr_scheduler.last_epoch + 1 < self.lr_scheduler.total_steps:
                    self.lr_scheduler.step()
                tl_sum += loss.item()
                tl_n += 1

            if step % val_every == 0 or step == n_steps:
                self.net.eval()
                with torch.no_grad():
                    vl = self._net_loss(*val).item()
                tl = tl_sum / max(tl_n, 1)
                tl_sum, tl_n, n_val = 0.0, 0, n_val + 1
                self.train_losses.append(tl)
                self.val_losses.append(vl)
                if vl < best_val:
                    best_val, patience = vl, 0
                    self.best_state_dict = {k: v.detach().clone()
                                            for k, v in self.net.state_dict().items()}
                    if checkpoint_path:                     # every new best is kept
                        self.save_model(checkpoint_path, state=self.best_state_dict)
                else:
                    patience += 1
                if verbose:
                    el = time.time() - t0
                    print(f"step {step}/{n_steps}  train {tl:.5f}  val {vl:.5f}  "
                          f"buffer {fill:,}  ({el:.0f}s, {el / step:.2f} s/step)",
                          flush=True)
                if save_every and checkpoint_path and n_val % save_every == 0:
                    self.save_model(checkpoint_path, state=self.best_state_dict)
                if patience >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at step {step} (best val {best_val:.5f})")
                    break

        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)
            if verbose:
                print(f"Restored best weights (val {best_val:.5f})")
        if checkpoint_path:
            self.save_model(checkpoint_path, state=self.best_state_dict)
        self._freeze()
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses,
                'val_every': val_every}

    # ---------------- loss ----------------

    def _event_tensors(self, params):
        dev = self.device
        col = lambda k: torch.stack([_as_t(e[k], dev).reshape(-1)[0] for e in params])
        zen, azi = col('zenith'), col('azimuth')
        return {'vertex': torch.stack([_as_t(e['position'], dev).reshape(3) for e in params]),
                'energy': col('energy'), 'zenith': zen, 'u': _unit_travel(zen, azi)}

    def _per_string(self, pts3, om_str, n_str, ev, chunk, ev_slice=None):
        """(n_events, n_strings, P, P), events in blocks of ~chunk (event, OM) rows."""
        n_pts, P = pts3.shape[0], self.P
        n_ev = ev['energy'].shape[0]
        B = max(int(chunk) // max(n_pts, 1), 1)
        out = []
        for b0 in range(0, n_ev, B):
            b1 = min(b0 + B, n_ev)
            nb = b1 - b0
            rep = lambda t: t[b0:b1].repeat_interleave(n_pts, 0)
            F = self.predict_fisher(pts3.repeat(nb, 1), rep(ev['vertex']),
                                    rep(ev['energy']), rep(ev['u']))
            key = (torch.arange(nb, device=pts3.device).repeat_interleave(n_pts)
                   * (n_str + 1) + om_str.repeat(nb))
            Fs = torch.zeros(nb * (n_str + 1), P, P, device=pts3.device,
                             dtype=F.dtype).index_add(0, key, F)
            out.append(Fs.reshape(nb, n_str + 1, P, P)[:, :n_str])
        return torch.cat(out)

    def _position_grad(self, pts3, om_str, n_str, ev, A, chunk):
        """dL/d(points_3d) from A = dL/dF_str, one block of events at a time."""
        leaf = pts3.detach().requires_grad_(True)
        n_pts, n_ev = leaf.shape[0], ev['energy'].shape[0]
        B = max(int(chunk) // max(n_pts, 1), 1)
        G = torch.zeros_like(leaf)
        for b0 in range(0, n_ev, B):
            b1 = min(b0 + B, n_ev)
            sub = {k: v[b0:b1] for k, v in ev.items()}
            s = (A[b0:b1].detach() * self._per_string(leaf, om_str, n_str, sub, chunk)).sum()
            G += torch.autograd.grad(s, leaf)[0]
        return G

    def __call__(self, geom_dict, **kwargs):
        """Same contract as FlowFisherResolutionLoss.__call__.

        kwargs: 'signal_event_params' or ('signal_sampler', 'num_events'),
                'fisher_res_metric' ('fom' | 'median' | 'mean'), 'use_relative_energy',
                'fisher_info_chunk_size' ((event, OM) rows per net pass).
        """
        if self.net is None:
            raise ValueError("train (fit) or load_model first")
        points_3d = geom_dict['points_3d']
        string_xy = geom_dict.get('string_xy', None)
        string_weights = geom_dict.get('string_weights', None)
        params = kwargs.get('signal_event_params', None)
        if params is None:
            sampler = kwargs.get('signal_sampler', None)
            if sampler is None:
                raise ValueError("provide signal_event_params or a signal_sampler")
            params = sampler.sample_events(int(kwargs.get('num_events', 100)))
        metric = kwargs.get('fisher_res_metric', 'fom')
        use_rel_e = bool(kwargs.get('use_relative_energy', False))
        chunk = int(kwargs.get('fisher_info_chunk_size', 65536))

        pts3 = _as_t(points_3d, self.device).reshape(-1, 3)
        om_str, n_str = _om_to_string(pts3, string_xy, self.device)
        ev = self._event_tensors(params)
        track = (bool(self.geometry_grads) and pts3.requires_grad
                 and torch.is_grad_enabled())
        chunked = track and self.geometry_grads == 'chunked'

        with torch.set_grad_enabled(track and not chunked):
            F_str = self._per_string(pts3 if track and not chunked else pts3.detach(),
                                     om_str, n_str, ev, chunk)
        F_in = F_str.detach().requires_grad_(True) if chunked else F_str
        if string_weights is None:
            F = F_in.sum(dim=1)
        else:
            w = torch.sigmoid(_as_t(string_weights, self.device))
            F = (w.reshape(1, -1, 1, 1) * F_in).sum(dim=1)

        res = resolution_from_fisher(F, self.fisher_params, ev['energy'], ev['zenith'],
                                     self.resolution_type, use_rel_e)
        ok = torch.isfinite(res) & (res > 1e-15)
        if ok.any():
            safe = torch.where(ok, torch.nan_to_num(res, nan=1e6, posinf=1e6).clamp_min(1e-15),
                               torch.full_like(res, 1e6))
            if metric == 'fom':
                total = 1.0 / torch.sqrt(torch.mean(1.0 / safe ** 2))
            elif metric == 'median':
                total = torch.median(safe)
            else:
                total = torch.mean(safe)
        else:
            total = torch.tensor(1.0, device=self.device, dtype=res.dtype, requires_grad=True)
        if self.print_loss:
            print(f'OMFisherNetLoss: {total.item():.6g} ({int(ok.sum())}/{len(res)} usable)')

        if chunked:
            A, = torch.autograd.grad(total, F_in, retain_graph=True, allow_unused=True)
            if A is not None:
                G = self._position_grad(pts3, om_str, n_str, ev, A, chunk)
                corr = (pts3 * G).sum()          # value 0, gradient G w.r.t. points_3d
                total = total + (corr - corr.detach())

        key = 'angular_resolution' if self.resolution_type == 'angular' else 'energy_resolution'
        return {f'{key}_loss': total, f'{key}_per_event': res,
                'resolution_params': params,
                'fisher_info_per_string_per_event': F_str.detach()}

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)
        cpu = lambda t: None if t is None else t.detach().cpu()
        torch.save({
            'net_state_dict': state if state is not None else self.net.state_dict(),
            'fisher_params': self.fisher_params, 'param_scales': self.param_scales,
            **{k: getattr(self, k) for k in self._CONTEXT_FLAGS},
            'width': self.width, 'depth': self.depth,
            'dropout': self.dropout, 'eps_floor': self.eps_floor,
            'weight_tau': self.weight_tau, 'tau': self.tau, 'target_mode': self.target_mode,
            'net_dtype': self.net_dtype,
            'context_mean': cpu(self.context_mean), 'context_std': cpu(self.context_std),
            'target_mean': cpu(self.target_mean), 'target_std': cpu(self.target_std),
            'tau_diag': cpu(self.tau_diag),
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
        }, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        for k in ('fisher_params', 'param_scales', *self._CONTEXT_FLAGS, 'width', 'depth',
                  'dropout', 'eps_floor', 'weight_tau', 'tau', 'target_mode', 'net_dtype',
                  'train_losses', 'val_losses'):
            if k in ck:
                setattr(self, k, ck[k])
            elif k in self._CONTEXT_FLAGS:
                print(f"Warning: {k} not found in checkpoint; using {getattr(self, k)!r}")
        self.build_network()
        self.net.load_state_dict(ck['net_state_dict'])
        for k in ('context_mean', 'context_std', 'target_mean', 'target_std', 'tau_diag'):
            setattr(self, k, None if ck.get(k) is None else ck[k].to(self.device))
        self._freeze()
        return self
