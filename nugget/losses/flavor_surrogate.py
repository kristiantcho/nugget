"""Differentiable surrogate g(event, geometry) ~ E[P(nu_mu) of the FlavorClassifier | event,
geometry], distilled from the classifier on nu_mu events in random sub-geometries of the dense grid.
Every input is computed in torch from the OM positions, so g is differentiable in the geometry."""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from nugget.losses.base_loss import LossFunction
from nugget.losses.flavor_classifier import FlavorClassifier
from nugget.surrogates.FlowMatchLY import _reseed_dataset_rng_in_worker, _resolve_pin_memory
from nugget.surrogates.FlowMatchReco import (
    FEATURE_SETS, AdaLNBlock, EncoderBlock, RecoFlowDataset, pad_events,
)

ENERGY_COLUMNS = {'neutrino': 'neutrino_energy', 'muon': 'muon_energy'}
VERTEX_COLUMNS = ('muon_x', 'muon_y', 'muon_z')
# the classifier's tokenisation settings the dataset copies from its data_config
TOKEN_KEYS = ('use_pulses', 'pulse_params', 'require_trigger', 'trigger_window',
              'trigger_min_pmts', 'max_tokens', 'feature_set', 'silent_tokens')
COT_CHERENKOV = 1.0 / math.sqrt(1.35 ** 2 - 1.0)   # phase index 1.35, as for the muon parquet
OM_FEATURES = ('s', 'd_perp', 'd', 'cos_vertex', 's_emit', 'dz', 'log_d_perp', 'log_d')
EVENT_FEATURES = ('log10_e', 'cos_zenith')


# --------------------------------------------------------------------------- #
#  Torch inputs                                                                #
# --------------------------------------------------------------------------- #

def om_features(om_pos, vertex, travel, length_scale=500.0):
    """(..., 3) OM positions relative to a track from `vertex` along `travel` -> (..., 8)
    OM_FEATURES: along-track and perpendicular distance, distance and cosine at the vertex,
    along-track position of the direct-light emission point and height above the vertex (all
    / length_scale), and log1p(distance / 10 m) of both distances."""
    u = travel / torch.linalg.norm(travel, dim=-1, keepdim=True).clamp_min(1e-12)
    r = om_pos - vertex
    s = (r * u).sum(-1)
    # +1 mm^2 keeps the gradients finite for an OM on the axis or at the vertex
    d_perp = torch.sqrt(((r - s.unsqueeze(-1) * u) ** 2).sum(-1) + 1e-6)
    d = torch.sqrt((r ** 2).sum(-1) + 1e-6)
    L = length_scale
    return torch.stack([s / L, d_perp / L, d / L, s / d, (s - d_perp * COT_CHERENKOV) / L,
                        r[..., 2] / L, torch.log1p(d_perp / 10.0), torch.log1p(d / 10.0)],
                       dim=-1)


def event_features(log10_e, travel, log10e_mean=0.0, log10e_std=1.0):
    """(B,) log10 E and (B, 3) travel directions -> (B, 2) EVENT_FEATURES."""
    u = travel / torch.linalg.norm(travel, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack([(log10_e - log10e_mean) / log10e_std, u[..., 2]], dim=-1)


def point_strings(points_3d, string_xy=None):
    """(n_pts,) string row of each point by exact (x, y) match (-1: on no string) and the number
    of rows. With string_xy the rows follow it, else they are the points' distinct (x, y)."""
    p = points_3d.detach()[:, :2]
    if string_xy is None:
        _, row = torch.unique(p, dim=0, return_inverse=True)
        return row, int(row.max()) + 1
    sxy = torch.as_tensor(string_xy).detach().to(p)
    _, code = torch.unique(torch.cat([sxy, p]), dim=0, return_inverse=True)
    table = torch.full((int(code.max()) + 1,), -1, dtype=torch.long, device=p.device)
    table[code[:len(sxy)]] = torch.arange(len(sxy), device=p.device)
    return table[code[len(sxy):]], len(sxy)


def string_layout(row, n_rows):
    """(n_pts,) string rows -> (n_rows, max points per string) point indices, -1 = padding."""
    keep = torch.nonzero(row >= 0).squeeze(1)
    r, order = torch.sort(row[keep], stable=True)
    keep = keep[order]
    counts = torch.bincount(r, minlength=n_rows)
    rank = torch.arange(len(r), device=row.device) - (torch.cumsum(counts, 0) - counts)[r]
    idx = torch.full((n_rows, max(int(counts.max()), 1)), -1, dtype=torch.long,
                     device=row.device)
    idx[r, rank] = keep
    return idx


def _event_tensors(event_params, device):
    """Event dicts (travel directions, as from the samplers) -> vertex (N, 3), travel (N, 3),
    log10 energy (N,)."""
    def t(x):
        return torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)

    V = torch.stack([t(e['position'])[:3] for e in event_params])
    if all(e.get('direction') is not None for e in event_params):
        U = torch.stack([t(e['direction'])[:3] for e in event_params])
    else:
        zen = torch.stack([t(e['zenith'])[0] for e in event_params])
        azi = torch.stack([t(e['azimuth'])[0] for e in event_params])
        U = torch.stack([torch.sin(zen) * torch.cos(azi), torch.sin(zen) * torch.sin(azi),
                         torch.cos(zen)], dim=1)
    E = torch.stack([t(e['energy'])[0] for e in event_params])
    return V, U, torch.log10(E)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class FlavorSurrogateDataset(Dataset):
    """nu_mu events of one parquet, each in its own random sub-geometry of the dense grid: the
    surrogate's inputs plus the teacher classifier's tokens of the event's simulated response.

    token_config is the teacher's data_config (tokenisation and default sub-geometries).
    undetected: 'redraw' an event left without pulses (the target is then E[P | detected]) or
    keep it with target 0 ('zero': E[P * detected]). deterministic: every batch's draws are a
    function of seed and its indices, so a validation set is the same on every pass.
    """

    def __init__(self, parquet_path, geometry_csv_path, token_config,
                 energy_column='neutrino_energy', num_events_per_epoch=None, seed=None,
                 uniform_energy_zenith=True, n_energy_bins=20, n_coszen_bins=20,
                 n_strings_range=None, om_drop_prob=None, om_drop_range=None,
                 undetected='redraw', parquet_dir_is_arrival=True, deterministic=False,
                 event_filter=None, max_redraws=1000, verbose=True):
        if undetected not in ('redraw', 'zero'):
            raise ValueError("undetected must be 'redraw' or 'zero'")
        masking = {k: token_config[k] if v is None else v for k, v in (
            ('n_strings_range', n_strings_range), ('om_drop_prob', om_drop_prob),
            ('om_drop_range', om_drop_range))}
        self.source = RecoFlowDataset(
            parquet_path, geometry_csv_path, num_events_per_epoch=num_events_per_epoch,
            seed=seed, energy_column=energy_column,
            uniform_energy_zenith=uniform_energy_zenith, n_energy_bins=n_energy_bins,
            n_coszen_bins=n_coszen_bins, event_filter=event_filter,
            event_columns=VERTEX_COLUMNS, verbose=verbose, **masking,
            **{k: token_config[k] for k in TOKEN_KEYS})
        src, geo = self.source, self.source.geometry
        self._seed = 0 if seed is None and deterministic else seed
        self.deterministic = bool(deterministic)
        self.undetected = undetected
        self.max_redraws = int(max_redraws)
        self.num_samples_per_epoch = src.num_samples_per_epoch
        self._n_om = int(geo.str_om_count.max())
        self._n_cols = len(FEATURE_SETS[src.feature_set]['columns'])
        self.vertex = np.stack([src.event_columns[c] for c in VERTEX_COLUMNS],
                               axis=1).astype(np.float32)
        # muon parquet zenith/azimuth point back to where the neutrino came from
        self.travel = -src._dir if parquet_dir_is_arrival else src._dir.copy()
        self.log10e = src._log10e
        self.config = {**src.masking, 'energy_column': energy_column,
                       'uniform_energy_zenith': bool(uniform_energy_zenith),
                       'undetected': undetected,
                       'parquet_dir_is_arrival': bool(parquet_dir_is_arrival),
                       'token_config': dict(token_config)}

    @property
    def _rng(self):
        return self.source._rng

    @_rng.setter
    def _rng(self, rng):
        self.source._rng = rng

    def __len__(self):
        return self.num_samples_per_epoch

    def get_batch(self, indices):
        """Dict: om_pos (B, S, O, 3), om_mask (B, S, O), str_mask (B, S), vertex, travel,
        log10_e, detected, event, and 'clf' = the detected rows' teacher tokens."""
        idx = np.asarray(indices).reshape(-1)
        if self.deterministic:
            self._rng = np.random.default_rng([int(self._seed), int(idx[0])])
        src, geo, rng = self.source, self.source.geometry, self._rng
        n = len(idx)
        ev, det = np.empty(n, dtype=np.int64), np.zeros(n, dtype=bool)
        toks, strings, active = [], [], []
        for i in range(n):
            # a fresh sub-geometry with every redraw, so given the geometry the recorded
            # events keep their true distribution
            for _ in range(self.max_redraws):
                act = geo.sample_active(rng, **src.masking)
                e = int(src._sample_events(1)[0])
                tok = src.tokens(e, om_active=act)
                if tok is not None or self.undetected == 'zero':
                    break
            else:
                raise RuntimeError(f"no event survived masking/trigger in {self.max_redraws} "
                                   "draws; use more strings")
            ev[i], det[i] = e, tok is not None
            if tok is not None:
                toks.append(tok)
            strings.append(np.flatnonzero(np.logical_or.reduceat(act, geo.str_om_start)))
            active.append(act)

        S, O = max(len(s) for s in strings), self._n_om
        om_pos = np.zeros((n, S, O, 3), dtype=np.float32)
        om_mask = np.zeros((n, S, O), dtype=bool)
        str_mask = np.zeros((n, S), dtype=bool)
        for i, (s, act) in enumerate(zip(strings, active)):
            on = np.arange(O) < geo.str_om_count[s][:, None]
            om = np.where(on, geo.str_om_start[s][:, None] + np.arange(O), 0)
            om_pos[i, :len(s)] = geo.om_pos[om]
            om_mask[i, :len(s)] = on & act[om]
            str_mask[i, :len(s)] = True
        clf = (pad_events(toks) if toks else
               (torch.zeros(0, 1, self._n_cols), torch.zeros(0, 1, dtype=torch.bool),
                torch.zeros(0, 3)))
        return {'om_pos': torch.from_numpy(om_pos), 'om_mask': torch.from_numpy(om_mask),
                'str_mask': torch.from_numpy(str_mask),
                'vertex': torch.from_numpy(self.vertex[ev]),
                'travel': torch.from_numpy(self.travel[ev]),
                'log10_e': torch.from_numpy(self.log10e[ev]),
                'detected': torch.from_numpy(det), 'event': torch.from_numpy(ev), 'clf': clf}

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        if isinstance(idx, (list, np.ndarray)):
            return self.get_batch(idx)
        return self.get_batch([idx])


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class _Pool(torch.nn.Module):
    """Masked mean and single-query attention pooling over the present tokens."""

    def __init__(self, d):
        super().__init__()
        self.key = torch.nn.Linear(d, d)
        self.query = torch.nn.Parameter(torch.zeros(d))
        self.scale = d ** -0.5

    def forward(self, h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(-2) / m.sum(-2).clamp_min(1.0)
        score = ((self.key(h) @ self.query) * self.scale).masked_fill(~mask, float('-inf'))
        a = torch.softmax(score, dim=-1)
        return torch.cat([mean, (a.unsqueeze(-1) * h).sum(-2)], dim=-1)


class _FinalLayer(torch.nn.Module):
    """adaLN-modulated LayerNorm -> one logit."""

    def __init__(self, width, cond_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mod = torch.nn.Linear(cond_dim, 2 * width)
        torch.nn.init.zeros_(self.mod.weight)
        torch.nn.init.zeros_(self.mod.bias)
        self.out = torch.nn.Linear(width, 1)

    def forward(self, h, c):
        shift, scale = self.mod(c).chunk(2, dim=-1)
        return self.out(self.norm(h) * (1.0 + scale) + shift).squeeze(-1)


class FlavorSurrogateNet(torch.nn.Module):
    """OM tokens -> attention within each string -> pooled string tokens -> attention across
    strings -> pooled event vector -> adaLN MLP conditioned on the event -> logit."""

    def __init__(self, d_model=64, om_depth=2, string_depth=4, n_heads=4, ffn_dim=256,
                 cond_dim=128, dropout=0.0, head_width=256, head_depth=4, mlp_ratio=2,
                 head_dropout=0.0):
        super().__init__()
        self.om_in = torch.nn.Sequential(torch.nn.Linear(len(OM_FEATURES), d_model),
                                         torch.nn.GELU(), torch.nn.Linear(d_model, d_model))
        self.om_blocks = torch.nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(om_depth)])
        self.om_pool = _Pool(d_model)
        self.str_in = torch.nn.Sequential(torch.nn.LayerNorm(2 * d_model),
                                          torch.nn.Linear(2 * d_model, d_model))
        self.str_blocks = torch.nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(string_depth)])
        self.str_pool = _Pool(d_model)
        self.ctx = torch.nn.Sequential(torch.nn.Linear(len(EVENT_FEATURES), cond_dim),
                                       torch.nn.SiLU(), torch.nn.Linear(cond_dim, cond_dim))
        self.head_in = torch.nn.Sequential(torch.nn.LayerNorm(2 * d_model),
                                           torch.nn.Linear(2 * d_model, head_width))
        self.head = torch.nn.ModuleList(
            [AdaLNBlock(head_width, cond_dim, mlp_ratio, head_dropout)
             for _ in range(head_depth)])
        self.out = _FinalLayer(head_width, cond_dim)

    def forward(self, om_feats, om_mask, str_mask, ev_feats):
        """(B, S, O, F) OM features, (B, S, O) and (B, S) presence masks, (B, E) event
        features."""
        B, S, O, _ = om_feats.shape
        # padded strings attend to their zero tokens (no all-masked rows); str_mask drops them
        om_mask = (om_mask | ~om_mask.any(-1, keepdim=True)).reshape(B * S, O)
        h = self.om_in(om_feats).reshape(B * S, O, -1)
        for blk in self.om_blocks:
            h = blk(h, om_mask[:, None, None, :])
        s = self.str_in(self.om_pool(h, om_mask).reshape(B, S, -1))
        for blk in self.str_blocks:
            s = blk(s, str_mask[:, None, None, :])
        c = self.ctx(ev_feats)
        x = self.head_in(self.str_pool(s, str_mask))
        for blk in self.head:
            x = blk(x, c)
        return self.out(x, c)


class FlavorSurrogate:
    """g(event, geometry) ~ E[P(nu_mu) | event, geometry] of a FlavorClassifier teacher."""

    _CONFIG_KEYS = ('energy_type', 'd_model', 'om_depth', 'string_depth', 'n_heads',
                    'ffn_dim', 'cond_dim', 'dropout', 'head_width', 'head_depth', 'mlp_ratio',
                    'head_dropout', 'length_scale', 'signal_prior', 'log10e_mean',
                    'log10e_std')

    # same dtype and optimisation plumbing as the classifier
    param_dtype = FlavorClassifier.param_dtype
    _prep = FlavorClassifier._prep
    _build_scheduler = FlavorClassifier._build_scheduler
    _scheduler_step = FlavorClassifier._scheduler_step

    def __init__(self, device=None, energy_type='neutrino', d_model=64, om_depth=2,
                 string_depth=4, n_heads=4, ffn_dim=256, cond_dim=128, dropout=0.0,
                 head_width=256, head_depth=4, mlp_ratio=2, head_dropout=0.0,
                 length_scale=500.0, signal_prior=None, learning_rate=3e-4,
                 weight_decay=1e-5, lr_schedule='warmup_cosine', warmup_steps=1000,
                 warmup_frac=0.1, min_lr=1e-5, amp=False):
        if energy_type not in ENERGY_COLUMNS:
            raise ValueError(f"energy_type must be one of {sorted(ENERGY_COLUMNS)}")
        if lr_schedule not in ('warmup_cosine', 'onecycle', 'constant'):
            raise ValueError("lr_schedule must be 'warmup_cosine', 'onecycle' or 'constant'")
        self.device = device if device is not None else torch.device('cpu')
        self.energy_type = energy_type
        self.d_model, self.om_depth, self.string_depth = d_model, om_depth, string_depth
        self.n_heads, self.ffn_dim, self.cond_dim, self.dropout = n_heads, ffn_dim, cond_dim, dropout
        self.head_width, self.head_depth, self.mlp_ratio = head_width, head_depth, mlp_ratio
        self.head_dropout, self.length_scale = head_dropout, length_scale
        # prior of the teacher's P(nu_mu) the surrogate learns (None = its training prior)
        self.signal_prior = signal_prior
        self.log10e_mean, self.log10e_std = None, None
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.lr_schedule, self.warmup_steps = lr_schedule, warmup_steps
        self.warmup_frac, self.min_lr = warmup_frac, min_lr
        self.amp = amp

        self.data_config = None
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_losses, self.val_losses, self.metrics = [], [], []
        self.best_state_dict = None
        self.is_trained = False

    def _autocast(self, amp=None):
        dev = torch.device(self.device).type
        on = self.amp if amp is None else amp
        return torch.autocast(dev, dtype=torch.bfloat16, enabled=bool(on) and dev == 'cuda')

    # ---------------- network ----------------

    def build_network(self):
        # explicit float32: nugget sets a float64 default dtype
        self.net = FlavorSurrogateNet(
            self.d_model, self.om_depth, self.string_depth, self.n_heads, self.ffn_dim,
            self.cond_dim, self.dropout, self.head_width, self.head_depth, self.mlp_ratio,
            self.head_dropout).to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        print(f"FlavorSurrogate: {self.energy_type} energy, d_model={self.d_model}, "
              f"OM/string depth {self.om_depth}/{self.string_depth}, head "
              f"{self.head_width}x{self.head_depth} "
              f"({sum(p.numel() for p in self.net.parameters()):,} params)")

    # ---------------- prediction ----------------

    def logits(self, om_pos, om_mask, str_mask, vertex, travel, log10_e, amp=None):
        """(B,) logits of the expected P(nu_mu) for OMs om_pos (B, S, O, 3) on strings (axis
        1; om_mask/str_mask mark the present ones) and events vertex, travel (B, 3), log10_e
        (B,)."""
        om_pos, vertex, travel = self._prep(om_pos), self._prep(vertex), self._prep(travel)
        feats = om_features(om_pos, vertex[:, None, None], travel[:, None, None],
                            self.length_scale)
        mu, sd = ((0.0, 1.0) if self.log10e_mean is None
                  else (self.log10e_mean, self.log10e_std))
        ctx = event_features(self._prep(log10_e), travel, mu, sd)
        with self._autocast(amp):
            z = self.net(feats, om_mask.to(self.device), str_mask.to(self.device), ctx)
        return z.to(self.param_dtype)

    def batch_logits(self, batch):
        return self.logits(batch['om_pos'], batch['om_mask'], batch['str_mask'],
                           batch['vertex'], batch['travel'], batch['log10_e'])

    def predict_geometry(self, points_3d, event_params, string_xy=None, events_per_batch=64,
                         amp=False):
        """(n_events,) expected P(nu_mu) of event_params (sampler dicts: travel directions,
        'energy' of self.energy_type) in the detector points_3d, differentiable in points_3d.
        Points form strings by exact (x, y)."""
        self.net.eval()
        pts = torch.as_tensor(points_3d, device=self.device)
        row, n_rows = point_strings(pts, string_xy)
        idx = string_layout(row, n_rows)
        om_pos, om_mask = pts[idx.clamp(min=0)], idx >= 0
        str_mask = om_mask.any(1)
        V, U, logE = _event_tensors(event_params, self.device)
        out = []
        for s in range(0, len(V), events_per_batch):
            B = len(V[s:s + events_per_batch])
            out.append(torch.sigmoid(self.logits(
                om_pos.expand(B, -1, -1, -1), om_mask.expand(B, -1, -1),
                str_mask.expand(B, -1), V[s:s + B], U[s:s + B], logE[s:s + B], amp=amp)))
        return torch.cat(out)

    # ---------------- training ----------------

    @torch.no_grad()
    def teacher_targets(self, teacher, batch):
        """(B,) the teacher's P(nu_mu) of each sampled response; 0 for undetected events."""
        det = batch['detected'].to(self.device)
        t = torch.zeros(len(det), device=self.device, dtype=self.param_dtype)
        if det.any():
            t[det] = teacher.predict_proba(*batch['clf'], signal_prior=self.signal_prior).to(
                self.param_dtype)
        return t

    @torch.no_grad()
    def evaluate(self, dataloader, teacher, max_batches=None, n_bins=10):
        """BCE, MSE and R^2 against the teacher's sampled P(nu_mu), and the calibration error
        over n_bins quantile bins of the prediction (mean prediction vs mean teacher P)."""
        self.net.eval()
        p, t = [], []
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            p.append(torch.sigmoid(self.batch_logits(batch)).float().cpu())
            t.append(self.teacher_targets(teacher, batch).float().cpu())
        p, t = (torch.cat(x).double().numpy() for x in (p, t))
        pc = np.clip(p, 1e-7, 1.0 - 1e-7)
        mse = float(np.mean((p - t) ** 2))
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        b = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, n_bins - 1)
        gap = np.abs(np.bincount(b, p, n_bins) - np.bincount(b, t, n_bins))
        return {'bce': float(np.mean(-(t * np.log(pc) + (1.0 - t) * np.log1p(-pc)))),
                'mse': mse, 'r2': 1.0 - mse / max(float(np.var(t)), 1e-12),
                'ece': float(gap.sum() / len(p)), 'mean_pred': float(p.mean()),
                'mean_target': float(t.mean()), 'n_events': int(len(p))}

    def train_with_dataloader(self, train_dataloader, teacher, val_dataloader=None, epochs=100,
                              verbose=True, early_stopping_patience=20, grad_clip=1.0,
                              save_every_n_epochs=None, checkpoint_path=None, val_batches=None):
        """BCE against the teacher's P(nu_mu) of one sampled response per (event, geometry),
        whose minimiser is the conditional mean. Early stopping and best weights on val BCE."""
        ds = train_dataloader.dataset
        if self.log10e_mean is None:
            self.log10e_mean, self.log10e_std = float(ds.log10e.mean()), float(ds.log10e.std())
        if self.net is None:
            self.build_network()
        self.data_config = dict(ds.config)
        self._build_scheduler(len(train_dataloader) * epochs)
        teacher.net.eval()

        best_val, patience = float('inf'), 0
        for epoch in range(epochs):
            self.net.train()
            t0 = time.time()
            tl, nb = 0.0, 0
            for batch in train_dataloader:
                target = self.teacher_targets(teacher, batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(self.batch_logits(batch), target)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                self.optimizer.step()
                self._scheduler_step()
                tl += loss.item()
                nb += 1
            tl /= max(nb, 1)
            self.train_losses.append(tl)

            m = None
            if val_dataloader is not None:
                m = self.evaluate(val_dataloader, teacher, max_batches=val_batches)
                self.val_losses.append(m['bce'])
                self.metrics.append({'epoch': epoch + 1, **m})
                if m['bce'] < best_val:
                    best_val, patience = m['bce'], 0
                    self.best_state_dict = {k: v.detach().clone()
                                            for k, v in self.net.state_dict().items()}
                else:
                    patience += 1

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs}  train {tl:.5f}"
                if m is not None:
                    msg += (f"  val {m['bce']:.5f}  mse {m['mse']:.5f}  r2 {m['r2']:.4f}  "
                            f"ece {m['ece']:.4f}  mean {m['mean_pred']:.4f}/"
                            f"{m['mean_target']:.4f}")
                print(msg + f"  lr {self.optimizer.param_groups[0]['lr']:.2e}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

            if val_dataloader is not None and patience >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (best val {best_val:.5f})")
                break

            if save_every_n_epochs and checkpoint_path and (
                    (epoch + 1) % save_every_n_epochs == 0 or epoch + 1 == epochs):
                d = os.path.dirname(checkpoint_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                self.save_model(checkpoint_path, state=self.best_state_dict)

        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)
            if verbose:
                print(f"Restored best weights (val {best_val:.5f})")
        self.is_trained = True
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses,
                'metrics': self.metrics}

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        torch.save({
            'config': {k: getattr(self, k) for k in self._CONFIG_KEYS},
            'net_state_dict': state if state is not None else self.net.state_dict(),
            'data_config': self.data_config,
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
            'metrics': self.metrics,
        }, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        for k, v in ck['config'].items():
            setattr(self, k, v)
        for k in ('data_config', 'train_losses', 'val_losses', 'metrics'):
            setattr(self, k, ck.get(k, getattr(self, k)))
        self.build_network()
        self.net.load_state_dict(ck['net_state_dict'])
        self.is_trained = True
        return self

    # ---------------- dataloaders ----------------

    def create_dataloader(self, parquet_path, teacher, geometry_csv_path,
                          num_events_per_epoch=None, batch_size=128, shuffle=True,
                          num_workers=0, seed=None, exclude_parquet=None, deterministic=False,
                          pin_memory=None, **data_kw):
        """Loader over a nu_mu parquet tokenised as `teacher` was trained; exclude_parquet
        leaves out its events (e.g. the teacher's test set). data_kw go to
        FlavorSurrogateDataset (sub-geometries, undetected, uniform_energy_zenith, ...)."""
        event_filter = None
        if exclude_parquet is not None:
            from nugget.utils.parquet_io import list_events
            event_filter = set(list_events(parquet_path)) - set(list_events(exclude_parquet))
        ds = FlavorSurrogateDataset(
            parquet_path, geometry_csv_path, teacher.data_config,
            energy_column=ENERGY_COLUMNS[self.energy_type],
            num_events_per_epoch=num_events_per_epoch, seed=seed, deterministic=deterministic,
            event_filter=event_filter, **data_kw)
        base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        kw = dict(dataset=ds, batch_size=None,
                  sampler=BatchSampler(base, batch_size=batch_size, drop_last=False),
                  num_workers=num_workers,
                  pin_memory=_resolve_pin_memory(self.device, pin_memory))
        if num_workers > 0:
            kw['worker_init_fn'] = _reseed_dataset_rng_in_worker
            kw['persistent_workers'] = True
        return DataLoader(**kw)

    def create_val_dataloader(self, parquet_path, teacher, geometry_csv_path,
                              num_events_per_epoch=None, batch_size=128, num_workers=0, seed=0,
                              **kw):
        """Fixed draws: every pass sees the same events, sub-geometries and responses."""
        return self.create_dataloader(
            parquet_path, teacher, geometry_csv_path, num_events_per_epoch=num_events_per_epoch,
            batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed,
            deterministic=True, **kw)


# --------------------------------------------------------------------------- #
#  Loss                                                                        #
# --------------------------------------------------------------------------- #

class FlavorSurrogateLoss(LossFunction):
    """1 - mean expected P(nu_mu) of the signal events in geom_dict['points_3d'].

    Returns 'flavor_surrogate_loss' and 'flavor_surrogate_prob_per_event'. kwargs:
    'signal_event_params' (or 'signal_sampler' + 'num_events'), 'flavor_surrogate_events_per_batch'
    (64), 'flavor_surrogate_event_weights' (per-event weights of the mean).
    """

    def __init__(self, surrogate, device=None):
        super().__init__(device)
        if isinstance(surrogate, str):
            surrogate = FlavorSurrogate(device=self.device).load_model(surrogate)
        self.surrogate = surrogate
        self.surrogate.net.requires_grad_(False)

    def __call__(self, geom_dict, **kwargs):
        params = kwargs.get('signal_event_params', None)
        if params is None:
            sampler = kwargs.get('signal_sampler', None)
            if sampler is None:
                raise ValueError("provide signal_event_params or a signal_sampler")
            params = sampler.sample_events(int(kwargs.get('num_events', 100)))
        p = self.surrogate.predict_geometry(
            geom_dict['points_3d'], params, string_xy=geom_dict.get('string_xy', None),
            events_per_batch=int(kwargs.get('flavor_surrogate_events_per_batch', 64)))
        w = kwargs.get('flavor_surrogate_event_weights', None)
        if w is None:
            mean = p.mean()
        else:
            w = torch.as_tensor(w, device=p.device, dtype=p.dtype)
            mean = (w * p).sum() / w.sum()
        return {'flavor_surrogate_loss': 1.0 - mean,
                'flavor_surrogate_prob_per_event': p}
