"""Riemannian flow matching for p(log10 E, direction | detector response) on R x S^2.

The pulse simulation and the 27 PMT token features follow K. Bas, MSc thesis (TUM, 2026),
Sec. 5.4-5.5 and Sec. 8.1, which in turn follow P-ONE Offline.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from nugget.surrogates.base_surrogate import Surrogate
from nugget.surrogates.FlowMatchLY import (
    TimeEmbedding, _reseed_dataset_rng_in_worker, _event_split,
)
from nugget.surrogates.FlowMatchATime import _ragged_indices

LOG_4PI = math.log(4.0 * math.pi)
LOG_2PI = math.log(2.0 * math.pi)

# Thesis Table 8.1 (times relative to the event's first pulse, positions to its pulse-count
# weighted centre), the charge within 100/500 ns of the PMT's first pulse, then the token
# type and the silent-string description. Hit PMTs have hit=1; a silent OM on a hit string
# has hit=string_token=0 and no orientation; a silent string sits at (x, y, mean z) of its
# active OMs with their count and vertical extent.
THESIS_FEATURES = (
    'x', 'y', 'z', 'dx', 'dy', 'dz', 'ux', 'uy', 'uz',
    'q_tot', 'q_max', 'q_mean', 'q_std', 'q10', 'q50', 'q90',
    'f_charge', 'n_pulses',
    't_first', 't10', 't20', 't50', 't80', 't90', 't_mean_q', 't_std_q', 't_span',
)
TOKEN_FLAGS = ('hit', 'string_token', 'n_om', 'z_span')
FEATURE_NAMES = THESIS_FEATURES + ('q_100', 'q_500') + TOKEN_FLAGS
COLUMN = {n: i for i, n in enumerate(FEATURE_NAMES)}
GLOBAL_NAMES = ('n_pmt', 'q_event', 'n_pulses_event')

# x, y, z lead every set because the position encoding reads them; the input layer sees the
# columns from `linear_from` on. 'essential' drops what is collinear or noise on the muon
# parquet: 64% of hit PMTs carry one pulse, the per-pulse charge spread is SPE noise,
# t10/t20 equal t_first, and t_std/t_span are dominated by afterpulses > 1 us late. For the
# energy it keeps q_max and n_pulses (merging/saturation in bright PMTs) and the prompt
# charges q_100/q_500, which exclude afterpulses.
FEATURE_SETS = {
    'essential': dict(columns=('x', 'y', 'z', 'dx', 'dy', 'dz', 'ux', 'uy', 'uz',
                               'q_tot', 'q_max', 'q_100', 'q_500', 'n_pulses',
                               't_first', 't50') + TOKEN_FLAGS,
                      linear_from=3),
    'thesis': dict(columns=THESIS_FEATURES + TOKEN_FLAGS, linear_from=0),
}
_LENGTH_COLS = {'x', 'y', 'z', 'dx', 'dy', 'dz', 'z_span'}
_CHARGE_COLS = {'q_tot', 'q_max', 'q_mean', 'q_std', 'q10', 'q50', 'q90', 'q_100', 'q_500'}
_COUNT_COLS = {'n_pulses', 'n_om'}
_TIME_COLS = {'t_first', 't10', 't20', 't50', 't80', 't90', 't_mean_q', 't_std_q', 't_span'}

# Thesis Table 5.3 (P-ONE Offline values); times in ns, charges in p.e.
PULSE_DEFAULTS = dict(
    tts=3.0, late_prob=0.03, late_mean=50.0,
    ap_prob=0.06, ap_weights=(0.37, 0.63), ap_means=(1254.0, 3193.0),
    ap_sigmas=(90.0, 587.1),
    merge_dt=0.2, q_mean=1.0, q_sigma=0.3, q_thr=0.25, q_sat=100.0,
)


# --------------------------------------------------------------------------- #
#  Photons -> pulses -> PMT tokens (numpy)                                     #
# --------------------------------------------------------------------------- #

def merge_close_pulses(lab, t, q, dt):
    """Merge the closest adjacent pulse pair on a PMT while its gap is <= dt.

    Inputs sorted by (lab, t). A merged pulse has the summed charge and the time of the
    larger-charge member (the earlier one on a tie).
    """
    while len(t) > 1:
        gap = np.diff(t)
        gap[lab[1:] != lab[:-1]] = np.inf
        close = gap <= dt
        if not close.any():
            break
        # A merge only ever widens the neighbouring gaps (the merged pulse keeps one of its
        # two times), so every local-minimum gap gets merged exactly as the sequential
        # closest-first procedure would merge it, and they can all go in one pass.
        left = np.concatenate([[np.inf], gap[:-1]])
        right = np.concatenate([gap[1:], [np.inf]])
        i = np.flatnonzero(close & (gap < left) & (gap <= right))
        t[i] = np.where(q[i] >= q[i + 1], t[i], t[i + 1])
        q[i] = q[i] + q[i + 1]
        keep = np.ones(len(t), dtype=bool)
        keep[i + 1] = False
        lab, t, q = lab[keep], t[keep], q[keep]
    return lab, t, q


class PulseSimulator:
    """Accepted photons -> reconstructed pulses (thesis Eqs. 5.9-5.13)."""

    def __init__(self, **params):
        unknown = set(params) - set(PULSE_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown pulse parameters {sorted(unknown)}")
        self.params = {**PULSE_DEFAULTS, **params}
        for k, v in self.params.items():
            setattr(self, k, v)

    def __call__(self, lab, t, rng):
        """(n,) PMT labels and photon times -> (lab, t, q) pulses sorted by (lab, t)."""
        lab = np.asarray(lab, dtype=np.int64)
        t = np.asarray(t, dtype=np.float64) + rng.normal(0.0, self.tts, len(t))
        late = rng.random(len(t)) < self.late_prob
        t[late] += rng.normal(self.late_mean, math.sqrt(2.0) * self.tts, int(late.sum()))
        ap = np.flatnonzero(rng.random(len(t)) < self.ap_prob)
        if len(ap):
            w = np.asarray(self.ap_weights, dtype=np.float64)
            k = rng.choice(len(w), size=len(ap), p=w / w.sum())
            t_ap = t[ap] + rng.normal(np.asarray(self.ap_means)[k],
                                      np.asarray(self.ap_sigmas)[k])
            lab = np.concatenate([lab, lab[ap]])
            t = np.concatenate([t, t_ap])
        # the Gaussian SPE charge goes negative ~1e-4 of the time; clip so the charge
        # weights used by the features stay non-negative
        q = np.clip(rng.normal(self.q_mean, self.q_sigma, len(t)), 0.0, None)
        order = np.lexsort((t, lab))
        lab, t, q = merge_close_pulses(lab[order], t[order], q[order], self.merge_dt)
        keep = q >= self.q_thr
        lab, t, q = lab[keep], t[keep], q[keep]
        sat = q > self.q_sat
        q[sat] = self.q_sat + (q[sat] - self.q_sat) * self.q_sat / q[sat]
        return lab, t, q


def om_triggered(om, pmt, t, window=10.0, min_pmts=3, chunk=200_000):
    """True if some OM has pulses on >= min_pmts distinct PMTs within `window` ns."""
    om = np.asarray(om, dtype=np.int64)
    pmt = np.asarray(pmt, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    if len(t) < min_pmts:
        return False
    P = int(pmt.max()) + 1
    uch, cinv = np.unique(om * P + pmt, return_inverse=True)
    uom = uch // P
    oms, n_ch = np.unique(uom, return_counts=True)
    cand = oms[n_ch >= min_pmts]
    if len(cand) == 0:
        return False
    sel = np.isin(om, cand)
    om, cinv, t = om[sel], cinv[sel], t[sel]
    # channel-major monotone key: channel c's pulses live in [c*span, (c+1)*span)
    t0 = t.min()
    span = t.max() - t0 + window + 1.0
    key = np.sort(cinv * span + (t - t0))
    lo_ch = np.searchsorted(uom, om, side='left')
    n_om = np.searchsorted(uom, om, side='right') - lo_ch
    m = int(n_om.max())
    for s in range(0, len(t), chunk):
        sl = slice(s, s + chunk)
        c = lo_ch[sl, None] + np.arange(m)
        lo = c * span + (t[sl, None] - t0)
        idx = np.minimum(np.searchsorted(key, lo, side='left'), len(key) - 1)
        hit = (np.arange(m) < n_om[sl, None]) & (key[idx] >= lo) & (key[idx] <= lo + window)
        if (hit.sum(1) >= min_pmts).any():
            return True
    return False


def pmt_features(tok, t, q, pos, dirs):
    """Pulses sorted by (tok, t), tok covering 0..n-1 -> (n, 29) raw PMT features
    (THESIS_FEATURES, q_100, q_500) and the (3,) globals (n_pmt, total charge, n_pulses)."""
    n = len(pos)
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    starts = np.searchsorted(tok, np.arange(n))
    ends = np.append(starts[1:], len(t))
    last = ends - 1
    npl = (ends - starts).astype(np.float64)

    q_tot = np.add.reduceat(q, starts)
    q_max = np.maximum.reduceat(q, starts)
    q_mean = q_tot / npl
    q_std = np.sqrt(np.maximum(np.add.reduceat(q * q, starts) / npl - q_mean ** 2, 0.0))
    qs = q[np.lexsort((q, tok))]
    q_pct = [qs[starts + np.clip(np.ceil(p * npl).astype(np.int64) - 1, 0, None)]
             for p in (0.1, 0.5, 0.9)]

    Q = max(float(q_tot.sum()), 1e-12)
    tol = 1e-9 * Q
    w = np.maximum(q_tot, 1e-12)
    cq = np.cumsum(q)
    before = cq[starts] - q[starts]

    def t_quantile(p):
        idx = np.searchsorted(cq, before + p * q_tot - tol, side='left')
        return t[np.clip(idx, starts, last)]

    t_first = t[starts]
    dtf = t - np.repeat(t_first, ends - starts)
    m1 = np.add.reduceat(q * dtf, starts) / w
    t_std = np.sqrt(np.maximum(np.add.reduceat(q * dtf * dtf, starts) / w - m1 ** 2, 0.0))
    q_100 = np.add.reduceat(q * (dtf <= 100.0), starts)
    q_500 = np.add.reduceat(q * (dtf <= 500.0), starts)

    center = (npl[:, None] * pos).sum(0) / npl.sum()
    t0 = t.min()

    feats = np.column_stack([
        pos, pos - center, dirs,
        q_tot, q_max, q_mean, q_std, *q_pct,
        q_tot / Q, npl,
        t_first - t0, *[t_quantile(p) - t0 for p in (0.1, 0.2, 0.5, 0.8, 0.9)],
        t_first + m1 - t0, t_std, t[last] - t_first,
        q_100, q_500,
    ])
    return feats.astype(np.float32), np.array([n, Q, len(t)], dtype=np.float32)


class DetectorGeometry:
    """PMT table sorted by (string, om, pmt), with its OM and string structure."""

    def __init__(self, string, om, pmt, pmt_pos, pmt_dir):
        import pandas as pd
        string, om, pmt = (np.asarray(x, dtype=np.int64) for x in (string, om, pmt))
        order = np.lexsort((pmt, om, string))
        self.string, self.om, self.pmt = string[order], om[order], pmt[order]
        self.pmt_pos = np.asarray(pmt_pos, dtype=np.float64)[order]
        self.pmt_dir = np.asarray(pmt_dir, dtype=np.float64)[order]
        new_om = np.concatenate([[True], (np.diff(self.string) != 0) | (np.diff(self.om) != 0)])
        self.pmt_om = np.cumsum(new_om) - 1
        om_start = np.flatnonzero(new_om)
        self.n_om = len(om_start)
        self.om_pos = (np.add.reduceat(self.pmt_pos, om_start, axis=0)
                       / np.diff(np.append(om_start, len(order)))[:, None])
        om_str = self.string[om_start]
        new_str = np.concatenate([[True], np.diff(om_str) != 0])
        self.om_string = np.cumsum(new_str) - 1
        self.str_om_start = np.flatnonzero(new_str)
        self.str_om_count = np.diff(np.append(self.str_om_start, self.n_om))
        self.n_strings = len(self.str_om_start)
        self.str_xy = (np.add.reduceat(self.om_pos[:, :2], self.str_om_start, axis=0)
                       / self.str_om_count[:, None])
        self._m_om, self._m_pmt = int(self.om.max()) + 1, int(self.pmt.max()) + 1
        self._index = pd.Index(self._key(self.string, self.om, self.pmt))

    @classmethod
    def from_csv(cls, path):
        import pandas as pd
        g = pd.read_csv(path, usecols=['string', 'om', 'pmt', 'om_x', 'om_y', 'om_z',
                                       'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z'])
        return cls(g.string, g.om, g.pmt, g[['om_x', 'om_y', 'om_z']].to_numpy(),
                   g[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy())

    def _key(self, s, o, p):
        return (s * self._m_om + o) * self._m_pmt + p

    def lookup(self, string, om, pmt):
        """PMT indices of (string, om, pmt); -1 where the geometry has no such PMT."""
        s, o, p = (np.asarray(x, dtype=np.int64) for x in (string, om, pmt))
        idx = self._index.get_indexer(self._key(s, o, p))
        idx[(o < 0) | (o >= self._m_om) | (p < 0) | (p >= self._m_pmt)] = -1
        return idx

    def sample_active(self, rng, n_strings_range=(10, 150), spread_range=(40.0, 1000.0),
                      om_drop_prob=0.0, om_drop_range=(0.0, 0.5)):
        """Random sub-geometry as a bool mask over OMs.

        n strings (uniform in n_strings_range, None = all) are drawn without replacement with
        weight exp(-d^2 / 2s^2) around a random string, s log-uniform in spread_range m: small
        s gives a compact cluster, large s scattered strings. With probability om_drop_prob
        each OM is then dropped with a probability drawn from om_drop_range.
        """
        keep = np.ones(self.n_strings, dtype=bool)
        if n_strings_range is not None:
            n = min(int(rng.integers(n_strings_range[0], n_strings_range[1] + 1)),
                    self.n_strings)
            s = math.exp(rng.uniform(math.log(spread_range[0]), math.log(spread_range[1])))
            c = self.str_xy[rng.integers(self.n_strings)]
            # Gumbel top-k = weighted sampling without replacement; done on log weights so
            # far strings keep a tiny but nonzero chance
            key = (-((self.str_xy - c) ** 2).sum(1) / (2.0 * s * s)
                   - np.log(-np.log(rng.random(self.n_strings))))
            keep[:] = False
            keep[np.argpartition(-key, n - 1)[:n]] = True
        act = keep[self.om_string]
        if om_drop_prob > 0 and rng.random() < om_drop_prob:
            act &= rng.random(self.n_om) >= rng.uniform(*om_drop_range)
        return act

    def silent_tokens(self, hit, center, om_active=None):
        """Token rows for the active silent OMs on hit strings and for the silent strings,
        each group ordered by distance to `center`. `hit` are the hit PMT indices."""
        act = np.ones(self.n_om, dtype=bool) if om_active is None else om_active
        hit_om = np.unique(self.pmt_om[hit])
        hit_str = np.unique(self.om_string[hit_om])

        oms = _ragged_indices(self.str_om_start[hit_str], self.str_om_count[hit_str])
        lit = np.zeros(self.n_om, dtype=bool)
        lit[hit_om] = True
        oms = oms[act[oms] & ~lit[oms]]
        oms = oms[np.argsort(np.linalg.norm(self.om_pos[oms] - center, axis=1), kind='stable')]
        om_rows = np.zeros((len(oms), len(FEATURE_NAMES)), dtype=np.float32)
        om_rows[:, 0:3] = self.om_pos[oms]
        om_rows[:, 3:6] = self.om_pos[oms] - center

        z = self.om_pos[:, 2]
        n = np.add.reduceat(act.astype(np.float64), self.str_om_start)
        z_mean = np.add.reduceat(np.where(act, z, 0.0), self.str_om_start) / np.maximum(n, 1.0)
        z_lo = np.minimum.reduceat(np.where(act, z, np.inf), self.str_om_start)
        z_hi = np.maximum.reduceat(np.where(act, z, -np.inf), self.str_om_start)
        s = np.flatnonzero(n > 0)
        s = s[~np.isin(s, hit_str, assume_unique=True)]
        s = s[np.argsort(np.linalg.norm(self.str_xy[s] - center[:2], axis=1), kind='stable')]
        str_rows = np.zeros((len(s), len(FEATURE_NAMES)), dtype=np.float32)
        str_rows[:, 0:2] = self.str_xy[s]
        str_rows[:, 2] = z_mean[s]
        str_rows[:, 3:6] = str_rows[:, 0:3] - center
        str_rows[:, COLUMN['string_token']] = 1.0
        str_rows[:, COLUMN['n_om']] = n[s]
        str_rows[:, COLUMN['z_span']] = z_hi[s] - z_lo[s]
        return om_rows, str_rows


def event_tokens(tok, t, q, geometry, max_tokens=None, feature_set='essential',
                 silent_tokens=False, om_active=None):
    """One event's pulses -> (n_tokens, n_columns) tokens of `feature_set` and (3,) globals.

    tok indexes geometry PMTs. silent_tokens adds a token per active silent OM on a hit
    string and per active silent string (om_active: bool per geometry OM, None = all).
    Over max_tokens the brightest hits win, then the silent OMs, then the silent strings,
    the silent ones nearest the event's pulse-weighted centre first.
    """
    tok = np.asarray(tok, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    order = np.lexsort((t, tok))
    used, lab = np.unique(tok[order], return_inverse=True)
    feats, glob = pmt_features(lab.reshape(-1), t[order], q[order],
                               geometry.pmt_pos[used], geometry.pmt_dir[used])
    rows = np.zeros((len(feats), len(FEATURE_NAMES)), dtype=np.float32)
    rows[:, :feats.shape[1]] = feats
    rows[:, COLUMN['hit']] = 1.0
    if max_tokens is not None and len(rows) > max_tokens:
        rows = rows[np.argsort(-rows[:, COLUMN['q_tot']], kind='stable')[:max_tokens]]
    if silent_tokens:
        center = (feats[0, 0:3] - feats[0, 3:6]).astype(np.float64)
        rows = np.concatenate([rows, *geometry.silent_tokens(used, center, om_active)])
        if max_tokens is not None:
            rows = rows[:max_tokens]
    cols = [COLUMN[c] for c in FEATURE_SETS[feature_set]['columns']]
    return rows[:, cols], glob


def pad_events(events):
    """List of (tokens, glob) -> (B, N, C) tokens, (B, N) bool mask, (B, 3) globals."""
    n = max(len(f) for f, _ in events)
    feats = np.zeros((len(events), n, events[0][0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(events), n), dtype=bool)
    for i, (f, _) in enumerate(events):
        feats[i, :len(f)] = f
        mask[i, :len(f)] = True
    glob = np.stack([g for _, g in events]).astype(np.float32)
    return torch.from_numpy(feats), torch.from_numpy(mask), torch.from_numpy(glob)


# --------------------------------------------------------------------------- #
#  Sphere geometry                                                             #
# --------------------------------------------------------------------------- #

def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _project(d, v):
    return v - (v * d).sum(-1, keepdim=True) * d


def _exp(d, v):
    """Exponential map on S^2."""
    n = v.norm(dim=-1, keepdim=True)
    return _normalize(torch.cos(n) * d + torch.sinc(n / math.pi) * v)


def _transport(x, y, v):
    """Parallel transport of v from T_x S^2 to T_y S^2 along the geodesic."""
    return v - ((y * v).sum(-1, keepdim=True)
                / (1.0 + (x * y).sum(-1, keepdim=True))) * (x + y)


def _tangent_basis(d):
    a = torch.zeros_like(d)
    a[..., 2] = 1.0
    alt = torch.zeros_like(d)
    alt[..., 0] = 1.0
    a = torch.where(d[..., 2:3].abs() > 0.9, alt, a)
    e1 = _normalize(_project(d, a))
    return e1, torch.cross(d, e1, dim=-1)


def _geodesic(d0, d1, t):
    """Point, velocity and validity mask at time t on the great circle d0 -> d1."""
    cos = (d0 * d1).sum(-1, keepdim=True)
    sin = torch.linalg.norm(torch.cross(d0, d1, dim=-1), dim=-1, keepdim=True)
    th = torch.atan2(sin, cos)
    t = t.reshape(-1, 1)
    s = sin.clamp(min=1e-6)
    small = th < 1e-3
    x = torch.where(small, d0 + t * (d1 - d0),
                    (torch.sin((1 - t) * th) * d0 + torch.sin(t * th) * d1) / s)
    u = torch.where(small, d1 - d0,
                    th * (torch.cos(t * th) * d1 - torch.cos((1 - t) * th) * d0) / s)
    x = _normalize(x)
    # antipodal endpoints have no unique geodesic (probability ~1e-8 under a uniform base)
    ok = ((sin > 1e-4) | (cos > 0)).squeeze(-1)
    return x, torch.where(ok.unsqueeze(-1), _project(x, u), torch.zeros_like(u)), ok


def _uniform_sphere(n, device, dtype, generator=None):
    return _normalize(torch.randn(n, 3, device=device, dtype=dtype, generator=generator))


# --------------------------------------------------------------------------- #
#  Network                                                                     #
# --------------------------------------------------------------------------- #

def sinusoidal_position_encoding(pos, d_model, length_scale=500.0):
    """Thesis Eqs. 8.4-8.6 with d_model // 6 frequencies per axis, zero-padded to d_model."""
    nf = d_model // 6
    u = (pos / length_scale + 1.5) * (1e4 / 3.0)
    j = torch.arange(nf, device=pos.device, dtype=pos.dtype)
    ang = u.unsqueeze(-1) * torch.pow(10000.0, -j / nf)
    enc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-3)
    return F.pad(enc, (0, d_model - 6 * nf))


class EncoderBlock(torch.nn.Module):
    """Pre-LN transformer block with a key-padding mask."""

    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(ffn_dim, d_model))

    def forward(self, h, attn_mask):
        B, N, D = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(B, N, 3, self.n_heads, D // self.n_heads) \
            .permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        h = h + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return h + self.ffn(self.norm2(h))


class PMTEncoder(torch.nn.Module):
    """Tokens -> transformer -> pooling over hit and silent tokens -> condition vector."""

    def __init__(self, feature_set='essential', d_model=96, depth=20, n_heads=4, ffn_dim=512,
                 cond_dim=256, dropout=0.0, length_scale=500.0, time_scale=1e4,
                 time_transform='linear'):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.length_scale, self.time_scale = d_model, length_scale, time_scale
        cols = FEATURE_SETS[feature_set]['columns']
        self.linear_from = FEATURE_SETS[feature_set]['linear_from']
        self.hit_col = cols.index('hit')
        lin = cols[self.linear_from:]
        asinh = time_transform == 'asinh'
        self.register_buffer('scale', torch.tensor(
            [1.0 / length_scale if c in _LENGTH_COLS
             else 1.0 / time_scale if c in _TIME_COLS and not asinh else 1.0 for c in lin]))
        self.register_buffer('log_q', torch.tensor([c in _CHARGE_COLS for c in lin]))
        self.register_buffer('log_n', torch.tensor([c in _COUNT_COLS for c in lin]))
        self.register_buffer('asinh_t', torch.tensor([c in _TIME_COLS and asinh for c in lin]))
        self.inp = torch.nn.Linear(len(lin), d_model)
        self.blocks = torch.nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(d_model)
        n_pool = 3 * d_model + len(GLOBAL_NAMES)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(n_pool), torch.nn.Linear(n_pool, cond_dim), torch.nn.Tanh(),
            torch.nn.Linear(cond_dim, cond_dim))

    def transform(self, f):
        """Fixed feature-wise transforms of thesis Eq. 8.2 (times optionally asinh)."""
        f = f[..., self.linear_from:]
        x = f * self.scale
        x = torch.where(self.log_q, 0.2 * torch.log1p(f.clamp(min=0.0)), x)
        x = torch.where(self.log_n, torch.log1p(f.clamp(min=0.0)) / 5.0, x)
        return torch.where(self.asinh_t, torch.asinh(f / self.time_scale), x)

    def forward(self, feats, mask, glob):
        h = self.inp(self.transform(feats)) + sinusoidal_position_encoding(
            feats[..., 0:3], self.d_model, self.length_scale)
        attn_mask = mask[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = self.norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)
        # pool hits and silent tokens separately so the silent ones, often the majority,
        # do not wash out the hits
        hit = m * (feats[..., self.hit_col:self.hit_col + 1] > 0.5).to(h.dtype)
        sil = m - hit
        n_hit = hit.sum(1).clamp(min=1.0)
        mean = (h * hit).sum(1) / n_hit
        var = (((h - mean.unsqueeze(1)) ** 2) * hit).sum(1) / n_hit
        mean_sil = (h * sil).sum(1) / sil.sum(1).clamp(min=1.0)
        # pooling alone cannot see how many PMTs fired or the total charge, which the
        # energy needs
        g = torch.cat([torch.log1p(glob[:, 0:1]) / 5.0, 0.2 * torch.log1p(glob[:, 1:2]),
                       torch.log1p(glob[:, 2:3]) / 5.0], dim=-1).to(h.dtype)
        return self.head(torch.cat([mean, var, mean_sil, g], dim=-1))


class AdaLNBlock(torch.nn.Module):
    """Residual MLP block with adaLN-Zero conditioning."""

    def __init__(self, width, cond_dim, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, mlp_ratio * width), torch.nn.SiLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(mlp_ratio * width, width))
        self.mod = torch.nn.Linear(cond_dim, 3 * width)
        torch.nn.init.zeros_(self.mod.weight)
        torch.nn.init.zeros_(self.mod.bias)

    def forward(self, h, c):
        shift, scale, gate = self.mod(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1.0 + scale) + shift)


class SphereVectorField(torch.nn.Module):
    """v(z, d, t | c) on R x S^2; the S^2 part is projected onto T_d S^2."""

    def __init__(self, cond_dim, width=256, depth=8, time_dim=64, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.t_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_dim, cond_dim), torch.nn.SiLU(),
            torch.nn.Linear(cond_dim, cond_dim))
        self.inp = torch.nn.Linear(4, width)
        self.blocks = torch.nn.ModuleList(
            [AdaLNBlock(width, cond_dim, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_mod = torch.nn.Linear(cond_dim, 2 * width)
        self.out = torch.nn.Linear(width, 4)
        for lin in (self.final_mod, self.out):
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, z, d, t, cond):
        c = F.silu(cond + self.t_mlp(self.time_emb(t)))
        h = self.inp(torch.cat([z, d], dim=-1))
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        o = self.out(self.norm(h) * (1.0 + scale) + shift)
        return o[:, :1], _project(d, o[:, 1:])


class RecoNet(torch.nn.Module):
    def __init__(self, encoder, field):
        super().__init__()
        self.encoder, self.field = encoder, field


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class RecoFlowDataset(Dataset):
    """Parquet events -> padded PMT tokens and (log10 E, direction). One event = one sample."""

    def __init__(self, parquet_path, geometry_csv_path, num_events_per_epoch=None, seed=None,
                 energy_column='neutrino_energy', use_pulses=True, pulse_params=None,
                 require_trigger=False, trigger_window=10.0, trigger_min_pmts=3,
                 n_strings_range=(10, 150), string_spread_range=(40.0, 1000.0),
                 om_drop_prob=0.5, om_drop_range=(0.0, 0.5),
                 max_tokens=None, feature_set='essential', silent_tokens=False,
                 uniform_energy_zenith=False, n_energy_bins=20, n_coszen_bins=20,
                 event_filter=None, max_redraws=1000, verbose=True):
        import pandas as pd
        import pyarrow.compute as pc
        from nugget.utils.parquet_io import (
            LEPTON_ALIASES, _filtered_batches, _iter_batches, _schema)

        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.pulses = PulseSimulator(**(pulse_params or {})) if use_pulses else None
        self.require_trigger = bool(require_trigger)
        self.trigger_window, self.trigger_min_pmts = float(trigger_window), int(trigger_min_pmts)
        self.masking = dict(
            n_strings_range=None if n_strings_range is None else tuple(map(int, n_strings_range)),
            spread_range=tuple(map(float, string_spread_range)),
            om_drop_prob=float(om_drop_prob), om_drop_range=tuple(map(float, om_drop_range)))
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.feature_set, self.silent_tokens = feature_set, bool(silent_tokens)
        self.max_redraws = int(max_redraws)
        self.config = dict(
            energy_column=energy_column, use_pulses=bool(use_pulses),
            pulse_params=None if self.pulses is None else dict(self.pulses.params),
            require_trigger=self.require_trigger, trigger_window=self.trigger_window,
            trigger_min_pmts=self.trigger_min_pmts, **self.masking,
            max_tokens=self.max_tokens, feature_set=self.feature_set,
            silent_tokens=self.silent_tokens,
            uniform_energy_zenith=bool(uniform_energy_zenith))

        self.geometry = DetectorGeometry.from_csv(geometry_csv_path)

        names = _schema(parquet_path).names
        ecol = (energy_column if energy_column in names
                else LEPTON_ALIASES.get(energy_column, energy_column))
        if 'times' in names:
            tcol = 'times'
        elif 'time' in names:
            tcol = 'time'
        else:
            raise ValueError(f"{parquet_path} has neither a 'times' (list mode) nor a "
                             "'time' (photon mode) column; light-yield files carry no times")
        cols = ['run_id', 'event_id', 'string', 'om', 'pmt', tcol, ecol, 'zenith', 'azimuth']
        batches = (_iter_batches(parquet_path, cols) if event_filter is None
                   else _filtered_batches(parquet_path, cols, event_filter))

        int_cols = {'run': 'run_id', 'event': 'event_id', 'string': 'string', 'om': 'om',
                    'pmt': 'pmt'}
        flt_cols = {'E': ecol, 'zen': 'zenith', 'azi': 'azimuth'}
        parts = {k: [] for k in [*int_cols, *flt_cols, 'n']}
        flat = []
        for b in batches:
            tc = b.column(tcol)
            if tcol == 'times':
                parts['n'].append(pc.list_value_length(tc).fill_null(0).to_numpy()
                                  .astype(np.int64))
                flat.append(pc.list_flatten(tc).to_numpy().astype(np.float32))
            else:
                parts['n'].append(np.ones(len(tc), np.int64))
                flat.append(tc.to_numpy().astype(np.float32))
            for k, c in int_cols.items():
                parts[k].append(b.column(c).to_numpy().astype(np.int64))
            for k, c in flt_cols.items():
                parts[k].append(b.column(c).to_numpy().astype(np.float64))
        if not flat:
            raise ValueError(f"no rows read from {parquet_path}")
        r = {k: np.concatenate(v) for k, v in parts.items()}
        self._times = np.concatenate(flat)
        del flat, parts
        ph_start = np.concatenate([[0], np.cumsum(r['n'])[:-1]]).astype(np.int64)

        g = self.geometry.lookup(r['string'], r['om'], r['pmt'])
        ok = (g >= 0) & (r['n'] > 0) & (r['E'] > 0)
        if verbose and (g < 0).any():
            print(f"RecoFlowDataset: dropped {int((g < 0).sum()):,} row(s) whose "
                  f"(string, om, pmt) is not in the geometry CSV")
        if not ok.any():
            raise ValueError("No usable rows: parquet and geometry CSV do not overlap.")
        g, ph_start = g[ok], ph_start[ok]
        r = {k: v[ok] for k, v in r.items()}

        codes = pd.DataFrame({'r': r['run'], 'e': r['event']}) \
            .groupby(['r', 'e'], sort=False).ngroup().to_numpy()
        if np.any(np.diff(codes) < 0):
            order = np.argsort(codes, kind='stable')
            codes, g, ph_start = codes[order], g[order], ph_start[order]
            r = {k: v[order] for k, v in r.items()}
        bounds = np.flatnonzero(np.diff(codes)) + 1
        self._ev_start = np.concatenate([[0], bounds]).astype(np.int64)
        self._ev_end = np.append(bounds, len(codes)).astype(np.int64)
        self._row_geo = g.astype(np.int32)
        self._row_ph_start = ph_start
        self._row_ph_count = r['n']

        first = self._ev_start
        zen, azi = r['zen'][first], r['azi'][first]
        st = np.sin(zen)
        self._log10e = np.log10(r['E'][first]).astype(np.float32)
        self._dir = np.stack([st * np.cos(azi), st * np.sin(azi), np.cos(zen)],
                             axis=1).astype(np.float32)
        self._zenith = zen.astype(np.float32)
        self._azimuth = azi.astype(np.float32)
        self._n_events = len(first)
        self.num_samples_per_epoch = (int(num_events_per_epoch) if num_events_per_epoch
                                      else self._n_events)

        self.uniform_energy_zenith = bool(uniform_energy_zenith)
        self._n_bins = 0
        if self.uniform_energy_zenith:
            self._build_bins(int(n_energy_bins), int(n_coszen_bins))

        if verbose:
            rows_per_ev = self._ev_end - self._ev_start
            print(f"RecoFlowDataset: {self._n_events:,} events, {len(codes):,} hit-PMT rows, "
                  f"{len(self._times):,} photons, {self.num_samples_per_epoch:,} events/epoch")
            print(f"  hit PMTs/event: median {np.median(rows_per_ev):.0f}, "
                  f"p99 {np.percentile(rows_per_ev, 99):.0f}, max {rows_per_ev.max()}; "
                  f"log10 E in [{self._log10e.min():.2f}, {self._log10e.max():.2f}]")
            print(f"  pulses={'on' if self.pulses is not None else 'off (photons)'}, "
                  f"trigger={'on' if self.require_trigger else 'off'}, "
                  f"features={self.feature_set}, "
                  f"silent tokens={'on' if self.silent_tokens else 'off'}, "
                  f"max_tokens={self.max_tokens}")
            print(f"  geometry: {self.geometry.n_strings} strings; per event "
                  f"{self.masking['n_strings_range'] or 'all'} strings, spread "
                  f"{self.masking['spread_range']} m, OM dropout "
                  f"{self.masking['om_drop_range']} with prob {self.masking['om_drop_prob']}")

    def _build_bins(self, n_e, n_c):
        def edges(v, nb):
            lo, hi = float(v.min()), float(v.max())
            return np.linspace(lo, hi if hi > lo else lo + 1e-6, nb + 1)

        le, cz = self._log10e, np.cos(self._zenith)
        ei = np.clip(np.digitize(le, edges(le, n_e)) - 1, 0, n_e - 1)
        ci = np.clip(np.digitize(cz, edges(cz, n_c)) - 1, 0, n_c - 1)
        flat = ei * n_c + ci
        order = np.argsort(flat, kind='stable')
        bounds = np.flatnonzero(np.diff(flat[order])) + 1
        self._bin_flat = order.astype(np.int64)
        self._bin_starts = np.concatenate([[0], bounds]).astype(np.int64)
        self._bin_lens = np.diff(np.append(self._bin_starts, len(order))).astype(np.int64)
        self._n_bins = len(self._bin_starts)

    def _sample_events(self, n):
        if self.uniform_energy_zenith and self._n_bins > 0:
            b = self._rng.integers(0, self._n_bins, size=n)
            offs = np.minimum((self._rng.random(n) * self._bin_lens[b]).astype(np.int64),
                              self._bin_lens[b] - 1)
            return self._bin_flat[self._bin_starts[b] + offs]
        return self._rng.integers(0, self._n_events, size=n)

    def target_stats(self):
        """(mu, sigma) of log10 E over the events."""
        return float(self._log10e.mean()), float(self._log10e.std())

    def tokens(self, e, masked=True):
        """Tokens of event e in a freshly sampled sub-geometry (masked=False: the whole
        geometry), after pulses and trigger; None if no hit survives."""
        rng, geo = self._rng, self.geometry
        rows = np.arange(self._ev_start[e], self._ev_end[e])
        g = self._row_geo[rows].astype(np.int64)
        om_active = None
        if masked:
            # drawn independently of the event, so given the geometry the events keep their
            # true (triggered) distribution; only how often each geometry is seen changes
            om_active = geo.sample_active(rng, **self.masking)
            keep = om_active[geo.pmt_om[g]]
            rows, g = rows[keep], g[keep]
            if len(rows) == 0:
                return None
        cnt = self._row_ph_count[rows]
        t = self._times[_ragged_indices(self._row_ph_start[rows], cnt)].astype(np.float64)
        tok = np.repeat(g, cnt)
        if self.pulses is not None:
            tok, t, q = self.pulses(tok, t, rng)
            if len(t) == 0:
                return None
        else:
            q = np.ones_like(t)
        if self.require_trigger and not om_triggered(
                geo.pmt_om[tok], geo.pmt[tok], t, self.trigger_window, self.trigger_min_pmts):
            return None
        return event_tokens(tok, t, q, geo, self.max_tokens, self.feature_set,
                            self.silent_tokens, om_active)

    def __len__(self):
        return self.num_samples_per_epoch

    def get_batch(self, indices, masked=True):
        n = len(np.asarray(indices).reshape(-1))
        out, evs = [], []
        for e in self._sample_events(n):
            # an event that loses every pulse or fails the trigger in the masked geometry
            # would not have been recorded, so draw a fresh event in its place
            for _ in range(self.max_redraws):
                tok = self.tokens(e, masked=masked)
                if tok is not None:
                    break
                e = self._sample_events(1)[0]
            else:
                raise RuntimeError(f"no event survived masking/trigger in "
                                   f"{self.max_redraws} draws; use more strings or a "
                                   f"wider string_spread_range")
            out.append(tok)
            evs.append(e)
        feats, mask, glob = pad_events(out)
        evs = np.asarray(evs)
        return (feats, mask, glob, torch.from_numpy(self._log10e[evs]),
                torch.from_numpy(self._dir[evs]))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(len(self))[idx]
        if isinstance(idx, (list, np.ndarray)):
            return self.get_batch(idx)
        return self.get_batch([idx])


# --------------------------------------------------------------------------- #
#  Model                                                                       #
# --------------------------------------------------------------------------- #

class FlowMatchReco(Surrogate):
    """Riemannian flow matching posterior p(log10 E, direction | PMT pulses) on R x S^2.

    The direction is the unit vector of the parquet (zenith, azimuth), i.e. the arrival
    direction for these files.
    """

    _CONFIG_KEYS = ('feature_set', 'd_model', 'encoder_depth', 'n_heads', 'ffn_dim',
                    'cond_dim', 'encoder_dropout', 'width', 'depth', 'time_dim', 'mlp_ratio',
                    'dropout', 'length_scale', 'time_scale', 'time_transform', 'sigma_min',
                    'direction_loss_weight', 'n_flow_samples')

    def __init__(self, device=None, feature_set='essential', d_model=96, encoder_depth=20,
                 n_heads=4, ffn_dim=512, cond_dim=256, encoder_dropout=0.0, width=256,
                 depth=8, time_dim=64, mlp_ratio=2, dropout=0.0, length_scale=500.0,
                 time_scale=1e4, time_transform='linear', sigma_min=1e-4,
                 direction_loss_weight=1.0, n_flow_samples=4, learning_rate=2e-4,
                 weight_decay=1e-5, lr_schedule='warmup_cosine', warmup_steps=2000,
                 warmup_frac=0.1, min_lr=1e-5, amp=False, **kwargs):
        super().__init__(device=device, dim=3)
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        if time_transform not in ('linear', 'asinh'):
            raise ValueError("time_transform must be 'linear' (thesis) or 'asinh'")
        if lr_schedule not in ('warmup_cosine', 'onecycle', 'constant'):
            raise ValueError("lr_schedule must be 'warmup_cosine', 'onecycle' or 'constant'")
        self.feature_set = feature_set
        self.d_model, self.encoder_depth, self.n_heads = d_model, encoder_depth, n_heads
        self.ffn_dim, self.cond_dim, self.encoder_dropout = ffn_dim, cond_dim, encoder_dropout
        self.width, self.depth, self.time_dim = width, depth, time_dim
        self.mlp_ratio, self.dropout = mlp_ratio, dropout
        self.length_scale, self.time_scale = length_scale, time_scale
        self.time_transform = time_transform
        self.sigma_min = sigma_min
        self.direction_loss_weight = direction_loss_weight
        self.n_flow_samples = int(n_flow_samples)
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.lr_schedule, self.warmup_steps = lr_schedule, warmup_steps
        self.warmup_frac, self.min_lr = warmup_frac, min_lr
        self.amp = amp

        self.target_mu = None      # mean of log10 E
        self.target_sigma = None   # std  of log10 E
        self.data_config = None
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_losses, self.val_losses, self.metrics = [], [], []
        self.best_state_dict = None
        self.is_trained = False

    # ---------------- dtype / device plumbing ----------------

    @property
    def param_dtype(self):
        return torch.float32 if self.net is None else next(self.net.parameters()).dtype

    def _prep(self, x):
        return x.to(device=self.device, dtype=self.param_dtype)

    def _autocast(self):
        dev = torch.device(self.device).type
        return torch.autocast(dev, dtype=torch.bfloat16, enabled=bool(self.amp) and dev == 'cuda')

    def to_z(self, log10_energy):
        return (log10_energy - self.target_mu) / self.target_sigma

    def from_z(self, z):
        return z * self.target_sigma + self.target_mu

    # ---------------- network ----------------

    def build_network(self):
        encoder = PMTEncoder(self.feature_set, self.d_model, self.encoder_depth, self.n_heads,
                             self.ffn_dim, self.cond_dim, self.encoder_dropout,
                             self.length_scale, self.time_scale, self.time_transform)
        field = SphereVectorField(self.cond_dim, self.width, self.depth, self.time_dim,
                                  self.mlp_ratio, self.dropout)
        # explicit float32: nugget sets a float64 default dtype, far too slow for this
        self.net = RecoNet(encoder, field).to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        n_enc = sum(p.numel() for p in encoder.parameters())
        n_fld = sum(p.numel() for p in field.parameters())
        print(f"FlowMatchReco: {self.feature_set} features, encoder d_model={self.d_model} "
              f"depth={self.encoder_depth} ({n_enc:,} params), field width={self.width} "
              f"depth={self.depth} ({n_fld:,} params)")

    def _build_scheduler(self, total_steps):
        if self.lr_schedule == 'onecycle':
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.learning_rate, total_steps=total_steps,
                pct_start=self.warmup_frac)
        elif self.lr_schedule == 'warmup_cosine':
            warm, floor = max(int(self.warmup_steps), 1), self.min_lr / self.learning_rate

            def factor(step):
                if step < warm:
                    return (step + 1) / warm
                p = min((step - warm) / max(total_steps - warm, 1), 1.0)
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        else:
            self.lr_scheduler = None

    def _scheduler_step(self):
        s = self.lr_scheduler
        if s is None:
            return
        if self.lr_schedule == 'onecycle' and s.last_epoch + 1 >= s.total_steps:
            return
        s.step()

    def tokenize(self, events, geometry, om_active=None):
        """List of (tok, t, q) pulse arrays, tok indexing `geometry` PMTs -> padded
        (feats, mask, glob), tokenised as in training (features, silent tokens, max_tokens)."""
        cfg = self.data_config or {}
        return pad_events([
            event_tokens(tok, t, q, geometry, cfg.get('max_tokens'), self.feature_set,
                         cfg.get('silent_tokens', False), om_active)
            for tok, t, q in events])

    def encode(self, feats, mask, glob):
        """(B, N, C) tokens, (B, N) mask, (B, 3) globals -> (B, cond_dim) condition."""
        with self._autocast():
            c = self.net.encoder(self._prep(feats), mask.to(self.device), self._prep(glob))
        return c.to(self.param_dtype)

    # ---------------- flow matching loss ----------------

    def fm_loss(self, feats, mask, glob, log10_energy, directions, generator=None):
        """Riemannian CFM loss; returns (total, (energy part, direction part))."""
        z1 = self.to_z(self._prep(log10_energy).reshape(-1, 1))
        d1 = _normalize(self._prep(directions).reshape(-1, 3))
        with self._autocast():
            c = self.net.encoder(self._prep(feats), mask.to(self.device), self._prep(glob))
        k = self.n_flow_samples
        if k > 1:
            c, z1, d1 = (x.repeat_interleave(k, dim=0) for x in (c, z1, d1))
        n, dev, dt = z1.shape[0], z1.device, z1.dtype
        z0 = torch.randn(n, 1, device=dev, dtype=dt, generator=generator)
        d0 = _uniform_sphere(n, dev, dt, generator)
        t = torch.rand(n, device=dev, dtype=dt, generator=generator)

        zt = (1.0 - (1.0 - self.sigma_min) * t.unsqueeze(1)) * z0 + t.unsqueeze(1) * z1
        uz = z1 - (1.0 - self.sigma_min) * z0
        dtt, ud, ok = _geodesic(d0, d1, t)
        with self._autocast():
            vz, vd = self.net.field(zt, dtt, t, c)
        le = ((vz.float() - uz) ** 2).mean()
        ld = (((vd.float() - ud) ** 2).sum(-1) * ok).sum() / ok.sum().clamp(min=1)
        return le + self.direction_loss_weight * ld, (le.detach(), ld.detach())

    # ---------------- ODE integration ----------------

    def _v_and_div(self, z, d, t, c):
        """Velocity and its Riemannian divergence on R x S^2."""
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            d = d.detach().requires_grad_(True)
            vz, vd = self.net.field(z, d, t, c)
            div = torch.autograd.grad(vz.sum(), z, retain_graph=True)[0].squeeze(-1)
            e1, e2 = _tangent_basis(d.detach())
            # the tangential trace of the ambient Jacobian of the projected field is its
            # surface divergence; autograd includes the derivative of the projection
            for e, keep in ((e1, True), (e2, False)):
                g = torch.autograd.grad((vd * e).sum(), d, retain_graph=keep)[0]
                div = div + (g * e).sum(-1)
        return vz.detach(), vd.detach(), div.detach()

    @torch.no_grad()
    def sample(self, feats=None, mask=None, glob=None, n_samples=1, n_steps=32,
               generator=None, cond=None):
        """(B, S) log10 E and (B, S, 3) direction samples (midpoint, exp-map steps)."""
        self.net.eval()
        c = self.encode(feats, mask, glob) if cond is None else self._prep(cond)
        B = c.shape[0]
        c = c.repeat_interleave(n_samples, dim=0)
        n, dev, dt_ = c.shape[0], c.device, c.dtype
        z = torch.randn(n, 1, device=dev, dtype=dt_, generator=generator)
        d = _uniform_sphere(n, dev, dt_, generator)
        h = 1.0 / n_steps
        for i in range(n_steps):
            t0 = torch.full((n,), i * h, device=dev, dtype=dt_)
            vz, vd = self.net.field(z, d, t0, c)
            zm, dm = z + 0.5 * h * vz, _exp(d, 0.5 * h * vd)
            vz, vd = self.net.field(zm, dm, t0 + 0.5 * h, c)
            z, d = z + h * vz, _exp(d, h * _transport(dm, d, vd))
        return self.from_z(z).reshape(B, n_samples), d.reshape(B, n_samples, 3)

    def log_prob(self, feats=None, mask=None, glob=None, log10_energy=None, directions=None,
                 n_steps=32, cond=None):
        """log p(log10 E, direction | event) per unit log10 E per steradian. Not differentiable.

        Add log(sin(zenith)) for a density in (log10 E, zenith, azimuth).
        """
        self.net.eval()
        with torch.no_grad():
            c = self.encode(feats, mask, glob) if cond is None else self._prep(cond)
        z = self.to_z(self._prep(log10_energy).reshape(-1, 1))
        d = _normalize(self._prep(directions).reshape(-1, 3))
        n, dev, dt_ = z.shape[0], z.device, z.dtype
        if c.shape[0] != n:
            raise ValueError(f"{c.shape[0]} conditions for {n} (energy, direction) points")
        acc = torch.zeros(n, device=dev, dtype=dt_)
        h = 1.0 / n_steps
        for i in reversed(range(n_steps)):
            t1 = torch.full((n,), (i + 1) * h, device=dev, dtype=dt_)
            with torch.no_grad():
                vz, vd = self.net.field(z, d, t1, c)
            zm, dm = z - 0.5 * h * vz, _exp(d, -0.5 * h * vd)
            vz, vd, div = self._v_and_div(zm, dm, t1 - 0.5 * h, c)
            z, d = z - h * vz, _exp(d, -h * _transport(dm, d, vd))
            acc = acc + h * div
        log_p0 = -0.5 * z.squeeze(-1) ** 2 - 0.5 * LOG_2PI - LOG_4PI
        return log_p0 - acc - math.log(self.target_sigma)

    @torch.no_grad()
    def reconstruct(self, feats, mask, glob, n_samples=128, n_steps=32, generator=None):
        """Point estimates: normalised mean sampled direction (thesis Eq. 8.12) and the
        median sampled log10 E."""
        le, d = self.sample(feats, mask, glob, n_samples, n_steps, generator)
        direction = _normalize(d.mean(1))
        return {
            'log10_energy': le.median(dim=1).values,
            'direction': direction,
            'zenith': torch.acos(direction[:, 2].clamp(-1.0, 1.0)),
            'azimuth': torch.remainder(torch.atan2(direction[:, 1], direction[:, 0]),
                                       2.0 * math.pi),
            'log10_energy_samples': le,
            'direction_samples': d,
        }

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches=10, n_samples=64, n_steps=32):
        """Median opening angle (deg), its mean over half-decade energy bins, and log10 E error."""
        ang, dle, le_true = [], [], []
        for i, (feats, mask, glob, le, d) in enumerate(dataloader):
            if i >= max_batches:
                break
            rec = self.reconstruct(feats, mask, glob, n_samples, n_steps)
            cos = (rec['direction'] * self._prep(d)).sum(-1).clamp(-1.0, 1.0)
            ang.append(torch.rad2deg(torch.acos(cos)).cpu())
            dle.append((rec['log10_energy'] - self._prep(le)).cpu())
            le_true.append(le.reshape(-1).cpu())
        ang, dle, le_true = (torch.cat(x).double().numpy() for x in (ang, dle, le_true))
        edges = np.arange(np.floor(2 * le_true.min()) / 2, le_true.max() + 0.5, 0.5)
        b = np.digitize(le_true, edges)
        per_bin = [np.median(ang[b == k]) for k in np.unique(b) if (b == k).sum() >= 10]
        return {
            'median_angle_deg': float(np.median(ang)),
            'macro_median_angle_deg': float(np.mean(per_bin)) if per_bin else float('nan'),
            'median_abs_dlog10e': float(np.median(np.abs(dle))),
            'dlog10e_68': float(np.percentile(np.abs(dle), 68)),
            'n_events': int(len(ang)),
        }

    # ---------------- training ----------------

    def fit_normalisers(self, dataset):
        self.target_mu, self.target_sigma = dataset.target_stats()
        print(f"  normalisers: log10 E mu={self.target_mu:.4f} sigma={self.target_sigma:.4f}")

    def train_with_dataloader(self, train_dataloader, val_dataloader=None, epochs=100,
                              verbose=True, early_stopping_patience=20, grad_clip=3.0,
                              save_every_n_epochs=None, checkpoint_path=None,
                              metrics_every_n_epochs=None, metrics_batches=10):
        if self.net is None:
            self.build_network()
        if self.target_mu is None:
            self.fit_normalisers(train_dataloader.dataset)
        cfg = getattr(train_dataloader.dataset, 'config', None)
        if cfg is not None:
            self.data_config = dict(cfg)
        self._build_scheduler(len(train_dataloader) * epochs)
        hist_e, hist_d = [], []

        best_val, patience = float('inf'), 0
        for epoch in range(epochs):
            self.net.train()
            t0 = time.time()
            sums, nb = np.zeros(3), 0
            for batch in train_dataloader:
                self.optimizer.zero_grad(set_to_none=True)
                loss, (le, ld) = self.fm_loss(*batch)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                self.optimizer.step()
                self._scheduler_step()
                sums += (loss.item(), le.item(), ld.item())
                nb += 1
            tl, tle, tld = sums / max(nb, 1)
            self.train_losses.append(tl)
            hist_e.append(tle)
            hist_d.append(tld)

            vl = None
            if val_dataloader is not None:
                self.net.eval()
                vs, vn = 0.0, 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        vs += self.fm_loss(*batch)[0].item()
                        vn += 1
                vl = vs / max(vn, 1)
                self.val_losses.append(vl)
                if vl < best_val:
                    best_val, patience = vl, 0
                    self.best_state_dict = {k: v.detach().clone()
                                            for k, v in self.net.state_dict().items()}
                else:
                    patience += 1

            if verbose:
                msg = (f"Epoch {epoch+1}/{epochs}  train {tl:.5f} (E {tle:.4f}, dir {tld:.4f})")
                if vl is not None:
                    msg += f"  val {vl:.5f}"
                print(msg + f"  lr {self.optimizer.param_groups[0]['lr']:.2e}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

            if (metrics_every_n_epochs and val_dataloader is not None
                    and (epoch + 1) % metrics_every_n_epochs == 0):
                m = self.evaluate(val_dataloader, max_batches=metrics_batches)
                self.metrics.append({'epoch': epoch + 1, **m})
                if verbose:
                    print(f"  val metrics: median angle {m['median_angle_deg']:.2f} deg, "
                          f"macro {m['macro_median_angle_deg']:.2f} deg, "
                          f"|dlog10E| median {m['median_abs_dlog10e']:.3f} "
                          f"68% {m['dlog10e_68']:.3f}", flush=True)

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
                'train_loss_energy': hist_e, 'train_loss_direction': hist_d,
                'metrics': self.metrics}

    # ---------------- persistence ----------------

    def save_model(self, filepath, state=None):
        torch.save({
            'config': {k: getattr(self, k) for k in self._CONFIG_KEYS},
            'net_state_dict': state if state is not None else self.net.state_dict(),
            'target_mu': self.target_mu, 'target_sigma': self.target_sigma,
            'data_config': self.data_config,
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
            'metrics': self.metrics,
        }, filepath)

    def load_model(self, filepath):
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        for k, v in ck['config'].items():
            setattr(self, k, v)
        for k in ('target_mu', 'target_sigma', 'data_config', 'train_losses', 'val_losses',
                  'metrics'):
            setattr(self, k, ck.get(k, getattr(self, k)))
        self.build_network()
        self.net.load_state_dict(ck['net_state_dict'])
        self.is_trained = True
        return self

    # ---------------- dataloaders ----------------

    def create_reco_parquet_dataloader(self, parquet_path, geometry_csv_path,
                                       num_events_per_epoch=None, batch_size=128,
                                       shuffle=True, num_workers=0, seed=None,
                                       energy_column='neutrino_energy', use_pulses=True,
                                       pulse_params=None, require_trigger=False,
                                       trigger_window=10.0, trigger_min_pmts=3,
                                       n_strings_range=(10, 150),
                                       string_spread_range=(40.0, 1000.0),
                                       om_drop_prob=0.5, om_drop_range=(0.0, 0.5),
                                       max_tokens=None, silent_tokens=False,
                                       uniform_energy_zenith=False, n_energy_bins=20,
                                       n_coszen_bins=20, test_save_path=None, test_frac=0.1,
                                       split_seed=None, pin_memory=None, event_frac=None):
        """Event-level loader. Each event is seen in its own random sub-geometry (see
        DetectorGeometry.sample_active); n_strings_range=None with om_drop_prob=0 keeps the
        whole geometry CSV. silent_tokens adds the active silent OMs on hit strings and the
        silent strings. uniform_energy_zenith flattens the implied prior in (log10 E, cos zen)."""
        train_filter = _event_split(parquet_path, event_frac, test_save_path, test_frac,
                                    split_seed if split_seed is not None else seed)
        ds = RecoFlowDataset(
            parquet_path, geometry_csv_path, num_events_per_epoch=num_events_per_epoch,
            seed=seed, energy_column=energy_column, use_pulses=use_pulses,
            pulse_params=pulse_params, require_trigger=require_trigger,
            trigger_window=trigger_window, trigger_min_pmts=trigger_min_pmts,
            n_strings_range=n_strings_range, string_spread_range=string_spread_range,
            om_drop_prob=om_drop_prob, om_drop_range=om_drop_range, max_tokens=max_tokens,
            feature_set=self.feature_set, silent_tokens=silent_tokens,
            uniform_energy_zenith=uniform_energy_zenith, n_energy_bins=n_energy_bins,
            n_coszen_bins=n_coszen_bins, event_filter=train_filter)

        if pin_memory is None:
            pin_memory = torch.device(self.device).type == 'cuda'
        base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        kw = dict(dataset=ds, batch_size=None,
                  sampler=BatchSampler(base, batch_size=batch_size, drop_last=False),
                  num_workers=num_workers, pin_memory=pin_memory)
        if num_workers > 0:
            kw['worker_init_fn'] = _reseed_dataset_rng_in_worker
            kw['persistent_workers'] = True
        return DataLoader(**kw)

    def create_reco_parquet_val_dataloader(self, parquet_path, geometry_csv_path,
                                           num_events_per_epoch=None, batch_size=128,
                                           num_workers=0, seed=0, **kw):
        return self.create_reco_parquet_dataloader(
            parquet_path=parquet_path, geometry_csv_path=geometry_csv_path,
            num_events_per_epoch=num_events_per_epoch, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, seed=seed, test_save_path=None, **kw)
