from nugget.surrogates.base_surrogate import Surrogate
from nugget.surrogates.HitClassifier import HitClassifier
from nugget.surrogates.FlowMatchLY import FlowMatchLY
from nugget.surrogates.FlowMatchATime import FlowMatchATime
import torch
import numpy as np
from pathlib import Path


CLIGHT = 0.299792458   # m/ns, vacuum

# Geometry the three models were trained on. Used to source the default PMT
# orientation template when the caller supplies only positions.
DEFAULT_GEOMETRY = (Path(__file__).resolve().parents[1]
                    / 'other' / '800_40_40_geom.csv')


class NuSmoothie(Surrogate):
    """Three models in one: hit classifier, light-yield flow and arrival-time flow.

    Evaluates the hurdle factorisation

        p(q, t | c) = pi(c) . p_LY(q | c, q>=1) . prod_j p_T(t_j | c)

    with pi from HitClassifier, p_LY from FlowMatchLY and p_T from FlowMatchATime.
    """

    _CTX_FLAGS = ('rich_rel_pos_mode', 'include_vertex_position',
                  'add_vertex_distance', 'add_distance_from_beam',
                  'add_dist_long', 'track_dir_is_arrival',
                  'add_pmt_direction', 'add_pmt_cosangle')

    def __init__(self, device=None, dim=3, domain_size=None,
                 hit_model=None, ly_model=None, atime_model=None,
                 hit_checkpoint=None, ly_checkpoint=None, atime_checkpoint=None,
                 particle_mode='track', patd_mode=False,
                 pmt_directions=None, geometry_csv_path=None,
                 n_pmt_per_om=None, pmt_mode='sum',
                 n_steps=64, ly_n_samples=32, batch_size=65536,
                 verbose=True, **kwargs):
        """
        Parameters
        ----------
        hit_model, ly_model, atime_model : model instance or None
            Already-constructed surrogates. Alternatively give ``*_checkpoint``
            paths and they are instantiated and loaded here.
        domain_size : float | (w, h) | None
            Only used when constructing from checkpoints; ``load_model`` restores
            each model's own value afterwards, so a mismatch here is harmless.
        particle_mode : {'track'}
            'cascade' is accepted by the constructor but every geometry path
            raises until the cascade feature builder exists.
        patd_mode : bool
            Default for ``light_yield_surrogate``; overridable per call.
        pmt_directions : Tensor, shape (N, 3) | (K, 3) | None
            Per-point orientations, or a K-direction OM template applied to every
            point. None reads the template from ``geometry_csv_path``, so a bare
            list of OM positions works with no orientation information.
        geometry_csv_path : path or None
            Geometry to take the default OM template from, via the pmt_dir_* of
            one optical module. Defaults to the geometry the models were trained
            on (``nugget/other/800_40_40_geom.csv``). Falls back to a Fibonacci
            sphere only if the file cannot be read.
        n_pmt_per_om : int or None
            Truncate the geometry template to this many PMTs, or set the Fibonacci
            count if there is no geometry. None keeps every PMT the geometry
            defines (16 for the training geometry).
        pmt_mode : {'sum', 'split'}
            How a template expansion is reported. 'sum' pools each OM's PMTs into
            one value per opt_point (length n_pts). 'split' keeps them separate,
            returning length ``n_pts * K`` ordered point-major -- reshape to
            ``(n_pts, K)`` to recover the grouping. Irrelevant when orientations
            are given one per point. Overridable per call.
        n_steps : int
            ODE steps for every flow call.
        ly_n_samples : int
            Monte-Carlo samples for E[q | c, q>=1]. Cost is
            ``n_pmt x ly_n_samples x n_steps`` network evaluations.
        """
        if dim != 3:
            raise ValueError("NeuralSabre only supports 3D geometry")
        if pmt_mode not in ('sum', 'split'):
            raise ValueError("pmt_mode must be 'sum' or 'split'")

        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.kwargs = kwargs

        self.particle_mode = particle_mode
        self.patd_mode = bool(patd_mode)
        self.pmt_mode = pmt_mode
        self.n_steps = int(n_steps)
        self.ly_n_samples = int(ly_n_samples)
        self.batch_size = int(batch_size)
        self.n_pmt_per_om = None if n_pmt_per_om is None else int(n_pmt_per_om)
        self.geometry_csv_path = (DEFAULT_GEOMETRY if geometry_csv_path is None
                                  else geometry_csv_path)
        self.refractive_index = float(kwargs.get('n_refraction', 1.33))
        self.v_mu = float(kwargs.get('v_mu', CLIGHT))
        self.poisson_rate_cap = float(kwargs.get('poisson_rate_cap', 1e8))

        ds = domain_size if domain_size is not None else 10000
        self.hit_model = self._load(hit_model, hit_checkpoint, HitClassifier, ds)
        self.ly_model = self._load(ly_model, ly_checkpoint, FlowMatchLY, ds)
        self.atime_model = self._load(atime_model, atime_checkpoint,
                                      FlowMatchATime, ds)

        if self.hit_model is None or self.ly_model is None:
            raise ValueError("NeuralSabre needs at least a hit model and an LY model")
        if self.patd_mode and self.atime_model is None:
            raise ValueError("patd_mode=True requires an arrival-time model")

        # refractive index follows the arrival-time model, whose t_geom defines it
        if self.atime_model is not None:
            self.refractive_index = float(self.atime_model.refractive_index)

        self._pmt_template_src = 'explicit'
        if pmt_directions is not None:
            pd = torch.as_tensor(pmt_directions, dtype=torch.float64)
            self._pmt_template = self._unit(pd.reshape(-1, 3).to(self.device))
        else:
            self._pmt_template = self._geometry_pmt_template(verbose=verbose)

        self._ctx_groups = self._group_contexts()
        if verbose:
            self._report()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _load(self, model, checkpoint, cls, domain_size):
        if model is not None:
            return model
        if checkpoint is None:
            return None
        m = cls(device=self.device, dim=3, domain_size=domain_size)
        m.load_model(checkpoint)
        if getattr(m, 'net', None) is not None:
            m.net.eval()
        return m

    @staticmethod
    def _unit(v):
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _named_models(self):
        pairs = [('hit', self.hit_model), ('ly', self.ly_model),
                 ('atime', self.atime_model)]
        return [(n, m) for n, m in pairs if m is not None]

    def _context_signature(self, m):
        """Two models with equal signatures produce identical build_context output.

        build_context returns UNNORMALISED features (each model applies its own
        standardiser internally), so differing normalisers do not block sharing --
        only differing feature flags, position scale or width do.
        """
        ds = m.domain_size
        if isinstance(ds, torch.Tensor):
            ds = ds.tolist() if ds.dim() > 0 else ds.item()
        ds = tuple(float(x) for x in ds) if isinstance(ds, (list, tuple)) else float(ds)
        return (tuple(bool(getattr(m, f)) for f in self._CTX_FLAGS),
                float(getattr(m, 'ly_eps', 1e-6)), ds, int(m.context_dim))

    def _group_contexts(self):
        """[(signature, [model names]), ...] -- one build_context call per group."""
        groups = {}
        for name, m in self._named_models():
            groups.setdefault(self._context_signature(m), []).append(name)
        return list(groups.items())

    def _report(self):
        print(f"NeuralSabre: particle_mode={self.particle_mode}, "
              f"patd_mode={self.patd_mode}, pmt_mode={self.pmt_mode}, "
              f"n={self.refractive_index}")
        for name, m in self._named_models():
            print(f"  {name:6s} context_dim={m.context_dim:3d} "
                  f"device={m.device} dtype={m.param_dtype}")
        print(f"  OM template: {self._pmt_template.shape[0]} PMT directions "
              f"from {self._pmt_template_src}")
        if len(self._ctx_groups) == 1:
            print(f"  all {len(self._named_models())} models share one context "
                  f"({self._ctx_groups[0][0][3]} features) -- built once per call")
        else:
            print(f"  {len(self._ctx_groups)} distinct context layouts -> "
                  f"{len(self._ctx_groups)} build_context calls per evaluation:")
            for sig, names in self._ctx_groups:
                print(f"    {sig[3]:3d} features: {', '.join(names)}")

    # ------------------------------------------------------------------
    # event parameters and geometry
    # ------------------------------------------------------------------

    def _parse_event_params(self, event_params, gradient_mode=False):
        """-> (vertex (3,), travel_dir (3,), energy ()) as float64 device tensors.

        ``travel_dir`` is the direction the particle MOVES. event_params follows
        the LightSabre convention where 'direction'/'zenith'/'azimuth' is already
        the travel direction; the per-model ``track_dir_is_arrival`` flip happens
        inside each model's own build_context, not here.
        """
        if event_params is None:
            raise ValueError("event_params must be provided")

        def t(x, name):
            if x is None:
                raise ValueError(f"event_params must contain '{name}'")
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=torch.float64).squeeze()
            return torch.tensor(x, dtype=torch.float64, device=self.device).squeeze()

        vertex = t(event_params.get('position'), 'position').reshape(3)
        energy = t(event_params.get('energy'), 'energy').reshape(())
        if gradient_mode:
            vertex = vertex.requires_grad_(True)
            energy = energy.requires_grad_(True)

        d = event_params.get('direction')
        if d is not None:
            direction = t(d, 'direction').reshape(3)
            if gradient_mode:
                direction = direction.requires_grad_(True)
        else:
            zen, azi = event_params.get('zenith'), event_params.get('azimuth')
            if zen is None or azi is None:
                raise ValueError("event_params needs 'direction' or "
                                 "both 'zenith' and 'azimuth'")
            th, ph = t(zen, 'zenith'), t(azi, 'azimuth')
            if gradient_mode:
                th = th.requires_grad_(True)
                ph = ph.requires_grad_(True)
            direction = torch.stack([torch.sin(th) * torch.cos(ph),
                                     torch.sin(th) * torch.sin(ph),
                                     torch.cos(th)])
        return vertex, self._unit(direction), energy

    @staticmethod
    def _dir_for(model, travel_dir):
        """The vector to hand `model` so that it recovers ``travel_dir`` internally.

        build_context and geometric_time both do
        ``track_dir = -direction if track_dir_is_arrival else direction``,
        so a model trained on arrival directions needs -travel here. Getting this
        backwards silently mirrors the event about its vertex: every downstream PMT
        looks upstream and P(hit) collapses. The flag can differ between the three
        checkpoints, so it is resolved per model.
        """
        return -travel_dir if getattr(model, 'track_dir_is_arrival', False) else travel_dir

    @staticmethod
    def pmt_directions_from_geometry(geometry_csv_path, string=None, om=None):
        """The PMT orientations of one optical module, in ``pmt`` order.

        Reads ``pmt_dir_x/y/z`` for a single (string, om) -- the first one present
        unless told otherwise. In the training geometry all 56,498 OMs carry an
        identical 16-direction pattern, so one module reproduces it exactly.

        Returns
        -------
        np.ndarray, shape (K, 3)
        """
        import pandas as pd
        cols = ['string', 'om', 'pmt', 'pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']
        g = pd.read_csv(geometry_csv_path, usecols=cols)
        s = g.string.iloc[0] if string is None else string
        o = g.om.iloc[0] if om is None else om
        sub = g[(g.string == s) & (g.om == o)].sort_values('pmt')
        if len(sub) == 0:
            raise ValueError(f"no PMTs for string={s}, om={o} in {geometry_csv_path}")
        return sub[['pmt_dir_x', 'pmt_dir_y', 'pmt_dir_z']].to_numpy(np.float64)

    def _geometry_pmt_template(self, verbose=True):
        """Default OM template: the training geometry's own PMT orientations."""
        try:
            d = self.pmt_directions_from_geometry(self.geometry_csv_path)
            self._pmt_template_src = str(self.geometry_csv_path)
        except Exception as exc:
            k = self.n_pmt_per_om or 16
            if verbose:
                print(f"NeuralSabre: could not read PMT directions from "
                      f"{self.geometry_csv_path} ({type(exc).__name__}: {exc}); "
                      f"falling back to a {k}-point Fibonacci sphere. Pass "
                      f"geometry_csv_path or pmt_directions to avoid this.")
            self._pmt_template_src = f'fibonacci({k})'
            return self._fibonacci_directions(k)

        t = torch.as_tensor(d, dtype=torch.float64, device=self.device)
        if self.n_pmt_per_om is not None and self.n_pmt_per_om < t.shape[0]:
            t = t[:self.n_pmt_per_om]
            self._pmt_template_src += f' (first {self.n_pmt_per_om} PMTs)'
        return self._unit(t)

    def _fibonacci_directions(self, k):
        """k roughly uniform directions on the sphere -- last-resort OM stand-in."""
        i = torch.arange(k, dtype=torch.float64, device=self.device) + 0.5
        cz = 1.0 - 2.0 * i / k
        r = torch.sqrt((1.0 - cz ** 2).clamp_min(0.0))
        phi = i * float(np.pi * (3.0 - np.sqrt(5.0)))
        return torch.stack([r * torch.cos(phi), r * torch.sin(phi), cz], dim=1)

    def _resolve_points(self, opt_point, pmt_directions=None, pmt_mode=None):
        """-> (pts (M,3), dirs (M,3), group (M,), n_out).

        ``group[j]`` indexes the output row PMT j contributes to. With
        pmt_mode='sum' that is the opt_point row (n_out = n_pts); with 'split'
        every PMT is its own output row (n_out = M), ordered point-major.
        """
        if opt_point is None:
            raise ValueError("opt_point must be provided")
        mode = self.pmt_mode if pmt_mode is None else pmt_mode
        if mode not in ('sum', 'split'):
            raise ValueError("pmt_mode must be 'sum' or 'split'")

        if not isinstance(opt_point, torch.Tensor):
            opt_point = torch.tensor(opt_point, dtype=torch.float64,
                                     device=self.device)
        pts = opt_point.to(device=self.device, dtype=torch.float64).reshape(-1, 3)
        n = pts.shape[0]

        pd = pmt_directions if pmt_directions is not None else self._pmt_template
        if not isinstance(pd, torch.Tensor):
            pd = torch.tensor(pd, dtype=torch.float64, device=self.device)
        pd = self._unit(pd.to(device=self.device,
                              dtype=torch.float64).reshape(-1, 3))

        if pd.shape[0] == n and n != 1:
            # one orientation per point: each point already is a single PMT
            return pts, pd, torch.arange(n, device=self.device), n

        # OM template: every point carries all K orientations, point-major
        k = pd.shape[0]
        pts_e = pts.repeat_interleave(k, dim=0)
        dirs_e = pd.repeat(n, 1)
        if mode == 'split':
            m = n * k
            return pts_e, dirs_e, torch.arange(m, device=self.device), m
        group = torch.arange(n, device=self.device).repeat_interleave(k)
        return pts_e, dirs_e, group, n

    def _track_geometry(self, pts, vertex, travel_dir):
        """Direct-Cherenkov geometry, consistent with FlowMatchATime.geometric_time.

        t_geom = [d_along + d_perp (n - cos_c)/sin_c] / c
               = s_emit / c + d_geom . n / c
        so s_emit and d_geom below are exactly the emission point and photon path
        length the arrival-time model's t_geom implies.
        """
        rel = pts - vertex.reshape(1, 3)
        d_along = (rel * travel_dir.reshape(1, 3)).sum(dim=1)
        perp = rel - d_along.unsqueeze(1) * travel_dir.reshape(1, 3)
        d_perp = torch.sqrt((perp ** 2).sum(dim=1) + 1e-12)

        cos_c = 1.0 / self.refractive_index
        sin_c = float(np.sqrt(max(1.0 - cos_c ** 2, 1e-12)))
        return d_along, d_perp, d_along - d_perp * (cos_c / sin_c), d_perp / sin_c

    def _require_track(self):
        if self.particle_mode != 'track':
            raise NotImplementedError(
                f"particle_mode='{self.particle_mode}' is not implemented. "
                "Cascades need their own context features (vertex distance and "
                "opening angle rather than track d_perp/d_long); add a "
                "_cascade_geometry counterpart to _track_geometry and a cascade "
                "branch in _contexts, then retrain the three models on cascades.")

    # ------------------------------------------------------------------
    # contexts
    # ------------------------------------------------------------------

    def _contexts(self, pts, dirs, vertex, travel_dir, energy):
        """-> {model name: context tensor on that model's device/dtype}."""
        self._require_track()
        m = pts.shape[0]
        vert = vertex.reshape(1, 3).expand(m, 3)
        en = energy.reshape(1).expand(m)

        out = {}
        for _, names in self._ctx_groups:
            ref = getattr(self, f'{names[0]}_model')
            # track_dir_is_arrival is part of the context signature, so every
            # model in a group wants the same convention as `ref`
            drc = self._dir_for(ref, travel_dir).reshape(1, 3).expand(m, 3)
            c = ref.build_context(pts, vert, en, pmt_directions=dirs,
                                  directions=drc)
            for nm in names:
                mdl = getattr(self, f'{nm}_model')
                out[nm] = c.to(device=mdl.device, dtype=mdl.param_dtype)
        return out

    def _chunks(self, m):
        b = max(int(self.batch_size), 1)
        return [(s, min(s + b, m)) for s in range(0, m, b)]

    # ------------------------------------------------------------------
    # the three model calls
    # ------------------------------------------------------------------

    def hit_prob(self, ctx_hit):
        """pi(c), calibrated to true detector occupancy."""
        out = torch.empty(ctx_hit.shape[0], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for s, e in self._chunks(ctx_hit.shape[0]):
                p = self.hit_model.predict_hit_prob(ctx_hit[s:e], calibrated=True)
                out[s:e] = p.detach().to(device=self.device, dtype=torch.float64)
        return out

    def expected_ly_given_hit(self, ctx_ly, n_samples=None, generator=None):
        """E[q | c, q>=1] by Monte Carlo over the light-yield flow."""
        ns = int(self.ly_n_samples if n_samples is None else n_samples)
        out = torch.empty(ctx_ly.shape[0], dtype=torch.float64, device=self.device)
        b = max(self.batch_size // max(ns, 1), 1)   # bound the internal repeat
        with torch.no_grad():
            for s in range(0, ctx_ly.shape[0], b):
                e = min(s + b, ctx_ly.shape[0])
                q = self.ly_model.expected_light_yield(
                    ctx_ly[s:e], n_samples=ns, n_steps=self.n_steps,
                    generator=generator)
                out[s:e] = q.detach().to(device=self.device, dtype=torch.float64)
        return out

    def sample_ly(self, ctx_ly, generator=None):
        """One draw of q >= 1 per PMT, conditional on being hit."""
        out = torch.empty(ctx_ly.shape[0], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for s, e in self._chunks(ctx_ly.shape[0]):
                q = self.ly_model.sample_light_yield(
                    ctx_ly[s:e], n_steps=self.n_steps, discrete=True,
                    generator=generator)
                out[s:e] = q.detach().to(device=self.device, dtype=torch.float64)
        return out

    def sample_time_residuals(self, ctx_rep, generator=None):
        """One t_res per row of an already-repeated context."""
        out = torch.empty(ctx_rep.shape[0], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for s, e in self._chunks(ctx_rep.shape[0]):
                t = self.atime_model.sample_time_residual(
                    ctx_rep[s:e], n_steps=self.n_steps, generator=generator)
                out[s:e] = t.detach().to(device=self.device, dtype=torch.float64)
        return out

    def _sanitize_rate_for_poisson(self, rate):
        rate = torch.nan_to_num(rate, nan=0.0, posinf=self.poisson_rate_cap,
                                neginf=0.0)
        return torch.clamp(rate, min=0.0, max=self.poisson_rate_cap)

    # ------------------------------------------------------------------
    # light yield
    # ------------------------------------------------------------------

    def __call__(self, track_pos=None, track_dir=None, track_energy=None,
                 om_positions=None, test_points=None, pmt_directions=None,
                 pmt_mode=None, sample=False, generator=None, **kwargs):
        """Expected photons at each position: pi(c) . E[q | c, q>=1].

        Mirrors LightSabre.__call__. With ``sample=True`` returns one Bernoulli x
        flow draw per position instead of the mean. Shape is (n_pts,) under
        pmt_mode='sum' and (n_pts * K,) under 'split'.
        """
        self._require_track()
        if om_positions is None:
            om_positions = test_points
        if track_pos is None or track_dir is None or track_energy is None:
            raise ValueError("track_pos, track_dir and track_energy are required")

        as_t = lambda x: (x.to(device=self.device, dtype=torch.float64)
                          if isinstance(x, torch.Tensor)
                          else torch.tensor(x, dtype=torch.float64,
                                            device=self.device))
        vertex = as_t(track_pos).reshape(3)
        travel = self._unit(as_t(track_dir).reshape(3))
        energy = as_t(track_energy).reshape(())

        pts, dirs, group, n_out = self._resolve_points(om_positions,
                                                        pmt_directions, pmt_mode)
        ctxs = self._contexts(pts, dirs, vertex, travel, energy)

        p = self.hit_prob(ctxs['hit'])
        if sample:
            u = torch.rand(p.shape[0], device=self.device, dtype=torch.float64,
                           generator=generator)
            q = torch.where(u < p, self.sample_ly(ctxs['ly'], generator),
                            torch.zeros_like(p))
        else:
            q = p * self.expected_ly_given_hit(ctxs['ly'], generator=generator)

        q = torch.nan_to_num(q, nan=0.0, posinf=self.poisson_rate_cap,
                             neginf=0.0).clamp(min=0.0)
        if n_out != q.shape[0]:
            q = torch.zeros(n_out, dtype=q.dtype,
                            device=q.device).index_add_(0, group, q)
        return q

    def call_batched(self, track_pos, track_dir, track_energy, om_positions,
                     pmt_directions=None, **kwargs):
        """(n_events, n_out) expected photons. Loops over events.

        Unlike LightSabre there is no closed form to broadcast over, so this is a
        convenience wrapper: the per-event work is already batched internally.
        """
        track_pos = torch.as_tensor(track_pos).reshape(-1, 3)
        track_dir = torch.as_tensor(track_dir).reshape(-1, 3)
        track_energy = torch.as_tensor(track_energy).reshape(-1)
        return torch.stack([
            self.__call__(track_pos=track_pos[i], track_dir=track_dir[i],
                          track_energy=track_energy[i],
                          om_positions=om_positions,
                          pmt_directions=pmt_directions, **kwargs)
            for i in range(track_pos.shape[0])])

    def light_yield_surrogate_batched(self, om_positions, event_params_list,
                                      **kwargs):
        """(n_events, n_out) expected photons for a list of event dicts."""
        rows = []
        for ep in event_params_list:
            vertex, travel, energy = self._parse_event_params(ep)
            rows.append(self.__call__(track_pos=vertex, track_dir=travel,
                                      track_energy=energy,
                                      om_positions=om_positions, **kwargs))
        return torch.stack(rows)

    def light_yield_surrogate(self, **kwargs):
        """Light yield at ``opt_point``, or arrival-time dicts in PATD mode.

        Parameters
        ----------
        event_params : dict
            'position', 'energy', and either 'direction' or 'zenith'/'azimuth'.
        opt_point : Tensor, shape (3,) | (n_pts, 3)
        patd_mode : bool, optional
            Overrides the constructor default.
        pmt_directions : Tensor, optional
            Per-point orientations or an OM template; see the constructor.
        pmt_mode : {'sum', 'split'}, optional
            Overrides the constructor default.
        sample : bool
            Light-yield mode only: draw instead of taking the mean.

        Returns
        -------
        Tensor (n_out,) in light-yield mode.
        dict for a single output row, else list[dict], in PATD mode -- same keys
        as LightSabrePATD.
        """
        self._require_track()
        event_params = kwargs.get('event_params')
        opt_point = kwargs.get('opt_point')
        patd = bool(kwargs.get('patd_mode', self.patd_mode))
        gradient_mode = bool(kwargs.get('gradient_mode',
                                        self.kwargs.get('gradient_mode', False)))

        vertex, travel, energy = self._parse_event_params(event_params,
                                                          gradient_mode)
        if patd:
            if self.atime_model is None:
                raise ValueError("patd_mode requires an arrival-time model")
            rest = {k: v for k, v in kwargs.items()
                    if k not in ('event_params', 'opt_point', 'patd_mode',
                                 'gradient_mode')}
            out = self._patd(opt_point, vertex, travel, energy, **rest)
            return out[0] if len(out) == 1 else out

        return self.__call__(track_pos=vertex, track_dir=travel,
                             track_energy=energy, om_positions=opt_point,
                             pmt_directions=kwargs.get('pmt_directions'),
                             pmt_mode=kwargs.get('pmt_mode'),
                             sample=kwargs.get('sample', False),
                             generator=kwargs.get('generator'))

    # ------------------------------------------------------------------
    # PATD
    # ------------------------------------------------------------------

    def _empty_patd_dict(self, expected_N=0.0):
        z = torch.tensor([], dtype=torch.float64, device=self.device)
        return {
            'hit_times': z, 'num_photons': 0, 'expected_photons': expected_N,
            'residual_times': z, 'geometric_times': z, 'vertex_times': z,
            'emission_points': torch.empty((0, 3), dtype=torch.float64,
                                           device=self.device),
            't_geom_min': torch.tensor(1e-6, dtype=torch.float64,
                                       device=self.device),
            'd_geom': z, 'patd_probs': None,
        }

    def _patd(self, opt_point, vertex, travel, energy, max_photons=None,
              get_patd_probs=False, generator=None, pmt_directions=None,
              pmt_mode=None, **kwargs):
        """One dict per output row, keys matching LightSabrePATD.

        Under pmt_mode='sum' a point expanding into several PMTs yields one dict
        holding that OM's photons pooled across its PMTs; under 'split' each PMT
        gets its own dict.
        """
        pts, dirs, group, n_out = self._resolve_points(opt_point, pmt_directions,
                                                        pmt_mode)
        ctxs = self._contexts(pts, dirs, vertex, travel, energy)

        p = self.hit_prob(ctxs['hit'])
        expected = p * self.expected_ly_given_hit(ctxs['ly'], generator=generator)

        # which PMTs fire, and how many photons each
        u = torch.rand(p.shape[0], device=self.device, dtype=torch.float64,
                       generator=generator)
        fired = u < p
        q = torch.zeros_like(p)
        if fired.any():
            q[fired] = self.sample_ly(ctxs['ly'][fired], generator=generator)
        if max_photons is not None:
            q = q.clamp(max=float(max_photons))
        n_ph = q.to(torch.int64)

        # geometry, exactly consistent with the model's own t_geom
        _, _, s_emit, d_geom = self._track_geometry(pts, vertex, travel)
        t_geom = self.atime_model.geometric_time(
            pts, vertex.reshape(1, 3).expand(pts.shape[0], 3),
            directions=self._dir_for(self.atime_model, travel)
                           .reshape(1, 3).expand(pts.shape[0], 3),
        ).to(device=self.device, dtype=torch.float64)

        # one flow call covering every photon of every PMT
        total = int(n_ph.sum().item())
        probs_all = None
        if total > 0:
            rep = torch.repeat_interleave(
                torch.arange(pts.shape[0], device=self.device), n_ph)
            t_res_all = self.sample_time_residuals(ctxs['atime'][rep],
                                                   generator=generator)
            if get_patd_probs:
                probs_all = torch.empty_like(t_res_all)
                with torch.no_grad():
                    for s, e in self._chunks(total):
                        lp = self.atime_model.log_prob_time_residual(
                            t_res_all[s:e].to(self.atime_model.param_dtype),
                            ctxs['atime'][rep[s:e]], n_steps=self.n_steps)
                        probs_all[s:e] = lp.exp().detach().to(
                            device=self.device, dtype=torch.float64)
            bounds = torch.cat([torch.zeros(1, dtype=torch.int64,
                                            device=self.device),
                                n_ph.cumsum(0)])
        else:
            t_res_all = None
            bounds = torch.zeros(pts.shape[0] + 1, dtype=torch.int64,
                                 device=self.device)

        emission = vertex.reshape(1, 3) + s_emit.unsqueeze(1) * travel.reshape(1, 3)

        results = []
        for g in range(n_out):
            members = torch.nonzero(group == g, as_tuple=True)[0]
            exp_g = float(expected[members].sum().item())
            spans = [(int(bounds[i].item()), int(bounds[i + 1].item()), int(i))
                     for i in members.tolist() if n_ph[i] > 0]
            if not spans:
                results.append(self._empty_patd_dict(exp_g))
                continue

            t_res = torch.cat([t_res_all[a:b] for a, b, _ in spans])
            tg = torch.cat([t_geom[i].expand(b - a) for a, b, i in spans])
            dg = torch.cat([d_geom[i].expand(b - a) for a, b, i in spans])
            se = torch.cat([s_emit[i].expand(b - a) for a, b, i in spans])
            em = torch.cat([emission[i].unsqueeze(0).expand(b - a, 3)
                            for a, b, i in spans])
            pr = (torch.cat([probs_all[a:b] for a, b, _ in spans])
                  if probs_all is not None else None)

            results.append({
                'hit_times': tg + t_res,
                'num_photons': int(sum(b - a for a, b, _ in spans)),
                'expected_photons': exp_g,
                'residual_times': t_res,
                'geometric_times': tg,
                'vertex_times': se / self.v_mu,
                'emission_points': em,
                't_geom_min': t_geom[members].min(),
                'd_geom': dg,
                'patd_probs': pr,
            })
        return results

    # ------------------------------------------------------------------
    # densities, for likelihood and Fisher work
    # ------------------------------------------------------------------

    def log_prob_event(self, opt_point, event_params, counts, times=None,
                       pmt_directions=None):
        """log p(q, t | c) per PMT under the hurdle factorisation.

        Needs one orientation per point (no OM pooling): a summed OM count is not
        what any of the three models was trained on.

        Parameters
        ----------
        counts : Tensor, shape (n_pts,)
            Observed photons per PMT. 0 contributes log(1 - pi).
        times : list of Tensor or None
            Per-PMT arrival times ``t_hit``; required for the timing term.

        Returns
        -------
        Tensor, shape (n_pts,)
        """
        self._require_track()
        vertex, travel, energy = self._parse_event_params(event_params)
        pts, dirs, group, n_out = self._resolve_points(opt_point, pmt_directions,
                                                        pmt_mode='split')
        ctxs = self._contexts(pts, dirs, vertex, travel, energy)

        counts = torch.as_tensor(counts, dtype=torch.float64,
                                 device=self.device).reshape(-1)
        if counts.shape[0] != pts.shape[0]:
            raise ValueError(f"counts has {counts.shape[0]} entries but there are "
                             f"{pts.shape[0]} PMTs; pass pmt_directions of shape "
                             f"(n_pts, 3) so points and PMTs correspond 1:1")

        p = self.hit_prob(ctxs['hit']).clamp(1e-30, 1 - 1e-15)
        out = torch.where(counts >= 1, torch.log(p), torch.log1p(-p))

        hit = counts >= 1
        if hit.any():
            with torch.no_grad():
                lp = self.ly_model.log_prob_light_yield(
                    counts[hit].to(self.ly_model.param_dtype),
                    ctxs['ly'][hit], n_steps=self.n_steps)
            out[hit] = out[hit] + lp.to(device=self.device, dtype=torch.float64)

        if times is not None:
            if self.atime_model is None:
                raise ValueError("times given but no arrival-time model")
            for i in range(pts.shape[0]):
                t_i = times[i]
                if t_i is None or len(t_i) == 0:
                    continue
                t_i = torch.as_tensor(t_i, dtype=torch.float64,
                                      device=self.device).reshape(-1)
                n_i = t_i.shape[0]
                with torch.no_grad():
                    lp = self.atime_model.log_prob_arrival_time(
                        t_i, ctxs['atime'][i].unsqueeze(0).expand(n_i, -1),
                        pts[i].unsqueeze(0).expand(n_i, 3),
                        vertex.unsqueeze(0).expand(n_i, 3),
                        directions=self._dir_for(self.atime_model, travel)
                                       .unsqueeze(0).expand(n_i, 3),
                        n_steps=self.n_steps)
                out[i] = out[i] + lp.sum().to(device=self.device,
                                              dtype=torch.float64)
        return out
