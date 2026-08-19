from nugget.geometries.base_geometry import Geometry
import torch
import numpy as np


class NFoldString(Geometry):
    """N-fold radially symmetric string geometry.

    Only a single angular slice ("fold") of width ``2*pi / n_folds`` is
    parameterized, each string in the slice has parameters for ``slice_radius`` and ``slice_angle``. 
    That slice is then mirrored ``n_folds`` times by rigid rotation about the z-axis to build the full detector, 
    with optional slice string weights mode.


    Parameters
    ----------
    n_folds : int
        Number of radial repetitions (N). The slice spans ``2*pi / n_folds``.
    strings_per_fold : int
        Number of independent strings inside one slice. The full geometry has
        ``n_folds * strings_per_fold`` strings.
    points_per_string : int
        Number of (optical module) points along each string.
    use_weights : bool
        If False, string weights are disabled entirely and no weight keys
        (``slice_weights`` / ``string_weights`` / ...) appear in the returned
        geometry dict. Forces ``active_weights_mode`` off.
    fold_offset : float
        Rigid angular offset applied to the whole detector, in radians.
    radius_margin : float
        Fraction of ``half_domain`` kept free at the outer edge when placing
        the initial slice strings. Only affects initialization -- radius is
        unconstrained during optimization.
    init_radius : float or None
        Optional maximum radius used when placing the initial slice strings.
    slice_init : {'sunflower', 'hex', 'random'}
        How the initial strings are laid out inside the wedge. 'sunflower'
        (default) uses a golden-angle spiral; 'hex' takes the points of a
        hexagonal lattice that fall in the wedge, so the mirrored detector
        starts on a true hex packing; 'random' samples uniformly over the
        wedge area. Initialization only -- the strings move freely afterwards.
    add_center_string : bool
        If True, one extra string is fixed at ``(x, y) = (0, 0)``, on top of
        the ``n_folds * strings_per_fold`` mirrored strings. Its xy position
        is a constant, not a parameter -- the origin is invariant under the
        fold rotation, so it has no orbit of its own the way every other
        string does. It uses the same points_per_string / z spacing as every
        other string, and (if ``use_weights``) gets its own optimizable weight.
    center_string_weight : float or None
        Starting raw weight for the center string when ``use_weights`` is set.
        Defaults to ``starting_weight``.

    """

    def __init__(
        self,
        device=None,
        dim=3,
        domain_size=2,
        n_folds=6,
        strings_per_fold=5,
        points_per_string=5,
        custom_z_spacing=None,
        use_weights: bool = True,
        random_weights=False,
        starting_weight=1.0,
        active_weights_mode: bool = False,
        fold_offset=0.0,
        radius_margin=0.0,
        init_radius=None,
        slice_init='sunflower',
        random_slice_init=False,
        add_center_string=False,
        center_string_weight=None,
        seed=None,
    ):
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        self.n_folds = int(n_folds)
        self.strings_per_fold = int(strings_per_fold)
        self.points_per_string = int(points_per_string)
        self.custom_z_spacing = custom_z_spacing
        self.use_weights = use_weights
        self.random_weights = random_weights
        self.starting_weight = starting_weight
        # Thresholding is meaningless without weights in the first place.
        self.active_weights_mode = active_weights_mode and use_weights
        self.fold_offset = float(fold_offset)
        self.radius_margin = float(radius_margin)
        self.init_radius = init_radius
        self.random_slice_init = random_slice_init
        # random_slice_init is the older boolean form of the same choice; keep
        # honouring it so existing callers are unaffected.
        self.slice_init = 'random' if random_slice_init else slice_init
        valid_inits = ('sunflower', 'hex', 'hexagonal', 'random')
        if str(self.slice_init).lower() not in valid_inits:
            raise ValueError(
                f"slice_init must be one of {valid_inits}, got {self.slice_init!r}"
            )
        self.seed = seed
        self.add_center_string = bool(add_center_string)
        self.center_string_weight = center_string_weight

        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.strings_per_fold < 1:
            raise ValueError("strings_per_fold must be >= 1")

        # Angular width of one fold.
        self.fold_angle = 2.0 * np.pi / self.n_folds
        # Total number of strings in the mirrored (full) geometry.
        self.n_strings = self.n_folds * self.strings_per_fold

        # Rotation angles of each fold copy, (n_folds,). Constant, no grad.
        self.fold_rotations = (
            torch.arange(self.n_folds, device=self.device, dtype=torch.float64)
            * self.fold_angle
            + self.fold_offset
        )

        # Outer radius the initial strings are spread over. Placement only --
        # the radius is unconstrained once optimization starts.
        if self.init_radius is not None:
            self.max_radius = float(self.init_radius)
        else:
            self.max_radius = self.half_domain * (1.0 - self.radius_margin)

    # ------------------------------------------------------------------
    # slice helpers
    # ------------------------------------------------------------------

    def _wrap_angle(self, slice_angle):
        """Wrap angles into ``[0, fold_angle)`` differentiably.

    
        """
        return torch.remainder(slice_angle, self.fold_angle)

    def _slice_to_full_xy(self, slice_radius, slice_angle):
        """Mirror the slice ``n_folds`` times -> full-detector xy.

        Returns ``(n_strings, 2)`` ordered fold-major, i.e. all strings of fold
        0 first, then fold 1, ... so that fold ``k``'s strings occupy the block
        ``[k * strings_per_fold : (k + 1) * strings_per_fold]``.

        """

        r = slice_radius
        a = self._wrap_angle(slice_angle)

        # (n_folds, strings_per_fold): each fold is the slice angle + its offset
        angles = a.unsqueeze(0) + self.fold_rotations.unsqueeze(1)
        radii = r.unsqueeze(0).expand(self.n_folds, -1)

        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        return torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)

    def _tile_weights(self, slice_weights):
        """Repeat one weight per slice string across all folds, (n_strings,).

        """
        return slice_weights.repeat(self.n_folds)

    def _hexagonal_slice_polar(self):
        """Initial slice strings taken from a hexagonal lattice.

        A full hexagonal lattice is generated ring by ring (the same
        centre-outward construction the base class uses), then only the points
        falling inside the first wedge are kept, ordered by radius. 

        The lattice spacing is chosen so that at least ``strings_per_fold``
        points land in the wedge; if the wedge still cannot supply enough, the
        remainder is topped up from the sunflower placement so the requested
        string count is always honoured.
        """
        n = self.strings_per_fold
        two_pi = 2.0 * np.pi

        # Start from a spacing that would put ~n points in one wedge, then
        # shrink until the wedge actually contains enough lattice sites.
        # Area of the wedge is (fold_angle / 2) * max_radius^2; a hex lattice
        # with spacing s has one point per (sqrt(3)/2) * s^2 of area.
        wedge_area = 0.5 * self.fold_angle * self.max_radius ** 2
        spacing = float(np.sqrt(max(wedge_area, 1e-12) / (max(n, 1) * np.sqrt(3) / 2)))
        spacing = max(spacing, 1e-6)

        sqrt3_half = np.sqrt(3.0) / 2.0
        eps = 1e-9

        sel_r, sel_a = [], []
        for _ in range(60):  # shrink-and-retry; converges well before this
            pts = []
            # Enough rings to cover max_radius at this spacing.
            n_rings = int(np.ceil(self.max_radius / (spacing * sqrt3_half))) + 2
            # Build sites straight from integer axial coordinates (q, r) rather
            # than by stepping around each ring: accumulating float steps drifts
            # by ~1e-8 over the outer rings, which would show up as the lattice
            # sitting slightly off its ideal ring.
            for q in range(-n_rings, n_rings + 1):
                lo = max(-n_rings, -q - n_rings)
                hi = min(n_rings, -q + n_rings)
                for rr in range(lo, hi + 1):
                    pts.append((
                        spacing * (q + 0.5 * rr),
                        spacing * sqrt3_half * rr,
                    ))

            sel_r, sel_a = [], []
            for (px, py) in pts:
                r = float(np.hypot(px, py))
                if r > self.max_radius + eps:
                    continue
                if r <= eps:
                    # r = 0 is invariant under rotation, so a string there
                    # would be mirrored onto itself n_folds times.
                    continue
                a = float(np.arctan2(py, px)) % two_pi
                # Half-open wedge [0, fold_angle): the trailing edge belongs to
                # the next fold, so excluding it prevents mirrored duplicates.
                if a < self.fold_angle - eps:
                    sel_r.append(r)
                    sel_a.append(a)

            if len(sel_r) >= n:
                break
            spacing *= 0.85  # too few sites in the wedge -> denser lattice

        order = np.argsort(np.asarray(sel_r, dtype=float), kind='stable')[:n]
        slice_radius = torch.tensor(
            [sel_r[i] for i in order], device=self.device, dtype=torch.float64
        )
        slice_angle = torch.tensor(
            [sel_a[i] for i in order], device=self.device, dtype=torch.float64
        )

        # Top up from the sunflower placement if the wedge could not supply
        # enough lattice sites (very large n or a very narrow fold).
        missing = n - int(slice_radius.numel())
        if missing > 0:
            fb_r, fb_a = self._sunflower_slice_polar()
            slice_radius = torch.cat([slice_radius, fb_r[-missing:]])
            slice_angle = torch.cat([slice_angle, fb_a[-missing:]])

        return slice_radius, slice_angle

    def _random_slice_polar(self):
        """Initial slice strings drawn uniformly over the wedge's area."""
        n = self.strings_per_fold
        if self.seed is not None:
            gen = torch.Generator(device='cpu').manual_seed(int(self.seed))
            rand = torch.rand(2, n, generator=gen, dtype=torch.float64).to(self.device)
        else:
            rand = torch.rand(2, n, device=self.device, dtype=torch.float64)
        # sqrt for uniform area density in the wedge
        slice_radius = self.max_radius * torch.sqrt(rand[0])
        slice_angle = self.fold_angle * rand[1]
        return slice_radius, slice_angle

    def _sunflower_slice_polar(self):
        """Initial slice strings on a golden-angle spiral filling the wedge."""
        n = self.strings_per_fold
        idx = torch.arange(n, device=self.device, dtype=torch.float64)
        # sqrt-spaced radii (uniform areal density) paired with angles spread
        # across the wedge, giving a sunflower-like slice.
        slice_radius = self.max_radius * torch.sqrt((idx + 0.5) / n)
        golden = np.pi * (3.0 - np.sqrt(5.0))
        slice_angle = torch.remainder(idx * golden, self.fold_angle)
        if n == 1:
            slice_angle = torch.full_like(slice_angle, self.fold_angle / 2.0)
        return slice_radius, slice_angle

    def _default_slice_polar(self):
        """Place the initial slice strings per the chosen ``slice_init`` mode."""
        mode = (self.slice_init or 'sunflower').lower()
        if mode in ('hex', 'hexagonal'):
            return self._hexagonal_slice_polar()
        if mode == 'random':
            return self._random_slice_polar()
        return self._sunflower_slice_polar()

    def _slice_from_symmetric_xy(self, string_xy):
        """Split an already-symmetric ``string_xy`` into a single slice.

        The geometry is assumed to already have the N-fold symmetry, and
        ``fold_offset`` is taken as given (it is a constructor argument, not
        something inferred). Each string is assigned to the fold its angle
        falls in; the strings in the first fold become the slice.

        Returns ``(slice_radius, slice_angle, slice_source_indices)``.
        """
        xy = self._as_tensor(string_xy).reshape(-1, 2)
        xy_np = xy.detach().cpu().numpy()

        r_np = np.hypot(xy_np[:, 0], xy_np[:, 1])
        a_np = np.mod(np.arctan2(xy_np[:, 1], xy_np[:, 0]) - self.fold_offset, 2.0 * np.pi)

        # Strings in the first wedge [0, fold_angle), innermost first.
        in_slice = np.where(a_np < self.fold_angle - 1e-12)[0]
        in_slice = in_slice[np.lexsort((a_np[in_slice], r_np[in_slice]))]
        reps = [int(i) for i in in_slice]

        slice_radius = torch.tensor(r_np[reps], device=self.device, dtype=torch.float64)
        slice_angle = torch.tensor(a_np[reps], device=self.device, dtype=torch.float64)
        return slice_radius, slice_angle, reps

    def _default_z_values(self, n_strings):
        if self.custom_z_spacing is not None:
            z_line = self.custom_z_spacing * (
                torch.arange(self.points_per_string, device=self.device, dtype=torch.float64)
                - (self.points_per_string - 1) / 2.0
            )
        else:
            z_line = torch.linspace(
                -self.half_domain,
                self.half_domain,
                self.points_per_string,
                device=self.device,
                dtype=torch.float64,
            )
        return z_line.repeat(n_strings)

    def _default_raw_weights(self, n):
        if not self.random_weights:
            return torch.ones(n, device=self.device, dtype=torch.float64) * self.starting_weight
        return torch.rand(n, device=self.device, dtype=torch.float64) * 8 - 4

    def _as_tensor(self, value, dtype=torch.float64):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=dtype)
        return torch.tensor(value, device=self.device, dtype=dtype)

    def _build_points(self, string_xy, z_values):
        """(n_strings, 2) + (n_strings * ppr,) -> (n_points, 3), differentiable."""
        n_strings = string_xy.shape[0]
        xy = string_xy.repeat_interleave(self.points_per_string, dim=0)
        z = z_values.reshape(-1, 1)
        if z.shape[0] != n_strings * self.points_per_string:
            raise ValueError(
                f"z_values has {z.shape[0]} entries, expected "
                f"{n_strings * self.points_per_string}"
            )
        return torch.cat([xy, z], dim=1)

    # ------------------------------------------------------------------
    # geometry API
    # ------------------------------------------------------------------

    def initialize_points(self, initial_geometry=None, **kwargs):
        """Initialize an N-fold symmetric string configuration.

        Parameters
        ----------
        initial_geometry : dict or None
            Optional pre-trained geometry to resume from. Recognizes the slice
            keys written by this class; falls back to defaults per missing key.
        """
        use_weights = kwargs.get('use_weights', self.use_weights)
        active_weights_mode = kwargs.get('active_weights_mode', self.active_weights_mode) and use_weights
        threshold = kwargs.get('weight_threshold', 0.7)
        add_center_string = kwargs.get('add_center_string', self.add_center_string)

        slice_radius = None
        slice_angle = None
        slice_z_values = None
        raw_weights = None
        raw_center_weight = None
        # Set when a slice is recovered from a full symmetric geometry: maps
        # each slice string back to the original string it came from.
        self._slice_source_indices = None

        if initial_geometry is not None:
            print("Using pre-trained N-fold string geometry as starting point")

            # Allow the fold count / slice occupancy to be restored too.
            if initial_geometry.get('n_folds', None) is not None:
                self.n_folds = int(initial_geometry['n_folds'])
                self.fold_angle = 2.0 * np.pi / self.n_folds
                self.fold_rotations = (
                    torch.arange(self.n_folds, device=self.device, dtype=torch.float64)
                    * self.fold_angle
                    + self.fold_offset
                )
            if initial_geometry.get('strings_per_fold', None) is not None:
                self.strings_per_fold = int(initial_geometry['strings_per_fold'])
            self.n_strings = self.n_folds * self.strings_per_fold

            if initial_geometry.get('slice_radius', None) is not None:
                slice_radius = self._as_tensor(initial_geometry['slice_radius']).reshape(-1)
                self.strings_per_fold = int(slice_radius.shape[0])
                self.n_strings = self.n_folds * self.strings_per_fold
            if initial_geometry.get('slice_angle', None) is not None:
                slice_angle = self._as_tensor(initial_geometry['slice_angle']).reshape(-1)

            # A plain symmetric geometry (only string_xy, no slice params):
            # take the strings in the first fold as the slice and infer
            # strings_per_fold from them. fold_offset is used as supplied.
            if (slice_radius is None and slice_angle is None
                    and initial_geometry.get('string_xy', None) is not None
                    and self.n_folds > 1):
                if initial_geometry.get('fold_offset', None) is not None:
                    self.fold_offset = float(initial_geometry['fold_offset'])
                    self.fold_rotations = (
                        torch.arange(self.n_folds, device=self.device, dtype=torch.float64)
                        * self.fold_angle
                        + self.fold_offset
                    )
                slice_radius, slice_angle, reps = self._slice_from_symmetric_xy(
                    initial_geometry['string_xy']
                )
                self.strings_per_fold = int(slice_radius.numel())
                self.n_strings = self.n_folds * self.strings_per_fold
                n_given = int(self._as_tensor(initial_geometry['string_xy']).reshape(-1, 2).shape[0])
                print(
                    f"Using {self.strings_per_fold} strings/fold from a symmetric geometry "
                    f"of {n_given} strings (fold_offset = {self.fold_offset:.6f} rad)"
                )
                if self.strings_per_fold * self.n_folds != n_given:
                    print(
                        f"  warning: {n_given} strings is not {self.n_folds} x "
                        f"{self.strings_per_fold}; the geometry may not be "
                        f"{self.n_folds}-fold symmetric about the given fold_offset"
                    )
                # Per-string values loaded below are indexed by the *original*
                # string order, so remember which originals we kept.
                self._slice_source_indices = reps

            # z: accept a single string's profile, a whole slice, or the full
            # detector (in which case the first fold is taken as the slice).
            expected_slice = self.strings_per_fold * self.points_per_string
            if initial_geometry.get('slice_z_values', None) is not None:
                cand = self._as_tensor(initial_geometry['slice_z_values']).reshape(-1)
                if cand.numel() == expected_slice:
                    slice_z_values = cand
                elif cand.numel() == self.points_per_string:
                    slice_z_values = cand.repeat(self.strings_per_fold)
            elif initial_geometry.get('z_values', None) is not None:
                cand = self._as_tensor(initial_geometry['z_values']).reshape(-1)
                if cand.numel() == self.points_per_string:
                    slice_z_values = cand.repeat(self.strings_per_fold)
                elif cand.numel() == expected_slice:
                    slice_z_values = cand
                elif self._slice_source_indices is not None:
                    # Slice recovered from a full geometry: gather each
                    # representative string's own z profile.
                    ppr = self.points_per_string
                    if cand.numel() >= (max(self._slice_source_indices) + 1) * ppr:
                        gather = torch.cat([
                            cand[k * ppr:(k + 1) * ppr] for k in self._slice_source_indices
                        ])
                        slice_z_values = gather
                elif cand.numel() == self.n_strings * self.points_per_string:
                    slice_z_values = cand[:expected_slice]

            src = self._slice_source_indices
            for key in ('old_slice_weights', 'slice_weights', 'old_string_weights',
                        'string_weights') if use_weights else ():
                if initial_geometry.get(key, None) is not None:
                    cand = self._as_tensor(initial_geometry[key]).reshape(-1)
                    if cand.numel() == self.strings_per_fold:
                        raw_weights = cand
                        break
                    if src is not None and cand.numel() > max(src):
                        # Slice recovered from a full geometry: take the weight
                        # of each representative string.
                        raw_weights = cand[torch.tensor(src, device=cand.device)]
                        break
                    if cand.numel() == self.n_strings:
                        # Full-detector weights: collapse back onto the slice.
                        raw_weights = cand[:self.strings_per_fold]
                        break

            if add_center_string and use_weights:
                for key in ('old_center_weight', 'center_weight'):
                    if initial_geometry.get(key, None) is not None:
                        raw_center_weight = self._as_tensor(initial_geometry[key]).reshape(-1)[:1]
                        break

        if slice_radius is None or slice_angle is None:
            default_r, default_a = self._default_slice_polar()
            slice_radius = default_r if slice_radius is None else slice_radius
            slice_angle = default_a if slice_angle is None else slice_angle
        if slice_z_values is None:
            slice_z_values = self._default_z_values(self.strings_per_fold)
        if use_weights and (raw_weights is None or raw_weights.numel() != self.strings_per_fold):
            raw_weights = self._default_raw_weights(self.strings_per_fold)
        if add_center_string and use_weights and (raw_center_weight is None or raw_center_weight.numel() != 1):
            if self.center_string_weight is not None:
                raw_center_weight = torch.tensor(
                    [self.center_string_weight], device=self.device, dtype=torch.float64
                )
            else:
                raw_center_weight = self._default_raw_weights(1)

        return self.update_points(
            slice_radius=slice_radius,
            slice_angle=slice_angle,
            slice_z_values=slice_z_values,
            slice_weights=raw_weights,
            old_slice_weights=raw_weights,
            use_weights=use_weights,
            active_weights_mode=active_weights_mode,
            weight_threshold=threshold,
            add_center_string=add_center_string,
            center_weight=raw_center_weight,
            old_center_weight=raw_center_weight,
        )

    def update_points(
        self,
        slice_radius,
        slice_angle,
        slice_z_values,
        slice_weights=None,
        old_slice_weights=None,
        center_weight=None,
        old_center_weight=None,
        **kwargs
    ):
        """Rebuild the full N-fold geometry from the current slice parameters.

        Parameters
        ----------
        slice_radius : torch.Tensor
            Radius of each string in the slice, ``(strings_per_fold,)``.
        slice_angle : torch.Tensor
            Angle of each string within the wedge, ``(strings_per_fold,)``.
            Wrapped into ``[0, fold_angle)``; crossing an edge re-enters from
            the other side.
        slice_z_values : torch.Tensor
            Z values for the slice's points,
            ``(strings_per_fold * points_per_string,)``.
        slice_weights : torch.Tensor or None
            One raw weight per slice string, shared across all folds. Ignored
            when weights are disabled.
        old_slice_weights : torch.Tensor or None
            Raw (pre-threshold) weights carried through ``active_weights_mode``.
        center_weight, old_center_weight : torch.Tensor or None
            Raw weight for the fixed center string (see ``add_center_string``),
            mirroring ``slice_weights`` / ``old_slice_weights``. Ignored unless
            ``add_center_string`` is set.

        Returns
        -------
        dict
            Full-detector geometry (``points_3d``, ``string_xy``, ``z_values``,
            ...) alongside the slice parameters, so the optimizer keeps
            stepping the slice while losses see the whole mirrored detector.
            Weight keys (``string_weights``, ``slice_weights``, ...) are present
            only when ``use_weights`` is set. If ``add_center_string`` is set, an
            extra string fixed at the origin is appended after the mirrored
            strings (its ``fold_indices`` / ``slice_indices`` entry is ``-1``,
            since it belongs to no fold).
        """
        use_weights = kwargs.get('use_weights', self.use_weights)
        active_weights_mode = kwargs.get('active_weights_mode', self.active_weights_mode) and use_weights
        threshold = kwargs.get('weight_threshold', 0.7)
        add_center_string = kwargs.get('add_center_string', self.add_center_string)

        if use_weights:
            # Backwards compatibility with dicts saved before the split.
            if old_slice_weights is None:
                old_slice_weights = kwargs.get('old_weights', None)
            if old_slice_weights is None:
                old_slice_weights = slice_weights
            if slice_weights is None:
                slice_weights = old_slice_weights
            if slice_weights is None:
                slice_weights = self._default_raw_weights(self.strings_per_fold)
                old_slice_weights = slice_weights

            if add_center_string:
                if old_center_weight is None:
                    old_center_weight = center_weight
                if center_weight is None:
                    center_weight = old_center_weight
                if center_weight is None:
                    start = self.starting_weight if self.center_string_weight is None else self.center_string_weight
                    center_weight = torch.tensor([start], device=self.device, dtype=torch.float64)
                    old_center_weight = center_weight

        # Keep the derived counts in sync if the slice size changed.
        self.strings_per_fold = int(slice_radius.shape[0])
        self.n_strings = self.n_folds * self.strings_per_fold

        # Mirror the slice N times -- this is where the symmetry is imposed.
        string_xy = self._slice_to_full_xy(slice_radius, slice_angle)

        # z profile is per-string and identical across folds, tiled fold-major
        # to match string_xy's ordering.
        z_values = slice_z_values.reshape(1, -1).expand(self.n_folds, -1).reshape(-1)

        n_total_strings = self.n_strings
        if add_center_string:
            # Fixed at the origin -- not a parameter, so it carries no grad and
            # is built fresh every call rather than threaded through as state.
            # Its z profile uses the same spacing/points_per_string as every
            # other string, for the same reason.
            center_xy = torch.zeros(1, 2, device=self.device, dtype=torch.float64)
            string_xy = torch.cat([string_xy, center_xy], dim=0)
            z_values = torch.cat([z_values, self._default_z_values(1)])
            n_total_strings = self.n_strings + 1

        points_3d = self._build_points(string_xy, z_values)

        string_indices = torch.arange(n_total_strings, device=self.device, dtype=torch.long)
        # Which fold each full-detector string came from, and which slice
        # string it is a copy of -- handy for plotting / analysis. The center
        # string (if any) belongs to no fold, marked with -1.
        fold_indices = torch.arange(
            self.n_folds, device=self.device, dtype=torch.long
        ).repeat_interleave(self.strings_per_fold)
        slice_indices = torch.arange(
            self.strings_per_fold, device=self.device, dtype=torch.long
        ).repeat(self.n_folds)
        if add_center_string:
            minus_one = torch.tensor([-1], device=self.device, dtype=torch.long)
            fold_indices = torch.cat([fold_indices, minus_one])
            slice_indices = torch.cat([slice_indices, minus_one])

        geom = {
            'points_3d': points_3d,
            'active_points': points_3d,
            'string_xy': string_xy,
            'z_values': z_values,
            'string_indices': string_indices,
            'active_string_indices': string_indices,
            'points_per_string_list': [self.points_per_string] * n_total_strings,
            # slice-level (optimizable) parameters
            'slice_radius': slice_radius,
            'slice_angle': slice_angle,
            'slice_z_values': slice_z_values,
            # bookkeeping
            'fold_indices': fold_indices,
            'slice_indices': slice_indices,
            'n_folds': self.n_folds,
            'strings_per_fold': self.strings_per_fold,
            'fold_angle': self.fold_angle,
            'fold_offset': self.fold_offset,
            'use_weights': use_weights,
            'add_center_string': add_center_string,
        }

        if not use_weights:
            return geom

        if active_weights_mode:
            slice_weights_to_return = 200 * (
                torch.sigmoid(old_slice_weights) > threshold
            ).to(dtype=torch.float64) - 100
            old_slice_weights_to_return = old_slice_weights
        else:
            slice_weights_to_return = old_slice_weights
            old_slice_weights_to_return = slice_weights_to_return

        string_weights = self._tile_weights(slice_weights_to_return)
        old_string_weights = self._tile_weights(old_slice_weights_to_return)

        geom.update({
            'slice_weights': slice_weights_to_return,
            'old_slice_weights': old_slice_weights_to_return,
            'active_weights_mode': active_weights_mode,
            'weight_threshold': threshold,
        })

        if add_center_string:
            if active_weights_mode:
                center_weight_to_return = 200 * (
                    torch.sigmoid(old_center_weight) > threshold
                ).to(dtype=torch.float64) - 100
                old_center_weight_to_return = old_center_weight
            else:
                center_weight_to_return = old_center_weight
                old_center_weight_to_return = center_weight_to_return

            string_weights = torch.cat([string_weights, center_weight_to_return])
            old_string_weights = torch.cat([old_string_weights, old_center_weight_to_return])
            geom.update({
                'center_weight': center_weight_to_return,
                'old_center_weight': old_center_weight_to_return,
            })

        geom['string_weights'] = string_weights
        geom['old_string_weights'] = old_string_weights
        return geom
