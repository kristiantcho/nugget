# NUGGET Codebase Reference

**NUGGET** = NeUtrino experiment Geometry optimization and General Evaluation Tool.
A PyTorch framework for optimizing neutrino detector geometries by combining differentiable physics surrogates with gradient-based geometry optimization.

---

## Table of Contents
1. [High-Level Concept](#1-high-level-concept)
2. [Directory Structure](#2-directory-structure)
3. [Geometry Optimization Pipeline](#3-geometry-optimization-pipeline)
4. [Geometries Module](#4-geometries-module)
5. [Surrogates Module](#5-surrogates-module)
6. [Losses Module](#6-losses-module)
7. [Samplers Module](#7-samplers-module)
8. [Utils Module](#8-utils-module)
9. [Key Data Structures](#9-key-data-structures)
10. [Typical Workflow (from notebook)](#10-typical-workflow-from-notebook)

---

## 1. High-Level Concept

The goal is to find the optimal placement of detector strings (cables with optical sensors) in a large water/ice volume to maximize detection performance for neutrino events. The pipeline is:

1. **Surrogates** approximate physics (light yield, LLR, timing) differentiably.
2. **Samplers** generate synthetic neutrino/muon events with physics parameters.
3. **Losses** score a given geometry using the surrogates + sampled events.
4. **Geometry** classes hold differentiable parameters (string XY positions, Z-spacings, weights).
5. **Optimizer** backpropagates through losses → geometry parameters, updating the detector layout.

---

## 2. Directory Structure

```
nugget/
├── setup.py
├── requirements.txt
├── nugget/
│   ├── __init__.py
│   ├── geometries/
│   │   ├── base_geometry.py        # Abstract base + hex/sunflower grid utilities
│   │   ├── FreePoints.py           # Unconstrained point cloud
│   │   ├── ContinuousString.py     # 1D path → 3D string positions
│   │   ├── DynamicString.py        # Variable points-per-string, dynamic spacing
│   │   ├── EvanescentString.py     # Learnable per-string weights (on/off)
│   │   └── SpaceString.py          # Fixed hex grid with learnable spacing
│   ├── surrogates/
│   │   ├── base_surrogate.py
│   │   ├── LightSabre.py           # Photon arrival time / light yield (primary)
│   │   ├── ChargeNet.py            # Neural net for charge/light yield prediction
│   │   ├── LLRnet.py               # Neural net classifier for LLR
│   │   ├── HitFlowNet.py           # Normalizing flow for hit time distributions
│   │   ├── HitFlow.py              # Wrapper + training for normalizing flows
│   │   ├── SkewedGaussian.py       # Analytical anisotropic Gaussian surrogate
│   │   ├── Uniform.py              # Constant baseline surrogate
│   │   ├── SymbolicReg.py          # Symbolic regression surrogate
│   │   ├── pandel.py / cpandel.py  # Pandel parametric photon timing model
│   ├── losses/
│   │   ├── base_loss.py
│   │   ├── LLR.py                  # Log-Likelihood Ratio loss
│   │   ├── SNR.py                  # Signal-to-Noise Ratio loss
│   │   ├── trigger.py              # Trigger efficiency loss
│   │   ├── effective_area.py       # Effective area loss
│   │   ├── fisher_info.py          # Fisher info + angular/energy resolution losses
│   │   ├── light_yield.py          # Direct light yield loss
│   │   ├── geometry_penalties.py   # Boundary, repulsion, ROV, sparsity penalties
│   │   ├── RBF.py
│   │   └── pointsource_fom.py      # Point source figure-of-merit
│   ├── samplers/
│   │   ├── base_sampler.py
│   │   ├── cyl_sampler.py          # Cylindrical detector sampler (primary)
│   │   └── toy_sampler.py          # Simple test sampler
│   ├── utils/
│   │   ├── basic_optimizer.py      # Core optimization loop with ALM support
│   │   ├── basic_evaluator.py      # Evaluation without gradients
│   │   ├── vis_tools.py            # Visualization + GIF generation
│   │   ├── data_tools.py           # Parquet I/O for event data
│   │   └── schedulers.py           # LR scheduler helpers
│   ├── other/
│   └── examples/
│       ├── modular_surrogate_notebook.ipynb   # Primary end-to-end demo
│       ├── train_hitflow.py
│       ├── train_chargenet.py
│       └── train_signal_only_llr.py
```

---

## 3. Geometry Optimization Pipeline

The complete pipeline from the reference notebook (`modular_surrogate_notebook.ipynb`):

### Step 1 — Instantiate surrogates and sampler
```python
# LightSabre: differentiable photon/light yield model
lightsabre = nugget.surrogates.LightSabre.LightSabre(
    use_poisson=True, num_track_points=2000, domain_size=2500, particle_mode='track')
light_yield_surrogate = lightsabre.light_yield_surrogate

# (Optional) PATD variant for photon arrival time distributions
light_yield_patd_surrogate = nugget.surrogates.LightSabre.LightSabrePATD(...).light_yield_surrogate

# CylinderSampler: generates muon/neutrino events intersecting a cylinder
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
    event_type='signal', domain_size=2000,
    E_min=1e2, E_max=1e8, energy_dist='log_uniform',
    find_exact_intersection=False, random_position_along_ray=True)
```

### Step 2 — (Optional) Load / train LLRnet
```python
llr_net = nugget.surrogates.LLRnet.LLRnet(
    domain_size=2500, dim=3, hidden_dims=[64,64,64,64],
    use_residual_connections=True, log_scale_ly=True, ...)
llr_net.load_model('best_charge_llr_model_v2')
```

### Step 3 — Instantiate geometry
```python
# EvanescentString is the standard geometry for string-selection optimization
geometry = nugget.geometries.EvanescentString.EvanescentString(
    hex_type='hexagonal', domain_size=1600, dim=3,
    n_strings=1027, points_per_string=20,
    custom_z_spacing=50, random_weights=True)
```

### Step 4 — Instantiate optimizer and visualizer
```python
visualizer = nugget.utils.vis_tools.Visualizer(
    dim=3, domain_size=1600, gif_temp_dir='./gif_temp')

optimizer = nugget.utils.basic_optimizer.Optimizer(
    device=geometry.device, geometry=geometry, visualizer=visualizer,
    conflict_free=False,
    use_alm=True,          # Augmented Lagrangian for constraint handling
    sigmoid_losses=True,   # Wrap losses in sigmoid for stable gradients
    sigmoid_softness=1,
    alm_params={'gamma': 1e-2, 'alpha': 0.95, 'epsilon': 1e-8})

optimizer.init_geometry(
    opt_list=[('string_weights', 0.5)])   # (parameter_name, learning_rate)
```

### Step 5 — Define losses, weights, and params
```python
# Physics losses
signal_yield_loss_func = nugget.losses.light_yield.WeightedLightYieldLoss(device=...)
weighted_energy_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
    device=..., resolution_type='energy',
    fisher_info_params=['position','energy','direction'])

# Geometry constraints / penalties
local_string_repulsion_penalty = nugget.losses.geometry_penalties.LocalStringRepulsionPenalty(...)
string_boundary_penalty = nugget.losses.geometry_penalties.StringBoundaryPenaltyCircle(...)
string_number_penalty = nugget.losses.geometry_penalties.StringNumberPenalty(...)
weighted_binarization_penalty = nugget.losses.geometry_penalties.WeightBinarizationPenalty(...)
rov_penalty = nugget.losses.geometry_penalties.ROVPenalty(
    rov_rec_width=230, rov_height=159.9, rov_tri_length=159.9)

loss_func_dict = {
    'energy_resolution_loss': weighted_energy_resolution_loss,
    'string_number_penalty': string_number_penalty,
    'rov_penalty': rov_penalty,
}
loss_weights_dict = {
    'energy_resolution_loss': 1e8,
    'string_number_penalty': 1,
    'rov_penalty': 1,
}
# Losses that get sigmoid applied (smooths gradients near 0)
loss_sigmoid_list = ['energy_resolution_loss', 'string_number_penalty', 'rov_penalty', ...]

loss_params = {
    'signal_event_params': pickle.load(...),         # Pre-sampled events (or sample live)
    'signal_surrogate_func': light_yield_surrogate,
    'signal_sampler': signal_sampler,
    'num_events': 100,
    'boundary_range': 1200,
    'precomputed_signal_yield_per_string': torch.load(...),
    'precomputed_fisher_info_per_string_per_event': torch.load(...),
    'constraints_list': ['rov_penalty', 'string_boundary_penalty', ...],  # ALM constraints
    'eva_min_num_strings': 70,
    'local_sharpness': 5,
    'boundary_sharpness': 10,
    ...
}
```

### Step 6 — Run optimization
```python
geom_dict = optimizer.optimize(
    loss_func_dict=loss_func_dict,
    loss_weights_dict=loss_weights_dict,
    loss_params_dict=loss_params,
    n_iter=1000,
    vis_kwargs=vis_kwargs,
    print_freq=5, vis_freq=10, gif_freq=5,
    sigmoid_loss_list=loss_sigmoid_list,
    save_geom_folder='./output_folder',
    save_geom_freq=100)
```

### Step 7 — Finalize and evaluate
```python
# Save animation
optimizer.visualizer.finalize_gif(gif_filename='output.gif', gif_fps=10)

# Evaluate without gradients
evaluator = nugget.utils.basic_evaluator.Evaluator(geometry=geometry, visualizer=visualizer)
result = evaluator.evaluate(
    geom_dict=optimizer.geom_dict,
    loss_func_dict=loss_func_dict,
    loss_params_dict=loss_params,
    vis_kwargs=vis_kwargs, visualize=True)
```

---

## 4. Geometries Module

All geometry classes inherit from `base_geometry.BaseGeometry` and return a **geom_dict** with consistent keys.

### BaseGeometry (`base_geometry.py`)
- `create_uniform_hexagonal_grid(n_points, optimal_spacing)` — centered hex lattice
- `create_circular_hexagonal_grid(n_points, optimal_spacing)` — hex packing in circle
- `create_sunflower_grid(n_points, optimal_spacing)` — golden-angle spiral
- `create_hybrid_hex_sunflower_grid(n_points, **kwargs)` — blend hex + sunflower via Hungarian matching
- `compare_geometries(geom1, geom2, ...)` — Hungarian-matched distance between two layouts

### EvanescentString (`EvanescentString.py`) — *primary geometry*
Strings with learnable sigmoid-gated weights. The key idea: each string has a scalar weight `w`; `sigmoid(w)` controls how much it contributes to losses, allowing the optimizer to "turn off" strings.

**Constructor params:**
- `n_strings` — initial number of strings (typically large, e.g. 1027)
- `points_per_string` — fixed PMTs per string (e.g. 20)
- `hex_type` — `'hexagonal'`, `'circular'`, `'sunflower'`
- `custom_z_spacing` — z-spacing between PMTs on a string
- `random_weights` — initialize weights randomly (promotes diversity)
- `starting_weight` — initial raw weight value

**Learned parameters (passed to optimizer):**
- `string_weights` — raw logit weights, `sigmoid(string_weights)` → effective contribution

**geom_dict keys:** `points_3d`, `string_xy`, `z_values`, `string_weights`, `string_indices`, `points_per_string_list`, `active_points`, `active_string_indices`

### SpaceString (`SpaceString.py`)
Hexagonal layout with learnable inter-string spacing. Use when you want to optimize spacing rather than string selection.

**Learned parameters:** `string_xy`, `string_spacing`, optionally `z_spacing` (if `optimize_z=True`)

### DynamicString (`DynamicString.py`)
Variable points-per-string with dynamic z-distribution. Supports `points_per_string` as a list or per-string array.

### ContinuousString (`ContinuousString.py`)
Parameterizes detector as a 1D path in 3D space — strings are located at evenly-spaced positions along the path.

### FreePoints (`FreePoints.py`)
Unconstrained point cloud. Simplest geometry; no string structure.

---

## 5. Surrogates Module

Surrogates are differentiable approximations of detector physics. All inherit from `base_surrogate.Surrogate`.

**Signature:** `surrogate(opt_point: Tensor, event_params: dict) -> dict`

### LightSabre (`LightSabre.py`) — *primary surrogate*
Analytical photon arrival time / light yield model based on Cherenkov emission geometry.

- `LightSabre` — basic light yield, outputs `light_yield` per detector point
- `LightSabrePATD` — photon arrival time distribution; supports single and batched detector positions

**Constructor params:**
- `use_poisson` — Poisson-sample photon counts
- `num_track_points` — discretization of muon track
- `domain_size` — detector domain radius
- `particle_mode` — `'track'` (muon) or `'cascade'`
- `use_perpendicular_distance_only` — skip along-track sampling, use foot-point geometry only

**Batched calling convention:**
- `opt_point` shape `(3,)` or `(1,3)` → returns a single `dict`
- `opt_point` shape `(n_pts, 3)` → returns `list[dict]` of length `n_pts`

Batch mode parallelises all geometry (`t_foot`, `foot_length`, `t_geom_min`, light yield, Poisson
sampling, and the full `(T × n_pts)` track-weight matrix) before the per-detector CPandel loop.

**Key internal methods:**
- `_parse_event_params` — normalise event dict → `(track_pos, track_dir, energy)`
- `_patd_single` — single-detector path (original logic)
- `_patd_batch` — multi-detector path with vectorised geometry
- `_compute_track_weights_batch` — `(T × n_pts)` emission-weight matrix (non-perp mode)
- `_sample_cpandel` — shared CPandel rvs + optional pdf call

**Usage:** Called per-event; fully differentiable wrt detector point positions and event parameters.

### LLRnet (`LLRnet.py`) — *trained neural surrogate*
MLP classifier trained to estimate `log P(signal)/P(background)` at each detector point given event parameters.

**Architecture:** Parallel Fourier-feature branches → residual MLP → sigmoid → LLR = log(p/(1-p))

**Key methods:**
- `create_signal_only_dataloader(signal_sampler, signal_surrogate_func, ...)` — build training data
- `train(train_loader, val_loader, ...)` — train the classifier
- `load_model(name)` / `save_model(name)` — checkpoint I/O
- `predict(features)` — inference

**Training data:** pairs of (detector_point, event_params, light_yield) labeled signal=1 / background=0

### ChargeNet (`ChargeNet.py`)
Neural net regression for charge/light yield prediction. Alternative to analytical LightSabre when data-driven accuracy is needed.

**Architecture:** Optional Fourier features → residual MLP → log-space output

### Pandel / CPandel (`pandel.py`, `cpandel.py`)
Parametric photon timing model (Pandel distribution). Used alongside LightSabrePATD for timing-based likelihood.

**CPandel constructor params:** `tau`, `lambda_s`, `lambda_a`, `v` (photon speed), `s`

### SkewedGaussian (`SkewedGaussian.py`)
Analytical anisotropic Gaussian: larger response ahead of muon direction (`sigma_front`), smaller behind (`sigma_back`), symmetric perpendicular (`sigma_perp`).

### Uniform (`Uniform.py`)
Returns a constant value — useful as a background baseline.

---

## 6. Losses Module

All losses inherit from `base_loss.LossFunction`. Return a `dict` where keys ending in `_loss` are minimized.

### WeightedLightYieldLoss (`light_yield.py`)
Maximizes total light yield summed over all active strings, weighted by `sigmoid(string_weights)`.

**loss_params keys:** `signal_event_params`, `signal_surrogate_func`, `num_events`, `precomputed_signal_yield_per_string`

### WeightedResolutionLoss (`fisher_info.py`)
Minimizes angular or energy resolution using Fisher information matrix.

- `resolution_type` — `'angular'` or `'energy'`
- `fisher_info_params` — which event parameters to differentiate (`['position','energy','direction']`)

**loss_params keys:** `signal_event_params`, `signal_surrogate_func`, `precomputed_fisher_info_per_string_per_event`, `use_relative_energy`, `skip_zero_response`

### WeightedLLRLoss (`LLR.py`)
Maximizes log-likelihood ratio summed over strings using LLRnet predictions.

**loss_params keys:** `llr_net`, `signal_event_params`, `background_event_params`, `signal_surrogate_func`

### SNRloss (`SNR.py`)
Maximizes `Σ(signal) / √Σ(background)`.

### TriggerLoss (`trigger.py`)
Sliding-bar trigger: scores how many detector points respond to a track within a bar window.

### EffectiveAreaLoss (`effective_area.py`)
Integrates muon detection probability along track chords through the detector cylinder.

### FoMLoss (`pointsource_fom.py`)
Combined point-source figure of merit = effective area × angular resolution.

### Geometry Penalties (`geometry_penalties.py`)
All return non-negative values (0 = satisfied constraint):

| Class | Purpose |
|---|---|
| `StringBoundaryPenaltyCircle` | Keep strings inside circular domain |
| `StringBoundaryPenaltySquare` | Keep strings inside square domain |
| `LocalStringRepulsionPenalty` | Prevent nearby strings from clustering |
| `StringRepulsionPenalty` | Global pairwise string repulsion |
| `StringWeightsPenalty` | L1-style penalty to encourage fewer active strings |
| `StringNumberPenalty` | Penalize if active string count exceeds target |
| `WeightBinarizationPenalty` | Push weights toward 0 or 1 (binary) |
| `ROVPenalty` | Penalize strings in ROV exclusion zones |
| `OrderPenalty` | Maintain point ordering along strings |
| `UniquePenalty` | Prevent duplicate point positions |

**ROVPenalty** is particularly important for P-ONE: enforces that strings avoid the space occupied by the remotely-operated vehicle during deployment. Parameters: `rov_rec_width`, `rov_height`, `rov_tri_length`.

---

## 7. Samplers Module

All samplers inherit from `base_sampler.Sampler`.

### CylinderSampler (`cyl_sampler.py`) — *primary sampler*
Generates neutrino-induced muon events that intersect a cylindrical detector volume.

**Constructor params:**
- `event_type` — `'signal'` or `'background'`
- `domain_size` — cylinder radius
- `E_min`, `E_max` — energy range in GeV
- `energy_dist` — `'log_uniform'` (standard) or other distributions
- `find_exact_intersection` — compute exact cylinder entry/exit (slower but more accurate)
- `random_position_along_ray` — randomize where along the track the event is "centered"
- `cos_range` — zenith angle range as cosine values

**Returns from `sample_events(n)`:** list of dicts with keys:
- `position` — 3D vertex position tensor
- `energy` — energy tensor
- `direction` — unit direction vector tensor
- `zenith`, `azimuth` — angles

**Returns from `sample_detector_points(n)`:** tensor of 3D coordinates inside cylinder

### ToySampler (`toy_sampler.py`)
Simplified sampler for debugging; events at fixed positions.

---

## 8. Utils Module

### Optimizer (`basic_optimizer.py`)
The core optimization engine.

**Constructor params:**
- `conflict_free` — use conflict-free multi-gradient (PCGrad) for multi-objective optimization
- `use_alm` — Augmented Lagrangian Method for hard constraints
- `sigmoid_losses` — wrap losses in sigmoid before weighting (prevents scale issues)
- `sigmoid_softness` — sigmoid temperature
- `alm_params` — dict with `gamma` (penalty growth), `alpha` (RMSprop decay), `epsilon`

**`init_geometry(opt_list, schedule_creator, schedule_params, geom_dict)`**
- `opt_list` — list of `(parameter_name, learning_rate)` tuples
- `geom_dict` — optional pre-loaded geometry to continue from

**`optimize(loss_func_dict, loss_weights_dict, loss_params_dict, n_iter, ...)`**

Key kwargs:
- `sigmoid_loss_list` — which losses get sigmoid applied
- `constraints_list` (inside `loss_params_dict`) — names of losses treated as ALM constraints (must = 0)
- `save_geom_folder`, `save_geom_freq` — checkpoint saves
- `vis_freq`, `gif_freq` — visualization cadence
- `continue_saving` — append to existing saved geometries vs overwrite

**ALM (Augmented Lagrangian Method):**
When `use_alm=True`, losses listed in `loss_params_dict['constraints_list']` become hard constraints enforced via:
```
L_ALM = Σ(objective losses) + Σ_c [ λ_c * c(x) + (μ_c/2) * c(x)² ]
```
where `λ_c` (Lagrange multipliers) and `μ_c` (penalties) are updated automatically.

### Evaluator (`basic_evaluator.py`)
Non-differentiable evaluation; wraps loss calls in `torch.no_grad()`.

**`evaluate(geom_dict, loss_func_dict, loss_params_dict, vis_kwargs, visualize)`** → dict of loss values

**`evaluate_multi(geom_dicts, ...)`** → list of result dicts

### Visualizer (`vis_tools.py`)
Generates matplotlib figures and assembles GIFs during optimization.

**Key plot types** (passed as `vis_kwargs['plot_types']`):
- `'loss_components'` — all loss values over iterations
- `'string_weights_scatter'` — XY scatter colored by string weight
- `'string_xy_rov_penalty'` — geometry with ROV exclusion zones overlaid
- `'signal_contour'` — spatial light yield heatmap
- `'energy_resolution_vs_energy'` — resolution binned by energy
- `'angular_resolution_vs_zenith'` — resolution vs zenith angle

**GIF workflow:**
1. Call `visualizer.cleanup_gif_temp_files()` before optimization
2. Optimizer auto-saves frames at `gif_freq`
3. Call `visualizer.finalize_gif(gif_filename, gif_fps)` after optimization

---

## 9. Key Data Structures

### geom_dict
Returned by `geometry.update_points(**geom_dict)` each iteration. Standard keys:

| Key | Shape | Description |
|---|---|---|
| `points_3d` | `(n_points, 3)` | All detector point 3D coordinates |
| `string_xy` | `(n_strings, 2)` | String horizontal positions |
| `z_values` | `(n_strings * pps,)` | Z-coordinates of all points |
| `string_weights` | `(n_strings,)` | Raw learnable weights (before sigmoid) |
| `string_indices` | `(n_strings,)` | String label per point |
| `points_per_string_list` | list | Points count per string |
| `active_points` | `(n_active, 3)` | Only points on active strings |
| `active_string_indices` | `(n_active,)` | String index for active points |

### event_params dict
Returned by `sampler.sample_events(n)` — list of per-event dicts:

| Key | Shape | Description |
|---|---|---|
| `position` | `(3,)` | Vertex position |
| `energy` | `(1,)` | Event energy in GeV |
| `direction` | `(3,)` | Unit direction vector |
| `zenith` | scalar | Zenith angle |
| `azimuth` | scalar | Azimuth angle |

### loss return dict
All loss functions return a dict where:
- Keys ending in `_loss` → minimized by optimizer
- Other keys → logged as metrics / used in visualization

---

## 10. Typical Workflow (from notebook)

The reference notebook `modular_surrogate_notebook.ipynb` demonstrates the canonical workflow:

### Pre-optimization (one-time)
1. Create `LightSabre` and `CylinderSampler`
2. Train `ChargeNet` on (event, detector_point) → light_yield pairs
3. Train `LLRnet` as signal/background classifier using trained ChargeNet or LightSabre
4. Pre-sample a large event set and pre-compute `light_yield_per_string` and `fisher_info_per_string_per_event` — these are saved as `.pt` files and loaded into `loss_params` for fast iteration

### Optimization
5. Instantiate `EvanescentString` geometry (large initial pool of strings)
6. Set up `Optimizer` with `use_alm=True` and `sigmoid_losses=True`
7. Register only `string_weights` for learning (XY positions and Z-spacing held fixed from hex grid)
8. Define physics losses (energy resolution, LLR) + penalty losses (ROV, boundary, repulsion, number, binarization)
9. Mark penalty losses as ALM constraints in `loss_params['constraints_list']`
10. Run `optimizer.optimize(n_iter=1000, ...)` — saves geometry checkpoints every N iterations
11. Finalize GIF animation

### Post-optimization
12. Load saved `geom_dict` from checkpoint
13. Evaluate with `Evaluator` for clean metrics
14. Analyze across multiple geometries using MDS visualization to find clusters of solutions

### Key design decisions in the notebook
- **Pre-computed per-string tensors** (`precomputed_signal_yield_per_string`, `precomputed_fisher_info_per_string_per_event`) enable fast inner-loop evaluation without re-running the full surrogate
- **Only `string_weights` optimized** (not `string_xy`) — starts from a dense hex grid and selects which strings to keep
- **ALM constraints** handle hard geometric constraints (ROV, boundary, repulsion, count) while physics losses are the objectives
- **Sigmoid wrapping** on all losses keeps gradients well-behaved regardless of loss scale
