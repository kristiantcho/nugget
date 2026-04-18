---
type: module
status: draft
sources:
  - ../../nugget/utils/vis_tools.py
updated: 2026-04-18
---

# vis_tools.py

## Purpose

Comprehensive visualization toolkit for geometry optimization. Provides rendering of detector geometries, loss histories, physics metrics (SNR, LLR, Fisher information, angular/energy resolution), surrogate model comparisons, and interactive 3D plots. Supports static matplotlib plots, GIF animation generation, and optional Plotly interactivity.

## Key Classes/Functions

### Utility Functions

- [`sph_to_cart(theta, phi)`](../../nugget/utils/vis_tools.py#L27) — Convert spherical (zenith, azimuth) to Cartesian 3D unit vector.

- [`cart_to_sph(vec)`](../../nugget/utils/vis_tools.py#L33) — Convert Cartesian 3D vector to spherical (theta, phi).

### Visualizer

[`Visualizer`](../../nugget/utils/vis_tools.py#L42) — Main visualization class for geometry optimization.

**Initialization & Utilities:**

- [`__init__(device, dim, domain_size, gif_temp_dir)`](../../nugget/utils/vis_tools.py#L123) — Initialize visualizer.
  - **device**: torch device for tensor operations.
  - **dim**: Space dimensionality (2 or 3).
  - **domain_size**: Domain size; geometry spans [-size/2, size/2]^dim.
  - **gif_temp_dir**: Optional directory for temporary GIF frame storage.

- [`_safe_tensor_convert(tensor_input, allow_none)`](../../nugget/utils/vis_tools.py#L45) — Clone, detach, and move tensors to CPU; pass other types through.

- [`_z_value_for_confidence(confidence_level)`](../../nugget/utils/vis_tools.py#L144) — Compute z-score for confidence intervals (e.g., 0.95 → 1.96).

- [`_compute_fom_from_resolution(values, min_resolution)`](../../nugget/utils/vis_tools.py#L167) — Compute figure-of-merit (FOM) and uncertainty from per-event resolutions.
  - FOM = sqrt(sum(1 / r_i^2)); uncertainty propagated via first-order approximation.

- [`_compute_pointsource_fom_from_resolution_and_aeff(res_values, aeff_values, min_resolution)`](../../nugget/utils/vis_tools.py#L193) — Compute FOM for point-source sensitivity (includes effective area).

- [`_pad_frames_to_max_size(frames, background_value)`](../../nugget/utils/vis_tools.py#L221) — Pad GIF frames to consistent size with white background.

**Plot Type Constants:** (50+ plot types defined as class constants)

- Loss tracking: `PLOT_LOSS`, `PLOT_UW_LOSS`, `PLOT_LOSS_COMPONENTS`, `PLOT_UW_LOSS_COMPONENTS`.
- Geometry: `PLOT_3D_POINTS`, `PLOT_STRING_XY`, `PLOT_Z_DIST`, `PLOT_STRING_DIST`.
- Physics metrics: `PLOT_SNR_HISTORY`, `PLOT_LLR_HISTORY`, `PLOT_ANGULAR_RESOLUTION`, `PLOT_ENERGY_RESOLUTION`, `PLOT_FISHER_INFO_CONTOUR`.
- LLR plots: `PLOT_LLR_CONTOUR`, `PLOT_SIGNAL_LLR_CONTOUR`, `PLOT_BACKGROUND_LLR_CONTOUR`, `PLOT_LLR_HISTOGRAM`.
- Surrogates: `PLOT_TRUE_FUNCTION`, `PLOT_INTERP_FUNCTION`, `PLOT_ERROR_FUNCTION`, `PLOT_SURROGATE_FUNCTION`.
- ALM: `PLOT_ALM_MU`, `PLOT_ALM_LAMBDA`.

**Core Visualization Methods:**

- `visualize_progress(**kwargs)` — Render specified plot types for the current optimization state.
  - **kwargs**: geometry dict, loss histories, vis parameters (plot_types, iteration, etc.).
  - Handles matplotlib figure creation, layout, and optional GIF frame capture.

- `visualize_multi_progress(geom_vis_kwargs, plot_types, make_gif, **kwargs)` — Render multiple geometries in a grid.
  - **geom_vis_kwargs**: Mapping geom_name → vis kwargs dict.
  - **plot_types**: Shared plot types across all geometries.

**GIF Generation:**

- GIF frames captured during optimization runs; images padded to consistent size, assembled into animated GIF.
- Temporary frame storage managed via `gif_temp_dir` or system temp directory.

### Coordinate Conversions

- **Spherical ↔ Cartesian**: Enables visualization of angular distributions and geometry projection plots.

## Visualization Pipeline

1. **Initialize**: Create Visualizer with device, dimensionality, domain size.
2. **Prepare data**: Convert all tensors to CPU numpy arrays; validate shapes.
3. **Create figure**: Set up matplotlib axes with subplots per plot type.
4. **Render plots**: Loop through requested plot types, calling specialized renderers (loss history, contour, histogram, etc.).
5. **Format axes**: Add labels, colorbars, legends; adjust scales (log, linear, etc.).
6. **Save/display**: Show inline (Jupyter) or save to file; optionally capture for GIF.
7. **Cleanup**: Clear matplotlib state for next iteration.

## Dependencies

- `torch` — tensor operations, device management.
- `numpy` — numerical operations, interpolation.
- `matplotlib` — figure creation, plotting, colormaps.
- `scipy.interpolate.griddata` — 2D surface interpolation for contour plots.
- `imageio` — GIF frame assembly.
- `plotly` (optional) — interactive 3D visualization (gracefully disabled if unavailable).
- `tempfile, os, glob, shutil` — GIF frame file management.

## Notes

- **Tensor safety**: All tensor inputs are cloned, detached, and moved to CPU to avoid in-place modifications and device mismatches.
- **Large file**: vis_tools.py (~8000 lines) contains many specialized plot renderers for different physics quantities.
- **GIF generation**: Frames padded to consistent size due to matplotlib bbox_inches='tight' variations.
- **Multi-geometry rendering**: Supports grid layouts for comparing different geometry configurations.
- **Physics-aware plots**: Includes SNR, LLR, Fisher information, resolution metrics specific to neutrino detector optimization.
- **ALM visualization**: Tracks and plots Lagrange multipliers (λ) and penalty parameters (μ) during constrained optimization.
- **Plotly fallback**: If Plotly unavailable, 3D plots still render via matplotlib.

## See also

- [[utils-basic_optimizer]] — optimization loop calls visualizer
- [[utils-basic_evaluator]] — evaluation loop calls visualizer
- [[concepts-geometry]] — detector geometry representation
- [[concepts-physics]] — SNR, LLR, Fisher information metrics

