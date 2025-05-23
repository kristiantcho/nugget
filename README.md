# *nugget*

*nugget* (NeUtrino experiement Geometry optimization and General Evaluation Tool) is a Python package for optimizing geometric configurations of detectors for astrophysical neutrino experiments.

## Features

- Multiple geometry optimization strategies (FreePoints, DynamicString, ContinuousString)
- Various loss functions for optimization (RBF interpolation, Signal-to-Noise Ratio)
- Built-in visualization tools for 2D and 3D configurations
- Learning rate schedulers for efficient optimization
- Surrogate functions for testing optimization strategies on different neutrino signal models

## Geometry Types

*nugget* implements three different geometry types for optimization:

### FreePoints
Points can be freely positioned in 3D space with no constraints on their arrangement. This provides maximum flexibility but may be less physically realistic for some detector designs.

### DynamicString
Points are arranged along vertical strings with fixed or free moving XY positions. The number of points on each string can also be optimized, allowing the algorithm to allocate more points to regions of interest (still under development). This geometry type closely resembles the structure of in-ice or underwater neutrino detectors like IceCube or P-ONE.

### ContinuousString
Similar to DynamicString, but points are distributed along a continuous path through the detector volume. This allows for smooth, continuous optimization of point positions and is useful for detectors with flexible sensor deployment options.

## Loss Functions

*nugget* provides two primary loss functions for optimization:

### RBF Interpolation Loss
This loss function uses Radial Basis Function (RBF) interpolation to reconstruct signal functions from detector points. It optimizes sensor placement to minimize the error between the true signal and the reconstructed signal. Key features:

- Employs Gaussian RBFs with configurable epsilon parameter
- Includes additional penalties for boundary constraints and point repulsion
- Supports sampling bias to focus on high-value regions
- Well-suited for general signal reconstruction problems

### Signal-to-Noise Ratio (SNR) Loss
This loss function maximizes the signal-to-noise ratio for detecting neutrino events in the presence of background noise. Key features:

- Optimizes detector geometry to maximize sensitivity to specific signal types
- Can operate with or without background functions
- Configurable signal and background scaling
- Supports neutrino event specific parameter optimization (up to two parameters)

Both loss functions support additional regularization terms including:
- Boundary constraints to keep points within the domain
- Repulsion penalties to prevent points from clustering
- String-specific penalties for the DynamicString and ContinuousString geometries

## Installation

```bash
# Clone the repository
git clone https://github.com/kristiantcho/nugget.git
cd nugget

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from nugget import GeoOptimizer

# Create a GeoOptimizer instance
rbf_optimizer = GeoOptimizer(
    dim=3,                  # 3D space
    domain_size=2.0,        # Domain size from -1 to 1
    epsilon=30.0,           # RBF kernel parameter
    num_iterations=100,     # Number of optimization iterations
    visualize_every=50      # Visualization frequency
)

# Run optimization for a continuous string geometry with 40 points
results = rbf_optimizer.optimize_geometry(
    geometry_type="continuous_string",
    num_points=40,
    loss_type="rbf_interpolation"
)

# Visualize the final optimized geometry
rbf_optimizer.visualize_results(results)
```

## Examples

See the `test_optimization.ipynb` notebook for detailed examples of different optimization strategies.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.


