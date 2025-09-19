# *nugget*

*nugget* (NeUtrino experiement Geometry optimization and General Evaluation Tool) is a comprehensive Python package for optimizing geometric configurations of detectors for astrophysical neutrino experiments such as IceCube, P-ONE, and similar deep-field detector arrays.

## Features

- **Multiple Geometry Types**: Four different optimization strategies (FreePoints, DynamicString, ContinuousString, EvanescentString)
- **Advanced Loss Functions**: Five loss functions including RBF interpolation, Signal-to-Noise Ratio (SNR), Weighted SNR, and Likelihood Ratio (LLR) loss
- **Comprehensive Visualization**: Interactive 2D and 3D plotting with GIF generation for optimization tracking
- **Flexible Learning Rate Scheduling**: Multiple scheduler types (cosine, exponential, step, linear)
- **Sophisticated Surrogate Models**: Neural network-based surrogate functions with Fourier features for testing optimization strategies
- **GPU/CPU Support**: Automatic device detection with PyTorch backend for accelerated computations
- **Fisher Information Analysis**: Built-in support for parameter estimation and sensitivity analysis

## Geometry Types

*nugget* implements four different geometry types for detector optimization:

### FreePoints
Points can be freely positioned in 3D space with no constraints on their arrangement. This provides maximum flexibility for detector configuration exploration and is ideal for initial concept studies or unconstrained optimization problems.

### DynamicString
Points are arranged along vertical strings with optimizable XY positions. The number of points on each string can be dynamically adjusted during optimization, allowing the algorithm to allocate sensing elements to regions of maximum information gain. This geometry closely resembles the structure of in-ice or underwater neutrino detectors like IceCube or P-ONE.

### ContinuousString
Points are distributed along continuous paths through the detector volume with smooth, differentiable string trajectories. This allows for continuous optimization of point positions along string paths and is particularly useful for detectors with flexible sensor deployment options or cable-based installations.

### EvanescentString
An advanced string geometry where string weights can be optimized, allowing strings to "fade out" (become evanescent) if they don't contribute significantly to the optimization objective. This provides automatic string pruning and optimal resource allocation for large-scale detector arrays.

## Loss Functions

*nugget* provides five sophisticated loss functions optimized for different neutrino detection scenarios:

### RBF Interpolation Loss
Uses Radial Basis Function interpolation to reconstruct signal functions from detector measurements. Optimizes sensor placement to minimize reconstruction error between true signals and detector-interpolated values. Key features:

- Gaussian RBF kernels with configurable epsilon parameter
- Boundary constraints and point repulsion penalties
- Sampling bias for focusing on high-value regions
- Ideal for general signal reconstruction and detector sensitivity studies

### Signal-to-Noise Ratio (SNR) Loss
Maximizes the signal-to-noise ratio for neutrino event detection in the presence of background noise. Features:

- Direct optimization of detector sensitivity to specific signal types
- Configurable signal and background scaling factors
- Support for background-free optimization scenarios
- Optimized for neutrino event parameter estimation

### Weighted SNR Loss
An advanced SNR loss that incorporates Fisher information matrix analysis for optimal parameter estimation. Features:

- Fisher information-based weighting for parameter sensitivity
- Support for multi-parameter optimization (position, direction, energy)
- Angular and energy resolution optimization
- Ideal for precision neutrino parameter reconstruction

### Likelihood Ratio (LLR) Loss
Optimizes detector geometry for maximum likelihood ratio discrimination between signal and background events. Features:

- Event-by-event likelihood ratio computation
- ROC curve optimization for event classification
- Support for realistic neutrino event simulation
- Ideal for neutrino event identification and background rejection

### Weighted LLR Loss
Combines likelihood ratio optimization with Fisher information weighting for comprehensive detector optimization.

All loss functions include:
- Boundary constraints to maintain physical detector domains
- Repulsion penalties to prevent sensor clustering
- String-specific penalties for string-based geometries
- GPU acceleration for large-scale optimizations

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for large optimizations)

### Install from source

```bash
# Clone the repository
git clone https://github.com/kristiantcho/nugget.git
cd nugget

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Dependencies
The package requires the following core dependencies:
- **PyTorch** (≥1.8.0): Deep learning backend with GPU support
- **NumPy** (≥1.19.0): Numerical computing
- **SciPy** (≥1.5.0): Scientific computing and optimization
- **Matplotlib** (≥3.3.0): Static plotting and visualization
- **tqdm** (≥4.45.0): Progress bars during optimization
- **Jupyter** (≥1.0.0): Interactive notebook support

Optional dependencies for enhanced functionality:
- **Plotly**: Interactive 3D visualizations
- **imageio**: GIF generation for optimization animations

## Quick Start

The package provides modular components for building custom neutrino detector optimization workflows:

### Using Geometry Classes Directly

```python
from nugget.utils.geometries import DynamicString, FreePoints
from nugget.utils.losses import RBFInterpolationLoss, SNRloss
import torch

# Set up device and domain
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
domain_size = 2.0

# Initialize geometry
geometry = DynamicString(
    device=device,
    dim=3,
    domain_size=domain_size,
    total_points=150,
    n_strings=30
)

# Initialize geometry points
geom_dict = geometry.initialize_points()
points = geom_dict['points']

# Set up loss function
loss_fn = RBFInterpolationLoss(
    device=device,
    epsilon=30.0,
    domain_size=domain_size
)
```

### Using Surrogate Models

```python
from nugget.utils.surrogates import SkewedGaussian

# Create surrogate model for testing
surrogate = SkewedGaussian(
    device=device,
    dim=3,
    domain_size=domain_size
)

# Generate test functions
test_points = torch.rand(2000, 3, device=device) * domain_size - domain_size/2
surrogate_funcs = surrogate.generate_batch(batch_size=10, test_points=test_points)
```

## Core Components

The package is organized into modular components that can be used independently or combined:

### Geometry Modules (`utils/geometries.py`)
- **FreePoints**: Unconstrained 3D point optimization
- **DynamicString**: Vertical string arrays with optimizable positions
- **ContinuousString**: Smooth string path optimization  
- **EvanescentString**: String arrays with learnable weights

### Loss Functions (`utils/losses.py`)
- **RBFInterpolationLoss**: Signal reconstruction optimization
- **SNRloss**: Signal-to-noise ratio maximization
- **WeightedSNRLoss**: Fisher information weighted optimization
- **WeightedLLRLoss**: Likelihood ratio optimization

### Surrogate Models (`utils/surrogates.py`)
- Neural network-based function approximation
- Fourier feature mappings for coordinate encoding
- Neutrino event simulation and parameter sampling

## Advanced Features

### Learning Rate Scheduling
Support for multiple learning rate schedulers to improve optimization convergence:
- **Cosine annealing**: Smooth learning rate decay
- **Exponential decay**: Traditional exponential scheduling  
- **Step scheduling**: Piecewise constant learning rates
- **Linear decay**: Linear learning rate reduction

### Visualization and Analysis
Comprehensive visualization tools for optimization analysis:
- **3D Interactive Plots**: Real-time detector geometry visualization
- **Optimization GIFs**: Animated optimization progress tracking
- **Loss History**: Detailed loss function evolution
- **SNR/LLR Analysis**: Signal detection performance metrics
- **Fisher Information**: Parameter estimation sensitivity maps

### GPU Acceleration
Full PyTorch backend with GPU support for:
- Large-scale geometry optimizations (>1000 detector elements)
- Neural network surrogate model training
- Batch processing of neutrino event simulations
- Fisher information matrix computations

## Project Structure

```
nugget/
├── __init__.py                 # Main package initialization
├── pyscripts/
│   └── GeoOptimizer.py        # Legacy optimization class (deprecated)
└── utils/
    ├── geometries.py          # Geometry implementations
    ├── losses.py              # Loss function implementations  
    ├── surrogates.py          # Surrogate model implementations
    ├── schedulers.py          # Learning rate schedulers
    └── vis_tools.py           # Visualization utilities
```

### Usage Pattern

The recommended approach is to use the modular components directly rather than the monolithic GeoOptimizer class:

1. **Choose a geometry type** from `utils/geometries.py`
2. **Select a loss function** from `utils/losses.py`  
3. **Set up optimization** using PyTorch optimizers
4. **Add visualization** using tools from `utils/vis_tools.py`
5. **Test with surrogates** from `utils/surrogates.py`




