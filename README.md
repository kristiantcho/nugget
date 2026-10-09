# *nugget*


**NeUtrino experiement Geometry optimization and General Evaluation Tool**

*nugget* is a Python package for optimizing detector geometries of neutrino observatories using machine learning and optimization techniques.

## Features

- **Geometry Types**: Multiple geometry parameterizations including continuous strings, dynamic strings, evanescent strings, N-fold strings, and free points
- **Loss Functions**: Various loss functions for optimization including Fisher information, light yield, effective area, and geometry constraints
- **Surrogate Models**: Neural network surrogates and analytic models for various neutrino event types
- **Sampling**: Event sampling tools for neutrino physics simulations
- **Utilities**: Optimization pipelines and visualization/data tools

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.8.0

### Install from source

1. Clone the repository:
```bash
git clone <repository-url>
cd nugget
```

2. Install the package:
```bash
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage Examples

For a comprehensive example of NUGGET's capabilities, see the Jupyter notebook:

- **`nugget/examples/new_example_notebook.ipynb`**: Demonstrates the complete workflow including:
  - Setting up geometry optimization with the dynamic string configuration
  - Using various loss functions (Fisher information, geometry constraints)
  - Visualization and analysis of optimization results

## Package Structure

- `geometries/`: Detector geometry types
- `losses/`: Loss functions 
- `surrogates/`: Surrogate models 
- `samplers/`: Event sampling utilities
- `utils/`: Optimizer pipeline and visualization/data tools

