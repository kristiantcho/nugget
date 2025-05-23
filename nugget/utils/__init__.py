"""
*nugget* utility modules for geometry optimization, loss functions, and visualization.
"""

# Import submodules to make them available through the utils namespace
from . import geometries
from . import losses
from . import schedulers
from . import surrogates
from . import vis_tools

# Import key classes for easier access
from .geometries import Geometry, FreePoints, DynamicString, ContinuousString
from .losses import LossFunction, RBFInterpolationLoss, SNRloss
from .surrogates import Surrogate, SkewedGaussian
from .schedulers import create_scheduler
from .vis_tools import Visualizer