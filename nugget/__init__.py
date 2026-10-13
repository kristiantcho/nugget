import torch

# Enforce float64 everywhere by default. Several loss computations (Fisher
# information matrix inversion, effective-area cross sections spanning many
# orders of magnitude in energy, flux weights from pyForwardFolding which
# forces jax_enable_x64) are numerically fragile in float32 and were a source
# of silent precision loss / NaN geometries when mixed with float32 event and
# geometry tensors. Setting this before any submodule builds a tensor makes
# float64 the default for every torch.zeros/torch.tensor/torch.rand/... call
# that does not explicitly override dtype.
torch.set_default_dtype(torch.float64)

# Import submodules to make them available
from . import surrogates
from . import samplers
from . import losses
from . import geometries
from . import utils