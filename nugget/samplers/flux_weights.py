import os
import numpy as np
import pandas as pd
import torch


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _model_for_dataset(conf, dataset_name):
    from pyForwardFolding.config import models_from_config

    models = models_from_config(conf)
    for hist in conf['histograms']:
        for model_name, ds_name in hist['models']:
            if ds_name == dataset_name:
                return models[hist['name']][model_name]
    raise KeyError(f"No model in the pyFF config uses dataset {dataset_name!r}")


def get_flux_weights(energy, zenith, azimuth, friend_config, pyff_config, params=None, pid=14, dataset=None, device=None):
    """Per-event flux Phi and dPhi/dparam from friEnd + pyForwardFolding, in GeV^-1 cm^-2 s^-1 sr^-1.

    zenith/azimuth are in the arrival convention (see sim_weights.event_arrays). params overrides
    the config's model_parameters. Returns (flux (N,), {param: grad (N,)}).
    """
    # JAX preallocates most of every visible GPU on first use.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    from friEnd.api import extend_dataframe, load_config
    from pyForwardFolding.backend import backend
    from pyForwardFolding.config import _load_config

    # Column names match the keys in nugget/other/friend_config.yaml and pyff_config.yaml.
    # fluxless_weight = 1 so friEnd/pyFF return the bare flux; the simulation weight is applied separately.
    df = pd.DataFrame({
        'energy': _to_numpy(energy),
        'zenith': _to_numpy(zenith),
        'azimuth': _to_numpy(azimuth),
        'pid': pid,
        'fluxless_weight': 1.0,
    })
    if isinstance(friend_config, str):
        friend_config = load_config(friend_config)
    df = extend_dataframe(df, friend_config)

    conf = _load_config(pyff_config)
    ds = conf['datasets'][0] if dataset is None else next(d for d in conf['datasets'] if d['name'] == dataset)
    inputs = {}
    for key, entry in ds['param_mapping'].items():
        vals = df[entry['df_key']].to_numpy()
        if entry.get('transform'):
            vals = getattr(backend, entry['transform'])(vals)
        inputs[key] = backend.asarray(vals)

    model = _model_for_dataset(conf, ds['name'])
    model_params = {**conf['model_parameters'], **(params or {})}

    def evaluate(p):
        return model.evaluate(inputs, p)

    def to_torch(a):
        return torch.as_tensor(np.array(a), dtype=torch.float64, device=device)

    flux = to_torch(evaluate(model_params))
    grad_flux = {}
    for name, g in jax.jacfwd(evaluate)(model_params).items():
        g = np.array(g)
        g[np.isnan(g)] = 0.0
        # Parameters the model does not depend on would make the Fisher matrix singular.
        if np.any(g != 0):
            grad_flux[name] = to_torch(g)
    return flux, grad_flux
