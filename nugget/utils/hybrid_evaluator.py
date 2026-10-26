"""Evaluator counterpart of `HybridOptimizer`, for one or many geometries.

Losses configured with ``loss_repeats`` / ``loss_devices`` / ``loss_accumulators`` in the loss params
are wrapped exactly as during optimization, and since evaluation runs under no_grad the accumulators
take their graph-free path: e.g. the analysis loss pools its histograms over all batches before the
Fisher head, so its value is the same quantity the optimizer minimised.
"""

import torch

from nugget.utils.basic_evaluator import Evaluator
from nugget.utils.hybrid_optimizer import wrap_losses

__all__ = ["HybridEvaluator"]


class _Seeded:
    """Reseed torch before every call, so each geometry sees the same random batches and vertices."""

    def __init__(self, loss_func, seed):
        self.loss_func = loss_func
        self.seed = int(seed)

    def __call__(self, geom_dict, **loss_params):
        torch.manual_seed(self.seed)
        return self.loss_func(geom_dict, **loss_params)


class HybridEvaluator(Evaluator):
    """`Evaluator` with the losses wrapped as in `HybridOptimizer`.

    seed (per call) reseeds torch before every loss call, so that comparisons between geometries use
    the same event batches and resampled vertices (common random numbers).
    """

    def __init__(self, *args, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = bool(verbose)

    def _wrap(self, loss_func_dict, loss_params_dict, seed=None, verbose=None):
        wrapped, _ = wrap_losses(loss_func_dict, loss_params_dict or {}, owner=self,
                                 verbose=self.verbose if verbose is None else verbose,
                                 label="HybridEvaluator")
        if seed is not None:
            wrapped = {name: _Seeded(func, seed) for name, func in wrapped.items()}
        return wrapped

    def evaluate(self, *, geom_dict, loss_func_dict, loss_params_dict=None, seed=None, **kwargs):
        return super().evaluate(geom_dict=geom_dict,
                                loss_func_dict=self._wrap(loss_func_dict, loss_params_dict, seed),
                                loss_params_dict=loss_params_dict, **kwargs)

    def evaluate_multi(self, *, geom_dicts, loss_func_dicts=None, loss_func_dict=None,
                       loss_params_dict=None, loss_params_dicts=None, seed=None, **kwargs):
        if loss_func_dicts is None and loss_func_dict is None:
            raise ValueError("Provide either loss_func_dicts or loss_func_dict.")
        if loss_func_dicts is not None and loss_func_dict is not None:
            raise ValueError("Provide only one of loss_func_dicts or loss_func_dict, not both.")
        loss_params_dicts = loss_params_dicts or {}
        wrapped = {}
        for i, name in enumerate(geom_dicts):
            params = {**(loss_params_dict or {}), **(loss_params_dicts.get(name, {}) or {})}
            funcs = loss_func_dicts[name] if loss_func_dicts is not None else loss_func_dict
            wrapped[name] = self._wrap(funcs, params, seed, verbose=self.verbose and i == 0)
        return super().evaluate_multi(geom_dicts=geom_dicts, loss_func_dicts=wrapped,
                                      loss_params_dict=loss_params_dict,
                                      loss_params_dicts=loss_params_dicts, **kwargs)
