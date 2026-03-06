from typing import Any, Dict, Optional, Tuple

import torch


class Evaluator:
    """Single-shot, no-grad evaluator for a fixed geometry.

    This is the evaluation counterpart to `basic_optimizer.Optimizer.optimize()`.
    It runs each loss function once on the provided geometry, prints the raw
    (unweighted, non-sigmoided) values, and optionally calls the visualizer.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        geometry: Any = None,
        visualizer: Any = None,
    ) -> None:
        self.device = device if device is not None else torch.device("cpu")
        self.geometry = geometry
        self.visualizer = visualizer

    @staticmethod
    def _unpack_loss_output(loss_name: str, loss_stuff: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Match Optimizer.optimize() conventions for loss function outputs."""
        extra_kwargs: Dict[str, Any] = {}
        loss_value: Optional[torch.Tensor]

        if isinstance(loss_stuff, dict):
            loss_value = loss_stuff.get(loss_name, None)
            extra_kwargs.update(loss_stuff)
        elif isinstance(loss_stuff, (tuple, list)):
            loss_value = loss_stuff[0] if len(loss_stuff) > 0 else None
            if loss_value is not None:
                extra_kwargs.update({loss_name: loss_value})
        else:
            loss_value = loss_stuff
            if loss_value is not None:
                extra_kwargs.update({loss_name: loss_value})

        return loss_value, extra_kwargs

    def evaluate(
        self,
        *,
        geom_dict: Dict[str, Any],
        loss_func_dict: Dict[str, Any],
        loss_params_dict: Optional[Dict[str, Any]] = None,
        print_result: bool = True,
        visualize: bool = False,
        make_gif: bool = False,
        vis_kwargs: Optional[Dict[str, Any]] = None,
        update_points: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate all losses once (no gradients).

        Parameters
        ----------
        geom_dict:
            Geometry dictionary to evaluate.
        loss_func_dict:
            Mapping loss_name -> callable(geom_dict, **loss_params_dict).
        loss_params_dict:
            Shared kwargs passed to every loss function.
        visualize:
            If True, calls `self.visualizer.visualize_progress(**vis_kwargs)` once.
        """

        if loss_params_dict is None:
            loss_params_dict = {}
        if vis_kwargs is None:
            vis_kwargs = {}

        # Work on a shallow copy so caller's dict isn't mutated accidentally.
        
        self.geom_dict = self.geometry.initialize_points(initial_geometry=geom_dict)

        losses: Dict[str, float] = {}

        with torch.no_grad():
            for loss_name, loss_func in loss_func_dict.items():
                loss_stuff = loss_func(self.geom_dict, **loss_params_dict)
                loss_value, extra = self._unpack_loss_output(loss_name, loss_stuff)
                vis_kwargs.update(extra)

                if loss_value is None:
                    print(f"Warning: {loss_name} did not return a valid loss value.")
                    continue

                losses[loss_name] = float(loss_value.detach().cpu().item())

        if update_points and self.geometry is not None and hasattr(self.geometry, "update_points"):
            self.geom_dict = self.geometry.update_points(**self.geom_dict)

        vis_kwargs.update(self.geom_dict)
        vis_kwargs.update(kwargs)

        # Visualizer defaults often include loss plots that expect a non-None
        # `loss_history`. Provide a single-point history for this evaluation.
        total_loss = float(sum(losses.values()))
        vis_kwargs.setdefault("loss_history", [total_loss])
        vis_kwargs.setdefault("iteration", 0)

        

        if self.visualizer is not None and visualize:
            vis_kwargs.update({"make_gif": bool(make_gif)})
            self.visualizer.visualize_progress(**vis_kwargs)
            
        if print_result:
            if len(losses) == 0:
                print("No valid losses returned.", flush=True)
            else:
                loss_str = " | ".join([f"{k}: {v:.6g}" for k, v in losses.items()])
                print(f"Eval | {loss_str}", flush=True)

        return {
            "geom_dict": self.geom_dict,
            "losses": losses,
            "vis_kwargs": vis_kwargs,
        }
