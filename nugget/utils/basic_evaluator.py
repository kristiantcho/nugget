from typing import Any, Dict, Optional, Tuple, Union

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

    def evaluate_multi(
        self,
        *,
        geom_dicts: Dict[str, Dict[str, Any]],
        loss_func_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
        loss_func_dict: Optional[Dict[str, Any]] = None,
        loss_params_dict: Optional[Dict[str, Any]] = None,
        loss_params_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
        print_result: bool = True,
        visualize: bool = False,
        make_gif: bool = False,
        vis_kwargs: Optional[Dict[str, Any]] = None,
        vis_kwargs_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
        update_points: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate multiple geometries in one call (no gradients).

        This is a multi-geometry counterpart to :meth:`evaluate`.

        Parameters
        ----------
        geom_dicts:
            Mapping geometry_name -> geom_dict.
        loss_func_dicts:
            Mapping geometry_name -> (loss_name -> callable).
        loss_func_dict:
            Shared mapping loss_name -> callable, applied to every geometry.
            Provide either ``loss_func_dict`` or ``loss_func_dicts``.
        loss_params_dict:
            Shared kwargs passed to every loss function for every geometry.
        loss_params_dicts:
            Optional mapping geometry_name -> kwargs passed to every loss
            function for that geometry (merged on top of ``loss_params_dict``).
        vis_kwargs:
            Shared visualization kwargs for all geometries (e.g. ``plot_types``).
        vis_kwargs_dicts:
            Optional mapping geometry_name -> visualization kwargs merged on top
            of ``vis_kwargs`` for that geometry.
        """

        if loss_params_dict is None:
            loss_params_dict = {}
        if vis_kwargs is None:
            vis_kwargs = {}
        if loss_params_dicts is None:
            loss_params_dicts = {}
        if vis_kwargs_dicts is None:
            vis_kwargs_dicts = {}

        if loss_func_dicts is None and loss_func_dict is None:
            raise ValueError("Provide either loss_func_dicts or loss_func_dict.")
        if loss_func_dicts is not None and loss_func_dict is not None:
            raise ValueError("Provide only one of loss_func_dicts or loss_func_dict, not both.")

        # Detect a common caller bug: reusing the same dict object for multiple geometries
        # (e.g. `loss_params_dicts[name] = loss_params` inside a loop). This causes per-geometry
        # values like precomputed tensors to be overwritten in-place, and can silently apply
        # the wrong data to a geometry.
        if loss_params_dicts:
            ids = [id(v) for v in loss_params_dicts.values() if isinstance(v, dict)]
            if len(ids) != len(set(ids)):
                print(
                    "Warning: loss_params_dicts reuses the same dict object across geometries. "
                    "Make a per-geometry copy (e.g. dict(loss_params) or copy.deepcopy(loss_params)) "
                    "to avoid applying the wrong precomputed inputs.",
                    flush=True,
                )

        # Enforce that plot_types are shared across geometries (user-requested).
        shared_plot_types = vis_kwargs.get("plot_types", None)
        for geom_name, per_vis in vis_kwargs_dicts.items():
            if per_vis is None:
                continue
            per_plot_types = per_vis.get("plot_types", None)
            if per_plot_types is None:
                continue
            if shared_plot_types is None:
                shared_plot_types = per_plot_types
            elif per_plot_types != shared_plot_types:
                raise ValueError(
                    "All geometries must use the same plot_types in multi-geometry mode. "
                    f"Mismatch at '{geom_name}'."
                )

        if make_gif:
            # Multi-geometry GIF layout is not implemented yet.
            # Disable rather than failing the entire evaluation.
            make_gif = False

        results: Dict[str, Any] = {}
        multi_vis_payload: Dict[str, Dict[str, Any]] = {}

        for geom_name, geom_dict in geom_dicts.items():
            if loss_func_dicts is not None:
                if geom_name not in loss_func_dicts:
                    raise ValueError(f"Missing loss_func_dict for geometry '{geom_name}'.")
                per_loss_func_dict = loss_func_dicts[geom_name]
            else:
                per_loss_func_dict = loss_func_dict or {}

            per_loss_params: Dict[str, Any] = dict(loss_params_dict)
            per_loss_params.update(loss_params_dicts.get(geom_name, {}) or {})

            per_vis_kwargs: Dict[str, Any] = dict(vis_kwargs)
            per_vis_kwargs.update(vis_kwargs_dicts.get(geom_name, {}) or {})
            if shared_plot_types is not None:
                per_vis_kwargs["plot_types"] = shared_plot_types

            geom_points = self.geometry.initialize_points(initial_geometry=geom_dict)

            losses: Dict[str, float] = {}
            with torch.no_grad():
                for loss_name, loss_func in per_loss_func_dict.items():
                    loss_stuff = loss_func(geom_points, **per_loss_params)
                    loss_value, extra = self._unpack_loss_output(loss_name, loss_stuff)
                    per_vis_kwargs.update(extra)

                    if loss_value is None:
                        print(f"Warning: {geom_name}.{loss_name} did not return a valid loss value.")
                        continue

                    losses[loss_name] = float(loss_value.detach().cpu().item())

            if update_points and self.geometry is not None and hasattr(self.geometry, "update_points"):
                geom_points = self.geometry.update_points(**geom_points)

            per_vis_kwargs.update(geom_points)
            per_vis_kwargs.update(kwargs)

            total_loss = float(sum(losses.values()))
            per_vis_kwargs.setdefault("loss_history", [total_loss])
            per_vis_kwargs.setdefault("iteration", 0)

            results[geom_name] = {
                "geom_dict": geom_points,
                "losses": losses,
                "vis_kwargs": per_vis_kwargs,
            }

            multi_vis_payload[geom_name] = per_vis_kwargs

            if print_result:
                if len(losses) == 0:
                    print(f"Eval[{geom_name}] | No valid losses returned.", flush=True)
                else:
                    loss_str = " | ".join([f"{k}: {v:.6g}" for k, v in losses.items()])
                    print(f"Eval[{geom_name}] | {loss_str}", flush=True)

        if self.visualizer is not None and visualize:
            if hasattr(self.visualizer, "visualize_multi_progress"):
                self.visualizer.visualize_multi_progress(
                    geom_vis_kwargs=multi_vis_payload,
                    plot_types=shared_plot_types,
                    make_gif=bool(make_gif),
                )
            else:
                # Fallback: render each geometry separately.
                for geom_name, per_vis_kwargs in multi_vis_payload.items():
                    per_vis_kwargs = dict(per_vis_kwargs)
                    per_vis_kwargs.setdefault("title", geom_name)
                    self.visualizer.visualize_progress(**per_vis_kwargs)

        return results
