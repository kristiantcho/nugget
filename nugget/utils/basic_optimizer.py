import torch
import numpy as np
from conflictfree.grad_operator import ConFIG_update
from conflictfree.weight_model import WeightModel, EqualWeight
from typing import Optional, Sequence, Union
import pickle
import os
import re

class Optimizer():
    
    def __init__(self, device=None, geometry=None, visualizer=None, conflict_free=False, use_custom_cf_weight=True, use_alm=False, alm_params=None, sigmoid_losses=False, sigmoid_softness=1.0):
        
        self.device=device if device is not None else torch.device('cpu')
        self.geometry = geometry
        self.visualizer = visualizer
        self.conflict_free = conflict_free
        self.use_custom_cf_weight = use_custom_cf_weight
        self.use_alm = use_alm
        self.sigmoid_losses = sigmoid_losses
        self.sigmoid_softness = sigmoid_softness  # Default softness for sigmoid losses (can be adjusted via loss_params_dict)
        
        # ALM parameters based on the algorithm in the image
        if alm_params is None:
            alm_params = {}
        self.alm_gamma = alm_params.get('gamma', 1e-2)  # global learning rate
        self.alm_alpha = alm_params.get('alpha', 0.9)   # discounting factor (default from RMSprop)
        self.alm_epsilon = alm_params.get('epsilon', 1e-8)  # numerical stability constant
        
        # ALM parameter bounds (None means no limit)
        self.alm_lambda_min = alm_params.get('lambda_min', None)  # Minimum value for Lagrange multipliers
        self.alm_lambda_max = alm_params.get('lambda_max', None)  # Maximum value for Lagrange multipliers
        self.alm_mu_min = alm_params.get('mu_min', None)  # Minimum value for penalty parameters
        self.alm_mu_max = alm_params.get('mu_max', None)  # Maximum value for penalty parameters
        
        # ALM state variables (initialized when constraints are set)
        self.alm_lambdas = {}  # Lagrange multipliers for each constraint
        self.alm_mus = {}      # penalty parameters for each constraint
        self.alm_v_lambda = {}  # weighted moving average for lambda gradients
        self.alm_v_mu = {}      # weighted moving average for mu gradients

    def init_geometry(self, opt_list=[('string_xy', 0.01)], schedule_creator=None, schedule_params=None, geom_dict=None):
        
        self.geom_dict = self.geometry.initialize_points(initial_geometry=geom_dict)
        self.optimizers = {}
        self.schedulers = {}
        for geo_aspect_name, lr in opt_list:
            geo_aspect = self.geom_dict.get(geo_aspect_name)
            print(f'Optimizing {geo_aspect_name} with {geo_aspect.shape} shape')
            geo_aspect.requires_grad = True
            self.geom_dict[geo_aspect_name] = geo_aspect
            geo_optimizer = torch.optim.Adam([geo_aspect], lr=lr)
            self.optimizers[geo_aspect_name] = geo_optimizer
            if schedule_creator is not None and schedule_params is not None:
                if geo_aspect_name in schedule_params:
                    params = schedule_params[geo_aspect_name]
                    geo_scheduler = schedule_creator(geo_optimizer, **params)
                    self.schedulers[geo_aspect_name] = geo_scheduler
            self.geom_dict = self.geometry.update_points(**self.geom_dict)
    
    def _initialize_alm_parameters(self):
        """Initialize ALM parameters for constraint loss components (preserves existing non-zero values)"""
        for constraint_name in self.constraints_list:
            # Initialize Lagrange multipliers (λ) only if not already set
            if constraint_name not in self.alm_lambdas:
                self.alm_lambdas[constraint_name] = torch.tensor(0.0, device=self.device, requires_grad=False)
            # Initialize penalty parameters (μ) only if not already set
            if constraint_name not in self.alm_mus:
                self.alm_mus[constraint_name] = torch.tensor(1.0, device=self.device, requires_grad=False)
            # Initialize weighted moving averages for gradients only if not already set
            if constraint_name not in self.alm_v_lambda:
                self.alm_v_lambda[constraint_name] = torch.tensor(0.0, device=self.device)
            if constraint_name not in self.alm_v_mu:
                self.alm_v_mu[constraint_name] = torch.tensor(0.0, device=self.device)
    
    def _update_alm_parameters(self):
        """Update ALM parameters according to the algorithm"""
        for constraint_name in self.constraints_list:
            if constraint_name in self.loss_dict and len(self.loss_dict[constraint_name]) > 0:
                # Get the latest constraint value C_i(θ)
                constraint_value = self.loss_dict[constraint_name][-1].detach().item()
                # Update weighted moving average for lambda gradient 
                lambda_grad_squared = constraint_value ** 2
                self.alm_v_lambda[constraint_name] = (self.alm_alpha * self.alm_v_lambda[constraint_name] + 
                                                    (1 - self.alm_alpha) * lambda_grad_squared)
                
                # Update μ 
                denominator = torch.sqrt(self.alm_v_lambda[constraint_name]) + self.alm_epsilon
                self.alm_mus[constraint_name] = self.alm_gamma / denominator
                
                # Apply bounds on μ if specified
                if self.alm_mu_min is not None:
                    self.alm_mus[constraint_name] = torch.clamp(self.alm_mus[constraint_name], min=self.alm_mu_min)
                if self.alm_mu_max is not None:
                    self.alm_mus[constraint_name] = torch.clamp(self.alm_mus[constraint_name], max=self.alm_mu_max)
                
                # Update λ 
                self.alm_lambdas[constraint_name] = (self.alm_lambdas[constraint_name] + 
                                                   self.alm_mus[constraint_name] * constraint_value)
                
                # Apply bounds on λ if specified
                if self.alm_lambda_min is not None:
                    self.alm_lambdas[constraint_name] = torch.clamp(self.alm_lambdas[constraint_name], min=self.alm_lambda_min)
                if self.alm_lambda_max is not None:
                    self.alm_lambdas[constraint_name] = torch.clamp(self.alm_lambdas[constraint_name], max=self.alm_lambda_max)
    
    def loss_update_step(self):
        
        total_loss = torch.tensor(0.0, device=self.device)
        if self.conflict_free:
            
            # Clear gradients first
            count = 0
            for geo_aspect_name, optimizer in self.optimizers.items():
                if self.alternate_freq is not None:
                    if not self.optimizer_phases[geo_aspect_name]:
                        continue
                geo_aspect = self.geom_dict[geo_aspect_name]
                count += 1
                optimizer.zero_grad()
                grads = []              
                # Compute gradients for each loss component separately
                for loss_name, loss_fn in self.loss_dict.items():
                    if self.loss_weights_dict.get(loss_name) == 0.0:
                        continue
                    
                    loss_value = loss_fn[-1]
                    if count == 1:    
                        total_loss += loss_value.item()
                    
                    # Apply ALM formulation for constraints or regular loss
                    # if self.use_alm and loss_name in self.constraints_list:
                    #     # Compute ALM loss: λC(θ) + (1/2)μC²(θ)
                    #     constraint_value = self.uw_loss_dict[loss_name][-1]
                    #     augmented_loss = (self.alm_lambdas[loss_name] * constraint_value + 
                    #                     0.5 * self.alm_mus[loss_name] * constraint_value ** 2)
                    #     augmented_loss.backward(retain_graph=True)
                    # else:
                    #     # Regular loss handling
                    loss_value.backward(retain_graph=True)
            
                        # Extract gradients manually for string_weights
                    if geo_aspect.grad is not None:
                        grad_vector = geo_aspect.grad.view(-1).clone()
                        grads.append(grad_vector)
                    
                        # Clear gradients for next loss component
                    geo_aspect.grad = None

                # Calculate conflict-free gradient direction
                if len(grads) > 0:
                    if self.use_custom_cf_weight:
                        weight_model = CustomWeight(self.loss_dict, self.loss_weights_dict)
                    else:
                        weight_model = EqualWeight()
                    g_config = ConFIG_update(grads, weight_model=weight_model)
                    
                    # Apply conflict-free gradients to string_weights manually
                    geo_aspect.grad = g_config.view_as(self.geom_dict[geo_aspect_name])
            for key in self.optimizers.keys():
                if self.alternate_freq is not None:
                    if not self.optimizer_phases[key]:
                        continue
                self.optimizers[key].step()
            
            # Update ALM parameters after parameter update (conflict-free case)
            if self.use_alm:
                self._update_alm_parameters()
        else:
            # Handle regular and (optionally) ALM-augmented losses.
            # Note: when ALM is enabled, constraint losses are expected to have been stored in
            # self.loss_dict already in augmented form (see optimize()).
            for loss_name, loss_fn in self.loss_dict.items():
                if self.loss_weights_dict.get(loss_name) != 0.0:
                    loss_value = loss_fn[-1]
                    total_loss += loss_value
                    loss_value.backward(retain_graph=True)
            # total_loss.backward()
            # Update parameters
            for key in self.optimizers.keys():
                if self.alternate_freq is not None:
                    if not self.optimizer_phases[key]:
                        continue
                self.optimizers[key].step()
            
            # Update ALM parameters after parameter update
            if self.use_alm:
                self._update_alm_parameters()
                
            total_loss = total_loss.item()

        return total_loss

    def _snapshot_geom_dict(self):
        """Create a pickle-friendly snapshot of the current geometry.

        Tensors are detached and moved to CPU to make snapshots easier to load
        on machines without GPU access.
        """
        snapshot = {}
        for key, value in self.geom_dict.items():
            if torch.is_tensor(value):
                snapshot[key] = value.detach().cpu().clone()
            else:
                snapshot[key] = value
        return snapshot

    def optimize(self, loss_func_dict, loss_dict={}, uw_loss_dict={}, loss_weights_dict = {}, loss_params_dict={}, n_iter=100, print_freq=10, vis_freq=None, vis_kwargs={}, gif_freq=None, **kwargs):
        
        self.loss_dict = loss_dict
        self.uw_loss_dict = uw_loss_dict
        self.vis_loss_dict = kwargs.get('vis_loss_dict', {})
        self.vis_uw_loss_dict = kwargs.get('vis_uw_loss_dict', {})
        self.alternate_freq = kwargs.get('alternate_freq', None)
        self.loss_weights_dict = loss_weights_dict
        self.cf_loss_weights_dict = kwargs.get('cf_loss_weights_dict', self.loss_weights_dict)
        self.loss_iterations_dict = kwargs.get('loss_iterations_dict', {})
        self.save_best_geom_file = kwargs.get('save_best_geom_file', None)
        self.save_last_geom = kwargs.get('save_last_geom', False)
        self.sample_every = loss_params_dict.get('sample_every', None)
        self.save_geom_folder = kwargs.get('save_geom_folder', None)
        self.save_geom_freq = kwargs.get('save_geom_freq', 100)
        self.continue_saving = kwargs.get('continue_saving', False)  # Whether to continue incrementing geom save index from existing files in save_geom_folder
        # Optional: only apply sigmoid to a subset of losses.
        # - If provided (list/tuple/set of strings), sigmoid is applied only to those loss names.
        # - If not provided, preserves legacy behavior (sigmoid applied to all losses when enabled).
        self.sigmoid_loss_list = kwargs.get('sigmoid_loss_list', loss_params_dict.get('sigmoid_loss_list', None))
        # Check both kwargs and loss_params_dict for constraints_list
        self.constraints_list = kwargs.get('constraints_list', loss_params_dict.get('constraints_list', []))
        
        # Initialize ALM parameters for constraints
        if self.use_alm:
            self._initialize_alm_parameters()

        # Optional: save intermediate geometry dictionaries.
        # Saves geom_0.pkl (initial) then geom_1.pkl, ... every save_geom_freq iterations.
        self._geom_save_enabled = self.save_geom_folder is not None
        self._geom_save_idx = 0
        if self._geom_save_enabled:
            if not isinstance(self.save_geom_folder, str):
                raise TypeError("save_geom_folder must be a string path")
            if not isinstance(self.save_geom_freq, int):
                raise TypeError("save_geom_freq must be an int")
            if self.save_geom_freq <= 0:
                raise ValueError("save_geom_freq must be a positive integer")
            os.makedirs(self.save_geom_folder, exist_ok=True)

            if self.continue_saving:
                existing_indices = []
                try:
                    for filename in os.listdir(self.save_geom_folder):
                        match = re.fullmatch(r"geom_(\d+)\.pkl", filename)
                        if match:
                            try:
                                existing_indices.append(int(match.group(1)))
                            except ValueError:
                                continue
                except FileNotFoundError:
                    existing_indices = []
                self._geom_save_idx = (max(existing_indices) + 1) if existing_indices else 0

            initial_path = os.path.join(self.save_geom_folder, f"geom_{self._geom_save_idx}.pkl")
            with open(initial_path, 'wb') as f:
                pickle.dump(self._snapshot_geom_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Initialize ALM history dictionaries
        self.alm_lambdas_history = {}
        self.alm_mus_history = {}
        if self.use_alm:
            for constraint_name in self.constraints_list:
                self.alm_lambdas_history[constraint_name] = []
                self.alm_mus_history[constraint_name] = []
        
        for key in loss_func_dict:
            if key not in self.loss_dict:
                self.loss_dict[key] = []
            if key not in self.vis_loss_dict:
                self.vis_loss_dict[key] = []
            if key not in self.uw_loss_dict:
                self.uw_loss_dict[key] = []
            if key not in self.vis_uw_loss_dict:
                self.vis_uw_loss_dict[key] = []
            if key not in self.loss_iterations_dict:
                self.loss_iterations_dict[key] = []
        self.total_loss = []
        if self.alternate_freq is not None:
            self.optimizer_phases = {}
            for key in self.optimizers:
                self.optimizer_phases[key] = False
        
        max_iter = max([len(v) for v in self.loss_iterations_dict.values()]) if len(self.loss_iterations_dict) > 0 else 0     
        for it in range(max_iter, max_iter+n_iter):
            vis_kwargs.update({'iteration': it})
            if self.sample_every is not None:
                if loss_params_dict.get('signal_sampler', None) is not None and it % self.sample_every == 0:
                    loss_params_dict['signal_event_params'] = loss_params_dict['signal_sampler'].sample_events(loss_params_dict.get('num_events', 100))
                    # print(f"Resampled events at iteration {it}")
                    if loss_params_dict.get('background_sampler', None) is not None:
                        loss_params_dict['background_event_params'] = loss_params_dict['background_sampler'].sample_events(loss_params_dict.get('num_events', 100))
            for key in self.loss_iterations_dict:
                self.loss_iterations_dict[key].append(it)
            if self.alternate_freq is not None:
                for ik, key in enumerate(self.optimizers):
                    if ik == 0 and it == 0:
                        self.optimizer_phases[key] = True
                    else:
                        if it % (ik+1)*self.alternate_freq == 0:
                            self.optimizer_phases[key] = True
                        else:
                            self.optimizer_phases[key] = False
            for key in self.optimizers.keys():
                if self.alternate_freq is not None:
                    if self.optimizer_phases[key]:
                        self.optimizers[key].zero_grad()
                else:
                    self.optimizers[key].zero_grad()
            for loss_name, loss_func in loss_func_dict.items():
                # params = loss_params_dict.get(loss_name, {})
                loss_stuff = loss_func(self.geom_dict, **loss_params_dict)
                if isinstance(loss_stuff, dict):
                    loss_value = loss_stuff.get(loss_name, None)
                    vis_kwargs.update(loss_stuff)
                elif isinstance(loss_stuff, tuple) or isinstance(loss_stuff, list):
                    loss_value = loss_stuff[0]
                    vis_kwargs.update({loss_name: loss_stuff[0]})
                else:
                    loss_value = loss_stuff
                    vis_kwargs.update({loss_name: loss_stuff})
                if loss_value is not None:
                    weight = self.loss_weights_dict.get(loss_name, 1.0)
                    # Store the per-loss objective term that will actually be used downstream.
                    # For ALM constraints, this is the augmented loss: λC(θ) + (1/2)μC(θ)^2.
                    weighted_loss = weight * loss_value
                    if self.sigmoid_losses:
                        apply_sigmoid = True
                        if self.sigmoid_loss_list is not None:
                            apply_sigmoid = loss_name in set(self.sigmoid_loss_list)
                        if apply_sigmoid:
                            weighted_loss = torch.sigmoid(self.sigmoid_softness * weighted_loss) - 0.5
                    if self.use_alm and loss_name in self.constraints_list:
                        weighted_loss = (
                            self.alm_lambdas[loss_name] * weighted_loss
                            + 0.5 * self.alm_mus[loss_name] * weighted_loss ** 2
                        )
                    if weight != 0.0:
                        self.loss_dict[loss_name].append(weighted_loss)
                        self.uw_loss_dict[loss_name].append(loss_value)
                    self.vis_loss_dict[loss_name].append(weighted_loss.item())
                    self.vis_uw_loss_dict[loss_name].append(loss_value.item())
                else:
                    print(f"Warning: {loss_name} did not return a valid loss value.")
                vis_kwargs['loss_dict'] = self.vis_loss_dict
                vis_kwargs['uw_loss_dict'] = self.vis_uw_loss_dict
                vis_kwargs['loss_weights_dict'] = self.loss_weights_dict
                vis_kwargs['loss_func_dict'] = loss_func_dict
                vis_kwargs['loss_iterations_dict'] = self.loss_iterations_dict
                if self.use_alm:
                    vis_kwargs['alm_lambdas_history'] = self.alm_lambdas_history
                    vis_kwargs['alm_mus_history'] = self.alm_mus_history
            self.total_loss.append(self.loss_update_step())
            
            # Update ALM history after loss update step
            if self.use_alm:
                for constraint_name in self.constraints_list:
                    self.alm_lambdas_history[constraint_name].append(self.alm_lambdas[constraint_name].item())
                    self.alm_mus_history[constraint_name].append(self.alm_mus[constraint_name].item())

            # Step the schedulers
            if len(self.schedulers) > 0:
                for key in self.schedulers.keys():
                    if self.alternate_freq is not None:
                        if not self.optimizer_phases[key]:
                            continue
                    self.schedulers[key].step()
            self.geom_dict = self.geometry.update_points(**self.geom_dict)

            if self._geom_save_enabled:
                local_step = (it - max_iter) + 1  # 1..n_iter for this optimize() call
                if local_step % self.save_geom_freq == 0:
                    self._geom_save_idx += 1
                    geom_path = os.path.join(self.save_geom_folder, f"geom_{self._geom_save_idx}.pkl")
                    with open(geom_path, 'wb') as f:
                        pickle.dump(self._snapshot_geom_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.save_best_geom_file is not None and (self.total_loss[-1] == min(self.total_loss) or self.save_last_geom):
                with open(self.save_best_geom_file, 'wb') as f:
                    pickle.dump(self.geom_dict, f)
            # print(self.geom_dict['string_weights'])
            vis_kwargs.update(self.geom_dict)
            if it % print_freq == 0 or it == n_iter - 1:
                # print('string weights:', self.geom_dict.get('string_weights'))
                loss_str = ' | '.join([f'{key}: {loss_fn[-1]:.4f}' if self.loss_weights_dict.get(key) != 0.0 else '' for key, loss_fn in self.loss_dict.items()])
                print(f'Iter {it+1}/{n_iter}, Total Loss: {self.total_loss[-1]:.4f} | {loss_str}', flush=True)
            
            if self.visualizer is not None and vis_freq is not None:
                if (it % vis_freq == 0 or it == n_iter - 1):
                    vis_kwargs.update({"make_gif": False})
                    self.visualizer.visualize_progress(**vis_kwargs)
            if self.visualizer is not None and gif_freq is not None:
                if (it % gif_freq == 0 or it == n_iter - 1):
                    vis_kwargs.update({"make_gif": True})
                    self.visualizer.visualize_progress(**vis_kwargs)
        
        return self.geom_dict
    
class CustomWeight(WeightModel):
    """
    A weight model that assigns equal weights to all gradients.
    """

    def __init__(self,
                 losses_dict: Optional[dict] = None,
                 weights_dict: Optional[dict] = None):
        super().__init__()
        self.losses_dict = losses_dict
        self.weights_dict = weights_dict

    def get_weights(
        self,
        gradients: torch.Tensor,
        losses: Optional[Sequence] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """
        Apply weights for the Configfree method.

        Parameters:
        -----------
        weights_dict : dict
            Dictionary of weights for each loss component.
        losses_dict : dict
            Dictionary of loss values for each loss component.
        device : torch.device or str
            Device to use for computations.
        """
        weights_tensor = torch.ones(len(self.losses_dict), device=device)
        for i, key in enumerate(self.losses_dict):
            if self.weights_dict is not None and key in self.weights_dict:
                weights_tensor[i] *= self.weights_dict[key]
            
        return weights_tensor