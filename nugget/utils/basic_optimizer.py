import torch
import numpy as np
from conflictfree.grad_operator import ConFIG_update
from conflictfree.weight_model import WeightModel, EqualWeight
from typing import Optional, Sequence, Union
import pickle

class Optimizer():
    
    def __init__(self, device=None, geometry=None, visualizer=None, conflict_free=False, use_custom_cf_weight=True):
        
        self.device=device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.geometry = geometry
        self.visualizer = visualizer
        self.conflict_free = conflict_free
        self.use_custom_cf_weight = use_custom_cf_weight

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
            if schedule_creator is not None:
                if geo_aspect_name in schedule_params:
                    params = schedule_params[geo_aspect_name]
                    geo_scheduler = schedule_creator(geo_optimizer, **params)
                    self.schedulers[geo_aspect_name] = geo_scheduler
            self.geom_dict = self.geometry.update_points(**self.geom_dict)
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
                    if count == 1:    
                        total_loss += loss_fn[-1].item()
                    
                    # Compute gradients for this loss component
                    loss_fn[-1].backward(retain_graph=True)
            
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
        else:
            for loss_name, loss_fn in self.loss_dict.items():
                if self.loss_weights_dict.get(loss_name) != 0.0:
                    total_loss += loss_fn[-1]
                    total_loss.backward(retain_graph=True)
                
            # Update parameters
            for key in self.optimizers.keys():
                if self.alternate_freq is not None:
                    if not self.optimizer_phases[key]:
                        continue
                self.optimizers[key].step()
            total_loss = total_loss.item()

        return total_loss

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
                    weighted_loss = weight * loss_value
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
            self.total_loss.append(self.loss_update_step())
            

            # Step the schedulers
            if len(self.schedulers) > 0:
                for key in self.schedulers.keys():
                    if self.alternate_freq is not None:
                        if not self.optimizer_phases[key]:
                            continue
                    self.schedulers[key].step()
            self.geom_dict = self.geometry.update_points(**self.geom_dict)
            if self.save_best_geom_file is not None and (self.total_loss[-1] == min(self.total_loss) or self.save_last_geom):
                with open(self.save_best_geom_file, 'wb') as f:
                    pickle.dump(self.geom_dict, f)
            # print(self.geom_dict['string_weights'])
            vis_kwargs.update(self.geom_dict)
            if it % print_freq == 0 or it == n_iter - 1:
                # print('string weights:', self.geom_dict.get('string_weights'))
                loss_str = ' | '.join([f'{key}: {loss_fn[-1]:.4f}' if self.loss_weights_dict.get(key) != 0.0 else '' for key, loss_fn in self.loss_dict.items()])
                print(f'Iter {it+1}/{n_iter}, Total Loss: {self.total_loss[-1]:.4f} | {loss_str}')
            
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