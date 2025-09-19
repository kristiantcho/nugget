import torch
import numpy as np
from conflictfree.grad_operator import ConFIG_update

class Optimizer():
    
    def __init__(self, device=None, geometry=None, visualizer=None, conflict_free=False):
        
        self.device=device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.geometry = geometry
        self.visualizer = visualizer
        self.conflict_free = conflict_free

    def init_geometry(self, opt_list=[('string_xy', 0.01)], schedule_creator=None, schedule_params=None, geom_dict=None):
        
        self.geom_dict = self.geometry.initialize_points(initial_geometry=geom_dict)
        self.optimizers = {}
        self.schedulers = {}
        for geo_aspect_name, lr in opt_list:

            geo_aspect = geom_dict.get(geo_aspect_name)
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
    
    def loss_update_step(self):
        
        count = 0
        total_loss = 0
        if self.conflict_free:
            
            # Clear gradients first
            for geo_aspect_name, optimizer in self.optimizers.items():
                count += 1
                optimizer.zero_grad()
                grads = []              
                # Compute gradients for each loss component separately
                for loss_fn in self.loss_dict.values():
                    total_loss += loss_fn[-1].item()
                    
                    # Compute gradients for this loss component
                    loss_fn.backward(retain_graph=True)
            
                        # Extract gradients manually for string_weights
                    if optimizer.grad is not None:
                        grad_vector = optimizer.grad.view(-1).clone()
                        grads.append(grad_vector)
                    
                        # Clear gradients for next loss component
                    optimizer.grad = None

                # Calculate conflict-free gradient direction
                if len(grads) > 0:
                    g_config = ConFIG_update(grads)
                    
                    # Apply conflict-free gradients to string_weights manually
                    optimizer.grad = g_config.view_as(self.geom_dict[geo_aspect_name])
        else:
            count += 1
            for loss_fn in self.loss_dict.values():
                total_loss += loss_fn[-1]
                total_loss.backward(retain_graph=True)
            # Update parameters
            for optimizer in self.optimizers.values():
                optimizer.step()
            total_loss = total_loss.item()

        return total_loss/count if count > 0 else total_loss

    def optimize(self, loss_func_dict, loss_dict={}, uw_loss_dict={}, loss_weights_dict = {}, loss_params_dict={}, n_iter=100, print_freq=10, vis_freq=None, vis_kwargs={}, gif_freq=None):
        
        self.loss_dict = loss_dict
        self.uw_loss_dict = uw_loss_dict
        for key in loss_func_dict:
            if key not in self.loss_dict:
                self.loss_dict[key] = []
        
        for it in range(n_iter):
            vis_kwargs.update({'iteration': it})
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            for loss_name, loss_func in loss_func_dict.items():
                params = loss_params_dict.get(loss_name, {})
                loss_stuff = loss_func(self.geom_dict, **params)
                if isinstance(loss_stuff, dict):
                    loss_value = loss_stuff.get(loss_name, None)
                    vis_kwargs.update(loss_stuff)
                elif isinstance(loss_stuff, tuple) or isinstance(loss_stuff, list):
                    loss_value = loss_stuff[0]
                else:
                    loss_value = loss_stuff
                if loss_value is not None:
                    weight = loss_weights_dict.get(loss_name, 1.0)
                    weighted_loss = weight * loss_value
                    self.loss_dict[loss_name].append(weighted_loss)
                    self.uw_loss_dict[loss_name].append(loss_value)
                else:
                    print(f"Warning: {loss_name} did not return a valid loss value.")
                vis_kwargs.update(loss_dict)
            self.total_loss.append(self.loss_step())
            
            # Step the schedulers
            if len(self.schedulers) > 0:
                for scheduler in self.schedulers.values():
                    scheduler.step()
            self.geom_dict = self.geometry.update_points(**self.geom_dict)
            
            vis_kwargs.update(self.geom_dict)
            if it % print_freq == 0 or it == n_iter - 1:
                loss_str = ' | '.join([f'{key}: {loss_fn[-1]:.4f}' for key, loss_fn in self.loss_dict.items()])
                print(f'Iter {it+1}/{n_iter}, Total Loss: {self.total_loss[-1]:.4f} | {loss_str}')
            
            if self.visualizer is not None and vis_freq is not None:
                if (it % vis_freq == 0 or it == n_iter - 1):
                    vis_kwargs.update({"make_gif": False})
                    self.visualizer.visualize_progress(**vis_kwargs)
            if self.visualizer is not None and gif_freq is not None:
                if (it % gif_freq == 0 or it == n_iter - 1):
                    vis_kwargs.update({"make_gif": True})
                    self.visualizer.update(**vis_kwargs)
        
        return self.geom_dict