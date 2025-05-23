import torch
import math
import numpy as np


class Scheduler:
    """Base class for learning rate schedulers in optimization."""
    
    def __init__(self, optimizer, device=None):
        """
        Initialize the scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled
        device : torch.device or None
            Device to use for computations.
        """
        self.optimizer = optimizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_lr = [group['lr'] for group in optimizer.param_groups]
        self.current_iteration = 0
    
    def step(self):
        """
        Update learning rate based on the current iteration.
        This method should be called after each optimization step.
        
        Returns:
        --------
        list
            Current learning rates for each parameter group
        """
        self.current_iteration += 1
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a dictionary.
        
        Returns:
        --------
        dict
            Scheduler state
        """
        return {
            'initial_lr': self.initial_lr,
            'current_iteration': self.current_iteration
        }
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from a dictionary.
        
        Parameters:
        -----------
        state_dict : dict
            State dictionary from a previous checkpoint
        """
        self.initial_lr = state_dict['initial_lr']
        self.current_iteration = state_dict['current_iteration']
    
    def get_lr(self):
        """
        Get current learning rate.
        
        Returns:
        --------
        list
            Current learning rates for each parameter group
        """
        return [group['lr'] for group in self.optimizer.param_groups]


class CosineScheduler(Scheduler):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer, num_iterations, eta_min=0, device=None):
        """
        Initialize the cosine scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled
        num_iterations : int
            Total number of iterations
        eta_min : float
            Minimum learning rate
        device : torch.device or None
            Device to use for computations.
        """
        super().__init__(optimizer, device=device)
        self.T_max = num_iterations
        self.eta_min = eta_min
    
    def step(self):
        """
        Update learning rate using cosine annealing.
        
        Returns:
        --------
        list
            Current learning rates for each parameter group
        """
        self.current_iteration += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.initial_lr[i] - self.eta_min) * (
                1 + math.cos(math.pi * self.current_iteration / self.T_max)
            ) / 2
            param_group['lr'] = lr
        
        return self.get_lr()
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a dictionary.
        
        Returns:
        --------
        dict
            Scheduler state
        """
        state = super().state_dict()
        state.update({
            'T_max': self.T_max,
            'eta_min': self.eta_min
        })
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from a dictionary.
        
        Parameters:
        -----------
        state_dict : dict
            State dictionary from a previous checkpoint
        """
        super().load_state_dict(state_dict)
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']


class StepScheduler(Scheduler):
    """Step learning rate scheduler."""
    
    def __init__(self, optimizer, step_size, gamma=0.1, device=None):
        """
        Initialize the step scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled
        step_size : int
            Period of learning rate decay (in iterations)
        gamma : float
            Multiplicative factor of learning rate decay
        device : torch.device or None
            Device to use for computations.
        """
        super().__init__(optimizer, device=device)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self):
        """
        Update learning rate using step decay.
        
        Returns:
        --------
        list
            Current learning rates for each parameter group
        """
        self.current_iteration += 1
        
        if self.current_iteration % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.gamma
        
        return self.get_lr()
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a dictionary.
        
        Returns:
        --------
        dict
            Scheduler state
        """
        state = super().state_dict()
        state.update({
            'step_size': self.step_size,
            'gamma': self.gamma
        })
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from a dictionary.
        
        Parameters:
        -----------
        state_dict : dict
            State dictionary from a previous checkpoint
        """
        super().load_state_dict(state_dict)
        self.step_size = state_dict['step_size']
        self.gamma = state_dict['gamma']


class ExponentialScheduler(Scheduler):
    """Exponential learning rate scheduler."""
    
    def __init__(self, optimizer, gamma=0.95, device=None):
        """
        Initialize the exponential scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled
        gamma : float
            Multiplicative factor of learning rate decay
        device : torch.device or None
            Device to use for computations.
        """
        super().__init__(optimizer, device=device)
        self.gamma = gamma
    
    def step(self):
        """
        Update learning rate using exponential decay.
        
        Returns:
        --------
        list
            Current learning rates for each parameter group
        """
        self.current_iteration += 1
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.gamma
        
        return self.get_lr()
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a dictionary.
        
        Returns:
        --------
        dict
            Scheduler state
        """
        state = super().state_dict()
        state.update({
            'gamma': self.gamma
        })
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from a dictionary.
        
        Parameters:
        -----------
        state_dict : dict
            State dictionary from a previous checkpoint
        """
        super().load_state_dict(state_dict)
        self.gamma = state_dict['gamma']


class LinearScheduler(Scheduler):
    """Linear learning rate scheduler."""
    
    def __init__(self, optimizer, num_iterations, end_factor=0.01, device=None):
        """
        Initialize the linear scheduler.
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate will be scheduled
        num_iterations : int
            Total number of iterations
        end_factor : float
            Target factor to reduce the learning rate to by the end of training
        device : torch.device or None
            Device to use for computations.
        """
        super().__init__(optimizer, device=device)
        self.num_iterations = num_iterations
        self.end_factor = end_factor
    
    def step(self):
        """
        Update learning rate using linear decay.
        
        Returns:
        --------
        list
            Current learning rates for each parameter group
        """
        self.current_iteration += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            progress = min(1.0, self.current_iteration / self.num_iterations)
            lr_factor = 1.0 - (1.0 - self.end_factor) * progress
            param_group['lr'] = self.initial_lr[i] * lr_factor
        
        return self.get_lr()
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a dictionary.
        
        Returns:
        --------
        dict
            Scheduler state
        """
        state = super().state_dict()
        state.update({
            'num_iterations': self.num_iterations,
            'end_factor': self.end_factor
        })
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state from a dictionary.
        
        Parameters:
        -----------
        state_dict : dict
            State dictionary from a previous checkpoint
        """
        super().load_state_dict(state_dict)
        self.num_iterations = state_dict['num_iterations']
        self.end_factor = state_dict['end_factor']


def create_scheduler(optimizer, num_iterations, scheduler_type=None, scheduler_params=None):
    """
    Create a learning rate scheduler for an optimizer.
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    num_iterations : int
        Total number of iterations
    scheduler_type : str or None
        Type of learning rate scheduler ('cosine', 'step', 'exp', 'linear', None)
    scheduler_params : dict or None
        Additional parameters for the scheduler
        
    Returns:
    --------
    Scheduler or None
        Learning rate scheduler or None if scheduler_type is None
    """
    # Don't use a scheduler if no type specified
    if scheduler_type is None:
        return None
        
    # Initialize default params if None
    if scheduler_params is None:
        scheduler_params = {}
        
    # Create appropriate scheduler
    if scheduler_type.lower() == 'cosine':
        # Cosine annealing scheduler
        eta_min = scheduler_params.get('eta_min', 0)
        return CosineScheduler(
            optimizer, num_iterations=num_iterations, eta_min=eta_min
        )
    elif scheduler_type.lower() == 'step':
        # Step scheduler that reduces LR by gamma every step_size iterations
        step_size = scheduler_params.get('step_size', num_iterations // 3)
        gamma = scheduler_params.get('gamma', 0.1)
        return StepScheduler(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type.lower() == 'exp':
        # Exponential decay scheduler
        gamma = scheduler_params.get('gamma', 0.95)
        return ExponentialScheduler(
            optimizer, gamma=gamma
        )
    elif scheduler_type.lower() == 'linear':
        # Linear decay scheduler
        end_factor = scheduler_params.get('end_factor', 0.01)
        return LinearScheduler(
            optimizer, num_iterations=num_iterations, end_factor=end_factor
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")