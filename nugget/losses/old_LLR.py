from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F
import numpy as np

class WeightedLLRLoss(LossFunction):
    """Loss function for weighted LLR optimization using surrogate functions and LLRnet or KLNet.
    
    This loss function supports two modes of operation:
    1. Traditional mode: Uses LLRnet with surrogate functions to generate events and compute LLR
    2. KLNet mode: Uses a pre-trained KLNet model to directly predict expected LLR values
    
    KLNet mode is more efficient as it bypasses event generation and directly predicts 
    expected LLR values at detector positions given event parameters.
    

    """
    
    def __init__(self, device=None, signal_scale=1.0, background_scale=1.0, 
                 add_noise=True, sig_noise_scale=0.1, bkg_noise_scale=0.1, num_samples=100, print_loss=False, llr_net=None, signal_surrogate_func=None, 
                 background_surrogate_func=None, signal_event_params=None, background_event_params=None,
                 batch_size_per_point=32, random_seed=None, keep_opt_point=True, signal_ratio = 0.5, use_bce_loss=False, boost_signal=True,
                 boost_signal_yield=False):
        """
        Initialize the weighted LLR loss function.
        
        Parameters:
        -----------
        device : torch.device or None
            Device to use for computations.
        repulsion_weight : float
            Weight for repulsion penalty between points on the same string.
        boundary_weight : float
            Weight for boundary penalty.
        eva_string_num_weight : float
            Weight for controlling number of active strings.
        eva_binary_weight : float
            Weight for encouraging binary string weights.
        eva_min_num_strings : int
            Minimum number of active strings.
        string_repulsion_weight : float
            Weight for repulsion between strings.
        max_local_rad : float
            Maximum radius for local string repulsion.
        path_repulsion_weight : float
            Weight for repulsion in path space.
        z_repulsion_weight : float
            Weight for z distance penalty.
        eva_weight : float
            Weight for evanescent string weights penalty.
        eva_boundary_weight : float
            Weight for string weights boundary penalty.
        min_dist : float
            Minimum distance threshold.
        domain_size : float
            Size of the domain.
        llr_weight : float
            Weight for the LLR loss term.
        signal_scale : float
            Scaling factor for signal values.
        background_scale : float
            Scaling factor for background values.
        add_noise : bool
            Whether to add noise to light yield values.
        noise_scale : float
            Standard deviation of noise to add to light yields.
        num_signal_samples : int
            Number of signal samples to generate per evaluation point.
        num_background_samples : int
            Number of background samples to generate per evaluation point.
        conflict_free : bool
            Whether to return losses as separate terms for multi-objective optimization.
        print_loss : bool
            Whether to print loss components during computation.
        llr_net : LLRnet or None
            Trained LLR network for computing log-likelihood ratios.
        signal_surrogate_func : callable or None
            Function that computes signal light yield from event parameters.
        background_surrogate_func : callable or None
            Function that computes background light yield from event parameters.
        signal_event_params : dict or None
            Dictionary containing signal event parameters.
        background_event_params : dict or None
            Dictionary containing background event parameters.
        batch_size_per_string : int
            Number of samples to generate per string for LLR computation.
        random_seed : int or None
            Random seed for reproducibility.
        use_klnet : bool
            Whether to use KLNet for expected LLR computation instead of traditional method.
        klnet_model : KLnet or None
            Trained KLNet model for computing expected LLR values directly.
        compute_fisher_info : bool
            Whether to compute Fisher Information with respect to signal light yield.
        fisher_info_weight : float
            Weight for Fisher Information term in the loss function.
        """
        super().__init__(device)
        
        
        self.signal_scale = signal_scale
        self.background_scale = background_scale
        self.add_noise = add_noise
        self.sig_noise_scale = sig_noise_scale
        self.bkg_noise_scale = bkg_noise_scale
        self.total_samples_per_point = num_samples
        self.print_loss = print_loss
        self.batch_size_per_point = batch_size_per_point
        self.random_seed = random_seed
        self.keep_opt_point = keep_opt_point
        
        # LLR computation components
        self.llr_net = llr_net
        self.signal_surrogate_func = signal_surrogate_func
        self.background_surrogate_func = background_surrogate_func
        self.signal_event_params = signal_event_params
        self.background_event_params = background_event_params
       
        self.signal_ratio = signal_ratio
        self.use_bce_loss = use_bce_loss
        self.boost_signal = boost_signal
        self.boost_signal_yield = boost_signal_yield

    
    def compute_llr_per_string(self, string_xy, n_strings, points_3d, compute_snr=False, compute_signal_yield = False,
                              signal_surrogate_func=None, background_surrogate_func=None, 
                              num_samples_per_point=100, compute_true_llr=False, signal_x_bias=0, background_x_bias=0, zenith_bias=1.5):
        """
        Compute expected LLR for each string using surrogate functions and noise modeling.
        Optionally compute SNR per string as well.
        
        Parameters:
        -----------
        string_xy : torch.Tensor
            XY positions of strings, shape (n_strings, 2).
        n_strings : int
            Number of strings.
        points_3d : torch.Tensor
            Points in 3D space for the geometry.
        compute_snr : bool
            Whether to also compute SNR per string using surrogate functions.
        signal_surrogate_func : callable or None
            Light yield surrogate function for signal events. Required if compute_snr=True.
        background_surrogate_func : callable or None
            Light yield surrogate function for background events. Required if compute_snr=True.
        num_samples_per_point : int
            Number of event samples to use per point when computing SNR.
            
        Returns:
        --------
        tuple
            (llr_per_string, signal_llr_per_string, background_llr_per_string, 
             snr_per_string, true_llr_per_string, true_signal_llr_per_string, 
             true_background_llr_per_string, signal_yield_per_string, fisher_info_per_string)
            - llr_per_string: Expected LLR values for each string, shape (n_strings,)
            - signal_llr_per_string: Signal LLR values for each string
            - background_llr_per_string: Background LLR values for each string  
            - snr_per_string: SNR values for each string (if compute_snr=True, else None)
            - true_llr_per_string: True LLR values (if compute_true_llr=True, else None)
            - true_signal_llr_per_string: True signal LLR values (if compute_true_llr=True, else None)
            - true_background_llr_per_string: True background LLR values (if compute_true_llr=True, else None)
            - signal_yield_per_string: Signal light yield values (if compute_signal_yield=True, else None)
            - fisher_info_per_string: Fisher Information values (if compute_fisher_info=True, else None)
        """
        # Validate SNR computation inputs
        if compute_snr:
            if signal_surrogate_func is None or background_surrogate_func is None:
                raise ValueError("Both signal_surrogate_func and background_surrogate_func must be provided when compute_snr=True")
        llr_per_string = torch.zeros(n_strings, device=self.device)
        snr_per_string = torch.zeros(n_strings, device=self.device) if compute_snr else None
        signal_yield_per_string = torch.zeros(n_strings, device=self.device) if compute_signal_yield else None
        true_llr_per_string = torch.zeros(n_strings, device=self.device) if compute_true_llr else None
        # Track signal and background LLR separately for loss computation
        signal_llr_per_string = torch.zeros(n_strings, device=self.device)
        background_llr_per_string = torch.zeros(n_strings, device=self.device)
        true_signal_llr_per_string = torch.zeros(n_strings, device=self.device) if compute_true_llr else None
        true_background_llr_per_string = torch.zeros(n_strings, device=self.device) if compute_true_llr else None
        
        # For BCE loss, we need to store predicted probabilities for signal and background separately
        signal_probs_per_string = torch.zeros(n_strings, device=self.device) if self.use_bce_loss else None
        background_probs_per_string = torch.zeros(n_strings, device=self.device) if self.use_bce_loss else None
        
        # For Fisher Information computation
        fisher_info_per_string = None
        if self.compute_fisher_info:
            n_params = len(self.fisher_info_params)
            fisher_info_per_string = torch.zeros(n_strings, n_params, n_params, device=self.device)
        # Set random seed for reproducibility if provided
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        if not self.use_klnet: 
            print(f"Computing LLR for {n_strings} strings with {self.total_samples_per_point} samples per point")
        else:
            print(f"Computing expected LLR for {n_strings} strings with {self.total_samples_per_point} samples per string using KLNet")
        
        # Cache surrogate functions for Fisher Information computation
        if self.compute_fisher_info:
            self._cached_signal_surrogate_func = signal_surrogate_func
            self._cached_background_surrogate_func = background_surrogate_func
        
        for s_idx in range(n_strings):
            # Create optimization points for this string
            # Sample points along the string (assuming strings extend in z-direction)
            mask = (points_3d[:, 1] == string_xy[s_idx][1]) & (points_3d[:, 0] == string_xy[s_idx][0])
            string_points = points_3d[mask]
            
            # Generate batch data using LLRnet's method
            # if not self.use_klnet:
            total_llr = 0.0
            total_signal_llr = 0.0
            total_background_llr = 0.0
            total_signal_prob = 0.0
            total_background_prob = 0.0
           
            total_true_llr = 0.0
            total_true_llr_signal = 0.0
            total_true_llr_background = 0.0
        
            for point in string_points:    
                # CORRECTED APPROACH: Generate mixed batches like in training
                # This computes the expected LLR by evaluating how the network responds
                # to a realistic mix of signal and background events at detector positions
                
                mixed_data_loader = self.llr_net.generate_batch_data_from_events(
                    optimization_points=[point],
                    signal_surrogate_func=self.signal_surrogate_func,
                    background_surrogate_func=self.background_surrogate_func,
                    signal_event_params=self.signal_event_params,
                    background_event_params=self.background_event_params,
                    batch_size=self.batch_size_per_point,
                    signal_ratio=self.signal_ratio,  # Use the configured signal ratio
                    add_noise=self.add_noise,
                    sig_noise_scale=self.sig_noise_scale,
                    bkg_noise_scale=self.bkg_noise_scale,
                    random_seed=self.random_seed,
                    keep_opt_point=self.keep_opt_point,
                    output_denoised=compute_true_llr,
                    samples_per_epoch=self.total_samples_per_point,
                    return_event_params=self.compute_fisher_info  # Get event params for Fisher Info
                )
                
                # Compute expected LLR for this string
               
                num_batches = 0
                batch_fisher_matrices = []
                with torch.no_grad():
                    for info in mixed_data_loader:
                        if self.compute_fisher_info:
                            if not compute_true_llr:
                                features, labels, event_params_batch = info
                            else:
                                features, labels, denoised_yield, event_params_batch = info
                        else:
                            if not compute_true_llr:
                                features, labels = info
                            else:
                                features, labels, denoised_yield = info
                        # Compute LLR for this mixed batch
                        batch_llr = self.llr_net.predict_log_likelihood_ratio(features)
                        
                        # Separate LLR by true labels for signal/background tracking
                        signal_mask = labels == 1
                        background_mask = labels == 0
                        
                        if signal_mask.any():
                            total_signal_llr += torch.mean(batch_llr[signal_mask]).item()
                        if background_mask.any():
                            total_background_llr += torch.mean(batch_llr[background_mask]).item()
                        
                        if self.use_bce_loss:
                            batch_pred = self.llr_net.predict_proba(features)
                            
                            if signal_mask.any():
                                total_signal_prob += torch.mean(batch_pred[signal_mask]).item()
                            if background_mask.any():
                                total_background_prob += torch.mean(batch_pred[background_mask]).item()
                        
                        # Expected LLR for this batch (weighted by signal_ratio)
                        # This correctly reflects how signal vs background events 
                        # are distinguished at this detector position
                        total_llr += torch.mean(batch_llr).item()
                        
                        # Compute Fisher Information matrix if requested
                        if self.compute_fisher_info and event_params_batch is not None:
                            # Filter event parameters for signal events only
                            signal_indices = torch.where(signal_mask)[0]
                            signal_event_params = [event_params_batch[i.item()] for i in signal_indices]
                            with torch.enable_grad():
                                batch_fisher_matrix = self._compute_fisher_information(signal_event_params, point)
                            batch_fisher_matrices.append(batch_fisher_matrix)
                        if compute_true_llr:
                            features = np.array(features).squeeze()
                            denoised_yield = denoised_yield.cpu().numpy().squeeze()
                            signal_mean_denoised_yield = np.mean(denoised_yield[signal_mask.cpu().numpy()])
                            background_mean_denoised_yield = np.mean(denoised_yield[background_mask.cpu().numpy()])
                            temp_total_true_llr = np.zeros_like(batch_llr)   
                            temp_total_true_llr += np.log(features[:, -2]) # energy llr
                            temp_total_true_llr += np.log((2+zenith_bias) / (1 + zenith_bias * (1 - np.abs(np.cos(features[:, -4]))))) # zenith llr
                            temp_total_true_llr += np.log(self.bkg_noise_scale/self.sig_noise_scale) # light yield noise llr
                            temp_total_true_llr += 0.5*(np.log(features[:,-1]) - np.log(background_mean_denoised_yield))**2 / self.bkg_noise_scale**2
                            temp_total_true_llr -= 0.5*(np.log(features[:,-1]) - np.log(signal_mean_denoised_yield))**2 / self.sig_noise_scale**2
                            if signal_x_bias != 0 or background_x_bias != 0: # spatial bias llr
                                temp_total_true_llr += (signal_x_bias - background_x_bias)*(features[:,0] + features[:,3]) + (background_x_bias**2 - signal_x_bias**2)/2
                            total_true_llr += np.mean(temp_total_true_llr)
                            total_true_llr_background += np.mean(temp_total_true_llr[background_mask])
                            total_true_llr_signal += np.mean(temp_total_true_llr[signal_mask])
                            

                        num_batches += 1
                        
                        

                # Average expected LLR across all batches for this string
                if num_batches > 0:
                    llr_per_string[s_idx] += total_llr / num_batches
                    signal_llr_per_string[s_idx] += total_signal_llr / num_batches
                    background_llr_per_string[s_idx] += total_background_llr / num_batches
                    if self.use_bce_loss:
                        signal_probs_per_string[s_idx] += total_signal_prob / num_batches
                        background_probs_per_string[s_idx] += total_background_prob / num_batches
                    if compute_true_llr:
                        true_llr_per_string[s_idx] += total_true_llr / num_batches
                        true_signal_llr_per_string[s_idx] += total_true_llr_signal / num_batches
                        true_background_llr_per_string[s_idx] += total_true_llr_background / num_batches
                    if self.compute_fisher_info and batch_fisher_matrices:
                        # Average Fisher Information matrices across batches
                        avg_fisher_matrix = torch.stack(batch_fisher_matrices).mean(dim=0)
                        fisher_info_per_string[s_idx] = avg_fisher_matrix
               
           
            
            # Compute SNR if requested
            if compute_snr or compute_signal_yield:
                
                    # Sample events for SNR computation using the same event parameters as LLR
                signal_indices = torch.randint(0, len(self.signal_event_params['position']), 
                                                (int(self.total_samples_per_point*0.5),), device=self.device)
                background_indices = torch.randint(0, len(self.background_event_params['position']), 
                                                    (self.total_samples_per_point-int(self.total_samples_per_point*0.5),), device=self.device)

                snr_for_string = torch.zeros(len(string_points), device=self.device)
                signal_yield_for_string = torch.zeros(len(string_points), device=self.device)
                
                for point_idx, point in enumerate(string_points):
                    # Compute signal light yield for this point
                    signal_yields = []
                    for sample_idx in range(len(signal_indices)):
                        event_idx = signal_indices[sample_idx].item()
                        signal_yield = signal_surrogate_func(
                            self.signal_event_params, point, event_idx
                        )
                        if self.add_noise:
                            signal_yield = signal_yield + signal_yield*torch.randn(size=signal_yield.shape, device=self.device) * self.sig_noise_scale
                        
                        signal_yields.append(signal_yield * self.signal_scale)
                    
                    # Compute background light yield for this point
                    background_yields = []
                    for sample_idx in range(len(background_indices)):
                        event_idx = background_indices[sample_idx].item()
                        background_yield = background_surrogate_func(
                            self.background_event_params, point, event_idx
                        )
                        if self.add_noise:
                            background_yield = background_yield + background_yield*torch.randn(size=background_yield.shape, device=self.device) * self.bkg_noise_scale
                        background_yields.append(background_yield * self.background_scale)
                    
                    # Convert to tensors and compute average
                    signal_yields = torch.stack(signal_yields) if signal_yields else torch.zeros(1, device=self.device)
                    background_yields = torch.stack(background_yields) if background_yields else torch.ones(1, device=self.device)
                    
                    avg_signal = torch.mean(signal_yields)
                    avg_background = torch.mean(background_yields)
                    
                    # Compute SNR: signal / sqrt(background) with epsilon to avoid division by zero
                    epsilon = 1e-10
                    snr_for_string[point_idx] = avg_signal / torch.sqrt(avg_background + epsilon)
                    signal_yield_for_string[point_idx] = avg_signal
                # summed SNR across all points on this string
                if compute_snr:
                    snr_per_string[s_idx] = torch.sum(snr_for_string)
                if compute_signal_yield:
                    signal_yield_per_string[s_idx] = torch.sum(signal_yield_for_string)
                        
             
           
        if self.print_loss:
            print(f"Expected LLR per string: {llr_per_string}")
            print(f"Signal LLR per string: {signal_llr_per_string}")
            print(f"Background LLR per string: {background_llr_per_string}")
            if compute_snr:
                print(f"SNR per string: {snr_per_string}")
            if self.use_bce_loss:
                print(f"Signal probs per string: {signal_probs_per_string}")
                print(f"Background probs per string: {background_probs_per_string}")

       
        return llr_per_string, signal_llr_per_string, background_llr_per_string, snr_per_string, true_llr_per_string, true_signal_llr_per_string, true_background_llr_per_string, signal_yield_per_string, fisher_info_per_string
       

    def set_fisher_info_params(self, params=['energy', 'azimuth', 'zenith']):
        """
        Set which parameters to compute Fisher Information for.
        
        Parameters:
        -----------
        params : list
            List of parameter names to compute Fisher Information for.
            Available options: ['energy', 'azimuth', 'zenith', 'position']
        """
        self.fisher_info_params = params
        
    def set_signal_surrogate_func(self, signal_surrogate_func):
        """
        Set the signal surrogate function for Fisher Information computation.
        
        Parameters:
        -----------
        signal_surrogate_func : callable
            Signal surrogate function that takes (event_params, detector_point, event_idx)
            and returns light yield mean value.
        """
        self.signal_surrogate_func = signal_surrogate_func

    def _compute_fisher_information(self, event_params_batch, detector_point):
        """
        Compute Fisher Information matrix with respect to specified parameters using Poisson light yield.
        
        Assumes light yield follows a Poisson distribution where the surrogate function output
        represents the mean (λ). For Poisson distributions, Fisher Information matrix with respect
        to parameters θ is: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features for the batch
        labels : torch.Tensor  
            True labels (0=background, 1=signal)
        event_params_batch : list
            Event parameters corresponding to each feature in the batch
        llr_values : torch.Tensor
            Log-likelihood ratio values for the batch
        detector_point : torch.Tensor
            Current detector point being evaluated
            
        Returns:
        --------
        torch.Tensor
            Fisher Information matrix for this batch (n_params, n_params)
        """
        n_params = len(self.fisher_info_params)
        fisher_matrix = torch.zeros(n_params, n_params, device=self.device)
        
        # if event_params_batch is None or len(event_params_batch) == 0:
        #     return fisher_matrix
            
      
        # # Extract signal events only (Fisher Info computed for signal parameters)
        # signal_mask = labels == 1
        # if not signal_mask.any():
        #     return fisher_matrix
            
        # # Get signal events
        # signal_indices = torch.where(signal_mask)[0]
        # if len(signal_indices) == 0:
        #     return fisher_matrix
            
        num_valid_events = 0
        
        # Compute Fisher Information for each signal event
        for batch_idx in range(len(event_params_batch)):
            
                
            event_params = event_params_batch[batch_idx]
            if event_params is None:
                continue
            
            # Extract parameter values for this event
            if not isinstance(event_params, dict):
                continue

            # Handle both dictionary and direct parameter formats
            if isinstance(event_params, dict):
                # Direct dictionary access
                energy = event_params.get('energy', torch.tensor(1.0, device=self.device))
                azimuth = event_params.get('azimuth', torch.tensor(0.0, device=self.device))
                zenith = event_params.get('zenith', torch.tensor(0.0, device=self.device))
                position = event_params.get('position', torch.zeros(3, device=self.device))
           
                
                # Ensure parameters require gradients
                grad_event_params = {}
                for param_name in self.fisher_info_params:
                  
                    if param_name == 'energy':
                        grad_event_params[param_name] = energy.clone().detach().requires_grad_(True)
                    elif param_name == 'azimuth':
                        grad_event_params[param_name] = azimuth.clone().detach().requires_grad_(True)
                    elif param_name == 'zenith':
                        grad_event_params[param_name] = zenith.clone().detach().requires_grad_(True)
                    else:
                        print(f'Using default value for {param_name}')
                        grad_event_params[param_name] = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Add position parameter
                grad_event_params['position'] = position.clone().detach()

                # Compute light yield mean (λ) using the signal surrogate function with gradients
                if hasattr(self, '_cached_signal_surrogate_func') and self._cached_signal_surrogate_func is not None:
                    light_yield_mean = self._cached_signal_surrogate_func(grad_event_params, detector_point, 0, using_gradient =True)
                elif hasattr(self, 'signal_surrogate_func') and self.signal_surrogate_func is not None:
                    light_yield_mean = self.signal_surrogate_func(grad_event_params, detector_point, 0, using_gradient=True)
                else:
                    # Skip Fisher Information computation if no surrogate function available
                    if self.print_loss and batch_idx == 0:  # Only print once per batch
                        print("Warning: No signal surrogate function available for Fisher Information")
                    continue
                    
                # Ensure light yield is positive for Poisson distribution
                # light_yield_mean = torch.clamp(light_yield_mean, min=1e-6)
                
                
                
                # Compute gradients with respect to each parameter
                param_gradients = []
                
                for param_name in self.fisher_info_params:
                    if param_name in grad_event_params:
                        param_tensor = grad_event_params[param_name]
                        
                        # Compute gradient ∂λ/∂θ
                     
                        grad_outputs = torch.ones_like(light_yield_mean)
                        # print(f'Computing gradient for {param_name} = {param_tensor}')
                        param_grad = torch.autograd.grad(
                            outputs=light_yield_mean,
                            inputs=param_tensor,
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=True,
                            only_inputs=True,
                            allow_unused=True
                        )[0]
                        # print(f'Param: {param_name}, Value: {param_tensor}, Grad: {param_grad}')
                        if param_grad is not None:
                            param_gradients.append(param_grad)
                        else:
                            param_gradients.append(torch.zeros_like(param_tensor))
                        # except Exception as grad_e:
                        #     if self.print_loss:
                        #         print(f"Warning: Gradient computation failed for {param_name}: {grad_e}")
                        #     param_gradients.append(torch.zeros_like(param_tensor))
                    else:
                        # Parameter not available, use zero gradient
                        print(f'Warning: Parameter {param_name} not found, using zero gradient.')
                        param_gradients.append(torch.tensor(0.0, device=self.device))
                
                # Compute Fisher Information matrix: I(θ_i, θ_j) = E[(∂λ/∂θ_i)(∂λ/∂θ_j)/λ]
                if len(param_gradients) == n_params:
                    for i_param in range(n_params):
                        for j_param in range(n_params):
                            try:
                                grad_i = param_gradients[i_param]
                                grad_j = param_gradients[j_param]
                                
                                # Fisher Information element: (∂λ/∂θ_i)(∂λ/∂θ_j)/λ
                                fisher_element = (grad_i * grad_j) / light_yield_mean
                                fisher_matrix[i_param, j_param] += fisher_element.item()
                            except Exception as e:
                                if self.print_loss:
                                    print(f"Warning: Fisher matrix element computation failed: {e}")
                                continue
                    
                    num_valid_events += 1
        
        # Average Fisher Information matrix across all valid signal events
        if num_valid_events > 0:
            fisher_matrix /= num_valid_events
            
        return fisher_matrix
                
       

    def __call__(self, string_xy=None, string_weights=None, num_strings=None, precomputed_llr_per_string=None,  precomputed_signal_yield_per_string=None,
                 points_3d=None, compute_snr=False, signal_surrogate_func=None, background_surrogate_func=None, precomputed_fisher_info_per_string=None,
                 num_samples_per_point=100, precomputed_signal_probs_per_string=None, 
                 precomputed_background_probs_per_string=None, precomputed_signal_llr=None, precomputed_background_llr=None, 
                 signal_x_bias = 0, background_x_bias = 0, compute_true_llr=False, points_per_string=1, **kwargs):
        """
        Compute the weighted LLR loss.
        
        Parameters:
        -----------
        points_3d : torch.Tensor or None
            Points in 3D space (not used directly, but kept for compatibility).
        signal_funcs : list of callable or None
            Not used directly (kept for compatibility).
        background_funcs : list of callable or None
            Not used directly (kept for compatibility).
        string_xy : torch.Tensor
            XY positions of strings, shape (n_strings, 2).
        points_per_string_list : list or None
            Number of points on each string (for compatibility).
        string_weights : torch.Tensor or None
            Weights for each string for weighted averaging.
        path_positions : torch.Tensor or None
            Positions along the continuous path (for compatibility).
        num_strings : int or None
            Number of strings in the geometry.
        precomputed_llr_per_string : torch.Tensor or None
            Precomputed LLR values per string. If provided, computation is skipped.
        points_3d : torch.Tensor or None
            Points in 3D space for the geometry.
        compute_snr : bool
            Whether to also compute SNR per string using surrogate functions.
        signal_surrogate_func : callable or None
            Light yield surrogate function for signal events. Required if compute_snr=True.
        background_surrogate_func : callable or None
            Light yield surrogate function for background events. Required if compute_snr=True.
        num_samples_per_point : int
            Number of event samples to use per point when computing SNR.
        points_per_string : list or int
            Number of points at each string.
        **kwargs : dict
            Additional arguments.
            
        Returns:
        --------
        tuple
            If compute_snr=False: (total_loss, weighted_avg_llr_no_penalties, llr_per_string)
            If compute_snr=True: (total_loss, weighted_avg_llr_no_penalties, llr_per_string, snr_per_string)
            - total_loss: Loss with all penalties applied
            - weighted_avg_llr_no_penalties: Weighted average LLR without penalties
            - llr_per_string: LLR values for each string
            - snr_per_string: SNR values for each string (only if compute_snr=True)
        """
        # Determine number of strings
        if num_strings is None:
            if string_weights is not None:
                num_strings = len(string_weights)
            elif string_xy is not None:
                num_strings = len(string_xy)
            else:
                raise ValueError("Cannot determine number of strings")
        
        # Validate string_xy
        if string_xy is None:
            raise ValueError("string_xy must be provided for LLR computation")
        
        # Use precomputed LLR if provided, otherwise compute it
        if precomputed_llr_per_string is not None and precomputed_background_llr is not None and precomputed_signal_llr is not None:
            llr_per_string = precomputed_llr_per_string
            signal_llr_per_string = precomputed_signal_llr
            background_llr_per_string = precomputed_background_llr
            
            
            # For now, assume we don't have precomputed signal/background LLR
            # signal_llr_per_string = torch.zeros_like(llr_per_string)
            # background_llr_per_string = torch.zeros_like(llr_per_string)
            if precomputed_signal_probs_per_string is not None and precomputed_background_probs_per_string is not None and self.use_bce_loss:
                signal_probs_per_string = precomputed_signal_probs_per_string
                background_probs_per_string = precomputed_background_probs_per_string
            if precomputed_signal_yield_per_string is not None:
                signal_yield_per_string = precomputed_signal_yield_per_string
            if precomputed_fisher_info_per_string is not None:
                fisher_info_per_string = precomputed_fisher_info_per_string
        else:
            llr_result = self.compute_llr_per_string(
                string_xy, num_strings, points_3d, compute_snr=compute_snr, 
                signal_surrogate_func=signal_surrogate_func,
                background_surrogate_func=background_surrogate_func,
                num_samples_per_point=num_samples_per_point, signal_x_bias=signal_x_bias,
                background_x_bias=background_x_bias, compute_true_llr=compute_true_llr
            )
            llr_per_string, signal_llr_per_string, background_llr_per_string, snr_per_string, true_llr_per_string, true_signal_llr_per_string, true_background_llr_per_string, signal_yield_per_string, fisher_info_per_string = llr_result

        if type(points_per_string) is int:
            points_per_string = [points_per_string] * num_strings
            points_per_string = torch.tensor(points_per_string, device=self.device)
        elif type(points_per_string) is list:
            points_per_string = torch.tensor(points_per_string, device=self.device)

        # Apply string weights if provided
        if string_weights is not None:
            # Apply sigmoid to get probabilities
            string_probs = string_weights
            clamped_string_probs = torch.sigmoid(string_probs)
            
            # Compute weighted average LLR for signal and background separately
            weighted_signal_llr = torch.mean(signal_llr_per_string * clamped_string_probs / points_per_string)
            weighted_background_llr = torch.mean(background_llr_per_string * clamped_string_probs / points_per_string)
            weighted_llr = torch.sum(llr_per_string * clamped_string_probs)  # Keep for compatibility

            # For BCE loss, compute weighted BCE using string weights as sample weights
            if self.use_bce_loss:
                # Create targets: 1 for signal, 0 for background
                signal_targets = torch.ones(num_strings, device=self.device)
                background_targets = torch.zeros(num_strings, device=self.device)
                
                # Convert probabilities back to logits for binary_cross_entropy_with_logits
                # logit(p) = log(p / (1 - p))
                eps = 1e-7  # Small epsilon to avoid log(0)
                signal_logits = torch.log((signal_probs_per_string + eps) / (1 - signal_probs_per_string + eps))
                background_logits = torch.log((background_probs_per_string + eps) / (1 - background_probs_per_string + eps))
                
                # Compute BCE with logits for signal and background separately, then combine
                signal_bce = F.binary_cross_entropy_with_logits(signal_logits, signal_targets, weight=clamped_string_probs, reduction='mean')
                background_bce = F.binary_cross_entropy_with_logits(background_logits, background_targets, weight=clamped_string_probs, reduction='mean')

                # Weighted BCE combining signal and background
                weighted_bce = (signal_bce + background_bce) / 2
        else:
            # If no weights provided, use simple average
            weighted_signal_llr = torch.mean(signal_llr_per_string)
            weighted_background_llr = torch.mean(background_llr_per_string)
            weighted_llr = torch.mean(llr_per_string)  # Keep for compatibility
            if self.use_bce_loss:
                # Create targets: 1 for signal, 0 for background
                signal_targets = torch.ones(num_strings, device=self.device)
                background_targets = torch.zeros(num_strings, device=self.device)
                
                # Convert probabilities back to logits for binary_cross_entropy_with_logits
                # logit(p) = log(p / (1 - p))
                eps = 1e-7  # Small epsilon to avoid log(0)
                signal_logits = torch.log((signal_probs_per_string + eps) / (1 - signal_probs_per_string + eps))
                background_logits = torch.log((background_probs_per_string + eps) / (1 - background_probs_per_string + eps))
                
                # Compute unweighted BCE with logits
                signal_bce = F.binary_cross_entropy_with_logits(signal_logits, signal_targets, reduction='mean')
                background_bce = F.binary_cross_entropy_with_logits(background_logits, background_targets, reduction='mean')
                
                # Average BCE
                weighted_bce = (signal_bce + background_bce) / 2
            clamped_string_probs = None
        
        # NEW LLR LOSS: Reciprocal of absolute difference between mean signal and background LLR
        if not self.use_bce_loss:
            # Compute absolute difference between signal and background LLR
            llr_difference = torch.abs(weighted_signal_llr - weighted_background_llr)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            # Loss is reciprocal of the difference (we want to maximize the difference)
            
            llr_loss = self.llr_weight / (llr_difference + epsilon)
            if self.boost_signal:
                # Boost signal LLR if configured
                llr_loss += self.signal_boost_weight/ weighted_signal_llr
            if self.boost_signal_yield and signal_yield_per_string is not None:
                llr_loss += self.signal_yield_boost_weight / torch.mean(signal_yield_per_string*clamped_string_probs/points_per_string)
        else:
            llr_loss = self.llr_weight * weighted_bce
        
        # Add Fisher Information term if computed
        if self.compute_fisher_info and fisher_info_per_string is not None:
            try:
                # Sum Fisher Information matrices across strings with optional weighting
                if string_weights is not None:
                    # Apply string weights to Fisher Information matrices
                    clamped_probs_for_fisher = torch.sigmoid(string_weights)
                    # Weight each matrix: sum over strings of (weight * fisher_matrix)
                    total_fisher_matrix = torch.zeros_like(fisher_info_per_string[0])
                    for s_idx in range(len(fisher_info_per_string)):
                        total_fisher_matrix += clamped_probs_for_fisher[s_idx] * fisher_info_per_string[s_idx]
                else:
                    # Simple sum across all strings
                    total_fisher_matrix = torch.sum(fisher_info_per_string, dim=0)
                
                # Compute log determinant of the total Fisher Information matrix
                # Add small regularization to ensure positive definiteness
                reg_matrix = torch.eye(total_fisher_matrix.size(0), device=self.device) * 1e-6
                regularized_fisher = total_fisher_matrix + reg_matrix
                
                # Compute log determinant
                try:
                    fisher_logdet = torch.det(regularized_fisher)
                    if torch.isfinite(fisher_logdet):
                        # Fisher Information bonus (negative log det to encourage higher Fisher Info)
                        fisher_bonus = self.fisher_info_weight / fisher_logdet
                        llr_loss += fisher_bonus
                        
                        if self.print_loss:
                            print(f"Fisher Information matrix trace: {torch.trace(total_fisher_matrix).item():.6f}")
                            print(f"Fisher Information log det: {fisher_logdet.item():.6f}")
                            print(f"Fisher Information bonus: {fisher_bonus.item():.6f}")
                    else:
                        if self.print_loss:
                            print("Warning: Fisher Information log determinant is not finite")
                except Exception as det_e:
                    if self.print_loss:
                        print(f"Warning: Fisher Information log determinant computation failed: {det_e}")
                        
            except Exception as fisher_e:
                if self.print_loss:
                    print(f"Warning: Fisher Information matrix processing failed: {fisher_e}")
        
        # Initialize total loss with LLR loss
        total_loss = llr_loss
        # if self.conflict_free:
        #     total_loss = [llr_loss]
        #     if self.print_loss:
        #         if not self.use_bce_loss:
        #             print(f"Signal LLR mean: {weighted_signal_llr.item():.6f}")
        #             print(f"Background LLR mean: {weighted_background_llr.item():.6f}")
        #             print(f"LLR difference: {torch.abs(weighted_signal_llr - weighted_background_llr).item():.6f}")
        #         print(f"Weighted LLR loss: {llr_loss.item()}")
        # elif self.print_loss:
        #     if not self.use_bce_loss:
        #         print(f"Signal LLR mean: {weighted_signal_llr.item():.6f}")
        #         print(f"Background LLR mean: {weighted_background_llr.item():.6f}")
        #         print(f"LLR difference: {torch.abs(weighted_signal_llr - weighted_background_llr).item():.6f}")
        #     print(f"Weighted LLR loss: {llr_loss.item()}")
        
        # # Compute penalties (same as WeightedSNRLoss)
        # if self.string_repulsion_weight > 0 and string_xy is not None and string_weights is not None:
        #     string_repulsion_penalty = self.compute_local_string_repulsion_penalty(string_xy, clamped_string_probs)
        #     if self.conflict_free:
        #         total_loss.append(string_repulsion_penalty)
        #     else:    
        #         total_loss += string_repulsion_penalty
        #     if self.print_loss:
        #         print(f"String repulsion penalty: {string_repulsion_penalty.item()}")
        
        # if self.eva_boundary_weight > 0 and string_weights is not None:
        #     string_weights_boundary_penalty = self.string_weights_boundary_penalty(clamped_string_probs)
        #     if self.conflict_free:
        #         total_loss.append(string_weights_boundary_penalty)
        #     else:
        #         total_loss += string_weights_boundary_penalty
        #     if self.print_loss:
        #         print(f"String weights boundary penalty: {string_weights_boundary_penalty.item()}")
            
        # if self.eva_string_num_weight > 0 and string_weights is not None:
        #     string_number_penalty = self.string_number_penalty(clamped_string_probs)
        #     if self.conflict_free:
        #         total_loss.append(string_number_penalty)
        #     else:
        #         total_loss += string_number_penalty
        #     if self.print_loss:
        #         print(f"String number penalty: {string_number_penalty.item()}")
            
        # if self.eva_binary_weight > 0 and string_weights is not None:
        #     binarization_penalty = self.weight_binarization_penalty(clamped_string_probs)
        #     if self.conflict_free:
        #         total_loss.append(binarization_penalty)
        #     else:
        #         total_loss += binarization_penalty
        #     if self.print_loss:
        #         print(f"Binarization penalty: {binarization_penalty.item()}")
        
        # # Add string weights penalty if using evanescent strings
        # if string_weights is not None and self.eva_weight > 0:
        #     string_weights_penalty = self.string_weights_penalty(clamped_string_probs)
        #     if self.conflict_free:
        #         total_loss.append(string_weights_penalty)
        #     else:
        #         total_loss += string_weights_penalty
        #     if self.print_loss:
        #         print(f"String weights penalty: {string_weights_penalty.item()}")
        
        # if self.print_loss:    
        #     if self.conflict_free:
        #         print(f"Total loss: {torch.stack(total_loss).sum().item()}")
        #     else:
        #         print(f"Total loss: {total_loss.item()}")
        
        # Prepare weighted Fisher Information output (if computed)
        if self.compute_fisher_info and 'fisher_info_per_string' in locals() and fisher_info_per_string is not None:
            weighted_fisher_info_value = total_fisher_matrix if 'total_fisher_matrix' in locals() else None
        else:
            weighted_fisher_info_value = None

        if compute_snr and points_3d is not None and not self.use_bce_loss:
            return total_loss, weighted_llr.item(), llr_per_string, snr_per_string, weighted_fisher_info_value
        elif compute_snr and points_3d is not None and self.use_bce_loss:
            return total_loss, weighted_llr.item(), weighted_bce.item(), llr_per_string, snr_per_string, signal_probs_per_string, background_probs_per_string, weighted_fisher_info_value
        elif points_3d is not None and not self.use_bce_loss:
            return total_loss, weighted_llr.item(), llr_per_string, weighted_fisher_info_value
        elif points_3d is not None and self.use_bce_loss:
            return total_loss, weighted_llr.item(), weighted_bce.item(), llr_per_string, snr_per_string, signal_probs_per_string, background_probs_per_string, weighted_fisher_info_value
        else:
            return total_loss, weighted_llr.item(), llr_per_string, weighted_fisher_info_value
