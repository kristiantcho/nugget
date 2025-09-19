from nugget.surrogates.base_surrogate import Surrogate
import torch
import numpy as np

class SymbolicReg(Surrogate):
    """
    Symbolic Regression surrogate model based on Julia implementation.
    
    This class implements an analytic amplitude surrogate that produces a 16-entry vector
    output based on symbolic regression from the Julia code. It calculates amplitude
    predictions using multiple rotation matrices and the symbolic regression formula.
    """
    
    def __init__(self, device=None, dim=3, domain_size=2, num_rotations=16):
        """
        Initialize the Symbolic Regression surrogate model.
        
        Parameters:
        -----------
        device : torch.device
            Device to run the model on (CPU or GPU)
        dim : int
            Dimension of the input space (must be 3D for this model)
        domain_size : int
            Length of the domain 
        num_rotations : int
            Number of rotation matrices to use (determines output vector size)
        """
        super().__init__(device=device, dim=dim, domain_size=domain_size)
        
        if dim != 3:
            raise ValueError("SymbolicReg model only supports 3D input space")
        
        self.num_rotations = num_rotations
        
        # Generate random rotation matrices
        self.rotation_matrices = self._generate_rotation_matrices(num_rotations)
        
        # Model constants from Julia code
        self.constants = {
            'c1': 7.2352023283837395e-6,
            'exp1': 0.03443865311003615,
            'c2': 12.39167568068273,
            'c3': 0.38839803112959387,
            'c4': 0.1738077225915957,
            'c5': 3.00545648970963,
            'exp2': 0.5743625057140219,
            'c6': 9.341833951320876
        }
    
    def _generate_rotation_matrices(self, num_rotations):
        """Generate rotation matrices that map uniformly distributed directions to [0,0,1]."""
        rotation_matrices = []
        
        # Generate uniformly distributed points on unit sphere using hexagonal packing
        directions = self._generate_uniform_sphere_directions(num_rotations)
        
        # Target direction [0, 0, 1]
        target = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
        for direction in directions:
            # Create rotation matrix that maps direction to [0,0,1]
            R = self._rotation_matrix_from_vectors(direction, target)
            rotation_matrices.append(R)
        
        return rotation_matrices
    
    def _generate_uniform_sphere_directions(self, n_points):
        """Generate uniformly distributed directions on unit sphere using Fibonacci spiral."""
        directions = []
        
        # Use Fibonacci spiral for uniform distribution on sphere
        golden_ratio = (1 + 5**0.5) / 2
        
        if n_points == 1:
            # Special case for single point
            directions.append(torch.tensor([0.0, 0.0, 1.0], device=self.device))
        else:
            for i in range(n_points):
                # Fibonacci spiral parameters
                theta = 2 * torch.pi * i / golden_ratio
                phi = torch.acos(1 - 2 * i / (n_points - 1))
                
                # Convert to Cartesian coordinates
                x = torch.sin(phi) * torch.cos(theta)
                y = torch.sin(phi) * torch.sin(theta)
                z = torch.cos(phi)
                
                direction = torch.tensor([x, y, z], device=self.device)
                directions.append(direction)
        
        return directions
    
    def _rotation_matrix_from_vectors(self, vec1, vec2):
        """
        Create rotation matrix that rotates vec1 to vec2.
        
        Parameters:
        -----------
        vec1 : torch.Tensor
            Source vector (3D)
        vec2 : torch.Tensor
            Target vector (3D)
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        # Normalize vectors
        v1 = vec1 / torch.norm(vec1)
        v2 = vec2 / torch.norm(vec2)
        
        # Check if vectors are already aligned
        if torch.allclose(v1, v2, atol=1e-6):
            return torch.eye(3, device=self.device)
        
        # Check if vectors are opposite
        if torch.allclose(v1, -v2, atol=1e-6):
            # Find an orthogonal vector
            if abs(v1[0]) < 0.9:
                orthogonal = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                orthogonal = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            
            # Create 180-degree rotation around orthogonal axis
            axis = torch.cross(v1, orthogonal)
            axis = axis / torch.norm(axis)
            return self._rodrigues_rotation_matrix(axis, torch.pi)
        
        # General case: use Rodrigues' rotation formula
        axis = torch.cross(v1, v2)
        axis = axis / torch.norm(axis)
        
        # Calculate rotation angle
        cos_angle = torch.dot(v1, v2)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        
        return self._rodrigues_rotation_matrix(axis, angle)
    
    def _rodrigues_rotation_matrix(self, axis, angle):
        """
        Create rotation matrix using Rodrigues' rotation formula.
        
        Parameters:
        -----------
        axis : torch.Tensor
            Rotation axis (unit vector)
        angle : torch.Tensor
            Rotation angle in radians
            
        Returns:
        --------
        torch.Tensor
            3x3 rotation matrix
        """
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Cross product matrix for axis
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]], 
            [-axis[1], axis[0], 0]
        ], device=self.device)
        
        # Rodrigues' formula
        R = torch.eye(3, device=self.device) + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
        
        return R
    
    def _cart_to_sph(self, xyz):
        """
        Convert Cartesian coordinates to spherical coordinates.
        
        Parameters:
        -----------
        xyz : torch.Tensor
            Cartesian coordinates (x, y, z)
            
        Returns:
        --------
        tuple : (theta, phi) spherical angles
        """
        x, y, z = xyz[0], xyz[1], xyz[2]
        
        # Ensure we don't divide by zero
        r = torch.norm(xyz)
        if r < 1e-10:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))  # polar angle
        phi = torch.atan2(y, x)  # azimuthal angle
        
        return theta, phi
    
    def _sr_amplitude(self, dist, energy, pos_ct, dir_ct, dphi):
        """
        Symbolic regression amplitude function from Julia code.
        
        Parameters:
        -----------
        dist : torch.Tensor
            Distance
        energy : torch.Tensor
            Energy
        pos_ct : torch.Tensor
            cos(theta) for position
        dir_ct : torch.Tensor
            cos(theta) for direction
        dphi : torch.Tensor
            Difference in phi angles
            
        Returns:
        --------
        torch.Tensor
            Amplitude value
        """
        c = self.constants
        
        # Calculate intermediate terms
        denominator_term1 = c['c1'] * (dist ** c['exp1'])
        denominator_term2 = c['c1'] * torch.cos(dphi)
        denominator = denominator_term1 + denominator_term2
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-10)
        
        log_term = torch.log(energy / denominator)
        
        # Calculate the complex denominator for the fraction
        complex_denom_term1 = c['exp1'] * (c['c2'] ** pos_ct) * (energy ** c['exp1'])
        complex_denom_term2 = c['exp1'] * dist
        complex_denom_term3 = (dir_ct - c['c3']) * (pos_ct - c['c4'])
        complex_denom = complex_denom_term1 + complex_denom_term2 + complex_denom_term3 + c['c5']
        
        # Avoid negative or zero values for power operation
        complex_denom = torch.clamp(complex_denom, min=1e-10)
        
        fraction = log_term / (complex_denom ** c['exp2'])
        
        exponent = pos_ct + fraction - c['c6']
        
        # Calculate 10^exponent
        amplitude = 10.0 ** exponent
        
        return amplitude
    
    def _create_model_input(self, position, direction, energy, target_position):
        """
        Create model input by applying rotation matrices and calculating features.
        
        This method transforms physical neutrino parameters into the standardized 5-dimensional
        feature vectors required by the symbolic regression model. The transformation involves:
        
        1. Computing the relative position vector from source to target
        2. Applying multiple rotation matrices to achieve rotational invariance
        3. Converting to spherical coordinates for each rotation
        4. Extracting the key geometric and physical features
        
        Parameters:
        -----------
        position : torch.Tensor
            Source position vector (3D Cartesian coordinates) where the neutrino originated
        direction : torch.Tensor  
            Direction vector (3D Cartesian, unit vector) of neutrino propagation
        energy : torch.Tensor
            Energy value of the neutrino event (scalar)
        target_position : torch.Tensor
            Target position vector (3D Cartesian coordinates) where we evaluate the model
            
        Returns:
        --------
        torch.Tensor
            Input buffer with shape (num_rotations, 5) where each row contains the 5 model features:
            
            dist: Euclidean distance from source to target position
                      This provides spatial scale information for the light yield prediction
                      
            energy: Neutrino energy (same for all rotations)
                      Fundamental physical parameter affecting light production
                      
            pos_ct: cos(theta_position) where theta_position is the polar angle 
                      of the rotated relative position vector in spherical coordinates
                      Range: [-1, 1], represents the z-component of the normalized relative position
                      
            dir_ct: cos(theta_direction) where theta_direction is the polar angle
                      of the rotated direction vector in spherical coordinates  
                      Range: [-1, 1], represents the z-component of the normalized direction
                      
            dphi: Difference in azimuthal angles (phi_position - phi_direction)
                      Range: [-π, π], captures the relative angular orientation between
                      the position and direction vectors in the rotated coordinate system
                      
        """
        # Calculate relative position
        rel_pos = position - target_position
        dist = torch.norm(rel_pos)
        
        # Normalize relative position
        if dist > 1e-10:
            rel_pos_norm = rel_pos / dist
        else:
            rel_pos_norm = torch.zeros_like(rel_pos)
        
        # Initialize input buffer
        input_buffer = torch.zeros(self.num_rotations, 5, device=self.device)
        
        for i, R in enumerate(self.rotation_matrices):
            # Apply rotation to position and direction
            pos_rot = torch.matmul(R, rel_pos_norm)
            dir_rot = torch.matmul(R, direction)
            
            # Normalize direction
            dir_rot = dir_rot / torch.norm(dir_rot)
            
            # Convert to spherical coordinates
            pos_rot_th, pos_rot_phi = self._cart_to_sph(pos_rot)
            dir_rot_th, dir_rot_phi = self._cart_to_sph(dir_rot)
            
            # Calculate dphi
            dphi = pos_rot_phi - dir_rot_phi
            
            # Store in input buffer: (dist, energy, cos(pos_rot_th), cos(dir_rot_th), dphi)
            input_buffer[i] = torch.tensor([
                dist,
                energy,
                torch.cos(pos_rot_th),
                torch.cos(dir_rot_th),
                dphi
            ], device=self.device)
        
        return input_buffer
    
    def __call__(self, test_points=None, position=None, direction=None, 
                 energy=None, theta=None, phi=None, **kwargs):
        """
        Generate symbolic regression model output.
        
        Parameters:
        -----------
        nu_num : int
            Number of functions to generate (not used in this implementation)
        test_points : torch.Tensor (optional)
            Points to evaluate at (if provided, used as target positions)
        position : torch.Tensor or None
            Source position. If None, random position is generated.
        direction : torch.Tensor or None
            Direction vector (3D Cartesian). If None, random direction is generated.
            Takes precedence over theta/phi if both are provided.
        energy : float or torch.Tensor or None
            Energy value. If None, random energy is generated.
        theta : float or torch.Tensor or None
            Polar angle in radians (0 to π). Used if direction is None.
        phi : float or torch.Tensor or None
            Azimuthal angle in radians (0 to 2π). Used if direction is None.
            
        Returns:
        --------
        function or tuple
            Function that evaluates the symbolic regression model at given points,
            or tuple with function and test values if test_points provided
        """
        # Generate random parameters if not provided
        if position is None:
            position = torch.rand(3, device=self.device) * self.domain_size - self.half_domain
        else:
            position = torch.tensor(position, device=self.device) if not isinstance(position, torch.Tensor) else position
        
        # Handle direction specification - Cartesian takes precedence over spherical
        if direction is None:
            if theta is not None and phi is not None:
                # Convert spherical coordinates to Cartesian
                theta_val = torch.tensor(theta, device=self.device) if not isinstance(theta, torch.Tensor) else theta
                phi_val = torch.tensor(phi, device=self.device) if not isinstance(phi, torch.Tensor) else phi
                
                # Convert to Cartesian coordinates
                direction = torch.tensor([
                    torch.sin(theta_val) * torch.cos(phi_val),
                    torch.sin(theta_val) * torch.sin(phi_val),
                    torch.cos(theta_val)
                ], device=self.device)
            else:
                # Random direction on unit sphere
                direction = torch.randn(3, device=self.device)
                direction = direction / torch.norm(direction)
        else:
            # Use provided Cartesian direction
            direction = torch.tensor(direction, device=self.device) if not isinstance(direction, torch.Tensor) else direction
            direction = direction / torch.norm(direction)
        
        if energy is None:
            # Random energy from power law distribution
            energy = self.sample_power_law(E_min=0.1, E_max=10.0, gamma=2.7)
        else:
            energy = torch.tensor(energy, device=self.device) if not isinstance(energy, torch.Tensor) else energy
        
        # Create closure function
        def symbolic_reg_function(target_points):
            """
            Evaluate symbolic regression model at target points.
            
            Parameters:
            -----------
            target_points : torch.Tensor
                Target points to evaluate at, shape (N, 3)
                
            Returns:
            --------
            torch.Tensor
                Model output with shape (N, num_rotations)
            """
            if target_points.dim() == 1:
                target_points = target_points.unsqueeze(0)
            
            n_points = target_points.shape[0]
            output = torch.zeros(n_points, self.num_rotations, device=self.device)
            
            for i, target_point in enumerate(target_points):
                # Create model input for this target point
                model_input = self._create_model_input(position, direction, energy, target_point)
                
                # Calculate amplitude for each rotation
                for j in range(self.num_rotations):
                    dist, eng, pos_ct, dir_ct, dphi = model_input[j]
                    amplitude = self._sr_amplitude(dist, eng, pos_ct, dir_ct, dphi)
                    output[i, j] = amplitude
            
            return output
        
        if test_points is not None:
            test_output = symbolic_reg_function(test_points)
            return symbolic_reg_function, [test_output]
        else:
            return symbolic_reg_function