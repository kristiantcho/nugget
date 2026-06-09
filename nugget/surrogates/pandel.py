import torch
# from scipy.special import  gammainc, gammaincinv, gamma, gammaln, hyp1f1
from torch.special import gammaln, gammainc
import numpy as np
import scipy

class Hyp1f1Function(torch.autograd.Function):
    """
    Differentiable wrapper for confluent hypergeometric function hyp1f1.
    Uses the derivative formula: d/dz hyp1f1(a, b, z) = (a/b) * hyp1f1(a+1, b+1, z)

    Uses the new-style setup_context API so that functorch transforms
    (jacrev, vmap, grad, jvp, linearize, ...) work correctly.
    """

    @staticmethod
    def forward(a, b, z):
        a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
        b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
        z_np = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z
        result = scipy.special.hyp1f1(a_np, b_np, z_np)
        return torch.as_tensor(result, dtype=z.dtype, device=z.device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, z = inputs
        ctx.save_for_backward(a, b, z)
        # Also store as plain attributes for the jvp hook — save_for_backward
        # tensors are unavailable during forward-mode AD (torch.func.jvp) because
        # the inputs are dual tensors at that point and cannot be saved via the
        # standard mechanism.
        ctx._a = a
        ctx._b = b
        ctx._z = z

    @staticmethod
    def backward(ctx, grad_output):
        a, b, z = ctx.saved_tensors
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        deriv_z = (a_np / b_np) * scipy.special.hyp1f1(a_np + 1, b_np + 1, z_np)
        deriv_z = torch.as_tensor(deriv_z, dtype=z.dtype, device=z.device)
        return None, None, grad_output * deriv_z

    @staticmethod
    def jvp(ctx, tangents_a, tangents_b, tangents_z):
        a = ctx._a
        b = ctx._b
        z = ctx._z
        if tangents_z is None:
            return None, None, torch.zeros_like(z)
        h = Hyp1f1Function.apply(a + 1.0, b + 1.0, z)
        return None, None, tangents_z * (a / b) * h

def hyp1f1(a, b, z):
    """
    Differentiable confluent hypergeometric function (Kummer's function).
    
    Args:
        a: first parameter
        b: second parameter  
        z: variable (will support gradients)
    
    Returns:
        hyp1f1(a, b, z) as a torch tensor with gradient support
    """
    # Convert inputs to tensors if needed
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=torch.float32)
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z, dtype=torch.float32)
    
    return Hyp1f1Function.apply(a, b, z)

# ---------------------------------------------------------------------
#  PANDEL
# ---------------------------------------------------------------------

class Pandel():
    def __init__(self, tau=557., lambda_s=33.3, lambda_a=98., v=0.3/1.3):
        self.tau = float(tau)
        self.lambda_s = float(lambda_s)
        self.lambda_a = float(lambda_a)
        self.v = float(v)

        # constant parameter
        self.rho = self.v / self.lambda_a + 1.0 / self.tau

    def pdf(self, t, d):
        t, d = torch.as_tensor(t), torch.as_tensor(d)
        xi = d / self.lambda_s

        return (self.rho ** xi / torch.exp(gammaln(xi))) * (t ** (xi - 1)) * torch.exp(-t * self.rho)

    def logpdf(self, t, d):
        t, d = torch.as_tensor(t), torch.as_tensor(d)
        xi = d / self.lambda_s

        return xi * torch.log(torch.tensor(self.rho)) - gammaln(xi) + (xi - 1) * torch.log(t) - t * self.rho

    def cdf(self, t, d):
        return gammainc(d / self.lambda_s, t * self.rho)

    def ppf(self, q, d):
        # Torch has no gammaincinv – must implement explicitly if needed
        raise NotImplementedError("Torch lacks gammaincinv, add custom solver if needed.")

    def rvs(self, d, size=None):
        # Gamma distribution
        d = torch.as_tensor(d)
        xi = d / self.lambda_s
        
        # Convert size to tuple format expected by PyTorch
        if size is None:
            sample_shape = torch.Size()
        elif isinstance(size, int):
            sample_shape = torch.Size([size])
        else:
            sample_shape = torch.Size(size) if not isinstance(size, torch.Size) else size
            
        gamma_sample = torch.distributions.Gamma(xi, self.rho).sample(sample_shape)
        return gamma_sample


# ---------------------------------------------------------------------
#  CPANDEL
# ---------------------------------------------------------------------

class CPandel():
    def __init__(self, tau=557., lambda_s=33.3, lambda_a=98., v=0.3/1.3, s=5.0):
        self.lambda_s = float(lambda_s)
        self.lambda_a = float(lambda_a)
        self.tau = float(tau)
        self.v = float(v)
        self.s = float(s)

        self.rho = self.v / self.lambda_a + 1.0 / self.tau
        self.pandel = Pandel(tau=tau, lambda_s=lambda_s, lambda_a=lambda_a, v=v)

    # Helper functions ------------------------------------------------

    @staticmethod
    def k(z):
        return 0.5 * (z * torch.sqrt(1 + z**2) + torch.log(z + torch.sqrt(1 + z**2)))

    @staticmethod
    def beta(z):
        return 0.5 * (z / torch.sqrt(1 + z**2) - 1)

    @staticmethod
    def N_1(beta):
        return beta / 12 * (20 * beta**2 + 30 * beta + 9)

    @staticmethod
    def N_2(beta):
        return beta**2 / 288 * (6160 * beta**4 + 18480 * beta**3 + 
                               19404 * beta**2 + 8028 * beta + 945)

    # Approximation branches -----------------------------------------

    def f1(self, xi, t, eta):
        # fully analytic region
        rho, s = self.rho, self.s
        term1 = hyp1f1(0.5 * xi, 0.5, 0.5 * eta**2) / torch.exp(gammaln(0.5 * (xi + 1)))
        term2 = torch.sqrt(torch.tensor(2.)) * eta * hyp1f1(0.5 * (xi + 1), 1.5, 0.5 * eta**2) / torch.exp(gammaln(0.5 * xi))

        pref = rho**xi * s**(xi - 1) * torch.exp(-t**2 / (2 * s**2)) / (2 ** ((1 + xi) / 2))
        return pref * (term1 - term2)

    def f2(self, xi, t, eta):
        return np.exp(self.rho**2 * self.s**2 / 2) * self.pandel.pdf(t, xi * self.lambda_s)

    def f3(self, xi, t, eta):
        z = -eta / torch.sqrt(4 * xi - 2)
        k = self.k(z)

        beta = self.beta(z)
        N1 = self.N_1(beta)
        N2 = self.N_2(beta)

        Phi = 1 - N1 / (2 * xi - 1) + N2 / (2 * xi - 1)**2

        alpha = (
            -t**2 / (2 * self.s**2)
            + eta**2 / 4
            - xi/2 + 1/4
            + k * (2*xi - 1)
            - 0.25 * torch.log(1 + z**2)
            - xi/2 * torch.log(torch.tensor(2.0))
            + (xi - 1)/2 * torch.log(2*xi - 1)
            + xi * torch.log(torch.tensor(self.rho))
            + (xi - 1) * torch.log(torch.tensor(self.s))
        )

        return torch.exp(alpha) / torch.exp(gammaln(xi)) * Phi

    def f4(self, xi, t, eta):
        z = eta / torch.sqrt(4 * xi - 2)
        k = self.k(z)
        beta = self.beta(z)

        N1 = self.N_1(beta)
        N2 = self.N_2(beta)

        U = torch.exp(xi/2 - 0.25) * (2 * xi - 1)**(-xi/2) * 2**((xi - 1)/2)
        Psi = 1 + N1/(2*xi - 1) + N2/(2*xi - 1)**2

        return (
            self.rho**xi
            * self.s**(xi - 1)
            * torch.exp(-t**2 / (2*self.s**2) + eta**2/4)
            / torch.sqrt(torch.tensor(2*torch.pi))
            * U
            * torch.exp(-k * (2*xi - 1))
            * (1 + z**2)**(-0.25)
            * Psi
        )

    def f5(self, xi, t, eta):
        return (
            (self.rho * self.s)**xi
            / torch.sqrt(torch.tensor(2 * torch.pi * self.s**2))
            * eta**(-xi)
            * torch.exp(-t**2 / (2 * self.s**2))
        )

    # ------------------------------------------------------------------
    # Full CPandel PDF with region masks (same logic as SciPy version)
    # ------------------------------------------------------------------

    def pdf(self, t, d):
        t = torch.as_tensor(t, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32)

        xi = d / self.lambda_s
        eta = self.rho * self.s - t / self.s

        # Boolean masks
        inner = (t > -5 * self.s) & (t < 30 * self.s) & (xi < 5 * self.s)
        left  = (t < self.rho * self.s**2)
        lower = (xi < 1)

        # Pick branch
        # (same logic as SciPy code)
        f = torch.zeros_like(t)

        f = torch.where(inner,               self.f1(xi, t, eta), f)
        f = torch.where(~inner & lower & ~left,  self.f2(xi, t, eta), f)
        f = torch.where(~inner & ~lower & ~left, self.f3(xi, t, eta), f)
        f = torch.where(~inner & ~lower & left,   self.f4(xi, t, eta), f)
        f = torch.where(~inner & lower & left,    self.f5(xi, t, eta), f)

        return f

    def rvs(self, d, size=None):
        # Pandel + Gaussian jitter
        base = self.pandel.rvs(d, size=size)
        noise = torch.randn_like(base) * self.s
        return base + noise
