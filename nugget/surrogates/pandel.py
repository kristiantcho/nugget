import torch
from torch import tensor
from torch.special import gammainc, gammaln

# Try to import hyp1f1 from torch.special (may not be available in all versions)
try:
    from torch.special import hyp1f1
except ImportError:
    # Fallback: use scipy if available
    try:
        from scipy.special import hyp1f1 as scipy_hyp1f1
        def hyp1f1(a, b, z):
            """Wrapper to convert scipy hyp1f1 to torch tensor"""
            import numpy as np
            result = scipy_hyp1f1(a.cpu().numpy() if isinstance(a, torch.Tensor) else a,
                                  b.cpu().numpy() if isinstance(b, torch.Tensor) else b,
                                  z.cpu().numpy() if isinstance(z, torch.Tensor) else z)
            return torch.as_tensor(result, device=z.device if isinstance(z, torch.Tensor) else 'cpu')
    except ImportError:
        # If neither available, define a stub that will raise an error if used
        def hyp1f1(a, b, z):
            raise NotImplementedError("hyp1f1 not available. Install scipy or use PyTorch version with hyp1f1 support.")

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

        return xi * torch.log(tensor(self.rho)) - gammaln(xi) + (xi - 1) * torch.log(t) - t * self.rho

    def cdf(self, t, d):
        return gammainc(d / self.lambda_s, t * self.rho)

    def ppf(self, q, d):
        # Torch has no gammaincinv – must implement explicitly if needed
        raise NotImplementedError("Torch lacks gammaincinv, add custom solver if needed.")

    def rvs(self, d, size=None):
        # Gamma distribution
        d = torch.as_tensor(d)
        xi = d / self.lambda_s
        gamma_sample = torch.distributions.Gamma(xi, self.rho).sample(size)
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
        term2 = torch.sqrt(tensor(2.)) * eta * hyp1f1(0.5 * (xi + 1), 1.5, 0.5 * eta**2) / torch.exp(gammaln(0.5 * xi))

        pref = rho**xi * s**(xi - 1) * torch.exp(-t**2 / (2 * s**2)) / (2 ** ((1 + xi) / 2))
        return pref * (term1 - term2)

    def f2(self, xi, t, eta):
        return torch.exp(self.rho**2 * self.s**2 / 2) * self.pandel.pdf(t, xi * self.lambda_s)

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
            - xi/2 * torch.log(tensor(2.0))
            + (xi - 1)/2 * torch.log(2*xi - 1)
            + xi * torch.log(tensor(self.rho))
            + (xi - 1) * torch.log(tensor(self.s))
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
            / torch.sqrt(tensor(2*torch.pi))
            * U
            * torch.exp(-k * (2*xi - 1))
            * (1 + z**2)**(-0.25)
            * Psi
        )

    def f5(self, xi, t, eta):
        return (
            (self.rho * self.s)**xi
            / torch.sqrt(tensor(2 * torch.pi * self.s**2))
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
