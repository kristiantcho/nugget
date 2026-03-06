from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.special import  gammainc, gammaincinv, gamma, gammaln, hyp1f1
import numpy as np
import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# code taken and adapted from https://github.com/philippeller/freeDOM

class pandel_gen(rv_continuous):
    """Pandel distribution
    
    Distribution Parameters:
    -----------
    
    t : variable of distribution (time)
    d : distance parameter (length)
    
    Instantiation parameters (optional):
    ------------------------------------

    tau : tau parameter (time)
    lambda_s : scattering length (length)
    lambda_a : absorption length (length)
    v : velocity (length/time)
    
    """
    
    def __init__(self, tau=557, lambda_s=33.3, lambda_a=98, v=0.3/1.3, **kwargs):
        super().__init__(**kwargs)
        self.lambda_s = lambda_s
        self.rho = v/lambda_a + 1/tau
    
    def _pdf(self, t, d):
        xi = d/self.lambda_s
        return self.rho**xi/gamma(xi) * np.power(t, xi-1) * np.exp(-t*self.rho)
    
    def _logpdf(self, t, d):
        xi = d/self.lambda_s
        return xi*np.log(self.rho) - gammaln(xi) + (xi-1)*np.log(t) - t*self.rho
    
    def _cdf(self, t, d):
        xi = d/self.lambda_s
        return gammainc(xi, t*self.rho)
    
    def _ppf(self, q, d):
        xi = d/self.lambda_s
        return gammaincinv(xi, q)/self.rho
    
    def freeze(self, *args, **kwds):
        frozen = rv_frozen(self, *args, **kwds)
        frozen.dist.lambda_s = self.lambda_s
        frozen.dist.rho = self.rho
        return frozen
    
pandel = pandel_gen(a=0.0, name='pandel')

class cpandel_gen(rv_continuous):
    """CPandel distribution
    
    Distribution Parameters:
    -----------
    
    t : variable of distribution (time)
    d : distance parameter (length)
    
    Instantiation parameters (optional):
    ------------------------------------

    tau : tau parameter (time)
    lambda_s : scattering length (length)
    lambda_a : absorption length (length)
    v : velocity (length/time)
    s : jitter (gauss covolution width)
    
    Notes:
    ------
    This implemets the function discussed in https://arxiv.org/pdf/0704.1706.pdf
    
    """
    
    def __init__(self, tau=557, lambda_s=33.3, lambda_a=98, v=0.3/1.3, s=5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_s = lambda_s
        self.rho = v/lambda_a + 1/tau
        self.s = s
        self.pandel = pandel_gen(a=0.0, tau=tau, lambda_a=lambda_a, lambda_s=lambda_s, v=v, name='pandel')
    
    def _pdf(self, t, d):
               
        xi = d/self.lambda_s
        eta = self.rho*self.s - t/self.s

        # regions:
        inner  = (t > -5 * self.s) & (t < 30 * self.s) & (xi < 5 * self.s)
        left = (t < self.rho * self.s**2)
        lower = (xi < 1)
        
        p = np.zeros_like(t)
        
        masks = []
        masks.append(inner)
        masks.append(lower & ~left & ~inner)
        masks.append(~lower & ~left & ~inner)
        masks.append(~lower & left & ~inner)
        masks.append(lower & left & ~inner)

        for mask, f in zip(masks, [self.f1, self.f2, self.f3, self.f4, self.f5]):
            
            t_array = len(t) > 1
            d_array = len(d) > 1
            
            if np.any(mask): p[mask] = f(xi[mask] if d_array else xi, t[mask] if t_array else t, eta[mask] if t_array else eta)

        return p
        
    @staticmethod
    def k(z):
        return 0.5 * (z * np.sqrt(1 + z**2) + np.log(z + np.sqrt(1 + z**2)))
    
    @staticmethod
    def beta(z):
        return 0.5 * (z / (np.sqrt(1 + z**2)) - 1)
    
    @staticmethod
    def N_1(beta):
        return beta / 12 * (20 * beta**2 + 30 * beta + 9)
    
    @staticmethod
    def N_2(beta):
        return beta**2 / 288 * (6160 * beta**4 + 18480 * beta**3 + 19404 * beta**2 + 8028 * beta + 945)

    def f1(self, xi, t, eta):
        ''' No approximation '''
        return self.rho**xi * self.s**(xi - 1) * np.exp(-t**2 / (2 * self.s**2)) / 2**((1 + xi) / 2) * (
                                        hyp1f1(0.5 * xi, 0.5, 0.5 * eta**2) / gamma(0.5 * (xi + 1)) 
                                        - np.sqrt(2) * eta * hyp1f1(0.5 * (xi + 1), 1.5, 0.5 * eta**2) / gamma(0.5 * xi)
                                        ) 
    
    def f2(self, xi, t, eta):
        return np.exp(self.rho**2 * self.s**2 / 2)*self.pandel.pdf(t, xi*self.lambda_s) 
    
    def f3(self, xi, t, eta):
        z = - eta / np.sqrt(4 * xi - 2)
        k = self.k(z)
        
        alpha = -t**2 / (2 * self.s**2) + eta**2 / 4 - xi/2 + 1/4 + k * (2 * xi - 1) - 1/4 * np.log(1 + z**2) - xi/2 * np.log(2) + (xi - 1) / 2 * np.log(2 * xi - 1) + xi * np.log(self.rho) + (xi - 1) * np.log(self.s)
        
        beta = self.beta(z)
        N_1 = self.N_1(beta)
        N_2 = self.N_1(beta)
        
        Phi = 1 - N_1 / (2 * xi - 1) + N_2 / (2 * xi - 1)**2
        
        return np.exp(alpha) / gamma(xi) * Phi
    
    def f4(self, xi, t, eta):
        z = eta / np.sqrt(4 * xi - 2)
        k = self.k(z)
        
        beta = self.beta(z)
        N_1 = self.N_1(beta)
        N_2 = self.N_1(beta)
        
        U = np.exp(xi/2 - 1/4) * (2 * xi - 1)**(-xi/2) * 2**((xi - 1)/2)
        
        Psi = 1 + N_1 / (2 * xi - 1) + N_2 / (2 * xi - 1)**2
        
        return self.rho**xi * self.s**(xi - 1) * np.exp(-t**2 / (2 * self.s**2) + eta**2 / 4) / np.sqrt(2 * np.pi) * U * np.exp(-k * (2 * xi -1)) * (1 + z**2)**-0.25 * Psi        

    def f5(self, xi, t, eta):
        return (self.rho * self.s)**xi / np.sqrt(2 * np.pi * self.s**2) * eta**-xi * np.exp(-t**2 / (2 * self.s**2))
    
    def _rvs(self, d, size=None, random_state=None):
        return self.pandel.rvs(d, size=size, random_state=random_state) + random_state.standard_normal(size) * self.s
    
    
    def freeze(self, *args, **kwds):
        frozen = rv_frozen(self, *args, **kwds)
        frozen.dist.lambda_s = self.lambda_s
        frozen.dist.rho = self.rho
        frozen.dist.s = self.s
        frozen.dist.pandel = self.pandel
        return frozen
    
cpandel = cpandel_gen(name='cpandel')