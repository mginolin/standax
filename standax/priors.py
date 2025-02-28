from jax.scipy import stats
import jax.numpy as jnp

class Prior():
    
    def __init__(self, which, loc, scale, **kwargs):
        """ """
        self._module = which
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)
        self._kwargs = kwargs
        
    # ============ #
    #    Priors    # 
    # ============ #    
    @classmethod
    def as_normal(cls, loc, scale):
        """ """
        return cls(stats.norm, loc, scale)

    @classmethod
    def as_uniform(cls, min, max):
        """ """
        return cls(stats.uniform, loc=min, scale=max-min)
        
    # ============ #
    #    Method    # 
    # ============ #
    def logpdf(self, x):
        return self._module.logpdf(x, loc=self.loc, scale=self.scale, **self._kwargs)

    def pdf(self, x):
        return self._module.pdf(x, loc=self.loc, scale=self.scale, **self._kwargs)

    def logcdf(self, x):
        return self._module.logcdf(x, loc=self.loc, scale=self.scale, **self._kwargs)

    def cdf(self, x):
        return self._module.cdf(x, loc=self.loc, scale=self.scale, **self._kwargs)
