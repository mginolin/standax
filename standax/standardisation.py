""" This is a simple module to standardized SNe~Ia without fitting cosmology """

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.scipy import stats
import warnings
from .covariance import CovMatrix

import jax.numpy as jnp


__all__ = ["standardise_snia", "Standardize"]


def standardise_snia(data, init=None, sigmaint_guess=0.1,
                        model="linear", xkeys=["x1","c"], ykey="mag",
                        nfetch=3, num_samples=500, verbose=False,
                        priors={}, **kwargs):
    """ top level standardisation function """
    
    this = Standardize.from_data(data, xkeys=xkeys, ykey=ykey,
                                     model=model, priors=priors)
    
    (best_params, sigmas, loss), mcmc = this.fit(init,
                                                  sigmaint = sigmaint_guess,
                                                  nfetch=nfetch,
                                                  num_samples=num_samples,
                                                  verbose=verbose,
                                                  **kwargs)
    
    return (best_params, sigmas, mcmc), this

def get_modelclass(which):
    """ """
    # - Input parser
    if type(which) is not str: # nothing to do.
        return which
        
    # - Models
    if which.lower() in ["linear"]:
        return LinearModel
        
    if which.lower() in ["brokenlinear", "brokenalpha", "brokenline"]:
        return BrokenLinearModel

    # - Failed
    raise NotImplementedError(f"{which=} standardisation model is not implemented")    

# =================== #
#                     #
#   Standardation     #
#                     #
# =================== #
class Standardize( object ):

    def __init__(self, y_obs, x_obs, cov,
                     model="linear", colnames=None,
                     index=None, priors={}):
        """ """
        self.y_obs = y_obs
        self.x_obs = x_obs
        self.cov = cov

        n_xobs = len(self.x_obs)
        n_points = len(self.y_obs)
        if colnames is None:
            colnames = ["y"] + [f"x{i}" for i in range(n_xobs)]
        elif len(colnames) != n_xobs+1:
            raise ValueError("input names do not match the number of observations")

        self._colnames = colnames
        if index is None:
            self._index = jnp.arange(n_points)
        else:
            self._index = index
            
        # modeling
        self.set_model(model, priors=priors)
        
    @classmethod
    def from_data(cls, data, ykey="mag", xkeys=["x1", "c"], 
                    model="linear", priors={},
                    **kwargs):
        """ """
        import itertools
        import numpy as np # jnp does not accept the str
        
        y_obs = jnp.asarray(data[ykey])
        x_obs = jnp.asarray(data[xkeys].values.T)
        cov = CovMatrix.from_data(data, ykey=ykey, xkeys=xkeys, **kwargs)
        colnames = list(np.hstack([np.atleast_1d(ykey), np.atleast_1d(xkeys)]))
        
        return cls(y_obs=y_obs, x_obs=x_obs, cov=cov, model=model,
                       colnames=colnames, index=data.index, priors=priors)

    # =============== #
    #   Method        #
    # =============== #
    def set_model(self, model, priors={}):
        """ """
        modelclass = get_modelclass(model) 
        self.model = modelclass.from_observations(self.y_obs, self.x_obs, 
                                                  cov=self.cov,
                                                  priors=priors)
    
    def get_data(self, as_dataframe=False):
        """ get the data entring the fit method with the correct format. """           
        
        if as_dataframe:
            import pandas
            datain = jnp.vstack([self.y_obs, self.x_obs, self.cov.y_err, self.cov.x_err])
            data_ = pandas.DataFrame(datain.T,
                                     columns=self._colnames + [f"{k}_err" for k in self._colnames],
                                     index=self._index)
        else:
            data_ = self.model.data

        return data_
    
    def fit(self, init=None, sigmaint=0.1, nfetch=1,
            fit_method="tncg", num_samples=500,
            verbose=False,
            **kwargs):
        """ fit the data
        
        * current implementation do not allow to fit for sigmaint *

        Parameters
        ----------
        init_coef: array
            initial guess for the fitted coefficient (M,)

        func: function
            function to minimize. Should return a float
            if None give, get_total_covchi2 is used and a warning it generated
           
        sigmaint: float, array, None
            intrinsic scatter fixed, or to fit:
            - None, means the sigmaint is fitted
            - float or array (N,) the scatter is fixed to the input value

        nfetch: int
            number of fetch_sigmaint loop. 0 means input sigmaint used.

        fit_method: func, str
            function used to minimize. 
            If string given it is assumed to be .fitter.fit_{fit_method}

        **kwargs fit_method options (e.g., niter, tol etc)

        Returns
        -------
        (params, loss)
            - params: dict, best fitted parameters
            - loss: array, loss history

        """
        from . import fitter

        # --------- #
        #   guess   #
        # --------- #
        guess = self.model.get_guess(init=init)
        if sigmaint is None:
            guess["sigmaint"] = 0.1
        else:
            kwargs |= dict(sigmaint=sigmaint)
            
        # --------- #
        #   fit     #
        # --------- #     
        if type(fit_method) is str:
            fit_method = getattr(fitter, f"fit_{fit_method}")

        if verbose:
            print(f"fitting using {fit_method} using guess: {guess}")
            print(f" -> fitting options {kwargs}")
            
            
        # sigmaint ignored if in init
        best_params, loss = fit_method(self.model.get_likelihood, guess, 
                                       **kwargs)
        
        sigmaint = best_params.pop("sigmaint", sigmaint) # gets either fitted or input
        sigmas = [sigmaint]
        if nfetch>0:
            for i in range(nfetch):
                sigmaint = self.model.fetch_sigmaint(params=best_params)
                kwargs["sigmaint"] = sigmaint
                if verbose:
                    print(f"reaching sigmaint: {sigmaint}")
                    
                best_params, loss = fit_method(self.model.get_likelihood, 
                                               best_params, # guess is former best param
                                               **kwargs) # includes new sigmaint
                if verbose:
                    print(f"leading to parameters: {best_params}")
                    
                sigmas.append(sigmaint)
                # retart nloops times.
            
        # --------- #
        #   cov     #
        # --------- #
        if num_samples>0:
            if verbose:
                print(f"running mcmc now with {num_samples} sampler (end 10% warmup)")

            if nfetch>=1:
                sigmaint_mcmc = jnp.hstack(sigmas)[1:].mean() # mean + some burning
            else:
                sigmaint_mcmc = sigmas[-1]
                
            if verbose:
                print(f"using sigmaint = {sigmaint:.2f} for mcmc")
                
            key = jax.random.PRNGKey( int(jnp.abs(loss[-1])*1000 ) )
            mcmc = self.model.nuts_sampling(key, params=best_params, sigmaint=sigmaint,
                                        num_warmup=int(0.1*num_samples),
                                        num_samples=num_samples,
                                        )
        else:
            mcmc = None
        #except:
        #    warnings.warn("mcmc sampling failed")
        #    mcmc = None
        
        # --------- #
        #  output   #
        # --------- #        
        return (best_params, sigmas, loss), mcmc

    def get_residuals(self, params, null_coefs=None, sigmaint=0,
                          as_dataframe=True, x_obs=True):
        """ """
        param_ = params.copy()
        # switch off coefficients
        if null_coefs is not None:
            coefs = param_["coefs"].copy()
            for i in jnp.atleast_1d(null_coefs):
                coefs = coefs.at[i].set(0)
            param_["coefs"] = coefs

        
        x_used = self.x_obs if x_obs else param_["x_model"]
        # Hubble Residuals
        
        y_model = self.model.get_ymodel(param_, x=x_used)
        hr = self.y_obs - y_model

        # Errors
        sigma_x_to_y = self.model.xvar_to_yvar(param_, self.cov) # variance
        sigma_tot = (self.cov.y_err**2 + sigmaint**2 + sigma_x_to_y)
        hr_err = jnp.sqrt(sigma_tot)

        if as_dataframe:
            import pandas
            return pandas.DataFrame({"magres": hr, "magres_err": hr_err},
                                    index=self._index)
        
        # output
        return hr, hr_err

    def show_residuals(self, params, xindex, data_x=None, null_coefs=None, sigmaint=0, ax=None,
                      facecolors="0.8", edgecolors="0.6", ecolor="0.8", index=None,
                       elw=1, lw=0.5, zorder=2,  x_obs=True, binms=8,
                       show_data=True,
                       bins=None, bincolor="navy", binlabel=None,
                       kde_levels=None, kdecolor="navy", kdelw=1, kdealpha=0.3,
                       
                       **kwargs):
        """ """
        import pandas
        import numpy as np

        def nmad(*args, **kwargs):
            """ stats.median_abs_deviation forcing scale='normal' """
            from scipy import stats
            return stats.median_abs_deviation(*args, scale="normal", **kwargs)
        
        df = self.get_data(as_dataframe=True)
        if type(xindex) is int:
            colname = df.columns[xindex]
        else:
            colname = xindex

        # data
        if data_x is None:
            data_x = df[[colname, colname+"_err"]]
            
        data_y = self.get_residuals(params, null_coefs=null_coefs, sigmaint=sigmaint,
                                        x_obs=x_obs,
                                        as_dataframe=True)
        data_ = data_y.join(data_x)
        if index is not None:
            data_ = data_.loc[index]
    
        # figure
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            
        # plot
        if show_data:
            ax.errorbar(data_[colname], data_["magres"], 
                        xerr=data_[f"{colname}_err"],
                        yerr=data_["magres_err"], 
                        ls="None", marker="None", 
                        ecolor=ecolor, lw=elw, zorder=zorder)
            
            ax.scatter(data_[colname], data_["magres"], zorder=zorder, 
                        facecolors=facecolors, edgecolors=edgecolors, 
                        lw=lw, **kwargs
                       )
        # binning ?
        if bins is not None:
            data_["xbin"] = pandas.cut(data_[colname], bins)
            gbin = data_.groupby("xbin")["magres"].agg(["median", nmad, "size"])
            gbin_mid = [i.mid for i in gbin.index]
            ax.errorbar(gbin_mid, gbin["median"], yerr=gbin["nmad"]/np.sqrt(gbin["size"]-1),
                       ls="None", marker="o", color=bincolor, zorder=zorder+4, ms=binms,
                            label=binlabel)
        
        if kde_levels is not None:
            import seaborn as sns
            sns.kdeplot(ax=ax, x=data_[colname], y=data_["magres"], #weights=1/ztfdata["localcolor"]**2,
                    levels=kde_levels, color=kdecolor, bw_adjust=1., zorder=zorder+3,
                    linewidths=kdelw, linestyles="-", alpha=kdealpha)
    
        return fig

# =================== #
#                     #
#     Models          #
#                     #
# =================== #
class StandardizationModel( object ):

    _REQUESTED_PRIORS = None
    
    def __init__(self, data, priors={}):
        """ """
        self._data = data
        self._priors = self._parse_priors(priors)
        self._prior_names = list(self._priors.keys())
        
    @classmethod
    def from_observations(cls, y_obs, x_obs, cov=None,
                              priors={}):
        """ """
        data = {"y_obs": y_obs,
                "x_obs": x_obs,
                "cov": cov}
        return cls(data, priors=priors)

    @classmethod
    def _parse_priors(cls, priors, default_range=[-1e3, 1e3]):
        """ """
        from .priors import Prior
        requested_key = cls._REQUESTED_PRIORS
        if requested_key is not None:
            backup = {k: Prior.as_uniform(*default_range) 
                            for k in cls._REQUESTED_PRIORS if k not in priors
                      }
        else:
            backup = {}

        full_priors = priors | backup
        return full_priors
    
    # ------------- #
    #   NEEDED      #
    # ------------- #   
    @staticmethod
    def get_ymodel(params, x):
        """ creates y from x given the input param (includes offset if any) """
        raise NotImplementedError("defined the model connecting x to y.")

    def get_guess(self, *args):
        """ """
        raise NotImplementedError(" you need to implemented your guess for this model ")

    @staticmethod
    def xvar_to_yvar(params, cov):
        """ """
        raise NotImplementedError(" you need to implemented the xvar to yvar convertion")

    # ------------- #
    #   Method      #
    # ------------- #
    def get_likelihood(self, params, sigmaint=0, **kwargs):
        """ """
        chi2, logdet = self.get_chi2_logdet(params, 
                                            data=self.data, 
                                            sigmaint=sigmaint, 
                                            **kwargs)
        # priors actually is -sum(logpdf)
        priors = self.get_priors(params).astype(float)
        
        return chi2 + logdet + priors

    def get_priors(self, params):
        """ """
        all_sums = tree_util.tree_map( lambda k: self.priors[k].logpdf(params[k]).sum(),
                                        self._prior_names
                                     )
        return -jnp.asarray(all_sums).sum()
        
    def fetch_sigmaint(self, params, guess=0.1):
        """ change the model (get_chi2) to reach a chi2/dof of 1.
        
        The model will be the basic chi2: y_model = x_obs*coefs + offset
        
        Parameters
        ----------
        param: dict, pytree
            fitted parameters
            - coefs (M,)
            - offset ()
        
        guess: float
            initial guess for sigmaint

        Returns
        -------
        sigmaint
            float
        """
        from .fitter import fit_adam        
        def _to_minimize_(sigmaint):
            # get_chi2(self, params, sigmaint=0):
            chi2 = self.get_chi2(params, sigmaint=sigmaint)
            return jnp.abs( chi2/self.ndof - 1 ) # get chi2/dof = 1

        guess = jnp.asarray( guess )
        best, loss = fit_adam(_to_minimize_, guess)
        return best

    def get_covmatrix(self, params, **kwargs):
        """ """
        # get the flatten shapes
        flatshapes = {k: v.flatten().shape[0] for k,v in params.items()}
        
        # compute the pytree hessian
        hess = jax.hessian(self.get_likelihood)(params, **kwargs)
        
        # create the 2d blocks
        hess_blocks = []
        for k1 in flatshapes.keys():
            # or hstack with reshape(k1, k2) ?
            block_ = jnp.vstack([hess[k1][k2].reshape(flatshapes[k2],flatshapes[k1])
                                 for k2 in flatshapes.keys()])
            hess_blocks.append(block_)
    
        # creates the 2d per-block hessian
        hess_2d = jnp.block(hess_blocks)
        # and finally get the covariance matrix
        return 2*jnp.linalg.inv(hess_2d)        
        
    @classmethod
    def get_chi2_logdet(cls, params, data, sigmaint=0):
        """ function that estimate the total chi2 for the model 
        
        Parameters
        ----------
        param: dict, pytree
            fitted parameters
            - coefs (M,)
            - offset ()
            - x_model (M, N)
            (- sigmaint )
            
        data: dict, pytree
            data to fit
            - x_obs (M, N)
            - y_obs (N,)
            - cov (covariance.CovMatrix)
    
        sigmaint: float
            default sigmaint it not included in param
            = ignored if param['sigmaint'] exists =
            
        Returns
        -------
        (chi2, logdet)
    
        """
        x_model = params["x_model"]
        y_model = cls.get_ymodel(params, x_model)
        
        delta_y = data["y_obs"] - y_model # (N,)
        delta_x = data["x_obs"] - x_model # (M, N)
    
        cov = data["cov"] # edcovariance.C
        sigmaint = params.get("sigmaint", sigmaint)
    
        # chi2 and logdet
        chi2, logdet = cov.get_chi2_logdet(delta_y, delta_x, sigmaint=sigmaint)
        
        return chi2, logdet

    def get_chi2(self, params, sigmaint=0):
        """ """
        chi2 = self._get_diag_chi2_(self.y_obs, x=self.x_obs, cov=self.cov,
                                   params=params, sigmaint=sigmaint)
        return chi2

    def nuts_sampling(self, rng_key, params, sigmaint=0, 
                      num_warmup=100, num_samples=1_000, 
                      prop_nuts={}, progress_bar=False, **kwargs):
        """ """
        from numpyro.infer import MCMC, NUTS
        
        def potential_fn(param_):
            # do not fit for x_model, creates biases...
            param_["x_model"] = params["x_model"]
            # coefs seems ok, tested ok simulation, resulting scatter ok with pull
            return 0.25 * self.get_likelihood(param_, sigmaint=sigmaint, **kwargs)
    
        # setting up
        sampler = NUTS(potential_fn=potential_fn, **prop_nuts)
        
        mcmc = MCMC(sampler, num_warmup=num_warmup, num_samples=num_samples, 
                    progress_bar=progress_bar)
    
        # init removes x_model if any
        init = params.copy()
        _ = init.pop("x_model", None)
        mcmc.run(rng_key, init_params=init)
        
        return mcmc
    
    # ------------- #
    #   Internal    #
    # ------------- #        
    @staticmethod
    def _test_data_(data):
        """ """
        if "y_obs" not in data or "x_obs" not in data:
            raise ValueError("x_obs or y_obs not in input data")
        if "cov" not in data:
            warnings.warn("No covariance in data")

    @classmethod
    def _get_diag_chi2_(cls, y_obs, x, cov, params, sigmaint=0):
        """ get the 'basic' chi2 with assuming modelfunc is linear (or linear combinations).
    
        Parameters
        ----------
        y_obs: array
            y values (N,)
    
        x: array
            x values (M, N)
    
        coefs: array
            amplitude (M,) of the individual x such that
            
            
        cov: covariance.CovMatrix
            covariance matrix
    
        offset: float, array
            offset, if array (N,)
    
        sigmaint: float array
            sigmaint, if array (N,)
        
        Returns
        -------
        float
            chi2
        """
        # diagonal only so far
        y_err, x_err = cov.y_err, cov.x_err
            
        # y model
        y_model = cls.get_ymodel(params, x)
    
        # chi2 using x_err to get y_err given the coefs.        
        sigma_x_to_y = cls.xvar_to_yvar(params, cov) # variance
        
        # total 1d pull
        y_res = (y_obs - y_model)**2 / (y_err**2 + sigmaint**2 + sigma_x_to_y)
        return jnp.sum(y_res) # chi2

    # ------------- #
    #  Properties   #
    # ------------- #
    @property
    def data(self):
        """ """
        return self._data

    @property
    def priors(self):
        """ """
        return self._priors
        
    @property
    def y_obs(self):
        """ """
        return self.data["y_obs"]
        
    @property        
    def x_obs(self):
        """ """
        return self.data["x_obs"]
        
    @property        
    def cov(self):
        """ """
        return self.data["cov"]
        
    @property
    def npoints(self):
        """ number of y datapoint (N) """
        return len(self.y_obs)
        
    @property
    def ncoefs(self):
        """ number of standardisation parameter (M) """
        return len(self.x_obs)

    @property        
    def ndof(self):
        """ """
        return self.npoints - (self.ncoefs + 1) # +1 for offset
        
# ================ #
#                  #
#   Linear Model   #
#                  #
# ================ #
class LinearModel( StandardizationModel ):
    #   NEEDED      #
    #def get_ymodel(params, x):
    #  Done
    #def get_guess(self, *args):
    #  Done
    #def xvar_to_yvar(params, cov):
    #  Done
    
    @staticmethod
    def get_ymodel(params, x):
        """ inputs params and output y_model, x_model """
        y_model = jnp.matmul(params["coefs"], x)
        return y_model + params["offset"]

    def get_guess(self, init=None):
        """ """
        # default
        guess = {"offset": jnp.asarray(0., dtype="float32"),
                "x_model": jnp.asarray(self.x_obs, dtype="float32").copy() ,
                "coefs": jnp.ones((self.ncoefs,), dtype="float32")}
        
        # to fit
        if init is not None:
            if type(init) is not dict: # you input coefs init
                # assuming init is a array for coefs
                guess |= {"coefs": jnp.asarray(init, dtype="float32") }
            else: # you input the whole input
                guess |= init
                
        return guess

    #@staticmethod
    #def xvar_to_yvar(params, cov):
    #    """ """
    #    y_var = jnp.dot(params["coefs"], cov.x_err**2 * params["coefs"][:,None])
    #    return y_var
        
    @classmethod
    def xvar_to_yvar(cls, params, cov):
        """ """
        p = {"coefs": params["coefs"]**2,
             "offset": 0}

        yvar = cls.get_ymodel(p, x=cov.x_err**2)
        return yvar

# ================ #
#                  #
#  Broken Linear   #
#                  #
# ================ #
class BrokenLinearModel( LinearModel ):
    #   NEEDED      #
    #def get_ymodel(params, x):
    #  Done
    #def get_guess(self, *args):
    #  Done
    #def xvar_to_yvar(params, cov):
    #  Done


    # - NEEDED - #
    @classmethod
    def get_ymodel(cls, params, x):
        """ """
        y_model_x1 = cls.get_brokenline(x[0], 
                                        *params["coefs"][:2],
                                        xbreak=params.get("xbreak")
                                        ) # (1,N)
        y_model_nox1 = params["coefs"][2:][:,None] * x[1:] # (M-1,N)
        
        # full y_model
        y_model = jnp.sum(jnp.vstack([y_model_x1, y_model_nox1]), 
                          axis=0) 
        
        return y_model + params["offset"]
        
    def get_guess(self, init=None, incl_xbreak=True):
        """ """
        # add xbreak in the guess
        guess = super().get_guess(init)
        if not incl_xbreak:
            return guess
        return {"xbreak": jnp.asarray(0., dtype="float32")} | guess

    @classmethod
    def xvar_to_yvar(cls, params, cov):
        """ """
        p = {"coefs": params["coefs"]**2,
             "xbreak": params["xbreak"],
             "offset": 0}

        yvar = cls.get_ymodel(p, x=cov.x_err**2)
        return yvar

    # - model difference - #
    @staticmethod
    def get_brokenline(x, alpha_low, alpha_high, xbreak=0):
        """ """
        delta_alpha = alpha_high-alpha_low
        return x * alpha_low + (jnp.abs(x-xbreak) + (x-xbreak))*delta_alpha/2

 
    # ------------- #
    #   Internal    #
    # ------------- #
    @property
    def ncoefs(self):
        """ number of standardisation parameter (M) """
        return len(self.x_obs) + 1
    
