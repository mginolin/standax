import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
from iminuit.cost import LeastSquares
import pandas
from scipy.stats import norm
from astropy import cosmology
import astropy
from . import standardisation

try:
    import jax.numpy as jnp
except ModuleNotFoundError:
    print('jax is not installed: the standax standardisation will not work')

def get_residuals (redshift, mag, mag_err, H0):
    cosmo = cosmology.FlatLambdaCDM(H0, 0.315)
    residuals = mag - np.array(cosmo.distmod(np.array(redshift)))
    return (residuals, mag_err)

def get_loglikelihood_hubble(redshift, data, data_err):
    def f(H0, sig):
        cosmo = cosmology.FlatLambdaCDM(H0, 0.315)
        predicted_distmod = cosmo.distmod(np.array(redshift)).value
        sig_tot = np.sqrt(sig**2+data_err**2)
        return np.sum(((data-predicted_distmod)/sig_tot)**2/2)+np.sum(np.log(sig_tot))+0.5*np.log(2*np.pi)*len(data)
    return f

def get_loglikelihood_tot(res, res_err, x1, c):
    def f(beta, alpha, const, sig):
        if sig < 0:
            return 1e99
        else:
            sig_tot = np.sqrt(sig**2+res_err**2)
            return np.sum(((res-beta*c+alpha*x1-const)/sig_tot)**2/2)+np.sum(np.log(sig_tot))+0.5*np.log(2*np.pi)*len(res)
    return f

def get_loglikelihood_cov_tot(res, x1, c, cov):
    def f(beta, alpha, const, sig):
        if sig < 0:
            return 1e99
        else:
            params = np.array([1, alpha, -beta])
            var_tot = np.dot(np.dot(cov,params), params.T)
            return -np.nansum(np.log(norm.pdf(res-beta*c+alpha*x1, loc=const, scale=np.sqrt(var_tot+sig**2))))
    return f

def get_chi2_cov_tot(res, x1, c, cov, beta, alpha, const, sig):
    params = np.array([1, alpha, -beta])
    var_tot = np.dot(np.dot(cov,params), params.T)
    return np.nansum((res-beta*c+alpha*x1-const)**2/(var_tot+sig**2))


def get_loglikelihood_cov_tot_step(mask, res, x1, c, cov):
    def f(beta, alpha, const, step, sig):
        if sig < 0:
            return 1e99
        else:
            mask_inv = mask == False
            params = np.array([1, alpha, -beta])
            var_tot = np.dot(np.dot(cov,params), params.T)
            pn = -np.nansum(np.log(norm.pdf(res[mask]-beta*c[mask]+alpha*x1[mask], loc=const+step/2, scale=np.sqrt(var_tot[mask]+sig**2))))
            pn_inv = -np.nansum(np.log(norm.pdf(res[mask_inv]-beta*c[mask_inv]+alpha*x1[mask_inv], loc=const-step/2, scale=np.sqrt(var_tot[mask_inv]+sig**2))))
            return pn + pn_inv
    return f

def get_loglikelihood_cov_tot_step_cdf(step_cdf, res, x1, c, cov):
    def f(beta, alpha, const, step, sig):
        if sig < 0:
            return 1e99
        else:
            steps = step*(step_cdf-0.5)
            params = np.array([1, alpha, -beta])
            var_tot = np.dot(np.dot(cov,params), params.T)
            pn = -np.nansum(np.log(norm.pdf(res-beta*c+alpha*x1, loc=const+steps, scale=np.sqrt(var_tot+sig**2))))
            return pn
    return f

def get_chi2_cov_tot_step(mask, res, x1, c, cov, beta, alpha, const, step, sig):
    mask_inv = mask == False
    params = np.array([1, alpha, -beta])
    var_tot = np.dot(np.dot(cov,params), params.T)
    pn = np.nansum((res[mask]-beta*c[mask]+alpha*x1[mask]- (const+step/2))**2/(var_tot[mask]+sig**2))
    pn_inv = np.nansum((res[mask_inv]-beta*c[mask_inv]+alpha*x1[mask_inv]-(const-step/2))**2/(var_tot[mask_inv]+sig**2))
    return pn + pn_inv

def get_loglikelihood_tot_step(mask, res, res_err, x1, c):
    def f(beta, alpha, const, step, sig):
        if sig < 0:
            return 1e99
        else:
            mask_inv = mask == False
            sig_tot = np.sqrt(sig**2+res_err**2)
            pn = np.sum(((res[mask]-beta*c[mask]+alpha*x1[mask]-(const+step/2))/sig_tot[mask])**2/2) + np.sum(np.log(sig_tot[mask])) + 0.5*np.log(2*np.pi)*len(res[mask])
            pn_inv = np.sum(((res[mask_inv]-beta*c[mask_inv]+alpha*x1[mask_inv]-(const-step/2))/sig_tot[mask_inv])**2/2) + np.sum(np.log(sig_tot[mask_inv])) + 0.5*np.log(2*np.pi)*len(res[mask_inv])
            return pn + pn_inv
    return f

def get_loglikelihood_step(mask, res, res_err, x1, c, beta, alpha):
    def f(const, step, sig):
        if sig < 0:
            return 1e99
        else:
            mask_inv = mask == False
            sig_tot = np.sqrt(sig**2+res_err**2)
            pn = np.sum(((res[mask]-beta*c[mask]+alpha*x1[mask]-(const+step/2))/sig_tot[mask])**2/2) + np.sum(np.log(sig_tot[mask])) + 0.5*np.log(2*np.pi)*len(res[mask])
            pn_inv = np.sum(((res[mask_inv]-beta*c[mask_inv]+alpha*x1[mask_inv]-(const-step/2))/sig_tot[mask_inv])**2/2) + np.sum(np.log(sig_tot[mask_inv])) + 0.5*np.log(2*np.pi)*len(res[mask_inv])
            return pn + pn_inv
    return f
    
    
class CleanDataset:
    
    """
    SN dataset cleaned from Hubble diagram outliers, assuming a flat LambdaCDM cosmology initialised with the Planck 2018 value for OmegaM
    """
    
    def __init__(self, data):
        self._data = data         

    
    @classmethod
    def from_dataset(cls, data):
        """
        Initialization with an existing panda dataset
        """
        return cls(data)
    
    def get_data(self, column = False):
        if column == False:
            return self.data
        else:
            return self.data[column]
    
    def reject_outliers(self, guessH0=100, guesssig=0.5):  ## change the level at which you detect outliers?
        """
        Outlier rejection with a Chauvenet criterion at 3 sigma
        """
        
        m = Minuit(get_loglikelihood_hubble(self.data['redshift'], self.data['mag'], self.data['mag_err']), 
                   H0=guessH0, sig=guesssig)
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
        cosmo = cosmology.FlatLambdaCDM(m.values[0], self.omega_m)
        cdf_cut = norm.cdf(self.data['mag'], loc=cosmo.distmod(np.array(self.data['redshift'])), 
                           scale=np.sqrt(self.data['mag_err']**2+m.values[1]**2))
        mask = (cdf_cut > 0.25/len(self.data['mag'])) & (cdf_cut < 1-0.25/len(self.data['mag']))
        return mask
    
    @property
    def data(self):
        return self._data
    
    @property
    def data_cut(self):
        if not hasattr(self, "_data_cut"):
            mask = self.reject_outliers()
            self._data_cut = self.data[mask]
        return self._data_cut
    
    @property
    def omega_m(self):
        if not hasattr(self, "_omega_m"):
            self._omega_m = 0.315
        return self._omega_m
    
    def set_omega_m(self, omega_m):
        self._omega_m = omega_m       
        
        
class HubbleResiduals:
    """
    SN dataset associated with an Hubble constant, residuals computed with this constant, colour/stretch/step correction
    """
    
    def __init__(self, data):
        self._data = data         
    
    @classmethod
    def from_dataset(cls, data):
        return cls(data)
    
    def get_data(self, column = False):
        if column == False:
            return self.data
        else:
            return self.data[column]
    
    def get_hubble_fit(self):
        m = Minuit(get_loglikelihood_hubble(self.data['redshift'], self.data['mag'], 
                                                   self.data['mag_err']),  H0=100, sig=0.5)
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()
        self.set_hubble_const = m.values[0]
        self.set_hubble_scatter = m.values[1]
    
    def get_residuals (self):
        cosmo = cosmology.FlatLambdaCDM(self.hubble_const, self.omega_m)
        return self.data['mag'] - np.array(cosmo.distmod(np.array(self.data['redshift'])))
    
    def correction(self, cov_beta_alpha=0): ##corrects the residuals with the fitted alpha, beta, and const or step
        """
        Corrects the residuals with the previously fitted colour/stretch (and eventually step)
        """
        if self.step==None:
            self.set_res_corr(self.res - self.beta*self.data['c'] + self.alpha*self.data['x1'] - self.const)
            self.set_mag_corr(self.data['mag'] - self.beta*self.data['c'] + self.alpha*self.data['x1'] - self.const)
            pcov = self.cov_params
            cov = self.cov
            var = cov[:,0,0] + self.beta**2*cov[:,2,2] + self.data['c']**2*pcov[0,0] + cov[:,2,2]*pcov[0,0] + self.alpha**2*cov[:,1,1] + self.data['x1']**2*pcov[1,1] + cov[:,1,1]*pcov[1,1] + pcov[2,2] + 2*(self.alpha*cov[:,0,1]-self.beta*cov[:,0,2]) - 2*(self.alpha*self.beta*cov[:,1,2]+self.data['x1']*self.data['c']*pcov[0,1]+cov[:,1,2]*pcov[0,1]) + self.data['c']*2*pcov[0,2] - self.data['x1']*2*pcov[1,2]
            self.set_mag_corr_err(np.sqrt(var))
        else:
            if self.broken_alpha == False:
                if self.smooth_step == False:
                    mask = self.stepmask
                    mask_inv = self.stepmask == False
                    res_corr = np.zeros(len(self.res))
                    mag_corr = np.zeros(len(self.res))
                    mag_corr_err = np.zeros(len(self.res))
                    res_corr[mask] = self.res[mask]-self.beta*self.data['c'][mask]+self.alpha*self.data['x1'][mask]-(self.const+self.step/2)
                    res_corr[mask_inv] = self.res[mask_inv]-self.beta*self.data['c'][mask_inv]+self.alpha*self.data['x1'][mask_inv]-(self.const-self.step/2)
                    mag_corr[mask] = self.data['mag'][mask]-self.beta*self.data['c'][mask]+self.alpha*self.data['x1'][mask]-(self.const+self.step/2)
                    mag_corr[mask_inv] = self.data['mag'][mask_inv]-self.beta*self.data['c'][mask_inv]+self.alpha*self.data['x1'][mask_inv]-(self.const-self.step/2)
                    self.set_res_corr(res_corr)
                    self.set_mag_corr(mag_corr)
                    mag_corr_err = np.zeros(len(self.data))
                    pcov = self.cov_params
                    if len(self.cov_params) == 3: #this is for the case of the step correction with fixed beta, alpha
                        pcov = np.zeros((4,4))
                        pcov[0,0] = self.beta_err**2
                        pcov[1,1] = self.alpha_err**2
                        pcov[0,1] = cov_beta_alpha
                        pcov[2,2] = self.cov_params[0,0]
                        pcov[3,3] = self.cov_params[1,1]
                        pcov[2,3] = self.cov_params[0,1]                
                    cov = self.cov
                    mag_corr_err[mask] = np.sqrt(cov[:,0,0][mask] + self.beta**2*cov[:,2,2][mask]  + self.data['c'][mask]**2*pcov[0,0] + cov[:,2,2][mask]*pcov[0,0] + self.alpha**2*cov[:,1,1][mask]  + self.data['x1'][mask] **2*pcov[1,1] + cov[:,1,1][mask]*pcov[1,1] + pcov[2,2] + pcov[3,3]/4 + 2*(self.alpha*cov[:,0,1][mask]-self.beta*cov[:,0,2][mask]) - 2*(self.alpha*self.beta*cov[:,1,2][mask]+self.data['x1'][mask]*self.data['c'][mask]*pcov[0,1]+cov[:,1,2][mask]*pcov[0,1]) + self.data['c'][mask]*(2*pcov[0,2]+pcov[0,3]) - self.data['x1'][mask]*(2*pcov[1,2]+pcov[1,3]) + pcov[2,3])
                    mag_corr_err[mask_inv] = np.sqrt(cov[:,0,0][mask_inv] + self.beta**2*cov[:,2,2][mask_inv] + self.data['c'][mask_inv]**2*pcov[0,0] + cov[:,2,2][mask_inv]*pcov[0,0] + self.alpha**2*cov[:,1,1][mask_inv] + self.data['x1'][mask_inv]**2*pcov[1,1] + cov[:,1,1][mask_inv]*pcov[1,1] + pcov[2,2] + pcov[3,3]/4 + 2*(self.alpha*cov[:,0,1][mask_inv]-self.beta*cov[:,0,2][mask_inv]) - 2*(self.alpha*self.beta*cov[:,1,2][mask_inv]+self.data['x1'][mask_inv]*self.data['c'][mask_inv]*pcov[0,1]+cov[:,1,2][mask_inv]*pcov[0,1]) + self.data['c'][mask_inv]*(2*pcov[0,2]+pcov[0,3]) - self.data['x1'][mask_inv]*(2*pcov[1,2]+pcov[1,3]) + pcov[2,3])
                else:
                    steps = np.zeros(len(self.res))
                    steps = self.step*(self.stepcdf-0.5)
                    res_corr = np.zeros(len(self.res))
                    mag_corr = np.zeros(len(self.res))
                    mag_corr_err = np.zeros(len(self.res))
                    res_corr = self.res-self.beta*self.data['c']+self.alpha*self.data['x1']-(self.const + steps)
                    mag_corr = self.data['mag']-self.beta*self.data['c']+self.alpha*self.data['x1']-(self.const + steps)
                    self.set_res_corr(res_corr)
                    self.set_mag_corr(mag_corr)
                    mag_corr_err = np.zeros(len(self.data))
                    pcov = self.cov_params
                    if len(self.cov_params) == 3: #this is for the case of the step correction with fixed beta, alpha
                        pcov = np.zeros((4,4))
                        pcov[0,0] = self.beta_err**2
                        pcov[1,1] = self.alpha_err**2
                        pcov[0,1] = cov_beta_alpha
                        pcov[2,2] = self.cov_params[0,0]
                        pcov[3,3] = self.cov_params[1,1]
                        pcov[2,3] = self.cov_params[0,1]                
                    cov = self.cov
                    mag_corr_err = np.sqrt(cov[:,0,0] + self.beta**2*cov[:,2,2] + self.data['c']**2*pcov[0,0] + cov[:,2,2]*pcov[0,0] + self.alpha**2*cov[:,1,1] + self.data['x1']**2*pcov[1,1] + cov[:,1,1]*pcov[1,1] + pcov[2,2] + pcov[3,3]*(self.stepcdf-0.5)**2 + 2*(self.alpha*cov[:,0,1]-self.beta*cov[:,0,2]) - 2*(self.alpha*self.beta*cov[:,1,2]+self.data['x1']*self.data['c']*pcov[0,1]+cov[:,1,2]*pcov[0,1]) + self.data['c']*(2*pcov[0,2]+pcov[0,3]*(self.stepcdf-0.5)) - self.data['x1']*(2*pcov[1,2]+pcov[1,3]*(self.stepcdf-0.5)) + pcov[2,3]*(self.stepcdf-0.5))
            else:
                if self.fit_method == 'loglikelihood':
                    print("Broken-alpha standardisation is not yet implemented for the regular loglikelihood method")
                else:
                    if self.smooth_step == False:
                        print("Broken-alpha standardisation is not yet implemented for the non-smooth step")
                    else:
                        steps = np.zeros(len(self.res))
                        steps = self.step*(self.stepcdf-0.5)
                        res_corr = np.zeros(len(self.res))
                        mag_corr = np.zeros(len(self.res))
                        mag_corr_err = np.zeros(len(self.res))
                        mask_x1_low = self.data['x1'] < self.x1break
                        mask_x1_high = mask_x1_low == False
                        alphas = np.zeros(len(self.res))
                        alphas[mask_x1_low] = self.alpha[0]
                        alphas[mask_x1_high] = self.alpha[1]
                        delta_alpha = self.alpha[1] - self.alpha[0]
                        alpha_corr = self.data['x1']*self.alpha[0]+(np.abs(self.data['x1']-self.x1break)+(self.data['x1']-self.x1break))*delta_alpha/2
                        res_corr = self.res-self.beta*self.data['c']+alpha_corr-(self.const + steps)
                        mag_corr = self.data['mag']-self.beta*self.data['c']+alpha_corr-(self.const + steps)
                        self.set_res_corr(res_corr)
                        self.set_mag_corr(mag_corr)
                        mag_corr_err = np.zeros(len(self.data))
                        pcov = self.cov_params               
                        cov = self.cov
                        alphas_var = np.zeros(len(self.res))
                        alphas_var[mask_x1_low] = pcov[1,1]
                        alphas_var[mask_x1_high] = pcov[2,2]
                        cov_alphas_beta = np.zeros(len(self.res))
                        cov_alphas_beta[mask_x1_low] = pcov[0,1]
                        cov_alphas_beta[mask_x1_high] = pcov[0,2]
                        cov_alphas_const = np.zeros(len(self.res))
                        cov_alphas_const[mask_x1_low] = pcov[3,1]
                        cov_alphas_const[mask_x1_high] = pcov[3,2]
                        cov_alphas_step = np.zeros(len(self.res))
                        cov_alphas_step[mask_x1_low] = pcov[4,1]
                        cov_alphas_step[mask_x1_high] = pcov[4,2]
                        mag_corr_err = np.sqrt(cov[:,0,0] + self.beta**2*cov[:,2,2] + self.data['c']**2*pcov[0,0] + cov[:,2,2]*pcov[0,0] + alphas**2*cov[:,1,1] + self.data['x1']**2*alphas_var + cov[:,1,1]*alphas_var + pcov[3,3] + pcov[4,4]*(self.stepcdf-0.5)**2 + 2*(alphas*cov[:,0,1]-self.beta*cov[:,0,2]) - 2*(alphas*self.beta*cov[:,1,2]+self.data['x1']*self.data['c']*cov_alphas_beta+cov[:,1,2]*cov_alphas_beta) + self.data['c']*(2*pcov[0,3]+pcov[0,4]*(self.stepcdf-0.5)) - self.data['x1']*(2*cov_alphas_const+cov_alphas_step*(self.stepcdf-0.5)) + pcov[3,4]*(self.stepcdf-0.5))
            self.set_mag_corr_err(mag_corr_err)

            
    def full_fit(self, guess_beta=3.2, guess_alpha=0.15, guess_const=-0.1, guess_sigma=0.15, force_sigmaint=False):  ## single constant for the whole sample
        """
        Fits for a colour and stretch correction of the residuals (beta and alpha parameters)
        """
        self.set_broken_alpha(False)
        if self.fit_method == 'loglikelihood':   
            m_res = Minuit(get_loglikelihood_cov_tot(self.res, self.data['x1'], self.data['c'], self.cov), beta=guess_beta, alpha=guess_alpha, const=guess_const, sig=guess_sigma)
            m_res.errordef = Minuit.LIKELIHOOD
            m_res.migrad()
            self.set_beta(m_res.values[0])
            self.set_beta_err(m_res.errors[0])
            self.set_alpha(m_res.values[1])
            self.set_alpha_err(m_res.errors[1])
            self.set_const(m_res.values[2])
            self.set_const_err(m_res.errors[2])
            self.set_sn_scatter(m_res.values[3])
            self.set_minuit(m_res)
            self.set_chi2(get_chi2_cov_tot(self.res, self.data['x1'], self.data['c'], self.cov, self.beta, self.alpha, self.const, self.sn_scatter)/(len(self.res)-4))
            self.set_cov_params(m_res.covariance)
            self.set_loglikelihood(m_res.fval)
        elif self.fit_method == 'standax':
            fit_data = pandas.DataFrame({'mag':self.res, 'x1':self.data['x1'], 'c':self.data['c'], 'x1_err':self.data['x1_err'], 'c_err':self.data['c_err'], 'mag_err':self.data['mag_err'], 'cov_c_x1':self.data['cov_x1_c'], 'cov_x1_mag':self.cov[:,0,1], 'cov_c_mag':self.cov[:,0,2]})
            if force_sigmaint == True:
                niter_sig = 0
            else:
                niter_sig = 10
            (fid, sigmaint, mcmc_l), this_l = standardisation.standardise_snia(fit_data, 
                                                      init=[guess_beta, -guess_alpha], 
                                                      xkeys=["c", "x1"], 
                                                      sigmaint_guess=guess_sigma, 
                                                      model="linear", 
                                                      nfetch=niter_sig, verbose=False,
                                                     lmbda=1e4, fit_method="tncg"
                                                                              )
            self.set_beta(fid['coefs'][0])
            self.set_alpha(-fid['coefs'][1])
            self.set_const(fid['offset'])
            self.set_sn_scatter(sigmaint[-1])
            cov_params = np.zeros((3,3))
            chain_beta = mcmc_l.get_samples()['coefs'][:,0]
            chain_alpha = -mcmc_l.get_samples()['coefs'][:,1]
            chain_offset = mcmc_l.get_samples()['offset'][:]
            len_chain = len(chain_beta)
            cov_params[0,0] = np.sum((chain_beta-self.beta)**2)/len_chain
            cov_params[1,1] = np.sum((chain_alpha-self.alpha)**2)/len_chain
            cov_params[2,2] = np.sum((chain_offset-self.const)**2)/len_chain
            cov_params[0,1] = np.sum((chain_beta-self.beta)*(chain_alpha-self.alpha))/len_chain
            cov_params[1,0] = cov_params[0,1]
            cov_params[0,2] = np.sum((chain_beta-self.beta)*(chain_offset-self.const))/len_chain
            cov_params[2,0] = cov_params[0,2]
            cov_params[1,2] = np.sum((chain_alpha-self.alpha)*(chain_offset-self.const))/len_chain
            cov_params[2,1] = cov_params[1,2]
            self.set_cov_params(cov_params)
            self.set_beta_err(np.sqrt(cov_params[0,0]))
            self.set_alpha_err(np.sqrt(cov_params[1,1]))
            self.set_const_err(np.sqrt(cov_params[2,2]))
            self.set_chi2(this_l.model.get_chi2(fid, sigmaint=self.sn_scatter))
            self.set_loglikelihood(this_l.model.get_likelihood(fid, sigmaint=self.sn_scatter)*0.5)
        self.correction()
    
    def fit_step(self, guess_beta=3.3, guess_alpha=0.15, guess_const=-0.13, guess_step=0.1, guess_sigma=0.15, force_sigmaint=False, 
                smooth_step=None):  ## two constants for two pops with a mask
        """
        Fits for a colour and stretch correction of the residuals, as well as a step
        A mask needs to cut the data in two to be able to compute a step
        """
        self.set_broken_alpha(False)
        if smooth_step is not None:
            self.set_smooth_step(smooth_step) 
        else:
            self.set_smooth_step(False) 
        if self.fit_method == 'loglikelihood':
            if self.smooth_step == False:
                m_res = Minuit(get_loglikelihood_cov_tot_step(self.stepmask, self.res, self.data['x1'],self.data['c'], self.cov), beta=guess_beta, alpha=guess_alpha, const=guess_const, step=guess_step, sig=guess_sigma)
            else:
                m_res = Minuit(get_loglikelihood_cov_tot_step_cdf(self.stepcdf, self.res, self.data['x1'], self.data['c'], self.cov), beta=guess_beta, alpha=guess_alpha, const=guess_const, step=guess_step, sig=guess_sigma)
            m_res.errordef = Minuit.LIKELIHOOD
            m_res.migrad()
            self.set_beta(m_res.values[0])
            self.set_beta_err(m_res.errors[0])
            self.set_alpha(m_res.values[1])
            self.set_alpha_err(m_res.errors[1])
            self.set_const(m_res.values[2])
            self.set_const_err(m_res.errors[2])
            self.set_step(m_res.values[3])
            self.set_step_err(m_res.errors[3])
            self.set_sn_scatter(m_res.values[4])
            self.set_minuit(m_res)
            self.set_loglikelihood(m_res.fval)
            self.set_cov_params(m_res.covariance)
            if self.smooth_step == False:
                self.set_chi2(get_chi2_cov_tot_step(self.stepmask, self.res, self.data['x1'], self.data['c'], self.cov, self.beta, self.alpha, self.const, self.step, self.sn_scatter)/(len(self.res)-5))
        elif self.fit_method == 'standax':
            if self.smooth_step == True:
                step = self.stepcdf-0.5
            else:
                step = self.stepmask*1-0.5
            step_err = np.ones(len(step))*1e-4*1.
            fit_data_step = pandas.DataFrame({'mag':self.res, 'x1':self.data['x1'], 'c':self.data['c'], 'x1_err':self.data['x1_err'], 'c_err':self.data['c_err'], 'mag_err':self.data['mag_err'], 'cov_c_x1':self.data['cov_x1_c'], 'cov_x1_mag':self.cov[:,0,1], 'cov_c_mag':self.cov[:,0,2], 'step':step, 'step_err':step_err, 'cov_step_mag':np.zeros(len(step))*1., 'cov_step_x1':np.zeros(len(step))*1., 'cov_step_c':np.zeros(len(step))*1.})
            if force_sigmaint == True:
                niter_sig = 0
            else:
                niter_sig = 10
            (fid, sigmaint, mcmc_l), this_l = standardisation.standardise_snia(fit_data_step, 
                                                      init=[guess_beta, -guess_alpha, 0.1], 
                                                      xkeys=["c", "x1", "step"], 
                                                      sigmaint_guess=guess_sigma, 
                                                      model="linear", 
                                                      nfetch=niter_sig, verbose=False,
                                                      lmbda=1e4, fit_method="tncg"
                                                                              )
            self.set_beta(fid['coefs'][0])
            self.set_alpha(-fid['coefs'][1])
            self.set_step(fid['coefs'][2])
            self.set_const(fid['offset'])
            self.set_sn_scatter(sigmaint[-1])
            cov_params = np.zeros((4,4))
            chain_beta = mcmc_l.get_samples()['coefs'][:,0]
            chain_alpha = -mcmc_l.get_samples()['coefs'][:,1]
            chain_offset = mcmc_l.get_samples()['offset'][:]
            chain_step = mcmc_l.get_samples()['coefs'][:,2]
            len_chain = len(chain_beta)
            cov_params[0,0] = np.sum((chain_beta-self.beta)**2)/len_chain
            cov_params[1,1] = np.sum((chain_alpha-self.alpha)**2)/len_chain
            cov_params[2,2] = np.sum((chain_offset-self.const)**2)/len_chain
            cov_params[3,3] = np.sum((chain_step-self.step)**2)/len_chain
            cov_params[0,1] = np.sum((chain_beta-self.beta)*(chain_alpha-self.alpha))/len_chain
            cov_params[1,0] = cov_params[0,1]
            cov_params[0,2] = np.sum((chain_beta-self.beta)*(chain_offset-self.const))/len_chain
            cov_params[2,0] = cov_params[0,2]
            cov_params[0,3] = np.sum((chain_beta-self.beta)*(chain_step-self.step))/len_chain
            cov_params[3,0] = cov_params[0,3]
            cov_params[1,2] = np.sum((chain_alpha-self.alpha)*(chain_offset-self.const))/len_chain
            cov_params[2,1] = cov_params[1,2]
            cov_params[1,3] = np.sum((chain_alpha-self.alpha)*(chain_step-self.step))/len_chain
            cov_params[3,1] = cov_params[1,3]
            cov_params[2,3] = np.sum((chain_offset-self.const)*(chain_step-self.step))/len_chain
            cov_params[3,2] = cov_params[2,3]
            self.set_cov_params(cov_params)
            self.set_beta_err(np.sqrt(cov_params[0,0]))
            self.set_alpha_err(np.sqrt(cov_params[1,1]))
            self.set_const_err(np.sqrt(cov_params[2,2]))
            self.set_step_err(np.sqrt(cov_params[3,3]))
            self.set_cov_params(cov_params)
            self.set_chi2(this_l.model.get_chi2(fid, sigmaint=self.sn_scatter))
            self.set_loglikelihood(this_l.model.get_likelihood(fid, sigmaint=self.sn_scatter)*0.5)
        self.correction()


    def fit_broken_alpha(self, guess_beta=3.3, guess_alpha_low=0.2, guess_alpha_high=0.05, guess_const=-0.13, guess_step=0.13, guess_x1break=-0.5, guess_sigma=0.15, force_sigmaint=False, smooth_step=None):  ## two constants for two pops with a mask
        """
        Fits for a colour and stretch correction of the residuals using the broken-alpha standardisation from Ginolin et al (2024a), as well as a step
        A mask needs to cut the data in two to be able to compute a step
        """
        self.set_broken_alpha(True)
        if smooth_step is not None:
            self.set_smooth_step(smooth_step) 
        else:
            self.set_smooth_step(False) 
        if self.fit_method == 'loglikelihood':
            print("Broken-alpha standardisation is not yet implemented for the regular likelihood method")
        elif self.fit_method == 'standax':
            if self.smooth_step == True:
                step = self.stepcdf-0.5
            else:
                step = self.stepmask*1-0.5
            step_err = np.ones(len(step))*1e-4*1.
            fit_data_step = pandas.DataFrame({'mag':self.res, 'x1':self.data['x1'], 'c':self.data['c'], 'x1_err':self.data['x1_err'], 'c_err':self.data['c_err'], 'mag_err':self.data['mag_err'], 'cov_c_x1':self.data['cov_x1_c'], 'cov_x1_mag':self.cov[:,0,1], 'cov_c_mag':self.cov[:,0,2], 'step':step, 'step_err':step_err, 'cov_step_mag':np.zeros(len(step))*1., 'cov_step_x1':np.zeros(len(step))*1., 'cov_step_c':np.zeros(len(step))*1.})
            if force_sigmaint == True:
                niter_sig = 0
            else:
                niter_sig = 10
            init = {"coefs": jnp.asarray([-guess_alpha_low, -guess_alpha_high, guess_beta, guess_step], dtype="float32"),
        "xbreak": jnp.asarray(guess_x1break, dtype="float32")}
            (fid, sigmaint, mcmc_l), this_l = standardisation.standardise_snia(fit_data_step, init=init, xkeys=["x1", "c", "step"], 
                                                  sigmaint_guess=guess_sigma, model="brokenlinear", nfetch=niter_sig, verbose=False,
                                                                              lmbda=1e4, fit_method="tncg")
            self.set_beta(fid['coefs'][2])
            self.set_alpha([-fid['coefs'][0], -fid['coefs'][1]])
            self.set_step(fid['coefs'][3])
            self.set_const(fid['offset'])
            self.set_x1break(fid['xbreak'])
            chain_x1break = mcmc_l.get_samples()['xbreak'][:]
            self.set_x1break_err(np.sqrt(np.sum((chain_x1break-self.x1break)**2)/len(chain_x1break)))
            self.set_sn_scatter(sigmaint[-1])
            cov_params = np.zeros((5,5))
            chain_beta = mcmc_l.get_samples()['coefs'][:,2]
            # Here we do not take into account the full covariances
            chain_alpha_low = -mcmc_l.get_samples()['coefs'][:,0]
            chain_alpha_high = -mcmc_l.get_samples()['coefs'][:,1]
            chain_offset = mcmc_l.get_samples()['offset'][:]
            chain_step = mcmc_l.get_samples()['coefs'][:,3]
            len_chain = len(chain_beta)
            cov_params[0,0] = np.sum((chain_beta-self.beta)**2)/len_chain
            cov_params[1,1] = np.sum((chain_alpha_low-self.alpha[0])**2)/len_chain
            cov_params[2,2] = np.sum((chain_alpha_high-self.alpha[1])**2)/len_chain
            cov_params[3,3] = np.sum((chain_offset-self.const)**2)/len_chain
            cov_params[4,4] = np.sum((chain_step-self.step)**2)/len_chain
            cov_params[0,1] = np.sum((chain_beta-self.beta)*(chain_alpha_low-self.alpha[0]))/len_chain
            cov_params[1,0] = cov_params[0,1]
            cov_params[0,2] = np.sum((chain_beta-self.beta)*(chain_alpha_high-self.alpha[1]))/len_chain
            cov_params[2,0] = cov_params[0,2]
            cov_params[0,3] = np.sum((chain_beta-self.beta)*(chain_offset-self.const))/len_chain
            cov_params[3,0] = cov_params[0,3]
            cov_params[0,4] = np.sum((chain_beta-self.beta)*(chain_step-self.step))/len_chain
            cov_params[4,0] = cov_params[0,4]
            cov_params[1,2] = np.sum((chain_alpha_low-self.alpha[0])*(chain_alpha_high-self.alpha[1]))/len_chain
            cov_params[2,1] = cov_params[1,2] 
            cov_params[1,3] = np.sum((chain_alpha_low-self.alpha[0])*(chain_offset-self.const))/len_chain
            cov_params[3,1] = cov_params[1,3]
            cov_params[1,4] = np.sum((chain_alpha_low-self.alpha[0])*(chain_step-self.step))/len_chain
            cov_params[4,1] = cov_params[1,4]
            cov_params[2,3] = np.sum((chain_alpha_high-self.alpha[1])*(chain_offset-self.const))/len_chain
            cov_params[3,2] = cov_params[2,3]
            cov_params[2,4] = np.sum((chain_alpha_high-self.alpha[1])*(chain_step-self.step))/len_chain
            cov_params[4,2] = cov_params[2,4]      
            cov_params[3,4] = np.sum((chain_offset-self.const)*(chain_step-self.step))/len_chain
            cov_params[4,3] = cov_params[3,4]
            self.set_cov_params(cov_params)
            self.set_beta_err(np.sqrt(cov_params[0,0]))
            self.set_alpha_err([np.sqrt(cov_params[1,1]), np.sqrt(cov_params[2,2])])
            self.set_const_err(np.sqrt(cov_params[3,3]))
            self.set_step_err(np.sqrt(cov_params[4,4]))
            self.set_chi2(this_l.model.get_chi2(fid, sigmaint=self.sn_scatter))
            self.set_loglikelihood(this_l.model.get_likelihood(fid, sigmaint=self.sn_scatter)*0.5)
        self.correction()
       
    def fit_step_fixed(self, beta, beta_err, alpha, alpha_err, cov_alpha_beta=0, guess_sigma=0.16, force_sigmaint=False, smooth_step=True):  ##fixed beta,alpha for comparison between masks
        """
        Fits for a step correction of the residuals for fixed (alpha, beta) parameters
        """
        if smooth_step is not None:
            self.set_smooth_step(smooth_step) 
        else:
            self.set_smooth_step(False)
        self.set_broken_alpha(False)
        if self.fit_method == 'loglikelihood':
            if self.smooth_step == False:
                m_res = Minuit(get_loglikelihood_step(self.stepmask, self.res, self.data['mag_err'], 
                            self.data['x1'],self.data['c'], beta, alpha), const=-0.1, step=0.1, sig=guess_sigma)
                m_res.errordef = Minuit.LIKELIHOOD
                m_res.migrad()
                self.set_beta(beta)
                self.set_beta_err(beta_err)
                self.set_alpha(alpha)
                self.set_alpha_err(alpha_err)
                self.set_const(m_res.values[0])
                self.set_const_err(m_res.errors[0])
                self.set_step(m_res.values[1])
                self.set_step_err(m_res.errors[1])
                self.set_sn_scatter(m_res.values[2])
                self.set_minuit(m_res)
                self.set_chi2(get_chi2_cov_tot_step(self.stepmask, self.res, self.data['x1'], self.data['c'], self.cov, self.beta, self.alpha, self.const, self.step, self.sn_scatter)/(len(self.res)-5))
            else:
                print("Smooth step not implemented yet with loglikelihood")
        elif self.fit_method == 'standax':
            if self.smooth_step == True:
                step = self.stepcdf-0.5
            else:
                step = self.stepmask*1-0.5
            step_err = np.ones(len(step))*1e-4*1.
            fit_data_step = pandas.DataFrame({'mag':self.res, 'mag_err':self.data['mag_err'], 'step':step, 'step_err':step_err, 'cov_step_mag':np.zeros(len(step))*1.})          
            (fid, sigmaint, mcmc_l), this_l = standardisation.standardise_snia(fit_data_step, init_coefs=[0.13], xkeys=['step'], #lmbda=10000, 
                                                                     sigmaint=guess_sigma, force_sigmaint=force_sigmaint)
            self.set_beta(beta)
            self.set_alpha(alpha)
            self.set_step(fid['coefs'][0])
            self.set_const(fid['offset'])
            chain_step = mcmc_l.get_samples()['coefs'][:,0]
            chain_offset = mcmc_l.get_samples()['offset'][:]
            len_chain = len(chain_step)
            cov_params[0,0] = beta_err**2
            cov_params[1,1] = alpha_err**2
            cov_params[2,2] = np.sum((chain_offset-self.const)**2)/len_chain
            cov_params[3,3] = np.sum((chain_step-self.step)**2)/len_chain
            cov_params[0,1] = 0
            cov_params[1,0] = cov_params[0,1]
            cov_params[0,2] = 0
            cov_params[2,0] = cov_params[0,2]
            cov_params[0,3] = 0
            cov_params[3,0] = cov_params[0,3]
            cov_params[1,2] = 0
            cov_params[2,1] = cov_params[1,2]
            cov_params[1,3] = 0
            cov_params[3,1] = cov_params[1,3]
            cov_params[2,3] = np.sum((chain_offset-self.const)*(chain_step-self.step))/len_chain
            cov_params[3,2] = cov_params[2,3]
            self.set_cov_params(cov_params)
            self.set_beta_err(np.sqrt(cov_params[0,0]))
            self.set_alpha_err(np.sqrt(cov_params[1,1]))
            self.set_const_err(np.sqrt(cov_params[2,2]))
            self.set_step_err(np.sqrt(cov_params[3,3]))
            self.set_sn_scatter(fid['sigmaint'][-1])
            cov_params = np.zeros((2,2))
            cov_params[:, [0, 1]] = cov_params[:, [1, 0]] #swapping columns to put them in the same order as the minuit routine
            cov_params[[0, 1], :] = cov_params[[1, 0], :]
            self.set_cov_params(cov_params)
        self.correction(cov_alpha_beta)
        
    
    def plot_fit(self, fontsize='large'):
        if self.step==None: 
            plt.figure(figsize=(16, 6))
            plt.subplot(121)
            plot_residuals(self.data['c'], self.res, self.data['mag_err'], xaxis='')
            plt.xlabel('Color $c$', fontsize=fontsize)
            plt.ylabel('Hubble residuals', fontsize=fontsize)
            xc = np.linspace(np.min(self.data['c'])*1.05, np.max(self.data['c'])*1.05, 200)
            plt.plot(xc, affine(xc, self.beta, self.const), color='tab:orange')
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.subplot(122)
            plot_residuals(self.data['x1'], self.res, self.data['mag_err'], xaxis='')
            plt.xlabel('Stretch $x_1$', fontsize=fontsize)
            plt.ylabel('', fontsize=fontsize)
            xx1 = np.linspace(np.min(self.data['x1'])*1.05, np.max(self.data['x1'])*1.05, 200)
            plt.plot(xx1, affine(xx1, -self.alpha, self.const), color='tab:orange')
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.show()
        else:
            print("The fitted lines plotted contain the mean of the two constants magnitudes fitted for each population, and thus are not the true correction")
            plt.figure(figsize=(16, 6))
            plt.subplot(121)
            plot_residuals(self.data['c'], self.res, self.data['mag_err'], xaxis='')
            plt.xlabel('Color $c$', fontsize=fontsize)
            plt.ylabel('Hubble residuals', fontsize=fontsize)
            xc = np.linspace(np.min(self.data['c'])*1.05, np.max(self.data['c'])*1.05, 200)
            plt.plot(xc, affine(xc, self.beta, self.const), color='tab:orange')
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.subplot(122)
            plot_residuals(self.data['x1'], self.res, self.data['mag_err'], xaxis='')
            plt.xlabel('Stretch $x_1$', fontsize=fontsize)
            plt.ylabel('', fontsize=fontsize)
            xx1 = np.linspace(np.min(self.data['x1'])*1.05, np.max(self.data['x1'])*1.05, 200)
            plt.plot(xx1, affine(xx1, -self.alpha, self.const), color='tab:orange')
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.show()
            
        
    @property
    def data(self):
        return self._data  
    
    @property
    def res(self):
        if not hasattr(self, "_res"):
            self._res = self.get_residuals()
        return self._res
       
    @property
    def res_corr(self):
        if not hasattr(self, "_res_corr"):
            self.correction()
        return self._res_corr
    
    def set_res_corr(self, res_corr):
        self._res_corr = res_corr
    
    @property
    def mag_corr(self):
        if not hasattr(self, "_mag_corr"):
            self.correction()
        return self._mag_corr
    
    def set_mag_corr(self, mag_corr):
        self._mag_corr = mag_corr
    
    @property
    def mag_corr_err(self):
        if not hasattr(self, "_mag_corr_err"):
            self.correction()
        return self._mag_corr_err
    
    def set_mag_corr_err(self, mag_corr_err):
        self._mag_corr_err = mag_corr_err
    
    @property
    def hubble_const(self):
        if self.block_hubble_const == False:
            if not hasattr(self, "_hubble_const"):
                m = Minuit(get_loglikelihood_hubble(self.data['redshift'], self.data['mag'], 
                                                   self.data['mag_err']),  H0=100, sig=0.5)
                m.errordef = Minuit.LIKELIHOOD
                m.migrad()
                self._hubble_const = m.values[0]
            return self._hubble_const
        else:
            if not hasattr(self, "_hubble_const"):
                raise ValueError("H0 value is blocked: either define one or change the value of 'hubble_const_block' to False")
            else:
                return self._hubble_const
            
    def set_hubble_const(self, hubble_const):
        self._hubble_const = hubble_const
    
    @property
    def block_hubble_const(self):
        if not hasattr(self, "_block_hubble_const"):
            self._block_hubble_const = True
        return self._block_hubble_const
    
    def set_block_hubble_const(self, block_hubble_const):
        self._block_hubble_const = block_hubble_const
        
    @property
    def omega_m(self):
        if not hasattr(self, "_omega_m"):
            self._omega_m = 0.315
        return self._omega_m
    
    def set_omega_m(self, omega_m):
        self._omega_m = omega_m
        
    @property
    def beta(self):
        if not hasattr(self, "_beta"):
            self.full_fit()
        return self._beta
    
    def set_beta(self, beta):
        self._beta = beta
        
    @property
    def beta_err(self):
        if not hasattr(self, "_beta_err"):
            self.full_fit()
        return self._beta_err
    
    def set_beta_err(self, beta_err):
        self._beta_err = beta_err
        
    @property
    def alpha(self):
        if not hasattr(self, "_alpha"):
            self.full_fit()
        return self._alpha
    
    def set_alpha(self, alpha):
        self._alpha = alpha
        
    @property
    def alpha_err(self):
        if not hasattr(self, "_alpha_err"):
            self.full_fit()
        return self._alpha_err
    
    def set_alpha_err(self, alpha_err):
        self._alpha_err = alpha_err
        
    @property
    def const(self):
        if not hasattr(self, "_const"):
            self.full_fit()
        return self._const
    
    def set_const(self, const):
        self._const = const
        
    @property
    def const_err(self):
        if not hasattr(self, "_const_err"):
            self.full_fit()
        return self._const_err
    
    def set_const_err(self, const_err):
        self._const_err = const_err
        
    @property
    def step(self):
        if not hasattr(self, "_step"):
            self._step = None
        return self._step
    
    def set_step(self, step):
        self._step = step
    
    @property
    def step_err(self):
        if not hasattr(self, "_step_err"):
            self._step_err = None
        return self._step_err
    
    def set_step_err(self, step_err):
        self._step_err = step_err
        
    @property
    def minuit(self):
        if not hasattr(self, "_minuit"):
            self.full_fit()
        return self._minuit
    
    def set_minuit(self, minuit):
        self._minuit = minuit
    
    @property
    def stepmask(self):
        if not hasattr(self, "_stepmask"):
            print('To fit a step a mask needs to be defined')
            self._stepmask = None
        return self._stepmask
    
    def set_stepmask(self, stepmask):
        self._stepmask = stepmask

    @property
    def stepcdf(self):
        if not hasattr(self, "_stepcdf"):
            print('To fit a smooth step the cdf array of the steps needs to be defined')
            self._stepcdf = None
        return self._stepcdf
    
    def set_stepcdf(self, stepcdf):
        self._stepcdf = stepcdf
               
    @property
    def hubble_scatter(self):
        if not hasattr(self, "_hubble_scatter"):
            m = Minuit(get_loglikelihood_hubble(self.data_cut['redshift'], self.mag, self.mag_err), 
                   H0=100, sig=0.5)
            m.errordef = Minuit.LIKELIHOOD
            m.migrad()
            self._hubble_scatter = m.values[1]
        return self._hubble_scatter
    
    def set_hubble_scatter(self, hubble_scatter):
        self._hubble_scatter = hubble_scatter 
        
    @property
    def sn_scatter(self):
        if not hasattr(self, "_sn_scatter"):
            self.full_fit()
        return self._sn_scatter
    
    def set_sn_scatter(self, sn_scatter):
        self._sn_scatter = sn_scatter 
    
    @property
    def chi2(self):
        if not hasattr(self, "_chi2"):
            self.full_fit()
        return self._chi2
    
    def set_chi2(self, chi2):
        self._chi2 = chi2
        
    @property
    def cov(self):
        """
        Covariance matrix between observed magnitude, stretch and color. The last index represents the number of observations
        """
        if not hasattr(self, "_cov"):
            cov1 = np.zeros((3, 3, len(self.res)))
            cov1[0,0] = np.array(self.data['mag_err']**2)
            cov1[1,1] = np.array(self.data['x1_err']**2)
            cov1[2,2] = np.array(self.data['c_err']**2)
            cov1[0,1] = -2.5*np.array(self.data['cov_x0_x1'])/(np.log(10)*self.data['x0'])
            cov1[1,0] = cov1[0,1]
            cov1[0,2] = -2.5*np.array(self.data['cov_x0_c'])/(np.log(10)*self.data['x0'])
            cov1[2,0] = cov1[0,2]
            cov1[1,2] = np.array(self.data['cov_x1_c'])
            cov1[2,1] = cov1[1,2]
            self._cov = np.moveaxis(cov1, [2], [0])
        return self._cov
    
    def set_cov(self, cov):
        self._cov = cov 

    @property
    def fit_method(self):
        if not hasattr(self, "_fit_method"):
            self._fit_method = "standax"
        return self._fit_method
    
    def set_fit_method(self, fit_method):
        self._fit_method = fit_method
        
    @property
    def cov_params(self):
        if not hasattr(self, "_cov_params"):
            self.full_fit()
        return self._cov_params
    
    def set_cov_params(self, cov_params):
        self._cov_params = cov_params

    @property
    def smooth_step(self):
        if not hasattr(self, "_smooth_step"):
            self._smooth_step = False
        return self._smooth_step
    
    def set_smooth_step(self, smooth_step):
        self._smooth_step = smooth_step

    @property
    def loglikelihood(self):
        if not hasattr(self, "_loglikelihood"):
            self.full_fit()
        return self._loglikelihood
    
    def set_loglikelihood(self, loglikelihood):
        self._loglikelihood = loglikelihood

    @property
    def x1break(self):
        if not hasattr(self, "_x1break"):
            self.fit_broken_alpha()
        return self._x1break
    
    def set_x1break(self, x1break):
        self._x1break = x1break

    @property
    def x1break_err(self):
        if not hasattr(self, "_x1break_err"):
            self.fit_broken_alpha()
        return self._x1break_err
    
    def set_x1break_err(self, x1break_err):
        self._x1break_err = x1break_err


    @property
    def broken_alpha(self):
        if not hasattr(self, "_broken_alpha"):
            self._broken_alpha = False
        return self._broken_alpha
    
    def set_broken_alpha(self, broken_alpha):
        self._broken_alpha = broken_alpha



def plot_residuals(redshift, residuals, residuals_err, color='tab:blue', color_err='lightblue', xaxis='', s=1, alpha=1):
    plt.errorbar(redshift, residuals, yerr=residuals_err, fmt='none', markersize=1, color=color_err, elinewidth=1, zorder=1,
                alpha=alpha)
    plt.scatter(redshift, residuals, marker='x', s=2, color=color, zorder=10, alpha=alpha)
    plt.plot([min(min(redshift),0), max(redshift)], [0, 0], color='Grey', alpha=0.5)
    plt.xlabel(xaxis)
    plt.ylabel('Hubble residuals')


def get_mean_std_qcut(res, param, nbins):
    labels, bins = pandas.qcut(param, nbins, labels=[str(k) for k in range(1,nbins+1)], retbins=True)
    center_bins = []
    means = []
    stdev = []
    for k in range(1,nbins+1):
        mask_qcut = labels == str(k)
        means.append(np.mean(res[mask_qcut]))
        stdev.append(np.std(res[mask_qcut], ddof=1))
        center_bins.append(np.mean(param[mask_qcut]))
    return (means, stdev, center_bins)

def plot_mean_std(res, res_err, param, nbins, title='', errors=True):
    (means, stdev, center_bins) = get_mean_std_qcut(res, param, nbins)
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.errorbar(param, res, yerr=res_err, fmt='none', markersize=1, color='lightblue', elinewidth=1, zorder=1)
    plt.scatter(param, res, marker='x', s=2, color='lightblue', zorder=10)
    plt.scatter(center_bins, means, marker='x', color='teal', zorder=10)
    if errors==True:
        plt.errorbar(center_bins, means, yerr=stdev/np.sqrt(len(param)), fmt='none')
    plt.plot([min(param)-0.01, max(param)+0.01], [0, 0], color='lightseagreen', alpha=0.3)
    plt.xlabel(title)
    plt.ylabel('Mean rediduals')
    plt.subplot(122)
    plt.scatter(center_bins, stdev, marker='x', color='teal')
    if errors==True:
        plt.errorbar(center_bins, stdev, yerr=stdev/np.sqrt(2*(len(param)-1)), fmt='none')
    plt.plot([min(param)-0.01, max(param)+0.01], [np.std(res), np.std(res)], color='lightseagreen', alpha=0.3)
    plt.xlabel(title)
    plt.ylabel('Standard deviation of rediduals')
    plt.show()
    
def plot_residuals_std(res, res_err, xtext=2, ytext=1.7):
    box = {'facecolor': 'none', 'edgecolor': 'black', 'boxstyle': 'round'}
    std = np.std(res)
    std_mad = astropy.stats.mad_std(res)
    moy = np.mean(res)
    xres = np.linspace(min(res)*1.5, max(res)*1.5, 200)
    alldata.plot_data(xres, res, res_err)
    plt.plot(xres, norm.pdf(xres, loc=moy, scale=std))
    plt.plot(xres, norm.pdf(xres, loc=moy, scale=std_mad))
    plt.xlabel('Residuals')
    t = '$\\sigma=$'+str(round(std, 3))+'\n$\\sigma_{MAD}=$'+str(round(std_mad, 3))
    plt.text(xtext, ytext, t, bbox=box, fontsize='large')

def plot_param(param, param_err, xlin, color_line='tab:blue', color_dot='lightblue', title='', dot_max=False):
    plt.plot(xlin, norm.pdf(xlin, param, param_err), label=title, color=color_line)
    if dot_max == False:
        dot_max = max(norm.pdf(xlin, param, param_err))*1.2
    plt.plot([param-param_err, param-param_err], [0, dot_max], linestyle='dotted', color=color_dot)
    plt.plot([param+param_err, param+param_err], [0, dot_max], linestyle='dotted', color=color_dot)


def line(data, a):
    return data*a

def affine(data, a, b):
    return data*a+b

def poly(data, a, b, c):
    return data*a+b+c*data**2

def plot_hubble(redshift, mag, mag_err, H0, Om, title=''):
    xred = np.linspace(min(redshift)*0.95, max(redshift)*1.05, 200)
    plt.errorbar(redshift, mag, yerr=mag_err, fmt='none', markersize=1, color='lightblue', elinewidth=1, zorder=1)
    plt.scatter(redshift, mag, marker='x', s=2, color='tab:blue', zorder=10)
    cosmo = cosmology.FlatLambdaCDM(H0, Om)
    plt.plot(xred, cosmo.distmod(xred), label='Hubble fit', color='tab:orange')
    plt.ylabel('Mag')
    plt.xlabel('Redshift $z$')
