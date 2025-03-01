import jax
import jax.numpy as jnp
import numpy as np

import warnings
# ============== #
#                #
#   Internal     #
#                #
# ============== #

def compute_cov_chi2(delta_y, flat_delta_x, cov_xx, cov_yy, cov_yx, sigmaint):
    """ fast chi2 and logdet computation from reshaped covariance matrix elements.

    Parameters
    ----------

    delta_y: jnp.array 
        residual between observed_y - model_y (N,)

    flat_delta_x: jnp.array 
        residual between observed_x - model_x (M*N)

    sigmaint: jnp.array
        intrinsic dispersion () or (N,)


    Returns
    -------
    chi2, logdet
    """

    C = jnp.block([
    [cov_yy + jnp.identity(len(delta_y))*sigmaint**2,               cov_yx],

    [cov_yx.T, cov_xx               ]])

    invC = jnp.linalg.inv(C)
    delta = jnp.concatenate([delta_y, flat_delta_x])
    sign, logdet = jnp.linalg.slogdet(invC)

    return jnp.matmul(delta.T,jnp.matmul(invC,delta)), -sign*logdet

def nodiag_elements( matrix2d ):
    """ """
    # numpy as strides
    matrix2d = np.asarray(matrix2d)
    m = matrix2d.shape[0]
    p, q = matrix2d.strides
    return np.lib.stride_tricks.as_strided(matrix2d[:,1:], (m-1,m), (p+q,q))

def is_matrix_diagonal( matrix2d ):
    """ test if the matrix is diagonal (all non-diagonal terms are null) """
    nodiags = nodiag_elements( matrix2d )    
    return (nodiags==0.).all()

# ============== #
#                #
#    I/O         #
#                #
# ============== #
def errors_to_diag_covmatrices(y_err, x_err):
    """ simplest implementation of the covariance matrix
    
    Parameters
    ----------
    y_err: array
        errors on the y-axis (N,)

    x_err: array
        errors on the x-axis (M, N)
    """
    ntargets = len(y_err)
    ncoefs = len(x_err)
    
    cov_yy = jnp.diag(jnp.asarray(y_err)**2)

    # cov_xx | fill with zero if 'mag_err' does not exist.
    cov_xx = jnp.diag( jnp.vstack(x_err).T.flatten()**2 ) # x1_err, c_err, x1_err, etc..

    # cov_yx | empty
    cov_yx = jnp.zeros( (ntargets, ncoefs*ntargets ) )

    return cov_yy, cov_yx, cov_xx

def dataframe_to_diag_covmatrices(data, ykey="mag", xkeys=["x1","c"], err_label="_err"):
    """ creates the cov_yy (diag), cov_xy (zeros) and cov_xx (diag) 
    matrices out of a dataframe.

    This follows the CovMatrix format.
    
    """
    # genrics 
    xkeys = np.atleast_1d(xkeys) # accepts xkey = str
    default_err = jnp.zeros( len(data) )
    
    # cov_yy | fill with zero if 'mag_err' does not exist.
    y_err = data.get(f"{ykey}{err_label}", default_err)
    # cov_xx | fill with zero if 'mag_err' does not exist.
    x_err = [jnp.asarray(data.get(f"{k}{err_label}", default_err )) for k in xkeys]
    
    return errors_to_diag_covmatrices(y_err, x_err)
    
def dataframe_to_covmatrices(data, ykey="mag", xkeys=["x1","c"], 
                             err_label="_err", cov_label="cov_",
                             diag_only=False,
                             **kwargs):
    """ creates the cov_yy, cov_xy and cov_xx matrices out of a dataframe

    Expected dataframe columns: format
    - {keyi}_err
    - cov_{keyi}_{keyj} | cov_{keyj}_{keyi}: the first found is used.
    

    Parameters
    ----------
    data: pandas.DataFrame
        pandas dataframe containing the data

    ykey: string
        name of the y-columns

    xkeys: list
        list of column containing the x-parameters

    err_label: string
        format for the error columns such that "{xkey}{err_label}"

    cov_label: string
        format for the covariance columns such that 
        "cov_{xkey1}{xkey2}"

    **kwargs goes to class' __init__

    Returns
    -------
    list
        list of three jax array
        - cov_yy: (N,N)
        - cov_xy: (N, N*M)
        - cov_xx: (N*M, N*M)

        ------------
        | yy | yx  |
        -----------|
             | xx  |
             |-----|

    """
    # should this be pure diag ?
    all_keys = np.append(np.atleast_1d(xkeys), ykey) # all input keys
    cov_entries = [f"{cov_label}{key1}_{key2}"
                       for key1 in all_keys
                       for key2 in all_keys
                    if key1!=key2]
    any_cov = np.any( data.columns.isin(cov_entries) )
    if diag_only or not any_cov:
        warnings.warn("diagonal covariance matrix, using dataframe_to_diag_covmatrices")
        # much faster
        return dataframe_to_diag_covmatrices(data, ykey=ykey, xkeys=xkeys, err_label=err_label)

    
    # -- Internal -- #
    def fill_diagonal(a, val):
        """ fast diagonal filler """
        assert a.ndim >= 2
        i, j = jnp.diag_indices(min(a.shape[-2:]))
        return a.at[..., i, j].set(val)
    # -------------- #
    
    # this method is sparse based, so ready for sparse analyses.    
    from jax.experimental import sparse

    data_ = data.copy() # work on the copy

    ntargets = len(data)
    mcoefs = len(xkeys)

    # make sure they are errors on x
    xerrkeys = [f"{k}_err" for k in xkeys]
    for k in xerrkeys:
        if k not in data_.columns:
            data_[k] = 0
    data_[xerrkeys] = data_[xerrkeys]**2 # variance
    
    #
    # cov_yy
    #
    y_err = data_.get(f"{ykey}{err_label}", np.zeros( ntargets))**2
    cov_yy = jnp.diag(jnp.asarray(y_err))

    #
    # cov_xy/cov_yx
    #
    keyscov_keys = [[i, covkey] 
            for i,xi in enumerate(xkeys)
            if (covkey := f"{cov_label}{xi}_{ykey}") in data_.columns or
               (covkey := f"{cov_label}{ykey}_{xi}") in data_.columns] 
            # never goes to second line if first accepted


    col_index = np.asarray([l[0] for l in keyscov_keys])
    cov_keys = [l[1] for l in keyscov_keys]


    row = np.arange(ntargets) * np.ones(mcoefs)[:,None]
    col = np.arange(ntargets)[:,None]*mcoefs + np.arange(mcoefs)
    covdata = np.zeros_like(col, dtype="float")
    covdata[:,col_index] = data[cov_keys].values
    coo_yx = sparse.COO((covdata.flatten(), 
                         row.T.flatten().astype(int), 
                         col.flatten()), 
                         shape=(ntargets,ntargets*mcoefs))
    cov_yx = coo_yx.todense()

    #
    # cov_xx
    #

    # make sure the x_err are in        
    keyscov_keys = [[(i,j),covkey] 
                    for i,xi in enumerate(xkeys)
                    for j,xj in enumerate(xkeys) 
                    if (covkey := f"{cov_label}{xj}_{xi}") in data_.columns]

    col_, row_ = np.asarray([l[0] for l in keyscov_keys]).T
    cov_keys = [l[1] for l in keyscov_keys]
    xerrkeys += cov_keys


    xdata = data_[xerrkeys].values
    # using sparse matrix tools
    xrow = np.append(np.arange(mcoefs),list(row_)) + np.arange(ntargets)[:,None]*mcoefs
    xcol = np.append(np.arange(mcoefs),list(col_)) + np.arange(ntargets)[:,None]*mcoefs
    coo_xx = sparse.COO( (xdata.flatten(), 
                       xrow.flatten(), 
                       xcol.flatten()), 
                       shape=(ntargets*mcoefs, ntargets*mcoefs) )

    # make it full and square per block
    dense_xx = coo_xx.todense()
    cov_xx = dense_xx + fill_diagonal(dense_xx, 0).T

    return cov_yy, cov_yx, cov_xx


# ============== #
#                #
#   Covariance   #
#                #
# ============== #

class CovMatrix( object ):
    
    def __init__(self, cov_yy=None, cov_yx=None, cov_xx=None):
        """ Covariance Matrix per block
        
        • ---- • ----- •
        •  yy  •  yx   •
        • ---- • ----- •
               •  xx   •
               • ----- •

        Parameters
        ----------
        cov_yy: array
            covariance matrix for the y-parameters (N, N)

        cov_yx: array
            covariance matrix between x- and y-parameters (N, N*M)

        cov_xx: array
            covariance matrix between x-parameters (M*N, M*N)
            
            
        """
        self.cov_yy = cov_yy
        self.cov_yx = cov_yx
        self.cov_xx = cov_xx

    @classmethod
    def from_data(cls, data,
                      ykey="mag", xkeys=["x1","c"],
                      **kwargs):
        """ loads the covariance matrix based on dataframe

        Parameters
        ----------
        data: pandas.DataFrame
            dataframe with the following format:
            - {keyi}_err
            - cov_{keyi}_{keyj} | cov_{keyj}_{keyi}: the first found is used.

        ykey: string
            name of the y-columns

        xkeys: list
            list of column containing the x-parameters

        **kwargs goes to dataframe_to_covmatrices

        Return
        ------
        self
        """
        cov_yy, cov_yx, cov_xx = dataframe_to_covmatrices(data,
                                                          ykey=ykey, xkeys=xkeys,
                                                          **kwargs)
        return cls(cov_yy=cov_yy, cov_yx=cov_yx, cov_xx=cov_xx)
 
    # =============== #
    #    Methods      #
    # =============== #
    def __call__(self, delta_y, delta_x, sigmaint=0, coefs=None, **kwargs):
        """ calls """
        chi2, logdet = self.get_chi2_logdet(delta_y, delta_x, sigmaint=sigmaint, **kwargs)
        if coefs:
            restriction = self.get_restriction(coefs=coefs, sigmaint=sigmaint)
        else:
            restriction = 0

        return chi2 + logdet + restriction
            
    def get(self, value, default=None):
        """ access instance variables. 
        This method follows the pytree/dict format 
        """
        return getattr(self, value, getattr(self, f"_{value}", default))

    def get_chi2_logdet(self, delta_y, delta_x, sigmaint=0):
        """ Compute the chi2 for a given residual vector 
        and the variable part of the logdet

        chi^2 evaluation is only O(N^2)
    
        Parameters
        ----------
        delta_y: jnp.array 
            residual between observed_y - model_y (N,)

        delta_x: jnp.array 
            residual between observed_x - model_x (M, N)

        sigmaint: float, jnp.array
            intrinsic dispersion () or (N,)

        Returns
        -------
        (chi2, logdet)
            - chi2: float
            - lodget: float
        """
        if self.is_diag:
            sigma2 = self.y_err**2 + sigmaint**2
            chi2_y = jnp.sum( delta_y**2 /sigma2 )
            chi2_x = jnp.sum( (delta_x/self.x_err)**2 )
            # chi2 & logdet
            chi2 = chi2_x + chi2_y
            logdet = jnp.sum( jnp.log(sigma2) ) 

        else:
            # F as x1_vec, x2_vec because (x1_0, x2_0, x1_1, x1_1 etc.)
            flat_delta_x = delta_x.ravel("F")
            chi2, logdet = compute_cov_chi2(delta_y=delta_y, 
                                            flat_delta_x=flat_delta_x, 
                                            cov_xx=self.cov_xx, 
                                            cov_yy=self.cov_yy, 
                                            cov_yx=self.cov_yx, 
                                            sigmaint=sigmaint)
            #chi2, logdet = compute_fastcov_chi2(q=self._q,
            #                                    lambda_=self._lambda,
            #                                    inv_cxx=self._inv_cxx, 
            #                                    cov_yx=self.cov_yx,
            #                                    delta_y=delta_y, #(N,)
            #                                    delta_x=flat_delta_x, #(N*M,)
            #                                    sigmaint=sigmaint)
        return chi2, logdet

    def get_restriction(self, coefs, sigmaint=0):
        """ get restriction term of the likelihood 

        Parameters
        ----------
        coefs: array
            array of coefficient (M,)

        sigmaint: float, array (N,)
            intrinsic dispersion if any

        Returns
        -------
        rest 
            - float
        """
        sigma2 = self.y_err**2 + sigmaint**2
        rest = jnp.log(1/self.x_err**2 + coefs[:,None]**2/sigma2 )
        return jnp.sum(rest)
        
    # =============== #
    #   Properties    #
    # =============== #
    # - shape
    @property
    def ntargets(self):
        """ number of targets '(N,)' """
        return self.cov_yy.shape[0]

    @property
    def ncoefs(self):
        """ number of coefficients '(M,)' """
        yxshape = self.cov_yx.shape
        return int( yxshape[1] / yxshape[0] )
        
    # - shortcut
    @property
    def x_err(self):
        """ sqrt of cov_xx diag reshaped (M, N) """
        x_err_flat = jnp.sqrt( jnp.diag(self.cov_xx) )
        return x_err_flat.reshape(self.ntargets, self.ncoefs).T
   
    @property
    def y_err(self):
        """ sqrt of cov_yy diag (N,) """
        return jnp.sqrt( jnp.diag(self.cov_yy) )

    @property
    def is_diag(self):
        """ this is true if cov_yy and cov_xx are diagonal and cov_yx is zeros """
        if not hasattr(self, "_is_diag") or self._is_diag is None:
            # check if diagonal as lazy property
            is_x_diag = is_matrix_diagonal(self.cov_xx)
            is_y_diag = is_matrix_diagonal(self.cov_yy)
            is_xy_null = bool( (np.array(self.cov_yx) == 0).all() )
            self._is_diag = is_x_diag & is_y_diag & is_xy_null
            
        return self._is_diag
