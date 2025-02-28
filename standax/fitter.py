import jax
import optax

def fit_adam(func, init_params,
             learning_rate=2e-3, niter=1500, 
             tol=1e-3,
             **kwargs):
    """ simple Adam gradient descent using optax.adam

    Parameters
    ----------
    func: function
        function to minimize. Should return a float.

    learning_rate: float
        learning rate of the gradient descent.
        (careful, results can be sensitive to this parameter)p
        
    init_param: 
        entry parameter of the input func

    niter: int
        maximum number of iterations

    tol: float
        targeted func variations below which the iteration will stop

    **kwargs other func entries 

    Returns
    -------
    list
        - best parameters
        - loss (array)        
    """

    # handle kwargs more easily
    func_ = lambda x: func(x, **kwargs)
    
    # Initialize the adam optimizer
    params = init_params
    optimizer = optax.adam(learning_rate)

    # Obtain the `opt_state` that contains statistics for the optimizer.
    opt_state = optimizer.init(params)
    
    grad_func = jax.jit(jax.grad( func_ )) # get the derivative
    
    # and do the gradient descent
    losses = []
    for i in range(niter):
        current_grads = grad_func(params)
        updates, opt_state = optimizer.update(current_grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append( func_(params) ) # store the loss function
        if tol is not None and (i>2 and ((losses[-2] - losses[-1]) < tol)):
            break
            
    return params, losses
