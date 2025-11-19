"""
    bi_dist_euler_maruyama(; t, sde_fn, init_dist, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Euler-Maruyama method
Euler-Maruyama methode is a method for the approximate numerical solution
of a stochastic differential equation (SDE). It simulates the solution
to an SDE by iteratively applying the Euler method to each time step,
incorporating a random perturbation to account for the diffusion term.
Equation: `dX_t = f(X_t, t) dt + g(X_t, t) dW_t`
where:
- \eqn`X_t` is the state of the system at time \eqn`t`.
- \eqn`f(X_t, t)` is the drift coefficient.
- \eqn`g(X_t, t)` is the diffusion coefficient.
- \eqn`dW_t` is a Wiener process (Brownian motion).
- `t`: A numeric vector representing the discretized time steps.
- `sde_fn`: A function that takes the current state and time as input and returns the drift and diffusion coefficients.
- `init_dist`: The initial distribution of the system.
- `shape`: A numeric vector specifying the shape of the output tensor.  Defaults to `NULL`.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector. Optional boolean array to mask observations.
- `create_obj`: Logical. If TRUE, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
- `validate_args`: A logical value indicating whether to validate the arguments. Defaults to `TRUE`.
- `sample`: A logical value that controls the function's behavior. If `TRUE`,
the function will directly draw samples from the distribution. If `FALSE`,
it will create a random variable within a model. Defaults to `FALSE`.
- `seed`: An integer used to set the random seed for reproducibility when
`sample = TRUE`. This argument has no effect when `sample = FALSE`, as
randomness is handled by the model's inference engine. Defaults to 0.
- `obs`: A numeric vector or array of observed values. If provided, the
random variable is conditioned on these values. If `NULL`, the variable is
treated as a latent (unobserved) variable. Defaults to `NULL`.
- `name`: A character string representing the name of the random variable
within a model. This is used to uniquely identify the variable. Defaults to 'x'.
# Returns
- When ``sample=FALSE`, a BI Euler-Maruyama distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Euler-Maruyama distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
ornstein_uhlenbeck_sde <- function(x, t) `
# This function models dX = -theta * X dt + sigma dW
theta <- 1.0
sigma <- 0.5
drift <- -theta * x
diffusion <- sigma
# Return a list of two elements: drift and diffusion
# reticulate will convert this to a Python tuple
return(list(drift, diffusion))
`
bi.dist.euler_maruyama(
t=c(0.0, 0.1, 0.2),
sde_fn = ornstein_uhlenbeck_sde,
init_dist=bi.dist.normal(0.0, 1.0, create_obj=TRUE),
sample = TRUE)
`
"""
function bi_dist_euler_maruyama(; 
    t,
    sde_fn,
    init_dist,
    validate_args=nothing,
    name="x",
    obs=nothing,
    mask=nothing,
    sample=false,
    seed=nothing,
    shape=(),
    event=0,
    create_obj=false,
    to_jax=true,
)
    if !isassigned(BI_INSTANCE)
        error("BayesianInference not initialized. Please call import_bi() first.")
    end

    # Convert shape to tuple if needed
    if shape isa AbstractVector
        py_shape = tuple(shape...)
    else
        py_shape = shape
    end

    jnp_t = JNP[].array(t)
    jnp_sde_fn = JNP[].array(sde_fn)
    jnp_init_dist = JNP[].array(init_dist)

    return BI_INSTANCE[].dist.euler_maruyama(
        t=jnp_t,
        sde_fn=jnp_sde_fn,
        init_dist=jnp_init_dist,
        validate_args=validate_args,
        name=name,
        obs=obs,
        mask=mask,
        sample=sample,
        seed=seed,
        shape=py_shape,
        event=event,
        create_obj=create_obj,
        to_jax=to_jax,
    )
end
