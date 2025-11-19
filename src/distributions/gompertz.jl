"""
    bi_dist_gompertz(; concentration, rate=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Gompertz Distribution
The Gompertz distribution is a distribution with support on the positive real line that is closely
related to the Gumbel distribution. This implementation follows the notation used in the Wikipedia
entry for the Gompertz distribution. See https://en.wikipedia.org/wiki/Gompertz_distribution.
- `concentration`: A positive numeric vector, matrix, or array representing the concentration parameter.
- `rate`: A positive numeric vector, matrix, or array representing the rate parameter.
- `shape`: A numeric vector representing the shape parameter.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions.
- `mask`: A boolean vector, matrix, or array representing an optional mask for observations.
- `create_obj`: Logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
- `validate_args`: Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
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
A BI Gompertz distribution object when ``sample=FALSE` (for model building).
A JAX array when ``sample=TRUE` (for direct sampling).
A BI distribution object when ``create_obj=TRUE` (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#gompertz`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.gompertz(concentration = 5., sample = TRUE)
`
"""
function bi_dist_gompertz(; 
    concentration,
    rate=1.0,
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

    jnp_concentration = JNP[].array(concentration)
    jnp_rate = JNP[].array(rate)

    return BI_INSTANCE[].dist.gompertz(
        concentration=jnp_concentration,
        rate=jnp_rate,
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
