"""
    bi_dist_gamma_poisson(; concentration, rate=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Gamma-Poisson Distribution
The Gamma-Poisson distribution, also known as the Negative Binomial distribution, models overdispersed count data. It arises from a hierarchical process where the rate parameter of a Poisson distribution is itself a random variable following a Gamma distribution. This structure allows the model to capture variability in count data that exceeds what is predicted by the Poisson distribution, making it suitable for applications like modeling RNA-sequencing data and microbial count.
- `concentration`: A numeric vector, matrix, or array representing the shape parameter (alpha) of the Gamma distribution.
- `rate`: A numeric vector, matrix, or array representing the rate parameter (beta) for the Gamma distribution.
- `shape`: A numeric vector used to shape the distribution. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: An optional boolean vector to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
- When ``sample=FALSE`, a BI Gamma-Poisson distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Gamma-Poisson distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#gammapoisson`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.gamma_poisson(concentration = 1, sample = TRUE)
`
"""
function bi_dist_gamma_poisson(; 
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

    return BI_INSTANCE[].dist.gamma_poisson(
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
