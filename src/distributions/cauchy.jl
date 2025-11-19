"""
    bi_dist_cauchy(; loc=0.0, scale=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Cauchy Distribution
Samples from a Cauchy distribution.
The Cauchy distribution, also known as the Lorentz distribution, is a continuous probability distribution
that arises frequently in various fields, including physics and statistics. It is characterized by its
heavy tails, which extend indefinitely. This means it has a higher probability of extreme values compared to the normal
distribution.
- `loc`: A numeric vector or scalar representing the location parameter. Defaults to 0.0.
- `scale`: A numeric vector or scalar representing the scale parameter. Must be positive. Defaults to 1.0.
- `shape`: A numeric vector specifying the shape of the distribution.  When `sample=False` (model building), this is used
with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
Defaults to `reticulate::py_none()`.
- `mask`: A logical vector, optional, to mask observations. Defaults to `reticulate::py_none()`.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample
site. Defaults to `FALSE`.
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
- When ``sample=FALSE`, a BI Cauchy distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Cauchy distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#cauchy`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.cauchy(sample = TRUE)
`
"""
function bi_dist_cauchy(; 
    loc=0.0,
    scale=1.0,
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

    jnp_loc = JNP[].array(loc)
    jnp_scale = JNP[].array(scale)

    return BI_INSTANCE[].dist.cauchy(
        loc=jnp_loc,
        scale=jnp_scale,
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
