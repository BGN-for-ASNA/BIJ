"""
    bi_dist_weibull(; scale, concentration, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Weibull distribution.
The Weibull distribution is a versatile distribution often used to model failure rates in engineering and reliability studies. It is characterized by its shape and scale parameters.
Equation: `f(x) = \frac`\beta``\alpha` \left(\frac`x``\alpha`\right)^`\beta - 1` e^`-\left(\frac`x``\alpha`\right)^`\beta`` \text` for ` x \ge 0`
where Equation: `\alpha` is the scale parameter and \eqn`\beta` is the shape parameter.
- `scale`: A numeric vector, matrix, or array representing the scale parameter of the Weibull distribution. Must be positive.
- `concentration`: A numeric vector, matrix, or array representing the shape parameter of the Weibull distribution. Must be positive.
- `shape`: A numeric vector.  This is used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
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
- When ``sample=FALSE`, a BI Weibull distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Weibull distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#weibull`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.weibull(scale = c(10, 10), concentration = c(1,1), sample = TRUE)
`
"""
function bi_dist_weibull(; 
    scale,
    concentration,
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

    jnp_scale = JNP[].array(scale)
    jnp_concentration = JNP[].array(concentration)

    return BI_INSTANCE[].dist.weibull(
        scale=jnp_scale,
        concentration=jnp_concentration,
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
