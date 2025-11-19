"""
    bi_dist_levy(; loc, scale, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Levy distribution.
Samples from a Levy distribution.
The probability density function is given by,
Equation: `f(x\mid \mu, c) = \sqrt`\frac`c``2\pi(x-\mu)^`3``` \exp\left(-\frac`c``2(x-\mu)`\right), \qquad x > \mu`
where Equation: `\mu` is the location parameter and Equation: `c` is the scale parameter.
- `loc`: A numeric vector, matrix, or array representing the location parameter.
- `scale`: A numeric vector, matrix, or array representing the scale parameter.
- `shape`: A numeric vector used for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector, matrix, or array to mask observations.
- `create_obj`: Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
- When ``sample=FALSE`, a BI Levy distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Levy distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#levy`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.levy( loc = 1, scale = 10,  sample = TRUE)
`
"""
function bi_dist_levy(; 
    loc,
    scale,
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

    return BI_INSTANCE[].dist.levy(
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
