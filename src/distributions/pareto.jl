"""
    bi_dist_pareto(; scale, alpha, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Pareto distribution.
The Pareto distribution is a power-law probability distribution that is often
used to model income, wealth, and the size of cities. It is defined by two
parameters: alpha (shape) and scale.
Equation: `f(x) = \frac`\alpha \cdot \text`scale`^`\alpha```x^`\alpha + 1`` \text` for ` x \geq \text`scale``
- `scale`: A numeric vector or single number representing the scale parameter of the Pareto distribution. Must be positive.
- `alpha`: A numeric vector or single number representing the shape parameter of the Pareto distribution. Must be positive.
- `shape`: A numeric vector. When \code{sample=FALSE} (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When \code{sample=TRUE} (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector. Optional boolean array to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Pareto distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Pareto distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#pareto`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.pareto(scale = c(0.2, 0.5, 0.8), alpha = c(-1.0, 0.5, 1.0), sample = TRUE)
`
"""
function bi_dist_pareto(; 
    scale,
    alpha,
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
    jnp_alpha = JNP[].array(alpha)

    return BI_INSTANCE[].dist.pareto(
        scale=jnp_scale,
        alpha=jnp_alpha,
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
