"""
    bi_dist_left_truncated_distribution(; base_dist, low=0.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a left-truncated distribution.
A left-truncated distribution is a probability distribution
obtained by restricting the support of another distribution
to values greater than a specified lower bound. This is useful
when dealing with data that is known to be greater than a certain value. All the "mass" below (or equal to) (a) is **excluded** (not just unobserved, but removed from the sample/analysis).
Equation: `f(x) = \begin`cases`
\frac`f(x)``P(X > \text`low`)` & \text`if ` x > \text`low` \\
0 & \text`otherwise`
\end`cases``
- `base_dist`: The base distribution to truncate. Must be univariate and have real support.
- `low`: The lower truncation bound. Values less than this are excluded from the distribution.
- `shape`: A numeric vector. When \code{sample=FALSE} (model building),
this is used with `.expand(shape)` to set the distribution's batch shape.
When ``sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw
JAX array of the given shape.
- `event`: The number of batch dimensions to reinterpret as event dimensions (used in model building).
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
- When ``sample=FALSE`: A BI LeftTruncatedDistribution distribution object (for model building).
- When ``sample=TRUE`: A JAX array of samples drawn from the LeftTruncatedDistribution distribution (for direct sampling).
- When ``create_obj=TRUE`: The raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#lefttruncateddistribution`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.left_truncated_distribution(
base_dist = bi.dist.normal(loc = 1, scale = 10 ,  create_obj = TRUE),
sample = TRUE)
`
"""
function bi_dist_left_truncated_distribution(; 
    base_dist,
    low=0.0,
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

    jnp_base_dist = JNP[].array(base_dist)
    jnp_low = JNP[].array(low)

    return BI_INSTANCE[].dist.left_truncated_distribution(
        base_dist=jnp_base_dist,
        low=jnp_low,
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
