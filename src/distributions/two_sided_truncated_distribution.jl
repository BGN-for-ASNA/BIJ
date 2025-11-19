"""
    bi_dist_two_sided_truncated_distribution(; base_dist, low=0.0, high=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Two-Sided Truncated Distribution
This distribution truncates a base distribution between two specified lower and upper bounds.
Equation: `f(x) = \begin`cases`
\frac`p(x)``P(\text`low` \le X \le \text`high`)` & \text`if ` \text`low` \le x \le \text`high` \\
0 & \text`otherwise`
\end`cases``
where Equation: `p(x)` is the probability density function of the base distribution.
- `base_dist`: The base distribution to truncate.
- `low`: The lower bound for truncation.
- `high`: The upper bound for truncation.
- `sample`: Logical; if `TRUE`, returns JAX array of samples.  Defaults to `FALSE`.
- `shape`: A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector to mask observations.
- `create_obj`: Logical; if `TRUE`, returns the raw BI distribution object. Defaults to `FALSE`.
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
- When ``sample=FALSE`, a BI Two-Sided Truncated distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Two-Sided Truncated distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#twosidedtruncateddistribution`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.two_sided_truncated_distribution(
base_dist = bi.dist.normal(0,1, create_obj = TRUE),
high = 0.5, low = 0.1, sample = TRUE)
`
"""
function bi_dist_two_sided_truncated_distribution(; 
    base_dist,
    low=0.0,
    high=1.0,
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
    jnp_high = JNP[].array(high)

    return BI_INSTANCE[].dist.two_sided_truncated_distribution(
        base_dist=jnp_base_dist,
        low=jnp_low,
        high=jnp_high,
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
