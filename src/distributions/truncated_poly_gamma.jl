"""
    bi_dist_truncated_polya_gamma(; batch_shape=(), validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Truncated PolyaGamma Distribution
Samples from a Truncated PolyaGamma distribution.
This distribution is a truncated version of the PolyaGamma distribution,
defined over the interval [0, truncation_point]. It is often used in
Bayesian non-parametric models.
Equation: `p(x) = \frac`1``Z` \exp\left( \sum_`n=0`^`N` \left( \log(2n+1) - 1.5 \log(x) - \frac`(2n+1)^2``4x` \right) \right)`
- `batch_shape`: A numeric vector specifying the shape of the batch dimension.
- `shape`: A numeric vector (e.g., `c(10)`) used to shape the distribution.
When `sample=False` (model building), this is used with `.expand(shape)` to set the
distribution's batch shape. When `sample=True` (direct sampling),
this is used as `sample_shape` to draw a raw JAX array of the
given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions.
- `mask`: A numeric vector, matrix, or array (e.g., a JAX array) of boolean values to mask observations.
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
- When ``sample=FALSE`, a BI Truncated PolyaGamma distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Truncated PolyaGamma distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.truncated_polya_gamma(batch_shape = c(), sample = TRUE)
`
"""
function bi_dist_truncated_polya_gamma(; 
    batch_shape=(),
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

    jnp_batch_shape = JNP[].array(batch_shape)

    return BI_INSTANCE[].dist.truncated_polya_gamma(
        batch_shape=jnp_batch_shape,
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
