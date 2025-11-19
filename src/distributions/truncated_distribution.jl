"""
    bi_dist_truncated_distribution(; base_dist, low=nothing, high=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Truncated Distribution
Samples from a Truncated Distribution.
This distribution represents a base distribution truncated between specified lower and upper bounds.
The truncation modifies the probability density function (PDF) of the base distribution,
effectively removing observations outside the defined interval.
Equation: `p(x) = \frac`p(x)``P(\text`lower` \le x \le \text`upper`)``
- `base_dist`: The base distribution to be truncated. This should be a univariate
distribution. Currently, only the following distributions are supported:
Cauchy, Laplace, Logistic, Normal, and StudentT.
- `low`: (float, jnp.ndarray, optional): The lower truncation point. If `None`, the distribution is only truncated on the right. Defaults to `None`.
- `high`: (float, jnp.ndarray, optional): The upper truncation point. If `None`, the distribution is only truncated on the left. Defaults to `None`.
- `shape`: A numeric vector (e.g., `c(10)`) specifying the shape. When \code{sample=FALSE}
(model building), this is used with `.expand(shape)` to set the distribution's
batch shape. When ``sample=TRUE` (direct sampling), this is used as `sample_shape`
to draw a raw JAX array of the given shape.
- `event`: The number of batch dimensions to reinterpret as event dimensions
(used in model building).
- `mask`: An optional boolean array to mask observations.
- `create_obj`: Logical; If `TRUE`, returns the raw BI distribution object
instead of creating a sample site. This is essential for building complex distributions
like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Truncated distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Truncated distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#truncateddistribution`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.truncated_distribution(
base_dist = bi.dist.normal(0,1, create_obj = TRUE),
high = 0.7,
low = 0.1,
sample = TRUE)
`
"""
function bi_dist_truncated_distribution(; 
    base_dist,
    low=nothing,
    high=nothing,
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

    return BI_INSTANCE[].dist.truncated_distribution(
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
