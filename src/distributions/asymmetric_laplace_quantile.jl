"""
    bi_dist_asymmetric_laplace_quantile(; loc=0.0, scale=1.0, quantile=0.5, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Asymmetric Laplace Quantile Distribution
Samples from a Bernoulli distribution parameterized by logits.
The Bernoulli distribution models a single binary event (success or failure),
parameterized by the log-odds ratio of success.  The probability of success
is given by the sigmoid function applied to the logit.
Equation: `f(x) = \frac`1``2 \sigma` \exp\left(-\frac`|x - \mu|``\sigma` \frac`1``q-1`\right) \left(1 - \frac`1``2q`\right)`
- `loc`: The location parameter of the distribution.
- `scale`: The scale parameter of the distribution.
- `quantile`: The quantile parameter, representing the proportion of
probability density to the left of the median. Must be between 0 and 1.
- `shape`: A numeric vector. When `sample=False` (model building), this is
used with `.expand(shape)` to set the distribution's batch shape. When
`sample=True` (direct sampling), this is used as `sample_shape` to draw a
raw JAX array of the given shape.
- `event`: The number of batch dimensions to reinterpret as event
dimensions (used in model building).
- `mask`: An optional boolean array to mask observations.
- `validate_args`: Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
- `create_obj`: If `TRUE`, returns the raw NumPyro distribution object
instead of creating a sample site. This is essential for building complex
distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Asymmetric Laplace Quantile distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Asymmetric Laplace Quantile distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplacequantile`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.asymmetric_laplace_quantile(sample = TRUE)
`
"""
function bi_dist_asymmetric_laplace_quantile(; 
    loc=0.0,
    scale=1.0,
    quantile=0.5,
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
    jnp_quantile = JNP[].array(quantile)

    return BI_INSTANCE[].dist.asymmetric_laplace_quantile(
        loc=jnp_loc,
        scale=jnp_scale,
        quantile=jnp_quantile,
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
