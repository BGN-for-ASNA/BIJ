"""
    bi_dist_gaussian_copula(; marginal_dist, correlation_matrix=nothing, correlation_cholesky=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Gaussian Copula Distribution
A distribution that links the `batch_shape[:-1]` of a marginal distribution with a multivariate Gaussian copula,
modelling the correlation between the axes. A copula is a multivariate distribution over the uniform distribution
on [0, 1]. The Gaussian copula links the marginal distributions through a multivariate normal distribution.
- `marginal_dist`: Distribution: Distribution whose last batch axis is to be coupled.
- `correlation_matrix`: array_like, optional: Correlation matrix of the coupling multivariate normal distribution. Defaults to `reticulate::py_none()`.
- `correlation_cholesky`: array_like, optional: Correlation Cholesky factor of the coupling multivariate normal distribution. Defaults to `reticulate::py_none()`.
- `shape`: numeric vector: A multi-purpose argument for shaping. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: int: The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: jnp.ndarray, bool, optional: Optional boolean array to mask observations. Defaults to `reticulate::py_none()`.
- `create_obj`: bool, optional: If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`. Defaults to `FALSE`.
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
- When ``sample=FALSE`, a BI Gaussian Copula distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Gaussian Copula distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.gaussian_copula(
marginal_dist = bi.dist.gamma(concentration = 1 ,  create_obj = TRUE) ,
correlation_matrix =  matrix(c(1.0, 0.7, 0.7, 1.0),, nrow = 2, byrow = TRUE),
sample = TRUE)
`
"""
function bi_dist_gaussian_copula(; 
    marginal_dist,
    correlation_matrix=nothing,
    correlation_cholesky=nothing,
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

    jnp_marginal_dist = JNP[].array(marginal_dist)
    jnp_correlation_matrix = JNP[].array(correlation_matrix)
    jnp_correlation_cholesky = JNP[].array(correlation_cholesky)

    return BI_INSTANCE[].dist.gaussian_copula(
        marginal_dist=jnp_marginal_dist,
        correlation_matrix=jnp_correlation_matrix,
        correlation_cholesky=jnp_correlation_cholesky,
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
