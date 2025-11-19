"""
    bi_dist_gaussian_copula_beta(; concentration1, concentration0, correlation_matrix=nothing, correlation_cholesky=nothing, validate_args=false, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Gaussian Copula Beta distribution.
This distribution combines a Gaussian copula with a Beta distribution.
The Gaussian copula models the dependence structure between random variables,
while the Beta distribution defines the marginal distributions of each variable.
- `concentration1`: A numeric vector or matrix representing the first shape parameter of the Beta distribution.
- `concentration0`: A numeric vector or matrix representing the second shape parameter of the Beta distribution.
- `correlation_matrix`: array_like, optional: Correlation matrix of the coupling multivariate normal distribution. Defaults to `reticulate::py_none()`.
- `correlation_cholesky`: A numeric vector, matrix, or array representing the Cholesky decomposition of the correlation matrix.
- `shape`: A numeric vector.  This is used as `sample_shape` to draw a raw JAX array of the given shape when `sample=True`.
- `event`: Integer indicating the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector. Optional boolean array to mask observations.
- `create_obj`: Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
- When ``sample=FALSE`, a BI Gaussian Copula Beta distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Gaussian Copula Beta distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#gaussiancopulabetadistribution`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.gaussian_copula_beta(
concentration1 = c(2.0, 3.0),
concentration0 = c(5.0, 3.0),
correlation_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
sample = TRUE)
`
"""
function bi_dist_gaussian_copula_beta(; 
    concentration1,
    concentration0,
    correlation_matrix=nothing,
    correlation_cholesky=nothing,
    validate_args=false,
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

    jnp_concentration1 = JNP[].array(concentration1)
    jnp_concentration0 = JNP[].array(concentration0)
    jnp_correlation_matrix = JNP[].array(correlation_matrix)
    jnp_correlation_cholesky = JNP[].array(correlation_cholesky)

    return BI_INSTANCE[].dist.gaussian_copula_beta(
        concentration1=jnp_concentration1,
        concentration0=jnp_concentration0,
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
