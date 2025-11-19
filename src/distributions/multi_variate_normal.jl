"""
    bi_dist_multivariate_normal(; loc=0.0, covariance_matrix=nothing, precision_matrix=nothing, scale_tril=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Multivariate Normal distribution.
The Multivariate Normal distribution, also known as the Gaussian distribution in multiple dimensions,
is a probability distribution that arises frequently in statistics and machine learning. It is
defined by its mean vector and covariance matrix, which describe the central tendency and
spread of the distribution, respectively.
Equation: `p(x) = \frac`1``\sqrt`(2\pi)^n |\Sigma|`` \exp\left(-\frac`1``2`(x - \mu)^T \Sigma^`-1` (x - \mu)\right)`
where:
- \eqn`x` is a \eqn`n`-dimensional vector of random variables.
- \eqn`\mu` is the mean vector.
- \eqn`\Sigma` is the covariance matrix.
@importFrom reticulate py_none tuple
- `loc`: A numeric vector representing the mean vector of the distribution.
- `covariance_matrix`: A numeric vector, matrix, or array representing the covariance matrix of the distribution. Must be positive definite.
- `precision_matrix`: A numeric vector, matrix, or array representing the precision matrix (inverse of the covariance matrix) of the distribution. Must be positive definite.
- `scale_tril`: A numeric vector, matrix, or array representing the lower triangular Cholesky decomposition of the covariance matrix.
- `shape`: A numeric vector representing the shape of the distribution.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector representing an optional boolean array to mask observations.
- `create_obj`: Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
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
- When ``sample=FALSE`, a BI Multivariate Normal distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Multivariate Normal distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#multivariate-normal`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.multivariate_normal(
loc =  c(1.0, 0.0, -2.0),
covariance_matrix = matrix(
c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5),
nrow = 3, byrow = TRUE),
sample = TRUE)
`
"""
function bi_dist_multivariate_normal(; 
    loc=0.0,
    covariance_matrix=nothing,
    precision_matrix=nothing,
    scale_tril=nothing,
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
    jnp_covariance_matrix = JNP[].array(covariance_matrix)
    jnp_precision_matrix = JNP[].array(precision_matrix)
    jnp_scale_tril = JNP[].array(scale_tril)

    return BI_INSTANCE[].dist.multivariate_normal(
        loc=jnp_loc,
        covariance_matrix=jnp_covariance_matrix,
        precision_matrix=jnp_precision_matrix,
        scale_tril=jnp_scale_tril,
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
