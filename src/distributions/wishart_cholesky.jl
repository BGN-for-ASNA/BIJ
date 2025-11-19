"""
    bi_dist_wishart_cholesky(; concentration, scale_matrix=nothing, rate_matrix=nothing, scale_tril=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Wishart Cholesky Distribution
The Wishart distribution is a multivariate distribution used as a prior distribution
for covariance matrices. This implementation represents the distribution in terms
of its Cholesky decomposition.
WishartCholesky Distribution
The Wishart distribution is a multivariate distribution used as a prior distribution
for covariance matrices. This implementation represents the distribution in terms
of its Cholesky decomposition.
- `concentration`: (numeric or vector) Positive concentration parameter analogous to the
concentration of a `Gamma` distribution. The concentration must be larger
than the dimensionality of the scale matrix.
- `scale_matrix`: (numeric vector, matrix, or array, optional) Scale matrix analogous to the inverse rate of a `Gamma`
distribution. If not provided, `rate_matrix` or `scale_tril` must be.
- `rate_matrix`: (numeric vector, matrix, or array, optional) Rate matrix anaologous to the rate of a `Gamma`
distribution. If not provided, `scale_matrix` or `scale_tril` must be.
- `scale_tril`: (numeric vector, matrix, or array, optional) Cholesky decomposition of the `scale_matrix`.
If not provided, `scale_matrix` or `rate_matrix` must be.
- `shape`: A numeric vector.  This is used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
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
- When ``sample=FALSE`, a BI Wishart Cholesky  distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Wishart Cholesky  distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.wishart_cholesky(
concentration = 5,
scale_matrix = matrix(c(1,0,0,1),
nrow = 2),
sample = TRUE)
`
"""
function bi_dist_wishart_cholesky(; 
    concentration,
    scale_matrix=nothing,
    rate_matrix=nothing,
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

    jnp_concentration = JNP[].array(concentration)
    jnp_scale_matrix = JNP[].array(scale_matrix)
    jnp_rate_matrix = JNP[].array(rate_matrix)
    jnp_scale_tril = JNP[].array(scale_tril)

    return BI_INSTANCE[].dist.wishart_cholesky(
        concentration=jnp_concentration,
        scale_matrix=jnp_scale_matrix,
        rate_matrix=jnp_rate_matrix,
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
