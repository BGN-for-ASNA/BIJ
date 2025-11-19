"""
    bi_dist_low_rank_multivariate_normal(; loc, cov_factor, cov_diag, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Low Rank Multivariate Normal Distribution
The *Low-Rank Multivariate Normal* (LRMVN) distribution is a parameterization of the multivariate normal distribution where the covariance matrix is expressed as a low-rank plus diagonal decomposition:
Equation: `
\Sigma = F F^\top + D
`
where $F$ is a low-rank matrix (capturing correlations) and $D$ is a diagonal matrix (capturing independent noise). This representation is often used in probabilistic modeling and variational inference to efficiently handle high-dimensional Gaussian distributions with structured covariance.
- `loc`: A numeric vector representing the mean vector.
- `cov_factor`: A numeric vector or matrix used to construct the covariance matrix.
- `cov_diag`: A numeric vector representing the diagonal elements of the covariance matrix.
- `validate_args`: Logical: Whether to validate parameter values.  Defaults to `reticulate::py_none()`.
- `shape`: Numeric vector. A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: Logical vector. Optional boolean array to mask observations.
- `create_obj`: Logical. If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Low Rank Multivariate Normal distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Low Rank Multivariate Normal distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#lowrankmultivariatenormal`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
event_size = 10
rank = 5
bi.dist.low_rank_multivariate_normal(
loc = bi.dist.normal(0,1,shape = c(event_size), sample = TRUE)*2,
cov_factor = bi.dist.normal(0,1,shape = c(event_size, rank), sample = TRUE),
cov_diag = bi.dist.normal(10,0.5,shape = c(event_size), sample = TRUE),
sample = TRUE)
`
"""
function bi_dist_low_rank_multivariate_normal(; 
    loc,
    cov_factor,
    cov_diag,
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
    jnp_cov_factor = JNP[].array(cov_factor)
    jnp_cov_diag = JNP[].array(cov_diag)

    return BI_INSTANCE[].dist.low_rank_multivariate_normal(
        loc=jnp_loc,
        cov_factor=jnp_cov_factor,
        cov_diag=jnp_cov_diag,
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
