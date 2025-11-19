"""
    bi_dist_asymmetric_laplace(; loc=0.0, scale=1.0, asymmetry=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Asymmetric Laplace distribution
Samples from an Asymmetric Laplace distribution.
The Asymmetric Laplace distribution is a generalization of the Laplace distribution,
where the two sides of the distribution are scaled differently. It is defined by
a location parameter (loc), a scale parameter (scale), and an asymmetry parameter (asymmetry).
- `loc`: A numeric vector or single numeric value representing the location parameter of the distribution. This corresponds to \eqn{\mu}.
- `scale`: A numeric vector or single numeric value representing the scale parameter of the distribution. This corresponds to \eqn{\sigma}.
- `asymmetry`: A numeric vector or single numeric value representing the asymmetry parameter of the distribution. This corresponds to \eqn{\kappa}.
- `shape`: A numeric vector specifying the shape of the output.  This is used to set the batch shape when \code{sample=FALSE} (model building) or as `sample_shape` to draw a raw JAX array when \code{sample=TRUE} (direct sampling).
- `event`: Integer specifying the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector indicating which observations to mask.
- `create_obj`: Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
When ``sample=TRUE`: A JAX array of samples drawn from the AsymmetricLaplace distribution (for direct sampling).
When ``create_obj=TRUE`: The raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.asymmetric_laplace(sample = TRUE)
`
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#asymmetriclaplace`
"""
function bi_dist_asymmetric_laplace(; 
    loc=0.0,
    scale=1.0,
    asymmetry=1.0,
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
    jnp_asymmetry = JNP[].array(asymmetry)

    return BI_INSTANCE[].dist.asymmetric_laplace(
        loc=jnp_loc,
        scale=jnp_scale,
        asymmetry=jnp_asymmetry,
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
