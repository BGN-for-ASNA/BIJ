"""
    bi_dist_lkj(; dimension, concentration=1.0, sample_method="onion", validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from an LKJ (Lewandowski, Kurowicka, Joe) distribution for correlation matrices.
The LKJ distribution is controlled by the concentration parameter \deqn{\eta} to make the probability
of the correlation matrix M proportional to Equation: `\det(M)^`\eta - 1``. When Equation: `\eta = 1`,
the distribution is uniform over correlation matrices.  When Equation: `\eta > 1`, the distribution favors
samples with large determinants. When Equation: `\eta < 1`, the distribution favors samples with small
determinants.
- `dimension`: An integer representing the dimension of the correlation matrices.
- `concentration`: A numeric vector representing the concentration/shape parameter of the distribution (often referred to as eta). Must be positive.
- `sample_method`: (str): Either "cvine" or "onion". Methods proposed offer the same distribution over correlation matrices. But they are different in how to generate samples. Defaults to "onion".
- `shape`: A numeric vector used for shaping. When `sample=False` (model building), this is used
with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling),
this is used as `sample_shape` to draw a raw JAX array of the given shape.
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
- When ``sample=FALSE`: A BI LKJ distribution object (for model building).
- When ``sample=TRUE`: A JAX array of samples drawn from the LKJ distribution (for direct sampling).
- When ``create_obj=TRUE`: The raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#lkj`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.lkj( dimension = 2, concentration=1.0, shape = c(1), sample = TRUE)
`
"""
function bi_dist_lkj(; 
    dimension,
    concentration=1.0,
    sample_method="onion",
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

    jnp_dimension = JNP[].array(dimension)
    jnp_concentration = JNP[].array(concentration)
    jnp_sample_method = JNP[].array(sample_method)

    return BI_INSTANCE[].dist.lkj(
        dimension=jnp_dimension,
        concentration=jnp_concentration,
        sample_method=jnp_sample_method,
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
