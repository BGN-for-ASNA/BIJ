"""
    bi_dist_beta_proportion(; mean, concentration, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Beta-Proportion distribution.
The Beta Proportion distribution is a reparameterization of the conventional
Beta distribution in terms of a the variate mean and a precision parameter. It's useful for modeling rates and proportions. It's essentially the same family as the standard Equation: `Beta(\alpha,\beta)`, but the mapping is:
Equation: ` \alpha = \mu , \kappa,\quad \beta = (1 - \mu), \kappa`.
- `mean`: A numeric vector, matrix, or array representing the mean of the BetaProportion distribution,
must be between 0 and 1.
- `concentration`: A numeric vector, matrix, or array representing the concentration parameter of the BetaProportion distribution.
- `shape`: A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the
distribution's batch shape. When `sample=True` (direct sampling),
this is used as `sample_shape` to draw a raw JAX array of the
given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event
dimensions (used in model building).
- `mask`: An optional boolean vector to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution
object instead of creating a sample site. This is essential for
building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Beta-Proportion distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Beta-Proportion distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.beta_proportion(0, 1, sample = TRUE)
`
"""
function bi_dist_beta_proportion(; 
    mean,
    concentration,
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

    jnp_mean = JNP[].array(mean)
    jnp_concentration = JNP[].array(concentration)

    return BI_INSTANCE[].dist.beta_proportion(
        mean=jnp_mean,
        concentration=jnp_concentration,
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
