"""
    bi_dist_chi2(; df, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Chi-squared distribution.
The Chi-squared distribution is a continuous probability distribution that arises
frequently in hypothesis testing, particularly in ANOVA and chi-squared tests.
It is defined by a single positive parameter, degrees of freedom (df), the number of independent standard normal variables
squared and summed, which determines the shape of the distribution.
- `df`: A numeric vector representing the degrees of freedom. Must be positive.
- `shape`: A numeric vector used for shaping. When `sample=FALSE` (model building),
this is used with `.expand(shape)` to set the distribution's batch shape.
When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event
dimensions (used in model building).
- `mask`: A logical vector, matrix, or array to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object
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
- When ``sample=FALSE`, a BI Chi-squared distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Chi-squared distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#chi2`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.chi2(c(0,2),sample = TRUE)
`
"""
function bi_dist_chi2(; 
    df,
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

    jnp_df = JNP[].array(df)

    return BI_INSTANCE[].dist.chi2(
        df=jnp_df,
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
