"""
    bi_dist_student_t(; df, loc=0.0, scale=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Student's t-distribution.
The Student's t-distribution is a probability distribution that arises in hypothesis testing involving the mean of a normally distributed population when the population standard deviation is unknown. It is similar to the normal distribution, but has heavier tails, making it more robust to outliers.
Equation: `f(x) = \frac`1``\Gamma(\nu/2) \sqrt`\nu \pi`` \left(1 + \frac`x^2``\nu`\right)^`-(\nu+1)/2``
- `df`: A numeric vector representing degrees of freedom, must be positive.
- `loc`: A numeric vector representing the location parameter, defaults to 0.0.
- `scale`: A numeric vector representing the scale parameter, defaults to 1.0.
- `shape`: A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector to mask observations.
- `create_obj`: Logical. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Student's t-distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Student's t-distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#studentt`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.student_t(df = 2, loc = 0, scale = 2, sample = TRUE)
`
"""
function bi_dist_student_t(; 
    df,
    loc=0.0,
    scale=1.0,
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
    jnp_loc = JNP[].array(loc)
    jnp_scale = JNP[].array(scale)

    return BI_INSTANCE[].dist.student_t(
        df=jnp_df,
        loc=jnp_loc,
        scale=jnp_scale,
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
