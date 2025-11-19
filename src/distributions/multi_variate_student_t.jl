"""
    bi_dist_multivariate_student_t(; df, loc=0.0, scale_tril=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Multivariate Student's t Distribution
The Multivariate Student's t distribution is a generalization of the Student's t
distribution to multiple dimensions. It is a heavy-tailed distribution that is
often used to model data that is not normally distributed.
Equation: `p(x) = \frac`1``B(df/2, n/2)` \frac`\Gamma(df/2 + n/2)``\Gamma(df/2)`
\left(1 + \frac`(x - \mu)^T \Sigma^`-1` (x - \mu)``df`\right)^`-(df + n)/2``
- `df`: A numeric vector representing degrees of freedom, must be positive.
- `loc`: A numeric vector representing the location vector (mean) of the distribution.
- `scale_tril`: A numeric matrix defining the scale (lower triangular matrix).
- `shape`: A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector. Optional boolean array to mask observations.
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
- When ``sample=FALSE`, a BI Multivariate Student's t distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Multivariate Student's t distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#multivariatestudentt`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.multivariate_student_t(
df = 2,
loc =  c(1.0, 0.0, -2.0),
scale_tril = chol(
matrix(c( 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5),
nrow = 3, byrow = TRUE)),
sample = TRUE)
`
"""
function bi_dist_multivariate_student_t(; 
    df,
    loc=0.0,
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

    jnp_df = JNP[].array(df)
    jnp_loc = JNP[].array(loc)
    jnp_scale_tril = JNP[].array(scale_tril)

    return BI_INSTANCE[].dist.multivariate_student_t(
        df=jnp_df,
        loc=jnp_loc,
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
