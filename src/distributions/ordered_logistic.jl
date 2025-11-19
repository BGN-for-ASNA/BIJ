"""
    bi_dist_ordered_logistic(; predictor, cutpoints, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Ordered Logistic Distribution
A categorical distribution with ordered outcomes. This distribution represents the probability of an event falling into one of several ordered categories, based on a predictor variable and a set of cutpoints. The probability of an event falling into a particular category is determined by the number of categories above it.
Equation: `P(Y = k) = \begin`cases`
1 & \text`if ` k = 0 \\
\frac`1``k` & \text`if ` k > 0
\end`cases``
- `predictor`: A numeric vector, matrix, or array representing the prediction in real domain; typically this is output of a linear model.
- `cutpoints`: A numeric vector, matrix, or array representing the positions in real domain to separate categories.
- `shape`: A numeric vector used to shape the distribution. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
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
- When ``sample=FALSE`, a BI Ordered Logistic distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Ordered Logistic distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#orderedlogistic`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.ordered_logistic(predictor = c(0.2, 0.5, 0.8), cutpoints = c(-1.0, 0.0, 1.0), sample = TRUE)
`
"""
function bi_dist_ordered_logistic(; 
    predictor,
    cutpoints,
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

    jnp_predictor = JNP[].array(predictor)
    jnp_cutpoints = JNP[].array(cutpoints)

    return BI_INSTANCE[].dist.ordered_logistic(
        predictor=jnp_predictor,
        cutpoints=jnp_cutpoints,
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
