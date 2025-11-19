"""
    bi_dist_zero_inflated_poisson(; gate, rate=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

A Zero Inflated Poisson distribution.
This distribution combines two Poisson processes: one with a rate parameter and another that generates only zeros.
The probability of observing a zero is determined by the 'gate' parameter, while the probability of observing a non-zero value is governed by the 'rate' parameter of the underlying Poisson distribution.
Equation: `P(X = k) = (1 - gate) * \frac`e^`-rate` rate^k``k!` + gate`
- `gate`: The gate parameter.
- `rate`: A numeric vector, matrix, or array representing the rate parameter of the underlying Poisson distribution.
- `shape`: A numeric vector used to shape the distribution. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: An optional boolean vector, matrix, or array to mask observations.
- `create_obj`: Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
- `sample`: Logical; If `TRUE`, draws samples from the distribution.
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
#' @return
- When ``sample=FALSE`, a BI Zero Inflated Poisson distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Zero Inflated Poisson distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.zero_inflated_poisson(gate=0.3, rate = 5, sample = TRUE)
`
"""
function bi_dist_zero_inflated_poisson(; 
    gate,
    rate=1.0,
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

    jnp_gate = JNP[].array(gate)
    jnp_rate = JNP[].array(rate)

    return BI_INSTANCE[].dist.zero_inflated_poisson(
        gate=jnp_gate,
        rate=jnp_rate,
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
