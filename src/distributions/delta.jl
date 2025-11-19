"""
    bi_dist_delta(; v=0.0, log_density=0.0, event_dim=0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

The Delta distribution.
The Delta distribution, also known as a point mass distribution, assigns probability 1 to a single point and 0 elsewhere.
It's useful for representing deterministic variables or as a building block for more complex distributions.
@importFrom reticulate py_none tuple
- `log_density`: The log probability density of the point mass. This is primarily for creating distributions that are non-normalized or for specific advanced use cases. For a standard delta distribution, this should be 0. Defaults to 0.0.
- `v`: A numeric vector representing the location of the point mass.
- `event_dim`: event_dim (A numeric vector, optional): The number of rightmost dimensions of `v` to interpret as event dimensions. Defaults to 0.
- `shape`: A numeric vector used for shaping. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A boolean vector to mask observations.
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
- When ``sample=FALSE`, a BI Delta distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Delta distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#delta`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.delta(v = 5, sample = TRUE)
`
"""
function bi_dist_delta(; 
    v=0.0,
    log_density=0.0,
    event_dim=0,
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

    jnp_v = JNP[].array(v)
    jnp_log_density = JNP[].array(log_density)
    jnp_event_dim = JNP[].array(event_dim)

    return BI_INSTANCE[].dist.delta(
        v=jnp_v,
        log_density=jnp_log_density,
        event_dim=jnp_event_dim,
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
