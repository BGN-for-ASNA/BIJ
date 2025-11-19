"""
    bi_dist_normal(; loc=0.0, scale=1.0, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Normal (Gaussian) distribution.

The Normal distribution is characterized by its mean (loc) and standard deviation (scale).
It's a continuous probability distribution that arises frequently in statistics and
probability theory.

# Arguments
- `loc`: Mean of the distribution. Defaults to 0.0.
- `scale`: Standard deviation of the distribution. Defaults to 1.0.
- `validate_args`: Whether to validate parameter values.
- `name`: Name of the random variable. Defaults to "x".
- `obs`: Observed values.
- `mask`: Mask for observations.
- `sample`: If true, returns samples. If false, creates a random variable. Defaults to `false`.
- `seed`: Random seed.
- `shape`: Shape of the distribution.
- `event`: Number of batch dimensions to reinterpret as event dimensions.
- `create_obj`: If true, returns the raw BI distribution object.
- `to_jax`: Whether to convert to JAX array.
"""
function bi_dist_normal(; 
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
    to_jax=true
)
    if !isassigned(BI_INSTANCE)
        error("BayesianInference not initialized. Please call import_bi() first.")
    end
    
    # Convert shape to tuple if needed
    py_shape = tuple(shape...)
    
    jnp_loc = JNP[].array(loc)
    jnp_scale = JNP[].array(scale)
    
    return BI_INSTANCE[].dist.normal(
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
        to_jax=to_jax
    )
end
