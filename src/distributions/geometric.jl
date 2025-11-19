"""
    bi_dist_geometric(; probs=nothing, logits=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Geometric distribution.
The Geometric distribution models the number of failures before the first success in a sequence of Bernoulli trials.
It is characterized by a single parameter, the probability of success on each trial.
- `probs`: A numeric vector, matrix, or array representing the probability of success on each trial. Must be between 0 and 1.
- `logits`: A numeric vector, matrix, or array representing the log-odds of success on each trial. `probs = jax.nn.sigmoid(logits)`.
- `shape`: A numeric vector specifying the shape of the output.  Used to set the distribution's batch shape when \code{sample=FALSE} (model building) or as `sample_shape` to draw a raw JAX array of the given shape when \code{sample=TRUE} (direct sampling).
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector, matrix, or array to mask observations.
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
When ``sample=FALSE`: A BI Geometric distribution object (for model building).
When ``sample=TRUE`: A JAX array of samples drawn from the Geometric distribution (for direct sampling).
When ``create_obj=TRUE`: The raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#geometric`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.geometric(logits = 0.5, sample = TRUE)
bi.dist.geometric(probs = 0.5, sample = TRUE)
`
"""
function bi_dist_geometric(; 
    probs=nothing,
    logits=nothing,
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

    jnp_probs = JNP[].array(probs)
    jnp_logits = JNP[].array(logits)

    return BI_INSTANCE[].dist.geometric(
        probs=jnp_probs,
        logits=jnp_logits,
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
