"""
    bi_dist_binomial(; total_count=1L, probs=nothing, logits=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Binomial distribution.
The Binomial distribution models the number of successes in a sequence of independent Bernoulli trials.
It represents the probability of obtaining exactly *k* successes in *n* trials, where each trial has a probability *p* of success.
Equation: `P(X = k) = \binom`n``k` p^k (1-p)^`n-k``
- `total_count`: (int): The number of trials *n*.
- `probs`: (numeric vector, optional): The probability of success *p* for each trial. Must be between 0 and 1.
- `logits`: (numeric vector, optional): The log-odds of success for each trial.
- `shape`: (numeric vector): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: (numeric vector of booleans, optional): Optional boolean array to mask observations.
- `create_obj`: (logical, optional): If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Binomial distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Binomial distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.binomial(probs = c(0.5,0.5), sample = TRUE)
bi.dist.binomial(logits = 1, sample = TRUE)
`
"""
function bi_dist_binomial(; 
    total_count=1L,
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

    jnp_total_count = JNP[].array(total_count)
    jnp_probs = JNP[].array(probs)
    jnp_logits = JNP[].array(logits)

    return BI_INSTANCE[].dist.binomial(
        total_count=jnp_total_count,
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
