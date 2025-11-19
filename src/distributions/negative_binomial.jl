"""
    bi_dist_negative_binomial(; total_count, probs, logits=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Negative Binomial distribution.
This distribution is parameterized as a Gamma-Poisson with a modified rate.
It represents the number of events occurring in a fixed amount of time or trials,
where each event has a probability of success.
Equation: `P(k) = \frac`\Gamma(k + \alpha)``\Gamma(k + 1) \Gamma(\alpha)` \left(\frac`\beta``\alpha + \beta`\right)^k \left(1 - \frac`\beta``\alpha + \beta`\right)^k`
- `total_count`: (int): The number of trials *n*.
- `probs`: A numeric vector, matrix, or array representing the probability of success for each Bernoulli trial. Must be between 0 and 1.
- `logits`: A numeric vector, matrix, or array representing the log-odds of success for each trial.
- `shape`: A numeric vector.  Used with `.expand(shape)` when `sample=False` (model building) to set the distribution's batch shape. When `sample=True` (direct sampling), used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: An optional logical vector to mask observations.
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
- When ``sample=FALSE`, a BI Negative Binomial distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Negative Binomial distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#negativebinomial2`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.negative_binomial(total_count = 100, probs = 0.5, sample = TRUE)
`
"""
function bi_dist_negative_binomial(; 
    total_count,
    probs,
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

    return BI_INSTANCE[].dist.negative_binomial(
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
