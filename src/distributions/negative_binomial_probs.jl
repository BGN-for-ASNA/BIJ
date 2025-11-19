"""
    bi_dist_negative_binomial_probs(; total_count, probs, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Sample from a Negative Binomial distribution with probabilities.
The Negative Binomial distribution models the number of failures before the first success in a sequence of independent Bernoulli trials.  It is characterized by two parameters: 'concentration' (r) and 'rate' (p).  In this implementation, the 'concentration' parameter is derived from 'total_count' and the 'rate' parameter is derived from 'probs'.
Equation: `P(k) = \binom`k + r - 1``k` p^r (1 - p)^k`
- `total_count`: A numeric vector, matrix, or array representing the parameter
controlling the shape of the distribution. Represents the total number of trials.
- `probs`: A numeric vector representing event probabilities. Must sum to 1.
- `shape`: A numeric vector.  When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector or array. Optional boolean array to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#negativebinomialprobs`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.negative_binomial_probs(probs =  c(0.2, 0.3, 0.5), total_count = 10, sample = TRUE)
`
"""
function bi_dist_negative_binomial_probs(; 
    total_count,
    probs,
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

    return BI_INSTANCE[].dist.negative_binomial_probs(
        total_count=jnp_total_count,
        probs=jnp_probs,
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
