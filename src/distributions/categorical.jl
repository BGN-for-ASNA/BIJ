"""
    bi_dist_categorical(; probs=nothing, logits=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Sample from a Categorical distribution.
The Categorical distribution, also known as the multinomial distribution,
describes the probability of different outcomes from a finite set of possibilities.
It is commonly used to model discrete choices or classifications.
Equation: `P(k) = \frac`e^`\log(p_k)```\sum_`j=1`^`K` e^`\log(p_j)```
where Equation: `p_k` is the probability of outcome Equation: `k`, and the sum is over all possible outcomes.
- `probs`: A numeric vector of probabilities for each category. Must sum to 1.
- `logits`: A numeric vector of  Log-odds of each category.
- `shape`: A numeric vector specifying the shape. When \code{sample=FALSE} (model building),
this is used with `.expand(shape)` to set the distribution's batch shape.
When ``sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array
of the given shape.
- `event`: The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: An optional boolean vector to mask observations.
- `create_obj`: Logical; If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
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
- When ``sample=FALSE`, a BI Categorical distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Categorical distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.categorical(probs = c(0.5,0.5), sample = TRUE, shape = c(3))
`
"""
function bi_dist_categorical(; 
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

    return BI_INSTANCE[].dist.categorical(
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
