"""
    bi_dist_relaxed_bernoulli(; temperature, probs=nothing, logits=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Relaxed Bernoulli distribution.
The Relaxed Bernoulli distribution is a continuous relaxation of the discrete Bernoulli distribution.
It's useful for variational inference and other applications where a differentiable approximation of the Bernoulli is needed.
The probability density function (PDF) is defined as:
Equation: `p(x) = \frac`1``2` \left( 1 + \tanh\left(\frac`x - \beta \log(\frac`p``1-p`)``1`\right) \right)`
- `temperature`: A numeric value representing the temperature parameter.
- `probs`: (jnp.ndarray, optional): The probability of success. Must be in the interval `[0, 1]`. Only one of `probs` or `logits` can be specified.
- `logits`: A numeric vector or matrix representing the logits parameter.
- `shape`: A numeric vector (e.g., `c(10)`) specifying the shape. When `sample=False` (model building), this is used
with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is
used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector or array to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site.
This is essential for building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI Relaxed Bernoulli distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Relaxed Bernoulli distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso \url`https://num.pyro.ai/en/stable/distributions.html#relaxedbernoulli`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.relaxed_bernoulli(temperature = c(1,1), logits = 0.0, sample = TRUE)
`
"""
function bi_dist_relaxed_bernoulli(; 
    temperature,
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

    jnp_temperature = JNP[].array(temperature)
    jnp_probs = JNP[].array(probs)
    jnp_logits = JNP[].array(logits)

    return BI_INSTANCE[].dist.relaxed_bernoulli(
        temperature=jnp_temperature,
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
