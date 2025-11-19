"""
    bi_dist_dirichlet(; concentration, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Samples from a Dirichlet distribution.
The Dirichlet distribution is a multivariate generalization of the Beta distribution.
It is a probability distribution over a simplex, which is a set of vectors where each element is non-negative and sums to one. It is often used as a prior distribution for categorical distributions.
Equation: `P(x_1, ..., x_K) = \frac`\Gamma(\sum_`i=1`^K \alpha_i)``\prod_`i=1`^K \Gamma(\alpha_i)` \prod_`i=1`^K x_i^`\alpha_i - 1``
- `concentration`: A numeric vector or array representing the concentration parameter(s) of the Dirichlet distribution. Must be positive.
- `shape`: A numeric vector specifying the shape of the distribution.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector or array to mask observations.
- `create_obj`: Logical; If TRUE, returns the raw BI distribution object instead of creating a sample site.
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
- When ``sample=FALSE`, a BI Dirichlet distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Dirichlet distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#dirichlet`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.dirichlet(concentration =  c(0.1,.9), sample = TRUE)
`
"""
function bi_dist_dirichlet(; 
    concentration,
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

    jnp_concentration = JNP[].array(concentration)

    return BI_INSTANCE[].dist.dirichlet(
        concentration=jnp_concentration,
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
