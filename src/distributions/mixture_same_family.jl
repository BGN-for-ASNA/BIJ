"""
    bi_dist_mixture_same_family(; mixing_distribution, component_distribution, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

A finite mixture of component distributions from the same family.
A *Mixture (Same-Family)* distribution is a finite mixture in which **all components come from the same parametric family** (for example, all Normal distributions but with different parameters), and are combined via mixing weights. The class is typically denoted as:
Let the mixture have (K) components. Let weights Equation: `w_i\ge0`, Equation: `\sum_`i=1`^K w_i = 1`. Let the component family have density Equation: `f(x \mid \theta_i)` for each component (i). Then the mixture's PDF is
Equation: `
f_X(x) = \sum_`i=1`^K w_i ; f(x \mid \theta_i).
`
where each $f(x \mid \theta_i)$ is from the same family with parameter $\theta_i$.
- `mixing_distribution`: A distribution specifying the weights for each mixture component.
The size of this distribution specifies the number of components in the mixture.
- `component_distribution`: A list of distributions representing the components of the mixture.
- `shape`: A numeric vector specifying the shape of the distribution.
- `event`: Integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector, matrix, or array to mask observations.
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
- When ``sample=FALSE`, a BI MixtureSameFamily distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the MixtureSameFamily distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#mixture-same-family`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.mixture_same_family(
mixing_distribution = bi.dist.categorical(probs = c(0.3, 0.7),create_obj = TRUE),
component_distribution = bi.dist.normal(0,1, shape = c(2), create_obj = TRUE),
sample = TRUE)
`
"""
function bi_dist_mixture_same_family(; 
    mixing_distribution,
    component_distribution,
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

    jnp_mixing_distribution = JNP[].array(mixing_distribution)
    jnp_component_distribution = JNP[].array(component_distribution)

    return BI_INSTANCE[].dist.mixture_same_family(
        mixing_distribution=jnp_mixing_distribution,
        component_distribution=jnp_component_distribution,
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
