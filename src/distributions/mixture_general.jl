"""
    bi_dist_mixture_general(; mixing_distribution, component_distributions, support=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

A finite mixture of component distributions from different families.
A mixture distribution is a probability distribution constructed by selecting one of several component distributions according to specified weights, and then drawing a sample from the chosen component. It allows modelling of heterogeneous populations and multimodal data.
- `mixing_distribution`: A `Categorical` distribution specifying the weights for each mixture component.
The size of this distribution specifies the number of components in the mixture.
- `component_distributions`: A list of distributions representing the components of the mixture.
- `support`: A constraint object specifying the support of the mixture distribution.
If not provided, the support will be inferred from the component distributions.
- `shape`: Numeric vector. A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: Integer. The number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: Logical vector. Optional boolean array to mask observations.
- `create_obj`: Logical. If True, returns the raw BI distribution object instead of creating a sample site. This is essential for building complex distributions like `MixtureSameFamily`.
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
- When ``sample=FALSE`, a BI MixtureGeneral distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the MixtureGeneral distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.mixture_general(
mixing_distribution = bi.dist.categorical(probs = c(0.3, 0, 7),create_obj = TRUE),
component_distributions = c(
bi.dist.normal(0,1,create_obj = TRUE),
bi.dist.normal(0,1,create_obj = TRUE),
bi.dist.normal(0,1,create_obj = TRUE)),
sample = TRUE)
`
"""
function bi_dist_mixture_general(; 
    mixing_distribution,
    component_distributions,
    support=nothing,
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
    jnp_component_distributions = JNP[].array(component_distributions)
    jnp_support = JNP[].array(support)

    return BI_INSTANCE[].dist.mixture_general(
        mixing_distribution=jnp_mixing_distribution,
        component_distributions=jnp_component_distributions,
        support=jnp_support,
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
