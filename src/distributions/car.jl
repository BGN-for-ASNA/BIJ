"""
    bi_dist_car(; loc, correlation, conditional_precision, adj_matrix, is_sparse=false, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Conditional Autoregressive (CAR) Distribution
The CAR distribution models a vector of variables where each variable is a linear
combination of its neighbors in a graph. The CAR model captures spatial dependence in areal data by modeling each observation as conditionally dependent on its neighbors.It specifies a joint distribution of a vector of random variables Equation: `\mathbf`y` = (y_1, y_2, \dots, y_N)` based on their conditional distributions, where each $y_i$ is conditionally independent of all other variables given its neighbors.
* **Application**: Widely used in disease mapping, environmental modeling, and spatial econometrics to account for spatial autocorrelation.
The CAR distribution is a special case of the multivariate normal distribution.
It is used to model spatial data, such as temperature or precipitation.
- `loc`: Numeric vector, matrix, or array representing the mean of the distribution.
- `correlation`: Numeric vector, matrix, or array representing the correlation between variables.
- `conditional_precision`: Numeric vector, matrix, or array representing the precision of the distribution.
- `adj_matrix`: Numeric vector, matrix, or array representing the adjacency matrix defining the graph.
- `is_sparse`: Logical indicating whether the adjacency matrix is sparse. Defaults to `FALSE`.
- `shape`: A numeric vector. When `sample=False` (model building), this is used with `.expand(shape)` to set the
distribution's batch shape. When `sample=True` (direct sampling),
this is used as `sample_shape` to draw a raw JAX array of the
given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event
dimensions (used in model building).
- `mask`: An optional boolean vector to mask observations.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution
object instead of creating a sample site. This is essential for
building complex distributions like `MixtureSameFamily`.
- `validate_args`: Logical indicating whether to validate arguments. Defaults to `reticulate::py_none()`.
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
- When ``sample=FALSE`, a BI CAR distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the CAR distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.car(
loc = c(1.,2.),
correlation = 0.9,
conditional_precision = 1.,
adj_matrix = matrix(c(1,0,0,1), nrow = 2),
sample = TRUE
)
`
"""
function bi_dist_car(; 
    loc,
    correlation,
    conditional_precision,
    adj_matrix,
    is_sparse=false,
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

    jnp_loc = JNP[].array(loc)
    jnp_correlation = JNP[].array(correlation)
    jnp_conditional_precision = JNP[].array(conditional_precision)
    jnp_adj_matrix = JNP[].array(adj_matrix)
    jnp_is_sparse = JNP[].array(is_sparse)

    return BI_INSTANCE[].dist.car(
        loc=jnp_loc,
        correlation=jnp_correlation,
        conditional_precision=jnp_conditional_precision,
        adj_matrix=jnp_adj_matrix,
        is_sparse=jnp_is_sparse,
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
