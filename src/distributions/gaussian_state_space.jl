"""
    bi_dist_gaussian_state_space(; num_steps, transition_matrix, covariance_matrix=nothing, precision_matrix=nothing, scale_tril=nothing, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Gaussian State Space Distribution
Samples from a Gaussian state space model.
Equation: `z_`t` = A z_`t - 1` + \epsilon_t \\ z_`t` = \sum_`k=1`^`t` A^`t-k` \epsilon_k`
where \eqn`z_t` is the state vector at step \eqn`t`, \eqn`A`
is the transition matrix, and \eqn`\epsilon` is the innovation noise.
- `num_steps`: An integer representing the number of steps.
- `transition_matrix`: A numeric vector, matrix, or array representing the state space transition matrix \eqn{A}.
- `covariance_matrix`: A numeric vector, matrix, or array representing the covariance of the innovation noise \eqn{\epsilon}.  Defaults to `reticulate::py_none()`.
- `precision_matrix`: A numeric vector, matrix, or array representing the precision matrix of the innovation noise \eqn{\epsilon}. Defaults to `reticulate::py_none()`.
- `scale_tril`: A numeric vector, matrix, or array representing the scale matrix of the innovation noise \eqn{\epsilon}. Defaults to `reticulate::py_none()`.
- `shape`: A numeric vector specifying the shape. When `sample=FALSE` (model building), this is used with `.expand(shape)` to set the distribution's batch shape. When `sample=TRUE` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions (used in model building).
- `mask`: A logical vector, matrix, or array representing an optional boolean array to mask observations. Defaults to `reticulate::py_none()`.
- `create_obj`: A logical value. If `TRUE`, returns the raw BI distribution object instead of creating a sample site. Defaults to `FALSE`.
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
- When `sample=FALSE`, a BI  Gaussian State Space distribution object (for model building).
- When `sample=TRUE`, a JAX array of samples drawn from the  Gaussian State Space distribution (for direct sampling).
- When `create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#gaussianstatespace`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
bi.dist.gaussian_state_space(
num_steps = 1,
transition_matrix = matrix(c(0.5), nrow = 1, byrow = TRUE),
covariance_matrix = matrix(c(1.0, 0.7, 0.7, 1.0), nrow = 2, byrow = TRUE),
sample = TRUE)
`
"""
function bi_dist_gaussian_state_space(; 
    num_steps,
    transition_matrix,
    covariance_matrix=nothing,
    precision_matrix=nothing,
    scale_tril=nothing,
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

    jnp_num_steps = JNP[].array(num_steps)
    jnp_transition_matrix = JNP[].array(transition_matrix)
    jnp_covariance_matrix = JNP[].array(covariance_matrix)
    jnp_precision_matrix = JNP[].array(precision_matrix)
    jnp_scale_tril = JNP[].array(scale_tril)

    return BI_INSTANCE[].dist.gaussian_state_space(
        num_steps=jnp_num_steps,
        transition_matrix=jnp_transition_matrix,
        covariance_matrix=jnp_covariance_matrix,
        precision_matrix=jnp_precision_matrix,
        scale_tril=jnp_scale_tril,
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
