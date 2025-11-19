"""
    bi_dist_matrix_normal(; loc, scale_tril_row, scale_tril_column, validate_args=nothing, name="x", obs=nothing, mask=nothing, sample=false, seed=nothing, shape=(), event=0, create_obj=false, to_jax=true)

Matrix Normal Distribution
Samples from a Matrix Normal distribution, which is a multivariate normal distribution over matrices.
The distribution is characterized by a location matrix and two lower triangular matrices that define the correlation structure.
The distribution is related to the multivariate normal distribution in the following way.
If Equation: `X ~ MN(loc,U,V)$ then $vec(X) ~ MVN(vec(loc), kron(V,U) )`.
- `loc`: A numeric vector, matrix, or array representing the location of the distribution.
- `scale_tril_row`: A numeric vector, matrix, or array representing the lower cholesky of rows correlation matrix.
- `scale_tril_column`: A numeric vector, matrix, or array representing the lower cholesky of columns correlation matrix.
- `shape`: A numeric vector specifying the shape of the distribution.  Must be a vector.
- `event`: An integer representing the number of batch dimensions to reinterpret as event dimensions.
- `mask`: A logical vector, matrix, or array (.BI_env$jnp$array) to mask observations.
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
- When ``sample=FALSE`, a BI Matrix Normal distribution object (for model building).
- When ``sample=TRUE`, a JAX array of samples drawn from the Matrix Normal distribution (for direct sampling).
- When ``create_obj=TRUE`, the raw BI distribution object (for advanced use cases).
@seealso This is a wrapper of  \url`https://num.pyro.ai/en/stable/distributions.html#matrixnormal_lowercase`
# Examples
\donttest`
library(BayesianInference)
m=importBI(platform='cpu')
n_rows= 3
n_cols = 4
loc = matrix(rep(0,n_rows*n_cols), nrow = n_rows, ncol = n_cols,byrow = TRUE)
U_row_cov =
matrix(c(1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0),
nrow = n_rows, ncol = n_rows,byrow = TRUE)
scale_tril_row = chol(U_row_cov)
V_col_cov = matrix(c(2.0, -0.8, 0.1, 0.4, -0.8, 2.0, 0.2, -0.2, 0.1,
0.2, 2.0, 0.0, 0.4, -0.2, 0.0, 2.0),
nrow = n_cols, ncol = n_cols,byrow = TRUE)
scale_tril_column = chol(V_col_cov)
bi.dist.matrix_normal(
loc = loc,
scale_tril_row = scale_tril_row,
scale_tril_column = scale_tril_column,
sample = TRUE
)
`
"""
function bi_dist_matrix_normal(; 
    loc,
    scale_tril_row,
    scale_tril_column,
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
    jnp_scale_tril_row = JNP[].array(scale_tril_row)
    jnp_scale_tril_column = JNP[].array(scale_tril_column)

    return BI_INSTANCE[].dist.matrix_normal(
        loc=jnp_loc,
        scale_tril_row=jnp_scale_tril_row,
        scale_tril_column=jnp_scale_tril_column,
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
