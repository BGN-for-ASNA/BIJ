module BayesianInference

using PythonCall

# Exported functions
export import_bi
export bi_dist_normal

# Global reference to the BI instance
const BI_INSTANCE = Ref{Py}()
const JNP = Ref{Py}()

"""
    import_bi(; platform="cpu", cores=nothing, rand_seed=true, deallocate=false, print_devices_found=true, backend="numpyro")

Initializes the BI Python module.

# Arguments
- `platform`: "cpu" or "gpu". Defaults to "cpu".
- `cores`: Number of CPU cores to use. Defaults to `nothing` (auto).
- `rand_seed`: Boolean. Random seed. Defaults to `true`.
- `deallocate`: Boolean. Whether memory should be deallocated when not in use. Defaults to `false`.
- `print_devices_found`: Boolean. Whether to print devices found. Defaults to `true`.
- `backend`: "numpyro" or "tfp". Defaults to "numpyro".
"""
function import_bi(; 
    platform="cpu", 
    cores=nothing, 
    rand_seed=true, 
    deallocate=false, 
    print_devices_found=true, 
    backend="numpyro"
)
    println("-"^52)
    println("Loading BI")
    println("-"^52)

    try
        # Import BI module
        bi_module = pyimport("BI")
        
        # Import jax.numpy
        jax_numpy = pyimport("jax.numpy")
        JNP[] = jax_numpy
        println("jax and jax.numpy have been imported.")

        # Initialize BI class
        # Note: PythonCall handles type conversion automatically for basic types
        bi_class = bi_module.bi
        
        bi_instance = bi_class(
            platform=platform,
            cores=cores,
            rand_seed=rand_seed,
            deallocate=deallocate,
            print_devices_found=print_devices_found
        )
        
        BI_INSTANCE[] = bi_instance
        
        return bi_instance
    catch e
        println("-"^52)
        println("An error occurred: ", e)
        println("-"^52)
        rethrow(e)
    end
end

# Include management functions
include("management.jl")

# Include distribution files
include("distributions/normal.jl")

end

include("distributions/asymmetric_laplace.jl")
include("distributions/asymmetric_laplace_quantile.jl")
include("distributions/bernoulli.jl")
include("distributions/beta.jl")
include("distributions/beta_binomial.jl")
include("distributions/beta_proportion.jl")
include("distributions/binomial.jl")
include("distributions/car.jl")
include("distributions/categorical.jl")
include("distributions/cauchy.jl")
include("distributions/chi2.jl")
include("distributions/delta.jl")
include("distributions/dirichlet.jl")
include("distributions/dirichlet_multinomial.jl")
include("distributions/discrete_uniform.jl")
include("distributions/eulermaruyama.jl")
include("distributions/exponential.jl")
include("distributions/gamma.jl")
include("distributions/gamma_poisson.jl")
include("distributions/gaussian_copula.jl")
include("distributions/gaussian_copula_beta.jl")
include("distributions/gaussian_random_walk.jl")
include("distributions/gaussian_state_space.jl")
include("distributions/geometric.jl")
include("distributions/gompertz.jl")
include("distributions/gumbel.jl")
include("distributions/halfcauchy.jl")
include("distributions/halfnormal.jl")
include("distributions/inverse_gamma.jl")
include("distributions/kumaraswamy.jl")
include("distributions/laplace.jl")
include("distributions/left_truncated_distribution.jl")
include("distributions/levy.jl")
include("distributions/lkj.jl")
include("distributions/logistic.jl")
include("distributions/log_normal.jl")
include("distributions/log_uniform.jl")
include("distributions/lower_truncated_powerlaw.jl")
include("distributions/low_rank_multivariate_normal.jl")
include("distributions/matrix_normal.jl")
include("distributions/mixture.jl")
include("distributions/mixture_general.jl")
include("distributions/mixture_same_family.jl")
include("distributions/multinomial.jl")
include("distributions/multi_nomial_logits.jl")
include("distributions/multi_nomial_probs.jl")
include("distributions/multi_variate_normal.jl")
include("distributions/multi_variate_student_t.jl")
include("distributions/negative_binomial.jl")
include("distributions/negative_binomial_logits.jl")
include("distributions/negative_binomial_probs.jl")
include("distributions/ordered_logistic.jl")
include("distributions/pareto.jl")
include("distributions/poisson.jl")
include("distributions/projected_normal.jl")
include("distributions/relaxed_bernoulli.jl")
include("distributions/relaxed_bernoulli_logits.jl")
include("distributions/right_truncated_distribution.jl")
include("distributions/soft_laplace.jl")
include("distributions/student_t.jl")
include("distributions/truncated_cauchy.jl")
include("distributions/truncated_distribution.jl")
include("distributions/truncated_normal.jl")
include("distributions/truncated_poly_gamma.jl")
include("distributions/two_sided_truncated_distribution.jl")
include("distributions/uniform.jl")
include("distributions/weibull.jl")
include("distributions/wishart.jl")
include("distributions/wishart_cholesky.jl")
include("distributions/zero_inflated_distribution.jl")
include("distributions/zero_inflated_negativebinomial2.jl")
include("distributions/zero_inflated_poisson.jl")
include("distributions/zero_sum_normal.jl")
