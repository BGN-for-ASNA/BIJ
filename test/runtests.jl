using Test
using BayesianInference
using PythonCall

# Helper to mimic R's matrix function
function r_matrix(data; nrow=nothing, ncol=nothing, byrow=false)
    if isnothing(nrow) && isnothing(ncol)
        nrow = length(data)
        ncol = 1
    elseif isnothing(nrow)
        nrow = div(length(data), ncol)
    elseif isnothing(ncol)
        ncol = div(length(data), nrow)
    end
    
    if byrow
        # Fill by row: reshape to (ncol, nrow) then transpose
        return permutedims(reshape(data, ncol, nrow))
    else
        # Fill by column (standard Julia/R default)
        return reshape(data, nrow, ncol)
    end
end

# Helper to convert Python output to Julia for comparison
function to_julia(res)
    # If it's a list (from .tolist()), convert to Vector
    # If it's a scalar, convert to number
    val = pyconvert(Any, res)
    return val
end

# Initialize BI
println("Initializing BI for tests...")
# User requested: m=importBI(platform='cpu', rand_seed = FALSE)
# Julia equivalent:
import_bi(platform="cpu", rand_seed=false)

@testset "BayesianInference Distributions" begin

    @testset "bi.dist.asymmetric_laplace" begin
        res = bi_dist_asymmetric_laplace(sample=true)
        @test isapprox(to_julia(res.tolist()), -0.2983, atol=0.0001)
    end

    @testset "bi.dist.asymmetric_laplace_quantile" begin
        res = bi_dist_asymmetric_laplace_quantile(sample=true)
        @test isapprox(to_julia(res.tolist()), -0.5967, atol=0.0001)
    end

    @testset "bi.dist.bernoulli" begin
        res = bi_dist_bernoulli(probs = 0.5, sample=true, seed = 5)
        @test isapprox(to_julia(res.tolist()), 0.00, atol=0.0001)
    end

    @testset "bi.dist.beta" begin
        res = bi_dist_beta(concentration1 = 0, concentration0 = 1, sample=true)
        @test isapprox(to_julia(res.tolist()), 2.22507386e-308, atol=1e-4)
    end

    @testset "bi.dist.beta_binomial" begin
        res = bi_dist_beta_binomial(0,1,sample=true)
        @test isapprox(to_julia(res.tolist()), 0.00, atol=0.0001)
    end

    @testset "bi.dist.beta_proportion" begin
        res =  bi_dist_beta_proportion(0, 1, sample=true)
        @test isapprox(to_julia(res.tolist()), 2.225074e-308, atol=1e-4)
    end

    @testset "bi.dist.binomial" begin
        res = bi_dist_binomial(probs = [0.5,0.5], sample=true)
        @test isapprox(to_julia(res.tolist()), [0,0], atol=1e-4)
    end

    @testset "bi.dist.car" begin
        res = bi_dist_car(loc = [1.,2.], correlation = 0.9, conditional_precision = 1., adj_matrix = r_matrix([1,0,0,1], nrow= 2), sample=true)
        @test isapprox(to_julia(res.tolist()), [0.34907   , -0.48164728], atol=1e-4)
    end

    @testset "bi.dist.categorical" begin
        res = bi_dist_categorical(probs = [0.5,0.5], sample=true, shape = [3])
        @test isapprox(to_julia(res.tolist()), [0,0,1], atol=1e-4)
    end

    @testset "bi.dist.cauchy" begin
        res = bi_dist_cauchy(sample=true)
        @test isapprox(to_julia(res.tolist()), -0.261929506653606, atol=1e-4)
    end

    @testset "bi.dist.chi2" begin
        res = bi_dist_chi2(0,sample=true)
        @test isapprox(to_julia(res.tolist()), 0.00, atol=1e-4)
    end

    @testset "bi.dist.delta" begin
        res = bi_dist_delta(v = 5, sample=true)
        @test isapprox(to_julia(res.tolist()), 5., atol=1e-4)
    end

    @testset "bi.dist.dirichlet" begin
        res = bi_dist_dirichlet(concentration = [0.1,.9],sample=true)
        @test isapprox(to_julia(res.tolist()), [9.98541738e-05, 9.99900146e-01], atol=1e-4)
    end

    @testset "bi.dist.dirichlet" begin
        res = bi_dist_dirichlet_multinomial(concentration = [0,1], sample=true, shape = [3])
        
        @test isapprox(to_julia(res.tolist()), [[0,1], [0,1], [0,1]], atol=1e-4)
    end

    @testset "bi.dist.discrete_uniform" begin
        res = bi_dist_discrete_uniform(sample=true)
        
        @test isapprox(to_julia(res.tolist()), 1, atol=1e-4)
    end

    @testset "bi.dist.euler_maruyama" begin
        ornstein_uhlenbeck_sde = function(x, t)
        # This function models dX = -theta * X dt + sigma dW
        theta = 1.0
        sigma = 0.5
        
        drift = -theta * x
        diffusion = sigma
        
        # Return a list of two elements: drift and diffusion
        # reticulate will convert this to a Python tuple
            return (drift, diffusion)
        end
        res = bi_dist_euler_maruyama(t=[0.0, 0.1, 0.2],
        sde_fn = ornstein_uhlenbeck_sde,
        init_dist=bi_dist_normal(0.0, 1.0, create_obj=TRUE), sample=true)
        @test isapprox(to_julia(res.tolist()), [-1.4008841 , -0.96353687, -0.94326995], atol=1e-4)
    end

    @testset "bi.dist.exponential" begin
        res = bi_dist_exponential(rate = [0.1,1,2],sample=true)
        
        @test isapprox(to_julia(res.tolist()), [5.42070555, 0.24372319, 1.68081713], atol=1e-4)
    end

    @testset "bi.dist.gamma_poisson" begin
        res = bi_dist_gamma_poisson(concentration = 1, sample=true)
        
        @test isapprox(to_julia(res.tolist()), 0, atol=1e-4)
    end

    @testset "bi.dist.gamma" begin
        res =  bi_dist_gamma(concentration = 1 , sample=true, seed = 0)
        
        @test isapprox(to_julia(res.tolist()), 0.47552933, atol=1e-4)
    end

    @testset "bi.dist.gaussian_random_walk" begin
        res =  bi_dist_gaussian_random_walk(scale = 1 , sample=true)
        
        @test isapprox(to_julia(res.tolist()), -0.205842139479643, atol=1e-4)
    end

    @testset "bi.dist.gaussian_random_walk" begin
        res =  bi_dist_gaussian_random_walk(scale = 1 , sample=true)
        @test isapprox(to_julia(res.tolist()), -0.205842139479643, atol=1e-4)
    end

    @testset "bi.dist.gaussian_state_space" begin
        res =  bi_dist_gaussian_state_space(
        num_steps = 1,
        transition_matrix = r_matrix([0.5], nrow= 1, byrow=true),
        covariance_matrix = r_matrix([1.0], nrow= 1, byrow=true),
        sample=true)
        
        @test isapprox(to_julia(res.tolist()), [-0.20584214], atol=1e-4)
    end

    @testset "bi.dist.geometric" begin
        res =  bi_dist_geometric(probs = 0.5 , sample=true)
        @test isapprox(to_julia(res.tolist()), 0.00, atol=1e-4)
    end

    @testset "bi.dist.gompertz" begin
        res =  bi_dist_gompertz(concentration = 0.5 , sample=true)
        r2 = round(r2, digits = 4)
        @test isapprox(to_julia(res.tolist()), 0.7344, atol=1e-4)
    end

    @testset "bi.dist.gumbel" begin
        res =  bi_dist_gumbel(loc = 0.5 , scale = 1., sample=true)
        r2 = round(r2, digits = 5)
        @test isapprox(to_julia(res.tolist()), 0.6379100, atol=1e-4)
    end

    @testset "bi.dist.gumbel" begin
        res =  bi_dist_gumbel(loc = 0.5 , scale = 1., sample=true)
        r2 = round(r2, digits = 5)
        @test isapprox(to_julia(res.tolist()), 0.6379100, atol=1e-4)
    end

    @testset "bi.dist.half_cauchy" begin
        res =  bi_dist_half_cauchy(scale = [0.5,0.5] , sample=true)
        @test isapprox(to_julia(res.tolist()), [0.13096475, 0.61892301], atol=1e-4)
    end

    @testset "bi.dist.half_normal" begin
        res =  bi_dist_half_normal(scale = [0.5,0.5] , sample=true)
        @test isapprox(to_julia(res.tolist()), [0.10292107, 0.39238289], atol=1e-4)
    end

    @testset "bi.dist.inverse_gamma" begin
        res =  bi_dist_inverse_gamma(concentration = [0.5,0.5] , sample=true)
        r2 = round(r2, digits = 5)
        @test isapprox(to_julia(res.tolist()), [10.19849, 131.85292], atol=1e-4)
    end

    @testset "bi.dist.kumaraswamy" begin
        res =  bi_dist_kumaraswamy(concentration1 = 1, concentration0 = 10, sample=true)
        @test isapprox(to_julia(res.tolist()), 0.083431146, atol=1e-4)
    end

    @testset "bi.dist.laplace" begin
        res =  bi_dist_laplace(loc = 1, scale = 10, sample=true, seed = 0)
        @test isapprox(to_julia(res.tolist()), 2.78033695, atol=1e-4)
    end

    @testset "bi.dist.left_truncated_distribution" begin
        res =  bi_dist_left_truncated_distribution( base_dist = bi_dist_normal(loc = 1, scale = 10 ,  create_obj=true),  sample=true)
        @test isapprox(to_julia(res.tolist()), 5.84732542, atol=1e-4)
    end

    @testset "bi.dist.levy" begin
        res =  bi_dist_levy( loc = 1, scale = 10,  sample=true)
        @test isapprox(to_julia(res.tolist()), 16.27547182, atol=1e-4)
    end

    @testset "bi.dist.lkj" begin
        res =  bi_dist_lkj( dimension = 2, concentration=1.0, shape = [1], sample=true)
        lst = [
        [
        [1.000000000, -0.502239437],
        [-0.502239437, 1.000000000]
        )
        )
        @test isapprox(to_julia(res.tolist()), lst, atol=1e-4)
    end

    @testset "bi.dist.lkj_cholesky" begin
        res =  bi_dist_lkj_cholesky( dimension = 2, concentration = 1.,  sample=true)
        r2[[2]] = round(r2[[2]], digits = 5)
        lst = [
        [1, 0.00],
        [-0.50224000, 0.86473000]
        )
        
        @test isapprox(to_julia(res.tolist()), lst, atol=1e-4)
    end

    @testset "bi.dist.log_uniform" begin
        res =  bi_dist_log_uniform( low = [1,1], high = [10,10],  sample=true)
        @test isapprox(to_julia(res.tolist()), [1,1], atol=1e-4)
    end

    @testset "bi.dist.logistic" begin
        res =  bi_dist_logistic( loc = [1,1], scale = [10,10],  sample=true)
        @test isapprox(to_julia(res.tolist()), [-2.2911032 , -11.87386776], atol=1e-4)
    end

    @testset "bi.dist.log_normal" begin
        res =  bi_dist_log_normal( loc = [1,1], scale = [10,10],  sample=true)
        @test isapprox(to_julia(res.tolist()), [0.34700316, 0.00106194], atol=1e-4)
    end

    @testset "bi.dist.log_normal" begin
        res =  bi_dist_log_normal( loc = [1,1], scale = [10,10],  sample=true)
        @test isapprox(to_julia(res.tolist()), [0.34700316, 0.00106194], atol=1e-4)
    end

    @testset "bi.dist.log_normal" begin
        res =  bi_dist_log_normal( loc = [1,1], scale = [10,10],  sample=true)
        @test isapprox(to_julia(res.tolist()), [0.34700316, 0.00106194], atol=1e-4)
    end

    @testset "bi.dist.low_rank_multivariate_normal" begin
        event_size = 10
        rank = 5
        
        res =  bi_dist_low_rank_multivariate_normal(
        loc = bi_dist_normal(0,1,shape = [event_size], sample=true)*2,
        cov_factor = bi_dist_normal(0,1,shape = [event_size, rank], sample=true),
        cov_diag = JNP[].exp(bi_dist_normal(0,1,shape = [event_size], sample=true)),
        sample=true)
        
        -3.64957438, -3.16951829, -0.7729603 , -5.94243277, -0.94424 ))
    end

    @testset "bi.dist.lower_truncated_power_law" begin
        res =  bi_dist_lower_truncated_power_law( alpha = [-2, 2], low = [1, 0.5],  sample=true)
        @test isapprox(to_julia(res.tolist()), [1.71956363, 0.46098571], atol=1e-4)
    end

    @testset "bi.dist.matrix_normal" begin
        n_rows= 3
        n_cols = 4
        loc = r_matrix(rep(0,n_rows*n_cols), nrow= n_rows, ncol = n_cols,byrow=true)
        
        U_row_cov = JNP[].array(r_matrix([1.0, 0.5, 0.2, 0.5, 1.0, 0.3, 0.2, 0.3, 1.0], nrow= n_rows, ncol = n_rows,byrow=true))
        scale_tril_row = JNP[].linalg$cholesky(U_row_cov)
        
        V_col_cov = JNP[].array(r_matrix([2.0, -0.8, 0.1, 0.4, -0.8, 2.0, 0.2, -0.2, 0.1, 0.2, 2.0, 0.0, 0.4, -0.2, 0.0, 2.0], nrow= n_cols, ncol = n_cols,byrow=true))
        scale_tril_column = JNP[].linalg$cholesky(V_col_cov)
        
        
        res =  bi_dist_matrix_normal( loc = loc, scale_tril_row = scale_tril_row, scale_tril_column = scale_tril_column, sample=true)
        lst = [
        [-0.291104745, -0.900730803, 2.383118964, 0.207682007],
        [-0.04650985, -0.90767574,  2.58010087,  0.52933797],
        [0.08241693, -1.42376624,  2.10489774, -1.86380361]
        )
        @test isapprox(to_julia(res.tolist()), lst, atol=1e-4)
    end

    @testset "bi.dist.mixture" begin
        res = bi_dist_mixture(
        mixing_distribution = bi_dist_categorical(probs = [0.3, 0, 7],create_obj=true),
        component_distributions = [bi_dist_normal(0,1,create_obj=true], bi_dist_normal(0,1,create_obj=true), bi_dist_normal(0,1,create_obj=true)),
        sample=true)
        @test isapprox(to_julia(res.tolist()), -0.24240651, atol=1e-4)
    end

    @testset "bi.dist.mixture_general" begin
        res = bi_dist_mixture_general(
        mixing_distribution = bi_dist_categorical(probs = [0.3, 0, 7],create_obj=true),
        component_distributions = [bi_dist_normal(0,1,create_obj=true], bi_dist_normal(0,1,create_obj=true), bi_dist_normal(0,1,create_obj=true)),
        sample=true)
        @test isapprox(to_julia(res.tolist()), -0.24240651, atol=1e-4)
    end

    @testset "bi.dist.mixture_same_family" begin
        res = bi_dist_mixture_same_family(
        mixing_distribution = bi_dist_categorical(probs = [0.3, 0.7],create_obj=true),
        component_distribution = bi_dist_normal(0,1, shape = [2], create_obj=true),
        sample=true)
        @test isapprox(to_julia(res.tolist()), 1.88002989, atol=1e-4)
    end

    @testset "bi.dist.multinomial_logits" begin
        res = bi_dist_multinomial_logits(
        logits =  [0.2, 0.3, 0.5],
        total_count = 10,
        sample=true)
        @test isapprox(to_julia(res.tolist()), [2, 5, 3], atol=1e-4)
    end

    @testset "bi.dist.multinomial_logits" begin
        res = bi_dist_multinomial_logits(
        logits =  [0.2, 0.3, 0.5],
        total_count = 10,
        sample=true)
        @test isapprox(to_julia(res.tolist()), [2, 5, 3], atol=1e-4)
    end

    @testset "bi.dist.multinomial_probs" begin
        res = bi_dist_multinomial_probs(
        probs =  [0.2, 0.3, 0.5],
        total_count = 10,
        sample=true)
        @test isapprox(to_julia(res.tolist()), [1,3,6], atol=1e-4)
    end

    @testset "bi.dist.multivariate_normal" begin
        res = bi_dist_multivariate_normal(
        loc =  [1.0, 0.0, -2.0],
        covariance_matrix = r_matrix([ 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5], nrow= 3, byrow=true),
        sample=true)
        @test isapprox(to_julia(res.tolist()), [0.708895254639994, -0.783775419788299, -0.713927354638493], atol=1e-4)
    end

    @testset "bi.dist.multivariate_student_t" begin
        res = bi_dist_multivariate_student_t(
        df = 2,
        loc =  [1.0, 0.0, -2.0],
        scale_tril = JNP[].linalg$cholesky(r_matrix([ 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5], nrow= 3, byrow=true)),
        sample=true)
        @test isapprox(to_julia(res.tolist()), [2.91015292,  0.36815279, -2.23324296], atol=1e-4)
    end

    @testset "bi.dist.multivariate_student_t" begin
        res = bi_dist_multivariate_student_t(
        df = 2,
        loc =  [1.0, 0.0, -2.0],
        scale_tril = JNP[].linalg$cholesky(r_matrix([ 2.0,  0.7, -0.3, 0.7,  1.0,  0.5, -0.3,  0.5,  1.5], nrow= 3, byrow=true)),
        sample=true)
        @test isapprox(to_julia(res.tolist()), [2.91015292,  0.36815279, -2.23324296], atol=1e-4)
    end

    @testset "bi.dist.multinomial" begin
        res = bi_dist_multinomial(
        logits =  [0.2, 0.3, 0.5],
        total_count = 10,
        sample=true)
        @test isapprox(to_julia(res.tolist()), [2, 5, 3], atol=1e-4)
    end

    @testset "bi.dist.negative_binomial_logits" begin
        res = bi_dist_negative_binomial_logits(
        logits =  [0.2, 0.3, 0.5],
        total_count = 10,
        sample=true)
        @test isapprox(to_julia(res.tolist()), [8.0, 17.0, 22.0], atol=1e-4)
    end

    @testset "bi.dist.negative_binomial_probs" begin
        res = bi_dist_negative_binomial_probs(
        probs =  [0.2, 0.3, 0.5],
        total_count = 10,
        sample=true)
        @test isapprox(to_julia(res.tolist()), [2.0,  6.0, 14.], atol=1e-4)
    end

    @testset "bi.dist.negative_binomial" begin
        res = bi_dist_negative_binomial(total_count = 100, probs = 0.5, sample=true)
        @test isapprox(to_julia(res.tolist()), 88, atol=1e-4)
    end

    @testset "bi.dist.normal" begin
        res = bi_dist_normal(
        loc = 0,
        scale = 2,
        sample=true)
        @test isapprox(to_julia(res.tolist()), -0.41168428, atol=1e-4)
    end

    @testset "bi.dist.ordered_logistic" begin
        res = bi_dist_ordered_logistic(
        predictor = [0.2, 0.5, 0.8],
        cutpoints = [-1.0, 0.0, 1.0],
        sample=true)
        @test isapprox(to_julia(res.tolist()), [1, 1, 3], atol=1e-4)
    end

    @testset "bi.dist.pareto" begin
        res = bi_dist_pareto(
        scale = [0.2, 0.5, 0.8],
        alpha = [-1.0, 0.5, 1.0],
        sample=true)
        @test isapprox(to_julia(res.tolist()), [ 0.11630858,  0.8140766 , 23.06902268], atol=1e-4)
    end

    @testset "bi.dist.poisson" begin
        res = bi_dist_poisson(rate = [0.2, 0.5, 0.8], sample=true)
        @test isapprox(to_julia(res.tolist()), [ 0, 0, 1], atol=1e-4)
    end

    @testset "bi.dist.projected_normal" begin
        res = bi_dist_projected_normal(concentration = [1.0, 3.0, 2.0], sample=true)
        @test isapprox(to_julia(res.tolist()), [0.17713475, 0.49410196, 0.85116774], atol=1e-4)
    end

    @testset "bi.dist.projected_normal" begin
        res = bi_dist_projected_normal(concentration = [1.0, 3.0, 2.0], sample=true)
        @test isapprox(to_julia(res.tolist()), [0.17713475, 0.49410196, 0.85116774], atol=1e-4)
    end

    @testset "bi.dist.relaxed_bernoulli" begin
        res = bi_dist_relaxed_bernoulli(temperature = 1, logits = 0.0, sample=true)
        @test isapprox(to_julia(res.tolist()), [0.41845711], atol=1e-4)
    end

    @testset "bi.dist.right_truncated_distribution" begin
        res = bi_dist_right_truncated_distribution(base_dist = bi_dist_normal(0,1, create_obj=true), high = 10, sample=true)
        @test isapprox(to_julia(res.tolist()), [-0.20584214], atol=1e-4)
    end

    @testset "bi.dist.soft_laplace" begin
        res = bi_dist_soft_laplace(loc = 0, scale = 2, sample=true)
        @test isapprox(to_julia(res.tolist()), [-0.51804666], atol=1e-4)
    end

    @testset "bi.dist.student_t" begin
        res = bi_dist_student_t(df = 2, loc = 0, scale = 2, sample=true)
        @test isapprox(to_julia(res.tolist()), [2.70136417], atol=1e-4)
    end

    @testset "bi.dist.truncated_cauchy" begin
        res = bi_dist_truncated_cauchy(loc = 0, scale = 2, low = 0, high = 1.5, sample=true)
        @test isapprox(to_julia(res.tolist()), [0.55196115], atol=1e-4)
    end

    @testset "bi.dist.truncated_distribution" begin
        res = bi_dist_truncated_distribution(base_dist = bi_dist_normal(0,1, create_obj=true), high = 0.7, low = 0.1, sample=true)
        @test isapprox(to_julia(res.tolist()), [0.33487084], atol=1e-4)
    end

    @testset "bi.dist.truncated_normal" begin
        res = bi_dist_truncated_normal(loc = 0, scale = 2, low = 0, high = 1.5, sample=true)
        @test isapprox(to_julia(res.tolist()), [0.58158364], atol=1e-4)
    end

    @testset "bi.dist.truncated_polya_gamma" begin
        res = bi_dist_truncated_polya_gamma(batch_shape = [], sample=true)
        @test isapprox(to_julia(res.tolist()), [0.13129763], atol=1e-4)
    end

    @testset "bi.dist.two_sided_truncated_distribution" begin
        res = bi_dist_two_sided_truncated_distribution(base_dist = bi_dist_normal(0,1, create_obj=true), high = 0.5, low = 0.1, sample=true)
        @test isapprox(to_julia(res.tolist()), [0.261847325], atol=1e-4)
    end

    @testset "bi.dist.uniform" begin
        res = bi_dist_uniform(low = 0, high = 1.5, sample=true)
        @test isapprox(to_julia(res.tolist()), [0.62768567], atol=1e-4)
    end

    @testset "bi.dist.weibull" begin
        res = bi_dist_weibull(scale = [10, 10], concentration = [1,1], sample=true)
        @test isapprox(to_julia(res.tolist()), [5.42070555, 2.43723185], atol=1e-4)
    end

    @testset "bi.dist.wishart" begin
        res = bi_dist_wishart(concentration = 5, scale_matrix = r_matrix([1,0,0,1], nrow= 2), sample=true)
        lst = [
        [5.81512786, -3.37817265],
        [-3.37817265, 9.33345547]
        )
        @test isapprox(to_julia(res.tolist()), lst, atol=1e-4)
    end

    @testset "bi.dist.wishart_cholesky" begin
        res = bi_dist_wishart_cholesky(concentration = 5, scale_matrix = r_matrix([1,0,0,1], nrow= 2), sample=true)
        lst = [
        [2.41145762,  0.],
        [-1.4008841 ,  2.71495473]
        )
        @test isapprox(to_julia(res.tolist()), lst, atol=1e-4)
    end

    @testset "bi.dist.zero_inflated_distribution" begin
        res = bi_dist_zero_inflated_distribution(base_dist = bi_dist_poisson(5, create_obj=true), gate=0.3, sample=true)
        @test isapprox(to_julia(res.tolist()), 4, atol=1e-4)
    end

    @testset "bi.dist.zero_inflated_negative_binomial" begin
        res = bi_dist_zero_inflated_negative_binomial(mean = 2, concentration = 1, gate=0.3, sample=true)
        @test isapprox(to_julia(res.tolist()), 1, atol=1e-4)
    end

    @testset "bi.dist.zero_inflated_poisson" begin
        res = bi_dist_zero_inflated_poisson(gate=0.3, rate = 5, sample=true)
        @test isapprox(to_julia(res.tolist()), 4, atol=1e-4)
    end

    @testset "bi.dist.zero_sum_normal" begin
        res = bi_dist_zero_sum_normal(scale=0.3, event_shape = [], sample=true)
        @test isapprox(to_julia(res.tolist()), -0.061752642, atol=1e-4)
    end

end
