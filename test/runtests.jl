using Test
using BayesianInference
using PythonCall

@testset "BayesianInference.jl" begin
    
    @testset "Module exports" begin
        # Test that main functions and macros are exported
        @test isdefined(BayesianInference, :importBI)
        @test isdefined(BayesianInference, :jnp)
        @test isdefined(BayesianInference, :jax)
        @test isdefined(BayesianInference, Symbol("@BI"))
        @test isdefined(BayesianInference, Symbol("@pyplot"))
    end
    
    @testset "Python interop basics" begin
        # Test that pybuiltins, pydict, pylist are accessible
        @test isdefined(BayesianInference, :pybuiltins)
        @test isdefined(BayesianInference, :pydict)
        @test isdefined(BayesianInference, :pylist)
    end
    
    @testset "BI Initialization" begin
        # Test basic initialization
        # Note: This may take time on first run as CondaPkg sets up the environment
        println("Initializing BI (may take time on first run)...")
        m = importBI(print_devices_found=false)
        @test !isnothing(m)
        @test pyhasattr(m, "dist")
        println("✓ BI initialized successfully")
    end
    
    @testset "JAX/NumPy availability" begin
        # Test that jax and jnp are properly initialized
        @test !isnothing(jnp)
        @test !isnothing(jax)
        
        # Test basic jnp operations
        arr = jnp.array([1, 2, 3])
        @test !isnothing(arr)
        @test pyconvert(Int, arr.shape[0]) == 3
    end
    
    @testset "InspectableFunction wrapper" begin
        # Test the @BI macro creates proper wrapper
        @BI function test_model(x, y)
            return x + y
        end
        
        @test typeof(test_model) == BayesianInference.InspectableFunction
        @test test_model(2, 3) == 5
    end
end
