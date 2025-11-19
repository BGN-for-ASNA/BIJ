using Pkg
using CondaPkg
using PythonCall

export setup_env, check_env, install_dependencies

"""
    check_env()

Check if the required Python dependencies are available.
Returns `true` if `jax`, `numpyro`, and `arviz` can be imported.
"""
function check_env()
    try
        pyimport("jax")
        pyimport("numpyro")
        pyimport("arviz")
        println("Virtual environment dependencies are available.")
        return true
    catch e
        println("Virtual environment dependencies are NOT available.")
        return false
    end
end

"""
    setup_env(; backend="cpu", env_name="BayesInference")

Sets up the Python environment with the required dependencies.
In Julia, this modifies the CondaPkg.toml or installs via pip into the current environment.

# Arguments
- `backend`: "cpu" or "gpu". Defaults to "cpu".
- `env_name`: Ignored in Julia as PythonCall manages the environment for the project. Kept for API compatibility.
"""
function setup_env(; backend="cpu", env_name="BayesInference")
    if !(backend in ["cpu", "gpu"])
        error("backend must be either 'cpu' or 'gpu'")
    end

    println("Setting up environment for backend: $backend")

    # Base dependencies
    base_packages = ["arviz", "numpyro", "BayesInference==0.0.30"]
    
    # Backend-specific dependencies
    if backend == "cpu"
        push!(base_packages, "jax", "jaxlib")
    else # gpu
        # Note: GPU installation often requires specific pip flags or wheels
        # mimicking the R code:
        push!(base_packages, "jax[cuda12_pip]==0.6.2")
    end

    println("Installing packages: ", join(base_packages, ", "))

    # We use CondaPkg's pip to install these. 
    # Note: modifying CondaPkg.toml programmatically is one way, but here we might just want to install them directly
    # to mimic the imperative nature of the R script.
    # However, CondaPkg is declarative. 
    
    # Strategy: Use pip directly in the environment managed by PythonCall
    pip = joinpath(dirname(PythonCall.C.CTX.exe_path), "pip")
    
    try
        run(`$pip install $(base_packages) --upgrade`)
        println("Installation complete. You may need to restart Julia for changes to take effect.")
    catch e
        println("Error during installation: ", e)
        rethrow(e)
    end
end

"""
    install_dependencies()

Re-installs the core dependencies (numpyro, jax, BayesInference).
"""
function install_dependencies()
    println("Re-installing dependencies...")
    pip = joinpath(dirname(PythonCall.C.CTX.exe_path), "pip")
    
    pkgs = ["numpyro", "jax", "BayesInference"]
    run(`$pip install $(pkgs) --upgrade --no-cache-dir`)
end
