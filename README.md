# BayesianInference.jl

A Julia wrapper for the [BayesInference (BI)](https://github.com/your-bi-repo) Python library, providing seamless Bayesian inference capabilities in Julia with JAX-based backends.

## Features

- 🚀 **Seamless Python-Julia Integration**: Call Python's BayesInference library directly from Julia with native syntax
- 🔧 **JAX Backend Support**: Leverage JAX and NumPyro for high-performance Bayesian inference
- 📊 **Matplotlib Integration**: Display plots directly in Julia with the `@pyplot` macro
- 🎯 **Model Definition Macro**: Use `@BI` to define Bayesian models with proper Python interoperability
- 🔢 **Array Interoperability**: Automatic conversion between Julia and JAX arrays

## Installation

### From Julia Registry (after registration)

```julia
using Pkg
Pkg.add("BayesianInference")
```

### Development Installation

```julia
using Pkg
Pkg.add(url="https://github.com/BGN-for-ASNA/BIJ")
```

Or clone the repository and activate it locally:

```bash
git clone https://github.com/BGN-for-ASNA/BIJ.git
cd BIJ
julia --project=.
```

Then in Julia:
```julia
using Pkg
Pkg.instantiate()
using BayesianInference
```

## Quick Start

```julia
using BayesianInference
using PythonCall

# Initialize BI
m = importBI()

# Generate some data
x = m.dist.normal(0, 1, shape=(100,), sample=true)
y = m.dist.normal(0.2 + 0.6 * x, 1.2, sample=true)

# Define a Bayesian linear regression model
@BI function linear_model(; x, y)
    alpha = m.dist.normal(loc=0, scale=1, name="alpha")
    beta  = m.dist.normal(loc=0, scale=1, name="beta")
    sigma = m.dist.exponential(1, name="sigma")
    mu = alpha + beta * x
    m.dist.normal(mu, sigma, obs=y)
end

# Fit the model
m.fit(linear_model, num_warmup=1000, num_samples=1000, num_chains=1)

# Display results
m.summary()

# Plot results with @pyplot
@pyplot begin
    m.plot_trace()
    plt.tight_layout()
end
```

## Key Components

### `importBI()`
Initialize the BayesInference Python module with configurable options:

```julia
m = importBI(
    platform="cpu",        # or "gpu", "tpu"
    cores=nothing,         # number of CPU cores
    rand_seed=true,        # set random seed
    backend="numpyro"      # backend to use
)
```

### `@BI` Macro
Define Bayesian models that are compatible with Python's inspection requirements:

```julia
@BI function my_model(data, params)
    # Your model definition
end
```

### `@pyplot` Macro
Execute matplotlib plotting commands and display results in Julia:

```julia
@pyplot begin
    plt.plot(x, y)
    plt.title("My Plot")
end
```

### JAX Integration
Use JAX's NumPy API directly:

```julia
# jnp and jax are available as global constants
arr = jnp.array([1, 2, 3, 4, 5])
result = jnp.sum(arr)
```

## Dependencies

- **Julia**: ≥ 1.12
- **PythonCall.jl**: For Python interoperability
- **CondaPkg.jl**: Automatic Python environment management

Python dependencies (installed automatically via CondaPkg):
- BayesInference
- JAX
- NumPyro
- Matplotlib

## Documentation

For detailed usage examples, see:
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [test/usage_example.ipynb](test/usage_example.ipynb) - Jupyter notebook examples

## Platform Support

- ✅ Linux
- ✅ macOS
- ✅ Windows

GPU support available on compatible systems with JAX GPU installation.

## License

This package is licensed under the [GNU General Public License v3.0](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite the underlying BayesInference library.

## Related Packages

- [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl) - Python interoperability
- [Turing.jl](https://github.com/TuringLang/Turing.jl) - Native Julia Bayesian inference

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/BGN-for-ASNA/BIJ/issues)
- See the [QUICKSTART.md](QUICKSTART.md) for troubleshooting tips
