# GPU Support

If an Nvidia GPU is available, the default CPU solve call

```julia
solve_unsteady(setup, V₀, p₀, tlims; kwargs...)
```

can now be replaced with the following:

```julia
using CUDA
solve_unsteady(
    setup, V₀, p₀, tlims;
    device = cu,
    kwargs...
)
```

This moves the arrays and sparse operators to the GPU, outsourcing all array operations to the GPU.

Limitations:

- [`DirectPressureSolver`](@ref) is currently not supported on the GPU. Use [`CGPressureSolver`](@ref) instead.
- Unsteady boundary conditions are currently not supported on the GPU.
- The code uses sparse matrices for discretization. For finer grids, these can take up a lot of memory on the GPU.
- This has not been tested with other GPU interfaces, such as
    - [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
    - [Metal.jl](https://github.com/JuliaGPU/Metal.jl)
    - [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl)
  If they start supporting sparse matrices and fast Fourier transforms they
  could also be used. Alternatively, IncompressibleNavierStokes may also be
  refactored to apply the operators without assembling any sparse arrays.
