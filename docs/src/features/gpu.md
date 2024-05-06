# GPU Support

IncompressibleNavierStokes supports various array types. The desired array type
only has to be passed to the [`Setup`](@ref) function:

```julia
using CUDA
setup = Setup(x...; kwargs..., ArrayType = CuArray)
```

All operators have been
made are backend agnostic by using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/).
Even if a GPU is not available, the operators are multithreaded if
Julia is started with multiple threads (e.g. `julia -t 4`)

- This has been tested with CUDA compatible GPUs.
- This has not been tested with other GPU interfaces, such as
    - [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
    - [Metal.jl](https://github.com/JuliaGPU/Metal.jl)
    - [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl)
  If they start supporting sparse matrices and fast Fourier transforms they
  could also be used. 

!!! note "`psolver_direct` on CUDA"
    To use a specialized linear solver for CUDA, make sure to install and
    `using` CUDA.jl and CUDSS.jl. Then `psolver_direct` will automatically use
    the CUDSS solver.
