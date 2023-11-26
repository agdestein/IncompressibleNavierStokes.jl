# GPU Support

IncompressibleNavierStokes supports various array types. The desired array type
only has to be passed to the [`Setup`](@ref) function. All operators have been
made are backend agnostic by using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/).
Even if a GPU is not available, the operators are multithreaded if  Julia is started with multiple threads (e.g. `julia -t 4`)

Limitations:

- [`DirectPressureSolver`](@ref) is currently used on the CPU with double precision. [`CGPressureSolver`](@ref) works on the GPU.
- This has not been tested with other GPU interfaces, such as
    - [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
    - [Metal.jl](https://github.com/JuliaGPU/Metal.jl)
    - [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl)
  If they start supporting sparse matrices and fast Fourier transforms they
  could also be used. 
