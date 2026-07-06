# GPU support and floating point precision

## Backends

All operators are matrix-free
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/)
kernels, so the same code runs on different backends. The desired backend
only has to be passed to the [`Setup`](@ref) function:

```julia
using CUDA
setup = Setup(; kwargs..., backend = CUDABackend())
```

On the CPU (the default backend), the kernels are multithreaded if Julia is
started with multiple threads (e.g. `julia -t auto`).

The package is developed and tested with CUDA-compatible GPUs. Other
KernelAbstractions backends
([AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl),
[Metal.jl](https://github.com/JuliaGPU/Metal.jl),
[oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl)) are untested; the
operators themselves are backend-agnostic, so the main constraint is the
[pressure solver](pressure.md) — the spectral solvers need FFT plans for the
given array type, and the direct solver needs a factorization
(`psolver_cg` is fully matrix-free and should work anywhere).

!!! note "`psolver_direct` on CUDA"
    To use a specialized direct solver for CUDA, install and `using` both
    CUDA.jl and CUDSS.jl. Then [`psolver_direct`](@ref) will automatically
    use the CUDSS solver.

## Floating point precision

IncompressibleNavierStokes generates efficient code for different floating
point precisions, such as

- Double precision (`Float64`)
- Single precision (`Float32`)
- Half precision (`Float16`)

To use single or half precision, all user input floats should be converted to
the desired type, starting with the grid vectors in [`Setup`](@ref). Mixing
different precisions causes unnecessary conversions and may break the code.

!!! note "GPU precision"
    For GPUs, single precision is preferred. `CUDA.jl`s `cu` converts to
    single precision.

!!! note "Pressure solvers"
    [`SparseArrays.jl`](https://github.com/JuliaSparse/SparseArrays.jl)s
    sparse matrix factorizations only support double precision, so
    [`psolver_direct`](@ref) only works for `Float64` on the CPU. Consider
    using an iterative solver such as [`psolver_cg`](@ref)
    when using single or half precision.
