# Floating point precision

IncompressibleNavierStokes generates efficient code for different floating
point precisions, such as

- Double precision (`Float64`)
- Single precision (`Float32`)
- Half precision (`Float16`)

To use single or half precision, all user input floats should be converted to
the desired type. Mixing different precisions causes unnecessary conversions
and may break the code. See the [tutorial](../generated/LidDrivenCavity2D.md) for
an example on how to enforce floating point type hygiene.

!!! note "GPU precision"
    For GPUs, single precision is preferred. `CUDA.jl`s `cu` converts to
    single precision.

!!! note "Pressure solvers"
    [`SparseArrays.jl`](https://github.com/JuliaSparse/SparseArrays.jl)s
    sparse matrix factorizations only support double precision.
    [`DirectPressureSolver`](@ref) only works for `Float64`. Consider
    using an iterative solver such as [`CGPressureSolver`](@ref)
    when using single or half precision.
