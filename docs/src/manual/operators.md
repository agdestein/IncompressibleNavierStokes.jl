```@meta
CurrentModule = IncompressibleNavierStokes
```

# Operators

All discrete operators are implemented as matrix-free kernels using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/)
and Cartesian indices, similar to
[WaterLily.jl](https://github.com/weymouth/WaterLily.jl/).
This allows for dimension- and backend-agnostic code; the same kernels run on
CPU and GPU. See this
[blog post](https://b-fg.github.io/research/2023-07-05-waterlily-on-gpu.html)
for how such kernels are written.

Each operator comes in two variants:

- A fast mutating variant with an exclamation mark (e.g.
  [`divergence!`](@ref)), which writes its result into a preallocated output
  array.
- A non-mutating variant (e.g. [`divergence`](@ref)), which allocates its
  output and is differentiable with reverse-mode automatic differentiation
  (see [Differentiating code](differentiability.md)).

The operators act on fields that include ghost volumes, so boundary
conditions must be applied first with [`apply_bc_u`](@ref) and friends (see
[Problem setup](setup.md)). Sparse matrix versions of the linear operators
are also available, see [Sparse matrices](matrices.md).

## API

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["operators.jl"]
```
