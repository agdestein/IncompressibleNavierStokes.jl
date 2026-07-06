```@meta
CurrentModule = IncompressibleNavierStokes
```

# Pressure solvers

The discrete pressure Poisson equation

```math
L p = W M F(u)
```

enforces divergence freeness at every Runge-Kutta stage (see [Spatial and
temporal discretization](discretization.md)). Since this is the only globally
coupled (and thus most expensive) part of a time step, choosing an
appropriate solver matters:

- [`psolver_spectral`](@ref): FFT-based solver. The fastest option, but
  requires a uniform grid with periodic boundary conditions in all
  directions.
- [`psolver_transform`](@ref): FFT/DCT-based solver for uniform grids with
  periodic and Dirichlet boundary conditions.
- [`psolver_direct`](@ref): sparse direct solver (factorize once, solve every
  step). Works for all grids and boundary conditions, but the factorization
  only supports `Float64` on the CPU, and memory usage can be prohibitive for
  large 3D problems. On CUDA arrays, loading
  [CUDSS.jl](https://github.com/exanauts/CUDSS.jl) makes this solver use a
  GPU factorization instead.
- [`psolver_cg`](@ref): matrix-free conjugate gradient solver. Works on all
  backends and precisions; the go-to fallback when the direct solver does
  not apply.
- [`psolver_cg_AMGX`](@ref): conjugate gradient solver with algebraic
  multigrid preconditioning from NVIDIA
  [AMGX](https://github.com/NVIDIA/AMGX) (requires CUDA and AMGX.jl).

The default ([`default_psolver`](@ref)) selects the spectral solver for
uniform periodic setups and the direct solver otherwise.

## API

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["pressure.jl"]
```
