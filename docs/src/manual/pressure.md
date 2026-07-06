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
- [`psolver_tridiagonal`](@ref): FFT/tri-diagonal solver for channel-like
  setups (one wall-bounded direction, periodic otherwise). The periodic
  directions must be uniform, but the wall-normal direction may be
  stretched. Works on the GPU.
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
uniform periodic setups, the tri-diagonal solver for channel-like setups,
and the direct solver otherwise.

## Preconditioning the matrix-free CG solver

[`psolver_cg`](@ref) is matrix-free: it only applies the Laplacian stencil
([`laplacian!`](@ref)), so it runs on any backend. It currently uses a
diagonal (Jacobi) preconditioner, which parallelizes trivially but does not
improve the mesh-dependent conditioning. GPU-friendly upgrades, roughly in
order of implementation effort:

- **Polynomial preconditioning** (Chebyshev or truncated Neumann series):
  applies the same Laplacian kernel a few times per iteration; no setup
  phase, no extra storage, fully matrix-free.
- **Fast-solver preconditioning**: use an FFT-based solver
  ([`psolver_spectral`](@ref), [`psolver_transform`](@ref), or
  [`psolver_tridiagonal`](@ref)) for a nearby constant-coefficient problem
  as the preconditioner. One FFT round-trip per iteration, and typically a
  mesh-independent iteration count when the grid is a smooth deformation of
  a uniform one.
- **Geometric multigrid**: matrix-free smoothers (damped Jacobi/Chebyshev)
  plus coarsening kernels; mesh-independent convergence for all supported
  grids and boundary conditions.
- **Algebraic multigrid**: needs the assembled matrix; on CUDA this already
  exists as [`psolver_cg_AMGX`](@ref).

Incomplete factorizations (IC(0)/ILU) are *not* a good fit: the triangular
solves are inherently sequential and perform poorly on GPUs.

## API

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["pressure.jl"]
```
