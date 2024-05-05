```@meta
CurrentModule = IncompressibleNavierStokes
```

# Pressure solvers

The discrete pressure Poisson equation
```math
L p = W M F(u)
```
enforces divergence freeness. There are multiple options for solving this
system.

```@docs
default_psolver
psolver_direct
psolver_cg
psolver_cg_matrix
psolver_spectral
psolver_spectral_lowmemory
pressure
pressure!
poisson
poisson!
applypressure!
project
project!
```
