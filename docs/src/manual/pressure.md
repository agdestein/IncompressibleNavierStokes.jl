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

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["pressure.jl"]
```
