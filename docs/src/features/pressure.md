# Pressure solvers

The discrete pressure Poisson equation
```math
L p = W M F(u)
```
enforces divergence freeness. There are multiple options for solving this
system.

```@docs
AbstractPressureSolver
DirectPressureSolver
CGPressureSolver
CGPressureSolverManual
SpectralPressureSolver
pressure_additional_solve
pressure_additional_solve!
pressure_poisson
pressure_poisson!
```
