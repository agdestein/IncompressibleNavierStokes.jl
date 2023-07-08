# Pressure solvers

The discrete pressure Poisson equation
```math
A p = f
```
enforces divergence freeness. There are three options for solving this system:

- [`DirectPressureSolver`](@ref) factorizes the Laplace matrix ``A`` such that
  the system can be solved for different right hand sides. This currently
  only works for double precision on the CPU.
- [`CGPressureSolver`](@ref) uses conjugate gradients to solve the system for
  different ``f``.
- [`SpectralPressureSolver`](@ref) solves the system in Fourier space, but only
  in the case of a uniform grid with periodic boundary conditions.
