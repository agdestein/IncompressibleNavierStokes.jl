```@meta
CurrentModule = IncompressibleNavierStokes
```

# Solvers

## Initial conditions

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["create_initial_conditions.jl"]
```

## Solvers

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["solver.jl"]
```

## Processors

Processors can be used to process the solution in [`solve_unsteady`](@ref) after
every time step.

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["processors.jl"]
```
