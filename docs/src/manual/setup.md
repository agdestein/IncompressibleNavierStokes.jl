```@meta
CurrentModule = IncompressibleNavierStokes
```

# Problem setup

## Boundary conditions

Each boundary has exactly one type of boundary conditions. For periodic
boundary conditions, the opposite boundary must also be periodic.

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages   = ["boundary_conditions.jl"]
```

## Setup

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages   = ["grid.jl"]
```

## Field initializers

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["initializers.jl"]
```
