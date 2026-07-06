```@meta
CurrentModule = IncompressibleNavierStokes
```

# Problem setup

A problem is defined by a call to [`Setup`](@ref), which precomputes all grid
quantities:

```julia
setup = Setup(;
    x = (range(0.0, 2.0, 129), range(0.0, 1.0, 65)),
    boundary_conditions = (;
        u = (
            (DirichletBC((1.0, 0.0)), PressureBC()), # x-left, x-right
            (DirichletBC(), DirichletBC()),          # y-bottom, y-top
        ),
    ),
)
```

## Grid

The tuple `x` contains one vector of volume boundary coordinates per
dimension (so a 2D grid with `N × M` volumes needs vectors of length `N + 1`
and `M + 1`). Uniform grids are created with `range`; non-uniform grids can
be created with [`stretched_grid`](@ref), [`cosine_grid`](@ref), or
[`tanh_grid`](@ref), or any custom monotone vector. The grid can be
visualized with [`plotgrid`](@ref).

Note that some functionality requires a uniform grid in one or more
directions: the spectral pressure solvers and the [spectral
quantities](postprocessing.md) require uniformity in the periodic directions.

## Boundary conditions

The available boundary condition types are [`PeriodicBC`](@ref),
[`DirichletBC`](@ref), [`SymmetricBC`](@ref), and [`PressureBC`](@ref).
They are given per field, dimension, and side:
`boundary_conditions.u[α][s]` is the boundary condition for the velocity in
direction `α` on the low (`s = 1`) or high (`s = 2`) side. Each boundary has
exactly one type of boundary condition, and for periodic boundary conditions
the opposite boundary must also be periodic. For simulations with a
[temperature equation](temperature.md), add a `temp` field with boundary
conditions for the temperature.

Boundary conditions are enforced by filling the ghost volumes of the field
arrays, see [Spatial and temporal discretization](discretization.md).

## Initial conditions

Fields are allocated with [`scalarfield`](@ref) and [`vectorfield`](@ref),
and initialized with [`velocityfield`](@ref) and [`temperaturefield`](@ref)
from a given analytical expression. For periodic boxes,
[`random_field`](@ref) creates a random velocity field with a prescribed
energy spectrum that is exactly divergence free on the staggered grid.

## API

### Setup and grid

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages   = ["grid.jl"]
```

### Boundary conditions

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages   = ["boundary_conditions.jl"]
```

### Field initializers

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["initializers.jl"]
```
