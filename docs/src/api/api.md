```@meta
CurrentModule = IncompressibleNavierStokes
```

# API Reference

## Force

```@docs
SteadyBodyForce
UnsteadyBodyForce
```

## Processors

```@docs
Logger
RealTimePlotter
VTKWriter
QuantityTracer
```

## Problems

```@docs
SteadyStateProblem
UnsteadyProblem
is_steady
```

## Setup

```@docs
Grid
Operators
SolverSettings
BoundaryConditions
Setup
```

## One-dimensional grids

```@docs
stretched_grid
cosine_grid
```

## Pressure solvers

```@docs
DirectPressureSolver
CGPressureSolver
FourierPressureSolver
```

## Main driver

```@docs
create_initial_conditions
set_bc_vectors!
solve
get_velocity
momentum!
```

## Plot

```@docs
plot_grid
plot_pressure
plot_velocity
plot_vorticity
plot_streamfunction
plot_tracers
```

## Visocosity Models

```@docs
LaminarModel
KEpsilonModel
MixingLengthModel
SmagorinskyModel
QRModel
```

## Convection Models

```@docs
NoRegConvectionModel
C2ConvectionModel
C4ConvectionModel
LerayConvectionModel
```
