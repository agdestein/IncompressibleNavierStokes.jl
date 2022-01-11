```@meta
CurrentModule = IncompressibleNavierStokes
```

# API Reference

## Grid

```@docs
create_grid
```

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
Case
Grid
Operators
Time
SolverSettings
BC
Setup
```

## Spatial

```@docs
nonuniform_grid
```

## Pressure solvers

```@docs
DirectPressureSolver
CGPressureSolver
FourierPressureSolver
```

## Main driver

```@docs
create_boundary_conditions
build_operators!
create_initial_conditions
set_bc_vectors!
solve
get_velocity
momentum!
```

## Plot

```@docs
plot_pressure
plot_streamfunction
plot_vorticity
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


## ODE methods

```@docs
AdamsBashforthCrankNicolsonMethod
OneLegMethod
ExplicitRungeKuttaMethod
ImplicitRungeKuttaMethod
runge_kutta_method
```


### Explicit Methods

```@docs
FE11
SSP22
SSP42
SSP33
SSP43
SSP104
rSSPs2
rSSPs3
Wray3
RK56
DOPRI6
```

### Implicit Methods

```@docs
BE11
SDIRK34
ISSPm2
ISSPs3
```

### Half explicit methods

```@docs
HEM3
HEM3BS
HEM5
```

### Classical Methods

```@docs
GL1
GL2
GL3
RIA1
RIA2
RIA3
RIIA1
RIIA2
RIIA3
LIIIA2
LIIIA3
```

### Chebyshev methods

```@docs
CHDIRK3
CHCONS3
CHC3
CHC5
```

### Miscellaneous Methods

```@docs
Mid22
MTE22
CN22
Heun33
RK33C2
RK33P2
RK44
RK44C2
RK44C23
RK44P2
```

### DSRK Methods

```@docs
DSso2
DSRK2
DSRK3
```

### "Non-SSP" Methods of Wong & Spiteri

```@docs
NSSP21
NSSP32
NSSP33
NSSP53
```
