```@meta
CurrentModule = IncompressibleNavierStokes
```

# API Reference

```@docs
IncompressibleNavierStokes
Setup
```

## Boundary conditions

```@docs
BoundaryConditions
get_bc_vectors
```

## Force

```@docs
SteadyBodyForce
UnsteadyBodyForce
```

## Grid

```@docs
dimension
Grid
cosine_grid
max_size
stretched_grid
```

## Visocosity Models

```@docs
AbstractViscosityModel
LaminarModel
MixingLengthModel
SmagorinskyModel
QRModel
```

## Convection Models

```@docs
AbstractConvectionModel
NoRegConvectionModel
C2ConvectionModel
C4ConvectionModel
LerayConvectionModel
```

## Momentum

```@docs
MomentumCache
bodyforce
bodyforce!
check_symmetry
compute_conservation
convection
convection!
convection_components
convection_components!
diffusion
diffusion!
momentum
momentum!
momentum_allstage
momentum_allstage!
strain_tensor
turbulent_K
turbulent_viscosity
```

## Operators

```@docs
Operators
operator_averaging
operator_convection_diffusion
operator_divergence
operator_interpolation
operator_postprocessing
operator_regularization
operator_turbulent_diffusion
operator_viscosity
```

## Postprocess

```@docs
get_streamfunction
get_velocity
get_vorticity
vorticity!
plot_force
plot_grid
plot_pressure
plot_streamfunction
plot_tracers
plot_velocity
plot_vorticity
save_vtk
```

## Preprocess

```@docs
create_initial_conditions
```

## Problems

```@docs
SteadyStateProblem
UnsteadyProblem
is_steady
```

## Processors

```@docs
AbstractProcessor
Logger
VTKWriter
QuantityTracer
StateObserver
initialize!
process!
finalize!
real_time_plot
```

## Solvers

```@docs
get_timestep
solve
solve_animate
```

### Pressure solvers

```@docs
AbstractPressureSolver
DirectPressureSolver
CGPressureSolver
FourierPressureSolver
pressure_additional_solve
pressure_additional_solve!
pressure_poisson
pressure_poisson!
```

## Time steppers

```@docs
AbstractODEMethod
AbstractRungeKuttaMethod
AdamsBashforthCrankNicolsonMethod
OneLegMethod
ExplicitRungeKuttaMethod
ImplicitRungeKuttaMethod

TimeStepper

AbstractODEMethodCache
ExplicitRungeKuttaCache
ImplicitRungeKuttaCache
AdamsBashforthCrankNicolsonCache
OneLegCache

change_time_stepper
isexplicit
lambda_conv_max
lambda_diff_max
needs_startup_method
nstage
ode_method_cache
runge_kutta_method
step
step!
```

## Utils

```@docs
filter_convection
filter_convection!
get_lims
```
