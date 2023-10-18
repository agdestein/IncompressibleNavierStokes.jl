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
```

## Grid

```@docs
Dimension
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
operator_filter
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
plot_velocity
plot_vorticity
save_vtk
```

## Preprocess

```@docs
create_initial_conditions
random_field
```

## Processors

```@docs
step_logger
vtk_writer
field_saver
field_plotter
energy_history_plotter
energy_spectrum_plotter
animator
```

## Solvers

```@docs
get_timestep
solve_unsteady
solve_steady_state
```

### Pressure solvers

```@docs
AbstractPressureSolver
DirectPressureSolver
CGPressureSolver
SpectralPressureSolver
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

isexplicit
lambda_conv_max
lambda_diff_max
nstage
ode_method_cache
runge_kutta_method
step
step!
```

## Filter
```@docs
create_top_hat_u
create_top_hat_v
create_top_hat_p
create_top_hat_velocity
```

## Utils

```@docs
filter_convection
filter_convection!
get_lims
plotmat
```
