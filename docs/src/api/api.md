```@meta
CurrentModule = IncompressibleNavierStokes
```

# API Reference

```@docs
IncompressibleNavierStokes
Setup
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
divergence
divergence!
vorticity
vorticity!
convection!
diffusion!
bodyforce!
momentum!
```

## Postprocess

```@docs
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
timestep
timestep!
```

## Utils

```@docs
get_lims
plotmat
```

## Other

```@docs
DirichletBC
SymmetricBC
PressureBC
mean_squared_error
cnn
relative_error
create_randloss
kinetic_energy
FourierLayer
create_callback
offset_p
momentum_allstage
momentum_allstage!
fno
offset_u
pressuregradient
pressuregradient!
```
