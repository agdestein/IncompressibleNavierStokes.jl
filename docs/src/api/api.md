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

## Convection Models

```@docs
AbstractConvectionModel
NoRegConvectionModel
C2ConvectionModel
C4ConvectionModel
LerayConvectionModel
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
timelogger
vtk_writer
fieldsaver
realtimeplotter
fieldplot
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

## Utils

```@docs
get_lims
plotmat
```
