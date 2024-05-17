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
max_size
cosine_grid
stretched_grid
tanh_grid
```

## Preprocess

```@docs
create_initial_conditions
random_field
```

## Processors

```@docs
processor
timelogger
observefield
vtk_writer
fieldsaver
realtimeplotter
fieldplot
energy_history_plot
energy_spectrum_plot
animator
```

## Solvers

```@docs
get_cfl_timestep!
solve_unsteady
solve_steady_state
```

## Utils

```@docs
save_vtk
plotgrid
get_lims
plotmat
```
