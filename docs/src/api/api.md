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

## Preprocess

```@docs
create_initial_conditions
random_field
```

## Processors

```@docs
processor
timelogger
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
get_timestep
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
