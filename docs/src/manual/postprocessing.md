```@meta
CurrentModule = IncompressibleNavierStokes
```

# Postprocessing

## Processors

Processors observe the solution inside [`solve_unsteady`](@ref) after every
time step, without having to store the full time history. They are passed as
a named tuple, and their outputs are returned with the same field names:

```julia
state, outputs = solve_unsteady(;
    setup, start, tlims, params,
    processors = (;
        log = timelogger(; nupdate = 100),
        ehist = realtimeplotter(; setup, plot = energy_history_plot, nupdate = 10),
        vtk = vtk_writer(; setup, dir = "output"),
    ),
)
```

The built-in processors are [`timelogger`](@ref),
[`realtimeplotter`](@ref)/[`animator`](@ref) (live plots and animations,
require a Makie backend), [`vtk_writer`](@ref) (for
[ParaView](https://www.paraview.org/)), [`fieldsaver`](@ref),
[`observefield`](@ref), and [`observespectrum`](@ref). Custom processors are
created with [`processor`](@ref).

Note that the `state` observable passed to a processor contains fields
living on the device; for GPU simulations you may have to move them to the
host with `Array` before processing.

## Plotting

Loading a [Makie](https://docs.makie.org/) backend (e.g. CairoMakie or
GLMakie) activates the plotting extension, which provides
[`fieldplot`](@ref), [`energy_history_plot`](@ref), and
[`energy_spectrum_plot`](@ref). These work both as standalone plots of a
state and as `plot` arguments to [`realtimeplotter`](@ref). Derived field
quantities such as vorticity and the Q-criterion are computed with
[`vorticity`](@ref), [`qcrit`](@ref), and [`kinetic_energy`](@ref); since
the velocity components live on staggered positions, they are interpolated
to the pressure points ([`interpolate_u_p`](@ref)) for visualization.

Fields can also be exported to VTK files with [`save_vtk`](@ref) and viewed
in ParaView.

## Energy spectra and turbulence statistics

For uniform periodic grids, [`energyspectrum`](@ref) computes the kinetic
energy spectrum ``\hat{e}(k)``, which can be compared with the theoretical
Kolmogorov scaling ``k^{-5/3}`` in 3D (or ``k^{-3}`` in 2D)
[Pope2000](@cite). [`turbulence_statistics`](@ref) computes quantities such
as the Taylor micro-scale and Kolmogorov length scale. The FFT-based
machinery is RFFT-based and accounts for the missing conjugate modes; reuse
[`spectral_stuff`](@ref) when computing many spectra on the same grid.

## API

### Processors

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["processors.jl"]
```

### Spectral quantities

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["spectral.jl"]
```

### Utils

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["utils.jl"]
```
