```@meta
CurrentModule = IncompressibleNavierStokes
```

# Solving unsteady problems

The main entry point of the package is [`solve_unsteady`](@ref):

```julia
state, outputs = solve_unsteady(;
    setup,
    start = (; u),
    tlims = (0.0, 1.0),
    params = (; viscosity = 1e-3),
)
```

It steps the initial `start` state (a named tuple of fields, e.g. `(; u)` or
`(; u, temp)`) through the time interval `tlims` and returns the final state
and the [processor](postprocessing.md) outputs.

## Forces and parameters

The right-hand side is a function `force!(force, state, t; setup, cache,
params...)` passed as the `force!` keyword. The built-in forces are
[`navierstokes!`](@ref) (convection and diffusion, the default) and
[`boussinesq!`](@ref) (adds a coupled [temperature equation](temperature.md)).
Physical parameters are passed in the `params` named tuple and forwarded to
`force!` as keyword arguments, e.g. `params = (; viscosity = 1e-3)` for
`navierstokes!`. Custom forces follow the same signature; see
`examples/Kolmogorov2D.jl` for a custom body force and
`examples/ChannelFlow.jl` for adding an eddy-viscosity
[closure model](les.md).

## Time step and method

By default, the time step is chosen adaptively from a CFL condition on the
convective and diffusive limits (tune with the `cfl` keyword); pass a fixed
`Δt` to disable this. The time integration method defaults to the low-storage
third-order Runge-Kutta method [`LMWray3`](@ref); any explicit Runge-Kutta
tableau from [`RKMethods`](@ref) can be passed via the `method` keyword. The
theory is discussed in [Spatial and temporal
discretization](discretization.md).

For finer control than `solve_unsteady`, the stepping primitives
[`create_stepper`](@ref) and [`timestep`](@ref)/[`timestep!`](@ref) can be
called directly, e.g. to write a custom (differentiable) simulation loop.

## API

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = ["solver.jl"]
```

## Time stepping

```@autodocs
Modules = [IncompressibleNavierStokes]
Pages = [
    "time_steppers/methods.jl",
    "time_steppers/step.jl",
    "time_steppers/step_explicit_runge_kutta.jl",
    "time_steppers/step_lmwray3.jl",
    "time_steppers/time_stepper_caches.jl",
]
```

## Runge-Kutta methods

```@autodocs
Modules = [IncompressibleNavierStokes, IncompressibleNavierStokes.RKMethods]
Pages = ["time_steppers/RKMethods.jl"]
```
