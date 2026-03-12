```@meta
CurrentModule = IncompressibleNavierStokes
```
# Temperature equation

IncompressibleNavierStokes.jl supports adding a temperature equation, which is
coupled back to the momentum equation through a gravity term
[Sanderse2023](@cite).

To enable the temperature equation, you need to add boundary conditions for
`temp` in setup:

```julia
setup = Setup(;
    kwargs...,
    boundary_conditions = (; 
        u = (bc...),
        temp = (bc...),
    ),
)
```

In `solve_unsteady`, add `force! = boussinesq!` and the following params:

```julia
solve_unsteady(;
    kwargs...,
    start = (; u, temp), # Initial vector field `u` and scalar field `temp`
    force! = boussinesq!,
    params = (;
        viscosity = 1e-3, # for diffusion of velocity
        conductivity = 1e-3, # for diffusion of temperature
        gravity = 1.0, # Gravity constant in momentum equation
        gdir = 2, # Direction of gravity
        dodissipation = true, # Redirect velocity dissipation to temperature
    ),
)
```
