# Temperature equation

IncompressibleNavierStokes.jl supports adding a temperature equation, which is
coupled back to the momentum equation through a gravity term.

To enable the temperature equation, you need to set the `temperature` keyword
in setup:

```julia
setup = Setup(
    args...;
    kwargs...,
    temperature = temperature_equation(; kwargs...),
)
```

```@docs
temperature_equation
```

Some operators are available for the temperature equation:
```@docs
gravity!
convection_diffusion_temp!
dissipation!
```
