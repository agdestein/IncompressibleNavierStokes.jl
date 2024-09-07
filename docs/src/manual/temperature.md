```@meta
CurrentModule = IncompressibleNavierStokes
```
# Temperature equation

IncompressibleNavierStokes.jl supports adding a temperature equation, which is
coupled back to the momentum equation through a gravity term
[Sanderse2023](@cite).

To enable the temperature equation, you need to set the `temperature` keyword
in setup:

```julia
setup = Setup(
    args...;
    kwargs...,
    temperature = temperature_equation(; kwargs...),
)
```

where `temperature_equation` can be configured as follows:

```@docs; canonical = false
temperature_equation
```
