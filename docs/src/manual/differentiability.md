```@meta
CurrentModule = IncompressibleNavierStokes
```

# Differentiating through the code

IncompressibleNavierStokes is
[reverse-mode differentiable](https://juliadiff.org/ChainRulesCore.jl/stable/index.html#Reverse-mode-AD-rules-(rrules)),
which means that you can back-propagate gradients through the code.
This comes at a cost however, as intermediate velocity fields need to be stored
in memory for use in the backward pass.  For this reason, many of the operators
come with a slow differentiable allocating non-mutating variant (e.g.
[`divergence`](@ref)) and fast non-differentiable non-allocating mutating
variant (e.g. [`divergence!`](@ref).)

!!! warning "Differentiable code"
    To make your code differentiable, you must use the differentiable versions
    of the operators (without the exclamation marks).

To differentiate the code, use [Zygote.jl](https://github.com/FluxML/Zygote.jl).

## Example: Gradient of kinetic energy

To differentiate outputs of a simulation with respect to the initial conditions,
make a time stepping loop composed of differentiable operations:

```julia
import IncompressibleNavierStokes as INS
setup = INS.Setup(0:0.01:1, 0:0.01:1; Re = 500.0)
psolver = INS.default_psolver(setup)
method = INS.RKMethods.RK44P2()
Δt = 0.01
nstep = 100
(; Iu) = setup.grid
function final_energy(u)
    stepper = INS.create_stepper(method; setup, psolver, u, temp = nothing, t = 0.0)
    for it = 1:nstep
        stepper = INS.timestep(method, stepper, Δt)
    end
    (; u) = stepper
    sum(abs2, u[1][Iu[1]]) / 2 + sum(abs2, u[2][Iu[2]]) / 2
end

u = INS.random_field(setup)

using Zygote
g, = Zygote.gradient(final_energy, u)

@show size.(u)
@show size.(g)
```

Now `g` is the gradient of `final_energy` with respect to the initial conditions
`u`, and consequently has the same size.

Note that every operation in the `final_energy` function is non-mutating and
thus differentiable.
