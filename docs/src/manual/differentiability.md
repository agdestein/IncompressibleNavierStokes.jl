```@meta
CurrentModule = IncompressibleNavierStokes
```

# Differentiating through the code

IncompressibleNavierStokes is
[reverse-mode differentiable](https://juliadiff.org/ChainRulesCore.jl/stable/index.html#Reverse-mode-AD-rules-(rrules)),
which means that you can back-propagate gradients through the code.
Two AD libraries are currently supported:
* **[Zygote.jl](https://github.com/FluxML/Zygote.jl)**: it is the default AD library in the Julia ecosystem and is the most widely used.
* **[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)**: currently has low coverage over the Julia programming language, however it is usually the most efficient if applicable.

## Automatic differentiation with Zygote

Zygote.jl is the default choice for AD backend because it is easy to understand, compatible with most of the Julia ecosystem and good with vectorized code and BLAS.
This comes at a cost however, as intermediate velocity fields need to be stored
in memory for use in the backward pass. For this reason, many of the operators
come in two versions: a slow differentiable allocating non-mutating variant (e.g.
[`divergence`](@ref)) and fast non-differentiable non-allocating mutating
variant (e.g. [`divergence!`](@ref).)

!!! warning "Zygote limitation: array mutation"
    To make your code differentiable, you must use the differentiable versions
    of the operators (without the exclamation marks).

### Example: Gradient of kinetic energy

To differentiate outputs of a simulation with respect to the initial conditions,
make a time stepping loop composed of differentiable operations:

```@example Differentiability
using IncompressibleNavierStokes

n = 100
ax = range(0, 1, n + 1)
setup = Setup(;
    x = (ax, ax),
    boundary_conditions = (;
        u = (
            (PeriodicBC(), PeriodicBC()),
            (PeriodicBC(), PeriodicBC()),
        ),
    ),
)
psolver = default_psolver(setup)
method = LMWray3()
Δt = 0.001
nstep = 100
(; Iu) = setup
function final_energy(u)
    state = (; u)
    stepper = create_stepper(method; setup, psolver, state, t = 0.0)
    for it = 1:nstep
        stepper = timestep(
            method, navierstokes, stepper, Δt;
            params = (; viscosity = 2e-3),
        )
    end
    (; u) = stepper.state
    E = sum(abs2, u[Iu[1], 1]) / 2n^2 + sum(abs2, u[Iu[2], 2]) / 2n^2
end

u = random_field(setup)
final_energy(u)
using Zygote
g, = Zygote.gradient(final_energy, u)

@show size(u) size(g)
```

Now `g` is the gradient of `final_energy` with respect to the initial conditions
`u`, and consequently has the same size.

Note that every operation in the `final_energy` function is non-mutating and
thus differentiable.

## Automatic differentiation with Enzyme

Enzyme.jl is highly-efficient and its ability to perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.
The downside is that restricts the user's defined f function to not do things like require garbage collection or calls to BLAS/LAPACK. However, mutation is supported, meaning that in-place f with fully mutating non-allocating code will work with Enzyme and this will be the most efficient adjoint implementation.

!!! warning "Enzyme limitation: vector returns"
    Enzyme's autodiff function can only handle functions with scalar output. To implement pullbacks for array-valued functions, use a mutating function that returns `nothing` and stores its result in one of the arguments, which must be passed wrapped in a Duplicated.
    In IncompressibleNavierStokes, we provide `enzyme_wrapper` to automatically wrap the function and its arguments in the correct way.

### Example: Gradient of the right-hand side

In this example we differentiate the right-hand side of the Navier-Stokes equations with respect to the velocity field `u`:

```@example Differentiability
using Enzyme
ax = range(0, 1, 101)
setup = Setup(;
    x = (ax, ax),
    boundary_conditions = (;
        u = (
            (PeriodicBC(), PeriodicBC()),
            (PeriodicBC(), PeriodicBC()),
        ),
    ),
)
psolver = default_psolver(setup)
u = random_field(setup)
dudt = similar(u)
t = 0.0
f! = right_hand_side!
```
Notice that we are using the mutating (in-place) version of the right-hand side function. This function can not be differentiate by Zygote, which requires the slower non-mutating version of the right-hand side.

We then define the `Dual` part of the input and output, required to store the adjoint values:

```@example Differentiability
ddudt = Enzyme.make_zero(dudt) .+ 1;
du = Enzyme.make_zero(u);
```
Remember that the derivative of the output (also called the *seed*) has to be set to $1$ in order to compute the gradient. In this case the output is the force, that we store mutating the value of `dudt` inside `right_hand_side!`.

Then we pack the parameters to be passed to `right_hand_side!`:

```@example Differentiability
viscosity = 2e-3
params = setup, psolver, viscosity;
params_ref = Ref(params);
```
Now, we call the `autodiff` function from Enzyme:

```@example Differentiability
Enzyme.autodiff(Enzyme.Reverse, f!, Duplicated(dudt,ddudt), Duplicated(u,du), Const(params_ref), Const(t))
```
Since we have passed a `Duplicated` object, the gradient of `u` is stored in `du`.

Finally, we can also compare its value with the one obtained by Zygote differentiating the out-of-place (non-mutating) version of the right-hand side:

```@example Differentiability
f = create_right_hand_side(setup, psolver)
_, zpull = Zygote.pullback(f, u, (; viscosity), 0.0);
@assert zpull(dudt)[1] == du
```
