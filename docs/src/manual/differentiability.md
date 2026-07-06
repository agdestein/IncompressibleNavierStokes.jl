```@meta
CurrentModule = IncompressibleNavierStokes
```

# Differentiating through the code

IncompressibleNavierStokes is reverse-mode differentiable, which means that
you can back-propagate gradients through a simulation, e.g. to optimize
initial conditions, force parameters, or neural closure models. Two AD
libraries are supported, both backed by the same hand-written adjoint
kernels:

- **[Zygote.jl](https://github.com/FluxML/Zygote.jl)**, via ChainRulesCore
  rrules. Zygote does not support array mutation, so your simulation code
  must be composed of the *non-mutating* operator variants (e.g.
  [`divergence`](@ref), [`timestep`](@ref)). These allocate intermediate
  fields that are kept in memory for the backward pass.
- **[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)**, via EnzymeCore
  rules. Enzyme differentiates the fast *mutating* variants (e.g.
  [`divergence!`](@ref), [`right_hand_side!`](@ref)) and is typically the
  most efficient option when it applies.

!!! warning "Enzyme limitation: array returns"
    Enzyme's `autodiff` requires functions with scalar output. To compute
    pullbacks of array-valued functions, use a mutating function that
    returns `nothing` and stores its result in an argument wrapped in
    `Duplicated`.

For complete, runnable demonstrations of both approaches — computing the
gradient of the final kinetic energy with respect to the initial conditions
with Zygote, and differentiating the right-hand side with Enzyme — see the
[differentiability example](../examples/generated/Differentiability.md).
