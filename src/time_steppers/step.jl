"""
    $FUNCTIONNAME(method; setup, psolver, u, temp, t, n)

Create time stepper.
"""
function create_stepper end

"""
    $FUNCTIONNAME(method, stepper, Δt; θ = nothing)

Perform one time step.

Non-mutating/allocating/out-of-place version.

See also [`timestep!`](@ref).
"""
function timestep end

"""
    $FUNCTIONNAME(method, stepper, Δt; θ = nothing, cache)

Perform one time step>

Mutating/non-allocating/in-place version.

See also [`timestep`](@ref).
"""
function timestep! end

include("step_one_leg.jl")
include("step_explicit_runge_kutta.jl")
include("step_lmwray3.jl")
