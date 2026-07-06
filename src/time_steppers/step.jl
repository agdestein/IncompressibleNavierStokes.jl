"""
    $FUNCTIONNAME(method; setup, psolver, state, t, n = 0)

Create time stepper.
"""
function create_stepper end

"""
    $FUNCTIONNAME(method, force, stepper, Δt; params = nothing)

Perform one time step.

Non-mutating/allocating/out-of-place version.

See also [`timestep!`](@ref).
"""
function timestep end

"""
    $FUNCTIONNAME(method, force!, stepper, Δt; params = nothing, ode_cache, force_cache)

Perform one time step.

Mutating/non-allocating/in-place version.

See also [`timestep`](@ref).
"""
function timestep! end

include("step_explicit_runge_kutta.jl")
include("step_lmwray3.jl")
