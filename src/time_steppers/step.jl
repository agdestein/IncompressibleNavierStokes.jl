"""
    step(stepper, Δt; bc_vectors = nothing)

Perform one time step.

Non-mutating/allocating/out-of-place version.

See also [`step!`](@ref).
"""
function timestep end
# step(stepper, Δt; bc_vectors = nothing) = step(stepper.method, stepper, Δt; bc_vectors = nothing)

"""
    step!(stepper, Δt; cache, momentum_cache, bc_vectors = nothing)

Perform one time step>

Mutating/non-allocating/in-place version.

See also [`step`](@ref).
"""
function timestep! end
# step!(stepper, Δt; kwargs...) = step!(stepper.method, stepper, Δt; kwargs...)

include("step_ab_cn.jl")
include("step_one_leg.jl")
include("step_explicit_runge_kutta.jl")
include("step_implicit_runge_kutta.jl")
