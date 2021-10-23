"""
    step!(stepper, Δt)

Take a step of size `Δt` with the given time stepper.
"""
function step! end

include("step_ab_cn.jl")
include("step_one_leg.jl")
include("step_explicit_runge_kutta.jl")
include("step_implicit_runge_kutta.jl")
