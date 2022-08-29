"""
    TimeStepper

Time stepper for solving ODEs.
"""
Base.@kwdef struct TimeStepper{M,T,N,S<:Setup{T,N},P<:AbstractPressureSolver{T}}
    method::M
    setup::S
    pressure_solver::P
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    Vₙ::Vector{T}
    pₙ::Vector{T}
    tₙ::T
end

AdamsBashforthCrankNicolsonStepper{T} = TimeStepper{AdamsBashforthCrankNicolsonMethod{T}}
OneLegStepper{T} = TimeStepper{OneLegMethod{T}}
ExplicitRungeKuttaStepper{T} = TimeStepper{ExplicitRungeKuttaMethod{T}}
ImplicitRungeKuttaStepper{T} = TimeStepper{ImplicitRungeKuttaMethod{T}}
