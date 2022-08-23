"""
    TimeStepper

Time stepper for solving ODEs.
"""
Base.@kwdef struct TimeStepper{M,T,N,VV,C,F,P}
    method::M
    setup::Setup{T,N,VV,C,F}
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
