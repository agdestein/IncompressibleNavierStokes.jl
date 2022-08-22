"""
    TimeStepper

Time stepper for solving ODEs.
"""
Base.@kwdef mutable struct TimeStepper{M,T,N,VV,C,F,P}
    method::M
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    Vₙ::Vector{T}
    pₙ::Vector{T}
    tₙ::T
    Δtₙ::T
    setup::Setup{T,N,VV,C,F}
    pressure_solver::P
    cache::AbstractODEMethodCache{T}
    momentum_cache::MomentumCache{T}
end

"""
    TimeStepper(method, setup, V₀, p₀, t, Δt) -> TimeStepper

Build associated time stepper from method.
"""
function TimeStepper(method::M, setup::Setup{T,N,VV,C,F}, pressure_solver::P, V₀, p₀, t, Δt) where {M,T,N,VV,C,F,P}
    # Initialize solution vectors (leave input intact)
    n = 0
    V = copy(V₀)
    p = copy(p₀)

    # Current solution
    Vₙ = copy(V)
    pₙ = copy(p)
    tₙ = t
    Δtₙ = Δt

    # Temporary variables
    cache = ode_method_cache(method, setup)
    momentum_cache = MomentumCache(setup)

    TimeStepper{M,T,N,VV,C,F,P}(; method, n, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, pressure_solver, cache, momentum_cache)
end

const AdamsBashforthCrankNicolsonStepper{S} = TimeStepper{AdamsBashforthCrankNicolsonMethod{S}}
const OneLegStepper{S} = TimeStepper{OneLegMethod{S}}
const ExplicitRungeKuttaStepper{S} = TimeStepper{ExplicitRungeKuttaMethod{S}}
const ImplicitRungeKuttaStepper{S} = TimeStepper{ImplicitRungeKuttaMethod{S}}
