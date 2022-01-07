"""
    TimeStepper

Time stepper for solving ODEs.
"""
Base.@kwdef mutable struct TimeStepper{M, T}
    method::M
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    Vₙ::Vector{T}
    pₙ::Vector{T}
    tₙ::T
    Δtₙ::T
    setup::Any # Setup{T}
    cache::AbstractODEMethodCache{T}
    momentum_cache::MomentumCache{T}
end

"""
    TimeStepper(method, setup, V₀, p₀, t, Δt) -> TimeStepper

Build associated time stepper from method.
"""
function TimeStepper(method::M, setup, V₀, p₀, t, Δt) where {M}
    T = eltype(V₀)

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

    TimeStepper{M, T}(; method, n, V, p, t, Vₙ, pₙ, tₙ, Δtₙ, setup, cache, momentum_cache)
end

const AdamsBashforthCrankNicolsonStepper{S, T} = TimeStepper{AdamsBashforthCrankNicolsonMethod{S}, T}
const OneLegStepper{S, T} = TimeStepper{OneLegMethod{S}, T}
const ExplicitRungeKuttaStepper{S, T} = TimeStepper{ExplicitRungeKuttaMethod{S}, T}
const ImplicitRungeKuttaStepper{S, T} = TimeStepper{ImplicitRungeKuttaMethod{S}, T}
