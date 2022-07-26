abstract type AbstractTimeStepper{T,N} end

"""
    ExplicitRungeKuttaStepper

Explicit Runge Kutta Time Stepper.
"""
Base.@kwdef struct ExplicitRungeKuttaStepper{T,N} <: AbstractTimeStepper{T,N}
    method::ExplicitRungeKuttaMethod{T}
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    setup::Setup{T,N}
end

"""
    ImplicitRungeKuttaStepper

Implicit Runge Kutta Time Stepper.
"""
Base.@kwdef struct ImplicitRungeKuttaStepper{T,N} <: AbstractTimeStepper{T,N}
    method::ImplicitRungeKuttaMethod{T}
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    setup::Setup{T,N}
end

"""
    AdamsBashforthCrankNicolsonStepper

Adams-Bashforth Crank-Nicolson Stepper.
"""
Base.@kwdef struct AdamsBashforthCrankNicolsonStepper{T,N} <: AbstractTimeStepper{T,N}
    method::AdamsBashforthCrankNicolsonMethod{T}
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    Vₙ::Vector{T}
    pₙ::Vector{T}
    tₙ::T
    setup::Setup{T,N}
end

"""
    OneLegStepper

One-leg Stepper.
"""
Base.@kwdef struct OneLegStepper{T,N} <: AbstractTimeStepper{T,N}
    method::OneLegMethod{T}
    n::Int = 0
    V::Vector{T}
    p::Vector{T}
    t::T
    Vₙ::Vector{T}
    pₙ::Vector{T}
    tₙ::T
    setup::Setup{T,N}
end

"""
    TimeStepper(method, setup, V₀, p₀, t, Δt) -> TimeStepper

Build associated time stepper from method.
"""
function time_stepper(method::ExplicitRungeKuttaMethod, setup::Setup{T,N}, V, p, t) where {T,N}
    # Initialize solution vectors (leave input intact)
    n = 0
    V = copy(V)
    p = copy(p)
    cache = ode_method_cache(method, setup)
    momentum_cache = MomentumCache(setup)
    stepper = ExplicitRungeKuttaStepper{T,N}(; method, n, V, p, t, setup)
    stepper, cache, momentum_cache
end

function time_stepper(method::ImplicitRungeKuttaMethod, setup::Setup{T,N}, V, p, t) where {T,N}
    # Initialize solution vectors (leave input intact)
    n = 0
    V = copy(V)
    p = copy(p)
    cache = ode_method_cache(method, setup)
    momentum_cache = MomentumCache(setup)
    stepper = ImplicitRungeKuttaStepper{T,N}(; method, n, V, p, t, setup)
    stepper, cache, momentum_cache
end

function time_stepper(method::AdamsBashforthCrankNicolsonMethod, setup::Setup{T,N}, V₀, p₀, t₀, Δt) where {T,N}
    # Initialize solution vectors (leave input intact)
    n = 0
    V = copy(V₀)
    p = copy(p₀)

    # Current solution
    Vₙ = copy(V)
    pₙ = copy(p)
    tₙ = t
    Δtₙ = Δt

    cache = ode_method_cache(method, setup)
    momentum_cache = MomentumCache(setup)
    stepper = AdamsBashforthCrankNicolsonStepper{T,N}(; method, n, V, p, t, setup)

    stepper, cache, momentum_cache
end

function time_stepper(method::OneLegMethod, setup::Setup{T,N}, V₀, p₀, t₀, Δt) where {T,N}
    # Initialize solution vectors (leave input intact)
    n = 0
    V = copy(V₀)
    p = copy(p₀)

    # Current solution
    Vₙ = copy(V)
    pₙ = copy(p)
    tₙ = t
    Δtₙ = Δt

    cache = ode_method_cache(method, setup)
    momentum_cache = MomentumCache(setup)
    stepper = OneLegStepper{T,N}(; method, n, V, p, t, setup)

    stepper, cache, momentum_cache
end

"""
    get_state(stepper)

Return the current state `(V, p, t)`.
"""
function get_state end

get_state(s::ExplicitRungeKuttaStepper) = (s.V, s.p, s.t)
get_state(s::ImplicitRungeKuttaStepper) = (s.V, s.p, s.t)
get_state(s::AdamsBashforthCrankNicolsonStepper) = (s.V, s.p, s.t)
get_state(s::OneLegStepper) = (s.V, s.p, s.t)
