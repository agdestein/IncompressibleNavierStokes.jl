"""
    AbstractODEMethodCache

Time stepper cache.
"""
abstract type AbstractODEMethodCache{T} end

Base.@kwdef struct ExplicitRungeKuttaCache{T} <: AbstractODEMethodCache{T}
    kV::Matrix{T}
    Vtemp::Vector{T}
    Vₙ::Vector{T}
    pₙ::Vector{T}
    F::Vector{T}
    ∇F::SparseMatrixCSC{T,Int}
    f::Vector{T}
    Δp::Vector{T}
end

Base.@kwdef struct ImplicitRungeKuttaCache{T} <: AbstractODEMethodCache{T}
    Vtotₙ::Vector{T}
    ptotₙ::Vector{T}
    Qⱼ::Vector{T}
    Fⱼ::Vector{T}
    ∇Fⱼ::SparseMatrixCSC{T,Int}
    fⱼ::Vector{T}
    F::Vector{T}
    ∇F::SparseMatrixCSC{T,Int}
    f::Vector{T}
    Δp::Vector{T}
    Gp::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    Is::SparseMatrixCSC{T,Int}
    Ω_sNV::SparseMatrixCSC{T,Int}
    A_ext::SparseMatrixCSC{T,Int}
    b_ext::SparseMatrixCSC{T,Int}
    c_ext::SparseMatrixCSC{T,Int}
    Gtot::SparseMatrixCSC{T,Int}
    Mtot::SparseMatrixCSC{T,Int}
    Ωtot::Vector{T}
    dfmom::SparseMatrixCSC{T,Int}
    Z::SparseMatrixCSC{T,Int}
end

Base.@kwdef mutable struct AdamsBashforthCrankNicolsonCache{T} <: AbstractODEMethodCache{T}
    cₙ::Vector{T}
    cₙ₋₁::Vector{T}
    F::Vector{T}
    f::Vector{T}
    Δp::Vector{T}
    Rr::Vector{T}
    b::Vector{T}
    bₙ::Vector{T}
    bₙ₊₁::Vector{T}
    Gpₙ::Vector{T}
    Diff_fact::Factorization{T}
    Δt::T
end

Base.@kwdef struct OneLegCache{T} <: AbstractODEMethodCache{T}
    Vₙ₋₁::Vector{T}
    pₙ₋₁::Vector{T}
    F::Vector{T}
    f::Vector{T}
    Δp::Vector{T}
    GΔp::Vector{T}
end

"""
    ode_method_cache(method, setup)

Get time stepper cache for the given ODE method.
"""
function ode_method_cache end

function ode_method_cache(::AdamsBashforthCrankNicolsonMethod{T}, setup) where {T}
    (; NV, Np) = setup.grid

    cₙ = zeros(T, NV)
    cₙ₋₁ = zeros(T, NV)
    F = zeros(T, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    Rr = zeros(T, NV)
    b = zeros(T, NV)
    bₙ = zeros(T, NV)
    bₙ₊₁ = zeros(T, NV)
    Gpₙ = zeros(T, NV)

    # Compute factorization at first time step (guaranteed since Δt > 0)
    Δt = 0
    Diff_fact = cholesky(spzeros(0, 0))

    AdamsBashforthCrankNicolsonCache{T}(;
        cₙ,
        cₙ₋₁,
        F,
        f,
        Δp,
        Rr,
        b,
        bₙ,
        bₙ₊₁,
        Gpₙ,
        Diff_fact,
        Δt,
    )
end

function ode_method_cache(::OneLegMethod{T}, setup) where {T}
    (; NV, Np) = setup.grid
    Vₙ₋₁ = zeros(T, NV)
    pₙ₋₁ = zeros(T, Np)
    F = zeros(T, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    GΔp = zeros(T, NV)
    OneLegCache{T}(; Vₙ₋₁, pₙ₋₁, F, f, Δp, GΔp)
end

function ode_method_cache(method::ExplicitRungeKuttaMethod{T}, setup) where {T}
    (; NV, Np) = setup.grid
    ns = nstage(method)
    kV = zeros(T, NV, ns)
    Vtemp = zeros(T, NV)
    Vₙ = zeros(T, NV)
    pₙ = zeros(T, Np)
    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    ExplicitRungeKuttaCache{T}(; kV, Vtemp, Vₙ, pₙ, F, ∇F, f, Δp)
end

function ode_method_cache(method::ImplicitRungeKuttaMethod{T}, setup) where {T}
    (; NV, Np, Ω) = setup.grid
    (; G, M) = setup.operators
    (; A, b, c) = method

    # Number of stages
    s = length(b)

    # Extend the Butcher tableau
    Is = sparse(I, s, s)
    Ω_sNV = kron(Is, spdiagm(Ω))
    A_ext = kron(A, sparse(I, NV, NV))
    b_ext = kron(b', sparse(I, NV, NV))
    c_ext = spdiagm(c)

    Vtotₙ = zeros(T, s * NV)
    ptotₙ = zeros(T, s * Np)
    Qⱼ = zeros(T, s * (NV + Np))

    Fⱼ = zeros(T, s * NV)
    ∇Fⱼ = spzeros(T, s * NV, s * NV)

    fⱼ = zeros(T, s * (NV + Np))

    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)
    Gp = zeros(T, NV)

    # Gradient operator (could also use 1 instead of c and later scale the pressure)
    Gtot = kron(A, G)

    # Divergence operator
    Mtot = kron(Is, M)

    # Finite volumes
    Ωtot = kron(ones(s), Ω)

    # Iteration matrix
    dfmom = spzeros(T, s * NV, s * NV)
    Z2 = spzeros(T, s * Np, s * Np)
    Z = [dfmom Gtot; Mtot Z2]

    ImplicitRungeKuttaCache{T}(;
        Vtotₙ,
        ptotₙ,
        Qⱼ,
        Fⱼ,
        ∇Fⱼ,
        fⱼ,
        F,
        ∇F,
        f,
        Δp,
        Gp,
        A,
        b,
        c,
        Is,
        Ω_sNV,
        A_ext,
        b_ext,
        c_ext,
        Gtot,
        Mtot,
        Ωtot,
        dfmom,
        Z,
    )
end
