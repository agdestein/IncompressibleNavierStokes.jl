"""
    AbstractODEMethodCache

Time stepper cache.
"""
abstract type AbstractODEMethodCache{T} end

Base.@kwdef struct ExplicitRungeKuttaCache{T} <: AbstractODEMethodCache{T}
    kV::Matrix{T}
    kp::Matrix{T}
    Vtemp::Vector{T}
    Vtemp2::Vector{T}
    F::Vector{T}
    ∇F::SparseMatrixCSC{T,Int}
    f::Vector{T}
    Δp::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
end

Base.@kwdef struct ImplicitRungeKuttaCache{T} <: AbstractODEMethodCache{T}
    Vtotₙ::Vector{T}
    ptotₙ::Vector{T}
    Vⱼ::Vector{T}
    pⱼ::Vector{T}
    Qⱼ::Vector{T}
    Fⱼ::Vector{T}
    ∇Fⱼ::SparseMatrixCSC{T,Int}
    f::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    c::Vector{T}
    s::Vector{T}
    Is::Vector{T}
    Ω_sNV::Vector{T}
    A_ext::Matrix{T}
    b_ext::Vector{T}
    c_ext::Vector{T}
end

Base.@kwdef struct AdamsBashforthCrankNicolsonCache{T} <: AbstractODEMethodCache{T}
    cₙ::Vector{T} 
    cₙ₋₁::Vector{T} 
    F::Vector{T}
    Δp::Vector{T}
    Rr::Vector{T}
    b::Vector{T}
    bₙ::Vector{T}
    bₙ₊₁::Vector{T}
    yDiffₙ::Vector{T}
    yDiffₙ₊₁::Vector{T}
    Gpₙ::Vector{T}
    Diff_fact::Factorization{T}
end

Base.@kwdef struct OneLegCache{T} <: AbstractODEMethodCache{T}
    Vₙ₋₁::Vector{T} 
    pₙ₋₁::Vector{T} 
    F::Vector{T}
    GΔp::Vector{T}
end

"""
    ode_method_cache(method, args...; kwargs...)

Get time stepper cache for the given ODE method.
"""
function ode_method_cache end

function ode_method_cache(method::AdamsBashforthCrankNicolsonMethod, setup)
    @unpack model = setup
    @unpack NV, Np, Ω⁻¹ = setup.grid
    @unpack Diff = setup.discretization
    @unpack Δt = setup.time
    @unpack θ = method

    T = typeof(Δt)

    cₙ = zeros(T, NV)
    cₙ₋₁ = zeros(T, NV)
    F = zeros(T, NV)
    Δp = zeros(T, Np)
    Rr = zeros(T, NV)
    b = zeros(T, NV)
    bₙ = zeros(T, NV)
    bₙ₊₁ = zeros(T, NV)
    yDiffₙ = zeros(T, NV)
    yDiffₙ₊₁ = zeros(T, NV)
    Gpₙ = zeros(T, NV)

    ## Additional for implicit time stepping diffusion
    if model isa LaminarModel
        # Implicit time-stepping for diffusion
        # FIXME: This only works if Δt is constant
        # LU decomposition
        Diff_fact = lu(sparse(I, NV, NV) - θ * Δt *  Diagonal(Ω⁻¹) * Diff)
    else
        Diff_fact = cholesky(spzeros(0, 0))
    end

    AdamsBashforthCrankNicolsonCache{T}(; cₙ, cₙ₋₁, F, Δp, Rr, b, bₙ, bₙ₊₁, yDiffₙ, yDiffₙ₊₁, Gpₙ, Diff_fact)
end

function ode_method_cache(::OneLegMethod, setup)
    @unpack NV, Np = setup.grid
    T = typeof(setup.time.Δt)
    Vₙ₋₁ = zeros(T, NV)
    pₙ₋₁ = zeros(T, Np)
    F = zeros(T, NV)
    GΔp = zeros(T, NV)
    OneLegCache{T}(; Vₙ₋₁, pₙ₋₁, F, GΔp)
end

function ode_method_cache(method::ExplicitRungeKuttaMethod, setup)
    # TODO: Decide where `T` is to be passed
    T = typeof(setup.time.Δt)

    @unpack NV, Np = setup.grid

    ns = nstage(method)
    kV = zeros(T, NV, ns)
    kp = zeros(T, Np, ns)
    Vtemp = zeros(T, NV)
    Vtemp2 = zeros(T, NV)
    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)

    # Get coefficients of RK method
    @unpack A, b, c = method

    # Shift Butcher tableau, as A[1, :] is always zero for explicit methods
    A = [A[2:end, :]; b']

    # Vector with time instances (1 is the time level of final step)
    c = [c[2:end]; 1]

    ExplicitRungeKuttaCache{T}(; kV, kp, Vtemp, Vtemp2, F, ∇F, f, Δp, A = T.(A), b = T.(b), c = T.(c))
end

function ode_method_cache(method::ImplicitRungeKuttaMethod, setup)
    # TODO: Decide where `T` is to be passed
    T = typeof(setup.time.Δt)

    @unpack Np, Ω = setup.grid

    # Get coefficients of RK method
    @unpack A, b, c = method

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
    Vⱼ = zeros(T, s * NV)
    pⱼ = zeros(T, s * Np)
    Qⱼ = zeros(T, s * (NV + Np))

    Fⱼ = zeros(T, s * NV)
    ∇Fⱼ = spzeros(T, s * NV, s * NV)

    f = zeros(T, s * (NV + Np))

    ImplicitRungeKuttaCache{T}(;
        Vtotₙ,
        ptotₙ,
        Vⱼ,
        pⱼ,
        Qⱼ,
        Fⱼ,
        ∇Fⱼ,
        f,
        A,
        b,
        c,
        s,
        Is,
        Ω_sNV,
        A_ext,
        b_ext,
        c_ext,
    )
end
