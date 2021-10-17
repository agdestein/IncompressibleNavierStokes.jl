"""
    TimeStepperCache

Time stepper cache.
"""
abstract type TimeStepperCache end

struct ExplicitRungeKuttaStepperCache{T} <: TimeStepperCache
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

struct ImplicitRungeKuttaStepperCache{T} <: TimeStepperCache
    Vtotₙ
    ptotₙ
    Vⱼ
    pⱼ
    Qⱼ
    Fⱼ
    ∇Fⱼ
    f
    A
    b
    c
    s
    Is
    Ω_sNV
    A_ext
    b_ext
    c_ext
end

struct AdamsBashforthCrankNicolsonStepperCache{T} <: TimeStepperCache
    F::Vector{T}
    Δp::Vector{T}
end

struct OneLegStepperCache{T} <: TimeStepperCache
    F::Vector{T}
    GΔp::Vector{T}
    Diff_fact::Factorization{T}
end

"""
    time_stepper_cache(stepper, args...; kwargs...)

Get time stepper cache for the given time stepper.
"""
function time_stepper_cache end

function time_stepper_cache(ts::AdamsBashforthCrankNicolsonStepper, setup)
    T = Float64
    @unpack model = setup
    @unpack NV, Np, Ω⁻¹ = setup.grid
    @unpack Diff = setup.discretization
    @unpack Δt = setup.Δt
    @unpack θ = ts

    F = zeros(NV)
    Δp = zeros(T, Np)

    ## Additional for implicit time stepping diffusion
    if model isa LaminarModel
        # Implicit time-stepping for diffusion
        # FIXME: This only works if Δt is constant
        # LU decomposition
        Diff_fact = lu(sparse(I, NV, NV) - θ * Δt *  Diagonal(Ω⁻¹) * Diff)
    else
        Diff_fact = cholesky(spzeros(0, 0))
    end

    AdamsBashforthCrankNicolsonStepperCache{T}(F, Δp, Diff_fact)
end

function time_stepper_cache(::OneLegStepper, setup)
    T = Float64
    @unpack NV = setup.grid
    F = zeros(T, NV)
    GΔp = zeros(T, NV)
    OneLegStepperCache{T}(F, GΔp)
end

function time_stepper_cache(stepper::ExplicitRungeKuttaStepper, setup)
    # TODO: Decide where `T` is to be passed
    T = Float64

    @unpack NV, Np = setup.grid

    ns = nstage(stepper)
    kV = zeros(T, NV, ns)
    kp = zeros(T, Np, ns)
    Vtemp = zeros(T, NV)
    Vtemp2 = zeros(T, NV)
    F = zeros(T, NV)
    ∇F = spzeros(T, NV, NV)
    f = zeros(T, Np)
    Δp = zeros(T, Np)

    # Get coefficients of RK method
    A, b, c, = tableau(stepper)

    # Shift Butcher tableau, as A[1, :] is always zero for explicit methods
    A = [A[2:end, :]; b']

    # Vector with time instances (1 is the time level of final step)
    c = [c[2:end]; 1]

    ExplicitRungeKuttaStepperCache{T}(kV, kp, Vtemp, Vtemp2, F, ∇F, f, Δp, T.(A), T.(b), T.(c))
end

function time_stepper_cache(stepper::ImplicitRungeKuttaStepper, setup)
    # TODO: Decide where `T` is to be passed
    T = Float64

    @unpack Np, Ω = setup.grid

    # Get coefficients of RK method
    A, b, c, = tableau(stepper)

    # Number of stages
    s = length(b)

    # Extend the Butcher tableau
    Is = sparse(I, s, s)
    Ω_sNV = kron(Is, spdiagm(Ω))
    A_ext = kron(A, sparse(I, NV, NV))
    b_ext = kron(b', sparse(I, NV, NV))
    c_ext = spdiagm(c)

    Vtotₙ = zeros(s * NV)
    ptotₙ = zeros(s * Np)
    Vⱼ = zeros(s * NV)
    pⱼ = zeros(s * Np)
    Qⱼ = zeros(s * (NV + Np))

    Fⱼ = zeros(s * NV)
    ∇Fⱼ = spzeros(s * NV, s * NV)

    f = zeros(s * (NV + Np))

    ImplicitRungeKuttaStepperCache{T}(
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
